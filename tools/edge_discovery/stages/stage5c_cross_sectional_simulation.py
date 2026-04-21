"""Stage 5c: Cross-sectional gate simulation.

Replays CrossSectionalGate against the filter-matched trade stream (Stage 5b
input). For each candidate (in chronological order), updates UniverseRVOLState
from OHLCV history and CrowdednessCounter from prior candidates, then asks the
gate allow/reject.

Produces:
- 07-cross-sectional-simulation.md with before/after aggregate metrics
- stage5c_simulation.json machine-readable summary
"""
from __future__ import annotations

from datetime import datetime, time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from services.cross_sectional.crowdedness_counter import CrowdednessCounter
from services.cross_sectional.universe_rvol import UniverseRVOLState
from services.cross_sectional.gate import CrossSectionalGate, Candidate
from tools.edge_discovery.report_writer import write_json_artifact, append_section


def simulate_filter(trades: pd.DataFrame, ohlcv: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Apply CrossSectionalGate to each trade (chronologically), return trades
    with `allowed` + `reject_reason` columns added.

    Replays all OHLCV bars chronologically to warm UniverseRVOLState before
    each trade's bar-close, then evaluates the gate for trades at that bar.
    """
    rvol_state = UniverseRVOLState(
        rolling_sessions=int(cfg["f1_rolling_window_sessions"]),
        min_sessions=int(cfg["f1_min_history_sessions"]),
    )
    crowd = CrowdednessCounter(window_min=int(cfg["f2_crowdedness_window_min"]))
    gate = CrossSectionalGate(cfg, rvol=rvol_state, crowdedness=crowd)

    # Sort trades chronologically
    trades = trades.copy()
    trades["decision_ts_parsed"] = pd.to_datetime(trades["decision_ts"], errors="coerce")
    trades = trades.sort_values("decision_ts_parsed").reset_index(drop=True)

    # Pre-index ohlcv by (date_only, mod) -> {symbol -> volume} for efficiency.
    # Keys are sorted chronologically so historical bars are replayed to warm
    # UniverseRVOLState before we evaluate the current trade's bar.
    ohlcv_by_bar: Dict[tuple, Dict[str, int]] = {}
    if len(ohlcv):
        for (date_only, mod), grp in ohlcv.groupby(["date_only", "mod"]):
            ohlcv_by_bar[(date_only, mod)] = dict(zip(grp["symbol"], grp["volume"].astype(int)))
    # Sort historical bars chronologically (by date, then mod-of-day)
    sorted_bar_keys = sorted(ohlcv_by_bar.keys())

    # Maintain cap_segment lookup from trades themselves (ohlcv doesn't have it)
    symbol_caps = dict(zip(trades["symbol_raw"], trades["cap_segment"]))

    # Group trades by bar for efficient evaluation
    trades_by_bar: Dict[tuple, List[int]] = {}
    for idx, row in trades.iterrows():
        ts = row["decision_ts_parsed"]
        if pd.isnull(ts):
            continue
        bar_key = (ts.date(), int(row["minute_of_day"]))
        trades_by_bar.setdefault(bar_key, []).append(idx)

    allowed: List[bool] = [True] * len(trades)
    reasons: List[str] = ["no_ts"] * len(trades)
    # Trade bars (dedup'd) in chronological order
    trade_bar_keys_sorted = sorted(trades_by_bar.keys())

    # Merge historical bar keys with trade bar keys in chronological order;
    # for each bar: feed volumes into rvol_state (on_bar_close), then evaluate
    # any trades at that bar in their original chronological order.
    all_bar_keys = sorted(set(sorted_bar_keys) | set(trade_bar_keys_sorted))
    seen_bars = set()
    for bar_key in all_bar_keys:
        date_only, mod = bar_key
        bar_ts = datetime.combine(date_only, time(hour=mod // 60, minute=mod % 60))
        # On-bar-close update
        bar_vols = ohlcv_by_bar.get(bar_key)
        if bar_vols:
            rvol_state.on_bar_close(
                ts=bar_ts,
                bar_volumes=bar_vols,
                symbol_caps=symbol_caps,
            )
        seen_bars.add(bar_key)
        # If any trades fire at this bar, evaluate them in chronological order
        trade_indices = trades_by_bar.get(bar_key)
        if not trade_indices:
            continue
        # Sort indices by decision_ts_parsed to ensure chronological order within bar
        trade_indices_sorted = sorted(
            trade_indices,
            key=lambda i: trades.iloc[i]["decision_ts_parsed"],
        )
        for idx in trade_indices_sorted:
            row = trades.iloc[idx]
            ts = row["decision_ts_parsed"].to_pydatetime()
            cand = Candidate(
                symbol=row["symbol_raw"],
                setup_type=row["setup"],
                cap_segment=row["cap_segment"],
                hour_bucket=row["hour_bucket"],
                decision_ts=ts,
            )
            ok, reason = gate.evaluate(cand)
            allowed[idx] = bool(ok)
            reasons[idx] = reason
            # Record in crowdedness AFTER evaluation (current candidate counts in
            # FUTURE candidates' crowdedness, but not its own)
            crowd.record(row["setup"], ts)

    trades = trades.copy()
    # Store as object dtype so DataFrame access returns native Python bool,
    # preserving `is True` / `is False` identity semantics for callers.
    trades["allowed"] = pd.Series(allowed, index=trades.index, dtype=object)
    trades["reject_reason"] = reasons
    return trades


def _aggregate_stats(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    pnl = df["total_trade_pnl"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0].abs()
    pf = float(wins.sum() / losses.sum()) if losses.sum() > 0 else float("inf")
    wr = 100 * len(wins) / len(df) if len(df) else 0.0
    daily = df.groupby("session_date_dt")["total_trade_pnl"].agg(["count", "sum"])
    sess_sharpe = float(daily["sum"].mean() / daily["sum"].std()) if daily["sum"].std() > 0 else 0.0
    losing_days = int((daily["sum"] < 0).sum())
    n_sessions = len(daily)
    return {
        "scenario": name,
        "n_trades": int(len(df)),
        "n_sessions": n_sessions,
        "trades_per_day": round(len(df) / n_sessions, 1) if n_sessions else 0.0,
        "total_pnl": round(float(pnl.sum()), 0),
        "pf": round(pf, 3) if pf != float("inf") else 999.0,
        "wr_pct": round(wr, 1),
        "session_sharpe": round(sess_sharpe, 3),
        "losing_days_pct": round(100 * losing_days / n_sessions, 1) if n_sessions else 0.0,
    }


def _rows_to_markdown(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "_(no rows)_"
    headers = list(rows[0].keys())
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    return "\n".join(out)


def run_stage5c(
    trades: pd.DataFrame,
    ohlcv: pd.DataFrame,
    cfg: Dict[str, Any],
    report_path: Path,
    summary_json: Path,
) -> Dict[str, Any]:
    """Run cross-sectional filter replay + emit markdown + JSON."""
    filtered = simulate_filter(trades, ohlcv, cfg)
    allowed_mask = filtered["allowed"].astype(bool)
    before = _aggregate_stats(filtered, "Before CrossSectionalGate")
    after = _aggregate_stats(filtered[allowed_mask], "After CrossSectionalGate (F1+F2)")
    reject_reasons = filtered[~allowed_mask]["reject_reason"].value_counts().to_dict()
    reject_reasons_top = [{"reason": k, "count": int(v)} for k, v in list(reject_reasons.items())[:10]]

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# Stage 5c — Cross-Sectional Filter Simulation",
        "",
        "**Purpose:** Replay F1 (RVOL cap-conditional) + F2 (crowdedness universal) "
        "filters on the Stage-5b trade stream. Report before/after aggregate metrics.",
        "",
        "## Scenarios",
        "",
        _rows_to_markdown([before, after]),
    ]
    report_path.write_text("\n".join(header) + "\n", encoding="utf-8")
    append_section(report_path, "## Top rejection reasons", _rows_to_markdown(reject_reasons_top))

    delta = {
        "n_trades_delta": after["n_trades"] - before["n_trades"],
        "trades_per_day_delta": round(after["trades_per_day"] - before["trades_per_day"], 1),
        "pf_delta": round(after["pf"] - before["pf"], 3),
        "session_sharpe_delta": round(after["session_sharpe"] - before["session_sharpe"], 3),
        "losing_days_pct_delta": round(after["losing_days_pct"] - before["losing_days_pct"], 1),
    }
    write_json_artifact(summary_json, {
        "stage": "5c",
        "cfg": cfg,
        "before": before,
        "after": after,
        "delta": delta,
        "rejection_reasons_top": reject_reasons_top,
    })
    return {"before": before, "after": after, "delta": delta, "rejection_reasons_top": reject_reasons_top}
