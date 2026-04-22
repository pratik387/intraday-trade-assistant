"""Stage 5e: Illiquid-aware Budgeted Selector simulation.

Replaces Stage 5d's FIFO cap-50 ConvictionGate with a multi-constraint
selector that reflects how pro NSE intraday systems operate for
illiquid/unknown-cap stocks (research 2026-04-22).

Constraints applied per candidate chronologically:
1. Opening-bell embargo (micro_cap/unknown during 9:15-9:30 hard-blocked)
2. Time-bucket quotas (opening 0% / morning 50% / lunch 15% / afternoon 30% / close 5%)
3. ADV_rupees size cap (position_notional <= 1.5% of 20d ADV for micro/unknown, 3% larger)
4. 5m bar participation cap (position_notional <= 10% of entry bar rupee volume)
5. Per-symbol rate limit (<=3 entries/min for micro/unknown)
6. Per-cap-segment concurrency (max 4 open for micro/unknown/small, unlimited larger)

Rationale: daily trade count is an EMERGENT output, not an input. Bound
liquidity/concurrency/time and the right count falls out naturally without
starving the afternoon like the old FIFO cap did.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from tools.edge_discovery.report_writer import append_section, write_json_artifact


# ---------------------------------------------------------------------------
# Time buckets (minute-of-day ranges)
# ---------------------------------------------------------------------------
# 555 = 9:15, 570 = 9:30, 690 = 11:30, 810 = 13:30, 900 = 15:00, 915 = 15:15
def _bucket_of(mod: int) -> str:
    if mod < 570:
        return "opening"
    if mod < 690:
        return "morning"
    if mod < 810:
        return "lunch"
    if mod < 900:
        return "afternoon"
    return "close"


def simulate_budgeted_selector(
    trades: pd.DataFrame,
    adv_map: Dict[Tuple[str, date], float],
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Apply illiquid-aware budgeted selection chronologically.

    Args:
        trades: DataFrame with at minimum these columns:
            symbol, setup_type, cap_segment, decision_ts (str or datetime),
            session_date_dt (date), minute_of_day (int), plan_notional (float,
            rupees), volume5 (float, shares), close5 (float, rupees),
            last_exit_ts (str or datetime) — used for concurrency tracking.
        adv_map: dict keyed by (symbol_without_NSE_prefix, date) → ADV_rupees_20d.
            Missing key treated as NaN → ADV cap skipped for that trade.
        cfg: config dict with keys:
            daily_budget (int, default 50)
            time_buckets (dict str→float, default {morning:.5, lunch:.15, afternoon:.3, close:.05})
            adv_cap_micro (float, default 0.015)
            adv_cap_large (float, default 0.03)
            bar_cap_pct (float, default 0.10)
            rate_limit_per_min (int, default 3)
            rate_limit_caps (list, default ["micro_cap","unknown"])
            max_concurrent_by_cap (dict str→int)
            embargo_caps (list, default ["micro_cap","unknown"])
            embargo_start_mod (int, default 555 = 9:15)
            embargo_end_mod (int, default 570 = 9:30)

    Returns:
        Copy of input with added columns: admitted (bool), reject_reason (str),
        time_bucket (str).
    """
    # Defaults
    cfg = {
        "daily_budget": 50,
        "time_buckets": {"morning": 0.50, "lunch": 0.15, "afternoon": 0.30, "close": 0.05},
        "adv_cap_micro": 0.015,
        "adv_cap_large": 0.03,
        "bar_cap_pct": 0.10,
        "rate_limit_per_min": 3,
        "rate_limit_caps": ["micro_cap", "unknown"],
        "max_concurrent_by_cap": {
            "micro_cap": 4,
            "unknown": 4,
            "small_cap": 4,
            "mid_cap": 100,
            "large_cap": 100,
        },
        "embargo_caps": ["micro_cap", "unknown"],
        "embargo_start_mod": 555,
        "embargo_end_mod": 570,
        **cfg,
    }

    total_daily_budget = int(cfg["daily_budget"])
    bucket_fracs = cfg["time_buckets"]
    adv_cap_micro = float(cfg["adv_cap_micro"])
    adv_cap_large = float(cfg["adv_cap_large"])
    bar_cap_pct = float(cfg["bar_cap_pct"])
    rate_limit = int(cfg["rate_limit_per_min"])
    rate_limit_caps = set(cfg["rate_limit_caps"])
    max_concurrent_by_cap = cfg["max_concurrent_by_cap"]
    embargo_caps = set(cfg["embargo_caps"])
    embargo_start = int(cfg["embargo_start_mod"])
    embargo_end = int(cfg["embargo_end_mod"])

    # Normalize timestamps
    t = trades.copy()
    t["dts_parsed"] = pd.to_datetime(t["decision_ts"], errors="coerce")
    t["lexit_ts_parsed"] = pd.to_datetime(t.get("last_exit_ts"), errors="coerce")
    t = t.sort_values("dts_parsed").reset_index(drop=True)

    # State
    bucket_counts: Dict[str, int] = defaultdict(int)
    symbol_entries: Dict[str, List[datetime]] = defaultdict(list)  # per-symbol entry times (rate limit)
    # Open positions per cap_segment: list of exit_ts (for concurrency check)
    open_by_cap: Dict[str, List[datetime]] = defaultdict(list)
    current_session: Any = None

    admitted: List[bool] = []
    reasons: List[str] = []
    buckets: List[str] = []

    for row in t.itertuples():
        ts = row.dts_parsed
        exit_ts = row.lexit_ts_parsed
        sess = row.session_date_dt
        cap = row.cap_segment
        sym_raw = str(row.symbol).replace("NSE:", "")
        notional = float(getattr(row, "plan_notional", 0) or 0)
        vol5 = float(getattr(row, "volume5", 0) or 0)
        close5 = float(getattr(row, "close5", 0) or 0)

        # Session boundary reset
        if sess != current_session:
            current_session = sess
            bucket_counts.clear()
            symbol_entries.clear()
            open_by_cap.clear()

        mod = int(row.minute_of_day) if hasattr(row, "minute_of_day") else (
            ts.hour * 60 + ts.minute if pd.notna(ts) else 0
        )
        bucket = _bucket_of(mod)
        buckets.append(bucket)

        # Purge expired open positions for this cap (those whose exit_ts < current ts)
        if pd.notna(ts):
            open_by_cap[cap] = [e for e in open_by_cap[cap] if pd.notna(e) and e > ts]

        # [1] Opening-bell embargo
        if cap in embargo_caps and embargo_start <= mod < embargo_end:
            reasons.append("embargo:opening_micro_unknown")
            admitted.append(False)
            continue

        # [2] Time-bucket quota
        bucket_quota = int(total_daily_budget * bucket_fracs.get(bucket, 0))
        if bucket_counts[bucket] >= bucket_quota:
            reasons.append(f"bucket_quota:{bucket}_{bucket_counts[bucket]}_of_{bucket_quota}")
            admitted.append(False)
            continue

        # [3] ADV_rupees size cap
        adv = adv_map.get((sym_raw, sess))
        if adv is not None and adv > 0 and notional > 0:
            cap_pct = adv_cap_micro if cap in ("micro_cap", "unknown") else adv_cap_large
            if notional > cap_pct * adv:
                reasons.append(f"adv_cap:notional_{int(notional)}_vs_{cap_pct*100:.1f}pct_adv_{int(adv)}")
                admitted.append(False)
                continue

        # [4] 5m bar participation cap
        bar_rupee = vol5 * close5
        if bar_rupee > 0 and notional > bar_cap_pct * bar_rupee:
            reasons.append(f"bar_cap:notional_{int(notional)}_vs_{int(bar_cap_pct*bar_rupee)}")
            admitted.append(False)
            continue

        # [5] Per-symbol rate limit (micro/unknown only)
        if cap in rate_limit_caps and pd.notna(ts):
            cutoff = ts - pd.Timedelta(seconds=60)
            symbol_entries[sym_raw] = [e for e in symbol_entries[sym_raw] if e > cutoff]
            if len(symbol_entries[sym_raw]) >= rate_limit:
                reasons.append(f"rate_limit:{rate_limit}_per_min")
                admitted.append(False)
                continue

        # [6] Per-cap-segment concurrency
        max_c = int(max_concurrent_by_cap.get(cap, 100))
        if len(open_by_cap[cap]) >= max_c:
            reasons.append(f"cap_concurrency:{cap}_{len(open_by_cap[cap])}_of_{max_c}")
            admitted.append(False)
            continue

        # ADMIT
        bucket_counts[bucket] += 1
        if cap in rate_limit_caps and pd.notna(ts):
            symbol_entries[sym_raw].append(ts)
        if pd.notna(exit_ts):
            open_by_cap[cap].append(exit_ts)
        admitted.append(True)
        reasons.append("admitted")

    t["admitted"] = admitted
    t["reject_reason"] = reasons
    t["time_bucket"] = buckets
    t = t.drop(columns=["dts_parsed", "lexit_ts_parsed"])
    return t


def _aggregate_stats(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    pnl = df["total_trade_pnl"] if "total_trade_pnl" in df.columns else df.get("pnl", pd.Series([0]))
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0].abs()
    pf = float(wins.sum() / losses.sum()) if losses.sum() > 0 else float("inf")
    wr = 100 * len(wins) / len(df) if len(df) else 0.0
    daily = df.groupby("session_date_dt")["total_trade_pnl"].agg(["count", "sum"]) if "session_date_dt" in df.columns else pd.DataFrame({"count": [0], "sum": [0]})
    sharpe = float(daily["sum"].mean() / daily["sum"].std()) if daily["sum"].std() > 0 else 0.0
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
        "session_sharpe": round(sharpe, 3),
        "losing_days_pct": round(100 * losing_days / n_sessions, 1) if n_sessions else 0.0,
    }


def _rows_to_markdown(rows):
    if not rows:
        return "_(no rows)_"
    headers = list(rows[0].keys())
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    return "\n".join(out)


def run_stage5e(
    trades: pd.DataFrame,
    adv_map: Dict[Tuple[str, date], float],
    cfg: Dict[str, Any],
    report_path: Path,
    summary_json: Path,
) -> Dict[str, Any]:
    """Run Stage 5e simulation, emit markdown + JSON report."""
    filtered = simulate_budgeted_selector(trades, adv_map, cfg)
    mask = filtered["admitted"].astype(bool)

    before = _aggregate_stats(filtered, "Before BudgetedSelector")
    after = _aggregate_stats(filtered[mask], "After BudgetedSelector")

    rej = filtered[~mask]["reject_reason"].value_counts().head(15).to_dict()
    rej_rows = [{"reason": k, "count": int(v)} for k, v in rej.items()]

    # Per-bucket admitted count
    bucket_dist = filtered[mask]["time_bucket"].value_counts().to_dict() if mask.sum() > 0 else {}
    bucket_rows = [{"bucket": k, "admitted_count": int(v)} for k, v in bucket_dist.items()]

    # Per cap_segment admitted
    cap_dist = filtered[mask]["cap_segment"].value_counts().to_dict() if mask.sum() > 0 else {}
    cap_rows = [{"cap_segment": k, "admitted_count": int(v)} for k, v in cap_dist.items()]

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# Stage 5e - Budgeted Selector Simulation (Illiquid-aware)",
        "",
        "Six-constraint selection replacing FIFO cap: opening embargo, time-bucket",
        "quotas, ADV cap, bar participation cap, per-symbol rate limit, cap-segment",
        "concurrency. Daily trade count is emergent, not an input.",
        "",
        "## Scenarios",
        "",
        _rows_to_markdown([before, after]),
    ]
    report_path.write_text("\n".join(header) + "\n", encoding="utf-8")
    append_section(report_path, "## Admitted by time bucket", _rows_to_markdown(bucket_rows))
    append_section(report_path, "## Admitted by cap segment", _rows_to_markdown(cap_rows))
    append_section(report_path, "## Top rejection reasons", _rows_to_markdown(rej_rows))

    delta = {
        "pf_delta": round(after["pf"] - before["pf"], 3),
        "sharpe_delta": round(after["session_sharpe"] - before["session_sharpe"], 3),
        "trades_per_day_delta": round(after["trades_per_day"] - before["trades_per_day"], 1),
    }
    write_json_artifact(summary_json, {
        "stage": "5e",
        "cfg": cfg,
        "before": before,
        "after": after,
        "delta": delta,
        "rejections": rej_rows,
        "admitted_by_bucket": bucket_rows,
        "admitted_by_cap": cap_rows,
    })
    return {"before": before, "after": after, "delta": delta}
