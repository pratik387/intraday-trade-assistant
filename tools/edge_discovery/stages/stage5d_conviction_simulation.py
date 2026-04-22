"""Stage 5d: Conviction gate simulation (backtest replay).

Replays ConvictionGate chronologically against a trade stream. Each trade is
scored by the injected scorer, evaluated by the gate, and marked admitted/
rejected in the returned DataFrame.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Protocol

import pandas as pd

from services.conviction.feature_spec import extract_features
from services.conviction.gate import ConvictionGate
from tools.edge_discovery.report_writer import write_json_artifact, append_section


class _ScorerLike(Protocol):
    def predict(self, feat: Dict[str, float]) -> float: ...


def simulate_conviction_filter(
    trades: pd.DataFrame, scorer: _ScorerLike, cfg: Dict[str, Any]
) -> pd.DataFrame:
    """Replay ConvictionGate on the trade stream chronologically.

    Adds columns: admitted (bool), predicted_r (float), reject_reason (str).
    """
    gate = ConvictionGate(cfg)
    t = trades.copy()
    t["_decision_ts_parsed"] = pd.to_datetime(t["decision_ts"], errors="coerce")
    t = t.sort_values("_decision_ts_parsed").reset_index(drop=True)

    preds: List[float] = []
    admitted: List[bool] = []
    reasons: List[str] = []

    for row in t.itertuples():
        row_dict = t.iloc[row.Index].to_dict()
        feat = extract_features(row_dict)
        symbol = row_dict.get("symbol", "")
        pred = float(scorer.predict({**feat, "symbol": symbol}))
        cand = {
            "symbol": symbol,
            "decision_ts": row_dict.get("decision_ts"),
            "session_date": row_dict.get("session_date_dt"),
        }
        ok, reason = gate.evaluate(cand, pred)
        preds.append(pred)
        admitted.append(ok)
        reasons.append(reason)

    t["predicted_r"] = preds
    t["admitted"] = admitted
    t["reject_reason"] = reasons
    t = t.drop(columns=["_decision_ts_parsed"])
    return t


def _aggregate_stats(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    pnl = df["total_trade_pnl"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0].abs()
    pf = float(wins.sum() / losses.sum()) if losses.sum() > 0 else float("inf")
    wr = 100 * len(wins) / len(df) if len(df) else 0.0
    daily = df.groupby("session_date_dt")["total_trade_pnl"].agg(["count", "sum"])
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


def _rows_to_markdown(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "_(no rows)_"
    headers = list(rows[0].keys())
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    return "\n".join(out)


def run_stage5d(
    trades: pd.DataFrame,
    scorer: _ScorerLike,
    cfg: Dict[str, Any],
    report_path: Path,
    summary_json: Path,
) -> Dict[str, Any]:
    """Full stage 5d run: simulate, aggregate, write report + JSON artifact."""
    filtered = simulate_conviction_filter(trades, scorer, cfg)
    before = _aggregate_stats(filtered, "Before ConvictionGate")
    after = _aggregate_stats(
        filtered[filtered["admitted"].astype(bool)], "After ConvictionGate"
    )
    rej_reasons = (
        filtered[~filtered["admitted"].astype(bool)]["reject_reason"]
        .value_counts()
        .head(10)
        .to_dict()
    )
    rej_rows = [{"reason": k, "count": int(v)} for k, v in rej_reasons.items()]

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# Stage 5d — Conviction Gate Simulation",
        "",
        "## Scenarios",
        "",
        _rows_to_markdown([before, after]),
    ]
    report_path.write_text("\n".join(header) + "\n", encoding="utf-8")
    append_section(report_path, "## Top rejection reasons", _rows_to_markdown(rej_rows))

    delta = {
        "n_trades_delta": after["n_trades"] - before["n_trades"],
        "pf_delta": round(after["pf"] - before["pf"], 3),
        "sharpe_delta": round(after["session_sharpe"] - before["session_sharpe"], 3),
    }
    write_json_artifact(
        summary_json,
        {
            "stage": "5d",
            "cfg": cfg,
            "before": before,
            "after": after,
            "delta": delta,
            "rejections": rej_rows,
        },
    )
    return {"before": before, "after": after, "delta": delta}
