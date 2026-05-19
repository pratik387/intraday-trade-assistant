"""Stage 5b: Ruleset simulation. Post-narrative-gate aggregation check.

Applies the set of APPROVED Stage-5 rules (each rule = setup + conditioner
key/values) as a union filter on the Discovery trades DataFrame, then reports
aggregate performance — PF, WR, session Sharpe, daily trade count, hour
distribution.

Why this exists: individual cells passing Stage 3 does NOT guarantee their
aggregate behavior is coherent. Overlapping cells, sub-population effects,
and mechanism assumptions that hold cell-by-cell can fail in combination.
Stage 5b catches this before Validation/Holdout waste OOS data on a degenerate
ruleset.

Per-trade match semantics: union (OR). A trade passes the filter iff at least
one approved rule matches (setup equality + all conditioner equalities).
"""
from pathlib import Path
from typing import Any, Dict, List
from collections import defaultdict

import pandas as pd

from tools.edge_discovery_legacy_gauntlet.report_writer import write_json_artifact, append_section


def apply_filter(trades: pd.DataFrame, approved_rules: List[Dict[str, Any]]) -> pd.DataFrame:
    """Return subset of `trades` matching at least one approved rule.

    Args:
        trades: DataFrame with setup_type + conditioner columns (regime,
                cap_segment, hour_bucket) + total_trade_pnl + session_date_dt
        approved_rules: list of {setup, conditions} where conditions is
                       list of (conditioner_key, value) tuples
    """
    rules_by_setup = defaultdict(list)
    for r in approved_rules:
        rules_by_setup[r["setup"]].append(r["conditions"])

    def matches(row):
        setup = row["setup_type"]
        if setup not in rules_by_setup:
            return False
        for conds in rules_by_setup[setup]:
            if all(row.get(k) == v for k, v in conds):
                return True
        return False

    mask = trades.apply(matches, axis=1)
    return trades[mask].copy()


def aggregate_stats(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    """PF / WR / session Sharpe / daily-count stats."""
    pnl = df["total_trade_pnl"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0].abs()
    pf = float(wins.sum() / losses.sum()) if losses.sum() > 0 else float("inf")
    wr = float(100 * len(wins) / len(df)) if len(df) else 0.0
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
        "median_daily_pnl": round(float(daily["sum"].median()), 0) if len(daily) else 0.0,
    }


def run_stage5b(
    trades: pd.DataFrame,
    approved_rules: List[Dict[str, Any]],
    report_path: Path,
    summary_json: Path,
) -> Dict[str, Any]:
    """Run ruleset simulation and write report + JSON.

    Args:
        trades: Discovery trades DataFrame from data_loader.load_run(...).trades
        approved_rules: list of {setup, conditions} from Stage 5 post-rejection
        report_path: markdown output path (e.g., '06-ruleset-simulation.md')
        summary_json: JSON artifact path

    Returns dict with scenarios + per-hour + per-setup breakdowns.
    """
    baseline = aggregate_stats(trades, "Baseline (raw wide-open, no filter)")

    filtered = apply_filter(trades, approved_rules)
    s_filter = aggregate_stats(filtered, "All approved rules (union filter)")
    s_excl_morning = aggregate_stats(
        filtered[~filtered["hour_bucket"].isin(["opening", "morning"])],
        "Exclude opening+morning entries",
    )
    s_late = aggregate_stats(filtered[filtered["hour_bucket"] == "late"], "Late-hour entries only")
    s_pm = aggregate_stats(
        filtered[filtered["hour_bucket"].isin(["afternoon", "late"])],
        "Afternoon+late entries only",
    )

    scenarios = [baseline, s_filter, s_excl_morning, s_late, s_pm]

    # Per-hour breakdown on filtered set
    per_hour_rows = []
    if len(filtered):
        for hb, grp in filtered.groupby("hour_bucket"):
            pnl = grp["total_trade_pnl"]
            wins = pnl[pnl > 0]
            losses = pnl[pnl < 0].abs()
            pf = float(wins.sum() / losses.sum()) if losses.sum() > 0 else float("inf")
            per_hour_rows.append({
                "hour_bucket": hb,
                "n": len(grp),
                "total_pnl": round(float(pnl.sum()), 0),
                "avg_pnl": round(float(pnl.mean()), 1),
                "pf": round(pf, 3) if pf != float("inf") else 999.0,
                "wr_pct": round(100 * len(wins) / len(grp), 1),
            })

    per_setup_rows = []
    if len(filtered):
        for setup, grp in filtered.groupby("setup_type"):
            pnl = grp["total_trade_pnl"]
            wins = pnl[pnl > 0]
            losses = pnl[pnl < 0].abs()
            pf = float(wins.sum() / losses.sum()) if losses.sum() > 0 else float("inf")
            per_setup_rows.append({
                "setup": setup,
                "n": len(grp),
                "total_pnl": round(float(pnl.sum()), 0),
                "avg_pnl": round(float(pnl.mean()), 1),
                "pf": round(pf, 3) if pf != float("inf") else 999.0,
                "wr_pct": round(100 * len(wins) / len(grp), 1),
            })

    # Write report manually — scenarios don't have PASS/FAIL semantics, so
    # write_stage_report's setup/status columns don't fit. Build markdown directly.
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        f"# Stage 5b — Ruleset Simulation",
        "",
        f"**Purpose:** Apply approved Stage-5 rules as union filter; report aggregate "
        f"PF / WR / session-Sharpe / daily-count to verify ruleset coherence before Validation.",
        "",
        f"**Approved rules simulated:** {len(approved_rules)}",
        "",
        "## Scenarios",
        "",
        _rows_to_markdown(scenarios),
    ]
    report_path.write_text("\n".join(header) + "\n", encoding="utf-8")
    append_section(
        report_path,
        "## Per-hour breakdown (filtered set, entry-time hour_bucket)",
        _rows_to_markdown(per_hour_rows),
    )
    append_section(
        report_path,
        "## Per-setup breakdown (filtered set)",
        _rows_to_markdown(per_setup_rows),
    )

    write_json_artifact(summary_json, {
        "stage": "5b",
        "n_approved_rules": len(approved_rules),
        "scenarios": scenarios,
        "per_hour": per_hour_rows,
        "per_setup": per_setup_rows,
    })

    return {"scenarios": scenarios, "per_hour": per_hour_rows, "per_setup": per_setup_rows}


def _rows_to_markdown(rows: List[Dict[str, Any]]) -> str:
    """Render list of dicts as a markdown table. Column order = dict insertion order."""
    if not rows:
        return "_(no rows)_"
    headers = list(rows[0].keys())
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    return "\n".join(out)
