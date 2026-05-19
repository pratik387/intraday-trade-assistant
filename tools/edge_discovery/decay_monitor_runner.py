"""Decay monitor runner — emits rolling PF status per shipped setup.

Inputs: per-setup trade parquet logs (path mapped in config).
Outputs: monthly PF series → compute_status (decay_monitor) → JSON.

All setup paths, column aliases, and the inf-PF display cap come from
config/pipelines/base_config.json -> edge_discovery.decay_monitor_runner.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from services.config_loader import load_base_config
from tools.edge_discovery.decay_monitor import DecayConfig, compute_status


_REPO = Path(__file__).resolve().parents[2]
_REPORT_DIR = _REPO / "reports" / "edge_discovery"


def compute_monthly_pf_from_trades(
    trades: pd.DataFrame,
    pnl_col: str,
    entry_time_col: str = "entry_time",
    inf_display_cap: float = 5.0,
) -> pd.Series:
    """Compute monthly PF from a flat trades DataFrame.

    PF = sum(positive returns) / sum(|negative returns|). When the denominator
    is zero (all-wins month), the result is capped at `inf_display_cap` to
    keep downstream rolling-mean arithmetic well-defined.
    """
    t = trades.copy()
    t[entry_time_col] = pd.to_datetime(t[entry_time_col])
    t["_month"] = t[entry_time_col].dt.to_period("M").dt.to_timestamp()
    out: Dict[pd.Timestamp, float] = {}
    for m, grp in t.groupby("_month"):
        pos = float(grp[grp[pnl_col] > 0][pnl_col].sum())
        neg = float(-grp[grp[pnl_col] < 0][pnl_col].sum())
        if neg > 0:
            pf = pos / neg
        elif pos > 0:
            pf = inf_display_cap
        else:
            pf = 0.0
        out[m] = pf
    return pd.Series(out).sort_index()


def _resolve_column(df: pd.DataFrame, aliases: List[str]) -> str:
    """Return the first alias present in df.columns; rename if needed."""
    for c in aliases:
        if c in df.columns:
            return c
    return ""


def run_decay_monitor_for_all_shipped() -> Dict[str, dict]:
    """Per shipped setup: load trade log, compute monthly PF, emit status."""
    cfg = load_base_config()
    edisc = cfg["edge_discovery"]
    decay_cfg_block = edisc["decay_monitor"]
    runner_cfg = edisc["decay_monitor_runner"]

    dc = DecayConfig(
        rolling_window_months=int(decay_cfg_block["rolling_window_months"]),
        caution_pf_threshold=float(decay_cfg_block["caution_pf_threshold"]),
        pause_pf_threshold=float(decay_cfg_block["pause_pf_threshold"]),
        retire_pf_threshold=float(decay_cfg_block["retire_pf_threshold"]),
        retire_consecutive_months=int(decay_cfg_block["retire_consecutive_months"]),
    )
    entry_aliases = list(runner_cfg["entry_time_column_aliases"])
    pnl_aliases = list(runner_cfg["pnl_column_aliases"])
    inf_display_cap = float(runner_cfg["inf_display_cap"])
    setup_to_pq_rel: Dict[str, str] = dict(runner_cfg["setup_trade_parquets"])

    results: Dict[str, dict] = {}
    for setup, rel_path in setup_to_pq_rel.items():
        pq = _REPO / rel_path
        if not pq.exists():
            print(f"[decay] SKIP {setup}: {rel_path} not found")
            continue
        df = pd.read_parquet(pq)
        entry_col = _resolve_column(df, entry_aliases)
        if not entry_col:
            print(f"[decay] SKIP {setup}: no entry_time column among {entry_aliases}")
            continue
        pnl_col = _resolve_column(df, pnl_aliases)
        if not pnl_col:
            print(f"[decay] SKIP {setup}: no return column among {pnl_aliases}")
            continue
        monthly_pf = compute_monthly_pf_from_trades(
            df, pnl_col=pnl_col, entry_time_col=entry_col,
            inf_display_cap=inf_display_cap,
        )
        status = compute_status(monthly_pf, dc)
        results[setup] = {
            "status": status.status,
            "rolling_pf": status.rolling_pf,
            "latest_month_pf": status.latest_month_pf,
            "consecutive_retire_months": status.consecutive_retire_months,
            "notes": status.notes,
            "monthly_pf": {ts.isoformat(): pf for ts, pf in monthly_pf.items()},
        }
        print(
            f"[decay] {setup}: {status.status} "
            f"(rolling_pf={status.rolling_pf:.2f}, latest={status.latest_month_pf:.2f})"
        )

    out_path = _REPORT_DIR / "decay_monitor.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    return results


if __name__ == "__main__":
    run_decay_monitor_for_all_shipped()
