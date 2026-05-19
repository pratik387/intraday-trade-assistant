"""Parity gate Target: gap_fade_short reproduction on Holdout window.

The live setup's Holdout statistics (PF=1.36, WR=70%, N=797) are captured in
config/configuration.json (setups.gap_fade_short._live_status). This script
loads reports/sub8_oos_holdout_clean/gap_fade_short.parquet and compares.

Hard gate: framework cannot be used to retire/ship anything until this passes.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from services.config_loader import load_base_config
from tools.edge_discovery.validation.parity_gate import compare_parity, ParityTolerance


_REPO = Path(__file__).resolve().parents[3]


def _load_live_baseline(setup_name: str) -> dict:
    """Parse PF/WR/N from setups.<name>._live_status text in configuration.json.

    setups live in config/configuration.json (NOT base_config.json — different file).
    """
    cfg_path = _REPO / "config" / "configuration.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    setup = cfg["setups"][setup_name]
    live_str = setup.get("_live_status", "")
    pf = float(re.search(r"PF=([0-9.]+)", live_str).group(1))
    wr = float(re.search(r"WR=([0-9.]+)", live_str).group(1)) / 100.0
    n = int(re.search(r"N=(\d+)", live_str).group(1))
    return {"pf": pf, "wr": wr, "n": n}


def _compute_framework_stats(parquet_path: Path, cap_segment: Optional[str] = None) -> dict:
    """Compute PF/WR/N from a live setup's trade parquet.

    Optionally filter by cap_segment to match production setup restrictions.
    """
    df = pd.read_parquet(parquet_path)
    if cap_segment is not None and "cap_segment" in df.columns:
        df = df[df["cap_segment"] == cap_segment]
    pnl_col = None
    for candidate in ("net_pnl_inr", "net_pnl", "pnl_net", "pnl", "net_return"):
        if candidate in df.columns:
            pnl_col = candidate
            break
    if pnl_col is None:
        raise KeyError(f"No PnL column found in {parquet_path} (cols: {df.columns.tolist()})")
    n = int(len(df))
    pos = df[df[pnl_col] > 0][pnl_col].sum()
    neg = -df[df[pnl_col] < 0][pnl_col].sum()
    pf = float(pos / neg) if neg > 0 else float("inf")
    wr = float((df[pnl_col] > 0).mean())
    return {"pf": pf, "wr": wr, "n": n}


def run_parity_gap_fade() -> dict:
    """Note: reads parity tolerance from edge_discovery config block in base_config.json."""
    cfg = load_base_config()
    pq_path = _REPO / "reports" / "sub8_oos_holdout_clean" / "gap_fade_short.parquet"
    if not pq_path.exists():
        pq_path = _REPO / "reports" / "sub8_oos_holdout" / "gap_fade_short.parquet"
    if not pq_path.exists():
        raise FileNotFoundError(f"gap_fade_short parquet not found at {pq_path}")
    framework = _compute_framework_stats(pq_path, cap_segment="small_cap")
    live = _load_live_baseline("gap_fade_short")
    tol_cfg = cfg["edge_discovery"]["parity_gate"]
    tol = ParityTolerance(
        pf_pct=float(tol_cfg["pf_tolerance_pct"]),
        wr_pp=float(tol_cfg["wr_tolerance_pp"]),
        n_pct=float(tol_cfg["n_tolerance_pct"]),
    )
    verdict = compare_parity(framework, live, tol)
    out = {
        "setup": "gap_fade_short",
        "live": live,
        "framework": framework,
        "verdict": {
            "passed": verdict.passed,
            "failures": verdict.failures,
            "pf_delta_pct": verdict.pf_delta_pct,
            "wr_delta_pp": verdict.wr_delta_pp,
            "n_delta_pct": verdict.n_delta_pct,
        },
    }
    print(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    run_parity_gap_fade()
