"""Production-aligned sanity for below_vwap_volume_revert_long.

Re-computes the sanity PF using the PRODUCTION universe filter (per-date,
matches consolidated_daily.feather based gate) instead of the existing
sanity's window-level coverage filter.

Findings ahead of running this script:
  - 260 OCI HO fires (Dec 2025-Apr 2026) ALL pass sanity's per-bar logic
    (verified empirically: 260/260 WOULD_FIRE under sanity-compute)
  - So the divergence is ENTIRELY at the universe-gate
  - This script: post-filter the existing sanity trade CSV using
    ProductionUniverseGate, re-compute PF, head-to-head against OCI

To produce a TRUE production-aligned sanity that also INCLUDES newly-tracked
symbols (currently dropped by the existing sanity's universe gate), the
underlying sanity loop must be rerun. That requires more invasive changes;
do it only if the post-filter check still leaves a gap.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from tools.sub9_research.production_universe import (
    ProductionUniverseGate,
    _normalize_symbol,
)


def _pf(s: pd.Series) -> float:
    g = float(s[s > 0].sum())
    l = float(-s[s < 0].sum())
    return (g / l) if l > 0 else float("inf")


def compute_oci_pf() -> dict:
    """Aggregate OCI HO trades (Dec 2025 - Apr 2026) from the two HO OCI runs."""
    runs_ho = ["20260521-214422_full", "20260521-163406_full"]
    rows = []
    for run_dir in runs_ho:
        p = _REPO_ROOT / run_dir
        if not p.exists():
            continue
        for date_dir in sorted(p.iterdir()):
            if not date_dir.is_dir():
                continue
            tr = date_dir / "trade_report.csv"
            if not tr.exists():
                continue
            try:
                df = pd.read_csv(tr)
                if "setup_type" in df.columns:
                    bv = df[df["setup_type"] == "below_vwap_volume_revert_long"]
                    for _, r in bv.iterrows():
                        pnl = pd.to_numeric(r.get("realized_pnl"), errors="coerce")
                        if pd.notna(pnl):
                            rows.append({
                                "session_date": date_dir.name,
                                "symbol": r.get("symbol"),
                                "realized_pnl": float(pnl),
                            })
            except Exception:
                pass
    oci_df = pd.DataFrame(rows)
    if oci_df.empty:
        return {"n": 0}
    s = oci_df["realized_pnl"]
    return {
        "n": len(oci_df),
        "pf": _pf(s),
        "wr": float((s > 0).mean() * 100),
        "mean": float(s.mean()),
        "net": float(s.sum()),
    }


def realign_sanity(
    sanity_csv: Path,
    window_start: date,
    window_end: date,
    cell_filter: dict,
) -> dict:
    """Apply ProductionUniverseGate to the existing sanity trades + re-compute PF."""
    df = pd.read_csv(sanity_csv)
    df["signal_date"] = pd.to_datetime(df["signal_date"]).dt.date
    df = df[(df["signal_date"] >= window_start) & (df["signal_date"] <= window_end)]
    for k, v in cell_filter.items():
        df = df[df[k] == v]
    df["bare"] = df["symbol"].apply(_normalize_symbol)

    gate = ProductionUniverseGate(
        accepted_caps={"unknown"},
        require_mis=True,
        require_mtf=False,
        min_trading_days_required=0,  # zeroed per Lesson #17
        min_daily_avg_volume=0,        # zeroed per Lesson #17
        exclude_etf=False,
    )

    print(f"Filtering {len(df)} sanity trades through ProductionUniverseGate...")
    accepted_mask = []
    for _, row in df.iterrows():
        accepted_mask.append(gate.is_eligible(row["bare"], row["signal_date"]))
    aligned_df = df[accepted_mask]
    rejected_df = df[[not x for x in accepted_mask]]

    s_orig = df["net_pnl_inr"]
    s_aligned = aligned_df["net_pnl_inr"]
    s_rejected = rejected_df["net_pnl_inr"]
    return {
        "original": {
            "n": len(df), "pf": _pf(s_orig),
            "wr": float((s_orig > 0).mean() * 100),
            "mean": float(s_orig.mean()),
            "net": float(s_orig.sum()),
        },
        "aligned": {
            "n": len(aligned_df), "pf": _pf(s_aligned),
            "wr": float((s_aligned > 0).mean() * 100) if len(aligned_df) else 0,
            "mean": float(s_aligned.mean()) if len(aligned_df) else 0,
            "net": float(s_aligned.sum()),
        },
        "rejected": {
            "n": len(rejected_df), "pf": _pf(s_rejected),
            "wr": float((s_rejected > 0).mean() * 100) if len(rejected_df) else 0,
            "mean": float(s_rejected.mean()) if len(rejected_df) else 0,
            "net": float(s_rejected.sum()),
        },
    }


def main():
    sanity_csv = _REPO_ROOT / "reports" / "sub9_sanity" / "_below_vwap_volume_revert_long_trades_holdout.csv"

    print("=" * 70)
    print("below_vwap_volume_revert_long — head-to-head: sanity vs production")
    print("=" * 70)
    print(f"Window: HO Dec 2025 - Apr 2026 (matches OCI run coverage)")
    print(f"Cell: cap_segment=unknown × vol_ratio_bin=gte_10 × hhmm_bucket=afternoon_1300_1500")
    print()

    san = realign_sanity(
        sanity_csv,
        window_start=date(2025, 12, 8),
        window_end=date(2026, 4, 30),
        cell_filter={
            "cap_segment": "unknown",
            "vol_ratio_bin": "gte_10",
            "hhmm_bucket": "afternoon_1300_1500",
        },
    )

    oci = compute_oci_pf()

    print()
    print(f"{'state':<35} {'n':>5} {'PF':>6} {'WR%':>5} {'mean':>9} {'NET':>11}")
    print("-" * 72)

    def _row(label, d):
        if d.get("n", 0) == 0:
            print(f"{label:<35} {0:>5}")
            return
        mean_s = f"Rs{d['mean']:+,.0f}"
        net_s = f"Rs{d['net']:+,.0f}"
        print(f"{label:<35} {d['n']:>5} {d['pf']:>6.3f} {d['wr']:>5.1f} {mean_s:>9} {net_s:>11}")

    _row("Sanity ORIGINAL (cell+window)", san["original"])
    _row("Sanity post-prod-gate (aligned)", san["aligned"])
    _row("Rejected by prod gate", san["rejected"])
    _row("OCI HO actual (ground truth)", oci)

    print()
    print("Interpretation:")
    print("- 'Sanity post-prod-gate' = sanity trades whose symbols PASS production universe filter")
    print("- 'OCI HO actual' = ground truth — what production actually fires")
    print()
    if oci.get("n"):
        d_san = san["aligned"]["pf"]
        d_oci = oci["pf"]
        print(f"Sanity-aligned PF: {d_san:.3f}")
        print(f"OCI actual PF:     {d_oci:.3f}")
        print(f"Gap:               {d_san - d_oci:+.3f}")
        print()
        print("NOTE: A residual gap can exist because sanity only retains trades that")
        print("originally PASSED sanity's universe filter (window-level coverage). Trades")
        print("on NEWLY-TRACKED symbols (5m archive only Dec 2025+) are MISSING from the")
        print("sanity CSV entirely; this script CAN'T recover them via post-filter. To")
        print("include them, the upstream sanity loop must be re-run without the")
        print("window-level coverage filter.")


if __name__ == "__main__":
    main()
