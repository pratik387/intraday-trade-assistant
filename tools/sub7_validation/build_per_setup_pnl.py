"""Build per-setup net PnL from trade_report.csv files (sub7-T9).

For each session's trade_report.csv, applies Indian intraday fee schedule
and groups by setup_type. Writes one parquet per setup with NET PnL.

CLI:
    python tools/sub7_validation/build_per_setup_pnl.py \\
        --oci-dir <path-to-OCI-output> \\
        --output-dir reports/sub7_validation/
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

# Indian intraday fee schedule (services/logging/trading_logger.py)
BROK_RATE = 0.0003
BROK_CAP = 20.0
STT_RATE = 0.00025
EXCH_RATE = 0.0000297
SEBI_RATE = 0.000001
IPFT_RATE = 0.000001
STAMP_RATE = 0.00003
GST_RATE = 0.18


def calc_fee(entry_price: float, exit_price: float, qty: int, side: str) -> float:
    """Compute round-trip fees for one Indian intraday equity trade."""
    if qty <= 0 or entry_price is None or exit_price is None:
        return 0.0
    if pd.isna(entry_price) or pd.isna(exit_price):
        return 0.0
    entry_to = float(entry_price) * int(qty)
    exit_to = float(exit_price) * int(qty)

    eb = min(BROK_RATE * entry_to, BROK_CAP)
    xb = min(BROK_RATE * exit_to, BROK_CAP)
    brok = eb + xb

    if side == "BUY":
        stt = exit_to * STT_RATE
        stamp = entry_to * STAMP_RATE
    else:
        stt = entry_to * STT_RATE
        stamp = exit_to * STAMP_RATE

    leg = entry_to + exit_to
    exch = leg * EXCH_RATE
    sebi = leg * SEBI_RATE
    ipft = leg * IPFT_RATE
    gst = (brok + exch + sebi + ipft) * GST_RATE

    return brok + stt + exch + sebi + ipft + stamp + gst


def build_net_per_setup(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to executed trades, compute fee + net PnL per row, return."""
    if df.empty or "executed" not in df.columns:
        return pd.DataFrame()
    mask = df["executed"] == True
    sub = df[mask].copy()
    if sub.empty:
        return sub
    sub["fee"] = sub.apply(
        lambda r: calc_fee(r.get("entry_price"), r.get("e1_price"),
                           int(r.get("qty", 0) or 0), r.get("side", "")),
        axis=1,
    )
    sub["net_pnl"] = sub["realized_pnl"].astype(float) - sub["fee"]
    return sub


def aggregate_oci_dir(oci_dir: Path) -> pd.DataFrame:
    """Walk OCI dir, load all trade_reports, return aggregated net DataFrame."""
    parts = []
    for f in sorted(glob.glob(f"{oci_dir}/*/trade_report.csv")):
        sess = Path(f).parent.name
        df = pd.read_csv(f, low_memory=False)
        if "realized_pnl" not in df.columns:
            continue
        sub = build_net_per_setup(df)
        if sub.empty:
            continue
        sub["session_date"] = sess
        # Select available columns only
        desired_cols = ["session_date", "setup_type", "realized_pnl",
                        "fee", "net_pnl", "qty", "entry_price", "e1_price",
                        "side", "decision_ts", "symbol",
                        "regime", "cap_segment", "rank_score"]
        available_cols = [c for c in desired_cols if c in sub.columns]
        parts.append(sub[available_cols])
    if not parts:
        raise SystemExit(f"[build_per_setup_pnl] no trade_reports under {oci_dir}")
    return pd.concat(parts, ignore_index=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oci-dir", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    oci_dir = Path(args.oci_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    big = aggregate_oci_dir(oci_dir)
    print(f"Loaded {len(big):,} executed trades from {oci_dir}")
    print(f"Setups present: {sorted(big['setup_type'].unique())}")
    for setup, grp in big.groupby("setup_type"):
        out_path = out_dir / f"{setup}.parquet"
        grp.to_parquet(out_path, index=False)
        print(f"  {setup}: {len(grp)} trades  net=Rs {int(grp['net_pnl'].sum()):,}  -> {out_path}")


if __name__ == "__main__":
    main()
