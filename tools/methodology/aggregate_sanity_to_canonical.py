"""Aggregate sanity Mode B trade CSVs into canonical-schema trade CSVs.

Mirror of `aggregate_oci_to_canonical.py` but reading sanity outputs.

CAVEAT (Lesson #13): sanity Mode B systematically OVER-ESTIMATES production
PF, especially for SHORT setups using entry_zone retest. Use the resulting
confidence cards as a NEGATIVE filter only:
  - Sanity card RED  -> setup is definitively dead (sanity inflates; production worse)
  - Sanity card GREEN -> CANNOT un-retire; run OCI structure code to verify

CAVEAT (Leverage): sanity CSVs use NAKED qty (un-leveraged). OCI CSVs
reflect production-traded sizes (effectively leveraged). Rs magnitudes are
NOT directly comparable across the two pipelines. However, PF, win-rate,
correlation clustering, and adjusted Sharpe SIGN are all scale-invariant
- the framework verdicts ARE comparable.

Output: per-setup canonical CSV at
    reports/sanity_canonical/<setup>_sanity_canonical.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SANITY_DIR = _REPO_ROOT / "reports" / "sub9_sanity"
_OUT_DIR = _REPO_ROOT / "reports" / "sanity_canonical"


# Setup -> (CSV files to concat in D-OOS-Holdout order, inferred side)
# Side inferred from setup-name convention (short/fade -> SHORT, long -> LONG).
SETUP_CSV_MAP: Dict[str, Dict] = {
    "capitulation_long_v2": {
        "files": [
            "_capitulation_long_v2_trades_discovery.csv",
            "_capitulation_long_v2_trades_oos.csv",
            "_capitulation_long_v2_trades_holdout.csv",
        ],
        "side": "LONG",
    },
    "pre_results_t1_fade_v2": {
        "files": [
            "_pre_results_t1_v2_trades_discovery.csv",
            "_pre_results_t1_v2_trades_oos.csv",
            "_pre_results_t1_v2_trades_holdout.csv",
        ],
        "side": "SHORT",
    },
    "block_deal_accumulation_long": {
        "files": [
            "_block_deal_accumulation_long_trades_discovery.csv",
            "_block_deal_accumulation_long_trades_oos.csv",
            "_block_deal_accumulation_long_trades_holdout.csv",
        ],
        "side": "LONG",
    },
    "volume_spike_reversal_midsession": {
        "files": [
            "_volume_spike_reversal_trades_discovery.csv",
            "_volume_spike_reversal_trades_oos_smallcap.csv",
            "_volume_spike_reversal_trades_holdout_smallcap.csv",
        ],
        "side": "SHORT",
    },
    "expiry_pin_strike_reversal": {
        "files": [
            "_expiry_pin_trades_discovery.csv",  # actually expiry_pin (no leading _)
        ],
        "side": "SHORT",
    },
}


# Some files don't have a leading underscore — handle gracefully
def _resolve_path(filename: str) -> Optional[Path]:
    p1 = _SANITY_DIR / filename
    if p1.exists():
        return p1
    # Try without leading underscore
    alt = filename.lstrip("_")
    p2 = _SANITY_DIR / alt
    if p2.exists():
        return p2
    return None


_DATE_COLS = ("signal_date", "trade_date", "T0_signal_date")
_SIDE_COLS = ("side", "direction")
_SAMEBAR_COLS = ("same_bar", "same_bar_exit")


def _pick_first(df: pd.DataFrame, candidates) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalize_columns(df: pd.DataFrame, setup_name: str, side: str) -> pd.DataFrame:
    """Map sanity columns to canonical schema (tolerant to schema drift across sanity scripts).

    Canonical schema (matches OCI):
      signal_date, symbol, side, entry_price, exit_price, qty, pnl_pct,
      exit_reason, same_bar, realized_pnl_inr, fee_inr, net_pnl_inr,
      setup_type, source
    """
    date_col = _pick_first(df, _DATE_COLS)
    if date_col is None:
        raise KeyError(f"no date column found; tried {_DATE_COLS}, got {list(df.columns)[:8]}...")
    side_col = _pick_first(df, _SIDE_COLS)
    samebar_col = _pick_first(df, _SAMEBAR_COLS)

    out = pd.DataFrame()
    out["signal_date"] = pd.to_datetime(df[date_col]).dt.date
    out["symbol"] = df["symbol"].astype(str)
    if side_col:
        out["side"] = df[side_col].astype(str).str.upper()
    else:
        out["side"] = side
    out["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")
    out["exit_price"] = pd.to_numeric(df["exit_price"], errors="coerce")
    out["qty"] = pd.to_numeric(df["qty"], errors="coerce").astype("Int64")
    out["pnl_pct"] = pd.to_numeric(df.get("pnl_pct", 0.0), errors="coerce")
    out["exit_reason"] = df.get("exit_reason", "unknown").astype(str)
    if samebar_col:
        out["same_bar"] = df[samebar_col].astype(bool)
    else:
        out["same_bar"] = False
    out["realized_pnl_inr"] = pd.to_numeric(df["realized_pnl"], errors="coerce")
    out["fee_inr"] = pd.to_numeric(df["fee"], errors="coerce")
    out["net_pnl_inr"] = pd.to_numeric(df["net_pnl"], errors="coerce")
    out["setup_type"] = setup_name
    out["source"] = "sanity"
    return out


def aggregate_setup(setup_name: str, spec: Dict) -> Optional[pd.DataFrame]:
    parts = []
    side = spec["side"]
    for fname in spec["files"]:
        p = _resolve_path(fname)
        if p is None:
            print(f"  [WARN] {setup_name}: missing {fname}", file=sys.stderr)
            continue
        raw = pd.read_csv(p)
        norm = _normalize_columns(raw, setup_name, side)
        # Drop rows with bad PnL
        norm = norm.dropna(subset=["net_pnl_inr", "signal_date"])
        parts.append(norm)
        print(f"  loaded {fname}: n={len(norm)}")
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values("signal_date").reset_index(drop=True)
    return df


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=_OUT_DIR)
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for setup_name, spec in SETUP_CSV_MAP.items():
        print(f"\n{setup_name}:")
        df = aggregate_setup(setup_name, spec)
        if df is None or len(df) == 0:
            print(f"  SKIP (no data)")
            continue
        out_path = args.out_dir / f"{setup_name}_sanity_canonical.csv"
        df.to_csv(out_path, index=False)
        net = df["net_pnl_inr"].sum()
        print(f"  wrote {out_path.name}: n={len(df)}, net_pnl_inr={net:+,.0f}")

    print(f"\nDone. Canonical CSVs in {args.out_dir}/")


if __name__ == "__main__":
    sys.exit(main() or 0)
