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

# CNC / delivery fee rates (Zerodha, verified 2026-05-21)
CNC_BROKERAGE_FLAT = 20.0        # Rs per side, flat
CNC_STT_RATE_SELL = 0.001        # 0.1% on sell value (delivery)
CNC_STAMP_RATE_BUY = 0.00015     # 0.015% on buy value (delivery)
CNC_TXN_RATE_PER_SIDE = 0.0000345
CNC_SEBI_RATE_PER_SIDE = 0.000001
CNC_GST_RATE = 0.18              # 18% on (brokerage + txn)

# MTF additional costs on top of base equity fee model (Zerodha rate card, 2026-05-21)
MTF_INTEREST_RATE_PER_DAY = 0.0004   # 0.04% per day on borrowed amount; from T+1
MTF_PLEDGE_FEE_INR_PER_ISIN = 15.0
MTF_UNPLEDGE_FEE_INR = 15.0
MTF_PLEDGE_GST_RATE = 0.18


def calc_fee(entry_price: float, exit_price: float, qty: int, side: str,
             mis_leverage: float = 1.0) -> float:
    """Compute round-trip fees for one Indian intraday equity trade.

    `qty` here is the ACTUAL share count traded (the broker holds exactly
    `qty` shares — MIS leverage only reduces margin requirement, it does NOT
    multiply your position). Fees scale with `qty * price`, never multiplied
    by leverage.

    The `mis_leverage` parameter is retained for backward compat with callers
    that pass it, but it scales qty internally — which is INCORRECT for the
    typical pipeline where `qty` is already actual shares. Leave at default
    1.0 unless you know your `qty` is base/unleveraged and broker holds
    `qty * leverage` (unusual — production capital_manager and trading_logger
    treat qty as actual). See `services/logging/trading_logger.py:63` for
    the authoritative fee model.

    Audit 2026-05-14: confirmed against production analytics.jsonl that
    fees use qty directly with no leverage multiplication. The 2026-05-13
    "fee undercount fix" in comprehensive_run_analyzer was about a separate
    reporting bug — not relevant to sanity/sweep scripts that mirror
    production trade sizing.
    """
    if qty <= 0 or entry_price is None or exit_price is None:
        return 0.0
    if pd.isna(entry_price) or pd.isna(exit_price):
        return 0.0
    lev = max(float(mis_leverage), 1.0)
    qty_actual = int(round(int(qty) * lev))
    entry_to = float(entry_price) * qty_actual
    exit_to = float(exit_price) * qty_actual

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


def calc_fee_cnc(buy_value_inr: float, sell_value_inr: float) -> float:
    """CNC (delivery) fee model for a single round-trip.

    Per-side breakdown (Zerodha rate card, verified 2026-05-21):
      - Brokerage: Rs 20 flat per side
      - STT: 0.10% on SELL value (delivery rate, distinct from intraday 0.025%)
      - Stamp: 0.015% on BUY value (delivery rate, distinct from intraday 0.003%)
      - Txn charges: 0.00345% per side (NSE)
      - SEBI: 0.0001% per side
      - GST: 18% on (brokerage + txn) per side

    Mirrors `tools/sub9_research/sanity_close_dn_overnight_long.py:calc_fee_cnc`
    so research / production reconcile. Pure function; returns Rs as float.
    """
    if buy_value_inr is None or sell_value_inr is None:
        return 0.0
    if pd.isna(buy_value_inr) or pd.isna(sell_value_inr):
        return 0.0
    if buy_value_inr <= 0 or sell_value_inr <= 0:
        return 0.0

    brokerage_buy = CNC_BROKERAGE_FLAT
    brokerage_sell = CNC_BROKERAGE_FLAT
    stt_sell = sell_value_inr * CNC_STT_RATE_SELL
    stamp_buy = buy_value_inr * CNC_STAMP_RATE_BUY
    txn_buy = buy_value_inr * CNC_TXN_RATE_PER_SIDE
    txn_sell = sell_value_inr * CNC_TXN_RATE_PER_SIDE
    sebi_buy = buy_value_inr * CNC_SEBI_RATE_PER_SIDE
    sebi_sell = sell_value_inr * CNC_SEBI_RATE_PER_SIDE
    gst_buy = (brokerage_buy + txn_buy) * CNC_GST_RATE
    gst_sell = (brokerage_sell + txn_sell) * CNC_GST_RATE
    return (
        brokerage_buy + brokerage_sell + stt_sell + stamp_buy +
        txn_buy + txn_sell + sebi_buy + sebi_sell + gst_buy + gst_sell
    )


def calc_fee_mtf(buy_value_inr: float, sell_value_inr: float,
                 margin_inr: float, hold_days: int) -> float:
    """Round-trip MTF fees including overnight interest.

    MTF fee structure on a Rs `buy_value_inr` notional position:
      - All same per-side CNC fees (brokerage, STT, stamp, txn, SEBI, GST)
        because fees scale on NOTIONAL, not on margin.
      - Overnight interest = (buy_value - margin) * 0.0004 * hold_days
        (0.04%/day on borrowed amount; from T+1).
      - Pledge fee: Rs 15 + GST per ISIN per pledge (1 pledge per BUY).
      - Unpledge fee: Rs 15 + GST per request (1 unpledge per SELL).

    For close_dn_overnight_long with 1-night hold (Mon-Thu BUYs): hold_days = 1.
    Friday BUYs exiting Monday: hold_days = 3 (3 calendar days of interest).

    Defensive: any non-positive value/margin returns 0.0 (mirrors calc_fee).
    """
    if buy_value_inr is None or sell_value_inr is None or margin_inr is None:
        return 0.0
    if pd.isna(buy_value_inr) or pd.isna(sell_value_inr) or pd.isna(margin_inr):
        return 0.0
    if buy_value_inr <= 0 or sell_value_inr <= 0 or margin_inr <= 0:
        return 0.0

    # Base equity fees (same as CNC)
    base = calc_fee_cnc(buy_value_inr, sell_value_inr)
    # MTF-specific
    borrowed = max(0.0, buy_value_inr - margin_inr)
    interest = borrowed * MTF_INTEREST_RATE_PER_DAY * max(0, int(hold_days))
    pledge = MTF_PLEDGE_FEE_INR_PER_ISIN * (1 + MTF_PLEDGE_GST_RATE)
    unpledge = MTF_UNPLEDGE_FEE_INR * (1 + MTF_PLEDGE_GST_RATE)
    return base + interest + pledge + unpledge


def calc_fee_by_mode(buy_value_inr: float, sell_value_inr: float, *,
                     mode: str,
                     margin_inr: float | None = None,
                     hold_days: int = 0) -> float:
    """Dispatch to the correct fee model.

    mode in {'delivery_cnc', 'mtf'}.
    For 'intraday_mis', use the legacy qty-based calc_fee() directly.
    """
    if mode == "delivery_cnc":
        return calc_fee_cnc(buy_value_inr, sell_value_inr)
    elif mode == "mtf":
        if margin_inr is None:
            raise ValueError("calc_fee_by_mode(mode='mtf') requires margin_inr")
        return calc_fee_mtf(buy_value_inr, sell_value_inr, margin_inr, hold_days)
    else:
        raise ValueError(f"calc_fee_by_mode: unsupported mode {mode!r} "
                         f"(use 'delivery_cnc' or 'mtf'; intraday_mis uses calc_fee())")


# Sanity bounds for intraday equity trades on NSE small/mid/large caps.
# Justification: NSE circuit-band ceilings are 5/10/20%; in extreme rare cases
# a single intraday move beyond 15% in one direction is plausible (small-cap
# upper-circuit chains) but anything beyond signals a data integrity bug
# (corporate-action mismatch, stale price feed, fat-finger in source CSV,
# split/bonus not adjusted in historical data). Trades crossing these bounds
# are dropped, not clipped — the position size derived from a bad price is
# itself wrong, so the whole row is unreliable.
INTRADAY_PRICE_RATIO_LO = 0.85   # exit cannot be < 85% of entry intraday
INTRADAY_PRICE_RATIO_HI = 1.15   # exit cannot be > 115% of entry intraday
# With Rs 1,000 risk and tiered T1 partial / T2 full exits (max T2 ~2-3R for
# most setups), legitimate winners cluster in [0R, 3R] = [0, Rs 3,000]. Rare
# bar-level runners can extend to 5R = Rs 5,000. ANY trade beyond 10R = Rs 10,000
# is a sizing bug (qty inflation from clamped risk_per_share) or data bug
# (corporate-action mismatch, stale price feed, fat-finger). 10R is the
# ABSOLUTE ceiling for legitimate trades — drop, don't clip.
MAX_PNL_R_MULTIPLE = 10          # |realized_pnl| > 10R = data/sizing bug
RISK_PER_TRADE_RUPEES = 1000     # mirrors config/configuration.json constant


def _drop_bad_priced_trades(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Drop trades with corporate-action/stale-price corruption.

    Returns (cleaned_df, n_dropped_ratio, n_dropped_pnl). Two-stage filter:
      1. exit/entry price ratio outside [0.85, 1.15] — implies the trade
         straddles a corporate action (split/bonus) or stale tick.
      2. |realized_pnl| > Rs 50,000 = 50R given Rs 1,000 risk-per-trade —
         independent floor catching qty-inflation bugs even when prices
         look benign.

    A trade hitting either condition has a sizing or pricing defect and
    must not contribute to PF/Sharpe/Stage 3 cell statistics."""
    n_before = len(df)
    if "entry_price" in df.columns and "e1_price" in df.columns:
        ep = df["entry_price"].astype(float)
        xp = df["e1_price"].astype(float)
        ratio = (xp / ep).where(ep > 0, 1.0)
        bad_ratio = (ratio < INTRADAY_PRICE_RATIO_LO) | (ratio > INTRADAY_PRICE_RATIO_HI)
    else:
        bad_ratio = pd.Series([False] * n_before, index=df.index)

    if "realized_pnl" in df.columns:
        bad_pnl = df["realized_pnl"].astype(float).abs() > MAX_PNL_R_MULTIPLE * RISK_PER_TRADE_RUPEES
    else:
        bad_pnl = pd.Series([False] * n_before, index=df.index)

    n_dropped_ratio = int(bad_ratio.sum())
    n_dropped_pnl = int((bad_pnl & ~bad_ratio).sum())  # avoid double-counting
    cleaned = df[~(bad_ratio | bad_pnl)].copy()
    return cleaned, n_dropped_ratio, n_dropped_pnl


def build_net_per_setup(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to executed trades, compute fee + net PnL per row, return.

    Drops corporate-action/sizing-bug rows via _drop_bad_priced_trades —
    these distort PF/Sharpe and don't represent tradeable outcomes."""
    if df.empty or "executed" not in df.columns:
        return pd.DataFrame()
    mask = df["executed"] == True
    sub = df[mask].copy()
    if sub.empty:
        return sub
    # Sanity-clean before fee math — bad rows shouldn't bias the aggregates.
    sub, n_ratio, n_pnl = _drop_bad_priced_trades(sub)
    if n_ratio + n_pnl > 0:
        # Log dropped count to stderr so the per-session driver can tally.
        import sys as _sys
        print(f"  [sanity-clean] dropped {n_ratio} bad-price-ratio + {n_pnl} bad-PnL rows",
              file=_sys.stderr)
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
        # Select available columns only. Phase-C-Stage4 update: preserve the
        # volatility / momentum features needed to compute Stage 3's
        # `volatility_regime` conditioner (master plan §3.3 — BB-width
        # tercile per stock × 20-day) plus the SHAP-flagged structural
        # drivers (adx5, range_pct, vol_x_*, body_pct).
        desired_cols = ["session_date", "setup_type", "realized_pnl",
                        "fee", "net_pnl", "qty", "entry_price", "e1_price",
                        "side", "decision_ts", "symbol",
                        "regime", "cap_segment", "rank_score",
                        # Structural drivers from Stage 4 SHAP analysis:
                        "bb_width_proxy", "adx5", "atr", "range_pct",
                        "vol_x_median", "vol_x_recent", "body_pct",
                        "minute_of_day", "day_of_week",
                        "first_bar_volume_ratio", "daily_trend_distance_pct"]
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
