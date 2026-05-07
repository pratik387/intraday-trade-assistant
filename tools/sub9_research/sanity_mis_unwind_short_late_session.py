"""Pre-coding sanity check for mis_unwind_short_late_session candidate.

Per sub-9 §3.3 brief gate (specs/2026-05-07-sub-project-9-brief-mis_unwind
_short_late_session.md): BEFORE writing detector code, simulate the late-
session retail MIS-long forced-unwind front-run on 24 monthly 2023-2024
5m enriched feathers.

Decision criterion (locked):
  PF >= 1.10  -> STRONG PROCEED
  1.0-1.10    -> marginal, proceed with caveat
  PF < 1.0    -> RETIRE candidate, do NOT write detector

Falsification gate (locked, brief Risks/falsification section):
  - WR < 20%  -> ABANDON (CNC-conversion squeeze still dominates)
  - n < 30    -> ABANDON (sample too thin)
  - Symbol overlap with prior failed mis_unwind_short > 60% -> ABANDON
    (the new mechanic has not actually changed the targeted population)

Mechanic — designed against the failure mode of the prior `mis_unwind_short`
(net PF 0.355, n=304, WR 9.2%):

  Universe: F&O 200 (assets/fno_liquid_200.csv), cap_segment in
            {mid_cap, small_cap}.

  At each 14:30, 14:35, 14:40, 14:45, 14:50, 14:55, 15:00 5m bar close,
  qualify symbol if ALL hold:
    - intraday return at bar close in [+1.5%, +4.0%]
        (heavy intraday accumulation, but not runaway — runaway candidates
         get CNC-converted to delivery and the unwind never fires)
    - bar close is 0.3-1.5% off intraday-high
        (off-the-high zone: mid-conviction holders, less likely to convert
         MIS->CNC; the prior detector targeted FRESH-high stocks, which
         attracted high-conviction CNC-conversion and got squeezed)
    - ret_3 (3-bar return) at bar close in [0.0%, +0.5%]
    - ret(bar) - ret(13:30) <= 0  (no fresh acceleration in last hour)
    - intraday cumulative volume rank >= 70th pct in qualifying universe
    - avg vol in last 3 bars (14:15-14:30 if firing at 14:30) <
        avg vol in 13:00-14:00 (accumulation exhausting)

  Entry: SHORT at the qualifying bar's close. Latch per (symbol, day):
         only the FIRST qualifying bar fires.
  Hard SL: max(intraday-high, entry * 1.012).
  T1: entry * 0.996 (~0.4% below entry) — partial 50%, move stop to
      entry (breakeven trail).
  T2 / final: VWAP if reached by 15:10.
  HARD time-stop: 15:10 forced exit at bar close, regardless of P&L
                  (5 min before broker auto-square 15:15).

Symbol overlap diagnostic vs prior failed setup:
  reports/sub8_phase1/mis_unwind_short.parquet has 304 trades, 185
  unique symbols. We compute the overlap of THIS sanity's traded
  symbols (set) vs that prior set, expressed as % of THIS sanity's
  set that was in the prior set. Brief gate: overlap < 60%.

Indian fee model: tools/sub7_validation/build_per_setup_pnl.calc_fee
(Zerodha intraday-equity rate card + STT/exch/SEBI/IPFT/stamp/GST).

Discovery period: 2023-01-01 -> 2024-12-31 (24 monthly feathers).

Usage:
    python tools/sub9_research/sanity_mis_unwind_short_late_session.py
"""
from __future__ import annotations

import sys
from datetime import date, time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from services.symbol_metadata import get_cap_segment           # noqa: E402
from tools.sub7_validation.build_per_setup_pnl import calc_fee  # noqa: E402


# ---- Locked params (per brief, not user-tunable for sanity) -------------
ALLOWED_CAPS = {"large_cap", "mid_cap", "small_cap"}  # F&O 200 is large_cap-dominant
# (143 large / 4 mid / 0 small per nse_all.json sanity); brief's mid+small-only
# restriction was a structural typo — leaves only 4 testable symbols. Same fix
# applied to options_vol_iv_rank_revert and earnings_day_intraday_fade earlier.

# Entry-window bars (5m bar closes within [14:30, 15:00] inclusive).
ENTRY_HHMM_LIST = ["14:30", "14:35", "14:40", "14:45", "14:50", "14:55", "15:00"]

# Reference HH:MM strings for derived signals.
HHMM_OPEN = "09:15"
HHMM_1330 = "13:30"
HHMM_1300 = "13:00"
HHMM_1400 = "14:00"

# Intraday-return gate at entry bar close.
INTRADAY_RET_LO_PCT = 1.5    # heavy accumulation floor
INTRADAY_RET_HI_PCT = 4.0    # exclude runaway (CNC-conversion magnets)

# Off-the-high gate (close vs intraday-high, % below high).
OFF_HIGH_LO_PCT = 0.3        # at-the-high excluded (high-conviction zone)
OFF_HIGH_HI_PCT = 1.5        # too far below excluded (already broken)

# 3-bar return at entry bar (mild positive only — late-comers still buying).
RET3_LO_PCT = 0.0
RET3_HI_PCT = 0.5

# Cumulative volume rank percentile inside qualifying universe (per session).
VOL_RANK_PCT_MIN = 70.0      # heavy intraday inventory accumulation

# Stop / target (locked per brief).
STOP_MIN_PCT = 1.2           # entry * 1.012 floor
T1_PCT = 0.4                 # entry * 0.996 — first target (50% partial, breakeven trail after)

# Hard time-stop bar (forced exit regardless of P&L).
HARD_TIMESTOP_HHMM = "15:10"

# Risk per trade — match other sub9 sanity scripts for comparable PnL units.
RISK_PER_TRADE_RUPEES = 1000

# Liquidity floor (matches other sub9 sanity scripts).
MIN_ADV_INR_CR = 3.0

# Prior failed mis_unwind_short trade parquet (for symbol-overlap diagnostic).
PRIOR_FAILED_TRADES_PARQUET = (
    _REPO_ROOT / "reports" / "sub8_phase1" / "mis_unwind_short.parquet"
)
SYMBOL_OVERLAP_GATE_PCT = 60.0   # > 60% means same population -> ABANDON


def _load_5m_for_month(yyyy: int, mm: int) -> pd.DataFrame:
    path = (
        _REPO_ROOT / "backtest-cache-download" / "monthly"
        / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather"
    )
    if not path.exists():
        return pd.DataFrame()
    return pd.read_feather(path)


def build_full_period_5m(universe: set, allowed_caps: set) -> pd.DataFrame:
    """Stream-load 24 monthly feathers with per-month F&O + cap filtering.

    Filtering happens INSIDE the per-month loop to keep peak memory low.
    A pre-filter to {universe x cap_segments} drops ~85% of bars before
    the cross-month concat/sort.
    """
    print("  loading 24 monthly 5m feathers (2023-01 .. 2024-12) ...")
    parts: List[pd.DataFrame] = []
    total_raw = 0
    for yyyy in (2023, 2024):
        for m in range(1, 13):
            mdf = _load_5m_for_month(yyyy, m)
            if mdf.empty:
                continue
            total_raw += len(mdf)
            # filter to F&O universe immediately (drops ~75% of bars)
            mdf = mdf[mdf["symbol"].isin(universe)]
            if mdf.empty:
                continue
            parts.append(mdf)
    if not parts:
        return pd.DataFrame()
    big = pd.concat(parts, ignore_index=True)
    print(f"  raw bars across 24 months: {total_raw:,}")
    print(f"  after F&O universe filter: {len(big):,}")

    # Cap-segment filter (per-symbol, can run on the smaller set)
    big["nse_symbol"] = "NSE:" + big["symbol"].astype(str)
    sym_to_cap = {
        s: get_cap_segment(s) for s in big["nse_symbol"].unique()
    }
    big["cap_segment"] = big["nse_symbol"].map(sym_to_cap)
    big = big[big["cap_segment"].isin(allowed_caps)].copy()
    print(f"  after cap_segment in {sorted(allowed_caps)}: {len(big):,}")

    big = big.sort_values(["symbol", "date"]).reset_index(drop=True)
    big["d"] = big["date"].dt.date
    big["hhmm"] = big["date"].dt.strftime("%H:%M")
    return big


def load_fno_universe() -> set:
    path = _REPO_ROOT / "assets" / "fno_liquid_200.csv"
    df = pd.read_csv(path)
    syms = df["symbol"].astype(str).str.replace("NSE:", "", regex=False).tolist()
    print(f"  F&O 200 universe: {len(syms)} symbols")
    return set(syms)


def load_daily_for_liquidity() -> pd.DataFrame:
    daily_path = _REPO_ROOT / "cache" / "preaggregate" / "consolidated_daily.feather"
    if not daily_path.exists():
        raise FileNotFoundError(f"{daily_path} missing")
    df = pd.read_feather(daily_path)
    df["ts"] = pd.to_datetime(df["ts"])
    if df["ts"].dt.tz is not None:
        df["ts"] = df["ts"].dt.tz_localize(None)
    df["d"] = df["ts"].dt.date
    df = df[(df["d"] >= date(2022, 11, 1)) & (df["d"] <= date(2024, 12, 31))]
    df = df[["symbol", "d", "close", "volume"]].copy()
    df["traded_value"] = df["close"] * df["volume"]
    df = df.sort_values(["symbol", "d"])
    df["adv_20d_cr"] = df.groupby("symbol")["traded_value"].transform(
        lambda v: v.shift(1).rolling(20).mean()
    ) / 1e7
    return df[["symbol", "d", "adv_20d_cr"]]


def load_prior_symbols() -> Optional[set]:
    """Load the unique symbol set traded by the PRIOR failed mis_unwind_short.

    Used for the §3.3 falsification overlap gate (< 60%).
    """
    if not PRIOR_FAILED_TRADES_PARQUET.exists():
        print(
            f"  WARNING: prior trade parquet not found at "
            f"{PRIOR_FAILED_TRADES_PARQUET} -- symbol-overlap diagnostic "
            "will be flagged as MANUAL CHECK."
        )
        return None
    df = pd.read_parquet(PRIOR_FAILED_TRADES_PARQUET)
    if "symbol" not in df.columns:
        print(
            f"  WARNING: 'symbol' column missing in {PRIOR_FAILED_TRADES_PARQUET} -- "
            "symbol-overlap diagnostic will be flagged as MANUAL CHECK."
        )
        return None
    syms = set(
        df["symbol"].astype(str).str.replace("NSE:", "", regex=False).unique().tolist()
    )
    print(
        f"  prior mis_unwind_short symbol set loaded: {len(syms)} unique "
        f"symbols ({len(df)} prior trades)"
    )
    return syms


# ---- Per-session feature engineering -------------------------------------

def _enrich_session_features(day_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-bar intraday features for one (symbol, session).

    Adds: open_anchor_price, intraday_high_so_far, intraday_ret_pct,
          off_high_pct, ret_3_pct, ret_at_1330_pct, cum_vol_so_far.
    Returns the input df with new columns. Caller is responsible for
    only using rows whose hhmm is in the entry window.
    """
    df = day_df.sort_values("date").reset_index(drop=True).copy()
    if df.empty:
        return df
    # Anchor = open of the session's first bar (typically 09:15).
    open_anchor = float(df.iloc[0]["open"])
    df["open_anchor"] = open_anchor

    # Running intraday high (from session start through current bar inclusive).
    df["intraday_high_so_far"] = df["high"].cummax()

    # Intraday return: close / open_anchor - 1 (in %).
    df["intraday_ret_pct"] = (df["close"] / open_anchor - 1.0) * 100.0

    # Off-the-high (% the close is BELOW the running intraday high).
    df["off_high_pct"] = (
        (df["intraday_high_so_far"] - df["close"]) / df["intraday_high_so_far"]
    ) * 100.0

    # 3-bar return: close vs close 3 bars ago (in %).
    df["close_3back"] = df["close"].shift(3)
    df["ret_3_pct"] = (df["close"] / df["close_3back"] - 1.0) * 100.0

    # Cumulative volume so far in the session.
    df["cum_vol"] = df["volume"].cumsum()

    # Intraday return at 13:30 (carry the 13:30 bar's intraday_ret_pct
    # forward across all subsequent bars). If 13:30 missing, NaN.
    if (df["hhmm"] == HHMM_1330).any():
        ret_at_1330 = float(
            df.loc[df["hhmm"] == HHMM_1330, "intraday_ret_pct"].iloc[0]
        )
    else:
        ret_at_1330 = np.nan
    df["ret_at_1330_pct"] = ret_at_1330

    # Volume sums for accumulation-exhausting check:
    #   avg_vol_13_14: mean volume of bars in [13:00, 14:00) — 12 bars max
    #   avg_vol_last3 (computed at entry bar): mean of last 3 bars (incl current)
    mask_13_14 = (df["hhmm"] >= HHMM_1300) & (df["hhmm"] < HHMM_1400)
    if mask_13_14.any():
        avg_13_14 = float(df.loc[mask_13_14, "volume"].mean())
    else:
        avg_13_14 = np.nan
    df["avg_vol_13_14"] = avg_13_14

    # 3-bar volume average ending at current bar (inclusive).
    df["avg_vol_last3"] = df["volume"].rolling(3, min_periods=3).mean()

    return df


# ---- Trigger discovery ---------------------------------------------------

def find_triggers(
    big5m: pd.DataFrame,
    universe: set,
    adv_table: pd.DataFrame,
) -> pd.DataFrame:
    """Per-session, per-symbol: enrich features, then for each entry-window
    bar, pre-compute eligibility flags (everything except the per-session
    cum-vol percentile rank, which requires cross-symbol comparison and is
    applied AFTER this filter).
    """
    # F&O + cap_segment filtering already applied in build_full_period_5m
    df = big5m.copy()
    print(f"    pre-filtered 5m bars: {len(df):,}")

    adv_idx = adv_table.set_index(["symbol", "d"])["adv_20d_cr"]
    df["adv_20d_cr"] = df.set_index(["symbol", "d"]).index.map(adv_idx).values
    df = df[df["adv_20d_cr"] >= MIN_ADV_INR_CR].copy()
    print(f"    adv_20d >= Rs {MIN_ADV_INR_CR}Cr: {len(df):,}")

    print("  enriching per-(symbol, session) intraday features ...")
    enriched_parts: List[pd.DataFrame] = []
    for (_sym, _d), grp in df.groupby(["symbol", "d"], sort=False):
        enriched_parts.append(_enrich_session_features(grp))
    enriched = pd.concat(enriched_parts, ignore_index=True)

    # Restrict to entry-window bars.
    enriched = enriched[enriched["hhmm"].isin(ENTRY_HHMM_LIST)].copy()
    print(f"    rows in entry window [14:30..15:00]: {len(enriched):,}")

    # Per-bar candidate flags (everything EXCEPT cross-symbol cum-vol rank).
    cond_intraday_ret = (
        (enriched["intraday_ret_pct"] >= INTRADAY_RET_LO_PCT)
        & (enriched["intraday_ret_pct"] <= INTRADAY_RET_HI_PCT)
    )
    cond_off_high = (
        (enriched["off_high_pct"] >= OFF_HIGH_LO_PCT)
        & (enriched["off_high_pct"] <= OFF_HIGH_HI_PCT)
    )
    cond_ret3 = (
        (enriched["ret_3_pct"] >= RET3_LO_PCT)
        & (enriched["ret_3_pct"] <= RET3_HI_PCT)
    )
    cond_no_fresh_accel = (
        (enriched["intraday_ret_pct"] - enriched["ret_at_1330_pct"]) <= 0.0
    )
    cond_vol_exhaust = enriched["avg_vol_last3"] < enriched["avg_vol_13_14"]

    pre_qual = (
        cond_intraday_ret
        & cond_off_high
        & cond_ret3
        & cond_no_fresh_accel
        & cond_vol_exhaust
    )
    enriched = enriched[pre_qual].copy()
    print(f"    rows passing 5/6 gates (excl. vol-rank): {len(enriched):,}")

    # Cross-symbol cum-vol rank: per (session, hhmm) within the qualifying
    # universe at THAT bar, compute percentile rank of cum_vol. Symbols at
    # >= 70th pct keep going.
    enriched["cum_vol_pct_rank"] = enriched.groupby(["d", "hhmm"])["cum_vol"].rank(
        pct=True, method="average"
    ) * 100.0
    enriched = enriched[enriched["cum_vol_pct_rank"] >= VOL_RANK_PCT_MIN].copy()
    print(f"    rows passing cum-vol rank >= {VOL_RANK_PCT_MIN}th pct: {len(enriched):,}")

    # Latch per (symbol, day): keep only the EARLIEST qualifying bar.
    enriched = enriched.sort_values(["symbol", "d", "date"])
    enriched = enriched.drop_duplicates(subset=["symbol", "d"], keep="first")
    print(f"    unique (symbol, day) trigger latches: {len(enriched):,}")

    return enriched.reset_index(drop=True)


# ---- Trade simulation ----------------------------------------------------

def simulate(
    triggers: pd.DataFrame,
    big5m: pd.DataFrame,
) -> pd.DataFrame:
    print("  simulating SHORT entries -> T1/SL/VWAP/time-stop ...")
    days_per_sym: Dict[str, pd.DataFrame] = {
        sym: g.sort_values("date").reset_index(drop=True)
        for sym, g in big5m.groupby("symbol")
    }

    trades: List[dict] = []
    n_no_session = n_no_entry_idx = n_traded = 0

    for _, t in triggers.iterrows():
        sym = t["symbol"]
        sd = t["d"]
        trig_ts = t["date"]

        sym_df = days_per_sym.get(sym)
        if sym_df is None:
            n_no_session += 1
            continue
        day_df = sym_df[sym_df["d"] == sd].reset_index(drop=True)
        if day_df.empty:
            n_no_session += 1
            continue

        idx_arr = day_df.index[day_df["date"] == trig_ts].tolist()
        if not idx_arr:
            n_no_entry_idx += 1
            continue
        entry_idx = idx_arr[0]
        entry_bar = day_df.iloc[entry_idx]

        # Entry: SHORT at the qualifying bar's CLOSE. (Intra-bar fills are
        # ambiguous for sanity; close-of-bar is the standard convention used
        # by other sub9 sanity scripts when the trigger is also the entry.)
        entry_price = float(entry_bar["close"])
        entry_ts = entry_bar["date"]

        # Stop = max(intraday-high so far, entry * (1 + STOP_MIN_PCT/100)).
        # intraday_high_so_far comes from the enriched trigger row.
        intra_high_at_entry = float(t["intraday_high_so_far"])
        sl_struct = intra_high_at_entry
        sl_min = entry_price * (1.0 + STOP_MIN_PCT / 100.0)
        hard_sl = max(sl_struct, sl_min)
        stop_distance = hard_sl - entry_price
        if stop_distance <= 0:
            continue

        # T1 (first target — 50% partial, breakeven trail after).
        t1_target = entry_price * (1.0 - T1_PCT / 100.0)

        # Forward bars: entry bar's CLOSE is the fill, so subsequent bars
        # (entry_idx + 1 onwards) are where SL/T1/T2/time-stop fire.
        forward = day_df.iloc[entry_idx + 1:].copy()
        if forward.empty:
            # No bars after entry — close out at entry (no PnL move possible).
            continue

        exit_ts = None
        exit_price = None
        exit_reason = None
        hit_t1 = False
        t1_exit_price: Optional[float] = None
        t1_exit_ts = None

        for _, bar in forward.iterrows():
            bar_ts = bar["date"]
            bar_hhmm = bar_ts.strftime("%H:%M")
            high = float(bar["high"])
            low = float(bar["low"])
            close_b = float(bar["close"])
            vwap_b = float(bar["vwap"]) if pd.notna(bar.get("vwap")) else np.nan

            # Active SL: hard_sl until T1 fills, then breakeven (entry).
            active_sl = entry_price if hit_t1 else hard_sl

            # SL check first (worst-case fill semantics — short blown up if
            # high reaches active_sl).
            if high >= active_sl:
                exit_ts = bar_ts
                exit_price = active_sl
                exit_reason = "breakeven_trail" if hit_t1 else "stop"
                break

            # T1 partial fill check (only first time it triggers).
            if (not hit_t1) and (low <= t1_target):
                hit_t1 = True
                t1_exit_price = t1_target
                t1_exit_ts = bar_ts

            # T2 = VWAP. After T1, exit remaining 50% if VWAP touched.
            # (Pre-T1 VWAP touches are not exits — we want T1 first.)
            if hit_t1 and pd.notna(vwap_b) and (low <= vwap_b):
                exit_ts = bar_ts
                exit_price = vwap_b
                exit_reason = "t2_vwap"
                break

            # Hard time stop at 15:10 — forced exit at bar's close, regardless
            # of P&L. This bar is INCLUDED (the 15:10 bar IS the exit bar).
            if bar_hhmm >= HARD_TIMESTOP_HHMM:
                exit_ts = bar_ts
                exit_price = close_b
                exit_reason = "time_stop_1510"
                break

        # Defensive fallback: if we walked off the day (no time-stop bar
        # available because forward window terminated early), close at the
        # last forward bar's close with reason 'eod'.
        if exit_price is None:
            last = forward.iloc[-1]
            exit_ts = last["date"]
            exit_price = float(last["close"])
            exit_reason = "eod"

        # ---- Position sizing & PnL ----
        qty = max(int(RISK_PER_TRADE_RUPEES / max(stop_distance, 1e-6)), 1)

        if hit_t1:
            qty_t1 = qty // 2
            qty_t2 = qty - qty_t1
            # SHORT: PnL = (entry - exit) * qty.
            pnl_t1 = (entry_price - t1_exit_price) * qty_t1
            pnl_t2 = (entry_price - exit_price) * qty_t2
            realized_pnl = pnl_t1 + pnl_t2
            fee_t1 = calc_fee(entry_price, t1_exit_price, qty_t1, "SELL")
            fee_t2 = calc_fee(entry_price, exit_price, qty_t2, "SELL")
            fee = fee_t1 + fee_t2
            blended_exit = (
                t1_exit_price * qty_t1 + exit_price * qty_t2
            ) / max(qty, 1)
        else:
            realized_pnl = (entry_price - exit_price) * qty
            fee = calc_fee(entry_price, exit_price, qty, "SELL")
            blended_exit = exit_price

        net_pnl = realized_pnl - fee

        trades.append({
            "T1_entry_date": sd,
            "symbol": "NSE:" + sym,
            "cap_segment": t["cap_segment"],
            "side": "SHORT",
            "entry_hhmm": t["hhmm"],
            "entry_ts": entry_ts,
            "entry_price": entry_price,
            "intraday_ret_pct": float(t["intraday_ret_pct"]),
            "off_high_pct": float(t["off_high_pct"]),
            "ret_3_pct": float(t["ret_3_pct"]),
            "ret_at_1330_pct": float(t["ret_at_1330_pct"]),
            "cum_vol_pct_rank": float(t["cum_vol_pct_rank"]),
            "intraday_high_at_entry": intra_high_at_entry,
            "hard_sl": hard_sl,
            "t1_target": t1_target,
            "stop_distance": stop_distance,
            "hit_t1": hit_t1,
            "exit_ts": exit_ts,
            "exit_price": blended_exit,
            "exit_reason": exit_reason,
            "qty": qty,
            "realized_pnl": realized_pnl,
            "fee": fee,
            "net_pnl": net_pnl,
        })
        n_traded += 1

    print(f"\n  no session in 5m:    {n_no_session}")
    print(f"  no entry bar idx:    {n_no_entry_idx}")
    print(f"  traded:              {n_traded}")
    return pd.DataFrame(trades)


# ---- Reporting ----------------------------------------------------------

def report(trades: pd.DataFrame, prior_symbols: Optional[set]) -> None:
    if trades.empty:
        print("\n[NO TRADES] sanity check returns 0 trades")
        return
    n = len(trades)
    npnl = trades["net_pnl"]
    wins = npnl[npnl > 0].sum()
    losses = npnl[npnl < 0].abs().sum()
    pf = round(float(wins / losses), 3) if losses > 0 else float("inf")
    daily = trades.groupby("T1_entry_date")["net_pnl"].sum()
    sharpe = (
        round(float(daily.mean() / daily.std()), 3)
        if daily.std() > 0 else 0.0
    )
    wr = round(float((npnl > 0).mean()) * 100, 1)

    print("\n=== mis_unwind_short_late_session -- pre-coding sanity check ===")
    print(f"Period: {trades['T1_entry_date'].min()} .. {trades['T1_entry_date'].max()}")
    print(f"Trades: n = {n}")
    print(f"Win rate: {wr}%")
    print(f"Gross PnL: Rs.{int(trades['realized_pnl'].sum()):,}")
    print(f"Fees:      Rs.{int(trades['fee'].sum()):,}")
    print(f"NET PnL:   Rs.{int(npnl.sum()):,}")
    print(f"NET PF:    {pf}")
    print(f"NET Sharpe (daily): {sharpe}")

    print("\nPer cap_segment:")
    for cap, grp in trades.groupby("cap_segment"):
        n2 = len(grp)
        w = grp["net_pnl"][grp["net_pnl"] > 0].sum()
        l = grp["net_pnl"][grp["net_pnl"] < 0].abs().sum()
        pf2 = round(float(w / l), 3) if l > 0 else float("inf")
        wr2 = round(float((grp["net_pnl"] > 0).mean()) * 100, 1)
        net = int(grp["net_pnl"].sum())
        print(f"  {cap:<12} n={n2:>4} PF={pf2:>5} WR={wr2:>5}% netPnL=Rs.{net:>10,}")

    print("\nPer month:")
    months = pd.to_datetime(trades["T1_entry_date"]).dt.to_period("M").astype(str)
    for m, grp in trades.assign(_m=months.values).groupby("_m"):
        n2 = len(grp)
        w = grp["net_pnl"][grp["net_pnl"] > 0].sum()
        l = grp["net_pnl"][grp["net_pnl"] < 0].abs().sum()
        pf2 = round(float(w / l), 3) if l > 0 else float("inf")
        wr2 = round(float((grp["net_pnl"] > 0).mean()) * 100, 1)
        net = int(grp["net_pnl"].sum())
        print(f"  {m}  n={n2:>4} PF={pf2:>5} WR={wr2:>5}% netPnL=Rs.{net:>10,}")

    print("\nExit-reason breakdown:")
    for rsn, grp in trades.groupby("exit_reason"):
        n2 = len(grp)
        avg = int(grp["net_pnl"].mean())
        print(f"  {rsn:<22} n={n2:>4} avg_net=Rs.{avg:>6,}")

    # ---- Symbol-overlap diagnostic vs prior failed mis_unwind_short ----
    print("\n--- Symbol-overlap diagnostic vs prior `mis_unwind_short` (failed) ---")
    this_syms = set(
        trades["symbol"].astype(str).str.replace("NSE:", "", regex=False).unique().tolist()
    )
    overlap_pct: Optional[float] = None
    if prior_symbols is None:
        print(
            "  prior trade parquet UNAVAILABLE -- MANUAL CHECK required: "
            "verify symbol overlap between this sanity's trades and the prior "
            "mis_unwind_short trade set."
        )
    else:
        inter = this_syms & prior_symbols
        union = this_syms | prior_symbols
        overlap_pct_of_this = (
            (len(inter) / len(this_syms)) * 100.0 if this_syms else 0.0
        )
        jaccard = (
            (len(inter) / len(union)) * 100.0 if union else 0.0
        )
        overlap_pct = round(overlap_pct_of_this, 1)
        print(f"  this sanity's unique symbols: {len(this_syms)}")
        print(f"  prior failed-setup unique symbols: {len(prior_symbols)}")
        print(f"  intersection: {len(inter)}")
        print(f"  overlap (% of THIS sanity's symbols also in prior): {overlap_pct}%")
        print(f"  Jaccard similarity:                                 {round(jaccard, 1)}%")
        print(f"  Brief gate: overlap < {SYMBOL_OVERLAP_GATE_PCT}% required")
        if overlap_pct >= SYMBOL_OVERLAP_GATE_PCT:
            print(
                f"  FAIL: overlap {overlap_pct}% >= {SYMBOL_OVERLAP_GATE_PCT}% "
                "-> targeted population is essentially the same as the prior "
                "failed setup. ABANDON candidate."
            )
        else:
            print(
                f"  PASS: overlap {overlap_pct}% < {SYMBOL_OVERLAP_GATE_PCT}% "
                "-> targeted population is materially different from prior."
            )

    # ---- VERDICT ----
    print("\n--- VERDICT ---")
    abandon_reasons: List[str] = []
    if wr < 20.0:
        abandon_reasons.append(f"WR {wr}% < 20% (CNC-conversion squeeze still dominates)")
    if n < 30:
        abandon_reasons.append(f"n {n} < 30 (sample too thin)")
    if overlap_pct is not None and overlap_pct >= SYMBOL_OVERLAP_GATE_PCT:
        abandon_reasons.append(
            f"symbol-overlap {overlap_pct}% >= {SYMBOL_OVERLAP_GATE_PCT}% "
            "(same population as prior failure)"
        )

    if pf >= 1.10 and not abandon_reasons:
        print(f"PF={pf} >= 1.10 -> STRONG PROCEED. Move to detector implementation.")
    elif pf >= 1.00 and not abandon_reasons:
        print(f"PF={pf} in [1.00, 1.10) -> marginal. Proceed with caveat.")
    elif abandon_reasons:
        print(f"PF={pf} | falsification gate(s) tripped -> ABANDON. Reasons:")
        for r in abandon_reasons:
            print(f"  - {r}")
    else:
        print(f"PF={pf} < 1.00 -> RETIRE candidate. Do not write detector code.")


def main():
    universe = load_fno_universe()
    big5m = build_full_period_5m(universe, ALLOWED_CAPS)
    if big5m.empty:
        print("[ABORT] no 5m feathers found")
        return
    adv_table = load_daily_for_liquidity()
    prior_symbols = load_prior_symbols()

    print("\nFinding triggers ...")
    triggers = find_triggers(big5m, universe, adv_table)
    print(f"\nTotal triggers (after latch): {len(triggers)}")
    if triggers.empty:
        print("[NO TRIGGERS] -- no symbols passed the 6-gate filter.")
        return

    print("\nSimulating entries -> exits:")
    trades = simulate(triggers, big5m)
    report(trades, prior_symbols)

    out = (
        _REPO_ROOT / "reports" / "sub9_sanity"
        / "mis_unwind_short_late_session_trades.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out, index=False)
    print(f"\nFull trade log: {out}")


if __name__ == "__main__":
    main()
