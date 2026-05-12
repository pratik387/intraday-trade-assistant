"""Pre-coding sanity check for closing_hour_reversal_short_redo candidate.

Per sub-9 round-6 §3.3 brief gate (specs/2026-05-07-sub-project-9-brief-
closing_hour_reversal_short_redo.md): BEFORE writing detector code,
simulate the operator-pump-unwind SHORT mechanic on the RETAIL-LIGHT
small/mid-cap NSE universe (NON-F&O, bottom-tercile cum-vol at 14:00).

Decision criteria (locked, brief §10):
  PF >= 1.10 AND n >= 500 AND symbol-overlap < 40% -> APPROVE detector
  PF in [1.00, 1.10) AND gates pass -> REVISE (probe vol-decline cutoff)
  Symbol-overlap >= 40% vs `mis_unwind_short_late_session` -> STRUCTURAL RETIRE
  n < 500 over Discovery 2yr -> STRUCTURAL RETIRE (round-6 floor failure)
  PF < 1.00 -> THESIS RETIRE (operator-pump-unwind falsified at fee scale)

Mechanic — designed to TARGET RETAIL-LIGHT non-F&O 200 small/mid-caps
where the prior `mis_unwind_short_late_session` (failed PF 0.367, n=1008)
could not differentiate population:

  Universe (data-broad with retail-light classifier):
    - cap_segment in {small_cap, mid_cap} (large_cap excluded — institutional
      shadow dominates; micro/unknown excluded — too thin / unclassified)
    - SYMBOL NOT in F&O 200 (assets/fno_liquid_200.csv) — zero F&O OI =
      zero F&O-driven retail flow
    - Retail-light classifier (Option A): bottom-tercile of
      cumulative_volume_at_14:00 / 20d_median_full_session_volume per
      session — only this subset

  At 14:30 5m bar close, qualify symbol if ALL hold:
    - Intraday return at 14:30 in [+1.5%, +3.5%] (rallying — but not
      runaway news event; brief §6 widened to 1.5-3.5% for post-redo)
    - Volume-decline gate: mean(vol 13:30-14:30) < 0.7 *
      mean(vol 09:30-12:00) — operator-pump signature

  Entry: SHORT at 14:30 5m bar's CLOSE. Latch per (symbol, day):
    only the first qualifying bar fires.
  Hard SL: max(intraday-high * 1.005, entry * 1.012).
  T1: 1R partial (50%) — breakeven trail after T1 fill.
  T2: 2R full close.
  HARD time-stop: 15:10 forced exit at bar close (5 min before MIS
    auto-square 15:15-15:20).

Symbol-overlap diagnostic vs prior failed setup:
  reports/sub9_sanity/mis_unwind_short_late_session_trades.csv has
  1,008 trades. Compute overlap as % of THIS sanity's (symbol, day)
  pairs that also appear in that prior trade list.
  Brief gate: overlap < 40%. Above -> STRUCTURAL RETIRE.

Cross-axis independence diagnostic:
  reports/sub7_validation/gap_fade_short.parquet (different time
  window 09:15-09:30, F&O 200 universe — should be NEAR ZERO overlap)
  reports/sub9_sanity/circuit_t1_fade_short_trades.csv (different
  trigger 10:30 single-bar — should be NEAR ZERO overlap)

Indian fee model: tools/sub7_validation/build_per_setup_pnl.calc_fee
(Zerodha intraday-equity rate card + STT/exch/SEBI/IPFT/stamp/GST).

Discovery period: 2023-01-01 -> 2024-12-31 (24 monthly feathers).

Usage:
    python tools/sub9_research/sanity_closing_hour_reversal_short_redo.py
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from services.symbol_metadata import get_cap_segment           # noqa: E402
from tools.sub7_validation.build_per_setup_pnl import calc_fee  # noqa: E402


# ---- Locked params (per brief §6 Mechanic; sanity-time non-tunable) -----
ALLOWED_CAPS = {"small_cap", "mid_cap"}    # large_cap excluded (operator-pump
                                           # absent in NIFTY100 institutional
                                           # shadow); unknown/micro excluded

# Entry single-screen at 14:30 5m bar close (brief §6 step 3 + sanity
# constraint: SHORT entry at 14:30 5m bar close is the locked param;
# brief's full mechanic allows 14:30..15:00 rollover-bar entries, but
# the sanity sticks to the single-bar version per task instruction).
ENTRY_HHMM = "14:30"
ENTRY_HHMM_INT = 14 * 100 + 30  # = 1430

# Reference HH:MM windows for derived signals.
HHMM_OPEN = "09:15"
HHMM_0930 = "09:30"
HHMM_1200 = "12:00"
HHMM_1330 = "13:30"
HHMM_1400 = "14:00"
HHMM_1430 = "14:30"

# Integer encodings for the per-row hhmm column (HH*100 + MM). Storing the
# 27M-row hhmm column as int16 (instead of 5-char unicode) avoids a 300+ MB
# allocation that OOMs on 16GB hosts during the R-multiplier sweep. Lex order
# of "HH:MM" strings is preserved by int order because the format is
# fixed-width; all comparisons (>=, <=, ==) translate 1:1.
def _hhmm_to_int(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 100 + int(m)

HHMM_INT_OPEN = _hhmm_to_int(HHMM_OPEN)
HHMM_INT_0930 = _hhmm_to_int(HHMM_0930)
HHMM_INT_1200 = _hhmm_to_int(HHMM_1200)
HHMM_INT_1330 = _hhmm_to_int(HHMM_1330)
HHMM_INT_1400 = _hhmm_to_int(HHMM_1400)
HHMM_INT_1430 = _hhmm_to_int(HHMM_1430)

# Intraday-return gate at entry bar close (brief §6 step 2 — locked
# 1.5-3.5% for the redo per task instruction; brief §6 widens to
# 1.0-5.0% but the task's locked spec narrows to 1.5-3.5%).
INTRADAY_RET_LO_PCT = 1.5    # heavy intraday rally floor
INTRADAY_RET_HI_PCT = 3.5    # exclude runaway news events

# Volume-decline gate (brief §6 step 2 — operator-pump signature).
VOL_DECLINE_RATIO = 0.7      # mean(13:30-14:30) < 0.7 * mean(09:30-12:00)

# Retail-light classifier — bottom-tercile cum-vol rank at 14:00.
RETAIL_LIGHT_PCT_MAX = 33.0  # bottom 33rd percentile (per session)

# Stop / target (per brief §6 step 4-5 + task locked params).
STOP_BUFFER_PCT_INTRA_HIGH = 0.5   # max(intraday-high * 1.005, ...)
STOP_MIN_PCT_OVER_ENTRY = 1.2      # max(..., entry * 1.012)

# R-multiplier constants for T1 / T2 / stop (CLI-overridable; default = 1R / 2R / 1R).
# T1 partial fill at entry - T1_R * stop_distance; T2 full close at entry - T2_R * stop_distance.
# STOP_R scales the computed hard_sl distance (1.0 = no scaling).
_R_MULTIPLE_T1 = 1.0
_R_MULTIPLE_T2 = 2.0
_R_MULTIPLE_STOP = 1.0

# Hard time-stop bar (forced exit regardless of P&L; 5 min before MIS
# auto-square 15:15-15:20).
HARD_TIMESTOP_HHMM = "15:10"
HARD_TIMESTOP_HHMM_INT = 15 * 100 + 10  # = 1510

# Risk per trade — match other sub9 sanity scripts for comparable PnL units.
RISK_PER_TRADE_RUPEES = 1000

# Liquidity floor (matches other sub9 sanity scripts for fee-scale parity).
MIN_ADV_INR_CR = 1.0   # lowered vs F&O scripts (Rs 3 Cr) because retail-light
                       # universe is by-definition lower-volume; brief §11 has
                       # no explicit floor. Rs 1 Cr keeps the universe tradeable
                       # while filtering pure penny noise.

# Round-6 acceptance gates (project-locked; brief §9 and §10).
N_FLOOR = 500
PF_FLOOR_APPROVE = 1.10
PF_FLOOR_MARGINAL = 1.00
SYMBOL_OVERLAP_GATE_PCT = 40.0    # brief §10 explicit STRUCTURAL RETIRE gate

# Prior failed mis_unwind_short_late_session trade CSV (for symbol-overlap).
PRIOR_FAILED_CSV = (
    _REPO_ROOT / "reports" / "sub9_sanity"
    / "mis_unwind_short_late_session_trades.csv"
)
# Cross-axis independence diagnostic sources.
GAP_FADE_PARQUET = (
    _REPO_ROOT / "reports" / "sub7_validation" / "gap_fade_short.parquet"
)
CIRCUIT_T1_CSV = (
    _REPO_ROOT / "reports" / "sub9_sanity" / "circuit_t1_fade_short_trades.csv"
)


# ---- Data loading -------------------------------------------------------

def _load_5m_for_month(yyyy: int, mm: int) -> pd.DataFrame:
    path = (
        _REPO_ROOT / "backtest-cache-download" / "monthly"
        / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather"
    )
    if not path.exists():
        return pd.DataFrame()
    return pd.read_feather(path)


def load_fno_universe() -> set:
    """Load F&O 200 list (for EXCLUSION per brief §7)."""
    path = _REPO_ROOT / "assets" / "fno_liquid_200.csv"
    df = pd.read_csv(path)
    col = "Symbol" if "Symbol" in df.columns else "symbol"
    syms = (
        df[col].astype(str).str.replace("NSE:", "", regex=False).str.strip().tolist()
    )
    print(f"  F&O 200 list (for EXCLUSION): {len(syms)} symbols")
    return set(syms)


def load_daily_for_liquidity_and_baseline() -> pd.DataFrame:
    """Daily ADV (Rs Cr) + 20d median full-session volume for each (symbol, d).

    The 20d median full-session volume is the denominator of the retail-light
    classifier (cum_vol_at_1400 / 20d_median_full_session_vol).
    """
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
    # ADV (Rs Cr) — 20-day rolling mean of traded_value, shifted by 1 (no peek).
    df["adv_20d_cr"] = df.groupby("symbol")["traded_value"].transform(
        lambda v: v.shift(1).rolling(20).mean()
    ) / 1e7
    # 20d median full-session volume (shifted; no peek).
    df["vol_20d_median"] = df.groupby("symbol")["volume"].transform(
        lambda v: v.shift(1).rolling(20).median()
    )
    return df[["symbol", "d", "adv_20d_cr", "vol_20d_median"]]


def build_full_period_5m(allowed_caps: set, fno_excl: set) -> pd.DataFrame:
    """Stream-load 24 monthly feathers; pre-filter to small/mid + non-F&O.

    Per brief §11 the universe is data-broad — we load ALL symbols, then
    drop F&O 200 (exclusion) and require cap_segment in {small, mid}.
    Filtering happens INSIDE the per-month loop to keep peak memory low.
    """
    print("  loading 24 monthly 5m feathers (2023-01 .. 2024-12) ...")

    # Pre-build the allowed-symbol set ONCE: any non-F&O symbol whose
    # cap_segment is in allowed_caps. Cheaper than per-row Python apply on 47M
    # rows, and fits inside memory.
    print("  building allowed-symbol set from nse_all.json cap segments ...")
    # We need a candidate symbol pool. Pull from one feather to enumerate.
    sample_path = (
        _REPO_ROOT / "backtest-cache-download" / "monthly"
        / "2023_01_5m_enriched.feather"
    )
    sample_df = pd.read_feather(sample_path, columns=["symbol"])
    sample_syms = set(sample_df["symbol"].unique().tolist())
    # Also pull 2024_06 to widen the candidate pool (newly listed symbols).
    extra_path = (
        _REPO_ROOT / "backtest-cache-download" / "monthly"
        / "2024_06_5m_enriched.feather"
    )
    if extra_path.exists():
        sample_syms |= set(
            pd.read_feather(extra_path, columns=["symbol"])["symbol"]
            .unique().tolist()
        )
    # Score each candidate and keep those passing both filters.
    allowed_syms: set = set()
    sym_to_cap: Dict[str, str] = {}
    for s in sample_syms:
        if s in fno_excl:
            continue
        cap = get_cap_segment("NSE:" + s)
        if cap in allowed_caps:
            allowed_syms.add(s)
            sym_to_cap[s] = cap
    print(
        f"  allowed-symbol set: {len(allowed_syms):,} symbols "
        f"(non-F&O + cap_segment in {sorted(allowed_caps)})"
    )

    # Only keep columns used downstream — drops indicator cols (vwap,
    # bb_width_proxy, adx, rsi) we don't need for this mechanic. Reduces
    # peak memory ~40% on the post-concat DataFrame.
    keep_cols = ["symbol", "date", "open", "high", "low", "close", "volume"]
    parts: List[pd.DataFrame] = []
    total_raw = 0
    for yyyy in (2023, 2024):
        for m in range(1, 13):
            mdf = _load_5m_for_month(yyyy, m)
            if mdf.empty:
                continue
            total_raw += len(mdf)
            # Single pass: keep only symbols in the pre-built allowed set.
            mdf = mdf[mdf["symbol"].isin(allowed_syms)]
            if mdf.empty:
                continue
            mdf = mdf[keep_cols].copy()
            # Downcast prices/volume to float32 to halve memory.
            for c in ("open", "high", "low", "close"):
                mdf[c] = mdf[c].astype("float32")
            mdf["volume"] = mdf["volume"].astype("float32")
            # IMPORTANT: keep symbol / cap_segment as plain object dtype here.
            # Concat-of-categoricals with disjoint category sets per month
            # forces a per-frame astype-to-object during concat, allocating a
            # 1.1M-row object array per month inside pandas' internal join
            # (numpy._core._exceptions._ArrayMemoryError on 16GB hosts during
            # R-multiplier sweeps). Categorize ONCE post-concat where the
            # category set is union'd in a single pass. Mechanic logic
            # unchanged — downstream code only reads .values / groupby keys.
            mdf["cap_segment"] = mdf["symbol"].map(sym_to_cap)
            parts.append(mdf)
    if not parts:
        return pd.DataFrame()
    big = pd.concat(parts, ignore_index=True, copy=False)
    # Now categorize symbol / cap_segment ONCE on the full union of values.
    big["symbol"] = big["symbol"].astype("category")
    big["cap_segment"] = big["cap_segment"].astype("category")
    print(f"  raw bars across 24 months: {total_raw:,}")
    print(f"  after non-F&O + cap filter: {len(big):,}")

    # Avoid full sort (memory-bound on 27M rows). Add d / hhmm directly,
    # then rely on per-(symbol, d) groupby which is order-tolerant within
    # the groupby (we sort inside _enrich_session_features anyway).
    big["d"] = big["date"].dt.date
    # IMPORTANT: store hhmm as int16 (HH*100 + MM), not 5-char unicode.
    # Allocating a 27M-row "HH:MM" string column needs ~300+ MB and OOMs on
    # 16GB hosts. int16 column is ~54 MB and supports identical lex-order
    # comparisons (the "HH:MM" format is fixed-width). All downstream usages
    # compare against HHMM_INT_* constants instead of the original strings.
    big["hhmm"] = (
        big["date"].dt.hour.astype("int16") * 100
        + big["date"].dt.minute.astype("int16")
    ).astype("int16")
    return big


# ---- Per-session feature engineering ------------------------------------

def _enrich_session_features(day_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-bar intraday features for one (symbol, session).

    Adds: intraday_high_so_far, intraday_ret_pct, cum_vol, vol_mean_0930_1200,
          vol_mean_1330_1430, intraday_high_at_1430.
    Returns the input df with new columns.
    """
    # Drop the explicit .copy(): sort_values + reset_index already returns a
    # new DataFrame, so the redundant .copy() doubles the per-group memory
    # footprint. With ~330K (symbol, d) groups this produced fatal heap
    # fragmentation on a 16GB host (1.46 KiB allocs failing inside pandas
    # block-manager copy). Patched 2026-05-07 round-6 closing_hour rerun.
    df = day_df.sort_values("date").reset_index(drop=True)
    if df.empty:
        return df
    open_anchor = float(df.iloc[0]["open"])
    df["open_anchor"] = open_anchor
    df["intraday_high_so_far"] = df["high"].cummax()
    df["intraday_ret_pct"] = (df["close"] / open_anchor - 1.0) * 100.0
    df["cum_vol"] = df["volume"].cumsum()

    # Volume-decline windows (note: bars are right-labeled close-of-bar; e.g.
    # the 09:30 bar covers 09:30-09:35. We use HH:MM >= 09:30 AND <= 12:00 for
    # the morning window, and >= 13:30 AND <= 14:30 for the late-morning
    # window. This is approximate but consistent across sessions.)
    mask_morning = (df["hhmm"] >= HHMM_INT_0930) & (df["hhmm"] <= HHMM_INT_1200)
    if mask_morning.any():
        avg_morning = float(df.loc[mask_morning, "volume"].mean())
    else:
        avg_morning = np.nan
    df["vol_mean_0930_1200"] = avg_morning

    mask_pre_entry = (df["hhmm"] >= HHMM_INT_1330) & (df["hhmm"] <= HHMM_INT_1430)
    if mask_pre_entry.any():
        avg_pre_entry = float(df.loc[mask_pre_entry, "volume"].mean())
    else:
        avg_pre_entry = np.nan
    df["vol_mean_1330_1430"] = avg_pre_entry

    # Cum vol AT 14:00 — for retail-light classifier denominator.
    if (df["hhmm"] == HHMM_INT_1400).any():
        cum_vol_1400 = float(
            df.loc[df["hhmm"] == HHMM_INT_1400, "cum_vol"].iloc[0]
        )
    else:
        cum_vol_1400 = np.nan
    df["cum_vol_at_1400"] = cum_vol_1400

    return df


# ---- Trigger discovery --------------------------------------------------

def find_triggers(
    big5m: pd.DataFrame,
    adv_table: pd.DataFrame,
) -> pd.DataFrame:
    """Per-session, per-symbol: enrich features, compute retail-light classifier
    rank cross-symbol, then apply 14:30 single-bar gates and latch.
    """
    df = big5m  # avoid full .copy() — costs ~1 GB on 27M rows
    print(f"    pre-filtered 5m bars (cap+non-F&O): {len(df):,}")

    # Attach ADV + 20d_median full-session volume via index.map (no merge).
    # df.merge on (symbol, d) realigns the 27M-row block manager and
    # internally copies the object/categorical symbol column twice (~400 MB
    # each), OOMing on 16GB hosts during the R-multiplier sweep. .map() on
    # a MultiIndex hits the right rows with a single hash lookup and
    # produces only the two new float columns (~220 MB). Equivalent semantics
    # for the (symbol, d) -> (adv, vol_median) lookup; mechanic unchanged.
    adv_idx = adv_table.set_index(["symbol", "d"])
    sym_d_idx = pd.MultiIndex.from_arrays([df["symbol"].astype(str), df["d"]])
    df = df.copy()  # allow column assignment without SettingWithCopy warnings
    df["adv_20d_cr"] = sym_d_idx.map(adv_idx["adv_20d_cr"]).astype("float32")
    df["vol_20d_median"] = sym_d_idx.map(adv_idx["vol_20d_median"]).astype("float32")
    del sym_d_idx, adv_idx
    import gc as _gc0
    _gc0.collect()

    # Liquidity floor (in-place boolean mask, no extra .copy()).
    df = df[df["adv_20d_cr"] >= MIN_ADV_INR_CR]
    print(f"    adv_20d >= Rs {MIN_ADV_INR_CR}Cr: {len(df):,}")

    print("  enriching per-(symbol, session) intraday features (vectorized) ...")
    # Vectorized rewrite 2026-05-07 round-6: the original per-group python
    # loop over ~330K (symbol, d) groups OOM'd on a 16GB host (heap
    # fragmentation from millions of small DataFrame ops). All per-session
    # features computable as groupby transforms / merges — constant memory.
    import gc
    _keep_cols = [
        c for c in df.columns
        if c in {"symbol", "d", "date", "hhmm",
                 "open", "high", "low", "close", "volume",
                 "adv_20d_cr", "vol_20d_median", "cap_segment"}
    ]
    enriched = df[_keep_cols].sort_values(["symbol", "d", "date"]).reset_index(drop=True)
    del df
    gc.collect()

    grp_sd = enriched.groupby(["symbol", "d"], sort=False)
    enriched["open_anchor"] = grp_sd["open"].transform("first")
    enriched["intraday_high_so_far"] = grp_sd["high"].cummax()
    enriched["intraday_ret_pct"] = (
        enriched["close"] / enriched["open_anchor"] - 1.0
    ) * 100.0
    enriched["cum_vol"] = grp_sd["volume"].cumsum()

    # Window-mean columns: compute per-(symbol, d) means on filtered subsets,
    # then broadcast back via map.
    mask_morn = (enriched["hhmm"] >= HHMM_INT_0930) & (enriched["hhmm"] <= HHMM_INT_1200)
    morn_mean = (
        enriched.loc[mask_morn]
        .groupby(["symbol", "d"], sort=False)["volume"]
        .mean()
    )
    enriched["vol_mean_0930_1200"] = enriched.set_index(["symbol", "d"]).index.map(morn_mean).values

    mask_pre = (enriched["hhmm"] >= HHMM_INT_1330) & (enriched["hhmm"] <= HHMM_INT_1430)
    pre_mean = (
        enriched.loc[mask_pre]
        .groupby(["symbol", "d"], sort=False)["volume"]
        .mean()
    )
    enriched["vol_mean_1330_1430"] = enriched.set_index(["symbol", "d"]).index.map(pre_mean).values

    # Cum vol AT 14:00 — first row matching hhmm==14:00 per (symbol, d)
    mask_1400 = enriched["hhmm"] == HHMM_INT_1400
    cv_1400 = (
        enriched.loc[mask_1400]
        .drop_duplicates(subset=["symbol", "d"], keep="first")
        .set_index(["symbol", "d"])["cum_vol"]
    )
    enriched["cum_vol_at_1400"] = enriched.set_index(["symbol", "d"]).index.map(cv_1400).values
    del morn_mean, pre_mean, cv_1400
    gc.collect()
    print(f"    enriched bars: {len(enriched):,}")

    # ---- Retail-light classifier (bottom-tercile cum-vol rank at 14:00) ----
    # Per session, rank symbols by (cum_vol_at_1400 / vol_20d_median).
    # Bottom 33rd percentile = retail-light. We compute rank ONCE per (symbol,
    # d), using only one row per session to avoid intra-day duplication.
    session_rows = (
        enriched.dropna(subset=["cum_vol_at_1400", "vol_20d_median"])
        .drop_duplicates(subset=["symbol", "d"], keep="first")
        [["symbol", "d", "cum_vol_at_1400", "vol_20d_median"]]
        .copy()
    )
    session_rows = session_rows[session_rows["vol_20d_median"] > 0]
    session_rows["vol_util_ratio"] = (
        session_rows["cum_vol_at_1400"] / session_rows["vol_20d_median"]
    )
    # Within each session, percentile-rank ASCENDING (low ratio -> low rank
    # -> bottom-tercile -> retail-light).
    session_rows["vol_util_pct_rank"] = session_rows.groupby("d")[
        "vol_util_ratio"
    ].rank(pct=True, method="average") * 100.0
    retail_light = session_rows[
        session_rows["vol_util_pct_rank"] <= RETAIL_LIGHT_PCT_MAX
    ][["symbol", "d", "vol_util_ratio", "vol_util_pct_rank"]].copy()
    print(
        f"    retail-light universe (bottom-{int(RETAIL_LIGHT_PCT_MAX)}% cum-vol "
        f"rank at 14:00): {len(retail_light):,} (symbol, day) cells"
    )

    # Restrict enriched bars to retail-light (symbol, d).
    enriched = enriched.merge(
        retail_light[["symbol", "d", "vol_util_pct_rank"]],
        on=["symbol", "d"], how="inner", validate="many_to_one"
    )
    print(f"    bars in retail-light universe: {len(enriched):,}")

    # Restrict to 14:30 entry-screen bar.
    entry_bars = enriched[enriched["hhmm"] == ENTRY_HHMM_INT].copy()
    print(f"    bars at {ENTRY_HHMM}: {len(entry_bars):,}")

    # Apply 14:30 gates.
    cond_intraday_ret = (
        (entry_bars["intraday_ret_pct"] >= INTRADAY_RET_LO_PCT)
        & (entry_bars["intraday_ret_pct"] <= INTRADAY_RET_HI_PCT)
    )
    n_after_rally = int(cond_intraday_ret.sum())
    print(
        f"    after intraday-ret in [{INTRADAY_RET_LO_PCT}, "
        f"{INTRADAY_RET_HI_PCT}]: {n_after_rally:,}"
    )

    cond_vol_decline = (
        entry_bars["vol_mean_1330_1430"]
        < VOL_DECLINE_RATIO * entry_bars["vol_mean_0930_1200"]
    ) & entry_bars["vol_mean_0930_1200"].notna() & entry_bars[
        "vol_mean_1330_1430"
    ].notna()
    triggers = entry_bars[cond_intraday_ret & cond_vol_decline].copy()
    print(
        f"    after volume-decline gate (vol(13:30-14:30) < "
        f"{VOL_DECLINE_RATIO} * vol(09:30-12:00)): {len(triggers):,}"
    )

    # Latch per (symbol, day) — there's only one 14:30 bar so this is
    # effectively a no-op, but guards against any duplication.
    triggers = triggers.sort_values(["symbol", "d", "date"]).drop_duplicates(
        subset=["symbol", "d"], keep="first"
    )
    print(f"    fired (after latch): {len(triggers):,}")

    return triggers.reset_index(drop=True)


# ---- Trade simulation ---------------------------------------------------

def simulate(triggers: pd.DataFrame, big5m: pd.DataFrame) -> pd.DataFrame:
    print("  simulating SHORT entries -> T1/SL/T2/time-stop ...")
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

        # Entry: SHORT at the 14:30 bar's CLOSE.
        entry_price = float(entry_bar["close"])
        entry_ts = entry_bar["date"]

        # Stop = max(intraday-high * 1.005, entry * 1.012), scaled by _R_MULTIPLE_STOP.
        intra_high_at_entry = float(t["intraday_high_so_far"])
        sl_struct = intra_high_at_entry * (1.0 + STOP_BUFFER_PCT_INTRA_HIGH / 100.0)
        sl_min = entry_price * (1.0 + STOP_MIN_PCT_OVER_ENTRY / 100.0)
        base_sl = max(sl_struct, sl_min)
        base_stop_distance = base_sl - entry_price
        if base_stop_distance <= 0:
            continue
        stop_distance = base_stop_distance * _R_MULTIPLE_STOP
        hard_sl = entry_price + stop_distance

        # T1 (T1_R partial - 50%) + T2 (T2_R full close).
        t1_target = entry_price - _R_MULTIPLE_T1 * stop_distance
        t2_target = entry_price - _R_MULTIPLE_T2 * stop_distance

        # Forward bars: entry bar's CLOSE is the fill, so subsequent bars
        # (entry_idx + 1 onwards) are where SL/T1/T2/time-stop fire.
        forward = day_df.iloc[entry_idx + 1:].copy()
        if forward.empty:
            continue

        exit_ts = None
        exit_price: Optional[float] = None
        exit_reason: Optional[str] = None
        hit_t1 = False
        t1_exit_price: Optional[float] = None
        t1_exit_ts = None

        for _, bar in forward.iterrows():
            bar_ts = bar["date"]
            bar_hhmm_int = bar_ts.hour * 100 + bar_ts.minute
            high = float(bar["high"])
            low = float(bar["low"])
            close_b = float(bar["close"])

            # Active SL: hard_sl until T1 fills, then breakeven (entry_price).
            active_sl = entry_price if hit_t1 else hard_sl

            # SL check first (worst-case fill semantics for SHORT).
            if high >= active_sl:
                exit_ts = bar_ts
                exit_price = active_sl
                exit_reason = "breakeven_trail" if hit_t1 else "stop"
                break

            # T1 partial fill (only first time).
            if (not hit_t1) and (low <= t1_target):
                hit_t1 = True
                t1_exit_price = t1_target
                t1_exit_ts = bar_ts

            # T2 full close (only after T1).
            if hit_t1 and (low <= t2_target):
                exit_ts = bar_ts
                exit_price = t2_target
                exit_reason = "t2"
                break

            # Hard time-stop at 15:10 — forced exit at bar's close.
            if bar_hhmm_int >= HARD_TIMESTOP_HHMM_INT:
                exit_ts = bar_ts
                exit_price = close_b
                exit_reason = "time_stop_1510"
                break

        # Defensive fallback (walked off the day with no time-stop bar).
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
            "entry_hhmm": ENTRY_HHMM,
            "entry_ts": entry_ts,
            "entry_price": entry_price,
            "intraday_ret_pct": float(t["intraday_ret_pct"]),
            "vol_mean_0930_1200": float(t["vol_mean_0930_1200"]),
            "vol_mean_1330_1430": float(t["vol_mean_1330_1430"]),
            "vol_decline_ratio": (
                float(t["vol_mean_1330_1430"] / t["vol_mean_0930_1200"])
                if t["vol_mean_0930_1200"] > 0 else np.nan
            ),
            "vol_util_pct_rank": float(t["vol_util_pct_rank"]),
            "intraday_high_at_entry": intra_high_at_entry,
            "hard_sl": hard_sl,
            "t1_target": t1_target,
            "t2_target": t2_target,
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


# ---- Diagnostics & reporting --------------------------------------------

def _load_overlap_set(
    path: Path, sym_col: str, date_col: str
) -> Optional[set]:
    """Load a (symbol, date) overlap set from a CSV/parquet trade list."""
    if not path.exists():
        return None
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if sym_col not in df.columns or date_col not in df.columns:
        return None
    df = df[[sym_col, date_col]].copy()
    df[sym_col] = (
        df[sym_col].astype(str).str.replace("NSE:", "", regex=False).str.strip()
    )
    df[date_col] = pd.to_datetime(df[date_col]).dt.date
    return set(zip(df[sym_col], df[date_col]))


def _trade_set(trades: pd.DataFrame) -> set:
    """(symbol bare, date) pairs from this sanity's trades."""
    return set(
        zip(
            trades["symbol"].astype(str)
            .str.replace("NSE:", "", regex=False)
            .str.strip()
            .tolist(),
            pd.to_datetime(trades["T1_entry_date"]).dt.date.tolist(),
        )
    )


def report(trades: pd.DataFrame) -> None:
    if trades.empty:
        print("\n[NO TRADES] sanity check returns 0 trades")
        print("--- VERDICT ---")
        print(f"n=0 < {N_FLOOR} -> STRUCTURAL RETIRE (round-6 floor failure)")
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

    print("\n=== closing_hour_reversal_short_redo -- pre-coding sanity check ===")
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

    # ---- MANDATORY symbol-overlap diagnostic vs mis_unwind_short_late_session
    print("\n--- Symbol-overlap diagnostic vs `mis_unwind_short_late_session` ---")
    print("    (BRIEF GATE: <40% of (symbol, day) overlap required)")
    this_pairs = _trade_set(trades)
    overlap_pct: Optional[float] = None
    prior_pairs = _load_overlap_set(
        PRIOR_FAILED_CSV, sym_col="symbol", date_col="T1_entry_date"
    )
    if prior_pairs is None:
        print(
            f"  WARNING: prior trade CSV not found at {PRIOR_FAILED_CSV} "
            "-- MANUAL CHECK required."
        )
    else:
        inter = this_pairs & prior_pairs
        overlap_pct_of_this = (
            (len(inter) / len(this_pairs)) * 100.0 if this_pairs else 0.0
        )
        overlap_pct = round(overlap_pct_of_this, 1)
        # Also report unique-symbol overlap (per brief §10 §11 narrative).
        this_syms = {p[0] for p in this_pairs}
        prior_syms = {p[0] for p in prior_pairs}
        sym_inter = this_syms & prior_syms
        sym_overlap_pct = round(
            (len(sym_inter) / len(this_syms)) * 100.0
            if this_syms else 0.0, 1
        )
        print(f"  this sanity (symbol, day) pairs: {len(this_pairs):,}")
        print(f"  prior failed-setup pairs:        {len(prior_pairs):,}")
        print(f"  intersecting (symbol, day):      {len(inter):,}")
        print(f"  PAIR-overlap (% of THIS in prior):   {overlap_pct}%")
        print(
            f"  SYMBOL-overlap (% of THIS unique syms in prior): "
            f"{sym_overlap_pct}%   ({len(sym_inter)}/{len(this_syms)})"
        )
        if overlap_pct >= SYMBOL_OVERLAP_GATE_PCT:
            print(
                f"  FAIL: pair-overlap {overlap_pct}% >= "
                f"{SYMBOL_OVERLAP_GATE_PCT}% -> STRUCTURAL RETIRE "
                "(population unchanged from prior failure)."
            )
        else:
            print(
                f"  PASS: pair-overlap {overlap_pct}% < "
                f"{SYMBOL_OVERLAP_GATE_PCT}% -> population is materially new."
            )

    # ---- Cross-axis independence diagnostic vs gap_fade_short + circuit_t1
    print("\n--- Cross-axis independence diagnostic ---")
    print("    (expect near-zero overlap — different time / different cap)")
    gap_pairs = _load_overlap_set(
        GAP_FADE_PARQUET, sym_col="symbol", date_col="session_date"
    )
    if gap_pairs is None:
        print(f"  gap_fade_short parquet not found at {GAP_FADE_PARQUET}")
    else:
        gap_inter = this_pairs & gap_pairs
        gap_pct = round(
            (len(gap_inter) / len(this_pairs)) * 100.0
            if this_pairs else 0.0, 2
        )
        print(
            f"  vs gap_fade_short (sub7 parquet):    "
            f"intersect={len(gap_inter)} ({gap_pct}% of this)"
        )

    circ_pairs = _load_overlap_set(
        CIRCUIT_T1_CSV, sym_col="symbol", date_col="T1_entry_date"
    )
    if circ_pairs is None:
        print(f"  circuit_t1 CSV not found at {CIRCUIT_T1_CSV}")
    else:
        circ_inter = this_pairs & circ_pairs
        circ_pct = round(
            (len(circ_inter) / len(this_pairs)) * 100.0
            if this_pairs else 0.0, 2
        )
        print(
            f"  vs circuit_t1_fade_short (sanity):   "
            f"intersect={len(circ_inter)} ({circ_pct}% of this)"
        )

    # ---- Top-20 contributing symbols (concentration check) ----
    print("\n--- Top-20 symbols by net PnL contribution ---")
    sym_grp = trades.groupby("symbol")["net_pnl"].agg(["sum", "count"])
    total_net = float(npnl.sum())
    sym_grp = sym_grp.sort_values("sum", ascending=False)
    for sym, row in sym_grp.head(20).iterrows():
        pct = (
            (float(row["sum"]) / total_net) * 100.0
            if abs(total_net) > 0 else 0.0
        )
        print(
            f"  {sym:<25} n={int(row['count']):>4}  "
            f"netPnL=Rs.{int(row['sum']):>10,}  ({pct:+.1f}% of total)"
        )

    # ---- VERDICT ----
    print("\n--- VERDICT ---")
    retire_reasons: List[str] = []
    if n < N_FLOOR:
        retire_reasons.append(
            f"n {n} < {N_FLOOR} (round-6 hard floor) -> STRUCTURAL RETIRE"
        )
    if overlap_pct is not None and overlap_pct >= SYMBOL_OVERLAP_GATE_PCT:
        retire_reasons.append(
            f"symbol-overlap {overlap_pct}% >= {SYMBOL_OVERLAP_GATE_PCT}% "
            "-> STRUCTURAL RETIRE (population unchanged)"
        )
    if pf < PF_FLOOR_MARGINAL:
        retire_reasons.append(
            f"PF {pf} < {PF_FLOOR_MARGINAL} -> THESIS RETIRE "
            "(operator-pump-unwind falsified)"
        )

    if retire_reasons:
        print("RETIRE -- gate failures:")
        for r in retire_reasons:
            print(f"  - {r}")
    elif pf >= PF_FLOOR_APPROVE:
        print(
            f"PF={pf} >= {PF_FLOOR_APPROVE} AND n={n} >= {N_FLOOR} AND "
            f"overlap < {SYMBOL_OVERLAP_GATE_PCT}% -> APPROVE detector."
        )
    else:
        print(
            f"PF={pf} in [{PF_FLOOR_MARGINAL}, {PF_FLOOR_APPROVE}) -> "
            "REVISE (probe vol-decline cutoff / cum-vol tercile)."
        )


def main():
    global _R_MULTIPLE_T1, _R_MULTIPLE_T2, _R_MULTIPLE_STOP

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--t1-r-mult", type=float, default=_R_MULTIPLE_T1,
                   help="T1 R-multiple (default 1.0)")
    p.add_argument("--t2-r-mult", type=float, default=_R_MULTIPLE_T2,
                   help="T2 R-multiple (default 2.0)")
    p.add_argument("--stop-r-mult", type=float, default=_R_MULTIPLE_STOP,
                   help="Stop R-multiple scaler (default 1.0)")
    p.add_argument("--out-suffix", default="",
                   help="suffix appended to output dir/file basename")
    args = p.parse_args()

    _R_MULTIPLE_T1 = float(args.t1_r_mult)
    _R_MULTIPLE_T2 = float(args.t2_r_mult)
    _R_MULTIPLE_STOP = float(args.stop_r_mult)

    # Sentinel-guard: refuse OOS/Holdout reads (Discovery only).
    end_d = datetime.strptime(args.end, "%Y-%m-%d").date()
    if end_d > date(2024, 12, 31):
        print("[ABORT] end past Discovery 2024-12-31; refusing OOS/Holdout reads")
        return 2
    start_d = datetime.strptime(args.start, "%Y-%m-%d").date()

    print(f"[sanity_chrs_redo] params: t1_r={_R_MULTIPLE_T1}, "
          f"t2_r={_R_MULTIPLE_T2}, stop_r={_R_MULTIPLE_STOP}, "
          f"period={start_d}..{end_d}, out_suffix={args.out_suffix!r}")

    fno_excl = load_fno_universe()
    big5m = build_full_period_5m(ALLOWED_CAPS, fno_excl)
    if big5m.empty:
        print("[ABORT] no 5m feathers found")
        return
    adv_table = load_daily_for_liquidity_and_baseline()

    print("\nFinding triggers ...")
    triggers = find_triggers(big5m, adv_table)
    print(f"\nTotal triggers (after latch): {len(triggers)}")
    if triggers.empty:
        print("[NO TRIGGERS] -- no symbols passed the retail-light + 14:30 gates.")
        print("--- VERDICT ---")
        print(
            f"n=0 < {N_FLOOR} -> STRUCTURAL RETIRE (round-6 floor failure)"
        )
        return

    print("\nSimulating entries -> exits:")
    trades = simulate(triggers, big5m)
    report(trades)

    suffix = args.out_suffix
    out_dir = _REPO_ROOT / "reports" / "sub9_sanity"
    if suffix:
        # Per-sweep output dir (matches step-4 spec).
        sweep_dir = out_dir / f"closing_hour_reversal_short_redo{suffix}"
        sweep_dir.mkdir(parents=True, exist_ok=True)
        out = sweep_dir / "trades.csv"
        # Also drop a summary.json with key metrics for aggregation.
        if not trades.empty:
            npnl = trades["net_pnl"]
            wins = float(npnl[npnl > 0].sum())
            losses = float(npnl[npnl < 0].abs().sum())
            gross = float(trades["realized_pnl"].sum())
            gross_w = float(trades["realized_pnl"][trades["realized_pnl"] > 0].sum())
            gross_l = float(
                trades["realized_pnl"][trades["realized_pnl"] < 0].abs().sum()
            )
            net_pf = (wins / losses) if losses > 0 else float("inf")
            gross_pf = (gross_w / gross_l) if gross_l > 0 else float("inf")
            daily = trades.groupby("T1_entry_date")["net_pnl"].sum()
            sharpe = (
                float(daily.mean() / daily.std()) if daily.std() > 0 else 0.0
            )
            import json as _json
            (sweep_dir / "summary.json").write_text(_json.dumps({
                "t1_r_mult": _R_MULTIPLE_T1,
                "t2_r_mult": _R_MULTIPLE_T2,
                "stop_r_mult": _R_MULTIPLE_STOP,
                "n_trades": int(len(trades)),
                "gross_pf": round(gross_pf, 4),
                "net_pf": round(net_pf, 4),
                "gross_pnl": round(gross, 2),
                "net_pnl": round(float(npnl.sum()), 2),
                "sharpe_daily": round(sharpe, 4),
                "win_rate_pct": round(float((npnl > 0).mean()) * 100, 2),
                "fees": round(float(trades["fee"].sum()), 2),
            }, indent=2, default=str))
            print(f"Sweep summary: {sweep_dir / 'summary.json'}")
    else:
        out = out_dir / "closing_hour_reversal_short_redo_trades.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out, index=False)
    print(f"\nFull trade log: {out}")


if __name__ == "__main__":
    main()
