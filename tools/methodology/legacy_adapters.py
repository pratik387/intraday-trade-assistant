"""Per-setup adapters that convert legacy sanity CSVs to canonical schema.

Each existing setup has a slightly different sanity CSV. Rather than make
the walk-forward CLI sniff schemas, we maintain explicit per-setup adapters
here. Adding a new setup = adding a new adapter function + entry in ADAPTERS.

Each adapter:
  - Takes a raw DataFrame as read from disk
  - Returns a DataFrame conforming to the canonical schema
    (tools.methodology.sanity_csv_schema.REQUIRED_COLUMNS + optionals)
  - Does NOT validate — caller is responsible for invoking validate() after

Contract: the adapter is the ONLY place setup-specific knowledge (side,
exit_reason vocabulary, column renames) is hardcoded. Walk-forward and
combine-trades trust the canonical output.
"""
from __future__ import annotations

from typing import Callable, Dict

import pandas as pd


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

def _ensure_nse_prefix(symbol_series: pd.Series) -> pd.Series:
    """Add 'NSE:' prefix if missing. Idempotent."""
    s = symbol_series.astype(str)
    return s.where(s.str.startswith("NSE:"), "NSE:" + s)


def _normalize_exit_reason(
    reason_series: pd.Series,
    same_bar_series: pd.Series,
    *,
    mapping: Dict[str, str],
) -> pd.Series:
    """Map legacy exit_reason vocabulary to canonical.

    For 'sl' / 'stop' / 'hard_sl' style values: if same_bar=True, emit
    'same_bar_sl'; otherwise emit 'sl'. mapping should specify the bare 'sl'
    mapping (e.g., {'stop': 'sl', 't2_full': 't2', ...}).
    """
    out = reason_series.astype(str).map(mapping)
    # For rows where mapping produced 'sl' AND same_bar is True, promote to
    # 'same_bar_sl'.
    sl_mask = (out == "sl") & same_bar_series.astype(bool)
    out = out.where(~sl_mask, "same_bar_sl")
    return out


# ---------------------------------------------------------------------------
# Adapter: pre_results_t1_fade
# ---------------------------------------------------------------------------

# Source: tools/sub9_research/sanity_pre_results_t1_fade_v2.py
# Trade CSVs: reports/sub9_sanity/_pre_results_t1_v2_trades_{discovery,oos,holdout}.csv
# Columns observed (2026-05-20):
#   symbol, signal_date, announce_date, announce_class, cap_segment,
#   prior_5d_ret_pct, entry_time, entry_price, hard_sl_initial, t1_target,
#   t2_target, exit_time, exit_price, exit_reason, qty, t1_booked,
#   realized_pnl, fee, net_pnl, pnl_pct, pnl_r, same_bar_exit
#
# - side: SHORT (T-1 pre-results de-risking fade)
# - pnl_pct is already raw per-share % return (sign-aware), verified
#   2026-05-20 against (entry-exit)/entry*100 for 3 samples.
# - exit_reason values: {stop, t2_full, time_stop}.

_PRE_RESULTS_EXIT_REASON_MAP = {
    "stop": "sl",
    "t2_full": "t2",
    "time_stop": "time_stop",
}


def adapt_pre_results_t1_fade(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize pre_results_t1_fade v2 trades CSV → canonical schema.

    Critical fix (2026-05-20): the legacy v2 sanity script emits a BUGGY
    pnl_pct for trades where T1 partial was booked then SL moved to
    breakeven fired. For those trades:
      - exit_price = entry_price (breakeven hit)
      - legacy pnl_pct = (entry-exit)/entry*100 = 0   ← WRONG (misses T1 profit)
      - actual blended return = realized_pnl / (entry * qty_total) * 100

    25% of pre_results_t1 trades are in this state (515/1940 Discovery rows).
    The adapter:
      1. Detects t1_booked=True rows
      2. Recomputes pnl_pct from realized_pnl (which IS correctly blended)
      3. Emits exit_reason='breakeven_stop' for the entry==exit subset
      4. Sets t1_partial_booked=True so validator skips the sign cross-check
    """
    out = pd.DataFrame()
    n = len(df)

    # Required columns
    out["signal_date"] = pd.to_datetime(df["signal_date"]).dt.date.astype(str)
    out["symbol"] = _ensure_nse_prefix(df["symbol"])
    out["side"] = "SHORT"
    entry = df["entry_price"].astype(float)
    exit_ = df["exit_price"].astype(float)
    qty = df["qty"].astype(int)
    out["entry_price"] = entry
    out["exit_price"] = exit_
    out["qty"] = qty

    # Detect multi-leg (T1 partial booked) rows and recompute pnl_pct
    t1_booked = df["t1_booked"].astype(bool) if "t1_booked" in df.columns else pd.Series(False, index=df.index)
    realized = df["realized_pnl"].astype(float) if "realized_pnl" in df.columns else None

    out["pnl_pct"] = df["pnl_pct"].astype(float)
    if realized is not None:
        # For t1_booked rows, recompute pnl_pct as blended return.
        # blended_pnl_pct = realized_pnl / (entry_price * qty_total) * 100
        notional = entry * qty
        blended = realized / notional * 100.0
        out.loc[t1_booked, "pnl_pct"] = blended.loc[t1_booked].values

    out["t1_partial_booked"] = t1_booked

    # Normalize exit_reason. For t1_booked rows with entry==exit and legacy
    # exit_reason='stop', emit 'breakeven_stop'. For others, use the normal map.
    base_reason = _normalize_exit_reason(
        df["exit_reason"], df["same_bar_exit"],
        mapping=_PRE_RESULTS_EXIT_REASON_MAP,
    )
    # breakeven_stop: t1_booked=True AND entry==exit AND legacy=='stop'
    bes_mask = t1_booked & (entry == exit_) & (df["exit_reason"] == "stop")
    base_reason = base_reason.where(~bes_mask, "breakeven_stop")
    out["exit_reason"] = base_reason
    out["same_bar"] = df["same_bar_exit"].astype(bool)

    # Optional columns (passthrough with renames)
    if "cap_segment" in df.columns:
        out["cap_segment"] = df["cap_segment"]
    if "entry_time" in df.columns:
        out["entry_ts"] = df["entry_time"]
    if "exit_time" in df.columns:
        out["exit_ts"] = df["exit_time"]
    if "realized_pnl" in df.columns:
        out["realized_pnl_inr"] = df["realized_pnl"].astype(float)
    if "fee" in df.columns:
        out["fee_inr"] = df["fee"].astype(float)
    if "net_pnl" in df.columns:
        out["net_pnl_inr"] = df["net_pnl"].astype(float)
    if "pnl_r" in df.columns:
        out["r_multiple"] = df["pnl_r"].astype(float)
    if "t1_target" in df.columns:
        out["t1_target"] = df["t1_target"].astype(float)
    if "t2_target" in df.columns:
        out["t2_target"] = df["t2_target"].astype(float)
    if "hard_sl_initial" in df.columns:
        out["hard_sl"] = df["hard_sl_initial"].astype(float)

    return out


# ---------------------------------------------------------------------------
# Adapter: capitulation_long_v2
# ---------------------------------------------------------------------------

# Source: tools/sub9_research/sanity_capitulation_long_v2.py
# Trade CSVs: reports/sub9_sanity/_capitulation_long_v2_trades_{discovery,oos,holdout}.csv
# Columns observed (2026-05-20):
#   trade_date, signal_ts, entry_ts, exit_ts, symbol, cap_segment, gap_pct,
#   entry_price, hard_sl, t1_target, t2_target, exit_price, exit_reason,
#   r_multiple, qty, realized_pnl, fee, net_pnl, same_bar, had_t1_partial
#
# - side: LONG (gap-down exhaustion fade LONG-side)
# - had_t1_partial is ALWAYS False in the v2 sanity outputs (no T1 partial
#   logic). No blended-pnl bug like pre_results_t1.
# - pnl_pct must be computed from entry/exit (not in CSV).
# - exit_reason values: stop, t2_full, time_stop, last_bar
# - 'last_bar' = path-walk-end fallback (sanity emits this when no other
#   condition fires; the last bar's close is used as exit). Treat as 'eod'.

_CAPITULATION_LONG_V2_EXIT_REASON_MAP = {
    "stop": "sl",
    "t2_full": "t2",
    "time_stop": "time_stop",
    "last_bar": "eod",
}


def adapt_capitulation_long_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize capitulation_long_v2 trades CSV → canonical schema.

    LONG-side gap-down exhaustion fade. No T1 partial logic in v2; pnl_pct
    computed cleanly from entry/exit with LONG sign convention.
    """
    out = pd.DataFrame()

    # Required columns
    out["signal_date"] = pd.to_datetime(df["trade_date"]).dt.date.astype(str)
    out["symbol"] = _ensure_nse_prefix(df["symbol"])
    out["side"] = "LONG"
    entry = df["entry_price"].astype(float)
    exit_ = df["exit_price"].astype(float)
    qty = df["qty"].astype(int)
    out["entry_price"] = entry
    out["exit_price"] = exit_
    out["qty"] = qty
    # LONG: pnl_pct = (exit - entry) / entry * 100
    out["pnl_pct"] = (exit_ - entry) / entry * 100.0

    out["exit_reason"] = _normalize_exit_reason(
        df["exit_reason"], df["same_bar"],
        mapping=_CAPITULATION_LONG_V2_EXIT_REASON_MAP,
    )
    out["same_bar"] = df["same_bar"].astype(bool)

    # Optional columns (passthrough with renames)
    if "cap_segment" in df.columns:
        out["cap_segment"] = df["cap_segment"]
    if "signal_ts" in df.columns:
        out["signal_ts"] = df["signal_ts"]
    if "entry_ts" in df.columns:
        out["entry_ts"] = df["entry_ts"]
    if "exit_ts" in df.columns:
        out["exit_ts"] = df["exit_ts"]
    if "realized_pnl" in df.columns:
        out["realized_pnl_inr"] = df["realized_pnl"].astype(float)
    if "fee" in df.columns:
        out["fee_inr"] = df["fee"].astype(float)
    if "net_pnl" in df.columns:
        out["net_pnl_inr"] = df["net_pnl"].astype(float)
    if "r_multiple" in df.columns:
        out["r_multiple"] = df["r_multiple"].astype(float)
    if "t1_target" in df.columns:
        out["t1_target"] = df["t1_target"].astype(float)
    if "t2_target" in df.columns:
        out["t2_target"] = df["t2_target"].astype(float)
    if "hard_sl" in df.columns:
        out["hard_sl"] = df["hard_sl"].astype(float)

    return out


# ---------------------------------------------------------------------------
# Adapter: mis_unwind_vwap_revert_short
# ---------------------------------------------------------------------------

# Trade CSVs: reports/sub9_sanity/_mis_unwind_locked_trades_{discovery,oos,holdout}.csv
# Columns observed (2026-05-20):
#   trade_date, signal_ts, symbol, cap_segment, rsi, vol_ratio, vwap_ext_pct,
#   entry_ts, entry_price, exit_ts, exit_price, exit_reason, same_bar,
#   R_per_share, r_multiple, qty, realized_pnl, fee, net_pnl, ym
#
# - side: SHORT (MIS auto-square-off retail-unwind fade in last 15min)
# - No T1 partial logic; no blended-pnl bug.
# - exit_reason values: stop, t2_full, time_stop
# - High same_bar rate (~71% Discovery) — confirms tight-SL fragility
#   noted in retired_setups.md retirement evidence.

_MIS_UNWIND_EXIT_REASON_MAP = {
    "stop": "sl",
    "t2_full": "t2",
    "time_stop": "time_stop",
}


def adapt_mis_unwind_vwap_revert_short(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize mis_unwind_vwap_revert_short trades CSV → canonical schema.

    SHORT-side late-session MIS auto-square unwind fade. No T1 partial logic.
    pnl_pct computed from entry/exit with SHORT sign convention.
    """
    out = pd.DataFrame()

    out["signal_date"] = pd.to_datetime(df["trade_date"]).dt.date.astype(str)
    out["symbol"] = _ensure_nse_prefix(df["symbol"])
    out["side"] = "SHORT"
    entry = df["entry_price"].astype(float)
    exit_ = df["exit_price"].astype(float)
    qty = df["qty"].astype(int)
    out["entry_price"] = entry
    out["exit_price"] = exit_
    out["qty"] = qty
    # SHORT: pnl_pct = (entry - exit) / entry * 100
    out["pnl_pct"] = (entry - exit_) / entry * 100.0

    out["exit_reason"] = _normalize_exit_reason(
        df["exit_reason"], df["same_bar"],
        mapping=_MIS_UNWIND_EXIT_REASON_MAP,
    )
    out["same_bar"] = df["same_bar"].astype(bool)

    # Optional passthrough
    if "cap_segment" in df.columns:
        out["cap_segment"] = df["cap_segment"]
    if "signal_ts" in df.columns:
        out["signal_ts"] = df["signal_ts"]
    if "entry_ts" in df.columns:
        out["entry_ts"] = df["entry_ts"]
    if "exit_ts" in df.columns:
        out["exit_ts"] = df["exit_ts"]
    if "realized_pnl" in df.columns:
        out["realized_pnl_inr"] = df["realized_pnl"].astype(float)
    if "fee" in df.columns:
        out["fee_inr"] = df["fee"].astype(float)
    if "net_pnl" in df.columns:
        out["net_pnl_inr"] = df["net_pnl"].astype(float)
    if "r_multiple" in df.columns:
        out["r_multiple"] = df["r_multiple"].astype(float)

    return out


# ---------------------------------------------------------------------------
# Adapter: circuit_release_fade_short
# ---------------------------------------------------------------------------

# Trade CSVs: reports/sub9_sanity/_circuit_release_fade_short_trades_{discovery,oos,holdout}.csv
# (base files — the _CLEAN/_MODE_A/_PHASE_A/_PHASE_B2 variants are intermediate
# experimentation outputs and NOT used here)
#
# Columns observed (2026-05-20):
#   trade_date, signal_ts, signal_close, symbol, side, signal_type,
#   day_high, day_low, day_close, pdc, day_gain_pct, close_off_high_pct,
#   entry_ts, entry_price, rejection_high, hard_sl, t1_target, t2_target,
#   R_per_share, qty, exit_ts, exit_price, exit_reason, mfe_r, mae_r,
#   close_at_1300, close_at_1400, close_at_1500, realized_pnl, fee, net_pnl,
#   cap_segment
#
# - side: SHORT (explicit column; verified)
# - No T1 partial logic; no blended-pnl bug.
# - No same_bar column — DERIVED from entry_ts == exit_ts
# - exit_reason values: stop, t2_full, time_stop, last_bar (path-walk-end)

_CIRCUIT_RELEASE_EXIT_REASON_MAP = {
    "stop": "sl",
    "t2_full": "t2",
    "time_stop": "time_stop",
    "last_bar": "eod",
}


def adapt_circuit_release_fade_short(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize circuit_release_fade_short trades CSV → canonical schema.

    SHORT-side failed-circuit-retest fade. No T1 partial logic.
    same_bar derived from entry_ts == exit_ts (sanity script doesn't emit it).
    """
    out = pd.DataFrame()

    out["signal_date"] = pd.to_datetime(df["trade_date"]).dt.date.astype(str)
    out["symbol"] = _ensure_nse_prefix(df["symbol"])
    # Trust the explicit side column (verified SHORT in samples)
    out["side"] = df["side"].astype(str).str.upper()
    entry = df["entry_price"].astype(float)
    exit_ = df["exit_price"].astype(float)
    qty = df["qty"].astype(int)
    out["entry_price"] = entry
    out["exit_price"] = exit_
    out["qty"] = qty
    # SHORT: pnl_pct = (entry - exit) / entry * 100
    out["pnl_pct"] = (entry - exit_) / entry * 100.0

    # Derive same_bar from entry_ts == exit_ts (script doesn't emit it)
    same_bar = pd.to_datetime(df["entry_ts"]) == pd.to_datetime(df["exit_ts"])

    out["exit_reason"] = _normalize_exit_reason(
        df["exit_reason"], same_bar,
        mapping=_CIRCUIT_RELEASE_EXIT_REASON_MAP,
    )
    out["same_bar"] = same_bar.astype(bool)

    # Optional passthrough
    if "cap_segment" in df.columns:
        out["cap_segment"] = df["cap_segment"]
    if "signal_ts" in df.columns:
        out["signal_ts"] = df["signal_ts"]
    if "entry_ts" in df.columns:
        out["entry_ts"] = df["entry_ts"]
    if "exit_ts" in df.columns:
        out["exit_ts"] = df["exit_ts"]
    if "realized_pnl" in df.columns:
        out["realized_pnl_inr"] = df["realized_pnl"].astype(float)
    if "fee" in df.columns:
        out["fee_inr"] = df["fee"].astype(float)
    if "net_pnl" in df.columns:
        out["net_pnl_inr"] = df["net_pnl"].astype(float)
    if "t1_target" in df.columns:
        out["t1_target"] = df["t1_target"].astype(float)
    if "t2_target" in df.columns:
        out["t2_target"] = df["t2_target"].astype(float)
    if "hard_sl" in df.columns:
        out["hard_sl"] = df["hard_sl"].astype(float)

    return out


# ---------------------------------------------------------------------------
# Adapter: capitulation_long_morning
# ---------------------------------------------------------------------------

# Trade CSVs: reports/sub9_sanity/capitulation_long_morning_trades.csv (Disc only,
#             Jan 2023 - Dec 2024) + ..._trades_holdout.csv (Oct 2025 - Apr 2026)
# NO OOS file — walk-forward will have empty windows for Jan-Sep 2025.
#
# Columns observed (2026-05-20):
#   T0_signal_date, symbol, bare_symbol, cap_segment, side, news_status,
#   gap_pct, pdc, open_09_15, low_09_15, vol_09_15, adv_20d_cr,
#   confirmation_ts, confirmation_hhmm, lower_wick_ratio, body_size_pct,
#   entry_ts, entry_price, hard_sl, atr_proxy, t1_target, t2_target,
#   hit_t1, exit_ts, exit_price, exit_reason, stop_distance, qty,
#   realized_pnl, fee, net_pnl, _month
#
# - side: LONG (explicit column)
# - symbol: ALREADY has NSE: prefix
# - hit_t1 marks T1-partial-booked trades, but UNLIKE pre_results_t1, the
#   sanity script encodes the BLENDED outcome in exit_price (effective
#   weighted-avg exit). So (exit-entry)/entry*100 already gives the correct
#   blended return for LONG. Verified 2026-05-20: row 0 breakeven_trail
#   pnl matches T1-leg-only profit (the breakeven leg contributes 0).
# - exit_reason 'breakeven_trail' = T1 booked, SL moved to BE, BE hit.
#   Maps to canonical 'breakeven_stop' (requires t1_partial_booked=True).

_CAPITULATION_LONG_MORNING_EXIT_REASON_MAP = {
    "stop": "sl",
    "t2": "t2",
    "time_stop": "time_stop",
    "breakeven_trail": "breakeven_stop",
}


def adapt_capitulation_long_morning(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize capitulation_long_morning trades CSV → canonical schema.

    LONG-side gap-down panic exhaustion fade. Has T1 partial logic, but
    sanity script encodes blended outcome in exit_price (unlike v2 sanity
    pre_results_t1 which has the buggy entry==exit pattern). So pnl_pct
    computed from prices is already correct; no recomputation needed.
    """
    out = pd.DataFrame()

    out["signal_date"] = pd.to_datetime(df["T0_signal_date"]).dt.date.astype(str)
    # Symbol already has NSE: prefix
    out["symbol"] = _ensure_nse_prefix(df["symbol"])
    out["side"] = df["side"].astype(str).str.upper()
    entry = df["entry_price"].astype(float)
    exit_ = df["exit_price"].astype(float)
    qty = df["qty"].astype(int)
    out["entry_price"] = entry
    out["exit_price"] = exit_
    out["qty"] = qty
    # LONG: pnl_pct = (exit - entry) / entry * 100
    # The sanity script encodes blended T1+exit in exit_price, so this is
    # already the correct blended return.
    out["pnl_pct"] = (exit_ - entry) / entry * 100.0

    # Derive same_bar from entry_ts == exit_ts (script doesn't emit it)
    same_bar = pd.to_datetime(df["entry_ts"]) == pd.to_datetime(df["exit_ts"])

    out["exit_reason"] = _normalize_exit_reason(
        df["exit_reason"], same_bar,
        mapping=_CAPITULATION_LONG_MORNING_EXIT_REASON_MAP,
    )
    out["same_bar"] = same_bar.astype(bool)

    # T1 partial booking flag (hit_t1 == True). Required by canonical schema
    # for any breakeven_stop exit_reason row.
    if "hit_t1" in df.columns:
        out["t1_partial_booked"] = df["hit_t1"].astype(bool)

    # Optional passthrough
    if "cap_segment" in df.columns:
        out["cap_segment"] = df["cap_segment"]
    if "confirmation_ts" in df.columns:
        out["signal_ts"] = df["confirmation_ts"]
    if "entry_ts" in df.columns:
        out["entry_ts"] = df["entry_ts"]
    if "exit_ts" in df.columns:
        out["exit_ts"] = df["exit_ts"]
    if "realized_pnl" in df.columns:
        out["realized_pnl_inr"] = df["realized_pnl"].astype(float)
    if "fee" in df.columns:
        out["fee_inr"] = df["fee"].astype(float)
    if "net_pnl" in df.columns:
        out["net_pnl_inr"] = df["net_pnl"].astype(float)
    if "t1_target" in df.columns:
        out["t1_target"] = df["t1_target"].astype(float)
    if "t2_target" in df.columns:
        out["t2_target"] = df["t2_target"].astype(float)
    if "hard_sl" in df.columns:
        out["hard_sl"] = df["hard_sl"].astype(float)

    return out


# ---------------------------------------------------------------------------
# Adapter: delivery_pct_anomaly_short (active)
# ---------------------------------------------------------------------------

# Trade CSVs: reports/sub9_sanity/nse_delivery_pct_anomaly_trades{,_oos,_holdout}.csv
# Note: file prefix is "nse_delivery_pct_anomaly" but config setup name is
# "delivery_pct_anomaly_short". Sanity is BIDIRECTIONAL (both LONG and SHORT
# trades) but the SHIPPED active setup is SHORT only. Adapter filters to
# SHORT only to match production.
#
# Columns observed (2026-05-20):
#   symbol, t_day, t1_date, side, t_day_close, open_09_15, high_09_15,
#   low_09_15, gap_pct, entry_ts, entry_price, hard_sl, stop_distance,
#   t1_target, t2_target, qty, hit_t1, t1_exit_price, exit_ts, exit_price,
#   exit_reason, gross_pnl, fee, net_pnl, mfe_r, mae_r,
#   close_at_1100..1525, signal_type, delivery_pct, daily_return_pct, adv_20d_cr
#
# - t_day: observation day (delivery_pct anomaly detected)
# - t1_date: trade day (next-day fade) — use as signal_date
# - hit_t1: T1 partial booking flag
# - exit_reasons: stop, t2, time_stop, eod
# - gross_pnl: rupee gross (not "realized_pnl" — different name from other adapters)

_DELIVERY_PCT_EXIT_REASON_MAP = {
    "stop": "sl",
    "t2": "t2",
    "time_stop": "time_stop",
    "eod": "eod",
}


def adapt_delivery_pct_anomaly_short(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize delivery_pct_anomaly trades CSV → canonical schema.

    Filters to SHORT trades only (matches shipped production setup
    setups.delivery_pct_anomaly_short).
    """
    # Filter to SHORT only
    df = df[df["side"].astype(str).str.upper() == "SHORT"].reset_index(drop=True)

    out = pd.DataFrame()
    out["signal_date"] = pd.to_datetime(df["t1_date"]).dt.date.astype(str)
    out["symbol"] = _ensure_nse_prefix(df["symbol"])
    out["side"] = "SHORT"
    entry = df["entry_price"].astype(float)
    exit_ = df["exit_price"].astype(float)
    qty = df["qty"].astype(int)
    out["entry_price"] = entry
    out["exit_price"] = exit_
    out["qty"] = qty
    # SHORT: pnl_pct = (entry - exit) / entry * 100
    # The sanity script encodes blended T1+exit in effective exit_price
    # (verified pattern matches capitulation_long_morning).
    out["pnl_pct"] = (entry - exit_) / entry * 100.0

    # Derive same_bar from entry_ts == exit_ts
    same_bar = pd.to_datetime(df["entry_ts"]) == pd.to_datetime(df["exit_ts"])

    out["exit_reason"] = _normalize_exit_reason(
        df["exit_reason"], same_bar,
        mapping=_DELIVERY_PCT_EXIT_REASON_MAP,
    )
    out["same_bar"] = same_bar.astype(bool)

    if "hit_t1" in df.columns:
        out["t1_partial_booked"] = df["hit_t1"].astype(bool)

    # Optional passthrough
    if "entry_ts" in df.columns:
        out["entry_ts"] = df["entry_ts"]
    if "exit_ts" in df.columns:
        out["exit_ts"] = df["exit_ts"]
    if "gross_pnl" in df.columns:
        out["realized_pnl_inr"] = df["gross_pnl"].astype(float)
    if "fee" in df.columns:
        out["fee_inr"] = df["fee"].astype(float)
    if "net_pnl" in df.columns:
        out["net_pnl_inr"] = df["net_pnl"].astype(float)
    if "t1_target" in df.columns:
        out["t1_target"] = df["t1_target"].astype(float)
    if "t2_target" in df.columns:
        out["t2_target"] = df["t2_target"].astype(float)
    if "hard_sl" in df.columns:
        out["hard_sl"] = df["hard_sl"].astype(float)

    return out


# ---------------------------------------------------------------------------
# Adapter: long_panic_gap_down (active)
# ---------------------------------------------------------------------------

# Trade CSVs: reports/sub9_sanity/long_panic_gap_down_trades{,_oos,_holdout}.csv
# Note: sanity outputs are AGGREGATE (10395+5094+1835 = 17324 trades). The
# shipped production setup is cell-locked to dist_from_pdl [-5%, -3%] which
# yields ~1412+333+126 = 1871 trades. Canonical walk-forward here reflects
# AGGREGATE, not the production cell.
#
# - side: LONG only
# - No T1 partial logic (no hit_t1 column)
# - exit_reason 't2_full_gap_fill' = T2 hit via gap-fill; maps to canonical t2

_LONG_PANIC_EXIT_REASON_MAP = {
    "stop": "sl",
    "time_stop": "time_stop",
    "t2_full_gap_fill": "t2",
    "last_bar": "eod",
}


def adapt_long_panic_gap_down(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize long_panic_gap_down trades CSV → canonical schema.

    LONG-side gap-down panic exhaustion fade with PDL-relative entry zone.
    Canonical output reflects AGGREGATE sanity (no cell filter); production
    uses narrower dist_from_pdl cell.
    """
    out = pd.DataFrame()
    out["signal_date"] = pd.to_datetime(df["T0_signal_date"]).dt.date.astype(str)
    out["symbol"] = _ensure_nse_prefix(df["symbol"])
    out["side"] = df["side"].astype(str).str.upper()
    entry = df["entry_price"].astype(float)
    exit_ = df["exit_price"].astype(float)
    qty = df["qty"].astype(int)
    out["entry_price"] = entry
    out["exit_price"] = exit_
    out["qty"] = qty
    out["pnl_pct"] = (exit_ - entry) / entry * 100.0  # LONG

    same_bar = pd.to_datetime(df["entry_ts"]) == pd.to_datetime(df["exit_ts"])
    out["exit_reason"] = _normalize_exit_reason(
        df["exit_reason"], same_bar,
        mapping=_LONG_PANIC_EXIT_REASON_MAP,
    )
    out["same_bar"] = same_bar.astype(bool)

    # Optional passthrough
    if "cap_segment" in df.columns:
        out["cap_segment"] = df["cap_segment"]
    if "entry_ts" in df.columns:
        out["entry_ts"] = df["entry_ts"]
    if "exit_ts" in df.columns:
        out["exit_ts"] = df["exit_ts"]
    if "realized_pnl" in df.columns:
        out["realized_pnl_inr"] = df["realized_pnl"].astype(float)
    if "fee" in df.columns:
        out["fee_inr"] = df["fee"].astype(float)
    if "net_pnl" in df.columns:
        out["net_pnl_inr"] = df["net_pnl"].astype(float)
    if "t1_target" in df.columns:
        out["t1_target"] = df["t1_target"].astype(float)
    if "t2_target" in df.columns:
        out["t2_target"] = df["t2_target"].astype(float)
    if "hard_sl" in df.columns:
        out["hard_sl"] = df["hard_sl"].astype(float)

    return out


# ---------------------------------------------------------------------------
# Adapter: or_window_failure_fade_short (active)
# ---------------------------------------------------------------------------

# Trade CSVs: reports/sub9_sanity/_or_window_failure_fade_trades_{discovery,oos,holdout}.csv
# Note: file prefix is "or_window_failure_fade" (no "_short"). Sanity is
# BIDIRECTIONAL (13654 LONG + 11340 SHORT in disc). Adapter filters to
# SHORT-only (matches shipped setups.or_window_failure_fade_short).
#
# - exit_reason: stop, time_stop, t2_full, last_bar
# - No T1 partial logic
# - signal_date = trade_date

_OR_WINDOW_EXIT_REASON_MAP = {
    "stop": "sl",
    "t2_full": "t2",
    "time_stop": "time_stop",
    "last_bar": "eod",
}


def adapt_or_window_failure_fade_short(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize or_window_failure_fade trades CSV → canonical schema.

    Filters to SHORT trades (matches shipped production setup
    setups.or_window_failure_fade_short).
    """
    df = df[df["side"].astype(str).str.upper() == "SHORT"].reset_index(drop=True)
    out = pd.DataFrame()
    out["signal_date"] = pd.to_datetime(df["trade_date"]).dt.date.astype(str)
    out["symbol"] = _ensure_nse_prefix(df["symbol"])
    out["side"] = "SHORT"
    entry = df["entry_price"].astype(float)
    exit_ = df["exit_price"].astype(float)
    qty = df["qty"].astype(int)
    out["entry_price"] = entry
    out["exit_price"] = exit_
    out["qty"] = qty
    out["pnl_pct"] = (entry - exit_) / entry * 100.0  # SHORT

    same_bar = pd.to_datetime(df["entry_ts"]) == pd.to_datetime(df["exit_ts"])
    out["exit_reason"] = _normalize_exit_reason(
        df["exit_reason"], same_bar,
        mapping=_OR_WINDOW_EXIT_REASON_MAP,
    )
    out["same_bar"] = same_bar.astype(bool)

    if "cap_segment" in df.columns:
        out["cap_segment"] = df["cap_segment"]
    if "entry_ts" in df.columns:
        out["entry_ts"] = df["entry_ts"]
    if "exit_ts" in df.columns:
        out["exit_ts"] = df["exit_ts"]
    if "realized_pnl" in df.columns:
        out["realized_pnl_inr"] = df["realized_pnl"].astype(float)
    if "fee" in df.columns:
        out["fee_inr"] = df["fee"].astype(float)
    if "net_pnl" in df.columns:
        out["net_pnl_inr"] = df["net_pnl"].astype(float)
    if "t1_target" in df.columns:
        out["t1_target"] = df["t1_target"].astype(float)
    if "t2_target" in df.columns:
        out["t2_target"] = df["t2_target"].astype(float)
    if "hard_sl" in df.columns:
        out["hard_sl"] = df["hard_sl"].astype(float)

    return out


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

# Maps setup_name → adapter function. Walk-forward toolkit consumers
# look up the adapter here.
ADAPTERS: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "pre_results_t1_fade": adapt_pre_results_t1_fade,
    "capitulation_long_v2": adapt_capitulation_long_v2,
    "mis_unwind_vwap_revert_short": adapt_mis_unwind_vwap_revert_short,
    "circuit_release_fade_short": adapt_circuit_release_fade_short,
    "capitulation_long_morning": adapt_capitulation_long_morning,
    "delivery_pct_anomaly_short": adapt_delivery_pct_anomaly_short,
    "long_panic_gap_down": adapt_long_panic_gap_down,
    "or_window_failure_fade_short": adapt_or_window_failure_fade_short,
}


def get_adapter(setup_name: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Look up the adapter for a setup. Raises KeyError with helpful message."""
    if setup_name not in ADAPTERS:
        raise KeyError(
            f"No legacy adapter registered for setup {setup_name!r}. "
            f"Registered: {sorted(ADAPTERS.keys())}. "
            f"Add an adapter in tools/methodology/legacy_adapters.py."
        )
    return ADAPTERS[setup_name]
