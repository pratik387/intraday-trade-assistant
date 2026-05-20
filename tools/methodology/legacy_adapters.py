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
# Adapter registry
# ---------------------------------------------------------------------------

# Maps setup_name → adapter function. Walk-forward toolkit consumers
# look up the adapter here.
ADAPTERS: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "pre_results_t1_fade": adapt_pre_results_t1_fade,
    "capitulation_long_v2": adapt_capitulation_long_v2,
    "mis_unwind_vwap_revert_short": adapt_mis_unwind_vwap_revert_short,
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
