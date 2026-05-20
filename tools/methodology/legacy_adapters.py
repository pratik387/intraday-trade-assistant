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
    """Normalize pre_results_t1_fade v2 trades CSV → canonical schema."""
    out = pd.DataFrame()

    # Required columns
    out["signal_date"] = pd.to_datetime(df["signal_date"]).dt.date.astype(str)
    out["symbol"] = _ensure_nse_prefix(df["symbol"])
    out["side"] = "SHORT"
    out["entry_price"] = df["entry_price"].astype(float)
    out["exit_price"] = df["exit_price"].astype(float)
    out["qty"] = df["qty"].astype(int)
    out["pnl_pct"] = df["pnl_pct"].astype(float)
    out["exit_reason"] = _normalize_exit_reason(
        df["exit_reason"], df["same_bar_exit"],
        mapping=_PRE_RESULTS_EXIT_REASON_MAP,
    )
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
# Adapter registry
# ---------------------------------------------------------------------------

# Maps setup_name → adapter function. Walk-forward toolkit consumers
# look up the adapter here.
ADAPTERS: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "pre_results_t1_fade": adapt_pre_results_t1_fade,
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
