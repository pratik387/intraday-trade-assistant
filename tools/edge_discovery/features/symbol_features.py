"""Tier-A symbol-level features: chart + cap + ADV + gap + bar shape.

Tier-A means: derivable from existing 5m feathers + nse_all.json + prior-day
OHLC without needing FII/DII/USD-INR/Crude/Calendar pipelines.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from tools.edge_discovery.types import Event


def _adv_bucket(adv: float) -> str:
    if adv < 100_000:
        return "adv_lt_100k"
    if adv < 500_000:
        return "adv_100k_500k"
    if adv < 2_000_000:
        return "adv_500k_2m"
    return "adv_gt_2m"


class SymbolFeaturesTierA:
    """Computes Tier-A symbol-level features for one event."""

    name = "symbol_features_tier_a"
    feature_names: List[str] = [
        "cap_segment",
        "adv_bucket",
        "mis_leverage",
        "dist_from_pdh_pct",
        "dist_from_pdl_pct",
        "prior_session_pct_change",
        "gap_pct",
        "bar_range_pct",
        "bar_body_pct",
        "bar_upper_wick_ratio",
        "bar_lower_wick_ratio",
        "vwap_distance_pct",
        "dist_from_20ema_pct",
        "dist_from_50ema_pct",
        "delivery_pct_t1",
    ]

    def compute(
        self,
        event: Event,
        bars: pd.DataFrame,
        symbol_meta: Optional[Dict[str, Any]] = None,
        pdh: Optional[float] = None,
        pdl: Optional[float] = None,
        prior_close: Optional[float] = None,
        adv_shares: Optional[float] = None,
        ema_20: Optional[float] = None,
        ema_50: Optional[float] = None,
        delivery_pct_t1: Optional[float] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        symbol_meta = symbol_meta or {}
        entry_idx_arr = bars.index[bars["date"] == event.event_time]
        if len(entry_idx_arr) == 0:
            entry_idx_arr = bars.index[bars["date"] >= event.event_time]
        if len(entry_idx_arr) == 0:
            return self._empty(symbol_meta)
        entry_i = int(entry_idx_arr[0])
        entry_bar = bars.iloc[entry_i]
        entry_close = float(entry_bar["close"])
        day_floor = event.event_time.floor("D")
        day_open_row = bars[bars["date"].dt.floor("D") == day_floor].iloc[0]
        day_open = float(day_open_row["open"])

        # bar shape
        bar_high, bar_low = float(entry_bar["high"]), float(entry_bar["low"])
        bar_open, bar_close = float(entry_bar["open"]), float(entry_bar["close"])
        bar_range = bar_high - bar_low
        bar_body = abs(bar_close - bar_open)
        upper_wick = bar_high - max(bar_open, bar_close)
        lower_wick = min(bar_open, bar_close) - bar_low
        range_pct = bar_range / entry_close if entry_close else 0.0
        body_pct = bar_body / entry_close if entry_close else 0.0
        upper_wick_ratio = (upper_wick / bar_range) if bar_range > 0 else 0.0
        lower_wick_ratio = (lower_wick / bar_range) if bar_range > 0 else 0.0

        out: Dict[str, Any] = {
            "cap_segment": symbol_meta.get("cap_segment", "unknown"),
            "adv_bucket": _adv_bucket(float(adv_shares) if adv_shares else 0.0),
            "mis_leverage": float(symbol_meta.get("mis_leverage", 1.0) or 1.0),
            "bar_range_pct": range_pct,
            "bar_body_pct": body_pct,
            "bar_upper_wick_ratio": upper_wick_ratio,
            "bar_lower_wick_ratio": lower_wick_ratio,
        }
        if pdh is not None and pdh > 0:
            out["dist_from_pdh_pct"] = (entry_close - pdh) / pdh
        else:
            out["dist_from_pdh_pct"] = np.nan
        if pdl is not None and pdl > 0:
            out["dist_from_pdl_pct"] = (entry_close - pdl) / pdl
        else:
            out["dist_from_pdl_pct"] = np.nan
        if prior_close is not None and prior_close > 0:
            out["gap_pct"] = (day_open - prior_close) / prior_close
            out["prior_session_pct_change"] = (prior_close - day_open) / day_open if day_open > 0 else np.nan
        else:
            out["gap_pct"] = np.nan
            out["prior_session_pct_change"] = np.nan

        # Session VWAP up to entry bar (volume-weighted typical price)
        same_day = bars[bars["date"].dt.floor("D") == day_floor]
        upto_entry = same_day[same_day["date"] <= event.event_time]
        if len(upto_entry) > 0 and upto_entry["volume"].sum() > 0:
            typical = (upto_entry["high"] + upto_entry["low"] + upto_entry["close"]) / 3.0
            vwap = float((typical * upto_entry["volume"]).sum() / upto_entry["volume"].sum())
            out["vwap_distance_pct"] = (entry_close - vwap) / vwap if vwap > 0 else np.nan
        else:
            out["vwap_distance_pct"] = np.nan

        # EMA distances passed in via kwargs (computed at universe-scaffold level)
        out["dist_from_20ema_pct"] = (entry_close - ema_20) / ema_20 if (ema_20 and ema_20 > 0) else np.nan
        out["dist_from_50ema_pct"] = (entry_close - ema_50) / ema_50 if (ema_50 and ema_50 > 0) else np.nan
        out["delivery_pct_t1"] = float(delivery_pct_t1) if delivery_pct_t1 is not None else np.nan
        return out

    def _empty(self, symbol_meta: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "cap_segment": symbol_meta.get("cap_segment", "unknown"),
            "adv_bucket": "adv_lt_100k",
            "mis_leverage": float(symbol_meta.get("mis_leverage", 1.0) or 1.0),
            **{k: np.nan for k in (
                "dist_from_pdh_pct", "dist_from_pdl_pct",
                "prior_session_pct_change", "gap_pct",
                "bar_range_pct", "bar_body_pct",
                "bar_upper_wick_ratio", "bar_lower_wick_ratio",
                "vwap_distance_pct",
                "dist_from_20ema_pct", "dist_from_50ema_pct",
                "delivery_pct_t1",
            )},
        }
