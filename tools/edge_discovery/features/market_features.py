"""Market-level + cross-asset features (Tier-B).

These are computed at universe-scaffold level (one value per session per
market context) and passed into the feature module via kwargs. Caller is
responsible for the data lookup (FII/DII, USD-INR, etc. — see Phase 3 data
adapters).
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from tools.edge_discovery.types import Event


class MarketFeaturesTierB:
    name = "market_features_tier_b"
    feature_names: List[str] = [
        "nifty_intraday_pct",
        "banknifty_intraday_pct",
        "banknifty_vs_nifty_relative_strength",
        "india_vix",
        "india_vix_5d_change",
        "advance_decline_ratio",
        "fii_net_flow_t1_inr_cr",
        "dii_net_flow_t1_inr_cr",
        "nifty_futures_basis_pct",
        "usd_inr_intraday_pct",
        "crude_intraday_pct",
    ]

    def compute(
        self,
        event: Event,
        bars: pd.DataFrame,
        nifty_intraday_pct: float = np.nan,
        banknifty_intraday_pct: float = np.nan,
        india_vix: float = np.nan,
        india_vix_5d_change: float = np.nan,
        advance_decline_ratio: float = np.nan,
        fii_net_flow_t1_inr_cr: float = np.nan,
        dii_net_flow_t1_inr_cr: float = np.nan,
        nifty_futures_basis_pct: float = np.nan,
        usd_inr_intraday_pct: float = np.nan,
        crude_intraday_pct: float = np.nan,
        **_: Any,
    ) -> Dict[str, Any]:
        nfty = float(nifty_intraday_pct) if not pd.isna(nifty_intraday_pct) else np.nan
        bn = float(banknifty_intraday_pct) if not pd.isna(banknifty_intraday_pct) else np.nan
        rel = (bn - nfty) if (not pd.isna(nfty) and not pd.isna(bn)) else np.nan
        return {
            "nifty_intraday_pct": nfty,
            "banknifty_intraday_pct": bn,
            "banknifty_vs_nifty_relative_strength": rel,
            "india_vix": float(india_vix) if not pd.isna(india_vix) else np.nan,
            "india_vix_5d_change": float(india_vix_5d_change) if not pd.isna(india_vix_5d_change) else np.nan,
            "advance_decline_ratio": float(advance_decline_ratio) if not pd.isna(advance_decline_ratio) else np.nan,
            "fii_net_flow_t1_inr_cr": float(fii_net_flow_t1_inr_cr) if not pd.isna(fii_net_flow_t1_inr_cr) else np.nan,
            "dii_net_flow_t1_inr_cr": float(dii_net_flow_t1_inr_cr) if not pd.isna(dii_net_flow_t1_inr_cr) else np.nan,
            "nifty_futures_basis_pct": float(nifty_futures_basis_pct) if not pd.isna(nifty_futures_basis_pct) else np.nan,
            "usd_inr_intraday_pct": float(usd_inr_intraday_pct) if not pd.isna(usd_inr_intraday_pct) else np.nan,
            "crude_intraday_pct": float(crude_intraday_pct) if not pd.isna(crude_intraday_pct) else np.nan,
        }
