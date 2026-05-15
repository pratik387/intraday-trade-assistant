import pandas as pd
import numpy as np
from datetime import datetime
from tools.edge_discovery.types import Event
from tools.edge_discovery.features.market_features import MarketFeaturesTierB


def test_market_features_emit_expected_keys():
    event = Event(symbol="X", event_time=pd.Timestamp("2024-06-15 10:00:00"), metadata={})
    mf = MarketFeaturesTierB()
    out = mf.compute(
        event, bars=pd.DataFrame(),
        nifty_intraday_pct=0.005,
        banknifty_intraday_pct=0.007,
        india_vix=14.5,
        india_vix_5d_change=-0.02,
        advance_decline_ratio=1.4,
        fii_net_flow_t1_inr_cr=-1200.0,
        dii_net_flow_t1_inr_cr=+800.0,
        nifty_futures_basis_pct=0.0008,
        usd_inr_intraday_pct=0.001,
        crude_intraday_pct=-0.012,
    )
    for k in (
        "nifty_intraday_pct", "banknifty_intraday_pct",
        "banknifty_vs_nifty_relative_strength",
        "india_vix", "india_vix_5d_change", "advance_decline_ratio",
        "fii_net_flow_t1_inr_cr", "dii_net_flow_t1_inr_cr",
        "nifty_futures_basis_pct", "usd_inr_intraday_pct", "crude_intraday_pct",
    ):
        assert k in out, f"missing {k}"


def test_banknifty_relative_strength_is_difference():
    event = Event(symbol="X", event_time=pd.Timestamp("2024-06-15 10:00:00"), metadata={})
    mf = MarketFeaturesTierB()
    out = mf.compute(event, bars=pd.DataFrame(),
                      nifty_intraday_pct=0.005, banknifty_intraday_pct=0.012,
                      india_vix=14.5, india_vix_5d_change=0.0, advance_decline_ratio=1.0,
                      fii_net_flow_t1_inr_cr=0.0, dii_net_flow_t1_inr_cr=0.0,
                      nifty_futures_basis_pct=0.0, usd_inr_intraday_pct=0.0,
                      crude_intraday_pct=0.0)
    assert abs(out["banknifty_vs_nifty_relative_strength"] - 0.007) < 1e-9
