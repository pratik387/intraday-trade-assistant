import pandas as pd
from tools.edge_discovery.outcomes.costs import ExecutionCosts


def _config_block() -> dict:
    return {
        "spread_by_cap_adv": {
            "small_cap": {"adv_lt_100k": 0.0015, "adv_100k_500k": 0.0008,
                          "adv_500k_2m": 0.0005, "adv_gt_2m": 0.0003},
            "mid_cap":   {"adv_lt_100k": 0.0008, "adv_100k_500k": 0.0005,
                          "adv_500k_2m": 0.0003, "adv_gt_2m": 0.0002},
            "large_cap": {"adv_lt_100k": 0.0004, "adv_100k_500k": 0.0003,
                          "adv_500k_2m": 0.0002, "adv_gt_2m": 0.0001},
        },
        "sl_slippage_bar_fraction": 0.5,
        "sl_slippage_normal_pct": 0.001,
        "market_impact_pct_per_pct_adv": 0.05,
        "market_impact_cap_pct": 0.05,
    }


def test_spread_lookup_small_cap_illiquid_returns_15bps():
    costs = ExecutionCosts(_config_block())
    s = costs.spread_pct(cap_segment="small_cap", adv_shares=50_000)
    assert abs(s - 0.0015) < 1e-9


def test_spread_lookup_large_cap_liquid_returns_1bps():
    costs = ExecutionCosts(_config_block())
    s = costs.spread_pct(cap_segment="large_cap", adv_shares=5_000_000)
    assert abs(s - 0.0001) < 1e-9


def test_market_impact_linear_then_capped():
    costs = ExecutionCosts(_config_block())
    assert abs(costs.market_impact_pct(order_shares=1000, adv_shares=100_000) - 0.0005) < 1e-9
    assert abs(costs.market_impact_pct(order_shares=2000, adv_shares=100_000) - 0.0010) < 1e-9
    assert abs(costs.market_impact_pct(order_shares=200_000, adv_shares=100_000) - 0.05) < 1e-9


def test_apply_round_trip_subtracts_both_sides():
    costs = ExecutionCosts(_config_block())
    net = costs.apply_round_trip(
        gross_return_pct=0.01,
        cap_segment="small_cap", adv_shares=50_000, order_shares=100,
        sl_hit=False, sl_bar_range_pct=None,
    )
    # 2 * 0.0015 spread + 2 * (100/50000 * 0.05 = 0.0001) impact = 0.0032
    assert abs(net - 0.0068) < 1e-6


def test_apply_round_trip_with_sl_hit_adds_slippage():
    costs = ExecutionCosts(_config_block())
    net = costs.apply_round_trip(
        gross_return_pct=-0.01,
        cap_segment="small_cap", adv_shares=50_000, order_shares=100,
        sl_hit=True, sl_bar_range_pct=0.02,
    )
    # gross -1%; spread 0.3%; impact ~0.02%; SL slip = 0.5 * 2% = 1%
    # net = -0.01 - 0.003 - 0.0002 - 0.01 = -0.0232
    assert abs(net - (-0.0232)) < 1e-6
