import pandas as pd

from tools.edge_discovery.decay_monitor_runner import compute_monthly_pf_from_trades


def test_monthly_pf_from_trade_pnl_series():
    trades = pd.DataFrame({
        "entry_time": pd.to_datetime([
            "2025-10-05", "2025-10-10", "2025-10-20",  # 2 wins, 1 loss
            "2025-11-05", "2025-11-10",                # 1 win, 1 loss
        ]),
        "net_return": [0.02, 0.03, -0.01, 0.02, -0.03],
    })
    monthly_pf = compute_monthly_pf_from_trades(trades, pnl_col="net_return")
    # 2025-10: pos=0.05, neg=0.01 → PF=5.0
    # 2025-11: pos=0.02, neg=0.03 → PF=0.667
    assert abs(monthly_pf[pd.Timestamp("2025-10-01")] - 5.0) < 1e-6
    assert abs(monthly_pf[pd.Timestamp("2025-11-01")] - (0.02 / 0.03)) < 1e-6


def test_all_wins_month_capped_at_inf_display_cap():
    trades = pd.DataFrame({
        "entry_time": pd.to_datetime(["2025-06-05", "2025-06-10"]),
        "net_return": [0.02, 0.03],
    })
    monthly_pf = compute_monthly_pf_from_trades(
        trades, pnl_col="net_return", inf_display_cap=7.5,
    )
    assert monthly_pf[pd.Timestamp("2025-06-01")] == 7.5
