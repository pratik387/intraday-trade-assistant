import pandas as pd
import numpy as np
from tools.edge_discovery.validation.walk_forward import walk_forward, WalkForwardConfig


def _make_trades(start: str = "2023-01-01", n_months: int = 24, trades_per_month: int = 50,
                 mean_ret: float = 0.005) -> pd.DataFrame:
    rows = []
    rng = np.random.RandomState(42)
    cur = pd.Timestamp(start)
    for _ in range(n_months):
        for _ in range(trades_per_month):
            r = rng.normal(mean_ret, 0.02)
            rows.append({"entry_time": cur, "net_return": r})
        cur = cur + pd.DateOffset(months=1)
    return pd.DataFrame(rows)


def test_walk_forward_stability_score_high_for_stable_series():
    trades = _make_trades(mean_ret=0.005, n_months=18)
    cfg = WalkForwardConfig(train_window_months=6, test_window_months=1, step_months=1)
    result = walk_forward(trades, cfg)
    assert result.stability_score >= 0.4


def test_walk_forward_emits_validation_pf_per_window():
    trades = _make_trades(n_months=12)
    cfg = WalkForwardConfig(train_window_months=6, test_window_months=1, step_months=1)
    result = walk_forward(trades, cfg)
    assert len(result.validation_pfs) >= 4
    assert all(pf >= 0 for pf in result.validation_pfs)
