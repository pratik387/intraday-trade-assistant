import pandas as pd
from tools.edge_discovery.decay_monitor import compute_status, DecayConfig


def _config() -> DecayConfig:
    return DecayConfig(
        rolling_window_months=6,
        caution_pf_threshold=1.20,
        pause_pf_threshold=1.00,
        retire_pf_threshold=0.80,
        retire_consecutive_months=2,
    )


def test_active_above_caution():
    monthly_pf = pd.Series({
        pd.Timestamp("2025-10-01"): 1.45,
        pd.Timestamp("2025-11-01"): 1.50,
        pd.Timestamp("2025-12-01"): 1.40,
    })
    status = compute_status(monthly_pf, _config())
    assert status.status == "ACTIVE"


def test_caution_when_pf_below_1_20():
    monthly_pf = pd.Series({
        pd.Timestamp("2025-10-01"): 1.30,
        pd.Timestamp("2025-11-01"): 1.10,
        pd.Timestamp("2025-12-01"): 1.05,
    })
    status = compute_status(monthly_pf, _config())
    assert status.status == "CAUTION"


def test_pause_when_pf_drops_below_1_00():
    monthly_pf = pd.Series({
        pd.Timestamp("2025-10-01"): 0.95,
    })
    status = compute_status(monthly_pf, _config())
    assert status.status == "PAUSED"


def test_retire_when_below_0_80_two_consecutive_months():
    monthly_pf = pd.Series({
        pd.Timestamp("2025-10-01"): 0.75,
        pd.Timestamp("2025-11-01"): 0.70,
    })
    status = compute_status(monthly_pf, _config())
    assert status.status == "RETIRED"


def test_not_retire_when_only_one_month_below_0_80():
    monthly_pf = pd.Series({
        pd.Timestamp("2025-10-01"): 0.75,
        pd.Timestamp("2025-11-01"): 0.85,
    })
    status = compute_status(monthly_pf, _config())
    assert status.status != "RETIRED"
