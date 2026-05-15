from datetime import date
import pandas as pd

from tools.edge_discovery.rule_orthogonality import (
    classify_candidate,
    check_against_policy_dates,
)


def test_classify_rule_orthogonal_candidate():
    classification = classify_candidate(
        name="long_panic_gap_down_smid",
        edge_source="retail cannot short cash small-caps without F&O",
        depends_on=[],
    )
    assert classification == "rule_orthogonal"


def test_classify_rule_dependent_candidate():
    classification = classify_candidate(
        name="stt_arbitrage",
        edge_source="STT rate differential between cash and futures",
        depends_on=["stt_rate"],
    )
    assert classification == "rule_dependent"


def test_policy_dates_known():
    pf_series = pd.Series({
        pd.Timestamp("2024-08-01"): 1.40,
        pd.Timestamp("2024-09-01"): 1.42,
        pd.Timestamp("2024-10-01"): 1.45,  # STT hike day
        pd.Timestamp("2024-11-01"): 0.60,  # 50%+ drop
    })
    breaks = check_against_policy_dates(pf_series, drop_threshold_pct=50.0)
    assert any(b["policy_date"] == date(2024, 10, 1) for b in breaks)
