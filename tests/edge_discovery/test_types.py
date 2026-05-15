import pandas as pd
import pytest
from tools.edge_discovery.types import Event, ConditionalOutcomeTable


def test_event_construction_requires_ist_naive_timestamp():
    ts = pd.Timestamp("2024-06-15 09:30:00")
    e = Event(symbol="RELIANCE", event_time=ts, metadata={"trigger": "gap_up"})
    assert e.symbol == "RELIANCE"
    assert e.event_time.tzinfo is None
    assert e.metadata["trigger"] == "gap_up"


def test_event_rejects_tz_aware_timestamp():
    ts = pd.Timestamp("2024-06-15 09:30:00", tz="Asia/Kolkata")
    with pytest.raises(ValueError, match="IST-naive"):
        Event(symbol="RELIANCE", event_time=ts, metadata={})


def test_conditional_outcome_table_holds_rows_df():
    rows = pd.DataFrame({"feature_a": [1, 2, 3], "outcome_ret_60m_post_cost": [0.01, -0.02, 0.005]})
    table = ConditionalOutcomeTable(rows=rows)
    assert len(table.rows) == 3
    assert "outcome_ret_60m_post_cost" in table.rows.columns


def test_conditional_outcome_table_slice_by_returns_aggregated_stats():
    rows = pd.DataFrame({
        "feature_x": ["a", "a", "b", "b"],
        "outcome_ret_60m_post_cost": [0.01, 0.03, -0.005, -0.01],
    })
    table = ConditionalOutcomeTable(rows=rows)
    sliced = table.slice_by("feature_x", outcome="outcome_ret_60m_post_cost")
    assert set(sliced.index.tolist()) == {"a", "b"}
    assert abs(sliced.loc["a", "mean"] - 0.02) < 1e-9
    assert abs(sliced.loc["b", "mean"] + 0.0075) < 1e-9
    assert sliced.loc["a", "n"] == 2
