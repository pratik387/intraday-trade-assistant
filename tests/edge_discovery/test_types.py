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


def test_top_edge_regions_ranks_by_effect_size_x_sqrt_n():
    rows = pd.DataFrame({
        "feature_a": ["x", "x", "x", "x", "y", "y", "y", "y", "z", "z"],
        "feature_b": ["p", "p", "q", "q", "p", "p", "q", "q", "p", "p"],
        "outcome_post_cost": [0.01, 0.02, 0.005, 0.01,    # x,p → mean 0.015 (n=2)
                              -0.01, -0.02, -0.005, -0.01,  # ...
                              0.001, 0.001],
    })
    table = ConditionalOutcomeTable(rows=rows)
    regions = table.top_edge_regions(
        outcome="outcome_post_cost",
        feature_names=["feature_a", "feature_b"],
        min_n=2,
        top_n=5,
    )
    assert len(regions) >= 1
    # The strongest region by effect-size × √n should be reported first
    top = regions[0]
    assert "feature_cut" in top
    assert "mean_return" in top
    assert "n" in top
    assert "t_proxy" in top
    # Verify ranking contract: top region has the highest t_proxy
    for i in range(len(regions) - 1):
        assert regions[i]["t_proxy"] >= regions[i + 1]["t_proxy"], (
            f"regions not sorted desc by t_proxy at index {i}"
        )


def test_top_edge_regions_raises_on_unknown_feature():
    rows = pd.DataFrame({
        "feature_a": ["x", "y"],
        "outcome_post_cost": [0.01, -0.01],
    })
    table = ConditionalOutcomeTable(rows=rows)
    with pytest.raises(KeyError, match="not in table columns"):
        table.top_edge_regions(
            outcome="outcome_post_cost",
            feature_names=["feature_a", "nonexistent_feature"],
            min_n=1,
        )
