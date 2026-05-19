import pandas as pd
import pytest
from datetime import date
from tools.edge_discovery.data_loader import load_5m_period


def test_load_5m_returns_dataframe_with_required_columns():
    df = load_5m_period(
        start=date(2024, 6, 1),
        end=date(2024, 6, 30),
        symbols={"RELIANCE", "TCS"},
    )
    if len(df) == 0:
        pytest.skip("June 2024 feather not present in working tree")
    assert isinstance(df, pd.DataFrame)
    assert {"symbol", "date", "open", "high", "low", "close", "volume"}.issubset(df.columns)
    assert df["date"].dt.tz is None  # IST-naive
    assert df["symbol"].nunique() <= 2  # filter applied
