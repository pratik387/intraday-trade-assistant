import pandas as pd
from datetime import datetime
from services.dispatch.tag_map import TagMap
from services.dispatch.fetch_scope import FetchScopeManager


def test_fetch_set_is_active_symbols():
    tm = TagMap()
    tm.add_universe("gap_fade_short", {"NSE:A", "NSE:B"})
    tm.open_window("gap_fade_short")
    scope = FetchScopeManager()
    assert scope.fetch_set(datetime(2024, 5, 3, 9, 30), tm) == {"NSE:A", "NSE:B"}


def test_fetch_set_empty_when_no_active_tags():
    tm = TagMap()
    scope = FetchScopeManager()
    assert scope.fetch_set(datetime(2024, 5, 3, 10, 30), tm) == set()


def test_backfill_needed_when_df_missing():
    scope = FetchScopeManager()
    assert scope.is_backfill_needed("NSE:A", df5_by_symbol={}, bar_ts=datetime(2024, 5, 3, 12, 0))


def test_backfill_needed_when_df_stale():
    scope = FetchScopeManager()
    df = pd.DataFrame({"close": [100.0]}, index=[pd.Timestamp("2024-05-03 10:25:00")])
    assert scope.is_backfill_needed("NSE:A", df5_by_symbol={"NSE:A": df}, bar_ts=datetime(2024, 5, 3, 12, 0))


def test_no_backfill_when_df_current():
    scope = FetchScopeManager()
    df = pd.DataFrame({"close": [100.0]}, index=[pd.Timestamp("2024-05-03 11:55:00")])
    assert not scope.is_backfill_needed("NSE:A", df5_by_symbol={"NSE:A": df}, bar_ts=datetime(2024, 5, 3, 12, 0))
