"""Decides what symbols to API-fetch per bar + backfill detection on re-entry."""
from __future__ import annotations

from datetime import datetime, timedelta
import pandas as pd
from services.dispatch.tag_map import TagMap


_ONE_BAR = timedelta(minutes=5)


class FetchScopeManager:
    def fetch_set(self, bar_ts: datetime, tag_map: TagMap) -> set:
        """Symbols to API-fetch this bar = currently-active-tag set."""
        return tag_map.active_symbols()

    def is_backfill_needed(
        self,
        sym: str,
        df5_by_symbol: dict,
        bar_ts: datetime,
    ) -> bool:
        """True if `sym` has no df_5m OR last bar is older than (bar_ts - 5min)."""
        df = df5_by_symbol.get(sym)
        if df is None or df.empty:
            return True
        last_ts = df.index[-1]
        if hasattr(last_ts, "to_pydatetime"):
            last_ts = last_ts.to_pydatetime()
        return last_ts < bar_ts - _ONE_BAR
