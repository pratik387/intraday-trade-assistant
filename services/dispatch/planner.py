"""Per-bar work plan assembly."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from services.dispatch.tag_map import TagMap


@dataclass
class Batch:
    items: list = field(default_factory=list)


class DispatchPlanner:
    def __init__(self, batch_size: int):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1; got {batch_size}")
        self._batch_size = batch_size

    def plan(
        self,
        bar_ts: datetime,
        tag_map: TagMap,
        df5_by_symbol: dict,
        levels_by_symbol: dict,
    ) -> list:
        items: list = []
        for sym in sorted(tag_map.active_symbols()):
            df5 = df5_by_symbol.get(sym)
            if df5 is None or df5.empty:
                continue
            levels = levels_by_symbol.get(sym, {})
            tags = tag_map.active_tags(sym)
            if not tags:
                continue
            items.append((sym, df5, levels, tags))

        batches: list = []
        for i in range(0, len(items), self._batch_size):
            batches.append(Batch(items=items[i : i + self._batch_size]))
        return batches
