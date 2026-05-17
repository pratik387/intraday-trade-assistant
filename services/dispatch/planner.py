"""Per-bar work plan assembly."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

from services.dispatch.tag_map import TagMap


# A single dispatch unit: symbol, its 5m data, its level cache, the set of
# detector names that should run against it, and its cap segment label.
# (sym, df5, levels, tags, cap_segment)
BatchItem = tuple  # (str, pd.DataFrame, dict, set, str)


@dataclass
class Batch:
    """One chunk of per-bar work for a ProcessPoolExecutor worker.

    bar_ts and session_date are bar-level metadata shared across all items.
    regime / regime_diagnostics were computed from index_df5 once per bar in
    the parent process and are forwarded to workers so detectors that read
    ctx.regime get consistent values without re-computing per worker.

    daily_dict maps symbol -> yesterday's daily DataFrame so detectors like
    circuit_t1_fade_short and delivery_pct_anomaly_short that read
    ctx.df_daily receive valid data.
    """
    items: list = field(default_factory=list)
    bar_ts: Optional[datetime] = None
    session_date: Optional[object] = None   # datetime.date
    regime: str = "chop"
    regime_diagnostics: Optional[dict] = None
    daily_dict: Optional[dict] = None       # sym -> pd.DataFrame of daily bars


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
        *,
        session_date=None,
        regime: str = "chop",
        cap_segment_map: Optional[dict] = None,
        regime_diagnostics: Optional[dict] = None,
        daily_dict: Optional[dict] = None,
    ) -> list:
        """Assemble per-bar work into ≤batch_size chunks.

        Each item in a Batch is a 5-tuple:
            (sym, df5, levels, tags, cap_segment)
        where cap_segment is looked up from cap_segment_map (if provided) or
        defaults to "unknown".

        Bar-level metadata (bar_ts, session_date, regime, regime_diagnostics)
        is stored on Batch and forwarded to worker.dispatch_worker_batch so
        workers can build a complete MarketContext without extra IPC.

        daily_dict (sym -> pd.DataFrame) is stored at the Batch level so
        workers can forward ctx.df_daily to detectors that need yesterday's
        daily bars (e.g. circuit_t1_fade_short, delivery_pct_anomaly_short).
        """
        cap_segment_map = cap_segment_map or {}
        items: list = []
        for sym in sorted(tag_map.active_symbols()):
            df5 = df5_by_symbol.get(sym)
            if df5 is None or df5.empty:
                continue
            levels = levels_by_symbol.get(sym, {})
            tags = tag_map.active_tags(sym)
            if not tags:
                continue
            cap_seg = cap_segment_map.get(sym, "unknown")
            items.append((sym, df5, levels, tags, cap_seg))

        batches: list = []
        for i in range(0, len(items), self._batch_size):
            batches.append(Batch(
                items=items[i : i + self._batch_size],
                bar_ts=bar_ts,
                session_date=session_date,
                regime=regime,
                regime_diagnostics=regime_diagnostics,
                daily_dict=daily_dict,
            ))
        return batches
