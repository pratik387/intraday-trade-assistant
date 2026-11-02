"""Results-day breadth fade SHORT — the up_spike fade, widened, on results days only.

Research (docs/experiment_ledger.jsonl, setup results_reaction_fade__breadth):
up_spike's edge sits on days the company's results are public (results_preopen)
or due (board_meeting_today). On those days the same fade also pays at a lower
spike threshold in the day window and at the full threshold after 14:00. Two
cells, each a setup of its own using this class:

  results_breadth_day_fade_short   r3 >= 3.5%, signal candle 09:45-14:00
  results_breadth_late_fade_short  r3 >= 5.5%, signal candle 14:05-15:00

Everything else is up_spike_fade_short's detector (vol burst > 5x the strictly
prior candles, >= 6 prior candles, day turnover >= Rs1.5cr, small/micro/unknown
caps, one fire per symbol-day) with two differences:

  1. COMPLETED candles only. Live, df_5m can carry the still-forming candle after
     the scanned bar; the cells were validated on completed candles, so rows
     newer than context.scan_bar_ts are dropped first. up_spike itself keeps its
     forming-candle behaviour (its forward record was earned that way).
  2. The results tag (services/results_context) must be in
     required_results_context.

Validation: Phase 2 pass (ledger 2026-11-02), fresh-pool one-shot PASS
2026-11-03 (fragile: t 1.16-1.30), real-time-priority re-run +Rs3.1-3.2k/month
added. PAPER ONLY (live_enabled false). Conflicts with up_spike are resolved by
config setup_groups (first signal wins the symbol-day, same-scan ties go to
up_spike), not here.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from config.logging_config import get_agent_logger
from services.results_context import EVENT_TAGS, results_context
from .data_models import MarketContext, StructureAnalysis
from .up_spike_fade_short_structure import UpSpikeFadeShortStructure

logger = get_agent_logger()

_REPO = Path(__file__).resolve().parents[1]


class ResultsBreadthFadeShortStructure(UpSpikeFadeShortStructure):
    """up_spike fade on completed candles, gated to results-tagged symbol-days."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # One class serves two setups, and the dispatch workers build detectors
        # from the raw setup config (no setup name injected), so each setup names
        # the setup its events belong to.
        self.structure_type = str(config["event_setup_type"])
        self.use_completed_bars_only = bool(config["use_completed_bars_only"])
        tags = config["required_results_context"]
        if not isinstance(tags, (list, tuple)) or not tags or not set(tags) <= EVENT_TAGS:
            raise ValueError(f"{self.structure_type}: required_results_context must be a non-empty "
                             f"list drawn from {sorted(EVENT_TAGS)}, got {tags!r}")
        self.required_results_context = frozenset(tags)
        # The same file the setup's event_feed validates at daemon start.
        self.calendar_path = _REPO / config["event_feed"]["path"]

    def detect(self, context: MarketContext) -> StructureAnalysis:
        def _reject(reason: str) -> StructureAnalysis:
            return StructureAnalysis(structure_detected=False, events=[], quality_score=0.0,
                                     rejection_reason=reason)

        tag = results_context(context.symbol, context.session_date, path=self.calendar_path)
        if tag not in self.required_results_context:
            return _reject(f"results_context={tag} not in {sorted(self.required_results_context)}")
        if self.use_completed_bars_only:
            if context.scan_bar_ts is None:
                # Fail closed: without the scanned bar's label the forming candle
                # cannot be told apart, and this cell is validated on completed ones.
                return _reject("scan_bar_ts unknown: cannot drop the forming candle")
            df = context.df_5m
            if df is not None and not df.empty and df.index[-1] > context.scan_bar_ts:
                df = df[df.index <= context.scan_bar_ts]
                if df.empty:
                    return _reject("no completed candle at or before the scanned bar")
                context = dataclasses.replace(
                    context, df_5m=df, timestamp=pd.Timestamp(df.index[-1]),
                    current_price=float(df["close"].iloc[-1]),
                )
        return super().detect(context)
