"""Edge-First Discovery Framework — main explorer.

Run(events, bar_data, symbol_meta, ...) → ConditionalOutcomeTable

Each event is enriched with features from each FeatureModule and outcomes from
each OutcomeModule. The resulting table is the conditional outcome distribution
that downstream slicers / edge-region detectors work on.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from tools.edge_discovery.features.base import FeatureModule
from tools.edge_discovery.outcomes.base import OutcomeModule
from tools.edge_discovery.types import ConditionalOutcomeTable, Event


class Explorer:
    def __init__(self, features: List[FeatureModule], outcomes: List[OutcomeModule]) -> None:
        self.features = features
        self.outcomes = outcomes

    def run(
        self,
        events: List[Event],
        bar_data: Dict[str, pd.DataFrame],
        symbol_meta: Dict[str, Dict[str, Any]],
        pdh_pdl_close_by_event: Optional[Dict[int, Dict[str, float]]] = None,
        adv_by_symbol: Optional[Dict[str, float]] = None,
        ema_by_event: Optional[Dict[int, Dict[str, float]]] = None,
        delivery_by_event: Optional[Dict[int, float]] = None,
    ) -> ConditionalOutcomeTable:
        rows: List[Dict[str, Any]] = []
        for i, ev in enumerate(events):
            bars = bar_data.get(ev.symbol)
            if bars is None or len(bars) == 0:
                continue
            meta = symbol_meta.get(ev.symbol, {})
            pdh = pdl = pclose = None
            if pdh_pdl_close_by_event and i in pdh_pdl_close_by_event:
                pdh = pdh_pdl_close_by_event[i].get("pdh")
                pdl = pdh_pdl_close_by_event[i].get("pdl")
                pclose = pdh_pdl_close_by_event[i].get("prior_close")
            adv = (adv_by_symbol or {}).get(ev.symbol)
            ema20 = ema50 = None
            if ema_by_event and i in ema_by_event:
                ema20 = ema_by_event[i].get("ema_20")
                ema50 = ema_by_event[i].get("ema_50")
            deliv = (delivery_by_event or {}).get(i)

            row: Dict[str, Any] = {
                "_event_idx": i,
                "symbol": ev.symbol,
                "event_time": ev.event_time,
                **{f"meta_{k}": v for k, v in ev.metadata.items()},
            }
            for fm in self.features:
                fvals = fm.compute(
                    ev, bars,
                    symbol_meta=meta, pdh=pdh, pdl=pdl, prior_close=pclose,
                    adv_shares=adv, ema_20=ema20, ema_50=ema50,
                    delivery_pct_t1=deliv,
                )
                row.update(fvals)
            for om in self.outcomes:
                ovals = om.compute(ev, bars)
                row.update(ovals)
            rows.append(row)
        return ConditionalOutcomeTable(rows=pd.DataFrame(rows))
