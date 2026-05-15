"""FeatureModule Protocol — every feature module implements this."""
from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable

import pandas as pd

from tools.edge_discovery.types import Event


@runtime_checkable
class FeatureModule(Protocol):
    name: str
    feature_names: List[str]

    def compute(self, event: Event, bars: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
        """Compute features for one event. kwargs may carry symbol_meta, pdh, pdl, etc."""
        ...
