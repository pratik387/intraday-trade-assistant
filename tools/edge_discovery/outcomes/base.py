"""OutcomeModule Protocol."""
from __future__ import annotations

from typing import Dict, Protocol, runtime_checkable

import pandas as pd

from tools.edge_discovery.types import Event


@runtime_checkable
class OutcomeModule(Protocol):
    name: str

    def compute(self, event: Event, bars: pd.DataFrame) -> Dict[str, float]:
        """Compute outcome values for one event given the symbol's 5m bars."""
        ...
