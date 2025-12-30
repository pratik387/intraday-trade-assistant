"""
Events module for market event tracking and risk management.

Provides:
- EventsLoader: Load and query static/dynamic event data
- NSEEventsFetcher: Fetch corporate actions from NSE
- EventsGate: Gate for blocking/reducing trades on event days
"""

from .events_loader import EventsLoader
from .nse_fetcher import NSEEventsFetcher

__all__ = ["EventsLoader", "NSEEventsFetcher"]
