# sidecar/__init__.py
"""
Sidecar Data Service
--------------------
Lightweight always-on service for collecting market data.

Components:
- data_collector.py: Main service (run as systemd unit)
- bootstrap.py: Reader for main engine to load sidecar data
- integration.py: Helper to integrate with ScreenerLive
"""

from .bootstrap import SidecarBootstrap, bootstrap_from_sidecar
from .integration import bootstrap_screener_from_sidecar, maybe_bootstrap_from_sidecar

__all__ = [
    "SidecarBootstrap",
    "bootstrap_from_sidecar",
    "bootstrap_screener_from_sidecar",
    "maybe_bootstrap_from_sidecar",
]
