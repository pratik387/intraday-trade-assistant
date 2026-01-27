"""
Market Data Service - Shared tick/bar distribution for deterministic multi-instance trading.

Architecture (Recommended):
  Standalone Market Data Service (separate process):
    - Connects to Zerodha WebSocket
    - Builds bars via BarBuilder
    - Publishes to Redis
    - No trading logic

  Trading Instances (all subscribers):
    - Subscribe to bars from Redis
    - Each has its own broker account
    - Identical market data = identical signals

Usage:
    # Terminal 1: Start Redis
    redis-server

    # Terminal 2: Start Market Data Service
    python -m market_data.market_data_service

    # Terminal 3: LIVE trading instance
    python main.py --shared-market-data

    # Terminal 4: PAPER trading instance
    python main.py --shared-market-data --paper-trading

    # Terminal 5: Another LIVE instance (different risk)
    python main.py --shared-market-data --risk-value 1000

Without --shared-market-data: standalone mode (existing behavior, no Redis)

Configuration (in configuration.json):
    "market_data_bus": {
        "mode": "standalone",      // or "subscriber" (all instances should be subscriber when using standalone service)
        "redis_url": "redis://localhost:6379/0",
        "publish_1m_bars": false,
        "ltp_mode": "standalone"   // follows mode
    }
"""

from .market_data_bus import MarketDataBus, BarEvent
from .shared_ltp import SharedLTPCache
from .bar_subscriber import BarSubscriber
from .factory import create_market_data_components

__all__ = [
    "MarketDataBus",
    "BarEvent",
    "SharedLTPCache",
    "BarSubscriber",
    "create_market_data_components",
]
