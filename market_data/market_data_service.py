#!/usr/bin/env python3
"""
Standalone Market Data Service - Single source of truth for all trading instances.

This service:
  - Connects to Zerodha WebSocket
  - Builds 1m/5m/15m bars via BarBuilder
  - Publishes completed bars to Redis
  - Updates LTP in Redis for all symbols

All trading instances (LIVE, PAPER, etc.) subscribe to this service
and receive identical market data, ensuring deterministic signals.

Usage:
    python -m market_data.market_data_service

    # Or with custom Redis URL
    python -m market_data.market_data_service --redis-url redis://192.168.1.100:6379/0

Requirements:
    - Redis server running
    - Zerodha API credentials (via environment or .env)
"""

import sys
import signal
import argparse
import threading
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.logging_config import get_agent_logger
from config.filters_setup import load_filters
from services.ingest.bar_builder import BarBuilder
from services.ingest.stream_client import WSClient
from services.ingest.tick_router import TickRouter
from market_data.market_data_bus import MarketDataBus

logger = get_agent_logger()


class MarketDataService:
    """
    Standalone service that publishes market data to Redis.

    All trading instances subscribe to this service for bars and LTP,
    ensuring they all see identical market data.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        publish_1m_bars: bool = False,
    ):
        self._redis_url = redis_url
        self._publish_1m = publish_1m_bars
        self._running = False

        # Load config for universe
        self._config = load_filters()

        # Create market data bus (publisher mode)
        self._bus = MarketDataBus(
            mode="publisher",
            redis_url=redis_url,
            publish_1m_bars=publish_1m_bars,
        )

        # Create bar builder with publish callbacks
        self._bar_builder = BarBuilder(
            bar_5m_span_minutes=5,
            on_1m_close=self._on_1m_close,
            on_5m_close=self._on_5m_close,
            on_15m_close=self._on_15m_close,
            index_symbols=self._get_index_symbols(),
        )

        # Will be initialized in start()
        self._ws = None
        self._router = None
        self._sdk = None

        # Stats
        self._tick_count = 0
        self._bar_1m_count = 0
        self._bar_5m_count = 0
        self._last_tick_time = None

    def _get_index_symbols(self) -> list:
        """Get index symbols from config."""
        return self._config.get("index_symbols", ["NSE:NIFTY 50", "NSE:NIFTY BANK"])

    def _load_universe(self) -> dict:
        """Load trading universe (token -> symbol mapping) from KiteClient."""
        # KiteClient loads and filters instruments from Kite API
        # Use its tok2sym mapping directly
        if self._sdk is None:
            logger.error("MDS | KiteClient not initialized")
            return {}
        token_map = self._sdk._tok2sym  # {token: "NSE:SYMBOL"}
        logger.info(f"MDS | Loaded {len(token_map)} instruments from KiteClient")
        return token_map

    def _on_tick(self, symbol: str, price: float, volume: float, ts: datetime) -> None:
        """Handle incoming tick - forward to bar builder and publish to Redis."""
        self._tick_count += 1
        self._last_tick_time = ts

        # Forward to bar builder
        self._bar_builder.on_tick(symbol, price, volume, ts)

        # Publish LTP to Redis hash (for point-in-time lookups)
        self._bus.update_ltp(symbol, price, ts)

        # Publish tick to Redis pub/sub (for real-time execution)
        self._bus.publish_tick(symbol, price, volume, ts)

    def _on_1m_close(self, symbol: str, bar) -> None:
        """Publish 1m bar to Redis."""
        self._bar_1m_count += 1
        if self._publish_1m:
            self._bus.publish_bar(symbol, "1m", bar)

    def _on_5m_close(self, symbol: str, bar) -> None:
        """Publish 5m bar to Redis."""
        self._bar_5m_count += 1
        self._bus.publish_bar(symbol, "5m", bar)
        logger.info(f"MDS | 5m bar: {symbol} close={bar.get('close', 0):.2f} @ {bar.name}")

    def _on_15m_close(self, symbol: str, bar) -> None:
        """Publish 15m bar to Redis."""
        self._bus.publish_bar(symbol, "15m", bar)

    def start(self) -> None:
        """Start the market data service."""
        logger.info("=" * 60)
        logger.info("MARKET DATA SERVICE - Starting")
        logger.info("=" * 60)
        logger.info(f"Redis URL: {self._redis_url}")
        logger.info(f"Publish 1m bars: {self._publish_1m}")

        # Initialize Kite SDK
        try:
            from broker.kite.kite_client import KiteClient
            self._sdk = KiteClient()
            logger.info("MDS | Kite SDK initialized")
        except Exception as e:
            logger.error(f"MDS | Failed to initialize Kite SDK: {e}")
            raise

        # Pre-warm daily cache and publish to Redis for subscriber instances
        self._prewarm_and_publish_daily_cache()

        # Load universe
        token_map = self._load_universe()
        if not token_map:
            raise RuntimeError("No instruments loaded")

        # Create WebSocket client and tick router
        self._router = TickRouter(
            on_tick=self._on_tick,
            token_to_symbol=token_map,
        )

        self._ws = WSClient(sdk=self._sdk, on_tick=self._bar_builder.on_tick)
        self._ws.on_message(self._router.handle_raw)

        # Subscribe to all symbols
        from services.ingest.subscription_manager import SubscriptionManager
        subs = SubscriptionManager(self._ws)
        subs.set_core(set(token_map.keys()))  # Pass token integers, not dict
        subs.start()

        # Start WebSocket
        self._ws.start()
        self._running = True

        logger.info(f"MDS | Subscribed to {len(token_map)} symbols")
        logger.info("MDS | Service started - publishing to Redis")

        # Start stats reporter
        self._start_stats_reporter()

    def _prewarm_and_publish_daily_cache(self) -> None:
        """
        Pre-warm daily OHLCV cache (rolling) and publish to Redis for subscribers.

        Uses rolling cache: loads yesterday's disk cache if today's doesn't exist.
        The 90% threshold in prewarm_daily_cache() skips API if cache is populated.
        Then publishes to Redis so subscriber instances load in ~1-3s.
        """
        dc_cfg = self._config.get("market_data_bus", {}).get("daily_cache_redis", {})
        if not dc_cfg.get("enabled", False):
            logger.info("MDS | daily_cache_redis disabled, skipping daily cache prewarm")
            return

        from services.state.daily_cache_persistence import DailyCachePersistence

        logger.info("MDS | Pre-warming daily OHLCV cache for Redis distribution...")

        # Rolling load: today's cache or yesterday's (2-5 sec)
        # Yesterday's cache has data through T-2 which is fine for startup.
        # At EOD, MDS builds today's daily bar from 1m data and appends it,
        # so tomorrow's rolling cache will have data through today.
        cache_persistence = DailyCachePersistence()
        cached_data = cache_persistence.load_latest()
        if cached_data:
            self._sdk.set_daily_cache(cached_data)

        # Only fetches from API if cache is empty (cold start, no rolling cache)
        result = self._sdk.prewarm_daily_cache(days=210)
        if result.get("source") == "api":
            cache_persistence.save(self._sdk.get_daily_cache())

        # Publish to Redis for subscriber instances
        today = datetime.now().date().isoformat()
        cache = self._sdk.get_daily_cache()
        if cache:
            self._bus.publish_daily_cache(today, cache, dc_cfg)
        else:
            logger.warning("MDS | Daily cache empty after prewarm, nothing to publish")

    def _start_stats_reporter(self) -> None:
        """Start background thread to report stats periodically."""
        def report_stats():
            while self._running:
                time.sleep(60)  # Report every minute
                if self._running:
                    logger.info(
                        f"MDS | Stats: ticks={self._tick_count}, "
                        f"1m_bars={self._bar_1m_count}, 5m_bars={self._bar_5m_count}, "
                        f"last_tick={self._last_tick_time}"
                    )

        t = threading.Thread(target=report_stats, daemon=True, name="MDS-Stats")
        t.start()

    def _build_and_save_eod_daily_bars(self) -> None:
        """
        Build today's daily OHLCV from bar_builder's 1m bars and append to rolling cache.

        At EOD the bar_builder has all 1m bars for the session. We aggregate them
        into a single daily bar per symbol (open/high/low/close/volume from 9:15-15:30),
        append to the rolling cache, and save to disk.

        Next day's startup loads this file → cache has data through today → fresh.
        No Kite historical API calls needed.
        """
        import pandas as pd
        from services.state.daily_cache_persistence import DailyCachePersistence

        logger.info("MDS | Building EOD daily bars from 1m intraday data...")

        today = pd.Timestamp(datetime.now().date())
        market_open = today + pd.Timedelta(hours=9, minutes=15)
        market_close = today + pd.Timedelta(hours=15, minutes=30)

        # Get current daily cache (loaded at startup from rolling file)
        daily_cache = self._sdk.get_daily_cache()

        updated = 0
        skipped = 0

        with self._bar_builder._lock:
            all_symbols = list(self._bar_builder._bars_1m.keys())

            for symbol in all_symbols:
                df_1m = self._bar_builder._bars_1m.get(symbol)
                if df_1m is None or df_1m.empty:
                    skipped += 1
                    continue

                # Filter to regular session only (9:15 to 15:30)
                session = df_1m[(df_1m.index >= market_open) & (df_1m.index < market_close)]
                if session.empty:
                    skipped += 1
                    continue

                # Aggregate 1m bars to daily OHLCV
                daily_row = pd.DataFrame({
                    "open": [float(session.iloc[0]["open"])],
                    "high": [float(session["high"].max())],
                    "low": [float(session["low"].min())],
                    "close": [float(session.iloc[-1]["close"])],
                    "volume": [float(session["volume"].sum())],
                }, index=[today])

                # Append to existing historical data for this symbol
                existing = daily_cache.get(symbol)
                if existing is not None and not existing.empty:
                    # Remove today if somehow already present
                    existing = existing[existing.index < today]
                    combined = pd.concat([existing, daily_row])
                    daily_cache[symbol] = combined.tail(210)
                else:
                    daily_cache[symbol] = daily_row

                updated += 1

        # Save updated cache
        self._sdk.set_daily_cache(daily_cache)
        persistence = DailyCachePersistence()
        persistence.save(daily_cache)

        # Publish updated cache to Redis
        dc_cfg = self._config.get("market_data_bus", {}).get("daily_cache_redis", {})
        if dc_cfg.get("enabled", False):
            today_str = datetime.now().date().isoformat()
            self._bus.publish_daily_cache(today_str, daily_cache, dc_cfg)

        logger.info(
            f"MDS | EOD daily bars complete: {updated} symbols updated, "
            f"{skipped} skipped (no session data), saved to disk"
        )

    def stop(self) -> None:
        """Stop the market data service."""
        logger.info("MDS | Stopping...")
        self._running = False

        if self._ws:
            try:
                self._ws.stop()
            except Exception:
                pass

        if self._bus:
            self._bus.shutdown()

        logger.info("MDS | Stopped")

    def run_forever(self) -> None:
        """Run until interrupted. Auto-shuts down at EOD after building daily bars."""
        self.start()

        # Handle Ctrl+C
        def signal_handler(sig, frame):
            logger.info("MDS | Interrupt received, shutting down...")
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Parse EOD shutdown time from config
        eod_time_str = self._config.get("market_data_bus", {}).get("mds_eod_shutdown_time")
        if not eod_time_str:
            raise ValueError("Missing config: market_data_bus.mds_eod_shutdown_time")
        eod_h, eod_m = map(int, eod_time_str.split(":"))
        eod_done = False

        logger.info(f"MDS | Running... EOD shutdown scheduled at {eod_time_str} IST")
        while self._running:
            time.sleep(1)

            if not eod_done:
                now = datetime.now()
                if now.hour > eod_h or (now.hour == eod_h and now.minute >= eod_m):
                    logger.info(f"MDS | EOD time reached ({eod_time_str}), building daily bars...")
                    try:
                        self._build_and_save_eod_daily_bars()
                    except Exception as e:
                        logger.error(f"MDS | EOD daily bar build failed: {e}", exc_info=True)
                    eod_done = True
                    logger.info("MDS | EOD complete, shutting down")
                    self.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Standalone Market Data Service - publishes bars to Redis for all trading instances"
    )
    parser.add_argument(
        "--redis-url",
        default="redis://localhost:6379/0",
        help="Redis connection URL (default: redis://localhost:6379/0)"
    )
    parser.add_argument(
        "--publish-1m",
        action="store_true",
        help="Also publish 1-minute bars (increases Redis traffic)"
    )
    args = parser.parse_args()

    service = MarketDataService(
        redis_url=args.redis_url,
        publish_1m_bars=args.publish_1m,
    )
    service.run_forever()


if __name__ == "__main__":
    main()
