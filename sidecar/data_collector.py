# sidecar/data_collector.py
"""
Sidecar Data Collector Service
------------------------------
Lightweight always-on service that collects market data independently of main engine.

Responsibilities:
1. Connect to Kite WebSocket at 09:00
2. Aggregate ticks into 5m bars (START-STAMPED)
3. Compute ORB levels at 09:35 (after 09:15-09:30 window)
4. Fetch PDH/PDL/PDC at startup
5. Persist everything to /data/ directory

Main engine reads from these files on startup for instant bootstrap.
"""

from __future__ import annotations

import json
import os
import sys
import signal
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.logging_config import get_agent_logger
from config.env_setup import env

logger = get_agent_logger()

# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sidecar"
BARS_DIR = DATA_DIR / "bars"
ORB_DIR = DATA_DIR / "orb"
LEVELS_DIR = DATA_DIR / "levels"
TICKS_DIR = DATA_DIR / "ticks"

# Ensure directories exist
for d in [DATA_DIR, BARS_DIR, ORB_DIR, LEVELS_DIR, TICKS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Bar Aggregator (simplified from bar_builder.py)
# ============================================================================

@dataclass
class TickAccumulator:
    """Accumulates ticks for current 5m bar."""
    open: float = 0.0
    high: float = 0.0
    low: float = float("inf")
    close: float = 0.0
    volume: float = 0.0           # Bar volume (from cumulative delta)
    vwap_num: float = 0.0         # For VWAP: sum(price * qty)
    vwap_den: float = 0.0         # For VWAP: sum(qty)
    bar_start: Optional[datetime] = None
    first_cumvol: float = 0.0     # Cumulative volume at bar start
    last_cumvol: float = 0.0      # Latest cumulative volume


class BarAggregator:
    """
    Aggregates ticks into 5m bars with START-STAMPED timestamps.
    Persists to Feather files periodically.

    Volume handling:
    - Bar volume = delta of cumulative volume_traded (accurate)
    - VWAP uses last_traded_quantity (per-tick qty)
    """

    def __init__(self, symbols: List[str], persist_interval: int = 300):
        self._symbols = set(symbols)
        self._persist_interval = persist_interval  # seconds
        self._lock = threading.RLock()

        # Current accumulating bars
        self._current: Dict[str, TickAccumulator] = {}

        # Completed 5m bars (in-memory, flushed periodically)
        self._bars: Dict[str, List[Dict]] = defaultdict(list)

        # Last persist time
        self._last_persist = time.time()

        # Today's date for file naming
        self._today = datetime.now().strftime("%Y%m%d")

    def _get_bar_start(self, ts: datetime) -> datetime:
        """Get the start timestamp for the 5m bar containing ts."""
        minute = (ts.minute // 5) * 5
        return ts.replace(minute=minute, second=0, microsecond=0)

    def on_tick(self, symbol: str, price: float, qty: float, ts: datetime,
                 cumulative_volume: float = 0) -> None:
        """
        Process a single tick.

        Args:
            symbol: Symbol like "NSE:RELIANCE"
            price: Last traded price
            qty: Last traded quantity (for VWAP)
            ts: Timestamp
            cumulative_volume: Cumulative day volume (for accurate bar volume)
        """
        if symbol not in self._symbols:
            return

        bar_start = self._get_bar_start(ts)

        with self._lock:
            acc = self._current.get(symbol)

            # New bar or first tick
            if acc is None or acc.bar_start != bar_start:
                # Close previous bar if exists
                if acc is not None and acc.bar_start is not None:
                    self._close_bar(symbol, acc)

                # Start new bar
                acc = TickAccumulator(
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=0,  # Will be computed from cumulative delta
                    vwap_num=price * qty,
                    vwap_den=qty,
                    bar_start=bar_start,
                    first_cumvol=cumulative_volume,
                    last_cumvol=cumulative_volume,
                )
                self._current[symbol] = acc
            else:
                # Update current bar
                acc.high = max(acc.high, price)
                acc.low = min(acc.low, price)
                acc.close = price
                acc.last_cumvol = cumulative_volume  # Track latest cumulative
                acc.vwap_num += price * qty
                acc.vwap_den += qty

    def _close_bar(self, symbol: str, acc: TickAccumulator) -> None:
        """Close a completed bar and add to history."""
        if acc.bar_start is None:
            return

        vwap = acc.vwap_num / acc.vwap_den if acc.vwap_den > 0 else acc.close

        # Compute volume from cumulative delta (more accurate than sum of qty)
        # Falls back to sum of qty if cumulative not available
        if acc.last_cumvol > 0 and acc.first_cumvol > 0:
            volume = acc.last_cumvol - acc.first_cumvol
        else:
            # Fallback: use sum of per-tick qty (vwap_den)
            volume = acc.vwap_den

        bar = {
            "timestamp": acc.bar_start.isoformat(),
            "open": acc.open,
            "high": acc.high,
            "low": acc.low if acc.low != float("inf") else acc.open,
            "close": acc.close,
            "volume": volume,
            "vwap": vwap,
        }

        self._bars[symbol].append(bar)

    def force_close_all(self) -> None:
        """Force close all current bars (called at end of session)."""
        with self._lock:
            for symbol, acc in list(self._current.items()):
                if acc.bar_start is not None:
                    self._close_bar(symbol, acc)
            self._current.clear()

    def get_bars_df(self, symbol: str) -> pd.DataFrame:
        """Get all completed bars for a symbol as DataFrame."""
        with self._lock:
            bars = self._bars.get(symbol, [])
            if not bars:
                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "vwap"])

            df = pd.DataFrame(bars)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
            return df

    def get_all_bars(self) -> Dict[str, pd.DataFrame]:
        """Get all bars for all symbols."""
        with self._lock:
            result = {}
            for symbol in self._bars:
                result[symbol] = self.get_bars_df(symbol)
            return result

    def persist(self) -> None:
        """Persist all bars to disk."""
        with self._lock:
            if not self._bars:
                return

            bars_file = BARS_DIR / f"bars_{self._today}.feather"

            # Combine all symbols into one DataFrame
            all_dfs = []
            for symbol, bars in self._bars.items():
                if bars:
                    df = pd.DataFrame(bars)
                    df["symbol"] = symbol
                    all_dfs.append(df)

            if all_dfs:
                combined = pd.concat(all_dfs, ignore_index=True)
                combined.to_feather(bars_file)
                logger.info(f"SIDECAR | Persisted {len(combined)} bars to {bars_file}")

            self._last_persist = time.time()

    def maybe_persist(self) -> None:
        """Persist if enough time has passed."""
        if time.time() - self._last_persist >= self._persist_interval:
            self.persist()


# ============================================================================
# ORB Calculator
# ============================================================================

class ORBCalculator:
    """Computes Opening Range Breakout levels from 09:15-09:30 bars."""

    def __init__(self, bar_aggregator: BarAggregator):
        self._aggregator = bar_aggregator
        self._orb_computed = False
        self._orb_levels: Dict[str, Dict[str, float]] = {}

    def compute_orb(self) -> Dict[str, Dict[str, float]]:
        """Compute ORB from accumulated bars."""
        if self._orb_computed:
            return self._orb_levels

        orb_start = datetime.now().replace(hour=9, minute=15, second=0, microsecond=0)
        orb_end = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)

        all_bars = self._aggregator.get_all_bars()

        for symbol, df in all_bars.items():
            if df.empty:
                self._orb_levels[symbol] = {"ORH": float("nan"), "ORL": float("nan")}
                continue

            # Filter to ORB window (START-STAMPED: 09:15, 09:20, 09:25)
            orb_bars = df[(df.index >= orb_start) & (df.index < orb_end)]

            if len(orb_bars) >= 3:  # Need all 3 bars for valid ORB
                orh = float(orb_bars["high"].max())
                orl = float(orb_bars["low"].min())
                self._orb_levels[symbol] = {"ORH": orh, "ORL": orl}
            else:
                self._orb_levels[symbol] = {"ORH": float("nan"), "ORL": float("nan")}

        self._orb_computed = True

        # Persist ORB
        self._persist_orb()

        valid_count = sum(1 for v in self._orb_levels.values()
                         if not pd.isna(v.get("ORH")))
        logger.info(f"SIDECAR | Computed ORB for {valid_count}/{len(self._orb_levels)} symbols")

        return self._orb_levels

    def _persist_orb(self) -> None:
        """Persist ORB levels to disk."""
        today = datetime.now().strftime("%Y%m%d")
        orb_file = ORB_DIR / f"orb_{today}.json"

        # Convert NaN to None for JSON serialization
        serializable = {}
        for symbol, levels in self._orb_levels.items():
            serializable[symbol] = {
                k: (None if pd.isna(v) else v)
                for k, v in levels.items()
            }

        data = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "computed_at": datetime.now().isoformat(),
            "levels": serializable
        }

        with open(orb_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"SIDECAR | Persisted ORB to {orb_file}")


# ============================================================================
# PDH/PDL/PDC Fetcher
# ============================================================================

class DailyLevelsFetcher:
    """Fetches and persists previous day's levels (PDH/PDL/PDC)."""

    def __init__(self, kite_client):
        self._kite = kite_client
        self._levels: Dict[str, Dict[str, float]] = {}

    def fetch_all(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Fetch PDH/PDL/PDC for all symbols."""
        logger.info(f"SIDECAR | Fetching daily levels for {len(symbols)} symbols...")

        for i, symbol in enumerate(symbols):
            try:
                levels = self._kite.get_prevday_levels(symbol)
                self._levels[symbol] = levels
            except Exception as e:
                logger.debug(f"SIDECAR | Failed to fetch levels for {symbol}: {e}")
                self._levels[symbol] = {
                    "PDH": float("nan"),
                    "PDL": float("nan"),
                    "PDC": float("nan")
                }

            # Progress logging
            if (i + 1) % 100 == 0:
                logger.info(f"SIDECAR | Daily levels progress: {i+1}/{len(symbols)}")

        # Persist
        self._persist()

        valid_count = sum(1 for v in self._levels.values()
                         if not pd.isna(v.get("PDH")))
        logger.info(f"SIDECAR | Fetched daily levels for {valid_count}/{len(symbols)} symbols")

        return self._levels

    def _persist(self) -> None:
        """Persist daily levels to disk."""
        today = datetime.now().strftime("%Y%m%d")
        levels_file = LEVELS_DIR / f"daily_{today}.json"

        # Convert NaN to None for JSON serialization
        serializable = {}
        for symbol, levels in self._levels.items():
            serializable[symbol] = {
                k: (None if pd.isna(v) else v)
                for k, v in levels.items()
            }

        data = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "fetched_at": datetime.now().isoformat(),
            "levels": serializable
        }

        with open(levels_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"SIDECAR | Persisted daily levels to {levels_file}")


# ============================================================================
# Tick Recorder (raw tick persistence)
# ============================================================================

class TickRecorder:
    """
    Records raw ticks to Parquet files for replay/recovery.

    Same format as main engine's TickRecorder for compatibility.
    Flushes periodically to avoid memory bloat.
    """

    def __init__(self, buffer_size: int = 50000):
        """
        Args:
            buffer_size: Ticks to buffer before flushing (~5-10 seconds of data)
        """
        self._buffer_size = buffer_size
        self._lock = threading.RLock()

        # Columnar buffers (more memory efficient than list of dicts)
        self._symbols: List[str] = []
        self._prices: List[float] = []
        self._qtys: List[int] = []
        self._cumvols: List[int] = []
        self._timestamps: List[datetime] = []

        self._today = datetime.now().strftime("%Y%m%d")
        self._part_num = 0
        self._tick_count = 0
        self._file_path = TICKS_DIR / f"ticks_{self._today}.parquet"

    def on_tick(self, symbol: str, price: float, qty: float,
                ts: datetime, cumulative_volume: float = 0) -> None:
        """Record a single tick."""
        with self._lock:
            self._symbols.append(symbol)
            self._prices.append(price)
            self._qtys.append(int(qty))
            self._cumvols.append(int(cumulative_volume))
            self._timestamps.append(ts)
            self._tick_count += 1

            if len(self._symbols) >= self._buffer_size:
                self._flush()

    def _flush(self) -> None:
        """Write buffered ticks to parquet part file."""
        with self._lock:
            if not self._symbols:
                return

            df = pd.DataFrame({
                "symbol": self._symbols,
                "price": pd.array(self._prices, dtype="float32"),
                "qty": pd.array(self._qtys, dtype="int32"),
                "volume": pd.array(self._cumvols, dtype="int64"),
                "ts": pd.to_datetime(self._timestamps),
            })

            # Write to part file
            part_path = self._file_path.with_suffix(f".part{self._part_num}.parquet")
            df.to_parquet(part_path, compression="snappy", index=False)
            self._part_num += 1

            logger.debug(f"SIDECAR | Flushed {len(self._symbols)} ticks to {part_path.name}")

            # Clear buffers
            self._symbols.clear()
            self._prices.clear()
            self._qtys.clear()
            self._cumvols.clear()
            self._timestamps.clear()

    def finalize(self) -> None:
        """Flush remaining buffer and merge part files."""
        self._flush()
        self._merge_parts()

        if self._file_path.exists():
            size_mb = self._file_path.stat().st_size / (1024 * 1024)
            logger.info(f"SIDECAR | Tick recording complete: {self._tick_count:,} ticks -> {size_mb:.1f} MB")

    def _merge_parts(self) -> None:
        """Merge all part files into single parquet file."""
        part_files = sorted(TICKS_DIR.glob(f"ticks_{self._today}.part*.parquet"))

        if not part_files:
            # No data - create empty file
            pd.DataFrame(columns=["symbol", "price", "qty", "volume", "ts"]).to_parquet(
                self._file_path, compression="snappy", index=False
            )
            return

        if len(part_files) == 1:
            # Just rename
            part_files[0].rename(self._file_path)
            return

        # Merge all parts
        dfs = [pd.read_parquet(p) for p in part_files]
        merged = pd.concat(dfs, ignore_index=True)
        merged.to_parquet(self._file_path, compression="snappy", index=False)

        # Clean up part files
        for p in part_files:
            p.unlink()

    @property
    def tick_count(self) -> int:
        return self._tick_count


# ============================================================================
# Main Sidecar Service
# ============================================================================

class SidecarService:
    """Main sidecar service orchestrating all components."""

    def __init__(self, symbols: List[str]):
        self._symbols = symbols
        self._running = False
        self._stop_event = threading.Event()

        # Components (initialized lazily)
        self._kite = None
        self._ticker = None
        self._bar_aggregator = None
        self._orb_calculator = None
        self._daily_fetcher = None
        self._tick_recorder = None

        # State
        self._orb_scheduled = False
        self._connected = False

    def _init_kite(self):
        """Initialize Kite client."""
        from broker.kite.kite_client import KiteClient
        self._kite = KiteClient()
        logger.info("SIDECAR | KiteClient initialized")

    def _on_ticks(self, ws, ticks):
        """WebSocket tick callback."""
        ts = datetime.now()
        for tick in ticks:
            token = tick.get("instrument_token")
            ltp = tick.get("last_price")
            # last_traded_quantity = per-tick volume (for VWAP calculation)
            # volume_traded = cumulative day volume (for accurate bar volume via delta)
            qty = tick.get("last_traded_quantity") or tick.get("last_quantity") or 0
            cumvol = tick.get("volume_traded") or 0

            if token and ltp:
                symbol = self._kite.get_token_map().get(token)
                if symbol:
                    # Feed to bar aggregator
                    self._bar_aggregator.on_tick(
                        symbol, float(ltp), float(qty), ts, float(cumvol)
                    )
                    # Record raw tick
                    self._tick_recorder.on_tick(
                        symbol, float(ltp), float(qty), ts, float(cumvol)
                    )

        # Maybe persist bars
        self._bar_aggregator.maybe_persist()

        # Check if we should compute ORB
        self._maybe_compute_orb()

    def _on_connect(self, ws, response):
        """WebSocket connect callback."""
        logger.info("SIDECAR | WebSocket connected")
        self._connected = True

        # Subscribe to all symbols
        tokens = self._kite.resolve_tokens(self._symbols)
        ws.subscribe(tokens)
        ws.set_mode(ws.MODE_LTP, tokens)
        logger.info(f"SIDECAR | Subscribed to {len(tokens)} tokens")

    def _on_close(self, ws, code, reason):
        """WebSocket close callback."""
        logger.warning(f"SIDECAR | WebSocket closed: {code} - {reason}")
        self._connected = False

    def _on_error(self, ws, code, reason):
        """WebSocket error callback."""
        logger.error(f"SIDECAR | WebSocket error: {code} - {reason}")

    def _maybe_compute_orb(self):
        """Compute ORB if it's time (after 09:35)."""
        if self._orb_scheduled:
            return

        now = datetime.now().time()
        if now >= dtime(9, 35):
            self._orb_scheduled = True
            self._orb_calculator.compute_orb()

    def start(self):
        """Start the sidecar service."""
        logger.info("SIDECAR | Starting...")
        self._running = True

        # Initialize Kite
        self._init_kite()

        # Initialize components
        self._bar_aggregator = BarAggregator(self._symbols)
        self._orb_calculator = ORBCalculator(self._bar_aggregator)
        self._daily_fetcher = DailyLevelsFetcher(self._kite)
        self._tick_recorder = TickRecorder()

        # Fetch daily levels first
        self._daily_fetcher.fetch_all(self._symbols)

        # Start WebSocket
        self._ticker = self._kite.make_ticker()
        self._ticker.on_ticks = self._on_ticks
        self._ticker.on_connect = self._on_connect
        self._ticker.on_close = self._on_close
        self._ticker.on_error = self._on_error

        # Connect in background thread
        self._ticker.connect(threaded=True)

        logger.info("SIDECAR | Started, waiting for market data...")

        # Main loop
        try:
            while self._running and not self._stop_event.is_set():
                time.sleep(1)

                # Check for EOD
                now = datetime.now().time()
                if now >= dtime(15, 35):
                    logger.info("SIDECAR | EOD reached, shutting down...")
                    break
        except KeyboardInterrupt:
            logger.info("SIDECAR | Interrupted")
        finally:
            self.stop()

    def stop(self):
        """Stop the sidecar service."""
        logger.info("SIDECAR | Stopping...")
        self._running = False
        self._stop_event.set()

        # Close all bars and persist
        if self._bar_aggregator:
            self._bar_aggregator.force_close_all()
            self._bar_aggregator.persist()

        # Finalize tick recording
        if self._tick_recorder:
            self._tick_recorder.finalize()

        # Stop WebSocket
        if self._ticker:
            try:
                self._ticker.close()
            except Exception as e:
                logger.warning(f"SIDECAR | Error closing ticker: {e}")

        logger.info("SIDECAR | Stopped")


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Main entry point for sidecar service."""
    logger.info("=" * 60)
    logger.info("SIDECAR | Data Collector Service Starting")
    logger.info("=" * 60)

    # Load symbols from configuration
    from config.filters_setup import load_filters
    cfg = load_filters()

    # Get core symbols (same as main engine uses)
    from broker.kite.kite_client import KiteClient
    kite = KiteClient()
    symbols = kite.list_equities()

    # Filter to configured universe if needed
    max_symbols = cfg.get("max_symbols", 500)
    symbols = symbols[:max_symbols]

    logger.info(f"SIDECAR | Loaded {len(symbols)} symbols")

    # Create and start service
    service = SidecarService(symbols)

    # Handle signals
    def signal_handler(signum, frame):
        logger.info(f"SIDECAR | Signal {signum} received")
        service.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start
    service.start()


if __name__ == "__main__":
    main()
