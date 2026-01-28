"""
Tick Recorder - Records market ticks to Parquet files for backtesting.

Can operate in two modes:
1. Direct mode: Called directly with tick data (used by publisher/LIVE)
2. Redis mode: Subscribes to Redis tick stream (standalone recording)

Output format matches sidecar for compatibility with upload tools:
- Path: data/sidecar/ticks/ticks_{YYYYMMDD}.parquet
- Columns: symbol, price, qty, volume, ts
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

try:
    from config.logging_config import get_agent_logger
    logger = get_agent_logger()
    if logger is None:
        raise ValueError()
except Exception:
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

# Output directory (matches sidecar for upload compatibility)
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sidecar" / "ticks"
DATA_DIR.mkdir(parents=True, exist_ok=True)


class TickRecorder:
    """
    Records ticks to Parquet files with periodic flushing.

    Thread-safe, buffers ticks in memory and flushes to disk periodically
    to avoid memory bloat. Uses part files for incremental writes.

    Usage:
        recorder = TickRecorder()
        recorder.on_tick("NSE:RELIANCE", 2450.0, 100, datetime.now(), 1500000)
        ...
        recorder.finalize()  # At end of day
    """

    def __init__(
        self,
        buffer_size: int = 50000,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize tick recorder.

        Args:
            buffer_size: Number of ticks to buffer before flushing (~5-10 seconds)
            output_dir: Output directory (default: data/sidecar/ticks)
        """
        self._buffer_size = buffer_size
        self._output_dir = Path(output_dir) if output_dir else DATA_DIR
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

        # Columnar buffers (memory efficient)
        self._symbols: List[str] = []
        self._prices: List[float] = []
        self._qtys: List[int] = []
        self._cumvols: List[int] = []
        self._timestamps: List[datetime] = []

        # File tracking
        self._today = datetime.now().strftime("%Y%m%d")
        self._part_num = 0
        self._tick_count = 0
        self._file_path = self._output_dir / f"ticks_{self._today}.parquet"

        logger.info(f"TICK_RECORDER | Initialized, output: {self._file_path}")

    def on_tick(
        self,
        symbol: str,
        price: float,
        qty: float,
        ts: datetime,
        cumulative_volume: float = 0,
    ) -> None:
        """
        Record a single tick.

        Args:
            symbol: Trading symbol (e.g., "NSE:RELIANCE")
            price: Last traded price
            qty: Last traded quantity
            ts: Tick timestamp
            cumulative_volume: Cumulative day volume (0 if not available)
        """
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

            logger.debug(f"TICK_RECORDER | Flushed {len(self._symbols)} ticks to {part_path.name}")

            # Clear buffers
            self._symbols.clear()
            self._prices.clear()
            self._qtys.clear()
            self._cumvols.clear()
            self._timestamps.clear()

    def finalize(self) -> Optional[Path]:
        """
        Flush remaining buffer and merge part files into final parquet.

        Returns:
            Path to final parquet file, or None if no data
        """
        self._flush()
        self._merge_parts()

        if self._file_path.exists():
            size_mb = self._file_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"TICK_RECORDER | Complete: {self._tick_count:,} ticks -> "
                f"{self._file_path.name} ({size_mb:.1f} MB)"
            )
            return self._file_path
        return None

    def _merge_parts(self) -> None:
        """Merge all part files into single parquet file."""
        part_files = sorted(self._output_dir.glob(f"ticks_{self._today}.part*.parquet"))

        if not part_files:
            # No data - create empty file for consistency
            pd.DataFrame(columns=["symbol", "price", "qty", "volume", "ts"]).to_parquet(
                self._file_path, compression="snappy", index=False
            )
            return

        if len(part_files) == 1:
            # Single part - just rename
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
        """Total ticks recorded."""
        return self._tick_count

    @property
    def output_path(self) -> Path:
        """Path to output file."""
        return self._file_path


class RedisTickRecorder:
    """
    Standalone tick recorder that subscribes to Redis tick stream.

    Use this when you want to record ticks without running the full engine.
    Subscribes to ticks:stream channel published by MDS.

    Usage:
        recorder = RedisTickRecorder("redis://localhost:6379/0")
        recorder.start()  # Blocks until stopped or EOD
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        buffer_size: int = 50000,
    ):
        """
        Initialize Redis tick recorder.

        Args:
            redis_url: Redis connection URL
            buffer_size: Ticks to buffer before flushing
        """
        self._redis_url = redis_url
        self._recorder = TickRecorder(buffer_size=buffer_size)
        self._running = False

        try:
            import redis
            self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
            self._redis.ping()
            logger.info(f"REDIS_TICK_RECORDER | Connected to {redis_url}")
        except ImportError:
            raise ImportError("redis package required: pip install redis")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Redis: {e}")

    def start(self, block: bool = True) -> None:
        """
        Start recording ticks from Redis.

        Args:
            block: If True, blocks until stopped. If False, runs in background thread.
        """
        self._running = True

        if block:
            self._run_loop()
        else:
            thread = threading.Thread(target=self._run_loop, daemon=True)
            thread.start()

    def _run_loop(self) -> None:
        """Main recording loop."""
        pubsub = self._redis.pubsub()
        pubsub.subscribe("ticks:stream")

        logger.info("REDIS_TICK_RECORDER | Started, listening for ticks...")

        try:
            for msg in pubsub.listen():
                if not self._running:
                    break

                if msg["type"] != "message":
                    continue

                try:
                    # Parse tick: "symbol|price|volume|ts_ns|send_ns"
                    parts = msg["data"].split("|")
                    if len(parts) < 4:
                        continue

                    symbol = parts[0]
                    price = float(parts[1])
                    qty = float(parts[2])
                    ts_ns = int(parts[3])
                    ts = datetime.fromtimestamp(ts_ns / 1_000_000)

                    self._recorder.on_tick(symbol, price, qty, ts, 0)

                except Exception:
                    # Don't log every parse failure
                    pass

        except Exception as e:
            if self._running:
                logger.error(f"REDIS_TICK_RECORDER | Error: {e}")
        finally:
            pubsub.close()

    def stop(self) -> Optional[Path]:
        """
        Stop recording and finalize output.

        Returns:
            Path to final parquet file
        """
        self._running = False
        return self._recorder.finalize()

    @property
    def tick_count(self) -> int:
        """Total ticks recorded."""
        return self._recorder.tick_count


# ============================================================================
# Standalone Entry Point
# ============================================================================

def main():
    """Run standalone Redis tick recorder."""
    import argparse
    import signal

    parser = argparse.ArgumentParser(description="Record ticks from Redis to Parquet")
    parser.add_argument(
        "--redis-url",
        default="redis://localhost:6379/0",
        help="Redis connection URL",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=50000,
        help="Ticks to buffer before flushing",
    )
    args = parser.parse_args()

    recorder = RedisTickRecorder(
        redis_url=args.redis_url,
        buffer_size=args.buffer_size,
    )

    def signal_handler(signum, frame):
        logger.info(f"REDIS_TICK_RECORDER | Signal {signum} received, stopping...")
        path = recorder.stop()
        if path:
            logger.info(f"REDIS_TICK_RECORDER | Output: {path}")
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    recorder.start(block=True)


if __name__ == "__main__":
    main()
