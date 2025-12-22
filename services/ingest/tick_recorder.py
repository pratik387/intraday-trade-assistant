"""
TickRecorder - Records raw tick data during paper trading sessions.

Usage:
    from services.ingest.tick_recorder import TickRecorder

    recorder = TickRecorder(output_dir="recordings")
    recorder.start()

    # ... run paper trading ...

    recorder.stop()

Output format (Parquet - columnar, compressed):
    Columns: symbol (string), price (float32), qty (int32), volume (int64), ts (datetime64[ms])
    ~12-18 bytes per tick compressed
"""
from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import pandas as pd
from config.logging_config import get_agent_logger
from services.ingest.tick_router import register_tick_listener_full, unregister_tick_listener_full

logger = get_agent_logger()


class TickRecorder:
    """
    Records raw tick data to Parquet files for later replay/analysis.

    Uses Parquet format for ~5x compression vs JSONL.
    Expected file size: ~50-100 MB for a full trading day.
    """

    def __init__(
        self,
        output_dir: str = "recordings",
        session_date: Optional[str] = None,
        buffer_size: int = 50000,  # Flush every 50k ticks (~5-10 seconds of data)
    ):
        """
        Initialize tick recorder.

        Args:
            output_dir: Directory to save recordings
            session_date: Date string (YYYY-MM-DD) for filename. Defaults to today.
            buffer_size: Number of ticks to buffer before flushing to disk
        """
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._session_date = session_date or datetime.now().strftime("%Y-%m-%d")
        self._buffer_size = buffer_size

        # Pre-allocate lists for columnar storage
        self._symbols: List[str] = []
        self._prices: List[float] = []
        self._qtys: List[int] = []
        self._volumes: List[int] = []  # Cumulative day volume
        self._timestamps: List[datetime] = []
        self._buffer_lock = threading.Lock()

        self._file_path: Optional[Path] = None
        self._part_num = 0  # For chunked writes

        self._running = False
        self._tick_count = 0
        self._start_time: Optional[datetime] = None

    def start(self) -> None:
        """Start recording ticks."""
        if self._running:
            logger.warning("TickRecorder already running")
            return

        # Create output file path
        timestamp = datetime.now().strftime("%H%M%S")
        self._file_path = self._output_dir / f"ticks_{self._session_date}_{timestamp}.parquet"

        self._running = True
        self._tick_count = 0
        self._part_num = 0
        self._start_time = datetime.now()

        # Register as full tick listener (includes volume)
        register_tick_listener_full(self._on_tick)

        logger.info(f"TickRecorder started: {self._file_path}")

    def stop(self) -> None:
        """Stop recording and flush remaining buffer."""
        if not self._running:
            return

        self._running = False

        # Unregister listener
        unregister_tick_listener_full(self._on_tick)

        # Flush remaining buffer
        self._flush()

        # Merge part files if any
        self._merge_parts()

        elapsed = (datetime.now() - self._start_time).total_seconds() if self._start_time else 0
        rate = self._tick_count / elapsed if elapsed > 0 else 0

        # Get file size
        size_mb = self._file_path.stat().st_size / (1024 * 1024) if self._file_path.exists() else 0

        logger.info(
            f"TickRecorder stopped: {self._tick_count:,} ticks in {elapsed:.1f}s "
            f"({rate:.0f} ticks/sec) -> {size_mb:.1f} MB"
        )

    def _on_tick(self, symbol: str, price: float, qty: float, volume: int, ts: datetime) -> None:
        """Callback for tick_router - called for each normalized tick with volume."""
        if not self._running:
            return

        with self._buffer_lock:
            self._symbols.append(symbol)
            self._prices.append(price)
            self._qtys.append(int(qty))
            self._volumes.append(volume)
            self._timestamps.append(ts)
            self._tick_count += 1

            if len(self._symbols) >= self._buffer_size:
                self._flush()

    def _flush(self) -> None:
        """Write buffered ticks to parquet file."""
        with self._buffer_lock:
            if not self._symbols:
                return

            # Create DataFrame from columnar data
            df = pd.DataFrame({
                "symbol": self._symbols,
                "price": pd.array(self._prices, dtype="float32"),
                "qty": pd.array(self._qtys, dtype="int32"),
                "volume": pd.array(self._volumes, dtype="int64"),
                "ts": pd.to_datetime(self._timestamps),
            })

            # Write to part file (will merge at end)
            part_path = self._file_path.with_suffix(f".part{self._part_num}.parquet")
            df.to_parquet(part_path, compression="snappy", index=False)
            self._part_num += 1

            # Clear buffers
            self._symbols.clear()
            self._prices.clear()
            self._qtys.clear()
            self._volumes.clear()
            self._timestamps.clear()

    def _merge_parts(self) -> None:
        """Merge all part files into single parquet file."""
        part_files = sorted(self._file_path.parent.glob(f"{self._file_path.stem}.part*.parquet"))

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
        """Number of ticks recorded so far."""
        return self._tick_count

    @property
    def file_path(self) -> Optional[Path]:
        """Path to current recording file."""
        return self._file_path


# Convenience function for quick recording
_global_recorder: Optional[TickRecorder] = None


def start_recording(output_dir: str = "recordings", session_date: Optional[str] = None) -> TickRecorder:
    """Start global tick recording."""
    global _global_recorder
    if _global_recorder and _global_recorder._running:
        logger.warning("Global recorder already running, stopping first")
        _global_recorder.stop()

    _global_recorder = TickRecorder(output_dir=output_dir, session_date=session_date)
    _global_recorder.start()
    return _global_recorder


def stop_recording() -> Optional[Path]:
    """Stop global tick recording and return file path."""
    global _global_recorder
    if not _global_recorder:
        return None

    file_path = _global_recorder.file_path
    _global_recorder.stop()
    _global_recorder = None
    return file_path
