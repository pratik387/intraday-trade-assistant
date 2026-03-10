"""
I1 Candle Recorder - Records Upstox WebSocket 1-minute candles to Parquet.

Captures the broker-constructed I1 (1-minute) OHLCV candles from the WebSocket
stream. These are used to verify whether WebSocket candles match the Historical
API 1m bars — critical for aligning backtest and live data sources.

Output: data/sidecar/i1_candles/i1_candles_{YYYYMMDD}.parquet
Schema: symbol, minute_ts, open, high, low, close, volume
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

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

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sidecar" / "i1_candles"
DATA_DIR.mkdir(parents=True, exist_ok=True)


class I1CandleRecorder:
    """
    Records Upstox WebSocket I1 (1-minute) candles to Parquet.

    Keeps only the LAST snapshot per (symbol, minute) — when the minute
    closes, the last received I1 candle is the completed bar.

    Thread-safe. Call finalize() at EOD to write the parquet file.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self._output_dir = Path(output_dir) if output_dir else DATA_DIR
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        # Key: (symbol, minute_ts_str) -> (open, high, low, close, volume)
        self._candles: Dict[Tuple[str, str], Tuple[float, float, float, float, int]] = {}

        self._today = datetime.now().strftime("%Y%m%d")
        self._file_path = self._output_dir / f"i1_candles_{self._today}.parquet"
        self._update_count = 0

        logger.info(f"I1_CANDLE_RECORDER | Initialized, output: {self._file_path}")

    def on_i1_candle(
        self,
        symbol: str,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: int,
        ts_str: str,
    ) -> None:
        """
        Record an I1 candle snapshot.

        Called on every tick — overwrites previous snapshot for same (symbol, minute).
        The last update before minute boundary is the completed candle.
        """
        if not ts_str or open_ == 0:
            return

        with self._lock:
            self._candles[(symbol, ts_str)] = (open_, high, low, close, volume)
            self._update_count += 1

    def finalize(self) -> Optional[Path]:
        """Write all candles to parquet and return the file path."""
        with self._lock:
            if not self._candles:
                logger.info("I1_CANDLE_RECORDER | No candles recorded")
                return None

            symbols = []
            timestamps = []
            opens = []
            highs = []
            lows = []
            closes = []
            volumes = []

            for (sym, ts_str), (o, h, l, c, v) in self._candles.items():
                symbols.append(sym)
                timestamps.append(ts_str)
                opens.append(o)
                highs.append(h)
                lows.append(l)
                closes.append(c)
                volumes.append(v)

            df = pd.DataFrame({
                "symbol": symbols,
                "minute_ts": timestamps,
                "open": pd.array(opens, dtype="float64"),
                "high": pd.array(highs, dtype="float64"),
                "low": pd.array(lows, dtype="float64"),
                "close": pd.array(closes, dtype="float64"),
                "volume": pd.array(volumes, dtype="int64"),
            })

            df.to_parquet(self._file_path, compression="snappy", index=False)

            unique_symbols = df["symbol"].nunique()
            unique_minutes = df["minute_ts"].nunique()
            size_mb = self._file_path.stat().st_size / (1024 * 1024)

            logger.info(
                f"I1_CANDLE_RECORDER | Complete: {len(df):,} candles "
                f"({unique_symbols} symbols, {unique_minutes} minutes) -> "
                f"{self._file_path.name} ({size_mb:.1f} MB), "
                f"{self._update_count:,} total updates"
            )
            return self._file_path

    @property
    def candle_count(self) -> int:
        """Number of unique (symbol, minute) candles recorded."""
        return len(self._candles)

    @property
    def output_path(self) -> Path:
        return self._file_path
