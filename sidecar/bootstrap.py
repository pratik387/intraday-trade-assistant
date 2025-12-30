# sidecar/bootstrap.py
"""
Bootstrap Reader for Main Engine
---------------------------------
Reads pre-computed data from sidecar's data files for instant startup.

Usage in main engine:
    from sidecar.bootstrap import SidecarBootstrap

    bootstrap = SidecarBootstrap()
    if bootstrap.is_available():
        orb_levels = bootstrap.load_orb()
        daily_levels = bootstrap.load_daily_levels()
        bars = bootstrap.load_bars()  # Dict[symbol, DataFrame]
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

import pandas as pd

try:
    from config.logging_config import get_agent_logger
    logger = get_agent_logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ============================================================================
# Paths (same as data_collector.py)
# ============================================================================

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sidecar"
BARS_DIR = DATA_DIR / "bars"
ORB_DIR = DATA_DIR / "orb"
LEVELS_DIR = DATA_DIR / "levels"
TICKS_DIR = DATA_DIR / "ticks"


# ============================================================================
# Bootstrap Reader
# ============================================================================

class SidecarBootstrap:
    """
    Reads data files written by sidecar for main engine bootstrap.

    Provides:
    - ORB levels (ORH, ORL)
    - Daily levels (PDH, PDL, PDC)
    - 5m bars for all symbols
    """

    def __init__(self, date: Optional[str] = None):
        """
        Initialize bootstrap reader.

        Args:
            date: Date string in YYYYMMDD format. Defaults to today.
        """
        self._date = date or datetime.now().strftime("%Y%m%d")
        self._date_iso = f"{self._date[:4]}-{self._date[4:6]}-{self._date[6:8]}"

    def is_available(self) -> bool:
        """Check if sidecar data is available for today."""
        bars_file = BARS_DIR / f"bars_{self._date}.feather"
        return bars_file.exists()

    def has_orb(self) -> bool:
        """Check if ORB data is available."""
        orb_file = ORB_DIR / f"orb_{self._date}.json"
        return orb_file.exists()

    def has_daily_levels(self) -> bool:
        """Check if daily levels are available."""
        levels_file = LEVELS_DIR / f"daily_{self._date}.json"
        return levels_file.exists()

    def load_orb(self) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Load ORB levels from sidecar.

        Returns:
            Dict mapping symbol -> {"ORH": float, "ORL": float}
            None values are converted to NaN.
        """
        orb_file = ORB_DIR / f"orb_{self._date}.json"

        if not orb_file.exists():
            logger.warning(f"BOOTSTRAP | ORB file not found: {orb_file}")
            return None

        try:
            with open(orb_file, "r") as f:
                data = json.load(f)

            # Verify date matches
            if data.get("date") != self._date_iso:
                logger.warning(f"BOOTSTRAP | ORB date mismatch: {data.get('date')} != {self._date_iso}")
                return None

            raw_levels = data.get("levels", {})

            # Convert None back to NaN
            levels = {}
            for symbol, lvls in raw_levels.items():
                levels[symbol] = {
                    k: (float("nan") if v is None else float(v))
                    for k, v in lvls.items()
                }

            valid_count = sum(1 for v in levels.values()
                             if not math.isnan(v.get("ORH", float("nan"))))
            logger.info(f"BOOTSTRAP | Loaded ORB for {len(levels)} symbols ({valid_count} valid)")

            return levels

        except Exception as e:
            logger.exception(f"BOOTSTRAP | Failed to load ORB: {e}")
            return None

    def load_daily_levels(self) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Load daily levels (PDH/PDL/PDC) from sidecar.

        Returns:
            Dict mapping symbol -> {"PDH": float, "PDL": float, "PDC": float}
        """
        levels_file = LEVELS_DIR / f"daily_{self._date}.json"

        if not levels_file.exists():
            logger.warning(f"BOOTSTRAP | Daily levels file not found: {levels_file}")
            return None

        try:
            with open(levels_file, "r") as f:
                data = json.load(f)

            # Verify date matches
            if data.get("date") != self._date_iso:
                logger.warning(f"BOOTSTRAP | Daily levels date mismatch")
                return None

            raw_levels = data.get("levels", {})

            # Convert None back to NaN
            levels = {}
            for symbol, lvls in raw_levels.items():
                levels[symbol] = {
                    k: (float("nan") if v is None else float(v))
                    for k, v in lvls.items()
                }

            valid_count = sum(1 for v in levels.values()
                             if not math.isnan(v.get("PDH", float("nan"))))
            logger.info(f"BOOTSTRAP | Loaded daily levels for {len(levels)} symbols ({valid_count} valid)")

            return levels

        except Exception as e:
            logger.exception(f"BOOTSTRAP | Failed to load daily levels: {e}")
            return None

    def load_bars(self, symbols: Optional[list] = None) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Load 5m bars from sidecar.

        Args:
            symbols: Optional list of symbols to load. If None, loads all.

        Returns:
            Dict mapping symbol -> DataFrame with OHLCV columns
            Index is datetime (START-STAMPED)
        """
        bars_file = BARS_DIR / f"bars_{self._date}.feather"

        if not bars_file.exists():
            logger.warning(f"BOOTSTRAP | Bars file not found: {bars_file}")
            return None

        try:
            df = pd.read_feather(bars_file)

            if df.empty:
                return {}

            # Parse timestamp
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Group by symbol
            result = {}
            grouped = df.groupby("symbol")

            for symbol, group in grouped:
                if symbols is not None and symbol not in symbols:
                    continue

                sym_df = group.set_index("timestamp").drop(columns=["symbol"], errors="ignore")
                sym_df = sym_df.sort_index()
                result[symbol] = sym_df

            logger.info(f"BOOTSTRAP | Loaded {sum(len(df) for df in result.values())} bars "
                       f"for {len(result)} symbols")

            return result

        except Exception as e:
            logger.exception(f"BOOTSTRAP | Failed to load bars: {e}")
            return None

    def load_bars_for_symbol(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load bars for a single symbol."""
        result = self.load_bars(symbols=[symbol])
        if result:
            return result.get(symbol)
        return None

    def has_ticks(self) -> bool:
        """Check if raw tick data is available."""
        ticks_file = TICKS_DIR / f"ticks_{self._date}.parquet"
        return ticks_file.exists()

    def load_ticks(self, symbols: Optional[list] = None,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """
        Load raw ticks from sidecar.

        Args:
            symbols: Optional list of symbols to filter
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            DataFrame with columns: symbol, price, qty, volume, ts
        """
        ticks_file = TICKS_DIR / f"ticks_{self._date}.parquet"

        if not ticks_file.exists():
            logger.warning(f"BOOTSTRAP | Ticks file not found: {ticks_file}")
            return None

        try:
            df = pd.read_parquet(ticks_file)

            if df.empty:
                return df

            # Filter by symbols if provided
            if symbols:
                df = df[df["symbol"].isin(symbols)]

            # Filter by time range if provided
            if start_time:
                df = df[df["ts"] >= start_time]
            if end_time:
                df = df[df["ts"] <= end_time]

            logger.info(f"BOOTSTRAP | Loaded {len(df):,} ticks")
            return df

        except Exception as e:
            logger.exception(f"BOOTSTRAP | Failed to load ticks: {e}")
            return None

    def get_status(self) -> Dict[str, Any]:
        """Get status of sidecar data availability."""
        bars_file = BARS_DIR / f"bars_{self._date}.feather"
        orb_file = ORB_DIR / f"orb_{self._date}.json"
        levels_file = LEVELS_DIR / f"daily_{self._date}.json"
        ticks_file = TICKS_DIR / f"ticks_{self._date}.parquet"

        status = {
            "date": self._date_iso,
            "bars": {
                "available": bars_file.exists(),
                "path": str(bars_file),
            },
            "orb": {
                "available": orb_file.exists(),
                "path": str(orb_file),
            },
            "daily_levels": {
                "available": levels_file.exists(),
                "path": str(levels_file),
            },
            "ticks": {
                "available": ticks_file.exists(),
                "path": str(ticks_file),
            }
        }

        # Add file sizes and modification times
        for key, file_path in [("bars", bars_file), ("orb", orb_file),
                               ("daily_levels", levels_file), ("ticks", ticks_file)]:
            if status[key]["available"]:
                try:
                    stat = file_path.stat()
                    status[key]["size_kb"] = round(stat.st_size / 1024, 2)
                    status[key]["modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
                except OSError:
                    pass

        return status


# ============================================================================
# Convenience Functions
# ============================================================================

def bootstrap_from_sidecar(bar_builder=None, orb_cache=None) -> Dict[str, Any]:
    """
    Convenience function to bootstrap main engine from sidecar data.

    Args:
        bar_builder: Optional BarBuilder instance to populate
        orb_cache: Optional dict to populate with ORB levels

    Returns:
        Dict with bootstrap results and counts
    """
    bootstrap = SidecarBootstrap()
    result = {
        "success": False,
        "orb_count": 0,
        "daily_count": 0,
        "bars_count": 0,
        "symbols_count": 0,
    }

    if not bootstrap.is_available():
        logger.info("BOOTSTRAP | Sidecar data not available")
        return result

    # Load ORB
    orb_levels = bootstrap.load_orb()
    if orb_levels and orb_cache is not None:
        orb_cache.update(orb_levels)
        result["orb_count"] = len(orb_levels)

    # Load daily levels
    daily_levels = bootstrap.load_daily_levels()
    if daily_levels:
        result["daily_count"] = len(daily_levels)

    # Load bars
    bars = bootstrap.load_bars()
    if bars:
        result["symbols_count"] = len(bars)
        result["bars_count"] = sum(len(df) for df in bars.values())

        # Populate bar_builder if provided
        if bar_builder is not None:
            for symbol, df in bars.items():
                # Inject bars into bar_builder's internal state
                if hasattr(bar_builder, "_bars_5m"):
                    bar_builder._bars_5m[symbol] = df.copy()

    result["success"] = True
    logger.info(f"BOOTSTRAP | Complete: {result['symbols_count']} symbols, "
               f"{result['bars_count']} bars, {result['orb_count']} ORB levels")

    return result
