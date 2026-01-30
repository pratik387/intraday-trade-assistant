"""
Daily Cache Persistence Layer

Provides disk persistence for daily OHLCV cache to survive server restarts.
On first startup of the day, cache is fetched from Kite API (~15 min).
On subsequent restarts, cache is loaded from disk (~2-5 sec).

Similar pattern to PositionPersistence - atomic writes, thread-safe, auto-cleanup.
"""
from __future__ import annotations

import pickle
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

try:
    from config.logging_config import get_agent_logger
    logger = get_agent_logger()
    if logger is None:
        raise ValueError("get_agent_logger returned None")
except Exception:
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)


class DailyCachePersistence:
    """
    Handles daily OHLCV cache persistence to survive restarts.

    Usage:
        persistence = DailyCachePersistence()

        # On startup - try to load from disk
        cached_data = persistence.load_today()
        if cached_data:
            sdk.set_daily_cache(cached_data)
        else:
            sdk.prewarm_daily_cache(days=210)
            persistence.save(sdk.get_daily_cache())
    """

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        """
        Initialize persistence layer.

        Args:
            cache_dir: Directory to store cache files. Defaults to cache/daily_cache/
        """
        if cache_dir is None:
            # Default: project_root/cache/daily_cache/
            cache_dir = Path(__file__).resolve().parents[2] / "cache" / "daily_cache"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _get_cache_file_path(self, date: Optional[str] = None) -> Path:
        """Get the cache file path for a specific date."""
        if date is None:
            date = datetime.now().date().isoformat()
        return self.cache_dir / f"daily_cache_{date}.pkl"

    def load_today(self) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Load today's daily cache from disk if it exists.

        Returns:
            Dict mapping symbol to DataFrame, or None if not found/corrupted
        """
        cache_file = self._get_cache_file_path()
        if not cache_file.exists():
            logger.info("DAILY_CACHE | No disk cache found for today")
            return None

        try:
            start_time = time.perf_counter()

            with self._lock:
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)

            cache = data.get("cache", {})
            cached_date = data.get("date")
            today = datetime.now().date().isoformat()

            # Verify cache is for today
            if cached_date != today:
                logger.warning(f"DAILY_CACHE | Cache date mismatch: {cached_date} != {today}")
                return None

            elapsed = time.perf_counter() - start_time
            logger.info(f"DAILY_CACHE | Loaded {len(cache)} symbols from disk in {elapsed:.2f}s")
            return cache

        except Exception as e:
            logger.warning(f"DAILY_CACHE | Failed to load from disk: {e}")
            return None

    def save(self, cache: Dict[str, pd.DataFrame]) -> bool:
        """
        Save daily cache to disk for persistence across restarts.

        Args:
            cache: Dict mapping symbol to DataFrame

        Returns:
            True if saved successfully, False otherwise
        """
        today = datetime.now().date().isoformat()
        cache_file = self._get_cache_file_path(today)
        temp_file = cache_file.with_suffix(".tmp")

        try:
            start_time = time.perf_counter()

            data = {
                "date": today,
                "cache": cache,
                "timestamp": datetime.now().isoformat(),
                "symbol_count": len(cache)
            }

            # Atomic write: write to temp, then rename
            with self._lock:
                with open(temp_file, "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                temp_file.replace(cache_file)  # Atomic on most systems

            elapsed = time.perf_counter() - start_time
            size_mb = cache_file.stat().st_size / (1024 * 1024)
            logger.info(f"DAILY_CACHE | Saved {len(cache)} symbols to disk ({size_mb:.1f} MB) in {elapsed:.2f}s")

            # Cleanup old cache files (keep only last 3 days)
            self._cleanup_old_files()

            return True

        except Exception as e:
            logger.error(f"DAILY_CACHE | Failed to save to disk: {e}")
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass
            return False

    def _cleanup_old_files(self, keep_days: int = 3) -> None:
        """Remove cache files older than keep_days."""
        try:
            today = datetime.now().date()
            for cache_file in self.cache_dir.glob("daily_cache_*.pkl"):
                try:
                    # Extract date from filename: daily_cache_2025-12-19.pkl
                    date_str = cache_file.stem.replace("daily_cache_", "")
                    file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    age_days = (today - file_date).days
                    if age_days > keep_days:
                        cache_file.unlink()
                        logger.debug(f"DAILY_CACHE | Deleted old cache file: {cache_file.name}")
                except (ValueError, OSError):
                    continue
        except Exception as e:
            logger.debug(f"DAILY_CACHE | Cleanup error: {e}")

    def load_latest(self) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Load the most recent daily cache (today's or yesterday's).

        Rolling cache design: MDS builds today's daily bar from 1m data at EOD
        and saves it. Next morning, this file is loaded — cache has 210 days
        ending at yesterday's close, which is current. No API fetch needed.

        On first day (cold start), today's cache won't exist and no rolling
        cache is available, so prewarm_daily_cache() does a full API fetch.

        Returns:
            Dict mapping symbol to DataFrame, or None if no recent cache
        """
        # Try today first (same as load_today)
        today_cache = self.load_today()
        if today_cache is not None:
            return today_cache

        # Fall back to most recent cache file
        cache_files = sorted(
            self.cache_dir.glob("daily_cache_*.pkl"),
            key=lambda p: p.stem,
            reverse=True,
        )

        for cache_file in cache_files:
            try:
                date_str = cache_file.stem.replace("daily_cache_", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                age_days = (datetime.now().date() - file_date).days

                # Only use caches from last 3 days (avoids stale weekend data)
                if age_days > 3:
                    continue

                start_time = time.perf_counter()

                with self._lock:
                    with open(cache_file, "rb") as f:
                        data = pickle.load(f)

                cache = data.get("cache", {})
                if not cache:
                    continue

                elapsed = time.perf_counter() - start_time
                logger.info(
                    f"DAILY_CACHE | Rolling load: {len(cache)} symbols from "
                    f"{cache_file.name} ({age_days} day(s) old) in {elapsed:.2f}s"
                )
                return cache

            except Exception as e:
                logger.warning(f"DAILY_CACHE | Failed to load {cache_file.name}: {e}")
                continue

        logger.info("DAILY_CACHE | No recent cache files found")
        return None

    def exists_today(self) -> bool:
        """Check if today's cache file exists."""
        return self._get_cache_file_path().exists()

    def get_cache_info(self) -> Dict[str, Any]:
        """Get info about current cache file."""
        cache_file = self._get_cache_file_path()
        if not cache_file.exists():
            return {"exists": False}

        try:
            stat = cache_file.stat()
            return {
                "exists": True,
                "path": str(cache_file),
                "size_mb": stat.st_size / (1024 * 1024),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            }
        except OSError:
            return {"exists": False}
