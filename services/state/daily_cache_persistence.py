"""
Daily Cache Persistence Layer

Provides disk persistence for daily OHLCV cache to survive server restarts.
On first startup of the day, cache is fetched from Kite API (~15 min).
On subsequent restarts, cache is loaded from disk (~2-5 sec).

Similar pattern to PositionPersistence - atomic writes, thread-safe, auto-cleanup.
"""
from __future__ import annotations

import os
import pickle
import threading
import time
from collections import Counter
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional

from utils.time_util import _now_naive_ist

_IS_WINDOWS = os.name == "nt"
if _IS_WINDOWS:
    import msvcrt
else:
    import fcntl

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

    Usage (see main._prewarm_daily_cache):
        p = DailyCachePersistence()
        with p.prewarm_lock(timeout_sec=...):
            cache = p.load_current(today=..., prev_trading_day=..., days=210)
            if cache:
                sdk.set_daily_cache(cache, days=210)
            sdk.prewarm_daily_cache(days=210)       # fetches only the gaps
            p.save(sdk.get_daily_cache(min_days=210), days=210, today=...)
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

    def load_current(
        self, *, today: date, prev_trading_day: date, days: int,
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """Newest snapshot that is CURRENT for `today`, else None (API fetch).

        A snapshot is only written right after a morning API fetch, so the file
        for day D holds bars up to D-1. Current therefore means, by content:
          - fetched at depth >= `days` (pre-fix snapshots carry no depth: rejected),
          - its most common last-bar date == `prev_trading_day`.
        The filename date is NOT trusted: yesterday's morning snapshot ends two
        sessions back and used to be loaded as current (2026-10 bug). Empty
        frames (failed fetches) are dropped so prewarm refetches them.
        """
        files = sorted(self.cache_dir.glob("daily_cache_*.pkl"), key=lambda p: p.stem, reverse=True)
        for cache_file in files:
            try:
                with self._lock:
                    with open(cache_file, "rb") as f:
                        data = pickle.load(f)
            except Exception as e:
                logger.warning(f"DAILY_CACHE | Failed to load {cache_file.name}: {e}")
                continue

            snap_days = data.get("days")
            cache = {k: v for k, v in (data.get("cache") or {}).items()
                     if v is not None and not v.empty}
            if not cache:
                continue
            if snap_days is None or int(snap_days) < days:
                logger.info(
                    "DAILY_CACHE | %s fetched at depth %s < %d -> not usable",
                    cache_file.name, snap_days, days,
                )
                continue
            last_bars = Counter(pd.Timestamp(v.index[-1]).date() for v in cache.values())
            last_bar = last_bars.most_common(1)[0][0]
            if last_bar != prev_trading_day:
                logger.info(
                    "DAILY_CACHE | %s ends %s, need %s for %s -> stale",
                    cache_file.name, last_bar, prev_trading_day, today,
                )
                continue
            # Illiquid names with no trade on the last session end earlier;
            # counted, not refetched (the snapshot itself was fetched this morning).
            n_lagging = sum(n for d, n in last_bars.items() if d < last_bar)
            logger.info(
                "DAILY_CACHE | Loaded %d symbols from %s (ends %s, depth %s, %d end earlier)",
                len(cache), cache_file.name, last_bar, snap_days, n_lagging,
            )
            return cache

        logger.info("DAILY_CACHE | No current snapshot for %s -> API fetch required", today)
        return None

    def save(self, cache: Dict[str, pd.DataFrame], *, days: int, today: date) -> bool:
        """Save the daily cache, recording the depth it was fetched at."""
        cache_file = self._get_cache_file_path(today.isoformat())
        temp_file = cache_file.with_suffix(".tmp")

        try:
            start_time = time.perf_counter()

            data = {
                "date": today.isoformat(),
                "days": int(days),
                "cache": cache,
                "timestamp": _now_naive_ist().isoformat(),
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
            self._cleanup_old_files(today)

            return True

        except Exception as e:
            logger.error(f"DAILY_CACHE | Failed to save to disk: {e}")
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass
            return False

    def _cleanup_old_files(self, today: date, keep_days: int = 3) -> None:
        """Remove cache files older than keep_days."""
        try:
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

    # --- cross-process prewarm lock ------------------------------------------
    # Both intraday daemons share one IP-limited Upstox budget (and, on the VM,
    # one cache dir via symlink). The first daemon fetches and saves; the second
    # blocks here, then loads that snapshot instead of fetching again.

    def try_prewarm_lock(self) -> bool:
        """Non-blocking attempt to take the prewarm lock."""
        fh = open(self.cache_dir / ".prewarm.lock", "a+b")
        try:
            if _IS_WINDOWS:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        self._lock_fh = fh
        return True

    def release_prewarm_lock(self) -> None:
        fh = getattr(self, "_lock_fh", None)
        if fh is None:
            return
        try:
            if _IS_WINDOWS:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()
            self._lock_fh = None

    @contextmanager
    def prewarm_lock(self, *, timeout_sec: float):
        """Hold the prewarm lock; waits up to `timeout_sec`, then raises."""
        deadline = time.monotonic() + timeout_sec
        waited = False
        while not self.try_prewarm_lock():
            if not waited:
                logger.info("DAILY_CACHE | Another process is prewarming; waiting for its snapshot")
                waited = True
            if time.monotonic() >= deadline:
                raise TimeoutError(f"DAILY_CACHE | prewarm lock not acquired in {timeout_sec:.0f}s")
            time.sleep(1.0)
        try:
            yield
        finally:
            self.release_prewarm_lock()

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
