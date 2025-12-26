"""
ORB (Opening Range) Cache Persistence Layer

Provides disk persistence for ORH/ORL levels cache to survive server restarts.
ORB levels are computed once at 09:40 from the 09:15-09:30 opening range bars.
On restart after 09:40, these bars may not be available from broker's intraday API.

Similar pattern to DailyCachePersistence - atomic writes, thread-safe, auto-cleanup.
"""
from __future__ import annotations

import json
import math
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from config.logging_config import get_agent_logger
    logger = get_agent_logger()
    if logger is None:
        raise ValueError("get_agent_logger returned None")
except Exception:
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)


class ORBCachePersistence:
    """
    Handles ORB levels cache persistence to survive restarts.

    The ORB levels (ORH/ORL + PDH/PDL/PDC) are computed once at 09:40
    and used for the entire trading day. If the server restarts after
    this time, the opening range bars may not be available from the
    broker's historical API.

    Usage:
        persistence = ORBCachePersistence()

        # On ORB computation (09:40)
        persistence.save(session_date, levels_by_symbol)

        # On startup - try to load from disk
        cached = persistence.load_today()
        if cached:
            screener._orb_levels_cache[session_date] = cached
    """

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        """
        Initialize persistence layer.

        Args:
            cache_dir: Directory to store cache files. Defaults to cache/orb_cache/
        """
        if cache_dir is None:
            # Default: project_root/cache/orb_cache/
            cache_dir = Path(__file__).resolve().parents[2] / "cache" / "orb_cache"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _get_cache_file_path(self, date: Optional[str] = None) -> Path:
        """Get the cache file path for a specific date."""
        if date is None:
            date = datetime.now().date().isoformat()
        return self.cache_dir / f"orb_levels_{date}.json"

    def load_today(self) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Load today's ORB levels cache from disk if it exists.

        Returns:
            Dict mapping symbol to levels dict (PDH/PDL/PDC/ORH/ORL),
            or None if not found/corrupted
        """
        today = datetime.now().date().isoformat()
        cache_file = self._get_cache_file_path(today)

        if not cache_file.exists():
            logger.debug("ORB_CACHE_PERSIST | No disk cache found for today")
            return None

        try:
            start_time = time.perf_counter()

            with self._lock:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

            cached_date = data.get("date")

            # Verify cache is for today
            if cached_date != today:
                logger.warning(f"ORB_CACHE_PERSIST | Cache date mismatch: {cached_date} != {today}")
                return None

            raw_levels = data.get("levels", {})

            # Convert None values back to float("nan") for consistency
            # (JSON doesn't support NaN, so we store as None and convert back)
            levels = {}
            for sym, lvls in raw_levels.items():
                levels[sym] = {
                    k: (float("nan") if v is None else v)
                    for k, v in lvls.items()
                }

            elapsed = time.perf_counter() - start_time

            # Count valid ORH/ORL entries (check for NaN, not None)
            valid_orb = sum(1 for v in levels.values()
                           if not math.isnan(v.get("ORH", float("nan")))
                           and not math.isnan(v.get("ORL", float("nan"))))

            logger.info(
                f"ORB_CACHE_PERSIST | Loaded {len(levels)} symbols from disk "
                f"({valid_orb} with valid ORH/ORL) in {elapsed:.2f}s"
            )
            return levels

        except Exception as e:
            logger.warning(f"ORB_CACHE_PERSIST | Failed to load from disk: {e}")
            return None

    def save(self, session_date: str, levels_by_symbol: Dict[str, Dict[str, float]]) -> bool:
        """
        Save ORB levels cache to disk for persistence across restarts.

        Args:
            session_date: Date string (YYYY-MM-DD) for the session
            levels_by_symbol: Dict mapping symbol to levels dict

        Returns:
            True if saved successfully, False otherwise
        """
        # Normalize date format
        if hasattr(session_date, 'isoformat'):
            session_date = session_date.isoformat()

        cache_file = self._get_cache_file_path(session_date)
        temp_file = cache_file.with_suffix(".tmp")

        try:
            start_time = time.perf_counter()

            # Convert any NaN values to None for JSON serialization
            clean_levels = {}
            for sym, lvls in levels_by_symbol.items():
                clean_levels[sym] = {
                    k: (None if v != v else v)  # NaN check: NaN != NaN
                    for k, v in lvls.items()
                }

            data = {
                "date": session_date,
                "levels": clean_levels,
                "timestamp": datetime.now().isoformat(),
                "symbol_count": len(clean_levels)
            }

            # Atomic write: write to temp, then rename
            with self._lock:
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                temp_file.replace(cache_file)  # Atomic on most systems

            elapsed = time.perf_counter() - start_time
            size_kb = cache_file.stat().st_size / 1024

            # Count valid entries
            valid_orb = sum(1 for v in clean_levels.values()
                           if v.get("ORH") is not None and v.get("ORL") is not None)

            logger.info(
                f"ORB_CACHE_PERSIST | Saved {len(clean_levels)} symbols to disk "
                f"({valid_orb} with valid ORH/ORL, {size_kb:.1f} KB) in {elapsed:.2f}s"
            )

            # Cleanup old cache files (keep only last 3 days)
            self._cleanup_old_files()

            return True

        except Exception as e:
            logger.error(f"ORB_CACHE_PERSIST | Failed to save to disk: {e}")
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
            for cache_file in self.cache_dir.glob("orb_levels_*.json"):
                try:
                    # Extract date from filename: orb_levels_2025-12-19.json
                    date_str = cache_file.stem.replace("orb_levels_", "")
                    file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    age_days = (today - file_date).days
                    if age_days > keep_days:
                        cache_file.unlink()
                        logger.debug(f"ORB_CACHE_PERSIST | Deleted old cache file: {cache_file.name}")
                except (ValueError, OSError):
                    continue
        except Exception as e:
            logger.debug(f"ORB_CACHE_PERSIST | Cleanup error: {e}")

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
                "size_kb": stat.st_size / 1024,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            }
        except OSError:
            return {"exists": False}
