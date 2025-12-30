"""
Session Upload Persistence Layer

Tracks which trading sessions have been uploaded to OCI.
Used for crash recovery: on startup, upload any sessions that weren't
uploaded due to crashes.

Follows the same pattern as PositionPersistence.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    from config.logging_config import get_agent_logger
    logger = get_agent_logger()
    if logger is None:
        raise ValueError("get_agent_logger returned None")
except Exception:
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)


class SessionUploadTracker:
    """
    Tracks uploaded sessions to OCI buckets.

    State file format:
    {
        "paper_20251229_073712": {
            "bucket": "paper-trading-logs",
            "uploaded_at": "2024-12-29T15:30:00",
            "files_count": 5
        },
        ...
    }
    """

    def __init__(self, state_dir: Path):
        """
        Initialize tracker.

        Args:
            state_dir: Directory for state file (typically project_root/state/)
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "uploaded_sessions.json"
        self._cache: Dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        """Load state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    self._cache = json.load(f)
                logger.debug(f"[UPLOAD_TRACKER] Loaded {len(self._cache)} uploaded sessions")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"[UPLOAD_TRACKER] Failed to load state: {e}")
                self._cache = {}
        else:
            self._cache = {}

    def _save(self) -> None:
        """Save state to file atomically."""
        temp_file = self.state_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w") as f:
                json.dump(self._cache, f, indent=2)
            temp_file.replace(self.state_file)
        except IOError as e:
            logger.error(f"[UPLOAD_TRACKER] Failed to save state: {e}")
            if temp_file.exists():
                temp_file.unlink()

    def is_uploaded(self, session_id: str) -> bool:
        """Check if a session has been uploaded."""
        return session_id in self._cache

    def mark_uploaded(self, session_id: str, bucket: str, files_count: int = 0) -> None:
        """
        Mark a session as uploaded.

        Args:
            session_id: Session folder name (e.g., "paper_20251229_073712")
            bucket: OCI bucket name
            files_count: Number of files uploaded
        """
        self._cache[session_id] = {
            "bucket": bucket,
            "uploaded_at": datetime.now().isoformat(),
            "files_count": files_count
        }
        self._save()
        logger.info(f"[UPLOAD_TRACKER] Marked as uploaded: {session_id} -> {bucket}")

    def get_pending_sessions(self, logs_dir: Path) -> List[Path]:
        """
        Find sessions that exist locally but haven't been uploaded.

        Args:
            logs_dir: Directory containing session folders (typically project_root/logs/)

        Returns:
            List of session directory paths that need uploading
        """
        pending = []
        logs_path = Path(logs_dir)

        if not logs_path.exists():
            return pending

        for item in logs_path.iterdir():
            if not item.is_dir():
                continue

            # Only consider paper_* and live_* directories
            if not (item.name.startswith("paper_") or item.name.startswith("live_")):
                continue

            # Must have events.jsonl to be a valid session
            events_file = item / "events.jsonl"
            if not events_file.exists():
                continue

            # Skip if already uploaded
            if self.is_uploaded(item.name):
                continue

            pending.append(item)

        # Sort by name (oldest first, since name contains timestamp)
        pending.sort(key=lambda p: p.name)

        if pending:
            logger.info(f"[UPLOAD_TRACKER] Found {len(pending)} pending sessions to upload")

        return pending

    def get_session_mode(self, session_id: str) -> str:
        """
        Determine mode (paper/live) from session ID.

        Args:
            session_id: Session folder name

        Returns:
            "paper" or "live"
        """
        if session_id.startswith("paper_"):
            return "paper"
        elif session_id.startswith("live_"):
            return "live"
        else:
            # Default to paper for safety
            return "paper"

    def cleanup_old_entries(self, keep_days: int = 30) -> int:
        """
        Remove old entries from the tracker to prevent unbounded growth.

        Args:
            keep_days: Keep entries from the last N days

        Returns:
            Number of entries removed
        """
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(days=keep_days)
        to_remove = []

        for session_id, info in self._cache.items():
            try:
                uploaded_at = datetime.fromisoformat(info.get("uploaded_at", ""))
                if uploaded_at < cutoff:
                    to_remove.append(session_id)
            except (ValueError, TypeError):
                # Can't parse date, keep it
                pass

        for session_id in to_remove:
            del self._cache[session_id]

        if to_remove:
            self._save()
            logger.info(f"[UPLOAD_TRACKER] Cleaned up {len(to_remove)} old entries")

        return len(to_remove)
