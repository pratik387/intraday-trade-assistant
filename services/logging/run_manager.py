# run_manager.py
"""
Run Management System
--------------------
Tracks which log sessions belong to the same trading run.
A "run" can span multiple days/sessions but represents a single logical execution.
"""

import json
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

class RunManager:
    """Manages trading run metadata and session grouping"""

    def __init__(self, logs_dir: Path):
        self.logs_dir = Path(logs_dir)
        self.manifest_file = self.logs_dir / "current_run.json"
        self.lock = threading.Lock()

    def start_new_run(self, run_description: str = "") -> str:
        """Start a new trading run and return run_id"""
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        manifest = {
            "run_id": run_id,
            "description": run_description,
            "start_time": datetime.now().isoformat(),
            "sessions": [],
            "status": "active"
        }

        with self.lock:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            with open(self.manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)

        return run_id

    def add_session_to_current_run(self, session_id: str) -> bool:
        """Add a session to the current active run"""
        with self.lock:
            if not self.manifest_file.exists():
                # Auto-start a run if none exists
                self.start_new_run("Auto-started run")

            try:
                with open(self.manifest_file, 'r') as f:
                    manifest = json.load(f)

                if session_id not in manifest.get("sessions", []):
                    manifest["sessions"].append(session_id)
                    manifest["last_updated"] = datetime.now().isoformat()

                with open(self.manifest_file, 'w') as f:
                    json.dump(manifest, f, indent=2)

                return True

            except (FileNotFoundError, json.JSONDecodeError):
                return False

    def get_current_run_sessions(self) -> List[str]:
        """Get list of session IDs in the current run"""
        with self.lock:
            if not self.manifest_file.exists():
                return []

            try:
                with open(self.manifest_file, 'r') as f:
                    manifest = json.load(f)
                return manifest.get("sessions", [])
            except (FileNotFoundError, json.JSONDecodeError):
                return []

    def end_current_run(self) -> Optional[Dict[str, Any]]:
        """Mark current run as completed and return manifest"""
        with self.lock:
            if not self.manifest_file.exists():
                return None

            try:
                with open(self.manifest_file, 'r') as f:
                    manifest = json.load(f)

                manifest["status"] = "completed"
                manifest["end_time"] = datetime.now().isoformat()

                # Archive the run
                archived_file = self.logs_dir / f"{manifest['run_id']}.json"
                with open(archived_file, 'w') as f:
                    json.dump(manifest, f, indent=2)

                # Remove current run file
                self.manifest_file.unlink()

                return manifest

            except (FileNotFoundError, json.JSONDecodeError):
                return None

    def get_current_run_info(self) -> Optional[Dict[str, Any]]:
        """Get info about the current active run"""
        with self.lock:
            if not self.manifest_file.exists():
                return None

            try:
                with open(self.manifest_file, 'r') as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                return None

# Global instance
_run_manager: Optional[RunManager] = None

def get_run_manager() -> RunManager:
    """Get the global run manager instance"""
    global _run_manager
    if _run_manager is None:
        # Use fixed path to avoid circular dependency with logging_config
        logs_root = Path(__file__).resolve().parents[2] / "logs"
        _run_manager = RunManager(logs_root)
    return _run_manager