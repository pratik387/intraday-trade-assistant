#!/usr/bin/env python3
"""
Upload Trading Session Logs to OCI Object Storage

Uploads a trading session folder (paper/live) to OCI bucket.
Can be run standalone or called from main.py on shutdown.

Usage:
    python oci/tools/upload_trading_session.py paper_20251229_073712
    python oci/tools/upload_trading_session.py logs/paper_20251229_073712
    python oci/tools/upload_trading_session.py  # Auto-discovers latest session
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import oci
    HAS_OCI = True
except ImportError:
    HAS_OCI = False

try:
    from config.logging_config import get_agent_logger
    logger = get_agent_logger()
    if logger is None:
        raise ValueError()
except Exception:
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)


# Bucket configuration
BUCKETS = {
    "paper": "paper-trading-logs",
    "live": "live-trading-logs"
}


def get_mode_from_session(session_id: str) -> str:
    """Determine mode (paper/live) from session ID."""
    if session_id.startswith("paper_"):
        return "paper"
    elif session_id.startswith("live_"):
        return "live"
    return "paper"  # Default to paper


def find_session_dir(session_arg: Optional[str]) -> Optional[Path]:
    """
    Find the session directory from argument or auto-discover latest.
    """
    if session_arg:
        # Try as-is first
        if Path(session_arg).exists():
            return Path(session_arg)

        # Try under logs/
        logs_path = ROOT / "logs" / session_arg
        if logs_path.exists():
            return logs_path

        # Try in current directory
        cwd_path = Path.cwd() / session_arg
        if cwd_path.exists():
            return cwd_path

        print(f"[ERROR] Session directory not found: {session_arg}")
        return None

    # Auto-discover latest session
    print("[INFO] No session specified, auto-discovering latest...")

    candidates = []
    logs_dir = ROOT / "logs"
    if logs_dir.exists():
        for item in logs_dir.iterdir():
            if item.is_dir() and (item.name.startswith("paper_") or item.name.startswith("live_")):
                events_file = item / "events.jsonl"
                if events_file.exists():
                    candidates.append(item)

    if not candidates:
        print("[ERROR] No session directories found")
        return None

    # Sort by modification time (newest first)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest = candidates[0]
    print(f"[INFO] Found latest session: {latest.name}")
    return latest


def run_analyzer(session_dir: Path) -> bool:
    """
    Run the comprehensive analyzer on the session.

    Returns:
        True if successful
    """
    print(f"\n[1/2] Running analyzer on {session_dir.name}...")

    analyzer_path = ROOT / "tools" / "live_trading_comprehensive_run_analyzer.py"
    if not analyzer_path.exists():
        print(f"  [WARN] Analyzer not found: {analyzer_path}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, str(analyzer_path), str(session_dir)],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=300
        )

        if result.returncode == 0:
            print("  [OK] Analysis completed")
            return True
        else:
            print(f"  [WARN] Analysis returned code {result.returncode}")
            if result.stderr:
                print(f"  {result.stderr[:200]}")
            return False

    except subprocess.TimeoutExpired:
        print("  [WARN] Analysis timed out")
        return False
    except Exception as e:
        print(f"  [WARN] Analysis error: {e}")
        return False


class SessionUploader:
    """Handles uploading session files to OCI."""

    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name

        if not HAS_OCI:
            raise ImportError("OCI SDK not installed. Run: pip install oci")

        print("Initializing OCI client...")
        self.config = oci.config.from_file()
        self.os_client = oci.object_storage.ObjectStorageClient(self.config)
        self.namespace = self.os_client.get_namespace().data
        print(f"Connected to OCI (namespace: {self.namespace})")

    def upload_file(self, local_path: Path, object_name: str) -> bool:
        """Upload a single file to OCI."""
        try:
            with open(local_path, 'rb') as f:
                self.os_client.put_object(
                    namespace_name=self.namespace,
                    bucket_name=self.bucket_name,
                    object_name=object_name,
                    put_object_body=f
                )
            return True
        except Exception as e:
            logger.warning(f"Failed to upload {local_path}: {e}")
            return False

    def upload_session(self, session_dir: Path, parallel: int = 5) -> Tuple[int, int]:
        """
        Upload all files in session directory.

        Args:
            session_dir: Path to session directory
            parallel: Number of parallel uploads

        Returns:
            Tuple of (uploaded_count, failed_count)
        """
        session_id = session_dir.name

        # Collect all files
        files_to_upload = []
        for file_path in session_dir.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(session_dir)
                object_name = f"{session_id}/{relative_path}"
                files_to_upload.append((file_path, object_name))

        if not files_to_upload:
            print("  No files to upload")
            return 0, 0

        print(f"  Uploading {len(files_to_upload)} files to {self.bucket_name}...")

        uploaded = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(self.upload_file, fp, on): (fp, on)
                for fp, on in files_to_upload
            }

            for future in as_completed(futures):
                if future.result():
                    uploaded += 1
                else:
                    failed += 1

                # Progress update
                total = uploaded + failed
                if total % 5 == 0 or total == len(files_to_upload):
                    print(f"  Progress: {total}/{len(files_to_upload)}", end='\r')

        print()  # New line after progress
        return uploaded, failed


def upload_session(session_dir: Path, mode: str) -> bool:
    """
    Upload a trading session to OCI.

    Args:
        session_dir: Path to session directory
        mode: "paper" or "live"

    Returns:
        True if upload successful
    """
    session_id = session_dir.name
    bucket = BUCKETS.get(mode, BUCKETS["paper"])

    print(f"\n{'='*60}")
    print(f"UPLOAD TRADING SESSION TO OCI")
    print(f"{'='*60}")
    print(f"Session: {session_id}")
    print(f"Mode: {mode}")
    print(f"Bucket: {bucket}")
    print(f"{'='*60}")

    # Step 1: Run analyzer
    run_analyzer(session_dir)

    # Step 2: Upload files
    print(f"\n[2/2] Uploading to OCI...")

    try:
        uploader = SessionUploader(bucket)
        uploaded, failed = uploader.upload_session(session_dir)

        if uploaded > 0:
            print(f"\n[OK] Uploaded {uploaded} files to oci://{bucket}/{session_id}/")
            if failed > 0:
                print(f"  [WARN] {failed} files failed to upload")
            return True
        else:
            print(f"\n[ERROR] No files uploaded")
            return False

    except ImportError as e:
        print(f"\n[ERROR] OCI SDK not available: {e}")
        return False
    except Exception as e:
        print(f"\n[ERROR] Upload failed: {e}")
        return False


def upload_pending_sessions(
    tracker,  # SessionUploadTracker
    logs_dir: Path,
    skip_current: Optional[str] = None
) -> int:
    """
    Upload any pending sessions from previous runs.

    Called on startup to handle crash recovery.

    Args:
        tracker: SessionUploadTracker instance
        logs_dir: Directory containing session folders
        skip_current: Current session ID to skip (still running)

    Returns:
        Number of sessions uploaded
    """
    pending = tracker.get_pending_sessions(logs_dir)

    # Skip current session if specified
    if skip_current:
        pending = [p for p in pending if p.name != skip_current]

    if not pending:
        return 0

    print(f"\n[STARTUP] Found {len(pending)} pending session(s) to upload")

    uploaded_count = 0
    for session_dir in pending:
        session_id = session_dir.name
        mode = get_mode_from_session(session_id)

        print(f"\n[STARTUP] Uploading previous session: {session_id}")

        try:
            if upload_session(session_dir, mode):
                bucket = BUCKETS.get(mode, BUCKETS["paper"])

                # Count files
                files_count = sum(1 for _ in session_dir.rglob('*') if _.is_file())

                tracker.mark_uploaded(session_id, bucket, files_count)
                uploaded_count += 1
        except Exception as e:
            print(f"[STARTUP] Failed to upload {session_id}: {e}")
            # Continue with next session

    return uploaded_count


def main():
    parser = argparse.ArgumentParser(
        description='Upload trading session logs to OCI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python oci/tools/upload_trading_session.py paper_20251229_073712
  python oci/tools/upload_trading_session.py logs/paper_20251229_073712
  python oci/tools/upload_trading_session.py  # Auto-discover latest
        """
    )

    parser.add_argument('session', nargs='?', help='Session folder name or path')
    parser.add_argument('--mode', choices=['paper', 'live'],
                        help='Override mode detection (default: auto-detect from session name)')
    parser.add_argument('--skip-analysis', action='store_true',
                        help='Skip running the analyzer')

    args = parser.parse_args()

    # Find session directory
    session_dir = find_session_dir(args.session)
    if not session_dir:
        print("\nUsage:")
        print("  python oci/tools/upload_trading_session.py paper_20251229_073712")
        print("  python oci/tools/upload_trading_session.py  # Auto-discover latest")
        return 1

    # Determine mode
    mode = args.mode or get_mode_from_session(session_dir.name)

    # Upload
    success = upload_session(session_dir, mode)

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
