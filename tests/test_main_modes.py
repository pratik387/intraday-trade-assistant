"""Tests for main.py --mode and --action CLI flags."""
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PY = _REPO_ROOT / ".venv" / "Scripts" / "python.exe"
if not _PY.exists():
    # Fallback for non-Windows or different venv layouts
    _PY = sys.executable


def _run(args: list, timeout: int = 30):
    return subprocess.run(
        [str(_PY), str(_REPO_ROOT / "main.py")] + args,
        capture_output=True, text=True, timeout=timeout, cwd=str(_REPO_ROOT),
    )


def test_intraday_mode_default_does_not_error_on_help():
    """--help should work and mention --mode flag."""
    r = _run(["--help"], timeout=10)
    assert r.returncode == 0
    assert "--mode" in r.stdout
    assert "--action" in r.stdout


def test_intraday_mode_with_overnight_action_errors():
    """--mode=intraday with --action=entry should be a usage error."""
    r = _run(["--mode", "intraday", "--action", "entry"], timeout=10)
    assert r.returncode == 2
    assert "intraday" in r.stderr.lower()
    assert "action" in r.stderr.lower()


def test_overnight_mode_with_run_action_errors():
    """--mode=overnight --action=run is a usage error (no daemon for overnight)."""
    r = _run(["--mode", "overnight", "--action", "run"], timeout=10)
    assert r.returncode == 2
    assert "daemon" in r.stderr.lower() or "no daemon" in r.stderr.lower()


def test_overnight_verify_exit_dry_run_exits_cleanly():
    """--dry-run --mode=overnight --action=verify-exit should exit code 0.

    No state file is required: handler returns early with zero settled.
    Verifies wiring works end-to-end through the CLI.
    """
    r = _run(
        ["--dry-run", "--mode", "overnight", "--action", "verify-exit",
         "--session-date", "2024-03-15"],
        timeout=60,
    )
    assert r.returncode == 0, f"stdout={r.stdout!r} stderr={r.stderr!r}"
    assert "overnight verify-exit" in r.stderr.lower() or "settled=" in r.stderr.lower()
