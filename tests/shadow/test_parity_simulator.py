"""Parity simulator tests (sub-project #4)."""
import subprocess
import sys
from pathlib import Path


def test_cli_help_runs():
    """CLI must respond to --help without ImportError."""
    result = subprocess.run(
        [sys.executable, "tools/shadow/parity_simulator.py", "--help"],
        capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "--gate-input" in result.stdout
    assert "--config" in result.stdout
    assert "--output" in result.stdout
