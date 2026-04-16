"""Unit tests for oci/docker/entrypoint.py gzip-on-upload helper.

Verifies that detector_rejections.jsonl and detector_accepts.jsonl get
compressed before upload, while files like events.jsonl (consumed by
existing post-processing) are left untouched.
"""

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT_PATH = ROOT / "oci" / "docker" / "entrypoint.py"


def _load_entrypoint_module():
    """Load entrypoint.py as a module without executing main() and
    without requiring oci SDK at import time (the module imports oci
    at module level — we stub it before loading)."""
    # Stub the oci import so module loads on dev machines without OCI SDK
    if "oci" not in sys.modules:
        import types
        oci_stub = types.ModuleType("oci")
        oci_stub.config = types.SimpleNamespace(from_file=lambda: {})
        oci_stub.object_storage = types.SimpleNamespace(
            ObjectStorageClient=lambda *a, **kw: None
        )
        oci_stub.exceptions = types.SimpleNamespace(ServiceError=Exception)
        sys.modules["oci"] = oci_stub

    spec = importlib.util.spec_from_file_location("oci_entrypoint", ENTRYPOINT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_gzip_helper_compresses_file_in_place(tmp_path):
    """_gzip_file_in_place reduces file size and removes the original."""
    mod = _load_entrypoint_module()

    src = tmp_path / "test.jsonl"
    # Build ~50KB of repetitive JSONL (gzips well)
    with open(src, "w") as f:
        for i in range(500):
            f.write(json.dumps({
                "ts": "2026-04-15T10:00:00",
                "symbol": "RELIANCE",
                "detector": "gap_fill_short",
                "reason": "gap_fill outside time window 0915-1030",
                "regime": "trend_up",
                "cap_segment": "large_cap",
            }) + "\n")
    size_before = src.stat().st_size

    new_path = mod._gzip_file_in_place(src)

    assert new_path.suffix == ".gz"
    assert new_path.name == "test.jsonl.gz"
    assert new_path.exists()
    assert not src.exists(), "Original file should have been deleted"

    size_after = new_path.stat().st_size
    assert size_after < size_before * 0.3, (
        f"Expected ~3:1+ compression on repetitive JSONL; "
        f"got {size_before} -> {size_after}"
    )

    # Verify contents are recoverable
    with gzip.open(new_path, "rt") as f:
        lines = f.read().strip().split("\n")
    assert len(lines) == 500
    assert json.loads(lines[0])["symbol"] == "RELIANCE"


def test_gzip_helper_skips_empty_files(tmp_path):
    """Empty files are not compressed (avoids creating useless .gz files)."""
    mod = _load_entrypoint_module()

    src = tmp_path / "empty.jsonl"
    src.touch()  # 0 bytes

    new_path = mod._gzip_file_in_place(src)

    assert new_path == src, "Empty file should be returned unchanged"
    assert src.exists(), "Empty file should not be deleted"
    assert not (tmp_path / "empty.jsonl.gz").exists()


def test_gzip_whitelist_includes_new_detector_logs():
    """The gzip whitelist must include both new detector log files
    plus agent.log; it must NOT include events.jsonl or other files
    that existing post-processing reads as plain text."""
    mod = _load_entrypoint_module()

    assert "detector_rejections.jsonl" in mod._GZIP_FILENAMES
    assert "detector_accepts.jsonl" in mod._GZIP_FILENAMES
    assert "agent.log" in mod._GZIP_FILENAMES

    # Critical: these MUST NOT be gzipped (post-processing depends on them)
    assert "events.jsonl" not in mod._GZIP_FILENAMES, (
        "events.jsonl is read by trading_logger.py + comprehensive_run_analyzer.py "
        "as plain text — gzipping it would break post-processing"
    )
    assert "analytics.jsonl" not in mod._GZIP_FILENAMES
    assert "screening.jsonl" not in mod._GZIP_FILENAMES
    assert "planning.jsonl" not in mod._GZIP_FILENAMES
