"""Tests for report writer helpers."""
import json
from pathlib import Path
import pandas as pd

from tools.edge_discovery.report_writer import (
    write_stage_report,
    append_section,
    write_json_artifact,
)


def test_write_stage_report_creates_file(tmp_path):
    path = tmp_path / "stage1.md"
    write_stage_report(
        path=path,
        stage_name="Stage 1 — Universe Pruning",
        criteria="N ≥ 500 AND PF ≥ 0.8",
        summary_rows=[
            {"setup": "a_long", "n": 600, "pf": 1.1, "passed": True},
            {"setup": "b_short", "n": 300, "pf": 0.5, "passed": False},
        ],
    )
    text = path.read_text(encoding="utf-8")
    assert "Stage 1" in text
    assert "N ≥ 500" in text
    assert "a_long" in text
    assert "b_short" in text
    assert "PASS" in text and "FAIL" in text


def test_append_section(tmp_path):
    path = tmp_path / "stage2.md"
    path.write_text("# Existing header\n", encoding="utf-8")
    append_section(path, "## New Section", "Some content")
    text = path.read_text(encoding="utf-8")
    assert "# Existing header" in text
    assert "## New Section" in text
    assert "Some content" in text


def test_write_json_artifact(tmp_path):
    path = tmp_path / "survivors.json"
    data = {"survivors": ["a_long"], "killed": ["b_short"]}
    write_json_artifact(path, data)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == data
