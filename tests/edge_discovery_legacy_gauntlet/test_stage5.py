"""Stage 5 tests: template generator for human narrative gate."""
from pathlib import Path
import pytest

from tools.edge_discovery_legacy_gauntlet.stages.stage5_narrative import generate_narrative_templates


def test_template_contains_all_required_sections(tmp_path):
    stage3_survivors = [
        {"setup": "setup_a_long", "rule": "setup_a_long__regime=trend_down"},
    ]
    stage3_details = [
        {
            "setup": "setup_a_long",
            "dim_count": 1, "conditioner": "regime", "cell_value": "trend_down",
            "n": 250, "pf": 1.45, "pf_h1": 1.3, "pf_h2": 1.2,
            "wr_pct": 55.0, "avg_pnl": 150, "passed": True,
        }
    ]
    out_dir = tmp_path / "narratives"
    generate_narrative_templates(stage3_survivors, stage3_details, out_dir)
    # One file per survivor (excluding index file)
    files = [p for p in out_dir.glob("*.md") if p.name != "00-index.md"]
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8")
    # Required sections
    assert "Canonical pro definition" in text
    assert "WHY does this work" in text
    assert "market participant" in text
    assert "[ ] APPROVED" in text
    assert "[ ] REJECTED" in text
    # Stats table present
    assert "n=250" in text or "N=250" in text or "| 250 |" in text
    assert "1.45" in text  # pf
