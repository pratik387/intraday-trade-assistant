"""End-to-end test: load a synthetic mini-run, execute stages 1-3-5,
verify output files exist."""
from datetime import date
from pathlib import Path
import pytest

from tests.edge_discovery_legacy_gauntlet.fixtures.make_fixtures import make_mini_run
from tools.edge_discovery_legacy_gauntlet.run_gauntlet import run_gauntlet_all


@pytest.fixture
def mini_run(tmp_path):
    """Larger synthetic fixture - needs enough trades for Stage 1 N>=500."""
    root = tmp_path / "mini_run"
    dates = []
    for m in range(1, 13):
        for d in [1, 8, 15, 22]:
            dates.append(f"2023-{m:02d}-{d:02d}")
            dates.append(f"2024-{m:02d}-{d:02d}")
    make_mini_run(root, dates, trades_per_session=10)
    return root


def test_run_gauntlet_all_produces_stage_reports(mini_run, tmp_path):
    out_dir = tmp_path / "out"
    cfg_dates = {
        "discovery_start": date(2023, 1, 1),
        "discovery_end": date(2024, 12, 31),
        "validation_start": date(2025, 1, 1),
        "validation_end": date(2025, 9, 30),
        "holdout_start": date(2025, 10, 1),
        "holdout_end": date(2026, 3, 31),
    }
    result = run_gauntlet_all(
        backtest_dir=mini_run,
        output_dir=out_dir,
        cfg_dates=cfg_dates,
    )
    # Reports written
    assert (out_dir / "01-universe-pruning.md").exists()
    assert (out_dir / "02-univariate-screening.md").exists()
    assert (out_dir / "03-conditional-edge.md").exists()
    assert (out_dir / "05-narrative-gate").is_dir()
    # JSON artifacts
    assert (out_dir / "stage1_survivors.json").exists()
    assert (out_dir / "stage2_survivors.json").exists()
    assert (out_dir / "stage3_survivors.json").exists()
    # Result struct
    assert "stage1_count" in result
    assert "stage2_count" in result
    assert "stage3_count" in result
    assert "narrative_templates_generated" in result
