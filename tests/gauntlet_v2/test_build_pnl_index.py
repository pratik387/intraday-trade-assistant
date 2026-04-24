"""build_pnl_index tests (sub5-T5)."""
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.parent
FIXTURES = Path(__file__).parent / "fixtures"


def test_build_pnl_index_produces_parquet_with_expected_shape(tmp_path):
    out = tmp_path / "pnl.parquet"
    result = subprocess.run(
        [sys.executable, "tools/gauntlet_v2/build_pnl_index.py",
         "--oci-dir", str(FIXTURES / "mini_oci_run"),
         "--output", str(out)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert out.exists()
    df = pd.read_parquet(out)
    assert len(df) == 5  # 3 + 2 trades across 2 sessions
    assert set(df.columns) == {"session_date", "ts", "symbol", "setup_type",
                                 "total_trade_pnl", "r_multiple", "gross_exit_qty"}
    # session_date must be string (not datetime) for stable join key
    assert df["session_date"].dtype == "object"
    # Look up one row to verify values
    row = df[(df["symbol"] == "NSE:A") & (df["session_date"] == "2025-01-02")].iloc[0]
    assert row["total_trade_pnl"] == 150.0
    assert row["r_multiple"] == 1.5


def test_duplicate_admit_keys_aborts(tmp_path):
    """Two trades with same (session_date, ts, symbol, setup_type) = corruption signal."""
    bad_dir = tmp_path / "bad_run" / "2025-01-02"
    bad_dir.mkdir(parents=True)
    (bad_dir / "trade_report.csv").write_text(
        "symbol,decision_ts,setup_type,realized_pnl,r_multiple,gross_exit_qty\n"
        "NSE:A,2025-01-02T09:20:00,premium_zone_short,150.0,1.5,10\n"
        "NSE:A,2025-01-02T09:20:00,premium_zone_short,200.0,2.0,10\n"  # duplicate key
    )
    out = tmp_path / "pnl.parquet"
    result = subprocess.run(
        [sys.executable, "tools/gauntlet_v2/build_pnl_index.py",
         "--oci-dir", str(tmp_path / "bad_run"),
         "--output", str(out)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 1
    assert "duplicate" in result.stderr.lower() or "duplicate" in result.stdout.lower()
