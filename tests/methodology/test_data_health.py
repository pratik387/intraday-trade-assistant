"""Tests for tools.methodology.data_health.

Verifies the 3 layers detect the kinds of bugs that previously caused
silent data drift (the earnings_day case).
"""
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.methodology.data_health import (
    check_trade_count_anomaly,
    check_source_provenance,
    check_trade_quality_drift,
    check_all,
    HealthIssue,
    HealthReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trades_uniform(start: date, end: date, per_quarter: int) -> pd.DataFrame:
    """Build trades evenly distributed across calendar quarters."""
    quarters = []
    d = start
    while d <= end:
        if d.weekday() < 5:  # weekday
            quarters.append({"signal_date": d.isoformat(), "exit_reason": "sl", "same_bar": False})
        d += timedelta(days=1)
    df = pd.DataFrame(quarters)
    return df


def _trades_with_quarter_distribution(distribution: dict) -> pd.DataFrame:
    """Build trades matching a target {quarter_label: count} distribution.

    Quarter labels like '2023-01' (Q1), '2023-04' (Q2), etc.
    """
    rows = []
    for q_label, count in distribution.items():
        year, month = q_label.split("-")
        y, m = int(year), int(month)
        # Distribute count across the 3 months of this quarter
        for i in range(count):
            day_offset = i % 28 + 1
            month_offset = (i // 28) % 3
            d = date(y, m + month_offset, day_offset)
            rows.append({
                "signal_date": d.isoformat(),
                "exit_reason": "sl",
                "same_bar": False,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Layer 1: trade-count anomaly
# ---------------------------------------------------------------------------

def test_layer1_passes_when_uniform_distribution():
    """Uniform trade distribution across 4+ quarters → no anomalies."""
    df = _trades_with_quarter_distribution({
        "2023-01": 200, "2023-04": 200, "2023-07": 200, "2023-10": 200,
        "2024-01": 200, "2024-04": 200,
    })
    issues = check_trade_count_anomaly(df, "test_setup")
    assert len(issues) == 0


def test_layer1_blocks_catastrophic_drop():
    """A 50%+ drop AND >3sigma below mean blocks (the earnings-day pattern).

    Stable baseline (12 quarters at 200) with sudden collapse to 23 is the
    announcements_fr scenario: long low-variance baseline + huge sudden drop.
    """
    distribution = {
        "2023-01": 200, "2023-04": 200, "2023-07": 200, "2023-10": 200,
        "2024-01": 200, "2024-04": 200, "2024-07": 200, "2024-10": 200,
        "2025-01": 200, "2025-04": 200, "2025-07": 200, "2025-10": 200,
        "2026-01": 23,    # CATASTROPHIC after 12 quarters of stable baseline
    }
    df = _trades_with_quarter_distribution(distribution)
    issues = check_trade_count_anomaly(df, "test_setup")
    block_issues = [i for i in issues if i.severity == "block"]
    assert len(block_issues) >= 1, f"expected block, got: {[i.code for i in issues]}"
    assert any(i.window_label == "2026-01" for i in block_issues)
    assert all(i.code == "trade_count.catastrophic_drop" for i in block_issues)


def test_layer1_does_not_block_event_driven_setup_natural_variability():
    """Event-driven setups (gap-down LONGs etc.) have high natural variance
    in trade count per quarter. Layer 1 must NOT flag normal regime-driven
    variability as catastrophic.

    Test: mean=80, sigma=49 with windows ranging 20-200. None should block
    even though some are >50% below mean — because they're within 3sigma."""
    df = _trades_with_quarter_distribution({
        "2023-01": 80, "2023-04": 29, "2023-07": 36, "2023-10": 130,
        "2024-01": 80, "2024-04": 50, "2024-07": 100, "2024-10": 75,
        "2025-01": 90, "2025-04": 197, "2025-07": 34, "2025-10": 20,
        "2026-01": 122,
    })
    issues = check_trade_count_anomaly(df, "test_setup")
    block_issues = [i for i in issues if i.severity == "block"]
    # No blocks — natural event-driven variance, even though several windows
    # are >50% below mean, they're all within ~3sigma
    assert len(block_issues) == 0, (
        f"Unexpected blocks on event-driven setup: "
        f"{[(i.window_label, i.metric_value, i.metric_baseline) for i in block_issues]}"
    )


def test_layer1_warns_on_outlier_but_not_block():
    """1-3σ outlier WARNS but does not block."""
    df = _trades_with_quarter_distribution({
        "2023-01": 200, "2023-04": 200, "2023-07": 200, "2023-10": 200,
        "2024-01": 200, "2024-04": 200, "2024-07": 200, "2024-10": 200,
        "2025-04": 350,  # 75% above mean — outlier but not catastrophic
    })
    issues = check_trade_count_anomaly(df, "test_setup", warn_sigma=2.0)
    warns = [i for i in issues if i.severity == "warn"]
    assert len(warns) >= 1
    assert all(i.code == "trade_count.outlier" for i in warns)
    blocks = [i for i in issues if i.severity == "block"]
    assert len(blocks) == 0


def test_layer1_skips_when_insufficient_data():
    """Below 100-row baseline → skip (no false positives on tiny samples)."""
    df = _trades_with_quarter_distribution({"2023-01": 10})
    issues = check_trade_count_anomaly(df, "test_setup", min_baseline_rows=100)
    assert len(issues) == 0


# ---------------------------------------------------------------------------
# Layer 2: source-provenance audit
# ---------------------------------------------------------------------------

def test_layer2_blocks_source_dropping_to_zero():
    """A source that goes from >100 rows → 0 rows is BLOCKED.

    This is THE earnings_day case: announcements_fr went from ~700/month
    to 0/month in April 2025.
    """
    rows = []
    # source A: 200 rows per month Jan 2024 - Sep 2024 (historical presence)
    for ym in ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05"]:
        y, m = ym.split("-")
        for i in range(200):
            rows.append({"announce_date": f"{y}-{m}-{(i % 28) + 1:02d}", "source": "source_A"})
    # source A: drops to 0 in 2024-06 (this is the catastrophe)
    for ym in ["2024-06", "2024-07"]:
        y, m = ym.split("-")
        for i in range(200):
            rows.append({"announce_date": f"{y}-{m}-{(i % 28) + 1:02d}", "source": "source_B"})

    df = pd.DataFrame(rows)
    issues = check_source_provenance(
        df, setup_name="test", date_column="announce_date", source_column="source",
        historical_threshold=100,
    )
    block_issues = [i for i in issues if i.severity == "block"]
    assert len(block_issues) >= 1
    assert all(i.code == "source_audit.source_dropped_to_zero" for i in block_issues)
    # The blocked windows should be 2024-06 and 2024-07 (source_A is gone)
    blocked_months = {i.window_label for i in block_issues}
    assert "2024-06" in blocked_months
    assert "2024-07" in blocked_months


def test_layer2_passes_when_no_zero_drops():
    """Sources present consistently across months → no anomalies."""
    rows = []
    for ym in ["2024-01", "2024-02", "2024-03"]:
        y, m = ym.split("-")
        for i in range(150):
            rows.append({"announce_date": f"{y}-{m}-{(i % 28) + 1:02d}", "source": "source_A"})

    df = pd.DataFrame(rows)
    issues = check_source_provenance(
        df, setup_name="test", date_column="announce_date", source_column="source",
        historical_threshold=100,
    )
    assert len(issues) == 0


def test_layer2_ignores_source_never_having_historical_presence():
    """A source that was only ever sporadic (<100 rows in any month) doesn't get flagged."""
    rows = []
    # Source A: always sporadic, never crosses threshold
    for ym in ["2024-01", "2024-02", "2024-03"]:
        y, m = ym.split("-")
        for i in range(10):  # only 10 rows/month
            rows.append({"announce_date": f"{y}-{m}-{(i % 28) + 1:02d}", "source": "source_A"})
    # Source B: zero in 2024-02 but never had threshold either
    rows.append({"announce_date": "2024-01-15", "source": "source_B"})

    df = pd.DataFrame(rows)
    issues = check_source_provenance(
        df, setup_name="test", date_column="announce_date", source_column="source",
        historical_threshold=100,
    )
    # Neither source ever hit threshold — no blocking
    block_issues = [i for i in issues if i.severity == "block"]
    assert len(block_issues) == 0


def test_layer2_warns_on_missing_columns():
    """No source column → warn (skip), not error."""
    df = pd.DataFrame({"announce_date": ["2024-01-15"]})
    issues = check_source_provenance(
        df, setup_name="test", date_column="announce_date",
        source_column="source",
    )
    assert len(issues) == 1
    assert issues[0].severity == "warn"
    assert "missing_source_column" in issues[0].code


# ---------------------------------------------------------------------------
# Layer 3: trade-quality drift
# ---------------------------------------------------------------------------

def test_layer3_passes_when_quality_metrics_stable():
    """Uniform exit_reason + same_bar rates across windows → no anomalies."""
    rows = []
    for ym in ["2023-01", "2023-04", "2023-07", "2023-10",
               "2024-01", "2024-04", "2024-07", "2024-10"]:
        y, m = ym.split("-")
        for i in range(100):
            rows.append({
                "signal_date": f"{y}-{m}-{(i % 28) + 1:02d}",
                "exit_reason": "sl" if i < 70 else "t2",
                "same_bar": (i % 10 == 0),  # 10% same_bar consistently
            })
    df = pd.DataFrame(rows)
    issues = check_trade_quality_drift(df, "test_setup")
    # May get small warnings depending on noise, but no major ones
    assert all(i.severity == "warn" for i in issues)


def test_layer3_warns_on_same_bar_rate_outlier():
    """A window with dramatically higher same_bar rate → WARN."""
    rows = []
    # Stable windows
    for ym in ["2023-01", "2023-04", "2023-07", "2023-10",
               "2024-01", "2024-04", "2024-07", "2024-10"]:
        y, m = ym.split("-")
        for i in range(100):
            rows.append({
                "signal_date": f"{y}-{m}-{(i % 28) + 1:02d}",
                "exit_reason": "sl",
                "same_bar": (i % 20 == 0),  # 5% same_bar
            })
    # Outlier window: 80% same_bar
    for i in range(100):
        rows.append({
            "signal_date": f"2025-01-{(i % 28) + 1:02d}",
            "exit_reason": "sl",
            "same_bar": (i < 80),  # 80% same_bar
        })
    df = pd.DataFrame(rows)
    issues = check_trade_quality_drift(df, "test_setup", warn_sigma=2.0)
    sb_issues = [i for i in issues if "same_bar_rate" in i.code]
    assert len(sb_issues) >= 1
    assert all(i.severity == "warn" for i in sb_issues)


def test_layer3_warns_on_missing_columns():
    """No exit_reason/same_bar → warn (skip)."""
    df = pd.DataFrame({"signal_date": ["2024-01-15"]})
    issues = check_trade_quality_drift(df, "test_setup")
    assert len(issues) == 1
    assert "missing_columns" in issues[0].code


# ---------------------------------------------------------------------------
# Combined check_all
# ---------------------------------------------------------------------------

def test_check_all_combines_all_layers():
    """check_all runs Layer 1 + Layer 3 (and Layer 2 if source data provided)."""
    distribution = {
        "2023-01": 200, "2023-04": 200, "2023-07": 200, "2023-10": 200,
        "2024-01": 200, "2024-04": 200, "2024-07": 200, "2024-10": 200,
        "2025-01": 200, "2025-04": 200, "2025-07": 200, "2025-10": 200,
        "2026-01": 23,  # Layer 1 should block
    }
    df = _trades_with_quarter_distribution(distribution)
    report = check_all(df, "test_setup")
    assert isinstance(report, HealthReport)
    assert report.has_blocking_issues, f"expected block; got issues: {[i.code for i in report.issues]}"
    layer1_blocks = [i for i in report.issues if i.layer == 1 and i.severity == "block"]
    assert len(layer1_blocks) >= 1


def test_check_all_writes_audit_log(tmp_path):
    """When audit_log_dir is provided, JSONL audit log is written."""
    distribution = {
        "2023-01": 200, "2023-04": 200, "2023-07": 200, "2023-10": 200,
        "2024-01": 200, "2024-04": 200, "2024-07": 200, "2024-10": 200,
        "2025-01": 200, "2025-04": 200, "2025-07": 200, "2025-10": 200,
        "2026-01": 23,
    }
    df = _trades_with_quarter_distribution(distribution)
    report = check_all(df, "test_setup", audit_log_dir=tmp_path)
    files = list(tmp_path.glob("test_setup_*.jsonl"))
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8")
    assert "trade_count.catastrophic_drop" in content


def test_check_all_clean_data_no_blocking():
    """Clean uniform trades → no blocking issues."""
    df = _trades_with_quarter_distribution({
        "2023-01": 200, "2023-04": 200, "2023-07": 200, "2023-10": 200,
        "2024-01": 200, "2024-04": 200, "2024-07": 200, "2024-10": 200,
    })
    df["exit_reason"] = "sl"
    df["same_bar"] = False
    report = check_all(df, "test_setup")
    assert not report.has_blocking_issues


def test_health_report_summary_format():
    """HealthReport.summary() includes blocking issues + warnings."""
    issues = [
        HealthIssue(severity="block", layer=1, code="trade_count.catastrophic_drop",
                    message="bad window", window_label="2025-04"),
        HealthIssue(severity="warn", layer=3, code="quality_drift.same_bar_rate_outlier",
                    message="elevated same_bar"),
    ]
    report = HealthReport(setup_name="test", issues=issues)
    summary = report.summary()
    assert "1 blocking" in summary
    assert "1 warnings" in summary
    assert "trade_count.catastrophic_drop" in summary
    assert "quality_drift.same_bar_rate_outlier" in summary
