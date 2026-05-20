"""Silent-drift detector for trade CSVs + their underlying data sources.

Three layers of defense, designed to catch the kind of bugs that destroyed
the earnings_day analysis (announcements_fr source going silently to zero
without any alert):

  Layer 1 — Trade-count anomaly (per-setup, per-period):
    For each non-overlapping window, compute trade count. Flag windows
    where the count is > N sigma from the in-sample mean OR > X% drop
    from the mean. The earnings_day April 2025 collapse (23 events vs
    ~250 typical) would be flagged here.

  Layer 2 — Source-provenance audit (per-source, per-period):
    For data files with a `source` column (e.g., earnings_events.parquet
    with source ∈ {announcements_fr, announcements_bmo, financial_results,
    board_meetings}): per (source, month), check that no source drops
    to zero rows when it historically had >100 rows. The announcements_fr
    case would be flagged here.

  Layer 3 — Trade-quality drift (per-setup, per-period):
    Exit-reason distribution, same_bar rate, T2-hit rate across windows.
    Flag windows where any metric is > N sigma from the mean. This catches
    scenarios where the *same* data is captured but the *trade behavior*
    shifts — could be regime change OR data quality.

Severity escalation:
  - BLOCK: walk-forward CLI refuses to run (Layer 1 trade-count >50%
    drop from in-sample mean; Layer 2 source going from >100 to 0 rows)
  - WARN: walk-forward CLI prints warning but proceeds (Layer 1/3 anomalies
    within 1-3 sigma; researcher decides)

Audit trail: per-run JSONL log at
  reports/data_health/<setup>_<YYYY-MM-DD>.jsonl
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Issue + Report
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HealthIssue:
    severity: str          # "block" | "warn"
    layer: int             # 1 | 2 | 3
    code: str              # stable identifier
    message: str
    metric_value: Optional[float] = None
    metric_baseline: Optional[float] = None
    window_label: Optional[str] = None    # e.g., "2025-04" or "Window 8 (2025-01..2025-03)"


@dataclass(frozen=True)
class HealthReport:
    setup_name: str
    issues: List[HealthIssue]

    @property
    def has_blocking_issues(self) -> bool:
        return any(i.severity == "block" for i in self.issues)

    @property
    def n_block(self) -> int:
        return sum(1 for i in self.issues if i.severity == "block")

    @property
    def n_warn(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warn")

    def summary(self) -> str:
        if not self.issues:
            return f"DATA HEALTH OK: {self.setup_name} — no anomalies"
        lines = [f"DATA HEALTH REPORT: {self.setup_name} — "
                 f"{self.n_block} blocking, {self.n_warn} warnings"]
        for issue in self.issues:
            tag = f"[{issue.severity.upper()}/L{issue.layer}/{issue.code}]"
            wnd = f" [{issue.window_label}]" if issue.window_label else ""
            metric_info = ""
            if issue.metric_value is not None and issue.metric_baseline is not None:
                metric_info = (
                    f" (value={issue.metric_value:.3g}, "
                    f"baseline={issue.metric_baseline:.3g})"
                )
            lines.append(f"  {tag}{wnd} {issue.message}{metric_info}")
        return "\n".join(lines)

    def to_jsonl(self) -> str:
        """One JSON object per issue. Suitable for append-only audit log."""
        out = []
        for issue in self.issues:
            d = asdict(issue)
            d["setup_name"] = self.setup_name
            d["checked_at"] = datetime.now().isoformat(timespec="seconds")
            out.append(json.dumps(d, ensure_ascii=False))
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Layer 1: Trade-count anomaly per period
# ---------------------------------------------------------------------------

def _bucket_by_quarter(dates: pd.Series) -> pd.Series:
    """Bucket dates into 3-month windows: 2023-Q1, 2023-Q2, etc.

    Uses calendar quarters anchored at Jan 1. Returns string labels like
    '2023-01' (start of quarter month).
    """
    ds = pd.to_datetime(dates)
    # Quarter start month: Jan=1, Apr=4, Jul=7, Oct=10
    q_month = ((ds.dt.month - 1) // 3) * 3 + 1
    return ds.dt.year.astype(str) + "-" + q_month.astype(str).str.zfill(2)


def check_trade_count_anomaly(
    trades_df: pd.DataFrame,
    setup_name: str,
    *,
    block_drop_pct: float = 0.50,
    warn_sigma: float = 2.0,
    min_baseline_rows: int = 100,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> List[HealthIssue]:
    """Layer 1: Flag periods with anomalous trade count.

    Args:
        trades_df: must have `signal_date` column (any date format pandas can parse)
        setup_name: for issue messages
        block_drop_pct: BLOCK if a window's count is <= (1-block_drop_pct) * mean.
                        Default 0.50 = 50% drop blocks.
        warn_sigma: WARN if a window's count is > warn_sigma * std from mean
        min_baseline_rows: skip checks if total trades < min_baseline_rows
        start_date / end_date: if provided, restrict analysis to trades in
            [start, end]. Filters out partial leading/trailing quarters that
            would generate false-positive anomalies.

    Returns:
        List of HealthIssue. Empty if no anomalies.
    """
    issues: List[HealthIssue] = []
    if "signal_date" not in trades_df.columns:
        issues.append(HealthIssue(
            severity="block", layer=1, code="trade_count.missing_signal_date",
            message="trades_df has no signal_date column — cannot bucket by period",
        ))
        return issues

    df = trades_df.copy()
    sd = pd.to_datetime(df["signal_date"]).dt.date
    df["_d"] = sd
    if start_date is not None:
        df = df[df["_d"] >= start_date]
    if end_date is not None:
        df = df[df["_d"] <= end_date]

    if len(df) < min_baseline_rows:
        return issues  # not enough data after filtering
    df["_q"] = _bucket_by_quarter(df["_d"])
    counts = df.groupby("_q").size().sort_index()
    if len(counts) < 4:
        return issues  # not enough windows to compute baseline

    mean = float(counts.mean())
    std = float(counts.std(ddof=0))

    for window_label, count in counts.items():
        # Block: anomalous drop below threshold
        if count <= mean * (1 - block_drop_pct):
            issues.append(HealthIssue(
                severity="block", layer=1, code="trade_count.catastrophic_drop",
                message=(
                    f"Window {window_label} has {int(count)} trades — "
                    f">{int(block_drop_pct*100)}% below mean ({mean:.0f}). "
                    f"Likely silent data issue. Investigate before running walk-forward."
                ),
                metric_value=float(count),
                metric_baseline=mean,
                window_label=window_label,
            ))
            continue  # don't also emit warn for the same window

        # Warn: anomalous in either direction (drift, not catastrophe)
        if std > 0 and abs(count - mean) > warn_sigma * std:
            direction = "high" if count > mean else "low"
            issues.append(HealthIssue(
                severity="warn", layer=1, code="trade_count.outlier",
                message=(
                    f"Window {window_label} has {int(count)} trades — "
                    f"{warn_sigma:.1f}sigma {direction} of mean ({mean:.0f}, sigma={std:.0f})"
                ),
                metric_value=float(count),
                metric_baseline=mean,
                window_label=window_label,
            ))

    return issues


# ---------------------------------------------------------------------------
# Layer 2: Source-provenance audit
# ---------------------------------------------------------------------------

def check_source_provenance(
    data_df: pd.DataFrame,
    *,
    setup_name: str,
    date_column: str,
    source_column: str = "source",
    historical_threshold: int = 100,
) -> List[HealthIssue]:
    """Layer 2: Flag (source, month) pairs where the source dropped to zero
    after historically having >`historical_threshold` rows.

    This is THE check that would have caught earnings_day's announcements_fr
    source going silently to zero in April 2025+.

    Args:
        data_df: must have `date_column` and `source_column`
        setup_name: for issue messages
        date_column: column name containing dates (e.g., 'announce_date')
        source_column: column name containing source labels
        historical_threshold: minimum row count for a source-month to be considered
                              "historically present" (default 100)

    Returns:
        List of HealthIssue. Empty if no anomalies.
    """
    issues: List[HealthIssue] = []

    if date_column not in data_df.columns:
        issues.append(HealthIssue(
            severity="warn", layer=2, code="source_audit.missing_date_column",
            message=f"data_df has no {date_column!r} column — cannot audit by month",
        ))
        return issues
    if source_column not in data_df.columns:
        issues.append(HealthIssue(
            severity="warn", layer=2, code="source_audit.missing_source_column",
            message=f"data_df has no {source_column!r} column — skip source-provenance audit",
        ))
        return issues

    df = data_df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df["_ym"] = df[date_column].dt.strftime("%Y-%m")

    sources = sorted(df[source_column].dropna().astype(str).unique().tolist())
    if not sources:
        return issues  # nothing to audit

    # For each source: compute per-month row count. Flag months where this
    # source went from >=threshold to 0 after appearing in the data.
    months_in_data = sorted(df["_ym"].unique().tolist())
    for src in sources:
        per_month = df[df[source_column] == src].groupby("_ym").size()
        # First month this source appeared with >= threshold rows
        big_months = per_month[per_month >= historical_threshold]
        if big_months.empty:
            continue  # source never had a substantial presence
        first_big = big_months.index.min()
        # Look at all months from first_big onward; flag any that are zero
        for ym in months_in_data:
            if ym < first_big:
                continue
            count = int(per_month.get(ym, 0))
            if count == 0:
                issues.append(HealthIssue(
                    severity="block", layer=2, code="source_audit.source_dropped_to_zero",
                    message=(
                        f"Source {src!r} dropped to 0 rows in {ym} after historically "
                        f"having >={historical_threshold} rows. "
                        f"This is exactly the announcements_fr-style silent data drift. "
                        f"Investigate the scraper for that endpoint."
                    ),
                    metric_value=0.0,
                    metric_baseline=float(big_months.mean()),
                    window_label=ym,
                ))

    return issues


# ---------------------------------------------------------------------------
# Layer 3: Trade-quality drift
# ---------------------------------------------------------------------------

def check_trade_quality_drift(
    trades_df: pd.DataFrame,
    setup_name: str,
    *,
    warn_sigma: float = 3.0,
    min_window_n: int = 30,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> List[HealthIssue]:
    """Layer 3: Flag periods with anomalous exit_reason distribution / same_bar %.

    Could be regime change OR data quality issue. Always WARN (never block) —
    distinguishing the two requires human judgment.

    Args:
        trades_df: must have `signal_date`, `exit_reason`, `same_bar` columns
        setup_name: for issue messages
        warn_sigma: threshold for outlier detection
        min_window_n: skip windows with fewer trades than this
        start_date / end_date: optional filter (as in check_trade_count_anomaly)
    """
    issues: List[HealthIssue] = []
    required = {"signal_date", "exit_reason", "same_bar"}
    missing = required - set(trades_df.columns)
    if missing:
        issues.append(HealthIssue(
            severity="warn", layer=3, code="quality_drift.missing_columns",
            message=f"Skipping Layer 3 — missing columns: {sorted(missing)}",
        ))
        return issues

    df = trades_df.copy()
    sd = pd.to_datetime(df["signal_date"]).dt.date
    df["_d"] = sd
    if start_date is not None:
        df = df[df["_d"] >= start_date]
    if end_date is not None:
        df = df[df["_d"] <= end_date]
    df["_q"] = _bucket_by_quarter(df["_d"])

    # Per-window: same_bar rate
    per_window_sb = df.groupby("_q").agg(
        n=("same_bar", "size"),
        same_bar_rate=("same_bar", "mean"),
    )
    per_window_sb = per_window_sb[per_window_sb["n"] >= min_window_n]
    if len(per_window_sb) < 4:
        return issues  # not enough windows

    sb_mean = per_window_sb["same_bar_rate"].mean()
    sb_std = per_window_sb["same_bar_rate"].std(ddof=0)
    if sb_std > 0:
        for window_label, row in per_window_sb.iterrows():
            if abs(row["same_bar_rate"] - sb_mean) > warn_sigma * sb_std:
                direction = "high" if row["same_bar_rate"] > sb_mean else "low"
                issues.append(HealthIssue(
                    severity="warn", layer=3, code="quality_drift.same_bar_rate_outlier",
                    message=(
                        f"Window {window_label} same_bar rate is {row['same_bar_rate']:.1%} — "
                        f"{warn_sigma:.1f}sigma {direction} of mean ({sb_mean:.1%}, sigma={sb_std:.1%}). "
                        f"Possible regime change OR sanity-script behavior drift."
                    ),
                    metric_value=float(row["same_bar_rate"]),
                    metric_baseline=float(sb_mean),
                    window_label=window_label,
                ))

    # Per-window: exit_reason distribution — check for major categories
    # We focus on SL vs T2 vs time_stop rates as the key health signals.
    for reason in ("sl", "t2", "time_stop"):
        rate_col = f"{reason}_rate"
        df[rate_col] = (df["exit_reason"] == reason).astype(int)
        per_w = df.groupby("_q").agg(n=(rate_col, "size"), rate=(rate_col, "mean"))
        per_w = per_w[per_w["n"] >= min_window_n]
        if len(per_w) < 4:
            continue
        m, s = per_w["rate"].mean(), per_w["rate"].std(ddof=0)
        if s == 0:
            continue
        for window_label, row in per_w.iterrows():
            if abs(row["rate"] - m) > warn_sigma * s:
                direction = "high" if row["rate"] > m else "low"
                issues.append(HealthIssue(
                    severity="warn", layer=3,
                    code=f"quality_drift.exit_reason_{reason}_outlier",
                    message=(
                        f"Window {window_label} {reason}-hit rate is {row['rate']:.1%} — "
                        f"{warn_sigma:.1f}sigma {direction} of mean ({m:.1%}, sigma={s:.1%}). "
                        f"Possible regime change."
                    ),
                    metric_value=float(row["rate"]),
                    metric_baseline=float(m),
                    window_label=window_label,
                ))

    return issues


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def check_all(
    trades_df: pd.DataFrame,
    setup_name: str,
    *,
    source_data_df: Optional[pd.DataFrame] = None,
    source_date_column: Optional[str] = None,
    source_column: str = "source",
    audit_log_dir: Optional[Path] = None,
    block_drop_pct: float = 0.50,
    warn_sigma_layer1: float = 2.0,
    warn_sigma_layer3: float = 3.0,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> HealthReport:
    """Run all 3 layers and return combined report.

    Layer 2 (source-provenance audit) is only run if `source_data_df` is
    provided AND it has the `source_column`.

    `start_date` and `end_date` (if provided) filter trades_df to that
    range before Layer 1 + Layer 3 analysis. Use these to align data_health
    bucketing with walk-forward's window range, preventing false-positive
    anomalies on partial leading/trailing quarters.

    If `audit_log_dir` is provided, writes a JSONL audit trail at
    `<audit_log_dir>/<setup_name>_<YYYY-MM-DD>.jsonl`.
    """
    issues = []

    # Layer 1
    issues.extend(check_trade_count_anomaly(
        trades_df, setup_name,
        block_drop_pct=block_drop_pct,
        warn_sigma=warn_sigma_layer1,
        start_date=start_date,
        end_date=end_date,
    ))

    # Layer 2 (optional — only if source data provided)
    if source_data_df is not None and source_date_column is not None:
        issues.extend(check_source_provenance(
            source_data_df,
            setup_name=setup_name,
            date_column=source_date_column,
            source_column=source_column,
        ))

    # Layer 3
    issues.extend(check_trade_quality_drift(
        trades_df, setup_name,
        warn_sigma=warn_sigma_layer3,
        start_date=start_date,
        end_date=end_date,
    ))

    report = HealthReport(setup_name=setup_name, issues=issues)

    # Persist audit log if requested
    if audit_log_dir is not None and issues:
        audit_log_dir = Path(audit_log_dir)
        audit_log_dir.mkdir(parents=True, exist_ok=True)
        out_path = audit_log_dir / f"{setup_name}_{date.today().isoformat()}.jsonl"
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(report.to_jsonl())
            f.write("\n")

    return report
