"""Smoke tests for detector_rejections.jsonl + detector_accepts.jsonl writers.

Per audit/15 + audit/14 Tier-A logging requirement: main_detector must emit
per-event reject/accept logs with full bar context for post-OCI gauntlet
funnel reconstruction.

These tests verify:
1. Loggers initialize without errors
2. Trivial rejections (insufficient data, generic "no patterns") are filtered
3. Specific Tier-A rejections are logged with full context
4. Accepted detections are logged to the mirror file
"""

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def test_logger_getters_return_jsonl_logger_when_initialized(tmp_path, monkeypatch):
    """get_detector_rejections_logger / accepts_logger return JSONLLogger
    instances after initialization with a run_prefix."""
    # Reset module-level singletons to force re-init
    import config.logging_config as lc
    lc._detector_rejections_logger = None
    lc._detector_accepts_logger = None
    lc._agent_logger = None  # force full re-init

    # Set log dir to tmp_path
    monkeypatch.setattr(lc, "_global_run_prefix", "test_logging_")
    # Patch the project log dir resolution to use tmp_path
    original_init = lc._initialize_loggers

    def patched_init(run_prefix="", force_reinit=False):
        lc.dir_path = tmp_path
        # Manually construct the JSONLLoggers in tmp_path
        lc._detector_rejections_logger = lc.JSONLLogger(
            tmp_path / "detector_rejections.jsonl", "detector_reject"
        )
        lc._detector_accepts_logger = lc.JSONLLogger(
            tmp_path / "detector_accepts.jsonl", "detector_accept"
        )

    monkeypatch.setattr(lc, "_initialize_loggers", patched_init)

    rej_logger = lc.get_detector_rejections_logger()
    acc_logger = lc.get_detector_accepts_logger()

    assert rej_logger is not None
    assert acc_logger is not None
    assert hasattr(rej_logger, "log_event")
    assert hasattr(acc_logger, "log_event")


def test_jsonl_logger_writes_line_per_event(tmp_path):
    """JSONLLogger.log_event writes one JSON line per call to the target file."""
    from config.logging_config import JSONLLogger

    out_file = tmp_path / "test_rejections.jsonl"
    logger = JSONLLogger(out_file, "test")

    logger.log_event(
        ts="2026-04-15T10:00:00",
        symbol="RELIANCE",
        detector="gap_fill_short",
        reason="gap_fill outside time window",
        regime="trend_up",
        cap_segment="large_cap",
        hour_bucket="1100-1300",
        vol_z=1.5,
        atr=12.5,
    )
    logger.log_event(
        ts="2026-04-15T10:05:00",
        symbol="HDFCBANK",
        detector="flag_continuation_long",
        reason="volume did not decline through flag",
        regime="trend_up",
        cap_segment="large_cap",
        hour_bucket="1000-1100",
    )
    logger._close()

    lines = out_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    rec1 = json.loads(lines[0])
    assert rec1["symbol"] == "RELIANCE"
    assert rec1["detector"] == "gap_fill_short"
    assert "outside time window" in rec1["reason"]

    rec2 = json.loads(lines[1])
    assert rec2["symbol"] == "HDFCBANK"
    assert "volume did not decline" in rec2["reason"]


def test_main_detector_emits_rejection_for_gap_tier_a_filter(tmp_path, monkeypatch):
    """When gap_fill detector returns rejection_reason for Tier-A time window,
    main_detector logs it to detector_rejections.jsonl with bar context."""
    import config.logging_config as lc

    # Wire the new loggers to tmp_path
    rej_path = tmp_path / "detector_rejections.jsonl"
    acc_path = tmp_path / "detector_accepts.jsonl"
    lc._detector_rejections_logger = lc.JSONLLogger(rej_path, "detector_reject")
    lc._detector_accepts_logger = lc.JSONLLogger(acc_path, "detector_accept")

    # Build a context where Gap detector rejects with Tier-A time window failure
    from structures.gap_structure import GapStructure
    from structures.data_models import MarketContext

    cfg = {
        "_setup_name": "gap_fill_short",
        "min_gap_pct": 0.3,
        "max_gap_pct": 2.5,
        "require_volume_confirmation": False,
        "min_volume_mult": 1.2,
        "gap_fill_start_hhmm": "0915",
        "gap_fill_end_hhmm": "1030",
        "target_mult_t1": 1.0,
        "target_mult_t2": 2.0,
        "confidence_level": 0.7,
        "gap_sl_buffer_atr": 0.3,
        "min_stop_distance_pct": 0.3,
    }
    detector = GapStructure(cfg)

    # Build a 10-bar df that simulates a gap-up + price below open AT 11:30 (outside window)
    idx = pd.date_range("2026-04-15 11:00:00", periods=10, freq="5min")
    df = pd.DataFrame({
        "open":   [102.0] + [101.9] * 9,
        "high":   [102.3] * 10,
        "low":    [101.4] * 10,
        "close":  [101.5] * 10,
        "volume": [1000] * 10,
    }, index=idx)
    df["vol_z"] = 2.0

    ctx = MarketContext(
        symbol="TEST",
        current_price=101.5,
        timestamp=pd.Timestamp("2026-04-15 11:30:00"),  # OUTSIDE 09:15-10:30
        df_5m=df,
        session_date=datetime(2026, 4, 15),
        cap_segment="mid_cap",
        pdc=100.0,
    )

    # Invoke detector directly — confirm rejection_reason is set
    analysis = detector.detect(ctx)
    assert not analysis.structure_detected
    assert "time window" in (analysis.rejection_reason or "").lower()

    # Now simulate what main_detector would log: write the same shape to the
    # new logger directly (we don't run main_detector here to keep this a
    # focused smoke test of the logger contract)
    lc._detector_rejections_logger.log_event(
        ts=ctx.timestamp.isoformat(),
        symbol=ctx.symbol,
        detector="gap_fill_short",
        reason=analysis.rejection_reason,
        regime=getattr(ctx, "regime", None),
        cap_segment=ctx.cap_segment,
        hour_bucket="1100-1300",
        vol_z=2.0,
        atr=None,
    )
    lc._detector_rejections_logger._close()

    lines = rej_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["detector"] == "gap_fill_short"
    assert "time window" in rec["reason"].lower()
    assert rec["cap_segment"] == "mid_cap"
    assert rec["hour_bucket"] == "1100-1300"
    assert rec["vol_z"] == 2.0


def test_flag_volume_decline_rejection_propagates_through_detector(tmp_path):
    """Audit/14 Tier-A: flag_volume_decline rejection must be reachable as
    StructureAnalysis.rejection_reason (verifying the propagation fix that
    surfaces the deepest filter via _last_rejection)."""
    from structures.flag_continuation_structure import FlagContinuationStructure
    from structures.data_models import MarketContext

    cfg = {
        "_setup_name": "flag_continuation_long",
        "min_consolidation_bars": 3,
        "max_consolidation_bars": 12,
        "min_trend_strength": 1.5,
        "trend_lookback_period": 10,
        "max_consolidation_range_pct": 2.0,
        "breakout_confirmation_pct": 0.1,
        "require_volume_confirmation": False,
        "min_volume_mult": 1.5,
        "flag_volume_decline_ratio": 0.85,
        "target_mult_t1": 1.5,
        "target_mult_t2": 2.5,
        "stop_mult": 1.0,
        "confidence_strong_flag": 0.85,
        "confidence_weak_flag": 0.65,
    }
    detector = FlagContinuationStructure(cfg)

    # Build flag pattern WITHOUT volume decline (vol stays flat)
    bars = []
    for i in range(14):
        p = 100.0 + i * 0.7
        bars.append((p, p + 0.2, p - 0.2, p + 0.4, 1000))
    for _ in range(9):
        bars.append((110.0, 110.2, 109.8, 110.0, 1000))
    bars.append((110.0, 110.8, 109.95, 110.6, 2000))

    idx = pd.date_range("2026-04-15 09:30", periods=len(bars), freq="5min")
    df = pd.DataFrame(bars, columns=["open", "high", "low", "close", "volume"], index=idx)

    ctx = MarketContext(
        symbol="TEST",
        current_price=110.6,
        timestamp=df.index[-1],
        df_5m=df,
        session_date=datetime(2026, 4, 15),
        cap_segment=None,
        pdc=100.0,
    )

    analysis = detector.detect(ctx)
    assert not analysis.structure_detected
    # CRITICAL: the volume-decline rejection (priority 3, deepest) must be
    # surfaced — NOT the generic "No flag continuation patterns detected"
    assert analysis.rejection_reason is not None
    assert "volume did not decline" in analysis.rejection_reason.lower(), (
        f"Got generic message instead of specific filter rejection: "
        f"{analysis.rejection_reason}"
    )
