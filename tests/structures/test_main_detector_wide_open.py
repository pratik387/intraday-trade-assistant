"""MainDetector._resolve_conflicts_and_prioritize wide_open bypass test.

Regression test for the bug discovered in the 2026-04-29 OCI Discovery run
where camarilla_l3_reversal (5,498 fires), pdh_pdl_sweep_reclaim (2,166), and
ema5_alert_pullback (3) emitted detector_accepts but ZERO of them reached
gate_input. Root cause: `_resolve_directional_conflicts` groups events by
0.5% price proximity and keeps only the highest-confidence event per group,
silently dropping newer detectors with conservative confidence values
(camarilla median confidence 0.138 vs pdh_pdl_reject's 0.662).

Under wide_open_mode the gauntlet must see every detector's signal so the
fix bypasses the dedup AND the max_detections_per_symbol cap entirely.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from structures.data_models import StructureAnalysis, StructureEvent


def _build_event(side: str, confidence: float, price: float,
                 structure_type: str = "test") -> StructureEvent:
    return StructureEvent(
        symbol="NSE:TEST",
        timestamp=pd.Timestamp("2024-06-06 14:00:00"),
        structure_type=structure_type,
        side=side,
        confidence=confidence,
        levels={"entry": price},
        context={},
        price=price,
    )


def _make_detector(wide_open: bool, max_detections: int = 5,
                   conflict_resolution_enabled: bool = True):
    """Construct a MainDetector skipping the heavy __init__ — we only need the
    instance attributes used by `_resolve_conflicts_and_prioritize`."""
    from structures.main_detector import MainDetector
    det = MainDetector.__new__(MainDetector)
    det._wide_open = wide_open
    det.max_detections_per_symbol = max_detections
    det.conflict_resolution_enabled = conflict_resolution_enabled
    det.priority_weights = {}
    return det


def test_wide_open_keeps_all_events_no_dedup():
    """Two detectors fire on the same direction at adjacent prices (within 0.5%);
    under wide_open both events MUST survive — no confidence-based dedup."""
    det = _make_detector(wide_open=True, max_detections=5)
    high_conf_event = _build_event("long", 0.95, 100.0, "pdh_pdl_reject")
    low_conf_event = _build_event("long", 0.13, 100.2, "camarilla_l3_reversal")  # 0.2% away
    all_detections = {
        "pdh_pdl_reject": StructureAnalysis(
            structure_detected=True, events=[high_conf_event], quality_score=95.0,
        ),
        "camarilla_l3_reversal": StructureAnalysis(
            structure_detected=True, events=[low_conf_event], quality_score=13.0,
        ),
    }
    result = det._resolve_conflicts_and_prioritize(all_detections, "NSE:TEST")
    assert len(result) == 2, (
        f"wide_open must preserve both events; got {len(result)}: "
        f"{[(e.structure_type, e.confidence) for e in result]}"
    )
    types = {e.structure_type for e in result}
    assert types == {"pdh_pdl_reject", "camarilla_l3_reversal"}


def test_wide_open_ignores_max_detections_cap():
    """With wide_open=true, the max_detections_per_symbol cap (default 5) is
    also bypassed — gauntlet must see every event."""
    det = _make_detector(wide_open=True, max_detections=2)
    events = []
    detections = {}
    for i in range(8):
        # 8 events on same side, all at distinct prices >0.5% apart
        e = _build_event("long", 0.5, 100.0 + i, structure_type=f"detector_{i}")
        events.append(e)
        detections[f"detector_{i}"] = StructureAnalysis(
            structure_detected=True, events=[e], quality_score=50.0,
        )
    result = det._resolve_conflicts_and_prioritize(detections, "NSE:TEST")
    assert len(result) == 8, (
        f"wide_open must ignore max_detections cap of 2; got {len(result)}"
    )


def test_strict_mode_still_dedups_and_caps():
    """Regression: when wide_open=false, the existing dedup + cap behavior
    is preserved — production trading must not change."""
    det = _make_detector(wide_open=False, max_detections=5)
    high_conf = _build_event("long", 0.95, 100.0, "pdh_pdl_reject")
    low_conf = _build_event("long", 0.13, 100.2, "camarilla_l3_reversal")  # 0.2% away
    detections = {
        "pdh_pdl_reject": StructureAnalysis(
            structure_detected=True, events=[high_conf], quality_score=95.0,
        ),
        "camarilla_l3_reversal": StructureAnalysis(
            structure_detected=True, events=[low_conf], quality_score=13.0,
        ),
    }
    result = det._resolve_conflicts_and_prioritize(detections, "NSE:TEST")
    # Original behavior: dedup keeps only the higher-confidence event
    assert len(result) == 1
    assert result[0].structure_type == "pdh_pdl_reject"
    assert result[0].confidence == 0.95


def test_strict_mode_keeps_distant_events():
    """Regression: when prices are >0.5% apart the dedup correctly keeps both.
    Verifies the bypass change didn't break the proximity logic itself."""
    det = _make_detector(wide_open=False, max_detections=5)
    e1 = _build_event("long", 0.95, 100.0, "pdh_pdl_reject")
    e2 = _build_event("long", 0.13, 105.0, "camarilla_l3_reversal")   # 5% away
    detections = {
        "pdh_pdl_reject": StructureAnalysis(
            structure_detected=True, events=[e1], quality_score=95.0,
        ),
        "camarilla_l3_reversal": StructureAnalysis(
            structure_detected=True, events=[e2], quality_score=13.0,
        ),
    }
    result = det._resolve_conflicts_and_prioritize(detections, "NSE:TEST")
    assert len(result) == 2
