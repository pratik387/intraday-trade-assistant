"""LiveGateChain integration tests — composed RuleFilter + CrossSectional + Conviction."""
import json
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from services.gate_chain.live_gate_chain import LiveGateChain


@pytest.fixture
def survivors_file(tmp_path):
    p = tmp_path / "survivors.json"
    p.write_text(json.dumps({
        "survivors": [
            {"setup": "premium_zone_short", "rule_id": "premium_zone_short__cap_segment+hour_bucket=small_cap+morning"},
            {"setup": "range_bounce_short", "rule_id": "range_bounce_short__regime=chop"},
        ]
    }), encoding="utf-8")
    return p


@pytest.fixture
def disabled_chain():
    cfg = {"live_gate_chain": {"enabled": False}}
    return LiveGateChain(cfg, project_root=Path("."))


def _make_candidate(symbol="SYM", setup="premium_zone_short", cap="small_cap",
                    hour="morning", regime="chop",
                    sess=None, ts=None, rank_score=1.0):
    return {
        "symbol": symbol,
        "setup_type": setup,
        "cap_segment": cap,
        "hour_bucket": hour,
        "regime": regime,
        "decision_ts": ts or datetime(2025, 1, 2, 10, 0),
        "session_date_dt": sess or date(2025, 1, 2),
        "minute_of_day": 600,
        "rank_score": float(rank_score),
    }


def test_disabled_chain_passes_all_through_unchanged(disabled_chain):
    cands = [_make_candidate(symbol=f"S{i}") for i in range(5)]
    out = disabled_chain.evaluate(cands)
    assert out == cands


def test_chain_with_real_components_drops_unmatched_setup(survivors_file, tmp_path, monkeypatch):
    """Use a real chain with mocked scorer; rule filter rejects unknown setup."""
    # Build a minimal valid config
    cfg = {
        "live_gate_chain": {"enabled": True},
        "rule_filter_gate": {"survivors_path": str(survivors_file)},
        "cross_sectional_gate": {
            "enabled": True,
            "f1_rvol_enabled": False,  # disable F1 so it doesn't reject
            "f1_rvol_threshold_pct": 90.0,
            "f1_applicable_caps": [],
            "f1_skip_hour_buckets": [],
            "f1_min_history_sessions": 5,
            "f1_rolling_window_sessions": 20,
            "f2_crowdedness_enabled": False,
            "f2_crowdedness_threshold": 100,
            "f2_crowdedness_window_min": 5,
        },
        "conviction_gate": {
            "enabled": True,
            "model_artifact": "models/conviction/2026-04-22-universal-xgboost.json",
            "feature_spec_path": "models/conviction/2026-04-22-feature-spec.json",
            "daily_cap": 50,
            "min_predicted_r": -100.0,  # admit all
        },
        "dedup_gate": {"enabled": False, "cooloff_bars": 6, "require_setup_change": True},
        "rank_pctl_min": 0.80,
    }
    chain = LiveGateChain(cfg, project_root=Path("."))

    # Candidate with unknown setup_type — should be dropped at rule filter
    bad = _make_candidate(setup="vwap_lose_short")
    good = _make_candidate(setup="premium_zone_short", cap="small_cap", hour="morning")
    out = chain.evaluate([bad, good])
    assert len(out) == 1
    assert out[0]["setup_type"] == "premium_zone_short"
    assert "rule_filter" in bad["gate_reject_reason"]


def test_conviction_cap_binds(survivors_file):
    """Cap=2 → only 2 candidates admitted regardless of how many pass earlier."""
    cfg = {
        "live_gate_chain": {"enabled": True},
        "rule_filter_gate": {"survivors_path": str(survivors_file)},
        "cross_sectional_gate": {
            "enabled": True,
            "f1_rvol_enabled": False, "f1_rvol_threshold_pct": 90.0,
            "f1_applicable_caps": [], "f1_skip_hour_buckets": [],
            "f1_min_history_sessions": 5, "f1_rolling_window_sessions": 20,
            "f2_crowdedness_enabled": False, "f2_crowdedness_threshold": 100,
            "f2_crowdedness_window_min": 5,
        },
        "conviction_gate": {
            "enabled": True,
            "model_artifact": "models/conviction/2026-04-22-universal-xgboost.json",
            "feature_spec_path": "models/conviction/2026-04-22-feature-spec.json",
            "daily_cap": 2,
            "min_predicted_r": -100.0,
        },
        "dedup_gate": {"enabled": False, "cooloff_bars": 6, "require_setup_change": True},
        "rank_pctl_min": 0.80,
    }
    chain = LiveGateChain(cfg, project_root=Path("."))
    cands = [
        _make_candidate(symbol=f"S{i}", setup="premium_zone_short", cap="small_cap", hour="morning")
        for i in range(5)
    ]
    out = chain.evaluate(cands)
    assert len(out) == 2  # cap binds
    rejected = [c for c in cands if c.get("gate_reject_reason", "").startswith("conviction")]
    assert len(rejected) == 3


def test_chain_stats_count_by_stage(survivors_file):
    cfg = {
        "live_gate_chain": {"enabled": True},
        "rule_filter_gate": {"survivors_path": str(survivors_file)},
        "cross_sectional_gate": {
            "enabled": True,
            "f1_rvol_enabled": False, "f1_rvol_threshold_pct": 90.0,
            "f1_applicable_caps": [], "f1_skip_hour_buckets": [],
            "f1_min_history_sessions": 5, "f1_rolling_window_sessions": 20,
            "f2_crowdedness_enabled": False, "f2_crowdedness_threshold": 100,
            "f2_crowdedness_window_min": 5,
        },
        "conviction_gate": {
            "enabled": True,
            "model_artifact": "models/conviction/2026-04-22-universal-xgboost.json",
            "feature_spec_path": "models/conviction/2026-04-22-feature-spec.json",
            "daily_cap": 50,
            "min_predicted_r": -100.0,
        },
        "dedup_gate": {"enabled": False, "cooloff_bars": 6, "require_setup_change": True},
        "rank_pctl_min": 0.80,
    }
    chain = LiveGateChain(cfg, project_root=Path("."))

    cands = [
        _make_candidate(setup="vwap_lose_short"),  # rule_filter drop
        _make_candidate(setup="vwap_lose_short"),  # rule_filter drop
        _make_candidate(setup="premium_zone_short", cap="small_cap", hour="morning"),  # admit
    ]
    chain.evaluate(cands)
    stats = chain.stats()
    assert stats["in"] == 3
    assert stats["rule_drop"] == 2
    assert stats["admitted"] == 1


def test_session_boundary_resets_conviction_cap(survivors_file):
    """Cap reached on day 1 → day 2 cap resets, more admitted."""
    cfg = {
        "live_gate_chain": {"enabled": True},
        "rule_filter_gate": {"survivors_path": str(survivors_file)},
        "cross_sectional_gate": {
            "enabled": True,
            "f1_rvol_enabled": False, "f1_rvol_threshold_pct": 90.0,
            "f1_applicable_caps": [], "f1_skip_hour_buckets": [],
            "f1_min_history_sessions": 5, "f1_rolling_window_sessions": 20,
            "f2_crowdedness_enabled": False, "f2_crowdedness_threshold": 100,
            "f2_crowdedness_window_min": 5,
        },
        "conviction_gate": {
            "enabled": True,
            "model_artifact": "models/conviction/2026-04-22-universal-xgboost.json",
            "feature_spec_path": "models/conviction/2026-04-22-feature-spec.json",
            "daily_cap": 1,
            "min_predicted_r": -100.0,
        },
        "dedup_gate": {"enabled": False, "cooloff_bars": 6, "require_setup_change": True},
        "rank_pctl_min": 0.80,
    }
    chain = LiveGateChain(cfg, project_root=Path("."))

    day1 = [
        _make_candidate(symbol="A", setup="premium_zone_short", cap="small_cap", hour="morning",
                        sess=date(2025, 1, 2)),
        _make_candidate(symbol="B", setup="premium_zone_short", cap="small_cap", hour="morning",
                        sess=date(2025, 1, 2)),
    ]
    out_d1 = chain.evaluate(day1)
    assert len(out_d1) == 1  # cap binds

    day2 = [
        _make_candidate(symbol="C", setup="premium_zone_short", cap="small_cap", hour="morning",
                        sess=date(2025, 1, 3)),
    ]
    out_d2 = chain.evaluate(day2)
    assert len(out_d2) == 1  # cap reset


def test_predicted_r_annotated_on_admitted(survivors_file):
    cfg = {
        "live_gate_chain": {"enabled": True},
        "rule_filter_gate": {"survivors_path": str(survivors_file)},
        "cross_sectional_gate": {
            "enabled": True,
            "f1_rvol_enabled": False, "f1_rvol_threshold_pct": 90.0,
            "f1_applicable_caps": [], "f1_skip_hour_buckets": [],
            "f1_min_history_sessions": 5, "f1_rolling_window_sessions": 20,
            "f2_crowdedness_enabled": False, "f2_crowdedness_threshold": 100,
            "f2_crowdedness_window_min": 5,
        },
        "conviction_gate": {
            "enabled": True,
            "model_artifact": "models/conviction/2026-04-22-universal-xgboost.json",
            "feature_spec_path": "models/conviction/2026-04-22-feature-spec.json",
            "daily_cap": 50,
            "min_predicted_r": -100.0,
        },
        "dedup_gate": {"enabled": False, "cooloff_bars": 6, "require_setup_change": True},
        "rank_pctl_min": 0.80,
    }
    chain = LiveGateChain(cfg, project_root=Path("."))
    cand = _make_candidate(setup="premium_zone_short", cap="small_cap", hour="morning")
    out = chain.evaluate([cand])
    assert len(out) == 1
    assert "predicted_r" in out[0]
    assert isinstance(out[0]["predicted_r"], float)


def test_f2_crowdedness_records_and_binds(survivors_file):
    """Stage 5c parity: every rule-filter-surviving candidate must be recorded
    in the crowdedness counter regardless of accept/reject. Without recording,
    F2 sliding-window count never grows, the gate never binds, and live admits
    flood the cap at the first bar instead of distributing across the day.
    """
    cfg = {
        "live_gate_chain": {"enabled": True},
        "rule_filter_gate": {"survivors_path": str(survivors_file)},
        "cross_sectional_gate": {
            "enabled": True,
            "f1_rvol_enabled": False, "f1_rvol_threshold_pct": 90.0,
            "f1_applicable_caps": [], "f1_skip_hour_buckets": [],
            "f1_min_history_sessions": 5, "f1_rolling_window_sessions": 20,
            "f2_crowdedness_enabled": True,
            "f2_crowdedness_threshold": 3,  # 3rd of same-setup-in-window rejected
            "f2_crowdedness_window_min": 5,
        },
        "conviction_gate": {
            "enabled": True,
            "model_artifact": "models/conviction/2026-04-22-universal-xgboost.json",
            "feature_spec_path": "models/conviction/2026-04-22-feature-spec.json",
            "daily_cap": 50,
            "min_predicted_r": -100.0,
        },
        "dedup_gate": {"enabled": False, "cooloff_bars": 6, "require_setup_change": True},
        "rank_pctl_min": 0.80,
    }
    chain = LiveGateChain(cfg, project_root=Path("."))
    # 5 same-setup candidates in same minute → after 3rd, F2 should bind
    base_ts = datetime(2025, 1, 2, 10, 0)
    cands = [
        _make_candidate(symbol=f"S{i}", setup="premium_zone_short",
                        cap="small_cap", hour="morning", ts=base_ts)
        for i in range(5)
    ]
    chain.evaluate(cands)
    # First 3 admit; remaining 2 must be cs_dropped (F2 crowded)
    assert chain.stats()["cs_drop"] == 2, chain.stats()


def test_empty_input_returns_empty(survivors_file):
    cfg = {"live_gate_chain": {"enabled": True}, "rule_filter_gate": {"survivors_path": str(survivors_file)},
           "cross_sectional_gate": {"enabled": True, "f1_rvol_enabled": False, "f1_rvol_threshold_pct": 90.0,
                                   "f1_applicable_caps": [], "f1_skip_hour_buckets": [],
                                   "f1_min_history_sessions": 5, "f1_rolling_window_sessions": 20,
                                   "f2_crowdedness_enabled": False, "f2_crowdedness_threshold": 100,
                                   "f2_crowdedness_window_min": 5},
           "conviction_gate": {"enabled": True,
                              "model_artifact": "models/conviction/2026-04-22-universal-xgboost.json",
                              "feature_spec_path": "models/conviction/2026-04-22-feature-spec.json",
                              "daily_cap": 50, "min_predicted_r": -100.0},
           "dedup_gate": {"enabled": False, "cooloff_bars": 6, "require_setup_change": True},
           "rank_pctl_min": 0.80}
    chain = LiveGateChain(cfg, project_root=Path("."))
    assert chain.evaluate([]) == []


def test_dedup_stage_blocks_second_same_setup(survivors_file):
    """Dedup stage D: when two same-symbol same-setup candidates pass the first
    three stages in the same bar, only the higher-ranked one is admitted; the
    other is dropped at dedup with reason 'dedup:cooloff ...' since bars_gap=0
    is less than cooloff_bars=6.
    """
    cfg = {
        "live_gate_chain": {"enabled": True},
        "rule_filter_gate": {"survivors_path": str(survivors_file)},
        "cross_sectional_gate": {
            "enabled": True,
            "f1_rvol_enabled": False, "f1_rvol_threshold_pct": 90.0,
            "f1_applicable_caps": [], "f1_skip_hour_buckets": [],
            "f1_min_history_sessions": 5, "f1_rolling_window_sessions": 20,
            "f2_crowdedness_enabled": False, "f2_crowdedness_threshold": 100,
            "f2_crowdedness_window_min": 5,
        },
        "conviction_gate": {
            "enabled": True,
            "model_artifact": "models/conviction/2026-04-22-universal-xgboost.json",
            "feature_spec_path": "models/conviction/2026-04-22-feature-spec.json",
            "daily_cap": 50,
            "min_predicted_r": -100.0,
        },
        "dedup_gate": {"enabled": True, "cooloff_bars": 6, "require_setup_change": True},
        "rank_pctl_min": 0.80,
    }
    chain = LiveGateChain(cfg, project_root=Path("."))

    # Two candidates, same symbol + same setup, same bar. First (higher rank)
    # admits; second rejected with cooloff.
    base_ts = datetime(2025, 1, 2, 10, 0)
    c1 = _make_candidate(symbol="SYMX", setup="premium_zone_short",
                         cap="small_cap", hour="morning", ts=base_ts,
                         rank_score=2.0)
    c2 = _make_candidate(symbol="SYMX", setup="premium_zone_short",
                         cap="small_cap", hour="morning", ts=base_ts,
                         rank_score=1.0)
    out = chain.evaluate([c1, c2])

    assert len(out) == 1
    assert out[0]["rank_score"] == 2.0
    # Second candidate annotated with dedup rejection
    rejected = [c for c in [c1, c2] if str(c.get("gate_reject_reason", "")).startswith("dedup")]
    assert len(rejected) == 1
    assert chain.stats()["dedup_drop"] == 1


def test_wide_open_mode_forces_passthrough_even_when_chain_enabled(survivors_file):
    """Sub-project #5 master kill-switch: when wide_open_mode=true at top-level
    config, evaluate() returns input unchanged even if live_gate_chain.enabled=true.
    Used by the OCI capture run to log the maximal pre-gate candidate pool."""
    cfg = {
        "wide_open_mode": True,            # ← master kill
        "live_gate_chain": {"enabled": True},
        "rule_filter_gate": {"survivors_path": str(survivors_file)},
        "cross_sectional_gate": {
            "enabled": True,
            "f1_rvol_enabled": False, "f1_rvol_threshold_pct": 90.0,
            "f1_applicable_caps": [], "f1_skip_hour_buckets": [],
            "f1_min_history_sessions": 5, "f1_rolling_window_sessions": 20,
            "f2_crowdedness_enabled": False, "f2_crowdedness_threshold": 100,
            "f2_crowdedness_window_min": 5,
        },
        "conviction_gate": {
            "enabled": True,
            "model_artifact": "models/conviction/2026-04-22-universal-xgboost.json",
            "feature_spec_path": "models/conviction/2026-04-22-feature-spec.json",
            "daily_cap": 2, "min_predicted_r": -100.0,
        },
        "dedup_gate": {"enabled": False, "cooloff_bars": 6, "require_setup_change": True},
        "rank_pctl_min": 0.80,
    }
    chain = LiveGateChain(cfg, project_root=Path("."))
    # 5 candidates with a setup_type NOT in survivors — would normally be rule_filter dropped
    cands = [
        _make_candidate(symbol=f"S{i}", setup="vwap_lose_short")
        for i in range(5)
    ]
    out = chain.evaluate(cands)
    # Passthrough: all 5 returned, no gate_reject_reason annotations
    assert out == cands
    for c in cands:
        assert "gate_reject_reason" not in c
