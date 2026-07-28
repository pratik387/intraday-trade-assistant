"""Tests for jobs.check_edge_integrity (edge-integrity monitors).

Spec: specs/2026-07-28-edge-integrity-monitors-design.md
Mirrors the conventions of tests/methodology/test_circuit_breaker_job.py:
pure-function unit tests on tmp_path fixtures + end-to-end run() tests
against a throwaway config file.
"""
import json
import math
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from jobs.check_edge_integrity import (
    ConfigError,
    PAUSED_STATE,
    check_data_source,
    check_factor_cluster,
    check_rule_watch,
    combine_cluster_daily_pnl,
    load_edge_integrity_config,
    load_member_daily_pnl,
    load_rule_changes,
    run,
    validate_preconditions,
)

TODAY = date(2026, 7, 28)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _rules_csv(tmp_path: Path, rows: list) -> Path:
    p = tmp_path / "data" / "rule_changes.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def _rule_row(effective: str, affects: str, **extra) -> dict:
    return {
        "effective_date": effective,
        "announced_date": effective,
        "category": extra.get("category", "fno_position"),
        "severity": extra.get("severity", "critical"),
        "affects": affects,
        "description": extra.get("description", "test circular"),
    }


def _feed_parquet(tmp_path: Path, rel: str, dates: list) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": pd.to_datetime(dates)}).to_parquet(p)
    return p


def _ledger(tmp_path: Path, setup_name: str, trades: list) -> Path:
    """Write a decay-tripwire-shaped ledger state file."""
    p = tmp_path / "state" / f"decay_tripwire_{setup_name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"setup_name": setup_name, "trades": trades}))
    return p


def _trade(pnl: float, d: str, attributed: bool = None) -> dict:
    t = {"net_pnl_inr": pnl, "ts_iso": f"{d}T09:30:00"}
    if attributed is not None:
        t["attributed"] = attributed
    return t


def _base_config(*, dry_run: bool = True, cluster_members=None) -> dict:
    members = cluster_members or ["member_a", "member_b"]
    cfg = {
        "setups": {
            "rule_dep_setup": {
                "enabled": True,
                "regulatory_sensitivity": "rule_dependent",
                "preconditions": [
                    {"type": "rule", "watch": "MWPL"},
                    {"type": "data_source",
                     "path": "data/feed.parquet",
                     "date_column": "date",
                     "max_staleness_days": 7,
                     "min_rows_per_week": 0},
                ],
            },
        },
        "edge_integrity": {
            "enabled": True,
            "dry_run": dry_run,
            "rule_watch_lookback_days": 90,
            "rule_changes_csv": "data/rule_changes.csv",
            "report_dir": "state",
            "factor_clusters": {
                "test_cluster": {
                    "members": members,
                    "lookback_days": 5,
                    "drawdown_threshold_sigma": 2.0,
                    "min_baseline_days": 10,
                },
            },
        },
    }
    for m in members:
        cfg["setups"][m] = {
            "enabled": False,
            "paper_enabled": True,
            "decay_tripwire": {
                "window_trades": 30, "pf_floor": 1.2, "sustained_weeks": 6,
                "state_file": f"state/decay_tripwire_{m}.json",
            },
        }
    return cfg


def _write_config(tmp_path: Path, cfg: dict) -> Path:
    p = tmp_path / "configuration.json"
    p.write_text(json.dumps(cfg, indent=2))
    return p


def _env(tmp_path: Path, *, dry_run: bool = True, rule_rows=None,
         feed_dates=None, ledgers=None) -> Path:
    """Build a full runnable environment under tmp_path; returns config path."""
    cfg = _base_config(dry_run=dry_run)
    _rules_csv(tmp_path, rule_rows if rule_rows is not None else [
        _rule_row("2024-01-01", "STT_drag"),  # old, unwatched
    ])
    _feed_parquet(tmp_path, "data/feed.parquet",
                  feed_dates if feed_dates is not None
                  else [(TODAY - timedelta(days=1)).isoformat()])
    for name, trades in (ledgers or {}).items():
        _ledger(tmp_path, name, trades)
    return _write_config(tmp_path, cfg)


# ---------------------------------------------------------------------------
# Startup validation / fail-fast
# ---------------------------------------------------------------------------

def test_missing_edge_integrity_block_fails_fast():
    with pytest.raises(ConfigError, match="edge_integrity"):
        load_edge_integrity_config({"setups": {}})


def test_missing_cluster_key_fails_fast():
    cfg = _base_config()
    del cfg["edge_integrity"]["factor_clusters"]["test_cluster"]["drawdown_threshold_sigma"]
    with pytest.raises(ConfigError, match="drawdown_threshold_sigma"):
        load_edge_integrity_config(cfg)


def test_missing_top_level_key_fails_fast():
    cfg = _base_config()
    del cfg["edge_integrity"]["rule_watch_lookback_days"]
    with pytest.raises(ConfigError, match="rule_watch_lookback_days"):
        load_edge_integrity_config(cfg)


def test_rule_dependent_setup_without_rule_precondition_is_validation_error():
    cfg = _base_config()
    cfg["setups"]["rule_dep_setup"]["preconditions"] = [
        {"type": "data_source", "path": "x.parquet", "date_column": "date",
         "max_staleness_days": 7, "min_rows_per_week": 0},
    ]
    errors = validate_preconditions(cfg)
    assert any("rule_dependent" in e for e in errors)


def test_rule_dependent_validation_error_aborts_run(tmp_path):
    config_path = _env(tmp_path)
    cfg = json.loads(config_path.read_text())
    cfg["setups"]["rule_dep_setup"]["preconditions"] = []
    config_path.write_text(json.dumps(cfg))
    with pytest.raises(ConfigError, match="rule_dependent"):
        run(config_path, None, TODAY, repo_root=tmp_path)


def test_malformed_data_source_precondition_is_validation_error():
    cfg = _base_config()
    cfg["setups"]["rule_dep_setup"]["preconditions"].append(
        {"type": "data_source", "path": "x.parquet"})  # missing keys
    errors = validate_preconditions(cfg)
    assert any("data_source" in e and "max_staleness_days" in e for e in errors)


# ---------------------------------------------------------------------------
# Monitor 1 — rule watch
# ---------------------------------------------------------------------------

def _rules_df(tmp_path, rows):
    return load_rule_changes(_rules_csv(tmp_path, rows))


def test_rule_watch_triggers_inside_window(tmp_path):
    df = _rules_df(tmp_path, [
        _rule_row((TODAY - timedelta(days=30)).isoformat(), "MWPL;single_stock_FO"),
    ])
    findings = check_rule_watch("s", [{"type": "rule", "watch": "MWPL"}],
                                df, TODAY, lookback_days=90)
    assert [f.action for f in findings] == ["pause"]
    assert findings[0].reason == "rule_watch:MWPL"


def test_rule_watch_ignores_rules_outside_window(tmp_path):
    df = _rules_df(tmp_path, [
        _rule_row((TODAY - timedelta(days=91)).isoformat(), "MWPL"),
    ])
    findings = check_rule_watch("s", [{"type": "rule", "watch": "MWPL"}],
                                df, TODAY, lookback_days=90)
    assert [f.action for f in findings] == ["ok"]


def test_rule_watch_boundary_day_triggers(tmp_path):
    """effective_date == today - lookback_days is inside the window (>=)."""
    df = _rules_df(tmp_path, [
        _rule_row((TODAY - timedelta(days=90)).isoformat(), "MWPL"),
    ])
    findings = check_rule_watch("s", [{"type": "rule", "watch": "MWPL"}],
                                df, TODAY, lookback_days=90)
    assert [f.action for f in findings] == ["pause"]


def test_rule_watch_future_effective_date_triggers(tmp_path):
    """A circular announced today with a future effective date must alert NOW
    (the Oct-2025 MWPL case: announced 125 days before effect)."""
    df = _rules_df(tmp_path, [
        _rule_row((TODAY + timedelta(days=60)).isoformat(), "MWPL"),
    ])
    findings = check_rule_watch("s", [{"type": "rule", "watch": "MWPL"}],
                                df, TODAY, lookback_days=90)
    assert [f.action for f in findings] == ["pause"]


def test_rule_watch_matches_whole_affects_tokens_only(tmp_path):
    """watch='MWPL' must not match affects token 'MWPL_other' (token equality,
    not substring)."""
    df = _rules_df(tmp_path, [
        _rule_row((TODAY - timedelta(days=10)).isoformat(), "MWPL_other;STT_drag"),
    ])
    findings = check_rule_watch("s", [{"type": "rule", "watch": "MWPL"}],
                                df, TODAY, lookback_days=90)
    assert [f.action for f in findings] == ["ok"]


# ---------------------------------------------------------------------------
# Monitor 2 — data-source health
# ---------------------------------------------------------------------------

def _ds_precondition(**overrides) -> dict:
    p = {"type": "data_source", "path": "data/feed.parquet",
         "date_column": "date", "max_staleness_days": 7,
         "min_rows_per_week": 0}
    p.update(overrides)
    return p


def test_staleness_math_fresh_feed_ok(tmp_path):
    _feed_parquet(tmp_path, "data/feed.parquet",
                  [(TODAY - timedelta(days=3)).isoformat()])
    f = check_data_source("s", _ds_precondition(), TODAY, tmp_path)
    assert f.action == "ok"
    assert f.detail["staleness_days"] == 3


def test_staleness_math_stale_feed_pauses(tmp_path):
    _feed_parquet(tmp_path, "data/feed.parquet",
                  [(TODAY - timedelta(days=10)).isoformat()])
    f = check_data_source("s", _ds_precondition(), TODAY, tmp_path)
    assert f.action == "pause"
    assert f.reason == "data_source_stale"
    assert f.detail["staleness_days"] == 10
    assert f.detail["max_staleness_days"] == 7


def test_staleness_boundary_exactly_max_is_ok(tmp_path):
    """staleness == max_staleness_days does NOT trigger (strict >)."""
    _feed_parquet(tmp_path, "data/feed.parquet",
                  [(TODAY - timedelta(days=7)).isoformat()])
    f = check_data_source("s", _ds_precondition(), TODAY, tmp_path)
    assert f.action == "ok"


def test_volume_collapse_detected(tmp_path):
    """Feed is fresh (one straggler row) but weekly volume collapsed —
    the announcements_fr pattern: file exists, feed is dead."""
    dates = [(TODAY - timedelta(days=60 + i)).isoformat() for i in range(50)]
    dates += [(TODAY - timedelta(days=1)).isoformat()]  # fresh but alone
    _feed_parquet(tmp_path, "data/feed.parquet", dates)
    f = check_data_source("s", _ds_precondition(min_rows_per_week=10),
                          TODAY, tmp_path)
    assert f.action == "pause"
    assert f.reason == "data_source_volume_collapse"
    assert f.detail["rows_last_week"] == 1


def test_volume_check_disabled_when_min_rows_zero(tmp_path):
    dates = [(TODAY - timedelta(days=1)).isoformat()]
    _feed_parquet(tmp_path, "data/feed.parquet", dates)
    f = check_data_source("s", _ds_precondition(min_rows_per_week=0),
                          TODAY, tmp_path)
    assert f.action == "ok"


def test_volume_healthy_feed_ok(tmp_path):
    dates = [(TODAY - timedelta(days=d)).isoformat()
             for d in range(1, 7) for _ in range(5)]
    _feed_parquet(tmp_path, "data/feed.parquet", dates)
    f = check_data_source("s", _ds_precondition(min_rows_per_week=10),
                          TODAY, tmp_path)
    assert f.action == "ok"
    assert f.detail["rows_last_week"] == 30


def test_missing_feed_file_pauses(tmp_path):
    f = check_data_source("s", _ds_precondition(path="data/nope.parquet"),
                          TODAY, tmp_path)
    assert f.action == "pause"
    assert f.reason == "data_source_missing"


def test_empty_feed_file_pauses(tmp_path):
    _feed_parquet(tmp_path, "data/feed.parquet", [])
    f = check_data_source("s", _ds_precondition(), TODAY, tmp_path)
    assert f.action == "pause"
    assert f.reason == "data_source_empty"


# ---------------------------------------------------------------------------
# Monitor 3 — factor-cluster tripwire
# ---------------------------------------------------------------------------

def test_member_daily_pnl_from_ledger_groups_by_day(tmp_path):
    _ledger(tmp_path, "m", [
        _trade(100.0, "2026-07-01"), _trade(-40.0, "2026-07-01"),
        _trade(25.0, "2026-07-02"),
    ])
    cfg = {"decay_tripwire": {"state_file": "state/decay_tripwire_m.json"}}
    s = load_member_daily_pnl("m", cfg, None, tmp_path)
    assert s.loc[date(2026, 7, 1)] == pytest.approx(60.0)
    assert s.loc[date(2026, 7, 2)] == pytest.approx(25.0)


def test_member_daily_pnl_excludes_attributed_mirror_rows(tmp_path):
    """Composite mirror rows (attributed=True) must NOT count — a portfolio
    view summing members would otherwise double-count one book position."""
    _ledger(tmp_path, "m", [
        _trade(100.0, "2026-07-01"),
        _trade(999.0, "2026-07-01", attributed=True),
    ])
    cfg = {"decay_tripwire": {"state_file": "state/decay_tripwire_m.json"}}
    s = load_member_daily_pnl("m", cfg, None, tmp_path)
    assert s.loc[date(2026, 7, 1)] == pytest.approx(100.0)


def test_member_daily_pnl_from_trades_csv(tmp_path):
    """Members without a decay-tripwire ledger fall back to the trades CSV
    (same schema as jobs/check_circuit_breakers.py)."""
    csv = tmp_path / "trade_report.csv"
    pd.DataFrame([
        {"signal_date": "2026-07-01", "setup_type": "m",
         "actual_pnl_after_charges": 10.0},
        {"signal_date": "2026-07-01", "setup_type": "m",
         "actual_pnl_after_charges": -4.0},
        {"signal_date": "2026-07-01", "setup_type": "other",
         "actual_pnl_after_charges": 999.0},
    ]).to_csv(csv, index=False)
    s = load_member_daily_pnl("m", {}, csv, tmp_path)
    assert s.loc[date(2026, 7, 1)] == pytest.approx(6.0)


def test_cluster_aggregation_is_daily_net_sum_across_members():
    """Combined daily PnL = per-day sum across members, absent member = 0
    (the July-2026 factor-study methodology)."""
    a = pd.Series({date(2026, 7, 1): 100.0, date(2026, 7, 2): -50.0})
    b = pd.Series({date(2026, 7, 2): 30.0, date(2026, 7, 3): 20.0})
    combined = combine_cluster_daily_pnl({"a": a, "b": b})
    assert combined.loc[date(2026, 7, 1)] == pytest.approx(100.0)
    assert combined.loc[date(2026, 7, 2)] == pytest.approx(-20.0)
    assert combined.loc[date(2026, 7, 3)] == pytest.approx(20.0)


def _cluster_cfg(**overrides) -> dict:
    c = {"members": ["a", "b"], "lookback_days": 5,
         "drawdown_threshold_sigma": 2.0, "min_baseline_days": 10}
    c.update(overrides)
    return c


def test_cluster_tripwire_breaches_on_drawdown():
    """20 calm days then 5 heavily-negative days -> z breach pauses ALL members."""
    days = [date(2026, 6, 1) + timedelta(days=i) for i in range(25)]
    vals = [100.0] * 10 + [-100.0] * 10 + [-3000.0] * 5
    a = pd.Series(dict(zip(days, vals)))
    findings, stats = check_factor_cluster("c", _cluster_cfg(),
                                           {"a": a, "b": pd.Series(dtype=float)})
    assert stats["status"] == "evaluated"
    # verify z against the documented formula
    combined = a
    mean, sigma = combined.mean(), combined.std(ddof=1)
    expected_z = (combined.iloc[-5:].sum() - 5 * mean) / (sigma * math.sqrt(5))
    assert stats["z"] == pytest.approx(expected_z)
    assert stats["breach"] is True
    assert {f.setup_name for f in findings} == {"a", "b"}
    assert all(f.action == "pause" for f in findings)


def test_cluster_tripwire_quiet_on_normal_variation():
    days = [date(2026, 6, 1) + timedelta(days=i) for i in range(30)]
    vals = [100.0 if i % 2 else -80.0 for i in range(30)]
    a = pd.Series(dict(zip(days, vals)))
    findings, stats = check_factor_cluster("c", _cluster_cfg(),
                                           {"a": a, "b": pd.Series(dtype=float)})
    assert stats["breach"] is False
    assert all(f.action == "ok" for f in findings)


def test_cluster_insufficient_data_skips():
    days = [date(2026, 6, 1) + timedelta(days=i) for i in range(5)]
    a = pd.Series(dict(zip(days, [-1000.0] * 5)))
    findings, stats = check_factor_cluster("c", _cluster_cfg(),
                                           {"a": a, "b": pd.Series(dtype=float)})
    assert stats["status"] == "insufficient_data"
    assert all(f.action == "skip" for f in findings)


# ---------------------------------------------------------------------------
# End-to-end run(): dry_run semantics + pause machinery
# ---------------------------------------------------------------------------

def _triggering_env(tmp_path: Path, *, dry_run: bool) -> Path:
    """Environment where the rule watch AND data staleness both trigger."""
    return _env(
        tmp_path, dry_run=dry_run,
        rule_rows=[_rule_row((TODAY - timedelta(days=5)).isoformat(),
                             "MWPL;single_stock_FO")],
        feed_dates=[(TODAY - timedelta(days=30)).isoformat()],  # stale
    )


def test_dry_run_never_mutates_state(tmp_path):
    config_path = _triggering_env(tmp_path, dry_run=True)
    before = config_path.read_text()

    summary = run(config_path, None, TODAY, repo_root=tmp_path)

    assert summary["dry_run"] is True
    assert summary["n_pause_triggers"] >= 2   # rule watch + stale feed
    assert summary["n_pauses_applied"] == 0
    assert config_path.read_text() == before  # config byte-identical
    cfg = json.loads(config_path.read_text())
    assert "cb_state" not in cfg["setups"]["rule_dep_setup"]
    # report JSON IS written even in dry-run
    report = json.loads(Path(summary["report_path"]).read_text())
    assert report["dry_run"] is True
    assert any(r.get("skip_cause") == "dry_run" for r in report["pauses"])


def test_live_mode_pauses_via_cb_state_machinery(tmp_path):
    config_path = _triggering_env(tmp_path, dry_run=False)

    summary = run(config_path, None, TODAY, repo_root=tmp_path)

    assert summary["n_pauses_applied"] == 1
    blk = json.loads(config_path.read_text())["setups"]["rule_dep_setup"]
    assert blk["cb_state"] == PAUSED_STATE
    assert blk["cb_disabled_at"] == TODAY.isoformat()
    assert blk["cb_disabled_reason"].startswith("edge_integrity:")


def test_live_mode_is_idempotent_no_repause(tmp_path):
    """Second run over an already-paused setup applies nothing (and never
    clobbers cb_disabled_at)."""
    config_path = _triggering_env(tmp_path, dry_run=False)
    run(config_path, None, TODAY, repo_root=tmp_path)
    first = json.loads(config_path.read_text())["setups"]["rule_dep_setup"]

    summary2 = run(config_path, None, TODAY + timedelta(days=1),
                   repo_root=tmp_path)

    assert summary2["n_pauses_applied"] == 0
    second = json.loads(config_path.read_text())["setups"]["rule_dep_setup"]
    assert second == first


def test_cluster_breach_pauses_all_members_live(tmp_path):
    days = [(date(2026, 6, 1) + timedelta(days=i)).isoformat() for i in range(25)]
    a_trades = [_trade(100.0, d) for d in days[:20]] + \
               [_trade(-3000.0, d) for d in days[20:]]
    b_trades = [_trade(50.0, d) for d in days[:20]] + \
               [_trade(-2000.0, d) for d in days[20:]]
    config_path = _env(tmp_path, dry_run=False,
                       ledgers={"member_a": a_trades, "member_b": b_trades})

    summary = run(config_path, None, TODAY, repo_root=tmp_path)

    assert summary["n_pauses_applied"] == 2
    cfg = json.loads(config_path.read_text())
    assert cfg["setups"]["member_a"]["cb_state"] == PAUSED_STATE
    assert cfg["setups"]["member_b"]["cb_state"] == PAUSED_STATE
    # rule_dep_setup did NOT trigger (default env: old rule, fresh feed)
    assert "cb_state" not in cfg["setups"]["rule_dep_setup"]


def test_enabled_false_short_circuits(tmp_path):
    config_path = _env(tmp_path)
    cfg = json.loads(config_path.read_text())
    cfg["edge_integrity"]["enabled"] = False
    config_path.write_text(json.dumps(cfg))
    assert run(config_path, None, TODAY, repo_root=tmp_path) == {"enabled": False}


def test_report_written_every_run(tmp_path):
    config_path = _env(tmp_path)
    summary = run(config_path, None, TODAY, repo_root=tmp_path)
    report_path = Path(summary["report_path"])
    assert report_path.name == f"edge_integrity_report_{TODAY.isoformat()}.json"
    report = json.loads(report_path.read_text())
    assert report["run_date"] == TODAY.isoformat()
    assert report["factor_clusters"][0]["cluster"] == "test_cluster"


# ---------------------------------------------------------------------------
# Pause state honored by entry points
# ---------------------------------------------------------------------------

def test_intraday_orchestrator_blocks_paused_precondition():
    from services.plan_orchestrator import _setup_should_fire
    assert _setup_should_fire(
        {"enabled": True, "cb_state": PAUSED_STATE}) == (False, 0.0)
    assert _setup_should_fire(
        {"enabled": True, "cb_state": "enabled"})[0] is True


def test_multiday_cron_entry_gate_flags_paused_setups():
    from services.execution.mtf_capitulation_handlers import _cb_paused_setups
    setups = [
        ("a", {"cb_state": PAUSED_STATE}),
        ("b", {"cb_state": "disabled"}),
        ("c", {}),
        ("d", {"cb_state": "enabled"}),
    ]
    assert _cb_paused_setups(setups) == ["a", "b"]


def test_unknown_cb_state_blocks_in_all_three_entry_gates():
    """REGRESSION (review 2026-07-28): the original blocklist let ANY unknown
    cb_state value fall through and fire at full size. All entry gates now use
    the shared allowlist (services.cb_state.is_cb_active) — an unrecognized or
    typo'd state must BLOCK new entries everywhere."""
    from services.cb_state import is_cb_active, ACTIVE_CB_STATES
    from services.plan_orchestrator import _setup_should_fire
    from services.execution.mtf_capitulation_handlers import _cb_paused_setups

    for weird in ("paused_manual", "under_review", "DISABLED", "enabled ",
                  "", None, 0):
        cfg = {"enabled": True, "cb_state": weird}
        assert is_cb_active(cfg) is False, weird
        assert _setup_should_fire(cfg) == (False, 0.0), weird
        assert _cb_paused_setups([("x", cfg)]) == ["x"], weird

    # Allowlist itself still passes, and a missing key defaults to active.
    for ok_state in ACTIVE_CB_STATES:
        assert is_cb_active({"cb_state": ok_state}) is True
    assert is_cb_active({}) is True


def test_overnight_entry_gate_blocks_unknown_cb_state():
    """Same regression class for the overnight run_entry filter, exercised at
    the helper level (run_entry uses is_cb_active on raw_config)."""
    from services.cb_state import is_cb_active

    class FakeSetup:
        def __init__(self, name, raw):
            self.name, self.raw_config = name, raw

    setups = [FakeSetup("good", {"cb_state": "enabled"}),
              FakeSetup("weird", {"cb_state": "pausd_precondition"})]  # typo
    paused = [s.name for s in setups if not is_cb_active(s.raw_config)]
    assert paused == ["weird"]
