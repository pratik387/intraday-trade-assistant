"""Results-day size tilt for up_spike_fade_short (built OFF 2026-11-05).

Pins:
  1. a malformed config block stops the daemon at startup;
  2. through the REAL plan builder, the multiplier reaches both the notional and
     the circuit-loss budget. The circuit cap set the size on every live up_spike
     trade of 11-03..05, so scaling the notional alone would change nothing;
  3. switched off, sizing and the RESULTS_CONTEXT record are what they were;
  4. every value the tagger can return has a multiplier.
"""
import inspect
import io
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import services.plan_orchestrator as po
import services.price_bands as pbm
import services.results_context as rc
from services.risk.intraday_sizing import size_intraday_position
from services.risk.results_size_tilt import (
    ResultsSizeTiltConfigError, results_size_multiplier, validate_results_size_tilt)

ROOT = Path(__file__).resolve().parents[2]
CFG = json.load(io.open(ROOT / "config" / "configuration.json", encoding="utf-8"))
UP = CFG["setups"]["up_spike_fade_short"]
SZ = CFG["intraday_sizing"]
MULT = {"results_preopen": 2.0, "board_meeting_today": 2.0, "none": 0.5, "unknown": 0.5}
PREMIUM = float(SZ["circuit_risk_auction_premium_pct"]) / 100.0


def _tilt(enabled=True, max_multiplier=2.0, **over):
    m = dict(MULT)
    m.update(over)
    return {"enabled": enabled, "max_multiplier": max_multiplier, "multipliers": m}


# --- the multiplier ----------------------------------------------------------

def test_absent_or_disabled_tilt_is_one():
    assert results_size_multiplier(None, None, "s") == 1.0
    for tag in sorted(rc.ALL_TAGS):
        assert results_size_multiplier(_tilt(enabled=False), tag, "s") == 1.0


@pytest.mark.parametrize("tag, want", sorted(MULT.items()))
def test_enabled_tilt_reads_the_tag(tag, want):
    assert results_size_multiplier(_tilt(), tag, "s") == want


def test_enabled_tilt_rejects_a_tag_the_tagger_cannot_return():
    for tag in (None, "results_intraday"):
        with pytest.raises(ResultsSizeTiltConfigError):
            results_size_multiplier(_tilt(), tag, "s")


@pytest.mark.parametrize("cfg", [
    [],                                                       # not an object
    {"max_multiplier": 2.0, "multipliers": MULT},             # enabled missing
    {"enabled": "false", "max_multiplier": 2.0, "multipliers": MULT},
    {"enabled": True, "multipliers": MULT},                   # max_multiplier missing
    _tilt(max_multiplier=0.0), _tilt(max_multiplier=True),
    {"enabled": True, "max_multiplier": 2.0},                 # multipliers missing
    {"enabled": True, "max_multiplier": 2.0, "multipliers": {k: v for k, v in MULT.items() if k != "unknown"}},
    {"enabled": True, "max_multiplier": 2.0, "multipliers": dict(MULT, results_intraday=2.0)},
    _tilt(none=0.0), _tilt(none=-0.5), _tilt(none=float("nan")), _tilt(none=float("inf")),
    _tilt(none=True), _tilt(none="0.5"),
    _tilt(results_preopen=20.0),                              # the typo the ceiling exists for
])
def test_malformed_blocks_are_config_errors(cfg):
    with pytest.raises(ResultsSizeTiltConfigError):
        validate_results_size_tilt(cfg, "s")


def test_comment_keys_inside_multipliers_are_allowed():
    validate_results_size_tilt(_tilt(_note="x"), "s")


# --- every tagger output has a multiplier -------------------------------------

@pytest.fixture
def cal(tmp_path):
    rows = [
        ("NSE:AMC", "2026-11-02 18:30", "AMC", "2026-11-03", None, "2026-11-02"),
        ("NSE:BMO", "2026-11-03 08:10", "BMO", "2026-11-03", None, "2026-11-03"),
        ("NSE:SCHED", "2026-11-03 09:00", "scheduled", "2026-11-03", "2026-11-03", "2026-11-03"),
        ("NSE:INTRA", "2026-11-03 13:05", "intraday", "2026-11-03", None, "2026-11-03"),
        ("NSE:LATE", "2026-11-03 19:00", "AMC", "2026-11-04", None, "2026-11-03"),
        ("NSE:OLD", "2026-10-27 13:00", "intraday", "2026-10-27", None, "2026-10-27"),
    ]
    df = pd.DataFrame(rows, columns=["symbol", "announce_time", "announce_time_class", "trade_date",
                                     "bm_date", "announce_date"])
    df["announce_time"] = pd.to_datetime(df["announce_time"])
    df["announce_date"] = pd.to_datetime(df["announce_date"])
    p = tmp_path / "earnings_events.parquet"
    df.to_parquet(p)
    rc._tags_for.cache_clear()
    yield p
    rc._tags_for.cache_clear()


def test_every_tag_the_tagger_returns_has_a_multiplier(cal, tmp_path):
    seen = {rc.results_context(s, "2026-11-03", path=cal)
            for s in ("NSE:AMC", "NSE:BMO", "NSE:SCHED", "NSE:INTRA", "NSE:LATE", "NSE:OLD", "NSE:NOPE")}
    seen.add(rc.results_context("NSE:AMC", "2026-11-03", path=tmp_path / "missing.parquet"))
    assert seen == rc.ALL_TAGS
    for tag in seen:
        results_size_multiplier(_tilt(), tag, "s")


# --- sizing arithmetic, composed as plan_orchestrator does -------------------

def _size(tilt, circuit_distance_frac, capital=200_000.0):
    book = float(SZ["book_size_multiplier"])
    return size_intraday_position(
        setup_name="up_spike_fade_short", sizing_mode="notional", entry_price=100.0,
        target_notional_pct=float(UP["target_notional_pct"]), total_capital_inr=capital,
        min_notional_inr=float(SZ["min_notional_pct_of_capital"]) * capital,
        max_notional_inr=float(SZ["max_notional_pct_of_capital"]) * capital,
        book_size_multiplier=book * tilt,
        circuit_risk_budget_inr=float(UP["circuit_risk_budget_pct_of_capital"]) * capital * book * tilt,
        circuit_distance_frac=circuit_distance_frac, circuit_premium_frac=PREMIUM)


def test_tilted_locked_loss_stays_inside_the_tilted_budget():
    budget = float(UP["circuit_risk_budget_pct_of_capital"]) * 200_000.0 * float(SZ["book_size_multiplier"])
    for dist in (0.0, 0.05, 0.0969, 0.187, 0.30):
        r = _size(2.0, dist)
        assert r.notional_inr * (dist + PREMIUM) <= 2 * budget + 1e-6


def test_max_notional_stays_a_hard_ceiling():
    # circuit 1% away: the 2x cap does not bind and 2 x the flat size exceeds the max
    capital = 200_000.0
    r = _size(2.0, 0.01, capital)
    assert r.clamped == "max"
    assert r.notional_inr == pytest.approx(float(SZ["max_notional_pct_of_capital"]) * capital, abs=100)


# --- the real plan builder ----------------------------------------------------

@pytest.fixture
def build(monkeypatch):
    """Run PlanOrchestrator._assemble_plan_dict for one up_spike short at Rs2L, with the
    band store, MIS lookup and results tagger stubbed: circuit 9.69% away (GUJALKALI 11-05)."""
    class _Store:
        def circuit_distance_frac(self, symbol, session_date, pdc, entry, bias):
            return 0.0969

        def circuit_price(self, symbol, session_date, pdc, bias):
            return 109.69

    monkeypatch.setattr(pbm, "get_price_band_store", lambda cfg: _Store())
    monkeypatch.setattr(po, "get_mis_info", lambda s: {"mis_enabled": True, "mis_leverage": 5.0})

    def run(tag, tilt_enabled):
        cfg = json.loads(json.dumps(CFG))
        cfg["total_capital_inr"] = 200_000.0
        cfg["setups"]["up_spike_fade_short"]["results_size_tilt"]["enabled"] = tilt_enabled
        monkeypatch.setattr(po, "_ROOT_CFG_CACHE", cfg)
        monkeypatch.setattr(rc, "results_context", lambda symbol, session_date: tag)
        tp = SimpleNamespace(
            trade_id="t1", confidence=1.0, entry_price=100.0,
            risk_params=SimpleNamespace(hard_sl=109.0, risk_per_share=9.0, atr=0.7),
            exit_levels=SimpleNamespace(targets=[], time_exit=None))
        return po.PlanOrchestrator()._assemble_plan_dict(
            symbol="NSE:X", setup_type="up_spike_fade_short", bias="short", regime="chop",
            cap_segment="small_cap", trade_plan=tp, df5m=pd.DataFrame({"volume": [1000]}),
            current_price=100.0, indicators={}, atr_val=0.7, vol_z_val=0.0,
            levels={"PDC": 92.0}, session_date=date(2026, 11, 5))
    return run


def test_off_sizes_every_tag_the_same_and_keeps_the_record(build):
    plans = {tag: build(tag, False) for tag in sorted(rc.ALL_TAGS)}
    sizes = {(p["sizing"]["qty"], p["sizing"]["notional"]) for p in plans.values()}
    assert len(sizes) == 1, sizes
    for tag, p in plans.items():
        assert p["sizing"]["qty"] > 0 and p["sizing"]["size_clamped"] == "circuit"
        assert p["sizing"]["size_results_tilt"] == 1.0
        assert p["results_context"] == tag        # log_results_context is on for up_spike


@pytest.mark.parametrize("tag, ratio", [("results_preopen", 2.0), ("board_meeting_today", 2.0),
                                        ("none", 0.5), ("unknown", 0.5)])
def test_on_really_scales_a_circuit_bound_trade(build, tag, ratio):
    off, on = build(tag, False), build(tag, True)
    assert on["sizing"]["size_clamped"] == "circuit"
    assert on["sizing"]["notional"] == pytest.approx(ratio * off["sizing"]["notional"], abs=100.0)
    assert on["sizing"]["size_results_tilt"] == ratio
    assert on["results_context"] == tag


# --- startup check ------------------------------------------------------------

def test_startup_runs_the_tilt_check_before_the_live_only_part():
    src = inspect.getsource(po.set_runtime_trading_mode)
    assert src.index("_check_results_size_tilts(setups)") < src.index('if mode != "live":')


def test_startup_check_rejects_a_malformed_block():
    with pytest.raises(po.OrchestratorConfigError, match="results_size_tilt"):
        po._check_results_size_tilts({"x": {"enabled": True, "results_size_tilt": {"enabled": True}}})


def test_startup_check_treats_a_null_block_as_absent():
    po._check_results_size_tilts({"x": {"enabled": True, "results_size_tilt": None}})


def test_startup_check_reads_the_calendar_only_when_on(monkeypatch, cal, tmp_path):
    monkeypatch.setattr(rc, "CALENDAR_PATH", cal)
    po._check_results_size_tilts({"x": {"enabled": True, "results_size_tilt": _tilt()}})
    monkeypatch.setattr(rc, "CALENDAR_PATH", tmp_path / "missing.parquet")
    po._check_results_size_tilts({"x": {"enabled": True, "results_size_tilt": _tilt(enabled=False)}})
    po._check_results_size_tilts({"x": {"enabled": False, "results_size_tilt": _tilt()}})
    with pytest.raises(po.OrchestratorConfigError, match="calendar"):
        po._check_results_size_tilts({"x": {"enabled": True, "results_size_tilt": _tilt()}})


def test_tagging_failure_is_unknown_not_a_lost_entry(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("calendar exploded")
    monkeypatch.setattr(rc, "results_context", boom)
    assert po._results_context_tag("NSE:X", "2026-11-05") == "unknown"


# --- real config --------------------------------------------------------------

def test_real_config_blocks_validate():
    blocks = {n: r["results_size_tilt"] for n, r in CFG["setups"].items()
              if isinstance(r, dict) and "results_size_tilt" in r}
    assert "up_spike_fade_short" in blocks
    for name, block in blocks.items():
        validate_results_size_tilt(block, name)
    po._check_results_size_tilts(CFG["setups"])


def test_up_spike_tilt_values_are_the_pre_registered_ones():
    m = UP["results_size_tilt"]["multipliers"]
    assert {k: v for k, v in m.items() if not k.startswith("_")} == MULT
