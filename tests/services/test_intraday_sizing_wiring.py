"""Config + wiring invariants for intraday sizing.

The unit behaviour lives in tests/services/risk/test_intraday_sizing.py. What
this file pins is that the REAL config and the REAL call path can't drift back
into the two defects measured on 2026-08-13:

  1. a setup with no `sizing_mode` inheriting stop-distance sizing by accident
     (or_window_failure_fade_short -> Rs112,822 median notional), and
  2. two sizing paths disagreeing (orchestrator sized the plan, the executor
     then silently re-sized notional-mode setups).
"""
import inspect
import io
import json
from pathlib import Path

import pytest

from services.risk.intraday_sizing import VALID_MODES

ROOT = Path(__file__).resolve().parents[2]
CFG = json.load(io.open(ROOT / "config" / "configuration.json", encoding="utf-8"))
SETUPS = CFG.get("setups") or {}
INTRADAY = {n: r for n, r in SETUPS.items() if str(r.get("horizon")) != "multi_day"}


def test_there_are_intraday_setups_to_check():
    assert INTRADAY, "fixture is vacuous — no intraday setups found"


@pytest.mark.parametrize("name", sorted(INTRADAY))
def test_every_intraday_setup_declares_a_valid_sizing_mode(name):
    """Including disabled ones: re-enabling must never resurrect the fall-through."""
    mode = INTRADAY[name].get("sizing_mode")
    assert mode is not None, f"{name} has no sizing_mode — it would size by accident"
    assert mode in VALID_MODES, f"{name} sizing_mode={mode!r} not in {VALID_MODES}"


@pytest.mark.parametrize("name", sorted(INTRADAY))
def test_notional_mode_setups_declare_their_target(name):
    if INTRADAY[name].get("sizing_mode") == "notional":
        tnp = INTRADAY[name].get("target_notional_pct")
        assert tnp and float(tnp) > 0, f"{name}: notional mode without target_notional_pct"


def test_intraday_sizing_block_is_present_and_fractional():
    s = CFG.get("intraday_sizing")
    assert s, "intraday_sizing missing from configuration.json"
    for k in ("vol_risk_budget_pct_of_capital", "stop_risk_budget_pct_of_capital",
              "min_notional_pct_of_capital", "max_notional_pct_of_capital"):
        assert k in s, f"intraday_sizing.{k} missing"
        assert 0 < float(s[k]) < 1, f"{k} must be a fraction of capital, got {s[k]}"


def test_the_two_risk_budgets_are_distinct_quantities():
    """vol budget = rupee 1-SD per ATR bar (~Rs100 on Rs5L); stop budget =
    rupee loss at the stop (~Rs1,000). Sharing one silently mis-sizes 10x."""
    s = CFG["intraday_sizing"]
    vol = float(s["vol_risk_budget_pct_of_capital"])
    stop = float(s["stop_risk_budget_pct_of_capital"])
    assert vol != stop, "the two budgets collapsed into one number"
    assert vol < stop, "a per-bar 1-SD move must be smaller than the loss at the stop"


def test_vol_target_is_not_active_until_at_signal_sigma_is_measured():
    """vol_target is implemented and unit-tested, but no setup may USE it until
    sigma at SIGNAL time is measured. The unconditional ATR distribution is the
    wrong population: all bars median 0.335%, opening hour 0.652%, yet the
    first real signal through the path sized on sigma 3.17%. Enabling it on
    that basis would be fitting a number to make the mechanism look right."""
    using = [n for n, r in INTRADAY.items() if r.get("sizing_mode") == "vol_target"]
    assert not using, (
        f"{using} use vol_target — calibrate from SIZING_OBS over a real run first")


def test_sizing_observations_are_logged_for_calibration():
    """SIZING_OBS on every plan is how the at-signal distribution gets collected."""
    import services.plan_orchestrator as po
    assert "SIZING_OBS" in inspect.getsource(po)


def test_min_notional_matches_the_capacity_floor():
    """If capital_management's floor is HIGHER, vol-targeted positions in
    volatile names get sized then shadowed into non-existence."""
    assert float(CFG["capital_management"]["min_notional_pct"]) <=         float(CFG["intraday_sizing"]["min_notional_pct_of_capital"])


def test_the_notional_clamp_is_ordered():
    s = CFG["intraday_sizing"]
    assert float(s["min_notional_pct_of_capital"]) < float(s["max_notional_pct_of_capital"])


def test_clamp_would_have_caught_the_or_window_accident():
    """Rs112,822 on the Rs5L paper book must be impossible under the new cap."""
    cap = 500_000.0
    max_notional = float(CFG["intraday_sizing"]["max_notional_pct_of_capital"]) * cap
    assert 112_822 > max_notional, "cap is too loose to prevent the measured accident"


def test_orchestrator_requires_runtime_capital_and_exposes_the_setter():
    import services.plan_orchestrator as po
    assert hasattr(po, "set_runtime_capital")
    src = inspect.getsource(po)
    assert "total_capital_inr" in src
    assert "intraday_sizing" in src


def test_set_runtime_capital_injects_into_the_config_the_orchestrator_reads():
    import services.plan_orchestrator as po
    po.set_runtime_capital(123456.0)
    assert po._load_root_config()["total_capital_inr"] == 123456.0


def test_main_calls_the_seam_before_planning():
    src = io.open(ROOT / "main.py", encoding="utf-8").read()
    assert "set_runtime_capital(capital_manager.total_capital)" in src, \
        "main.py must publish the run's capital or every plan raises"


def test_executor_no_longer_contains_a_second_sizing_path():
    """Comments may still explain the removal; CODE must not re-size."""
    from services.execution import trigger_aware_executor as tae
    code = []
    for ln in inspect.getsource(tae).splitlines():
        stripped = ln.strip()
        if stripped.startswith("#"):
            continue
        code.append(ln.split("#", 1)[0])
    code = "\n".join(code)
    assert "NOTIONAL_SIZE" not in code, "executor still re-sizes — two sizing paths again"
    assert "target_notional_pct" not in code, \
        "executor still reads target_notional_pct; sizing belongs to the orchestrator"
