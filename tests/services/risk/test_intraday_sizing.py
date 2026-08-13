"""Intraday sizing: no fall-through, one clamp, ex-ante vol only."""
import pytest

from services.risk.intraday_sizing import (
    IntradaySizingResult, SizingConfigError, VALID_MODES,
    resolve_sizing_mode, sigma_pct_from_atr, size_intraday_position,
)

MIN_N, MAX_N = 20_000.0, 100_000.0


def _size(**kw):
    base = dict(setup_name="s", entry_price=100.0,
                min_notional_inr=MIN_N, max_notional_inr=MAX_N)
    base.update(kw)
    return size_intraday_position(**base)


# --- the fall-through defect -------------------------------------------------

def test_missing_sizing_mode_raises():
    """or_window_failure_fade_short had no sizing_mode and got Rs112k/trade."""
    with pytest.raises(SizingConfigError, match="sizing_mode is missing"):
        resolve_sizing_mode({}, "or_window_failure_fade_short")


def test_unknown_sizing_mode_raises():
    with pytest.raises(SizingConfigError, match="not one of"):
        resolve_sizing_mode({"sizing_mode": "whatever"}, "s")


@pytest.mark.parametrize("mode", VALID_MODES)
def test_valid_modes_resolve(mode):
    assert resolve_sizing_mode({"sizing_mode": mode}, "s") == mode


def test_size_rejects_unknown_mode():
    with pytest.raises(SizingConfigError):
        _size(sizing_mode="nonsense", vol_risk_budget_inr=1000.0)


# --- vol targeting equalises risk -------------------------------------------

def test_vol_target_sizes_inversely_to_volatility():
    """The whole point: equal rupee risk, not equal notional."""
    calm = _size(sizing_mode="vol_target", vol_risk_budget_inr=1000.0, atr=2.0)   # 2%
    wild = _size(sizing_mode="vol_target", vol_risk_budget_inr=1000.0, atr=6.0)   # 6%
    assert calm.notional_inr == pytest.approx(50_000, rel=1e-3)
    assert wild.notional_inr == pytest.approx(16_666, rel=1e-2) or wild.reason == "below_min_notional"
    # 3x the vol => 1/3 the notional => same risk contribution
    if wild.reason == "ok":
        assert calm.notional_inr == pytest.approx(3 * wild.notional_inr, rel=1e-2)


def test_vol_target_equalises_rupee_risk_while_unclamped():
    """Each position contributes the same rupee 1-SD move — the whole point."""
    out = [_size(sizing_mode="vol_target", vol_risk_budget_inr=1000.0, atr=a, entry_price=200.0)
           for a in (4.0, 6.0, 8.0)]  # sigma 2%, 3%, 4% -> Rs50k, 33k, 25k: all in band
    assert [r.reason for r in out] == ["ok"] * 3
    assert [r.clamped for r in out] == [None] * 3
    for r in out:
        assert r.notional_inr * (r.sigma_pct / 100.0) == pytest.approx(1000.0, rel=1e-2)


def test_the_clamp_deliberately_breaks_equalisation_at_the_extremes():
    """A very calm name would need a position bigger than the cap allows, so it
    carries LESS than the target risk. That is the cap doing its job, not a bug."""
    calm = _size(sizing_mode="vol_target", vol_risk_budget_inr=1000.0, atr=1.0, entry_price=200.0)
    assert calm.clamped == "max"
    assert calm.notional_inr * (calm.sigma_pct / 100.0) < 1000.0


def test_vol_target_refuses_when_atr_missing():
    """No ex-ante vol => refuse, never silently fall back to a vol-blind size."""
    r = _size(sizing_mode="vol_target", vol_risk_budget_inr=1000.0, atr=None)
    assert r.qty == 0 and r.reason == "atr_missing"


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_vol_target_refuses_nonpositive_atr(bad):
    assert _size(sizing_mode="vol_target", vol_risk_budget_inr=1000.0, atr=bad).reason == "atr_missing"


def test_vol_target_requires_a_budget():
    with pytest.raises(SizingConfigError, match="vol_risk_budget_inr"):
        _size(sizing_mode="vol_target", atr=2.0)


# --- the clamp is what makes Rs112k impossible ------------------------------

def test_tight_stop_cannot_mint_a_giant_position():
    """The exact or_window failure: rps 0.10 on a Rs100 stock."""
    r = _size(sizing_mode="risk", stop_risk_budget_inr=1000.0, risk_per_share=0.10)
    assert r.notional_inr == MAX_N, "clamp did not bind"
    assert r.clamped == "max"


def test_every_mode_passes_through_the_same_clamp():
    huge = [
        _size(sizing_mode="vol_target", vol_risk_budget_inr=100_000.0, atr=1.0),
        _size(sizing_mode="notional", target_notional_pct=5.0, total_capital_inr=500_000.0),
        _size(sizing_mode="risk", stop_risk_budget_inr=100_000.0, risk_per_share=0.01),
    ]
    assert [r.notional_inr for r in huge] == [MAX_N] * 3
    assert [r.clamped for r in huge] == ["max"] * 3


def test_below_floor_is_reported_not_rounded_up():
    r = _size(sizing_mode="notional", target_notional_pct=0.001, total_capital_inr=500_000.0)
    assert r.qty == 0 and r.reason == "below_min_notional"


def test_inverted_clamp_config_raises():
    with pytest.raises(SizingConfigError, match="max_notional_inr"):
        _size(sizing_mode="notional", target_notional_pct=0.06,
              total_capital_inr=500_000.0, min_notional_inr=100.0, max_notional_inr=50.0)


# --- notional + risk modes ---------------------------------------------------

def test_notional_mode_matches_the_old_behaviour():
    """6% of Rs5L = Rs30k — the observed median, so this is a faithful port."""
    r = _size(sizing_mode="notional", target_notional_pct=0.06, total_capital_inr=500_000.0)
    assert r.notional_inr == pytest.approx(30_000, rel=1e-6)
    assert r.qty == 300


def test_notional_mode_requires_its_inputs():
    with pytest.raises(SizingConfigError, match="target_notional_pct"):
        _size(sizing_mode="notional", total_capital_inr=500_000.0)


def test_risk_mode_is_now_explicit_and_still_works():
    r = _size(sizing_mode="risk", stop_risk_budget_inr=1000.0, risk_per_share=2.0)
    assert r.qty == 500 and r.notional_inr == pytest.approx(50_000)


def test_risk_mode_with_no_stop_distance_is_zero_not_infinite():
    assert _size(sizing_mode="risk", stop_risk_budget_inr=1000.0, risk_per_share=0.0).qty == 0


# --- degenerate inputs -------------------------------------------------------

@pytest.mark.parametrize("px", [0.0, -5.0, None])
def test_bad_entry_price_sizes_zero(px):
    assert _size(sizing_mode="vol_target", vol_risk_budget_inr=1000.0, atr=2.0,
                 entry_price=px).qty == 0


def test_qty_never_exceeds_the_clamped_notional():
    r = _size(sizing_mode="vol_target", vol_risk_budget_inr=1000.0, atr=2.0, entry_price=333.0)
    assert r.qty * 333.0 <= MAX_N and r.notional_inr <= MAX_N


def test_sigma_helper_is_ex_ante_and_total():
    assert sigma_pct_from_atr(atr=3.0, price=150.0) == pytest.approx(2.0)
    for bad in (None, 0, -1):
        assert sigma_pct_from_atr(atr=bad, price=100.0) is None
        assert sigma_pct_from_atr(atr=1.0, price=bad) is None


def test_result_carries_the_mode_and_sigma_for_logging():
    r = _size(sizing_mode="vol_target", vol_risk_budget_inr=1000.0, atr=2.5)
    assert isinstance(r, IntradaySizingResult)
    assert r.mode == "vol_target" and r.sigma_pct == pytest.approx(2.5)
