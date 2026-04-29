"""Expiry-Pin Strike Reversal detector unit tests.

Covers the spec from
specs/2026-04-29-expiry_pin_strike_reversal-plan.md (§ Phase B1).

Mechanic:
  Fires only on F&O expiry sessions (NIFTY weekly Thursday pre-2025-09 /
  Tuesday post-2025-09; monthly = last weekly of month). On expiry day,
  after 13:30 IST, when NIFTY spot is ≥0.3% from the pin (highest aggregate
  CE+PE OI strike), and RSI(14) on the symbol's 5m chart shows decay through
  the 70/30 thresholds.

Tests cover:
  - happy-path SHORT: spot above pin, RSI overbought decay
  - happy-path LONG: spot below pin, RSI oversold decay
  - non-expiry day → no fire
  - before 13:30 on expiry day → no fire
  - spot too close to pin (≤0.3%) → no fire
  - symbol outside heavyweights universe → no fire
  - RSI without decay → no fire
  - first-trigger latch
  - directional side selection (parametric)
  - monthly-expiry path
  - plan emits hard_sl + tiered T1/T2 (LONG + SHORT)
  - wide_open bypasses RSI decay (mandatory gates STAY)
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import pandas as pd
import pytest

from structures.expiry_pin_strike_reversal_structure import (
    ExpiryPinStrikeReversalStructure,
    _compute_rsi,
)
from structures.data_models import MarketContext


def _cfg(**overrides):
    cfg = {
        "_setup_name": "expiry_pin_strike_reversal",
        "enabled": True,
        "active_window_start": "13:30",
        "active_window_end": "15:15",
        "min_spot_distance_to_pin_pct": 0.3,
        "pin_index": "NIFTY",
        "expiry_types": ["weekly", "monthly"],
        "nifty_heavyweights_csv": "assets/nifty_heavyweights.csv",
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "stop_atr_multiplier": 0.5,
        "t1_qty_pct": 0.5,
        "t1_target_frac": 0.5,
        "allowed_sides": ["long", "short"],
        "allowed_cap_segments": ["large_cap", "mid_cap", "small_cap"],
        "min_bars_required": 30,
        "entry_zone_pct": 0.10,
        "entry_zone_mode": "symmetric",
        "min_stop_distance_pct": 0.3,
    }
    cfg.update(overrides)
    return cfg


class _StubOILoader:
    """Test stub: hard-coded pin strike + expiry calendar.

    Records every (session_date, expiry) pair queried in `self.queries`
    so tests can assert which date the detector actually looked up.
    """

    def __init__(
        self,
        pin_strike: Optional[float] = 23100.0,
        is_expiry: bool = True,
        weekly_unavailable: bool = False,
        monthly_pin: Optional[float] = None,
    ):
        self._pin_strike = pin_strike
        self._is_expiry = is_expiry
        # weekly_unavailable raises ValueError (parquet exists, no weekly
        # contracts at this expiry mode). For real "parquet missing for this
        # date", a stub would raise FileNotFoundError instead — see the
        # date-walk-back behavior tested in test_walks_back_when_parquet_missing.
        self._weekly_unavailable = weekly_unavailable
        self._monthly_pin = monthly_pin
        self.queries = []

    def find_max_oi_strike(self, session_date, symbol="NIFTY", expiry="weekly", oi_root=None):
        self.queries.append((session_date, expiry))
        if expiry == "weekly":
            if self._weekly_unavailable:
                raise ValueError("no weekly contracts at this mode")
            if self._pin_strike is None:
                raise ValueError("no contracts")
            return float(self._pin_strike)
        if expiry == "monthly":
            if self._monthly_pin is None:
                raise ValueError("no monthly contracts")
            return float(self._monthly_pin)
        raise ValueError(f"unknown expiry {expiry}")

    def is_expiry_day(self, session_date) -> bool:
        return bool(self._is_expiry)


def _build_session_5m(
    session_date: date,
    current_time: str = "14:00:00",
    n_bars: int = 40,
    close: float = 1500.0,
    rsi_target_prior: Optional[float] = None,
    rsi_target_current: Optional[float] = None,
):
    """Build a 40-bar 5m OHLCV df ending at `current_time` on `session_date`.

    If rsi_target_(prior|current) are set, the close series is hand-crafted so
    Wilder-RSI(14) at the prior bar matches `rsi_target_prior` and at the last
    bar matches `rsi_target_current` (within ±1.5).
    """
    end_ts = pd.Timestamp(f"{session_date.isoformat()} {current_time}")
    idx = pd.date_range(end_ts - pd.Timedelta(minutes=5 * (n_bars - 1)),
                        periods=n_bars, freq="5min")

    if rsi_target_prior is None or rsi_target_current is None:
        # Default: flat-ish chop near `close`
        closes = pd.Series([close + (i - n_bars / 2) * 0.05 for i in range(n_bars)])
    else:
        # Generate a noisy trend so RSI(14) has both gains and losses to
        # average over (avg_loss=0 → RSI=NA, would break detection).
        # Pattern: uptrend-with-pullbacks for overbought, downtrend-with-rallies
        # for oversold. Then bisect-tune the last bar to land RSI on target.
        if rsi_target_prior > 50:
            # Net uptrend (+1.0 per bar) with 30% pullback bars (-0.3)
            closes = []
            cur = close - (n_bars - 1) * 0.7   # start lower so end ≈ close
            for i in range(n_bars):
                if i % 3 == 0 and i > 0:
                    cur -= 0.3   # pullback
                else:
                    cur += 1.0
                closes.append(cur)
            closes = pd.Series(closes)
            if rsi_target_current < rsi_target_prior:
                closes = _calibrate_rsi_decay(
                    closes, rsi_target_prior, rsi_target_current, "overbought",
                )
        else:
            # Net downtrend (-1.0 per bar) with 30% rally bars (+0.3)
            closes = []
            cur = close + (n_bars - 1) * 0.7
            for i in range(n_bars):
                if i % 3 == 0 and i > 0:
                    cur += 0.3   # rally
                else:
                    cur -= 1.0
                closes.append(cur)
            closes = pd.Series(closes)
            if rsi_target_current > rsi_target_prior:
                closes = _calibrate_rsi_decay(
                    closes, rsi_target_prior, rsi_target_current, "oversold",
                )

    rows = []
    for ts, c in zip(idx, closes):
        rows.append({
            "ts": ts,
            "open": float(c) - 0.05,
            "high": float(c) + 0.10,
            "low": float(c) - 0.10,
            "close": float(c),
            "volume": 10_000,
        })
    return pd.DataFrame(rows).set_index("ts")


def _calibrate_rsi_decay(closes: pd.Series, target_prior: float,
                         target_current: float, regime: str) -> pd.Series:
    """Tune the LAST close so RSI(14) ≈ target_current while RSI(14) at the
    prior bar ≈ target_prior. Bisect over the magnitude of the last move.
    """
    n = len(closes)
    base_prior_close = float(closes.iloc[-2])
    if regime == "overbought":
        # We want the last close to dip — search a dip magnitude ∈ [0, prior]
        lo, hi = 0.0, base_prior_close * 0.10
        for _ in range(40):
            mid = (lo + hi) / 2
            tweaked = closes.copy()
            tweaked.iloc[-1] = base_prior_close - mid
            rsi_ser = _compute_rsi(tweaked, 14).dropna()
            if len(rsi_ser) < 2:
                lo = mid
                continue
            cur_rsi = float(rsi_ser.iloc[-1])
            if cur_rsi > target_current:
                lo = mid   # need a bigger dip
            else:
                hi = mid
        tweaked = closes.copy()
        tweaked.iloc[-1] = base_prior_close - (lo + hi) / 2
        return tweaked
    else:   # oversold — last close should bounce up
        lo, hi = 0.0, base_prior_close * 0.10
        for _ in range(40):
            mid = (lo + hi) / 2
            tweaked = closes.copy()
            tweaked.iloc[-1] = base_prior_close + mid
            rsi_ser = _compute_rsi(tweaked, 14).dropna()
            if len(rsi_ser) < 2:
                lo = mid
                continue
            cur_rsi = float(rsi_ser.iloc[-1])
            if cur_rsi < target_current:
                lo = mid   # need a bigger bounce
            else:
                hi = mid
        tweaked = closes.copy()
        tweaked.iloc[-1] = base_prior_close + (lo + hi) / 2
        return tweaked


def _ctx(
    df,
    session_date: date,
    symbol: str = "NSE:HDFCBANK",
    cap_segment: str = "large_cap",
    nifty_spot: Optional[float] = 23200.0,
    atr: float = 5.0,
    extra_indicators: Optional[dict] = None,
):
    last_ts = df.index[-1]
    indicators = {"atr": atr}
    if nifty_spot is not None:
        indicators["nifty_spot"] = float(nifty_spot)
    if extra_indicators:
        indicators.update(extra_indicators)
    return MarketContext(
        symbol=symbol,
        current_price=float(df["close"].iloc[-1]),
        timestamp=last_ts,
        df_5m=df,
        session_date=datetime(session_date.year, session_date.month, session_date.day),
        cap_segment=cap_segment,
        regime="chop",
        indicators=indicators,
    )


# =============================================================================
# Happy-path tests
# =============================================================================

def test_fires_short_on_canonical_expiry_thursday_above_pin_with_rsi_decay():
    """SHORT: spot 23200 (0.43% above pin 23100), RSI 75 → 68 (overbought decay)."""
    sd = date(2024, 6, 6)   # Thursday — pre-2025-09 weekly NIFTY expiry
    det = ExpiryPinStrikeReversalStructure(
        _cfg(), oi_loader=_StubOILoader(pin_strike=23100.0, is_expiry=True),
    )
    df = _build_session_5m(
        sd, current_time="14:00:00", n_bars=40, close=1500.0,
        rsi_target_prior=75.0, rsi_target_current=68.0,
    )
    ctx = _ctx(df, sd, nifty_spot=23200.0)
    res = det.detect(ctx)
    assert res.structure_detected is True, f"expected fire: {res.rejection_reason}"
    assert res.events[0].side == "short"
    assert res.events[0].levels["pin_strike"] == 23100.0
    assert res.events[0].levels["nifty_spot"] == 23200.0


def test_fires_long_on_expiry_thursday_below_pin_with_rsi_decay():
    """LONG: spot 22900 (0.43% below pin 23000), RSI 28 → 32 (oversold decay)."""
    sd = date(2024, 6, 6)
    det = ExpiryPinStrikeReversalStructure(
        _cfg(), oi_loader=_StubOILoader(pin_strike=23000.0, is_expiry=True),
    )
    df = _build_session_5m(
        sd, current_time="14:00:00", n_bars=40, close=1500.0,
        rsi_target_prior=28.0, rsi_target_current=32.0,
    )
    ctx = _ctx(df, sd, nifty_spot=22900.0)
    res = det.detect(ctx)
    assert res.structure_detected is True, f"expected fire: {res.rejection_reason}"
    assert res.events[0].side == "long"


# =============================================================================
# Calendar / window guards
# =============================================================================

def test_does_not_fire_on_non_expiry_day():
    """Wednesday 2024-06-05 — is_expiry_day stubbed False."""
    sd = date(2024, 6, 5)
    det = ExpiryPinStrikeReversalStructure(
        _cfg(), oi_loader=_StubOILoader(pin_strike=23100.0, is_expiry=False),
    )
    df = _build_session_5m(
        sd, current_time="14:00:00", n_bars=40, close=1500.0,
        rsi_target_prior=75.0, rsi_target_current=68.0,
    )
    res = det.detect(_ctx(df, sd, nifty_spot=23200.0))
    assert res.structure_detected is False
    assert "expiry" in (res.rejection_reason or "").lower()


def test_does_not_fire_before_active_window_on_expiry_day():
    """current_time=12:30 on expiry Thursday — pre-13:30 active window."""
    sd = date(2024, 6, 6)
    det = ExpiryPinStrikeReversalStructure(
        _cfg(), oi_loader=_StubOILoader(pin_strike=23100.0, is_expiry=True),
    )
    df = _build_session_5m(
        sd, current_time="12:30:00", n_bars=40, close=1500.0,
        rsi_target_prior=75.0, rsi_target_current=68.0,
    )
    res = det.detect(_ctx(df, sd, nifty_spot=23200.0))
    assert res.structure_detected is False
    assert "window" in (res.rejection_reason or "").lower()


# =============================================================================
# Distance / universe guards
# =============================================================================

def test_does_not_fire_when_spot_too_close_to_pin():
    """spot=23110, pin=23100 → distance 0.043% < 0.3% min."""
    sd = date(2024, 6, 6)
    det = ExpiryPinStrikeReversalStructure(
        _cfg(), oi_loader=_StubOILoader(pin_strike=23100.0, is_expiry=True),
    )
    df = _build_session_5m(
        sd, current_time="14:00:00", n_bars=40, close=1500.0,
        rsi_target_prior=75.0, rsi_target_current=68.0,
    )
    res = det.detect(_ctx(df, sd, nifty_spot=23110.0))
    assert res.structure_detected is False
    assert "spot too close" in (res.rejection_reason or "").lower()


def test_does_not_fire_when_symbol_outside_heavyweights():
    sd = date(2024, 6, 6)
    det = ExpiryPinStrikeReversalStructure(
        _cfg(), oi_loader=_StubOILoader(pin_strike=23100.0, is_expiry=True),
    )
    df = _build_session_5m(
        sd, current_time="14:00:00", n_bars=40, close=1500.0,
        rsi_target_prior=75.0, rsi_target_current=68.0,
    )
    ctx = _ctx(df, sd, symbol="NSE:RANDOMSTOCK", nifty_spot=23200.0)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "heavyweight" in (res.rejection_reason or "").lower()


# =============================================================================
# RSI decay confirmation
# =============================================================================

def test_does_not_fire_when_rsi_still_overbought():
    """RSI prior=75, current=72 — both above 70 (no decay through threshold)."""
    sd = date(2024, 6, 6)
    det = ExpiryPinStrikeReversalStructure(
        _cfg(), oi_loader=_StubOILoader(pin_strike=23100.0, is_expiry=True),
    )
    df = _build_session_5m(
        sd, current_time="14:00:00", n_bars=40, close=1500.0,
        rsi_target_prior=75.0, rsi_target_current=72.0,
    )
    res = det.detect(_ctx(df, sd, nifty_spot=23200.0))
    assert res.structure_detected is False
    assert "rsi" in (res.rejection_reason or "").lower()


# =============================================================================
# First-trigger latch
# =============================================================================

def test_first_trigger_latch_prevents_same_day_double_fire():
    sd = date(2024, 6, 6)
    det = ExpiryPinStrikeReversalStructure(
        _cfg(), oi_loader=_StubOILoader(pin_strike=23100.0, is_expiry=True),
    )
    df = _build_session_5m(
        sd, current_time="14:00:00", n_bars=40, close=1500.0,
        rsi_target_prior=75.0, rsi_target_current=68.0,
    )
    res1 = det.detect(_ctx(df, sd, nifty_spot=23200.0))
    assert res1.structure_detected is True

    df2 = _build_session_5m(
        sd, current_time="14:30:00", n_bars=40, close=1500.0,
        rsi_target_prior=75.0, rsi_target_current=68.0,
    )
    res2 = det.detect(_ctx(df2, sd, nifty_spot=23200.0))
    assert res2.structure_detected is False
    assert "latch" in (res2.rejection_reason or "").lower()


# =============================================================================
# Directional side selection (parametric)
# =============================================================================

@pytest.mark.parametrize(
    "spot,pin,expected_side,prior,current",
    [
        (23200.0, 23100.0, "short", 75.0, 68.0),  # spot above → SHORT
        (22900.0, 23000.0, "long", 28.0, 32.0),    # spot below → LONG
    ],
)
def test_side_selection_above_pin_yields_short_below_pin_yields_long(
    spot, pin, expected_side, prior, current,
):
    sd = date(2024, 6, 6)
    det = ExpiryPinStrikeReversalStructure(
        _cfg(), oi_loader=_StubOILoader(pin_strike=pin, is_expiry=True),
    )
    df = _build_session_5m(
        sd, current_time="14:00:00", n_bars=40, close=1500.0,
        rsi_target_prior=prior, rsi_target_current=current,
    )
    res = det.detect(_ctx(df, sd, nifty_spot=spot))
    assert res.structure_detected is True, res.rejection_reason
    assert res.events[0].side == expected_side


# =============================================================================
# Monthly expiry path
# =============================================================================

def test_fires_on_monthly_expiry_when_weekly_absent():
    """Last Thursday of June 2024 — weekly snapshot raises, monthly returns 23100."""
    sd = date(2024, 6, 27)
    stub = _StubOILoader(
        pin_strike=None, is_expiry=True, weekly_unavailable=True, monthly_pin=23100.0,
    )
    det = ExpiryPinStrikeReversalStructure(_cfg(), oi_loader=stub)
    df = _build_session_5m(
        sd, current_time="14:00:00", n_bars=40, close=1500.0,
        rsi_target_prior=75.0, rsi_target_current=68.0,
    )
    res = det.detect(_ctx(df, sd, nifty_spot=23200.0))
    assert res.structure_detected is True, f"expected fire: {res.rejection_reason}"
    assert res.events[0].side == "short"
    assert res.events[0].levels["pin_strike"] == 23100.0


# =============================================================================
# D-1 lookup (live-trading parity)
# =============================================================================

def test_uses_prior_session_oi_not_same_session():
    """The detector must look up D-1's bhavcopy, NOT D's. Live traders don't
    have today's settlement OI mid-session (NSE publishes ~6PM after close)."""
    sd = date(2024, 6, 6)   # Thursday weekly expiry
    stub = _StubOILoader(pin_strike=23100.0, is_expiry=True)
    det = ExpiryPinStrikeReversalStructure(_cfg(), oi_loader=stub)
    df = _build_session_5m(
        sd, current_time="14:00:00", n_bars=40, close=1500.0,
        rsi_target_prior=75.0, rsi_target_current=68.0,
    )
    res = det.detect(_ctx(df, sd, nifty_spot=23200.0))
    assert res.structure_detected is True
    # First query should be D-1 (2024-06-05), NEVER session_date itself.
    queried_dates = {q[0] for q in stub.queries}
    assert sd not in queried_dates, (
        f"detector must not query same-day OI; queries={stub.queries}"
    )
    assert date(2024, 6, 5) in queried_dates, (
        f"detector must query D-1; queries={stub.queries}"
    )


def test_walks_back_when_parquet_missing():
    """If D-1's bhavcopy is missing (e.g., a holiday gap or backfill gap),
    detector walks back day-by-day up to MAX_LOOKBACK_DAYS=7."""
    sd = date(2024, 6, 10)   # Monday after a Thursday expiry

    # Stub: raise FileNotFoundError for the first 3 days back, succeed at D-4.
    class _GapStub:
        def __init__(self):
            self.queries = []
        def find_max_oi_strike(self, session_date, symbol="NIFTY", expiry="weekly", oi_root=None):
            self.queries.append((session_date, expiry))
            from datetime import timedelta
            offset_days = (sd - session_date).days
            if offset_days <= 3:
                raise FileNotFoundError(f"no parquet for {session_date}")
            return 23100.0
        def is_expiry_day(self, session_date):
            # Monday isn't an expiry day in real life, but stub yes for the
            # focused walk-back test.
            return True

    stub = _GapStub()
    det = ExpiryPinStrikeReversalStructure(_cfg(), oi_loader=stub)
    df = _build_session_5m(
        sd, current_time="14:00:00", n_bars=40, close=1500.0,
        rsi_target_prior=75.0, rsi_target_current=68.0,
    )
    res = det.detect(_ctx(df, sd, nifty_spot=23200.0))
    assert res.structure_detected is True, (
        f"detector should walk back to D-4: {res.rejection_reason}"
    )
    assert res.events[0].levels["pin_strike"] == 23100.0
    # Should have queried D-1, D-2, D-3 (all FileNotFound) then D-4 (success).
    queried_offsets = sorted({(sd - q[0]).days for q in stub.queries})
    assert queried_offsets[:4] == [1, 2, 3, 4], (
        f"expected D-1..D-4 queries, got offsets {queried_offsets}"
    )


def test_walk_back_caps_at_seven_days():
    """If no parquet within 7 days back, return None — no fire."""
    sd = date(2024, 6, 6)

    class _AllMissingStub:
        queries = []
        def find_max_oi_strike(self, session_date, symbol="NIFTY", expiry="weekly", oi_root=None):
            self.queries.append((session_date, expiry))
            raise FileNotFoundError(f"no parquet anywhere")
        def is_expiry_day(self, session_date):
            return True

    stub = _AllMissingStub()
    det = ExpiryPinStrikeReversalStructure(_cfg(), oi_loader=stub)
    df = _build_session_5m(
        sd, current_time="14:00:00", n_bars=40, close=1500.0,
        rsi_target_prior=75.0, rsi_target_current=68.0,
    )
    res = det.detect(_ctx(df, sd, nifty_spot=23200.0))
    assert res.structure_detected is False
    assert "pin strike unavailable" in (res.rejection_reason or "").lower()
    # Should have stopped at D-7, not gone deeper.
    queried_offsets = sorted({(sd - q[0]).days for q in stub.queries})
    assert max(queried_offsets) == 7, (
        f"expected walk-back to cap at 7 days, got max offset {max(queried_offsets)}"
    )


# =============================================================================
# Plan emission
# =============================================================================

def test_plan_emits_hard_sl_and_tiered_t1_t2_for_short():
    """SHORT plan: hard_sl > entry, T1 < entry, T2 ≤ T1, qty_pct splits."""
    sd = date(2024, 6, 6)
    det = ExpiryPinStrikeReversalStructure(
        _cfg(), oi_loader=_StubOILoader(pin_strike=23100.0, is_expiry=True),
    )
    df = _build_session_5m(
        sd, current_time="14:00:00", n_bars=40, close=1500.0,
        rsi_target_prior=75.0, rsi_target_current=68.0,
    )
    plan = det.plan_short_strategy(_ctx(df, sd, nifty_spot=23200.0))
    assert plan is not None
    assert plan.side == "short"
    entry = plan.entry_price
    assert plan.risk_params.hard_sl > entry, "short stop must be above entry"
    targets = plan.exit_levels.targets
    assert len(targets) == 2
    assert targets[0]["name"] == "T1" and targets[0]["qty_pct"] == 0.5
    assert targets[1]["name"] == "T2" and targets[1]["qty_pct"] == 0.5
    assert targets[0]["level"] < entry, "T1 must be below entry on short"
    assert targets[1]["level"] <= targets[0]["level"], "T2 must be further than T1"


def test_plan_emits_hard_sl_and_tiered_t1_t2_for_long():
    """LONG mirror."""
    sd = date(2024, 6, 6)
    det = ExpiryPinStrikeReversalStructure(
        _cfg(), oi_loader=_StubOILoader(pin_strike=23000.0, is_expiry=True),
    )
    df = _build_session_5m(
        sd, current_time="14:00:00", n_bars=40, close=1500.0,
        rsi_target_prior=28.0, rsi_target_current=32.0,
    )
    plan = det.plan_long_strategy(_ctx(df, sd, nifty_spot=22900.0))
    assert plan is not None
    assert plan.side == "long"
    entry = plan.entry_price
    assert plan.risk_params.hard_sl < entry, "long stop must be below entry"
    targets = plan.exit_levels.targets
    assert targets[0]["level"] > entry, "T1 must be above entry on long"
    assert targets[1]["level"] >= targets[0]["level"], "T2 must be further than T1"


# =============================================================================
# Wide-open mode bypass — RSI decay only
# =============================================================================

def test_wide_open_bypasses_rsi_decay_filter(monkeypatch):
    """Under wide_open: RSI never crossed thresholds (neutral) → still fires.
    Mandatory gates (expiry-day, active window, min distance, heavyweight)
    STAY enforced."""
    import structures.expiry_pin_strike_reversal_structure as mod
    monkeypatch.setattr(mod, "_is_wide_open", lambda: True)
    sd = date(2024, 6, 6)
    det = ExpiryPinStrikeReversalStructure(
        _cfg(), oi_loader=_StubOILoader(pin_strike=23100.0, is_expiry=True),
    )
    df = _build_session_5m(
        sd, current_time="14:00:00", n_bars=40, close=1500.0,
        rsi_target_prior=55.0, rsi_target_current=50.0,   # neutral RSI, no decay
    )
    res = det.detect(_ctx(df, sd, nifty_spot=23200.0))
    assert res.structure_detected is True, (
        f"wide_open should bypass RSI decay: {res.rejection_reason}"
    )


def test_wide_open_does_NOT_bypass_expiry_day_check(monkeypatch):
    """Wide_open must NOT bypass the expiry-day calendar gate — the gamma pin
    effect doesn't exist on non-expiry days, so the thesis evaporates."""
    import structures.expiry_pin_strike_reversal_structure as mod
    monkeypatch.setattr(mod, "_is_wide_open", lambda: True)
    sd = date(2024, 6, 5)   # Wednesday
    det = ExpiryPinStrikeReversalStructure(
        _cfg(), oi_loader=_StubOILoader(pin_strike=23100.0, is_expiry=False),
    )
    df = _build_session_5m(
        sd, current_time="14:00:00", n_bars=40, close=1500.0,
        rsi_target_prior=75.0, rsi_target_current=68.0,
    )
    res = det.detect(_ctx(df, sd, nifty_spot=23200.0))
    assert res.structure_detected is False
    assert "expiry" in (res.rejection_reason or "").lower()


def test_wide_open_does_NOT_bypass_min_distance(monkeypatch):
    """Wide_open must NOT bypass the min spot distance — too close to pin
    means the magnet-pull thesis has nowhere to play out."""
    import structures.expiry_pin_strike_reversal_structure as mod
    monkeypatch.setattr(mod, "_is_wide_open", lambda: True)
    sd = date(2024, 6, 6)
    det = ExpiryPinStrikeReversalStructure(
        _cfg(), oi_loader=_StubOILoader(pin_strike=23100.0, is_expiry=True),
    )
    df = _build_session_5m(
        sd, current_time="14:00:00", n_bars=40, close=1500.0,
        rsi_target_prior=75.0, rsi_target_current=68.0,
    )
    res = det.detect(_ctx(df, sd, nifty_spot=23110.0))   # 0.043% from pin
    assert res.structure_detected is False
    assert "spot too close" in (res.rejection_reason or "").lower()
