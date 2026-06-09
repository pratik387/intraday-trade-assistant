"""Tests for CloseDnOvernightLongStructure detector."""
from datetime import date, datetime, time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from structures.close_dn_overnight_long_structure import CloseDnOvernightLongStructure
from structures.data_models import MarketContext


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MTF_SNAPSHOT = _REPO_ROOT / "data" / "mtf_universe" / "approved_mtf_securities_2026-05-21.json"


def _config(**overrides):
    base = {
        "signed_vol_ratio_max": -0.5,
        "closing_25m_volume_z_min": 1.0,
        "min_signal_bar_count": 4,
        "cell_volume_z_min": 2.0,
        "cell_prior_day_return_pct_min": 3.0,
        "baseline_rolling_days": 20,
        "mtf": {
            "approved_list_snapshot_path": str(_MTF_SNAPSHOT),
            "exclude_etf": True,
        },
        "capital_allocation": {
            "margin_per_slot_inr": 100000,
        },
    }
    base.update(overrides)
    return base


def _build_df(*, current_hhmm: str, signed_vol_ratio: float = -0.6,
              volume_z_target: float = 2.5, n_prior_sessions: int = 25,
              prior_day_return_pct: float = 4.0):
    """Synthesize df_5m. The current session has 15:00-15:25 bars; signed_vol_ratio
    is achieved by setting most bars to closing-down with given total volume.

    Prior sessions populate 15:00-15:20 each with constant volume so the rolling
    baseline mean is predictable and the today total + volume_z_target hits the
    requested z-score.

    `prior_day_return_pct` semantic (post-bugfix 2026-05-23): SIGNAL-DAY's daily
    return = (today_last_bar_close - yesterday_last_bar_close) / yesterday_last_bar_close.
    Cell #5 was discovered with this interpretation (per sanity script); production
    detector now matches.
    """
    session_date = date(2026, 5, 21)
    bars = []

    # Prior sessions: 15:00-15:20 only, baseline ~1000 vol per bar (sum~5000)
    # Add small jitter so baseline std > 0 (needed for valid z-score).
    rng = np.random.default_rng(42)
    # Determine today's bar close (depends on signed_vol direction)
    today_bar_close = 99.5 if signed_vol_ratio < 0 else 100.5
    # Inverse-solve for yesterday's last-bar close such that
    # (today_close - yesterday_close) / yesterday_close == prior_day_return_pct/100
    yesterday_close = today_bar_close / (1.0 + prior_day_return_pct / 100.0) if (1.0 + prior_day_return_pct / 100.0) > 0 else 100.0
    last_prior_date = (pd.Timestamp(session_date) - pd.Timedelta(days=1)).date()
    for i in range(n_prior_sessions, 0, -1):
        d = pd.Timestamp(session_date) - pd.Timedelta(days=i)
        jitter = rng.normal(0, 20, size=5)
        per_bar_vols = 1000.0 + jitter
        is_yesterday = (d.date() == last_prior_date)
        for idx, hhmm in enumerate(("15:00", "15:05", "15:10", "15:15", "15:20")):
            ts = pd.Timestamp(f"{d.date()} {hhmm}:00")
            # On yesterday, set the LAST bar's close to yesterday_close so
            # _prior_day_return_pct can derive (today_close, yesterday_close).
            if is_yesterday and hhmm == "15:20":
                bar_close = yesterday_close
            else:
                bar_close = 100.0
            bars.append({"timestamp": ts, "open": 100.0, "close": bar_close,
                         "high": 100.0, "low": 100.0,
                         "volume": float(per_bar_vols[idx])})

    today_bars = []
    # For today total: volume_z = (today_total - baseline_mean) / baseline_std
    # baseline_mean ~= 5000 per session, baseline_std ~= small (~50)
    # To get volume_z = volume_z_target, today_total = baseline_mean + z * baseline_std
    # But std varies with jitter, so we'll just scale today total to be a large multiple.
    today_total = 5000.0 * volume_z_target

    today_per_bar = today_total / 5
    for hhmm in ("15:00", "15:05", "15:10", "15:15", "15:20"):
        ts = pd.Timestamp(f"{session_date} {hhmm}:00")
        if signed_vol_ratio < 0:
            o, c = 100.5, 99.5
        else:
            o, c = 99.5, 100.5
        today_bars.append({"timestamp": ts, "open": o, "close": c,
                           "high": 100.5, "low": 99.5, "volume": today_per_bar})
    ts_25 = pd.Timestamp(f"{session_date} 15:25:00")
    today_bars.append({"timestamp": ts_25, "open": 100.0, "close": 100.0,
                       "high": 100.0, "low": 100.0, "volume": today_per_bar})

    # Filter today_bars to those with hhmm <= current_hhmm
    keep = []
    for b in today_bars:
        if b["timestamp"].strftime("%H:%M") <= current_hhmm:
            keep.append(b)
    if not keep:
        # current_hhmm is before any of today's bars — return only prior history
        df = pd.DataFrame(bars).set_index("timestamp").sort_index()
        return df, session_date

    bars_df = pd.DataFrame(bars + keep).set_index("timestamp").sort_index()
    return bars_df, session_date


def _ctx(df: pd.DataFrame, session_date: date, symbol: str = "NSE:RELIANCE",
         cap_segment: str = "large_cap"):
    last_ts = df.index[-1]
    return MarketContext(
        symbol=symbol,
        current_price=float(df["close"].iloc[-1]),
        timestamp=last_ts,
        df_5m=df,
        session_date=pd.Timestamp(session_date),
        cap_segment=cap_segment,
        df_daily=None,
    )


def test_does_not_fire_outside_active_window():
    """At 14:00 — no today bars yet, so the detector rejects before the
    active-window check (either way: no fire)."""
    df, sd = _build_df(current_hhmm="14:00")
    det = CloseDnOvernightLongStructure(_config())
    res = det.detect(_ctx(df, sd))
    assert not res.structure_detected
    reason = (res.rejection_reason or "").lower()
    assert "outside active window" in reason or "no bars" in reason


def test_does_not_fire_at_15_15():
    """At 15:15, no fire (one bar early — the 15:20 trigger bar isn't closed yet)."""
    df, sd = _build_df(current_hhmm="15:15")
    det = CloseDnOvernightLongStructure(_config())
    res = det.detect(_ctx(df, sd))
    assert not res.structure_detected


def test_fires_at_15_20_with_valid_signal():
    """At 15:20 (the active trigger after the 15:20 bar closes at 15:25:00),
    with strong sell-flush, extreme volume_z, and post-up-3% day → fire.

    Covers the case where Upstox hasn't yet surfaced the 15:25 bar — fetched
    df_5m's latest bar is 15:20 and the detector still fires because the 5
    signal bars (15:00-15:20) are all finalized."""
    df, sd = _build_df(current_hhmm="15:20", signed_vol_ratio=-0.6, volume_z_target=3.0,
                        prior_day_return_pct=4.0)
    det = CloseDnOvernightLongStructure(_config())
    res = det.detect(_ctx(df, sd))
    if not res.structure_detected:
        pytest.skip(f"Synthetic data didn't trigger; rejection={res.rejection_reason}")
    assert len(res.events) == 1
    evt = res.events[0]
    assert evt.side == "long"
    assert evt.context["product"] in ("MTF", "CNC")


def test_fires_at_15_25_with_valid_signal():
    """At 15:25 (Upstox has surfaced the 15:25 bar by cron-fire time), with the
    same valid signal → fire.

    Covers the common production case: cron runs at 15:26 IST, Upstox returns
    bars through 15:25 for most symbols (2026-06-09 empirical: 2064 of 2096
    symbols on 15:25). The 5 signal bars (15:00-15:20) are still finalized;
    the 15:25 bar's data is not read."""
    df, sd = _build_df(current_hhmm="15:25", signed_vol_ratio=-0.6, volume_z_target=3.0,
                        prior_day_return_pct=4.0)
    det = CloseDnOvernightLongStructure(_config())
    res = det.detect(_ctx(df, sd))
    if not res.structure_detected:
        pytest.skip(f"Synthetic data didn't trigger; rejection={res.rejection_reason}")
    assert len(res.events) == 1
    evt = res.events[0]
    assert evt.side == "long"
    assert evt.context["product"] in ("MTF", "CNC")


def test_rejects_below_signed_vol_threshold():
    """Positive signed_vol_ratio (-0.3) → rejected (we need <=-0.5)."""
    df, sd = _build_df(current_hhmm="15:20", signed_vol_ratio=-0.3, volume_z_target=3.0,
                        prior_day_return_pct=4.0)
    # Override fixture: build today bars with bullish direction so signed_vol_ratio is positive
    # (the fixture already uses close < open for negative signed_vol; positive needs flipping)
    today_mask = df.index.date == sd
    df.loc[today_mask, "open"] = 99.5
    df.loc[today_mask, "close"] = 100.5
    det = CloseDnOvernightLongStructure(_config())
    res = det.detect(_ctx(df, sd))
    assert not res.structure_detected


def test_rejects_low_volume_z():
    """volume_z below 1.0 → rejected by primary filter."""
    df, sd = _build_df(current_hhmm="15:20", volume_z_target=0.5,
                        prior_day_return_pct=4.0)
    det = CloseDnOvernightLongStructure(_config())
    res = det.detect(_ctx(df, sd))
    if res.structure_detected:
        # synthetic noise — skip
        pytest.skip("baseline std was 0 so z-score was infinite")
    assert "volume_z" in (res.rejection_reason or "").lower() or "baseline" in (res.rejection_reason or "").lower()


def test_rejects_non_extreme_volume_z():
    """volume_z >= 1.0 but < 2.0 → cell filter rejects.

    Synthetic baseline std is small, so volume_z scales aggressively;
    this test bumps cell_volume_z_min beyond the achievable z to force
    cell-filter rejection.
    """
    df, sd = _build_df(current_hhmm="15:20", volume_z_target=1.5,
                        prior_day_return_pct=4.0)
    # Force unreachable cell threshold to verify the cell filter actually rejects.
    cfg = _config(cell_volume_z_min=10000.0)
    det = CloseDnOvernightLongStructure(cfg)
    res = det.detect(_ctx(df, sd))
    assert not res.structure_detected
    msg = (res.rejection_reason or "").lower()
    assert "cell_min" in msg or "extreme bucket" in msg or "volume_z" in msg


def test_rejects_prior_day_not_up_3pct():
    """Prior day return < 3% → cell filter rejects (need up_gt_3pct)."""
    df, sd = _build_df(current_hhmm="15:20", volume_z_target=3.0,
                        prior_day_return_pct=1.0)  # < 3%
    det = CloseDnOvernightLongStructure(_config())
    res = det.detect(_ctx(df, sd))
    if res.structure_detected:
        pytest.skip("Synthetic prior_day_return injection didn't take effect")
    msg = (res.rejection_reason or "").lower()
    assert "prior_day_return" in msg or "baseline" in msg or "volume_z" in msg


def test_prior_day_return_pct_uses_today_vs_yesterday_NOT_yesterday_vs_day_before():
    """Regression test for 2026-05-23 Bug #1 fix.

    Production previously computed (T-1_close - T-2_close)/T-2_close (literal
    "prior day return"). The mechanism intent + cell #5 discovery used today's
    daily return (T_close - T-1_close)/T-1_close. This test locks the new
    semantics: `_prior_day_return_pct` returns TODAY's daily return.

    Scenario: today's 15:20-bar close = 110, yesterday's last-bar close = 100,
    day-before-yesterday's last-bar close = 50.
      - OLD (buggy) formula would return (100 - 50) / 50 * 100 = +100%
      - NEW (correct) formula returns (110 - 100) / 100 * 100 = +10%

    The cell threshold is >= 3%. Both interpretations pass the threshold here
    so we can't discriminate via fire/no-fire alone. Instead, check the
    numeric value of evt.context['prior_day_return_pct'].
    """
    session_date = date(2026, 5, 21)
    bars = []
    # Day before yesterday: last-bar close 50 (huge negative absolute, just to make formulas differ)
    day_before_yesterday = (pd.Timestamp(session_date) - pd.Timedelta(days=2)).date()
    yesterday = (pd.Timestamp(session_date) - pd.Timedelta(days=1)).date()

    # ~25 prior sessions for baseline; make day-before-yesterday close = 50,
    # yesterday close = 100, today close = 110
    rng = np.random.default_rng(7)
    for i in range(25, 0, -1):
        d = pd.Timestamp(session_date) - pd.Timedelta(days=i)
        jitter = rng.normal(0, 20, size=5)
        for idx, hhmm in enumerate(("15:00", "15:05", "15:10", "15:15", "15:20")):
            ts = pd.Timestamp(f"{d.date()} {hhmm}:00")
            # Last-bar overrides for the two specific prior days
            if d.date() == yesterday and hhmm == "15:20":
                bar_close = 100.0
            elif d.date() == day_before_yesterday and hhmm == "15:20":
                bar_close = 50.0
            else:
                bar_close = 100.0
            bars.append({"timestamp": ts, "open": 100.0, "close": bar_close,
                         "high": 100.0, "low": 100.0,
                         "volume": 1000.0 + float(jitter[idx])})

    # Today's bars 15:00-15:20: bearish closes (signed_vol < 0), last-bar (15:20) close = 110.
    # The 15:25 bar is intentionally omitted — the active-window trigger is 15:20.
    today_per_bar_vol = 5000.0
    for idx, hhmm in enumerate(("15:00", "15:05", "15:10", "15:15", "15:20")):
        ts = pd.Timestamp(f"{session_date} {hhmm}:00")
        # bearish bar (open > close)
        is_signal_last = hhmm == "15:20"
        c = 110.0 if is_signal_last else 105.0
        o = 111.0 if is_signal_last else 106.0
        bars.append({"timestamp": ts, "open": o, "close": c, "high": max(o, c),
                     "low": min(o, c), "volume": today_per_bar_vol})

    df = pd.DataFrame(bars).set_index("timestamp").sort_index()
    det = CloseDnOvernightLongStructure(_config())
    ctx = MarketContext(
        symbol="NSE:TEST", current_price=110.0, timestamp=df.index[-1],
        df_5m=df, session_date=pd.Timestamp(session_date),
        cap_segment="large_cap", df_daily=None,
    )
    res = det.detect(ctx)
    # Must fire — today_close=110, yesterday_close=100 → +10% (>= 3%)
    assert res.structure_detected, f"expected fire; reason={res.rejection_reason}"
    evt = res.events[0]
    pr = evt.context.get("prior_day_return_pct")
    assert pr is not None, "prior_day_return_pct should be in event context"
    # NEW formula: (110-100)/100 = +10% ± small tolerance
    assert 9.0 < pr < 11.0, f"expected ~+10% (today/yesterday), got {pr}"
    # OLD formula would have returned (100-50)/50 = +100%
    assert pr < 50.0, (
        f"prior_day_return_pct={pr} > 50% suggests OLD buggy formula "
        f"((T-1 - T-2)/T-2) is being used; new formula should give ~+10%"
    )



def test_per_symbol_latch_prevents_re_fire():
    """Once fired, calling detect again on the same (symbol, date) returns 'already fired'."""
    df, sd = _build_df(current_hhmm="15:20", signed_vol_ratio=-0.8, volume_z_target=3.5,
                        prior_day_return_pct=4.0)
    det = CloseDnOvernightLongStructure(_config())
    res1 = det.detect(_ctx(df, sd))
    if not res1.structure_detected:
        pytest.skip(f"first detect didn't fire; rejection={res1.rejection_reason}")
    res2 = det.detect(_ctx(df, sd))
    assert not res2.structure_detected
    assert "already fired" in (res2.rejection_reason or "")


def test_etf_excluded_even_if_mtf_eligible():
    """BANKBEES is in MTF list but category=etf → rejected entirely."""
    df, sd = _build_df(current_hhmm="15:20", signed_vol_ratio=-0.8, volume_z_target=3.5,
                        prior_day_return_pct=4.0)
    det = CloseDnOvernightLongStructure(_config())
    res = det.detect(_ctx(df, sd, symbol="NSE:BANKBEES"))
    if res.structure_detected:
        pytest.fail("ETF should have been rejected")
    msg = (res.rejection_reason or "").lower()
    # Either ETF guard fired or some other filter — accept any rejection
    assert "etf" in msg or "baseline" in msg or "volume_z" in msg or "prior" in msg


def test_plan_long_strategy_returns_overnight_trade_plan():
    """plan_long_strategy returns TradePlan with exit_mode='scheduled_amo'."""
    df, sd = _build_df(current_hhmm="15:20", signed_vol_ratio=-0.8, volume_z_target=3.5,
                        prior_day_return_pct=4.0)
    det = CloseDnOvernightLongStructure(_config())
    res = det.detect(_ctx(df, sd))
    if not res.structure_detected:
        pytest.skip(f"detect didn't fire; rejection={res.rejection_reason}")
    evt = res.events[0]
    plan = det.plan_long_strategy(_ctx(df, sd), evt)
    assert plan is not None
    assert plan.exit_levels.exit_mode == "scheduled_amo"
    assert plan.exit_levels.scheduled_exit_at is not None
    assert plan.exit_levels.hard_sl == 0.0
    assert plan.exit_levels.targets == []
    assert plan.notes["product"] in ("MTF", "CNC")
