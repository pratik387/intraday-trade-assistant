"""Tests for services.cross_sectional_ranker.CrossSectionalRanker.

Deterministic synthetic panels: verify the ranker selects exactly the names
that satisfy ALL cell conditions (bottom loser_pct x adv_tier x turnover-shock),
discriminates each filter, honors CA exclusion, and fails fast on missing config.
"""
from datetime import date, timedelta
import pandas as pd
import pytest

from services.cross_sectional_ranker import CrossSectionalRanker


def _cfg(**over):
    base = {
        "selection_mode": "trailing_loser_decile",
        "lookback_days": 5, "loser_pct": 0.25, "adv_tier": 1, "adv_tier_count": 5,
        "turnover_shock_min": 2.0, "shock_lookback_days": 20, "adv_floor_inr": 2_000_000,
        "min_price": 5.0, "min_universe_symbols_per_day": 20, "hold_days": 2,
        "exclude_ca_in_hold_window": True,
    }
    base.update(over)
    return base


def _cfg_low(**over):
    """Config for the near_period_low selection mode (C1 low52)."""
    base = {
        "selection_mode": "near_period_low",
        "low_lookback_days": 20, "dist_low_max": 0.02,
        "adv_tier": 1, "adv_tier_count": 5,
        "turnover_shock_min": 2.0, "shock_lookback_days": 20, "adv_floor_inr": 2_000_000,
        "min_price": 5.0, "min_universe_symbols_per_day": 20, "hold_days": 2,
        "exclude_ca_in_hold_window": True,
    }
    base.update(over)
    return base


def _biz_days(end: date, n: int):
    out = []
    d = end
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return sorted(out)


def _panel(session_date: date):
    """28 symbols x 25 business days. Controlled losers vs fillers."""
    days = _biz_days(session_date, 25)
    rows = []
    # 25 filler symbols: flat price 100, varied volume to spread ADV tiers, no loss/shock
    for i in range(25):
        vol = 30000 + i * 8000           # spreads ADV so qcut has 5 tiers
        for d in days:
            rows.append((d, f"FILL{i:02d}", 100.0, 101.0, 99.0, 100.0, vol))
    # helper to add a "loser" symbol with a 5-day decline + optional day-t volume spike
    def loser(sym, base_vol, shock, deepest):
        drop = 0.12 if deepest else 0.10  # deepest loser ranks lowest
        for j, d in enumerate(days):
            if j >= len(days) - 5:
                frac = (j - (len(days) - 6)) / 5.0
                close = 100.0 * (1 - drop * frac)
            else:
                close = 100.0
            v = base_vol
            if shock and d == session_date:
                v = base_vol * 3          # day-t turnover spike -> tshock ~3
            rows.append((d, sym, close, close * 1.01, close * 0.99, close, v))
    loser("LOSER1", 25000, shock=True, deepest=True)     # tier1 + shock + deepest -> SELECT
    loser("LOSER_NOSHOCK", 25000, shock=False, deepest=False)  # tier1, no shock -> reject
    loser("LOSER_HIGHADV", 400000, shock=True, deepest=False)  # shock but high ADV (tier5) -> reject
    df = pd.DataFrame(rows, columns=["date", "symbol", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    return df


SD = date(2025, 6, 16)  # a Monday
MTF = {f"FILL{i:02d}" for i in range(25)} | {"LOSER1", "LOSER_NOSHOCK", "LOSER_HIGHADV"}


def test_selects_loser_with_shock_in_tier1():
    r = CrossSectionalRanker(_cfg())
    basket = r.rank(_panel(SD), SD, MTF)
    syms = {b["symbol"] for b in basket}
    assert "LOSER1" in syms
    # every selected name satisfies ALL cell conditions
    for b in basket:
        assert b["rank_pct"] <= 0.25
        assert b["adv_tier"] == 1
        assert b["tshock"] >= 2.0


def test_rejects_loser_without_shock():
    r = CrossSectionalRanker(_cfg())
    syms = {b["symbol"] for b in r.rank(_panel(SD), SD, MTF)}
    assert "LOSER_NOSHOCK" not in syms


def test_rejects_shock_loser_in_wrong_adv_tier():
    r = CrossSectionalRanker(_cfg())
    syms = {b["symbol"] for b in r.rank(_panel(SD), SD, MTF)}
    assert "LOSER_HIGHADV" not in syms


def test_non_mtf_eligible_excluded():
    r = CrossSectionalRanker(_cfg())
    mtf_minus = MTF - {"LOSER1"}
    syms = {b["symbol"] for b in r.rank(_panel(SD), SD, mtf_minus)}
    assert "LOSER1" not in syms


def test_thin_universe_returns_empty():
    r = CrossSectionalRanker(_cfg(min_universe_symbols_per_day=40))
    assert r.rank(_panel(SD), SD, MTF) == []


def test_ca_in_hold_window_excludes():
    r = CrossSectionalRanker(_cfg())
    ca = {"LOSER1": [SD + timedelta(days=1)]}  # ex-date inside the 2-day hold window
    syms = {b["symbol"] for b in r.rank(_panel(SD), SD, MTF, ca_ex_dates=ca)}
    assert "LOSER1" not in syms


def test_config_fail_fast_on_missing_key():
    bad = _cfg()
    del bad["turnover_shock_min"]
    with pytest.raises(KeyError):
        CrossSectionalRanker(bad)


# --- near_period_low selection mode (C1 low52) ---

def test_low_mode_selects_near_low_with_shock_in_tier1():
    r = CrossSectionalRanker(_cfg_low())
    basket = r.rank(_panel(SD), SD, MTF)
    syms = {b["symbol"] for b in basket}
    assert "LOSER1" in syms  # near its 20d low + tier1 + shock
    for b in basket:
        assert b["trail_ret"] <= 0.02  # signal = dist-from-low within threshold
        assert b["adv_tier"] == 1
        assert b["tshock"] >= 2.0


def test_low_mode_rejects_no_shock():
    r = CrossSectionalRanker(_cfg_low())
    syms = {b["symbol"] for b in r.rank(_panel(SD), SD, MTF)}
    assert "LOSER_NOSHOCK" not in syms  # near low but no turnover shock


def test_low_mode_rejects_wrong_adv_tier():
    r = CrossSectionalRanker(_cfg_low())
    syms = {b["symbol"] for b in r.rank(_panel(SD), SD, MTF)}
    assert "LOSER_HIGHADV" not in syms  # near low + shock but tier-5


def test_invalid_selection_mode_raises():
    with pytest.raises(ValueError):
        CrossSectionalRanker(_cfg(selection_mode="momentum_decile"))


def test_low_mode_fail_fast_on_missing_key():
    bad = _cfg_low()
    del bad["dist_low_max"]
    with pytest.raises(KeyError):
        CrossSectionalRanker(bad)
