"""Tests for MultiDayCompositeSelector ordering (Phase 4, 2026-08-12).

`composite` ordering promotes consensus names, but consensus does not predict
(121 deduped positions: pooled consensus -0.448% vs solo -0.049%, t=-0.52,
permutation p=0.69). Production therefore orders by a date-salted hash, keeping
composite computed-and-logged but decision-free.

Ordering is load-bearing now: with cluster caps, 40 of 121 historical positions
get dropped by a cap, so the order decides which trades the book actually gets.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.multiday_composite_selector import MultiDayCompositeSelector  # noqa: E402

BASE = {"max_new_per_day": 100, "max_concurrent": 200, "cap_score_clip": 3.0,
        "tiebreaker": "tshock", "slot_ranking_mode": "unbiased_hash"}


def _cfg(**over):
    c = dict(BASE); c.update(over); return c


def _cand(sym, cap, tshock=1.0, close=100.0):
    return {"symbol": f"NSE:{sym}", "cap_score": cap, "tshock": tshock,
            "close": close, "trail_ret": -0.1, "sigma20_pct": 3.0}


def _baskets(pairs, setup="s1"):
    return {setup: [_cand(s, c) for s, c in pairs]}


D = date(2026, 8, 12)


def test_unknown_mode_fails_at_construction():
    with pytest.raises(ValueError):
        MultiDayCompositeSelector(_cfg(slot_ranking_mode="magic"))


def test_missing_mode_key_fails_fast():
    c = dict(BASE); c.pop("slot_ranking_mode")
    with pytest.raises(KeyError):
        MultiDayCompositeSelector(c)


def test_hash_mode_requires_session_date():
    s = MultiDayCompositeSelector(_cfg())
    with pytest.raises(ValueError):
        s.select(_baskets([("A", 3.0)]), held_symbols=set(), weights={"s1": 1.0}, limit=5)


def test_hash_order_ignores_composite():
    """The point: a higher cap_score must NOT buy a better slot."""
    s = MultiDayCompositeSelector(_cfg())
    out = s.select(_baskets([("AAA", 0.1), ("BBB", 3.0), ("CCC", 1.5)]),
                   held_symbols=set(), weights={"s1": 1.0}, limit=3, session_date=D)
    import hashlib
    expected = sorted(["AAA", "BBB", "CCC"],
                      key=lambda b: hashlib.sha1(f"{D.isoformat()}|{b}".encode()).hexdigest())
    assert [r["bare"] for r in out] == expected


def test_composite_mode_still_orders_by_score():
    s = MultiDayCompositeSelector(_cfg(slot_ranking_mode="composite"))
    out = s.select(_baskets([("AAA", 0.1), ("BBB", 3.0), ("CCC", 1.5)]),
                   held_symbols=set(), weights={"s1": 1.0}, limit=3, session_date=D)
    assert [r["bare"] for r in out] == ["BBB", "CCC", "AAA"]


def test_composite_is_still_computed_under_hash_ordering():
    """It must remain scoreable against the random baseline."""
    s = MultiDayCompositeSelector(_cfg())
    out = s.select(_baskets([("AAA", 2.0)]), held_symbols=set(),
                   weights={"s1": 1.0}, limit=5, session_date=D)
    assert out[0]["composite"] == pytest.approx(2.0)


def test_hash_order_is_deterministic_within_a_session():
    """Idempotent re-runs of the cron must pick the same names."""
    s = MultiDayCompositeSelector(_cfg())
    kw = dict(held_symbols=set(), weights={"s1": 1.0}, limit=3, session_date=D)
    a = s.select(_baskets([("AAA", 1), ("BBB", 2), ("CCC", 3)]), **kw)
    b = s.select(_baskets([("AAA", 1), ("BBB", 2), ("CCC", 3)]), **kw)
    assert [r["bare"] for r in a] == [r["bare"] for r in b]


def test_hash_order_reshuffles_across_days():
    """No symbol may be permanently favoured."""
    s = MultiDayCompositeSelector(_cfg())
    syms = [(c * 3, 1.0) for c in "ABCDEFGH"]
    kw = dict(held_symbols=set(), weights={"s1": 1.0}, limit=8)
    orders = {d.isoformat(): [r["bare"] for r in s.select(_baskets(syms), session_date=d, **kw)]
              for d in (date(2026, 8, 10), date(2026, 8, 11), date(2026, 8, 12))}
    assert len(set(tuple(v) for v in orders.values())) > 1


def test_dedupe_and_held_exclusion_survive_the_ranking_change():
    """Ordering changed; dedupe/held semantics must not have."""
    s = MultiDayCompositeSelector(_cfg())
    baskets = {"s1": [_cand("AAA", 2.0), _cand("BBB", 1.0)],
               "s2": [_cand("AAA", 1.0)]}          # AAA seen twice -> one row
    out = s.select(baskets, held_symbols={"BBB"}, weights={"s1": 1.0, "s2": 1.0},
                   limit=10, session_date=D)
    assert [r["bare"] for r in out] == ["AAA"]
    assert sorted(out[0]["contributors"]) == ["s1", "s2"]
    assert out[0]["composite"] == pytest.approx(3.0)   # summed across contributors


def test_limit_is_respected_under_hash_ordering():
    s = MultiDayCompositeSelector(_cfg())
    out = s.select(_baskets([(c * 3, 1.0) for c in "ABCDEF"]), held_symbols=set(),
                   weights={"s1": 1.0}, limit=2, session_date=D)
    assert len(out) == 2
