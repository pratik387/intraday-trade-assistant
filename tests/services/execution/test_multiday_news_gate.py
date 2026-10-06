"""The multiday news gate in the entry handler (design:
specs/2026-10-06-multiday-news-skip-and-bulk-tilt-design.md).

Evidence it encodes (research baselines, 9,012 trades 2023-01..2026-04): a
capitulation BUY whose signal day is a results reaction day (or the day before
a scheduled results meeting), on a drop still shallower than -20% vs the 50-day
SMA, is a loser (median -2.8k..-4.4k, win 33-35%) in every year and every
setup; removing it = +169k/+648k/+723k/+3k by year. A bulk deal in the window
is the opposite: +12k/+2k/+5k/+18k vs quiet, so it overrides the skip and
(stage 3) doubles the risk budget.

What these tests pin:
  - observe-only: enabled=false tags and logs WOULD_SKIP but removes nothing
  - enabled=true removes the R3 name from the basket (slot passes on)
  - deep drop keeps; bulk overrides; SMA None keeps (fail open)
  - feeds missing/stale -> gate off, every candidate tilt 1.0, nothing removed
  - the tilt map is keyed by bare symbol and carries the bulk multiplier
  - config keys are read with [] (a missing key raises)
"""
from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest

from services.execution import mtf_capitulation_handlers as H

NOW = pd.Timestamp("2026-10-06 15:35:00")
TODAY = pd.Timestamp("2026-10-06")

GATE = {
    "enabled": False, "results_days_before": 1, "results_days_after": 0, "shallow_drop_sma50_pct": -20.0,
    "bulk_overrides_skip": True, "bulk_deal_window_bdays": 1, "filing_cutoff_hhmm": "15:30",
    "results_cluster_categories": ["results", "investor/analyst", "board meeting"],
    "scheduled_purpose_regex": "financial result",
    "category_patterns": [["results", "financial result|results|outcome of board meeting.*result"],
                          ["board meeting", "board meeting"]],
    "bulk_deal_risk_multiplier": 2.0, "sma_days": 50,
    "feeds": {"announcements": {"path": "news/ann.parquet", "date_column": "an_dt", "max_staleness_days": 4},
              "event_calendar": {"path": "news/cal.parquet", "date_column": "fetched_at", "max_staleness_days": 4},
              "bulk_deals": {"path": "news/bulk.parquet", "date_column": "date", "max_staleness_days": 6}},
    "refresh": {"news_module": "x", "news_lookback_days": 21, "news_forward_days": 10, "news_sleep_secs": 1,
                "news_timeout_sec": 10, "bulk_module": "y", "bulk_lookback_days": 21, "bulk_sleep_secs": 1,
                "bulk_timeout_sec": 10},
}


def _write_feeds(root: Path, *, results_for=(), bulk_for=(), scheduled_for=()):
    (root / "news").mkdir(parents=True, exist_ok=True)
    ann = pd.DataFrame({
        "symbol": list(results_for) or ["ZZZ"],
        "an_dt": [pd.Timestamp("2026-10-06 09:05:00")] * (len(results_for) or 1),
        "desc": ["Financial Results"] * (len(results_for) or 1) if results_for else ["General Updates"],
        "text": [""] * (len(results_for) or 1),
    })
    ann.to_parquet(root / "news" / "ann.parquet", index=False)
    cal = pd.DataFrame({
        "symbol": list(scheduled_for) or ["ZZZ"],
        "meeting_date": [pd.Timestamp("2026-10-07")] * (len(scheduled_for) or 1),
        "purpose": ["Financial Results"] * (len(scheduled_for) or 1),
        "bm_desc": [""] * (len(scheduled_for) or 1),
        "fetched_at": [NOW] * (len(scheduled_for) or 1),
    })
    cal.to_parquet(root / "news" / "cal.parquet", index=False)
    bulk = pd.DataFrame({
        "date": [TODAY] * (len(bulk_for) or 1),
        "symbol": list(bulk_for) or ["ZZZ"],
        "side": ["SELL"] * (len(bulk_for) or 1),
    })
    bulk.to_parquet(root / "news" / "bulk.parquet", index=False)


def _cand(sym, dist):
    return {"symbol": f"NSE:{sym}", "cap_score": 1.0, "tshock": 3.0, "trail_ret": -0.1,
            "rank_pct": 0.05, "close": 100.0, "dist_sma_pct": dist}


def _run(tmp_path, gate_over, baskets):
    gate = copy.deepcopy(GATE); gate.update(gate_over)
    active = [("zscore_oversold_revert_long", {"news_gate": gate}),
              ("low52_capitulation_revert_long", {"news_gate": gate})]
    summary = {}
    m = H._apply_news_gate(baskets, active, today=TODAY, now=NOW, repo_root=tmp_path, summary=summary)
    return m, summary


def test_observe_only_tags_and_keeps_everything(tmp_path):
    _write_feeds(tmp_path, results_for=["RES"])
    baskets = {"zscore_oversold_revert_long": [_cand("RES", -7.0), _cand("QUIET", -7.0)],
               "low52_capitulation_revert_long": []}
    m, s = _run(tmp_path, {"enabled": False}, baskets)
    assert [c["symbol"] for c in baskets["zscore_oversold_revert_long"]] == ["NSE:RES", "NSE:QUIET"]
    assert s["news_gate_would_skip"] == 1 and s["news_gate_skipped"] == 0
    assert m["RES"]["log"]["verdict"] == "skip" and m["RES"]["log"]["enabled"] is False
    assert m["QUIET"]["log"]["verdict"] == "take"


def test_enabled_removes_the_results_day_name(tmp_path):
    _write_feeds(tmp_path, results_for=["RES"])
    baskets = {"zscore_oversold_revert_long": [_cand("RES", -7.0), _cand("NEXT", -9.0)],
               "low52_capitulation_revert_long": []}
    m, s = _run(tmp_path, {"enabled": True}, baskets)
    assert [c["symbol"] for c in baskets["zscore_oversold_revert_long"]] == ["NSE:NEXT"]
    assert s["news_gate_skipped"] == 1
    assert m["RES"]["log"]["reason"].startswith("results_reaction")


def test_deep_drop_on_results_is_kept(tmp_path):
    _write_feeds(tmp_path, results_for=["RES"])
    baskets = {"zscore_oversold_revert_long": [_cand("RES", -26.0)], "low52_capitulation_revert_long": []}
    m, s = _run(tmp_path, {"enabled": True}, baskets)
    assert len(baskets["zscore_oversold_revert_long"]) == 1
    assert m["RES"]["log"]["reason"] == "deep_drop"


def test_sma_unavailable_fails_open(tmp_path):
    _write_feeds(tmp_path, results_for=["RES"])
    baskets = {"zscore_oversold_revert_long": [_cand("RES", None)], "low52_capitulation_revert_long": []}
    m, s = _run(tmp_path, {"enabled": True}, baskets)
    assert len(baskets["zscore_oversold_revert_long"]) == 1
    assert m["RES"]["log"]["reason"] == "sma50_unavailable"


def test_scheduled_results_tomorrow_is_the_day_before_leg(tmp_path):
    _write_feeds(tmp_path, scheduled_for=["SCH"])
    baskets = {"zscore_oversold_revert_long": [_cand("SCH", -5.0)], "low52_capitulation_revert_long": []}
    m, s = _run(tmp_path, {"enabled": True}, baskets)
    assert baskets["zscore_oversold_revert_long"] == []
    assert "results_scheduled" in m["SCH"]["log"]["reason"]


def test_bulk_deal_overrides_the_skip_and_sets_the_tilt(tmp_path):
    _write_feeds(tmp_path, results_for=["BOTH"], bulk_for=["BOTH", "BULKONLY"])
    baskets = {"zscore_oversold_revert_long": [_cand("BOTH", -5.0), _cand("BULKONLY", -5.0)],
               "low52_capitulation_revert_long": []}
    m, s = _run(tmp_path, {"enabled": True}, baskets)
    assert len(baskets["zscore_oversold_revert_long"]) == 2
    assert m["BOTH"]["log"]["reason"] == "bulk_overrides"
    assert m["BOTH"]["tilt"] == 2.0 and m["BULKONLY"]["tilt"] == 2.0
    assert m["BULKONLY"]["log"]["bulk_side"] == "SELL"


def test_missing_feed_turns_the_gate_off_for_the_pass(tmp_path):
    # no feeds written at all
    baskets = {"zscore_oversold_revert_long": [_cand("RES", -7.0)], "low52_capitulation_revert_long": []}
    m, s = _run(tmp_path, {"enabled": True}, baskets)
    assert len(baskets["zscore_oversold_revert_long"]) == 1
    assert "news_gate_off" in s and m["RES"]["tilt"] == 1.0 and m["RES"]["log"]["gate"] == "off"


def test_stale_feed_turns_the_gate_off(tmp_path):
    _write_feeds(tmp_path, results_for=["RES"])
    # age the announcements feed past max_staleness_days
    old = pd.read_parquet(tmp_path / "news" / "ann.parquet")
    old["an_dt"] = pd.Timestamp("2026-09-01 09:00:00")
    old.to_parquet(tmp_path / "news" / "ann.parquet", index=False)
    baskets = {"zscore_oversold_revert_long": [_cand("RES", -7.0)], "low52_capitulation_revert_long": []}
    m, s = _run(tmp_path, {"enabled": True}, baskets)
    assert len(baskets["zscore_oversold_revert_long"]) == 1 and "announcements" in s["news_gate_off"]


def test_config_keys_are_required(tmp_path):
    _write_feeds(tmp_path, results_for=["RES"])
    gate = copy.deepcopy(GATE); del gate["shallow_drop_sma50_pct"]
    active = [("zscore_oversold_revert_long", {"news_gate": gate})]
    with pytest.raises(KeyError):
        H._apply_news_gate({"zscore_oversold_revert_long": [_cand("RES", -7.0)]}, active,
                           today=TODAY, now=NOW, repo_root=tmp_path, summary={})


def test_tilt_is_applied_in_sizing_by_bare_symbol():
    """The composite selector emits NEW dicts; the sizing loop must find the
    tilt by bare symbol, not on the basket row."""
    src = (Path(H.__file__)).read_text(encoding="utf-8")
    i = src.index("news_by_bare.get(bare)")
    assert "risk_inr *= _tilt" in src[i:i + 800]
    assert src.index("news_by_bare = _apply_news_gate(") < src.index("chosen = selector.select(")
