"""Integration tests for the MTF capitulation basket executor.

Exercises the full lifecycle end-to-end against a synthetic clean daily feather,
a fake MTF universe (so a controlled loser is leverageable), and a fake broker:
  rank+AMO-BUY (run_eod) -> fill at open (run_verify_entries) -> MOC-SELL exit
  on the K-day close (run_eod). DRY_RUN is forced so the feather + archive 5m
  paths are used deterministically (no live API / wall clock).
"""
from datetime import date, time, timedelta
from pathlib import Path

import pandas as pd
import pytest

import services.execution.mtf_capitulation_handlers as mod
from services.execution.mtf_capitulation_handlers import (
    run_eod, run_verify_entries, _add_trading_days, _next_trading_day,
)

SD = date(2025, 6, 16)  # Monday (signal day T)


# ---- composite-selection helpers ------------------------------------------

def _two_setup_config(tmp_path):
    def _block(state_name, weight=1.0):
        return {
            "horizon": "multi_day", "enabled": False, "paper_enabled": True,
            "selection_mode": "trailing_loser_decile", "lookback_days": 5, "loser_pct": 0.1,
            "adv_tier": 1, "adv_tier_count": 5, "turnover_shock_min": 2.0,
            "shock_lookback_days": 20, "adv_floor_inr": 2_000_000, "min_price": 5.0,
            "min_universe_symbols_per_day": 20, "hold_days": 2,
            "exclude_ca_in_hold_window": False, "ca_events_path": "",
            "composite_weight": weight, "cap_score_clip": 3.0,
            "mtf": {"approved_list_snapshot_path": "data/mtf_universe/approved_mtf_securities_2026-05-21.json",
                    "interest_pct_per_day": 0.0004, "exclude_etf": True,
                    "fallback_to_cnc_if_not_mtf": True, "stale_snapshot_warn_days": 7},
            "capital_allocation": {"state_file": str(tmp_path / f"{state_name}.json"),
                                   "max_concurrent_slots": 100, "margin_per_slot_inr": 100000,
                                   "max_new_positions_per_day": 100},
        }
    return {
        "setups": {"A2": _block("a2_slots"), "C1": _block("c1_slots")},
        "multi_day_portfolio": {"max_new_per_day": 100, "max_concurrent": 200,
                                "cap_score_clip": 3.0, "tiebreaker": "tshock",
                                "selection_log_path": str(tmp_path / "sel.jsonl")},
    }


def _stub_broker_amo():
    from unittest.mock import MagicMock
    b = MagicMock()
    b.place_order.return_value = "AMO1"
    b._dry_session_date = None
    return b


# ---- fakes ----------------------------------------------------------------

class _FakeMtfInfo:
    def __init__(self, leverage):
        self.leverage = leverage
        self.category = "non_fo"


class _FakeMtf:
    """Mimics MtfUniverse: _by_symbol, is_eligible, lookup."""
    def __init__(self, *_a, **_k):
        syms = [f"FILL{i:02d}" for i in range(25)] + ["LOSER1", "LOSER_NOSHOCK", "LOSER_HIGHADV"]
        self._by_symbol = {s: _FakeMtfInfo(3.0) for s in syms}

    def all_symbols(self):
        return set(self._by_symbol.keys())

    def is_eligible(self, sym, *, exclude_etf=True):
        return sym.upper().replace("NSE:", "") in self._by_symbol

    def lookup(self, sym):
        return self._by_symbol.get(sym.upper().replace("NSE:", ""))


class _FakeBroker:
    def __init__(self, enriched):
        self._dry_session_date = SD
        self._enriched = enriched
        self.orders = []

    def place_order(self, **kw):
        self.orders.append(kw)
        return f"ORD-{len(self.orders)}"

    def _load_enriched_5m(self):
        return self._enriched


def _biz_days(end, n):
    out, d = [], end
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return sorted(out)


def _daily_panel():
    """28 symbols x 25 business days; LOSER1 = deepest loser + tier1 + shock."""
    days = _biz_days(SD, 25)
    rows = []
    for i in range(25):
        vol = 30000 + i * 8000
        for d in days:
            rows.append((pd.Timestamp(d), f"FILL{i:02d}", 100.0, 101.0, 99.0, 100.0, vol))

    def loser(sym, base_vol, shock, deepest):
        drop = 0.12 if deepest else 0.10
        for j, d in enumerate(days):
            if j >= len(days) - 5:
                frac = (j - (len(days) - 6)) / 5.0
                close = 100.0 * (1 - drop * frac)
            else:
                close = 100.0
            v = base_vol * 3 if (shock and d == SD) else base_vol
            rows.append((pd.Timestamp(d), sym, close, close * 1.01, close * 0.99, close, v))
    loser("LOSER1", 25000, shock=True, deepest=True)
    loser("LOSER_NOSHOCK", 25000, shock=False, deepest=False)
    loser("LOSER_HIGHADV", 400000, shock=True, deepest=False)
    return pd.DataFrame(rows, columns=["date", "symbol", "open", "high", "low", "close", "volume"])


def _enriched_5m(entry_day, exit_day, open_px, close_px):
    """5m archive for LOSER1 with an entry-day 09:15 bar + exit-day 15:25 bar."""
    idx = [pd.Timestamp.combine(entry_day, time(9, 15)),
           pd.Timestamp.combine(exit_day, time(15, 25))]
    df = pd.DataFrame(
        {"open": [open_px, close_px], "high": [open_px, close_px],
         "low": [open_px, close_px], "close": [open_px, close_px], "volume": [1, 1]},
        index=pd.DatetimeIndex(idx),
    )
    return {"LOSER1": df}


def _cfg(tmp_path):
    return {
        "setups": {
            "mtf_capitulation_revert_long": {
                "enabled": False, "paper_enabled": True,
                "horizon": "multi_day", "selection_mode": "trailing_loser_decile",
                "data_source": "cache/preaggregate/clean_daily_from5m.feather",
                "lookback_days": 5, "loser_pct": 0.05, "adv_tier": 1, "adv_tier_count": 5,
                "turnover_shock_min": 2.0, "shock_lookback_days": 20,
                "adv_floor_inr": 2_000_000, "min_price": 5.0,
                "min_universe_symbols_per_day": 20, "hold_days": 2,
                "exclude_ca_in_hold_window": True,
                "ca_events_path": "data/corporate_actions/does_not_exist.parquet",
                "composite_weight": 1.0, "cap_score_clip": 3.0,
                "capital_allocation": {
                    "state_file": str(tmp_path / "state" / "mtf_capitulation_slots.json"),
                    "max_concurrent_slots": 100, "max_new_positions_per_day": 100,
                    "margin_per_slot_inr": 100000,
                },
                "mtf": {
                    "approved_list_snapshot_path": "ignored_monkeypatched.json",
                    "exclude_etf": True, "fallback_to_cnc_if_not_mtf": True,
                    "interest_pct_per_day": 0.0004,
                },
            }
        },
        "multi_day_portfolio": {"max_new_per_day": 100, "max_concurrent": 200,
                                "cap_score_clip": 3.0, "tiebreaker": "tshock",
                                "selection_log_path": str(tmp_path / "sel.jsonl")},
    }


def _write_feather(repo_root):
    p = repo_root / "cache" / "preaggregate" / "clean_daily_from5m.feather"
    p.parent.mkdir(parents=True, exist_ok=True)
    _daily_panel().to_feather(p)


@pytest.fixture(autouse=True)
def _force_dry_and_fake_mtf(monkeypatch):
    monkeypatch.setattr(mod, "_is_dry_run", lambda broker: True)
    monkeypatch.setattr(mod, "MtfUniverse", _FakeMtf)


# ---- tests ----------------------------------------------------------------

def test_add_trading_days():
    assert _add_trading_days(date(2025, 6, 16), 2) == date(2025, 6, 18)  # Mon -> Wed
    assert _add_trading_days(date(2025, 6, 19), 2) == date(2025, 6, 23)  # Thu -> Mon (skip wknd)


def test_run_eod_not_eligible_exits_clean(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["setups"]["mtf_capitulation_revert_long"]["paper_enabled"] = False
    s = run_eod(cfg, _FakeBroker({}), now_ist=pd.Timestamp("2025-06-16 15:25"),
                repo_root=tmp_path)
    assert s["entered_count"] == 0 and s["exited_count"] == 0


def test_run_eod_phase_exits_only_places_no_entries(tmp_path):
    """phase='exits' (the pre-close square-off pass) must NOT rank or place AMO
    BUYs — that's the post-close entry pass's job. Required so live can exit at
    the close AND compute the signal on the complete day-T bar (separate jobs)."""
    _write_feather(tmp_path)
    cfg = _cfg(tmp_path)
    broker = _FakeBroker({})
    s = run_eod(cfg, broker, now_ist=pd.Timestamp("2025-06-16 15:25"),
                repo_root=tmp_path, phase="exits")
    assert s["entered_count"] == 0
    amo = [o for o in broker.orders if o.get("variety") == "amo" and o["side"] == "BUY"]
    assert amo == []


def test_run_eod_phase_entries_only_places_amo(tmp_path):
    """phase='entries' (the post-close pass) ranks + places AMO BUYs."""
    _write_feather(tmp_path)
    cfg = _cfg(tmp_path)
    broker = _FakeBroker({})
    s = run_eod(cfg, broker, now_ist=pd.Timestamp("2025-06-16 15:35"),
                repo_root=tmp_path, phase="entries")
    assert s["entered_count"] == 1
    amo = [o for o in broker.orders if o.get("variety") == "amo" and o["side"] == "BUY"]
    assert len(amo) == 1 and amo[0]["symbol"] == "NSE:LOSER1"


def test_run_eod_default_phase_both_unchanged(tmp_path):
    """Default phase='both' preserves the combined behavior (replay/backtest)."""
    _write_feather(tmp_path)
    cfg = _cfg(tmp_path)
    broker = _FakeBroker({})
    s = run_eod(cfg, broker, now_ist=pd.Timestamp("2025-06-16 15:25"), repo_root=tmp_path)
    assert s["entered_count"] == 1


def test_run_eod_enters_loser_basket(tmp_path):
    _write_feather(tmp_path)
    cfg = _cfg(tmp_path)
    broker = _FakeBroker({})
    s = run_eod(cfg, broker, now_ist=pd.Timestamp("2025-06-16 15:25"), repo_root=tmp_path)
    assert s["entered_count"] == 1
    ev = s["events"][0]
    assert ev["symbol"] == "NSE:LOSER1"
    assert ev["product"] == "MTF"
    assert ev["entry_date"] == _next_trading_day(SD).isoformat()       # T+1
    assert ev["exit_on_date"] == _add_trading_days(_next_trading_day(SD), 2).isoformat()
    # An AMO BUY was placed.
    amo = [o for o in broker.orders if o["variety"] == "amo" and o["side"] == "BUY"]
    assert len(amo) == 1 and amo[0]["symbol"] == "NSE:LOSER1"


def test_full_lifecycle_entry_fill_exit(tmp_path):
    _write_feather(tmp_path)
    cfg = _cfg(tmp_path)
    entry_day = _next_trading_day(SD)                 # 2025-06-17 Tue
    exit_day = _add_trading_days(entry_day, 2)        # 2025-06-19 Thu
    broker = _FakeBroker(_enriched_5m(entry_day, exit_day, open_px=88.0, close_px=95.0))

    # 1) signal + AMO BUY
    run_eod(cfg, broker, now_ist=pd.Timestamp("2025-06-16 15:25"), repo_root=tmp_path)

    # 2) fill at entry-day open
    sf = run_verify_entries(cfg, broker, now_ist=pd.Timestamp(f"{entry_day} 09:30"))
    assert sf["filled_count"] == 1
    assert sf["events"][0]["entry_fill"] == 88.0

    # 3) exit at K-day close (MOC). Positive move 88 -> 95 nets profit after fees.
    se = run_eod(cfg, broker, now_ist=pd.Timestamp(f"{exit_day} 15:25"), repo_root=tmp_path)
    assert se["exited_count"] == 1
    exit_ev = next(e for e in se["events"] if e.get("symbol") == "NSE:LOSER1")
    assert exit_ev["entry"] == 88.0 and exit_ev["exit"] == 95.0
    assert exit_ev["net_pnl"] > 0

    # position store is now empty (entry was the only name; exit removed it).
    from services.state.position_persistence import PositionPersistence
    pdir = mod._position_state_dir(cfg["setups"]["mtf_capitulation_revert_long"])
    assert PositionPersistence(pdir).load_snapshot() == {}


def test_concurrency_cap_blocks_entries(tmp_path):
    _write_feather(tmp_path)
    cfg = _cfg(tmp_path)
    # Concurrency is now capped at the FAMILY level (multi_day_portfolio), since
    # the composite pass sizes one shared book across all multi-day setups.
    cfg["multi_day_portfolio"]["max_concurrent"] = 0
    s = run_eod(cfg, _FakeBroker({}), now_ist=pd.Timestamp("2025-06-16 15:25"),
                repo_root=tmp_path)
    assert s["entered_count"] == 0


def test_max_new_per_day_cap(tmp_path):
    _write_feather(tmp_path)
    cfg = _cfg(tmp_path)
    # New-per-day is now capped at the FAMILY level (multi_day_portfolio).
    cfg["multi_day_portfolio"]["max_new_per_day"] = 0
    s = run_eod(cfg, _FakeBroker({}), now_ist=pd.Timestamp("2025-06-16 15:25"),
                repo_root=tmp_path)
    assert s["entered_count"] == 0


def test_idempotent_no_duplicate_entry(tmp_path):
    _write_feather(tmp_path)
    cfg = _cfg(tmp_path)
    broker = _FakeBroker({})
    run_eod(cfg, broker, now_ist=pd.Timestamp("2025-06-16 15:25"), repo_root=tmp_path)
    # Re-run same signal day: LOSER1 already held => no second entry.
    s2 = run_eod(cfg, broker, now_ist=pd.Timestamp("2025-06-16 15:40"), repo_root=tmp_path)
    assert s2["entered_count"] == 0


def test_batch_runs_both_setups(tmp_path):
    """A2 (trailing_loser) + C1 low52 (near_period_low) both run via the generalized
    executor, each into its OWN position store."""
    _write_feather(tmp_path)
    cfg = _cfg(tmp_path)
    base = cfg["setups"]["mtf_capitulation_revert_long"]
    low = {k: v for k, v in base.items() if k not in ("lookback_days", "loser_pct")}
    low["selection_mode"] = "near_period_low"
    low["low_lookback_days"] = 20
    low["dist_low_max"] = 0.02
    low["capital_allocation"] = {**base["capital_allocation"],
                                 "state_file": str(tmp_path / "state" / "low52_slots.json")}
    cfg["setups"]["low52_capitulation_revert_long"] = low

    s = run_eod(cfg, _FakeBroker({}), now_ist=pd.Timestamp("2025-06-16 15:25"), repo_root=tmp_path)
    # LOSER1 qualifies for both triggers -> deduped by composite selector to ONE
    # position (cross-day held-union dedupe), owned by one setup with both as
    # contributors.
    assert s["entered_count"] == 1
    fired = {e["setup"] for e in s["events"]}
    assert fired <= {"mtf_capitulation_revert_long", "low52_capitulation_revert_long"}
    ev = s["events"][0]
    assert sorted(ev["contributors"]) == ["low52_capitulation_revert_long",
                                          "mtf_capitulation_revert_long"]


def test_composite_entries_dedup_two_setups_one_position(monkeypatch, tmp_path):
    import services.execution.mtf_capitulation_handlers as mh

    # Two multi-day setups both flag SHARED; A2 also flags AONLY.
    cfg = _two_setup_config(tmp_path)  # helper below
    monkeypatch.setattr(mh, "_eligible_multiday_setups",
                        lambda config, *, paper_mode: [("A2", cfg["setups"]["A2"]),
                                                       ("C1", cfg["setups"]["C1"])])
    monkeypatch.setattr(mh, "_decay_paused", lambda name, raw: False)
    monkeypatch.setattr(mh, "_prewarm_daily_universe", lambda setups, broker: None)

    # Stub each setup's ranker output (cap_score-bearing baskets).
    def fake_rank_for(name):
        if name == "A2":
            return [{"symbol": "SHARED", "cap_score": 1.0, "tshock": 3.0, "close": 100.0,
                     "trail_ret": -0.12, "adv_tier": 1, "rank_pct": 0.01},
                    {"symbol": "AONLY", "cap_score": 0.5, "tshock": 2.5, "close": 50.0,
                     "trail_ret": -0.10, "adv_tier": 1, "rank_pct": 0.04}]
        return [{"symbol": "SHARED", "cap_score": 1.0, "tshock": 2.0, "close": 100.0,
                 "trail_ret": -0.09, "adv_tier": 1, "rank_pct": 0.02}]
    monkeypatch.setattr(mh, "_rank_basket_for_setup",
                        lambda name, raw, broker, today, ca_ex_dates, repo_root: fake_rank_for(name))

    broker = _stub_broker_amo()  # place_order returns a fake order id
    summary = mh.run_eod(cfg, broker, now_ist=pd.Timestamp("2026-06-22 15:35:00"),
                         paper_mode=True, phase="entries")

    # SHARED entered ONCE (deduped), owner = the higher weighted cap_score (tie -> deterministic).
    placed = [e for e in summary["events"]]
    symbols = sorted(e["symbol"] for e in placed)
    assert symbols == ["NSE:AONLY", "NSE:SHARED"]
    shared = next(e for e in placed if e["symbol"] == "NSE:SHARED")
    assert sorted(shared["contributors"]) == ["A2", "C1"]

    # SHARED persisted in exactly one store, with contributors tagged.
    from services.state.position_persistence import PositionPersistence
    a2_store = PositionPersistence(mh._position_state_dir(cfg["setups"]["A2"]))
    c1_store = PositionPersistence(mh._position_state_dir(cfg["setups"]["C1"]))
    in_a2 = a2_store.get_position("NSE:SHARED") is not None
    in_c1 = c1_store.get_position("NSE:SHARED") is not None
    assert in_a2 ^ in_c1  # exactly one store holds it
    owner_store = a2_store if in_a2 else c1_store
    pos = owner_store.get_position("NSE:SHARED")
    assert sorted(pos.state["contributors"]) == ["A2", "C1"]


def test_composite_entries_skip_held_union(monkeypatch, tmp_path):
    import services.execution.mtf_capitulation_handlers as mh
    cfg = _two_setup_config(tmp_path)
    # Pre-seed SHARED as held in C1's store (still inside its hold window).
    from services.state.position_persistence import PositionPersistence
    c1_store = PositionPersistence(mh._position_state_dir(cfg["setups"]["C1"]))
    c1_store.save_position(symbol="NSE:SHARED", side="BUY", qty=10, avg_price=100.0,
                           trade_id="t", entry_date="2026-06-20", exit_on_date="2026-06-24",
                           product="MTF", state={"qty": 10})
    monkeypatch.setattr(mh, "_eligible_multiday_setups",
                        lambda config, *, paper_mode: [("A2", cfg["setups"]["A2"]),
                                                       ("C1", cfg["setups"]["C1"])])
    monkeypatch.setattr(mh, "_decay_paused", lambda name, raw: False)
    monkeypatch.setattr(mh, "_prewarm_daily_universe", lambda setups, broker: None)
    monkeypatch.setattr(mh, "_rank_basket_for_setup",
                        lambda name, raw, broker, today, ca_ex_dates, repo_root:
                        [{"symbol": "SHARED", "cap_score": 5.0, "tshock": 3.0, "close": 100.0,
                          "trail_ret": -0.2, "adv_tier": 1, "rank_pct": 0.01}])
    broker = _stub_broker_amo()
    summary = mh.run_eod(cfg, broker, now_ist=pd.Timestamp("2026-06-22 15:35:00"),
                         paper_mode=True, phase="entries")
    assert summary["entered_count"] == 0  # SHARED already held -> excluded


def test_selection_diagnostics_logged_per_setup_symbol(monkeypatch, tmp_path):
    import json as _json
    import services.execution.mtf_capitulation_handlers as mh
    cfg = _two_setup_config(tmp_path)
    log_path = tmp_path / "sel.jsonl"
    cfg["multi_day_portfolio"]["selection_log_path"] = str(log_path)
    monkeypatch.setattr(mh, "_eligible_multiday_setups",
                        lambda config, *, paper_mode: [("A2", cfg["setups"]["A2"]),
                                                       ("C1", cfg["setups"]["C1"])])
    monkeypatch.setattr(mh, "_decay_paused", lambda name, raw: False)
    monkeypatch.setattr(mh, "_prewarm_daily_universe", lambda setups, broker: None)
    monkeypatch.setattr(mh, "_rank_basket_for_setup",
                        lambda name, raw, broker, today, ca_ex_dates, repo_root:
                        ([{"symbol": "SHARED", "cap_score": 1.0, "tshock": 3.0, "close": 100.0,
                           "trail_ret": -0.12, "adv_tier": 1, "rank_pct": 0.01}]
                         if name == "C1" else
                         [{"symbol": "SHARED", "cap_score": 1.0, "tshock": 3.0, "close": 100.0,
                           "trail_ret": -0.12, "adv_tier": 1, "rank_pct": 0.01},
                          {"symbol": "AONLY", "cap_score": 0.5, "tshock": 2.5, "close": 50.0,
                           "trail_ret": -0.10, "adv_tier": 1, "rank_pct": 0.04}]))
    broker = _stub_broker_amo()
    mh.run_eod(cfg, broker, now_ist=pd.Timestamp("2026-06-22 15:35:00"),
               paper_mode=True, phase="entries")
    rows = [_json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    # one row per (setup, symbol): A2/SHARED, A2/AONLY, C1/SHARED
    keyed = {(r["setup"], r["symbol"]): r for r in rows}
    assert set(keyed) == {("A2", "SHARED"), ("A2", "AONLY"), ("C1", "SHARED")}
    assert keyed[("A2", "SHARED")]["cap_score"] == 1.0
    assert keyed[("A2", "SHARED")]["session_date"] == "2026-06-22"
    assert keyed[("A2", "SHARED")]["consensus_count"] == 2  # flagged by A2 + C1
    assert keyed[("A2", "AONLY")]["consensus_count"] == 1
    assert keyed[("A2", "SHARED")]["chosen"] is True


def test_exit_feeds_all_contributors_tripwires(monkeypatch, tmp_path):
    import services.execution.mtf_capitulation_handlers as mh
    from services.state.position_persistence import PositionPersistence
    from services.risk.decay_tripwire import DecayTripwire

    cfg = _two_setup_config(tmp_path)
    # give both setups a decay tripwire
    for n in ("A2", "C1"):
        cfg["setups"][n]["decay_tripwire"] = {
            "window_trades": 30, "pf_floor": 1.2, "sustained_weeks": 6,
            "state_file": str(tmp_path / f"tw_{n}.json")}
    # Seed an owned-by-A2 position that exits today, tagged contributors=[A2,C1].
    a2_store = PositionPersistence(mh._position_state_dir(cfg["setups"]["A2"]))
    a2_store.save_position(symbol="NSE:SHARED", side="BUY", qty=10, avg_price=100.0,
                           trade_id="A2_2026-06-20_SHARED", entry_date="2026-06-20",
                           exit_on_date="2026-06-22", product="MTF",
                           state={"qty": 10, "leverage": 2.5, "entry_fill_price": 100.0,
                                  "contributors": ["A2", "C1"]})
    monkeypatch.setattr(mh, "_eligible_multiday_setups",
                        lambda config, *, paper_mode: [("A2", cfg["setups"]["A2"]),
                                                       ("C1", cfg["setups"]["C1"])])
    monkeypatch.setattr(mh, "_paper_close_price", lambda b, s, d: 110.0)  # +10% exit
    broker = _stub_broker_amo()
    mh.run_eod(cfg, broker, now_ist=pd.Timestamp("2026-06-22 15:28:00"),
               paper_mode=True, phase="exits")
    # both A2 and C1 tripwires recorded one trade
    for n in ("A2", "C1"):
        tw = DecayTripwire(setup_name=n, state_path=tmp_path / f"tw_{n}.json",
                           window_trades=30, pf_floor=1.2, sustained_weeks=6)
        assert len(tw._trades) == 1  # noqa: SLF001
        assert tw._trades[0].net_pnl_inr > 0  # +10% gross, profitable
    # Attribution flag: the position is OWNED by A2's store — A2's row is the
    # real book trade (attributed False), C1's row is a MIRROR (True) so pooled
    # views can exclude it (one position counts once).
    tw_a2 = DecayTripwire(setup_name="A2", state_path=tmp_path / "tw_A2.json",
                          window_trades=30, pf_floor=1.2, sustained_weeks=6)
    tw_c1 = DecayTripwire(setup_name="C1", state_path=tmp_path / "tw_C1.json",
                          window_trades=30, pf_floor=1.2, sustained_weeks=6)
    assert tw_a2._trades[0].attributed is False  # noqa: SLF001
    assert tw_c1._trades[0].attributed is True   # noqa: SLF001
