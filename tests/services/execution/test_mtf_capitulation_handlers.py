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
                "data_source": "cache/preaggregate/clean_daily_from5m.feather",
                "lookback_days": 5, "loser_pct": 0.05, "adv_tier": 1, "adv_tier_count": 5,
                "turnover_shock_min": 2.0, "shock_lookback_days": 20,
                "adv_floor_inr": 2_000_000, "min_price": 5.0,
                "min_universe_symbols_per_day": 20, "hold_days": 2,
                "exclude_ca_in_hold_window": True,
                "ca_events_path": "data/corporate_actions/does_not_exist.parquet",
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
        }
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
    cfg["setups"]["mtf_capitulation_revert_long"]["capital_allocation"]["max_concurrent_slots"] = 0
    s = run_eod(cfg, _FakeBroker({}), now_ist=pd.Timestamp("2025-06-16 15:25"),
                repo_root=tmp_path)
    assert s["entered_count"] == 0


def test_max_new_per_day_cap(tmp_path):
    _write_feather(tmp_path)
    cfg = _cfg(tmp_path)
    cfg["setups"]["mtf_capitulation_revert_long"]["capital_allocation"]["max_new_positions_per_day"] = 0
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
