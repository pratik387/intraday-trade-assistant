"""Overnight live canary (Risk Charter restart rules) and the paper mirror under a live pause.

- With live paused (enabled=False, paper_enabled=True) the LIVE cron still detects
  and logs every fire, because the paper mirror is rebuilt from those log lines,
  but places no order.
- Canary on: live orders only for names under max_adv_inr ADV, as CNC, at
  notional_inr; stopped by the monthly stop or by the SPRT accepting 'no edge'.
"""
import json
import logging
import math
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import services.execution.overnight_handlers as oh  # noqa: E402
import services.dispatch.setup_registry as sr  # noqa: E402
from tests.services.execution.test_overnight_handlers import _minimal_config  # noqa: E402

TODAY = "2026-11-04"
NOW = pd.Timestamp(f"{TODAY} 15:26:00")
SETUP = "close_dn_overnight_long"
SPRT = {"mu1_pct": 0.44, "sd_pct": 2.4, "alpha": 0.10, "beta": 0.20}
CANARY = {"enabled": True, "start_date": "2026-11-01", "max_adv_inr": 2500000,
          "notional_inr": 15000, "min_notional_inr": 10000, "monthly_stop_inr": -3000,
          "sprt": SPRT}
LIVE_TEST_SPRT = {"mu1_pct": 0.35, "sd_pct": 2.2, "alpha": 0.10, "beta": 0.20}
LIVE_TEST = {"enabled": False, "start_date": "2026-11-02", "monthly_stop_inr": -5000,
             "sprt": LIVE_TEST_SPRT}


class _Detector:
    """Fires on every symbol as MTF at 3x, like the real detector on an MTF name."""
    def __init__(self, cfg):
        pass

    def detect(self, ctx):
        oh.logger.info("close_dn_overnight_long fired | symbol=%s product=MTF lev=3.00", ctx)
        evt = SimpleNamespace(context={"product": "MTF", "leverage": 3.0})
        return SimpleNamespace(structure_detected=True, events=[evt])

    def plan_long_strategy(self, ctx, evt):
        return SimpleNamespace(qty=750, entry_price=100.0)


def _trade(ts, pnl, entry=100.0, qty=150):
    return {"net_pnl_inr": pnl, "ts_iso": ts, "entry_price": entry, "qty": qty}


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A live run_entry with fake detection and a recording broker; ADV per symbol below."""
    adv = {"NSE:THIN": 1_500_000, "NSE:THICK": 40_000_000, "NSE:NOADV": None}
    root = tmp_path / "repo"
    (root / "data" / "close_dn_baseline").mkdir(parents=True)
    (root / "data" / "close_dn_baseline" / "candidates_latest.json").write_text(json.dumps({
        "session_date": TODAY,
        "candidates": [{"symbol": s, "adv_inr": a, "prior_close": 100.0, "prev_prior_close": 95.0}
                       for s, a in adv.items()]}))
    monkeypatch.setattr(oh, "__file__", str(root / "services" / "execution" / "overnight_handlers.py"))
    monkeypatch.setattr(sr, "_import_path", lambda p: _Detector)
    monkeypatch.setattr(oh, "_build_market_context", lambda broker, sym, *a, **k: sym)
    monkeypatch.setattr(oh, "_safe_quote", lambda broker, sym: (100.0, 120.0, 80.0))
    monkeypatch.setattr(oh, "_instrument_tick", lambda broker, sym, cfg: 0.05)
    monkeypatch.setattr(oh, "_safe_top_of_book", lambda broker, sym: (None, None, None, None))
    monkeypatch.setattr(oh, "_entry_cutoff_reached", lambda cfg, paper: False)
    monkeypatch.setattr(oh, "_live_poll_fill_ex", lambda broker, oid, timeout_sec: (100.0, {"status": "COMPLETE"}))
    monkeypatch.setattr(oh, "_reconcile_unattached_buys", lambda *a, **k: 0)
    placed = []

    def _buy(broker, **kw):
        placed.append(kw)
        return f"B{len(placed)}"
    monkeypatch.setattr(oh, "_place_buy", _buy)

    cfg = _minimal_config(tmp_path / "overnight_slots.json")
    sc = cfg["setups"][SETUP]
    sc["capital_allocation"]["max_new_positions_per_day"] = 4
    sc["entry_reconcile_delay_sec"] = 0
    sc["live_canary"] = dict(CANARY)
    sc["live_test"] = dict(LIVE_TEST, state_file=str(tmp_path / "live_test_state.json"))
    Path(sc["live_test"]["state_file"]).write_text(json.dumps(oh.new_live_test_state(SETUP, sc["live_test"])))
    sc["live_products_allowed"] = ["CNC", "MTF"]
    ledger = tmp_path / "tripwire_live.json"
    sc["decay_tripwire"] = {"window_trades": 30, "pf_floor": 1.2, "sustained_weeks": 3,
                            "state_file": str(ledger), "archive_entries_before": None,
                            "archive_label": None, "archive_regime": None}
    broker = MagicMock()
    broker._data_sdk = None

    def write_ledger(trades):
        ledger.write_text(json.dumps({"setup_name": SETUP, "window_trades": 30, "pf_floor": 1.2,
                                      "sustained_weeks": 3, "first_below_floor_ts": None,
                                      "paused_since": None, "trades": trades}))

    def run():
        return oh.run_entry(cfg, broker, now_ist=NOW, paper_mode=False)
    return SimpleNamespace(cfg=cfg, sc=sc, run=run, placed=placed, write_ledger=write_ledger,
                           cand_path=root / "data" / "close_dn_baseline" / "candidates_latest.json")


def test_live_paused_still_logs_fires_for_the_paper_mirror(env, caplog):
    env.sc["enabled"] = False
    with caplog.at_level(logging.INFO):
        s = env.run()
    assert env.placed == []
    assert s["mirror_only_count"] == 3
    fired = [r for r in caplog.records if "close_dn_overnight_long fired" in r.getMessage()]
    assert len(fired) == 3                      # the lines tools/overnight_paper_slippage.py parses


def test_live_paused_mirror_survives_a_tripped_tripwire(env, caplog):
    env.sc["enabled"] = True
    env.sc["live_canary"]["enabled"] = False
    env.sc["live_test"]["enabled"] = True
    env.write_ledger([])
    led = Path(env.sc["decay_tripwire"]["state_file"])
    led.write_text(json.dumps({**json.loads(led.read_text()), "paused_since": "2026-11-02T09:16:00"}))
    s = env.run()
    assert env.placed == [] and s["mirror_only_count"] == 3


def test_canary_takes_only_thin_names_as_cnc_at_canary_size(env):
    env.sc["enabled"] = True
    env.write_ledger([])
    s = env.run()
    assert [p["symbol"] for p in env.placed] == ["NSE:THIN"]
    (p,) = env.placed
    assert p["product"] == "CNC"
    assert p["qty"] * p["limit_price"] <= CANARY["notional_inr"]
    assert p["qty"] * p["limit_price"] >= CANARY["min_notional_inr"]
    assert s["canary_adv_skipped"] == 2         # THICK (Rs4cr) and NOADV (unknown)
    slots = json.loads(Path(env.sc["capital_allocation"]["state_file"]).read_text())["slots"]
    (taken,) = [x for x in slots if x["status"] != "free"]
    assert taken["product"] == "CNC" and taken["leverage"] == 1.0
    assert taken["margin_inr"] == pytest.approx(p["qty"] * p["limit_price"])


def test_canary_floor_skips_when_one_share_exceeds_the_notional(env, monkeypatch):
    env.sc["enabled"] = True
    env.write_ledger([])
    monkeypatch.setattr(oh, "_safe_quote", lambda broker, sym: (16000.0, 18000.0, 14000.0))
    env.run()
    assert env.placed == []


def test_monthly_stop_blocks_live_but_not_the_mirror(env, caplog):
    env.sc["enabled"] = True
    env.write_ledger([_trade("2026-11-02T09:16:00", -1800), _trade("2026-11-03T09:16:00", -1300)])
    with caplog.at_level(logging.INFO):
        s = env.run()
    assert env.placed == [] and s["mirror_only_count"] == 3
    assert any(r.levelno == logging.CRITICAL and "monthly stop" in r.getMessage() for r in caplog.records)


def test_losses_before_the_canary_start_do_not_count(env):
    env.sc["enabled"] = True
    env.write_ledger([_trade("2026-10-28T09:16:00", -9000), _trade("2026-11-01T09:16:00", -9000)])
    env.run()
    assert [p["symbol"] for p in env.placed] == ["NSE:THIN"]


@pytest.mark.parametrize("bad", [{"start_date": None}, {"start_date": "soon"},
                                 {"min_notional_inr": 20000}, {"monthly_stop_inr": 3000},
                                 {"sprt": {**SPRT, "sd_pct": 0}}, {"sprt": {**SPRT, "alpha": 1.0}}])
def test_unusable_canary_config_blocks_live_but_keeps_the_mirror(env, caplog, bad):
    env.sc["enabled"] = True
    env.write_ledger([])
    env.sc["live_canary"].update(bad)
    with caplog.at_level(logging.INFO):
        s = env.run()
    assert env.placed == [] and s["mirror_only_count"] == 3
    assert any(r.levelno == logging.CRITICAL and "cannot be evaluated" in r.getMessage()
               for r in caplog.records)


def test_missing_ledger_blocks_live_entries(env):
    env.sc["enabled"] = True                    # no write_ledger: the file does not exist
    s = env.run()
    assert env.placed == [] and s["mirror_only_count"] == 3


def test_corrupt_ledger_blocks_live_but_the_cron_and_mirror_survive(env):
    env.sc["enabled"] = True
    Path(env.sc["decay_tripwire"]["state_file"]).write_text("{not json")
    s = env.run()
    assert env.placed == [] and s["mirror_only_count"] == 3


def test_net_only_ledger_rows_do_not_break_the_gate(env):
    env.sc["enabled"] = True
    env.write_ledger([{"net_pnl_inr": 0.0, "ts_iso": "2026-11-03T09:30:00"}])
    env.run()
    assert [p["symbol"] for p in env.placed] == ["NSE:THIN"]
    # ...and they still count toward the monthly stop
    assert not oh.canary_gate(CANARY, [{"net_pnl_inr": -3100.0, "ts_iso": "2026-11-03T09:30:00"}],
                              date(2026, 11, 20))[0]


@pytest.mark.parametrize("adv, taken", [(120_000, True), (80_000, False)])
def test_participation_cap_uses_the_canary_floor(env, adv, taken):
    """10% of ADV: Rs1.2L ADV caps the order at ~Rs12k (above the Rs10k canary floor:
    shrunk, taken); Rs0.8L caps it at Rs8k (skipped)."""
    env.sc["enabled"] = True
    env.write_ledger([])
    env.sc["participation_cap"] = {"enabled": True, "max_participation_pct": 0.1,
                                   "skip_when_adv_missing": True,
                                   "min_notional_after_cap_inr": 25000,
                                   "min_notional_after_cap_inr_cnc": 25000,
                                   "fallback_to_cnc_below_mtf_floor": False,
                                   "max_tick_pct_of_price": 0.2, "min_qty_after_cap": 1}
    cands = env.cand_path
    data = json.loads(cands.read_text())
    for c in data["candidates"]:
        if c["symbol"] == "NSE:THIN":
            c["adv_inr"] = adv
    cands.write_text(json.dumps(data))
    env.run()
    if taken:
        (p,) = env.placed
        assert 10_000 <= p["qty"] * p["limit_price"] <= 0.1 * adv
    else:
        assert env.placed == []


def test_live_test_is_full_size_live(env):
    env.sc["enabled"] = True
    env.sc["live_canary"]["enabled"] = False
    env.sc["live_test"]["enabled"] = True
    env.write_ledger([])
    env.run()
    assert sorted(p["symbol"] for p in env.placed) == ["NSE:NOADV", "NSE:THICK", "NSE:THIN"]
    assert {p["product"] for p in env.placed} == {"MTF"}


def test_paper_mode_ignores_the_canary(env, monkeypatch):
    env.sc["enabled"] = True
    monkeypatch.setattr(oh, "_paper_fill_price_entry", lambda broker, sym, d: 100.0)
    oh.run_entry(env.cfg, MagicMock(_data_sdk=None), now_ist=NOW, paper_mode=True)
    assert len(env.placed) == 3 and {p["product"] for p in env.placed} == {"MTF"}


# ---- the gate itself ------------------------------------------------------------

def _gate(trades, today=date(2026, 11, 20)):
    return oh.canary_gate(CANARY, trades, today)


def test_gate_monthly_stop_is_calendar_month_to_date():
    big = lambda pnl: _trade("2026-11-05T09:16:00", pnl, qty=10_000)   # tiny % so the SPRT stays quiet
    assert not _gate([big(-3000)])[0]
    assert _gate([big(-2999)])[0]
    assert _gate([big(-3000)], today=date(2026, 12, 1))[0]             # a new month starts clean


def test_gate_sprt_thresholds():
    lo, hi = math.log(0.2 / 0.9), math.log(0.8 / 0.1)
    # each zero-return trade moves the LLR by -mu1^2/(2 sd^2) = -0.0168: 'no edge' after 90
    zero = lambda n: [_trade(f"2026-11-{2 + i % 18:02d}T09:16:00", 0.0) for i in range(n)]
    step = -(0.44 ** 2) / (2 * 2.4 ** 2)
    assert oh.canary_sprt_llr([0.0] * 10, SPRT) == pytest.approx(10 * step)
    n_h0 = math.ceil(lo / step)
    ok, why = _gate(zero(n_h0))
    assert not ok and "no edge" in why
    ok, why = _gate(zero(n_h0 - 1))
    assert ok and "running" in why
    # a run of +1.0% trades accepts 'edge' (allowed, logged for a manual decision)
    win = [_trade(f"2026-11-{2 + i % 18:02d}T09:16:00", 150.0) for i in range(60)]
    assert oh.canary_sprt_llr([1.0] * 60, SPRT) >= hi
    ok, why = oh.canary_gate({**CANARY, "monthly_stop_inr": -1e9}, win, date(2026, 11, 20))
    assert ok and "accepted 'edge'" in why


def test_real_config_is_live_cnc_only_at_full_size_under_the_live_test():
    cfg = json.loads((_REPO_ROOT / "config" / "configuration.json").read_text(encoding="utf-8"))
    sc = cfg["setups"][SETUP]
    c = sc["live_canary"]
    # live stopped by the operator 2026-11-04; the paper mirror keeps running
    assert sc["enabled"] is False and sc["paper_enabled"] is True
    assert sc["live_products_allowed"] == ["CNC"]
    assert c["enabled"] is False and c["start_date"] is None
    lt, tw = sc["live_test"], sc["decay_tripwire"]
    assert lt["enabled"] is True and lt["monthly_stop_inr"] == -5000 and lt["sprt"] == LIVE_TEST_SPRT
    # the test counts settles after start_date; earlier entries route to the archive
    assert lt["start_date"] == tw["archive_entries_before"] and lt["state_file"]
    assert tw["archive_label"] and tw["archive_regime"] and tw["sustained_weeks"] == 6
    assert c["max_adv_inr"] == 2_500_000 and 10_000 <= c["min_notional_inr"] <= c["notional_inr"] <= 15_000
    assert c["sprt"] == SPRT and c["monthly_stop_inr"] == -3000


# ---- live product allowlist ---------------------------------------------------------

class _MixedDetector(_Detector):
    """THIN is not on the MTF list (CNC); THICK and NOADV are MTF names."""
    def detect(self, ctx):
        prod = "CNC" if ctx == "NSE:THIN" else "MTF"
        oh.logger.info("close_dn_overnight_long fired | symbol=%s product=%s", ctx, prod)
        evt = SimpleNamespace(context={"product": prod, "leverage": 1.0 if prod == "CNC" else 3.0})
        return SimpleNamespace(structure_detected=True, events=[evt])


def test_cnc_only_live_skips_mtf_fires_and_keeps_full_size(env, monkeypatch):
    monkeypatch.setattr(sr, "_import_path", lambda p: _MixedDetector)
    env.sc["enabled"] = True
    env.sc["live_canary"]["enabled"] = False
    env.sc["live_test"]["enabled"] = True
    env.sc["live_products_allowed"] = ["CNC"]
    env.write_ledger([])
    s = env.run()
    assert [(p["symbol"], p["product"]) for p in env.placed] == [("NSE:THIN", "CNC")]
    assert env.placed[0]["qty"] == 750                  # the plan's full size, not the canary's
    assert s["product_skipped"] == 2
    slots = json.loads(Path(env.sc["capital_allocation"]["state_file"]).read_text())["slots"]
    assert [x["symbol"] for x in slots if x["status"] != "free"] == ["NSE:THIN"]


def test_skipped_mtf_fire_does_not_use_up_the_daily_cap(env, monkeypatch):
    monkeypatch.setattr(sr, "_import_path", lambda p: _MixedDetector)
    env.sc["enabled"] = True
    env.sc["live_canary"]["enabled"] = False
    env.sc["live_test"]["enabled"] = True
    env.sc["live_products_allowed"] = ["CNC"]
    env.sc["capital_allocation"]["max_new_positions_per_day"] = 1
    env.write_ledger([])
    env.run()
    assert [p["symbol"] for p in env.placed] == ["NSE:THIN"]


@pytest.mark.parametrize("bad", [None, "CNC", [], ["MIS"]])
def test_unusable_live_products_blocks_live_but_keeps_the_mirror(env, bad):
    env.sc["enabled"] = True
    env.sc["live_canary"]["enabled"] = False
    env.sc["live_test"]["enabled"] = True
    if bad is None:
        del env.sc["live_products_allowed"]
    else:
        env.sc["live_products_allowed"] = bad
    env.write_ledger([])
    s = env.run()
    assert env.placed == [] and s["mirror_only_count"] == 3


def test_paper_mode_ignores_the_product_allowlist(env, monkeypatch):
    monkeypatch.setattr(sr, "_import_path", lambda p: _MixedDetector)
    monkeypatch.setattr(oh, "_paper_fill_price_entry", lambda broker, sym, d: 100.0)
    env.sc["live_products_allowed"] = ["CNC"]
    oh.run_entry(env.cfg, MagicMock(_data_sdk=None), now_ist=NOW, paper_mode=True)
    assert len(env.placed) == 3


# ---- full-size live test (live_test) ------------------------------------------------

def test_no_live_mandate_blocks_live_but_keeps_the_mirror(env, caplog):
    env.sc["enabled"] = True
    env.sc["live_canary"]["enabled"] = False
    env.sc["live_test"]["enabled"] = False
    env.write_ledger([])
    with caplog.at_level(logging.CRITICAL, logger=oh.logger.name):
        s = env.run()
    assert env.placed == [] and s["mirror_only_count"] == 3
    assert "neither live_canary nor live_test" in caplog.text


def test_both_mandates_on_blocks_live(env):
    env.sc["enabled"] = True
    env.sc["live_test"]["enabled"] = True            # the canary is on in the fixture
    env.write_ledger([])
    s = env.run()
    assert env.placed == [] and s["mirror_only_count"] == 3


@pytest.mark.parametrize("bad", [None, {"monthly_stop_inr": 5000}, {"start_date": None},
                                 {"sprt": {**LIVE_TEST_SPRT, "alpha": 0}}])
def test_unusable_live_test_blocks_live_but_keeps_the_mirror(env, bad):
    env.sc["enabled"] = True
    env.sc["live_canary"]["enabled"] = False
    if bad is None:
        del env.sc["live_test"]
    else:
        env.sc["live_test"].update({"enabled": True, **bad})
    env.write_ledger([])
    s = env.run()
    assert env.placed == [] and s["mirror_only_count"] == 3


def test_live_test_monthly_stop_blocks_live(env):
    env.sc["enabled"] = True
    env.sc["live_canary"]["enabled"] = False
    env.sc["live_test"]["enabled"] = True
    env.write_ledger([_trade("2026-11-03T09:16:00", -3000, qty=10_000),
                      _trade("2026-11-04T09:16:00", -2000, qty=10_000)])
    s = env.run()
    assert env.placed == [] and "monthly stop" in s["live_test"][SETUP]


def test_live_test_ignores_settles_on_or_before_its_start(env):
    env.sc["enabled"] = True
    env.sc["live_canary"]["enabled"] = False
    env.sc["live_test"]["enabled"] = True
    env.write_ledger([_trade("2026-11-02T09:16:00", -9000), _trade("2026-10-30T09:16:00", -9000)])
    env.run()
    assert len(env.placed) == 3


def test_sprt_no_edge_is_final_even_if_later_settles_recover():
    lt = {**LIVE_TEST, "monthly_stop_inr": -1e9}
    step = -(0.35 ** 2) / (2 * 2.2 ** 2)
    n_h0 = math.ceil(math.log(0.2 / 0.9) / step)
    rows = [_trade(f"2026-11-{3 + i % 25:02d}T09:16:00", 0.0) for i in range(n_h0)]
    ok, why = oh.canary_gate(lt, rows, date(2026, 12, 20))
    assert not ok and "no edge" in why
    # positions still open when entries stopped settle as big winners: the total LLR
    # climbs back inside the band, but the decided test stays decided
    later = rows + [_trade("2026-12-15T09:16:00", 1500.0)] * 3
    assert oh.canary_sprt_llr([0.0] * n_h0 + [10.0] * 3, LIVE_TEST_SPRT) > math.log(0.2 / 0.9)
    ok, why = oh.canary_gate(lt, later, date(2026, 12, 20))
    assert not ok and "no edge" in why


def test_sprt_decision_reports_the_first_crossing():
    dec, n, llr, lo, hi = oh.sprt_decision([3.0] * 50, LIVE_TEST_SPRT)
    assert dec == "edge" and n < 50 and llr >= hi
    assert oh.sprt_decision([0.1, -0.1], LIVE_TEST_SPRT)[0] is None


# ---- live test state, integrity checks ---------------------------------------------

def _live_test_on(env):
    env.sc["enabled"] = True
    env.sc["live_canary"]["enabled"] = False
    env.sc["live_test"]["enabled"] = True


def test_live_test_without_its_state_file_blocks_live(env):
    _live_test_on(env)
    Path(env.sc["live_test"]["state_file"]).unlink()
    env.write_ledger([])
    s = env.run()
    assert env.placed == [] and "FileNotFoundError" in s["live_test"][SETUP]


def test_a_changed_registered_parameter_blocks_live(env):
    _live_test_on(env)
    env.sc["live_test"]["sprt"] = {**LIVE_TEST_SPRT, "mu1_pct": 0.30}
    env.write_ledger([])
    s = env.run()
    assert env.placed == [] and "registered parameters changed" in s["live_test"][SETUP]


def test_a_start_date_after_today_blocks_live(env):
    _live_test_on(env)
    env.sc["live_test"]["start_date"] = "2026-11-05"
    Path(env.sc["live_test"]["state_file"]).write_text(json.dumps(oh.new_live_test_state(SETUP, env.sc["live_test"])))
    env.write_ledger([])
    s = env.run()
    assert env.placed == [] and "after today" in s["live_test"][SETUP]


def test_start_date_must_be_the_archive_boundary(env):
    _live_test_on(env)
    env.sc["decay_tripwire"].update(archive_entries_before="2026-11-03", archive_label="pre",
                                    archive_regime="before the test")
    env.write_ledger([])
    s = env.run()                                    # start 11-02 vs boundary 11-03
    assert env.placed == [] and "must equal" in s["live_test"][SETUP]


def test_a_non_finite_ledger_row_blocks_live(env):
    _live_test_on(env)
    env.write_ledger([_trade("2026-11-03T09:16:00", float("nan"))])
    s = env.run()
    assert env.placed == [] and "non-finite" in s["live_test"][SETUP]


def test_a_failed_ledger_write_blocks_live_until_the_marker_is_cleared(env):
    _live_test_on(env)
    env.write_ledger([])
    marker = Path(env.sc["decay_tripwire"]["state_file"] + ".WRITE_FAILED")
    marker.write_text("{}\n")
    s = env.run()
    assert env.placed == [] and "missing from the ledger" in s["live_test"][SETUP]
    marker.unlink()
    env.run()
    assert len(env.placed) == 3


def test_sprt_state_survives_the_ledger_trim(tmp_path):
    """Reviewer case: an undecided test whose oldest rows the ledger trims away. Read
    from the trimmed file the test would wrongly accept 'edge'; the state file keeps
    the true walk (n=201, still undecided)."""
    from datetime import timedelta
    from services.risk.decay_tripwire import DecayTripwire
    lt = dict(LIVE_TEST, enabled=True, monthly_stop_inr=-1e9, state_file=str(tmp_path / "st.json"))
    Path(lt["state_file"]).write_text(json.dumps(oh.new_live_test_state(SETUP, lt)))
    ledger = tmp_path / "led.json"
    tw = DecayTripwire(setup_name=SETUP, state_path=ledger, window_trades=30, pf_floor=1.2,
                       sustained_weeks=6, archive_entries_before=None, archive_label=None,
                       archive_regime=None)
    rets = [-0.2] * 51 + [0.4] * 150                 # LLR -1.38 after 51, +1.06 after 201
    day0 = date(2026, 11, 3)
    for i in range(0, len(rets), 6):                 # six settles a morning, gate at 15:26
        d = day0 + timedelta(days=i // 6)
        for r in rets[i:i + 6]:
            tw.record_trade(net_pnl_inr=r * 250.0, ts_iso=f"{d.isoformat()}T09:16:00",
                            entry_price=100.0, qty=250, entry_date=(d - timedelta(days=1)).isoformat())
        trades = json.loads(ledger.read_text())["trades"]
        ok, why, dec = oh.live_test_gate(SETUP, lt, trades, d)
    assert len(trades) < len(rets)                   # the ledger did trim
    assert oh.sprt_decision(rets, LIVE_TEST_SPRT)[0] is None
    assert ok and dec is None and "running: 201 live trades" in why
    assert oh.canary_gate(lt, trades, d)[1].startswith("SPRT accepted 'edge'")   # the stateless read
    st = json.loads(Path(lt["state_file"]).read_text())
    assert st["n"] == 201 and st["llr"] == pytest.approx(oh.canary_sprt_llr(rets, LIVE_TEST_SPRT))


def test_a_decided_test_stays_decided_in_the_state(tmp_path):
    lt = dict(LIVE_TEST, enabled=True, monthly_stop_inr=-1e9, state_file=str(tmp_path / "st.json"))
    Path(lt["state_file"]).write_text(json.dumps(oh.new_live_test_state(SETUP, lt)))
    losers = [_trade(f"2026-11-{4 + i // 6:02d}T09:{16 + i % 6:02d}:00", -250.0, qty=250) for i in range(60)]
    ok, why, dec = oh.live_test_gate(SETUP, lt, losers, date(2026, 11, 20))
    assert not ok and dec == "no edge"
    winners = losers + [_trade("2026-11-25T09:16:00", 5000.0, qty=250)] * 5
    ok, why, dec = oh.live_test_gate(SETUP, lt, winners, date(2026, 11, 25))
    assert not ok and dec == "no edge"


def test_alpha_plus_beta_must_stay_below_one(env):
    _live_test_on(env)
    env.sc["live_test"]["sprt"] = {**LIVE_TEST_SPRT, "alpha": 0.6, "beta": 0.5}
    env.write_ledger([])
    s = env.run()
    assert env.placed == [] and "alpha + beta < 1" in s["live_test"][SETUP]
