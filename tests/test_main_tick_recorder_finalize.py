"""The session's tick recorder is flushed periodically and finalized at EOD.

Until 2026-11-02 _run_until_eod looked the recorder up with locals(), where it
does not exist, so it was never finalized: only full 50k-tick part files reached
disk (live 10-30 recorded nothing; the HUHTAMAKI 10-28 tape after 13:24 is lost).
"""
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import main


class _Recorder:
    tick_count = 7

    def __init__(self):
        self.flushes, self.finalized = 0, 0

    def flush(self):
        self.flushes += 1

    def finalize(self):
        self.finalized += 1


def _session(polls_before_exit):
    screener = SimpleNamespace(_request_exit=False, start=lambda: None, stop=lambda: None,
                               subs=None, symbol_map={}, _eod_done=True)
    left = {"n": polls_before_exit}

    def _sleep(_):
        left["n"] -= 1
        if left["n"] <= 0:
            screener._request_exit = True
    trader = SimpleNamespace(_lock=threading.Lock(), pending_trades={}, stop=lambda: None)
    positions = SimpleNamespace(list_open=lambda: {})
    return screener, trader, positions, _sleep


def test_recorder_is_finalized_at_eod_and_flushed_on_its_interval(monkeypatch):
    screener, trader, positions, _sleep = _session(polls_before_exit=5)
    clock = {"t": 0.0}

    def _sleep_and_tick(s):
        clock["t"] += 1.0
        _sleep(s)
    monkeypatch.setattr(main.time, "sleep", _sleep_and_tick)
    monkeypatch.setattr(main.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(main.signal, "signal", lambda *a: None)
    rec = _Recorder()
    main._run_until_eod(screener, MagicMock(), trader, None, None, positions,
                        tick_recorder=rec, tick_flush_sec=2.0)
    assert rec.finalized == 1
    assert rec.flushes == 2                     # at t=2 and t=4 of the 5 polls


def test_no_recorder_is_fine(monkeypatch):
    screener, trader, positions, _sleep = _session(polls_before_exit=1)
    monkeypatch.setattr(main.time, "sleep", _sleep)
    monkeypatch.setattr(main.signal, "signal", lambda *a: None)
    main._run_until_eod(screener, MagicMock(), trader, None, None, positions)


def test_code_version_names_the_commit_and_config():
    v = main._code_version()
    assert v.startswith("sha=") and "dirty=" in v and "config_sha256=" in v


def test_make_tick_recorder_validates_the_flush_interval(tmp_path, monkeypatch):
    import pytest
    import market_data.tick_recorder as tr
    monkeypatch.setattr(tr, "DATA_DIR", tmp_path)
    assert main._make_tick_recorder({"enabled": False, "buffer_size": 10, "flush_interval_sec": 300}, "live") == (None, None)
    with pytest.raises(ValueError, match="flush_interval_sec"):
        main._make_tick_recorder({"enabled": True, "buffer_size": 10, "flush_interval_sec": 0}, "live")
    rec, sec = main._make_tick_recorder({"enabled": True, "buffer_size": 10, "flush_interval_sec": 300}, "live")
    assert rec is not None and sec == 300.0


def test_a_recorder_without_a_flush_interval_is_refused_before_the_session():
    import pytest
    screener, trader, positions, _sleep = _session(polls_before_exit=1)
    with pytest.raises(ValueError, match="tick_flush_sec"):
        main._run_until_eod(screener, MagicMock(), trader, None, None, positions,
                            tick_recorder=_Recorder(), tick_flush_sec=None)
