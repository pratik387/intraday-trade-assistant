"""tools/multiday_shadow_eval.py rebuilds a candidate's return as production
trades it: buy the open of the session after the signal, sell the close
hold_days sessions after that (exit_on_date = entry_date + hold_days trading
days). Before 2026-11-03 it sold one session early."""
import sys
import types

import pandas as pd
import pytest

from tools import multiday_shadow_eval as se


@pytest.fixture
def fwd(monkeypatch):
    idx = pd.to_datetime(["2026-10-29", "2026-10-30", "2026-11-02", "2026-11-03", "2026-11-04"])
    df = pd.DataFrame({"open": [10.0, 100.0, 1.0, 1.0, 1.0],
                       "close": [10.0, 101.0, 102.0, 110.0, 120.0]}, index=idx)

    class FakeSdk:
        def get_daily(self, sym, days):
            return df
    fake = types.ModuleType("broker.upstox.upstox_data_client")
    fake.UpstoxDataClient = FakeSdk
    monkeypatch.setitem(sys.modules, "broker.upstox.upstox_data_client", fake)
    return se.build_return_fn(["X"], None, None)


def test_exit_is_hold_days_sessions_after_entry(fwd):
    # signal 10-29 -> entry open 10-30 (100) -> hold 2 -> exit close 11-03 (110)
    assert fwd("X", pd.Timestamp("2026-10-29").date(), 2) == pytest.approx(10.0)


def test_no_return_when_exit_session_is_not_in_the_data(fwd):
    assert fwd("X", pd.Timestamp("2026-11-02").date(), 2) is None
