from datetime import date
import pandas as pd
from tools.edge_discovery.types import Event
from tools.edge_discovery.features.event_features import EventCalendarFeatures


def test_thursday_is_expiry_day():
    event = Event(symbol="X", event_time=pd.Timestamp("2024-06-13 10:00:00"), metadata={})  # a Thursday
    ef = EventCalendarFeatures()
    out = ef.compute(event, bars=pd.DataFrame())
    assert out["is_expiry_day"] is True
    assert out["is_expiry_week"] is True


def test_monday_not_expiry_but_still_expiry_week():
    event = Event(symbol="X", event_time=pd.Timestamp("2024-06-10 10:00:00"), metadata={})  # Monday before Thu 13th
    ef = EventCalendarFeatures()
    out = ef.compute(event, bars=pd.DataFrame())
    assert out["is_expiry_day"] is False
    assert out["is_expiry_week"] is True


def test_last_thursday_of_month_is_monthly_expiry():
    event = Event(symbol="X", event_time=pd.Timestamp("2024-06-27 10:00:00"), metadata={})  # last Thu of June 2024
    ef = EventCalendarFeatures()
    out = ef.compute(event, bars=pd.DataFrame())
    assert out["is_monthly_expiry_day"] is True


def test_days_to_earnings_when_provided():
    event = Event(symbol="X", event_time=pd.Timestamp("2024-06-15 10:00:00"), metadata={})
    ef = EventCalendarFeatures()
    out = ef.compute(event, bars=pd.DataFrame(), next_earnings_date=date(2024, 6, 20))
    assert out["days_to_next_earnings"] == 5


def test_rbi_policy_day_lookup():
    event = Event(symbol="X", event_time=pd.Timestamp("2024-06-07 10:00:00"), metadata={})  # 7-Jun-2024 was RBI MPC
    ef = EventCalendarFeatures(rbi_policy_dates={date(2024, 6, 7)})
    out = ef.compute(event, bars=pd.DataFrame())
    assert out["is_rbi_policy_day"] is True
