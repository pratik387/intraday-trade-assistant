"""The in-progress Upstox 5m candle can be dropped before detection (2026-11-02, flag off)."""
import io, json
from pathlib import Path

import pandas as pd

from services.screener_live import drop_in_progress_bars


def test_rows_after_the_scanned_bar_are_dropped():
    idx = pd.date_range("2026-10-30 10:30", periods=5, freq="5min")     # 10:30 .. 10:50
    df = pd.DataFrame({"close": range(5)}, index=idx)
    out = drop_in_progress_bars(df, pd.Timestamp("2026-10-30 10:45"))     # 10:50 is forming
    assert list(out.index.strftime("%H:%M")) == ["10:30", "10:35", "10:40", "10:45"]


def test_a_frame_with_only_closed_bars_is_unchanged():
    idx = pd.date_range("2026-10-30 09:15", periods=3, freq="5min")
    df = pd.DataFrame({"close": [1, 2, 3]}, index=idx)
    assert drop_in_progress_bars(df, idx[-1]).equals(df)


def test_flag_is_in_config_and_off():
    cfg = json.load(io.open(Path(__file__).resolve().parents[2] / "config" / "configuration.json",
                            encoding="utf-8"))
    assert cfg["api_5m_bars"]["drop_in_progress_bar"] is False
