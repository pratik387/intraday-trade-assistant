"""level_pipeline.screen() wide_open_mode bypass (sub5-T1)."""
import pandas as pd

from pipelines.level_pipeline import LevelPipeline


def _make_pipe(wide_open: bool) -> LevelPipeline:
    """Construct a LevelPipeline and inject wide_open_mode into cfg."""
    pipe = LevelPipeline()
    pipe.cfg = dict(pipe.cfg)  # don't mutate the cached config
    pipe.cfg["wide_open_mode"] = wide_open
    return pipe


def _dummy_df5m(ts_str: str = "2025-01-02 12:30") -> pd.DataFrame:
    idx = pd.date_range(ts_str, periods=5, freq="5min")
    return pd.DataFrame({
        "open": [100.0] * 5, "high": [101.0] * 5, "low": [99.0] * 5,
        "close": [100.0] * 5, "volume": [1000] * 5, "vwap": [100.0] * 5,
    }, index=idx)


def test_wide_open_bypasses_screen_at_lunch_block():
    """At 12:30 (lunch block), normal flow rejects with time_window_fail.
    wide_open_mode=true must short-circuit and pass."""
    pipe = _make_pipe(wide_open=True)
    df5m = _dummy_df5m("2025-01-02 12:30")
    result = pipe.screen(
        symbol="NSE:SYM",
        df5m=df5m,
        features={"atr": 1.0},
        levels={},
        now=df5m.index[-1],
    )
    assert result.passed is True
    assert result.reasons == ["wide_open_mode:bypass"]


def test_wide_open_false_preserves_existing_rejections():
    """wide_open_mode=false must keep the existing time_window_fail behavior at 12:30."""
    pipe = _make_pipe(wide_open=False)
    df5m = _dummy_df5m("2025-01-02 12:30")
    result = pipe.screen(
        symbol="NSE:SYM",
        df5m=df5m,
        features={"atr": 1.0},
        levels={},
        now=df5m.index[-1],
    )
    assert result.passed is False
    assert any("time_window_fail" in r for r in result.reasons)
