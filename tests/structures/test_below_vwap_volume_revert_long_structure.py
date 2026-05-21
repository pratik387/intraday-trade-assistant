"""below_vwap_volume_revert_long detector unit tests.

Cell locked at: cap_segment=unknown × vol_ratio_bin=gte_10 × hhmm_bucket=afternoon_1300_1500
Disc PF 1.59 / OOS PF 1.78 / HO PF 1.61 (3+ years, n=3,712 pooled).
Spec: specs/2026-05-21-below_vwap_volume_revert_long-paper-trade-spec.md
"""
import pandas as pd
import pytest

from structures.below_vwap_volume_revert_long_structure import (
    BelowVwapVolumeRevertLongStructure,
)
from structures.data_models import MarketContext


def _cfg():
    return {
        "_setup_name": "below_vwap_volume_revert_long",
        "enabled": False,
        "paper_enabled": True,
        "active_window_start": "10:00",
        "active_window_end": "14:55",
        "vwap_dev_pct_max": -2.0,
        "vol_ratio_min": 3.0,
        "cell_lock_cap_segment": "unknown",
        "cell_lock_vol_ratio_min": 10.0,
        "cell_lock_hhmm_min": "13:00",
        "cell_lock_hhmm_max": "14:55",
        "min_signal_bar_notional_rs": 500000,
        "t1_r_multiple": 1.5,
        "t2_r_multiple": 2.0,
        "sl_buffer_below_bar_low_pct": 0.2,
        "min_stop_pct": 1.0,
        "t1_partial_qty_pct": 0.5,
        "time_stop_at": "14:30",
        "min_bars_required": 12,
        "entry_zone_pct": 0.3,
        "entry_zone_mode": "symmetric",
        "min_stop_distance_pct": 0.5,
    }


def test_detector_instantiates_from_config():
    """Detector accepts the config block without raising."""
    det = BelowVwapVolumeRevertLongStructure(_cfg())
    assert det.structure_type == "below_vwap_volume_revert_long"


def _make_df(*, n_bars=20, hhmm_str="13:30", session_date=None,
             close_seq=None, volume_seq=None,
             open_val=100.0, high_val=100.5, low_val=99.5):
    """Build a session-shape df_5m with n_bars 5-minute bars ending at hhmm_str.

    Bars start at 09:15. close_seq / volume_seq override last-bar values if given.
    """
    if session_date is None:
        session_date = pd.Timestamp("2026-05-20").date()
    base_ts = pd.Timestamp(f"{session_date} 09:15:00")
    timestamps = [base_ts + pd.Timedelta(minutes=5 * i) for i in range(n_bars)]
    # Use closes that produce VWAP near 100 (default) so we have headroom
    closes = [100.0] * n_bars if close_seq is None else list(close_seq)
    vols = [10000] * n_bars if volume_seq is None else list(volume_seq)
    df = pd.DataFrame({
        "open": [open_val] * n_bars,
        "high": [high_val] * n_bars,
        "low": [low_val] * n_bars,
        "close": closes,
        "volume": vols,
    }, index=timestamps)
    # Stamp the requested final hhmm on the last bar
    final_h, final_m = hhmm_str.split(":")
    df.index = df.index[:-1].append(
        pd.DatetimeIndex([pd.Timestamp(f"{session_date} {hhmm_str}:00")])
    )
    return df


def test_rejects_outside_active_window():
    """09:30 bar (before active_window_start=10:00) is rejected."""
    det = BelowVwapVolumeRevertLongStructure(_cfg())
    df = _make_df(n_bars=12, hhmm_str="09:30")  # 12 bars, last stamped 09:30
    ctx = MarketContext(
        symbol="TEST", current_price=100.0, timestamp=df.index[-1],
        df_5m=df, session_date=df.index[-1].date(),
    )
    r = det.detect(ctx)
    assert not r.structure_detected
    assert "active" in (r.rejection_reason or "").lower() or "window" in (r.rejection_reason or "").lower()
