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
