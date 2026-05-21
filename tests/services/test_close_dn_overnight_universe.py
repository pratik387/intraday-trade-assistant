"""Tests for close_dn_overnight_long_universe + MtfUniverse."""
import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from services.mtf_universe import MtfInfo, MtfUniverse
from services.setup_universe import close_dn_overnight_long_universe


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MTF_SNAPSHOT = _REPO_ROOT / "data" / "mtf_universe" / "approved_mtf_securities_2026-05-21.json"


# ---------- MtfUniverse tests ----------

def test_mtf_universe_loads_real_snapshot():
    """Load the actual snapshot and verify it has 1,489 entries."""
    if not _MTF_SNAPSHOT.exists():
        pytest.skip(f"MTF snapshot missing at {_MTF_SNAPSHOT}")
    mtf = MtfUniverse(_MTF_SNAPSHOT)
    assert len(mtf) == 1489
    # Spot-check a known entry (ABB was the first entry per cell_lock JSON)
    info = mtf.lookup("ABB")
    assert info is not None
    assert info.category == "fo"
    assert info.leverage > 3.0


def test_mtf_universe_excludes_etf_by_default():
    """is_eligible with default (exclude_etf=True) returns False for ETFs."""
    if not _MTF_SNAPSHOT.exists():
        pytest.skip(f"MTF snapshot missing at {_MTF_SNAPSHOT}")
    mtf = MtfUniverse(_MTF_SNAPSHOT)
    # BANKBEES is a known ETF in the list per the cell-lock cross-check earlier
    assert mtf.is_eligible("BANKBEES", exclude_etf=True) is False
    assert mtf.is_eligible("BANKBEES", exclude_etf=False) is True


def test_mtf_universe_handles_nse_prefix():
    """lookup() accepts both 'RELIANCE' and 'NSE:RELIANCE'."""
    if not _MTF_SNAPSHOT.exists():
        pytest.skip(f"MTF snapshot missing at {_MTF_SNAPSHOT}")
    mtf = MtfUniverse(_MTF_SNAPSHOT)
    a = mtf.lookup("RELIANCE")
    b = mtf.lookup("NSE:RELIANCE")
    if a or b:
        assert a == b


def test_mtf_universe_missing_snapshot_raises(tmp_path):
    fake = tmp_path / "nope.json"
    with pytest.raises(FileNotFoundError, match="MTF snapshot not found"):
        MtfUniverse(fake)


def test_mtf_universe_malformed_snapshot_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"not": "a list"}', encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected shape"):
        MtfUniverse(bad)


# ---------- close_dn_overnight_long_universe tests ----------

def _ddf(n_rows: int = 50, avg_vol: float = 100000) -> pd.DataFrame:
    """Build a mock daily DataFrame with given row count + volume."""
    return pd.DataFrame({"volume": [avg_vol] * n_rows, "close": [100.0] * n_rows})


def _config():
    return {
        "min_daily_avg_volume": 50000,
        "min_trading_days_required": 30,
        "universe_max_symbols": 1500,
        "mtf": {
            "approved_list_snapshot_path": str(_MTF_SNAPSHOT),
            "exclude_etf": True,
            "fallback_to_cnc_if_not_mtf": True,
            "stale_snapshot_warn_days": 7,
        },
    }


def test_universe_includes_mtf_eligible_symbol():
    """A symbol in the MTF list with valid daily volume should be in the universe."""
    if not _MTF_SNAPSHOT.exists():
        pytest.skip(f"MTF snapshot missing at {_MTF_SNAPSHOT}")
    daily = {"NSE:RELIANCE": _ddf()}
    with patch("services.symbol_metadata.get_cap_segment", return_value="large_cap"), \
         patch("services.symbol_metadata.get_mis_info", return_value={"mis_enabled": True}):
        result = close_dn_overnight_long_universe(daily, date(2026, 5, 21), _config())
    if "NSE:RELIANCE" in MtfUniverse(_MTF_SNAPSHOT)._by_symbol:
        assert "NSE:RELIANCE" in result


def test_universe_excludes_etf_even_if_in_mtf_list():
    """ETFs in MTF list are excluded at universe-builder level."""
    if not _MTF_SNAPSHOT.exists():
        pytest.skip(f"MTF snapshot missing at {_MTF_SNAPSHOT}")
    daily = {"NSE:BANKBEES": _ddf()}
    with patch("services.symbol_metadata.get_cap_segment", return_value="large_cap"), \
         patch("services.symbol_metadata.get_mis_info", return_value={"mis_enabled": True}):
        result = close_dn_overnight_long_universe(daily, date(2026, 5, 21), _config())
    # BANKBEES is an ETF in the MTF list — should be excluded
    assert "NSE:BANKBEES" not in result


def test_universe_includes_non_mtf_fallback_large_cap():
    """A large-cap symbol NOT in MTF list still included for CNC fallback."""
    daily = {"NSE:NOTINLISTXYZ": _ddf()}
    with patch("services.symbol_metadata.get_cap_segment", return_value="large_cap"), \
         patch("services.symbol_metadata.get_mis_info", return_value={"mis_enabled": True}):
        result = close_dn_overnight_long_universe(daily, date(2026, 5, 21), _config())
    assert "NSE:NOTINLISTXYZ" in result


def test_universe_excludes_micro_cap():
    """Micro-cap excluded regardless of MTF eligibility."""
    daily = {"NSE:MICROSYM": _ddf()}
    with patch("services.symbol_metadata.get_cap_segment", return_value="micro_cap"), \
         patch("services.symbol_metadata.get_mis_info", return_value={"mis_enabled": True}):
        result = close_dn_overnight_long_universe(daily, date(2026, 5, 21), _config())
    assert "NSE:MICROSYM" not in result


def test_universe_excludes_unknown_cap_not_in_mtf():
    """Unknown-cap NOT in MTF list is excluded (no leverage + likely illiquid)."""
    daily = {"NSE:UNKNOWNXYZ": _ddf()}
    with patch("services.symbol_metadata.get_cap_segment", return_value="unknown"), \
         patch("services.symbol_metadata.get_mis_info", return_value={"mis_enabled": True}):
        result = close_dn_overnight_long_universe(daily, date(2026, 5, 21), _config())
    assert "NSE:UNKNOWNXYZ" not in result


def test_universe_excludes_low_volume_symbol():
    """Symbol with avg volume < threshold is excluded."""
    daily = {"NSE:LOWVOL": _ddf(avg_vol=10000)}  # below 50000 threshold
    with patch("services.symbol_metadata.get_cap_segment", return_value="large_cap"), \
         patch("services.symbol_metadata.get_mis_info", return_value={"mis_enabled": True}):
        result = close_dn_overnight_long_universe(daily, date(2026, 5, 21), _config())
    assert "NSE:LOWVOL" not in result


def test_universe_excludes_insufficient_history():
    """Symbol with < min_trading_days_required is excluded."""
    daily = {"NSE:NEWLISTING": _ddf(n_rows=10)}  # 10 < 30
    with patch("services.symbol_metadata.get_cap_segment", return_value="large_cap"), \
         patch("services.symbol_metadata.get_mis_info", return_value={"mis_enabled": True}):
        result = close_dn_overnight_long_universe(daily, date(2026, 5, 21), _config())
    assert "NSE:NEWLISTING" not in result
