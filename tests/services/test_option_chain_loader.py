"""option_chain_loader tests (sub8 plan #6 Phase A4 + A5).

Eight tests covering the loader API:
  1. load_oi_snapshot returns a DataFrame with the canonical schema.
  2. find_max_oi_strike returns the correct strike for NIFTY weekly.
  3. is_expiry_day is delegated from universe_filter (no duplication).
  4. load_oi_snapshot raises OISnapshotMissing on a missing session file.
  5. Cached calls are sub-100ms (LRU performance gate).
  6. Tied OI strikes break deterministically (lower strike wins).
  7. is_monthly_expiry distinguishes monthly from weekly Thursdays.
  8. oi_root override redirects reads to a custom parquet root.
"""
from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from services import option_chain_loader as ocl


@pytest.fixture(autouse=True)
def _reset_cache():
    """Clear the module-level LRU before each test so perf tests are honest."""
    ocl.clear_cache()
    yield
    ocl.clear_cache()


def _write_oi_parquet(tmp_root: Path, session_date: date, rows: list[dict]) -> Path:
    """Helper: write a parquet at the canonical
    <root>/<YYYY>/<MM>/<YYYY-MM-DD>.parquet path."""
    out = (
        tmp_root
        / f"{session_date.year:04d}"
        / f"{session_date.month:02d}"
        / f"{session_date.isoformat()}.parquet"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    return out


def _make_chain(
    session_date: date,
    expiry: date,
    symbol: str,
    strikes_oi: dict[float, tuple[int, int]],
) -> list[dict]:
    """Build canonical-schema rows for one (symbol, expiry).
    strikes_oi maps strike -> (CE_oi, PE_oi)."""
    out = []
    for strike, (ce_oi, pe_oi) in strikes_oi.items():
        out.append({
            "session_date": session_date,
            "symbol": symbol,
            "expiry_date": expiry,
            "strike": float(strike),
            "option_type": "CE",
            "oi": int(ce_oi),
            "oi_change": 0,
            "vol": 0,
            "ltp": 1.0,
            "settlement_price": 1.0,
            "iv": pd.NA,
        })
        out.append({
            "session_date": session_date,
            "symbol": symbol,
            "expiry_date": expiry,
            "strike": float(strike),
            "option_type": "PE",
            "oi": int(pe_oi),
            "oi_change": 0,
            "vol": 0,
            "ltp": 1.0,
            "settlement_price": 1.0,
            "iv": pd.NA,
        })
    return out


# ---------------------------------------------------------------------------
# 1. load_oi_snapshot returns a DataFrame with canonical schema
# ---------------------------------------------------------------------------
def test_load_oi_snapshot_returns_canonical_schema(tmp_path: Path):
    sd = date(2024, 6, 6)
    rows = _make_chain(sd, sd, "NIFTY", {23000: (10_000, 5_000)})
    _write_oi_parquet(tmp_path, sd, rows)

    df = ocl.load_oi_snapshot(sd, oi_root=tmp_path)

    expected_cols = {
        "session_date", "symbol", "expiry_date", "strike", "option_type",
        "oi", "oi_change", "vol", "ltp", "settlement_price", "iv",
    }
    assert expected_cols <= set(df.columns)
    assert len(df) == 2  # 1 strike × {CE, PE}
    assert set(df["option_type"].unique()) == {"CE", "PE"}


# ---------------------------------------------------------------------------
# 2. find_max_oi_strike returns correct strike for NIFTY weekly
# ---------------------------------------------------------------------------
def test_find_max_oi_strike_returns_argmax(tmp_path: Path):
    sd = date(2024, 6, 6)  # Thursday — weekly NIFTY expiry pre-2025-09
    # Pin should clearly be at 23200 (highest CE+PE OI sum)
    rows = _make_chain(sd, sd, "NIFTY", {
        23000: (5_000, 4_000),    # sum 9_000
        23100: (10_000, 8_000),   # sum 18_000
        23200: (50_000, 60_000),  # sum 110_000  <-- argmax
        23300: (15_000, 12_000),  # sum 27_000
    })
    _write_oi_parquet(tmp_path, sd, rows)

    strike = ocl.find_max_oi_strike(sd, symbol="NIFTY", expiry="weekly",
                                    oi_root=tmp_path)
    assert strike == 23200.0


# ---------------------------------------------------------------------------
# 3. is_expiry_day is the universe_filter export (no duplication)
# ---------------------------------------------------------------------------
def test_is_expiry_day_is_universe_filter_export():
    """Sanity: ocl.is_expiry_day is the same callable as
    services.universe_filter.is_expiry_day — proves no copy."""
    from services.universe_filter import is_expiry_day as canonical
    assert ocl.is_expiry_day is canonical
    # And it actually works on a known Thursday pre-2025-09
    assert ocl.is_expiry_day(date(2024, 6, 6)) is True
    assert ocl.is_expiry_day(date(2024, 6, 5)) is False  # Wednesday


# ---------------------------------------------------------------------------
# 4. raises OISnapshotMissing on missing session file
# ---------------------------------------------------------------------------
def test_load_oi_snapshot_raises_on_missing_file(tmp_path: Path):
    sd = date(2024, 6, 6)
    # No file written
    with pytest.raises(ocl.OISnapshotMissing):
        ocl.load_oi_snapshot(sd, oi_root=tmp_path)
    # And it's a FileNotFoundError subclass for callers using broader except
    assert issubclass(ocl.OISnapshotMissing, FileNotFoundError)


# ---------------------------------------------------------------------------
# 5. Cached calls under 100ms (perf gate — proves LRU works)
# ---------------------------------------------------------------------------
def test_cached_load_under_100ms(tmp_path: Path):
    sd = date(2024, 6, 6)
    # Fairly large frame so the first read is appreciable, second hit is cache
    rows = _make_chain(sd, sd, "NIFTY",
                       {float(k): (1_000, 1_000) for k in range(20_000, 26_000, 50)})
    _write_oi_parquet(tmp_path, sd, rows)

    # Prime the cache
    ocl.load_oi_snapshot(sd, oi_root=tmp_path)

    t0 = time.perf_counter()
    for _ in range(10):
        ocl.load_oi_snapshot(sd, oi_root=tmp_path)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    # 10 cache hits should average well under 100ms total.
    assert elapsed_ms < 100, f"cached reads too slow: {elapsed_ms:.1f}ms for 10 hits"


# ---------------------------------------------------------------------------
# 6. Tied strikes break deterministically — lower strike wins
# ---------------------------------------------------------------------------
def test_tied_oi_strikes_break_to_lower_strike(tmp_path: Path):
    sd = date(2024, 6, 6)
    # Two strikes with identical CE+PE totals
    rows = _make_chain(sd, sd, "NIFTY", {
        23200: (50_000, 50_000),  # sum 100_000
        23300: (40_000, 60_000),  # sum 100_000  -- tied
        23100: (10_000, 10_000),  # smaller
    })
    _write_oi_parquet(tmp_path, sd, rows)

    strike = ocl.find_max_oi_strike(sd, symbol="NIFTY", expiry="weekly",
                                    oi_root=tmp_path)
    assert strike == 23200.0  # lower wins


# ---------------------------------------------------------------------------
# 7. is_monthly_expiry: distinguishes monthly from weekly Thursdays
# ---------------------------------------------------------------------------
def test_is_monthly_expiry_separates_monthly_from_weekly():
    # 2024-06-27 is the LAST Thursday of June 2024 — monthly expiry.
    # 2024-06-06 is the FIRST Thursday of June 2024 — weekly only.
    assert ocl.is_monthly_expiry(date(2024, 6, 27)) is True
    assert ocl.is_monthly_expiry(date(2024, 6, 6)) is False
    # A non-expiry day is never monthly.
    assert ocl.is_monthly_expiry(date(2024, 6, 5)) is False


# ---------------------------------------------------------------------------
# 8. oi_root override redirects reads
# ---------------------------------------------------------------------------
def test_oi_root_override_redirects_reads(tmp_path: Path):
    """Two separate roots → loader reads from whichever path was passed."""
    sd = date(2024, 6, 6)
    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"

    rows_a = _make_chain(sd, sd, "NIFTY", {23000: (10_000, 10_000)})
    rows_b = _make_chain(sd, sd, "NIFTY", {24000: (50_000, 50_000)})
    _write_oi_parquet(root_a, sd, rows_a)
    _write_oi_parquet(root_b, sd, rows_b)

    pin_a = ocl.find_max_oi_strike(sd, "NIFTY", "weekly", oi_root=root_a)
    pin_b = ocl.find_max_oi_strike(sd, "NIFTY", "weekly", oi_root=root_b)
    assert pin_a == 23000.0
    assert pin_b == 24000.0
