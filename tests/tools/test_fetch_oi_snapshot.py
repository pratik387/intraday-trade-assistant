"""fetch_oi_snapshot tests (sub8 plan #6 Phase A2 + A2.3).

Three HTTP-mocked tests for the daily ingestion CLI:
  1. test_ingests_single_day_to_parquet — fixture CSV → parquet at canonical path,
     correct schema.
  2. test_skip_existing_does_not_redownload — pre-existing parquet → no
     download_fn call, returns same path.
  3. test_validation_fails_on_null_oi — corrupted bhavcopy → ValueError raised
     by _validate (before parquet write).

The tests use dependency injection (download_fn / parse_fn args) so we never
actually hit NSE during testing.
"""
from __future__ import annotations

import io
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from tools.option_chain import _nse_bhavcopy_client as bc
from tools.option_chain.fetch_oi_snapshot import (
    _parquet_path,
    ingest_one_session,
)


def _make_legacy_bhavcopy_zip(rows: list[dict]) -> bytes:
    """Build a fake legacy-schema NSE F&O bhavcopy ZIP (returns raw bytes)."""
    df = pd.DataFrame(rows)
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("fo06JUN2024bhav.csv", csv_bytes)
    return zip_buf.getvalue()


def _legacy_rows(n_strikes: int = 110, expiry: str = "06-JUN-2024") -> list[dict]:
    """Generate n_strikes × {CE, PE} legacy-schema rows (well above the 100-floor)."""
    rows: list[dict] = []
    for i in range(n_strikes):
        strike = 22_000 + i * 50
        for opt in ("CE", "PE"):
            rows.append({
                "INSTRUMENT": "OPTIDX",
                "SYMBOL": "NIFTY",
                "EXPIRY_DT": expiry,
                "STRIKE_PR": strike,
                "OPTION_TYP": opt,
                "OPEN": 100.0,
                "HIGH": 110.0,
                "LOW": 90.0,
                "CLOSE": 100.0,
                "SETTLE_PR": 100.0,
                "CONTRACTS": 100,
                "VAL_INLAKH": 1.0,
                "OPEN_INT": 1_000 + i,
                "CHG_IN_OI": 0,
                "TIMESTAMP": "06-JUN-2024",
            })
    return rows


# ---------------------------------------------------------------------------
# 1. Single-day ingest — fixture CSV → parquet at canonical path
# ---------------------------------------------------------------------------
def test_ingests_single_day_to_parquet(tmp_path: Path):
    sd = date(2024, 6, 6)
    raw_zip = _make_legacy_bhavcopy_zip(_legacy_rows(n_strikes=110))

    def fake_download(session_date: date) -> bytes:
        assert session_date == sd
        return raw_zip

    out_path = ingest_one_session(
        sd, out_root=tmp_path,
        download_fn=fake_download,
        parse_fn=bc.parse_bhavcopy,   # exercise the real parser
    )

    expected = _parquet_path(tmp_path, sd)
    assert out_path == expected
    assert out_path.exists()
    assert out_path.parent.name == "06"
    assert out_path.parent.parent.name == "2024"

    df = pd.read_parquet(out_path)
    expected_cols = {
        "session_date", "symbol", "expiry_date", "strike", "option_type",
        "oi", "oi_change", "vol", "ltp", "settlement_price", "iv",
    }
    assert expected_cols <= set(df.columns)
    assert len(df) == 110 * 2   # all options retained
    assert set(df["option_type"].unique()) == {"CE", "PE"}


# ---------------------------------------------------------------------------
# 2. skip_existing=True does not call download_fn when parquet exists
# ---------------------------------------------------------------------------
def test_skip_existing_does_not_redownload(tmp_path: Path):
    sd = date(2024, 6, 6)
    # Pre-create the parquet at the canonical path
    out = _parquet_path(tmp_path, sd)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"a": [1]}).to_parquet(out, index=False)

    call_count = {"n": 0}

    def fail_if_called(session_date: date) -> bytes:
        call_count["n"] += 1
        raise AssertionError("download_fn must NOT be invoked when skip_existing=True")

    returned = ingest_one_session(
        sd, out_root=tmp_path, skip_existing=True,
        download_fn=fail_if_called,
        parse_fn=bc.parse_bhavcopy,
    )
    assert returned == out
    assert call_count["n"] == 0


# ---------------------------------------------------------------------------
# 3. _validate fails when OI column is null
# ---------------------------------------------------------------------------
def test_validation_fails_on_null_oi(tmp_path: Path):
    sd = date(2024, 6, 6)
    rows = _legacy_rows(n_strikes=110)
    # Null out the OI on the first 10 contracts to trigger _validate's null-OI check
    for r in rows[:10]:
        r["OPEN_INT"] = ""   # parses to NaN

    raw_zip = _make_legacy_bhavcopy_zip(rows)

    def fake_download(session_date: date) -> bytes:
        return raw_zip

    # We hand-craft a parser that preserves the empty OI as NaN so
    # _validate trips on null OI rather than the parser silently filling 0.
    def parse_keep_nulls(raw_bytes: bytes, session_date_arg: date):
        """Mirror parse_bhavcopy but keep oi NaN where source was empty."""
        zf = zipfile.ZipFile(io.BytesIO(raw_bytes))
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        df = pd.read_csv(io.BytesIO(zf.read(csv_name)))
        df = df.rename(columns={
            "INSTRUMENT": "instrument", "SYMBOL": "symbol",
            "EXPIRY_DT": "expiry_date", "STRIKE_PR": "strike",
            "OPTION_TYP": "option_type", "OPEN_INT": "oi",
            "CHG_IN_OI": "oi_change", "CONTRACTS": "vol",
            "CLOSE": "ltp", "SETTLE_PR": "settlement_price",
        })
        df = df[df["instrument"].isin(["OPTIDX", "OPTSTK"])].copy()
        df["session_date"] = session_date_arg
        df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce").dt.date
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        # IMPORTANT: do NOT fillna(0) — keep nulls so _validate sees them
        df["oi"] = pd.to_numeric(df["oi"], errors="coerce")
        df["oi_change"] = pd.to_numeric(df["oi_change"], errors="coerce").fillna(0).astype("int64")
        df["vol"] = pd.to_numeric(df["vol"], errors="coerce").fillna(0).astype("int64")
        df["ltp"] = pd.to_numeric(df["ltp"], errors="coerce")
        df["settlement_price"] = pd.to_numeric(df["settlement_price"], errors="coerce")
        df["iv"] = pd.NA
        df = df[df["option_type"].isin(["CE", "PE"])].copy()
        df = df[[
            "session_date", "symbol", "expiry_date", "strike", "option_type",
            "oi", "oi_change", "vol", "ltp", "settlement_price", "iv",
        ]].reset_index(drop=True)
        return bc.BhavcopyParseResult(rows=df, schema="legacy")

    with pytest.raises(ValueError, match="null OI"):
        ingest_one_session(
            sd, out_root=tmp_path,
            download_fn=fake_download,
            parse_fn=parse_keep_nulls,
        )

    # And no parquet was written
    assert not _parquet_path(tmp_path, sd).exists()
