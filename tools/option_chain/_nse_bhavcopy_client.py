"""NSE F&O bhavcopy HTTP client + CSV parser.

Internal module — public ingestion lives in fetch_oi_snapshot.py and the
loader API in services/option_chain_loader.py.

The NSE published two URL schemes for the daily F&O bhavcopy:
  - LEGACY (pre-2024-01-01):
      https://archives.nseindia.com/content/historical/DERIVATIVES/<YYYY>/<MMM>/fo<DDMMMYYYY>bhav.csv.zip
    e.g.  fo28DEC2023bhav.csv.zip
  - NEW (2024-01-01 onwards):
      https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_<YYYYMMDD>.csv.zip
    e.g.  BhavCopy_NSE_FO_20240101.csv.zip

The historical-archive endpoints are unauthenticated and not rate-limited
for backfill (vs the live nseindia.com endpoints which require a
JSESSIONID cookie + frequent CAPTCHA). The legacy archive remained reachable
post-2024 cutover for old dates.

This module exposes two functions:
  - download_bhavcopy(session_date) -> bytes
      Returns the raw ZIP bytes. Raises BhavcopyNotFound on a session that
      isn't a trading day (404), or on any other transport error.
  - parse_bhavcopy(raw_bytes, session_date) -> pd.DataFrame
      Unzips the ZIP, parses the CSV, filters to options only (OPTIDX +
      OPTSTK), normalizes columns to the canonical schema:
        session_date, symbol, expiry_date, strike, option_type (CE/PE),
        oi, oi_change, vol, ltp, settlement_price, iv (NULL).

The actual backfill is the user's deferred step — this client is a
standalone module that the user can invoke via fetch_oi_snapshot.py once
network access to NSE is established.
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

import pandas as pd

# urllib is part of stdlib — no extra dependency.
from urllib import request as _urlreq
from urllib.error import HTTPError as _HTTPError, URLError as _URLError


class BhavcopyNotFound(Exception):
    """Raised when the requested session has no bhavcopy (holiday / weekend / 404)."""


# Cutover: 2024-01-01 onwards uses the new BhavCopy_NSE_FO scheme.
_NEW_SCHEME_CUTOVER = date(2024, 1, 1)

_LEGACY_URL_TEMPLATE = (
    "https://archives.nseindia.com/content/historical/DERIVATIVES/"
    "{year}/{mon_upper}/fo{ddmmmyyyy}bhav.csv.zip"
)
_NEW_URL_TEMPLATE = (
    "https://nsearchives.nseindia.com/content/fo/"
    "BhavCopy_NSE_FO_0_0_0_{yyyymmdd}_F_0000.csv.zip"
)
# A real browser User-Agent — NSE's archive endpoints reject curl/python-urllib defaults.
_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


def _bhavcopy_url(session_date: date) -> str:
    """Choose the right URL scheme based on the session date."""
    if session_date >= _NEW_SCHEME_CUTOVER:
        return _NEW_URL_TEMPLATE.format(yyyymmdd=session_date.strftime("%Y%m%d"))
    # Legacy template uses uppercase abbreviated month and DDMMMYYYY date.
    mon_upper = session_date.strftime("%b").upper()
    return _LEGACY_URL_TEMPLATE.format(
        year=session_date.year,
        mon_upper=mon_upper,
        ddmmmyyyy=session_date.strftime("%d") + mon_upper + session_date.strftime("%Y"),
    )


def download_bhavcopy(
    session_date: date,
    *,
    timeout_seconds: float = 30.0,
    url_override: Optional[str] = None,
) -> bytes:
    """Download the NSE F&O bhavcopy ZIP for `session_date`.

    Returns the raw ZIP bytes. Raises BhavcopyNotFound on 404 (non-trading
    day) or any other transport-layer error. Callers should catch that and
    skip the session.

    `url_override` is for tests — pass a local file:// or fixture URL.
    """
    url = url_override if url_override is not None else _bhavcopy_url(session_date)
    req = _urlreq.Request(url, headers=_DEFAULT_HEADERS)
    try:
        with _urlreq.urlopen(req, timeout=timeout_seconds) as resp:
            data = resp.read()
            if not data:
                raise BhavcopyNotFound(f"empty response for {session_date}: {url}")
            return data
    except _HTTPError as e:
        if e.code == 404:
            raise BhavcopyNotFound(f"404 for {session_date}: {url}") from e
        raise BhavcopyNotFound(
            f"HTTPError {e.code} for {session_date}: {url}"
        ) from e
    except _URLError as e:
        raise BhavcopyNotFound(f"URLError for {session_date}: {url} — {e}") from e


@dataclass
class BhavcopyParseResult:
    """Result of parse_bhavcopy — a DataFrame and the inferred schema variant."""

    rows: pd.DataFrame
    schema: str   # "legacy" or "new"


_LEGACY_COLUMN_MAP = {
    # legacy CSV header → canonical
    "INSTRUMENT": "instrument",
    "SYMBOL": "symbol",
    "EXPIRY_DT": "expiry_date",
    "STRIKE_PR": "strike",
    "OPTION_TYP": "option_type",
    "OPEN_INT": "oi",
    "CHG_IN_OI": "oi_change",
    "CONTRACTS": "vol",
    "CLOSE": "ltp",
    "SETTLE_PR": "settlement_price",
}
_NEW_COLUMN_MAP = {
    # new BhavCopy_NSE_FO header → canonical
    "FinInstrmTp": "instrument",
    "TckrSymb": "symbol",
    "XpryDt": "expiry_date",
    "StrkPric": "strike",
    "OptnTp": "option_type",
    "OpnIntrst": "oi",
    "ChngInOpnIntrst": "oi_change",
    "TtlTradgVol": "vol",
    "ClsPric": "ltp",
    "SttlmPric": "settlement_price",
}


def parse_bhavcopy(raw_bytes: bytes, session_date: date) -> BhavcopyParseResult:
    """Unzip, parse, filter to options only, normalize columns.

    Returns a DataFrame with the canonical schema:
      session_date, symbol, expiry_date, strike, option_type (CE/PE),
      oi, oi_change, vol, ltp, settlement_price, iv (NULL)

    Raises ValueError if the ZIP is malformed or the schema doesn't match
    either known variant.
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw_bytes))
    except zipfile.BadZipFile as e:
        raise ValueError(f"malformed ZIP for {session_date}") from e
    csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
    if not csv_names:
        raise ValueError(f"no CSV in ZIP for {session_date}")
    raw_csv = zf.read(csv_names[0])
    df = pd.read_csv(io.BytesIO(raw_csv))

    # Detect schema variant by header columns
    cols = set(df.columns)
    if {"INSTRUMENT", "SYMBOL", "EXPIRY_DT"} <= cols:
        schema = "legacy"
        df = df.rename(columns=_LEGACY_COLUMN_MAP)
    elif {"FinInstrmTp", "TckrSymb", "XpryDt"} <= cols:
        schema = "new"
        df = df.rename(columns=_NEW_COLUMN_MAP)
    else:
        raise ValueError(
            f"unrecognized bhavcopy schema for {session_date}; "
            f"got cols: {sorted(cols)[:10]}"
        )

    # Filter to options only (OPTIDX + OPTSTK; legacy uses OPTIDX/OPTSTK,
    # new uses STO + IDO option-instrument codes).
    if schema == "legacy":
        df = df[df["instrument"].isin(["OPTIDX", "OPTSTK"])].copy()
    else:
        # New scheme: FinInstrmTp = STO (stock options) or IDO (index options).
        df = df[df["instrument"].isin(["STO", "IDO"])].copy()

    # Normalize types
    df["session_date"] = session_date
    df["expiry_date"] = pd.to_datetime(
        df["expiry_date"], errors="coerce"
    ).dt.date
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["oi"] = pd.to_numeric(df["oi"], errors="coerce").fillna(0).astype("int64")
    df["oi_change"] = pd.to_numeric(df["oi_change"], errors="coerce").fillna(0).astype("int64")
    df["vol"] = pd.to_numeric(df["vol"], errors="coerce").fillna(0).astype("int64")
    df["ltp"] = pd.to_numeric(df["ltp"], errors="coerce")
    df["settlement_price"] = pd.to_numeric(df["settlement_price"], errors="coerce")
    df["iv"] = pd.NA   # not in bhavcopy; placeholder for future

    # CE/PE normalization (legacy uses 'CE'/'PE'; new uses 'Call'/'Put')
    if schema == "new":
        df["option_type"] = df["option_type"].str.upper().map(
            {"CALL": "CE", "PUT": "PE"}
        ).fillna(df["option_type"])

    df = df[df["option_type"].isin(["CE", "PE"])].copy()

    # Final canonical column order
    canonical_cols = [
        "session_date", "symbol", "expiry_date", "strike", "option_type",
        "oi", "oi_change", "vol", "ltp", "settlement_price", "iv",
    ]
    df = df[canonical_cols].reset_index(drop=True)
    return BhavcopyParseResult(rows=df, schema=schema)
