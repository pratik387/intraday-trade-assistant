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
  - parse_bhavcopy(raw_bytes, session_date) -> BhavcopyParseResult
      Unzips the ZIP, parses the CSV, splits rows into:
        * options   (OPTIDX/OPTSTK legacy, IDO/STO new)  -> .rows
        * futures   (FUTIDX/FUTSTK legacy, IDF/STF new)  -> .futures
      Normalizes columns to canonical schemas. Options schema:
        session_date, symbol, expiry_date, strike, option_type (CE/PE),
        oi, oi_change, vol, ltp, settlement_price, iv (NULL).
      Futures schema:
        session_date, symbol, instrument_type ('FUTSTK'/'FUTIDX'),
        expiry_date, contract_type (None), strike (None),
        open, high, low, close, settle, oi, vol.

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
    """Result of parse_bhavcopy — options DataFrame, futures DataFrame, schema."""

    rows: pd.DataFrame                # options rows (canonical schema)
    schema: str                       # "legacy" or "new"
    futures: pd.DataFrame = None      # futures rows (canonical schema)


# Both options and futures need OHLC for the futures rows but only close/ltp
# for the options rows. We extend the column maps accordingly so a single
# rename pass handles both subsets of the same underlying CSV.
_LEGACY_COLUMN_MAP = {
    # legacy CSV header → canonical
    "INSTRUMENT": "instrument",
    "SYMBOL": "symbol",
    "EXPIRY_DT": "expiry_date",
    "STRIKE_PR": "strike",
    "OPTION_TYP": "option_type",
    "OPEN": "open",
    "HIGH": "high",
    "LOW": "low",
    "CLOSE": "close",
    "OPEN_INT": "oi",
    "CHG_IN_OI": "oi_change",
    "CONTRACTS": "vol",
    "SETTLE_PR": "settlement_price",
}
_NEW_COLUMN_MAP = {
    # new BhavCopy_NSE_FO header → canonical
    "FinInstrmTp": "instrument",
    "TckrSymb": "symbol",
    "XpryDt": "expiry_date",
    "StrkPric": "strike",
    "OptnTp": "option_type",
    "OpnPric": "open",
    "HghPric": "high",
    "LwPric": "low",
    "ClsPric": "close",
    "OpnIntrst": "oi",
    "ChngInOpnIntrst": "oi_change",
    "TtlTradgVol": "vol",
    "SttlmPric": "settlement_price",
}

# Maps the (schema, raw instrument code) onto the canonical instrument_type
# label we want in the futures parquet. Index futures and stock futures
# share the same OHLC schema but live in two distinct codes per scheme.
_FUTURES_INSTRUMENT_LABEL = {
    ("legacy", "FUTSTK"): "FUTSTK",
    ("legacy", "FUTIDX"): "FUTIDX",
    ("new", "STF"): "FUTSTK",
    ("new", "IDF"): "FUTIDX",
}


def parse_bhavcopy(raw_bytes: bytes, session_date: date) -> BhavcopyParseResult:
    """Unzip, parse, split into options + futures, normalize columns.

    Returns BhavcopyParseResult with:
      .rows    — options DataFrame, canonical schema:
                   session_date, symbol, expiry_date, strike,
                   option_type (CE/PE), oi, oi_change, vol, ltp,
                   settlement_price, iv (NULL)
      .futures — futures DataFrame, canonical schema:
                   symbol, instrument_type ('FUTSTK'/'FUTIDX'),
                   expiry_date, contract_type (None), strike (None),
                   open, high, low, close, settle, oi, vol, session_date
      .schema  — "legacy" or "new"

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
    raw_df = pd.read_csv(io.BytesIO(raw_csv))

    # Detect schema variant by header columns
    cols = set(raw_df.columns)
    if {"INSTRUMENT", "SYMBOL", "EXPIRY_DT"} <= cols:
        schema = "legacy"
        raw_df = raw_df.rename(columns=_LEGACY_COLUMN_MAP)
        opt_codes = ["OPTIDX", "OPTSTK"]
        fut_codes = ["FUTSTK", "FUTIDX"]
    elif {"FinInstrmTp", "TckrSymb", "XpryDt"} <= cols:
        schema = "new"
        raw_df = raw_df.rename(columns=_NEW_COLUMN_MAP)
        opt_codes = ["STO", "IDO"]
        fut_codes = ["STF", "IDF"]
    else:
        raise ValueError(
            f"unrecognized bhavcopy schema for {session_date}; "
            f"got cols: {sorted(cols)[:10]}"
        )

    # ---- Options subset (preserved behaviour from the original parser) ----
    df = raw_df[raw_df["instrument"].isin(opt_codes)].copy()
    df["session_date"] = session_date
    df["expiry_date"] = pd.to_datetime(
        df["expiry_date"], errors="coerce"
    ).dt.date
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["oi"] = pd.to_numeric(df["oi"], errors="coerce").fillna(0).astype("int64")
    df["oi_change"] = pd.to_numeric(df["oi_change"], errors="coerce").fillna(0).astype("int64")
    df["vol"] = pd.to_numeric(df["vol"], errors="coerce").fillna(0).astype("int64")
    # 'close' is the canonical name; the historical option_chain parquets
    # call it 'ltp' for compatibility with the loader API.
    df["ltp"] = pd.to_numeric(df["close"], errors="coerce")
    df["settlement_price"] = pd.to_numeric(df["settlement_price"], errors="coerce")
    df["iv"] = pd.NA   # not in bhavcopy; placeholder for future

    # CE/PE normalization (legacy uses 'CE'/'PE'; new uses 'Call'/'Put')
    if schema == "new":
        df["option_type"] = df["option_type"].astype("string").str.upper().map(
            {"CALL": "CE", "PUT": "PE"}
        ).fillna(df["option_type"])

    df = df[df["option_type"].isin(["CE", "PE"])].copy()

    options_cols = [
        "session_date", "symbol", "expiry_date", "strike", "option_type",
        "oi", "oi_change", "vol", "ltp", "settlement_price", "iv",
    ]
    options_df = df[options_cols].reset_index(drop=True)

    # ---- Futures subset (FUTSTK + FUTIDX) ----
    fdf = raw_df[raw_df["instrument"].isin(fut_codes)].copy()
    if not fdf.empty:
        fdf["session_date"] = session_date
        fdf["instrument_type"] = fdf["instrument"].map(
            lambda code: _FUTURES_INSTRUMENT_LABEL.get((schema, code), code)
        )
        fdf["expiry_date"] = pd.to_datetime(
            fdf["expiry_date"], errors="coerce"
        ).dt.date
        # Futures rows have no strike / no contract_type — explicit None
        # columns keep the parquet schema stable across days.
        fdf["contract_type"] = None
        fdf["strike"] = pd.NA
        for col in ("open", "high", "low", "close"):
            fdf[col] = pd.to_numeric(fdf.get(col), errors="coerce")
        fdf["settle"] = pd.to_numeric(fdf["settlement_price"], errors="coerce")
        fdf["oi"] = pd.to_numeric(fdf["oi"], errors="coerce").fillna(0).astype("int64")
        fdf["vol"] = pd.to_numeric(fdf["vol"], errors="coerce").fillna(0).astype("int64")

        futures_cols = [
            "symbol", "instrument_type", "expiry_date", "contract_type",
            "strike", "open", "high", "low", "close", "settle",
            "oi", "vol", "session_date",
        ]
        futures_df = fdf[futures_cols].reset_index(drop=True)
    else:
        futures_df = pd.DataFrame(columns=[
            "symbol", "instrument_type", "expiry_date", "contract_type",
            "strike", "open", "high", "low", "close", "settle",
            "oi", "vol", "session_date",
        ])

    return BhavcopyParseResult(rows=options_df, schema=schema, futures=futures_df)
