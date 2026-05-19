"""NSE F&O ban-list daily history scraper.

CLI:
    python tools/fno_ban_history/fetch_fno_ban_list.py \\
        --start 2025-10-01 --end 2026-04-30

Output:
    data/fno_ban_history/fno_ban_events.parquet

Schema (one row per (symbol, ban_date) ENTRY EVENT — i.e. the first day a
symbol appears on the ban list after being absent the prior trading day):

    symbol                (NSE F&O underlying ticker, bare — no NSE: prefix)
    ban_date              (date — effective trade-date on which the symbol is
                           in the ban list; IST-naive)
    ban_entry_time        (time — for intraday entries; pd.NaT for EOD)
    ban_exit_time         (time — when ban lifted; pd.NaT if still pending or
                           EOD-only signal — populated when an EOD-list symbol
                           transitions OFF the next-day list)
    mwpl_pct_at_entry     (float — MWPL utilisation % from the daily CSV if
                           the column is present; pd.NA otherwise)
    event_type            ('eod' | 'intraday')
    entry_snapshot_index  (int — 1..4 for intraday snapshots; pd.NA for EOD)

Sources:

1. EOD (primary, fully working):
    https://nsearchives.nseindia.com/archives/fo/sec_ban/fo_secban_<DDMMYYYY>.csv
   Each archive file is the F&O ban list EFFECTIVE on the date embedded in
   the URL (verified by inspecting the "Trade Date" header inside the file —
   it equals the URL date). One file per trading day. Header text:
     ``Securities in Ban For Trade Date <DD-MON-YYYY>:``
   followed by rows of ``<sr_no>,<SYMBOL>`` (no header row, no MWPL column
   in the public archive file as of May 2026). The "today" alias
   ``https://nsearchives.nseindia.com/content/fo/fo_secban.csv`` returns the
   same format but is only the latest day — not addressable historically.

   Per-day "ban entry" events are derived by comparing the symbol set on day
   D to day D-1 (prior trading day): symbols on D but not on D-1 are entry
   events.

2. Intraday snapshots (BEST-EFFORT — stub):
   Per the SEBI Nov 3 2025 framework, NSE runs 4 random intraday MWPL
   checks. Speculation in the brief is that NSE may publish per-snapshot
   intraday ban-list CSVs at:
     ``https://nsearchives.nseindia.com/content/fo/fo_secban_intraday_*.csv``
   We probed (May 2026):
     - ``fo_secban_intraday.csv``
     - ``fo_secban_intraday_1..4.csv``
     - ``fo_secban_intraday_<DDMMYYYY>.csv``
     - ``fo_secban_<DDMMYYYY>_1.csv``
   All returned 404. We have not yet found a stable intraday endpoint.

   ``_fetch_intraday_snapshots()`` is left as a TODO stub returning [] so the
   EOD pipeline is unblocked. When the intraday endpoint is reverse-
   engineered (e.g. via Wayback Machine on a high-volatility day, or by
   inspecting NSE's F&O surveillance page network requests during market
   hours), update that function and the entries it produces will be merged
   into the same parquet with ``event_type='intraday'`` and
   ``entry_snapshot_index`` ∈ {1..4}.

Politeness:
- ≥2-sec sleep between archive file downloads.
- Exponential backoff (1.5x, ceiling 60s) on 429/5xx.
- curl_cffi chrome-impersonation + NSE Akamai bootstrap (home -> referrer).
- Skip non-trading days (Sat/Sun + ``assets/nse_holidays.json``).

Incremental:
- If the output parquet already exists, the existing ``max(ban_date)`` is
  read and only ``[max + 1 trading day, end]`` is fetched. New rows are
  merged with existing, deduped on ``(symbol, ban_date)`` keeping first.

Mirrors the politeness/retry/Akamai-bootstrap pattern from
``tools/asm_gsm_history/fetch_asm_gsm.py``.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from curl_cffi import requests as crequests  # type: ignore
    _HAS_CURL_CFFI = True
except ImportError:  # pragma: no cover
    _HAS_CURL_CFFI = False


# ---------------------------------------------------------------------------
# Constants — paths / endpoints.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OUT_PATH = _REPO_ROOT / "data" / "fno_ban_history" / "fno_ban_events.parquet"
_README_PATH = _REPO_ROOT / "data" / "fno_ban_history" / "_README.md"
_HOLIDAYS_PATH = _REPO_ROOT / "assets" / "nse_holidays.json"

_NSE_HOME = "https://www.nseindia.com/"
_NSE_REFERRER = "https://www.nseindia.com/all-reports"
# Archive endpoint — the URL date is the EFFECTIVE ban date.
_NSE_ARCHIVE_BASE = "https://nsearchives.nseindia.com/archives/fo/sec_ban"
_NSE_ARCHIVE_PATTERN = "fo_secban_{ddmmyyyy}.csv"
# "Today" alias — only used as a fallback when the archive URL 404s on the
# CURRENT day (NSE sometimes lags archiving by a few hours after publish).
_NSE_TODAY_URL = "https://nsearchives.nseindia.com/content/fo/fo_secban.csv"

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}


# ---------------------------------------------------------------------------
# Helpers — date / trading day.
# ---------------------------------------------------------------------------

def load_nse_holidays(path: Path = _HOLIDAYS_PATH) -> set[date]:
    """Parse NSE holidays JSON. Returns a set of holiday dates."""
    if not path.exists():
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return set()
    out: set[date] = set()
    if isinstance(payload, list):
        for it in payload:
            s = (it or {}).get("tradingDate", "")
            try:
                out.add(datetime.strptime(s, "%d-%b-%Y").date())
            except (ValueError, TypeError):
                continue
    return out


def is_trading_day(d: date, holidays: set[date]) -> bool:
    """Mon-Fri and not in holidays."""
    if d.weekday() >= 5:
        return False
    if d in holidays:
        return False
    return True


def trading_days(start: date, end: date, holidays: set[date]) -> list[date]:
    """All trading days in [start, end] inclusive."""
    out: list[date] = []
    d = start
    while d <= end:
        if is_trading_day(d, holidays):
            out.append(d)
        d += timedelta(days=1)
    return out


def next_trading_day(d: date, holidays: set[date]) -> date:
    """Smallest trading day strictly greater than d."""
    cur = d + timedelta(days=1)
    while not is_trading_day(cur, holidays):
        cur += timedelta(days=1)
    return cur


def prev_trading_day(d: date, holidays: set[date]) -> date:
    """Largest trading day strictly less than d."""
    cur = d - timedelta(days=1)
    while not is_trading_day(cur, holidays):
        cur -= timedelta(days=1)
    return cur


# ---------------------------------------------------------------------------
# Symbol normalisation.
# ---------------------------------------------------------------------------

def _normalise_symbol(raw: str) -> Optional[str]:
    """Strip exchange prefixes and uppercase. Returns None on empty/invalid."""
    if raw is None:
        return None
    s = str(raw).strip().upper()
    if not s:
        return None
    # Drop exchange prefix like 'NSE:RELIANCE' or 'BSE:RELIANCE'
    if ":" in s:
        s = s.split(":", 1)[1].strip()
    # Remove surrounding quotes
    s = s.strip().strip('"').strip("'")
    if not s or s in ("NAN", "NIL", "NONE", "NA"):
        return None
    # NSE F&O underlyings are alphanumeric (allow & and -)
    if not re.match(r"^[A-Z0-9&\-]+$", s):
        return None
    return s


# ---------------------------------------------------------------------------
# NSE client — Akamai-bootstrapped.
# ---------------------------------------------------------------------------

class NSEClient:
    """NSE archive downloader with chrome-impersonation + bootstrap."""

    def __init__(
        self,
        sleep_secs: float = 2.0,
        max_backoff_secs: float = 60.0,
        timeout_secs: float = 30.0,
        impersonate: str = "chrome120",
    ) -> None:
        if not _HAS_CURL_CFFI:
            raise RuntimeError(
                "curl_cffi is required for NSE scraping (Akamai bot "
                "challenge); pip install curl_cffi"
            )
        self.sleep_secs = sleep_secs
        self.max_backoff_secs = max_backoff_secs
        self.timeout_secs = timeout_secs
        self.session = crequests.Session(impersonate=impersonate)
        self._bootstrap()

    def _bootstrap(self) -> None:
        """Warm up the Akamai cookie jar."""
        try:
            self.session.get(_NSE_HOME, timeout=self.timeout_secs)
        except Exception as e:
            print(
                f"[fno_ban/nse] home bootstrap warn: {e}", file=sys.stderr,
            )
        time.sleep(2)
        try:
            self.session.get(
                _NSE_REFERRER,
                headers={"Referer": _NSE_HOME},
                timeout=self.timeout_secs,
            )
        except Exception as e:
            print(
                f"[fno_ban/nse] referrer bootstrap warn: {e}", file=sys.stderr,
            )
        time.sleep(2)

    def get_csv(
        self, url: str, max_retries: int = 4,
    ) -> Optional[bytes]:
        """Download a CSV. Returns bytes on 200, None on 404 or exhausted."""
        backoff = self.sleep_secs
        for attempt in range(1, max_retries + 1):
            try:
                r = self.session.get(
                    url,
                    headers={"Referer": _NSE_REFERRER},
                    timeout=self.timeout_secs,
                )
                if r.status_code == 200:
                    ct = (r.headers.get("content-type") or "").lower()
                    if "html" in ct:
                        # 200 with HTML body = NSE soft-404
                        return None
                    return r.content
                if r.status_code == 404:
                    return None
                if r.status_code in (401, 403, 503):
                    print(
                        f"[fno_ban/nse] {r.status_code} on {url} attempt "
                        f"{attempt}; re-bootstrapping",
                        file=sys.stderr,
                    )
                    self._bootstrap()
                    time.sleep(backoff)
                    backoff = min(backoff * 1.5, self.max_backoff_secs)
                    continue
                if r.status_code == 429 or 500 <= r.status_code < 600:
                    retry_after = r.headers.get("Retry-After")
                    sleep_for = (
                        float(retry_after)
                        if retry_after and retry_after.isdigit()
                        else backoff
                    )
                    print(
                        f"[fno_ban/nse] {r.status_code} on {url} attempt "
                        f"{attempt}; sleeping {sleep_for:.1f}s",
                        file=sys.stderr,
                    )
                    time.sleep(sleep_for)
                    backoff = min(backoff * 1.5, self.max_backoff_secs)
                    continue
                print(
                    f"[fno_ban/nse] HTTP {r.status_code} on {url}; giving up",
                    file=sys.stderr,
                )
                return None
            except Exception as e:
                print(
                    f"[fno_ban/nse] transport err {url} attempt {attempt}: {e}",
                    file=sys.stderr,
                )
                time.sleep(backoff)
                backoff = min(backoff * 1.5, self.max_backoff_secs)
        return None


# ---------------------------------------------------------------------------
# CSV parsing.
# ---------------------------------------------------------------------------

# Header line patterns we may encounter:
#   "Securities in Ban For Trade Date 15-MAY-2026:"
#   "Securities in Ban For Trade Date 15-MAY-2026: NIL"
_HEADER_TRADE_DATE_PAT = re.compile(
    r"Trade\s+Date\s+(\d{1,2}-[A-Z]{3}-\d{4})",
    re.IGNORECASE,
)


def parse_fo_secban_csv(raw: bytes) -> tuple[Optional[date], list[dict]]:
    """Parse a fo_secban CSV. Returns (trade_date_in_file, rows).

    rows = list of {symbol, mwpl_pct} dicts. mwpl_pct is None when the CSV
    does not include a utilisation column (the public archive format does
    not — only sr_no + symbol).
    """
    if not raw:
        return None, []
    text: Optional[str] = None
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        return None, []
    if not text.strip():
        return None, []

    # Extract the trade date from the header line (if present).
    trade_date: Optional[date] = None
    m = _HEADER_TRADE_DATE_PAT.search(text)
    if m:
        try:
            trade_date = datetime.strptime(m.group(1), "%d-%b-%Y").date()
        except ValueError:
            trade_date = None

    rows: list[dict] = []
    # The CSV body is: optional header line, then rows of "<n>,<SYMBOL>" or
    # "<n>,<SYMBOL>,<mwpl_pct>". A "NIL" trailing token on the header line
    # means the list is empty.
    upper = text.upper()
    if "NIL" in upper and "TRADE DATE" in upper:
        # Examine the header line: if "NIL" follows the colon, no rows.
        head_line = text.splitlines()[0] if text.splitlines() else ""
        if re.search(r":\s*NIL\b", head_line, re.IGNORECASE):
            return trade_date, []

    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if "TRADE DATE" in ln.upper():
            continue  # header
        # Skip any "Sr. No.,Symbol" style header row defensively
        if ln.lower().startswith("sr") and "symbol" in ln.lower():
            continue
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 2:
            continue
        # parts[0] is sr_no; parts[1] is symbol; parts[2] (optional) is MWPL%
        sym = _normalise_symbol(parts[1])
        if not sym:
            continue
        mwpl: Optional[float] = None
        if len(parts) >= 3 and parts[2]:
            try:
                # Strip a trailing '%' if NSE ever adds one
                mwpl = float(parts[2].rstrip("%").strip())
            except ValueError:
                mwpl = None
        rows.append({"symbol": sym, "mwpl_pct": mwpl})
    return trade_date, rows


# ---------------------------------------------------------------------------
# Intraday snapshots (STUB).
# ---------------------------------------------------------------------------

def _fetch_intraday_snapshots(
    client: NSEClient, d: date,
) -> list[dict]:
    """TODO: BEST-EFFORT stub for the post-Nov-3-2025 intraday ban snapshots.

    NSE may publish 1-4 intraday ban-list snapshots per trading day under the
    new SEBI 4-random-checks framework. We probed (May 2026):
        - ``content/fo/fo_secban_intraday.csv``
        - ``content/fo/fo_secban_intraday_1..4.csv``
        - ``content/fo/fo_secban_intraday_<DDMMYYYY>.csv``
        - ``content/fo/fo_secban_<DDMMYYYY>_1.csv``
    All returned 404. The actual endpoint has not yet been reverse-engineered.

    When found, this should return a list of:
        {"symbol": str, "entry_snapshot_index": int (1..4),
         "ban_entry_time": datetime.time, "mwpl_pct_at_entry": float | None}

    For now: returns [].
    """
    return []


# ---------------------------------------------------------------------------
# End-to-end ingestion.
# ---------------------------------------------------------------------------

def _archive_url(d: date) -> str:
    return (
        f"{_NSE_ARCHIVE_BASE}/"
        f"{_NSE_ARCHIVE_PATTERN.format(ddmmyyyy=d.strftime('%d%m%Y'))}"
    )


def fetch_fno_ban_window(
    start: date, end: date, *,
    holidays: set[date],
    sleep_secs: float = 2.0,
    include_intraday: bool = False,
) -> tuple[list[dict], dict]:
    """Download fo_secban CSVs for every trading day in [start, end].

    Returns (events, stats). Each event row matches the output schema:
        symbol, ban_date, ban_entry_time, ban_exit_time, mwpl_pct_at_entry,
        event_type, entry_snapshot_index.

    Only ENTRY events are emitted: symbols on the ban list on day D that
    were NOT on the ban list on day (D-1 prior trading day). For EOD events,
    ``ban_exit_time`` is filled in when the symbol later disappears from the
    list (the first trading day it is absent).
    """
    client = NSEClient(sleep_secs=sleep_secs)
    days = trading_days(start, end, holidays)
    stats = {
        "days_attempted": 0,
        "days_with_csv": 0,
        "days_404": 0,
        "days_failed": 0,
        "days_empty_list": 0,
        "total_banned_rows_seen": 0,
        "eod_entry_events": 0,
        "intraday_entry_events": 0,
    }

    # Daily snapshot map: date -> {symbol -> {mwpl_pct}}
    daily_snapshots: dict[date, dict[str, dict]] = {}
    # Intraday snapshot map: date -> list of stub event dicts
    intraday_events_by_date: dict[date, list[dict]] = {}

    total = len(days)
    for di, d in enumerate(days, start=1):
        stats["days_attempted"] += 1
        url = _archive_url(d)
        raw = client.get_csv(url)
        if raw is None:
            stats["days_404"] += 1
            print(
                f"[fno_ban] [{di}/{total}] {d} archive 404",
                file=sys.stderr,
            )
            time.sleep(sleep_secs)
            continue
        trade_date_in_file, rows = parse_fo_secban_csv(raw)
        if trade_date_in_file is not None and trade_date_in_file != d:
            print(
                f"[fno_ban] WARN {d}: file Trade Date is "
                f"{trade_date_in_file} (URL date mismatch)",
                file=sys.stderr,
            )
        stats["days_with_csv"] += 1
        if not rows:
            stats["days_empty_list"] += 1
        stats["total_banned_rows_seen"] += len(rows)
        daily_snapshots[d] = {r["symbol"]: {"mwpl_pct": r["mwpl_pct"]} for r in rows}

        # Intraday best-effort
        if include_intraday:
            intraday = _fetch_intraday_snapshots(client, d)
            if intraday:
                intraday_events_by_date[d] = intraday
                stats["intraday_entry_events"] += len(intraday)

        if di % 25 == 0 or di == total:
            print(
                f"[fno_ban] [{di}/{total}] {d} cumulative days_with_csv="
                f"{stats['days_with_csv']} banned_rows_seen="
                f"{stats['total_banned_rows_seen']}",
                file=sys.stderr,
            )
        time.sleep(sleep_secs)

    # Compute EOD entry events: symbol in snapshot[d] AND NOT in snapshot[d_prev]
    events: list[dict] = []
    sorted_days = sorted(daily_snapshots.keys())
    for i, d in enumerate(sorted_days):
        cur = set(daily_snapshots[d].keys())
        # Find prior trading day's snapshot (if we fetched it)
        prev_snap: set[str] = set()
        # Walk back through `days` (the input window) to find the prior fetched day
        if i > 0:
            d_prev = sorted_days[i - 1]
            # Only treat as adjacent if d_prev is the actual prior trading day
            # within the window (gap may exist if we 404'd a day — in that
            # case we treat the symbols as freshly-entered).
            expected_prev = prev_trading_day(d, holidays)
            if d_prev == expected_prev:
                prev_snap = set(daily_snapshots[d_prev].keys())
            # else: treat as a fresh window, all symbols are "entries"
        for sym in cur:
            if sym in prev_snap:
                continue
            mwpl = daily_snapshots[d][sym].get("mwpl_pct")
            events.append({
                "symbol": sym,
                "ban_date": d,
                "ban_entry_time": pd.NaT,
                "ban_exit_time": pd.NaT,
                "mwpl_pct_at_entry": mwpl if mwpl is not None else pd.NA,
                "event_type": "eod",
                "entry_snapshot_index": pd.NA,
            })
            stats["eod_entry_events"] += 1

    # Compute ban_exit_time for EOD events: the first day the symbol is
    # absent from the list (within the window). Set the time to market open
    # (09:15) of that day as the exit signal — downstream filters key off
    # the date primarily.
    if events:
        # Build per-symbol sorted ban-day lists
        sym_days: dict[str, list[date]] = {}
        for d in sorted_days:
            for sym in daily_snapshots[d].keys():
                sym_days.setdefault(sym, []).append(d)
        for ev in events:
            if ev["event_type"] != "eod":
                continue
            sym = ev["symbol"]
            entry_d = ev["ban_date"]
            days_in_ban = sym_days.get(sym, [])
            # The symbol is "in ban" on each date in days_in_ban (which are
            # snapshot dates). Find the run starting at entry_d, then the
            # first trading day after the run-end where the symbol is absent.
            # Run-end = last contiguous fetched date where sym is still in.
            run_end = entry_d
            for d in days_in_ban:
                if d < entry_d:
                    continue
                # Walk forward via expected trading-day sequence
                if d == entry_d:
                    run_end = d
                    continue
                if d == next_trading_day(run_end, holidays):
                    run_end = d
                else:
                    break
            # Exit date = first trading day after run_end that we DID fetch
            # and on which sym is NOT present. If we never saw such a day
            # in-window, leave NaT.
            exit_d: Optional[date] = None
            cursor = next_trading_day(run_end, holidays)
            while cursor <= end:
                if cursor in daily_snapshots:
                    if sym not in daily_snapshots[cursor]:
                        exit_d = cursor
                        break
                else:
                    # We didn't fetch that day — bail (NaT)
                    break
                cursor = next_trading_day(cursor, holidays)
            if exit_d is not None:
                ev["ban_exit_time"] = datetime.combine(exit_d, dt_time(9, 15))

    # Merge intraday events (stub returns []; future-ready)
    for d, intraday_rows in intraday_events_by_date.items():
        for r in intraday_rows:
            events.append({
                "symbol": r["symbol"],
                "ban_date": d,
                "ban_entry_time": r.get("ban_entry_time"),
                "ban_exit_time": pd.NaT,
                "mwpl_pct_at_entry": r.get("mwpl_pct_at_entry", pd.NA),
                "event_type": "intraday",
                "entry_snapshot_index": r.get("entry_snapshot_index", pd.NA),
            })

    return events, stats


# ---------------------------------------------------------------------------
# Persistence (incremental merge + dedupe).
# ---------------------------------------------------------------------------

_OUTPUT_COLUMNS = [
    "symbol",
    "ban_date",
    "ban_entry_time",
    "ban_exit_time",
    "mwpl_pct_at_entry",
    "event_type",
    "entry_snapshot_index",
]


def _read_existing(out_path: Path) -> Optional[pd.DataFrame]:
    if not out_path.exists():
        return None
    try:
        df = pd.read_parquet(out_path)
    except Exception as e:
        print(
            f"[fno_ban] WARN: could not read existing {out_path}: {e}; "
            f"treating as empty",
            file=sys.stderr,
        )
        return None
    if df.empty:
        return df
    if "ban_date" in df.columns:
        df["ban_date"] = pd.to_datetime(df["ban_date"]).dt.date
    return df


def write_events(
    new_rows: list[dict],
    *,
    existing: Optional[pd.DataFrame],
    out_path: Path,
) -> pd.DataFrame:
    """Merge new_rows with existing parquet, dedupe on (symbol, ban_date)."""
    new_df = pd.DataFrame(new_rows) if new_rows else pd.DataFrame(columns=_OUTPUT_COLUMNS)
    if existing is not None and not existing.empty:
        for c in _OUTPUT_COLUMNS:
            if c not in existing.columns:
                existing[c] = pd.NA
        for c in _OUTPUT_COLUMNS:
            if c not in new_df.columns:
                new_df[c] = pd.NA
        merged = pd.concat([existing[_OUTPUT_COLUMNS], new_df[_OUTPUT_COLUMNS]], ignore_index=True)
    else:
        for c in _OUTPUT_COLUMNS:
            if c not in new_df.columns:
                new_df[c] = pd.NA
        merged = new_df[_OUTPUT_COLUMNS].copy()

    if not merged.empty:
        # Dedupe — keep last so newer fetches win (e.g. exit time fill-ins)
        merged = merged.drop_duplicates(
            subset=["symbol", "ban_date", "event_type", "entry_snapshot_index"],
            keep="last",
        )
        merged["ban_date"] = pd.to_datetime(merged["ban_date"]).dt.date
        merged = merged.sort_values(["ban_date", "event_type", "symbol"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    return merged


def write_readme(
    df: pd.DataFrame, *,
    start: date,
    end: date,
    stats: dict,
    include_intraday: bool,
    out_path: Path = _README_PATH,
) -> None:
    """Write a brief data-limitations / coverage note."""
    n = len(df)
    if n:
        n_dates = df["ban_date"].nunique()
        n_symbols = df["symbol"].nunique()
        by_type = df.groupby("event_type").size().to_dict()
        date_min = df["ban_date"].min()
        date_max = df["ban_date"].max()
    else:
        n_dates = 0
        n_symbols = 0
        by_type = {}
        date_min = None
        date_max = None
    text = f"""# F&O Ban-List Event History — Data Notes

Output:    `data/fno_ban_history/fno_ban_events.parquet`
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST
Window:    {start} -> {end}

## Coverage

- Total events:        {n:,}
- Distinct ban_dates:  {n_dates:,}
- Distinct symbols:    {n_symbols:,}
- Date span (data):    {date_min} -> {date_max}
- By event_type:       {by_type}

## Sources

### EOD (primary)
`https://nsearchives.nseindia.com/archives/fo/sec_ban/fo_secban_<DDMMYYYY>.csv`

The archive file's URL date == the effective ban date (verified from the
file's "Trade Date" header text). One file per trading day. File format:
header line `Securities in Ban For Trade Date <DD-MON-YYYY>:` (or
`...: NIL`) followed by lines `<sr_no>,<SYMBOL>`. The public archive does
not include the MWPL% column; `mwpl_pct_at_entry` is therefore `<NA>`
unless intraday snapshots are wired up later.

### Intraday (stub — TODO)
Per the SEBI Nov 3 2025 framework NSE may publish 1-4 intraday ban-list
snapshots when random MWPL checks trigger a ban entry. The endpoint has
not been found in our May-2026 probe (404 on all guessed paths under
`content/fo/fo_secban_intraday_*.csv`). The function
`_fetch_intraday_snapshots()` returns [] until reverse-engineered. When
populated, events will be merged with `event_type='intraday'` and
`entry_snapshot_index` in 1..4.

## Stats

```json
{json.dumps(stats, indent=2, default=str)}
```

Intraday probe: {'enabled (stub returns [])' if include_intraday else 'disabled'}

## Schema

- `symbol` (string — NSE F&O underlying, bare ticker)
- `ban_date` (date — IST-naive, EFFECTIVE trade date the symbol is in ban)
- `ban_entry_time` (Timestamp — NaT for EOD; HH:MM for intraday entries)
- `ban_exit_time` (Timestamp — first trading day after the ban-run end on
   which the symbol is absent from the list; 09:15 IST market-open marker;
   NaT if still pending at end-of-window or not yet observed)
- `mwpl_pct_at_entry` (float — MWPL% utilisation at entry; `<NA>` if not in
   source file)
- `event_type` ('eod' | 'intraday')
- `entry_snapshot_index` (int 1..4 for intraday; `<NA>` for EOD)

## Limitations

1. **EOD MWPL is NA**: the public archive file omits the MWPL% column.
   To recover MWPL at entry, cross-reference with the intraday snapshot
   (once endpoint is found) or scrape the NSE end-of-day FAOII bhavcopy.

2. **Intraday endpoint TBC**: probed `fo_secban_intraday_*.csv` variants
   all return 404 as of May 2026. Path needs to be discovered by network-
   tracing NSE's F&O surveillance page during market hours on a known
   intraday-entry day (e.g. a documented entry day from broker chatter).

3. **Ban exit timing**: `ban_exit_time` is set to 09:15 of the first
   trading day the symbol is absent. The actual cash-segment behaviour
   (lift of ban) is at market open of that date — fine for daily / 5m
   backtests.

4. **Window-edge entries**: a symbol present in the FIRST fetched day of
   the window is treated as a "fresh entry" (no prior-day snapshot to
   compare against). For accurate entry timing at the window-start
   boundary, include 1-2 trading days of pre-window context.
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def _parse_date_arg(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill NSE F&O daily ban-list history into "
            "data/fno_ban_history/fno_ban_events.parquet."
        )
    )
    parser.add_argument(
        "--start", type=_parse_date_arg, required=True,
        help="range start (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", type=_parse_date_arg, required=True,
        help="range end inclusive (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output", type=Path, default=_OUT_PATH,
        help=f"parquet output (default: {_OUT_PATH})",
    )
    parser.add_argument(
        "--readme-path", type=Path, default=_README_PATH,
        help=f"readme path (default: {_README_PATH})",
    )
    parser.add_argument(
        "--sleep-secs", type=float, default=2.0,
        help="sleep between archive downloads (default 2.0)",
    )
    parser.add_argument(
        "--include-intraday", action="store_true",
        help=(
            "attempt intraday snapshot fetch via the stub "
            "(_fetch_intraday_snapshots — currently returns []). Off by "
            "default; flip on when endpoint is wired up."
        ),
    )
    parser.add_argument(
        "--force-full", action="store_true",
        help=(
            "ignore the existing parquet and refetch the full [start, end] "
            "window. Default is incremental: only fetch days after existing "
            "max(ban_date)."
        ),
    )
    args = parser.parse_args(argv)
    if args.start > args.end:
        parser.error("--start must be <= --end")

    holidays = load_nse_holidays()
    print(
        f"[fno_ban] holidays={len(holidays)} window={args.start} -> {args.end}",
        file=sys.stderr,
    )

    existing: Optional[pd.DataFrame] = None if args.force_full else _read_existing(args.output)
    effective_start = args.start
    if existing is not None and not existing.empty and "ban_date" in existing.columns:
        max_existing = max(existing["ban_date"])
        candidate_start = next_trading_day(max_existing, holidays)
        if candidate_start > effective_start:
            effective_start = candidate_start
        print(
            f"[fno_ban] incremental: existing max(ban_date)={max_existing} -> "
            f"effective_start={effective_start}",
            file=sys.stderr,
        )

    if effective_start > args.end:
        print(
            f"[fno_ban] nothing to fetch ({effective_start} > {args.end}); "
            f"rewriting existing parquet unchanged",
            file=sys.stderr,
        )
        df = write_events([], existing=existing, out_path=args.output)
        write_readme(
            df, start=args.start, end=args.end, stats={"skipped": True},
            include_intraday=args.include_intraday,
            out_path=args.readme_path,
        )
        return 0 if not df.empty else 4

    events, stats = fetch_fno_ban_window(
        effective_start, args.end,
        holidays=holidays,
        sleep_secs=args.sleep_secs,
        include_intraday=args.include_intraday,
    )
    print(f"[fno_ban] new events={len(events)} stats={stats}", file=sys.stderr)

    df = write_events(events, existing=existing, out_path=args.output)
    write_readme(
        df,
        start=args.start, end=args.end,
        stats=stats,
        include_intraday=args.include_intraday,
        out_path=args.readme_path,
    )

    by_year = {}
    if not df.empty:
        years = pd.to_datetime(df["ban_date"]).dt.year
        by_year = years.value_counts().sort_index().to_dict()
    print(
        f"[fno_ban] DONE\n"
        f"  range:       {args.start} -> {args.end}\n"
        f"  fetched:     {effective_start} -> {args.end}\n"
        f"  stats:       {stats}\n"
        f"  total rows:  {len(df)}\n"
        f"  by_year:     {by_year}\n"
        f"  out:         {args.output}\n"
        f"  readme:      {args.readme_path}",
        file=sys.stderr,
    )
    return 0 if not df.empty else 4


if __name__ == "__main__":
    sys.exit(main())
