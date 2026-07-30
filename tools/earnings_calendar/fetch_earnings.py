"""NSE earnings-announcement calendar scraper (Path C: multi-endpoint).

Path C scope (per `specs/2026-05-07-earnings_calendar_expansion_audit.md`):

1. **Existing** ``/api/corporates-financial-results`` — realised filings stream.
   Kept for backwards compatibility, but the F&O-200 filter is dropped at
   scrape time (universe is data-broad at this stage; cell selection happens
   at the gauntlet layer).

2. **NEW** ``/api/corporate-board-meetings`` — forward-scheduled board
   meetings. Resolves the broadcast-lag BMO classification gap because
   ``bm_date`` is the **scheduled meeting date** (independent of when the
   filing eventually broadcasts). Earnings-relevant meetings are detected
   by purpose substring match (Financial Results / Audited / Quarterly /
   Annual / Dividend).

3. **NEW** ``/api/corporate-announcements?subject=Outcome+of+Board+Meeting``
   — Reg-30 30-minute disclosure stream. Captures BMO-class outcomes as they
   are filed (07:30-09:00 IST), not when broadcast queue flushes at 09:15+.

4. **NEW** ``/api/corporate-announcements?subject=Financial+Result+Updates``
   — companion stream to (3); same Reg-30 mechanism, different NSE subject
   tag. NSE uses both interchangeably across the 2022-2026 window.
   **DEAD since ~2025-03**: NSE retired the "Financial Result Updates"
   subject; the stream migrated to "Outcome of Board Meeting" on the same
   endpoint with the same ``an_dt`` second-precision timestamps.

5. **NEW (2026-07-28 repair)** ``announcements_obm`` — same
   ``/api/corporate-announcements?subject=Outcome+of+Board+Meeting`` stream,
   but rows are matched against the board-meetings *earnings roster*
   (symbol, bm_date +/- 1 day) instead of the ``_ann_is_earnings``
   attchmntText filter. The text filter drops most post-2025-03 OBM
   earnings rows (companies stopped writing "financial result" into the
   attachment text); roster matching recovers them with real Reg-30
   timestamps.

6. **NEW (2026-07-28 repair)** ``announcements_bse`` — BSE
   ``api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w``
   (strCat=Result, subcategory=Financial Results; NEWS_DT millisecond
   timestamps). Secondary/validator source (cross-validation showed <=165s
   BSE-vs-NSE deltas). Symbols are mapped SCRIP_CD -> ISIN (via the BSE
   scrip master, cached) -> NSE symbol (via ISIN history in the existing
   parquet + board-meetings rows), then roster-matched like (5).

CLI:
    python tools/earnings_calendar/fetch_earnings.py \\
        --start 2022-12-01 --end 2026-04-30
    python tools/earnings_calendar/fetch_earnings.py \\
        --start 2025-01-01 --end 2025-12-31  # incremental top-up

Output:
    data/earnings_calendar/earnings_events.parquet

Schema (one row per (symbol, announce_date) — sources are merged with a
priority order so the best-quality timestamp wins):
    symbol               (NSE_PREFIX, e.g. "NSE:RELIANCE")
    announce_date        (date, IST-naive — the day the market saw the news)
    announce_time        (datetime, IST-naive, full broadcast/intimation timestamp)
    announce_time_class  ("BMO" | "AMC" | "intraday" | "scheduled")
                         "scheduled" = derived from board-meetings forward
                         scrape, no exact timestamp; treat as BMO for T-1
                         anchoring downstream.
    trade_date           (date — same as announce_date for BMO/scheduled,
                          next-trading-day for AMC, NaT for intraday)
    result_type          ("quarterly" | "annual")
    relating_to          (raw NSE relatingTo where available)
    audited              ("Audited" | "Un-Audited" | None)
    company_name         (raw NSE companyName)
    isin                 (raw NSE isin, optional)
    consolidated         (raw NSE consolidated flag, optional)
    source               ("financial_results" | "board_meetings" |
                          "announcements_bmo" | "announcements_fr" |
                          "announcements_obm" | "announcements_bse")
    bm_date              (date, optional — scheduled board-meeting date)
    bm_purpose           (str, optional — board-meeting purpose text)

Politeness:
- ≥5 sec sleep between requests (configurable via --sleep-secs)
- Exponential backoff on 429 / 5xx (1.5x up to 60s ceiling), respecting
  any Retry-After header.
- Endpoints silently cap at ~600-1000 results per window — we chunk by
  half-month windows.

Result-type heuristic:
    relatingTo == "Fourth Quarter"  AND  audited in {"Audited"}  ->  "annual"
    otherwise                                                    ->  "quarterly"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants — read from config-ish locations (no magic numbers in trading
# logic; this is a one-shot scraper, but values come from CLI / file paths).
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OUT_PATH = _REPO_ROOT / "data" / "earnings_calendar" / "earnings_events.parquet"
_FNO_UNIVERSE_PATH = _REPO_ROOT / "assets" / "fno_liquid_200.csv"
_HOLIDAYS_PATH = _REPO_ROOT / "assets" / "nse_holidays.json"

_BASE_URL = "https://www.nseindia.com"
_BOOTSTRAP_FR_URL = (
    "https://www.nseindia.com/companies-listing/corporate-filings-financial-results"
)
_BOOTSTRAP_BM_URL = (
    "https://www.nseindia.com/companies-listing/corporate-filings-board-meetings"
)
_BOOTSTRAP_ANN_URL = (
    "https://www.nseindia.com/companies-listing/corporate-filings-announcements"
)
_API_FR_URL = "https://www.nseindia.com/api/corporates-financial-results"
_API_BM_URL = "https://www.nseindia.com/api/corporate-board-meetings"
_API_ANN_URL = "https://www.nseindia.com/api/corporate-announcements"

# --- BSE (secondary/validator source, 2026-07-28 repair) ---
_API_BSE_ANN_URL = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
_API_BSE_SCRIP_MASTER_URL = "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w"
_BSE_SCRIP_MASTER_CACHE = (
    _REPO_ROOT / "data" / "earnings_calendar" / "bse_scrip_master.json"
)
_BSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.bseindia.com/",
    "Origin": "https://www.bseindia.com",
}
# Roster matching: an OBM/BSE announcement is treated as an earnings event
# when the symbol has a board-meetings earnings roster entry within this
# many days of the announcement date (results are commonly filed post-
# midnight or the meeting slips by a day).
_ROSTER_TOLERANCE_DAYS = 1
# NSE announcements endpoint silently caps large windows (~600-1000 rows
# observed). When a window returns >= this many raw rows we suspect a cap
# and recursively split the window down to single days.
_NSE_ANN_SPLIT_THRESHOLD = 800
# BSE paging: hard ceiling on pages per window (page size 50 observed;
# ROWCNT in Table1 gives the window total). 100 pages = 5000 rows/window.
_BSE_MAX_PAGES = 100

# NSE subject tags that flag earnings-related announcements. Both spellings
# observed live in 2024-05 sample; "Board Meeting Outcome" returned 0 rows
# while "Outcome of Board Meeting" returned 305 rows — we use the latter.
_ANN_SUBJECTS = (
    "Outcome of Board Meeting",
    "Financial Result Updates",
)

# Board-meeting purpose keywords that flag an earnings-related meeting.
# Match is case-insensitive substring against `bm_purpose` AND `bm_desc`.
_BM_EARNINGS_KEYWORDS = (
    "financial result",
    "quarterly result",
    "audited financial",
    "un-audited financial",
    "unaudited financial",
    "annual result",
    "quarterly financial",
    "yearly audited",
    "yearly financial",
    "half year",
    "half-year",
)

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": _BOOTSTRAP_FR_URL,
    "Connection": "keep-alive",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Dest": "empty",
}

# Announcement-time classification thresholds (IST, 24-hour clock).
# NSE pre-open is 09:00, market open 09:15, market close 15:30.
# Treat 09:15-15:30 as intraday; <09:00 as BMO; >=15:30 as AMC.
# (09:00-09:15 is technically pre-open — also BMO for the purposes of fade.)
_BMO_HOUR_END = 9          # filings before 09:00 IST
_INTRADAY_END_HOUR = 15
_INTRADAY_END_MIN = 30


# ---------------------------------------------------------------------------
# NSE HTTP client (cookie-bootstrap + polite retry).
# ---------------------------------------------------------------------------

class NSESession:
    """Thin wrapper around requests.Session with cookie bootstrap + backoff.

    Bootstraps cookies against multiple deep-link pages so the session is
    valid for *all* corporate-filings endpoints (financial-results,
    board-meetings, announcements).
    """

    def __init__(
        self,
        sleep_secs: float = 5.0,
        max_backoff_secs: float = 60.0,
        timeout_secs: float = 30.0,
    ) -> None:
        self.sleep_secs = sleep_secs
        self.max_backoff_secs = max_backoff_secs
        self.timeout_secs = timeout_secs
        self.session = requests.Session()
        self.session.headers.update(_DEFAULT_HEADERS)
        self._bootstrap()

    def _bootstrap(self) -> None:
        """Visit deep-link pages to seed cookies for all endpoints used."""
        for url in (_BOOTSTRAP_FR_URL, _BOOTSTRAP_BM_URL, _BOOTSTRAP_ANN_URL):
            try:
                self.session.get(url, timeout=self.timeout_secs)
            except requests.RequestException:
                continue
        if not self.session.cookies:
            try:
                self.session.get(_BASE_URL, timeout=self.timeout_secs)
            except requests.RequestException:
                pass
        # Brief pause so the cookie-set request doesn't look automated.
        time.sleep(2)

    def get_json(
        self,
        url: str,
        params: Optional[dict] = None,
        max_retries: int = 5,
    ) -> Optional[list | dict]:
        """GET `url` with backoff. Returns parsed JSON, or None after exhaustion."""
        backoff = self.sleep_secs
        last_status = None
        for attempt in range(1, max_retries + 1):
            try:
                r = self.session.get(
                    url, params=params, timeout=self.timeout_secs
                )
                last_status = r.status_code
                if r.status_code == 200:
                    try:
                        return r.json()
                    except ValueError:
                        # NSE occasionally serves an HTML challenge; re-bootstrap.
                        print(
                            f"[earnings] non-JSON 200 on attempt {attempt}; "
                            f"re-bootstrapping",
                            file=sys.stderr,
                        )
                        self._bootstrap()
                        time.sleep(backoff)
                        backoff = min(backoff * 1.5, self.max_backoff_secs)
                        continue
                if r.status_code in (401, 403):
                    print(
                        f"[earnings] {r.status_code} on attempt {attempt}; "
                        f"re-bootstrapping cookies",
                        file=sys.stderr,
                    )
                    self._bootstrap()
                    time.sleep(backoff)
                    backoff = min(backoff * 1.5, self.max_backoff_secs)
                    continue
                if r.status_code == 429 or 500 <= r.status_code < 600:
                    retry_after = r.headers.get("Retry-After")
                    sleep_for = (
                        float(retry_after) if retry_after and retry_after.isdigit()
                        else backoff
                    )
                    print(
                        f"[earnings] {r.status_code} on attempt {attempt}; "
                        f"sleeping {sleep_for:.1f}s",
                        file=sys.stderr,
                    )
                    time.sleep(sleep_for)
                    backoff = min(backoff * 1.5, self.max_backoff_secs)
                    continue
                # Other status — give up on this URL.
                print(
                    f"[earnings] HTTP {r.status_code} on {url} params={params}; "
                    f"giving up",
                    file=sys.stderr,
                )
                return None
            except requests.RequestException as e:
                print(
                    f"[earnings] transport error attempt {attempt}: {e}",
                    file=sys.stderr,
                )
                time.sleep(backoff)
                backoff = min(backoff * 1.5, self.max_backoff_secs)
        print(
            f"[earnings] exhausted retries for {url} params={params} "
            f"last_status={last_status}",
            file=sys.stderr,
        )
        return None


class BSESession:
    """Polite BSE API client (no cookie bootstrap needed; static headers)."""

    def __init__(
        self,
        sleep_secs: float = 4.0,
        max_backoff_secs: float = 60.0,
        timeout_secs: float = 45.0,
    ) -> None:
        self.sleep_secs = sleep_secs
        self.max_backoff_secs = max_backoff_secs
        self.timeout_secs = timeout_secs
        self.session = requests.Session()
        self.session.headers.update(_BSE_HEADERS)

    def get_json(
        self,
        url: str,
        params: Optional[dict] = None,
        max_retries: int = 5,
    ) -> Optional[list | dict]:
        backoff = self.sleep_secs
        last_status = None
        for attempt in range(1, max_retries + 1):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout_secs)
                last_status = r.status_code
                if r.status_code == 200:
                    try:
                        return r.json()
                    except ValueError:
                        print(
                            f"[earnings:bse] non-JSON 200 on attempt {attempt}",
                            file=sys.stderr,
                        )
                        time.sleep(backoff)
                        backoff = min(backoff * 1.5, self.max_backoff_secs)
                        continue
                if r.status_code == 429 or 500 <= r.status_code < 600:
                    retry_after = r.headers.get("Retry-After")
                    sleep_for = (
                        float(retry_after) if retry_after and retry_after.isdigit()
                        else backoff
                    )
                    print(
                        f"[earnings:bse] {r.status_code} on attempt {attempt}; "
                        f"sleeping {sleep_for:.1f}s",
                        file=sys.stderr,
                    )
                    time.sleep(sleep_for)
                    backoff = min(backoff * 1.5, self.max_backoff_secs)
                    continue
                print(
                    f"[earnings:bse] HTTP {r.status_code} on {url} "
                    f"params={params}; giving up",
                    file=sys.stderr,
                )
                return None
            except requests.RequestException as e:
                print(
                    f"[earnings:bse] transport error attempt {attempt}: {e}",
                    file=sys.stderr,
                )
                time.sleep(backoff)
                backoff = min(backoff * 1.5, self.max_backoff_secs)
        print(
            f"[earnings:bse] exhausted retries for {url} params={params} "
            f"last_status={last_status}",
            file=sys.stderr,
        )
        return None


# ---------------------------------------------------------------------------
# F&O universe + trading-day arithmetic.
# ---------------------------------------------------------------------------

def load_fno_universe(path: Path = _FNO_UNIVERSE_PATH) -> set[str]:
    """Returns a set of bare NSE ticker symbols (e.g. {"RELIANCE", "TCS", ...}).

    Used only when ``--restrict-to-fno`` is supplied. Per Path C, the default
    is to NOT filter at scrape time — the gauntlet layer handles cell
    selection downstream.
    """
    df = pd.read_csv(path)
    col = df.columns[0]
    syms = df[col].dropna().astype(str).str.strip().tolist()
    bare = set()
    for s in syms:
        if s.upper().startswith("NSE:"):
            bare.add(s.split(":", 1)[1].upper())
        else:
            bare.add(s.upper())
    return bare


def load_nse_holidays(path: Path = _HOLIDAYS_PATH) -> set[date]:
    """Parse the NSE holidays JSON, return a set of holiday dates."""
    if not path.exists():
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return set()
    out: set[date] = set()
    items: list = []
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        for v in payload.values():
            if isinstance(v, list):
                items.extend(v)
    for item in items:
        if not isinstance(item, dict):
            continue
        # NSE holiday entries have "tradingDate" like "27-Jan-2022"
        for fld in ("tradingDate", "tradingDay", "holidayDate"):
            if fld in item:
                try:
                    out.add(datetime.strptime(item[fld], "%d-%b-%Y").date())
                except ValueError:
                    pass
                break
    return out


def next_trading_day(d: date, holidays: set[date]) -> date:
    """Return next NSE trading day strictly after `d`."""
    cur = d + timedelta(days=1)
    while cur.weekday() >= 5 or cur in holidays:  # 5=Sat, 6=Sun
        cur += timedelta(days=1)
    return cur


def prev_trading_day(d: date, holidays: set[date]) -> date:
    """Return previous NSE trading day strictly before `d`."""
    cur = d - timedelta(days=1)
    while cur.weekday() >= 5 or cur in holidays:
        cur -= timedelta(days=1)
    return cur


# ---------------------------------------------------------------------------
# Filing parsing + classification.
# ---------------------------------------------------------------------------

def _parse_announce_ts(item: dict) -> Optional[datetime]:
    """Extract the most authoritative timestamp.

    Priority: broadCastDate > exchdisstime > filingDate (FR endpoint)
              an_dt > exchdisstime > sort_date (announcements endpoint)
              bm_timestamp > sysTime (board-meetings endpoint)
    NSE format: "29-Jan-2022 16:20:24" or "29-Jan-2022 16:20"
                or ISO "2024-04-30 23:59:37".
    """
    for fld in (
        "broadCastDate", "exchdisstime", "filingDate",  # FR schema
        "an_dt", "sort_date",                           # announcements schema
        "bm_timestamp", "sysTime",                      # board-meetings schema
    ):
        raw = item.get(fld)
        if not raw or not isinstance(raw, str):
            continue
        for fmt in (
            "%d-%b-%Y %H:%M:%S",
            "%d-%b-%Y %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
        ):
            try:
                return datetime.strptime(raw.strip(), fmt)
            except ValueError:
                continue
    return None


def _parse_bm_date(raw: Optional[str]) -> Optional[date]:
    """Parse the NSE board-meetings ``bm_date`` field, e.g. ``15-Apr-2024``."""
    if not raw or not isinstance(raw, str):
        return None
    for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _classify_announce_time(ts: datetime) -> str:
    """Classify announcement time into BMO / AMC / intraday."""
    if ts.hour < _BMO_HOUR_END:
        return "BMO"
    if ts.hour > _INTRADAY_END_HOUR or (
        ts.hour == _INTRADAY_END_HOUR and ts.minute >= _INTRADAY_END_MIN
    ):
        return "AMC"
    return "intraday"


def _classify_result_type(item: dict) -> str:
    """Heuristic: Q4 + Audited filing is the annual print."""
    relating_to = (item.get("relatingTo") or "").strip()
    audited = (item.get("audited") or "").strip()
    if relating_to == "Fourth Quarter" and audited.lower().startswith("audited"):
        return "annual"
    return "quarterly"


def _bm_is_earnings(purpose: str, desc: str) -> bool:
    """True if a board-meeting row is earnings-related (vs fund-raising etc)."""
    blob = f"{purpose or ''} {desc or ''}".lower()
    return any(kw in blob for kw in _BM_EARNINGS_KEYWORDS)


def _bm_result_type(purpose: str, desc: str) -> str:
    """Heuristic for board-meetings: keyword 'audited'/'annual'/'yearly' -> annual."""
    blob = f"{purpose or ''} {desc or ''}".lower()
    if any(kw in blob for kw in (
        "audited financial",
        "annual result",
        "yearly audited",
        "yearly financial",
        "year ended",
    )):
        return "annual"
    return "quarterly"


def _ann_is_earnings(item: dict) -> bool:
    """True if an announcement row is a financial-results / BMO event.

    The endpoint returns rows for both subjects we asked for. We further
    filter by attchmntText to drop non-earnings outcomes (e.g. fund-raising
    BMOs) and by file-name pattern to drop press-release-only outcomes.
    """
    desc = (item.get("desc") or "").lower()
    if "financial result" in desc:
        return True
    # 'Outcome of Board Meeting' is the umbrella category; filter further
    # by attchmntText for earnings-relevant outcomes.
    text = (item.get("attchmntText") or "").lower()
    if "financial result" in text or "audited financial" in text:
        return True
    # Sometimes attchmntText is empty; fall back to attchmntFile name.
    fname = (item.get("attchmntFile") or "").lower()
    if "financialresult" in fname or "_fr_" in fname or "fr." in fname:
        return True
    return False


def _ann_result_type(item: dict) -> str:
    """Heuristic for announcement rows."""
    text = (
        (item.get("attchmntText") or "")
        + " " + (item.get("desc") or "")
    ).lower()
    if "audited financial" in text or "year ended" in text or "yearly" in text:
        return "annual"
    return "quarterly"


# ---------------------------------------------------------------------------
# Earnings roster (board-meetings derived) + BSE symbol mapping.
# ---------------------------------------------------------------------------

def _bare_symbol(sym: str) -> str:
    """"NSE:RELIANCE" -> "RELIANCE" (idempotent for bare symbols)."""
    s = (sym or "").strip().upper()
    return s.split(":", 1)[1] if ":" in s else s


def build_earnings_roster(
    bm_rows: Iterable[dict],
    parquet_path: Optional[Path] = None,
) -> dict[str, set[date]]:
    """Build {bare_symbol -> set of earnings board-meeting dates}.

    Sources: freshly-scraped ``parse_board_meetings`` rows (pre-dedupe) plus
    any ``bm_date``-bearing rows already persisted in the parquet. This is
    the reference set for roster-matching OBM / BSE announcement rows.
    """
    roster: dict[str, set[date]] = {}

    def _add(sym: str, d) -> None:
        if not sym or d is None:
            return
        if isinstance(d, datetime):
            d = d.date()
        elif not isinstance(d, date):
            try:
                d = pd.Timestamp(d).date()
            except (ValueError, TypeError):
                return
        roster.setdefault(sym, set()).add(d)

    for r in bm_rows:
        _add(_bare_symbol(r.get("symbol", "")), r.get("bm_date"))

    if parquet_path is not None and parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path, columns=["symbol", "bm_date"])
        except (OSError, ValueError, KeyError) as e:
            print(
                f"[earnings] roster: could not read {parquet_path}: {e}",
                file=sys.stderr,
            )
            df = None
        if df is not None:
            for sym, bmd in df.dropna(subset=["bm_date"]).itertuples(index=False):
                _add(_bare_symbol(sym), bmd)
    return roster


def roster_match(roster: dict[str, set[date]], bare_sym: str, d: date) -> bool:
    """True if `d` is within +/- _ROSTER_TOLERANCE_DAYS of a roster bm_date."""
    ds = roster.get(bare_sym)
    if not ds:
        return False
    for off in range(-_ROSTER_TOLERANCE_DAYS, _ROSTER_TOLERANCE_DAYS + 1):
        if d + timedelta(days=off) in ds:
            return True
    return False


def load_bse_scrip_to_isin(
    bse: BSESession,
    cache_path: Path = _BSE_SCRIP_MASTER_CACHE,
) -> dict[str, str]:
    """{SCRIP_CD (str) -> ISIN} from the BSE active-equity scrip master.

    Fetched once and cached to `cache_path` (JSON) — the master is ~1.7MB
    and changes slowly.
    """
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if isinstance(cached, dict) and cached:
                return cached
        except (OSError, json.JSONDecodeError):
            pass
    payload = bse.get_json(
        _API_BSE_SCRIP_MASTER_URL,
        params={
            "Group": "", "Scripcode": "", "industry": "",
            "segment": "Equity", "status": "Active",
        },
    )
    out: dict[str, str] = {}
    if isinstance(payload, list):
        for it in payload:
            scrip = str(it.get("SCRIP_CD") or "").strip()
            isin = (it.get("ISIN_NUMBER") or "").strip()
            if scrip and isin:
                out[scrip] = isin
    if out:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(out, f)
    else:
        print(
            "[earnings:bse] scrip master fetch returned no rows — BSE "
            "symbol mapping unavailable",
            file=sys.stderr,
        )
    return out


def build_isin_to_symbol(parquet_path: Path = _OUT_PATH) -> dict[str, str]:
    """{ISIN -> bare NSE symbol} from the existing parquet's ISIN history.

    ISINs that map to >1 NSE symbol in history (renames captured as two
    symbols) are dropped as ambiguous.
    """
    if not parquet_path.exists():
        return {}
    try:
        df = pd.read_parquet(parquet_path, columns=["symbol", "isin"])
    except (OSError, ValueError, KeyError) as e:
        print(
            f"[earnings] isin map: could not read {parquet_path}: {e}",
            file=sys.stderr,
        )
        return {}
    df = df.dropna(subset=["isin", "symbol"])
    df["bare"] = df["symbol"].map(_bare_symbol)
    grp = df.groupby("isin")["bare"].agg(["nunique", "last"])
    ambiguous = int((grp["nunique"] > 1).sum())
    if ambiguous:
        print(
            f"[earnings:bse] dropping {ambiguous} ambiguous ISIN->symbol "
            f"mappings",
            file=sys.stderr,
        )
    return grp.loc[grp["nunique"] == 1, "last"].to_dict()


# ---------------------------------------------------------------------------
# Per-source parsers — each returns a list of normalised event-rows.
# ---------------------------------------------------------------------------

def parse_financial_results(
    items: Iterable[dict],
    fno_universe: Optional[set[str]],
    holidays: set[date],
) -> list[dict]:
    """Parse rows from ``/api/corporates-financial-results``.

    ``fno_universe``: if not None, restrict to bare NSE symbols in this set.
    """
    rows: list[dict] = []
    for it in items:
        sym_raw = (it.get("symbol") or "").strip().upper()
        if not sym_raw:
            continue
        if fno_universe is not None and sym_raw not in fno_universe:
            continue
        ts = _parse_announce_ts(it)
        if ts is None:
            continue
        announce_date = ts.date()
        announce_class = _classify_announce_time(ts)
        if announce_class == "BMO":
            trade_date = announce_date
        elif announce_class == "AMC":
            trade_date = next_trading_day(announce_date, holidays)
        else:
            trade_date = None  # intraday — exclude from sanity
        rows.append(
            {
                "symbol": f"NSE:{sym_raw}",
                "announce_date": announce_date,
                "announce_time": ts,
                "announce_time_class": announce_class,
                "trade_date": trade_date,
                "result_type": _classify_result_type(it),
                "relating_to": it.get("relatingTo"),
                "audited": it.get("audited"),
                "company_name": it.get("companyName"),
                "isin": it.get("isin"),
                "consolidated": it.get("consolidated"),
                "source": "financial_results",
                "bm_date": None,
                "bm_purpose": None,
            }
        )
    return rows


def parse_board_meetings(
    items: Iterable[dict],
    fno_universe: Optional[set[str]],
    holidays: set[date],
) -> list[dict]:
    """Parse rows from ``/api/corporate-board-meetings``.

    Each row is a forward-scheduled board meeting. We keep only earnings-
    related meetings (purpose contains "Financial Results" / "Audited" etc).

    For these rows:
      - ``announce_date`` = ``bm_date`` (the day market hears the result).
      - ``announce_time_class`` = "scheduled" — we don't know the timing of
        the result release on the meeting day. Downstream sanity should
        either treat "scheduled" rows as BMO (the dominant pattern: most
        Indian companies post-meeting-end at 17:00-22:00 IST = AMC) OR
        merge with announcements_bmo rows for true timing.
      - ``trade_date`` = ``bm_date`` itself (T-0 of announcement; the brief
        operates on T-1). We let the sanity stage compute T-1 from this.
    """
    rows: list[dict] = []
    for it in items:
        sym_raw = (it.get("bm_symbol") or "").strip().upper()
        if not sym_raw:
            continue
        if fno_universe is not None and sym_raw not in fno_universe:
            continue
        bm_dt = _parse_bm_date(it.get("bm_date"))
        if bm_dt is None:
            continue
        purpose = it.get("bm_purpose") or ""
        desc = it.get("bm_desc") or ""
        if not _bm_is_earnings(purpose, desc):
            continue
        # Synthetic announce_time: meeting day + a sentinel hour. We use the
        # bm_timestamp (intimation timestamp — when company filed the
        # forward notice) as a fallback so downstream timestamp-based logic
        # still works, but we mark class as "scheduled" so sanity can
        # special-case if needed.
        intimation_ts = _parse_announce_ts(it)
        # synthetic: bm_date with HH:MM = 09:00 (pre-open boundary)
        synth_ts = datetime.combine(bm_dt, datetime.min.time()).replace(hour=9, minute=0)
        rows.append(
            {
                "symbol": f"NSE:{sym_raw}",
                "announce_date": bm_dt,
                "announce_time": synth_ts,
                "announce_time_class": "scheduled",
                "trade_date": bm_dt,
                "result_type": _bm_result_type(purpose, desc),
                "relating_to": None,
                "audited": (
                    "Audited"
                    if "audited" in purpose.lower() or "audited" in desc.lower()
                    else None
                ),
                "company_name": it.get("sm_name"),
                "isin": it.get("sm_isin"),
                "consolidated": None,
                "source": "board_meetings",
                "bm_date": bm_dt,
                "bm_purpose": purpose[:200] if purpose else None,
                "_intimation_time": intimation_ts,  # internal — dropped at write
            }
        )
    return rows


def parse_announcements(
    items: Iterable[dict],
    fno_universe: Optional[set[str]],
    holidays: set[date],
    *,
    source_tag: str,
    earnings_roster: Optional[dict[str, set[date]]] = None,
) -> list[dict]:
    """Parse rows from ``/api/corporate-announcements``.

    Drops non-earnings outcomes (fund-raising, dividend-only, etc).

    Filtering modes:
      - ``earnings_roster is None`` (legacy): keep rows passing the
        ``_ann_is_earnings`` attchmntText/filename filter.
      - ``earnings_roster`` provided (2026-07-28 repair, ``announcements_obm``):
        keep rows whose (symbol, announce_date) matches a board-meetings
        earnings roster entry within +/- 1 day — the text filter drops most
        post-2025-03 OBM earnings rows.
    """
    rows: list[dict] = []
    for it in items:
        sym_raw = (it.get("symbol") or "").strip().upper()
        if not sym_raw:
            continue
        if fno_universe is not None and sym_raw not in fno_universe:
            continue
        ts = _parse_announce_ts(it)
        if ts is None:
            continue
        if earnings_roster is not None:
            if not roster_match(earnings_roster, sym_raw, ts.date()):
                continue
        elif not _ann_is_earnings(it):
            continue
        announce_date = ts.date()
        announce_class = _classify_announce_time(ts)
        if announce_class == "BMO":
            trade_date = announce_date
        elif announce_class == "AMC":
            trade_date = next_trading_day(announce_date, holidays)
        else:
            trade_date = None  # intraday
        rows.append(
            {
                "symbol": f"NSE:{sym_raw}",
                "announce_date": announce_date,
                "announce_time": ts,
                "announce_time_class": announce_class,
                "trade_date": trade_date,
                "result_type": _ann_result_type(it),
                "relating_to": None,
                "audited": None,
                "company_name": it.get("sm_name"),
                "isin": it.get("sm_isin"),
                "consolidated": None,
                "source": source_tag,
                "bm_date": None,
                "bm_purpose": None,
            }
        )
    return rows


def _parse_bse_ts(item: dict) -> Optional[datetime]:
    """BSE announcement timestamp: ISO with milliseconds, e.g.
    ``2025-04-16T21:45:31.377``. Priority: NEWS_DT > News_submission_dt >
    DT_TM > DissemDT."""
    for fld in ("NEWS_DT", "News_submission_dt", "DT_TM", "DissemDT"):
        raw = item.get(fld)
        if not raw or not isinstance(raw, str):
            continue
        try:
            ts = datetime.fromisoformat(raw.strip())
        except ValueError:
            continue
        # Drop sub-second precision — schema is second-precision IST-naive.
        return ts.replace(microsecond=0)
    return None


def _bse_result_type(item: dict) -> str:
    """Heuristic for BSE Result/Financial Results rows."""
    text = (
        (item.get("NEWSSUB") or "") + " " + (item.get("HEADLINE") or "")
        + " " + (item.get("MORE") or "")
    ).lower()
    if "audited financial" in text or "year ended" in text or "yearly" in text:
        return "annual"
    return "quarterly"


def parse_bse_announcements(
    items: Iterable[dict],
    scrip_to_isin: dict[str, str],
    isin_to_symbol: dict[str, str],
    earnings_roster: dict[str, set[date]],
    fno_universe: Optional[set[str]],
    holidays: set[date],
    stats: Optional[dict] = None,
) -> list[dict]:
    """Parse rows from the BSE AnnSubCategoryGetData Result/Financial
    Results feed.

    Symbol mapping: SCRIP_CD -> ISIN (BSE scrip master) -> bare NSE symbol
    (ISIN history in the parquet). Unmappable rows are dropped (counted in
    ``stats``). Rows must roster-match a board-meetings earnings entry
    (+/- 1 day) — this drops result *presentations*, notes-updates and
    stale-year revision filings.
    """
    rows: list[dict] = []
    for it in items:
        scrip = str(it.get("SCRIP_CD") or "").strip()
        isin = scrip_to_isin.get(scrip)
        sym_raw = isin_to_symbol.get(isin) if isin else None
        if not sym_raw:
            if stats is not None:
                stats["unmapped_rows"] = stats.get("unmapped_rows", 0) + 1
            continue
        if fno_universe is not None and sym_raw not in fno_universe:
            continue
        ts = _parse_bse_ts(it)
        if ts is None:
            continue
        if not roster_match(earnings_roster, sym_raw, ts.date()):
            if stats is not None:
                stats["roster_rejected"] = stats.get("roster_rejected", 0) + 1
            continue
        announce_date = ts.date()
        announce_class = _classify_announce_time(ts)
        if announce_class == "BMO":
            trade_date = announce_date
        elif announce_class == "AMC":
            trade_date = next_trading_day(announce_date, holidays)
        else:
            trade_date = None  # intraday
        rows.append(
            {
                "symbol": f"NSE:{sym_raw}",
                "announce_date": announce_date,
                "announce_time": ts,
                "announce_time_class": announce_class,
                "trade_date": trade_date,
                "result_type": _bse_result_type(it),
                "relating_to": None,
                "audited": None,
                "company_name": it.get("SLONGNAME"),
                "isin": isin,
                "consolidated": None,
                "source": "announcements_bse",
                "bm_date": None,
                "bm_purpose": None,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Per-source-day dedup with priority.
# ---------------------------------------------------------------------------

# Priority for "best" event when multiple sources fire on the same
# (symbol, announce_date). Higher value = preferred (kept on conflict).
# Logic: the announcements stream gives the most authoritative timestamp
# (Reg-30 disclosure clock), the financial-results endpoint gives the
# broadcast timestamp (lagged), board-meetings gives only a synthetic 09:00.
#
# 2026-07-28 repair sources (announcements_obm, announcements_bse) carry
# real timestamps but rank BELOW every pre-existing real-timestamp source
# and ABOVE the synthetic board_meetings rows: this makes the repair
# strictly additive — re-running over 2023-2024 windows can never displace
# an existing row, it can only upgrade synthetic "scheduled" rows.
_SOURCE_PRIORITY = {
    "announcements_fr": 6,
    "announcements_bmo": 5,
    "financial_results": 4,
    "announcements_obm": 3,
    "announcements_bse": 2,
    "board_meetings": 1,
}


def dedupe_per_symbol_day(rows: list[dict]) -> list[dict]:
    """Collapse rows on the same (symbol, announce_date) keyed by source priority.

    Sources rank: announcements_fr > announcements_bmo > financial_results >
    board_meetings. Within a source, prefer Consolidated > Standalone, then
    earliest announce_time.
    """
    if not rows:
        return rows
    df = pd.DataFrame(rows)
    df["_source_priority"] = df["source"].map(_SOURCE_PRIORITY).fillna(0).astype(int)
    df["_consolidated_priority"] = df["consolidated"].apply(
        lambda x: 0 if isinstance(x, str) and x.lower().startswith("consolid")
        else 1
    )
    # Sort: highest source_priority first; within that, consolidated first;
    # then earliest announce_time.
    df = df.sort_values(
        by=[
            "symbol", "announce_date",
            "_source_priority", "_consolidated_priority", "announce_time",
        ],
        ascending=[True, True, False, True, True],
    )
    df = df.drop_duplicates(subset=["symbol", "announce_date"], keep="first")
    df = df.drop(columns=["_source_priority", "_consolidated_priority"])
    if "_intimation_time" in df.columns:
        df = df.drop(columns=["_intimation_time"])
    return df.to_dict("records")


# ---------------------------------------------------------------------------
# Date-range scraping — half-month chunks.
# ---------------------------------------------------------------------------

def _fmt_nse_date(d: date) -> str:
    return d.strftime("%d-%m-%Y")


def _half_month_chunks(start: date, end: date) -> list[tuple[date, date]]:
    """Yield (chunk_start, chunk_end) pairs covering [start, end] in ~15-day windows.

    Splitting at month-day-15 keeps each window under the NSE result cap
    (~600 rows/window observed empirically).
    """
    chunks: list[tuple[date, date]] = []
    cur = start
    while cur <= end:
        # End of half-month containing `cur`.
        if cur.day <= 15:
            half_end = cur.replace(day=15)
        else:
            # Last day of month.
            if cur.month == 12:
                next_first = cur.replace(year=cur.year + 1, month=1, day=1)
            else:
                next_first = cur.replace(month=cur.month + 1, day=1)
            half_end = next_first - timedelta(days=1)
        chunk_end = min(half_end, end)
        chunks.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)
    return chunks


def _weekly_chunks(start: date, end: date) -> list[tuple[date, date]]:
    """Yield (chunk_start, chunk_end) pairs covering [start, end] in 7-day windows.

    Used for the announcements endpoint, which can return >1000 rows even
    in a half-month window across the full universe.
    """
    chunks: list[tuple[date, date]] = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=6), end)
        chunks.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)
    return chunks


def fetch_financial_results_range(
    nse: NSESession,
    start: date,
    end: date,
    *,
    fno_universe: Optional[set[str]],
    holidays: set[date],
    sleep_secs: float,
) -> tuple[list[dict], dict]:
    """Scrape ``/api/corporates-financial-results`` for [start, end]."""
    chunks = _half_month_chunks(start, end)
    stats = {
        "endpoint": "financial_results",
        "chunks_attempted": 0,
        "chunks_ok": 0,
        "chunks_failed": 0,
        "raw_rows": 0,
        "kept_rows": 0,
    }
    all_rows: list[dict] = []
    for i, (cs, ce) in enumerate(chunks, start=1):
        stats["chunks_attempted"] += 1
        params = {
            "index": "equities",
            "from_date": _fmt_nse_date(cs),
            "to_date": _fmt_nse_date(ce),
            "period": "Quarterly",
        }
        # NB: dropped `fo_sec=true` — universe broadening per Path C.
        # If the caller still wants F&O-only, they pass fno_universe and
        # the parse step filters there.
        print(f"[FR] [{i}/{len(chunks)}] {cs} -> {ce} ...", file=sys.stderr)
        payload = nse.get_json(_API_FR_URL, params=params)
        if payload is None:
            stats["chunks_failed"] += 1
            time.sleep(sleep_secs)
            continue
        items = payload if isinstance(payload, list) else []
        stats["raw_rows"] += len(items)
        rows = parse_financial_results(items, fno_universe, holidays)
        stats["kept_rows"] += len(rows)
        all_rows.extend(rows)
        stats["chunks_ok"] += 1
        print(
            f"[FR]    raw={len(items)} kept={len(rows)} "
            f"running_total={len(all_rows)}",
            file=sys.stderr,
        )
        time.sleep(sleep_secs)
    return all_rows, stats


def fetch_board_meetings_range(
    nse: NSESession,
    start: date,
    end: date,
    *,
    fno_universe: Optional[set[str]],
    holidays: set[date],
    sleep_secs: float,
) -> tuple[list[dict], dict]:
    """Scrape ``/api/corporate-board-meetings`` for [start, end].

    The endpoint's ``from_date``/``to_date`` filter on ``bm_date``
    (scheduled meeting), not intimation date. So we can iterate the meeting
    date range directly.
    """
    chunks = _half_month_chunks(start, end)
    stats = {
        "endpoint": "board_meetings",
        "chunks_attempted": 0,
        "chunks_ok": 0,
        "chunks_failed": 0,
        "raw_rows": 0,
        "kept_rows": 0,
    }
    all_rows: list[dict] = []
    for i, (cs, ce) in enumerate(chunks, start=1):
        stats["chunks_attempted"] += 1
        params = {
            "index": "equities",
            "from_date": _fmt_nse_date(cs),
            "to_date": _fmt_nse_date(ce),
        }
        print(f"[BM] [{i}/{len(chunks)}] {cs} -> {ce} ...", file=sys.stderr)
        payload = nse.get_json(_API_BM_URL, params=params)
        if payload is None:
            stats["chunks_failed"] += 1
            time.sleep(sleep_secs)
            continue
        items = payload if isinstance(payload, list) else []
        stats["raw_rows"] += len(items)
        rows = parse_board_meetings(items, fno_universe, holidays)
        stats["kept_rows"] += len(rows)
        all_rows.extend(rows)
        stats["chunks_ok"] += 1
        print(
            f"[BM]    raw={len(items)} kept_earnings={len(rows)} "
            f"running_total={len(all_rows)}",
            file=sys.stderr,
        )
        time.sleep(sleep_secs)
    return all_rows, stats


def fetch_announcements_range(
    nse: NSESession,
    start: date,
    end: date,
    *,
    fno_universe: Optional[set[str]],
    holidays: set[date],
    sleep_secs: float,
    subject: str,
    source_tag: str,
) -> tuple[list[dict], dict]:
    """Scrape ``/api/corporate-announcements`` for [start, end] with subject filter."""
    chunks = _weekly_chunks(start, end)
    stats = {
        "endpoint": f"announcements:{subject}",
        "chunks_attempted": 0,
        "chunks_ok": 0,
        "chunks_failed": 0,
        "raw_rows": 0,
        "kept_rows": 0,
    }
    all_rows: list[dict] = []
    for i, (cs, ce) in enumerate(chunks, start=1):
        stats["chunks_attempted"] += 1
        params = {
            "index": "equities",
            "from_date": _fmt_nse_date(cs),
            "to_date": _fmt_nse_date(ce),
            "subject": subject,
        }
        print(
            f"[ANN:{subject[:18]}] [{i}/{len(chunks)}] {cs} -> {ce} ...",
            file=sys.stderr,
        )
        payload = nse.get_json(_API_ANN_URL, params=params)
        if payload is None:
            stats["chunks_failed"] += 1
            time.sleep(sleep_secs)
            continue
        items = payload if isinstance(payload, list) else []
        stats["raw_rows"] += len(items)
        rows = parse_announcements(
            items, fno_universe, holidays, source_tag=source_tag,
        )
        stats["kept_rows"] += len(rows)
        all_rows.extend(rows)
        stats["chunks_ok"] += 1
        print(
            f"[ANN]    raw={len(items)} kept={len(rows)} "
            f"running_total={len(all_rows)}",
            file=sys.stderr,
        )
        time.sleep(sleep_secs)
    return all_rows, stats


def _fetch_ann_window_adaptive(
    nse: NSESession,
    cs: date,
    ce: date,
    *,
    subject: str,
    sleep_secs: float,
    stats: dict,
) -> list[dict]:
    """Fetch one announcements window; split recursively if the response
    looks silently capped (>= _NSE_ANN_SPLIT_THRESHOLD raw rows)."""
    stats["chunks_attempted"] += 1
    params = {
        "index": "equities",
        "from_date": _fmt_nse_date(cs),
        "to_date": _fmt_nse_date(ce),
        "subject": subject,
    }
    payload = nse.get_json(_API_ANN_URL, params=params)
    time.sleep(sleep_secs)
    if payload is None:
        stats["chunks_failed"] += 1
        return []
    items = payload if isinstance(payload, list) else []
    stats["chunks_ok"] += 1
    if len(items) >= _NSE_ANN_SPLIT_THRESHOLD and cs < ce:
        # Suspected silent cap — split the window and re-fetch both halves.
        mid = cs + timedelta(days=(ce - cs).days // 2)
        print(
            f"[ANN-OBM] window {cs}->{ce} returned {len(items)} rows "
            f"(>= cap threshold); splitting at {mid}",
            file=sys.stderr,
        )
        stats["windows_split"] = stats.get("windows_split", 0) + 1
        left = _fetch_ann_window_adaptive(
            nse, cs, mid, subject=subject, sleep_secs=sleep_secs, stats=stats,
        )
        right = _fetch_ann_window_adaptive(
            nse, mid + timedelta(days=1), ce,
            subject=subject, sleep_secs=sleep_secs, stats=stats,
        )
        return left + right
    if len(items) >= _NSE_ANN_SPLIT_THRESHOLD:
        print(
            f"[ANN-OBM] WARNING single-day {cs} returned {len(items)} rows "
            f"(>= cap threshold) — possible truncation, cannot split further",
            file=sys.stderr,
        )
        stats["days_possibly_truncated"] = (
            stats.get("days_possibly_truncated", 0) + 1
        )
    stats["raw_rows"] += len(items)
    return items


def fetch_announcements_obm_range(
    nse: NSESession,
    start: date,
    end: date,
    *,
    earnings_roster: dict[str, set[date]],
    fno_universe: Optional[set[str]],
    holidays: set[date],
    sleep_secs: float,
) -> tuple[list[dict], dict]:
    """Scrape the OBM subject stream, roster-matched (source=announcements_obm).

    Weekly windows with adaptive per-day splitting on peak result weeks
    (the endpoint silently caps large responses).
    """
    subject = "Outcome of Board Meeting"
    chunks = _weekly_chunks(start, end)
    stats = {
        "endpoint": "announcements_obm",
        "chunks_attempted": 0,
        "chunks_ok": 0,
        "chunks_failed": 0,
        "raw_rows": 0,
        "kept_rows": 0,
    }
    all_rows: list[dict] = []
    for i, (cs, ce) in enumerate(chunks, start=1):
        print(f"[ANN-OBM] [{i}/{len(chunks)}] {cs} -> {ce} ...", file=sys.stderr)
        items = _fetch_ann_window_adaptive(
            nse, cs, ce, subject=subject, sleep_secs=sleep_secs, stats=stats,
        )
        rows = parse_announcements(
            items, fno_universe, holidays,
            source_tag="announcements_obm", earnings_roster=earnings_roster,
        )
        stats["kept_rows"] += len(rows)
        all_rows.extend(rows)
        print(
            f"[ANN-OBM]    raw={len(items)} kept={len(rows)} "
            f"running_total={len(all_rows)}",
            file=sys.stderr,
        )
    return all_rows, stats


def fetch_bse_announcements_range(
    bse: BSESession,
    start: date,
    end: date,
    *,
    scrip_to_isin: dict[str, str],
    isin_to_symbol: dict[str, str],
    earnings_roster: dict[str, set[date]],
    fno_universe: Optional[set[str]],
    holidays: set[date],
    sleep_secs: float,
) -> tuple[list[dict], dict]:
    """Scrape BSE Result / Financial Results announcements (paged, weekly
    windows). Source tag: ``announcements_bse``."""
    chunks = _weekly_chunks(start, end)
    stats = {
        "endpoint": "announcements_bse",
        "chunks_attempted": 0,
        "chunks_ok": 0,
        "chunks_failed": 0,
        "pages_fetched": 0,
        "raw_rows": 0,
        "kept_rows": 0,
        "unmapped_rows": 0,
        "roster_rejected": 0,
    }
    all_rows: list[dict] = []
    for i, (cs, ce) in enumerate(chunks, start=1):
        stats["chunks_attempted"] += 1
        print(f"[BSE] [{i}/{len(chunks)}] {cs} -> {ce} ...", file=sys.stderr)
        items: list[dict] = []
        rowcnt: Optional[int] = None
        page_ok = False
        for pageno in range(1, _BSE_MAX_PAGES + 1):
            params = {
                "pageno": pageno,
                "strCat": "Result",
                "strPrevDate": cs.strftime("%Y%m%d"),
                "strToDate": ce.strftime("%Y%m%d"),
                "strScrip": "",
                "strSearch": "P",
                "strType": "C",
                "subcategory": "Financial Results",
            }
            payload = bse.get_json(_API_BSE_ANN_URL, params=params)
            time.sleep(sleep_secs)
            if payload is None:
                break
            page_ok = True
            stats["pages_fetched"] += 1
            page_rows = (
                payload.get("Table", []) if isinstance(payload, dict) else []
            )
            if rowcnt is None and isinstance(payload, dict):
                t1 = payload.get("Table1") or []
                if t1 and isinstance(t1[0], dict):
                    try:
                        rowcnt = int(t1[0].get("ROWCNT"))
                    except (TypeError, ValueError):
                        rowcnt = None
            if not page_rows:
                break
            items.extend(page_rows)
            if rowcnt is not None and len(items) >= rowcnt:
                break
        if not page_ok:
            stats["chunks_failed"] += 1
            continue
        stats["chunks_ok"] += 1
        stats["raw_rows"] += len(items)
        rows = parse_bse_announcements(
            items, scrip_to_isin, isin_to_symbol, earnings_roster,
            fno_universe, holidays, stats=stats,
        )
        stats["kept_rows"] += len(rows)
        all_rows.extend(rows)
        print(
            f"[BSE]    raw={len(items)} (rowcnt={rowcnt}) kept={len(rows)} "
            f"running_total={len(all_rows)}",
            file=sys.stderr,
        )
    return all_rows, stats


def fetch_all_sources(
    start: date,
    end: date,
    *,
    fno_universe: Optional[set[str]],
    holidays: set[date],
    sleep_secs: float = 5.0,
    sources: Iterable[str] = ("financial_results", "board_meetings", "announcements"),
    roster_parquet_path: Optional[Path] = _OUT_PATH,
) -> tuple[list[dict], list[dict]]:
    """Scrape all configured endpoints for [start, end].

    Returns (all_rows, stats_per_source). Per-endpoint failure is isolated:
    if one endpoint returns 0 rows or errors out, we still get rows from
    the others — preserving backwards compatibility.

    ``roster_parquet_path``: parquet used to seed the board-meetings
    earnings roster (and the ISIN->symbol map) for the repair sources
    ``announcements_obm`` / ``announcements_bse``.
    """
    nse = NSESession(sleep_secs=sleep_secs)
    all_rows: list[dict] = []
    all_stats: list[dict] = []
    bm_rows_for_roster: list[dict] = []

    sources = set(sources)

    if "financial_results" in sources:
        try:
            rows, stats = fetch_financial_results_range(
                nse, start, end,
                fno_universe=fno_universe, holidays=holidays,
                sleep_secs=sleep_secs,
            )
            all_rows.extend(rows)
            all_stats.append(stats)
        except Exception as e:
            print(
                f"[earnings] financial_results scrape FAILED: {e}; continuing",
                file=sys.stderr,
            )
            all_stats.append({"endpoint": "financial_results", "error": str(e)})

    if "board_meetings" in sources:
        try:
            rows, stats = fetch_board_meetings_range(
                nse, start, end,
                fno_universe=fno_universe, holidays=holidays,
                sleep_secs=sleep_secs,
            )
            all_rows.extend(rows)
            bm_rows_for_roster = rows
            all_stats.append(stats)
        except Exception as e:
            print(
                f"[earnings] board_meetings scrape FAILED: {e}; continuing",
                file=sys.stderr,
            )
            all_stats.append({"endpoint": "board_meetings", "error": str(e)})

    if "announcements" in sources:
        for subject in _ANN_SUBJECTS:
            tag = (
                "announcements_bmo"
                if subject == "Outcome of Board Meeting"
                else "announcements_fr"
            )
            try:
                rows, stats = fetch_announcements_range(
                    nse, start, end,
                    fno_universe=fno_universe, holidays=holidays,
                    sleep_secs=sleep_secs,
                    subject=subject, source_tag=tag,
                )
                all_rows.extend(rows)
                all_stats.append(stats)
            except Exception as e:
                print(
                    f"[earnings] announcements({subject}) scrape FAILED: {e}; "
                    f"continuing",
                    file=sys.stderr,
                )
                all_stats.append(
                    {"endpoint": f"announcements:{subject}", "error": str(e)}
                )

    # --- 2026-07-28 repair sources (roster-matched) ---
    repair_sources = {"announcements_obm", "announcements_bse"} & sources
    if repair_sources:
        earnings_roster = build_earnings_roster(
            bm_rows_for_roster, roster_parquet_path,
        )
        print(
            f"[earnings] roster: {sum(len(v) for v in earnings_roster.values())} "
            f"(symbol, bm_date) entries across {len(earnings_roster)} symbols",
            file=sys.stderr,
        )
        if not earnings_roster:
            print(
                "[earnings] WARNING: empty earnings roster — repair sources "
                "will keep 0 rows",
                file=sys.stderr,
            )

    if "announcements_obm" in sources:
        try:
            rows, stats = fetch_announcements_obm_range(
                nse, start, end,
                earnings_roster=earnings_roster,
                fno_universe=fno_universe, holidays=holidays,
                sleep_secs=sleep_secs,
            )
            all_rows.extend(rows)
            all_stats.append(stats)
        except Exception as e:
            print(
                f"[earnings] announcements_obm scrape FAILED: {e}; continuing",
                file=sys.stderr,
            )
            all_stats.append({"endpoint": "announcements_obm", "error": str(e)})

    if "announcements_bse" in sources:
        try:
            bse = BSESession(sleep_secs=sleep_secs)
            scrip_to_isin = load_bse_scrip_to_isin(bse)
            isin_to_symbol = build_isin_to_symbol(
                roster_parquet_path if roster_parquet_path else _OUT_PATH,
            )
            print(
                f"[earnings:bse] scrip_master={len(scrip_to_isin)} scrips; "
                f"isin_map={len(isin_to_symbol)} symbols",
                file=sys.stderr,
            )
            rows, stats = fetch_bse_announcements_range(
                bse, start, end,
                scrip_to_isin=scrip_to_isin,
                isin_to_symbol=isin_to_symbol,
                earnings_roster=earnings_roster,
                fno_universe=fno_universe, holidays=holidays,
                sleep_secs=sleep_secs,
            )
            all_rows.extend(rows)
            all_stats.append(stats)
        except Exception as e:
            print(
                f"[earnings] announcements_bse scrape FAILED: {e}; continuing",
                file=sys.stderr,
            )
            all_stats.append({"endpoint": "announcements_bse", "error": str(e)})

    return all_rows, all_stats


# Legacy alias preserved for backwards compatibility (older drivers may
# still import ``fetch_range``).
def fetch_range(
    start: date,
    end: date,
    *,
    fno_universe: set[str],
    holidays: set[date],
    sleep_secs: float = 5.0,
) -> tuple[list[dict], dict]:
    """Legacy API: scrape only the financial-results endpoint with F&O filter."""
    nse = NSESession(sleep_secs=sleep_secs)
    rows, stats = fetch_financial_results_range(
        nse, start, end,
        fno_universe=fno_universe, holidays=holidays,
        sleep_secs=sleep_secs,
    )
    legacy_stats = {
        "chunks_attempted": stats.get("chunks_attempted", 0),
        "chunks_ok": stats.get("chunks_ok", 0),
        "chunks_failed": stats.get("chunks_failed", 0),
        "raw_filings": stats.get("raw_rows", 0),
        "fno_filings_kept": stats.get("kept_rows", 0),
    }
    return rows, legacy_stats


# ---------------------------------------------------------------------------
# Persistence — incremental merge with existing parquet.
# ---------------------------------------------------------------------------

_OUT_COLUMNS = [
    "symbol", "announce_date", "announce_time", "announce_time_class",
    "trade_date", "result_type", "relating_to", "audited",
    "company_name", "isin", "consolidated",
    "source", "bm_date", "bm_purpose",
]


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing columns from the canonical list (default None)."""
    for c in _OUT_COLUMNS:
        if c not in df.columns:
            df[c] = None
    # Reorder canonically.
    return df[_OUT_COLUMNS]


def write_events(
    rows: list[dict],
    out_path: Path = _OUT_PATH,
    *,
    merge_existing: bool = True,
) -> pd.DataFrame:
    """Dedupe + (optionally) merge with existing parquet, write to disk."""
    df_new = pd.DataFrame(rows) if rows else pd.DataFrame(columns=_OUT_COLUMNS)
    df_new = _ensure_columns(df_new)

    if merge_existing and out_path.exists():
        try:
            df_old = pd.read_parquet(out_path)
            df_old = _ensure_columns(df_old)
            # Existing rows that pre-date Path C have no source label —
            # treat them as ``financial_results`` so the dedup priority
            # math is well-defined (the new announcements rows will win
            # ties as they should).
            df_old["source"] = df_old["source"].fillna("financial_results")
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        except (OSError, ValueError) as e:
            print(
                f"[earnings] could not read existing {out_path}: {e}; "
                f"overwriting",
                file=sys.stderr,
            )
            df_all = df_new
    else:
        df_all = df_new

    if df_all.empty:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_all.to_parquet(out_path, index=False)
        return df_all

    # Final dedup at write-time (covers re-runs that overlap windows).
    df_all = pd.DataFrame(dedupe_per_symbol_day(df_all.to_dict("records")))
    df_all = _ensure_columns(df_all)
    df_all = df_all.sort_values(["announce_date", "symbol"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(out_path, index=False)
    return df_all


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def _parse_date_arg(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill NSE earnings-announcement calendar via 4 endpoints "
            "(Path C). Default = full universe, no F&O restriction."
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
        "--out-path", type=Path, default=_OUT_PATH,
        help=f"parquet output path (default: {_OUT_PATH})",
    )
    parser.add_argument(
        "--restrict-to-fno", action="store_true",
        help=(
            "Filter scraped rows to symbols listed in --fno-universe CSV. "
            "Default (Path C) is to scrape the full NSE equities universe; "
            "downstream gauntlet handles cell selection."
        ),
    )
    parser.add_argument(
        "--fno-universe", type=Path, default=_FNO_UNIVERSE_PATH,
        help=f"F&O universe CSV (default: {_FNO_UNIVERSE_PATH})",
    )
    parser.add_argument(
        "--sleep-secs", type=float, default=5.0,
        help="sleep between requests (>=5 recommended for NSE politeness)",
    )
    parser.add_argument(
        "--no-merge-existing", action="store_true",
        help="overwrite the parquet instead of merging with existing rows",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="financial_results,board_meetings,announcements",
        help=(
            "Comma-separated list of source endpoints to scrape. "
            "Choices: financial_results, board_meetings, announcements, "
            "announcements_obm, announcements_bse"
        ),
    )
    args = parser.parse_args(argv)

    if args.start > args.end:
        parser.error("--start must be <= --end")

    sources = tuple(s.strip() for s in args.sources.split(",") if s.strip())
    valid = {
        "financial_results", "board_meetings", "announcements",
        "announcements_obm", "announcements_bse",
    }
    bad = set(sources) - valid
    if bad:
        parser.error(f"unknown source(s) in --sources: {sorted(bad)}; valid={sorted(valid)}")

    fno = load_fno_universe(args.fno_universe) if args.restrict_to_fno else None
    holidays = load_nse_holidays()
    print(
        f"[earnings] universe: "
        f"{'F&O '+str(len(fno)) if fno else 'ALL NSE equities'} ; "
        f"holidays loaded: {len(holidays)} ; "
        f"sources: {sources}",
        file=sys.stderr,
    )

    rows, stats_list = fetch_all_sources(
        args.start, args.end,
        fno_universe=fno, holidays=holidays,
        sleep_secs=args.sleep_secs, sources=sources,
        roster_parquet_path=args.out_path,
    )

    df = write_events(
        rows, args.out_path, merge_existing=not args.no_merge_existing,
    )

    # Summary
    if not df.empty:
        cls_counts = df["announce_time_class"].value_counts().to_dict()
        rt_counts = df["result_type"].value_counts().to_dict()
        src_counts = df["source"].value_counts().to_dict() if "source" in df.columns else {}
    else:
        cls_counts = {}
        rt_counts = {}
        src_counts = {}
    print(
        f"[earnings] DONE\n"
        f"  range:           {args.start} -> {args.end}\n"
        f"  per-source stats: {stats_list}\n"
        f"  events_after_dedup: {len(df)}\n"
        f"  unique_symbols:  {df['symbol'].nunique() if not df.empty else 0}\n"
        f"  announce_class:  {cls_counts}\n"
        f"  result_type:     {rt_counts}\n"
        f"  source:          {src_counts}\n"
        f"  out:             {args.out_path}"
    )
    n_failed = sum(s.get("chunks_failed", 0) for s in stats_list if isinstance(s, dict))
    return 0 if n_failed == 0 else 4


if __name__ == "__main__":
    sys.exit(main())
