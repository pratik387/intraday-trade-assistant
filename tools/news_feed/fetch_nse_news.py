"""NSE whole-market corporate-announcements + scheduled-board-meeting feeds.

Feeds the multiday results-day skip (design:
``specs/2026-10-06-multiday-news-skip-and-bulk-tilt-design.md`` section 3).

CLI (matches the ``services/event_feeds.py::refresh_event_feed`` contract,
which invokes ``python -m <refresh_module> --start --end --sleep-secs``):

    python -m tools.news_feed.fetch_nse_news \\
        --start 2026-10-05 --end 2026-10-06 --sleep-secs 1 --forward-days 10

Two streams, two parquet files under ``--out-dir`` (default ``data/news``):

1. ``nse_announcements.parquet`` — ``GET /api/corporate-announcements
   ?index=equities&from_date&to_date`` with NO symbol parameter, i.e. the
   whole market. One call per calendar day of [start, end] (NSE caps rows per
   response; a day is ~500-1,100 rows and stays under it). Merged with the
   existing file and deduplicated on (symbol, an_dt, desc); rows outside the
   fetched window are never dropped.

       symbol        bare NSE ticker as published (e.g. "KARAMTARA")
       an_dt         IST-naive pandas Timestamp of the filing ("06-Oct-2026 16:16:44")
       filing_date   IST-naive Timestamp normalised to midnight (the date column)
       desc          NSE subject, e.g. "Outcome of Board Meeting"
       text          attchmntText truncated to 500 chars
       company_name  sm_name
       source        "nse_announcements"

2. ``nse_event_calendar.parquet`` — ``GET /api/event-calendar?index=equities
   &from_date&to_date`` for [end, end + forward_days]: scheduled board
   meetings ("Financial Results", "Financial Results/Dividend", ...). Rows in
   the re-fetched date range REPLACE the stored rows for that range
   (meetings get rescheduled); rows outside it are kept. Deduplicated on
   (symbol, meeting_date, purpose).

       symbol        bare NSE ticker
       meeting_date  IST-naive Timestamp (midnight)
       purpose       e.g. "Financial Results"
       bm_desc       free text
       company_name  company
       fetched_at    IST-naive Timestamp of the scrape (metadata only)

Anti-bot: NSE needs cookies from a GET of the home page with a Chrome
User-Agent (plain ``requests`` clears it from the PC and the VM). The session
re-bootstraps on 401/403/429, on an empty body and on a non-JSON 200, with
exponential backoff. A single bad chunk is logged and skipped; the process
exits non-zero only when EVERY chunk of a stream failed.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.time_util import _now_naive_ist  # noqa: E402

# ---------------------------------------------------------------------------
# Constants: endpoints, headers, file names. CLI flags override the paths.
# ---------------------------------------------------------------------------

_NSE_HOME = "https://www.nseindia.com/"
_NSE_ANN_REFERER = (
    "https://www.nseindia.com/companies-listing/corporate-filings-announcements"
)
_NSE_ANN_API = "https://www.nseindia.com/api/corporate-announcements"
_NSE_CAL_API = "https://www.nseindia.com/api/event-calendar"

_DEFAULT_OUT_DIR = _REPO_ROOT / "data" / "news"
_ANN_FILE = "nse_announcements.parquet"
_CAL_FILE = "nse_event_calendar.parquet"

_ANN_SOURCE = "nse_announcements"
_TEXT_MAX_CHARS = 500

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": _NSE_ANN_REFERER,
    "Connection": "keep-alive",
}

# One event-calendar call covers up to this many days (154 rows for 26 days
# was fine; chunking keeps a long --forward-days away from the response cap).
_CAL_CHUNK_DAYS = 7

ANN_COLUMNS = [
    "symbol", "an_dt", "filing_date", "desc", "text", "company_name", "source",
]
ANN_DEDUPE_KEYS = ["symbol", "an_dt", "desc"]

CAL_COLUMNS = [
    "symbol", "meeting_date", "purpose", "bm_desc", "company_name", "fetched_at",
]
CAL_DEDUPE_KEYS = ["symbol", "meeting_date", "purpose"]


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def _fmt_nse(d: date) -> str:
    """NSE query-param date format: DD-MM-YYYY."""
    return d.strftime("%d-%m-%Y")


def day_chunks(start: date, end: date) -> list[date]:
    """Every calendar day in [start, end] inclusive — one API call each."""
    if start > end:
        raise ValueError(f"start {start} > end {end}")
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def window_chunks(start: date, end: date, chunk_days: int) -> list[tuple[date, date]]:
    """[start, end] split into inclusive windows of at most `chunk_days`."""
    out: list[tuple[date, date]] = []
    cur = start
    while cur <= end:
        ce = min(cur + timedelta(days=chunk_days - 1), end)
        out.append((cur, ce))
        cur = ce + timedelta(days=1)
    return out


def parse_an_dt(s: Optional[str]) -> Optional[pd.Timestamp]:
    """'06-Oct-2026 16:16:44' -> IST-naive Timestamp. None when unparseable.

    NSE timestamps are IST wall-clock with no zone; they are stored as-is
    (IST-naive) per CLAUDE.md rule 2.
    """
    if not s or not isinstance(s, str):
        return None
    raw = s.strip()
    for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%d-%b-%Y"):
        try:
            return pd.Timestamp(datetime.strptime(raw, fmt))
        except ValueError:
            continue
    return None


def parse_nse_day(s: Optional[str]) -> Optional[pd.Timestamp]:
    """'06-Oct-2026' -> IST-naive Timestamp at midnight. None when unparseable."""
    if not s or not isinstance(s, str):
        return None
    raw = s.strip()
    for fmt in ("%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            return pd.Timestamp(datetime.strptime(raw, fmt))
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# NSE HTTP session: cookie bootstrap + re-bootstrap + backoff.
# ---------------------------------------------------------------------------

class NSENewsSession:
    """requests.Session wrapper that bootstraps NSE cookies and retries."""

    def __init__(
        self,
        *,
        sleep_secs: float,
        max_backoff_secs: float = 60.0,
        timeout_secs: float = 30.0,
        max_retries: int = 5,
    ) -> None:
        self.sleep_secs = float(sleep_secs)
        self.max_backoff_secs = max_backoff_secs
        self.timeout_secs = timeout_secs
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update(_HEADERS)
        self.bootstrap_count = 0
        self._bootstrap()

    def _bootstrap(self) -> None:
        """GET the home page so NSE sets its session cookies."""
        self.bootstrap_count += 1
        try:
            self.session.get(_NSE_HOME, timeout=self.timeout_secs)
        except requests.RequestException as e:
            print(f"[nse_news] bootstrap warn: {e}", file=sys.stderr)

    def _rebootstrap_and_wait(self, why: str, attempt: int, backoff: float) -> float:
        print(
            f"[nse_news] {why} on attempt {attempt}; re-bootstrapping cookies, "
            f"sleeping {backoff:.1f}s",
            file=sys.stderr,
        )
        self._bootstrap()
        time.sleep(backoff)
        return min(backoff * 1.5, self.max_backoff_secs)

    def get_json(self, url: str, params: dict) -> Optional[list | dict]:
        """GET `url` and return parsed JSON, or None after exhausting retries."""
        backoff = max(self.sleep_secs, 1.0)
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout_secs)
            except requests.RequestException as e:
                print(
                    f"[nse_news] transport error attempt {attempt}: {e}",
                    file=sys.stderr,
                )
                time.sleep(backoff)
                backoff = min(backoff * 1.5, self.max_backoff_secs)
                continue

            if r.status_code == 200:
                if not r.content or not r.content.strip():
                    backoff = self._rebootstrap_and_wait("empty body", attempt, backoff)
                    continue
                try:
                    return r.json()
                except ValueError:
                    backoff = self._rebootstrap_and_wait("non-JSON 200", attempt, backoff)
                    continue

            if r.status_code in (401, 403, 429):
                retry_after = r.headers.get("Retry-After")
                if retry_after and str(retry_after).isdigit():
                    backoff = max(backoff, float(retry_after))
                backoff = self._rebootstrap_and_wait(
                    f"HTTP {r.status_code}", attempt, backoff
                )
                continue

            if 500 <= r.status_code < 600:
                print(
                    f"[nse_news] HTTP {r.status_code} attempt {attempt}; "
                    f"sleeping {backoff:.1f}s",
                    file=sys.stderr,
                )
                time.sleep(backoff)
                backoff = min(backoff * 1.5, self.max_backoff_secs)
                continue

            print(
                f"[nse_news] HTTP {r.status_code} on {url} params={params}; giving up",
                file=sys.stderr,
            )
            return None

        print(
            f"[nse_news] exhausted {self.max_retries} retries for {url} params={params}",
            file=sys.stderr,
        )
        return None


def _rows_from_payload(payload) -> Optional[list]:
    """NSE returns either a bare list or {'data': [...]}. None = unusable."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return data
        if payload.get("error") or payload.get("showMessage"):
            print(f"[nse_news] error payload: {payload}", file=sys.stderr)
            return None
    return None


# ---------------------------------------------------------------------------
# Parsers (pure).
# ---------------------------------------------------------------------------

def parse_announcements(items: Iterable[dict]) -> list[dict]:
    """Map /api/corporate-announcements rows to the ANN_COLUMNS schema."""
    out: list[dict] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        sym = (it.get("symbol") or "").strip().upper()
        if not sym:
            continue
        ts = parse_an_dt(it.get("an_dt")) or parse_an_dt(it.get("sort_date"))
        if ts is None:
            continue
        text = it.get("attchmntText") or ""
        if not isinstance(text, str):
            text = str(text)
        out.append(
            {
                "symbol": sym,
                "an_dt": ts,
                "filing_date": ts.normalize(),
                "desc": (it.get("desc") or "").strip(),
                "text": text.strip()[:_TEXT_MAX_CHARS],
                "company_name": (it.get("sm_name") or "").strip(),
                "source": _ANN_SOURCE,
            }
        )
    return out


def parse_event_calendar(items: Iterable[dict], *, fetched_at: pd.Timestamp) -> list[dict]:
    """Map /api/event-calendar rows to the CAL_COLUMNS schema."""
    out: list[dict] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        sym = (it.get("symbol") or "").strip().upper()
        if not sym:
            continue
        md = parse_nse_day(it.get("date"))
        if md is None:
            continue
        out.append(
            {
                "symbol": sym,
                "meeting_date": md,
                "purpose": (it.get("purpose") or "").strip(),
                "bm_desc": (it.get("bm_desc") or "").strip(),
                "company_name": (it.get("company") or "").strip(),
                "fetched_at": fetched_at,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Fetch loops.
# ---------------------------------------------------------------------------

def fetch_announcements_range(
    session: NSENewsSession, start: date, end: date, *, sleep_secs: float,
) -> tuple[list[dict], dict]:
    """One /api/corporate-announcements call per calendar day in [start, end]."""
    days = day_chunks(start, end)
    stats = {"chunks_attempted": 0, "chunks_ok": 0, "chunks_failed": 0, "raw_records": 0}
    rows: list[dict] = []
    for i, d in enumerate(days, start=1):
        stats["chunks_attempted"] += 1
        params = {"index": "equities", "from_date": _fmt_nse(d), "to_date": _fmt_nse(d)}
        payload = session.get_json(_NSE_ANN_API, params)
        items = _rows_from_payload(payload) if payload is not None else None
        if items is None:
            stats["chunks_failed"] += 1
            print(f"[nse_news/ann] [{i}/{len(days)}] {d} FAILED; skipping", file=sys.stderr)
        else:
            parsed = parse_announcements(items)
            rows.extend(parsed)
            stats["chunks_ok"] += 1
            stats["raw_records"] += len(items)
            print(
                f"[nse_news/ann] [{i}/{len(days)}] {d} raw={len(items)} kept={len(parsed)}",
                file=sys.stderr,
            )
        if i < len(days):
            time.sleep(sleep_secs)
    return rows, stats


def fetch_event_calendar_range(
    session: NSENewsSession, start: date, end: date, *, sleep_secs: float,
    fetched_at: pd.Timestamp,
) -> tuple[list[dict], dict]:
    """/api/event-calendar over [start, end] in windows of _CAL_CHUNK_DAYS."""
    chunks = window_chunks(start, end, _CAL_CHUNK_DAYS)
    stats = {
        "chunks_attempted": 0, "chunks_ok": 0, "chunks_failed": 0, "raw_records": 0,
        # Windows that came back OK: only these ranges get replaced on disk, so
        # a failed sub-window never wipes the meetings already stored for it.
        "ok_windows": [],
    }
    rows: list[dict] = []
    for i, (cs, ce) in enumerate(chunks, start=1):
        stats["chunks_attempted"] += 1
        params = {"index": "equities", "from_date": _fmt_nse(cs), "to_date": _fmt_nse(ce)}
        payload = session.get_json(_NSE_CAL_API, params)
        items = _rows_from_payload(payload) if payload is not None else None
        if items is None:
            stats["chunks_failed"] += 1
            print(
                f"[nse_news/cal] [{i}/{len(chunks)}] {cs}->{ce} FAILED; skipping",
                file=sys.stderr,
            )
        else:
            parsed = parse_event_calendar(items, fetched_at=fetched_at)
            rows.extend(parsed)
            stats["chunks_ok"] += 1
            stats["ok_windows"].append((cs, ce))
            stats["raw_records"] += len(items)
            print(
                f"[nse_news/cal] [{i}/{len(chunks)}] {cs}->{ce} raw={len(items)} "
                f"kept={len(parsed)}",
                file=sys.stderr,
            )
        if i < len(chunks):
            time.sleep(sleep_secs)
    return rows, stats


# ---------------------------------------------------------------------------
# Persistence (merge semantics differ per stream).
# ---------------------------------------------------------------------------

def _read_existing(path: Path, columns: list[str], label: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError) as e:
        print(f"[nse_news/{label}] could not read {path}: {e}; overwriting", file=sys.stderr)
        return pd.DataFrame(columns=columns)


def _coerce_ann(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reindex(columns=ANN_COLUMNS)
    df["an_dt"] = pd.to_datetime(df["an_dt"])
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    for c in ("symbol", "desc", "text", "company_name", "source"):
        df[c] = df[c].astype("string").fillna("").astype(object)
    return df


def _coerce_cal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reindex(columns=CAL_COLUMNS)
    df["meeting_date"] = pd.to_datetime(df["meeting_date"])
    df["fetched_at"] = pd.to_datetime(df["fetched_at"])
    for c in ("symbol", "purpose", "bm_desc", "company_name"):
        df[c] = df[c].astype("string").fillna("").astype(object)
    return df


def merge_announcements(rows: list[dict], out_path: Path) -> pd.DataFrame:
    """Union new rows with the existing file; dedupe on (symbol, an_dt, desc).

    Rows already on disk are never dropped, whatever window was fetched.
    """
    df_new = pd.DataFrame(rows, columns=ANN_COLUMNS)
    df_old = _read_existing(out_path, ANN_COLUMNS, "ann")
    parts = [p for p in (_coerce_ann(df_old), _coerce_ann(df_new)) if not p.empty]
    df_all = pd.concat(parts, ignore_index=True) if parts else _coerce_ann(df_new)
    if not df_all.empty:
        df_all = (
            df_all.drop_duplicates(subset=ANN_DEDUPE_KEYS, keep="last")
            .sort_values(["an_dt", "symbol", "desc"])
            .reset_index(drop=True)
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(out_path, index=False)
    return df_all


def merge_event_calendar(
    rows: list[dict], out_path: Path, *, replace_windows: list[tuple[date, date]],
) -> pd.DataFrame:
    """Replace stored rows whose meeting_date falls inside any of the
    successfully fetched `replace_windows` (inclusive) with the freshly fetched
    rows (meetings get rescheduled); keep every other stored row.
    Dedupe on (symbol, meeting_date, purpose).
    """
    df_new = _coerce_cal(pd.DataFrame(rows, columns=CAL_COLUMNS))
    df_old = _coerce_cal(_read_existing(out_path, CAL_COLUMNS, "cal"))
    if not df_old.empty and replace_windows:
        in_range = pd.Series(False, index=df_old.index)
        for ws, we in replace_windows:
            lo, hi = pd.Timestamp(ws), pd.Timestamp(we)
            in_range |= (df_old["meeting_date"] >= lo) & (df_old["meeting_date"] <= hi)
        df_old = df_old.loc[~in_range]
    parts = [p for p in (df_old, df_new) if not p.empty]
    df_all = pd.concat(parts, ignore_index=True) if parts else df_new
    if not df_all.empty:
        df_all = (
            df_all.drop_duplicates(subset=CAL_DEDUPE_KEYS, keep="last")
            .sort_values(["meeting_date", "symbol", "purpose"])
            .reset_index(drop=True)
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(out_path, index=False)
    return df_all


def _summary_line(label: str, fetched: int, df: pd.DataFrame, date_col: str, stats: dict) -> str:
    if df.empty:
        span = "min=None max=None"
    else:
        span = f"min={df[date_col].min().date()} max={df[date_col].max().date()}"
    return (
        f"[nse_news] {label}: fetched={fetched} after_merge={len(df)} {span} "
        f"chunks ok/failed={stats['chunks_ok']}/{stats['chunks_failed']}"
    )


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def _parse_date_arg(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh NSE whole-market corporate announcements and the scheduled "
            "board-meeting calendar into data/news/*.parquet."
        )
    )
    parser.add_argument("--start", type=_parse_date_arg, required=True,
                        help="announcements range start (YYYY-MM-DD)")
    parser.add_argument("--end", type=_parse_date_arg, required=True,
                        help="announcements range end inclusive (YYYY-MM-DD); also the "
                             "anchor day for the forward event-calendar window")
    parser.add_argument("--sleep-secs", type=float, required=True,
                        help="sleep between NSE calls (seconds)")
    parser.add_argument("--forward-days", type=int, required=True,
                        help="event calendar is fetched for [end, end + forward-days]")
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR,
                        help=f"directory for the two parquet files (path default: "
                             f"{_DEFAULT_OUT_DIR})")
    parser.add_argument("--skip-announcements", action="store_true",
                        help="skip the announcements stream")
    parser.add_argument("--skip-calendar", action="store_true",
                        help="skip the event-calendar stream")
    args = parser.parse_args(argv)

    if args.start > args.end:
        parser.error("--start must be <= --end")
    if args.forward_days < 0:
        parser.error("--forward-days must be >= 0")
    if args.sleep_secs < 0:
        parser.error("--sleep-secs must be >= 0")

    out_dir: Path = args.out_dir
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    fetched_at = _now_naive_ist()

    session = NSENewsSession(sleep_secs=args.sleep_secs)
    total_failures = 0

    if not args.skip_announcements:
        rows, stats = fetch_announcements_range(
            session, args.start, args.end, sleep_secs=args.sleep_secs,
        )
        df = merge_announcements(rows, out_dir / _ANN_FILE)
        print(_summary_line("announcements", len(rows), df, "filing_date", stats))
        if stats["chunks_attempted"] > 0 and stats["chunks_ok"] == 0:
            total_failures += 1

    if not args.skip_calendar:
        cal_start = args.end
        cal_end = args.end + timedelta(days=args.forward_days)
        rows, stats = fetch_event_calendar_range(
            session, cal_start, cal_end, sleep_secs=args.sleep_secs, fetched_at=fetched_at,
        )
        if stats["chunks_ok"] > 0:
            df = merge_event_calendar(
                rows, out_dir / _CAL_FILE, replace_windows=stats["ok_windows"],
            )
        else:
            # Nothing usable came back: do NOT wipe the stored range.
            df = _coerce_cal(_read_existing(out_dir / _CAL_FILE, CAL_COLUMNS, "cal"))
            total_failures += 1
        print(_summary_line("event_calendar", len(rows), df, "meeting_date", stats))

    return 0 if total_failures == 0 else 4


if __name__ == "__main__":
    sys.exit(main())
