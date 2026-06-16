"""NSE split / bonus ex-date scraper.

Pulls BONUS and FACE-VALUE-SPLIT corporate-action rows from the NSE
``/api/corporates-corporateActions`` endpoint and writes them to
``data/corporate_actions/split_bonus_events.parquet`` — the CA-events file the
multi-day CNC/MTF executor reads (mtf_capitulation_handlers._load_ca_ex_dates)
to EXCLUDE names with a split/bonus ex-date inside the K-day hold (the broker's
qty adjustment on an ex-date would otherwise corrupt the position's PnL).

Mirrors tools/dividend_ex_date/fetch_dividends.py (same NSE endpoint, cookie
bootstrap, half-month chunking, polite retry) — only the subject filter + parser
differ. Per-subject fetch (BONUS, SPLIT) keeps result sets tiny (splits/bonuses
are rare), well under NSE's ~1000-row/window cap, so completeness is never at
risk from dividend-season volume.

Output columns (symbol + ex_date are the keys the executor reads):
    symbol          (str, "NSE:<ticker>")
    ex_date         (date, IST-naive)
    ca_type         ("bonus" | "split")
    ratio           (str | None, e.g. "1:1" / "10:1", best-effort)
    record_date     (date, IST-naive | None)
    source          ("NSE_corp_actions")
    isin            (str | None)
    company_name    (str | None)
    subject         (str, raw NSE subject)
    series          (str | None)
    broadcast_date  (datetime, IST-naive | None)

NOTE: the exact NSE `subject` query tokens can be lax — the parser double-checks
the subject text, and the run summary prints per-type counts; if a type comes
back 0, verify the token on a live run.

CLI:
    python tools/corporate_actions/fetch_split_bonus.py --start 2023-01-01 --end 2026-06-30
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import requests

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OUT_PATH = _REPO_ROOT / "data" / "corporate_actions" / "split_bonus_events.parquet"

_BASE_URL = "https://www.nseindia.com"
_BOOTSTRAP_CA_URL = "https://www.nseindia.com/companies-listing/corporate-filings-actions"
_BOOTSTRAP_ANN_URL = "https://www.nseindia.com/companies-listing/corporate-filings-announcements"
_API_CA_URL = "https://www.nseindia.com/api/corporates-corporateActions"

# NSE corp-action subject query tokens for the two CA types we exclude on.
_SUBJECTS = ("BONUS", "SPLIT")

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": _BOOTSTRAP_CA_URL,
    "Connection": "keep-alive",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Dest": "empty",
}

_OUT_COLUMNS = [
    "symbol", "ex_date", "ca_type", "ratio", "record_date",
    "source", "isin", "company_name", "subject", "series", "broadcast_date",
]

_NULL_TOKENS = {"", "-", "N/A", "NA", "null", "NULL", "None"}


# ---------------------------------------------------------------------------
# Field parsing (mirrors fetch_dividends)
# ---------------------------------------------------------------------------

def _parse_nse_date(raw: Optional[str]) -> Optional[date]:
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip()
    if s in _NULL_TOKENS:
        return None
    for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _parse_nse_datetime(raw: Optional[str]) -> Optional[datetime]:
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip()
    if s in _NULL_TOKENS:
        return None
    for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    d = _parse_nse_date(s)
    return datetime.combine(d, datetime.min.time()) if d is not None else None


def classify_ca_type(subject: str) -> Optional[str]:
    """Map an NSE subject to 'bonus' | 'split' | None (not a split/bonus)."""
    s = (subject or "").lower()
    if "bonus" in s:
        return "bonus"
    if "split" in s or "sub-division" in s or "sub division" in s or "subdivision" in s:
        return "split"
    return None


_RATIO_RE = re.compile(r"(\d+)\s*:\s*(\d+)")
_SPLIT_FROM_TO_RE = re.compile(
    r"from\s*(?:rs\.?|re\.?|inr)?\s*\.?\s*(\d+(?:\.\d+)?)\s*/?-?\s*to\s*"
    r"(?:rs\.?|re\.?|inr)?\s*\.?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def extract_ratio(subject: str) -> Optional[str]:
    """Best-effort ratio string (e.g. '1:1', '10:1'). None if not parseable.
    Metadata only — exclusion keys on ex_date, not ratio."""
    if not subject:
        return None
    m = _RATIO_RE.search(subject)
    if m:
        return f"{m.group(1)}:{m.group(2)}"
    m = _SPLIT_FROM_TO_RE.search(subject)
    if m:
        # face-value split "from 10 to 1" -> 10:1
        a, b = m.group(1), m.group(2)
        a = a[:-2] if a.endswith(".0") else a
        b = b[:-2] if b.endswith(".0") else b
        return f"{a}:{b}"
    return None


def parse_split_bonus_actions(items: Iterable[dict]) -> List[dict]:
    """Convert raw NSE corp-actions rows into normalised split/bonus events.

    Keeps only rows whose subject classifies as bonus/split AND has an ex-date
    (double-checks the subject, since the API's subject filter can be lax)."""
    rows: List[dict] = []
    for it in items:
        sym = (it.get("symbol") or "").strip().upper()
        if not sym:
            continue
        subject = (it.get("subject") or "").strip()
        ca_type = classify_ca_type(subject)
        if ca_type is None:
            continue
        ex_date = _parse_nse_date(it.get("exDate"))
        if ex_date is None:
            continue
        series = it.get("series")
        if isinstance(series, str):
            series = series.strip() or None
        rows.append({
            "symbol": f"NSE:{sym}",
            "ex_date": ex_date,
            "ca_type": ca_type,
            "ratio": extract_ratio(subject),
            "record_date": _parse_nse_date(it.get("recDate")),
            "source": "NSE_corp_actions",
            "isin": (it.get("isin") or None) or None,
            "company_name": it.get("comp"),
            "subject": subject,
            "series": series,
            "broadcast_date": _parse_nse_datetime(it.get("caBroadcastDate")),
        })
    return rows


# ---------------------------------------------------------------------------
# NSE HTTP client (cookie-bootstrap + polite retry) — mirrors fetch_dividends
# ---------------------------------------------------------------------------

class NSESession:
    def __init__(self, sleep_secs: float = 5.0, max_backoff_secs: float = 60.0,
                 timeout_secs: float = 30.0) -> None:
        self.sleep_secs = sleep_secs
        self.max_backoff_secs = max_backoff_secs
        self.timeout_secs = timeout_secs
        self.session = requests.Session()
        self.session.headers.update(_DEFAULT_HEADERS)
        self._bootstrap()

    def _bootstrap(self) -> None:
        for url in (_BOOTSTRAP_CA_URL, _BOOTSTRAP_ANN_URL):
            try:
                self.session.get(url, timeout=self.timeout_secs)
            except requests.RequestException:
                continue
        if not self.session.cookies:
            try:
                self.session.get(_BASE_URL, timeout=self.timeout_secs)
            except requests.RequestException:
                pass
        time.sleep(2)

    def get_json(self, url: str, params: Optional[dict] = None, max_retries: int = 5):
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
                        self._bootstrap(); time.sleep(backoff)
                        backoff = min(backoff * 1.5, self.max_backoff_secs); continue
                if r.status_code in (401, 403):
                    self._bootstrap(); time.sleep(backoff)
                    backoff = min(backoff * 1.5, self.max_backoff_secs); continue
                if r.status_code == 429 or 500 <= r.status_code < 600:
                    ra = r.headers.get("Retry-After")
                    sleep_for = float(ra) if ra and ra.isdigit() else backoff
                    print(f"[split_bonus] {r.status_code} attempt {attempt}; sleeping {sleep_for:.1f}s", file=sys.stderr)
                    time.sleep(sleep_for)
                    backoff = min(backoff * 1.5, self.max_backoff_secs); continue
                print(f"[split_bonus] HTTP {r.status_code} {url} params={params}; giving up", file=sys.stderr)
                return None
            except requests.RequestException as e:
                print(f"[split_bonus] transport error attempt {attempt}: {e}", file=sys.stderr)
                time.sleep(backoff)
                backoff = min(backoff * 1.5, self.max_backoff_secs)
        print(f"[split_bonus] exhausted retries {url} params={params} last={last_status}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Date-range scraping
# ---------------------------------------------------------------------------

def _fmt_nse_date(d: date) -> str:
    return d.strftime("%d-%m-%Y")


def _half_month_chunks(start: date, end: date) -> List[tuple]:
    chunks = []
    cur = start
    while cur <= end:
        if cur.day <= 15:
            half_end = cur.replace(day=15)
        else:
            if cur.month == 12:
                next_first = cur.replace(year=cur.year + 1, month=1, day=1)
            else:
                next_first = cur.replace(month=cur.month + 1, day=1)
            half_end = next_first - timedelta(days=1)
        chunks.append((cur, min(half_end, end)))
        cur = min(half_end, end) + timedelta(days=1)
    return chunks


def fetch_split_bonus_range(nse: NSESession, start: date, end: date, *, sleep_secs: float):
    chunks = _half_month_chunks(start, end)
    stats = {"chunks_attempted": 0, "chunks_ok": 0, "chunks_failed": 0, "raw_rows": 0, "kept_rows": 0}
    all_rows: List[dict] = []
    for subject in _SUBJECTS:
        for i, (cs, ce) in enumerate(chunks, start=1):
            stats["chunks_attempted"] += 1
            params = {"index": "equities", "from_date": _fmt_nse_date(cs),
                      "to_date": _fmt_nse_date(ce), "subject": subject}
            print(f"[split_bonus] {subject} [{i}/{len(chunks)}] {cs} -> {ce} ...", file=sys.stderr)
            payload = nse.get_json(_API_CA_URL, params=params)
            if payload is None:
                stats["chunks_failed"] += 1; time.sleep(sleep_secs); continue
            items = payload if isinstance(payload, list) else []
            stats["raw_rows"] += len(items)
            rows = parse_split_bonus_actions(items)
            stats["kept_rows"] += len(rows)
            all_rows.extend(rows)
            stats["chunks_ok"] += 1
            time.sleep(sleep_secs)
    return all_rows, stats


# ---------------------------------------------------------------------------
# Dedup + persistence
# ---------------------------------------------------------------------------

def dedupe(rows: List[dict]) -> List[dict]:
    if not rows:
        return rows
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["symbol", "ex_date", "ca_type", "broadcast_date"],
                        ascending=[True, True, True, False], na_position="last")
    df = df.drop_duplicates(subset=["symbol", "ex_date", "ca_type", "subject"], keep="first")
    return df.to_dict("records")


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in _OUT_COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df[_OUT_COLUMNS]


def write_events(rows: List[dict], out_path: Path = _OUT_PATH, *, merge_existing: bool = True) -> pd.DataFrame:
    df_new = _ensure_columns(pd.DataFrame(rows) if rows else pd.DataFrame(columns=_OUT_COLUMNS))
    if merge_existing and out_path.exists():
        try:
            df_all = pd.concat([_ensure_columns(pd.read_parquet(out_path)), df_new], ignore_index=True)
        except (OSError, ValueError):
            df_all = df_new
    else:
        df_all = df_new
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if df_all.empty:
        df_all.to_parquet(out_path, index=False)
        return df_all
    df_all = _ensure_columns(pd.DataFrame(dedupe(df_all.to_dict("records"))))
    df_all = df_all.sort_values(["ex_date", "symbol"]).reset_index(drop=True)
    df_all.to_parquet(out_path, index=False)
    return df_all


def _parse_date_arg(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Backfill NSE split/bonus ex-dates -> data/corporate_actions/split_bonus_events.parquet")
    p.add_argument("--start", type=_parse_date_arg, required=True, help="range start (YYYY-MM-DD)")
    p.add_argument("--end", type=_parse_date_arg, required=True, help="range end inclusive (YYYY-MM-DD)")
    p.add_argument("--out-path", type=Path, default=_OUT_PATH)
    p.add_argument("--sleep-secs", type=float, default=5.0, help=">=5 recommended for NSE politeness")
    p.add_argument("--no-merge-existing", action="store_true")
    args = p.parse_args(argv)
    if args.start > args.end:
        p.error("--start must be <= --end")

    nse = NSESession(sleep_secs=args.sleep_secs)
    try:
        rows, stats = fetch_split_bonus_range(nse, args.start, args.end, sleep_secs=args.sleep_secs)
    except Exception as e:
        print(f"[split_bonus] scrape FAILED: {e}", file=sys.stderr)
        rows, stats = [], {"error": str(e)}

    df = write_events(rows, args.out_path, merge_existing=not args.no_merge_existing)
    by_type = df["ca_type"].value_counts().to_dict() if not df.empty else {}
    per_year = (pd.to_datetime(df["ex_date"]).dt.year.value_counts().sort_index().to_dict()
                if not df.empty else {})
    print(f"[split_bonus] DONE\n  range: {args.start} -> {args.end}\n  stats: {stats}\n"
          f"  events_after_dedup: {len(df)}\n  by_type: {by_type}\n  per_year: {per_year}\n"
          f"  out: {args.out_path}")
    return 0 if stats.get("chunks_failed", 0) == 0 else 4


if __name__ == "__main__":
    sys.exit(main())
