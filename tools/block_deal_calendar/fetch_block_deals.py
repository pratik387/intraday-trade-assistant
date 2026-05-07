"""NSE + BSE block-deal historical scraper.

CLI:
    python tools/block_deal_calendar/fetch_block_deals.py \\
        --start 2023-01-01 --end 2026-04-30

Output:
    data/block_deals/block_deals_events.parquet

Schema (one row per disclosed block-deal trade — multi-line entries kept as
distinct rows; aggregation by (trade_date, symbol, side) happens at sanity
stage):

    trade_date         (date — block-deal session date, IST-naive)
    symbol             (string, "NSE:<TICKER>" form)
    raw_symbol         (raw NSE symbol or BSE ScripName as published)
    client_name        (raw — institutional name disclosed)
    buy_or_sell        ('BUY' or 'SELL')
    qty                (int — shares)
    trade_price        (float — block trade price / weighted-avg price)
    trade_value_cr     (float — qty * price / 1e7, in INR crore)
    exchange           ('NSE' or 'BSE')
    company_name       (raw exchange-published company / scrip name)

Sources:
- NSE: ``/api/historicalOR/bulk-block-short-deals?optionType=block_deals
       &from=DD-MM-YYYY&to=DD-MM-YYYY``
       Anti-bot: requires _abck/bm_sz cookies seeded from
       /report-detail/display-bulk-and-block-deals. We use curl_cffi
       chrome-impersonation to clear the Akamai challenge. The endpoint
       returns at most ~70 rows per response (hard cap), so we chunk by week.

- BSE: ``api.bseindia.com/BseIndiaAPI/api/BulknBlockBETADwnld/w``
       Returns CSV. No row-cap up to 4yr range. Single-request fetch for the
       whole window. Endpoint discovered from
       www.bseindia.com/assets/includenew/js/main-*.js (Angular bundle).

Adapts the politeness / retry pattern from
``tools/earnings_calendar/fetch_earnings.py``.
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests

try:
    from curl_cffi import requests as crequests  # type: ignore
    _HAS_CURL_CFFI = True
except ImportError:  # pragma: no cover
    _HAS_CURL_CFFI = False


# ---------------------------------------------------------------------------
# Constants — paths / endpoints. CLI flags can override.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OUT_PATH = _REPO_ROOT / "data" / "block_deals" / "block_deals_events.parquet"
_FNO_UNIVERSE_PATH = _REPO_ROOT / "assets" / "fno_liquid_200.csv"

_NSE_BASE = "https://www.nseindia.com"
_NSE_HOME = "https://www.nseindia.com/"
_NSE_DETAIL = "https://www.nseindia.com/report-detail/display-bulk-and-block-deals"
_NSE_API = "https://www.nseindia.com/api/historicalOR/bulk-block-short-deals"

_BSE_DOWNLOAD = (
    "https://api.bseindia.com/BseIndiaAPI/api/BulknBlockBETADwnld/w"
)
_BSE_REFERER = (
    "https://www.bseindia.com/markets/equity/EQReports/BulknBlockDeals?flag=2"
)

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

# NSE response hard-cap observed empirically: ~70 rows per response. We chunk
# weekly (max ~10-30 deals/week typical) to stay safely below the cap, and
# recursively split any window that *did* hit the cap (which indicates the
# response was truncated and we need finer granularity).
_NSE_CHUNK_DAYS = 7
_NSE_ROW_CAP = 70


# ---------------------------------------------------------------------------
# Generic helpers.
# ---------------------------------------------------------------------------

def _fmt_nse(d: date) -> str:
    """NSE date param format: DD-MM-YYYY."""
    return d.strftime("%d-%m-%Y")


def _fmt_bse(d: date) -> str:
    """BSE date param format: DD/MM/YYYY."""
    return d.strftime("%d/%m/%Y")


def _parse_nse_date(s: str) -> Optional[date]:
    """NSE response BD_DT_DATE: '23-JAN-2025'."""
    try:
        return datetime.strptime(s.strip(), "%d-%b-%Y").date()
    except (ValueError, AttributeError):
        return None


def _parse_bse_date(s: str) -> Optional[date]:
    """BSE CSV Deal Date — accepts both 'DD/MM/YYYY' and 'DD-MM-YYYY' (BSE
    response uses both depending on the year window)."""
    if not s:
        return None
    raw = s.strip()
    for fmt in ("%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def _week_chunks(start: date, end: date, chunk_days: int) -> list[tuple[date, date]]:
    """Yield (chunk_start, chunk_end) pairs covering [start, end] in `chunk_days` windows."""
    chunks: list[tuple[date, date]] = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), end)
        chunks.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)
    return chunks


# ---------------------------------------------------------------------------
# NSE client — uses curl_cffi for chrome impersonation (clears Akamai _abck).
# ---------------------------------------------------------------------------

class NSEBlockDealClient:
    """NSE block-deal historical scraper. Bootstraps Akamai cookies."""

    def __init__(
        self,
        sleep_secs: float = 5.0,
        max_backoff_secs: float = 60.0,
        timeout_secs: float = 30.0,
        impersonate: str = "chrome120",
    ) -> None:
        if not _HAS_CURL_CFFI:
            raise RuntimeError(
                "curl_cffi is required for NSE block-deal scraping (Akamai bot "
                "challenge); pip install curl_cffi"
            )
        self.sleep_secs = sleep_secs
        self.max_backoff_secs = max_backoff_secs
        self.timeout_secs = timeout_secs
        self.session = crequests.Session(impersonate=impersonate)
        self._bootstrap()

    def _bootstrap(self) -> None:
        """Visit home + report-detail to seed _abck/bm_sz cookies."""
        try:
            self.session.get(_NSE_HOME, timeout=self.timeout_secs)
        except Exception as e:
            print(f"[block_deals/nse] home bootstrap warn: {e}", file=sys.stderr)
        time.sleep(2)
        try:
            self.session.get(
                _NSE_DETAIL,
                headers={"Referer": _NSE_HOME},
                timeout=self.timeout_secs,
            )
        except Exception as e:
            print(f"[block_deals/nse] detail bootstrap warn: {e}", file=sys.stderr)
        time.sleep(2)

    def get_window(
        self, start: date, end: date, max_retries: int = 5,
    ) -> Optional[list[dict]]:
        """GET block-deals for [start, end] inclusive. Returns list of raw rows."""
        params = {
            "optionType": "block_deals",
            "from": _fmt_nse(start),
            "to": _fmt_nse(end),
        }
        backoff = self.sleep_secs
        for attempt in range(1, max_retries + 1):
            try:
                r = self.session.get(
                    _NSE_API,
                    params=params,
                    headers={"Referer": _NSE_DETAIL},
                    timeout=self.timeout_secs,
                )
                if r.status_code == 200:
                    try:
                        payload = r.json()
                    except Exception:
                        # Likely an Akamai HTML challenge; re-bootstrap.
                        print(
                            f"[block_deals/nse] non-JSON 200 attempt {attempt} "
                            f"window {start}->{end}; re-bootstrapping",
                            file=sys.stderr,
                        )
                        self._bootstrap()
                        time.sleep(backoff)
                        backoff = min(backoff * 1.5, self.max_backoff_secs)
                        continue
                    if isinstance(payload, dict):
                        if "data" in payload and isinstance(payload["data"], list):
                            return payload["data"]
                        if payload.get("error"):
                            print(
                                f"[block_deals/nse] error payload: "
                                f"{payload.get('showMessage') or payload}",
                                file=sys.stderr,
                            )
                            return []
                    return []
                if r.status_code in (401, 403, 503):
                    print(
                        f"[block_deals/nse] {r.status_code} attempt {attempt} "
                        f"window {start}->{end}; re-bootstrapping",
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
                        f"[block_deals/nse] {r.status_code} attempt {attempt}; "
                        f"sleeping {sleep_for:.1f}s",
                        file=sys.stderr,
                    )
                    time.sleep(sleep_for)
                    backoff = min(backoff * 1.5, self.max_backoff_secs)
                    continue
                print(
                    f"[block_deals/nse] HTTP {r.status_code} window "
                    f"{start}->{end}; giving up",
                    file=sys.stderr,
                )
                return None
            except Exception as e:
                print(
                    f"[block_deals/nse] transport error attempt {attempt}: {e}",
                    file=sys.stderr,
                )
                time.sleep(backoff)
                backoff = min(backoff * 1.5, self.max_backoff_secs)
        print(
            f"[block_deals/nse] exhausted retries window {start}->{end}",
            file=sys.stderr,
        )
        return None


def parse_nse_rows(items: Iterable[dict]) -> list[dict]:
    """Map NSE response rows -> normalized schema."""
    out: list[dict] = []
    for it in items:
        d = _parse_nse_date(it.get("BD_DT_DATE", ""))
        if d is None:
            continue
        sym = (it.get("BD_SYMBOL") or "").strip().upper()
        if not sym:
            continue
        side_raw = (it.get("BD_BUY_SELL") or "").strip().upper()
        if side_raw not in {"BUY", "SELL"}:
            # Sometimes BD_BUY_SELL = 'P'/'S' on NSE; normalize.
            if side_raw == "P":
                side = "BUY"
            elif side_raw == "S":
                side = "SELL"
            else:
                continue
        else:
            side = side_raw
        try:
            qty = int(it.get("BD_QTY_TRD") or 0)
        except (TypeError, ValueError):
            continue
        try:
            price = float(it.get("BD_TP_WATP") or 0.0)
        except (TypeError, ValueError):
            continue
        if qty <= 0 or price <= 0.0:
            continue
        value_cr = qty * price / 1e7
        out.append(
            {
                "trade_date": d,
                "symbol": f"NSE:{sym}",
                "raw_symbol": sym,
                "client_name": (it.get("BD_CLIENT_NAME") or "").strip(),
                "buy_or_sell": side,
                "qty": qty,
                "trade_price": price,
                "trade_value_cr": round(value_cr, 4),
                "exchange": "NSE",
                "company_name": (it.get("BD_SCRIP_NAME") or "").strip(),
            }
        )
    return out


# ---------------------------------------------------------------------------
# BSE client — plain CSV download, no Akamai.
# ---------------------------------------------------------------------------

class BSEBlockDealClient:
    """BSE block-deal historical scraper. Plain CSV download endpoint."""

    def __init__(
        self,
        sleep_secs: float = 3.0,
        max_backoff_secs: float = 60.0,
        timeout_secs: float = 60.0,
    ) -> None:
        self.sleep_secs = sleep_secs
        self.max_backoff_secs = max_backoff_secs
        self.timeout_secs = timeout_secs
        self.session = requests.Session()
        self.session.headers.update(
            {
                **_BROWSER_HEADERS,
                "Accept": "text/csv, */*",
                "Origin": "https://www.bseindia.com",
                "Referer": _BSE_REFERER,
            }
        )

    def get_window(
        self, start: date, end: date, max_retries: int = 5,
    ) -> Optional[str]:
        """Fetch block-deal CSV for [start, end] inclusive."""
        params = {
            "DealType": "2",  # 2 = Block, 1 = Bulk
            "sc_code": "",
            "FDate": _fmt_bse(start),
            "TDate": _fmt_bse(end),
        }
        backoff = self.sleep_secs
        for attempt in range(1, max_retries + 1):
            try:
                r = self.session.get(
                    _BSE_DOWNLOAD, params=params, timeout=self.timeout_secs
                )
                if r.status_code == 200:
                    text = r.text
                    if not text:
                        return ""
                    # Sanity: first line should look like a CSV header.
                    if text.lstrip().lower().startswith("<"):
                        # HTML — usually a transient 'page moved'; retry once.
                        print(
                            f"[block_deals/bse] HTML body attempt {attempt}; "
                            f"retrying",
                            file=sys.stderr,
                        )
                        time.sleep(backoff)
                        backoff = min(backoff * 1.5, self.max_backoff_secs)
                        continue
                    return text
                if r.status_code == 429 or 500 <= r.status_code < 600:
                    retry_after = r.headers.get("Retry-After")
                    sleep_for = (
                        float(retry_after) if retry_after and retry_after.isdigit()
                        else backoff
                    )
                    print(
                        f"[block_deals/bse] {r.status_code} attempt {attempt}; "
                        f"sleeping {sleep_for:.1f}s",
                        file=sys.stderr,
                    )
                    time.sleep(sleep_for)
                    backoff = min(backoff * 1.5, self.max_backoff_secs)
                    continue
                print(
                    f"[block_deals/bse] HTTP {r.status_code} window "
                    f"{start}->{end}; giving up",
                    file=sys.stderr,
                )
                return None
            except requests.RequestException as e:
                print(
                    f"[block_deals/bse] transport error attempt {attempt}: {e}",
                    file=sys.stderr,
                )
                time.sleep(backoff)
                backoff = min(backoff * 1.5, self.max_backoff_secs)
        print(
            f"[block_deals/bse] exhausted retries window {start}->{end}",
            file=sys.stderr,
        )
        return None


def parse_bse_csv(text: str, fno_universe: set[str]) -> list[dict]:
    """Parse BSE block-deals CSV.

    BSE CSV columns:
        Deal Date, Security Code, Company, Client Name, Deal Type, Quantity, Price

    Deal Type values: P (Purchase / Buy), S (Sell). Maps to NSE BD_SYMBOL via
    `Company` column (BSE publishes the NSE-style symbol there for dual-listed
    stocks).

    fno_universe: set of bare NSE tickers used to attach the "NSE:<sym>" form
    when the BSE company string matches a known NSE ticker. Non-matched names
    keep symbol = f"BSE:{Security Code}".
    """
    out: list[dict] = []
    if not text:
        return out
    reader = csv.reader(io.StringIO(text))
    header_seen = False
    for row in reader:
        if not row:
            continue
        if not header_seen:
            # Detect/skip header
            joined = ",".join(row).lower()
            if "deal date" in joined or "trade date" in joined:
                header_seen = True
                continue
            header_seen = True
            # fall through with first row treated as data
        if len(row) < 7:
            continue
        deal_date_s, sec_code, company, client_name, deal_type, qty_s, price_s = (
            row[0], row[1], row[2], row[3], row[4], row[5], row[6]
        )
        d = _parse_bse_date(deal_date_s)
        if d is None:
            continue
        side_raw = (deal_type or "").strip().upper()
        if side_raw == "P":
            side = "BUY"
        elif side_raw == "S":
            side = "SELL"
        else:
            continue
        try:
            qty = int(float((qty_s or "0").replace(",", "").strip()))
        except (TypeError, ValueError):
            continue
        try:
            price = float((price_s or "0").replace(",", "").strip())
        except (TypeError, ValueError):
            continue
        if qty <= 0 or price <= 0:
            continue
        bse_company = (company or "").strip().upper()
        # Normalize symbol: BSE often publishes the NSE-style 'short_name';
        # if it matches an NSE F&O ticker, attach NSE: prefix.
        if bse_company in fno_universe:
            symbol = f"NSE:{bse_company}"
        else:
            # Keep BSE-prefixed for cross-validation; downstream filter can
            # discard non-NSE-mapped tickers if F&O 200 alignment is required.
            symbol = f"BSE:{bse_company}" if bse_company else f"BSE:{sec_code.strip()}"
        out.append(
            {
                "trade_date": d,
                "symbol": symbol,
                "raw_symbol": bse_company,
                "client_name": (client_name or "").strip(),
                "buy_or_sell": side,
                "qty": qty,
                "trade_price": price,
                "trade_value_cr": round(qty * price / 1e7, 4),
                "exchange": "BSE",
                "company_name": bse_company,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Universe loader (mirror earnings_calendar pattern).
# ---------------------------------------------------------------------------

def load_fno_universe(path: Path = _FNO_UNIVERSE_PATH) -> set[str]:
    """Returns a set of bare NSE ticker symbols (e.g. {'RELIANCE', 'TCS', ...})."""
    df = pd.read_csv(path)
    col = df.columns[0]
    syms = df[col].dropna().astype(str).str.strip().tolist()
    bare: set[str] = set()
    for s in syms:
        if s.upper().startswith("NSE:"):
            bare.add(s.split(":", 1)[1].upper())
        else:
            bare.add(s.upper())
    return bare


# ---------------------------------------------------------------------------
# End-to-end ingestion.
# ---------------------------------------------------------------------------

def _fetch_nse_window_recursive(
    client: "NSEBlockDealClient",
    start: date,
    end: date,
    *,
    sleep_secs: float,
    stats: dict,
    depth: int = 0,
) -> list[dict]:
    """Fetch a window, splitting recursively if the 70-row cap is hit.

    The NSE endpoint hard-caps at ~70 rows per response. When we hit that
    threshold the response is truncated; we split the window in half and
    recurse. Single-day windows are accepted even if they hit the cap (a
    single trading session can legitimately have 70+ block deals during
    earnings season).
    """
    items = client.get_window(start, end)
    if items is None:
        stats["chunks_failed"] += 1
        time.sleep(sleep_secs)
        return []

    n = len(items)
    span_days = (end - start).days + 1
    indent = "  " * depth
    if n >= _NSE_ROW_CAP and span_days > 1:
        print(
            f"[block_deals/nse]{indent}  HIT-CAP {n} rows on {start}->{end} "
            f"(span={span_days}d); splitting",
            file=sys.stderr,
        )
        time.sleep(sleep_secs)
        mid = start + timedelta(days=span_days // 2 - 1)
        left = _fetch_nse_window_recursive(
            client, start, mid,
            sleep_secs=sleep_secs, stats=stats, depth=depth + 1,
        )
        right = _fetch_nse_window_recursive(
            client, mid + timedelta(days=1), end,
            sleep_secs=sleep_secs, stats=stats, depth=depth + 1,
        )
        # Dedup by (BD_DT_DATE, BD_SYMBOL, BD_CLIENT_NAME, BD_BUY_SELL,
        # BD_QTY_TRD, BD_TP_WATP) just in case the split overlaps weirdly.
        # Splits are non-overlapping by construction, so this is defensive.
        return left + right

    if n >= _NSE_ROW_CAP:
        print(
            f"[block_deals/nse]{indent}  WARN cap-hit on single day {start} "
            f"({n} rows) — accepted",
            file=sys.stderr,
        )

    rows = parse_nse_rows(items)
    stats["raw_records"] += n
    print(
        f"[block_deals/nse]{indent}  raw={n} kept={len(rows)} window={start}->{end}",
        file=sys.stderr,
    )
    time.sleep(sleep_secs)
    return rows


def fetch_nse_range(
    start: date, end: date, *, sleep_secs: float = 5.0,
) -> tuple[list[dict], dict]:
    """Scrape NSE block-deals across [start, end]. Returns (rows, stats).

    Splits at 7-day chunks; recursively bisects any window that hits the
    ~70-row response cap.
    """
    client = NSEBlockDealClient(sleep_secs=sleep_secs)
    chunks = _week_chunks(start, end, _NSE_CHUNK_DAYS)
    stats = {
        "chunks_attempted": 0,
        "chunks_ok": 0,
        "chunks_failed": 0,
        "raw_records": 0,
    }
    all_rows: list[dict] = []
    for i, (cs, ce) in enumerate(chunks, start=1):
        stats["chunks_attempted"] += 1
        print(
            f"[block_deals/nse] [{i}/{len(chunks)}] {cs} -> {ce} ...",
            file=sys.stderr,
        )
        rows_before = stats["chunks_failed"]
        rows = _fetch_nse_window_recursive(
            client, cs, ce,
            sleep_secs=sleep_secs, stats=stats,
        )
        if stats["chunks_failed"] > rows_before:
            # already counted inside helper; nothing more to do
            continue
        all_rows.extend(rows)
        stats["chunks_ok"] += 1
        print(
            f"[block_deals/nse]   running_total={len(all_rows)}",
            file=sys.stderr,
        )
    return all_rows, stats


def fetch_bse_range(
    start: date,
    end: date,
    *,
    fno_universe: set[str],
    sleep_secs: float = 3.0,
) -> tuple[list[dict], dict]:
    """Scrape BSE block-deals across [start, end] (single CSV download)."""
    client = BSEBlockDealClient(sleep_secs=sleep_secs)
    stats = {
        "chunks_attempted": 1,
        "chunks_ok": 0,
        "chunks_failed": 0,
        "raw_records": 0,
    }
    print(f"[block_deals/bse] downloading {start} -> {end} ...", file=sys.stderr)
    text = client.get_window(start, end)
    if text is None:
        stats["chunks_failed"] = 1
        return [], stats
    rows = parse_bse_csv(text, fno_universe)
    # Approximate raw count from CSV (lines minus header)
    line_count = len(text.strip().split("\n"))
    stats["raw_records"] = max(0, line_count - 1)
    stats["chunks_ok"] = 1
    print(
        f"[block_deals/bse]   raw_lines={line_count} kept={len(rows)}",
        file=sys.stderr,
    )
    return rows, stats


# ---------------------------------------------------------------------------
# Persistence.
# ---------------------------------------------------------------------------

_OUTPUT_COLUMNS = [
    "trade_date",
    "symbol",
    "raw_symbol",
    "client_name",
    "buy_or_sell",
    "qty",
    "trade_price",
    "trade_value_cr",
    "exchange",
    "company_name",
]


def write_events(
    rows: list[dict],
    out_path: Path = _OUT_PATH,
    *,
    merge_existing: bool = True,
) -> pd.DataFrame:
    """Dedupe + (optionally) merge with existing parquet, write to disk."""
    df_new = (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(columns=_OUTPUT_COLUMNS)
    )
    if merge_existing and out_path.exists():
        try:
            df_old = pd.read_parquet(out_path)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        except (OSError, ValueError) as e:
            print(
                f"[block_deals] could not read existing {out_path}: {e}; "
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

    # Final dedup (re-runs that overlap the same windows).
    dedup_keys = [
        "trade_date", "symbol", "client_name", "buy_or_sell",
        "qty", "trade_price", "exchange",
    ]
    df_all = df_all.drop_duplicates(subset=dedup_keys, keep="first")
    df_all = df_all.sort_values(
        ["trade_date", "exchange", "symbol", "buy_or_sell"]
    ).reset_index(drop=True)
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
            "Backfill NSE+BSE block-deal events to "
            "data/block_deals/block_deals_events.parquet."
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
        "--fno-universe", type=Path, default=_FNO_UNIVERSE_PATH,
        help=f"F&O universe CSV (default: {_FNO_UNIVERSE_PATH})",
    )
    parser.add_argument(
        "--nse-sleep-secs", type=float, default=5.0,
        help="sleep between NSE chunk requests (>=5 recommended)",
    )
    parser.add_argument(
        "--bse-sleep-secs", type=float, default=3.0,
        help="sleep between BSE chunk requests",
    )
    parser.add_argument(
        "--skip-nse", action="store_true", help="skip NSE ingestion",
    )
    parser.add_argument(
        "--skip-bse", action="store_true", help="skip BSE ingestion",
    )
    parser.add_argument(
        "--no-merge-existing", action="store_true",
        help="overwrite parquet instead of merging with existing rows",
    )
    args = parser.parse_args(argv)

    if args.start > args.end:
        parser.error("--start must be <= --end")

    fno = load_fno_universe(args.fno_universe)
    print(
        f"[block_deals] F&O universe size: {len(fno)} ; "
        f"window: {args.start} -> {args.end}",
        file=sys.stderr,
    )

    all_rows: list[dict] = []
    stats_combined = {"nse": {}, "bse": {}}

    if not args.skip_nse:
        nse_rows, nse_stats = fetch_nse_range(
            args.start, args.end, sleep_secs=args.nse_sleep_secs,
        )
        stats_combined["nse"] = nse_stats
        all_rows.extend(nse_rows)

    if not args.skip_bse:
        bse_rows, bse_stats = fetch_bse_range(
            args.start, args.end,
            fno_universe=fno,
            sleep_secs=args.bse_sleep_secs,
        )
        stats_combined["bse"] = bse_stats
        all_rows.extend(bse_rows)

    df = write_events(
        all_rows, args.out_path,
        merge_existing=not args.no_merge_existing,
    )

    # Summary
    if not df.empty:
        df["_y"] = pd.to_datetime(df["trade_date"]).dt.year
        by_year = df.groupby(["_y", "exchange"]).size().to_dict()
        df = df.drop(columns=["_y"])
    else:
        by_year = {}
    print(
        f"[block_deals] DONE\n"
        f"  range:        {args.start} -> {args.end}\n"
        f"  nse_stats:    {stats_combined.get('nse')}\n"
        f"  bse_stats:    {stats_combined.get('bse')}\n"
        f"  total rows:   {len(df)}\n"
        f"  by_year_xch:  {by_year}\n"
        f"  out:          {args.out_path}"
    )
    nse_failed = stats_combined.get("nse", {}).get("chunks_failed", 0)
    bse_failed = stats_combined.get("bse", {}).get("chunks_failed", 0)
    return 0 if (nse_failed + bse_failed) == 0 else 4


if __name__ == "__main__":
    sys.exit(main())
