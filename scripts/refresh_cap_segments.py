"""Refresh cap-segment classification from NSE Indices.

Pulls the four NSE Indices constituent CSVs (NIFTY 100, NIFTY Midcap 150,
NIFTY Smallcap 250, NIFTY Microcap 250) — the SEBI-aligned market-cap
classification — and produces:

    data/cap_segments/cap_segments_latest.json    (used by live/paper)
    data/cap_segments/cap_segments_<YYYY-MM-DD>.json  (dated archive)

The dated copy lets backtests look up the snapshot in effect on a given
session_date, avoiding look-ahead bias (a stock's classification in 2023
may differ from today's).

Run weekly via cron (Sunday morning is fine — NSE rebalances semi-annually
on Mar/Sep, but lightweight to refresh more often).

Usage:
    python scripts/refresh_cap_segments.py
    python scripts/refresh_cap_segments.py --out-dir data/cap_segments

Source URLs are stable, no auth required. CSV format:
    Company Name, Industry, Symbol, Series, ISIN Code
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import requests

logger = logging.getLogger("refresh_cap_segments")


# NSE Indices canonical URLs. These are the SEBI/AMFI-aligned classification
# (market cap top 100 / 101-250 / 251-500 / 501-750). Anything outside the
# 750 is treated as "unknown" downstream (very illiquid / new listings).
SOURCES: List[Dict[str, str]] = [
    {"segment": "large_cap", "url": "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv"},
    {"segment": "mid_cap",   "url": "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap150list.csv"},
    {"segment": "small_cap", "url": "https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv"},
    {"segment": "micro_cap", "url": "https://www.niftyindices.com/IndexConstituent/ind_niftymicrocap250list.csv"},
]


HEADERS = {
    # niftyindices.com doesn't require auth, but a browser-like UA reduces
    # rate-limit risk if the script ever runs in tight loops.
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/csv,*/*;q=0.9",
}


def _fetch_csv(url: str, *, timeout: int = 20) -> Optional[str]:
    """Fetch CSV body. Returns None if response is not CSV-shaped (some NSE
    URLs intermittently 302 to HTML pages — we tolerate that and skip)."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        logger.warning("fetch failed for %s: %s", url, e)
        return None

    body = resp.text or ""
    # Heuristic: NSE Indices CSV starts with the header "Company Name"
    if not body.lstrip().lower().startswith("company name"):
        logger.warning(
            "non-CSV response for %s (first 80 chars: %r) — skipping",
            url, body[:80],
        )
        return None
    return body


def _parse_constituents(csv_text: str) -> List[str]:
    """Pull the 'Symbol' column from a NSE Indices CSV.

    Format:
        Company Name, Industry, Symbol, Series, ISIN Code
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    symbols: List[str] = []
    for row in reader:
        sym = (row.get("Symbol") or row.get("symbol") or "").strip().upper()
        series = (row.get("Series") or "").strip().upper()
        if not sym:
            continue
        # Only equity series (skip BE — limited trading, ETFs, etc.)
        if series and series not in ("EQ", "BE"):
            continue
        symbols.append(sym)
    return symbols


def build_classification() -> Dict[str, str]:
    """Return {NSE_symbol: cap_segment}.

    Precedence: large_cap > mid_cap > small_cap > micro_cap.
    A symbol can appear in only one list under NSE's methodology, but the
    explicit precedence guards against any overlap from index transition
    snapshots.
    """
    classification: Dict[str, str] = {}
    counts: Dict[str, int] = {s["segment"]: 0 for s in SOURCES}

    # Iterate in precedence order so first-write wins.
    for src in SOURCES:
        seg = src["segment"]
        csv_text = _fetch_csv(src["url"])
        if csv_text is None:
            logger.warning("skipping %s — fetch returned non-CSV", seg)
            continue
        symbols = _parse_constituents(csv_text)
        for sym in symbols:
            key = f"NSE:{sym}"
            if key not in classification:
                classification[key] = seg
                counts[seg] += 1
        logger.info("%-10s | %d symbols from %s", seg, len(symbols), src["url"].rsplit("/", 1)[-1])

    logger.info(
        "TOTALS | large=%d mid=%d small=%d micro=%d | grand=%d",
        counts["large_cap"], counts["mid_cap"], counts["small_cap"],
        counts["micro_cap"], len(classification),
    )
    return classification


def write_outputs(classification: Dict[str, str], out_dir: Path, today: date) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_path = out_dir / "cap_segments_latest.json"
    dated_path = out_dir / f"cap_segments_{today.isoformat()}.json"

    payload = {
        "generated_at": today.isoformat(),
        "source": "niftyindices.com (NIFTY 100 / Midcap 150 / Smallcap 250 / Microcap 250)",
        "n_symbols": len(classification),
        "classification": classification,
    }
    blob = json.dumps(payload, indent=2, sort_keys=True)
    latest_path.write_text(blob, encoding="utf-8")
    dated_path.write_text(blob, encoding="utf-8")
    logger.info("wrote %s and %s (%d symbols)", latest_path, dated_path, len(classification))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path,
                   default=Path(__file__).resolve().parent.parent / "data" / "cap_segments",
                   help="Directory to write cap_segments_latest.json and dated snapshot")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    try:
        classification = build_classification()
    except Exception as e:
        logger.error("aborting — classification build failed: %s", e)
        return 1

    if not classification:
        logger.error("classification is empty; refusing to overwrite outputs")
        return 1

    write_outputs(classification, args.out_dir, date.today())
    return 0


if __name__ == "__main__":
    sys.exit(main())
