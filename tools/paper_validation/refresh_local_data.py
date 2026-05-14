"""Refresh local data parquets that paper/live setups depend on.

Run this manually before starting paper/live. Without it, setups that
depend on local parquets silently no-fire on every signal.

Current scope:
  * `delivery_pct_anomaly_short` needs:
      - data/delivery_pct/delivery_history.parquet  (NSE bhavcopy archive)
      - data/cross_day_rvol/rvol_baseline.parquet   (derived from monthly 5m feathers)

  * `gap_fade_short` + `circuit_t1_fade_short` need NO local parquets — they
    use Kite Historical API exclusively.

Usage:
    python -m tools.paper_validation.refresh_local_data
    python -m tools.paper_validation.refresh_local_data --skip-delivery
    python -m tools.paper_validation.refresh_local_data --skip-rvol --start 2023-01-01

Runtime expectations (measured 2026-05-14, broadband):
  * delivery_pct full rebuild: ~3-5 minutes (network-bound; NSE archive
    fetches with 6 workers, ~875 weekdays for 2023-01..today).
  * cross_day_rvol full rebuild: ~1-2 minutes IF the current month's
    `backtest-cache-download/monthly/{YYYY_MM}_5m_enriched.feather`
    already exists. **If that feather is missing for the current month
    (e.g., May 2026), this tool reports the gap but cannot rebuild it
    on its own — the upstream monthly-feather build is a separate
    pipeline.** Without a fresh monthly feather for the current month,
    cross_day_rvol cannot extend past last month's end, and
    delivery_pct silently no-fires every signal in live.

Exit codes:
  0 — both freshness goals reached (or skipped per flags)
  1 — at least one parquet is still stale after the run; safe to ignore
      for backtest, but DO NOT start live/paper with this state.

This script writes a freshness report to stdout you can grep before
launching main.py:
    DATA_REFRESH | delivery_pct max_date=2026-05-13 OK
    DATA_REFRESH | cross_day_rvol max_date=2026-04-30 STALE_14d
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd


_REPO = Path(__file__).resolve().parents[2]
_DELIVERY_PARQUET = _REPO / "data" / "delivery_pct" / "delivery_history.parquet"
_RVOL_PARQUET = _REPO / "data" / "cross_day_rvol" / "rvol_baseline.parquet"
_MONTHLY_DIR = _REPO / "backtest-cache-download" / "monthly"

log = logging.getLogger("refresh_local_data")


# ---------------------------------------------------------------------------
# Freshness probes
# ---------------------------------------------------------------------------


def _parquet_max_date(parquet_path: Path) -> "date | None":
    """Return the maximum `date` column in the parquet, or None if missing."""
    if not parquet_path.exists():
        return None
    try:
        df = pd.read_parquet(parquet_path, columns=["date"])
        if df.empty:
            return None
        max_dt = pd.to_datetime(df["date"]).max()
        return max_dt.date() if hasattr(max_dt, "date") else max_dt
    except Exception as e:
        log.warning("could not read %s: %s", parquet_path, e)
        return None


def _last_market_day(today: date) -> date:
    """Return the most recent weekday strictly before `today`. Holiday-naive."""
    d = today - timedelta(days=1)
    while d.weekday() >= 5:  # 5=Sat, 6=Sun
        d -= timedelta(days=1)
    return d


def _staleness_days(parquet_max: "date | None", target: date) -> int:
    """How many days behind target is the parquet."""
    if parquet_max is None:
        return 9999
    return (target - parquet_max).days


# ---------------------------------------------------------------------------
# delivery_pct refresh
# ---------------------------------------------------------------------------


def _fetch_delivery_to(out_path: Path, start: str, end: str) -> int:
    """Subprocess wrapper: fetch delivery_pct for [start, end] to `out_path`.

    The underlying tool rewrites the file at out_path (not incremental).
    """
    cmd = [
        sys.executable,
        "-m", "tools.delivery_pct.fetch_delivery",
        "--start", start,
        "--end", end,
        "--out", str(out_path),
        "--workers", "6",
        "--log-level", "INFO",
    ]
    log.info("delivery_pct: invoking %s", " ".join(cmd))
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=str(_REPO)).returncode
    log.info("delivery_pct: done in %.1fs (rc=%d)", time.time() - t0, rc)
    return rc


def refresh_delivery(start: str, end: str, *, rebuild: bool = False) -> int:
    """Refresh delivery_pct, incremental by default.

    Strategy:
      1. If existing parquet missing or `--rebuild` flag → full fetch [start, end].
      2. If existing parquet already covers `end` → no-op.
      3. Else → fetch only (existing_max + 1)..end into a temp parquet,
         merge with existing, dedupe on (symbol, date, series), write back.

    A 1-day top-up takes ~3-10 seconds. A full cold rebuild takes 3-5 minutes.
    """
    end_d = datetime.strptime(end, "%Y-%m-%d").date()
    existing_max = _parquet_max_date(_DELIVERY_PARQUET)

    # Cold start or forced rebuild
    if rebuild or existing_max is None:
        log.info("delivery_pct: %s — full rebuild for [%s, %s]",
                 "rebuild forced" if rebuild else "no existing parquet",
                 start, end)
        return _fetch_delivery_to(_DELIVERY_PARQUET, start, end)

    # Already current
    if existing_max >= end_d:
        log.info("delivery_pct: already current (max_date=%s ≥ target=%s) — skip fetch",
                 existing_max, end_d)
        return 0

    # Incremental top-up
    new_start = (existing_max + timedelta(days=1)).isoformat()
    log.info("delivery_pct: incremental top-up [%s..%s] (existing max=%s)",
             new_start, end, existing_max)

    temp_path = _DELIVERY_PARQUET.parent / f"_temp_delivery_{int(time.time())}.parquet"
    try:
        rc = _fetch_delivery_to(temp_path, new_start, end)
        if rc != 0:
            log.error("delivery_pct: fetch_delivery returned rc=%d — not merging", rc)
            return rc
        if not temp_path.exists():
            log.error("delivery_pct: temp parquet missing after fetch — fetch returned 0 rows?")
            return 1

        # Merge: existing + new, dedupe on natural key
        existing = pd.read_parquet(_DELIVERY_PARQUET)
        new = pd.read_parquet(temp_path)
        before_rows = len(existing)
        merged = pd.concat([existing, new], ignore_index=True)
        merged = merged.drop_duplicates(
            subset=["symbol", "date", "series"], keep="last",
        )
        merged = merged.sort_values(["date", "symbol", "series"]).reset_index(drop=True)
        merged.to_parquet(_DELIVERY_PARQUET, index=False)
        log.info(
            "delivery_pct: merged +%d new rows (%d → %d total). max_date=%s",
            len(new), before_rows, len(merged),
            pd.to_datetime(merged["date"]).max().date(),
        )
        return 0
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# cross_day_rvol refresh
# ---------------------------------------------------------------------------


def _months_in_range(start: date, end: date) -> "list[date]":
    """First-of-month dates from start's month to end's month, inclusive."""
    cur = date(start.year, start.month, 1)
    final = date(end.year, end.month, 1)
    out = []
    while cur <= final:
        out.append(cur)
        cur = (cur + timedelta(days=32)).replace(day=1)
    return out


def refresh_rvol(start: str, end: str) -> int:
    """Re-build cross_day_rvol baseline from monthly 5m feathers.

    Requires `backtest-cache-download/monthly/{YYYY_MM}_5m_enriched.feather`
    to exist for every month from (start - 40 days) through end.
    Returns 0 on success, 1 if monthly-feather gaps prevent rebuild.
    """
    start_d = datetime.strptime(start, "%Y-%m-%d").date()
    end_d = datetime.strptime(end, "%Y-%m-%d").date()
    # Tool needs ROLLING_DAYS=20 days of prior history → pad start by 40 days
    needed_months = _months_in_range(start_d - timedelta(days=40), end_d)
    missing = []
    for m in needed_months:
        fp = _MONTHLY_DIR / f"{m.year}_{m.month:02d}_5m_enriched.feather"
        if not fp.exists():
            missing.append(fp.name)
    if missing:
        log.error(
            "cross_day_rvol: cannot rebuild — missing monthly feathers: %s\n"
            "  Build these via tools/precompute_5m_cache.py (or pull from OCI cache) before rerunning.\n"
            "  Without fresh feathers for the current month, the baseline cannot extend past last month-end\n"
            "  and delivery_pct_anomaly_short will silently no-fire every signal in live.",
            ", ".join(missing),
        )
        return 1

    cmd = [sys.executable, str(_REPO / "tools" / "cross_day_rvol" / "build_baseline.py"),
           start, end]
    log.info("cross_day_rvol: invoking %s", " ".join(cmd))
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=str(_REPO)).returncode
    log.info("cross_day_rvol: done in %.1fs (rc=%d)", time.time() - t0, rc)
    return rc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--start",
        default="2023-01-01",
        help="Start date YYYY-MM-DD (default: 2023-01-01 — full history rebuild)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date YYYY-MM-DD inclusive (default: last weekday before today)",
    )
    parser.add_argument(
        "--skip-delivery", action="store_true",
        help="Skip the delivery_pct refresh",
    )
    parser.add_argument(
        "--skip-rvol", action="store_true",
        help="Skip the cross_day_rvol rebuild",
    )
    parser.add_argument(
        "--rebuild-delivery", action="store_true",
        help="Force full delivery_pct rebuild from --start (default: incremental top-up)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    today = date.today()
    target = _last_market_day(today)
    end_str = args.end or target.isoformat()

    log.info("REFRESH | today=%s target=%s start=%s end=%s",
             today, target, args.start, end_str)

    delivery_rc = 0
    if not args.skip_delivery:
        log.info("=== delivery_pct refresh ===")
        delivery_rc = refresh_delivery(args.start, end_str, rebuild=args.rebuild_delivery)
    else:
        log.info("delivery_pct refresh SKIPPED per flag")

    rvol_rc = 0
    if not args.skip_rvol:
        log.info("=== cross_day_rvol refresh ===")
        rvol_rc = refresh_rvol(args.start, end_str)
    else:
        log.info("cross_day_rvol refresh SKIPPED per flag")

    # Final freshness report
    log.info("=== freshness report ===")
    final_rc = 0
    for label, fp in (("delivery_pct", _DELIVERY_PARQUET),
                      ("cross_day_rvol", _RVOL_PARQUET)):
        max_d = _parquet_max_date(fp)
        if max_d is None:
            print(f"DATA_REFRESH | {label}: parquet MISSING — setup will silently no-fire")
            final_rc = 1
            continue
        stale = _staleness_days(max_d, target)
        if stale <= 1:
            print(f"DATA_REFRESH | {label} max_date={max_d} OK")
        else:
            print(f"DATA_REFRESH | {label} max_date={max_d} STALE_{stale}d")
            final_rc = 1

    if final_rc == 0:
        log.info("All parquets fresh — safe to start paper/live")
    else:
        log.error("Some parquets still stale — fix before starting live")
    return final_rc


if __name__ == "__main__":
    sys.exit(main())
