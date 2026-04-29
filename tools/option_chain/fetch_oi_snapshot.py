"""Daily NSE F&O bhavcopy → option_chain parquet ingestion.

CLI:
    python tools/option_chain/fetch_oi_snapshot.py --session-date 2024-06-06
    python tools/option_chain/fetch_oi_snapshot.py --start 2023-01-02 --end 2026-04-29

Output tree:
    data/option_chain/<YYYY>/<MM>/<YYYY-MM-DD>.parquet

Per specs/2026-04-29-expiry_pin_strike_reversal-plan.md Phase A2 + A3.
The actual full-history backfill (~800 sessions) is the user's deferred
step — this module is the executable infrastructure to run it when ready.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from tools.option_chain._nse_bhavcopy_client import (
    BhavcopyNotFound,
    download_bhavcopy,
    parse_bhavcopy,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUT_ROOT = _REPO_ROOT / "data" / "option_chain"


def _parquet_path(out_root: Path, session_date: date) -> Path:
    """data/option_chain/<YYYY>/<MM>/<YYYY-MM-DD>.parquet"""
    return (
        out_root
        / f"{session_date.year:04d}"
        / f"{session_date.month:02d}"
        / f"{session_date.isoformat()}.parquet"
    )


def _validate(df: pd.DataFrame, session_date: date) -> None:
    """Raise ValueError on incomplete bhavcopy (per Task A2.3 spec).

    - At least 100 contracts (sanity floor)
    - No null OI, strike, option_type
    - All expiry_date >= session_date (settled-future rules)
    """
    if len(df) < 100:
        raise ValueError(
            f"too few contracts in bhavcopy for {session_date}: {len(df)} < 100"
        )
    if df["oi"].isna().any():
        raise ValueError(f"null OI present for {session_date}")
    if df["strike"].isna().any():
        raise ValueError(f"null strike present for {session_date}")
    if df["option_type"].isna().any():
        raise ValueError(f"null option_type present for {session_date}")
    bad_expiry = df[df["expiry_date"] < session_date]
    if len(bad_expiry) > 0:
        raise ValueError(
            f"{len(bad_expiry)} rows with expiry_date < session_date for {session_date}"
        )


def ingest_one_session(
    session_date: date,
    out_root: Path = _DEFAULT_OUT_ROOT,
    *,
    skip_existing: bool = True,
    download_fn=download_bhavcopy,
    parse_fn=parse_bhavcopy,
) -> Path:
    """Download + parse + validate + write parquet for one session.

    Returns the parquet path. Raises BhavcopyNotFound for non-trading days
    and ValueError for malformed/incomplete data.

    `download_fn` and `parse_fn` are dependency-injection points for tests
    (HTTP-mocked or fixture-based).
    """
    out_path = _parquet_path(out_root, session_date)
    if skip_existing and out_path.exists():
        return out_path
    raw = download_fn(session_date)
    result = parse_fn(raw, session_date)
    df = result.rows
    _validate(df, session_date)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Use pandas' default parquet engine (pyarrow if available, else fastparquet).
    df.to_parquet(out_path, index=False)
    return out_path


def ingest_range(
    start: date,
    end: date,
    out_root: Path = _DEFAULT_OUT_ROOT,
    *,
    skip_existing: bool = True,
    skip_weekends: bool = True,
    download_fn=download_bhavcopy,
    parse_fn=parse_bhavcopy,
) -> dict:
    """Iterate sessions [start, end] inclusive; ingest each.

    Returns a summary dict: {"sessions_attempted", "sessions_written",
    "sessions_skipped_existing", "sessions_holiday_or_weekend"}.

    Tolerates BhavcopyNotFound (counted as holiday/weekend; doesn't abort).
    Tolerates ValueError on parse/validation (counted as failed; logged to stderr).
    """
    summary = {
        "sessions_attempted": 0,
        "sessions_written": 0,
        "sessions_skipped_existing": 0,
        "sessions_holiday_or_weekend": 0,
        "sessions_failed": 0,
    }
    cur = start
    while cur <= end:
        if skip_weekends and cur.weekday() >= 5:   # Sat=5, Sun=6
            cur += timedelta(days=1)
            continue
        summary["sessions_attempted"] += 1
        out_path = _parquet_path(out_root, cur)
        if skip_existing and out_path.exists():
            summary["sessions_skipped_existing"] += 1
            cur += timedelta(days=1)
            continue
        try:
            ingest_one_session(
                cur, out_root,
                skip_existing=skip_existing,
                download_fn=download_fn, parse_fn=parse_fn,
            )
            summary["sessions_written"] += 1
        except BhavcopyNotFound:
            summary["sessions_holiday_or_weekend"] += 1
        except Exception as e:   # noqa: BLE001 — broad on purpose; backfill must continue
            summary["sessions_failed"] += 1
            print(
                f"[fetch_oi_snapshot] {cur}: failed — {type(e).__name__}: {e}",
                file=sys.stderr,
            )
        cur += timedelta(days=1)
    return summary


def _parse_date_arg(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest NSE F&O bhavcopy into option_chain parquet store."
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--session-date", type=_parse_date_arg,
                   help="single session to ingest (YYYY-MM-DD)")
    g.add_argument("--start", type=_parse_date_arg,
                   help="range start (YYYY-MM-DD); requires --end")
    parser.add_argument("--end", type=_parse_date_arg,
                        help="range end inclusive (YYYY-MM-DD)")
    parser.add_argument("--out-root", type=Path, default=_DEFAULT_OUT_ROOT,
                        help=f"parquet output root (default: {_DEFAULT_OUT_ROOT})")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="re-ingest sessions whose parquet already exists")
    args = parser.parse_args(argv)

    skip_existing = not args.no_skip_existing
    if args.session_date is not None:
        try:
            path = ingest_one_session(
                args.session_date, args.out_root, skip_existing=skip_existing,
            )
            print(f"OK: {path}")
            return 0
        except BhavcopyNotFound as e:
            print(f"NOT_FOUND: {e}", file=sys.stderr)
            return 2
        except ValueError as e:
            print(f"INVALID: {e}", file=sys.stderr)
            return 3
    else:
        if args.end is None:
            parser.error("--start requires --end")
        summary = ingest_range(
            args.start, args.end, args.out_root, skip_existing=skip_existing,
        )
        print(
            f"Range {args.start} -> {args.end}: "
            f"attempted={summary['sessions_attempted']} "
            f"written={summary['sessions_written']} "
            f"skipped_existing={summary['sessions_skipped_existing']} "
            f"holiday_or_404={summary['sessions_holiday_or_weekend']} "
            f"failed={summary['sessions_failed']}"
        )
        return 0 if summary["sessions_failed"] == 0 else 4


if __name__ == "__main__":
    sys.exit(main())
