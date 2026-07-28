"""Repair the post-2025-03 earnings announcement-timestamp hole.

Background (2026-07-28 scoping): NSE retired the "Financial Result Updates"
announcements subject ~2025-03; the Reg-30 earnings stream migrated to the
"Outcome of Board Meeting" subject on the same endpoint. The legacy
``_ann_is_earnings`` text filter drops most OBM earnings rows, so events in
the 2025-03 -> present window degraded to synthetic ``scheduled`` (09:00)
board-meetings rows, or were missed entirely.

This driver:
  1. Backs up ``earnings_events.parquet`` to
     ``earnings_events_pre_repair_2026-07-28.parquet`` (permanent artifact —
     the PEAD falsifier-3 cohort comparison needs both versions).
  2. Re-scrapes the window from three sources: ``board_meetings`` (roster +
     synthetic fallback), ``announcements_obm`` (NSE OBM subject,
     roster-matched) and ``announcements_bse`` (BSE Financial Results,
     scrip->ISIN->symbol mapped, roster-matched).
  3. Merges via the existing incremental-merge machinery (dedupe priority
     keeps every pre-existing row; repair sources only upgrade synthetic
     rows or add new events).
  4. Drops stale ``scheduled`` rows superseded by a real-timestamped row on
     bm_date+1 (post-midnight filings land on the next calendar day so the
     (symbol, announce_date) dedupe key does not collide).
  5. Prints a before/after report: per-quarter announce_time_class mix,
     class changes, new events, trade_date shifts.

Usage:
    # additivity assertion (small 2024 window, operates on a scratch COPY):
    python tools/earnings_calendar/repair_post_fr_window.py --verify-additive

    # full repair:
    python tools/earnings_calendar/repair_post_fr_window.py \
        --start 2025-03-01 --end 2026-07-28 --sleep-secs 4
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_earnings as fe  # noqa: E402

_BACKUP_NAME = "earnings_events_pre_repair_2026-07-28.parquet"
_REPAIR_SOURCES = ("board_meetings", "announcements_obm", "announcements_bse")


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def _norm_date(v):
    """Normalise parquet date-ish cells (date / Timestamp / NaT) to date|None."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    try:
        ts = pd.Timestamp(v)
    except (ValueError, TypeError):
        return None
    return None if pd.isna(ts) else ts.date()


def _snapshot(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.copy()
    df["_ad"] = df["announce_date"].map(_norm_date)
    df["_td"] = df["trade_date"].map(_norm_date)
    df["_key"] = list(zip(df["symbol"], df["_ad"]))
    return df


def _quarter_class_mix(df: pd.DataFrame) -> pd.DataFrame:
    q = pd.PeriodIndex(pd.to_datetime(df["_ad"]), freq="Q")
    return (
        df.assign(_q=q)
        .pivot_table(
            index="_q", columns="announce_time_class", values="symbol",
            aggfunc="count", fill_value=0,
        )
    )


def drop_superseded_scheduled(
    df: pd.DataFrame,
    min_announce_date: Optional[date] = None,
) -> tuple[pd.DataFrame, int]:
    """Drop ``scheduled`` rows whose meeting produced a real-timestamped row
    on the NEXT calendar day (post-midnight filing => different dedupe key,
    so the stale synthetic row survives the merge).

    Same-day collisions are already resolved by dedupe priority.

    ``min_announce_date``: only drop scheduled rows on/after this date. The
    repair must leave the pre-death window untouched (frozen control region
    for the PEAD falsifier-3 cohort comparison), so the caller passes the
    repair-window start.
    """
    ads = df["announce_date"].map(_norm_date)
    real_keys = set(
        zip(
            df.loc[df["announce_time_class"] != "scheduled", "symbol"],
            ads[df["announce_time_class"] != "scheduled"],
        )
    )
    is_sched = df["announce_time_class"] == "scheduled"
    if min_announce_date is not None:
        in_window = pd.Series(
            [ad is not None and ad >= min_announce_date for ad in ads],
            index=df.index,
        )
        is_sched = is_sched & in_window
    next_day_real = pd.Series(
        [
            (sym, ad + timedelta(days=1)) in real_keys if ad is not None else False
            for sym, ad in zip(df["symbol"], ads)
        ],
        index=df.index,
    )
    drop_mask = is_sched & next_day_real
    n = int(drop_mask.sum())
    return df.loc[~drop_mask].reset_index(drop=True), n


# ---------------------------------------------------------------------------
# Additivity assertion (Lesson: repair must not change pre-death rows).
# ---------------------------------------------------------------------------

def verify_additive(sleep_secs: float) -> int:
    """Scrape a small healthy-era window with the NEW sources into a COPY of
    the parquet; assert every pre-existing row is byte-identical."""
    start, end = date(2024, 7, 22), date(2024, 7, 26)
    scratch = Path(tempfile.mkdtemp(prefix="earnings_additive_check_"))
    copy_path = scratch / "earnings_events.parquet"
    shutil.copy2(fe._OUT_PATH, copy_path)
    print(f"[additive-check] window {start} -> {end}; scratch copy {copy_path}")

    before = _snapshot(copy_path)
    holidays = fe.load_nse_holidays()
    rows, stats = fe.fetch_all_sources(
        start, end,
        fno_universe=None, holidays=holidays,
        sleep_secs=sleep_secs, sources=_REPAIR_SOURCES,
        roster_parquet_path=copy_path,
    )
    fe.write_events(rows, copy_path, merge_existing=True)
    after = _snapshot(copy_path)

    cmp_cols = [
        "announce_time", "announce_time_class", "trade_date",
        "result_type", "source",
    ]
    b = before.set_index("_key")
    a = after.set_index("_key")
    missing = b.index.difference(a.index)
    common = b.index.intersection(a.index)
    changed = set()
    detail = []
    for c in cmp_cols:
        bc = b.loc[common, c]
        ac = a.loc[common, c]
        neq = ~((bc == ac) | (bc.isna() & ac.isna()))
        if neq.any():
            changed.update(common[neq])
            detail.extend((c, k, bc[k], ac[k]) for k in common[neq][:10])
    # Invariant: repair sources may ONLY upgrade synthetic "scheduled"
    # (board_meetings) rows. Any change to a real-timestamped row is a
    # violation of the dedupe-priority additivity guarantee.
    upgrades = {
        k for k in changed
        if b.loc[[k], "announce_time_class"].iloc[0] == "scheduled"
    }
    violations = changed - upgrades
    n_new = len(a.index.difference(b.index))
    print(f"[additive-check] scrape stats: {stats}")
    print(
        f"[additive-check] pre-existing rows: {len(b)}; disappeared: "
        f"{len(missing)}; scheduled->real upgrades (allowed): {len(upgrades)}; "
        f"real-row changes (violations): {len(violations)}; "
        f"new rows added: {n_new}"
    )
    if len(missing) or violations:
        for item in detail[:20]:
            if item[1] in violations:
                print(f"[additive-check]   VIOLATION {item}")
        for k in list(missing)[:20]:
            print(f"[additive-check]   MISSING {k}")
        print("[additive-check] FAIL — repair is NOT additive")
        return 1
    print(
        "[additive-check] PASS — no real-timestamped row changed "
        "(synthetic scheduled upgrades are the repair working as designed)"
    )
    return 0


# ---------------------------------------------------------------------------
# Full repair.
# ---------------------------------------------------------------------------

def run_repair(start: date, end: date, sleep_secs: float) -> int:
    parquet = fe._OUT_PATH
    backup = parquet.with_name(_BACKUP_NAME)
    if not backup.exists():
        shutil.copy2(parquet, backup)
        print(f"[repair] backup written: {backup}")
    else:
        print(f"[repair] backup already exists (kept): {backup}")

    pre = _snapshot(parquet)
    holidays = fe.load_nse_holidays()
    print(
        f"[repair] window {start} -> {end}; sleep {sleep_secs}s; "
        f"sources {_REPAIR_SOURCES}; pre-repair rows={len(pre)}"
    )

    rows, stats = fe.fetch_all_sources(
        start, end,
        fno_universe=None, holidays=holidays,
        sleep_secs=sleep_secs, sources=_REPAIR_SOURCES,
        roster_parquet_path=parquet,
    )
    print(f"[repair] scraped {len(rows)} candidate rows; merging ...")
    df_after = fe.write_events(rows, parquet, merge_existing=True)

    df_after, n_superseded = drop_superseded_scheduled(
        df_after, min_announce_date=start,
    )
    if n_superseded:
        df_after = fe._ensure_columns(df_after)
        df_after = df_after.sort_values(
            ["announce_date", "symbol"]
        ).reset_index(drop=True)
        df_after.to_parquet(parquet, index=False)
        print(
            f"[repair] dropped {n_superseded} stale scheduled rows "
            f"(superseded by a real row on bm_date+1); parquet rewritten"
        )

    post = _snapshot(parquet)
    _report(pre, post, stats, n_superseded, start, end)
    n_failed = sum(
        s.get("chunks_failed", 0) for s in stats if isinstance(s, dict)
    )
    return 0 if n_failed == 0 else 4


def _report(
    pre: pd.DataFrame,
    post: pd.DataFrame,
    stats: list,
    n_superseded: int,
    start: date,
    end: date,
) -> None:
    pd.set_option("display.width", 200)
    print("\n" + "=" * 72)
    print("REPAIR REPORT — earnings announce-timestamp repair "
          f"({start} -> {end})")
    print("=" * 72)

    print("\n--- per-quarter announce_time_class mix: BEFORE ---")
    print(_quarter_class_mix(pre))
    print("\n--- per-quarter announce_time_class mix: AFTER ---")
    print(_quarter_class_mix(post))

    b = pre.set_index("_key")
    a = post.set_index("_key")
    common = b.index.intersection(a.index)
    new_keys = a.index.difference(b.index)
    gone_keys = b.index.difference(a.index)

    cls_changed = common[
        (b.loc[common, "announce_time_class"] != a.loc[common, "announce_time_class"])
    ]
    td_b = b.loc[common, "_td"]
    td_a = a.loc[common, "_td"]
    td_changed = common[~((td_b == td_a) | (td_b.isna() & td_a.isna()))]

    def _by_quarter(keys) -> pd.Series:
        if len(keys) == 0:
            return pd.Series(dtype=int)
        qs = pd.PeriodIndex(
            pd.to_datetime([k[1] for k in keys]), freq="Q"
        )
        return pd.Series(1, index=qs).groupby(level=0).sum()

    print(f"\nevents BEFORE: {len(b)}   AFTER: {len(a)}")
    print(f"NEW events recovered:        {len(new_keys)}")
    print(_by_quarter(new_keys).to_string())
    print(f"\nclass corrected (same key):  {len(cls_changed)}")
    print(_by_quarter(cls_changed).to_string())
    if len(cls_changed):
        trans = (
            b.loc[cls_changed, "announce_time_class"]
            + " -> " + a.loc[cls_changed, "announce_time_class"]
        ).value_counts()
        print("class transitions:")
        print(trans.to_string())
    print(f"\ntrade_date shifted (same key): {len(td_changed)}")
    print(_by_quarter(td_changed).to_string())
    print(f"\nstale scheduled rows dropped (bm_date+1 supersede): {n_superseded}")
    print(f"rows removed total (should equal supersede count): {len(gone_keys)}")

    still = post[post["announce_time_class"] == "scheduled"]
    print(f"\nevents still 'scheduled' AFTER repair: {len(still)}")
    if len(still):
        qs = pd.PeriodIndex(pd.to_datetime(still["_ad"]), freq="Q")
        print(still.groupby(qs)["symbol"].count().to_string())

    print("\n--- per-source scrape stats ---")
    for s in stats:
        print(f"  {s}")
    print("\n--- source mix AFTER ---")
    print(post["source"].value_counts().to_string())
    print("=" * 72)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--start", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
                   default=date(2025, 3, 1))
    p.add_argument("--end", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
                   default=date(2026, 7, 28))
    p.add_argument("--sleep-secs", type=float, default=4.0,
                   help="politeness sleep between requests (>=4)")
    p.add_argument("--verify-additive", action="store_true",
                   help="run the 2024 additivity assertion on a scratch copy "
                        "and exit (no write to the real parquet)")
    args = p.parse_args(argv)

    if args.sleep_secs < 4.0:
        p.error("--sleep-secs must be >= 4 (politeness)")

    # Backup FIRST — even the verify mode must not run before the permanent
    # pre-repair artifact exists.
    parquet = fe._OUT_PATH
    backup = parquet.with_name(_BACKUP_NAME)
    if parquet.exists() and not backup.exists():
        shutil.copy2(parquet, backup)
        print(f"[repair] backup written: {backup}")

    if args.verify_additive:
        return verify_additive(args.sleep_secs)
    return run_repair(args.start, args.end, args.sleep_secs)


if __name__ == "__main__":
    sys.exit(main())
