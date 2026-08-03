"""Event-data feed refresh + staleness validation for event-driven setups.

Why this module exists
----------------------
On 2026-08-03 `earnings_downshock_continuation_short` ran its first live paper
session and logged ``DISPATCH_BUILD_UNIVERSE | ... | 0 symbols`` on every bar.
The setup was registered and dispatching correctly. The cause was that
``data/earnings_calendar/`` had never been deployed to the host at all — ``data/``
is gitignored, so the parquet was never shipped.

A missing event feed means no reaction day is ever flagged, which means an empty
universe forever, which means zero fires — and **nothing raised**. 150 qualifying
announcements existed that session. A second failure compounds it: the feed is a
static file, so even when present it silently goes stale and stops producing
signals.

Both failure modes are silent, which is what makes them dangerous. This module
makes them loud.

A setup opts in by declaring an ``event_feed`` block in its config; nothing here
is per-setup, so future event-driven setups inherit refresh + validation for free.
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class FeedStatus:
    """Outcome of validating one event feed."""

    ok: bool
    reason: str          # 'ok' | 'missing' | 'stale' | 'unreadable'
    message: str
    newest: Optional[pd.Timestamp] = None
    age_days: Optional[int] = None
    rows: Optional[int] = None


def validate_event_feed(
    path: Path,
    *,
    date_column: str,
    max_staleness_days: int,
    now: pd.Timestamp,
) -> FeedStatus:
    """Check that an event feed exists and is recent enough to be usable.

    `now` is injected rather than read from the clock so this is deterministic
    and testable (and so callers pass IST-naive time per CLAUDE.md rule 2).
    """
    if not path.exists():
        return FeedStatus(
            ok=False,
            reason="missing",
            message=(
                f"FEED MISSING at {path}. The setup is ENABLED but will build an "
                f"EMPTY universe every bar and can never fire. Note data/ is "
                f"gitignored, so this file must be copied to the host explicitly."
            ),
        )

    try:
        df = pd.read_parquet(path, columns=[date_column])
        newest = pd.to_datetime(df[date_column]).max()
    except Exception as e:  # unreadable/corrupt/missing column
        return FeedStatus(
            ok=False,
            reason="unreadable",
            message=f"could not read {path}: {type(e).__name__}: {e}",
        )

    if pd.isna(newest):
        return FeedStatus(
            ok=False,
            reason="unreadable",
            message=f"{path} has no parseable values in '{date_column}'",
            rows=len(df),
        )

    age_days = int((now.normalize() - newest.normalize()).days)
    base = f"newest={newest.date()} age={age_days}d rows={len(df)}"

    if age_days > max_staleness_days:
        return FeedStatus(
            ok=False,
            reason="stale",
            message=(
                f"{base} — STALE (limit {max_staleness_days}d). This setup needs "
                f"the PREVIOUS TRADING DAY's events; it will likely fire nothing."
            ),
            newest=newest,
            age_days=age_days,
            rows=len(df),
        )

    return FeedStatus(True, "ok", base, newest=newest, age_days=age_days, rows=len(df))


def refresh_event_feed(
    feed: dict,
    *,
    repo_root: Path,
    now: pd.Timestamp,
) -> tuple[bool, str]:
    """Run a feed's refresh command. Returns (ok, message).

    Never raises: a feed refresh must not be able to kill the trading daemon.
    The subsequent `validate_event_feed` call is what decides whether the feed is
    actually usable, so a failed refresh degrades to "possibly stale", not a crash.
    """
    lookback = int(feed["refresh_lookback_days"])
    start = (now - timedelta(days=lookback)).strftime("%Y-%m-%d")
    end = now.strftime("%Y-%m-%d")
    try:
        completed = subprocess.run(
            [
                sys.executable, "-m", feed["refresh_module"],
                "--start", start, "--end", end,
                "--sleep-secs", str(feed["refresh_sleep_secs"]),
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=int(feed["refresh_timeout_sec"]),
        )
    except subprocess.TimeoutExpired:
        return False, f"refresh TIMED OUT after {feed['refresh_timeout_sec']}s ({start}..{end})"
    except Exception as e:
        return False, f"refresh raised {type(e).__name__}: {e}"

    if completed.returncode != 0:
        return False, (
            f"refresh FAILED rc={completed.returncode} ({start}..{end}): "
            f"{(completed.stderr or '')[-400:]}"
        )
    return True, f"refresh ok ({start}..{end})"


def refresh_and_validate_all(cfg: dict, *, repo_root: Path, now: pd.Timestamp, logger) -> None:
    """Refresh + validate the feed of every ENABLED setup that declares one.

    Call at daemon start for paper/live ONLY. Never in backtest: the archive must
    stay frozen at whatever it contains, and a network scrape mid-replay would be
    both wrong and non-reproducible.
    """
    for name, raw in (cfg.get("setups") or {}).items():
        if not isinstance(raw, dict) or not raw.get("enabled", False):
            continue
        feed = raw.get("event_feed")
        if not feed:
            continue

        label = f"EVENT_FEED | {name}"

        if feed.get("refresh_on_start", False):
            logger.info(f"{label} | refreshing {feed['path']}")
            ok, msg = refresh_event_feed(feed, repo_root=repo_root, now=now)
            (logger.info if ok else logger.error)(f"{label} | {msg}")

        status = validate_event_feed(
            repo_root / feed["path"],
            date_column=feed["date_column"],
            max_staleness_days=int(feed["max_staleness_days"]),
            now=now,
        )
        if status.ok:
            logger.info(f"{label} | {status.message}")
        else:
            # CRITICAL: the setup is enabled and cannot produce signals. This is
            # exactly the condition that went unnoticed for a full session.
            logger.critical(f"{label} | {status.message}")
