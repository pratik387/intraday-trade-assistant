"""Load regenerated wide-open backtest output into a canonical trade table.

Critical correctness property: only is_final_exit=True rows from analytics.jsonl
are kept. T1 partial exits are dropped here so all downstream stages see the
true per-trade outcome.

Hour bucket mapping (from minute_of_day):
  555-600 (9:15-10:00) → opening
  600-720 (10:00-12:00) → morning
  720-780 (12:00-13:00) → lunch
  780-870 (13:00-14:30) → afternoon
  870+ (14:30+)         → late
"""
from dataclasses import dataclass
from datetime import date
import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

from tools.edge_discovery_legacy_gauntlet.periods import assign_fy


def _hour_bucket(minute_of_day):
    if minute_of_day is None or pd.isna(minute_of_day):
        return None
    m = int(minute_of_day)
    # NSE market opens at 9:15 IST (minute 555). Bars before this (pre-market,
    # or malformed data) return None — they're not a valid trading bucket.
    if m < 555:
        return None
    if m < 600:
        return "opening"
    if m < 720:
        return "morning"
    if m < 780:
        return "lunch"
    if m < 870:
        return "afternoon"
    return "late"


@dataclass
class GauntletData:
    """Canonical trade table + metadata."""
    trades: pd.DataFrame
    sessions_loaded: int
    sessions_skipped: int


def load_run(run_dir: Path) -> GauntletData:
    """Load every session directory under run_dir, filtering to final exits.

    Each session dir is expected to contain:
      - analytics.jsonl (required, has PnL + outcome)
      - trade_report.csv (optional, provides cap_segment + minute_of_day)

    Returns GauntletData with one row per completed trade.
    """
    run_dir = Path(run_dir)
    session_dirs = sorted(
        d for d in run_dir.iterdir()
        if d.is_dir() and (d / "analytics.jsonl").exists() and d.name[:4].isdigit()
    )
    if not session_dirs:
        raise ValueError(f"No session directories found in {run_dir}")

    all_trades: List[pd.DataFrame] = []
    skipped = 0

    for sdir in session_dirs:
        session = sdir.name
        jl = sdir / "analytics.jsonl"
        csv = sdir / "trade_report.csv"

        analytics_rows = []
        with open(jl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not rec.get("is_final_exit"):
                    continue
                analytics_rows.append(rec)

        if not analytics_rows:
            skipped += 1
            continue

        df = pd.DataFrame(analytics_rows)
        df["session_date"] = session

        # Prefer total_trade_pnl; fall back to pnl
        if "total_trade_pnl" not in df.columns:
            df["total_trade_pnl"] = df["pnl"]
        else:
            df["total_trade_pnl"] = df["total_trade_pnl"].fillna(df["pnl"])

        # Join cap_segment + minute_of_day from trade_report.csv
        if csv.exists():
            try:
                tr = pd.read_csv(
                    csv,
                    usecols=lambda c: c in {"trade_id", "cap_segment", "minute_of_day"},
                )
                df = df.merge(tr, on="trade_id", how="left", suffixes=("", "_tr"))
            except Exception as e:
                # Silent failure would mean the whole session loses cap_segment +
                # minute_of_day. Log so operators notice during a gauntlet run.
                logger.warning(
                    "Failed to join trade_report.csv for session %s: %s. "
                    "cap_segment / hour_bucket will be null for this session.",
                    session, e,
                )

        if "cap_segment" not in df.columns:
            df["cap_segment"] = None
        if "minute_of_day" not in df.columns:
            df["minute_of_day"] = None

        df["hour_bucket"] = df["minute_of_day"].apply(_hour_bucket)

        all_trades.append(df)

    if not all_trades:
        raise ValueError(f"No final-exit trades found in any session under {run_dir}")

    merged = pd.concat(all_trades, ignore_index=True)
    merged["session_date_dt"] = pd.to_datetime(merged["session_date"]).dt.date
    merged["fy"] = merged["session_date_dt"].apply(assign_fy)

    return GauntletData(
        trades=merged,
        sessions_loaded=len(session_dirs) - skipped,
        sessions_skipped=skipped,
    )
