"""Phase 2 empirical signature for mpc_day_intraday_reversal.

Per docs/setup_lifecycle.md Stage 2: quantify the mechanism's footprint in raw
data BEFORE writing the sanity script. Kill criterion: net drift < 0.1% means
no signal -> abandon (no methodology rescues a non-existent edge).

# What this measures

For each MPC announcement date in 2023-01 to 2024-12 (Discovery window), and
for each symbol in the rate-sensitive cohort (~37 stocks per brief §3):

  - move_10_30_pct = (close(10:30) - pre_anchor) / pre_anchor * 100
    where pre_anchor = close(09:55) [last bar before announcement]
  - move_30_120_pct = (close(12:00) - close(10:30)) / close(10:30) * 100

Baseline: same metrics on N random non-MPC trading days from same window
(matched cohort, matched days-of-week, ~50 random days).

# Verdict criteria

The brief proposes a 10:00-10:30 overshoot + 10:30-12:00 partial revert.
Two signatures must show on MPC days:

  1. OVERSHOOT: mean |move_10_30_pct| on MPC days > mean |move_10_30_pct|
     on random days by a meaningful margin (e.g., 50% larger).

  2. REVERSION: correlation(move_10_30, move_30_120) on MPC days is
     materially MORE NEGATIVE than on random days. Pure mean-reversion
     would give corr = -1.0; pure trend-continuation gives +1.0; random
     gives ~0.

If neither signature shows AND the effect-size threshold is breached
(< 0.1% drift, no negative correlation), KILL.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MONTHLY_DIR = _REPO_ROOT / "backtest-cache-download" / "monthly"


# ---------------------------------------------------------------------------
# Inputs (locked from brief §3)
# ---------------------------------------------------------------------------

# RBI MPC announcement dates 2023-2024. 10:00 IST press conference convention.
# Sourced from RBI press release archive (rbi.org.in).
MPC_DATES = [
    date(2023, 2, 8),
    date(2023, 4, 6),
    date(2023, 6, 8),
    date(2023, 8, 10),
    date(2023, 10, 6),
    date(2023, 12, 8),
    date(2024, 2, 8),
    date(2024, 4, 5),
    date(2024, 6, 7),
    date(2024, 8, 8),
    date(2024, 10, 9),
    date(2024, 12, 6),
]

# Rate-sensitive cohort (~37 names, locked from brief §3).
COHORT = [
    # Banks (13)
    "HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK", "INDUSINDBK",
    "BANDHANBNK", "IDFCFIRSTB", "FEDERALBNK", "BANKBARODA", "PNB",
    "CANBK", "BANKINDIA",
    # NBFCs (8)
    "BAJFINANCE", "BAJAJFINSV", "CHOLAFIN", "MUTHOOTFIN", "MANAPPURAM",
    "LICHSGFIN", "SHRIRAMFIN", "M&MFIN",
    # Auto (8)
    "MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "EICHERMOT",
    "HEROMOTOCO", "ASHOKLEY", "TVSMOTOR",
    # Real estate (8)
    "DLF", "GODREJPROP", "OBEROIRLTY", "PRESTIGE", "BRIGADE",
    "PHOENIXLTD", "LODHA", "MACROTECH",
]

# Time-of-day markers
PRE_ANCHOR_HHMM = "09:55"     # last 5m bar before 10:00 announcement
T_OVERSHOOT_END = "10:30"      # end of 10:00-10:30 overshoot window
T_REVERT_END = "12:00"         # end of 10:30-12:00 reversion window

# Baseline sampling
N_BASELINE_DAYS = 50           # random non-MPC trading days from same window
BASELINE_SEED = 42

# Verdict thresholds (per lesson #3 Phase 2)
MIN_DRIFT_PCT = 0.1            # net drift below this = no signal


# ---------------------------------------------------------------------------
# Bar loading
# ---------------------------------------------------------------------------

def _load_5m_for_month_indexed(yyyy: int, mm: int) -> pd.DataFrame:
    """Load month's 5m feather and pre-index by (symbol, date_str, hhmm_str).

    Returns a DataFrame with a single computed column close_at_hhmm that the
    caller can pivot or lookup-by-key. The full month is filtered to the
    cohort symbols only (massive prune from 1.8M rows -> ~50k rows).
    """
    p = _MONTHLY_DIR / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_feather(p, columns=["date", "symbol", "close"])
    df = df[df["symbol"].isin(COHORT)]
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df["d"] = df["date"].dt.date
    df["hhmm"] = df["date"].dt.strftime("%H:%M")
    return df


def _close_lookup_table(df_month_filtered: pd.DataFrame) -> dict:
    """Build dict[(symbol, d, hhmm)] -> close for fast O(1) lookups."""
    if df_month_filtered.empty:
        return {}
    return dict(zip(
        zip(df_month_filtered["symbol"], df_month_filtered["d"],
            df_month_filtered["hhmm"]),
        df_month_filtered["close"].astype(float),
    ))


def measure_session_from_lookup(
    lookup: dict, symbol: str, session: date,
) -> Tuple[float, float, float]:
    """Return (pre_anchor, move_10_30_pct, move_30_120_pct) using dict lookup.

    NaN tuple if any price missing.
    """
    pre = lookup.get((symbol, session, PRE_ANCHOR_HHMM))
    t30 = lookup.get((symbol, session, T_OVERSHOOT_END))
    t120 = lookup.get((symbol, session, T_REVERT_END))
    if pre is None or t30 is None or t120 is None or pre <= 0:
        return float("nan"), float("nan"), float("nan")
    move_10_30 = (t30 - pre) / pre * 100.0
    move_30_120 = (t120 - t30) / t30 * 100.0
    return pre, move_10_30, move_30_120


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def measure_event_set(
    dates: List[date], label: str,
) -> pd.DataFrame:
    """Measure (cohort × dates) into one long DataFrame.

    Returns rows: signal_date, symbol, pre_anchor, move_10_30_pct,
    move_30_120_pct, label.
    """
    # Group dates by (yyyy, mm) to load each month once
    by_month: dict = {}
    for d in dates:
        by_month.setdefault((d.year, d.month), []).append(d)

    rows: List[dict] = []
    for (yyyy, mm), dates_in_month in by_month.items():
        df_month = _load_5m_for_month_indexed(yyyy, mm)
        if df_month.empty:
            print(f"  [warn] no 5m feather for {yyyy}-{mm:02d}; skipping {len(dates_in_month)} dates", flush=True)
            continue
        lookup = _close_lookup_table(df_month)
        print(f"  [{label}] loaded {yyyy}-{mm:02d}: {len(df_month):,} cohort bars, {len(dates_in_month)} dates", flush=True)
        for session in dates_in_month:
            for symbol in COHORT:
                pre, m10_30, m30_120 = measure_session_from_lookup(lookup, symbol, session)
                if pd.isna(pre):
                    continue
                rows.append({
                    "signal_date": session,
                    "symbol": symbol,
                    "pre_anchor": pre,
                    "move_10_30_pct": m10_30,
                    "move_30_120_pct": m30_120,
                    "label": label,
                })
    return pd.DataFrame(rows)


def sample_baseline_dates(rng: np.random.Generator, n: int) -> List[date]:
    """Sample N random non-MPC trading days from 2023-2024.

    Filters: weekday, not within ±2 days of an MPC date (to avoid bleed),
    skip Jan 1-2 / Dec 25 (holiday risk).
    """
    pool: List[date] = []
    for yyyy in (2023, 2024):
        for mm in range(1, 13):
            for d_int in range(1, 32):
                try:
                    d = date(yyyy, mm, d_int)
                except ValueError:
                    continue
                if d.weekday() >= 5:
                    continue   # weekend
                # Exclude ±2 days of any MPC date
                if any(abs((d - md).days) <= 2 for md in MPC_DATES):
                    continue
                if (mm == 1 and d_int <= 2) or (mm == 12 and d_int >= 24):
                    continue
                pool.append(d)
    chosen_idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return sorted(pool[i] for i in chosen_idx)


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def print_signature(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        print(f"  {label}: NO DATA")
        return {}
    m10_30 = df["move_10_30_pct"].dropna()
    m30_120 = df["move_30_120_pct"].dropna()
    abs_overshoot = m10_30.abs()
    corr = float(m10_30.corr(m30_120)) if len(m10_30) > 1 else float("nan")
    stats = {
        "n_observations": int(len(df)),
        "n_unique_dates": int(df["signal_date"].nunique()),
        "n_unique_symbols": int(df["symbol"].nunique()),
        "mean_move_10_30_pct": float(m10_30.mean()),
        "mean_abs_move_10_30_pct": float(abs_overshoot.mean()),
        "median_abs_move_10_30_pct": float(abs_overshoot.median()),
        "p75_abs_move_10_30_pct": float(abs_overshoot.quantile(0.75)),
        "p95_abs_move_10_30_pct": float(abs_overshoot.quantile(0.95)),
        "mean_move_30_120_pct": float(m30_120.mean()),
        "corr_overshoot_vs_revert": corr,
    }
    print(f"\n  --- {label} (n={stats['n_observations']}, "
          f"dates={stats['n_unique_dates']}, syms={stats['n_unique_symbols']}) ---")
    print(f"    mean move_10_30_pct        : {stats['mean_move_10_30_pct']:+.3f}%")
    print(f"    mean |move_10_30_pct|       : {stats['mean_abs_move_10_30_pct']:.3f}%")
    print(f"    median |move_10_30_pct|     : {stats['median_abs_move_10_30_pct']:.3f}%")
    print(f"    p75 |move_10_30_pct|        : {stats['p75_abs_move_10_30_pct']:.3f}%")
    print(f"    p95 |move_10_30_pct|        : {stats['p95_abs_move_10_30_pct']:.3f}%")
    print(f"    mean move_30_120_pct       : {stats['mean_move_30_120_pct']:+.3f}%")
    print(f"    corr(10_30 vs 30_120)      : {stats['corr_overshoot_vs_revert']:+.3f}")
    return stats


def main() -> int:
    print("=" * 75)
    print("Phase 2 signature: mpc_day_intraday_reversal")
    print("=" * 75)
    print(f"  MPC dates: {len(MPC_DATES)} ({MPC_DATES[0]} .. {MPC_DATES[-1]})")
    print(f"  Cohort: {len(COHORT)} symbols")

    print("\n[1/2] Measuring MPC days...")
    mpc_df = measure_event_set(MPC_DATES, label="mpc")

    print("\n[2/2] Measuring baseline (random non-MPC days)...")
    rng = np.random.default_rng(BASELINE_SEED)
    baseline_dates = sample_baseline_dates(rng, N_BASELINE_DAYS)
    baseline_df = measure_event_set(baseline_dates, label="baseline")

    print("\n" + "=" * 75)
    print("SIGNATURE COMPARISON")
    print("=" * 75)
    mpc_stats = print_signature(mpc_df, "MPC days")
    base_stats = print_signature(baseline_df, "Random non-MPC days")

    print("\n" + "=" * 75)
    print("EFFECT SIZES")
    print("=" * 75)
    if mpc_stats and base_stats:
        overshoot_ratio = (mpc_stats["mean_abs_move_10_30_pct"]
                          / max(base_stats["mean_abs_move_10_30_pct"], 1e-9))
        delta_corr = (mpc_stats["corr_overshoot_vs_revert"]
                     - base_stats["corr_overshoot_vs_revert"])
        print(f"  overshoot ratio (MPC / baseline mean |move_10_30|): {overshoot_ratio:.2f}x")
        print(f"  delta correlation (MPC - baseline)                : {delta_corr:+.3f}")
        print(f"    (negative delta = MPC days more mean-reverting)")

        # Verdict per Lesson #3 Phase 2
        signal_present = (
            mpc_stats["mean_abs_move_10_30_pct"] >= MIN_DRIFT_PCT
            and (overshoot_ratio >= 1.3 or delta_corr <= -0.10)
        )
        verdict = "PROCEED TO STAGE 3" if signal_present else "KILL — no measurable signal"
        print(f"\n  VERDICT: {verdict}")

    out_path = _REPO_ROOT / "reports" / "sub9_sanity" / "_phase2_mpc_day_signature.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full = pd.concat([mpc_df, baseline_df], ignore_index=True)
    full.to_csv(out_path, index=False)
    print(f"\n  Raw measurements written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
