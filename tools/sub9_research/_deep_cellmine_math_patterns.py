"""Deep cell mining for retired Tier-A math-pattern sanities.

Two patterns were retired by aggregate PF after a shallow 3-dim cell scan:
  - Inside Bar Breakout: 293,025 trades, PF=0.738 aggregate
  - RSI Extreme Reversal: 87,415 trades, PF=0.765 aggregate

This script extends the cell-mining surface to ~12 dimensions per pattern,
adds bucket dims derived from raw trade fields, and reports SURVIVORS
(n>=100, PF>=1.20, Sh>0) and SHIP-ELIGIBLE (gauntlet-v2 gates) cells.

Usage:
    python tools/sub9_research/_deep_cellmine_math_patterns.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from _cell_mine_tier_a import (  # type: ignore  # noqa: E402
    N_MIN_SHIP,
    N_MIN_SURVIVOR,
    PF_MIN_SHIP,
    PF_MIN_SURVIVOR,
    SHARPE_MIN_SHIP,
    SHARPE_MIN_SURVIVOR,
    LOSING_MONTHS_PCT_MAX_SHIP,
    TOP_MONTH_CONCENTRATION_MAX,
    pf_of,
    scan_cells,
    _sharpe_of,
    _monthly_stats,
)

_REPO = Path(__file__).resolve().parents[2]
_OUT_DIR = _REPO / "reports" / "sub9_sanity"


# ---------------------------------------------------------------------------
# Inside Bar dimension prep
# ---------------------------------------------------------------------------


def _bucket_pct(values: pd.Series, edges, labels) -> pd.Series:
    return pd.cut(values, bins=edges, labels=labels, right=False, include_lowest=True)


def prep_inside_bar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Dates / calendar
    d = pd.to_datetime(df["T0_signal_date"])
    df["_session_date"] = d.dt.date
    df["_month"] = d.dt.strftime("%Y-%m")
    df["dow"] = d.dt.day_name().str[:3]
    df["month"] = d.dt.strftime("%b")
    df["quarter"] = "Q" + d.dt.quarter.astype(str)

    # Mother size (% of mother_low)
    mother_size_pct = (df["mother_high"] - df["mother_low"]) / df["mother_low"] * 100.0
    df["mother_size_bucket"] = _bucket_pct(
        mother_size_pct,
        edges=[-np.inf, 0.5, 1.0, 2.0, 3.0, np.inf],
        labels=["<0.5%", "0.5-1%", "1-2%", "2-3%", "3%+"],
    )

    # Inside position within the mother bar
    rng = (df["mother_high"] - df["mother_low"]).replace(0, np.nan)
    inside_pos = (df["inside_low"] - df["mother_low"]) / rng
    df["inside_pos_bucket"] = _bucket_pct(
        inside_pos,
        edges=[-np.inf, 0.33, 0.67, np.inf],
        labels=["lower-third", "mid", "upper-third"],
    )

    # R size (% of entry price)
    r_pct = df["stop_distance"] / df["entry_price"] * 100.0
    df["R_size_bucket"] = _bucket_pct(
        r_pct,
        edges=[-np.inf, 0.5, 1.0, 2.0, 3.0, np.inf],
        labels=["<0.5%", "0.5-1%", "1-2%", "2-3%", "3%+"],
    )

    df["hit_t1_str"] = df["hit_t1"].astype(str)

    # Entry hour band — round entry_hhmm down to the hour
    hhmm = df["entry_hhmm"].astype(str)
    hour_str = hhmm.str.slice(0, 2)
    df["entry_hour_bucket"] = hour_str + ":00"

    return df


# ---------------------------------------------------------------------------
# RSI dimension prep
# ---------------------------------------------------------------------------


def prep_rsi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    d = pd.to_datetime(df["session_date"])
    df["_session_date"] = d.dt.date
    df["_month"] = d.dt.strftime("%Y-%m")
    df["dow"] = d.dt.day_name().str[:3]
    df["month"] = d.dt.strftime("%b")
    df["quarter"] = "Q" + d.dt.quarter.astype(str)

    # Fine RSI severity (direction-aware)
    rsi = df["rsi_trigger"]
    side = df["direction"].astype(str).str.lower()

    long_bins = [-np.inf, 15, 20, 25, 30, np.inf]
    long_lbl = ["L<15", "L15-20", "L20-25", "L25-30", "L30+"]
    short_bins = [-np.inf, 70, 75, 80, 85, np.inf]
    short_lbl = ["S<=70", "S70-75", "S75-80", "S80-85", "S85+"]

    long_bucket = pd.cut(rsi, bins=long_bins, labels=long_lbl, right=False)
    short_bucket = pd.cut(rsi, bins=short_bins, labels=short_lbl, right=False)

    df["rsi_severity_fine"] = np.where(side == "long", long_bucket.astype(str),
                                       short_bucket.astype(str))
    df.loc[df["rsi_severity_fine"] == "nan", "rsi_severity_fine"] = np.nan

    # R size (% of entry)
    r_pct = df["stop_distance"] / df["entry_price"] * 100.0
    df["R_size_bucket"] = _bucket_pct(
        r_pct,
        edges=[-np.inf, 0.5, 1.0, 2.0, 3.0, np.inf],
        labels=["<0.5%", "0.5-1%", "1-2%", "2-3%", "3%+"],
    )

    df["t1_hit_str"] = df["t1_hit"].astype(str)

    # Confirmation lag (min) bucket
    trig = pd.to_datetime(df["trigger_ts"])
    conf = pd.to_datetime(df["confirmation_ts"])
    lag_min = (conf - trig).dt.total_seconds() / 60.0
    df["confirmation_lag_min"] = pd.cut(
        lag_min,
        bins=[-np.inf, 15, 30, 45, 60, np.inf],
        labels=["<=15", "15-30", "30-45", "45-60", "60+"],
    )

    return df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_block(title: str, df: pd.DataFrame, top_n: int = 20) -> None:
    print(f"\n  {title}: {len(df):,}")
    if df.empty:
        print("    [none]")
        return
    for _, r in df.head(top_n).iterrows():
        print(
            f"    [{r['dims']}] {r['cell']}  "
            f"n={r['n']:,} PF={r['pf']:.3f} WR={r['wr']:.1f}% "
            f"Sh={r['sharpe']:.2f} mo[win={r['win_mo_pct']}% "
            f"lose={r['lose_mo_pct']}% top={r['top_mo_pct']}%] "
            f"NET={r['net']:,.0f}"
        )


def run_pattern(name: str, csv: Path, dims: list[str], prep_fn) -> dict:
    print(f"\n{'='*78}\n{name.upper()}\n{'='*78}")
    if not csv.exists():
        print(f"  MISSING: {csv}")
        return {}

    df = pd.read_csv(csv)
    print(f"  rows: {len(df):,}")
    df = prep_fn(df)

    # Available dims after prep
    have = [d for d in dims if d in df.columns]
    missing = [d for d in dims if d not in df.columns]
    if missing:
        print(f"  missing dims: {missing}")
    print(f"  scanning {len(have)} dims: {have}")

    pnl = "net_pnl"
    print(
        f"  AGGREGATE: n={len(df):,}  PF={pf_of(df[pnl]):.3f}  "
        f"WR={100 * (df[pnl] > 0).mean():.1f}%"
    )
    agg_sharpe = _sharpe_of(df.groupby("_session_date")[pnl].sum())
    agg_m = _monthly_stats(df, pnl, "_month")
    print(
        f"  AGGREGATE Sh(daily)={agg_sharpe:.3f}  monthly[n={agg_m['n_months']} "
        f"win={agg_m['win_mo_pct']}% lose={agg_m['lose_mo_pct']}% "
        f"top={agg_m['top_mo_pct']}%]"
    )

    cells = scan_cells(
        df,
        have,
        pnl,
        max_combo=3,
        date_col="_session_date",
        month_col="_month",
    )
    print(f"  total cells scanned: {len(cells):,}")

    survivors_mask = (
        (cells["n"] >= N_MIN_SURVIVOR)
        & (cells["pf"] >= PF_MIN_SURVIVOR)
        & (cells["sharpe"] >= SHARPE_MIN_SURVIVOR)
    )
    survivors = cells[survivors_mask].sort_values(
        ["pf", "n"], ascending=[False, False]
    )

    ship_mask = (
        (cells["n"] >= N_MIN_SHIP)
        & (cells["pf"] >= PF_MIN_SHIP)
        & (cells["sharpe"] >= SHARPE_MIN_SHIP)
        & (cells["lose_mo_pct"] <= LOSING_MONTHS_PCT_MAX_SHIP)
        & (cells["top_mo_pct"] < TOP_MONTH_CONCENTRATION_MAX)
        & (cells["win_mo_pct"] >= 55.0)
    )
    ship = cells[ship_mask].sort_values(["pf", "n"], ascending=[False, False])

    _print_block(
        f"SURVIVORS (n>={N_MIN_SURVIVOR}, PF>={PF_MIN_SURVIVOR}, Sh>0)",
        survivors,
        top_n=10,
    )
    _print_block(
        f"SHIP-ELIGIBLE (n>={N_MIN_SHIP}, PF>={PF_MIN_SHIP}, Sh>={SHARPE_MIN_SHIP}, "
        f"win_mo>=55%, lose_mo<={LOSING_MONTHS_PCT_MAX_SHIP}%, "
        f"top_mo<{TOP_MONTH_CONCENTRATION_MAX}%)",
        ship,
        top_n=10,
    )

    # Persist survivors
    out = _OUT_DIR / f"_deep_cellmine_{name}_survivors.csv"
    survivors.to_csv(out, index=False)
    print(f"\n  wrote {out.relative_to(_REPO)} ({len(survivors):,} rows)")

    return {
        "name": name,
        "total_cells": len(cells),
        "n_survivors": len(survivors),
        "n_ship_eligible": len(ship),
        "top10_pf": survivors.head(10),
        "ship_top10": ship.head(10),
    }


def main():
    # IMPORTANT: only ex-ante (pre-trade) discriminators. Post-hoc outcome cols
    # (exit_reason, hit_t1) leak forward and give tautological PF=inf cells.
    inside_bar_dims = [
        "direction",
        "cap_segment",
        "time_bucket",
        "mother_size_bucket",
        "inside_pos_bucket",
        "R_size_bucket",
        "dow",
        "month",
        "quarter",
        "entry_hour_bucket",
    ]
    rsi_dims = [
        "direction",
        "cap_segment",
        "tod_bucket",
        "rsi_severity",
        "rsi_severity_fine",
        "R_size_bucket",
        "dow",
        "month",
        "quarter",
        "confirmation_lag_min",
    ]

    results = []
    results.append(
        run_pattern(
            "inside_bar_breakout",
            _OUT_DIR / "inside_bar_breakout_trades.csv",
            inside_bar_dims,
            prep_inside_bar,
        )
    )
    results.append(
        run_pattern(
            "rsi_extreme_reversal",
            _OUT_DIR / "rsi_extreme_reversal_trades.csv",
            rsi_dims,
            prep_rsi,
        )
    )

    # Final summary
    print(f"\n\n{'='*78}\nFINAL SUMMARY\n{'='*78}")
    for r in results:
        if not r:
            continue
        print(
            f"  {r['name']}: cells_scanned={r['total_cells']:,}  "
            f"survivors={r['n_survivors']:,}  ship_eligible={r['n_ship_eligible']:,}"
        )


if __name__ == "__main__":
    main()
