"""Does the min-notional floor drop good trades? Reconstructs ADV to find out.

close_dn caps each order at `max_participation_pct` of the name's ADV, then
SKIPS the name entirely if the capped notional falls below
`min_notional_after_cap_inr` (Rs25,000). On 2026-09-07 that floor removed 5 of
9 fires — the day cap, slots and cash never bound — so it is now the binding
constraint on the book, not capital.

The floor's stated justification is that "the Rs20/order brokerage cap stops
helping below ~Rs67k notional while delivery STT stays proportional". That is
the MTF regime. Live measurement across 128 fills puts CNC at 0.22% of notional
FLAT down to Rs1,465 with no fixed component (Zerodha delivery brokerage is
Rs0), and every one of the five names dropped that day was CNC.

The research ledgers carry no ADV column, so the cap cannot be replayed
directly. This rebuilds it the way production does — median DAILY RUPEE
TURNOVER over the prior sessions, from the same monthly 5m feathers the sanity
scripts read — and then asks the only question that matters: do the names the
floor removes earn more or less than the ones it keeps?

ADV is computed strictly from sessions BEFORE the signal date, matching
close_dn_baseline_build._adv_for_symbol. Measuring participation against a
window that includes the trade day is look-ahead, and these setups fire on
volume spikes, so the trade day is systematically atypical.

Usage:
    python tools/sub9_research/floor_ablation_close_dn.py
    python tools/sub9_research/floor_ablation_close_dn.py --floors 0 10000 25000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from tools.sub9_research.rank_ablation_close_dn_overnight import (  # noqa: E402
    load_cell, SPLITS, PNL_COL, profit_factor,
)

MONTHLY_DIR = _REPO_ROOT / "backtest-cache-download" / "monthly"
ADV_SESSIONS = 30           # production: session_date-30d .. session_date-1d
MIN_SESSIONS = 10           # production returns None below this
PARTICIPATION_PCT = 0.01    # config max_participation_pct
DESIRED_NOTIONAL = 25_000.0  # CNC desired at the new Rs25k slot
LEDGER_NOTIONAL = 100_000.0


def build_turnover() -> pd.DataFrame:
    """Per (symbol, date) rupee turnover from the monthly 5m feathers."""
    frames = []
    for p in sorted(MONTHLY_DIR.glob("*_5m_enriched.feather")):
        try:
            df = pd.read_feather(p, columns=["date", "symbol", "close", "volume"])
        except Exception:
            continue
        df["d"] = pd.to_datetime(df["date"]).dt.date
        df["t"] = df["close"].astype(float) * df["volume"].astype(float)
        frames.append(df.groupby(["symbol", "d"], observed=True)["t"].sum().reset_index())
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.groupby(["symbol", "d"], observed=True)["t"].sum().reset_index()


def adv_lookup(turnover: pd.DataFrame) -> dict:
    """{(symbol, date): trailing median daily turnover, sessions BEFORE date}."""
    adv = {}
    for sym, g in turnover.groupby("symbol", observed=True):
        g = g.sort_values("d")
        vals = g["t"].tolist()
        dates = g["d"].tolist()
        for i, dt in enumerate(dates):
            prior = [v for v in vals[max(0, i - ADV_SESSIONS):i] if v > 0]
            if len(prior) >= MIN_SESSIONS:
                adv[(sym, dt)] = float(np.median(prior))
    return adv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--floors", type=float, nargs="+",
                    default=[0.0, 5_000.0, 10_000.0, 15_000.0, 25_000.0])
    args = ap.parse_args()

    print("Reconstructing ADV from monthly 5m feathers "
          "(median daily rupee turnover, %d prior sessions)..." % ADV_SESSIONS)
    turnover = build_turnover()
    if turnover.empty:
        print("  no monthly feathers found under %s" % MONTHLY_DIR)
        return 1
    adv = adv_lookup(turnover)
    print("  turnover rows %d | ADV points %d\n" % (len(turnover), len(adv)))

    for split in SPLITS:
        df = load_cell(split)
        if df.empty:
            continue
        df = df.copy()
        df["bare"] = df["symbol"].astype(str).str.replace("NSE:", "", regex=False)
        df["adv"] = [adv.get((s, d)) for s, d in zip(df["bare"], df["signal_date"])]
        have = df[df["adv"].notna()].copy()
        if have.empty:
            print("=== %s: no ADV overlap ===\n" % split.upper())
            continue
        have["capped"] = np.minimum(DESIRED_NOTIONAL, PARTICIPATION_PCT * have["adv"])
        have["ret"] = have[PNL_COL] / LEDGER_NOTIONAL

        print("=== %s | %d fires, ADV resolved for %d (%.0f%%) ===" % (
            split.upper(), len(df), len(have), 100 * len(have) / len(df)))
        print("  %-10s %6s %8s %10s %11s %10s" % (
            "floor", "kept", "skipped", "skip %", "kept mean", "SKIPPED mean"))
        for f in args.floors:
            keep = have[have["capped"] >= f]
            skip = have[have["capped"] < f]
            km = 100 * keep["ret"].mean() if len(keep) else float("nan")
            sm = 100 * skip["ret"].mean() if len(skip) else float("nan")
            print("  %-10s %6d %8d %9.0f%% %+10.3f%% %+11s" % (
                format(f, ",.0f"), len(keep), len(skip),
                100 * len(skip) / len(have),
                km, ("%.3f%%" % sm) if len(skip) else "-"))
        # what the floor currently in production removes
        cur = have[have["capped"] < 25_000.0]
        if len(cur):
            print("\n  names the Rs25,000 floor removes: n=%d  mean %+.3f%%  PF %.3f" % (
                len(cur), 100 * cur["ret"].mean(), profit_factor(cur[PNL_COL])))
            kept = have[have["capped"] >= 25_000.0]
            print("  names it keeps                  : n=%d  mean %+.3f%%  PF %.3f" % (
                len(kept), 100 * kept["ret"].mean(), profit_factor(kept[PNL_COL])))
            # tick-relative cost of the removed names
            px = cur["entry_price"].astype(float)
            tick = np.where(px < 250, 0.05, 0.05)
            cur = cur.assign(tick_pct=100 * tick / px)
            print("  removed names, one tick as %% of price: "
                  "median %.3f%%  p90 %.3f%%  max %.3f%%" % (
                      cur["tick_pct"].median(), cur["tick_pct"].quantile(0.9),
                      cur["tick_pct"].max()))
            for thr in (0.25, 0.5, 1.0):
                clean = cur[cur["tick_pct"] <= thr]
                if len(clean):
                    print("    of those, tick <= %.2f%%: n=%-4d mean %+.3f%%  PF %.3f" % (
                        thr, len(clean), 100 * clean["ret"].mean(),
                        profit_factor(clean[PNL_COL])))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
