"""Monster-conditioning validation for the 4 multiday capitulation setups.

Hypothesis (pre-stated 2026-07-21, from a 17-feature screen with 3-period
sign-consistency): fires on HIGH-volatility (ATR14%) + DEEP-trailing-weakness
(dist below SMA50) names carry essentially all of the basket's net edge; the
low-ATR/shallow half is flat-to-negative after fees.

Anti-bias guards:
- Rules are PARAMETER-FREE (no thresholds fitted on Discovery):
    RULE-X: within each day's per-setup fires, keep ATR14% >= day-median AND
            dist_sma50 <= day-median (day-relative; implementable in the
            composite selector which already sees the day's candidates).
    RULE-T: keep fires with ATR14% >= trailing-120d median of FIRES' ATR and
            dist_sma50 <= trailing-120d median (absolute, uses only past
            fires; first 120d of the panel warms up).
- Features computed strictly from data through the signal close (no lookahead).
- Fees: real MTF model (Rs20/order + pledge + interest) at the Rs1L live-plan
  sizing, identical to the target-exit study.
- Evaluation on all 3 chronological periods (Disc 2023-24 / OOS 2025 /
  Recent 2026); the feature screen itself is the only fitted step and is
  charged via Harvey-Liu M=64 (17 features x 4 setups screened, rounded up).
- Baseline geometry: K-day close exit (no target exits) — the filter is
  orthogonal to the exit rule.

Outputs: per-setup + pooled trades CSVs (signal_date, net_pnl_inr, ...) for
baseline/RULE-X/RULE-T baskets under reports/sub9_sanity/, and a console
summary. Confidence cards are rendered by the companion runner.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "sub9_research"))

from multiday_target_exit_study import (  # noqa: E402
    CLEAN, CELLS, MARGIN, LEV, load_eligible, build_trades,
)
from tools.sub7_validation.build_per_setup_pnl import calc_fee_mtf  # noqa: E402

OUT_DIR = ROOT / "reports" / "sub9_sanity"
TRAIL_DAYS = 120


def wilder_rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    ru = d.clip(lower=0.0).ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    rd = (-d).clip(lower=0.0).ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    return 100.0 - 100.0 / (1.0 + ru / rd)


def add_features(dd: pd.DataFrame) -> pd.DataFrame:
    """ATR14% + dist_sma50 at the signal close (no lookahead)."""
    dd = dd.sort_values(["symbol", "date"]).copy()
    g = dd.groupby("symbol", sort=False)
    c, h, l = dd["close"], dd["high"], dd["low"]
    prev_c = g["close"].shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    dd["_tr"] = tr
    dd["f_atr14_pct"] = dd.groupby("symbol", sort=False)["_tr"].transform(
        lambda s: s.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()) / c * 100
    m50 = g["close"].transform(lambda s: s.rolling(50, min_periods=50).mean())
    dd["f_dist_sma50"] = (c / m50 - 1) * 100
    return dd.drop(columns=["_tr"])


def net_pnl_row(entry: float, exit_px: float, hold_days: int) -> float:
    notional = MARGIN * LEV
    qty = notional / entry
    gross = (exit_px - entry) * qty
    fees = calc_fee_mtf(notional, float(exit_px * qty), MARGIN, int(hold_days))
    return gross - fees


def period(dates) -> np.ndarray:
    y = pd.to_datetime(dates).dt.year
    return np.where(y <= 2024, "Disc23-24", np.where(y == 2025, "OOS25", "Rec26"))


def rule_x_mask(sel: pd.DataFrame) -> pd.Series:
    """Day-relative: ATR >= day-median AND depth <= day-median (>= half kept
    per condition; single-fire days pass both trivially — that is the honest
    live behavior of a day-relative rule)."""
    atr_med = sel.groupby("date")["f_atr14_pct"].transform("median")
    d50_med = sel.groupby("date")["f_dist_sma50"].transform("median")
    return (sel["f_atr14_pct"] >= atr_med) & (sel["f_dist_sma50"] <= d50_med)


def rule_t_mask(sel: pd.DataFrame) -> pd.Series:
    """Trailing-absolute: vs the previous TRAIL_DAYS calendar days of fires'
    feature medians (excludes today; NaN warmup rows are dropped)."""
    sel = sel.sort_values("date")
    daily = sel.groupby("date")[["f_atr14_pct", "f_dist_sma50"]].median()
    trail_atr = daily["f_atr14_pct"].rolling(f"{TRAIL_DAYS}D", min_periods=20).median().shift(1)
    trail_d50 = daily["f_dist_sma50"].rolling(f"{TRAIL_DAYS}D", min_periods=20).median().shift(1)
    ta = sel["date"].map(trail_atr)
    td = sel["date"].map(trail_d50)
    return (sel["f_atr14_pct"] >= ta) & (sel["f_dist_sma50"] <= td) & ta.notna() & td.notna()


def pf(x: np.ndarray) -> float:
    w = x[x > 0].sum()
    l = -x[x < 0].sum()
    return w / l if l > 0 else np.inf


def summarize(tag: str, df: pd.DataFrame) -> None:
    for p in ["Disc23-24", "OOS25", "Rec26", "ALL"]:
        sub = df if p == "ALL" else df[df["per"] == p]
        if len(sub) < 20:
            print(f"    {tag:<10} {p:<9} n={len(sub):5d}  (too small)")
            continue
        x = sub["net_pnl_inr"].values
        print(f"    {tag:<10} {p:<9} n={len(sub):5d} net={x.sum():>12,.0f} "
              f"PF={pf(x):5.2f} avg={x.mean():>8,.0f} win%={100*(x>0).mean():4.1f}")


def main() -> None:
    dd = pd.read_feather(CLEAN)
    dd["date"] = pd.to_datetime(dd["date"])
    dd["bare"] = dd["symbol"].astype(str).str.replace("NSE:", "", regex=False).str.upper()
    dd = add_features(dd)
    elig = load_eligible()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pooled = {"baseline": [], "rule_x": [], "rule_t": []}
    for name, (mode, params) in CELLS.items():
        sel, K = build_trades(dd, elig, mode, params)
        sel = sel.copy()
        sel["hold_days"] = K + 1
        sel["net_pnl_inr"] = [
            net_pnl_row(e, x, K + 1)
            for e, x in zip(sel["open_next"].values, sel["close_exit"].values)
        ]
        sel["per"] = period(sel["date_next"])
        sel["signal_date"] = pd.to_datetime(sel["date"]).dt.date
        mx = rule_x_mask(sel)
        mt = rule_t_mask(sel)
        print(f"\n=== {name} (K={K}) fires={len(sel)} "
              f"keep_x={mx.mean()*100:.0f}% keep_t={mt.mean()*100:.0f}%")
        cols = ["signal_date", "symbol", "net_pnl_inr", "per",
                "f_atr14_pct", "f_dist_sma50"]
        baskets = {"baseline": sel, "rule_x": sel[mx], "rule_t": sel[mt]}
        for tag, df in baskets.items():
            summarize(tag, df)
            out = df[cols].copy()
            out.insert(0, "setup", name)
            out.to_csv(OUT_DIR / f"_monster_cond_{name}_{tag}.csv", index=False)
            pooled[tag].append(out)

    print("\n=== POOLED (4 setups) ===")
    for tag, parts in pooled.items():
        alldf = pd.concat(parts, ignore_index=True)
        summarize(tag, alldf)
        alldf.to_csv(OUT_DIR / f"_monster_cond_POOLED_{tag}.csv", index=False)
    print(f"\nCSVs -> {OUT_DIR}/_monster_cond_*.csv")


if __name__ == "__main__":
    main()
