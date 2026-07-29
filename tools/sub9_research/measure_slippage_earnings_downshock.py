"""
measure_slippage_earnings_downshock.py
======================================
MEASUREMENT (not a backtest).  Estimate the REAL per-side execution slippage for the
EXACT trade cohort produced by Stage-4 of `earnings_downshock_continuation_short`.

WHY THIS EXISTS
---------------
Stage-4 (`sanity_earnings_downshock_continuation_short.py`) showed that real Zerodha
MIS fees are only ~0.083% of notional for this cohort.  That makes SLIPPAGE the entire
verdict:

    net +0.512%/trade (PF 1.50) @ 20bp/side
    net -0.085%/trade (PF 0.93) @ 50bp/side
    break-even                  ~46bp/side
    brief falsifier #1 (net < +0.15%/trade)  trips at ~38bp/side

Everything so far ASSUMED the slippage number.  This script measures it.

PRE-COMMITTED DECISION BANDS (fixed BEFORE the measurement was run)
-------------------------------------------------------------------
    > ~38 bp/side  -> FAILS pre-registered falsifier #1 -> KILL
    ~20 to ~38 bp  -> survives, thin
    < ~20 bp       -> comfortable

THE COHORT
----------
`reports/sub9_sanity/_earnings_downshock_continuation_short_trades_discovery.csv` (n=220).
Every trade is a SHORT:
    entry leg : decision at the CLOSE of the 09:15-09:20 5m bar (the "09:20 print").
                The first executable minute for a 09:20:00 market order is the
                09:20-09:21 1m bar.  ENTRY FILL BAR = 09:20 (1m).
    exit  leg : cover at the session close = close of the 15:25-15:30 5m bar.
                The last executable minute is the 15:29-15:30 1m bar.
                EXIT FILL BAR = 15:29 (1m).
(Verified against the CSV: entry_price == 1m close at 09:19, exit_price == 1m close at
15:29, for spot-checked trades.)

DATA
----
`backtest-cache-download/monthly/YYYY_MM_1m.feather` -- 1-minute OHLCV, IST-naive `ts`.
Cohort spans 2023-01-20 .. 2024-11-19, fully inside 1m coverage (2023_01 .. 2026_04).
No signal in this cohort is >= 2026-05-01.
`data/tick_archive/` holds ONE day (2026-01-20) -- outside the cohort, and the cohort has
no trade on it, so true tick/depth data is UNAVAILABLE for this cohort.  Every method
below is therefore a PROXY built from 1-minute bars.  See the "WHAT THIS CANNOT SEE"
section at the bottom of the printed report.

METHODS (each with its bias direction stated)
---------------------------------------------
M1  INTRA-BAR RANGE.  (high-low)/close of the fill bar, in bps.  Half of that
    ("half-range") is the classic upper-bound proxy for half-spread + immediate impact:
    a market order into the bar fills somewhere in [low, high]; mid-to-adverse-extreme is
    half the range.  BIAS: UPWARD for the half-range on volatile bars (range contains
    genuine drift, not only spread), but DOWNWARD if the true book is wider than the
    traded range (thin books can print a narrow range while quoting a wide spread).

M2  ROLL (1984) implied effective spread.  s = 2*sqrt(-cov(r_t, r_{t-1})) on 1m returns;
    per-side (effective half-spread) = s/2 = sqrt(-cov).  Only estimable when the serial
    covariance is negative.  BIAS: DOWNWARD in aggregate because the non-estimable
    (positive-cov, i.e. trending) symbol-days are dropped -- and a -8% earnings name at
    09:20 is exactly a trending tape.  Also assumes no informed-trade drift, which is
    violated here.  Fraction estimable is reported.

M3  TICK-SIZE FLOOR.  Half of one tick over the price, in bps.  The tick is INFERRED
    empirically per symbol-day (gcd of the day's price prints in paise) rather than
    assumed, because NSE runs Rs 0.05 and Rs 0.01 ticks depending on the security/era.
    This is a HARD FLOOR: no fill can beat it.  BIAS: pure lower bound.

M4  ADJACENT-PRINT DISPERSION.  |close_t - open_{t+1}| / close_t in bps at the fill bar
    (the "gap between consecutive prints" -- a direct read on how far price jumps when
    the book is thin), plus the stdev of 1m closes across the 09:15-09:25 entry window.
    BIAS: the close-to-next-open gap is a good spread proxy on thin names (bid-ask
    bounce between prints) but mixes in real drift; UPWARD-biased on fast tapes.

M5  TURNOVER-RELATIVE SIZE.  Our Rs 1L notional as a share of the fill bar's rupee
    turnover (1m) and of the 5m window's turnover.  Not a bp estimate by itself -- it
    tells us whether SPREAD or IMPACT dominates.  Rupee turnover uses typical price
    ((h+l+c)/3) * volume.  BIAS: 1m bar volume from the cache may under-count for the
    most illiquid names, which would OVERSTATE our participation.

M6  MARKET IMPACT (square-root law).  M1-M4 are all SPREAD proxies -- none of them
    contains our own order, so none of them can see impact.  M5 shows our Rs 1L is a
    double-digit share of a fill MINUTE's turnover on adv_low, so impact cannot be
    ignored.  Two independent impact estimates:
      M6a  Almgren-style daily square-root law:
              impact_bps = Y * sigma_daily_bps * sqrt(Q / ADV20)
           sigma_daily from the Parkinson estimator on the trade day's own high/low
           (range / (2*sqrt(ln 2))), ADV20 = the cohort CSV's rupee adv20, Q = Rs 1L.
           Reported at Y=0.5 (Almgren et al. calibration) and Y=1.0 (stress).
      M6b  Micro "bar-sweep": taking a fraction f of a bar's traded volume walks you
           roughly f of that bar's half-range.  impact_bps = min(1, f_5m) * half_range_5m,
           where f_5m is our share of the 5-minute execution block's turnover.
           This is the more pessimistic, more local read.
    BIAS: M6a assumes an order worked over a normal horizon at a sane participation
    rate.  A naive single-minute market order would be WORSE.  M6b assumes we work the
    order across the 5m block; a single-minute sweep would again be worse.

FINAL COMBINE = per-side SPREAD (half-spread, from M2 with the M3 tick floor)
                + per-side IMPACT (M6).  M1 half-range is retained only as a sanity
                ceiling to report against, NOT as a cap.

OUTPUT
------
    reports/sub9_sanity/_earnings_downshock_slippage.csv   (per trade, per leg, all methods)
    stdout report                                          (tables + verdict)

Run:  .venv/Scripts/python tools/sub9_research/measure_slippage_earnings_downshock.py

FEASIBILITY CORRECTION (2026-07-29) -- parameterised EXIT LEG
--------------------------------------------------------------
Point 5 of "WHAT THIS MEASUREMENT CANNOT SEE" below became the binding constraint:
Zerodha auto-squares MIS equity from ~15:20, so the 15:29 exit leg measured by the
default run is UNREACHABLE.  `--cohort-csv` / `--out-csv` / `--exit-fill-minute` /
`--exit-5m-window` / `--exit-window` / `--exit-roll-window` let the SAME triangulation
be re-run against the feasible 15:15 exit, so the exit leg's slippage is MEASURED at
15:15 rather than assumed equal to the 15:29 measurement.
Defaults are unchanged -> the original run is reproducible with no arguments.
This is a feasibility correction, NOT an exit-time sweep (lesson #31).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------
# LOCKED MEASUREMENT CONSTANTS (declared up-front; nothing below re-tunes them)
# ----------------------------------------------------------------------------------
LOCKED = {
    # cohort
    "cohort_csv": "reports/sub9_sanity/_earnings_downshock_continuation_short_trades_discovery.csv",
    "out_csv": "reports/sub9_sanity/_earnings_downshock_slippage.csv",
    "monthly_1m_glob": "backtest-cache-download/monthly/{ym}_1m.feather",

    # legs -- the 1m bar a live market order would actually execute inside
    "entry_fill_minute": "09:20",     # first executable minute after the 09:20 decision
    "exit_fill_minute": "15:29",      # last executable minute of the 15:25-15:30 bar

    # windows
    "entry_window": ("09:15", "09:25"),      # dispersion window (brief SS -- the open ramp)
    "entry_roll_window": ("09:15", "10:15"), # Roll needs >= ~30 obs to be meaningful
    "exit_window": ("15:20", "15:29"),
    "exit_roll_window": ("14:30", "15:29"),
    "entry_5m_window": ("09:20", "09:24"),   # the 5m block we would work the order in
    "exit_5m_window": ("15:25", "15:29"),

    "roll_min_obs": 25,

    # sizing (must match Stage-4)
    "notional_inr": 100_000.0,

    # participation flag threshold
    "participation_flag_pct": 10.0,   # our order >10% of the fill minute's turnover

    # M6 impact
    "impact_Y_base": 0.5,             # Almgren et al. square-root-law coefficient
    "impact_Y_stress": 1.0,           # stress calibration
    "parkinson_k": 2.0 * math.sqrt(math.log(2.0)),   # range -> sigma

    # pre-committed verdict bands (bp per side)
    "band_kill_above": 38.0,
    "band_thin_above": 20.0,

    # Stage-4 economics, for the re-computation
    "fees_pct_of_notional": None,     # derived from the cohort CSV, not assumed
}


# ----------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------
def _t(s: str) -> pd.Timestamp:
    return pd.Timestamp(s).time()


def bps(x) -> float:
    return float(x) * 10_000.0


def infer_tick(prices: np.ndarray) -> float:
    """Infer the NSE tick for a symbol-day from the gcd of its price prints (in paise).

    Returns 0.05 or 0.01.  Anything whose gcd is a multiple of 5 paise is a 5-paise-tick
    security for our purposes; everything else is 1 paise.  Derived, never assumed.
    """
    p = np.unique(np.round(np.asarray(prices, dtype=float) * 100.0).astype(np.int64))
    p = p[p > 0]
    if p.size < 2:
        return 0.05
    g = 0
    for v in p:
        g = math.gcd(g, int(v))
        if g == 1:
            break
    if g <= 0:
        return 0.05
    return 0.05 if (g % 5 == 0) else 0.01


def roll_half_spread_bps(closes: np.ndarray, min_obs: int) -> tuple[float, int, bool]:
    """Roll (1984).  Returns (per-side bps, n_obs, estimable)."""
    c = np.asarray(closes, dtype=float)
    c = c[np.isfinite(c) & (c > 0)]
    if c.size < min_obs + 2:
        return (np.nan, int(c.size), False)
    r = np.diff(np.log(c))
    if r.size < min_obs:
        return (np.nan, int(r.size), False)
    cov = float(np.cov(r[1:], r[:-1], ddof=1)[0, 1])
    if not np.isfinite(cov) or cov >= 0:
        return (np.nan, int(r.size), False)      # not estimable: trending tape
    half = math.sqrt(-cov)                        # s/2 in log-return units
    return (bps(half), int(r.size), True)


def window(df: pd.DataFrame, lo: str, hi: str) -> pd.DataFrame:
    tt = df["ts"].dt.time
    return df[(tt >= _t(lo)) & (tt <= _t(hi))]


def bar_at(df: pd.DataFrame, hhmm: str) -> pd.Series | None:
    m = df[df["ts"].dt.time == _t(hhmm)]
    return None if m.empty else m.iloc[0]


def next_bar_after(df: pd.DataFrame, hhmm: str) -> pd.Series | None:
    m = df[df["ts"].dt.time > _t(hhmm)]
    return None if m.empty else m.iloc[0]


def q(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return dict(n=0, median=np.nan, p25=np.nan, p75=np.nan, p90=np.nan, mean=np.nan)
    return dict(n=int(s.size), median=float(s.median()), p25=float(s.quantile(.25)),
                p75=float(s.quantile(.75)), p90=float(s.quantile(.90)), mean=float(s.mean()))


# ----------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="slippage measurement (see module docstring)")
    ap.add_argument("--cohort-csv", default=None)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--exit-fill-minute", default=None, metavar="HH:MM",
                    help="the 1m bar a live cover order executes inside "
                         "(15:29 = session close; 15:14 = the feasible 15:15 MIS exit)")
    ap.add_argument("--exit-5m-window", default=None, metavar="HH:MM,HH:MM")
    ap.add_argument("--exit-window", default=None, metavar="HH:MM,HH:MM")
    ap.add_argument("--exit-roll-window", default=None, metavar="HH:MM,HH:MM")
    a = ap.parse_args()
    if a.cohort_csv:
        LOCKED["cohort_csv"] = a.cohort_csv
    if a.out_csv:
        LOCKED["out_csv"] = a.out_csv
    if a.exit_fill_minute:
        LOCKED["exit_fill_minute"] = a.exit_fill_minute
    for key, val in (("exit_5m_window", a.exit_5m_window),
                     ("exit_window", a.exit_window),
                     ("exit_roll_window", a.exit_roll_window)):
        if val:
            lo, hi = val.split(",")
            LOCKED[key] = (lo.strip(), hi.strip())
    print(f"[cfg] cohort={LOCKED['cohort_csv']}")
    print(f"[cfg] exit leg: fill_minute={LOCKED['exit_fill_minute']}  "
          f"5m_block={LOCKED['exit_5m_window']}  window={LOCKED['exit_window']}  "
          f"roll={LOCKED['exit_roll_window']}")

    root = Path(__file__).resolve().parents[2]
    tr = pd.read_csv(root / LOCKED["cohort_csv"])
    tr["sym"] = tr["symbol"].str.replace("NSE:", "", regex=False)
    tr["entry_date"] = pd.to_datetime(tr["entry_date"]).dt.date

    print("=" * 100)
    print("SLIPPAGE MEASUREMENT -- earnings_downshock_continuation_short (Stage-4 cohort)")
    print("=" * 100)
    print(f"cohort rows              : {len(tr)}")
    print(f"cohort entry_date range  : {min(tr.entry_date)} .. {max(tr.entry_date)}")
    print(f"unique symbols           : {tr['sym'].nunique()}")
    print(f"ADV tiers                : {dict(tr['adv_tier'].value_counts())}")
    print(f"mean notional deployed   : Rs {(tr.entry_price*tr.qty).mean():,.0f}")
    print(f"gross pnl_pct mean       : {tr.pnl_pct.mean():+.4f}%")
    assert max(tr.entry_date) < pd.Timestamp("2026-05-01").date(), "cohort leaks past 2026-05-01"

    # ------------------------------------------------------------------ load 1m
    tr["ym"] = pd.to_datetime(tr["entry_date"]).dt.strftime("%Y_%m")
    need = tr.groupby("ym")["sym"].apply(lambda s: set(s)).to_dict()

    frames = {}
    cov_rows = []
    for ym, syms in sorted(need.items()):
        p = root / LOCKED["monthly_1m_glob"].format(ym=ym)
        if not p.exists():
            cov_rows.append(dict(ym=ym, file="MISSING", rows=0, syms_needed=len(syms), syms_found=0))
            continue
        d = pd.read_feather(p, columns=["ts", "symbol", "open", "high", "low", "close", "volume"])
        d = d[d["symbol"].isin(syms)]
        d["d"] = d["ts"].dt.date
        frames[ym] = d
        cov_rows.append(dict(ym=ym, file=p.name, rows=len(d),
                             syms_needed=len(syms), syms_found=d["symbol"].nunique()))
    cov = pd.DataFrame(cov_rows)
    print("\n--- 1m DATA COVERAGE (filtered to cohort symbols) ---")
    print(cov.to_string(index=False))
    print(f"total 1m rows pulled     : {sum(len(f) for f in frames.values()):,}")

    # ------------------------------------------------------------------ per-trade
    rows = []
    for _, t in tr.iterrows():
        d = frames.get(t["ym"])
        rec = dict(signal_date=t["signal_date"], symbol=t["symbol"], sym=t["sym"],
                   entry_date=t["entry_date"], adv_tier=t["adv_tier"], adv20=t["adv20"],
                   entry_price=t["entry_price"], exit_price=t["exit_price"], qty=t["qty"],
                   pnl_pct=t["pnl_pct"], bars_found=0)
        if d is None:
            rows.append(rec)
            continue
        day = d[(d["symbol"] == t["sym"]) & (d["d"] == t["entry_date"])].sort_values("ts")
        rec["bars_found"] = len(day)
        if day.empty:
            rows.append(rec)
            continue

        tick = infer_tick(pd.concat([day["open"], day["high"], day["low"], day["close"]]).values)
        rec["tick"] = tick
        # units sanity: reconstruct the day's rupee turnover from 1m, compare to adv20
        rec["day_turnover_inr"] = float(
            (((day["high"] + day["low"] + day["close"]) / 3.0) * day["volume"]).sum())

        for leg, fill_min, win, roll_win, w5 in (
            ("entry", LOCKED["entry_fill_minute"], LOCKED["entry_window"],
             LOCKED["entry_roll_window"], LOCKED["entry_5m_window"]),
            ("exit", LOCKED["exit_fill_minute"], LOCKED["exit_window"],
             LOCKED["exit_roll_window"], LOCKED["exit_5m_window"]),
        ):
            b = bar_at(day, fill_min)
            if b is None:
                continue
            c = float(b["close"])
            if not (c > 0):
                continue

            # M1 -- intra-bar range
            rng = (float(b["high"]) - float(b["low"])) / c
            rec[f"{leg}_m1_range_bps"] = bps(rng)
            rec[f"{leg}_m1_halfrange_bps"] = bps(rng) / 2.0

            # M2 -- Roll
            w = window(day, *roll_win)
            hb, nobs, ok = roll_half_spread_bps(w["close"].values, LOCKED["roll_min_obs"])
            rec[f"{leg}_m2_roll_half_bps"] = hb
            rec[f"{leg}_m2_roll_nobs"] = nobs
            rec[f"{leg}_m2_roll_estimable"] = ok

            # M3 -- half-tick floor
            rec[f"{leg}_m3_halftick_bps"] = bps((tick / 2.0) / c)

            # M4 -- adjacent-print dispersion
            nb = next_bar_after(day, fill_min)
            if nb is not None:
                rec[f"{leg}_m4_gap_bps"] = bps(abs(float(nb["open"]) - c) / c)
            ww = window(day, *win)
            if len(ww) >= 3:
                rec[f"{leg}_m4_winstd_bps"] = bps(float(ww["close"].std()) / c)
                rec[f"{leg}_m4_winrange_bps"] = bps(
                    (float(ww["high"].max()) - float(ww["low"].min())) / c)
                gaps = (ww["open"].shift(-1) - ww["close"]).abs() / ww["close"]
                rec[f"{leg}_m4_meangap_bps"] = bps(float(gaps.dropna().mean()))

            # M5 -- participation
            typ = (float(b["high"]) + float(b["low"]) + c) / 3.0
            to1 = typ * float(b["volume"])
            rec[f"{leg}_turnover_1m_inr"] = to1
            rec[f"{leg}_partic_1m_pct"] = (LOCKED["notional_inr"] / to1 * 100.0) if to1 > 0 else np.inf
            w5df = window(day, *w5)
            typ5 = ((w5df["high"] + w5df["low"] + w5df["close"]) / 3.0)
            to5 = float((typ5 * w5df["volume"]).sum())
            rec[f"{leg}_turnover_5m_inr"] = to5
            rec[f"{leg}_partic_5m_pct"] = (LOCKED["notional_inr"] / to5 * 100.0) if to5 > 0 else np.inf
            rec[f"{leg}_zero_vol_fill_bar"] = bool(float(b["volume"]) <= 0)

            # M7 -- MODEL-FREE implementation shortfall vs the assumed mark.
            #   M7a: fill at the OPEN of the fill minute (a 09:20:00 / 15:29:00 market
            #        order hits the first print of that minute).
            #   M7b: fill at the VWAP of the 5m execution block (a realistically worked
            #        order).  Signed so that POSITIVE = adverse for this SHORT.
            mark = float(t["entry_price"]) if leg == "entry" else float(t["exit_price"])
            if mark > 0:
                first = float(b["open"])
                # short: entry adverse if we sell BELOW the mark; exit adverse if we buy ABOVE
                sgn = 1.0 if leg == "entry" else -1.0
                rec[f"{leg}_m7a_firstprint_bps"] = bps(sgn * (mark - first) / mark)
                rec[f"{leg}_firstprint"] = first
                if len(w5df) and float(w5df["volume"].sum()) > 0:
                    typ5v = ((w5df["high"] + w5df["low"] + w5df["close"]) / 3.0)
                    vwap5 = float((typ5v * w5df["volume"]).sum() / w5df["volume"].sum())
                    rec[f"{leg}_m7b_vwap_bps"] = bps(sgn * (mark - vwap5) / mark)
                    rec[f"{leg}_vwap5"] = vwap5

            # M6b -- micro bar-sweep impact over the 5m execution block
            if len(w5df) and to5 > 0:
                hi5, lo5 = float(w5df["high"].max()), float(w5df["low"].min())
                halfrng5 = bps((hi5 - lo5) / c) / 2.0
                f5 = min(1.0, LOCKED["notional_inr"] / to5)
                rec[f"{leg}_m6b_impact_bps"] = f5 * halfrng5
                rec[f"{leg}_block5_halfrange_bps"] = halfrng5

        # M6a -- daily square-root law (leg-independent: same Q, ADV, sigma)
        dh, dl, dc = float(t["day_high"]), float(t["day_low"]), float(t["day_close"])
        adv = float(t["adv20"])
        if dc > 0 and adv > 0 and dh > dl > 0:
            sigma_bps = bps((dh - dl) / dc) / LOCKED["parkinson_k"]
            partic_adv = LOCKED["notional_inr"] / adv
            rec["sigma_daily_bps"] = sigma_bps
            rec["partic_adv_pct"] = partic_adv * 100.0
            rec["m6a_impact_bps"] = LOCKED["impact_Y_base"] * sigma_bps * math.sqrt(partic_adv)
            rec["m6a_impact_stress_bps"] = LOCKED["impact_Y_stress"] * sigma_bps * math.sqrt(partic_adv)

        rows.append(rec)

    out = pd.DataFrame(rows)
    outp = root / LOCKED["out_csv"]
    out.to_csv(outp, index=False)
    print(f"\nwrote per-trade slippage measurements -> {outp}  ({out.shape[0]} rows x {out.shape[1]} cols)")

    m = out[out["bars_found"] > 0].copy()
    print(f"trades with 1m bars      : {len(m)} / {len(out)}  "
          f"({len(m)/len(out)*100:.1f}%)")
    for leg in ("entry", "exit"):
        col = f"{leg}_m1_range_bps"
        print(f"  {leg:<5} fill bar present   : {int(m[col].notna().sum())}")

    # units sanity: adv20 must be RUPEE turnover for the M6a sqrt-law to be valid
    ratio = (m["day_turnover_inr"] / m["adv20"]).replace([np.inf, -np.inf], np.nan).dropna()
    print(f"\n  UNITS SANITY  day rupee-turnover(from 1m) / adv20 : "
          f"median {ratio.median():.2f}  p25 {ratio.quantile(.25):.2f}  p75 {ratio.quantile(.75):.2f}")
    print("    (a ratio of order 1-5 confirms adv20 is RUPEE turnover, as M6a assumes;"
          " these are shock days so >1 is expected)")

    # ------------------------------------------------------------------ tables
    def table(metric_col: str, label: str, note: str):
        print(f"\n--- {label} (bp per side) ---")
        print(f"    {note}")
        hdr = f"    {'leg':<7}{'tier':<9}{'n':>5}{'p25':>9}{'median':>9}{'p75':>9}{'p90':>9}{'mean':>9}"
        print(hdr)
        for leg in ("entry", "exit"):
            c = f"{leg}_{metric_col}"
            if c not in m.columns:
                continue
            for tier in ("adv_low", "adv_mid", "ALL"):
                sub = m if tier == "ALL" else m[m["adv_tier"] == tier]
                s = q(sub[c])
                print(f"    {leg:<7}{tier:<9}{s['n']:>5}{s['p25']:>9.1f}{s['median']:>9.1f}"
                      f"{s['p75']:>9.1f}{s['p90']:>9.1f}{s['mean']:>9.1f}")

    table("m1_halfrange_bps", "M1  INTRA-BAR HALF-RANGE",
          "upper bound on half-spread+impact; UPWARD-biased by genuine intrabar drift")
    table("m1_range_bps", "M1b INTRA-BAR FULL RANGE (reference)",
          "the full high-low of the fill bar")

    print(f"\n--- M2  ROLL (1984) estimability ---")
    for leg in ("entry", "exit"):
        c = f"{leg}_m2_roll_estimable"
        if c in m.columns:
            v = m[c].fillna(False)
            for tier in ("adv_low", "adv_mid", "ALL"):
                sub = m if tier == "ALL" else m[m["adv_tier"] == tier]
                vv = sub[c].fillna(False)
                print(f"    {leg:<7}{tier:<9} estimable {int(vv.sum())}/{len(vv)} "
                      f"= {vv.mean()*100:5.1f}%  (neg serial cov)")
    table("m2_roll_half_bps", "M2  ROLL implied EFFECTIVE HALF-SPREAD",
          "estimable subset only -> DOWNWARD-biased (trending/informed tapes dropped)")

    table("m3_halftick_bps", "M3  HALF-TICK FLOOR",
          "hard lower bound; tick inferred per symbol-day from price gcd")
    print("    tick mix:", dict(m["tick"].value_counts()) if "tick" in m.columns else "n/a")

    table("m4_gap_bps", "M4  ADJACENT-PRINT GAP |close_t - open_t+1|",
          "bid-ask bounce proxy at the fill bar; mixes in real drift (UPWARD-biased)")
    table("m4_meangap_bps", "M4b MEAN ADJACENT-PRINT GAP over the leg window",
          "same, averaged over the window -> less noisy")
    table("m4_winstd_bps", "M4c STDEV of 1m closes over the leg window",
          "dispersion of the tape, not a spread; context only")

    print("\n--- M5  TURNOVER-RELATIVE SIZE (Rs 1L order as % of bar turnover) ---")
    hdr = f"    {'leg':<7}{'tier':<9}{'n':>5}{'p25':>9}{'median':>9}{'p75':>9}{'p90':>9}"
    for horizon in ("1m", "5m"):
        print(f"  horizon={horizon}")
        print(hdr)
        for leg in ("entry", "exit"):
            c = f"{leg}_partic_{horizon}_pct"
            if c not in m.columns:
                continue
            for tier in ("adv_low", "adv_mid", "ALL"):
                sub = m if tier == "ALL" else m[m["adv_tier"] == tier]
                s = q(sub[c].replace([np.inf, -np.inf], np.nan))
                print(f"    {leg:<7}{tier:<9}{s['n']:>5}{s['p25']:>9.1f}{s['median']:>9.1f}"
                      f"{s['p75']:>9.1f}{s['p90']:>9.1f}")
    thr = LOCKED["participation_flag_pct"]
    for leg in ("entry", "exit"):
        c = f"{leg}_partic_1m_pct"
        if c in m.columns:
            v = pd.to_numeric(m[c], errors="coerce")
            print(f"    {leg}: trades where Rs1L > {thr:.0f}% of the fill MINUTE turnover: "
                  f"{int((v > thr).sum())}/{int(v.notna().sum())} "
                  f"({(v > thr).mean()*100:.1f}%)")
        c5 = f"{leg}_partic_5m_pct"
        if c5 in m.columns:
            v5 = pd.to_numeric(m[c5], errors="coerce")
            print(f"    {leg}: trades where Rs1L > {thr:.0f}% of the fill 5-MIN turnover: "
                  f"{int((v5 > thr).sum())}/{int(v5.notna().sum())} "
                  f"({(v5 > thr).mean()*100:.1f}%)")

    table("m6b_impact_bps", "M6b MICRO BAR-SWEEP IMPACT (share of 5m block x block half-range)",
          "local impact of working the order across the 5-minute execution block")

    table("m7a_firstprint_bps", "M7a MODEL-FREE SHORTFALL: fill at the FIRST PRINT of the fill minute",
          "signed, +ve = adverse to the SHORT.  No model at all -- this is a real tape price.")
    table("m7b_vwap_bps", "M7b MODEL-FREE SHORTFALL: fill at the 5m-block VWAP (worked order)",
          "signed, +ve = adverse.  Where the volume actually traded vs our assumed mark.")
    for leg in ("entry", "exit"):
        for c, lbl in ((f"{leg}_m7a_firstprint_bps", "first-print"),
                       (f"{leg}_m7b_vwap_bps", "block-VWAP")):
            if c in m.columns:
                v = pd.to_numeric(m[c], errors="coerce").dropna()
                print(f"    {leg:<6}{lbl:<12} adverse (>0) on {float((v>0).mean()*100):5.1f}% of trades; "
                      f"mean {v.mean():+6.1f} bp; mean of adverse-only {v[v>0].mean():+6.1f} bp")
    print("\n--- M6a DAILY SQUARE-ROOT-LAW IMPACT (leg-independent) ---")
    print(f"    {'metric':<26}{'tier':<9}{'n':>5}{'p25':>9}{'median':>9}{'p75':>9}{'p90':>9}{'mean':>9}")
    for cname, lbl in (("partic_adv_pct", "Rs1L as % of ADV20"),
                       ("sigma_daily_bps", "sigma_daily (bps, Parkinson)"),
                       ("m6a_impact_bps", "impact bps  Y=0.5"),
                       ("m6a_impact_stress_bps", "impact bps  Y=1.0 stress")):
        for tier in ("adv_low", "adv_mid", "ALL"):
            sub = m if tier == "ALL" else m[m["adv_tier"] == tier]
            s = q(sub.get(cname))
            print(f"    {lbl:<26}{tier:<9}{s['n']:>5}{s['p25']:>9.1f}{s['median']:>9.1f}"
                  f"{s['p75']:>9.1f}{s['p90']:>9.1f}{s['mean']:>9.1f}")

    # ------------------------------------------------------------------ triangulate
    print("\n" + "=" * 100)
    print("TRIANGULATION  =  per-side SPREAD  +  per-side IMPACT")
    print("=" * 100)
    print("  SPREAD  : Roll half-spread where estimable; else the tier x leg median Roll")
    print("            (the non-estimable cases are TRENDING tapes, i.e. if anything wider);")
    print("            floored at the half-tick, and cross-checked against M4 gap/2.")
    print("  IMPACT  : max(M6a Y=0.5 daily sqrt-law, M6b 5m bar-sweep) -- the more")
    print("            pessimistic of the two independent impact reads, per trade.")
    print("  A STRESS variant uses M1 half-range (the intra-bar upper bound) as spread")
    print("  and the Y=1.0 impact.")

    est_cols, cons_cols, stress_cols = [], [], []
    for leg in ("entry", "exit"):
        f3 = pd.to_numeric(m.get(f"{leg}_m3_halftick_bps"), errors="coerce")
        r2 = pd.to_numeric(m.get(f"{leg}_m2_roll_half_bps"), errors="coerce")
        h1 = pd.to_numeric(m.get(f"{leg}_m1_halfrange_bps"), errors="coerce")
        i6b = pd.to_numeric(m.get(f"{leg}_m6b_impact_bps"), errors="coerce")
        i6a = pd.to_numeric(m.get("m6a_impact_bps"), errors="coerce")
        i6as = pd.to_numeric(m.get("m6a_impact_stress_bps"), errors="coerce")

        # fill non-estimable Roll with the tier x leg median (documented above)
        r2f = r2.copy()
        for tier in ("adv_low", "adv_mid"):
            msk = (m["adv_tier"] == tier)
            r2f.loc[msk & r2f.isna()] = float(r2[msk].median())
        spread = np.maximum(r2f.fillna(float(r2.median())), f3)
        impact = pd.concat([i6a, i6b], axis=1).max(axis=1, skipna=True)

        g4 = pd.to_numeric(m.get(f"{leg}_m4_meangap_bps"), errors="coerce")
        impact_stress = pd.concat([i6as, i6b], axis=1).max(axis=1, skipna=True).fillna(0.0)

        m[f"{leg}_spread_bps"] = spread
        m[f"{leg}_impact_bps"] = impact
        m[f"{leg}_slip_bps"] = spread + impact.fillna(0.0)
        # CONSERVATIVE: treat the FULL adjacent-print gap as the half-spread (not gap/2),
        # and use the Y=1.0 impact.  Plausible-pessimistic, not an upper bound.
        m[f"{leg}_slip_cons_bps"] = np.maximum(spread, g4.fillna(0.0)) + impact_stress
        # STRESS: the intra-bar half-range as the half-spread = definitional upper bound.
        m[f"{leg}_slip_stress_bps"] = np.maximum(h1, spread) + impact_stress
        est_cols.append(f"{leg}_slip_bps")
        cons_cols.append(f"{leg}_slip_cons_bps")
        stress_cols.append(f"{leg}_slip_stress_bps")

    m["roundtrip_blend_bps"] = m[est_cols].sum(axis=1, min_count=1)
    m["perside_blend_bps"] = m["roundtrip_blend_bps"] / 2.0
    m["roundtrip_cons_bps"] = m[cons_cols].sum(axis=1, min_count=1)
    m["perside_cons_bps"] = m["roundtrip_cons_bps"] / 2.0
    m["roundtrip_stress_bps"] = m[stress_cols].sum(axis=1, min_count=1)
    m["perside_stress_bps"] = m["roundtrip_stress_bps"] / 2.0

    for lbl, cols, ps in (("CENTRAL", est_cols, "perside_blend_bps"),
                          ("CONSERV", cons_cols, "perside_cons_bps"),
                          ("STRESS ", stress_cols, "perside_stress_bps")):
        print(f"\n--- {lbl} per-side slippage estimate (bp) ---")
        print(f"    {'leg':<7}{'tier':<9}{'n':>5}{'p25':>9}{'median':>9}{'p75':>9}{'p90':>9}{'mean':>9}")
        for c in cols:
            leg = c.split("_")[0]
            for tier in ("adv_low", "adv_mid", "ALL"):
                sub = m if tier == "ALL" else m[m["adv_tier"] == tier]
                s = q(sub[c])
                print(f"    {leg:<7}{tier:<9}{s['n']:>5}{s['p25']:>9.1f}{s['median']:>9.1f}"
                      f"{s['p75']:>9.1f}{s['p90']:>9.1f}{s['mean']:>9.1f}")
        for tier in ("adv_low", "adv_mid", "ALL"):
            sub = m if tier == "ALL" else m[m["adv_tier"] == tier]
            s = q(sub[ps])
            print(f"    {'RT/2':<7}{tier:<9}{s['n']:>5}{s['p25']:>9.1f}{s['median']:>9.1f}"
                  f"{s['p75']:>9.1f}{s['p90']:>9.1f}{s['mean']:>9.1f}")

    # decomposition
    print("\n--- decomposition of the CENTRAL estimate (mean bp) ---")
    print(f"    {'leg':<7}{'tier':<9}{'spread':>9}{'impact':>9}{'total':>9}")
    for leg in ("entry", "exit"):
        for tier in ("adv_low", "adv_mid", "ALL"):
            sub = m if tier == "ALL" else m[m["adv_tier"] == tier]
            print(f"    {leg:<7}{tier:<9}{sub[f'{leg}_spread_bps'].mean():>9.1f}"
                  f"{sub[f'{leg}_impact_bps'].mean():>9.1f}{sub[f'{leg}_slip_bps'].mean():>9.1f}")

    # ------------------------------------------------------------------ re-price
    print("\n" + "=" * 100)
    print("RE-PRICED EXPECTANCY AT THE MEASURED SLIPPAGE")
    print("=" * 100)
    fees_pct = float((tr["fees_0bp"] / (tr["entry_price"] * tr["qty"]) * 100.0).mean())
    print(f"real Zerodha MIS fees (Stage-4)   : {fees_pct:.4f}% of notional per round trip")

    def reprice(bp_per_side: float, sub: pd.DataFrame) -> dict:
        s = bp_per_side / 10_000.0
        e = sub["entry_price"] * (1.0 - s)    # short sells LOWER
        x = sub["exit_price"] * (1.0 + s)     # covers HIGHER
        qty = sub["qty"]
        gross = (e - x) * qty
        # fees scale with turnover; Stage-4 fee is ~0.083% of notional -> re-apply on slipped
        turn = (e + x) * qty
        fee = turn * (fees_pct / 100.0) / 2.0
        net = gross - fee
        notional = sub["entry_price"] * qty
        npct = net / notional * 100.0
        win = net > 0
        pf = net[net > 0].sum() / abs(net[net < 0].sum()) if (net < 0).any() else np.inf
        return dict(n=len(sub), net_pct=float(npct.mean()), pf=float(pf),
                    winrate=float(win.mean() * 100.0), total=float(net.sum()))

    # use the measured cohort only, so the comparison is apples-to-apples
    carry = ["signal_date", "symbol", "perside_blend_bps", "perside_cons_bps",
             "perside_stress_bps",
             "entry_spread_bps", "exit_spread_bps", "entry_impact_bps", "exit_impact_bps",
             "entry_vwap5", "exit_vwap5", "entry_firstprint", "exit_firstprint"]
    base = tr.merge(m[[c for c in carry if c in m.columns]],
                    on=["signal_date", "symbol"], how="inner")
    print(f"repriced on the {len(base)} trades with 1m data\n")
    print(f"    {'bp/side':>9}{'n':>6}{'net%/trade':>13}{'PF':>8}{'win%':>8}{'total Rs':>12}   verdict-band")
    for b in (0, 10, 15, 20, 25, 30, 35, 38, 40, 45, 46, 50, 60):
        r = reprice(float(b), base)
        band = ("COMFORTABLE" if b < LOCKED["band_thin_above"]
                else "THIN" if b <= LOCKED["band_kill_above"] else "KILL-ZONE")
        print(f"    {b:>9}{r['n']:>6}{r['net_pct']:>13.4f}{r['pf']:>8.2f}"
              f"{r['winrate']:>8.1f}{r['total']:>12,.0f}   {band}")

    # variable, per-trade measured slippage
    def reprice_var(sub: pd.DataFrame, col: str) -> dict:
        ss = sub[col] / 10_000.0
        ee = sub["entry_price"] * (1 - ss)
        xx = sub["exit_price"] * (1 + ss)
        gg = (ee - xx) * sub["qty"]
        ff = (ee + xx) * sub["qty"] * (fees_pct / 100.0) / 2.0
        nn = gg - ff
        nnp = nn / (sub["entry_price"] * sub["qty"]) * 100.0
        ppf = nn[nn > 0].sum() / abs(nn[nn < 0].sum()) if (nn < 0).any() else np.inf
        return dict(net_pct=float(nnp.mean()), pf=float(ppf),
                    win=float((nn > 0).mean() * 100), total=float(nn.sum()))

    results = {}
    for lbl, col in (("CENTRAL", "perside_blend_bps"),
                     ("CONSERVATIVE", "perside_cons_bps"),
                     ("STRESS", "perside_stress_bps")):
        print(f"\n    PER-TRADE MEASURED slippage [{lbl}] (not a flat bp): "
              f"mean {base[col].mean():.1f} bp/side, median {base[col].median():.1f} bp/side")
        r = reprice_var(base, col)
        results[lbl] = (float(base[col].mean()), float(base[col].median()), r)
        print(f"      -> net {r['net_pct']:+.4f}%/trade   PF {r['pf']:.2f}   "
              f"win {r['win']:.1f}%   total Rs {r['total']:,.0f}")
        for tier in ("adv_low", "adv_mid"):
            sb = base[base["adv_tier"] == tier]
            rr = reprice_var(sb, col)
            print(f"      {tier:<9} n={len(sb):<4} measured {sb[col].mean():5.1f} bp/side "
                  f"-> net {rr['net_pct']:+.4f}%/trade  PF {rr['pf']:.2f}")
    npct_mean = results["CENTRAL"][2]["net_pct"]

    # ---- MODEL-FREE fill scenarios: real tape prices + the measured spread crossing ----
    print("\n" + "-" * 100)
    print("MODEL-FREE FILL SCENARIOS  (real tape prices, + measured half-spread to cross)")
    print("-" * 100)
    for lbl, ecol, xcol in (("block-VWAP fill", "entry_vwap5", "exit_vwap5"),
                            ("first-print fill", "entry_firstprint", "exit_firstprint")):
        for add_spread in (False, True):
            b2 = base.dropna(subset=[ecol, xcol]).copy()
            se = (b2["entry_spread_bps"] / 10_000.0) if add_spread else 0.0
            sx = (b2["exit_spread_bps"] / 10_000.0) if add_spread else 0.0
            e = b2[ecol] * (1.0 - se)
            x = b2[xcol] * (1.0 + sx)
            g = (e - x) * b2["qty"]
            f = (e + x) * b2["qty"] * (fees_pct / 100.0) / 2.0
            n = g - f
            npc = n / (b2["entry_price"] * b2["qty"]) * 100.0
            pf2 = n[n > 0].sum() / abs(n[n < 0].sum()) if (n < 0).any() else np.inf
            implied = (((b2["entry_price"] - e) / b2["entry_price"]
                        + (x - b2["exit_price"]) / b2["exit_price"]) / 2.0 * 10_000.0)
            tag = "+ half-spread" if add_spread else "raw (no spread)"
            print(f"  {lbl:<17} {tag:<16} n={len(b2):<4} "
                  f"implied {implied.mean():+6.1f} bp/side (median {implied.median():+6.1f})  "
                  f"net {npc.mean():+.4f}%/trade  PF {pf2:.2f}  win {float((n>0).mean()*100):.1f}%")
            if add_spread:
                results[f"MF::{lbl}"] = (float(implied.mean()), float(implied.median()),
                                         dict(net_pct=float(npc.mean()), pf=float(pf2),
                                              win=float((n > 0).mean() * 100), total=float(n.sum())))

    # ------------------------------------------------------------------ verdict
    print("\n" + "=" * 100)
    print("VERDICT vs PRE-COMMITTED BANDS")
    print("=" * 100)
    print(f"  pre-committed:  >{LOCKED['band_kill_above']:.0f} bp/side = KILL (falsifier #1) | "
          f"{LOCKED['band_thin_above']:.0f}-{LOCKED['band_kill_above']:.0f} = THIN | "
          f"<{LOCKED['band_thin_above']:.0f} = COMFORTABLE")

    def band_of(v: float) -> str:
        return ("KILL" if v > LOCKED["band_kill_above"]
                else "THIN-BUT-ALIVE" if v > LOCKED["band_thin_above"] else "COMFORTABLE")

    for lbl in [k for k in results if k in ("CENTRAL", "CONSERVATIVE", "STRESS") or k.startswith("MF::")]:
        mn, md, r = results[lbl]
        print(f"\n  [{lbl}]  per-side mean {mn:.1f} bp -> {band_of(mn)}"
              f"   |  median {md:.1f} bp -> {band_of(md)}")
        print(f"           net {r['net_pct']:+.4f}%/trade, PF {r['pf']:.2f}  "
              f"(falsifier #1 = net < +0.15%) -> "
              f"{'PASS' if r['net_pct'] >= 0.15 else 'FAIL'}")

    # per-tier bands on the CENTRAL estimate
    print("\n  per-ADV-tier (CENTRAL):")
    for tier in ("adv_low", "adv_mid"):
        sb = base[base["adv_tier"] == tier]
        v = float(sb["perside_blend_bps"].mean())
        rr = reprice_var(sb, "perside_blend_bps")
        print(f"    {tier:<9} {v:5.1f} bp/side -> {band_of(v):<15} "
              f"net {rr['net_pct']:+.4f}%/trade PF {rr['pf']:.2f} -> "
              f"{'PASS' if rr['net_pct'] >= 0.15 else 'FAIL'}")

    # re-write the deliverable CSV now that the derived per-trade estimates exist
    m.to_csv(outp, index=False)
    print(f"\n  re-wrote {outp} with derived estimates "
          f"({m.shape[0]} rows x {m.shape[1]} cols)")

    print("\n" + "=" * 100)
    print("WHAT THIS MEASUREMENT CANNOT SEE (read before acting on it)")
    print("=" * 100)
    for line in (
        "1. NO DEPTH DATA.  data/tick_archive/ holds ONE day (2026-01-20), outside this",
        "   cohort entirely.  There is no L1/L2 book, no quoted spread, no queue depth for",
        "   ANY of these 220 trades.  Every number above is inferred from 1-minute OHLCV.",
        "2. QUEUE POSITION / PARTIAL FILLS are invisible.  A marketable order that only",
        "   partially fills at 09:20 leaves residual size to chase, which is strictly worse",
        "   than any of these estimates.",
        "3. THE EARNINGS-SHOCK BOOK.  A name down 8%+ on results at 09:20 plausibly quotes a",
        "   far wider spread than its own Roll estimate implies, because Roll is computed on",
        "   TRADED prices -- it only sees trades that happened, not the width nobody crossed.",
        "   Non-estimable Roll (27.7% of entry legs) is exactly the trending/informed tape",
        "   where this bias is worst, and it was filled with the tier median (optimistic).",
        "4. SHORT-SIDE BORROW/BAN.  Intraday shorting is fine on MIS, but an F&O-ban or a",
        "   circuit-locked down move makes the SHORT unfillable at any price.  Stage-4 flags",
        "   exist for circuit/locked but a real broker rejection is not modelled here.",
        ("5. MIS AUTO-SQUARE-OFF.  Zerodha squares MIS equity off from ~15:20, before this"
         if LOCKED["exit_fill_minute"] >= "15:20" else
         "5. MIS AUTO-SQUARE-OFF is RESPECTED by this run: the exit leg is measured at"),
        (f"   cohort's 15:25-15:30 exit bar.  The measured 'close' leg may not be the leg you"
         if LOCKED["exit_fill_minute"] >= "15:20" else
         f"   {LOCKED['exit_fill_minute']}, inside every relevant broker's pre-square-off window."),
        ("   actually get -- a 15:20 forced exit is a different (usually worse) print."
         if LOCKED["exit_fill_minute"] >= "15:20" else
         "   A broker-initiated square-off is still not modelled (we exit voluntarily first)."),
        "6. OUR OWN FOOTPRINT.  M1-M4 read a tape that did NOT contain our order.  Only M6",
        "   models impact, and it does so parametrically, not empirically.",
        "",
        "HONEST BOTTOM LINE: 1-minute OHLCV can bound the SPREAD to roughly the right order",
        "of magnitude and can size the IMPACT parametrically.  It cannot resolve the",
        "worst-case scenario -- a wide, thin, one-sided book on a post-earnings gap-down at",
        "09:20 -- because that information only exists in quote/depth data we do not have.",
        "If the CENTRAL estimate had landed near the 38bp falsifier this would be",
        "unresolvable without live/paper execution.  Where it actually lands relative to the",
        "bands is stated above; the margin to the band is the thing to judge, not the point",
        "estimate.",
    ):
        print("  " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
