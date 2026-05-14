"""C4a/C4b sanity — realistic-execution simulation of gap-down NIFTY-50 setups.

Setups (pre-registered 2026-05-14):
  C4a gap_down_reversal_long:
    gap_pct <= -0.5%  AND  09:20 close > 09:15 open  ->  LONG at 09:20 close
  C4b gap_down_continuation_short:
    gap_pct <= -0.5%  AND  09:20 close < 09:15 open  ->  SHORT at 09:20 close

Shared mechanics:
  Universe: NIFTY-50 constituents only.
  Hard SL: combined 09:15+09:20 bars' low (long) or high (short), +/- 0.10% buffer.
  T1: +0.5R, 50% qty partial exit, BE trail after.
  T2: +1.0R, full exit.
  Time stop: 13:00 IST.
  Fees: tools.sub7_validation.build_per_setup_pnl.calc_fee (Indian intraday MIS).

Regimes:
  pre_rule  = 2023-01-01 .. 2025-01-31  (before Feb-1-2025 option-premium-upfront)
  post_rule = 2025-02-01 .. 2026-04-30  (after rule effective)

War sub-split within post_rule:
  pre_war   = 2025-02-01 .. 2025-12-31
  war       = 2026-01-01 .. 2026-04-30  (Feb-28..Apr-8 active conflict)

Pre-registered pass criteria:
  post_rule:   PF >= 1.30, n >= 200, WR >= 50%
  pre_rule:    PF >= 1.10, n >= 500  (sanity that signal isn't ONLY post-rule artifact)
  delta_pf >= 0.15 (post - pre)

Usage:
    python -m tools.research.post_sebi.sanity_gap_down_intraday
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.sub7_validation.build_per_setup_pnl import calc_fee  # noqa: E402

_FEATHER_DIR = _REPO / "backtest-cache-download" / "monthly"
_NIFTY50_CSV = _REPO / "assets" / "ind_nifty50list.csv"
_OUT_DIR = _REPO / "reports" / "research" / "post_sebi" / "gap_down_intraday"
_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Pre-registered parameters (locked before run)
GAP_PCT_MAX = -0.005           # gap_pct must be <= -0.5%
SL_BUFFER_PCT = 0.001          # 10 bps beyond combined high/low
T1_R_MULTIPLE = 0.5            # T1 partial at +0.5R
T1_PARTIAL_PCT = 0.5           # 50% qty exit at T1
T2_R_MULTIPLE = 1.0            # T2 full at +1.0R
TIME_STOP_HHMM = "13:00"       # exit no later than 13:00 IST
RISK_PER_TRADE_RUPEES = 1000   # qty sized so |entry - SL| * qty = 1000

# Regime split
RULE_DATE = pd.Timestamp("2025-02-01").date()
WAR_START = pd.Timestamp("2026-01-01").date()


def load_nifty50() -> List[str]:
    df = pd.read_csv(_NIFTY50_CSV)
    return sorted(df["Symbol"].dropna().astype(str).str.strip().unique().tolist())


def load_all_5m(symbols: set) -> pd.DataFrame:
    """Load monthly 5m feathers, filtering to `symbols` per-file to avoid OOM."""
    feathers = sorted(_FEATHER_DIR.glob("20*_5m_enriched.feather"))
    print(f"  loading {len(feathers)} monthly 5m feathers (filtered to NIFTY-50) ...")
    parts = []
    for fp in feathers:
        try:
            df = pd.read_feather(fp)
            df = df[df["symbol"].isin(symbols)]
            if not df.empty:
                parts.append(df[["symbol", "date", "open", "high", "low", "close", "volume"]].copy())
        except Exception as e:
            print(f"  WARN: {fp.name}: {e}")
    big = pd.concat(parts, ignore_index=True)
    big["date"] = pd.to_datetime(big["date"])
    big = big.sort_values(["symbol", "date"]).reset_index(drop=True)
    big["d"] = big["date"].dt.date
    big["hhmm"] = big["date"].dt.strftime("%H:%M")
    print(f"  loaded {len(big):,} NIFTY-50 bars")
    return big


def build_pdc_map(big: pd.DataFrame) -> Dict[Tuple[str, date], float]:
    """Per-symbol PDC: last close of prior session_date."""
    print("  building PDC map ...")
    daily_close = (
        big.groupby(["symbol", "d"])
        .agg(last_close=("close", "last"))
        .reset_index()
    )
    daily_close["pdc"] = daily_close.groupby("symbol")["last_close"].shift(1)
    daily_close = daily_close.dropna(subset=["pdc"])
    return {
        (r["symbol"], r["d"]): float(r["pdc"])
        for _, r in daily_close.iterrows()
    }


def simulate(big: pd.DataFrame, pdc_map: Dict, symbols: set) -> pd.DataFrame:
    """Walk each (symbol, session) and emit one trade row per qualifying event."""
    trades: List[dict] = []
    sessions = big[big["symbol"].isin(symbols)].groupby(["symbol", "d"])
    print(f"  iterating {len(sessions):,} (symbol, session) groups ...")

    n_no_pdc = n_no_915 = n_no_920 = n_gap_fail = n_zero_R = 0
    n_long = n_short = 0

    for (sym, sd), grp in sessions:
        pdc = pdc_map.get((sym, sd))
        if pdc is None or pdc <= 0:
            n_no_pdc += 1
            continue
        grp = grp.sort_values("date")
        bar_915 = grp[grp["hhmm"] == "09:15"]
        bar_920 = grp[grp["hhmm"] == "09:20"]
        if bar_915.empty:
            n_no_915 += 1; continue
        if bar_920.empty:
            n_no_920 += 1; continue

        open_915 = float(bar_915.iloc[0]["open"])
        close_915 = float(bar_915.iloc[0]["close"])
        high_915 = float(bar_915.iloc[0]["high"])
        low_915 = float(bar_915.iloc[0]["low"])
        open_920 = float(bar_920.iloc[0]["open"])
        close_920 = float(bar_920.iloc[0]["close"])
        high_920 = float(bar_920.iloc[0]["high"])
        low_920 = float(bar_920.iloc[0]["low"])

        if open_915 <= 0:
            continue
        gap_pct = (open_915 - pdc) / pdc
        if gap_pct > GAP_PCT_MAX:
            n_gap_fail += 1; continue

        # Direction by first-bar
        bar1_up = close_920 > open_920
        bar1_dn = close_920 < open_920
        if not (bar1_up or bar1_dn):  # doji — skip
            continue

        if bar1_up:
            setup = "c4a_gap_down_reversal_long"
            direction = "long"
            entry_price = close_920
            combined_low = min(low_915, low_920)
            hard_sl = combined_low * (1 - SL_BUFFER_PCT)
            stop_distance = entry_price - hard_sl
        else:
            setup = "c4b_gap_down_continuation_short"
            direction = "short"
            entry_price = close_920
            combined_high = max(high_915, high_920)
            hard_sl = combined_high * (1 + SL_BUFFER_PCT)
            stop_distance = hard_sl - entry_price

        if stop_distance <= 0:
            n_zero_R += 1; continue
        R = stop_distance

        if direction == "long":
            t1 = entry_price + T1_R_MULTIPLE * R
            t2 = entry_price + T2_R_MULTIPLE * R
        else:
            t1 = entry_price - T1_R_MULTIPLE * R
            t2 = entry_price - T2_R_MULTIPLE * R

        qty = max(int(RISK_PER_TRADE_RUPEES / R), 1)
        qty_at_t1 = int(qty * T1_PARTIAL_PCT)
        qty_runner = qty - qty_at_t1

        # Walk forward from 09:25 to 13:00
        post = grp[grp["date"] > bar_920.iloc[0]["date"]].copy()
        if post.empty:
            continue

        t1_hit = False
        t1_exit_price = None
        t2_exit_price = None
        sl_exit_price = None
        time_exit_price = None
        exit_reasons = []

        for _, bar in post.iterrows():
            ts = bar["date"]
            high = float(bar["high"])
            low = float(bar["low"])

            # After T1 hit, stop trails to break-even (entry). Before T1, hard SL.
            effective_sl = entry_price if t1_hit else hard_sl

            if direction == "long":
                if low <= effective_sl:
                    sl_exit_price = effective_sl
                    exit_reasons.append("be_trail" if t1_hit else "sl")
                    break
                if not t1_hit and high >= t1:
                    t1_hit = True
                    t1_exit_price = t1
                    exit_reasons.append("t1_partial")
                if high >= t2:
                    t2_exit_price = t2
                    exit_reasons.append("t2_full")
                    break
                if ts.strftime("%H:%M") >= TIME_STOP_HHMM:
                    time_exit_price = float(bar["close"])
                    exit_reasons.append("time_stop")
                    break
            else:  # short
                if high >= effective_sl:
                    sl_exit_price = effective_sl
                    exit_reasons.append("be_trail" if t1_hit else "sl")
                    break
                if not t1_hit and low <= t1:
                    t1_hit = True
                    t1_exit_price = t1
                    exit_reasons.append("t1_partial")
                if low <= t2:
                    t2_exit_price = t2
                    exit_reasons.append("t2_full")
                    break
                if ts.strftime("%H:%M") >= TIME_STOP_HHMM:
                    time_exit_price = float(bar["close"])
                    exit_reasons.append("time_stop")
                    break

        # Compute PnL pieces
        pnl = 0.0
        # T1 leg
        if t1_hit and t1_exit_price is not None:
            if direction == "long":
                pnl += (t1_exit_price - entry_price) * qty_at_t1
            else:
                pnl += (entry_price - t1_exit_price) * qty_at_t1
        # Runner leg
        if t2_exit_price is not None:
            # T2 hit (post T1 or direct)
            runner_qty = qty_runner if t1_hit else qty  # if T1 not hit, runner is full
            if direction == "long":
                pnl += (t2_exit_price - entry_price) * runner_qty
            else:
                pnl += (entry_price - t2_exit_price) * runner_qty
        elif sl_exit_price is not None:
            runner_qty = qty_runner if t1_hit else qty
            if direction == "long":
                pnl += (sl_exit_price - entry_price) * runner_qty
            else:
                pnl += (entry_price - sl_exit_price) * runner_qty
        elif time_exit_price is not None:
            runner_qty = qty_runner if t1_hit else qty
            if direction == "long":
                pnl += (time_exit_price - entry_price) * runner_qty
            else:
                pnl += (entry_price - time_exit_price) * runner_qty
        else:
            # ran off the end of post; close at last bar
            last = post.iloc[-1]
            close_last = float(last["close"])
            runner_qty = qty_runner if t1_hit else qty
            if direction == "long":
                pnl += (close_last - entry_price) * runner_qty
            else:
                pnl += (entry_price - close_last) * runner_qty
            exit_reasons.append("last_bar")

        # Fee (simplified: one round-trip on average fill)
        avg_exit = (
            t2_exit_price or sl_exit_price or time_exit_price
            or (float(post.iloc[-1]["close"]) if not post.empty else entry_price)
        )
        leg = "BUY" if direction == "long" else "SELL"
        fee = calc_fee(entry_price, avg_exit, qty, leg)
        net_pnl = pnl - fee

        if direction == "long":
            n_long += 1
        else:
            n_short += 1

        trades.append({
            "setup": setup,
            "direction": direction,
            "symbol": sym,
            "session_date": sd,
            "pdc": pdc,
            "open_915": open_915,
            "close_920": close_920,
            "gap_pct": gap_pct,
            "entry_price": entry_price,
            "hard_sl": hard_sl,
            "t1_target": t1,
            "t2_target": t2,
            "stop_distance": stop_distance,
            "qty": qty,
            "t1_hit": t1_hit,
            "exit_reason": "+".join(exit_reasons) if exit_reasons else "none",
            "realized_pnl": pnl,
            "fee": fee,
            "net_pnl": net_pnl,
        })

    print(f"\n  qualified events:")
    print(f"    no_pdc={n_no_pdc}, no_09:15={n_no_915}, no_09:20={n_no_920}, "
          f"gap_fail={n_gap_fail}, zero_R={n_zero_R}")
    print(f"    LONG (C4a): {n_long:,}, SHORT (C4b): {n_short:,}")
    return pd.DataFrame(trades)


def metrics(df: pd.DataFrame) -> Dict:
    if df.empty:
        return dict(n=0, wr=0.0, pf=0.0, net=0.0, sharpe=0.0, gw=0.0, gl=0.0)
    wins = df[df["net_pnl"] > 0]
    losses = df[df["net_pnl"] <= 0]
    gw = float(wins["net_pnl"].sum())
    gl = float(-losses["net_pnl"].sum())
    pf = gw / gl if gl > 0 else float("inf")
    daily = df.groupby("session_date")["net_pnl"].sum()
    sharpe = float(daily.mean() / daily.std()) if daily.std() > 0 else 0.0
    return dict(
        n=len(df),
        wr=100.0 * len(wins) / len(df),
        pf=pf,
        net=float(df["net_pnl"].sum()),
        sharpe=sharpe,
        gw=gw, gl=gl,
    )


def split_and_report(trades: pd.DataFrame) -> None:
    trades = trades.copy()
    trades["session_date"] = pd.to_datetime(trades["session_date"]).dt.date
    trades["regime"] = trades["session_date"].apply(
        lambda d: "pre_rule" if d < RULE_DATE else "post_rule"
    )
    trades["sub_period"] = trades.apply(
        lambda r: (
            "pre_rule" if r["regime"] == "pre_rule"
            else ("war" if r["session_date"] >= WAR_START else "pre_war")
        ),
        axis=1,
    )

    for setup in ("c4a_gap_down_reversal_long", "c4b_gap_down_continuation_short"):
        print()
        print("=" * 82)
        print(setup)
        print("=" * 82)
        sub = trades[trades["setup"] == setup]
        if sub.empty:
            print("  NO TRADES")
            continue

        # By regime
        for label, period in [
            ("FULL (2023-01 to 2026-04)", sub),
            ("pre_rule (2023-01 to 2025-01)", sub[sub["regime"] == "pre_rule"]),
            ("post_rule (2025-02 to 2026-04)", sub[sub["regime"] == "post_rule"]),
            ("post_rule.pre_war (2025-02 to 2025-12)", sub[sub["sub_period"] == "pre_war"]),
            ("post_rule.war (2026-01 to 2026-04)", sub[sub["sub_period"] == "war"]),
        ]:
            m = metrics(period)
            pfd = f"{m['pf']:.3f}" if m["pf"] != float("inf") else "inf"
            print(f"  {label:<42s} n={m['n']:>4}  PF={pfd:>6}  "
                  f"WR={m['wr']:>5.1f}%  NET=Rs.{m['net']:>10,.0f}  "
                  f"Sharpe={m['sharpe']:>5.2f}")

        # Pre-registered falsifier
        m_pre = metrics(sub[sub["regime"] == "pre_rule"])
        m_post = metrics(sub[sub["regime"] == "post_rule"])
        delta = m_post["pf"] - m_pre["pf"]
        pass_post = (m_post["pf"] >= 1.30 and m_post["n"] >= 200 and m_post["wr"] >= 50.0)
        pass_pre = (m_pre["pf"] >= 1.10 and m_pre["n"] >= 500)
        pass_delta = delta >= 0.15

        print(f"\n  Pre-registered falsifier check:")
        print(f"    post >= 1.30 PF, n>=200, WR>=50%:  {'PASS' if pass_post else 'FAIL'}")
        print(f"    pre  >= 1.10 PF, n>=500:           {'PASS' if pass_pre else 'FAIL'}")
        print(f"    delta_pf >= 0.15:                  {'PASS' if pass_delta else 'FAIL'} "
              f"(actual: {delta:+.3f})")
        all_pass = pass_post and pass_pre and pass_delta
        print(f"    OVERALL VERDICT:                   {'>>> PASS <<<' if all_pass else 'FAIL'}")

        # Per-month breakdown
        print(f"\n  Per-month breakdown:")
        sub2 = sub.copy()
        sub2["month"] = pd.to_datetime(sub2["session_date"]).dt.strftime("%Y-%m")
        for mth, grp in sub2.groupby("month"):
            mm = metrics(grp)
            pfd = f"{mm['pf']:.2f}" if mm['pf'] != float("inf") else "inf"
            print(f"    {mth}  n={mm['n']:>4}  PF={pfd:>5}  "
                  f"WR={mm['wr']:>5.1f}%  NET=Rs.{mm['net']:>8,.0f}")

        # Exit-reason mix
        print(f"\n  Exit-reason mix (post_rule):")
        post = sub[sub["regime"] == "post_rule"]
        if not post.empty:
            for reason, grp in post.groupby("exit_reason"):
                avg = float(grp["net_pnl"].mean())
                print(f"    {reason:<30s} n={len(grp):>4}  avg_net=Rs.{avg:>7,.0f}")


def main():
    print(f"NIFTY-50 universe: {_NIFTY50_CSV.name}")
    symbols = set(load_nifty50())
    print(f"  {len(symbols)} symbols loaded")
    big = load_all_5m(symbols)
    pdc_map = build_pdc_map(big)
    trades = simulate(big, pdc_map, symbols)
    if trades.empty:
        print("\n[NO TRADES] something went wrong with simulation")
        sys.exit(1)
    out_path = _OUT_DIR / "gap_down_intraday_trades.parquet"
    trades.to_parquet(out_path, index=False)
    print(f"\n  trades saved: {out_path}")
    split_and_report(trades)


if __name__ == "__main__":
    main()
