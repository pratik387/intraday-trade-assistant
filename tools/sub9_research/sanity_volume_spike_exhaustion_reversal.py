"""Pre-coding sanity check for volume_spike_exhaustion_reversal candidate.

Per sub-9 §3.3 brief gate (specs/2026-05-06-sub-project-9-brief-volume_
spike_exhaustion_reversal.md): BEFORE writing detector code, simulate
the volume-spike + wick-rejection mean-reversion fade on 12 months of
2024 5m enriched feathers.

Decision criterion (from brief):
  PF >= 1.10  -> strong proceed
  1.0-1.10    -> marginal, proceed with caveat
  PF < 1.0    -> retire candidate, do NOT write detector

Long-side ship gate: LONG-side PF must be >= SHORT-side PF * 0.85.
Per task instruction, SHORT-only run first (LONG variant deferred).

Usage:
    python tools/sub9_research/sanity_volume_spike_exhaustion_reversal.py

Mechanic (per locked brief params):
  - Per 5m bar in [11:00, 15:05] IST:
      vol_mean_20 = rolling 20-bar mean of volume (excl current)
      vol_std_20  = rolling 20-bar stdev of volume
      vol_z = (volume - vol_mean_20) / vol_std_20
      bar_range = high - low
      upper_wick_frac = (high - close) / bar_range
  - SHORT trigger: vol_z >= 3.0 AND upper_wick_frac >= 0.6 AND red bar (close<open)
  - Reversal-confirmation bar (NEXT bar): close < open AND close < trigger close
  - Entry: confirmation bar's CLOSE (per brief Mechanic step 3)
  - Hard SL: trigger high + 0.3% buffer; min stop 0.6% of entry
  - T1 (50% qty): 1.0R move (R = stop_distance)
  - T2 (50% qty): 2.0R move OR bar-midpoint fallback if T2 not hit by 30 min
  - Time stop: 6 bars (~30 min) or 15:15 IST hard stop
  - Latch per (symbol, day, side)
  - Universe: F&O 200, mid_cap + small_cap

Note on "30 minutes" wording in user task: 30 minutes = 6 5m bars =
TIME_STOP_BARS exactly, so the brief's bar-midpoint fallback at T2
end-of-window is implemented as: if no T2 hit by the final bar, take
bar-midpoint exit at that bar.

Per task: SHORT-only run; LONG path is implemented but disabled to
preserve the explicit instruction "DEFER long for sanity, run SHORT-
only first". Set RUN_LONG_SIDE = True to also produce the long-side
diagnostic.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from services.symbol_metadata import get_cap_segment           # noqa: E402
from tools.sub7_validation.build_per_setup_pnl import calc_fee  # noqa: E402


# ---- Locked params (research-defensible per brief §3.3 / Mechanic) ----
ALLOWED_CAPS = {"mid_cap", "small_cap"}    # large_cap muted edge; micro_cap thin (brief Universe)
VOL_Z_THRESHOLD = 3.0                      # 3-sigma volume anomaly (brief Mechanic step 1)
VOL_WINDOW_BARS = 20                       # rolling 20-bar vol mean/stdev
WICK_FRAC_THRESHOLD = 0.6                  # 60% rejection wick (brief Mechanic step 1)
ENTRY_WINDOW_START_HHMM = "11:00"          # 11:00 IST — exclude opening institutional flow
ENTRY_WINDOW_END_HHMM = "15:05"            # 15:05 IST — exclude MIS auto-square contamination
TIME_STOP_BARS = 6                         # 6 bars (~30 min) per brief Mechanic step 5
TIME_STOP_HARD_HHMM = "15:15"              # absolute hard stop (5 min before MIS auto-square 15:20)
SL_BUFFER_PCT = 0.3                        # 0.3% buffer
MIN_STOP_PCT = 0.6                         # min stop 0.6% of entry
T1_R_MULTIPLE = 1.0                        # T1 = 1.0R move
T2_R_MULTIPLE = 2.0                        # T2 = 2.0R move
USE_BAR_MIDPOINT_FALLBACK = False          # User+research decision 2026-05-06: time-stop CLOSE,
                                           # NOT bar-midpoint. "Midpoint" has no published precedent;
                                           # time-stop close is the documented Indian-bracket-order
                                           # convention (Tradejini, Share India). See spec
                                           # 2026-05-06-sub-project-9-design-decisions-research.md Q4.
USE_BREAKEVEN_TRAIL_AFTER_T1 = True        # Q8 user decision: 50/50 tiered with breakeven trail
                                           # after T1 fill. After T1 hits, move SL on T2 leg to
                                           # breakeven (entry_price). Indian retail-pro convention
                                           # per Tradejini/Share India/Samco/Stratzy.
RISK_PER_TRADE_RUPEES = 1000               # match circuit_t1_fade_short sanity
MIN_ADV_INR_CR = 3.0                       # 20-day avg traded value >= Rs 3 Cr (brief Universe liquidity)

# Q3 user decision 2026-05-06: BIDIRECTIONAL from day 1 (overrides task's
# "SHORT-only first"). LONG side: vol_z >= 3.0 + lower_wick_frac >= 0.6 +
# bullish reversal next bar (capitulation bottom-fishing).
RUN_LONG_SIDE = True

# Q1 user decision: ENTRY at NEXT BAR'S OPEN (not confirmation-bar close).
# Streak/AlgoTest standard convention; eliminates look-ahead.
ENTRY_ON_NEXT_BAR_OPEN = True


def _load_5m_for_month(yyyy: int, mm: int) -> pd.DataFrame:
    path = _REPO_ROOT / "backtest-cache-download" / "monthly" / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_feather(path)


def build_full_year_5m() -> pd.DataFrame:
    print("  loading 24 monthly 5m feathers (2023-01 .. 2024-12) ...")
    parts: List[pd.DataFrame] = []
    for yyyy in (2023, 2024):
        for m in range(1, 13):
            mdf = _load_5m_for_month(yyyy, m)
            if not mdf.empty:
                parts.append(mdf)
    if not parts:
        return pd.DataFrame()
    big = pd.concat(parts, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    big["d"] = big["date"].dt.date
    print(f"  total 5m bars: {len(big):,}")
    return big


def load_fno_universe() -> set:
    path = _REPO_ROOT / "assets" / "fno_liquid_200.csv"
    df = pd.read_csv(path)
    syms = df["symbol"].astype(str).str.replace("NSE:", "", regex=False).tolist()
    print(f"  F&O 200 universe: {len(syms)} symbols")
    return set(syms)


def load_daily_for_liquidity() -> pd.DataFrame:
    daily_path = _REPO_ROOT / "cache" / "preaggregate" / "consolidated_daily.feather"
    if not daily_path.exists():
        raise FileNotFoundError(f"{daily_path} missing")
    df = pd.read_feather(daily_path)
    df["ts"] = pd.to_datetime(df["ts"])
    if df["ts"].dt.tz is not None:
        df["ts"] = df["ts"].dt.tz_localize(None)
    df["d"] = df["ts"].dt.date
    df = df[(df["d"] >= date(2022, 11, 1)) & (df["d"] <= date(2024, 12, 31))]
    df = df[["symbol", "d", "close", "volume"]].copy()
    df["traded_value"] = df["close"] * df["volume"]
    df = df.sort_values(["symbol", "d"])
    df["adv_20d_cr"] = df.groupby("symbol")["traded_value"].transform(
        lambda v: v.shift(1).rolling(20).mean()
    ) / 1e7
    return df[["symbol", "d", "adv_20d_cr"]]


def find_triggers(big5m: pd.DataFrame, universe: set, adv_table: pd.DataFrame) -> pd.DataFrame:
    """Compute vol_z + wick fracs per bar; identify trigger bars."""
    print("  filtering 5m bars to F&O 200 universe ...")
    df = big5m[big5m["symbol"].isin(universe)].copy()
    print(f"    universe-filtered 5m bars: {len(df):,}")

    df["nse_symbol"] = "NSE:" + df["symbol"].astype(str)
    df["cap_segment"] = df["nse_symbol"].apply(get_cap_segment)
    df = df[df["cap_segment"].isin(ALLOWED_CAPS)].copy()
    print(f"    cap_segment in {sorted(ALLOWED_CAPS)}: {len(df):,}")

    adv_idx = adv_table.set_index(["symbol", "d"])["adv_20d_cr"]
    df["adv_20d_cr"] = df.set_index(["symbol", "d"]).index.map(adv_idx).values
    df = df[df["adv_20d_cr"] >= MIN_ADV_INR_CR].copy()
    print(f"    adv_20d >= Rs {MIN_ADV_INR_CR}Cr: {len(df):,}")

    df = df.sort_values(["symbol", "d", "date"]).reset_index(drop=True)

    # rolling 20-bar vol mean/stdev EXCLUDING current bar (per brief)
    df["vol_mean_20"] = df.groupby(["symbol", "d"])["volume"].transform(
        lambda s: s.shift(1).rolling(VOL_WINDOW_BARS, min_periods=VOL_WINDOW_BARS).mean()
    )
    df["vol_std_20"] = df.groupby(["symbol", "d"])["volume"].transform(
        lambda s: s.shift(1).rolling(VOL_WINDOW_BARS, min_periods=VOL_WINDOW_BARS).std()
    )
    df["vol_z"] = (df["volume"] - df["vol_mean_20"]) / df["vol_std_20"]
    df = df.dropna(subset=["vol_std_20", "vol_z"])
    df = df[df["vol_std_20"] > 0]

    df["range"] = (df["high"] - df["low"]).clip(lower=1e-9)
    df["upper_wick_frac"] = (df["high"] - df["close"]) / df["range"]
    df["lower_wick_frac"] = (df["close"] - df["low"]) / df["range"]
    df["bar_midpoint"] = (df["high"] + df["low"]) / 2.0

    df["hhmm"] = df["date"].dt.strftime("%H:%M")
    df = df[(df["hhmm"] >= ENTRY_WINDOW_START_HHMM) & (df["hhmm"] <= ENTRY_WINDOW_END_HHMM)]

    # SHORT trigger
    short = df[
        (df["vol_z"] >= VOL_Z_THRESHOLD)
        & (df["upper_wick_frac"] >= WICK_FRAC_THRESHOLD)
        & (df["close"] < df["open"])
    ].copy()
    short["side"] = "SHORT"

    triggers = [short]
    if RUN_LONG_SIDE:
        long_ = df[
            (df["vol_z"] >= VOL_Z_THRESHOLD)
            & (df["lower_wick_frac"] >= WICK_FRAC_THRESHOLD)
            & (df["close"] > df["open"])
        ].copy()
        long_["side"] = "LONG"
        triggers.append(long_)
        print(f"    LONG triggers (vol_z>=3 + lower-wick>=0.6 + green): {len(long_):,}")
    print(f"    SHORT triggers (vol_z>=3 + upper-wick>=0.6 + red): {len(short):,}")

    return pd.concat(triggers, ignore_index=True)


def simulate(triggers: pd.DataFrame, big5m: pd.DataFrame) -> pd.DataFrame:
    print("  simulating entries -> targets/stop/time-stop ...")
    days_per_sym: Dict[str, pd.DataFrame] = {
        sym: g.sort_values("date").reset_index(drop=True)
        for sym, g in big5m.groupby("symbol")
    }

    # latch
    triggers = triggers.sort_values(["symbol", "d", "side", "date"])
    triggers = triggers.drop_duplicates(subset=["symbol", "d", "side"], keep="first")

    trades: List[dict] = []
    n_no_next = n_no_bars_after = n_traded = 0

    for _, t in triggers.iterrows():
        sym = t["symbol"]; sd = t["d"]; side = t["side"]
        trig_ts = t["date"]
        trig_high = float(t["high"]); trig_low = float(t["low"])
        trig_mid = float(t["bar_midpoint"])
        trig_close = float(t["close"])

        sym_df = days_per_sym.get(sym)
        if sym_df is None:
            n_no_next += 1; continue
        day_df = sym_df[sym_df["d"] == sd].reset_index(drop=True)
        if day_df.empty:
            n_no_next += 1; continue

        idx_arr = day_df.index[day_df["date"] == trig_ts].tolist()
        if not idx_arr:
            n_no_next += 1; continue
        trig_idx = idx_arr[0]
        if trig_idx + 1 >= len(day_df):
            n_no_next += 1; continue

        # confirmation bar = next 5m bar (used to verify reversal direction)
        conf = day_df.iloc[trig_idx + 1]
        c_open = float(conf["open"]); c_close = float(conf["close"])

        if side == "SHORT":
            ok = (c_close < c_open) and (c_close < trig_close)
        else:
            ok = (c_close > c_open) and (c_close > trig_close)
        if not ok:
            continue

        # Q1 user decision 2026-05-06: ENTRY at NEXT-BAR OPEN, not confirmation
        # bar's close. Streak/AlgoTest "Signal candle = Trade candle - 1" model
        # — confirmation completes at idx+1 close, trade enters at idx+2 open.
        if trig_idx + 2 >= len(day_df):
            n_no_next += 1; continue
        entry_bar = day_df.iloc[trig_idx + 2]
        entry_price = float(entry_bar["open"])
        entry_ts = entry_bar["date"]

        # hard SL
        if side == "SHORT":
            sl_struct = trig_high * (1.0 + SL_BUFFER_PCT / 100.0)
            sl_min = entry_price * (1.0 + MIN_STOP_PCT / 100.0)
            hard_sl = max(sl_struct, sl_min)
            stop_distance = hard_sl - entry_price
        else:
            sl_struct = trig_low * (1.0 - SL_BUFFER_PCT / 100.0)
            sl_min = entry_price * (1.0 - MIN_STOP_PCT / 100.0)
            hard_sl = min(sl_struct, sl_min)
            stop_distance = entry_price - hard_sl
        if stop_distance <= 0:
            continue

        # T1 / T2 = R-multiples (brief Mechanic step 5)
        if side == "SHORT":
            t1_target = entry_price - T1_R_MULTIPLE * stop_distance
            t2_target = entry_price - T2_R_MULTIPLE * stop_distance
        else:
            t1_target = entry_price + T1_R_MULTIPLE * stop_distance
            t2_target = entry_price + T2_R_MULTIPLE * stop_distance

        # Forward-exit window starts AFTER entry bar (entry is filled at entry_bar's
        # open, so use intrabar high/low of entry_bar + subsequent bars for hit
        # detection). Per Q1 user decision the forward starts at trig_idx+2.
        forward = day_df.iloc[trig_idx + 2: trig_idx + 2 + TIME_STOP_BARS].copy()
        if forward.empty:
            n_no_bars_after += 1; continue

        exit_ts = None; exit_price = None; exit_reason = None
        hit_t1 = False; t1_exit_price = None; t1_exit_ts = None

        for _, bar in forward.iterrows():
            ts = bar["date"]; high = float(bar["high"]); low = float(bar["low"])
            close = float(bar["close"])
            # Q8 breakeven trail: after T1 fills, SL on remaining T2 leg moves to
            # entry_price. Active SL = entry_price post-T1, hard_sl pre-T1.
            active_sl = (entry_price if (hit_t1 and USE_BREAKEVEN_TRAIL_AFTER_T1) else hard_sl)
            if side == "SHORT":
                if high >= active_sl:
                    exit_ts = ts; exit_price = active_sl
                    exit_reason = "breakeven_trail" if hit_t1 else "stop"
                    break
                if not hit_t1 and low <= t1_target:
                    hit_t1 = True; t1_exit_price = t1_target; t1_exit_ts = ts
                if hit_t1 and low <= t2_target:
                    exit_ts = ts; exit_price = t2_target; exit_reason = "t2"
                    break
            else:
                if low <= active_sl:
                    exit_ts = ts; exit_price = active_sl
                    exit_reason = "breakeven_trail" if hit_t1 else "stop"
                    break
                if not hit_t1 and high >= t1_target:
                    hit_t1 = True; t1_exit_price = t1_target; t1_exit_ts = ts
                if hit_t1 and high >= t2_target:
                    exit_ts = ts; exit_price = t2_target; exit_reason = "t2"
                    break

            if ts.strftime("%H:%M") >= TIME_STOP_HARD_HHMM:
                exit_ts = ts; exit_price = close; exit_reason = "time_stop_hard"
                break

        if exit_price is None:
            # Q4 user decision 2026-05-06: time-stop CLOSE (NOT bar-midpoint).
            # Bar-midpoint has no published precedent; time-stop close is the
            # documented Indian-bracket-order convention.
            last = forward.iloc[-1]
            exit_ts = last["date"]; exit_price = float(last["close"])
            exit_reason = "time_stop_bars"

        qty = max(int(RISK_PER_TRADE_RUPEES / max(stop_distance, 1e-6)), 1)
        if hit_t1:
            qty_t1 = qty // 2; qty_t2 = qty - qty_t1
            if side == "SHORT":
                pnl_t1 = (entry_price - t1_exit_price) * qty_t1
                pnl_t2 = (entry_price - exit_price) * qty_t2
            else:
                pnl_t1 = (t1_exit_price - entry_price) * qty_t1
                pnl_t2 = (exit_price - entry_price) * qty_t2
            realized_pnl = pnl_t1 + pnl_t2
            fee_t1 = calc_fee(entry_price, t1_exit_price, qty_t1, "SELL" if side == "SHORT" else "BUY")
            fee_t2 = calc_fee(entry_price, exit_price, qty_t2, "SELL" if side == "SHORT" else "BUY")
            fee = fee_t1 + fee_t2
            blended_exit = (t1_exit_price * qty_t1 + exit_price * qty_t2) / max(qty, 1)
        else:
            if side == "SHORT":
                realized_pnl = (entry_price - exit_price) * qty
            else:
                realized_pnl = (exit_price - entry_price) * qty
            fee = calc_fee(entry_price, exit_price, qty, "SELL" if side == "SHORT" else "BUY")
            blended_exit = exit_price

        net_pnl = realized_pnl - fee

        trades.append({
            "T1_entry_date": sd,
            "symbol": "NSE:" + sym,
            "cap_segment": t["cap_segment"],
            "side": side,
            "trigger_ts": trig_ts,
            "trigger_vol_z": float(t["vol_z"]),
            "trigger_upper_wick_frac": float(t["upper_wick_frac"]),
            "trigger_lower_wick_frac": float(t["lower_wick_frac"]),
            "trigger_bar_midpoint": trig_mid,
            "entry_ts": entry_ts,
            "entry_price": entry_price,
            "hard_sl": hard_sl,
            "t1_target": t1_target,
            "t2_target": t2_target,
            "hit_t1": hit_t1,
            "exit_ts": exit_ts,
            "exit_price": blended_exit,
            "exit_reason": exit_reason,
            "stop_distance": stop_distance,
            "qty": qty,
            "realized_pnl": realized_pnl,
            "fee": fee,
            "net_pnl": net_pnl,
        })
        n_traded += 1

    print(f"\n  no next bar:        {n_no_next}")
    print(f"  no bars after entry:{n_no_bars_after}")
    print(f"  traded:             {n_traded}")
    return pd.DataFrame(trades)


def report(trades: pd.DataFrame) -> None:
    if trades.empty:
        print("\n[NO TRADES] sanity check returns 0 trades")
        return
    n = len(trades)
    npnl = trades["net_pnl"]
    wins = npnl[npnl > 0].sum()
    losses = npnl[npnl < 0].abs().sum()
    pf = round(wins / losses, 3) if losses > 0 else float("inf")
    daily = trades.groupby("T1_entry_date")["net_pnl"].sum()
    sharpe = round(daily.mean() / daily.std(), 3) if daily.std() > 0 else 0.0
    wr = round(float((npnl > 0).mean()) * 100, 1)

    print("\n=== volume_spike_exhaustion_reversal -- pre-coding sanity check ===")
    print(f"Period: {trades['T1_entry_date'].min()} .. {trades['T1_entry_date'].max()}")
    print(f"Trades: n = {n}")
    print(f"Win rate: {wr}%")
    print(f"Gross PnL: Rs.{int(trades['realized_pnl'].sum()):,}")
    print(f"Fees:      Rs.{int(trades['fee'].sum()):,}")
    print(f"NET PnL:   Rs.{int(npnl.sum()):,}")
    print(f"NET PF:    {pf}")
    print(f"NET Sharpe (daily): {sharpe}")

    print("\nPer side:")
    short_pf = None; long_pf = None
    for sd, grp in trades.groupby("side"):
        n2 = len(grp)
        w = grp["net_pnl"][grp["net_pnl"] > 0].sum()
        l = grp["net_pnl"][grp["net_pnl"] < 0].abs().sum()
        pf2 = round(w / l, 3) if l > 0 else float("inf")
        wr2 = round(float((grp["net_pnl"] > 0).mean()) * 100, 1)
        net = int(grp["net_pnl"].sum())
        print(f"  {sd:<6} n={n2:>4} PF={pf2:>5} WR={wr2:>5}% netPnL=Rs.{net:>10,}")
        if sd == "SHORT": short_pf = pf2
        if sd == "LONG":  long_pf = pf2

    print("\nPer cap_segment:")
    for cap, grp in trades.groupby("cap_segment"):
        n2 = len(grp)
        w = grp["net_pnl"][grp["net_pnl"] > 0].sum()
        l = grp["net_pnl"][grp["net_pnl"] < 0].abs().sum()
        pf2 = round(w / l, 3) if l > 0 else float("inf")
        net = int(grp["net_pnl"].sum())
        print(f"  {cap:<12} n={n2:>4} PF={pf2:>5} netPnL=Rs.{net:>10,}")

    print("\nExit-reason breakdown:")
    for rsn, grp in trades.groupby("exit_reason"):
        print(f"  {rsn:<22} n={len(grp):>4} avg_net=Rs.{int(grp['net_pnl'].mean()):>6,}")

    print("\n--- VERDICT ---")
    if pf >= 1.10:
        print(f"PF={pf} >= 1.10 -> STRONG PROCEED. Move to detector implementation.")
    elif pf >= 1.00:
        print(f"PF={pf} in [1.00, 1.10) -> marginal. Proceed with caveat.")
    else:
        print(f"PF={pf} < 1.00 -> RETIRE candidate. Do not write detector code.")

    if short_pf is not None and long_pf is not None and short_pf > 0:
        ratio = round(long_pf / short_pf, 3) if isinstance(long_pf, (int, float)) else 0.0
        print(f"\nLong/Short PF ratio: {ratio} (gate: >= 0.85 for bidirectional ship)")
        if isinstance(long_pf, (int, float)) and long_pf >= short_pf * 0.85:
            print("  LONG-side passes ship gate.")
        else:
            print("  LONG-side FAILS ship gate -> SHORT-only ship.")
    else:
        print("\n(LONG side disabled per task instruction — re-run with RUN_LONG_SIDE=True for bidirectional diagnostic.)")


def main():
    big5m = build_full_year_5m()
    universe = load_fno_universe()
    adv_table = load_daily_for_liquidity()

    print("\nFinding triggers ...")
    triggers = find_triggers(big5m, universe, adv_table)
    print(f"\nTotal triggers (before latch): {len(triggers)}")
    if triggers.empty:
        return

    print("\nSimulating entries -> exits:")
    trades = simulate(triggers, big5m)
    report(trades)

    out = _REPO_ROOT / "reports" / "sub9_sanity" / "volume_spike_exhaustion_reversal_trades.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out, index=False)
    print(f"\nFull trade log: {out}")


if __name__ == "__main__":
    main()
