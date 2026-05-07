"""Pre-coding sanity check for sector_pair_convergence_intraday candidate.

Per sub-9 §3.3 brief gate (specs/2026-05-07-sub-project-9-brief-sector_
pair_convergence_intraday.md): BEFORE writing detector code, simulate the
2-leg intra-sector spread-convergence trade on 2 years (2023-01..2024-12)
of 5m enriched feathers.

Mechanic (per locked brief params §6):
  - Daily T-1 leader/laggard ranking per sector (trailing 20d cumret).
  - At 11:00 IST 5m bar close, per sector compute:
      leader_ret  = (leader_close_11:00 - leader_open_09:15) / leader_open_09:15
      laggard_ret = (laggard_close_11:00 - laggard_open_09:15) / laggard_open_09:15
      spread_bps  = (leader_ret - laggard_ret) * 10000
  - Trigger: spread_bps >= 80 (anti-noise floor; brief Mechanic step 3).
  - Confirmation gates: F&O membership for SHORT leg (borrow proxy).
  - Entry: simultaneously LONG laggard + SHORT leader at 11:05 5m bar OPEN
      (the next 5m bar after 11:00). The brief §6 step 5 says "11:00 close",
      but per the volume_spike sibling Q1 decision (no look-ahead) we enter
      at 11:05 open. This is the conservative Indian-bracket convention.
  - Stop: per-leg 0.6% adverse OR spread widens by >=150% of trigger spread.
  - T1 (50% qty both legs): spread converges to 0 (mid of historical p50).
  - T2 (50% qty both legs): spread overshoots to opposite-sign 50% of trigger.
  - Hard exit: 14:30 IST -- both legs squared regardless of P&L.
  - Latch: one fire per (sector, session, T+0).
  - 2-leg friction: calc_fee called TWICE -- once per leg round-trip.

Decision criterion (from brief §9):
  Phase 1: n >= 500 over 2 years -- ABORT-RETIRE if below.
  Phase 2 (only if Phase 1 passes):
    NET PF >= 1.10  -> strong proceed
    1.0-1.10        -> marginal
    NET PF < 1.0    -> retire

Period: 2023-01-01 .. 2024-12-31 (24 months).

Usage:
    python tools/sub9_research/sanity_sector_pair_convergence_intraday.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from services.symbol_metadata import get_cap_segment           # noqa: E402
from tools.sub7_validation.build_per_setup_pnl import calc_fee  # noqa: E402


# ---- Locked params (research-defensible per brief §6) ----
SPREAD_MIN_BPS = 80.0                       # anti-noise floor (brief Mechanic step 3)
LEADER_LOOKBACK_DAYS = 20                   # trailing 20d cumret for leader/laggard rank
SECTOR_TREND_BAND_PCT = 0.005               # |sector_5m_ret| <= 0.5% (brief gate 4)
SL_PCT_PER_LEG = 0.006                      # 0.6% per-leg hard stop
SPREAD_KILL_MULTIPLIER = 1.5                # spread widens by >=150% of trigger
EXCLUDED_SECTOR = "NSE_NIFTY_50"            # cross-sector heavyweights -- excluded per brief
ENTRY_BAR_HHMM = "11:00"                    # trigger evaluation bar
ENTRY_NEXT_BAR_HHMM = "11:05"               # entry executes at next bar's OPEN
HARD_EXIT_HHMM = "14:30"                    # non-negotiable hard square
NOTIONAL_PER_LEG_INR = 50000.0              # ₹50K per leg (brief §6 step 5)
N_FLOOR_PHASE1 = 500                        # abort threshold
PHASE1_PERIOD = (date(2023, 1, 1), date(2024, 12, 31))


_KEEP_COLS = ["date", "symbol", "open", "high", "low", "close", "volume"]


def _load_5m_for_month(yyyy: int, mm: int, symbols: Optional[set] = None) -> pd.DataFrame:
    path = _REPO_ROOT / "backtest-cache-download" / "monthly" / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_feather(path, columns=_KEEP_COLS)
    if symbols is not None:
        df = df[df["symbol"].isin(symbols)]
    # Downcast numerics to save memory
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], downcast="float")
    df["volume"] = pd.to_numeric(df["volume"], downcast="integer")
    return df


def build_full_period_5m(symbols: Optional[set] = None) -> pd.DataFrame:
    """Concatenate 24 monthly 5m feathers (2023-01..2024-12).
    Restrict to `symbols` set if provided (saves a lot of memory)."""
    print("  loading 24 monthly 5m feathers (2023-01 .. 2024-12) ...")
    parts: List[pd.DataFrame] = []
    for yyyy in (2023, 2024):
        for m in range(1, 13):
            mdf = _load_5m_for_month(yyyy, m, symbols=symbols)
            if not mdf.empty:
                parts.append(mdf)
    if not parts:
        return pd.DataFrame()
    big = pd.concat(parts, ignore_index=True)
    big = big.sort_values(["symbol", "date"]).reset_index(drop=True)
    big["d"] = big["date"].dt.date
    print(f"  total 5m bars: {len(big):,}")
    return big


def load_sector_map() -> Dict[str, str]:
    """Load NSE:SYMBOL -> NSE_NIFTY_<SECTOR> mapping; strip NSE: prefix."""
    path = _REPO_ROOT / "assets" / "stock_sector_map.json"
    raw = json.load(open(path))
    out = {}
    for k, v in raw.items():
        sym = k.replace("NSE:", "")
        out[sym] = v
    print(f"  sector map: {len(out)} symbols across {len(set(out.values()))} sectors")
    return out


def load_fno_universe() -> set:
    """F&O 200 universe -- proxy for SHORT borrow availability."""
    path = _REPO_ROOT / "assets" / "fno_liquid_200.csv"
    df = pd.read_csv(path)
    syms = df["symbol"].astype(str).str.replace("NSE:", "", regex=False).tolist()
    print(f"  F&O 200 universe: {len(syms)} symbols")
    return set(syms)


def build_daily_close_series(big5m: pd.DataFrame) -> pd.DataFrame:
    """For leader/laggard ranking: per (symbol, date), get day's close.
    Returns long-format frame: symbol, d, day_close.
    """
    print("  building daily close series for ranking ...")
    g = big5m.groupby(["symbol", "d"], as_index=False).agg(
        day_open=("open", "first"),
        day_close=("close", "last"),
    )
    return g


def rank_leader_laggard_per_sector(
    daily: pd.DataFrame,
    sector_map: Dict[str, str],
    fno_universe: set,
) -> pd.DataFrame:
    """For each (sector, T+0 date), rank constituents by trailing-20d cumret
    using T-1's close as the right boundary (no look-ahead).

    Returns: sector, d, leader_symbol, laggard_symbol.
    """
    print("  ranking leader/laggard per sector per day (trailing 20d cumret) ...")
    df = daily.copy()
    df["sector"] = df["symbol"].map(sector_map)
    df = df[df["sector"].notna() & (df["sector"] != EXCLUDED_SECTOR)].copy()
    df = df[df["symbol"].isin(fno_universe)].copy()  # F&O membership for borrow proxy

    df = df.sort_values(["symbol", "d"]).reset_index(drop=True)
    # 20d cumret based on day_close: ratio of today's close / close 20 trading days ago.
    df["close_lag20"] = df.groupby("symbol")["day_close"].shift(LEADER_LOOKBACK_DAYS)
    df["cumret_20d"] = df["day_close"] / df["close_lag20"] - 1.0
    df = df.dropna(subset=["cumret_20d"])

    # The ranking for T+0 must use T-1 cumret (no look-ahead). Shift cumret by 1 within symbol.
    df["cumret_20d_t1"] = df.groupby("symbol")["cumret_20d"].shift(1)
    df = df.dropna(subset=["cumret_20d_t1"])

    out_rows: List[dict] = []
    for (sector, d), grp in df.groupby(["sector", "d"]):
        if len(grp) < 3:
            continue  # need >= 3 constituents
        sub = grp.sort_values("cumret_20d_t1")
        laggard = sub.iloc[0]["symbol"]
        leader = sub.iloc[-1]["symbol"]
        out_rows.append({
            "sector": sector,
            "d": d,
            "leader_symbol": leader,
            "laggard_symbol": laggard,
            "leader_cumret_20d_t1": float(sub.iloc[-1]["cumret_20d_t1"]),
            "laggard_cumret_20d_t1": float(sub.iloc[0]["cumret_20d_t1"]),
            "n_constituents": len(grp),
        })
    rk = pd.DataFrame(out_rows)
    print(f"    leader/laggard pairs across all (sector, day): {len(rk):,}")
    print(f"    distinct sectors: {rk['sector'].nunique() if not rk.empty else 0}")
    return rk


def compute_intraday_returns_at_1100(big5m: pd.DataFrame) -> pd.DataFrame:
    """Per (symbol, date), compute intraday_ret = (close@11:00 - open@09:15) / open@09:15.
    Also return open@11:05 (entry price), 11:00 close, day_open for SL.
    Memory-efficient: computes hhmm only once, slices instead of sorting.
    """
    print("  computing intraday_ret @ 11:00 + 11:05 entry open per (symbol, day) ...")
    # big5m is already sorted by (symbol, date) in build_full_period_5m.
    hhmm = big5m["date"].dt.strftime("%H:%M")

    # day_open = first bar (09:15) per (symbol, d). big5m is sorted -> use 09:15 directly.
    mask_open = hhmm == "09:15"
    day_open_df = big5m.loc[mask_open, ["symbol", "d", "open"]].rename(columns={"open": "day_open"})

    mask_1100 = hhmm == ENTRY_BAR_HHMM
    bar_1100 = big5m.loc[mask_1100, ["symbol", "d", "date", "close"]].rename(
        columns={"close": "close_1100", "date": "ts_1100"}
    )

    mask_1105 = hhmm == ENTRY_NEXT_BAR_HHMM
    bar_1105 = big5m.loc[mask_1105, ["symbol", "d", "date", "open"]].rename(
        columns={"open": "entry_open_1105", "date": "ts_1105"}
    )

    merged = bar_1100.merge(day_open_df, on=["symbol", "d"], how="inner")
    merged = merged.merge(bar_1105, on=["symbol", "d"], how="inner")
    merged["intraday_ret"] = merged["close_1100"] / merged["day_open"] - 1.0
    print(f"    intraday rows (sym, day) with valid 09:15 + 11:00 + 11:05: {len(merged):,}")
    return merged


def evaluate_pairs_phase1(
    rk: pd.DataFrame,
    intra: pd.DataFrame,
    fno_universe: set,
) -> pd.DataFrame:
    """Phase 1 -- build candidate pair triggers per (sector, day).
    Counts only; NO P&L yet.

    Filters in order:
    A) leader+laggard both have valid 11:00 intraday_ret + 11:05 entry bar
    B) spread_bps >= SPREAD_MIN_BPS (anti-noise floor) -- BUT also handle direction:
       if intraday_ret(leader_T-1ranked) < intraday_ret(laggard_T-1ranked),
       i.e. ranked-leader is now lagging at 11:00, then by brief §5 the labels
       swap (always trade the spread-contraction direction).
    C) leader (the SHORT leg) must be in F&O 200.

    Returns enriched candidate frame (ready for Phase 2 if floor met).
    """
    print("  evaluating Phase-1 trigger gates ...")
    intra_idx = intra.set_index(["symbol", "d"])

    rows: List[dict] = []
    n_input = len(rk)
    n_missing_intra = 0
    n_below_floor = 0
    n_kept = 0

    for r in rk.itertuples(index=False):
        sector = r.sector
        d = r.d
        leader_t1 = r.leader_symbol
        laggard_t1 = r.laggard_symbol

        try:
            ld = intra_idx.loc[(leader_t1, d)]
            la = intra_idx.loc[(laggard_t1, d)]
        except KeyError:
            n_missing_intra += 1
            continue

        # If either symbol lacks an 11:00 or 11:05 bar (data hole), skip.
        # iloc/scalar access -- single-row Series since (symbol, d) unique.
        ld_ret = float(ld["intraday_ret"]) if pd.notna(ld["intraday_ret"]) else None
        la_ret = float(la["intraday_ret"]) if pd.notna(la["intraday_ret"]) else None
        if ld_ret is None or la_ret is None:
            n_missing_intra += 1
            continue

        # Direction: always trade spread-CONTRACTION. Identify actual leader/laggard
        # by 11:00 intraday returns (per brief §5 -- "labels swap" if rank inverted).
        if ld_ret >= la_ret:
            actual_leader = leader_t1
            actual_laggard = laggard_t1
            actual_leader_ret = ld_ret
            actual_laggard_ret = la_ret
            ld_row = ld
            la_row = la
        else:
            actual_leader = laggard_t1
            actual_laggard = leader_t1
            actual_leader_ret = la_ret
            actual_laggard_ret = ld_ret
            ld_row = la
            la_row = ld

        spread_bps = (actual_leader_ret - actual_laggard_ret) * 10000.0

        # Anti-noise floor.
        if spread_bps < SPREAD_MIN_BPS:
            n_below_floor += 1
            continue

        # Leader (SHORT leg) must be in F&O 200.
        if actual_leader not in fno_universe:
            continue

        rows.append({
            "sector": sector,
            "d": d,
            "leader_symbol": actual_leader,
            "laggard_symbol": actual_laggard,
            "spread_bps_at_entry": spread_bps,
            "leader_ret_1100": actual_leader_ret,
            "laggard_ret_1100": actual_laggard_ret,
            "leader_close_1100": float(ld_row["close_1100"]),
            "laggard_close_1100": float(la_row["close_1100"]),
            "leader_entry_open_1105": float(ld_row["entry_open_1105"]),
            "laggard_entry_open_1105": float(la_row["entry_open_1105"]),
            "leader_day_open": float(ld_row["day_open"]),
            "laggard_day_open": float(la_row["day_open"]),
        })
        n_kept += 1

    print(f"    input (sector, day) pairs:                {n_input:,}")
    print(f"    missing 11:00/11:05 intraday data:         {n_missing_intra:,}")
    print(f"    below {int(SPREAD_MIN_BPS)} bps spread floor:                {n_below_floor:,}")
    print(f"    kept candidate pair-events:               {n_kept:,}")
    return pd.DataFrame(rows)


def simulate_phase2(
    candidates: pd.DataFrame,
    big5m: pd.DataFrame,
) -> pd.DataFrame:
    """Phase 2: simulate 2-leg pair trades with calc_fee TWICE per fire."""
    print("  simulating 2-leg pair trades (LONG laggard + SHORT leader) ...")
    # Pre-group bars by (symbol, d) for fast forward-bar lookup
    days_per_sym: Dict[str, pd.DataFrame] = {
        sym: g.sort_values("date").reset_index(drop=True)
        for sym, g in big5m.groupby("symbol")
    }

    trades: List[dict] = []
    n_no_forward = 0

    for r in candidates.itertuples(index=False):
        sector = r.sector
        d = r.d
        leader = r.leader_symbol
        laggard = r.laggard_symbol
        spread_at_entry = float(r.spread_bps_at_entry)

        ld_df = days_per_sym.get(leader)
        la_df = days_per_sym.get(laggard)
        if ld_df is None or la_df is None:
            n_no_forward += 1
            continue
        ld_day = ld_df[ld_df["d"] == d].reset_index(drop=True)
        la_day = la_df[la_df["d"] == d].reset_index(drop=True)
        if ld_day.empty or la_day.empty:
            n_no_forward += 1
            continue

        # Locate the 11:05 entry bar index in each leg's intraday frame.
        ld_day["hhmm"] = ld_day["date"].dt.strftime("%H:%M")
        la_day["hhmm"] = la_day["date"].dt.strftime("%H:%M")
        ld_entry_idx = ld_day.index[ld_day["hhmm"] == ENTRY_NEXT_BAR_HHMM].tolist()
        la_entry_idx = la_day.index[la_day["hhmm"] == ENTRY_NEXT_BAR_HHMM].tolist()
        if not ld_entry_idx or not la_entry_idx:
            n_no_forward += 1
            continue

        leader_entry_price = float(r.leader_entry_open_1105)
        laggard_entry_price = float(r.laggard_entry_open_1105)

        # Position sizing: ₹50K notional per leg.
        leader_qty = max(int(NOTIONAL_PER_LEG_INR / max(leader_entry_price, 1e-6)), 1)
        laggard_qty = max(int(NOTIONAL_PER_LEG_INR / max(laggard_entry_price, 1e-6)), 1)

        # Per-leg 0.6% hard SL.
        leader_hard_sl = leader_entry_price * (1.0 + SL_PCT_PER_LEG)   # SHORT: stop above
        laggard_hard_sl = laggard_entry_price * (1.0 - SL_PCT_PER_LEG) # LONG: stop below

        # Spread-kill: spread widens by >=150% of trigger spread.
        spread_kill_bps = spread_at_entry * (1.0 + SPREAD_KILL_MULTIPLIER)

        # T1: spread converges to 0 bps. T2: spread overshoots to -spread_at_entry/2 (50% overshoot).
        t1_spread_bps = 0.0
        t2_spread_bps = -spread_at_entry * 0.5

        # Walk forward bar-by-bar. We need synchronized timestamps between the two legs.
        # Use leader's bars as the time base; align each bar to laggard by hhmm match
        # within the same day. Both feathers come from the same source so timestamps match.
        ld_fwd = ld_day.iloc[ld_entry_idx[0]:].reset_index(drop=True)
        la_fwd = la_day.iloc[la_entry_idx[0]:].reset_index(drop=True)
        # Inner-merge on date to get only common bars.
        merged = pd.merge(
            ld_fwd[["date", "hhmm", "open", "high", "low", "close"]].rename(
                columns={"open": "ld_open", "high": "ld_high", "low": "ld_low", "close": "ld_close"}
            ),
            la_fwd[["date", "open", "high", "low", "close"]].rename(
                columns={"open": "la_open", "high": "la_high", "low": "la_low", "close": "la_close"}
            ),
            on="date", how="inner",
        )
        if merged.empty:
            n_no_forward += 1
            continue

        exit_ts: Optional[pd.Timestamp] = None
        leader_exit_price: Optional[float] = None
        laggard_exit_price: Optional[float] = None
        exit_reason = "hard_time"
        hit_t1 = False
        t1_leader_exit: Optional[float] = None
        t1_laggard_exit: Optional[float] = None

        leader_day_open_for_spread = float(r.leader_day_open)
        laggard_day_open_for_spread = float(r.laggard_day_open)

        for i, bar in merged.iterrows():
            ts = bar["date"]
            hhmm = bar["hhmm"]
            ld_high = float(bar["ld_high"]); ld_low = float(bar["ld_low"]); ld_close = float(bar["ld_close"])
            la_high = float(bar["la_high"]); la_low = float(bar["la_low"]); la_close = float(bar["la_close"])

            # PER-LEG hard SL check (worst-case: high triggers SHORT stop, low triggers LONG stop).
            if ld_high >= leader_hard_sl or la_low <= laggard_hard_sl:
                exit_ts = ts
                # Conservative fill at the SL trigger price for the broken leg,
                # the other leg exits at this bar's close.
                if ld_high >= leader_hard_sl and la_low <= laggard_hard_sl:
                    leader_exit_price = leader_hard_sl
                    laggard_exit_price = laggard_hard_sl
                    exit_reason = "stop_both_legs"
                elif ld_high >= leader_hard_sl:
                    leader_exit_price = leader_hard_sl
                    laggard_exit_price = la_close
                    exit_reason = "stop_leader_short"
                else:
                    laggard_exit_price = laggard_hard_sl
                    leader_exit_price = ld_close
                    exit_reason = "stop_laggard_long"
                break

            # Spread-state evaluation against bar close.
            ld_ret_now = ld_close / leader_day_open_for_spread - 1.0
            la_ret_now = la_close / laggard_day_open_for_spread - 1.0
            spread_bps_now = (ld_ret_now - la_ret_now) * 10000.0

            # Spread-kill: spread widens beyond kill threshold.
            if spread_bps_now >= spread_kill_bps:
                exit_ts = ts
                leader_exit_price = ld_close
                laggard_exit_price = la_close
                exit_reason = "spread_kill"
                break

            # T1 hit: spread converges to <= 0 bps (50% qty exit).
            if not hit_t1 and spread_bps_now <= t1_spread_bps:
                hit_t1 = True
                t1_leader_exit = ld_close
                t1_laggard_exit = la_close

            # T2 hit (only after T1): spread overshoots to <= t2_spread_bps.
            if hit_t1 and spread_bps_now <= t2_spread_bps:
                exit_ts = ts
                leader_exit_price = ld_close
                laggard_exit_price = la_close
                exit_reason = "t2"
                break

            # Hard 14:30 time exit.
            if hhmm >= HARD_EXIT_HHMM:
                exit_ts = ts
                leader_exit_price = ld_close
                laggard_exit_price = la_close
                exit_reason = "t1_hard_time" if hit_t1 else "hard_time"
                break

        if leader_exit_price is None or laggard_exit_price is None:
            # Ran past last bar without exit -- fill at last bar's close.
            last = merged.iloc[-1]
            exit_ts = last["date"]
            leader_exit_price = float(last["ld_close"])
            laggard_exit_price = float(last["la_close"])
            exit_reason = "last_bar"

        # 50/50 tiered exit math: if T1 hit, half the qty exits at T1 prices,
        # other half at final exit prices. Otherwise full qty at final prices.
        if hit_t1 and t1_leader_exit is not None and t1_laggard_exit is not None:
            ld_qty1 = leader_qty // 2
            ld_qty2 = leader_qty - ld_qty1
            la_qty1 = laggard_qty // 2
            la_qty2 = laggard_qty - la_qty1
            # SHORT leader: pnl = (entry - exit) * qty
            ld_pnl = (leader_entry_price - t1_leader_exit) * ld_qty1 + \
                     (leader_entry_price - leader_exit_price) * ld_qty2
            # LONG laggard: pnl = (exit - entry) * qty
            la_pnl = (t1_laggard_exit - laggard_entry_price) * la_qty1 + \
                     (laggard_exit_price - laggard_entry_price) * la_qty2
            ld_fee = (
                calc_fee(leader_entry_price, t1_leader_exit, ld_qty1, "SELL")
                + calc_fee(leader_entry_price, leader_exit_price, ld_qty2, "SELL")
            )
            la_fee = (
                calc_fee(laggard_entry_price, t1_laggard_exit, la_qty1, "BUY")
                + calc_fee(laggard_entry_price, laggard_exit_price, la_qty2, "BUY")
            )
            blended_leader_exit = (
                t1_leader_exit * ld_qty1 + leader_exit_price * ld_qty2
            ) / max(leader_qty, 1)
            blended_laggard_exit = (
                t1_laggard_exit * la_qty1 + laggard_exit_price * la_qty2
            ) / max(laggard_qty, 1)
        else:
            ld_pnl = (leader_entry_price - leader_exit_price) * leader_qty
            la_pnl = (laggard_exit_price - laggard_entry_price) * laggard_qty
            ld_fee = calc_fee(leader_entry_price, leader_exit_price, leader_qty, "SELL")
            la_fee = calc_fee(laggard_entry_price, laggard_exit_price, laggard_qty, "BUY")
            blended_leader_exit = leader_exit_price
            blended_laggard_exit = laggard_exit_price

        # 2-leg friction: calc_fee called TWICE (leader-SHORT round-trip + laggard-LONG round-trip).
        fee_total = ld_fee + la_fee
        realized_pnl = ld_pnl + la_pnl
        net_pnl = realized_pnl - fee_total

        # Final spread (using bar at exit close).
        ld_ret_exit = blended_leader_exit / leader_day_open_for_spread - 1.0
        la_ret_exit = blended_laggard_exit / laggard_day_open_for_spread - 1.0
        spread_bps_at_exit = (ld_ret_exit - la_ret_exit) * 10000.0

        nse_leader = "NSE:" + leader
        nse_laggard = "NSE:" + laggard
        leader_cap = get_cap_segment(nse_leader)
        laggard_cap = get_cap_segment(nse_laggard)

        trades.append({
            "T0_session_date": d,
            "sector": sector,
            "leader_symbol": nse_leader,        # SHORT leg
            "laggard_symbol": nse_laggard,      # LONG leg
            "leader_cap": leader_cap,
            "laggard_cap": laggard_cap,
            "spread_at_entry_bps": spread_at_entry,
            "spread_at_exit_bps": spread_bps_at_exit,
            "leader_entry_price": leader_entry_price,
            "leader_exit_price": blended_leader_exit,
            "leader_qty": leader_qty,
            "leader_realized_pnl": ld_pnl,
            "leader_fee": ld_fee,
            "laggard_entry_price": laggard_entry_price,
            "laggard_exit_price": blended_laggard_exit,
            "laggard_qty": laggard_qty,
            "laggard_realized_pnl": la_pnl,
            "laggard_fee": la_fee,
            "fee_total": fee_total,             # 2-leg friction (both calc_fee calls)
            "realized_pnl": realized_pnl,
            "net_pnl": net_pnl,
            "exit_ts": exit_ts,
            "exit_reason": exit_reason,
            "hit_t1": hit_t1,
        })

    print(f"\n  no_forward (data hole):     {n_no_forward}")
    print(f"  traded:                     {len(trades)}")
    return pd.DataFrame(trades)


def report(trades: pd.DataFrame) -> None:
    if trades.empty:
        print("\n[NO TRADES] Phase 2 returns 0 rows -- abort downstream analysis")
        return
    n = len(trades)
    npnl = trades["net_pnl"]
    wins = npnl[npnl > 0].sum()
    losses = npnl[npnl < 0].abs().sum()
    pf = round(wins / losses, 3) if losses > 0 else float("inf")
    daily = trades.groupby("T0_session_date")["net_pnl"].sum()
    sharpe = round(daily.mean() / daily.std(), 3) if daily.std() > 0 else 0.0
    wr = round(float((npnl > 0).mean()) * 100.0, 1)

    print("\n=== sector_pair_convergence_intraday -- pre-coding sanity check ===")
    print(f"Period: {trades['T0_session_date'].min()} .. {trades['T0_session_date'].max()}")
    print(f"Trades (pair-events): n = {n}")
    print(f"Win rate:           {wr}%")
    print(f"Gross PnL:          Rs.{int(trades['realized_pnl'].sum()):,}")
    print(f"Fees (2-leg):       Rs.{int(trades['fee_total'].sum()):,}")
    print(f"NET PnL:            Rs.{int(npnl.sum()):,}")
    print(f"NET PF:             {pf}")
    print(f"NET Sharpe (daily): {sharpe}")

    # Per-leg PF (LONG laggard vs SHORT leader)
    leader_pnl = trades["leader_realized_pnl"] - trades["leader_fee"]
    laggard_pnl = trades["laggard_realized_pnl"] - trades["laggard_fee"]
    def _pf(s):
        w = s[s > 0].sum(); l = s[s < 0].abs().sum()
        return round(w / l, 3) if l > 0 else float("inf")
    short_pf = _pf(leader_pnl)
    long_pf = _pf(laggard_pnl)
    short_wr = round(float((leader_pnl > 0).mean()) * 100.0, 1)
    long_wr = round(float((laggard_pnl > 0).mean()) * 100.0, 1)
    print(f"\nPer-leg standalone PF (informational):")
    print(f"  SHORT leader leg: PF={short_pf} WR={short_wr}% net=Rs.{int(leader_pnl.sum()):,}")
    print(f"  LONG laggard leg: PF={long_pf}  WR={long_wr}% net=Rs.{int(laggard_pnl.sum()):,}")
    if isinstance(long_pf, (int, float)) and isinstance(short_pf, (int, float)) and short_pf > 0:
        wr_delta = abs(long_wr - short_wr)
        print(f"  |WR delta|: {round(wr_delta, 1)} pp (gate: <=10pp per brief §9 risk #3)")

    print("\nPer sector:")
    for sec, grp in trades.groupby("sector"):
        n2 = len(grp)
        w = grp["net_pnl"][grp["net_pnl"] > 0].sum()
        l = grp["net_pnl"][grp["net_pnl"] < 0].abs().sum()
        pf2 = round(w / l, 3) if l > 0 else float("inf")
        net = int(grp["net_pnl"].sum())
        share = round(100.0 * n2 / n, 1)
        print(f"  {sec:<22} n={n2:>4} ({share:>5}%) PF={pf2:>6} netPnL=Rs.{net:>10,}")

    print("\nPer month:")
    trades["yyyymm"] = pd.to_datetime(trades["T0_session_date"]).dt.strftime("%Y-%m")
    for ym, grp in trades.groupby("yyyymm"):
        n2 = len(grp)
        w = grp["net_pnl"][grp["net_pnl"] > 0].sum()
        l = grp["net_pnl"][grp["net_pnl"] < 0].abs().sum()
        pf2 = round(w / l, 3) if l > 0 else float("inf")
        print(f"  {ym}  n={n2:>3} PF={pf2:>6}")

    print("\nExit-reason breakdown:")
    for rsn, grp in trades.groupby("exit_reason"):
        n2 = len(grp)
        avg = int(grp["net_pnl"].mean())
        print(f"  {rsn:<22} n={n2:>4} avg_net=Rs.{avg:>6,}")

    # Friction-vs-spread analysis
    avg_fee = trades["fee_total"].mean()
    avg_gross = trades["realized_pnl"].mean()
    avg_spread = trades["spread_at_entry_bps"].mean()
    avg_notional = (trades["leader_entry_price"] * trades["leader_qty"]).mean() + \
                   (trades["laggard_entry_price"] * trades["laggard_qty"]).mean()
    fee_bps = (avg_fee / max(avg_notional, 1e-6)) * 10000.0
    print(f"\nFriction-vs-spread:")
    print(f"  avg trigger spread:  {round(avg_spread, 1)} bps")
    print(f"  avg gross PnL:       Rs.{int(avg_gross):,}")
    print(f"  avg fee (2-leg):     Rs.{int(avg_fee):,}  (~{round(fee_bps, 1)} bps of total notional)")
    print(f"  avg notional/trade:  Rs.{int(avg_notional):,}")

    # Symbol-overlap check (informational): trades hit which leader+laggard symbols?
    print(f"\nSymbol diversity:")
    print(f"  distinct leader symbols (SHORT leg): {trades['leader_symbol'].nunique()}")
    print(f"  distinct laggard symbols (LONG leg): {trades['laggard_symbol'].nunique()}")
    print(f"  distinct unique symbols (any leg):   {pd.concat([trades['leader_symbol'], trades['laggard_symbol']]).nunique()}")

    print("\n--- VERDICT ---")
    if pf >= 1.10:
        print(f"NET PF={pf} >= 1.10 -> STRONG PROCEED. Sanity passed.")
        print("  Caveat: 2-leg order-router coupling work is non-trivial -- factor into APPROVE decision.")
    elif pf >= 1.00:
        print(f"NET PF={pf} in [1.00, 1.10) -> MARGINAL. Brief retire-or-proceed user call.")
    else:
        print(f"NET PF={pf} < 1.00 -> THESIS RETIRE. Edge does not survive 2-leg friction.")


def main():
    print("=== Loading data sources ===")
    sector_map = load_sector_map()
    fno_universe = load_fno_universe()
    # Universe = sector-mapped symbols (excl NSE_NIFTY_50 fallback) intersect F&O 200.
    # Restricting load to this set (~140 symbols) keeps memory in budget.
    eligible_syms = {sym for sym, sec in sector_map.items()
                     if sec != EXCLUDED_SECTOR and sym in fno_universe}
    print(f"  eligible universe (mapped & F&O & not NSE_NIFTY_50): {len(eligible_syms)} symbols")
    big5m = build_full_period_5m(symbols=eligible_syms)
    if big5m.empty:
        print("ABORT: no 5m feathers loaded.")
        return

    # Restrict to sanity period (defensive; feathers should already be in range).
    big5m = big5m[(big5m["d"] >= PHASE1_PERIOD[0]) & (big5m["d"] <= PHASE1_PERIOD[1])].copy()
    print(f"  bars after period filter: {len(big5m):,}")

    print("\n=== Daily ranking ===")
    daily = build_daily_close_series(big5m)
    rk = rank_leader_laggard_per_sector(daily, sector_map, fno_universe)
    if rk.empty:
        print("ABORT: no leader/laggard rankings produced.")
        return

    print("\n=== Phase 1 -- counts only ===")
    intra = compute_intraday_returns_at_1100(big5m)
    candidates = evaluate_pairs_phase1(rk, intra, fno_universe)

    n_phase1 = len(candidates)
    print(f"\nPHASE 1 final pair-events: {n_phase1}")
    if n_phase1 < N_FLOOR_PHASE1:
        print(f"\n--- VERDICT (Phase 1) ---")
        print(f"n={n_phase1} < {N_FLOOR_PHASE1} floor -> ABORT-RETIRE per brief §9 risk #1.")
        print("  Do NOT run Phase 2. Brief retires for insufficient signal frequency.")
        out_dir = _REPO_ROOT / "reports" / "sub9_sanity"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Persist phase-1 candidates so user can inspect why floor missed.
        candidates.to_csv(out_dir / "sector_pair_convergence_intraday_phase1_candidates.csv", index=False)
        return
    print(f"PHASE 1 PASS (n={n_phase1} >= {N_FLOOR_PHASE1}) -> proceed to Phase 2.")

    print("\n=== Phase 2 -- 2-leg P&L simulation ===")
    trades = simulate_phase2(candidates, big5m)
    report(trades)

    out = _REPO_ROOT / "reports" / "sub9_sanity" / "sector_pair_convergence_intraday_trades.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out, index=False)
    print(f"\nFull trade log: {out}")


if __name__ == "__main__":
    main()
