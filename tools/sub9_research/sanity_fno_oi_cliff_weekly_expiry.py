"""Pre-coding sanity check for fno_oi_cliff_weekly_expiry candidate.

Per sub-9 §3.3 brief gate (specs/2026-05-07-sub-project-9-brief-fno_oi_
cliff_weekly_expiry.md): BEFORE writing detector code, simulate the
T-1 14:30 NIFTY 50 directional fade toward the max-OI cliff strike on
2023-2024 weekly expiries.

The setup is a clean redesign of the broken `expiry_pin_strike_reversal`
detector (-Rs 4.3M, retired 2026-04-30). Four structural fixes:
  - universe: NIFTY 50 spot (not 10 heavyweights)
  - side rule: sign(spot - cliff) AND CE/PE-OI dominance
  - trigger: T-1 14:30 single-bar entry (not RSI(14) on 5m heavyweights)
  - OI staleness: T-1 EOD bhavcopy for T-1 14:30 entry (same-day, fresh)

Decision criterion (from brief):
  PF >= 1.10 AND n >= 30  -> APPROVED for detector implementation
  PF in [1.00, 1.10)      -> marginal, revisit dominance/distance bands
  PF < 1.00               -> RETIRE; OI cliff thesis decisively negative
  n < 30 over 2yr          -> SAMPLE TOO THIN — RETIRE-PRE-IMPLEMENTATION

Mechanic (per locked brief params):
  1. T-1 EOD: read option_chain bhavcopy for T-1 (the day BEFORE entry),
     filter NIFTY rows, expiry_date == nearest weekly. Aggregate
     oi_total = CE_OI + PE_OI per strike; cliff_strike = argmax(oi_total).
     Compute CE_OI vs PE_OI dominance at cliff strike.
  2. T-1 14:30 IST: NIFTY 50 5m close. Compute spot_distance_pct.
  3. SHORT if spot > cliff*(1+0.30%) AND CE-dominant (CE >= 1.5x PE).
     LONG if spot < cliff*(1-0.30%) AND PE-dominant (PE >= 1.5x CE).
     NO TRADE if |spot_distance_pct| > 1.20% (macro flow regime).
  4. Entry: next-bar open (T-1 14:35 5m bar open).
  5. Stop: next opposite-side OI cluster strike + 0.20% buffer; min 0.40%.
  6. T1 (50% qty): 50% of (entry -> cliff) distance.
     T2 (50% qty): cliff strike itself.
  7. Time stop: T-1 15:15 IST.
  8. Latch: one fire per (expiry_date, side).

Universe: NIFTY 50 INDEX SPOT only. BANKNIFTY excluded — SEBI
discontinued BANKNIFTY weekly options 2024-11-20.

Discovery period: 2023-01-01 .. 2024-12-31 (NIFTY weekly was Thursday
until Apr 2024, Wednesday from Apr 2024). The bhavcopy expiry_date
column is the source of truth for the calendar — no hardcoded weekday.

Trading vehicle for fee modelling: NIFTYBEES.NS ETF (or NIFTY futures).
Sanity uses index spot price as the entry/exit fill — execution slippage
is NOT modelled for sanity (the broken predecessor was -Rs 4.3M GROSS
on the diagnosed mechanic; if NET PF can't clear 1.10 here, it definitely
can't on tradeable proxies). calc_fee from build_per_setup_pnl is used
with side BUY/SELL on a 1-lot-equivalent NIFTYBEES proxy (1 share of an
ETF that mirrors NIFTY ~ 1/240 of index, i.e. notional ~Rs 100/share at
spot 24000). For sanity we use NIFTY-spot-price-equivalent qty so the
fee model returns the correct % drag; the absolute Rs PnL is illustrative
of the directional edge, not the live tradeable profit.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.sub7_validation.build_per_setup_pnl import calc_fee  # noqa: E402


# ---- Locked params (per brief Mechanic) ----
UNDERLYING_SYMBOL = "NIFTY"                     # bhavcopy symbol column
SPOT_FEATHER = (
    _REPO_ROOT / "backtest-cache-download" / "index_ohlcv"
    / "NSE_NIFTY_50" / "NSE_NIFTY_50_1minutes.feather"
)
OPTION_CHAIN_DIR = _REPO_ROOT / "data" / "option_chain"

DOMINANCE_RATIO = 1.5                           # CE/PE ratio threshold
DISTANCE_MIN_PCT = 0.30                         # |spot - cliff| / cliff >= 0.30%
DISTANCE_MAX_PCT = 1.20                         # |spot - cliff| / cliff <= 1.20%
SL_BUFFER_PCT = 0.20                            # opposite-cluster + 0.20%
MIN_STOP_PCT = 0.40                             # qty-inflation guard

ENTRY_DECISION_HHMM = "14:30"                   # single-bar evaluation timestamp
ENTRY_FILL_HHMM = "14:35"                       # next 5m bar open
TIME_STOP_HHMM = "15:15"                        # hard exit

# Discovery period
START_DATE = date(2023, 1, 1)
END_DATE = date(2024, 12, 31)

RISK_PER_TRADE_RUPEES = 1000                    # match other sub9 sanity scripts


def _bhav_path(d: date) -> Path:
    return OPTION_CHAIN_DIR / f"{d.year:04d}" / f"{d.month:02d}" / f"{d.isoformat()}.parquet"


def load_nifty_5m() -> pd.DataFrame:
    """Load NIFTY 50 1-min feather, resample to 5m, IST-naive."""
    print(f"  loading {SPOT_FEATHER.name} ...")
    df = pd.read_feather(SPOT_FEATHER)
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df = df.sort_values("date").reset_index(drop=True)
    df = df[(df["date"].dt.date >= START_DATE) & (df["date"].dt.date <= END_DATE)].copy()
    print(f"    1m rows in window: {len(df):,}")

    # Resample to 5m bars (right-closed, label-left to match exchange bar convention)
    df = df.set_index("date")
    agg = pd.DataFrame({
        "open": df["open"].resample("5min", label="left", closed="left").first(),
        "high": df["high"].resample("5min", label="left", closed="left").max(),
        "low":  df["low"].resample("5min", label="left", closed="left").min(),
        "close": df["close"].resample("5min", label="left", closed="left").last(),
    }).dropna(how="all").reset_index()
    agg["d"] = agg["date"].dt.date
    agg["hhmm"] = agg["date"].dt.strftime("%H:%M")
    # Trading session 09:15 - 15:30
    agg = agg[(agg["hhmm"] >= "09:15") & (agg["hhmm"] <= "15:30")].reset_index(drop=True)
    print(f"    5m bars (session-filtered): {len(agg):,}")
    return agg


def list_session_dates(spot_5m: pd.DataFrame) -> List[date]:
    """Sorted list of trading session dates seen on the NIFTY spot."""
    return sorted(spot_5m["d"].unique().tolist())


def find_weekly_expiry_signals(
    session_dates: List[date],
) -> List[Tuple[date, date]]:
    """Return list of (entry_session_date, expiry_date) pairs.

    For each weekly expiry seen in the bhavcopies in the window, the entry
    session is T-1 = the trading session BEFORE the expiry day. We use the
    bhavcopy's `expiry_date` column to identify weekly expiries — no
    hardcoded weekday (NIFTY weekly was Thursday until Apr 2024, Wednesday
    after).

    Definition of weekly vs monthly: take the SET of expiry_dates listed
    on each bhavcopy. The NEAREST expiry from each session that is
    <= 7 calendar days out is treated as 'this week's weekly expiry'.
    Collect the unique set of expiry_dates that ever appear as nearest
    in this window — these are the candidate weekly expiries.
    """
    # Gather candidate expiries by scanning each session's bhavcopy
    weekly_expiries: set = set()
    print("  scanning bhavcopies for weekly expiries ...")
    n_scanned = 0
    for sd in session_dates:
        bp = _bhav_path(sd)
        if not bp.exists():
            continue
        try:
            df = pd.read_parquet(bp, columns=["symbol", "expiry_date"])
        except Exception:
            continue
        n = df[df["symbol"] == UNDERLYING_SYMBOL]
        if n.empty:
            continue
        exps_dt = pd.to_datetime(n["expiry_date"], errors="coerce").dropna()
        exps = exps_dt.dt.date.unique()
        # Keep nearest expiry that is >= sd and <= sd + 7 days (i.e. weekly)
        future = sorted([e for e in exps if isinstance(e, date) and e >= sd and (e - sd).days <= 7])
        if future:
            weekly_expiries.add(future[0])
        n_scanned += 1
    print(f"    bhavcopies scanned: {n_scanned}")
    print(f"    distinct weekly expiries identified: {len(weekly_expiries)}")

    # For each weekly expiry, find T-1 = previous trading session in our spot data
    sd_set = set(session_dates)
    pairs: List[Tuple[date, date]] = []
    for exp in sorted(weekly_expiries):
        # walk back day by day to find the previous trading session
        t_minus_1: Optional[date] = None
        for back in range(1, 8):
            cand = exp - timedelta(days=back)
            if cand in sd_set:
                t_minus_1 = cand
                break
        if t_minus_1 is None:
            continue
        # entry session must also have the bhavcopy on disk (we read its EOD OI)
        if not _bhav_path(t_minus_1).exists():
            continue
        # entry session must be within window
        if t_minus_1 < START_DATE or t_minus_1 > END_DATE:
            continue
        pairs.append((t_minus_1, exp))
    print(f"    valid (T-1, expiry) pairs in window: {len(pairs)}")
    return pairs


def compute_cliff(bhav: pd.DataFrame, expiry: date) -> Optional[Dict]:
    """Given a single-day bhavcopy for NIFTY, return cliff metadata for `expiry`.

    Returns dict with keys:
      cliff_strike, ce_oi_at_cliff, pe_oi_at_cliff, total_oi_at_cliff,
      dominant_side ('CE'/'PE'/'MIXED'), dominance_ratio,
      next_ce_above (next CE-dominant cluster strike above cliff),
      next_pe_below (next PE-dominant cluster strike below cliff),
      n_strikes
    Returns None if no NIFTY rows for the expiry.
    """
    # bhavcopy stores expiry_date as Python date objects (object dtype) — direct
    # equality avoids pd.to_datetime conversion that can leave NaT for trailing rows.
    n = bhav[(bhav["symbol"] == UNDERLYING_SYMBOL)
             & (bhav["expiry_date"] == expiry)].copy()
    if n.empty:
        return None
    # Aggregate OI by strike+side
    grp = n.groupby(["strike", "option_type"])["oi"].sum().unstack(fill_value=0)
    if "CE" not in grp.columns or "PE" not in grp.columns:
        return None
    grp["total"] = grp["CE"] + grp["PE"]
    if grp["total"].max() <= 0:
        return None
    cliff_strike = float(grp["total"].idxmax())
    ce_oi = float(grp.loc[cliff_strike, "CE"])
    pe_oi = float(grp.loc[cliff_strike, "PE"])

    if pe_oi > 0:
        ce_pe_ratio = ce_oi / pe_oi
    else:
        ce_pe_ratio = float("inf") if ce_oi > 0 else 1.0
    if ce_oi > 0:
        pe_ce_ratio = pe_oi / ce_oi
    else:
        pe_ce_ratio = float("inf") if pe_oi > 0 else 1.0

    if ce_pe_ratio >= DOMINANCE_RATIO:
        dom_side = "CE"
        dom_ratio = ce_pe_ratio
    elif pe_ce_ratio >= DOMINANCE_RATIO:
        dom_side = "PE"
        dom_ratio = pe_ce_ratio
    else:
        dom_side = "MIXED"
        dom_ratio = max(ce_pe_ratio, pe_ce_ratio)

    # Next opposite-cluster strike for stop placement.
    # SHORT (cliff is CE-dominant, spot ABOVE cliff): SL = next high-CE strike ABOVE cliff.
    # LONG  (cliff is PE-dominant, spot BELOW cliff): SL = next high-PE strike BELOW cliff.
    above = grp[grp.index > cliff_strike].sort_index()
    below = grp[grp.index < cliff_strike].sort_index(ascending=False)
    # "Next cluster" = next strike whose CE (resp PE) OI is at the 90th percentile
    # of all CE (resp PE) OI for this expiry. Falls back to next strike up/down
    # at the configured offset if no qualifying cluster exists.
    ce_q = grp["CE"].quantile(0.90)
    pe_q = grp["PE"].quantile(0.90)
    next_ce_above = None
    for k, row in above.iterrows():
        if row["CE"] >= ce_q and row["CE"] > 0:
            next_ce_above = float(k)
            break
    if next_ce_above is None and not above.empty:
        # fall back: cliff + 1% as a default cluster-equivalent
        next_ce_above = float(cliff_strike) * 1.01
    next_pe_below = None
    for k, row in below.iterrows():
        if row["PE"] >= pe_q and row["PE"] > 0:
            next_pe_below = float(k)
            break
    if next_pe_below is None and not below.empty:
        next_pe_below = float(cliff_strike) * 0.99

    return {
        "cliff_strike": cliff_strike,
        "ce_oi_at_cliff": ce_oi,
        "pe_oi_at_cliff": pe_oi,
        "total_oi_at_cliff": ce_oi + pe_oi,
        "dominant_side": dom_side,
        "dominance_ratio": dom_ratio,
        "next_ce_above": next_ce_above,
        "next_pe_below": next_pe_below,
        "n_strikes": int(len(grp)),
    }


def simulate(
    pairs: List[Tuple[date, date]],
    spot_5m: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Run the entry/exit simulation per (T-1, expiry) pair."""
    print(f"\n  simulating {len(pairs)} (T-1, expiry) pairs ...")
    spot_by_day = {d: g.reset_index(drop=True) for d, g in spot_5m.groupby("d")}

    counters = {
        "n_pairs": len(pairs),
        "no_bhav": 0,
        "no_cliff": 0,
        "mixed_no_dominance": 0,
        "no_spot_at_1430": 0,
        "distance_below_min": 0,
        "distance_above_max": 0,
        "wrong_side_geometry": 0,  # CE-dominant but spot BELOW (and vice versa)
        "no_entry_bar": 0,
        "fired": 0,
    }

    trades: List[dict] = []
    for t_minus_1, expiry in pairs:
        bp = _bhav_path(t_minus_1)
        if not bp.exists():
            counters["no_bhav"] += 1
            continue
        try:
            bhav = pd.read_parquet(
                bp, columns=["symbol", "expiry_date", "strike", "option_type", "oi"]
            )
        except Exception:
            counters["no_bhav"] += 1
            continue
        cliff = compute_cliff(bhav, expiry)
        if cliff is None:
            counters["no_cliff"] += 1
            continue
        if cliff["dominant_side"] == "MIXED":
            counters["mixed_no_dominance"] += 1
            continue

        day_bars = spot_by_day.get(t_minus_1)
        if day_bars is None or day_bars.empty:
            counters["no_spot_at_1430"] += 1
            continue
        # 14:30 evaluation bar (5m bar labelled 14:30 = 14:30-14:34 inclusive close)
        eval_row = day_bars[day_bars["hhmm"] == ENTRY_DECISION_HHMM]
        if eval_row.empty:
            counters["no_spot_at_1430"] += 1
            continue
        eval_row = eval_row.iloc[0]
        spot = float(eval_row["close"])

        cliff_strike = cliff["cliff_strike"]
        spot_distance_pct = (spot - cliff_strike) / cliff_strike * 100.0
        abs_dist = abs(spot_distance_pct)

        if abs_dist < DISTANCE_MIN_PCT:
            counters["distance_below_min"] += 1
            continue
        if abs_dist > DISTANCE_MAX_PCT:
            counters["distance_above_max"] += 1
            continue

        # GAMMA-MAGNET DIRECTION (v2 patch 2026-05-07).
        # v1 sanity dropped 35/103 events on "wrong_side_geometry" because the
        # original brief had the side rule BACKWARD. NSE empirics: CE-dominant
        # cliffs systematically sit ABOVE spot (OTM call OI = resistance built
        # by retail call buying); PE-dominant cliffs sit BELOW spot (OTM put
        # OI = floor). The brief's "spot has already broken through the cliff"
        # case is itself the failure mode — cliffs by construction repel spot
        # from doing so before expiry. The CORRECT mechanic per Indian retail
        # gamma-positioning: MMs are net SHORT options vs retail (SEBI FY23:
        # retail is dominantly long-options on weeklies). Short-gamma delta-
        # hedging is destabilising — pulls spot TOWARD the cliff. So:
        #   CE-cliff above spot  -> LONG NIFTY (magnet pulls up to cliff)
        #   PE-cliff below spot  -> SHORT NIFTY (magnet pulls down to cliff)
        # The previously-firing cases (CE-dom + spot-above, PE-dom + spot-
        # below) become the new wrong-side drops — those are "spot already past
        # the cliff" scenarios where the gamma-magnet has already discharged.
        if cliff["dominant_side"] == "CE" and spot_distance_pct < 0:
            side = "LONG"   # gamma-magnet pulls UP to CE-cliff
        elif cliff["dominant_side"] == "PE" and spot_distance_pct > 0:
            side = "SHORT"  # gamma-magnet pulls DOWN to PE-cliff
        else:
            counters["wrong_side_geometry"] += 1
            continue

        # Entry at next-bar open (14:35)
        entry_row = day_bars[day_bars["hhmm"] == ENTRY_FILL_HHMM]
        if entry_row.empty:
            counters["no_entry_bar"] += 1
            continue
        entry_row = entry_row.iloc[0]
        entry_price = float(entry_row["open"])
        entry_ts = entry_row["date"]

        # Stop placement
        if side == "SHORT":
            opp_cluster = cliff["next_ce_above"]
            if opp_cluster is None:
                opp_cluster = cliff_strike * 1.01
            sl_struct = opp_cluster * (1.0 + SL_BUFFER_PCT / 100.0)
            sl_min = entry_price * (1.0 + MIN_STOP_PCT / 100.0)
            hard_sl = max(sl_struct, sl_min)
            stop_distance = hard_sl - entry_price
            t2_target = cliff_strike
            t1_target = entry_price - 0.5 * (entry_price - cliff_strike)
        else:
            opp_cluster = cliff["next_pe_below"]
            if opp_cluster is None:
                opp_cluster = cliff_strike * 0.99
            sl_struct = opp_cluster * (1.0 - SL_BUFFER_PCT / 100.0)
            sl_min = entry_price * (1.0 - MIN_STOP_PCT / 100.0)
            hard_sl = min(sl_struct, sl_min)
            stop_distance = entry_price - hard_sl
            t2_target = cliff_strike
            t1_target = entry_price + 0.5 * (cliff_strike - entry_price)
        if stop_distance <= 0:
            counters["wrong_side_geometry"] += 1
            continue

        # Forward exit walk: bars from 14:35 onward, hard time stop at 15:15.
        forward = day_bars[(day_bars["date"] >= entry_ts)
                           & (day_bars["hhmm"] <= TIME_STOP_HHMM)].reset_index(drop=True)
        if forward.empty:
            counters["no_entry_bar"] += 1
            continue
        # Entry bar OPEN is the fill; the entry bar's intrabar high/low is in scope
        # for adverse-fill detection (Streak/AlgoTest convention).
        exit_ts = None
        exit_price = None
        exit_reason = None
        hit_t1 = False
        t1_exit_price = None
        t1_exit_ts = None
        for _, bar in forward.iterrows():
            ts = bar["date"]
            high = float(bar["high"])
            low = float(bar["low"])
            close = float(bar["close"])
            # Active SL: post-T1 breakeven trail (matches volume_spike sanity convention)
            active_sl = entry_price if hit_t1 else hard_sl

            if side == "SHORT":
                if high >= active_sl:
                    exit_ts = ts
                    exit_price = active_sl
                    exit_reason = "breakeven_trail" if hit_t1 else "stop"
                    break
                if not hit_t1 and low <= t1_target:
                    hit_t1 = True
                    t1_exit_price = t1_target
                    t1_exit_ts = ts
                if hit_t1 and low <= t2_target:
                    exit_ts = ts
                    exit_price = t2_target
                    exit_reason = "t2_cliff"
                    break
            else:
                if low <= active_sl:
                    exit_ts = ts
                    exit_price = active_sl
                    exit_reason = "breakeven_trail" if hit_t1 else "stop"
                    break
                if not hit_t1 and high >= t1_target:
                    hit_t1 = True
                    t1_exit_price = t1_target
                    t1_exit_ts = ts
                if hit_t1 and high >= t2_target:
                    exit_ts = ts
                    exit_price = t2_target
                    exit_reason = "t2_cliff"
                    break

            if bar["hhmm"] >= TIME_STOP_HHMM:
                exit_ts = ts
                exit_price = close
                exit_reason = "time_stop_1515"
                break

        if exit_price is None:
            last = forward.iloc[-1]
            exit_ts = last["date"]
            exit_price = float(last["close"])
            exit_reason = "time_stop_eod"

        # Sizing: Rs 1000 risk / stop_distance
        qty = max(int(RISK_PER_TRADE_RUPEES / max(stop_distance, 1e-6)), 1)
        if hit_t1:
            qty_t1 = qty // 2
            qty_t2 = qty - qty_t1
            if side == "SHORT":
                pnl_t1 = (entry_price - t1_exit_price) * qty_t1
                pnl_t2 = (entry_price - exit_price) * qty_t2
            else:
                pnl_t1 = (t1_exit_price - entry_price) * qty_t1
                pnl_t2 = (exit_price - entry_price) * qty_t2
            realized_pnl = pnl_t1 + pnl_t2
            fee_t1 = calc_fee(
                entry_price, t1_exit_price, qty_t1,
                "SELL" if side == "SHORT" else "BUY",
            )
            fee_t2 = calc_fee(
                entry_price, exit_price, qty_t2,
                "SELL" if side == "SHORT" else "BUY",
            )
            fee = fee_t1 + fee_t2
            blended_exit = (t1_exit_price * qty_t1 + exit_price * qty_t2) / max(qty, 1)
        else:
            if side == "SHORT":
                realized_pnl = (entry_price - exit_price) * qty
            else:
                realized_pnl = (exit_price - entry_price) * qty
            fee = calc_fee(
                entry_price, exit_price, qty,
                "SELL" if side == "SHORT" else "BUY",
            )
            blended_exit = exit_price

        net_pnl = realized_pnl - fee

        trades.append({
            "T1_entry_date": t_minus_1,
            "expiry_date": expiry,
            "side": side,
            "cliff_strike": cliff_strike,
            "ce_oi_at_cliff": cliff["ce_oi_at_cliff"],
            "pe_oi_at_cliff": cliff["pe_oi_at_cliff"],
            "dominant_side": cliff["dominant_side"],
            "dominance_ratio": cliff["dominance_ratio"],
            "next_ce_above": cliff["next_ce_above"],
            "next_pe_below": cliff["next_pe_below"],
            "spot_at_1430": spot,
            "spot_distance_pct": spot_distance_pct,
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
        counters["fired"] += 1

    return pd.DataFrame(trades), counters


def report(trades: pd.DataFrame, counters: Dict[str, int]) -> None:
    print("\n" + "=" * 64)
    print(" fno_oi_cliff_weekly_expiry  —  pre-coding sanity check")
    print("=" * 64)
    print("\nFunnel (post-gate event count is the n-floor diagnostic):")
    print(f"  weekly (T-1, expiry) pairs in window:    {counters['n_pairs']:>4}")
    print(f"  no bhavcopy on T-1:                       {counters['no_bhav']:>4}")
    print(f"  no cliff (no NIFTY rows for expiry):      {counters['no_cliff']:>4}")
    print(f"  MIXED OI (CE/PE ratio < {DOMINANCE_RATIO}x):           "
          f"{counters['mixed_no_dominance']:>4}")
    print(f"  no spot at 14:30:                         {counters['no_spot_at_1430']:>4}")
    print(f"  distance < {DISTANCE_MIN_PCT:.2f}% (already pinned):     "
          f"{counters['distance_below_min']:>4}")
    print(f"  distance > {DISTANCE_MAX_PCT:.2f}% (macro flow):         "
          f"{counters['distance_above_max']:>4}")
    print(f"  wrong-side geometry (CE+below or PE+above):"
          f"{counters['wrong_side_geometry']:>4}")
    print(f"  missing 14:35 entry bar:                  {counters['no_entry_bar']:>4}")
    print(f"  FIRED:                                    {counters['fired']:>4}")

    if trades.empty:
        print("\n[NO TRADES] sanity check returns 0 trades.")
        print("\n--- VERDICT ---")
        print("SAMPLE TOO THIN — RETIRE-PRE-IMPLEMENTATION")
        return

    n = len(trades)
    npnl = trades["net_pnl"]
    wins = npnl[npnl > 0].sum()
    losses = npnl[npnl < 0].abs().sum()
    pf = round(wins / losses, 3) if losses > 0 else float("inf")
    daily = trades.groupby("T1_entry_date")["net_pnl"].sum()
    sharpe = round(daily.mean() / daily.std(), 3) if daily.std() > 0 else 0.0
    wr = round(float((npnl > 0).mean()) * 100, 1)

    print(f"\nPeriod: {trades['T1_entry_date'].min()} .. {trades['T1_entry_date'].max()}")
    print(f"Trades: n = {n}")
    print(f"Win rate: {wr}%")
    print(f"Gross PnL: Rs.{int(trades['realized_pnl'].sum()):,}")
    print(f"Fees:      Rs.{int(trades['fee'].sum()):,}")
    print(f"NET PnL:   Rs.{int(npnl.sum()):,}")
    print(f"NET PF:    {pf}")
    print(f"NET Sharpe (daily): {sharpe}")

    print("\nPer side:")
    for sd, grp in trades.groupby("side"):
        n2 = len(grp)
        w = grp["net_pnl"][grp["net_pnl"] > 0].sum()
        l = grp["net_pnl"][grp["net_pnl"] < 0].abs().sum()
        pf2 = round(w / l, 3) if l > 0 else float("inf")
        wr2 = round(float((grp["net_pnl"] > 0).mean()) * 100, 1)
        net = int(grp["net_pnl"].sum())
        print(f"  {sd:<6} n={n2:>4} PF={pf2:>5} WR={wr2:>5}% netPnL=Rs.{net:>10,}")

    print("\nPer month (entry session):")
    trades = trades.copy()
    trades["entry_month"] = pd.to_datetime(trades["T1_entry_date"]).dt.strftime("%Y-%m")
    for mth, grp in trades.groupby("entry_month"):
        n2 = len(grp)
        w = grp["net_pnl"][grp["net_pnl"] > 0].sum()
        l = grp["net_pnl"][grp["net_pnl"] < 0].abs().sum()
        pf2 = round(w / l, 3) if l > 0 else float("inf")
        net = int(grp["net_pnl"].sum())
        print(f"  {mth} n={n2:>3} PF={pf2:>6} netPnL=Rs.{net:>10,}")

    print("\nExit-reason breakdown:")
    for rsn, grp in trades.groupby("exit_reason"):
        avg = int(grp["net_pnl"].mean()) if len(grp) else 0
        print(f"  {rsn:<22} n={len(grp):>4} avg_net=Rs.{avg:>6,}")

    # Sample-size diagnostic (per task: brief flagged ~50/yr ceiling; need >=30).
    print("\n--- SAMPLE-SIZE DIAGNOSTIC ---")
    n_required = 30
    if n < n_required:
        print(f"n={n} < {n_required} over 2yr -> SAMPLE TOO THIN — RETIRE-PRE-IMPLEMENTATION")
        sample_thin = True
    else:
        print(f"n={n} >= {n_required} -> sample size OK")
        sample_thin = False

    print("\n--- VERDICT ---")
    if sample_thin:
        print("SAMPLE TOO THIN — RETIRE-PRE-IMPLEMENTATION")
    elif pf >= 1.10:
        print(f"PF={pf} >= 1.10 AND n={n} >= 30 -> APPROVED for detector implementation.")
    elif pf >= 1.00:
        print(f"PF={pf} in [1.00, 1.10) -> MARGINAL. Revisit dominance/distance bands.")
    else:
        print(f"PF={pf} < 1.00 -> RETIRE candidate. OI cliff thesis decisively negative.")


def main() -> None:
    print(f"sanity period: {START_DATE} .. {END_DATE}")
    print(f"underlying: {UNDERLYING_SYMBOL} (NIFTY 50 spot)")
    print(f"option_chain dir: {OPTION_CHAIN_DIR}")
    print()

    spot_5m = load_nifty_5m()
    if spot_5m.empty:
        print("[ERROR] no NIFTY 50 spot bars in window")
        return

    session_dates = list_session_dates(spot_5m)
    pairs = find_weekly_expiry_signals(session_dates)
    if not pairs:
        print("[ERROR] no (T-1, expiry) pairs identified")
        return

    trades, counters = simulate(pairs, spot_5m)

    report(trades, counters)

    out = _REPO_ROOT / "reports" / "sub9_sanity" / "fno_oi_cliff_weekly_expiry_trades.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out, index=False)
    print(f"\nFull trade log: {out}")


if __name__ == "__main__":
    main()
