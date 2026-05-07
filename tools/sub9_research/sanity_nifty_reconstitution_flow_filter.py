"""Round-5 sub-9 candidate: nifty_reconstitution_window — flow-magnitude filter.

The vanilla per-index sanity (`sanity_nifty_reconstitution_fade.py`) revealed
a clean monotonic PF gradient by index identity:

  NIFTY 50          PF=6.33 (n=8)   — strongest
  NIFTY Next 50     PF=1.51 (n=23)  — passes
  NIFTY 500         PF=0.60 (n=139) — drag
  NIFTY Midcap 150  PF=0.61 (n=88)  — drag
  NIFTY Smallcap 250PF=0.64 (n=172) — drag

Hypothesis: index identity is a proxy. The real driver is FORCED PASSIVE
FLOW MAGNITUDE per included stock, which differs ~100x across indices:

  NIFTY 50 inclusion   ~ Rs 4,000 cr / stock (200000 cr AUM / 50)
  Smallcap 250 inclusion ~ Rs 8 cr / stock  (2000 cr AUM / 250)

Right "ship" rule isn't index-level — it's flow-to-ADV (forced flow as a
ratio of normal traded value). High-flow-to-ADV events from any index
should produce a pin/squeeze; low-flow-to-ADV NIFTY 50 events (e.g.
RELIANCE-class) shouldn't.

This script computes flow_to_adv_ratio per inclusion and re-runs Window B
(T-1 14:00 fade) — the better of the two windows in the original sanity —
bucketed by ratio. Reports per-bucket PF/WR/n + per-index breakdown to
test whether "big flow from any index" generalizes.

Decision criterion (round-5 standard):
  ratio bucket with PF >= 1.10 AND n >= 30 AND multi-index support -> ship
  Only NIFTY 50 events pass regardless of ratio -> structural retire (it's
    the index identity that drives, not flow magnitude)
  No bucket passes -> retire decisively

Approximate AUM tracking per index (publicly disclosed AMFI/ETF data, Mar
2024 — these are coarse approximations; equal-weight per-stock flow
understates NIFTY 50's heaviest weights):

  NIFTY 50            Rs 2,00,000 cr
  NIFTY Next 50       Rs    30,000 cr
  NIFTY Bank          Rs    25,000 cr
  NIFTY 500           Rs     8,000 cr
  NIFTY Midcap 150    Rs     6,000 cr
  NIFTY Smallcap 250  Rs     2,000 cr
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.sub7_validation.build_per_setup_pnl import calc_fee  # noqa: E402


# ---- Locked params (matches Window B of original fade sanity) ----
DISCOVERY_START = date(2023, 1, 1)
DISCOVERY_END   = date(2024, 12, 31)

ALLOWED_INDICES = {
    "NIFTY 50", "NIFTY Next 50", "NIFTY Bank",
    "NIFTY 500", "NIFTY Midcap 150", "NIFTY Smallcap 250",
}

# Window B: T-1 14:00 -> 15:10
ENTRY_HHMM = "14:00"
EXIT_HHMM  = "15:10"

# Stop: 1.0% hard stop above entry (SHORT)
HARD_STOP_PCT = 1.0

# Targets (R-multiples)
T1_R_MULTIPLE = 0.5
T2_R_MULTIPLE = 1.5
T1_PARTIAL_FRACTION = 0.5

USE_BREAKEVEN_TRAIL_AFTER_T1 = True
RISK_PER_TRADE_RUPEES = 1000

# ADV window: 20 trading days strictly BEFORE effective_date
ADV_LOOKBACK_DAYS = 20

# Approximate passive AUM tracked (Rs cr) per index — Mar 2024 disclosed
# AMFI / ETF holdings. Equal-weight per-stock flow is a coarse proxy but
# acceptable for ranking ratio buckets.
INDEX_AUM_CR = {
    "NIFTY 50":           200000.0,
    "NIFTY Next 50":       30000.0,
    "NIFTY Bank":          25000.0,
    "NIFTY 500":            8000.0,
    "NIFTY Midcap 150":     6000.0,
    "NIFTY Smallcap 250":   2000.0,
}
INDEX_SIZE = {
    "NIFTY 50":            50,
    "NIFTY Next 50":       50,
    "NIFTY Bank":          12,
    "NIFTY 500":          500,
    "NIFTY Midcap 150":   150,
    "NIFTY Smallcap 250": 250,
}

# Bucket cutoffs on flow_to_adv_ratio (forced flow / 20d ADV in cr)
# Reads as "how many days of normal volume the forced buying represents".
BUCKET_EDGES = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, float("inf")]
BUCKET_LABELS = [
    "[0.0,0.5)", "[0.5,1.0)", "[1.0,2.0)", "[2.0,3.0)",
    "[3.0,5.0)", "[5.0,10.0)", "[10.0,inf)",
]


# ============================================================
# Loaders
# ============================================================

def load_events() -> pd.DataFrame:
    """Load reconstitution events; restrict to inclusions in allowed indices,
    inside the Discovery window."""
    path = _REPO_ROOT / "data" / "index_reconstitution" / "events.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing")
    df = pd.read_parquet(path)
    df["announcement_date"] = pd.to_datetime(df["announcement_date"]).dt.date
    df["effective_date"]    = pd.to_datetime(df["effective_date"]).dt.date

    print(f"  raw events: {len(df)}")
    incl = df[df["action"] == "inclusion"].copy()
    print(f"  inclusions: {len(incl)}")

    incl = incl[incl["index_name"].isin(ALLOWED_INDICES)]
    print(f"  inclusions in allowed indices: {len(incl)}")

    incl = incl[
        (incl["effective_date"] >= DISCOVERY_START)
        & (incl["effective_date"] <= DISCOVERY_END)
    ]
    print(f"  inclusions in Discovery {DISCOVERY_START} .. {DISCOVERY_END}: {len(incl)}")
    return incl.reset_index(drop=True)


def load_consolidated_daily() -> pd.DataFrame:
    """Load consolidated daily OHLCV — used for 20d pre-effective ADV (in Rs cr)."""
    path = _REPO_ROOT / "cache" / "preaggregate" / "consolidated_daily.feather"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing")
    df = pd.read_feather(path)
    # Schema: ts, open, high, low, close, volume, symbol
    df["d"] = pd.to_datetime(df["ts"]).dt.date
    # Traded value in Rs (close * volume); /1e7 -> Rs cr
    df["traded_value_cr"] = (df["close"] * df["volume"]) / 1e7
    return df[["symbol", "d", "traded_value_cr"]].sort_values(["symbol", "d"]).reset_index(drop=True)


def _load_5m_for_month(yyyy: int, mm: int) -> pd.DataFrame:
    path = _REPO_ROOT / "backtest-cache-download" / "monthly" / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_feather(path)


def build_5m_for_events(events: pd.DataFrame) -> pd.DataFrame:
    """Load 5m bars for the months covering each event's T-1."""
    needed_months = set()
    for ed in events["effective_date"]:
        needed_months.add((ed.year, ed.month))
        prev_day = ed - timedelta(days=1)
        for _ in range(5):
            needed_months.add((prev_day.year, prev_day.month))
            prev_day -= timedelta(days=1)

    print(f"  loading {len(needed_months)} monthly 5m feathers ...")
    parts: List[pd.DataFrame] = []
    universe_syms = set(events["symbol"].str.replace("NSE:", "", regex=False).unique())

    for yyyy, mm in sorted(needed_months):
        mdf = _load_5m_for_month(yyyy, mm)
        if mdf.empty:
            continue
        mdf = mdf[mdf["symbol"].isin(universe_syms)]
        if not mdf.empty:
            parts.append(mdf)

    if not parts:
        return pd.DataFrame()

    big = pd.concat(parts, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    big["d"] = big["date"].dt.date
    print(f"  total event-relevant 5m bars: {len(big):,}")
    return big


# ============================================================
# Flow / ADV computation
# ============================================================

def compute_forced_flow_cr(index_name: str) -> float:
    """Approximate per-stock forced passive flow in Rs cr.

    Equal-weight proxy: aum / index_size. Coarse but acceptable for
    ratio-bucket ranking; for top-of-NIFTY 50 (RELIANCE-class) this
    UNDERSTATES forced flow because of cap-weighting, but those names
    also have outsized ADV so the ratio is what matters.
    """
    aum = INDEX_AUM_CR.get(index_name)
    sz  = INDEX_SIZE.get(index_name)
    if aum is None or sz is None or sz <= 0:
        return float("nan")
    return aum / sz


def compute_adv_20d_cr(daily_df: pd.DataFrame, symbol: str, effective_d: date) -> Optional[float]:
    """20-trading-day average traded value (Rs cr) STRICTLY BEFORE effective_d."""
    sym = symbol.replace("NSE:", "")
    g = daily_df[daily_df["symbol"] == sym]
    if g.empty:
        return None
    g = g[g["d"] < effective_d]
    if len(g) == 0:
        return None
    g = g.sort_values("d").tail(ADV_LOOKBACK_DAYS)
    if len(g) < 5:  # need at least 5 days
        return None
    return float(g["traded_value_cr"].mean())


def assign_bucket(ratio: float) -> str:
    if pd.isna(ratio):
        return "NA"
    for i in range(len(BUCKET_EDGES) - 1):
        lo, hi = BUCKET_EDGES[i], BUCKET_EDGES[i + 1]
        if lo <= ratio < hi:
            return BUCKET_LABELS[i]
    return BUCKET_LABELS[-1]


# ============================================================
# Window B simulation (T-1 14:00 -> 15:10 SHORT fade)
# ============================================================

def _prev_trading_day(sym_df: pd.DataFrame, target_d: date) -> Optional[date]:
    days_before = sym_df[sym_df["d"] < target_d]["d"].unique()
    if len(days_before) == 0:
        return None
    return max(days_before)


def simulate_window_b(events: pd.DataFrame, big5m: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run Window B (T-1 14:00 5m close SHORT -> 15:10) across all events."""
    days_per_sym: Dict[str, pd.DataFrame] = {
        sym: g.sort_values("date").reset_index(drop=True)
        for sym, g in big5m.groupby("symbol")
    }

    trades: List[dict] = []
    funnel = {
        "events_total": len(events),
        "no_5m_data": 0,
        "no_target_day": 0,
        "no_entry_bar": 0,
        "no_exit_window": 0,
        "fired": 0,
    }

    for _, ev in events.iterrows():
        raw_sym = ev["symbol"]
        sym = raw_sym.replace("NSE:", "")
        eff_d = ev["effective_date"]
        idx_name = ev["index_name"]
        forced_flow = ev["forced_flow_cr"]
        adv_cr = ev["adv_20d_cr"]
        ratio = ev["flow_to_adv_ratio"]
        bucket = ev["flow_bucket"]

        sym_df = days_per_sym.get(sym)
        if sym_df is None or sym_df.empty:
            funnel["no_5m_data"] += 1
            continue

        target_d = _prev_trading_day(sym_df, eff_d)
        if target_d is None:
            funnel["no_target_day"] += 1
            continue

        day_df = sym_df[sym_df["d"] == target_d].sort_values("date").reset_index(drop=True)
        if day_df.empty:
            funnel["no_target_day"] += 1
            continue

        day_df["hhmm"] = day_df["date"].dt.strftime("%H:%M")

        entry_rows = day_df[day_df["hhmm"] == ENTRY_HHMM]
        if entry_rows.empty:
            funnel["no_entry_bar"] += 1
            continue
        entry_row = entry_rows.iloc[0]
        entry_ts = entry_row["date"]
        entry_close = float(entry_row["close"])
        entry_price = entry_close

        hard_sl = entry_price * (1.0 + HARD_STOP_PCT / 100.0)
        stop_distance = hard_sl - entry_price
        t1_target = entry_price - T1_R_MULTIPLE * stop_distance
        t2_target = entry_price - T2_R_MULTIPLE * stop_distance

        entry_idx_arr = day_df.index[day_df["date"] == entry_ts].tolist()
        if not entry_idx_arr:
            funnel["no_entry_bar"] += 1
            continue
        entry_idx = entry_idx_arr[0]
        forward = day_df.iloc[entry_idx + 1:].copy()
        forward = forward[forward["hhmm"] <= EXIT_HHMM]
        if forward.empty:
            funnel["no_exit_window"] += 1
            continue

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
            hhmm = bar["hhmm"]

            active_sl = entry_price if (hit_t1 and USE_BREAKEVEN_TRAIL_AFTER_T1) else hard_sl

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
                exit_reason = "t2"
                break
            if hhmm >= EXIT_HHMM:
                exit_ts = ts
                exit_price = close
                exit_reason = "time_stop"
                break

        if exit_price is None:
            last = forward.iloc[-1]
            exit_ts = last["date"]
            exit_price = float(last["close"])
            exit_reason = "time_stop"

        qty = max(int(RISK_PER_TRADE_RUPEES / max(stop_distance, 1e-6)), 1)

        if hit_t1:
            qty_t1 = max(int(qty * T1_PARTIAL_FRACTION), 1)
            qty_t2 = max(qty - qty_t1, 0)
            pnl_t1 = (entry_price - t1_exit_price) * qty_t1
            pnl_t2 = (entry_price - exit_price) * qty_t2 if qty_t2 > 0 else 0.0
            realized_pnl = pnl_t1 + pnl_t2
            fee_t1 = calc_fee(entry_price, t1_exit_price, qty_t1, "SELL")
            fee_t2 = calc_fee(entry_price, exit_price, qty_t2, "SELL") if qty_t2 > 0 else 0.0
            fee = fee_t1 + fee_t2
            blended_exit = (
                (t1_exit_price * qty_t1 + exit_price * qty_t2) / max(qty, 1)
                if qty_t2 > 0 else t1_exit_price
            )
        else:
            realized_pnl = (entry_price - exit_price) * qty
            fee = calc_fee(entry_price, exit_price, qty, "SELL")
            blended_exit = exit_price

        net_pnl = realized_pnl - fee

        trades.append({
            "symbol": "NSE:" + sym,
            "index_name": idx_name,
            "announcement_date": ev["announcement_date"],
            "effective_date": eff_d,
            "trade_date": target_d,
            "entry_ts": entry_ts,
            "entry_price": entry_price,
            "hard_sl": hard_sl,
            "t1_target": t1_target,
            "t2_target": t2_target,
            "hit_t1": hit_t1,
            "t1_exit_price": t1_exit_price,
            "t1_exit_ts": t1_exit_ts,
            "exit_ts": exit_ts,
            "exit_price": blended_exit,
            "exit_reason": exit_reason,
            "stop_distance": stop_distance,
            "qty": qty,
            "realized_pnl": realized_pnl,
            "fee": fee,
            "net_pnl": net_pnl,
            "forced_flow_cr": forced_flow,
            "adv_20d_cr": adv_cr,
            "flow_to_adv_ratio": ratio,
            "flow_bucket": bucket,
        })
        funnel["fired"] += 1

    return pd.DataFrame(trades), funnel


# ============================================================
# Reporting
# ============================================================

def _print_funnel(funnel: dict) -> None:
    print(f"\n  Funnel:")
    print(f"    events in Discovery     : {funnel['events_total']}")
    print(f"    no 5m data for symbol   : {funnel['no_5m_data']}")
    print(f"    no target trading day   : {funnel['no_target_day']}")
    print(f"    no entry bar at HHMM    : {funnel['no_entry_bar']}")
    print(f"    no exit window bars     : {funnel['no_exit_window']}")
    print(f"    -> FIRED                : {funnel['fired']}")


def _bucket_summary(trades: pd.DataFrame, label: str) -> dict:
    if trades.empty:
        print(f"\n[{label}] no trades")
        return {"label": label, "n": 0, "pf": None, "wr": None, "sharpe": None,
                "net_pnl": 0.0}
    n = len(trades)
    npnl = trades["net_pnl"]
    wins = npnl[npnl > 0].sum()
    losses = npnl[npnl < 0].abs().sum()
    pf = round(wins / losses, 3) if losses > 0 else float("inf")
    wr = round(float((npnl > 0).mean()) * 100, 1)
    daily = trades.groupby("trade_date")["net_pnl"].sum()
    sharpe = round(daily.mean() / daily.std(), 3) if daily.std() > 0 else 0.0

    print(f"\n=== {label} ===")
    print(f"  n         : {n}")
    print(f"  WR        : {wr}%")
    print(f"  Gross PnL : Rs.{int(trades['realized_pnl'].sum()):,}")
    print(f"  Fees      : Rs.{int(trades['fee'].sum()):,}")
    print(f"  NET PnL   : Rs.{int(npnl.sum()):,}")
    print(f"  NET PF    : {pf}")
    print(f"  Sharpe(d) : {sharpe}")
    print(f"  Avg net   : Rs.{int(npnl.mean()):,}")

    print("  Per index:")
    for idx_name, grp in trades.groupby("index_name"):
        n2 = len(grp)
        w = grp["net_pnl"][grp["net_pnl"] > 0].sum()
        l = grp["net_pnl"][grp["net_pnl"] < 0].abs().sum()
        pf2 = round(w / l, 3) if l > 0 else float("inf")
        wr2 = round(float((grp["net_pnl"] > 0).mean()) * 100, 1)
        net = int(grp["net_pnl"].sum())
        print(f"    {idx_name:<20} n={n2:>3} PF={pf2:>6} WR={wr2:>5}% net=Rs.{net:>10,}")

    return {"label": label, "n": n, "pf": pf, "wr": wr, "sharpe": sharpe,
            "net_pnl": float(npnl.sum())}


def _print_ratio_distribution(events: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("FLOW-TO-ADV RATIO DISTRIBUTION (all 303 Discovery inclusions)")
    print("=" * 70)
    valid = events.dropna(subset=["flow_to_adv_ratio"])
    print(f"  events with computable ratio: {len(valid)} / {len(events)}")
    if valid.empty:
        return

    print(f"  ratio min={valid['flow_to_adv_ratio'].min():.3f}  "
          f"median={valid['flow_to_adv_ratio'].median():.3f}  "
          f"max={valid['flow_to_adv_ratio'].max():.3f}")
    print(f"  forced_flow_cr min={valid['forced_flow_cr'].min():.1f}  "
          f"median={valid['forced_flow_cr'].median():.1f}  "
          f"max={valid['forced_flow_cr'].max():.1f}")
    print(f"  adv_20d_cr min={valid['adv_20d_cr'].min():.2f}  "
          f"median={valid['adv_20d_cr'].median():.2f}  "
          f"max={valid['adv_20d_cr'].max():.2f}")

    print("\n  Histogram by bucket:")
    bucket_counts = events["flow_bucket"].value_counts().reindex(
        BUCKET_LABELS + ["NA"], fill_value=0
    )
    total = len(events)
    for lbl in BUCKET_LABELS + ["NA"]:
        cnt = int(bucket_counts.get(lbl, 0))
        if cnt == 0:
            continue
        bar = "#" * int(60 * cnt / max(total, 1))
        print(f"    {lbl:<14} {cnt:>4}  {bar}")

    print("\n  Per-index ratio summary:")
    for idx_name, grp in events.groupby("index_name"):
        v = grp.dropna(subset=["flow_to_adv_ratio"])
        if v.empty:
            print(f"    {idx_name:<20} n={len(grp):>3} (no ratio computable)")
            continue
        print(f"    {idx_name:<20} n={len(grp):>3} "
              f"flow_cr={grp['forced_flow_cr'].iloc[0]:>7.1f} "
              f"ratio: min={v['flow_to_adv_ratio'].min():>6.2f} "
              f"med={v['flow_to_adv_ratio'].median():>6.2f} "
              f"max={v['flow_to_adv_ratio'].max():>7.2f}")


def _bucket_verdict(label: str, summary: dict, n_indices: int) -> str:
    n = summary["n"]
    pf = summary["pf"]
    if n < 30:
        return f"[{label}] n={n} < 30 — sample too tight."
    if pf is None or pf < 1.10:
        return f"[{label}] PF={pf} < 1.10 — fails PF gate."
    if n_indices < 2:
        return (f"[{label}] PASSES PF/n gate but only {n_indices} index "
                f"contributes — single-index dominance, NOT generalized.")
    return (f"[{label}] PASSES round-5 gate: n={n}, PF={pf}, "
            f"{n_indices} indices contribute. Real flow-magnitude edge.")


# ============================================================
# Main
# ============================================================

def main():
    print("=== nifty_reconstitution_window — flow-magnitude filter sanity ===")
    print(f"Discovery window: {DISCOVERY_START} .. {DISCOVERY_END}")
    print(f"Window: T-1 {ENTRY_HHMM} -> {EXIT_HHMM} SHORT (best of original sanity)")

    print("\nLoading reconstitution events ...")
    events = load_events()
    if events.empty:
        print("No qualifying events; aborting.")
        return

    print("\nLoading consolidated daily for ADV ...")
    daily = load_consolidated_daily()
    print(f"  daily rows: {len(daily):,}, symbols: {daily['symbol'].nunique()}")

    print("\nComputing forced_flow_cr & adv_20d_cr & flow_to_adv_ratio ...")
    flows = []
    advs = []
    for _, ev in events.iterrows():
        ff = compute_forced_flow_cr(ev["index_name"])
        adv = compute_adv_20d_cr(daily, ev["symbol"], ev["effective_date"])
        flows.append(ff)
        advs.append(adv)
    events["forced_flow_cr"] = flows
    events["adv_20d_cr"] = advs
    events["flow_to_adv_ratio"] = [
        (f / a) if (a is not None and a > 0 and not pd.isna(f)) else np.nan
        for f, a in zip(flows, advs)
    ]
    events["flow_bucket"] = events["flow_to_adv_ratio"].apply(assign_bucket)

    _print_ratio_distribution(events)

    print("\nLoading 5m feathers for event months ...")
    big5m = build_5m_for_events(events)
    if big5m.empty:
        print("No 5m bars for any event symbol; aborting.")
        return

    print("\nRunning Window B simulation across ALL events ...")
    trades, funnel = simulate_window_b(events, big5m)
    _print_funnel(funnel)

    if trades.empty:
        print("No trades fired; aborting.")
        return

    # Overall summary
    overall = _bucket_summary(trades, "OVERALL (all buckets, all indices)")

    # Per-bucket summaries
    print("\n" + "=" * 70)
    print("PER-BUCKET PERFORMANCE")
    print("=" * 70)

    bucket_summaries: Dict[str, dict] = {}
    bucket_index_counts: Dict[str, int] = {}
    for lbl in BUCKET_LABELS:
        grp = trades[trades["flow_bucket"] == lbl]
        if grp.empty:
            print(f"\n[{lbl}] no trades")
            bucket_summaries[lbl] = {"n": 0, "pf": None}
            bucket_index_counts[lbl] = 0
            continue
        s = _bucket_summary(grp, f"BUCKET ratio {lbl}")
        bucket_summaries[lbl] = s
        bucket_index_counts[lbl] = grp["index_name"].nunique()

    # NA bucket (no ratio computable — likely no daily data)
    na_grp = trades[trades["flow_bucket"] == "NA"]
    if not na_grp.empty:
        _bucket_summary(na_grp, "BUCKET NA (no ADV computable)")

    # Bucket verdicts
    print("\n" + "=" * 70)
    print("PER-BUCKET VERDICTS")
    print("=" * 70)
    passing_buckets: List[str] = []
    for lbl in BUCKET_LABELS:
        s = bucket_summaries.get(lbl, {"n": 0, "pf": None})
        nidx = bucket_index_counts.get(lbl, 0)
        v = _bucket_verdict(lbl, s, nidx)
        print(v)
        if (s.get("n", 0) >= 30 and s.get("pf") is not None and s["pf"] >= 1.10):
            passing_buckets.append(lbl)

    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT (round-5 candidate: nifty_reconstitution_window)")
    print("=" * 70)

    if not passing_buckets:
        # Determine what NIFTY 50 alone is doing
        nifty50 = trades[trades["index_name"] == "NIFTY 50"]
        if not nifty50.empty:
            n50_pf = (nifty50["net_pnl"][nifty50["net_pnl"] > 0].sum() /
                      max(nifty50["net_pnl"][nifty50["net_pnl"] < 0].abs().sum(), 1e-6))
            print(f"NIFTY 50 standalone: n={len(nifty50)}, PF={n50_pf:.3f}")
        print(
            "RETIRE — flow-magnitude filter does NOT generalize.\n"
            "  No ratio bucket meets PF>=1.10 + n>=30 jointly.\n"
            "  Either index identity is the proxy (only NIFTY 50 worked, n too\n"
            "  small) or the original NIFTY 50 PF was sample-noise.\n"
            "  Round-5 candidate retired."
        )
    else:
        # Check multi-index support per passing bucket
        multi_pass = [lbl for lbl in passing_buckets if bucket_index_counts[lbl] >= 2]
        if multi_pass:
            print(
                f"SHIP flow-magnitude cell. Buckets passing strict gate with "
                f"multi-index support: {multi_pass}.\n"
                f"  This is real edge — high-flow events from multiple indices\n"
                f"  produce the pin/squeeze, not just NIFTY 50 identity."
            )
        else:
            print(
                f"BORDERLINE: buckets {passing_buckets} pass PF/n strictly but\n"
                f"  only one index dominates each. Likely confirmation that\n"
                f"  index identity (NIFTY 50) is the driver, not flow magnitude.\n"
                f"  Recommend RETIRE — flow filter doesn't add generalization."
            )

    # Persist trades CSV
    out_dir = _REPO_ROOT / "reports" / "sub9_sanity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "nifty_reconstitution_flow_filter_trades.csv"
    trades.to_csv(out_csv, index=False)
    print(f"\nFull trade log: {out_csv}")


if __name__ == "__main__":
    main()
