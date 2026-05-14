"""C4 PROXY SANITY — Pre-open auction follow-through, using 09:15 open as proxy.

Hypothesis (pre-registered 2026-05-14):
  Feb 1, 2025 SEBI rule eliminated leverage on long options. Retail option-buying
  flow that used to position pre-open (low-premium-cost speculative calls) is
  now gone or much smaller. Pre-open / opening behavior in NIFTY-50 names
  should be more "honest" post-rule, so gap_pct + first-5m-bar direction
  should predict next 30/60/120-min returns better in the post-rule regime.

What we test:
  For each (NIFTY-50 symbol, session), compute:
    * gap_pct      = (09:15 open - PDC) / PDC
    * bar1_dir     = sign(09:20 close - 09:15 open)
    * fwd_30m_pct  = (09:45 close - 09:15 open) / 09:15 open
    * fwd_60m_pct  = (10:15 close - 09:15 open) / 09:15 open
    * fwd_120m_pct = (11:15 close - 09:15 open) / 09:15 open

  Cells = (gap_dir, bar1_dir) quadrants × (|gap_pct| threshold).
    - gap_up_bar_up    = momentum continuation LONG
    - gap_up_bar_down  = exhaustion fade SHORT  (already gap_fade territory)
    - gap_dn_bar_up    = reversal LONG
    - gap_dn_bar_down  = continuation SHORT

  Outcomes per cell × regime × horizon:
    - n (sample size)
    - directional_acc (% of trades that closed in predicted direction)
    - avg_pct (mean forward return in predicted direction)
    - PF (gross_win / gross_loss after Indian intraday MIS fees)

  Regimes split at Feb 1, 2025:
    - PRE-RULE (2023-01 to 2025-01-31): baseline
    - POST-RULE (2025-02-01 to 2026-04-30): treatment

Pre-registered falsifiers:
  * Hypothesis confirmed IF:
      - In >=1 cell × horizon: post-rule PF >= pre-rule PF + 0.15
      - And post-rule n >= 200 per arm (LONG/SHORT)
      - And post-rule directional_acc >= 56% (vs 50% baseline)
  * Hypothesis rejected if no cell meets all three.

This is a PROXY test — true auction discovery price (09:08 IEP) is unavailable.
The 09:15 open is a coarse proxy. If proxy version shows lift, escalate to
paid-vendor data (TickData / Refinitiv). If proxy shows nothing, retire candidate.

Usage:
  python -m tools.research.post_sebi.sanity_preopen_proxy
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Dict, List

import pandas as pd

_REPO = Path(__file__).resolve().parents[3]
_FEATHER_DIR = _REPO / "backtest-cache-download" / "monthly"
_NIFTY50_CSV = _REPO / "assets" / "ind_nifty50list.csv"
_OUT_DIR = _REPO / "reports" / "research" / "post_sebi" / "preopen_proxy"
_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Indian intraday fee model: simplified 0.10% one-way for sanity-check.
# (We're testing directional edge, not absolute PnL precision. Final ship
# decision uses report_utils.calculate_single_trade_charges with MIS leverage.)
_FEE_PCT_ONE_WAY = 0.0010

# Regime split
_RULE_DATE = pd.Timestamp("2025-02-01")

# Gap threshold buckets (basis points)
_GAP_BUCKETS = [
    ("any",       0.0,  100.0),   # any gap, even tiny
    ("0.3pct",    0.003, 100.0),  # |gap| >= 0.3%
    ("0.5pct",    0.005, 100.0),  # |gap| >= 0.5%
    ("1.0pct",    0.010, 100.0),  # |gap| >= 1.0%
]


def load_nifty50() -> List[str]:
    df = pd.read_csv(_NIFTY50_CSV)
    return sorted(df["Symbol"].dropna().astype(str).str.strip().unique().tolist())


def _gap_dir(gp: float) -> str:
    return "up" if gp > 0 else "down"


def _bar_dir(open_: float, close_: float) -> str:
    return "up" if close_ > open_ else "down"


def _quadrant(gap_d: str, bar_d: str) -> str:
    return f"gap_{gap_d}_bar_{bar_d}"


def build_events(feather_paths: List[Path], symbols: set) -> pd.DataFrame:
    """For each (symbol, session), produce one event row with gap_pct,
    bar1 direction, and forward returns at 30/60/120 min.

    Loads ALL feathers into memory first (concat), so prior-session PDC
    lookups can cross month boundaries.
    """
    # Concat all feathers first so PDC can come from prior monthly file
    all_dfs = []
    for fp in feather_paths:
        try:
            df = pd.read_feather(fp)
        except Exception as e:
            print(f"  WARN: could not read {fp.name}: {e}")
            continue
        df = df[df["symbol"].isin(symbols)]
        if not df.empty:
            all_dfs.append(df)
    if not all_dfs:
        return pd.DataFrame()
    big = pd.concat(all_dfs, ignore_index=True)
    big["date"] = pd.to_datetime(big["date"])
    big = big.sort_values(["symbol", "date"]).reset_index(drop=True)
    big["session_date"] = big["date"].dt.date
    big["hhmm"] = big["date"].dt.hour * 100 + big["date"].dt.minute
    print(f"  loaded {len(big):,} bars across {big['symbol'].nunique()} symbols")

    # Per-symbol PDC table: last close per (symbol, session_date)
    daily_close = (
        big.groupby(["symbol", "session_date"], sort=True)
        .agg(last_close=("close", "last"))
        .reset_index()
    )
    # Shift to get prior session's close
    daily_close["pdc"] = (
        daily_close.groupby("symbol")["last_close"].shift(1)
    )
    pdc_map = daily_close.set_index(["symbol", "session_date"])["pdc"].to_dict()

    # Pivot session bars: pick out 915, 920, 940, 1010, 1110 for each (sym, session)
    big_pivot = big[big["hhmm"].isin([915, 920, 940, 1010, 1110])].copy()
    rows = []
    for (sym, sd), grp in big_pivot.groupby(["symbol", "session_date"]):
        try:
            pdc = pdc_map.get((sym, sd))
            if pdc is None or pd.isna(pdc) or pdc <= 0:
                continue
            bars = {int(r["hhmm"]): r for _, r in grp.iterrows()}
            if 915 not in bars or 920 not in bars or 940 not in bars or 1010 not in bars:
                continue
            open_915 = float(bars[915]["open"])
            close_920 = float(bars[920]["close"])
            close_30m = float(bars[940]["close"])
            close_60m = float(bars[1010]["close"])
            close_120m = float(bars[1110]["close"]) if 1110 in bars else float("nan")
            if open_915 <= 0:
                continue

            gap_pct = (open_915 - float(pdc)) / float(pdc)
            bar1_ret = (close_920 - open_915) / open_915
            fwd_30 = (close_30m - open_915) / open_915
            fwd_60 = (close_60m - open_915) / open_915
            fwd_120 = (close_120m - open_915) / open_915 if pd.notna(close_120m) else float("nan")

            rows.append({
                "symbol": sym,
                "session_date": sd,
                "pdc": float(pdc),
                "open_915": open_915,
                "close_920": close_920,
                "gap_pct": gap_pct,
                "bar1_ret": bar1_ret,
                "gap_dir": _gap_dir(gap_pct),
                "bar_dir": _bar_dir(open_915, close_920),
                "quadrant": _quadrant(_gap_dir(gap_pct), _bar_dir(open_915, close_920)),
                "fwd_30m_pct": fwd_30,
                "fwd_60m_pct": fwd_60,
                "fwd_120m_pct": fwd_120,
            })
        except (KeyError, ValueError, TypeError):
            continue

    return pd.DataFrame(rows)


def predicted_direction(quadrant: str) -> str:
    """The hypothesis-suggested direction for each quadrant."""
    # gap_up_bar_up    -> LONG (momentum continuation)
    # gap_up_bar_down  -> SHORT (exhaustion fade)
    # gap_dn_bar_up    -> LONG (reversal)
    # gap_dn_bar_down  -> SHORT (continuation)
    return "long" if quadrant.endswith("bar_up") else "short"


def compute_pnl_for_direction(ret_pct: float, direction: str) -> float:
    """Net return in pct after one round-trip of fees (~0.20% total)."""
    signed = ret_pct if direction == "long" else -ret_pct
    return signed - 2 * _FEE_PCT_ONE_WAY


def metrics_block(df: pd.DataFrame, horizon_col: str) -> Dict:
    if df.empty:
        return dict(n=0, acc=0.0, avg=0.0, pf=0.0, net_avg=0.0)
    df = df.dropna(subset=[horizon_col]).copy()
    if df.empty:
        return dict(n=0, acc=0.0, avg=0.0, pf=0.0, net_avg=0.0)
    df["direction"] = df["quadrant"].apply(predicted_direction)
    df["net_ret_pct"] = df.apply(
        lambda r: compute_pnl_for_direction(r[horizon_col], r["direction"]),
        axis=1,
    )
    wins = df[df["net_ret_pct"] > 0]
    losses = df[df["net_ret_pct"] <= 0]
    gw = float(wins["net_ret_pct"].sum())
    gl = float(-losses["net_ret_pct"].sum())
    pf = gw / gl if gl > 0 else float("inf")
    return dict(
        n=len(df),
        acc=100.0 * len(wins) / len(df),
        avg=100.0 * float(df[horizon_col].mean()),
        net_avg=100.0 * float(df["net_ret_pct"].mean()),
        pf=pf,
    )


def main():
    print(f"NIFTY-50 list: {_NIFTY50_CSV.name}")
    symbols = set(load_nifty50())
    print(f"  {len(symbols)} symbols loaded")

    feathers = sorted(_FEATHER_DIR.glob("20*_5m_enriched.feather"))
    print(f"\n5m feathers: {len(feathers)} monthly files")
    if not feathers:
        print("  ERROR: no monthly feathers found")
        sys.exit(1)

    print("\nBuilding event table (one row per symbol-session)...")
    events = build_events(feathers, symbols)
    print(f"  total events: {len(events):,}")

    events["session_ts"] = pd.to_datetime(events["session_date"])
    events["regime"] = events["session_ts"].apply(
        lambda d: "pre_rule" if d < _RULE_DATE else "post_rule"
    )
    print(f"  pre-rule events:  {(events['regime']=='pre_rule').sum():,}")
    print(f"  post-rule events: {(events['regime']=='post_rule').sum():,}")

    # ---- Per (regime × gap_bucket × quadrant × horizon) metrics ----
    out_rows = []
    horizons = ["fwd_30m_pct", "fwd_60m_pct", "fwd_120m_pct"]
    quadrants = ["gap_up_bar_up", "gap_up_bar_down", "gap_down_bar_up", "gap_down_bar_down"]
    regimes = ["pre_rule", "post_rule"]

    for regime in regimes:
        sub_reg = events[events["regime"] == regime]
        for bucket_name, gmin, gmax in _GAP_BUCKETS:
            mask = (sub_reg["gap_pct"].abs() >= gmin) & (sub_reg["gap_pct"].abs() <= gmax)
            sub_b = sub_reg[mask]
            for quad in quadrants:
                sub_q = sub_b[sub_b["quadrant"] == quad]
                for hz in horizons:
                    m = metrics_block(sub_q, hz)
                    out_rows.append({
                        "regime": regime,
                        "gap_bucket": bucket_name,
                        "quadrant": quad,
                        "horizon": hz.replace("fwd_", "").replace("_pct", ""),
                        "n": m["n"],
                        "acc_pct": round(m["acc"], 2),
                        "avg_ret_pct": round(m["avg"], 4),
                        "net_avg_pct": round(m["net_avg"], 4),
                        "pf": round(m["pf"], 3) if m["pf"] != float("inf") else None,
                    })

    results = pd.DataFrame(out_rows)
    results_path = _OUT_DIR / "preopen_proxy_results.csv"
    results.to_csv(results_path, index=False)
    events_path = _OUT_DIR / "preopen_proxy_events.parquet"
    events.to_parquet(events_path, index=False)
    print(f"\nResults: {results_path}")
    print(f"Events:  {events_path}")

    # ---- Pre-registered falsifier check ----
    print("\n" + "=" * 78)
    print("PRE-REGISTERED FALSIFIER CHECK")
    print("=" * 78)

    pf_pivot = results.pivot_table(
        index=["gap_bucket", "quadrant", "horizon"],
        columns="regime",
        values=["pf", "n", "acc_pct"],
    )

    promising = []
    for (bucket, quad, hz), row in pf_pivot.iterrows():
        try:
            pre_pf = row.get(("pf", "pre_rule"), 0) or 0
            post_pf = row.get(("pf", "post_rule"), 0) or 0
            post_n = row.get(("n", "post_rule"), 0) or 0
            post_acc = row.get(("acc_pct", "post_rule"), 0) or 0
            pre_n = row.get(("n", "pre_rule"), 0) or 0
            delta_pf = post_pf - pre_pf
            if delta_pf >= 0.15 and post_n >= 200 and post_acc >= 56.0:
                promising.append({
                    "bucket": bucket, "quadrant": quad, "horizon": hz,
                    "pre_pf": pre_pf, "post_pf": post_pf,
                    "delta_pf": round(delta_pf, 3),
                    "post_n": int(post_n), "pre_n": int(pre_n),
                    "post_acc": round(post_acc, 1),
                })
        except Exception:
            continue

    if not promising:
        print("\nNO CELL passes the pre-registered falsifier:")
        print("   delta_pf >= 0.15 AND post_n >= 200 AND post_acc >= 56%")
        print("\nHypothesis REJECTED for proxy data.")
        print("Recommendation: retire C4 unless paid-vendor data changes the picture.")
    else:
        print(f"\n{len(promising)} cells pass the pre-registered falsifier:\n")
        prom_df = pd.DataFrame(promising)
        print(prom_df.to_string(index=False))
        print("\nHypothesis CONFIRMED for proxy data.")
        print("Recommendation: escalate to richer-data investigation.")
        prom_df.to_csv(_OUT_DIR / "preopen_proxy_promising_cells.csv", index=False)

    # ---- Summary table by regime ----
    print("\n" + "=" * 78)
    print("SUMMARY TABLE (gap_bucket=0.5pct, horizon=60m)")
    print("=" * 78)
    summary = results[
        (results["gap_bucket"] == "0.5pct") & (results["horizon"] == "60m")
    ].sort_values(["quadrant", "regime"])
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
