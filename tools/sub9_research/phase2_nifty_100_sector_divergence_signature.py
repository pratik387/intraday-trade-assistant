# tools/sub9_research/phase2_nifty_100_sector_divergence_signature.py
#
# Phase 2 empirical signature for `nifty_100_sector_divergence_intraday_revert`.
# See specs/2026-05-22-brief-nifty_100_sector_divergence_intraday_revert.md
#
# This script measures the *signature* of stock-vs-sector intraday divergence
# mean-revert across NIFTY 100 names. It is a SIGNATURE MEASUREMENT script
# (no exit walk, no PnL_pct). It records relative-return drift at +60min,
# +120min, and to 13:30 close, so the parent can evaluate Falsifier #1
# (volume-flow signature) and Falsifier #2 (mean-revert direction).
#
# Structural template: phase2_5day_RSI_VWAP_absorb_continuation_signature.py
#
# Mechanism (restated, BILATERAL):
#   - Universe: NIFTY 100 (top-100 by market_cap_cr, mis_enabled=True), per
#     assets/nifty_100_universe.json. Each symbol carries a sector_id.
#   - For each (sector_id, 5m_bar) on a given session_date, compute the
#     sector basket intraday return as a MARKET-CAP-WEIGHTED mean of
#     constituent intraday returns. A sector is skipped on that date if it
#     has < 3 constituent symbols with valid bars (too noisy basket).
#   - For each (sym, 5m_bar) in [10:30, 13:30] IST:
#       stock_intraday_return = (close[i] / open_at_0915) - 1
#       sector_intraday_return = sector basket return at bar i
#       divergence = stock_intraday_return - sector_intraday_return
#       vol_ratio = vols[i] / mean(vols[:i])     # excludes current bar
#     Trigger if |divergence| >= 0.01 (1%) AND vol_ratio >= 1.5x.
#     First-fire-per-day-per-stock latch.
#   - Direction: SHORT if divergence > 0 (outpacer), LONG if divergence < 0
#     (underperformer). Mean-revert thesis.
#
# Target measurements (informational, NOT exit logic):
#   - post_60m_rel_ret  = (stock_ret[bar+12] - sector_ret[bar+12])
#                       - (stock_ret[bar]   - sector_ret[bar])
#   - post_120m_rel_ret = (stock_ret[bar+24] - sector_ret[bar+24])
#                       - (stock_ret[bar]   - sector_ret[bar])
#   - drift_to_1330_rel_ret = (stock_ret[1325] - sector_ret[1325])
#                           - (stock_ret[sig]  - sector_ret[sig])
#   For mean-revert to hold: signed-mean (signal_sign * post_X_rel_ret) < 0.
#
# Anti-bias guards (Lesson #5):
#   1. Day-aggregate look-ahead: each per-bar metric uses ONLY bars[:i+1].
#      Sector basket is computed PER BAR i across constituent stocks present
#      at the bar — no aggregate-day return used to drive trigger.
#   2. vol_baseline excludes current bar (mean of bars[:i], not bars[:i+1]).
#   3. First-fire-per-day latch per (sym, date).
#   4. ProductionUniverseGate per-date with accepted_caps={'large_cap'},
#      require_mis=True (Lesson #19).
#   5. Sector basket includes ONLY symbols that pass the per-date universe
#      gate AND have a valid 0915 open bar AND a valid bar at the current
#      time-slot. No look-ahead into future bars.
#   6. Regime gate: discovery=2023-01..2024-12 / oos=2025-01..2025-12 /
#      holdout=2026-01..2026-04 (standard 3-window split). Also pre/post
#      2023 split for ETF AUM regime sensitivity per brief.
#   7. NO exit walk performed in this script (pure signature measurement,
#      similar to phase2_5day_RSI_VWAP_absorb_continuation_signature).
#
# Pre-registration discipline (brief Falsifiers):
#   - Falsifier #1 (volume-flow): median vol_ratio across signal events
#       must be >= 1.5x; fraction with vol_ratio < 1.2 must be <= 40%.
#   - Falsifier #2 (mean-revert direction): signed-mean
#       `signal_sign * post_120m_rel_ret` must be < 0 (NEGATIVE) with
#       absolute mean magnitude >= 0.0020 (20 bps).
#   - Falsifier #3 (regime): pre/post-2023 split — both periods evaluated.
#
"""Phase 2 empirical signature — nifty_100_sector_divergence_intraday_revert."""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, time as dtime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from tools.sub9_research.production_universe import ProductionUniverseGate  # noqa: E402

# -----------------------------------------------------------------------------
# CONFIG — NO hardcoded defaults inside the logic; every knob declared here.
# -----------------------------------------------------------------------------
CONFIG: Dict[str, object] = {
    # Date window (full history; regime splits applied at analysis time)
    "window_start": date(2023, 1, 2),
    "window_end":   date(2026, 4, 30),

    # Standard 3-window split (matches setup_lifecycle convention)
    "discovery_start":  date(2023, 1, 1),
    "discovery_end":    date(2024, 12, 31),
    "oos_start":        date(2025, 1, 1),
    "oos_end":          date(2025, 12, 31),
    "holdout_start":    date(2026, 1, 1),
    "holdout_end":      date(2026, 4, 30),

    # Universe (large_cap only — NIFTY 100 by construction)
    "accepted_caps": {"large_cap"},
    "require_mis":   True,

    # Intraday signal scan window
    "sig_window_start":  dtime(10, 30),
    "sig_window_end":    dtime(13, 30),   # INCLUSIVE on this side; we still
                                          # require bar+24 (~120min) to land
                                          # before 15:25 so deepest scan bar
                                          # is 13:25.

    # Trigger thresholds (brief §5)
    "divergence_abs_min":   0.01,         # 1% absolute divergence
    "vol_ratio_min":        1.5,          # institutional-flow gate

    # Sector basket guardrails
    "min_constituents_per_sector_per_day": 3,

    # Target measurement bars
    # 5m bars: bar+12 = +60 min, bar+24 = +120 min
    "post_60m_bars":  12,
    "post_120m_bars": 24,
    "drift_target_time": dtime(13, 25),   # 5m bar closing 13:25 == 13:30 IST

    # Regime cuts
    "regime_2023_cut":   date(2023, 1, 1),   # brief notes pre-2023 ETF AUM lower
    "sebi_oct2025_cut":  date(2025, 10, 1),

    # Falsifier thresholds (pre-registered per brief §2)
    "falsifier1_median_vr_min":   1.5,
    "falsifier1_frac_lt12_max":   0.40,
    "falsifier2_min_n":           200,
    "falsifier2_signed_mean_max": 0.0,   # must be NEGATIVE (mean-revert)
    "falsifier2_abs_min":         0.0020,  # 20bps absolute magnitude

    # Paths
    "monthly_5m_dir":  _REPO_ROOT / "backtest-cache-download" / "monthly",
    "universe_path":   _REPO_ROOT / "assets" / "nifty_100_universe.json",
    "out_csv":         _REPO_ROOT / "reports" / "sub9_sanity"
                                   / "_phase2_nifty_100_sector_divergence_signature.csv",
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _months_between(d0: date, d1: date) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    y, m = d0.year, d0.month
    while (y, m) <= (d1.year, d1.month):
        out.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def _ensure_naive_ist(ts_col: pd.Series) -> pd.Series:
    if isinstance(ts_col.dtype, pd.DatetimeTZDtype):
        return ts_col.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return ts_col


# -----------------------------------------------------------------------------
# Universe loader
# -----------------------------------------------------------------------------
def load_universe(path: Path) -> Tuple[Dict[str, str], Dict[str, float]]:
    """Return (sym -> sector_id, sym -> market_cap_cr) for NIFTY 100 names.

    Note: drops the _50 residual cohort per Phase 1 recommendation (high
    intra-sector heterogeneity makes basket noisy).
    """
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    sym_to_sector: Dict[str, str] = {}
    sym_to_mcap: Dict[str, float] = {}
    for row in doc["universe"]:
        sym = row["symbol"]
        sid = row["sector_id"]
        if sid == "NSE_NIFTY_50":
            # Phase 1 recommended drop — residual cohort is noisy
            continue
        sym_to_sector[sym] = sid
        sym_to_mcap[sym] = float(row.get("market_cap_cr", 0.0))
    return sym_to_sector, sym_to_mcap


# -----------------------------------------------------------------------------
# Per-day evaluation
#
# For one session_date d:
#   1. Build a long-form table per (sym, bar_time) -> open_at_0915, close[i],
#      volume[i], stock_intraday_return.
#   2. For each bar_time, compute sector_basket_return per sector_id as
#      market-cap-weighted mean of constituent stock_intraday_return values
#      (skip sectors with < 3 constituents at that bar).
#   3. For each (sym, bar_time) in [10:30, 13:30]:
#         divergence = stock_ret - sector_ret
#         vol_ratio  = vol[i] / mean(vol[:i])
#         Trigger if |divergence| >= 0.01 AND vol_ratio >= 1.5x.
#      First-fire per (sym).
#   4. For each fire, compute post_60m_rel_ret / post_120m_rel_ret /
#      drift_to_1330_rel_ret (informational).
# -----------------------------------------------------------------------------
def evaluate_day(
    day_df: pd.DataFrame,
    sym_to_sector: Dict[str, str],
    sym_to_mcap: Dict[str, float],
    eligible_syms: set,
    d: date,
) -> List[dict]:
    if day_df.empty:
        return []

    # Confine to regular session
    day_df = day_df[(day_df["time"] >= dtime(9, 15)) & (day_df["time"] <= dtime(15, 25))]
    if day_df.empty:
        return []

    # Restrict to NIFTY 100 universe AND eligible (per-date gate)
    day_df = day_df[
        day_df["symbol"].isin(sym_to_sector.keys())
        & day_df["symbol"].isin(eligible_syms)
    ]
    if day_df.empty:
        return []

    # 0915 open per symbol (anchor)
    bar_0915 = day_df[day_df["time"] == dtime(9, 15)][["symbol", "open"]].rename(
        columns={"open": "open_0915"}
    )
    if bar_0915.empty:
        return []
    day_df = day_df.merge(bar_0915, on="symbol", how="inner")
    if day_df.empty:
        return []

    # stock_intraday_return at each bar = close / open_0915 - 1
    day_df["stock_ret"] = day_df["close"] / day_df["open_0915"] - 1.0

    # Add sector_id + mcap
    day_df["sector_id"] = day_df["symbol"].map(sym_to_sector)
    day_df["mcap"] = day_df["symbol"].map(sym_to_mcap)
    day_df = day_df.dropna(subset=["sector_id"])

    # Per (sector_id, time) market-cap-weighted basket return
    min_const = int(CONFIG["min_constituents_per_sector_per_day"])

    def _wmean(group: pd.DataFrame) -> Optional[float]:
        w = group["mcap"].to_numpy(dtype=np.float64)
        r = group["stock_ret"].to_numpy(dtype=np.float64)
        if len(r) < min_const:
            return None
        total = float(w.sum())
        if total <= 0:
            return None
        return float((w * r).sum() / total)

    sect_grp = day_df.groupby(["sector_id", "time"], sort=False)
    sect_rows = []
    for (sid, t), grp in sect_grp:
        wr = _wmean(grp)
        if wr is None:
            continue
        sect_rows.append({"sector_id": sid, "time": t, "sector_ret": wr,
                          "sector_n_constituents": int(len(grp))})
    if not sect_rows:
        return []
    sect_df = pd.DataFrame(sect_rows)

    # Join sector returns back onto day_df by (sector_id, time)
    day_df = day_df.merge(sect_df, on=["sector_id", "time"], how="inner")
    if day_df.empty:
        return []

    # Now per-symbol scan in [10:30, 13:30]; first-fire latch
    sw_start = CONFIG["sig_window_start"]
    sw_end   = CONFIG["sig_window_end"]
    div_min  = float(CONFIG["divergence_abs_min"])
    vr_min   = float(CONFIG["vol_ratio_min"])
    p60      = int(CONFIG["post_60m_bars"])
    p120     = int(CONFIG["post_120m_bars"])
    tgt_time = CONFIG["drift_target_time"]

    out: List[dict] = []
    for sym, sym_df in day_df.groupby("symbol", sort=False):
        sym_df = sym_df.sort_values("time").reset_index(drop=True)
        if sym_df.empty:
            continue

        times = sym_df["time"].tolist()
        closes = sym_df["close"].to_numpy(dtype=np.float64)
        vols = sym_df["volume"].to_numpy(dtype=np.float64)
        stock_rets = sym_df["stock_ret"].to_numpy(dtype=np.float64)
        sector_rets = sym_df["sector_ret"].to_numpy(dtype=np.float64)
        sector_ns = sym_df["sector_n_constituents"].to_numpy(dtype=np.int64)
        sid = sym_df["sector_id"].iloc[0]
        ts_col = sym_df["date"].tolist()
        n = len(sym_df)
        if n < 2:
            continue

        # vol_baseline excludes current bar
        cum_vol_prior = np.cumsum(vols) - vols
        idx_arr = np.arange(n, dtype=np.float64)
        vol_baseline = np.full(n, np.nan, dtype=np.float64)
        pos = idx_arr > 0
        vol_baseline[pos] = cum_vol_prior[pos] / idx_arr[pos]

        # Find 13:25 row (drift target) — if missing, skip drift_to_1330 metric
        tgt_rows = sym_df[sym_df["time"] == tgt_time]
        if not tgt_rows.empty:
            tgt_stock_ret = float(tgt_rows.iloc[0]["stock_ret"])
            tgt_sector_ret = float(tgt_rows.iloc[0]["sector_ret"])
            has_drift = True
        else:
            tgt_stock_ret = float("nan")
            tgt_sector_ret = float("nan")
            has_drift = False

        # First-fire scan
        for i in range(1, n):
            t = times[i]
            if t < sw_start or t > sw_end:
                continue
            if np.isnan(vol_baseline[i]) or vol_baseline[i] <= 0:
                continue
            div = stock_rets[i] - sector_rets[i]
            if not np.isfinite(div):
                continue
            if abs(div) < div_min:
                continue
            vr = float(vols[i] / vol_baseline[i])
            if vr < vr_min:
                continue

            # Direction: mean-revert
            sig_dir = "SHORT" if div > 0 else "LONG"
            sig_sign = 1.0 if div > 0 else -1.0   # SHORT=+1, LONG=-1

            # Post-window relative returns
            def _post_rel(k: int) -> float:
                j = i + k
                if j >= n:
                    return float("nan")
                cur_rel = stock_rets[i] - sector_rets[i]
                fut_rel = stock_rets[j] - sector_rets[j]
                return float(fut_rel - cur_rel)

            post60 = _post_rel(p60)
            post120 = _post_rel(p120)

            if has_drift:
                cur_rel = stock_rets[i] - sector_rets[i]
                tgt_rel = tgt_stock_ret - tgt_sector_ret
                drift_1330 = float(tgt_rel - cur_rel)
            else:
                drift_1330 = float("nan")

            out.append({
                "signal_ts": pd.Timestamp(ts_col[i]).isoformat(),
                "session_date": d.isoformat(),
                "symbol": sym,
                "sector_id": sid,
                "sector_n_constituents": int(sector_ns[i]),
                "signal_bar_time": str(times[i]),
                "stock_intraday_return": float(stock_rets[i]),
                "sector_intraday_return": float(sector_rets[i]),
                "divergence": float(div),
                "vol_ratio": vr,
                "post_60m_rel_ret": post60,
                "post_120m_rel_ret": post120,
                "drift_to_1330_rel_ret": drift_1330,
                "signal_direction": sig_dir,
                "signal_sign": float(sig_sign),
            })
            break  # first-fire per day per sym

    return out


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def run() -> int:
    print("=" * 80)
    print("Phase 2 empirical signature -- nifty_100_sector_divergence_intraday_revert")
    print(f"Window: {CONFIG['window_start']} -> {CONFIG['window_end']}")
    print(f"Caps:   {sorted(CONFIG['accepted_caps'])}")
    print(f"Signal: |stock_ret - sector_ret| >= {CONFIG['divergence_abs_min']*100:.2f}% "
          f"AND vol_ratio >= {CONFIG['vol_ratio_min']:.2f}x")
    print(f"Window: [{CONFIG['sig_window_start']}, {CONFIG['sig_window_end']}]  "
          f"first-fire per day per sym")
    print(f"Drift:  bar+12 (~60m), bar+24 (~120m), bar -> 13:25 (~13:30 close)")
    print("=" * 80)

    # 1. Load NIFTY 100 universe + sector map (drop _50 residual)
    _log("Loading NIFTY 100 universe ...")
    sym_to_sector, sym_to_mcap = load_universe(Path(CONFIG["universe_path"]))
    _log(f"  active universe: {len(sym_to_sector)} symbols across "
         f"{len(set(sym_to_sector.values()))} sectors (NIFTY_50 residual dropped)")

    from collections import Counter
    sect_ct = Counter(sym_to_sector.values())
    for sid, n in sorted(sect_ct.items(), key=lambda kv: -kv[1]):
        print(f"    {sid:30s} n={n}")

    # 2. ProductionUniverseGate (large_cap, mis)
    gate = ProductionUniverseGate(
        accepted_caps=CONFIG["accepted_caps"],
        require_mis=CONFIG["require_mis"],
        min_trading_days_required=0,
        min_daily_avg_volume=0,
    )

    # 3. Per-month 5m scan
    records: List[dict] = []
    months = _months_between(CONFIG["window_start"], CONFIG["window_end"])
    nifty100_set = set(sym_to_sector.keys())

    for (yy, mm) in months:
        path = Path(CONFIG["monthly_5m_dir"]) / f"{yy:04d}_{mm:02d}_5m_enriched.feather"
        if not path.exists():
            _log(f"  skip {yy:04d}-{mm:02d} (no 5m feather)")
            continue

        df = pd.read_feather(
            path,
            columns=["symbol", "date", "open", "high", "low", "close", "volume"],
        )
        df["date"] = _ensure_naive_ist(df["date"])
        df["day"] = df["date"].dt.date
        df["time"] = df["date"].dt.time
        df = df[(df["day"] >= CONFIG["window_start"]) & (df["day"] <= CONFIG["window_end"])]
        if df.empty:
            continue

        # Restrict to NIFTY 100 (massive prune)
        df = df[df["symbol"].isin(nifty100_set)]
        if df.empty:
            continue

        month_fires = 0
        for d, day_grp in df.groupby("day", sort=True):
            # Per-date universe gate
            day_syms_universe = set(day_grp["symbol"].unique())
            eligible_syms = {
                s for s in day_syms_universe if gate.is_eligible(s, d)
            }
            if not eligible_syms:
                continue
            recs = evaluate_day(
                day_grp.copy(), sym_to_sector, sym_to_mcap, eligible_syms, d
            )
            records.extend(recs)
            month_fires += len(recs)

        _log(f"  {yy:04d}-{mm:02d}: fires={month_fires}")

    print()
    print("=" * 80)
    print(f"TOTAL FIRES: {len(records):,}")
    print("=" * 80)

    if not records:
        print("\nNO RECORDS COLLECTED -- abort.")
        return 1

    df_sig = pd.DataFrame.from_records(records)
    df_sig["session_date"] = pd.to_datetime(df_sig["session_date"]).dt.date

    # Persist raw output
    out_path = Path(CONFIG["out_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_sig.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")

    # Add splits
    dt_dates = pd.to_datetime(df_sig["session_date"])
    cut_2023 = pd.Timestamp(CONFIG["regime_2023_cut"])
    cut_sebi = pd.Timestamp(CONFIG["sebi_oct2025_cut"])
    df_sig["pre_post_2023"] = np.where(dt_dates < cut_2023, "pre", "post")
    df_sig["pre_post_sebi_oct2025"] = np.where(dt_dates < cut_sebi, "pre", "post")

    def _window(start: date, end: date) -> pd.Series:
        return (dt_dates >= pd.Timestamp(start)) & (dt_dates <= pd.Timestamp(end))

    df_sig["window"] = "OTHER"
    df_sig.loc[_window(CONFIG["discovery_start"], CONFIG["discovery_end"]), "window"] = "DISCOVERY"
    df_sig.loc[_window(CONFIG["oos_start"],       CONFIG["oos_end"]),       "window"] = "OOS"
    df_sig.loc[_window(CONFIG["holdout_start"],   CONFIG["holdout_end"]),   "window"] = "HOLDOUT"

    # =========================================================================
    # STEP 1 — Falsifier #1: volume-flow signature
    # =========================================================================
    print()
    print("=" * 80)
    print("STEP 1 -- Falsifier #1 (volume-flow signature)")
    print("=" * 80)

    vr_med = float(df_sig["vol_ratio"].median())
    vr_mean = float(df_sig["vol_ratio"].mean())
    frac_lt12 = float((df_sig["vol_ratio"] < 1.2).mean())

    print(f"  n total fires:                 {len(df_sig):,}")
    print(f"  median vol_ratio:              {vr_med:.4f}  (Falsifier #1: >= {CONFIG['falsifier1_median_vr_min']:.2f})")
    print(f"  mean   vol_ratio:              {vr_mean:.4f}")
    print(f"  fraction with vol_ratio < 1.2: {frac_lt12*100:.2f}%  (Falsifier #1: <= {CONFIG['falsifier1_frac_lt12_max']*100:.0f}%)")

    f1_median_ok = vr_med >= float(CONFIG["falsifier1_median_vr_min"])
    f1_frac_ok = frac_lt12 <= float(CONFIG["falsifier1_frac_lt12_max"])
    f1_pass = f1_median_ok and f1_frac_ok
    if f1_pass:
        print("  FALSIFIER #1: PASS")
    else:
        print("  FALSIFIER #1: FAIL")

    # =========================================================================
    # STEP 2 — Falsifier #2: mean-revert direction (post_120m signed mean)
    # =========================================================================
    print()
    print("=" * 80)
    print("STEP 2 -- Falsifier #2 (mean-revert direction)")
    print("=" * 80)

    df_valid = df_sig.dropna(subset=["post_120m_rel_ret"]).copy()
    n_valid = len(df_valid)
    if n_valid == 0:
        print("  No valid post-120m observations.")
        signed_mean_120 = float("nan")
        f2_pass = False
    else:
        df_valid["signed_post_120m"] = df_valid["signal_sign"] * df_valid["post_120m_rel_ret"]
        signed_mean_120 = float(df_valid["signed_post_120m"].mean())
        signed_mean_60 = float(
            (df_valid["signal_sign"] * df_valid["post_60m_rel_ret"]).mean()
        )
        signed_mean_drift = float(
            (df_valid["signal_sign"] * df_valid["drift_to_1330_rel_ret"]).mean()
        )
        print(f"  n_valid (post-120m):                      {n_valid:,}")
        print(f"  signed_mean post_60m  (must be < 0):       {signed_mean_60:+.6f}")
        print(f"  signed_mean post_120m (must be < 0):       {signed_mean_120:+.6f}")
        print(f"  signed_mean drift_to_1330 (must be < 0):   {signed_mean_drift:+.6f}")
        print(f"  Falsifier #2 thresholds: signed_mean <= {CONFIG['falsifier2_signed_mean_max']:.4f}, "
              f"|mean| >= {CONFIG['falsifier2_abs_min']:.4f}")
        f2_pass = (
            (signed_mean_120 <= float(CONFIG["falsifier2_signed_mean_max"]))
            and (abs(signed_mean_120) >= float(CONFIG["falsifier2_abs_min"]))
            and (n_valid >= int(CONFIG["falsifier2_min_n"]))
        )
        print(f"  FALSIFIER #2: {'PASS' if f2_pass else 'FAIL'}")

    # =========================================================================
    # STEP 3 — Splits
    # =========================================================================
    print()
    print("=" * 80)
    print("STEP 3 -- Cohort splits")
    print("=" * 80)

    def _block(label: str, sub: pd.DataFrame) -> None:
        n = len(sub)
        if n == 0:
            print(f"  {label:<40} n=0")
            return
        med_vr = float(sub["vol_ratio"].median())
        mean_div = float(sub["divergence"].mean())
        sm60 = float((sub["signal_sign"] * sub["post_60m_rel_ret"]).mean())
        sm120 = float((sub["signal_sign"] * sub["post_120m_rel_ret"]).mean())
        smd = float((sub["signal_sign"] * sub["drift_to_1330_rel_ret"]).mean())
        print(f"  {label:<40} n={n:>6,}  med_vr={med_vr:.3f}  mean_div={mean_div:+.4f}  "
              f"signed60={sm60:+.5f}  signed120={sm120:+.5f}  signedDrift={smd:+.5f}")

    print()
    print("  --- by 3-window split (discovery/oos/holdout) ---")
    for w in ["DISCOVERY", "OOS", "HOLDOUT"]:
        _block(f"window={w}", df_sig[df_sig["window"] == w])

    print()
    print("  --- by pre/post 2023 (ETF AUM regime sensitivity) ---")
    for tag in ["pre", "post"]:
        _block(f"pre_post_2023={tag}", df_sig[df_sig["pre_post_2023"] == tag])

    print()
    print("  --- by pre/post SEBI Oct 2025 ---")
    for tag in ["pre", "post"]:
        _block(f"pre_post_sebi={tag}", df_sig[df_sig["pre_post_sebi_oct2025"] == tag])

    print()
    print("  --- by signal_direction ---")
    for tag in ["SHORT", "LONG"]:
        _block(f"signal_direction={tag}", df_sig[df_sig["signal_direction"] == tag])

    print()
    print("  --- by sector_id ---")
    for sid in sorted(df_sig["sector_id"].unique()):
        _block(f"sector_id={sid}", df_sig[df_sig["sector_id"] == sid])

    # Divergence magnitude buckets
    def _div_bucket(v: float) -> str:
        a = abs(v)
        if a < 0.015:
            return "1.0-1.5pct"
        if a < 0.020:
            return "1.5-2.0pct"
        if a < 0.030:
            return "2.0-3.0pct"
        return ">=3.0pct"

    df_sig["div_bucket"] = df_sig["divergence"].map(_div_bucket)
    print()
    print("  --- by divergence magnitude bucket ---")
    for b in ["1.0-1.5pct", "1.5-2.0pct", "2.0-3.0pct", ">=3.0pct"]:
        _block(f"div_bucket={b}", df_sig[df_sig["div_bucket"] == b])

    # Vol-ratio buckets
    def _vr_bucket(v: float) -> str:
        if v < 2.0:
            return "1.5-2.0"
        if v < 3.0:
            return "2.0-3.0"
        if v < 5.0:
            return "3.0-5.0"
        return ">=5.0"

    df_sig["vr_bucket"] = df_sig["vol_ratio"].map(_vr_bucket)
    print()
    print("  --- by vol_ratio bucket ---")
    for b in ["1.5-2.0", "2.0-3.0", "3.0-5.0", ">=5.0"]:
        _block(f"vr_bucket={b}", df_sig[df_sig["vr_bucket"] == b])

    # =========================================================================
    # VERDICT
    # =========================================================================
    print()
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)

    if not f1_pass:
        verdict = "KILL"
        reason = (
            f"Falsifier #1 FAIL -- median vol_ratio {vr_med:.3f} "
            f"(>= {CONFIG['falsifier1_median_vr_min']} required) OR "
            f"frac<1.2 {frac_lt12*100:.2f}% (<= {CONFIG['falsifier1_frac_lt12_max']*100:.0f}% required). "
            f"Divergences are noise, not institutional flow."
        )
    elif not f2_pass:
        verdict = "KILL"
        reason = (
            f"Falsifier #2 FAIL -- signed_mean post_120m {signed_mean_120:+.5f} "
            f"(<= {CONFIG['falsifier2_signed_mean_max']:.4f} AND "
            f"|mean| >= {CONFIG['falsifier2_abs_min']:.4f} required). "
            f"Mean-revert direction not confirmed -- mechanism wrong."
        )
    else:
        verdict = "PROCEED to Phase 3 (panel signature)"
        reason = (
            f"Falsifier #1 PASS (median vr {vr_med:.3f}, frac<1.2 {frac_lt12*100:.2f}%), "
            f"Falsifier #2 PASS (signed_mean post_120m {signed_mean_120:+.5f}). "
            f"Build exit walk + per-regime PF CIs next."
        )

    print(f"  {verdict}")
    print(f"  reason: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
