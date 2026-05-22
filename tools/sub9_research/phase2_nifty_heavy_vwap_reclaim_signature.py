"""
tools/sub9_research/phase2_nifty_heavy_vwap_reclaim_signature.py

Phase 2 empirical signature for nifty_heavy_vwap_reclaim_long candidate.
See specs/2026-05-22-brief-nifty_heavy_vwap_reclaim_long.md

Mechanism (from brief):
    NIFTY-50 top-weight heavyweights trading below VWAP at 13:00-14:00 IST
    that reclaim VWAP after 14:30 on rising volume see passive ETF +
    index-rebalance close-flow push price 0.3-0.5% above VWAP into 15:15.

Footprint measurement only — NO fees, NO leverage, NO exits, NO PF.
Output: per-event CSV + signed-mean drift delta vs baseline, with
regime/period splits.

Anti-bias guards (Lesson #5):
  1. VWAP computed cumulatively from bars[:i+1] — no day-aggregate look-ahead
  2. Volume baseline excludes current bar (cumulative .shift(1) equivalent)
  3. No exits modeled (Phase 2 is footprint only, not validation)
  4. ProductionUniverseGate used for per-date universe (Lesson #19)
"""
from __future__ import annotations

import sys
from datetime import date, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from tools.sub9_research.production_universe import ProductionUniverseGate  # noqa: E402

# ---------------------------------------------------------------------------
# CONFIG  —  NO hardcoded defaults inside the logic; every knob declared here.
# ---------------------------------------------------------------------------
CONFIG = {
    # Window
    "START_DATE": date(2023, 1, 2),
    "END_DATE":   date(2026, 4, 30),

    # Universe inputs
    "HEAVYWEIGHTS_CSV": _REPO_ROOT / "assets" / "nifty_heavyweights.csv",
    "ACCEPTED_CAPS":    {"large_cap"},   # large_cap-only per brief
    "REQUIRE_MIS":      True,            # production universe builder default

    # Eligibility window (must be entirely below VWAP at every bar close)
    "ELIG_WINDOW_START": dtime(13, 0),    # inclusive
    "ELIG_WINDOW_END":   dtime(14, 0),    # exclusive (bars 13:00, 13:05, ..., 13:55)

    # Signal-scan window
    "SIG_WINDOW_START":  dtime(14, 30),   # inclusive
    "SIG_WINDOW_END":    dtime(15, 0),    # exclusive (bars 14:30, 14:35, ..., 14:55)

    # Signal triggers
    "VOL_RATIO_MIN":     1.3,             # vol_curr / mean(vol_prior) >= 1.3

    # Target bar (close-of-day price for ret_to_1515 measurement)
    # 5m bar labeled 15:10 = covers 15:10-15:15, its close IS the 15:15 mark.
    "TARGET_BAR_TIME":   dtime(15, 10),

    # Baseline-only fallback "signal_bar_close" for eligible-no-trigger rows:
    "BASELINE_BAR_TIME": dtime(14, 30),

    # Regime cutoffs
    "REGIME_2024_CUT":   date(2024, 1, 1),
    "PERIOD_20260527":   date(2026, 5, 27),

    # Acceptance criteria (Stage 2 floor per setup_lifecycle.md)
    "MIN_DRIFT_DELTA_PCT": 0.10,
    "MIN_N_SIGNAL":        200,

    # Output paths
    "OUT_CSV": _REPO_ROOT / "reports" / "sub9_sanity" / "_phase2_nifty_heavy_vwap_reclaim_signature.csv",
    "MONTHLY_DIR": _REPO_ROOT / "backtest-cache-download" / "monthly",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_heavyweights() -> list[str]:
    df = pd.read_csv(CONFIG["HEAVYWEIGHTS_CSV"])
    return [str(s).strip().upper() for s in df["symbol"].tolist()]


def _months_in_window() -> list[tuple[int, int]]:
    months = []
    y, m = CONFIG["START_DATE"].year, CONFIG["START_DATE"].month
    end_y, end_m = CONFIG["END_DATE"].year, CONFIG["END_DATE"].month
    while (y, m) <= (end_y, end_m):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def _ensure_naive_ist(ts_col: pd.Series) -> pd.Series:
    """Drop tz from a datetime column (legacy data may be tz-aware)."""
    if pd.api.types.is_datetime64tz_dtype(ts_col):
        return ts_col.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return ts_col


def _compute_cumulative_vwap(bars: pd.DataFrame) -> np.ndarray:
    """
    VWAP[i] = sum(typical_price[:i+1] * volume[:i+1]) / sum(volume[:i+1])
    No look-ahead — typical_price uses bar's own H/L/C.
    Returns float ndarray same length as bars.
    """
    tp = (bars["high"].to_numpy() + bars["low"].to_numpy() + bars["close"].to_numpy()) / 3.0
    vol = bars["volume"].to_numpy().astype(np.float64)
    pv_cum = np.cumsum(tp * vol)
    v_cum  = np.cumsum(vol)
    # protect against zero-volume openings (shouldn't happen for heavyweights)
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.where(v_cum > 0, pv_cum / v_cum, np.nan)
    return vwap


def _evaluate_symbol_day(bars: pd.DataFrame, sym: str, d: date) -> dict | None:
    """
    Returns one record (dict) or None if neither eligible+signal NOR eligible+baseline.

    Logic:
      1. cumulative VWAP at each bar (using bars[:i+1])
      2. eligibility: ALL bars in [13:00, 14:00) close < vwap_at_bar
      3. signal scan: first bar in [14:30, 15:00) with close >= vwap_at_bar AND
         vol_ratio = vol_curr / mean(vol of bars BEFORE current bar today) >= 1.3
      4. if signal found → kind='signal', signal_bar_close = signal bar close
         else (eligible only) → kind='baseline', signal_bar_close = 14:30 bar close
      5. ret_to_1515 = (close_at_15:10_bar - signal_bar_close) / signal_bar_close * 100
    """
    bars = bars.sort_values("date").reset_index(drop=True)
    if bars.empty:
        return None

    # Restrict to regular session (drop pre-open / extended bars if any)
    bars = bars[(bars["date"].dt.time >= dtime(9, 15)) & (bars["date"].dt.time <= dtime(15, 25))]
    if bars.empty:
        return None
    bars = bars.reset_index(drop=True)

    vwap = _compute_cumulative_vwap(bars)
    closes = bars["close"].to_numpy()
    vols = bars["volume"].to_numpy().astype(np.float64)
    times = bars["date"].dt.time.to_numpy()

    # ------- Eligibility check (13:00-14:00) -------
    elig_mask = np.array([
        (CONFIG["ELIG_WINDOW_START"] <= t < CONFIG["ELIG_WINDOW_END"])
        for t in times
    ])
    if not elig_mask.any():
        return None  # missing data in eligibility window
    if not np.all(closes[elig_mask] < vwap[elig_mask]):
        return None  # not eligible — at least one bar was at/above VWAP

    # ------- Target bar close (15:10) -------
    tgt_idx = np.where(times == CONFIG["TARGET_BAR_TIME"])[0]
    if tgt_idx.size == 0:
        return None
    target_close = float(closes[tgt_idx[0]])

    # ------- Signal scan (14:30-15:00) -------
    sig_window_idx = np.where(
        np.array([
            (CONFIG["SIG_WINDOW_START"] <= t < CONFIG["SIG_WINDOW_END"])
            for t in times
        ])
    )[0]
    if sig_window_idx.size == 0:
        return None

    signal_idx = None
    vol_ratio_at_signal = None
    for i in sig_window_idx:
        prior_vol = vols[:i]
        if prior_vol.size == 0:
            continue
        prior_mean = float(prior_vol.mean())
        if prior_mean <= 0:
            continue
        vol_ratio = float(vols[i] / prior_mean)
        if closes[i] >= vwap[i] and vol_ratio >= CONFIG["VOL_RATIO_MIN"]:
            signal_idx = int(i)
            vol_ratio_at_signal = vol_ratio
            break

    # Regime / period buckets
    regime = "post" if d >= CONFIG["REGIME_2024_CUT"] else "pre"
    period = "post" if d >= CONFIG["PERIOD_20260527"] else "pre"

    if signal_idx is not None:
        sig_close = float(closes[signal_idx])
        ret_to_1515 = (target_close - sig_close) / sig_close * 100.0
        return {
            "kind": "signal",
            "symbol": sym,
            "date": d.isoformat(),
            "signal_bar_ts": pd.Timestamp(bars["date"].iloc[signal_idx]).isoformat(),
            "signal_bar_close": sig_close,
            "vwap_at_signal": float(vwap[signal_idx]),
            "vol_ratio": vol_ratio_at_signal,
            "ret_to_1515": ret_to_1515,
            "regime_pre_post_2024": regime,
            "period_pre_post_20260527": period,
        }

    # ------- Baseline (eligible, no signal trigger) -------
    base_idx = np.where(times == CONFIG["BASELINE_BAR_TIME"])[0]
    if base_idx.size == 0:
        return None
    bi = int(base_idx[0])
    base_close = float(closes[bi])
    ret_to_1515 = (target_close - base_close) / base_close * 100.0
    return {
        "kind": "baseline",
        "symbol": sym,
        "date": d.isoformat(),
        "signal_bar_ts": pd.Timestamp(bars["date"].iloc[bi]).isoformat(),
        "signal_bar_close": base_close,
        "vwap_at_signal": float(vwap[bi]),
        "vol_ratio": float("nan"),
        "ret_to_1515": ret_to_1515,
        "regime_pre_post_2024": regime,
        "period_pre_post_20260527": period,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 70)
    print("Phase 2 empirical signature — nifty_heavy_vwap_reclaim_long")
    print("=" * 70)
    print(f"Window: {CONFIG['START_DATE']} -> {CONFIG['END_DATE']}")
    print(f"Drift acceptance threshold: +{CONFIG['MIN_DRIFT_DELTA_PCT']:.2f}%")
    print(f"Min n_signal: {CONFIG['MIN_N_SIGNAL']}")

    heavyweights = _load_heavyweights()
    print(f"Heavyweights loaded: {len(heavyweights)} symbols  {heavyweights}")

    gate = ProductionUniverseGate(
        accepted_caps=CONFIG["ACCEPTED_CAPS"],
        require_mis=CONFIG["REQUIRE_MIS"],
        min_trading_days_required=0,   # Lesson #17
        min_daily_avg_volume=0,        # Lesson #17
    )

    records: list[dict] = []
    months = _months_in_window()
    print(f"Months to process: {len(months)} (from {months[0]} to {months[-1]})")

    for (yy, mm) in months:
        path = CONFIG["MONTHLY_DIR"] / f"{yy:04d}_{mm:02d}_5m_enriched.feather"
        if not path.exists():
            print(f"  [SKIP] {path.name} not on disk")
            continue
        df = pd.read_feather(
            path,
            columns=["symbol", "date", "open", "high", "low", "close", "volume"],
        )
        df["date"] = _ensure_naive_ist(df["date"])
        # Filter universe early to keep this tractable
        df = df[df["symbol"].isin(heavyweights)]
        if df.empty:
            print(f"  [EMPTY] {path.name} — no heavyweight bars")
            continue

        df["day"] = df["date"].dt.date
        # Apply per-(sym, day) date-range filter
        mask = (df["day"] >= CONFIG["START_DATE"]) & (df["day"] <= CONFIG["END_DATE"])
        df = df[mask]

        n_evals = 0
        n_signal = 0
        n_baseline = 0
        for sym, sym_df in df.groupby("symbol"):
            for day_d, day_bars in sym_df.groupby("day"):
                # ProductionUniverseGate per-date eligibility (Lesson #19)
                if not gate.is_eligible(sym, day_d):
                    continue
                n_evals += 1
                rec = _evaluate_symbol_day(day_bars, sym, day_d)
                if rec is None:
                    continue
                records.append(rec)
                if rec["kind"] == "signal":
                    n_signal += 1
                else:
                    n_baseline += 1
        print(f"  {path.name}: evals={n_evals}  signal={n_signal}  baseline={n_baseline}")

    if not records:
        print("\nNO RECORDS COLLECTED — abort.")
        return 1

    out_df = pd.DataFrame.from_records(records)
    CONFIG["OUT_CSV"].parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(CONFIG["OUT_CSV"], index=False)
    print(f"\nWrote {len(out_df):,} rows -> {CONFIG['OUT_CSV']}")

    # ---------------- Reporting ----------------
    sig = out_df[out_df["kind"] == "signal"]
    base = out_df[out_df["kind"] == "baseline"]

    def _agg(df_in: pd.DataFrame) -> tuple[int, float]:
        n = int(len(df_in))
        mean = float(df_in["ret_to_1515"].mean()) if n > 0 else float("nan")
        return n, mean

    n_sig, mu_sig = _agg(sig)
    n_base, mu_base = _agg(base)
    delta = mu_sig - mu_base if (n_sig > 0 and n_base > 0) else float("nan")

    print()
    print("## Phase 2 Empirical Signature — nifty_heavy_vwap_reclaim_long")
    print()
    print("### Aggregate")
    print(f"- Signal events (n): {n_sig}")
    print(f"- Baseline events (n): {n_base}")
    print(f"- Signal mean ret_to_1515: {mu_sig:+.4f}%")
    print(f"- Baseline mean ret_to_1515: {mu_base:+.4f}%")
    print(f"- DRIFT DELTA: {delta:+.4f}%")
    print()

    # Pre/Post 2024
    print("### Pre/Post 2024 split")
    print("| Cohort | n_signal | n_baseline | signal_mean | baseline_mean | delta |")
    print("|---|---|---|---|---|---|")
    for cohort in ("pre", "post"):
        s = sig[sig["regime_pre_post_2024"] == cohort]
        b = base[base["regime_pre_post_2024"] == cohort]
        ns, ms = _agg(s)
        nb, mb = _agg(b)
        d = ms - mb if (ns > 0 and nb > 0) else float("nan")
        print(f"| {cohort}_2024 | {ns} | {nb} | {ms:+.4f}% | {mb:+.4f}% | {d:+.4f}% |")
    print()

    # Pre/Post 2026-05-27
    print("### Pre/Post 2026-05-27 split (NSE methodology change)")
    print("| Cohort | n_signal | n_baseline | signal_mean | baseline_mean | delta |")
    print("|---|---|---|---|---|---|")
    for cohort in ("pre", "post"):
        s = sig[sig["period_pre_post_20260527"] == cohort]
        b = base[base["period_pre_post_20260527"] == cohort]
        ns, ms = _agg(s)
        nb, mb = _agg(b)
        d = ms - mb if (ns > 0 and nb > 0) else float("nan")
        print(f"| {cohort}_20260527 | {ns} | {nb} | {ms:+.4f}% | {mb:+.4f}% | {d:+.4f}% |")
    print()

    # Per-symbol breakdown (signal only)
    print("### Per-symbol signal breakdown")
    if n_sig > 0:
        per_sym = sig.groupby("symbol")["ret_to_1515"].agg(["count", "mean", "median"]).round(4)
        per_sym = per_sym.sort_values("count", ascending=False)
        print(per_sym.to_string())
    print()

    # Verdict
    passed_drift = (not np.isnan(delta)) and (delta >= CONFIG["MIN_DRIFT_DELTA_PCT"])
    passed_n = n_sig >= CONFIG["MIN_N_SIGNAL"]
    print("### Verdict")
    if passed_drift and passed_n:
        print(f"PROCEED to Phase 3 — drift {delta:+.4f}% >= floor {CONFIG['MIN_DRIFT_DELTA_PCT']:+.2f}% AND n_signal {n_sig} >= {CONFIG['MIN_N_SIGNAL']}")
    else:
        reasons = []
        if not passed_drift:
            reasons.append(f"drift {delta:+.4f}% < floor {CONFIG['MIN_DRIFT_DELTA_PCT']:+.2f}%")
        if not passed_n:
            reasons.append(f"n_signal {n_sig} < {CONFIG['MIN_N_SIGNAL']}")
        print(f"KILL — {' AND '.join(reasons)}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
