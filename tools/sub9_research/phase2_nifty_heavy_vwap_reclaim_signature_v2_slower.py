"""
tools/sub9_research/phase2_nifty_heavy_vwap_reclaim_signature_v2_slower.py

A1 SLOWER-TIMESCALE VARIANT of phase2_nifty_heavy_vwap_reclaim_signature.py.

Only difference from the original: target measurement bar shifted from the
15:10 5m bar (close = 15:15 IST) to the 15:20 5m bar (close = 15:25 IST,
the LAST 5m bar of the regular session).

All other logic (eligibility, signal scan, anti-bias guards, universe,
production gate, regime/period splits) is BYTE-IDENTICAL.

Acceptance unchanged: drift delta >= +0.10% (Stage 2 floor).
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
    "ACCEPTED_CAPS":    {"large_cap"},
    "REQUIRE_MIS":      True,

    # Eligibility window
    "ELIG_WINDOW_START": dtime(13, 0),
    "ELIG_WINDOW_END":   dtime(14, 0),

    # Signal-scan window
    "SIG_WINDOW_START":  dtime(14, 30),
    "SIG_WINDOW_END":    dtime(15, 0),

    # Signal triggers
    "VOL_RATIO_MIN":     1.3,

    # *** ONLY DIFFERENCE FROM ORIGINAL ***
    # Target bar = 15:20 5m bar (covers 15:20-15:25, its close IS 15:25 IST,
    # the LAST 5m bar of the regular session).
    "TARGET_BAR_TIME":   dtime(15, 20),

    # Baseline-only fallback "signal_bar_close" for eligible-no-trigger rows:
    "BASELINE_BAR_TIME": dtime(14, 30),

    # Regime cutoffs
    "REGIME_2024_CUT":   date(2024, 1, 1),
    "PERIOD_20260527":   date(2026, 5, 27),

    # Acceptance criteria
    "MIN_DRIFT_DELTA_PCT": 0.10,
    "MIN_N_SIGNAL":        200,

    # Output paths (variant)
    "OUT_CSV": _REPO_ROOT / "reports" / "sub9_sanity" / "_phase2_nifty_heavy_vwap_reclaim_signature_v2_slower.csv",
    "MONTHLY_DIR": _REPO_ROOT / "backtest-cache-download" / "monthly",
}

# Output return column name (kept generic for clarity)
RET_COL = "ret_to_1525"


# ---------------------------------------------------------------------------
# Helpers (verbatim from original)
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
    if pd.api.types.is_datetime64tz_dtype(ts_col):
        return ts_col.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return ts_col


def _compute_cumulative_vwap(bars: pd.DataFrame) -> np.ndarray:
    tp = (bars["high"].to_numpy() + bars["low"].to_numpy() + bars["close"].to_numpy()) / 3.0
    vol = bars["volume"].to_numpy().astype(np.float64)
    pv_cum = np.cumsum(tp * vol)
    v_cum  = np.cumsum(vol)
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.where(v_cum > 0, pv_cum / v_cum, np.nan)
    return vwap


def _evaluate_symbol_day(bars: pd.DataFrame, sym: str, d: date) -> dict | None:
    bars = bars.sort_values("date").reset_index(drop=True)
    if bars.empty:
        return None

    bars = bars[(bars["date"].dt.time >= dtime(9, 15)) & (bars["date"].dt.time <= dtime(15, 25))]
    if bars.empty:
        return None
    bars = bars.reset_index(drop=True)

    vwap = _compute_cumulative_vwap(bars)
    closes = bars["close"].to_numpy()
    vols = bars["volume"].to_numpy().astype(np.float64)
    times = bars["date"].dt.time.to_numpy()

    # Eligibility
    elig_mask = np.array([
        (CONFIG["ELIG_WINDOW_START"] <= t < CONFIG["ELIG_WINDOW_END"])
        for t in times
    ])
    if not elig_mask.any():
        return None
    if not np.all(closes[elig_mask] < vwap[elig_mask]):
        return None

    # Target bar (15:20 close = 15:25 IST)
    tgt_idx = np.where(times == CONFIG["TARGET_BAR_TIME"])[0]
    if tgt_idx.size == 0:
        return None
    target_close = float(closes[tgt_idx[0]])

    # Signal scan
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

    regime = "post" if d >= CONFIG["REGIME_2024_CUT"] else "pre"
    period = "post" if d >= CONFIG["PERIOD_20260527"] else "pre"

    if signal_idx is not None:
        sig_close = float(closes[signal_idx])
        ret_to_target = (target_close - sig_close) / sig_close * 100.0
        return {
            "kind": "signal",
            "symbol": sym,
            "date": d.isoformat(),
            "signal_bar_ts": pd.Timestamp(bars["date"].iloc[signal_idx]).isoformat(),
            "signal_bar_close": sig_close,
            "vwap_at_signal": float(vwap[signal_idx]),
            "vol_ratio": vol_ratio_at_signal,
            RET_COL: ret_to_target,
            "regime_pre_post_2024": regime,
            "period_pre_post_20260527": period,
        }

    base_idx = np.where(times == CONFIG["BASELINE_BAR_TIME"])[0]
    if base_idx.size == 0:
        return None
    bi = int(base_idx[0])
    base_close = float(closes[bi])
    ret_to_target = (target_close - base_close) / base_close * 100.0
    return {
        "kind": "baseline",
        "symbol": sym,
        "date": d.isoformat(),
        "signal_bar_ts": pd.Timestamp(bars["date"].iloc[bi]).isoformat(),
        "signal_bar_close": base_close,
        "vwap_at_signal": float(vwap[bi]),
        "vol_ratio": float("nan"),
        RET_COL: ret_to_target,
        "regime_pre_post_2024": regime,
        "period_pre_post_20260527": period,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 70)
    print("A1 SLOWER VARIANT — nifty_heavy_vwap_reclaim_long @ 15:25 exit")
    print("=" * 70)
    print(f"Window: {CONFIG['START_DATE']} -> {CONFIG['END_DATE']}")
    print(f"Target bar: {CONFIG['TARGET_BAR_TIME']} (5m bar close = 15:25 IST)")
    print(f"Drift acceptance threshold: +{CONFIG['MIN_DRIFT_DELTA_PCT']:.2f}%")

    heavyweights = _load_heavyweights()
    print(f"Heavyweights loaded: {len(heavyweights)} symbols")

    gate = ProductionUniverseGate(
        accepted_caps=CONFIG["ACCEPTED_CAPS"],
        require_mis=CONFIG["REQUIRE_MIS"],
        min_trading_days_required=0,
        min_daily_avg_volume=0,
    )

    records: list[dict] = []
    months = _months_in_window()

    for (yy, mm) in months:
        path = CONFIG["MONTHLY_DIR"] / f"{yy:04d}_{mm:02d}_5m_enriched.feather"
        if not path.exists():
            continue
        df = pd.read_feather(
            path,
            columns=["symbol", "date", "open", "high", "low", "close", "volume"],
        )
        df["date"] = _ensure_naive_ist(df["date"])
        df = df[df["symbol"].isin(heavyweights)]
        if df.empty:
            continue
        df["day"] = df["date"].dt.date
        mask = (df["day"] >= CONFIG["START_DATE"]) & (df["day"] <= CONFIG["END_DATE"])
        df = df[mask]

        n_evals = 0; n_signal = 0; n_baseline = 0
        for sym, sym_df in df.groupby("symbol"):
            for day_d, day_bars in sym_df.groupby("day"):
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

    sig = out_df[out_df["kind"] == "signal"]
    base = out_df[out_df["kind"] == "baseline"]

    def _agg(df_in: pd.DataFrame) -> tuple[int, float]:
        n = int(len(df_in))
        mean = float(df_in[RET_COL].mean()) if n > 0 else float("nan")
        return n, mean

    n_sig, mu_sig = _agg(sig)
    n_base, mu_base = _agg(base)
    delta = mu_sig - mu_base if (n_sig > 0 and n_base > 0) else float("nan")

    print()
    print("## A1 Variant Empirical Signature — nifty_heavy_vwap_reclaim_long @ 15:25 exit")
    print()
    print("### Aggregate")
    print(f"- Signal events (n): {n_sig}")
    print(f"- Baseline events (n): {n_base}")
    print(f"- Signal mean {RET_COL}: {mu_sig:+.4f}%")
    print(f"- Baseline mean {RET_COL}: {mu_base:+.4f}%")
    print(f"- DRIFT DELTA: {delta:+.4f}%")
    print()

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

    print("### Pre/Post 2026-05-27 split")
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

    passed_drift = (not np.isnan(delta)) and (delta >= CONFIG["MIN_DRIFT_DELTA_PCT"])
    passed_n = n_sig >= CONFIG["MIN_N_SIGNAL"]
    print("### Verdict")
    if passed_drift and passed_n:
        print(f"UN-KILL CANDIDATE — drift {delta:+.4f}% >= floor {CONFIG['MIN_DRIFT_DELTA_PCT']:+.2f}% AND n_signal {n_sig} >= {CONFIG['MIN_N_SIGNAL']}")
    else:
        reasons = []
        if not passed_drift:
            reasons.append(f"drift {delta:+.4f}% < floor {CONFIG['MIN_DRIFT_DELTA_PCT']:+.2f}%")
        if not passed_n:
            reasons.append(f"n_signal {n_sig} < {CONFIG['MIN_N_SIGNAL']}")
        print(f"CONFIRM-KILL — {' AND '.join(reasons)}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
