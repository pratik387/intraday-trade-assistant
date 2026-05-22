"""tools/sub9_research/phase2_pre_results_t0_signature_v2_slower.py

B1 + B3 SLOWER-TIMESCALE VARIANTS of phase2_pre_results_t0_signature.py.

Only difference from the original: target measurement bar shifted from the
12:55 5m bar (close = 13:00 IST) to the 14:50 5m bar (close = 14:55 IST,
last 5m bar before the 14:55 intraday MIS-square-off considerations).

All other logic (signal definition, anti-bias guards, AMC v2 filter,
ADV universe, ProductionUniverseGate, regime splits) is BYTE-IDENTICAL.

Reports BOTH:
  - B1: SHORT direction (delta as published); pass if delta <= -0.15%
  - B3: LONG direction (Lesson #1 inverse-edge check); LONG_delta = -delta;
        pass if LONG_delta >= +0.15%
"""
from __future__ import annotations

import sys
from datetime import date, time as dtime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from tools.sub9_research.production_universe import ProductionUniverseGate  # noqa: E402

# -----------------------------------------------------------------------------
# CONFIG (only difference vs original: target_close_time -> 14:50)
# -----------------------------------------------------------------------------
CONFIG: Dict[str, object] = {
    "window_start": date(2023, 1, 2),
    "window_end": date(2026, 4, 30),

    "top_n_adv": 200,
    "adv_lookback_days": 20,
    "accepted_caps": {"large_cap", "mid_cap", "unknown"},
    "require_mis": True,

    "signal_window_start": dtime(10, 0),
    "signal_window_end": dtime(11, 0),
    "price_breakout_ratio": 1.005,
    "vol_ratio_threshold": 1.3,
    "baseline_anchor": dtime(11, 0),

    # *** ONLY DIFFERENCE FROM ORIGINAL ***
    # Target = 14:50 5m bar (covers 14:50-14:55, its close IS 14:55 IST,
    # last bar before intraday MIS square-off).
    "target_close_time": dtime(14, 50),

    "amc_classes": ("AMC", "scheduled"),

    "regime_2024_cut": date(2024, 1, 1),
    "sebi_oct2025_cut": date(2025, 10, 1),

    # SHORT acceptance (B1)
    "drift_delta_max": -0.15,
    "n_signal_min": 200,
    # LONG acceptance (B3, Lesson #1 inverse-edge check)
    "long_drift_delta_min": 0.15,

    "earnings_path": _REPO_ROOT / "data" / "earnings_calendar" / "earnings_events.parquet",
    "daily_path": _REPO_ROOT / "backtest-cache-download" / "consolidated_daily.feather",
    "monthly_5m_dir": _REPO_ROOT / "backtest-cache-download" / "monthly",
    "out_csv": _REPO_ROOT / "reports" / "sub9_sanity" / "_phase2_pre_results_t0_signature_v2_slower.csv",
}

RET_COL = "ret_to_1455"


def _normalize_symbol(s: str) -> str:
    if ":" in s:
        s = s.split(":")[-1]
    if "." in s:
        s = s.split(".")[0]
    return s


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


def load_amc_earnings() -> pd.DataFrame:
    df = pd.read_parquet(CONFIG["earnings_path"])
    df["announce_date"] = pd.to_datetime(df["announce_date"]).dt.date
    df = df[df["announce_time_class"].isin(CONFIG["amc_classes"])].copy()
    df = df[(df["announce_date"] >= CONFIG["window_start"]) & (df["announce_date"] <= CONFIG["window_end"])].copy()
    df["symbol_bare"] = df["symbol"].apply(_normalize_symbol)
    src_priority = {
        "financial_results": 0,
        "announcements_fr": 1,
        "announcements_bmo": 2,
        "board_meetings": 3,
    }
    df["src_rank"] = df["source"].map(src_priority).fillna(99).astype(int)
    df = df.sort_values(["symbol_bare", "announce_date", "src_rank"])
    df = df.drop_duplicates(subset=["symbol_bare", "announce_date"], keep="first")
    print(f"Loaded {len(df):,} AMC/scheduled earnings events")
    return df[["symbol_bare", "announce_date", "announce_time_class", "source"]]


def build_adv_universe() -> Dict[date, set]:
    print("\n=== Building per-date top-N ADV universe ===")
    df = pd.read_feather(CONFIG["daily_path"])
    df["d"] = pd.to_datetime(df["ts"]).dt.date
    df["turnover"] = df["close"].astype(float) * df["volume"].astype(float)
    df = df[df["d"] <= CONFIG["window_end"]].copy()
    df = df.sort_values(["symbol", "d"])
    df["adv"] = (
        df.groupby("symbol")["turnover"]
        .rolling(window=CONFIG["adv_lookback_days"], min_periods=CONFIG["adv_lookback_days"])
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["adv_shifted"] = df.groupby("symbol")["adv"].shift(1)
    sub = df[(df["d"] >= CONFIG["window_start"]) & df["adv_shifted"].notna()].copy()

    out: Dict[date, set] = {}
    top_n = int(CONFIG["top_n_adv"])
    for d, grp in sub.groupby("d"):
        top = grp.nlargest(top_n, "adv_shifted")
        out[d] = set(top["symbol"].tolist())
    print(f"  built ADV universe for {len(out):,} session dates (top-{top_n} each)")
    return out


def _load_month_5m(yy: int, mm: int) -> Optional[pd.DataFrame]:
    path = Path(CONFIG["monthly_5m_dir"]) / f"{yy:04d}_{mm:02d}_5m_enriched.feather"
    if not path.exists():
        return None
    df = pd.read_feather(path, columns=["symbol", "date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df["day"] = df["date"].dt.date
    df["time"] = df["date"].dt.time
    return df


def compute_signals_for_day(
    sym_bars: pd.DataFrame,
    earnings_row: dict,
) -> Optional[dict]:
    sym_bars = sym_bars.sort_values("date").reset_index(drop=True)
    if sym_bars.empty:
        return None

    target_t = CONFIG["target_close_time"]
    target_bars = sym_bars[sym_bars["time"] == target_t]
    if target_bars.empty:
        return None
    target_close = float(target_bars.iloc[0]["close"])

    times = sym_bars["time"].tolist()
    highs = sym_bars["high"].to_numpy(dtype=float)
    closes = sym_bars["close"].to_numpy(dtype=float)
    vols = sym_bars["volume"].to_numpy(dtype=float)

    cum_high = np.maximum.accumulate(highs)
    morning_high = np.full_like(highs, np.nan, dtype=float)
    if len(highs) > 1:
        morning_high[1:] = cum_high[:-1]

    cum_vol = np.cumsum(vols)
    vol_baseline = np.full_like(vols, np.nan, dtype=float)
    if len(vols) > 1:
        idx = np.arange(1, len(vols))
        vol_baseline[1:] = cum_vol[:-1] / idx

    sw_start = CONFIG["signal_window_start"]
    sw_end = CONFIG["signal_window_end"]

    signal_idx = None
    for i, t in enumerate(times):
        if t < sw_start or t >= sw_end:
            continue
        if np.isnan(vol_baseline[i]) or vol_baseline[i] <= 0:
            continue
        if np.isnan(morning_high[i]) or morning_high[i] <= 0:
            continue
        price_ok = closes[i] >= morning_high[i] * CONFIG["price_breakout_ratio"]
        vol_ratio = vols[i] / vol_baseline[i]
        vol_ok = vol_ratio >= CONFIG["vol_ratio_threshold"]
        if price_ok and vol_ok:
            signal_idx = i
            break

    if signal_idx is not None:
        signal_bar_close = float(closes[signal_idx])
        ret = (target_close - signal_bar_close) / signal_bar_close * 100.0
        return {
            "symbol": earnings_row["symbol_bare"],
            "date": earnings_row["announce_date"],
            "is_signal": True,
            "signal_bar_ts": sym_bars.iloc[signal_idx]["date"].to_pydatetime(),
            "signal_bar_time": str(times[signal_idx]),
            "signal_bar_close": signal_bar_close,
            "morning_high": float(morning_high[signal_idx]),
            "vol_ratio": float(vols[signal_idx] / vol_baseline[signal_idx]),
            RET_COL: ret,
            "target_close_1455": target_close,
            "announce_time_class_at_signal": earnings_row["announce_time_class"],
            "source_of_announcement": earnings_row["source"],
        }

    anchor_bars = sym_bars[sym_bars["time"] == CONFIG["baseline_anchor"]]
    if anchor_bars.empty:
        return None
    anchor_close = float(anchor_bars.iloc[0]["close"])
    ret = (target_close - anchor_close) / anchor_close * 100.0
    return {
        "symbol": earnings_row["symbol_bare"],
        "date": earnings_row["announce_date"],
        "is_signal": False,
        "signal_bar_ts": None,
        "signal_bar_time": None,
        "signal_bar_close": anchor_close,
        "morning_high": None,
        "vol_ratio": None,
        RET_COL: ret,
        "target_close_1455": target_close,
        "announce_time_class_at_signal": earnings_row["announce_time_class"],
        "source_of_announcement": earnings_row["source"],
    }


def run() -> int:
    print("=" * 80)
    print("B1+B3 SLOWER VARIANT — pre_results_t0_morning_accumulation @ 14:55 exit")
    print(f"Window: {CONFIG['window_start']} -> {CONFIG['window_end']}")
    print("=" * 80)

    earn = load_amc_earnings()
    if earn.empty:
        print("ERROR: no AMC earnings events in window.")
        return 2

    gate = ProductionUniverseGate(
        accepted_caps=CONFIG["accepted_caps"],
        require_mis=CONFIG["require_mis"],
        min_trading_days_required=0,
        min_daily_avg_volume=0,
    )

    adv_universe = build_adv_universe()

    months = _months_between(CONFIG["window_start"], CONFIG["window_end"])
    earn_by_month: Dict[Tuple[int, int], list] = {}
    for _, row in earn.iterrows():
        d = row["announce_date"]
        key = (d.year, d.month)
        earn_by_month.setdefault(key, []).append(row.to_dict())

    records: List[dict] = []
    total_evaluated = 0
    total_rejected_universe = 0
    total_no_bars = 0
    total_no_target = 0

    for (yy, mm) in months:
        rows_this_month = earn_by_month.get((yy, mm), [])
        if not rows_this_month:
            continue
        path = Path(CONFIG["monthly_5m_dir"]) / f"{yy:04d}_{mm:02d}_5m_enriched.feather"
        if not path.exists():
            continue
        print(f"  processing {yy:04d}-{mm:02d}: {len(rows_this_month)} AMC earnings rows", flush=True)

        eligible_rows = []
        for row in rows_this_month:
            d = row["announce_date"]
            sym = row["symbol_bare"]
            adv_uni = adv_universe.get(d)
            if adv_uni is None or sym not in adv_uni:
                total_rejected_universe += 1
                continue
            if not gate.is_eligible(sym, d):
                total_rejected_universe += 1
                continue
            eligible_rows.append(row)

        if not eligible_rows:
            continue

        eligible_syms = {r["symbol_bare"] for r in eligible_rows}
        df5 = _load_month_5m(yy, mm)
        if df5 is None:
            continue
        df5 = df5[df5["symbol"].isin(eligible_syms)]
        if df5.empty:
            continue

        gb = df5.groupby(["symbol", "day"])
        for row in eligible_rows:
            sym = row["symbol_bare"]
            d = row["announce_date"]
            total_evaluated += 1
            try:
                bars = gb.get_group((sym, d))
            except KeyError:
                total_no_bars += 1
                continue
            rec = compute_signals_for_day(bars, row)
            if rec is None:
                total_no_target += 1
                continue
            records.append(rec)

    print("\n=== Pipeline tally ===")
    print(f"  evaluated (in ADV univ & MIS):     {total_evaluated:,}")
    print(f"  rejected (universe/gate):          {total_rejected_universe:,}")
    print(f"  missing 5m bars:                   {total_no_bars:,}")
    print(f"  missing 11:00 or 14:50 anchor:     {total_no_target:,}")
    print(f"  recorded:                          {len(records):,}")

    if not records:
        print("\nERROR: no records collected.")
        return 2

    df = pd.DataFrame(records)
    out_path = Path(CONFIG["out_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df):,} rows -> {out_path}")

    sig = df[df["is_signal"]]
    base = df[~df["is_signal"]]
    n_sig = len(sig)
    n_base = len(base)
    sig_mean = sig[RET_COL].mean() if n_sig else float("nan")
    base_mean = base[RET_COL].mean() if n_base else float("nan")
    delta = sig_mean - base_mean if (n_sig and n_base) else float("nan")

    print("\n" + "=" * 80)
    print("AGGREGATE")
    print("=" * 80)
    print(f"  Signal events (n):      {n_sig:,}")
    print(f"  Baseline events (n):    {n_base:,}")
    print(f"  Signal mean {RET_COL}:    {sig_mean:+.4f}%")
    print(f"  Baseline mean {RET_COL}:  {base_mean:+.4f}%")
    print(f"  DRIFT DELTA (SHORT, B1):  {delta:+.4f}%  [must be <= {CONFIG['drift_delta_max']:.2f}% for SHORT]")
    print(f"  LONG_DELTA (B3, =-delta): {-delta:+.4f}%  [must be >= +{CONFIG['long_drift_delta_min']:.2f}% for LONG]")

    def split_block(name: str, mask_sig: pd.Series, mask_base: pd.Series) -> dict:
        s = sig[mask_sig]
        b = base[mask_base]
        ns, nb = len(s), len(b)
        sm = s[RET_COL].mean() if ns else float("nan")
        bm = b[RET_COL].mean() if nb else float("nan")
        dl = sm - bm if (ns and nb) else float("nan")
        return {"split": name, "n_signal": ns, "n_baseline": nb, "signal_mean": sm, "baseline_mean": bm, "delta": dl}

    splits: List[dict] = []
    sig_dates = pd.to_datetime(sig["date"])
    base_dates = pd.to_datetime(base["date"])

    cut_2024 = pd.Timestamp(CONFIG["regime_2024_cut"])
    cut_sebi = pd.Timestamp(CONFIG["sebi_oct2025_cut"])

    splits.append(split_block("pre_2024", sig_dates < cut_2024, base_dates < cut_2024))
    splits.append(split_block("post_2024", sig_dates >= cut_2024, base_dates >= cut_2024))
    splits.append(split_block("pre_sebi_oct2025", sig_dates < cut_sebi, base_dates < cut_sebi))
    splits.append(split_block("post_sebi_oct2025", sig_dates >= cut_sebi, base_dates >= cut_sebi))

    for src in ["announcements_fr", "announcements_bmo", "board_meetings", "financial_results"]:
        splits.append(split_block(f"source={src}", sig["source_of_announcement"] == src, base["source_of_announcement"] == src))

    cap_lookup_all = gate._load_nse_all()
    def _cap(s):
        row = cap_lookup_all.get(s)
        return row.cap_segment if row else "unknown"
    sig_cap = sig["symbol"].map(_cap)
    base_cap = base["symbol"].map(_cap)
    for cap_val in ["large_cap", "mid_cap", "unknown"]:
        splits.append(split_block(f"cap={cap_val}", sig_cap == cap_val, base_cap == cap_val))

    print("\n" + "=" * 80)
    print("SPLITS (drift delta per cohort) — SHORT direction (B1)")
    print("=" * 80)
    print(f"{'split':<28}{'n_sig':>8}{'n_base':>8}{'sig_mean':>12}{'base_mean':>12}{'delta':>12}{'LONG_dlt':>12}")
    for r in splits:
        long_d = -r["delta"] if not np.isnan(r["delta"]) else float("nan")
        print(
            f"{r['split']:<28}"
            f"{r['n_signal']:>8d}"
            f"{r['n_baseline']:>8d}"
            f"{r['signal_mean']:>12.4f}"
            f"{r['baseline_mean']:>12.4f}"
            f"{r['delta']:>12.4f}"
            f"{long_d:>12.4f}"
        )

    print("\n" + "=" * 80)
    print("VERDICTS")
    print("=" * 80)
    # B1
    drift_ok = (not np.isnan(delta)) and (delta <= CONFIG["drift_delta_max"])
    n_ok = n_sig >= int(CONFIG["n_signal_min"])
    if not drift_ok or not n_ok:
        b1 = "CONFIRM-KILL"
        why = []
        if not drift_ok:
            why.append(f"SHORT delta {delta:+.4f}% > {CONFIG['drift_delta_max']:.2f}%")
        if not n_ok:
            why.append(f"n {n_sig} < {CONFIG['n_signal_min']}")
        print(f"  B1 (SHORT @ 14:55): {b1}  — {' AND '.join(why)}")
    else:
        post_sebi = next((s for s in splits if s["split"] == "post_sebi_oct2025"), None)
        sign_flip = bool(post_sebi and not np.isnan(post_sebi["delta"]) and post_sebi["delta"] > 0)
        if sign_flip:
            print(f"  B1 (SHORT @ 14:55): DEFER — post-SEBI cohort delta {post_sebi['delta']:+.4f}% positive")
        else:
            print(f"  B1 (SHORT @ 14:55): UN-KILL CANDIDATE — drift {delta:+.4f}%, n={n_sig}")

    # B3 (LONG inverse-edge check)
    long_delta = -delta
    long_ok = (not np.isnan(long_delta)) and (long_delta >= CONFIG["long_drift_delta_min"])
    if long_ok and n_ok:
        print(f"  B3 (LONG @ 14:55):  NEW-LONG-CANDIDATE-FOUND — LONG_delta {long_delta:+.4f}% >= +{CONFIG['long_drift_delta_min']:.2f}%, n={n_sig}")
    else:
        why = []
        if not long_ok:
            why.append(f"LONG_delta {long_delta:+.4f}% < +{CONFIG['long_drift_delta_min']:.2f}%")
        if not n_ok:
            why.append(f"n {n_sig} < {CONFIG['n_signal_min']}")
        print(f"  B3 (LONG @ 14:55):  NO-INVERSE-EDGE — {' AND '.join(why)}")

    return 0


if __name__ == "__main__":
    sys.exit(run())
