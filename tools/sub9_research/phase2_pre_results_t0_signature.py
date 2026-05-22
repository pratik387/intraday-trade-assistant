# tools/sub9_research/phase2_pre_results_t0_signature.py
#
# Phase 2 empirical signature for pre_results_t0_morning_accumulation_fade_short candidate.
# See specs/2026-05-22-brief-pre_results_t0_morning_accumulation_fade_short.md
#
# Anti-bias guards (Lesson #5):
#   1. morning_high computed from bars[:i] (rolling, EXCLUDES current bar to make the
#      "close >= morning_high * 1.005" breakout structurally meaningful) - no look-ahead
#   2. Volume baseline excludes current bar
#   3. {AMC, scheduled} v2 filter from Day 1 (Lesson #11)
#   4. ProductionUniverseGate used for per-date universe (Lesson #19)
#   5. Source-priority audit printed before signature run
#
# Mechanism (from brief):
#   On AMC results-day, F&O underlyings with announced AMC results that print a
#   10:00-11:00 retail-FOMO accumulation move (price >= morning_high * 1.005 with
#   vol_ratio >= 1.3x cumulative-prior-mean) get SHORT-faded in 11:30-13:00.
#
# Acceptance:
#   - DRIFT DELTA (signal_mean - baseline_mean) <= -0.15% (SHORT direction)
#   - n_signal >= 200
#   - No cohort sign-flip on post-SEBI-Oct-2025 cohort
"""Phase 2 empirical signature - pre_results_t0_morning_accumulation_fade_short."""
from __future__ import annotations

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
# CONFIG (no hardcoded defaults in trading logic - all explicit here)
# -----------------------------------------------------------------------------
CONFIG: Dict[str, object] = {
    # Date window
    "window_start": date(2023, 1, 2),
    "window_end": date(2026, 4, 30),

    # F&O universe proxy
    "top_n_adv": 200,                       # top-N symbols by ADV at each session start
    "adv_lookback_days": 20,                # 20-day rolling ADV
    "accepted_caps": {"large_cap", "mid_cap", "unknown"},
    "require_mis": True,                    # production-aligned

    # Signal definition
    "signal_window_start": dtime(10, 0),    # 10:00 inclusive
    "signal_window_end": dtime(11, 0),      # 11:00 exclusive (10:55 bar is last)
    "price_breakout_ratio": 1.005,          # close >= morning_high * 1.005
    "vol_ratio_threshold": 1.3,             # volume / prior-mean >= 1.3
    # Baseline anchor (for symbol-days with NO signal)
    "baseline_anchor": dtime(11, 0),        # use 11:00 5m bar close
    # Target horizon
    "target_close_time": dtime(12, 55),     # 12:55 bar close == "13:00" for 5m bars

    # Earnings filter (v2)
    "amc_classes": ("AMC", "scheduled"),

    # Regime splits
    "regime_2024_cut": date(2024, 1, 1),    # passive-AUM-explosion concern
    "sebi_oct2025_cut": date(2025, 10, 1),  # SEBI Oct 2025 cutover

    # Acceptance
    "drift_delta_max": -0.15,               # signal_mean - baseline_mean must be <= -0.15%
    "n_signal_min": 200,

    # Paths
    "earnings_path": _REPO_ROOT / "data" / "earnings_calendar" / "earnings_events.parquet",
    "daily_path": _REPO_ROOT / "backtest-cache-download" / "consolidated_daily.feather",
    "monthly_5m_dir": _REPO_ROOT / "backtest-cache-download" / "monthly",
    "out_csv": _REPO_ROOT / "reports" / "sub9_sanity" / "_phase2_pre_results_t0_signature.csv",
    "audit_csv": _REPO_ROOT / "reports" / "sub9_sanity" / "_phase2_pre_results_t0_source_audit.csv",
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _normalize_symbol(s: str) -> str:
    """Strip NSE: prefix and .NS suffix to a bare symbol."""
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


# -----------------------------------------------------------------------------
# Source-priority audit
# -----------------------------------------------------------------------------
def run_source_audit() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Print per-month source x announce_time_class table. Returns (full_audit, v2_recovery)."""
    earnings_path = CONFIG["earnings_path"]
    df = pd.read_parquet(earnings_path)
    df["announce_date"] = pd.to_datetime(df["announce_date"])
    df["year_month"] = df["announce_date"].dt.to_period("M")

    audit = (
        df.groupby(["year_month", "source", "announce_time_class"])
        .size()
        .unstack(fill_value=0)
    )

    print("\n=== Source-priority audit (per year-month) ===")
    print(audit.tail(60).to_string())

    out_path = Path(CONFIG["audit_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out_path)
    print(f"\nFull audit saved: {out_path}")

    # Recovery check: v2 filter must produce >= 200 AMC/scheduled events per year (2023-2026)
    df["year"] = df["announce_date"].dt.year
    v2 = df[df["announce_time_class"].isin(CONFIG["amc_classes"])].copy()
    recovery = v2.groupby(["year", "source"]).size().unstack(fill_value=0)
    print("\n=== v2 {AMC, scheduled} recovery per year x source ===")
    print(recovery.to_string())

    return audit, recovery


# -----------------------------------------------------------------------------
# Earnings filter
# -----------------------------------------------------------------------------
def load_amc_earnings() -> pd.DataFrame:
    """Return DataFrame[symbol_bare, announce_date, announce_time_class, source]
    for AMC + scheduled events in window."""
    df = pd.read_parquet(CONFIG["earnings_path"])
    df["announce_date"] = pd.to_datetime(df["announce_date"]).dt.date
    df = df[df["announce_time_class"].isin(CONFIG["amc_classes"])].copy()
    df = df[(df["announce_date"] >= CONFIG["window_start"]) & (df["announce_date"] <= CONFIG["window_end"])].copy()
    df["symbol_bare"] = df["symbol"].apply(_normalize_symbol)
    # Some symbols may have multiple rows (financial_results + announcements_bmo).
    # For source attribution we keep highest-priority (financial_results > announcements_fr >
    # announcements_bmo > board_meetings) when duplicated on same (sym, date).
    src_priority = {
        "financial_results": 0,
        "announcements_fr": 1,
        "announcements_bmo": 2,
        "board_meetings": 3,
    }
    df["src_rank"] = df["source"].map(src_priority).fillna(99).astype(int)
    df = df.sort_values(["symbol_bare", "announce_date", "src_rank"])
    df = df.drop_duplicates(subset=["symbol_bare", "announce_date"], keep="first")
    print(f"Loaded {len(df):,} AMC/scheduled earnings events for {CONFIG['window_start']}..{CONFIG['window_end']}")
    return df[["symbol_bare", "announce_date", "announce_time_class", "source"]]


# -----------------------------------------------------------------------------
# F&O proxy universe: top-N by ADV
# -----------------------------------------------------------------------------
def build_adv_universe() -> Dict[date, set]:
    """Return {session_date -> set of top-N bare symbols by 20-day ADV}.

    ADV = rolling mean of (close * volume) over `adv_lookback_days` prior trading days.
    Computed strictly on dates < session_date (no look-ahead).
    """
    print("\n=== Building per-date top-N ADV universe ===")
    df = pd.read_feather(CONFIG["daily_path"])
    df["d"] = pd.to_datetime(df["ts"]).dt.date
    df["turnover"] = df["close"].astype(float) * df["volume"].astype(float)
    # Restrict to a slightly wider window so we can compute the lookback rolling
    pad_start = CONFIG["window_start"]
    # rolling 20-day -> we need ~30 trading days before window_start to start clean
    df = df[df["d"] <= CONFIG["window_end"]].copy()
    df = df.sort_values(["symbol", "d"])
    # Rolling ADV per symbol
    df["adv"] = (
        df.groupby("symbol")["turnover"]
        .rolling(window=CONFIG["adv_lookback_days"], min_periods=CONFIG["adv_lookback_days"])
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Build per-date universe: ADV ranked using DATA UP TO d-1 (prior day's rolling window).
    # The 20-day window ending on d itself uses d as a member -> shift one day to avoid look-ahead.
    df["adv_shifted"] = df.groupby("symbol")["adv"].shift(1)

    # Filter to study window + drop NaN ADVs
    sub = df[(df["d"] >= pad_start) & df["adv_shifted"].notna()].copy()

    out: Dict[date, set] = {}
    top_n = int(CONFIG["top_n_adv"])
    for d, grp in sub.groupby("d"):
        top = grp.nlargest(top_n, "adv_shifted")
        out[d] = set(top["symbol"].tolist())
    print(f"  built ADV universe for {len(out):,} session dates (top-{top_n} each)")
    return out


# -----------------------------------------------------------------------------
# 5m bar processing
# -----------------------------------------------------------------------------
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
    """For a single (symbol, day) DataFrame of intraday 5m bars (sorted asc by time),
    detect the FIRST signal in [10:00, 11:00) and compute ret_to_1300.

    If no signal triggers, returns a 'baseline' record using 11:00 bar close as anchor.
    Returns None if neither anchor is present (e.g., partial-day data).
    """
    # Sort safety
    sym_bars = sym_bars.sort_values("date").reset_index(drop=True)
    if sym_bars.empty:
        return None

    target_t = CONFIG["target_close_time"]
    # 12:55 5m bar close == represents the 12:55-13:00 candle close
    target_bars = sym_bars[sym_bars["time"] == target_t]
    if target_bars.empty:
        return None
    target_close = float(target_bars.iloc[0]["close"])

    # Build cumulative arrays for ANY bar i: high_max[0..i], vol_baseline (mean over [0..i-1])
    # Then evaluate in [10:00, 11:00) the FIRST i meeting the trigger.
    times = sym_bars["time"].tolist()
    highs = sym_bars["high"].to_numpy(dtype=float)
    closes = sym_bars["close"].to_numpy(dtype=float)
    vols = sym_bars["volume"].to_numpy(dtype=float)

    # rolling morning_high from PRIOR bars only (no look-ahead, no self-reference).
    # By construction close <= high <= cum_max_so_far, so a "close >= morning_high*1.005"
    # breakout requires comparing against the max of bars STRICTLY before the current one.
    cum_high = np.maximum.accumulate(highs)
    morning_high = np.full_like(highs, np.nan, dtype=float)
    if len(highs) > 1:
        morning_high[1:] = cum_high[:-1]  # at bar i: max(high[0..i-1])

    # vol_baseline at bar i = mean(vols[0..i-1]) (excludes current bar)
    cum_vol = np.cumsum(vols)
    # at index i, sum of bars [0..i-1] = cum_vol[i-1]; for i==0 undefined.
    vol_baseline = np.full_like(vols, np.nan, dtype=float)
    if len(vols) > 1:
        # mean of prior bars = cum_vol[i-1] / i
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
            "ret_to_1300": ret,
            "target_close_1300": target_close,
            "announce_time_class_at_signal": earnings_row["announce_time_class"],
            "source_of_announcement": earnings_row["source"],
        }

    # Baseline: use 11:00 bar close (no signal triggered in 10:00-11:00 window)
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
        "signal_bar_close": anchor_close,  # uses 11:00 anchor for baseline
        "morning_high": None,
        "vol_ratio": None,
        "ret_to_1300": ret,
        "target_close_1300": target_close,
        "announce_time_class_at_signal": earnings_row["announce_time_class"],
        "source_of_announcement": earnings_row["source"],
    }


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def run() -> int:
    print("=" * 80)
    print("Phase 2 empirical signature - pre_results_t0_morning_accumulation_fade_short")
    print(f"Window: {CONFIG['window_start']} -> {CONFIG['window_end']}")
    print("=" * 80)

    # --- Source-priority audit (Lesson #11) ---
    audit_df, v2_recovery = run_source_audit()

    # Audit verdict
    v2_2025 = int(v2_recovery.loc[2025].sum()) if 2025 in v2_recovery.index else 0
    v2_2026 = int(v2_recovery.loc[2026].sum()) if 2026 in v2_recovery.index else 0
    if v2_2025 < 200 or v2_2026 < 200:
        print(f"\nAUDIT WARNING: 2025 AMC events n={v2_2025}, 2026 n={v2_2026}")
        print("Low recovery may indicate another source rotation broke the v2 filter.")
    else:
        print(f"\n2025+ AMC events recoverable via v2 filter: YES (2025: {v2_2025}, 2026: {v2_2026})")

    # --- Load earnings ---
    earn = load_amc_earnings()
    if earn.empty:
        print("ERROR: no AMC earnings events in window after v2 filter. STOP.")
        return 2

    # --- ProductionUniverseGate (Lesson #19) ---
    gate = ProductionUniverseGate(
        accepted_caps=CONFIG["accepted_caps"],
        require_mis=CONFIG["require_mis"],
        min_trading_days_required=0,
        min_daily_avg_volume=0,
    )

    # --- ADV proxy F&O universe ---
    adv_universe = build_adv_universe()

    # --- Iterate by month to keep memory in check ---
    months = _months_between(CONFIG["window_start"], CONFIG["window_end"])
    earn_by_month: Dict[Tuple[int, int], pd.DataFrame] = {}
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
            print(f"  skipping {yy:04d}-{mm:02d} - no 5m feather")
            continue
        print(f"  processing {yy:04d}-{mm:02d}: {len(rows_this_month)} AMC earnings rows", flush=True)

        # Filter earnings rows down to those whose symbol is in the ADV universe on that date
        # AND that passes ProductionUniverseGate
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
            print(f"    no eligible (sym,date) in {yy:04d}-{mm:02d}")
            continue

        # Load 5m feather, restrict to needed symbols and dates
        eligible_syms = {r["symbol_bare"] for r in eligible_rows}
        df5 = _load_month_5m(yy, mm)
        if df5 is None:
            continue
        df5 = df5[df5["symbol"].isin(eligible_syms)]
        if df5.empty:
            print(f"    no 5m data for eligible symbols in {yy:04d}-{mm:02d}")
            continue

        # Group by (symbol, day) once for fast lookup
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
    print(f"  total earnings rows in window:     {len(earn):,}")
    print(f"  evaluated (in ADV univ & MIS):     {total_evaluated:,}")
    print(f"  rejected (universe/gate):          {total_rejected_universe:,}")
    print(f"  missing 5m bars:                   {total_no_bars:,}")
    print(f"  missing 11:00 or 12:55 anchor:     {total_no_target:,}")
    print(f"  recorded:                          {len(records):,}")

    if not records:
        print("\nERROR: no records collected. STOP.")
        return 2

    df = pd.DataFrame(records)

    # Save CSV
    out_path = Path(CONFIG["out_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Add helpful columns
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df):,} rows -> {out_path}")

    # ---- Summary ----
    sig = df[df["is_signal"]]
    base = df[~df["is_signal"]]
    n_sig = len(sig)
    n_base = len(base)
    sig_mean = sig["ret_to_1300"].mean() if n_sig else float("nan")
    base_mean = base["ret_to_1300"].mean() if n_base else float("nan")
    delta = sig_mean - base_mean if (n_sig and n_base) else float("nan")
    sig_median = sig["ret_to_1300"].median() if n_sig else float("nan")
    base_median = base["ret_to_1300"].median() if n_base else float("nan")

    print("\n" + "=" * 80)
    print("AGGREGATE")
    print("=" * 80)
    print(f"  Signal events (n):      {n_sig:,}")
    print(f"  Baseline events (n):    {n_base:,}")
    print(f"  Signal mean ret_to_1300:  {sig_mean:+.4f}%  (median {sig_median:+.4f}%)")
    print(f"  Baseline mean ret_to_1300:{base_mean:+.4f}%  (median {base_median:+.4f}%)")
    print(f"  DRIFT DELTA (signal-baseline): {delta:+.4f}%  [must be <= {CONFIG['drift_delta_max']:.2f}% for SHORT]")

    # ---- Splits ----
    def split_block(name: str, mask_sig: pd.Series, mask_base: pd.Series) -> dict:
        s = sig[mask_sig]
        b = base[mask_base]
        ns, nb = len(s), len(b)
        sm = s["ret_to_1300"].mean() if ns else float("nan")
        bm = b["ret_to_1300"].mean() if nb else float("nan")
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

    # cap split: re-derive cap from nse_all via gate's internal map
    cap_lookup_all = gate._load_nse_all()
    def _cap(s):
        row = cap_lookup_all.get(s)
        return row.cap_segment if row else "unknown"
    sig_cap = sig["symbol"].map(_cap)
    base_cap = base["symbol"].map(_cap)
    for cap_val in ["large_cap", "mid_cap", "unknown"]:
        splits.append(split_block(f"cap={cap_val}", sig_cap == cap_val, base_cap == cap_val))

    print("\n" + "=" * 80)
    print("SPLITS (drift delta per cohort)")
    print("=" * 80)
    print(f"{'split':<28}{'n_sig':>8}{'n_base':>8}{'sig_mean':>12}{'base_mean':>12}{'delta':>12}")
    for r in splits:
        print(
            f"{r['split']:<28}"
            f"{r['n_signal']:>8d}"
            f"{r['n_baseline']:>8d}"
            f"{r['signal_mean']:>12.4f}"
            f"{r['baseline_mean']:>12.4f}"
            f"{r['delta']:>12.4f}"
        )

    # ---- Verdict ----
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    drift_ok = (not np.isnan(delta)) and (delta <= CONFIG["drift_delta_max"])
    n_ok = n_sig >= int(CONFIG["n_signal_min"])

    # Sign flip in post-SEBI cohort?
    post_sebi = next((s for s in splits if s["split"] == "post_sebi_oct2025"), None)
    sign_flip = False
    if post_sebi and not np.isnan(post_sebi["delta"]):
        if post_sebi["delta"] > 0:
            sign_flip = True

    verdict = "UNKNOWN"
    reason = ""
    if not drift_ok:
        verdict = "KILL"
        reason = f"net delta {delta:+.4f}% > {CONFIG['drift_delta_max']:.2f}% (insufficient SHORT footprint)"
    elif not n_ok:
        verdict = "KILL"
        reason = f"n_signal {n_sig} < {CONFIG['n_signal_min']}"
    elif sign_flip:
        verdict = "DEFER"
        reason = f"post-SEBI-Oct-2025 cohort delta is POSITIVE ({post_sebi['delta']:+.4f}%) - regulatory regime risk"
    else:
        verdict = "PROCEED to Phase 3"
        reason = f"net delta {delta:+.4f}% (<= {CONFIG['drift_delta_max']:.2f}%), n_signal {n_sig} (>= {CONFIG['n_signal_min']}), no post-SEBI sign flip"

    print(f"  {verdict}")
    print(f"  reason: {reason}")

    return 0


if __name__ == "__main__":
    sys.exit(run())
