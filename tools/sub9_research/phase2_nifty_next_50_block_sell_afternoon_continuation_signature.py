"""Phase 2 empirical signature: nifty_next_50_block_sell_afternoon_continuation_short.

RE-TEST on NIFTY Next 50 (rank 51-100) after NIFTY 100 KILL (2026-05-22).
Brief: specs/2026-05-22-brief-nifty_100_above_VWAP_sustained_volume_climax_fade_short.md
(REFRAMED 2026-05-22 to nifty_100_block_sell_afternoon_continuation_short)
Original NIFTY 100 script:
  tools/sub9_research/phase2_nifty_100_block_sell_afternoon_continuation_signature.py

Hypothesis: Block-sell continuation was KILLed on top-100 (large institutions in
mega-caps already efficiently distributed). The "becoming-large" cohort (rank 51-100)
has thinner books -- block-sell residual supply may have larger market impact in
afternoon session, producing measurable SHORT drift to MIS auto-square.

CAVEAT: block-deal events are sparse on Next 50 (smaller mcap = fewer Rs 25cr block
prints). Expect n_discovery far below NIFTY 100's 86 signals -- may be too thin for
Phase 5.

Mechanism: NIFTY Next 50 large-cap MIS-eligible stocks with NSE block-deal SELL print of
trade_value >= Rs 25 cr on T+0. Hypothesis: residual institutional supply works into the
lit order book between 14:30 and 15:25, producing negative drift (SHORT continuation).

DATA CAVEAT (CRITICAL):
  `data/block_deals/block_deals_events.parquet` carries only `trade_date` (no intraday
  timestamp). Per SEBI rule, any block trade >= Rs 25 cr is in either the morning
  (08:45-09:00) or afternoon (14:05-14:20) regulated window. We cannot distinguish.
  Falsifier #1 is replaced by a "clean-SELL vs SELL-with-paired-BUY-same-date" proxy
  cohort split (see column `clean_sell_no_paired_buy`).

Entry-bar convention (project standard):
  * 5m bars are labeled by START time. Bar labeled "14:25" covers wall-clock
    14:25:00 -> 14:30:00 (close is "14:30 wall-clock").
  * `signal_close` = close of bar labeled "14:25" (latest data available at 14:30).
  * `close_at_1520` = close of bar labeled "15:20" (covers 15:20-15:25 wall-clock;
    matches MIS auto-square reference 15:25).

ANTI-BIAS GUARDS (audit checklist):
  * Lesson #5 (#1): no day-aggregate look-ahead. Intraday volume baseline uses ONLY
    bars from prior 5 trading days; current-day volume in `late_session_vol_ratio`
    sums bars 14:20-15:25 of T+0 only (no future-period leakage relative to exit).
  * Lesson #5 (#2): volume baseline excludes current bar where applicable; prior-5d
    baseline is computed BEFORE the signal date.
  * Lesson #19: ProductionUniverseGate applied per-date (accepted_caps=large_cap,
    require_mis=True). Avoids window-level survivorship.
  * Block-deal signal uses end-of-day disclosed data, which IS published intraday by
    NSE within mandatory disclosure windows -- BUT, since on-disk data has no
    intraday timestamp we conservatively treat it as available at session close.
    This script measures the empirical drift signature ONLY; a forward strategy
    must restrict to block prints whose disclosure preceded the entry bar.

  * First-fire-per-day-per-stock latch (one row per (sym, date) in signal cohort).

Universe: NIFTY Next 50 from assets/nifty_next_50_universe.json filtered via
ProductionUniverseGate(accepted_caps={"large_cap"}, require_mis=True).

NO exit walk -- pure signature measurement.
"""
from __future__ import annotations

import json
import sys
from datetime import date, time as dtime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from tools.sub9_research.production_universe import ProductionUniverseGate  # noqa: E402

# ---------------------------------------------------------------- windows
DISCOVERY_START = date(2023, 1, 1)
DISCOVERY_END = date(2024, 12, 31)
OOS_START = date(2025, 1, 1)
OOS_END = date(2025, 12, 31)
HOLDOUT_START = date(2026, 1, 1)
HOLDOUT_END = date(2026, 4, 30)
FULL_START = DISCOVERY_START
FULL_END = HOLDOUT_END

SEBI_REGIME_BREAK = date(2025, 12, 7)  # block-deal reform: min 10cr->25cr, band 1%->3%

# ---------------------------------------------------------------- thresholds
BLOCK_NOTIONAL_MIN_CR = 25.0  # post-Dec-2025 SEBI floor; applied universally for cohort comparability

# Notional buckets (per brief)
NOTIONAL_BUCKETS: List[Tuple[float, float, str]] = [
    (25.0, 50.0, "25-50cr"),
    (50.0, 100.0, "50-100cr"),
    (100.0, float("inf"), ">=100cr"),
]

# Bar labels (5m bars; labels = start-of-bar wall-clock)
BAR_SIGNAL = dtime(14, 25)   # covers 14:25 -> 14:30; close = wall-clock 14:30
BAR_EXIT = dtime(15, 20)     # covers 15:20 -> 15:25; close = wall-clock 15:25 (MIS square)

# Bars 14:20-15:25 wall-clock = labels 14:20, 14:25, 14:30, ..., 15:20 (inclusive)
LATE_SESSION_LABELS: List[dtime] = []
_t = dtime(14, 20)
while _t <= dtime(15, 20):
    LATE_SESSION_LABELS.append(_t)
    h, m = _t.hour, _t.minute + 5
    if m >= 60:
        h, m = h + 1, m - 60
    _t = dtime(h, m)

# Prior-N-day baseline for late-session volume
BASELINE_LOOKBACK_DAYS = 5

# Paths
BLOCK_DEALS_PATH = _REPO_ROOT / "data" / "block_deals" / "block_deals_events.parquet"
UNIVERSE_PATH = _REPO_ROOT / "assets" / "nifty_next_50_universe.json"
SECTOR_MAP_PATH = _REPO_ROOT / "assets" / "stock_sector_map.json"
MONTHLY_DIR = _REPO_ROOT / "backtest-cache-download" / "monthly"
OUTPUT_CSV = _REPO_ROOT / "reports" / "sub9_sanity" / "_phase2_nifty_next_50_block_sell_afternoon_continuation_signature.csv"


def _normalize_symbol(s: str) -> str:
    if not isinstance(s, str):
        return ""
    if ":" in s:
        s = s.split(":")[-1]
    if "." in s:
        s = s.split(".")[0]
    return s


def _months_in_window(start: date, end: date) -> List[Tuple[int, int]]:
    months: List[Tuple[int, int]] = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def _load_next50_universe() -> Tuple[Set[str], Dict[str, str]]:
    """Returns (bare_symbols_set, sym_to_sector_id)."""
    with open(UNIVERSE_PATH, encoding="utf-8") as f:
        d = json.load(f)
    items = d["universe"]
    bare: Set[str] = set()
    sec_map: Dict[str, str] = {}
    for it in items:
        sym = _normalize_symbol(it.get("symbol", ""))
        if sym:
            bare.add(sym)
            sec = it.get("sector_id")
            if sec:
                sec_map[sym] = sec
    return bare, sec_map


def _load_sector_map_fallback() -> Dict[str, str]:
    """Optional fallback sector map; keys may be 'NSE:SYM'."""
    if not SECTOR_MAP_PATH.exists():
        return {}
    with open(SECTOR_MAP_PATH, encoding="utf-8") as f:
        d = json.load(f)
    out: Dict[str, str] = {}
    for k, v in d.items():
        if k == "__meta__":
            continue
        bare = _normalize_symbol(k)
        if bare and isinstance(v, str):
            out[bare] = v
    return out


def _notional_bucket(cr: float) -> str:
    for lo, hi, label in NOTIONAL_BUCKETS:
        if lo <= cr < hi:
            return label
    return "unbucketed"


def _load_block_signals(universe: Set[str]) -> pd.DataFrame:
    """Load block-deal events, filter to NSE NIFTY-Next-50 prints with trade_value >= 25 cr.

    Returns dataframe with columns: trade_date(date), symbol(bare), buy_or_sell,
    trade_value_cr, notional_bucket.
    """
    df = pd.read_parquet(BLOCK_DEALS_PATH)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df = df[(df["trade_date"] >= FULL_START) & (df["trade_date"] <= FULL_END)]
    # Restrict to NSE exchange (BSE rows are excluded per brief)
    df = df[df["exchange"].astype(str).str.upper() == "NSE"]
    # Normalize symbol to bare
    df["sym"] = df["symbol"].astype(str).apply(_normalize_symbol)
    # Filter to NIFTY Next 50
    df = df[df["sym"].isin(universe)]
    # Normalize side
    df["side"] = df["buy_or_sell"].astype(str).str.upper().str.strip()
    # Notional threshold
    df = df[df["trade_value_cr"] >= BLOCK_NOTIONAL_MIN_CR]
    df = df.reset_index(drop=True)
    return df[["trade_date", "sym", "side", "trade_value_cr"]]


def _build_signal_universe(
    block_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate to (sym, trade_date) granularity.

    Signal cohort  : has >=1 SELL print >=25cr on (sym, date), regardless of paired BUY.
    `clean_sell_no_paired_buy` flag = True if NO BUY print >=25cr on same (sym, date).

    Returns:
      sig_df: one row per (sym, date) in signal cohort, columns:
              [trade_date, sym, block_trade_value_cr, clean_sell_no_paired_buy]
      block_dates_by_sym: lookup for baseline (sym, date) candidates that we want to
              EXCLUDE from baseline cohort (any block print, BUY or SELL, on that date).
    """
    sells = block_df[block_df["side"] == "SELL"]
    if sells.empty:
        return pd.DataFrame(columns=["trade_date", "sym", "block_trade_value_cr", "clean_sell_no_paired_buy"]), block_df
    # Per (sym, date): max SELL notional, max BUY notional
    sells_agg = (
        sells.groupby(["sym", "trade_date"], as_index=False)["trade_value_cr"]
        .max()
        .rename(columns={"trade_value_cr": "block_trade_value_cr"})
    )
    buys = block_df[block_df["side"] == "BUY"]
    buys_agg = (
        buys.groupby(["sym", "trade_date"], as_index=False)["trade_value_cr"]
        .max()
        .rename(columns={"trade_value_cr": "max_buy_cr"})
        if not buys.empty
        else pd.DataFrame(columns=["sym", "trade_date", "max_buy_cr"])
    )
    sig_df = sells_agg.merge(buys_agg, on=["sym", "trade_date"], how="left")
    sig_df["clean_sell_no_paired_buy"] = sig_df["max_buy_cr"].isna()
    sig_df = sig_df.drop(columns=["max_buy_cr"])
    return sig_df, block_df


def _sebi_regime(d: date) -> str:
    return "post_2025_12_07" if d >= SEBI_REGIME_BREAK else "pre_2025_12_07"


def _resolve_window(d: date) -> str:
    if DISCOVERY_START <= d <= DISCOVERY_END:
        return "discovery"
    if OOS_START <= d <= OOS_END:
        return "oos"
    if HOLDOUT_START <= d <= HOLDOUT_END:
        return "holdout"
    return "out_of_scope"


def _load_5m_for_months(months: List[Tuple[int, int]], universe: Set[str]) -> pd.DataFrame:
    """Load monthly 5m feathers, restrict to NIFTY Next 50, normalize time/date."""
    frames = []
    cols = ["symbol", "date", "open", "high", "low", "close", "volume"]
    for (yy, mm) in months:
        p = MONTHLY_DIR / f"{yy:04d}_{mm:02d}_5m_enriched.feather"
        if not p.exists():
            continue
        print(f"  loading {p.name} ...", flush=True)
        df = pd.read_feather(p, columns=cols)
        df["sym"] = df["symbol"].astype(str).apply(_normalize_symbol)
        df = df[df["sym"].isin(universe)]
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"])
        if df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)
        df["d"] = df["date"].dt.date
        df["time"] = df["date"].dt.time
        frames.append(df[["sym", "date", "d", "time", "open", "high", "low", "close", "volume"]])
    if not frames:
        return pd.DataFrame(columns=["sym", "date", "d", "time", "open", "high", "low", "close", "volume"])
    out = pd.concat(frames, ignore_index=True)
    return out


def _build_late_session_vol_table(bars: pd.DataFrame) -> pd.DataFrame:
    """Sum of volume over LATE_SESSION_LABELS per (sym, day).

    Lesson #5 #1/#2: only bars labeled 14:20..15:20 of that day are summed. No look-ahead
    across days; baseline (prior 5 days) is computed AFTER this aggregation.
    """
    late = bars[bars["time"].isin(LATE_SESSION_LABELS)]
    if late.empty:
        return pd.DataFrame(columns=["sym", "d", "late_vol_sum"])
    agg = (
        late.groupby(["sym", "d"], as_index=False)["volume"]
        .sum()
        .rename(columns={"volume": "late_vol_sum"})
    )
    return agg


def _compute_baseline_vol(late_vol: pd.DataFrame) -> pd.DataFrame:
    """Per (sym, d): mean of late_vol_sum over the prior BASELINE_LOOKBACK_DAYS trading
    days for that sym (strictly d' < d). Returns rows aligned to late_vol."""
    if late_vol.empty:
        late_vol = late_vol.copy()
        late_vol["baseline_late_vol_mean"] = pd.Series(dtype=float)
        return late_vol
    out_rows = []
    # Sort by sym, d
    late_vol = late_vol.sort_values(["sym", "d"]).reset_index(drop=True)
    for sym, grp in late_vol.groupby("sym", sort=False):
        grp = grp.sort_values("d").reset_index(drop=True)
        vols = grp["late_vol_sum"].tolist()
        ds = grp["d"].tolist()
        for i in range(len(grp)):
            prior = vols[max(0, i - BASELINE_LOOKBACK_DAYS):i]  # strictly prior, excludes i
            base = float(sum(prior) / len(prior)) if prior else float("nan")
            out_rows.append({"sym": sym, "d": ds[i], "late_vol_sum": vols[i], "baseline_late_vol_mean": base})
    return pd.DataFrame(out_rows)


def main() -> int:
    print("Phase 2 empirical signature: nifty_next_50_block_sell_afternoon_continuation_short")
    print(f"  windows: discovery {DISCOVERY_START}->{DISCOVERY_END}, oos {OOS_START}->{OOS_END}, holdout {HOLDOUT_START}->{HOLDOUT_END}")
    print(f"  block notional floor: Rs {BLOCK_NOTIONAL_MIN_CR} cr (SEBI post-reform floor)")
    print(f"  SEBI regime break: {SEBI_REGIME_BREAK}")

    # 1) Universe + sector map
    universe, sector_map_pri = _load_next50_universe()
    sector_map_fb = _load_sector_map_fallback()
    print(f"  NIFTY Next 50 universe loaded: n={len(universe)}")

    # 2) Production universe gate (large_cap + MIS)
    gate = ProductionUniverseGate(
        accepted_caps={"large_cap"},
        require_mis=True,
        min_trading_days_required=0,
        min_daily_avg_volume=0,
    )

    # 3) Block-deal signals
    block_df = _load_block_signals(universe)
    print(f"  NSE block prints (>= Rs {BLOCK_NOTIONAL_MIN_CR} cr) in NIFTY Next 50, full window: n={len(block_df)}")
    sig_df, _ = _build_signal_universe(block_df)
    print(f"  signal cohort (>=1 SELL print >=25cr per (sym,date)): n={len(sig_df)} (unique sym,date pairs)")

    # 4) Bar data (load full window once -- NIFTY Next 50 only, fits in memory comfortably)
    months = _months_in_window(FULL_START, FULL_END)
    bars = _load_5m_for_months(months, universe)
    print(f"  5m bar rows (NIFTY Next 50, full window): {len(bars):,}")
    if bars.empty:
        print("  no bar data. abort.")
        return 1

    # 5) Late-session volume per (sym, d) + baseline (prior-5d, strictly past)
    late_vol = _build_late_session_vol_table(bars)
    late_vol = _compute_baseline_vol(late_vol)
    print(f"  late-session volume rows: {len(late_vol):,}")

    # 6) Pivot signal bar (14:25) close and exit bar (15:20) close per (sym, d)
    sig_bar = (
        bars[bars["time"] == BAR_SIGNAL]
        .loc[:, ["sym", "d", "close"]]
        .rename(columns={"close": "signal_close"})
    )
    exit_bar = (
        bars[bars["time"] == BAR_EXIT]
        .loc[:, ["sym", "d", "close"]]
        .rename(columns={"close": "close_at_1520"})
    )
    bar_anchor = sig_bar.merge(exit_bar, on=["sym", "d"], how="inner")
    bar_anchor = bar_anchor.merge(late_vol, on=["sym", "d"], how="left")
    print(f"  bar anchor rows (have 14:25 + 15:20 bars same day): {len(bar_anchor):,}")

    # 7) Signal records
    sig_records: List[Dict] = []
    for _, row in sig_df.iterrows():
        sym = row["sym"]
        d = row["trade_date"]
        if not gate.is_eligible(sym, d):
            continue
        anchor = bar_anchor[(bar_anchor["sym"] == sym) & (bar_anchor["d"] == d)]
        if anchor.empty:
            continue
        ar = anchor.iloc[0]
        signal_close = float(ar["signal_close"])
        if signal_close <= 0:
            continue
        close_at_1520 = float(ar["close_at_1520"])
        ret_to_1525 = (close_at_1520 - signal_close) / signal_close * 100.0
        late_v = ar.get("late_vol_sum")
        base_v = ar.get("baseline_late_vol_mean")
        if pd.notna(late_v) and pd.notna(base_v) and base_v > 0:
            late_ratio = float(late_v) / float(base_v)
        else:
            late_ratio = float("nan")
        sector_id = sector_map_pri.get(sym) or sector_map_fb.get(sym) or "unknown"
        sig_records.append({
            "signal_date": d,
            "symbol": sym,
            "sector_id": sector_id,
            "block_trade_value_cr": float(row["block_trade_value_cr"]),
            "notional_bucket": _notional_bucket(float(row["block_trade_value_cr"])),
            "deal_type": "SELL",
            "signal_close": signal_close,
            "close_at_1520": close_at_1520,
            "ret_to_1525": ret_to_1525,
            "late_session_vol_ratio": late_ratio,
            "clean_sell_no_paired_buy": bool(row["clean_sell_no_paired_buy"]),
            "sebi_regime": _sebi_regime(d),
            "window": _resolve_window(d),
            "is_signal": True,
            "is_baseline": False,
        })

    # 8) Baseline records: NIFTY Next 50 (sym, d) with NO block print of any side >=25cr
    # Build a set of (sym, d) to EXCLUDE from baseline (any block row in block_df).
    block_pairs: Set[Tuple[str, date]] = set(
        (r["sym"], r["trade_date"]) for _, r in block_df[["sym", "trade_date"]].iterrows()
    )
    print(f"  block-print (sym,date) pairs to exclude from baseline: {len(block_pairs)}")

    # Build per (sym, d) baseline records using the bar_anchor table.
    # Apply universe gate per-date. This is large; cache gate decisions per (sym, d).
    sig_set: Set[Tuple[str, date]] = set((r["symbol"], r["signal_date"]) for r in sig_records)
    baseline_records: List[Dict] = []
    gate_cache: Dict[Tuple[str, date], bool] = {}

    # Pre-filter bar_anchor to NIFTY Next 50 (already restricted but defensive)
    ba_iter = bar_anchor[bar_anchor["sym"].isin(universe)]
    n_total = len(ba_iter)
    print(f"  scanning {n_total:,} (sym,d) bar-anchor rows for baseline candidates ...", flush=True)
    progress_interval = max(1, n_total // 20)
    for idx, row in enumerate(ba_iter.itertuples(index=False)):
        if idx and idx % progress_interval == 0:
            print(f"    baseline scan: {idx:,}/{n_total:,}", flush=True)
        sym = row.sym
        d = row.d
        if (sym, d) in block_pairs:
            continue
        if (sym, d) in sig_set:
            continue
        key = (sym, d)
        elig = gate_cache.get(key)
        if elig is None:
            elig = gate.is_eligible(sym, d)
            gate_cache[key] = elig
        if not elig:
            continue
        signal_close = float(row.signal_close)
        if signal_close <= 0:
            continue
        close_at_1520 = float(row.close_at_1520)
        ret_to_1525 = (close_at_1520 - signal_close) / signal_close * 100.0
        late_v = row.late_vol_sum
        base_v = row.baseline_late_vol_mean
        if pd.notna(late_v) and pd.notna(base_v) and base_v > 0:
            late_ratio = float(late_v) / float(base_v)
        else:
            late_ratio = float("nan")
        sector_id = sector_map_pri.get(sym) or sector_map_fb.get(sym) or "unknown"
        baseline_records.append({
            "signal_date": d,
            "symbol": sym,
            "sector_id": sector_id,
            "block_trade_value_cr": float("nan"),
            "notional_bucket": "baseline",
            "deal_type": "NONE",
            "signal_close": signal_close,
            "close_at_1520": close_at_1520,
            "ret_to_1525": ret_to_1525,
            "late_session_vol_ratio": late_ratio,
            "clean_sell_no_paired_buy": False,
            "sebi_regime": _sebi_regime(d),
            "window": _resolve_window(d),
            "is_signal": False,
            "is_baseline": True,
        })

    # 9) Combine + write
    all_records = sig_records + baseline_records
    if not all_records:
        print("  no records. abort.")
        return 1
    df = pd.DataFrame(all_records)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  wrote {OUTPUT_CSV.relative_to(_REPO_ROOT)}  rows={len(df):,}")

    # 10) Summary reports
    print("\n=== Cohort sizes by window ===")
    print(df.groupby(["window", "is_signal"]).size().unstack(fill_value=0).to_string())

    print("\n=== Signal-cohort drift (ret_to_1525, SHORT direction -> negative is good) ===")
    sig_only = df[df["is_signal"]].copy()
    base_only = df[df["is_baseline"]].copy()
    for win in ("discovery", "oos", "holdout"):
        s = sig_only[sig_only["window"] == win]["ret_to_1525"].dropna()
        b = base_only[base_only["window"] == win]["ret_to_1525"].dropna()
        if len(s) == 0 and len(b) == 0:
            continue
        print(f"  {win:9s} signal n={len(s):5d} mean={s.mean():+.4f}% med={s.median():+.4f}%  baseline n={len(b):7d} mean={b.mean():+.4f}% med={b.median():+.4f}%")

    print("\n=== Signal cohort: SEBI regime split (Falsifier #3) ===")
    for reg in ("pre_2025_12_07", "post_2025_12_07"):
        s = sig_only[sig_only["sebi_regime"] == reg]["ret_to_1525"].dropna()
        if len(s) == 0:
            continue
        neg = (s < 0).mean()
        print(f"  {reg:18s} n={len(s):5d} mean={s.mean():+.4f}% med={s.median():+.4f}% pct_neg={neg:.3f}")

    print("\n=== Signal cohort: clean-SELL vs SELL-with-paired-BUY (Falsifier #1 proxy) ===")
    for clean_flag in (True, False):
        s = sig_only[sig_only["clean_sell_no_paired_buy"] == clean_flag]["ret_to_1525"].dropna()
        if len(s) == 0:
            continue
        neg = (s < 0).mean()
        label = "clean_SELL" if clean_flag else "paired_with_BUY"
        print(f"  {label:18s} n={len(s):5d} mean={s.mean():+.4f}% med={s.median():+.4f}% pct_neg={neg:.3f}")

    print("\n=== Signal cohort: notional buckets ===")
    for _, _, label in NOTIONAL_BUCKETS:
        s = sig_only[sig_only["notional_bucket"] == label]["ret_to_1525"].dropna()
        if len(s) == 0:
            continue
        print(f"  {label:10s} n={len(s):5d} mean={s.mean():+.4f}% med={s.median():+.4f}%")

    print("\n=== Falsifier #2: late_session_vol_ratio (signal vs baseline) ===")
    s_ratio = sig_only["late_session_vol_ratio"].dropna()
    b_ratio = base_only["late_session_vol_ratio"].dropna()
    if len(s_ratio) > 0:
        print(f"  signal   n={len(s_ratio):5d} mean={s_ratio.mean():.3f} med={s_ratio.median():.3f} pct>=1.3={(s_ratio>=1.3).mean():.3f}")
    if len(b_ratio) > 0:
        print(f"  baseline n={len(b_ratio):7d} mean={b_ratio.mean():.3f} med={b_ratio.median():.3f} pct>=1.3={(b_ratio>=1.3).mean():.3f}")

    print("\n=== Sector split (signal cohort, discovery) ===")
    s_disc = sig_only[sig_only["window"] == "discovery"]
    if len(s_disc) > 0:
        sec_g = s_disc.groupby("sector_id")["ret_to_1525"].agg(["count", "mean", "median"]).sort_values("count", ascending=False).head(20)
        print(sec_g.round(4).to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
