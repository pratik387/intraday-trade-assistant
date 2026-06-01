"""Phase 2 empirical signature — mis_unwind_short_late_session.

Brief: specs/2026-05-07-sub-project-9-brief-mis_unwind_short_late_session.md
Phase 1 verdict (2026-06-01): PROCEED to Phase 2 (with concerns).

# Mechanism (from brief §6)

At 14:30 IST, screen F&O 200 mid+small_cap. Symbol qualifies if ALL hold:
  1. intraday return at 14:30 ∈ [+1.5%, +4.0%]
  2. close at 14:30 is 0.3-1.5% off intraday-high (off-the-high zone)
  3. ret_3 (last 3 bars) at 14:30 ∈ [0.0%, +0.5%]
     AND ret(14:30) − ret(13:30) ≤ 0 (peak distribution)
  4. intraday cumulative volume rank ≥ 70th percentile of qualifying universe
  5. avg vol(14:15-14:30) < avg vol(13:00-14:00) (accumulation exhausting)

Phase 2 goal: measure mean forward return from 14:30 to 15:10 on signal vs
baseline (matched universe, not meeting the 6-condition gate). NO fees, NO
leverage, NO exits — raw drift only.

Kill criterion per setup_lifecycle.md: signal mean drift < 0.1% absolute
(i.e. for SHORT, mean drift > -0.1%) → signal doesn't exist; abandon.

# Anti-bias guards (Lesson #5)

  1. All features computed from bars[:i+1] only — no look-ahead.
  2. Volume baselines use bars STRICTLY PRIOR to the relevant window (no
     current-bar leakage in vol_ratio computation).
  3. Cap segment via production lookup (services.symbol_metadata.get_cap_segment)
     — same path the live detector would use.
  4. F&O 200 universe is a static asset file — no survivorship bias from
     "what's currently liquid."
  5. First-fire-per-day latch (one fire per (symbol, date) max).
  6. ret_3 and 13:30 reference are computed from session bars only (intraday
     reset, no prior-day leakage).

# Discovery window

2023-01-02 to 2024-12-31 (2-year Discovery). Sufficient for n>=200 floor at
the brief's expected 5 trades/month rate. OOS / Holdout remain unobserved
at Phase 2.

Usage:
    .venv/Scripts/python -m tools.sub9_research.phase2_mis_unwind_short_late_session_signature

Output:
    reports/sub9_sanity/_phase2_mis_unwind_short_late_session_signature.csv

Note: 2023-2024 backtest data is in backtest-cache-download/monthly/<YYYY>_<MM>_5m_enriched.feather
"""
from __future__ import annotations

import sys
from datetime import date, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from services.symbol_metadata import get_cap_segment  # noqa: E402

# ── CONFIG (locked at Phase 2; no late tuning per brief §6) ─────────────
ALLOWED_CAPS = {"mid_cap", "small_cap"}
UNIVERSE_FILE = _REPO / "assets" / "fno_liquid_200.csv"

# 6-condition gate (per brief §6)
INTRADAY_RET_MIN = 0.015         # +1.5%
INTRADAY_RET_MAX = 0.040         # +4.0%
OFF_HIGH_MIN = 0.003             # 0.3% off intraday high
OFF_HIGH_MAX = 0.015             # 1.5% off intraday high
RET_3_MIN = 0.000                # 0.0%
RET_3_MAX = 0.005                # +0.5%
VOL_RANK_PCT_MIN = 0.70          # ≥70th percentile of qualifying universe
LATE_VS_EARLY_VOL_MAX_RATIO = 1.00  # avg vol(14:15-14:30) < avg vol(13:00-14:00)

# Window timestamps (IST-naive)
T_SIGNAL = dtime(14, 30)
T_REF_FOR_DELTA = dtime(13, 30)
EARLY_VOL_START = dtime(13, 0)
EARLY_VOL_END = dtime(14, 0)
LATE_VOL_START = dtime(14, 15)
LATE_VOL_END = dtime(14, 30)
FORWARD_TIMESTAMPS = [
    dtime(14, 35), dtime(14, 40), dtime(14, 45),
    dtime(14, 50), dtime(14, 55), dtime(15, 0),
    dtime(15, 5), dtime(15, 10),
]

DISCOVERY_START = date(2023, 1, 2)
DISCOVERY_END = date(2024, 12, 31)

OUT_CSV = _REPO / "reports" / "sub9_sanity" / "_phase2_mis_unwind_short_late_session_signature.csv"
FEATHER_DIR = _REPO / "backtest-cache-download" / "monthly"

# ── Universe construction ──────────────────────────────────────────────
def build_universe() -> set[str]:
    """F&O 200 stocks with cap_segment in {mid_cap, small_cap}."""
    df = pd.read_csv(UNIVERSE_FILE)
    syms = []
    for raw in df["symbol"]:
        try:
            if get_cap_segment(raw) in ALLOWED_CAPS:
                # feather uses bare symbol (no NSE: prefix); brief uses NSE:SYM
                syms.append(raw.replace("NSE:", ""))
        except Exception:
            continue
    return set(syms)


# ── Per-day, per-symbol signal evaluation ──────────────────────────────
def evaluate_session(
    session_date: date, universe: set[str], df_month: pd.DataFrame
) -> list[dict]:
    """Returns list of dict rows: one per qualifying (symbol, date)."""
    rows = []
    day = df_month[df_month["date"].dt.date == session_date]
    if day.empty:
        return rows

    # First pass: compute per-symbol features at 14:30
    per_symbol_features: dict[str, dict] = {}
    for sym, g in day.groupby("symbol"):
        if sym not in universe:
            continue
        g = g.sort_values("date").reset_index(drop=True)
        # Find bar at 14:30 (single 5m bar exactly)
        mask_signal = g["date"].dt.time == T_SIGNAL
        if not mask_signal.any():
            continue
        signal_bar = g[mask_signal].iloc[0]
        # All bars up to and INCLUDING 14:30
        bars_to_signal = g[g["date"].dt.time <= T_SIGNAL]
        if len(bars_to_signal) < 4:  # need at least 09:15 + 3 more
            continue

        open_price = bars_to_signal.iloc[0]["open"]
        signal_close = float(signal_bar["close"])
        intraday_high = float(bars_to_signal["high"].max())

        if open_price <= 0 or intraday_high <= 0:
            continue

        # Feature 1: intraday return at 14:30
        intraday_ret = (signal_close - open_price) / open_price
        # Feature 2: off-high
        off_high = (intraday_high - signal_close) / intraday_high
        # Feature 3: ret_3 (last 3 bars: signal bar + 2 prior)
        if len(bars_to_signal) < 3:
            continue
        ret_3 = (
            (signal_close - bars_to_signal.iloc[-3]["close"])
            / bars_to_signal.iloc[-3]["close"]
        )
        # ret(14:30) − ret(13:30)
        mask_1330 = g["date"].dt.time == T_REF_FOR_DELTA
        if not mask_1330.any():
            continue
        close_1330 = float(g[mask_1330].iloc[0]["close"])
        ret_to_1330 = (close_1330 - open_price) / open_price
        ret_delta_1430_minus_1330 = intraday_ret - ret_to_1330

        # Feature 4: intraday cumulative volume (ranked per-day across universe later)
        cum_volume = float(bars_to_signal["volume"].sum())

        # Feature 5: vol(14:15-14:30) vs vol(13:00-14:00)
        late_bars = bars_to_signal[
            (bars_to_signal["date"].dt.time >= LATE_VOL_START)
            & (bars_to_signal["date"].dt.time <= LATE_VOL_END)
        ]
        early_bars = bars_to_signal[
            (bars_to_signal["date"].dt.time >= EARLY_VOL_START)
            & (bars_to_signal["date"].dt.time < EARLY_VOL_END)
        ]
        if early_bars.empty or late_bars.empty:
            continue
        late_avg_vol = float(late_bars["volume"].mean())
        early_avg_vol = float(early_bars["volume"].mean())
        if early_avg_vol <= 0:
            continue
        late_over_early = late_avg_vol / early_avg_vol

        # Forward returns
        forward_returns = {}
        for t in FORWARD_TIMESTAMPS:
            mask = g["date"].dt.time == t
            if mask.any():
                fwd_close = float(g[mask].iloc[0]["close"])
                forward_returns[t.strftime("%H%M")] = (fwd_close - signal_close) / signal_close
            else:
                forward_returns[t.strftime("%H%M")] = np.nan

        per_symbol_features[sym] = {
            "intraday_ret": intraday_ret,
            "off_high": off_high,
            "ret_3": ret_3,
            "ret_delta_1430_minus_1330": ret_delta_1430_minus_1330,
            "cum_volume": cum_volume,
            "late_over_early": late_over_early,
            "signal_close": signal_close,
            "intraday_high": intraday_high,
            "forward_returns": forward_returns,
        }

    if not per_symbol_features:
        return rows

    # Compute volume rank percentile WITHIN universe-of-the-day
    # (using cumulative volume up to 14:30 as the "intraday cumulative volume" proxy)
    cum_vols = pd.Series({s: f["cum_volume"] for s, f in per_symbol_features.items()})
    vol_ranks = cum_vols.rank(pct=True)

    # Second pass: apply 6-condition gate; emit rows
    for sym, feats in per_symbol_features.items():
        passes = (
            INTRADAY_RET_MIN <= feats["intraday_ret"] <= INTRADAY_RET_MAX
            and OFF_HIGH_MIN <= feats["off_high"] <= OFF_HIGH_MAX
            and RET_3_MIN <= feats["ret_3"] <= RET_3_MAX
            and feats["ret_delta_1430_minus_1330"] <= 0
            and float(vol_ranks[sym]) >= VOL_RANK_PCT_MIN
            and feats["late_over_early"] < LATE_VS_EARLY_VOL_MAX_RATIO
        )
        row = {
            "session_date": session_date.isoformat(),
            "symbol": sym,
            "signal": int(passes),
            "intraday_ret": feats["intraday_ret"],
            "off_high": feats["off_high"],
            "ret_3": feats["ret_3"],
            "ret_delta_1430_minus_1330": feats["ret_delta_1430_minus_1330"],
            "vol_rank_pct": float(vol_ranks[sym]),
            "late_over_early": feats["late_over_early"],
            "signal_close": feats["signal_close"],
            "intraday_high": feats["intraday_high"],
        }
        for t_key, fwd in feats["forward_returns"].items():
            row[f"ret_to_{t_key}"] = fwd
        rows.append(row)
    return rows


def main():
    print(f"Phase 2 — mis_unwind_short_late_session (Discovery {DISCOVERY_START} to {DISCOVERY_END})")
    universe = build_universe()
    print(f"Universe (F&O 200 mid+small_cap): {len(universe)} symbols")

    months_needed = []
    cur = pd.Timestamp(DISCOVERY_START).to_period("M")
    end = pd.Timestamp(DISCOVERY_END).to_period("M")
    while cur <= end:
        months_needed.append(cur)
        cur += 1

    all_rows = []
    for ym in months_needed:
        feather_path = FEATHER_DIR / f"{ym.year}_{ym.month:02d}_5m_enriched.feather"
        if not feather_path.exists():
            print(f"  missing: {feather_path.name}")
            continue
        df_month = pd.read_feather(feather_path)
        if df_month.empty:
            continue
        df_month["date"] = pd.to_datetime(df_month["date"])
        # Universe filter early to shrink rows
        df_month = df_month[df_month["symbol"].isin(universe)]
        if df_month.empty:
            continue
        # Per-session
        sessions = sorted(df_month["date"].dt.date.unique())
        sessions = [s for s in sessions if DISCOVERY_START <= s <= DISCOVERY_END]
        for s in sessions:
            rows = evaluate_session(s, universe, df_month)
            all_rows.extend(rows)
        print(f"  {ym}: {len(sessions)} sessions processed")

    out = pd.DataFrame(all_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    n_signal = int(out["signal"].sum())
    n_baseline = int((out["signal"] == 0).sum())
    print()
    print(f"Total candidates evaluated: {len(out)}")
    print(f"  signal (passes 6-condition gate): {n_signal}")
    print(f"  baseline (in universe but fails gate): {n_baseline}")

    if n_signal == 0:
        print("KILL: no signal events.")
        return

    for t_key in ("1435", "1440", "1445", "1450", "1455", "1500", "1505", "1510"):
        col = f"ret_to_{t_key}"
        if col not in out.columns:
            continue
        sig_ret = out.loc[out["signal"] == 1, col].dropna() * 100.0
        base_ret = out.loc[out["signal"] == 0, col].dropna() * 100.0
        delta = sig_ret.mean() - base_ret.mean()
        print(
            f"  {t_key}: signal mean={sig_ret.mean():+.3f}% (n={len(sig_ret)})  "
            f"baseline mean={base_ret.mean():+.3f}% (n={len(base_ret)})  "
            f"delta={delta:+.3f}%"
        )

    # Per-year breakdown on 15:10 endpoint
    if n_signal > 0 and "ret_to_1510" in out.columns:
        print()
        print("Per-year breakdown (signal ret_to_1510):")
        sig = out[out["signal"] == 1].copy()
        sig["session_date"] = pd.to_datetime(sig["session_date"])
        sig["year"] = sig["session_date"].dt.year
        for year, g in sig.groupby("year"):
            r = g["ret_to_1510"].dropna() * 100.0
            wr_short = (r < 0).mean() * 100.0 if len(r) else 0
            print(f"  {year}: n={len(r)}  mean={r.mean():+.3f}%  median={r.median():+.3f}%  short_WR={wr_short:.1f}%")

    print(f"\nOutput: {OUT_CSV.relative_to(_REPO)}")


if __name__ == "__main__":
    main()
