"""Pre-coding sanity check for C-03 Upper-Circuit Release Spike-and-Fade.

Candidate spec: `specs/2026-05-16-new-setup-candidates.md` -> CANDIDATE-03.

REVISED MECHANISM (clarified from candidate spec):
  Indian NSE upper-circuit limits (5/10/20%) cap the day's maximum price.
  Once a small-cap hits its upper band, the level becomes the day's ceiling
  for the rest of the session. The candidate spec's "circuit release" framing
  is technically incorrect (circuits don't dynamically lift) - the real
  intraday pattern is:

    1. Stock pins upper circuit EARLY (morning retail-FOMO surge into level)
    2. Sellers appear later (profit-taking, news, market weakness) -> price
       drops 1-2% from the pin
    3. Retail buyers re-engage, price RE-TESTS the pin from below
    4. If the re-test FAILS (price can't break day high; new buyers exhausted),
       a cascade-down ensues as FOMO buyers panic-sell

  We trade step 4: SHORT the failed re-test.

  Distinct from `circuit_t1_fade_short` (active in prod) which fades NEXT-DAY
  after circuit-pin (different time horizon, same underlying edge premise).

  Mechanism is Indian-microstructure-specific (NSE has hard circuit bands;
  US/EU markets have soft "limit-up/limit-down" auctions that behave differently).

UNIVERSE + FILTERS:
  - small_cap + mid_cap (where retail FOMO concentrates)
  - MIS-eligible (we trade MIS)
  - Symbol in 5m feather cache

DAY FILTER (suggests morning circuit-pin):
  - day_high / PDC >= 1.045  (>= 4.5% intraday gain, consistent with 5/10/20% bands)
  - day_high reached by 10:30 IST  (morning pin, not late-day spike)
  - day_high - day_close <= 1.5% of day_high  (stock stayed elevated, not crashed)
                                                BUT not perfectly pinned to high either

DETECTION:
  - Find first 5m bar AFTER 12:00 where bar.high >= day_high * 0.997
    (price re-tested within 0.3% of day high in afternoon)
  - Require: bar.close <= bar.high * 0.997 (rejection - close below the test high)
  - Require: bar.volume >= prior 5-bar median volume (real test, not noise)

ENTRY:
  - SHORT at the close of the rejection bar
  - SL = bar.high * 1.003  (0.3% above the rejection high)
  - T1 = entry - 1R; T2 = entry - 2R
  - Time stop = 15:10

DECISION CRITERION (sub9 retire-pre-data, adapted):
  n_total < 200                       -> STRUCTURAL RETIRE-PRE-DATA
  n_total >= 200 AND PF >= 1.10       -> STRONG PROCEED -> cell-mine + R-sweep
  n_total >= 200 AND PF in [1.0, 1.10) -> MARGINAL -> cell selection may rescue
  n_total >= 200 AND PF < 1.0          -> THESIS RETIRE

Usage:
    .venv/Scripts/python tools/sub9_research/sanity_circuit_release_fade.py
    .venv/Scripts/python tools/sub9_research/sanity_circuit_release_fade.py --window oos
    .venv/Scripts/python tools/sub9_research/sanity_circuit_release_fade.py --window holdout
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from services.symbol_metadata import get_cap_segment, get_mis_info  # noqa: E402
from tools.sub7_validation.build_per_setup_pnl import calc_fee       # noqa: E402


# ----- Config knobs -----

ALLOWED_CAPS = {"small_cap", "mid_cap"}

# Day filter (heuristic upper-circuit pin signature)
# 2026-05-18 fix: removed two more look-ahead biases:
#   - "morning_high < day_high * 0.999" rejected days where EOD high > morning_high
#     (impossible to know at signal time — production uses session_high_so_far)
#   - "day_gain_pct >= 4.5%" was computed from EOD day_high, not session_high at signal bar
# Both now use session_high_so_far at signal time, matching production
# (structures/circuit_release_fade_short_structure.py:144-160).
MIN_DAY_GAIN_PCT = 4.5            # session_high_so_far/PDC - 1 >= 4.5% AT SIGNAL TIME
DAY_HIGH_BY_HHMM = "10:30"        # morning bars: morning_high = max(high before this time)
MORNING_HIGH_TOLERANCE_PCT = 0.1  # session_high_so_far * (1 - tol/100) <= morning_high (no new high since)

# Re-test detection
RETEST_AFTER_HHMM = "12:00"       # only consider re-tests after this time (afternoon)
RETEST_TOL_PCT = 0.3              # bar.high >= session_high_so_far * (1 - tol/100) qualifies as re-test
REJECTION_CLOSE_PCT = 0.3         # bar.close <= bar.high * (1 - this/100) = rejection
VOLUME_CONFIRM_BARS = 5           # rolling median of last N bars

# Optional strictness filter (proposed remediation 2026-05-18):
# Requires LOW retracement from morning_high BEFORE the retest is allowed.
# Filters out "consolidation near morning_high → breakout" (the 197 OCI-only
# losers in circuit_release) and keeps only "real weakness then retest"
# patterns. Set to 0 to disable.
#   min_low_between_morning_and_signal <= morning_high * (1 - this_pct/100)
MIN_RETRACE_PCT_FROM_MORNING_HIGH = 0.0  # default off; sweep 0/1.5/2.0/3.0

# Trade geometry
SL_PCT_ABOVE_REJECTION_HIGH = 0.3
T1_R = 1.0
T2_R = 2.0
EXIT_BAR_HHMM = "15:10"
RISK_PER_TRADE_RUPEES = 1000

# Entry-zone semantics (mode A only). Default Mode B = next-bar-open (idealized).
ENTRY_MODE = "B"                # "A" | "B" — set by --entry-mode
ENTRY_ZONE_PCT = 0.3            # symmetric, matches setups.circuit_release_fade_short.entry_zone_pct
TRIGGER_EXPIRY_BARS = 3         # 15 minutes = 3 x 5m bars (matches trigger_expiry_minutes=15)

# Windows
WINDOWS = {
    "discovery": (date(2023, 1, 1), date(2024, 12, 31)),
    "oos":       (date(2025, 1, 1), date(2025, 9, 30)),
    "holdout":   (date(2025, 10, 1), date(2026, 4, 30)),
}

WINDOW_START = WINDOWS["discovery"][0]
WINDOW_END = WINDOWS["discovery"][1]
WINDOW_LABEL = "discovery"


# ----- Loaders -----

def _months_in_window() -> List[tuple]:
    months: List[tuple] = []
    y, m = WINDOW_START.year, WINDOW_START.month
    while (y, m) <= (WINDOW_END.year, WINDOW_END.month):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def _load_5m_for_month(yyyy: int, mm: int) -> pd.DataFrame:
    p = _REPO_ROOT / "backtest-cache-download" / "monthly" / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_feather(p)


def load_daily_with_pdc() -> pd.DataFrame:
    """Load daily OHLCV from consolidated cache, compute PDC (prior close)."""
    print("  loading consolidated_daily.feather ...")
    p = _REPO_ROOT / "cache" / "preaggregate" / "consolidated_daily.feather"
    df = pd.read_feather(p)
    df["ts"] = pd.to_datetime(df["ts"])
    if df["ts"].dt.tz is not None:
        df["ts"] = df["ts"].dt.tz_localize(None)
    df["d"] = df["ts"].dt.date
    df = df[(df["d"] >= WINDOW_START) & (df["d"] <= WINDOW_END)]
    df = df.sort_values(["symbol", "d"])
    df["pdc"] = df.groupby("symbol")["close"].shift(1)
    df = df.dropna(subset=["pdc"]).copy()
    df["day_gain_pct"] = (df["high"] / df["pdc"] - 1.0) * 100.0
    df["close_off_high_pct"] = (df["high"] - df["close"]) / df["high"] * 100.0
    print(f"    daily rows in window: {len(df):,} | symbols: {df['symbol'].nunique()}")
    return df


def filter_circuit_pin_days(daily: pd.DataFrame) -> pd.DataFrame:
    """Filter daily rows to candidate circuit-pin days (heuristic detection).

    Only filters on day_gain_pct (known at entry time via morning-pin detection).
    close_off_high_pct is computed for cell-mining metadata only, not used as a filter.
    """
    print(f"  filtering for circuit-pin signature ...")
    n0 = len(daily)
    df = daily[daily["day_gain_pct"] >= MIN_DAY_GAIN_PCT]
    print(f"    day_gain >= {MIN_DAY_GAIN_PCT}%: {len(df):,} (was {n0:,})")
    return df


def _is_mis_eligible(bare_symbol: str) -> bool:
    nse_sym = f"NSE:{bare_symbol}"
    try:
        return bool(get_mis_info(nse_sym).get("mis_enabled", False))
    except Exception:
        return False


def _cap_segment(bare_symbol: str) -> str:
    try:
        return get_cap_segment(f"NSE:{bare_symbol}")
    except Exception:
        return "unknown"


# ----- Simulation -----

def _simulate_one(daily_row, day_bars: pd.DataFrame) -> Optional[dict]:
    """Simulate one circuit-pin-failed-retest SHORT trade.

    All "as-of-signal-time" checks use session_high_so_far (running max of all
    bars from session open through the current candidate bar) rather than the
    EOD day_high. This matches production logic in
    structures/circuit_release_fade_short_structure.py:144-160 and removes the
    look-ahead bias that made sanity reject failed-pin days post-hoc.
    """
    pdc = float(daily_row["pdc"])

    if day_bars.empty:
        return None

    day_bars = day_bars.copy()
    day_bars["hhmm"] = day_bars["date"].dt.strftime("%H%M").astype(int)
    day_bars = day_bars.sort_values("date").reset_index(drop=True)

    # Morning bars used to compute morning_high (no look-ahead — morning is
    # locked by 10:30, so reading max(high) of morning bars is fine).
    morning_cutoff = int(DAY_HIGH_BY_HHMM.replace(":", ""))
    morning_bars = day_bars[day_bars["hhmm"] <= morning_cutoff]
    if morning_bars.empty:
        return None
    morning_high = float(morning_bars["high"].max())

    # Iterate afternoon bars; at each candidate, compute session_high_so_far
    # using only data available AT that bar (production-equivalent).
    retest_cutoff = int(RETEST_AFTER_HHMM.replace(":", ""))
    exit_cutoff = int(EXIT_BAR_HHMM.replace(":", ""))
    afternoon = day_bars[(day_bars["hhmm"] >= retest_cutoff) & (day_bars["hhmm"] <= exit_cutoff)].reset_index(drop=True)
    if afternoon.empty:
        return None

    # Precompute running max(high) over the full intraday series for fast
    # session_high_so_far lookup at any signal bar.
    day_bars["session_high_so_far"] = day_bars["high"].cummax()
    # Same for cumulative min(low) used by optional strictness filter.
    day_bars["session_low_after_morning"] = day_bars["low"].where(
        day_bars["hhmm"] > morning_cutoff
    ).cummin()

    # Build a lookup: afternoon bar timestamp -> session metrics at that bar.
    # Use index alignment by date column.
    sess_high_map = dict(zip(day_bars["date"], day_bars["session_high_so_far"]))
    sess_low_after_morning_map = dict(zip(day_bars["date"], day_bars["session_low_after_morning"]))

    rejection_idx = None
    session_high_at_signal = None
    for i, bar in afternoon.iterrows():
        session_high = float(sess_high_map[bar["date"]])

        # Production check 1: day_gain_pct using session_high (not EOD)
        if (session_high / pdc - 1.0) * 100.0 < MIN_DAY_GAIN_PCT:
            continue
        # Production check 2: morning_high still holds (no new high since)
        if morning_high < session_high * (1.0 - MORNING_HIGH_TOLERANCE_PCT / 100.0):
            continue
        # Production check 3: retest threshold against session_high
        retest_threshold = session_high * (1.0 - RETEST_TOL_PCT / 100.0)
        if bar["high"] < retest_threshold:
            continue
        # Rejection: close meaningfully below this bar's high
        if bar["close"] > bar["high"] * (1.0 - REJECTION_CLOSE_PCT / 100.0):
            continue
        # Volume confirmation: bar volume >= recent median
        lookback_start = max(0, i - VOLUME_CONFIRM_BARS)
        recent_vol_median = float(afternoon.iloc[lookback_start:i]["volume"].median()) if i > 0 else 0.0
        if not (recent_vol_median > 0 and bar["volume"] >= recent_vol_median):
            continue
        # Optional strictness filter: require low retracement from morning_high
        # BEFORE the retest is allowed (proposed remediation 2026-05-18).
        if MIN_RETRACE_PCT_FROM_MORNING_HIGH > 0.0:
            sess_low = sess_low_after_morning_map.get(bar["date"])
            if sess_low is None or pd.isna(sess_low):
                continue
            required_low = morning_high * (1.0 - MIN_RETRACE_PCT_FROM_MORNING_HIGH / 100.0)
            if sess_low > required_low:
                continue

        rejection_idx = i
        session_high_at_signal = session_high
        break

    if rejection_idx is None:
        return None

    rej_bar = afternoon.iloc[rejection_idx]
    rejection_high = float(rej_bar["high"])
    signal_close_px = float(rej_bar["close"])
    signal_ts = rej_bar["date"]

    # Entry semantics — see ENTRY_MODE module constant.
    # Mode B (default, idealized): fill at next-bar OPEN. Production's actual
    #   behavior pre-2026-05-18 — over-states PF for momentum signals that
    #   never re-test the rejection level.
    # Mode A (tick-zone-touch): walk subsequent bars; fill when range
    #   intersects entry_zone (signal_close ± ENTRY_ZONE_PCT). EXPIRE after
    #   TRIGGER_EXPIRY_BARS without touch. Matches production's tick-aware
    #   trigger if we ship that fix.
    if rejection_idx + 1 >= len(afternoon):
        return None  # no next bar to enter on

    if ENTRY_MODE == "B":
        entry_bar = afternoon.iloc[rejection_idx + 1]
        entry_ts = entry_bar["date"]
        entry_price = float(entry_bar["open"])
        entry_offset = 1  # bars after rejection_idx
    else:
        # Mode A
        zone_min = signal_close_px * (1.0 - ENTRY_ZONE_PCT / 100.0)
        zone_max = signal_close_px * (1.0 + ENTRY_ZONE_PCT / 100.0)
        entry_price = None
        entry_ts = None
        entry_offset = None
        for j in range(1, min(1 + TRIGGER_EXPIRY_BARS, len(afternoon) - rejection_idx)):
            cand = afternoon.iloc[rejection_idx + j]
            cand_open = float(cand["open"])
            cand_high = float(cand["high"])
            cand_low = float(cand["low"])
            # If open is already in zone, fill at open
            if zone_min <= cand_open <= zone_max:
                entry_price = cand_open
            elif cand_low <= zone_max and cand_high >= zone_min:
                # Range intersects zone — for SHORT, conservative fill = zone_min
                # (price crossed up into zone from below, fill on entry-zone cross)
                entry_price = max(cand_low, zone_min)
            if entry_price is not None:
                entry_ts = cand["date"]
                entry_offset = j
                break
        if entry_price is None:
            return None  # EXPIRED — no zone touch within trigger window

    hard_sl = rejection_high * (1.0 + SL_PCT_ABOVE_REJECTION_HIGH / 100.0)
    R = hard_sl - entry_price
    if R <= 0:
        return None

    t1_target = entry_price - T1_R * R
    t2_target = entry_price - T2_R * R

    # Path walk from ENTRY bar onwards (entry at bar's open; same bar's high/low
    # eligible for SL/T2; entry_offset is 1 for Mode B, 1-3 for Mode A).
    after = afternoon.iloc[rejection_idx + entry_offset:].copy()

    mfe_price = entry_price  # for SHORT, lower = more favorable
    mae_price = entry_price  # for SHORT, higher = more adverse
    closes_at_hhmm: Dict[int, float] = {}

    baseline_exit_ts = None
    baseline_exit_price = None
    baseline_exit_reason = None

    for bar in after.itertuples(index=False):
        ts = bar.date
        hi = float(bar.high)
        lo = float(bar.low)
        cl = float(bar.close)
        hhmm = int(bar.hhmm)

        mfe_price = min(mfe_price, lo)
        mae_price = max(mae_price, hi)
        closes_at_hhmm[hhmm] = cl

        if baseline_exit_price is None:
            if hi >= hard_sl:
                baseline_exit_ts, baseline_exit_price, baseline_exit_reason = ts, hard_sl, "stop"
            elif lo <= t2_target:
                baseline_exit_ts, baseline_exit_price, baseline_exit_reason = ts, t2_target, "t2_full"
            elif ts.strftime("%H:%M") >= EXIT_BAR_HHMM:
                baseline_exit_ts, baseline_exit_price, baseline_exit_reason = ts, cl, "time_stop"

    if baseline_exit_price is None:
        last = after.iloc[-1]
        baseline_exit_ts = last["date"]
        baseline_exit_price = float(last["close"])
        baseline_exit_reason = "last_bar"

    mfe_r = (entry_price - mfe_price) / R
    mae_r = (mae_price - entry_price) / R

    def _close_at(target_hhmm: int) -> float:
        eligible = [v for k, v in closes_at_hhmm.items() if k <= target_hhmm]
        return float(eligible[-1]) if eligible else float("nan")

    qty = max(int(RISK_PER_TRADE_RUPEES / max(R, 1e-6)), 1)
    realized_pnl = (entry_price - baseline_exit_price) * qty
    fee = calc_fee(entry_price, baseline_exit_price, qty, "SELL")
    net_pnl = realized_pnl - fee

    return {
        "trade_date": daily_row["d"],
        "signal_ts": signal_ts,
        "signal_close": signal_close_px,
        "symbol": daily_row["symbol"],
        "side": "SHORT",
        "signal_type": "circuit_release_failed_retest",
        "day_high": float(daily_row["high"]),       # EOD (metadata only, not used for filtering)
        "day_low": float(daily_row["low"]),         # EOD (metadata only)
        "session_high_at_signal": session_high_at_signal,
        "day_close": float(daily_row["close"]),
        "pdc": pdc,
        "day_gain_pct": float(daily_row["day_gain_pct"]),
        "close_off_high_pct": float(daily_row["close_off_high_pct"]),
        "entry_ts": entry_ts,
        "entry_price": entry_price,
        "rejection_high": rejection_high,
        "hard_sl": hard_sl,
        "t1_target": t1_target,
        "t2_target": t2_target,
        "R_per_share": R,
        "qty": qty,
        "exit_ts": baseline_exit_ts,
        "exit_price": baseline_exit_price,
        "exit_reason": baseline_exit_reason,
        "mfe_r": mfe_r,
        "mae_r": mae_r,
        "close_at_1300": _close_at(1300),
        "close_at_1400": _close_at(1400),
        "close_at_1500": _close_at(1500),
        "realized_pnl": realized_pnl,
        "fee": fee,
        "net_pnl": net_pnl,
    }


# ----- Driver -----

def main() -> int:
    global WINDOW_START, WINDOW_END, WINDOW_LABEL

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", choices=list(WINDOWS.keys()), default="discovery")
    parser.add_argument("--out", default=None)
    parser.add_argument("--entry-mode", choices=["B", "A"], default="B",
                        help="Entry model: B=next-bar-open (default, idealized); "
                             "A=tick-zone-touch (walk subsequent bars, fill when range "
                             "intersects entry_zone, EXPIRE if no touch within 3 bars)")
    args = parser.parse_args()
    global ENTRY_MODE
    ENTRY_MODE = args.entry_mode

    WINDOW_LABEL = args.window
    WINDOW_START, WINDOW_END = WINDOWS[WINDOW_LABEL]

    out_path = Path(args.out) if args.out else (
        _REPO_ROOT / "reports" / "sub9_sanity"
        / f"_circuit_release_fade_short_trades_{WINDOW_LABEL}.csv"
    )

    print(f"== sanity_circuit_release_fade ({WINDOW_LABEL}: {WINDOW_START} -> {WINDOW_END}) ==")

    # 1. Daily data with PDC
    daily = load_daily_with_pdc()
    # 2. Filter to circuit-pin days
    candidates = filter_circuit_pin_days(daily)

    # 3. Filter by cap segment + MIS
    print(f"  filtering by cap segment + MIS-eligibility ...")
    candidates = candidates.copy()
    candidates["cap_segment"] = candidates["symbol"].apply(_cap_segment)
    candidates["_mis"] = candidates["symbol"].apply(_is_mis_eligible)
    candidates = candidates[
        candidates["cap_segment"].isin(ALLOWED_CAPS) & candidates["_mis"]
    ]
    print(f"    in {sorted(ALLOWED_CAPS)}+MIS: {len(candidates):,}")
    print(f"  cap distribution:\n{candidates['cap_segment'].value_counts().to_string()}")

    if candidates.empty:
        print("  NO CANDIDATES - exit")
        return 0

    # 4. Load 5m bars for ONLY the candidate (symbol, day) keys
    candidate_keys = set(zip(candidates["symbol"], candidates["d"]))
    symbols_needed = set(candidates["symbol"].unique())
    print(f"  loading 5m feathers; filtering to {len(candidate_keys)} candidate (symbol,day) keys ...")

    bars_by_key: Dict[tuple, pd.DataFrame] = {}
    for (y, m) in _months_in_window():
        mdf = _load_5m_for_month(y, m)
        if mdf.empty:
            continue
        mdf = mdf[mdf["symbol"].isin(symbols_needed)]
        if mdf.empty:
            continue
        mdf["d"] = mdf["date"].dt.date
        mdf["_key"] = list(zip(mdf["symbol"], mdf["d"]))
        mdf = mdf[mdf["_key"].isin(candidate_keys)]
        if mdf.empty:
            continue
        for (s, dd), grp in mdf.groupby(["symbol", "d"], sort=False):
            bars_by_key[(s, dd)] = grp.sort_values("date").reset_index(drop=True)
        del mdf
    print(f"    (symbol, day) keys loaded: {len(bars_by_key):,}")

    # 5. Simulate
    print(f"  simulating {len(candidates):,} candidate days ...")
    trades: List[dict] = []
    skipped_no_bars = 0
    skipped_no_morning_pin = 0
    skipped_no_retest = 0

    for _, drow in candidates.iterrows():
        key = (drow["symbol"], drow["d"])
        day_bars = bars_by_key.get(key)
        if day_bars is None or day_bars.empty:
            skipped_no_bars += 1
            continue
        trade = _simulate_one(drow, day_bars)
        if trade is None:
            skipped_no_retest += 1
            continue
        trade["cap_segment"] = drow["cap_segment"]
        trades.append(trade)

    print(f"    trades fired:        {len(trades):,}")
    print(f"    skipped no 5m bars:  {skipped_no_bars:,}")
    print(f"    skipped no retest:   {skipped_no_retest:,}")

    if not trades:
        print("  NO TRADES FIRED - exit")
        return 0

    tdf = pd.DataFrame(trades)

    # 6. Aggregate verdict
    n = len(tdf)
    wins = int((tdf["net_pnl"] > 0).sum())
    wr = 100.0 * wins / n
    net = float(tdf["net_pnl"].sum())
    gross_wins = float(tdf[tdf["net_pnl"] > 0]["net_pnl"].sum())
    gross_losses = -float(tdf[tdf["net_pnl"] < 0]["net_pnl"].sum())
    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")
    avg_win = gross_wins / wins if wins else 0.0
    avg_loss = gross_losses / max(n - wins, 1)
    sharpe = (tdf["net_pnl"].mean() / tdf["net_pnl"].std(ddof=1)) * (252 ** 0.5) if tdf["net_pnl"].std(ddof=1) > 0 else 0.0
    exit_dist = tdf["exit_reason"].value_counts(normalize=True).to_dict()

    print()
    print("=" * 80)
    print(f"  C-03 CIRCUIT RELEASE FADE (SHORT) - {WINDOW_LABEL.upper()} VERDICT")
    print("=" * 80)
    print(f"  n trades:               {n:,}")
    print(f"  win rate:               {wr:.1f}%  ({wins} wins / {n - wins} losses)")
    print(f"  Profit Factor:          {pf:.3f}")
    print(f"  NET PnL (after fees):   Rs. {net:>+12,.0f}")
    print(f"  Avg win / avg loss:     Rs. {avg_win:>+8,.0f}  /  Rs. {avg_loss:>+8,.0f}")
    print(f"  Annualized Sharpe:      {sharpe:.2f}")
    print(f"  Exit reason mix:        {exit_dist}")
    print()

    print("  Per-cap-segment breakdown:")
    cap_grp = tdf.groupby("cap_segment").agg(
        n=("net_pnl", "count"),
        net=("net_pnl", "sum"),
        wr=("net_pnl", lambda s: 100.0 * (s > 0).sum() / len(s)),
    )
    cap_grp["pf"] = tdf.groupby("cap_segment").apply(
        lambda s: (s[s["net_pnl"] > 0]["net_pnl"].sum() /
                   max(-s[s["net_pnl"] < 0]["net_pnl"].sum(), 1e-9))
    )
    print(cap_grp.round(2).to_string())
    print()

    if n < 200:
        verdict = "STRUCTURAL RETIRE-PRE-DATA (n < 200)"
    elif pf >= 1.10:
        verdict = "STRONG PROCEED -> cell-mine + R-sweep"
    elif pf >= 1.0:
        verdict = "MARGINAL -> cell selection may rescue"
    else:
        verdict = "THESIS RETIRE (PF < 1.0)"
    print(f"  VERDICT: {verdict}")
    print()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tdf.to_csv(out_path, index=False)
    print(f"  trades CSV saved: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
