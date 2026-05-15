"""Per-setup qualifying-universe contribution.

Architectural problem (2026-05-12): Stage-0 EnergyScanner ranks ~1500 symbols
down to top-1000 by intraday features (vol_z, vwap_distance, ret_1). This
works for momentum setups, but cross-day / event-driven setups have
qualifying signals INVISIBLE to Stage-0:

  - circuit_t1_fade_short: needs yesterday's circuit hit (cross-day)
  - delivery_pct_anomaly_short: needs yesterday's delivery_pct (cross-day)
  - gap_fade_short: needs gap-from-PDC (cross-day; Stage-0 ignores explicitly)

Without this module, qualifying symbols for these setups are silently
dropped at Stage-0 because their intraday rank is below the top-1000 cap.

Solution: each setup contributes a deterministic universe of qualifying
symbols for the session. Screener unions this with Stage-0 shortlist so
qualifying symbols are NEVER lost. Stage-0 keeps optimizing for momentum;
setup universes guarantee event-driven visibility.

These functions are pure data-driven — no detector instance needed.
"""
from __future__ import annotations

import logging
from datetime import date, time
from pathlib import Path
from typing import Any, Dict, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)

# Tunable safety cap on the union. Stage-0 cap is 1000 (500 long + 500 short).
# We allow up to MAX_EXTRA per cross-day setup; cap final union to prevent
# runaway memory if e.g. delivery_pct anomaly day flags 500 symbols.
MAX_EXTRA_PER_SETUP = 200


# ---------------------------------------------------------------------------
# circuit_t1_fade_short — yesterday's upper-circuit hit
# ---------------------------------------------------------------------------

def circuit_t1_universe(
    daily_dict: Dict[str, pd.DataFrame],
    session_date: date,
    config: Dict[str, Any],
) -> Set[str]:
    """Symbols where T-1 hit an upper circuit (qualifying T0 for T+1 fade).

    Mirrors the sanity-script + production-detector logic:
      - T-1 day close >= min_pct_change% above prior close (catches 5/10/20% bands)
      - close >= 99.5% of day high (clamped at top)
      - day volume >= 1.5x trailing 20-day avg
      - cap_segment in allowed (mid/small)
    The detector itself does fresh checks; this universe just ensures the
    symbol survives Stage-0 to reach the detector.
    """
    if not daily_dict:
        return set()

    min_pct = float(config.get("t0_min_pct_change", 4.5))
    high_to_close_min = float(config.get("t0_high_to_close_min", 0.995))
    min_vol_ratio = float(config.get("t0_min_vol_vs_20d", 1.5))

    # daily_dict values come pre-filtered to dates < session_date by the
    # broker (see broker/mock/mock_broker.py: get_daily — slices strictly
    # before session_date). So the last row is T-1 (yesterday).
    qual: Set[str] = set()
    for sym, ddf in daily_dict.items():
        if ddf is None or ddf.empty or len(ddf) < 21:
            continue
        try:
            t_minus_1 = ddf.iloc[-1]
            window_20 = ddf.iloc[-21:-1]   # 20-day window ENDING at T-2

            prev_close = float(window_20["close"].iloc[-1])
            t1_close = float(t_minus_1["close"])
            t1_high = float(t_minus_1["high"])
            t1_vol = float(t_minus_1["volume"])
            vol_avg_20d = float(window_20["volume"].mean())

            if prev_close <= 0 or vol_avg_20d <= 0:
                continue
            pct_change = (t1_close - prev_close) / prev_close * 100.0
            high_to_close = t1_close / t1_high if t1_high > 0 else 0.0
            vol_ratio = t1_vol / vol_avg_20d

            if (pct_change >= min_pct
                    and high_to_close >= high_to_close_min
                    and vol_ratio >= min_vol_ratio):
                qual.add(sym)
                if len(qual) >= MAX_EXTRA_PER_SETUP:
                    break
        except Exception:
            continue
    logger.info("setup_universe.circuit_t1: %d qualifying symbols on %s",
                len(qual), session_date)
    return qual


# earnings_day_universe removed 2026-05-14 (earnings_day_intraday_fade retired
# — see docs/retired_setups.md).


# ---------------------------------------------------------------------------
# delivery_pct_anomaly_short — yesterday's delivery_pct anomaly
# ---------------------------------------------------------------------------

def delivery_pct_universe(
    daily_dict: Dict[str, pd.DataFrame],
    session_date: date,
    config: Dict[str, Any],
) -> Set[str]:
    """Symbols with prior-day LOW delivery_pct + price pump (qualifying for T+1 fade).

    The validated cell-mined signal is LOW delivery (<=20%) + same-day pump
    (>=3% return) = retail/operator pump signature → T+1 fade.
    Earlier version used `delivery_pct_min=50.0` which filtered for HIGH delivery —
    the inverse of the actual signal — populating the universe with non-qualifiers
    and excluding the symbols the detector wants to short.

    Daily DataFrames must already be enriched with `delivery_pct` column
    (services.delivery_pct_enrichment runs this at seed time).
    """
    if not daily_dict:
        return set()

    delivery_max = float(config.get("delivery_pct_max", 20.0))
    min_prior_day_return_pct = float(config.get("min_prior_day_return_pct", 3.0))
    qual: Set[str] = set()
    for sym, ddf in daily_dict.items():
        if ddf is None or len(ddf) < 2 or "delivery_pct" not in ddf.columns:
            continue
        try:
            # daily_dict is pre-filtered < session_date — last row is T-1, prior is T-2.
            t_minus_1 = ddf.iloc[-1]
            t_minus_2 = ddf.iloc[-2]
            dp_val = t_minus_1.get("delivery_pct") if hasattr(t_minus_1, "get") else t_minus_1["delivery_pct"]
            dp = float(dp_val) if pd.notna(dp_val) else 100.0
            if dp > delivery_max:
                continue
            # Day-over-day return: (T-1 close - T-2 close) / T-2 close.
            # Matches sanity's daily_return_pct field — the pump-fade signal is
            # the T-1 day-on-day pump >= 3% combined with low delivery_pct.
            close_t1 = float(t_minus_1["close"]) if pd.notna(t_minus_1["close"]) else 0.0
            close_t2 = float(t_minus_2["close"]) if pd.notna(t_minus_2["close"]) else 0.0
            if close_t2 <= 0:
                continue
            ret_pct = (close_t1 - close_t2) / close_t2 * 100.0
            if ret_pct < min_prior_day_return_pct:
                continue
            qual.add(sym)
            if len(qual) >= MAX_EXTRA_PER_SETUP:
                break
        except Exception:
            continue
    logger.info("setup_universe.delivery_pct: %d qualifying symbols on %s",
                len(qual), session_date)
    return qual


# ---------------------------------------------------------------------------
# gap_fade_short — today's gap-up from PDC (computed at 09:15 bar)
# ---------------------------------------------------------------------------

def long_panic_gap_down_universe(
    df5_today_by_symbol: Dict[str, pd.DataFrame],
    daily_dict: Dict[str, pd.DataFrame],
    session_date: date,
    config: Dict[str, Any],
    cap_map: Dict[str, str],
) -> Set[str]:
    """Symbols passing the BROADER long_panic_gap_down filter at the 09:15 bar.

    Detector gates (broader filter — superset of narrow Cell B band):
      - cap_segment in allowed (small/mid)
      - first-5m gap_pct <= gap_pct_max (e.g. -1%)
      - dist_from_pdh <= dist_from_pdh_pct_max (e.g. -5.5%)
      - dist_from_pdl <= broader_dist_from_pdl_pct_max (e.g. -1.25%)

    Returns the qualifying symbol set. Caller is responsible for adding to
    the screener's scan extras AND for seeding services.regime_density_tracker
    so the detector's regime guard activates BEFORE any per-symbol fire.

    Built lazily once the 09:15 bar is observed; cached for the session.
    """
    allowed_caps = set(config.get("allowed_cap_segments", ["small_cap", "mid_cap"]))
    gap_pct_max = float(config["gap_pct_max"])
    dist_pdh_max = float(config["dist_from_pdh_pct_max"])
    broader_dist_pdl_max = float(config["broader_dist_from_pdl_pct_max"])

    qual: Set[str] = set()
    for sym, df5 in df5_today_by_symbol.items():
        if df5 is None or df5.empty:
            continue
        if cap_map.get(sym) not in allowed_caps:
            continue
        # df5_today_by_symbol despite its name contains warmup bars from
        # prior sessions (screener cache spans many days). Filter to TODAY
        # before reading the 09:15 bar.
        try:
            today_bars = df5[df5.index.date == session_date]
        except Exception:
            continue
        if today_bars.empty:
            continue
        first_bar = today_bars.iloc[0]
        try:
            today_open = float(first_bar["open"])
            today_close = float(first_bar["close"])
        except Exception:
            continue
        ddf = daily_dict.get(sym)
        if ddf is None or ddf.empty:
            continue
        try:
            pdc = float(ddf.iloc[-1]["close"])
            pdh = float(ddf.iloc[-1]["high"])
            pdl = float(ddf.iloc[-1]["low"])
        except Exception:
            continue
        if pdc <= 0 or pdh <= 0 or pdl <= 0:
            continue
        gap_pct = ((today_open - pdc) / pdc) * 100.0
        if gap_pct > gap_pct_max:
            continue
        dist_pdh = ((today_close - pdh) / pdh) * 100.0
        if dist_pdh > dist_pdh_max:
            continue
        dist_pdl = ((today_close - pdl) / pdl) * 100.0
        if dist_pdl > broader_dist_pdl_max:
            continue
        qual.add(sym)
        if len(qual) >= MAX_EXTRA_PER_SETUP:
            break
    logger.info(
        "setup_universe.long_panic_gap_down: %d qualifying panic-gap-downs on %s",
        len(qual), session_date,
    )
    return qual


def gap_fade_universe(
    df5_today_by_symbol: Dict[str, pd.DataFrame],
    daily_dict: Dict[str, pd.DataFrame],
    session_date: date,
    config: Dict[str, Any],
    cap_map: Dict[str, str],
) -> Set[str]:
    """Symbols with qualifying morning gap-up at 09:15 bar.

    Detector gates:
      - cap_segment in allowed (default: small_cap)
      - (open - PDC) / PDC ∈ [min_gap_pct, max_gap_pct] (as fraction, e.g. 0.015)

    Caller should invoke this once per session AFTER 09:15 bar is available,
    then add the resulting symbols to the shortlist for that bar and all
    subsequent 09:20/09:25 fires (gap_fade window).
    """
    allowed_caps = set(config.get("allowed_cap_segments", ["small_cap"]))
    min_gap_pct = float(config.get("min_gap_pct_above_pdc", 1.5)) / 100.0
    max_gap_pct = float(config.get("max_gap_pct_above_pdc", 8.0)) / 100.0

    qual: Set[str] = set()
    for sym, df5 in df5_today_by_symbol.items():
        if df5 is None or df5.empty:
            continue
        if cap_map.get(sym) not in allowed_caps:
            continue
        # First bar of session = 09:15
        first_bar = df5.iloc[0]
        try:
            today_open = float(first_bar["open"])
        except Exception:
            continue
        # PDC from daily_dict (already pre-filtered < session_date — last row is T-1).
        ddf = daily_dict.get(sym)
        if ddf is None or ddf.empty:
            continue
        try:
            pdc = float(ddf.iloc[-1]["close"])
        except Exception:
            continue
        if pdc <= 0:
            continue
        gap = (today_open - pdc) / pdc
        if min_gap_pct <= gap <= max_gap_pct:
            qual.add(sym)
            if len(qual) >= MAX_EXTRA_PER_SETUP:
                break
    logger.info("setup_universe.gap_fade: %d qualifying gappers on %s",
                len(qual), session_date)
    return qual


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def compute_static_universes(
    setups_cfg: Dict[str, Dict[str, Any]],
    daily_dict: Dict[str, pd.DataFrame],
    session_date: date,
) -> Dict[str, Set[str]]:
    """Compute session-start universes (cross-day setups only).

    gap_fade is dynamic (needs 09:15 bar) — call gap_fade_universe separately.

    Returns dict: {setup_name: set_of_symbols}. Setups with no universe
    contribution (e.g. gap_fade, or any setup that relies on Stage-0
    momentum) are absent from the result.
    """
    out: Dict[str, Set[str]] = {}
    if not setups_cfg or not daily_dict:
        return out

    # circuit_t1
    cfg = (setups_cfg.get("circuit_t1_fade_short") or {})
    if cfg.get("enabled"):
        out["circuit_t1_fade_short"] = circuit_t1_universe(daily_dict, session_date, cfg)

    # delivery_pct
    cfg = (setups_cfg.get("delivery_pct_anomaly_short") or {})
    if cfg.get("enabled"):
        out["delivery_pct_anomaly_short"] = delivery_pct_universe(daily_dict, session_date, cfg)

    return out
