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

# Repo root, so config-relative data paths (e.g. the MIS-short eligibility map)
# resolve regardless of the run harness's cwd. The full-backtest harness
# (tools/_gen_full.py) runs main.py from a different cwd than a repo-root
# dry-run; a bare relative path silently missed there -> fail-closed ->
# up_spike_fade_short universe=0 in the OCI full run while a repo-root
# single-day run loaded it fine (Lesson #26: relative paths in config are
# cwd-fragile; resolve against repo root).
_REPO_ROOT = Path(__file__).resolve().parent.parent

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

    # Illiquidity filter (pre-reg specs/2026-06-03-gap_fade_short-illiquidity-filter-prereg.md):
    # gap_fade's fade edge concentrates in illiquid small-caps (illiquid-tertile PF ~2.0-3.7
    # vs liquid ~1.0, validated Disc/OOS/HO on OCI). Drift-robust: keep the bottom ADV-quantile
    # among THIS session's gappers. DEFAULT OFF (quantile 1.0 = keep all) until paper A/B
    # promotes it (set illiquid_filter_enabled=true + illiquid_adv_quantile=0.33).
    illiquid_enabled = bool(config.get("illiquid_filter_enabled", False))
    illiquid_q = float(config.get("illiquid_adv_quantile", 1.0))

    cands: list = []  # (sym, adv20_turnover) for qualifying gappers
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
        if not (min_gap_pct <= gap <= max_gap_pct):
            continue
        # Trailing-20d ADV (turnover) from daily_dict (rows already < session_date — no look-ahead).
        # Guard short history (<21 rows) and missing/zero volume — those would yield a
        # spuriously tiny ADV and look "most illiquid", so set NaN to exclude from the filter.
        try:
            if len(ddf) >= 21 and "volume" in ddf.columns:
                tail = ddf.tail(20)
                adv20 = float((tail["close"] * tail["volume"]).mean())
                if not (adv20 > 0.0):
                    adv20 = float("nan")
            else:
                adv20 = float("nan")
        except Exception:
            adv20 = float("nan")
        cands.append((sym, adv20))

    if illiquid_enabled and 0.0 < illiquid_q < 1.0:
        advs = [a for _, a in cands if a == a]  # finite only
        if advs:
            thr = float(pd.Series(advs).quantile(illiquid_q))  # pd already imported; no hot-path numpy import
            cands = [(s, a) for (s, a) in cands if (a == a and a <= thr)]
            # Deterministic cap: sort most-illiquid first so the MAX_EXTRA_PER_SETUP
            # truncation is reproducible live vs backtest regardless of symbol arrival order.
            cands.sort(key=lambda sa: sa[1])

    qual: Set[str] = set()
    for sym, _adv in cands:
        qual.add(sym)
        if len(qual) >= MAX_EXTRA_PER_SETUP:
            break
    logger.info("setup_universe.gap_fade: %d qualifying gappers on %s (illiquid_filter=%s, q=%s)",
                len(qual), session_date, illiquid_enabled, illiquid_q)
    return qual


# ---------------------------------------------------------------------------
# below_vwap_volume_revert_long — 3D cell-lock static universe
# ---------------------------------------------------------------------------

def below_vwap_volume_revert_long_universe(
    daily_dict: Dict[str, pd.DataFrame],
    session_date: date,
    config: Dict[str, Any],
) -> Set[str]:
    """Symbols matching the 3D cell lock: cap_segment=unknown, MIS-eligible,
    minimum daily average volume, minimum trading-days coverage.

    Static universe — no intraday signal required at session start. The
    detector's per-bar filters (vwap_dev, vol_ratio, hhmm) determine when
    the signal actually fires.

    The cap=unknown cohort in nse_all.json has ~1,000 MIS-eligible symbols.
    The default MAX_EXTRA_PER_SETUP=200 cap silently dropped 81% of the
    eligible universe (smoke test 2026-05-21). Cap is now setup-overridable
    via config.universe_max_symbols (default high enough to include the full
    cohort).

    Spec: specs/2026-05-21-below_vwap_volume_revert_long-paper-trade-spec.md
    """
    from services.symbol_metadata import get_cap_segment, get_mis_info

    required_cap = str(config["cell_lock_cap_segment"])
    min_daily_avg_vol = float(config["min_daily_avg_volume"])
    min_days = int(config["min_trading_days_required"])
    max_symbols = int(config.get("universe_max_symbols", 1500))

    qual: Set[str] = set()
    for sym, ddf in daily_dict.items():
        bare = sym.replace("NSE:", "")
        nse_sym = f"NSE:{bare}"
        try:
            if get_cap_segment(nse_sym) != required_cap:
                continue
            if not get_mis_info(nse_sym).get("mis_enabled", False):
                continue
        except Exception:
            continue
        if ddf is None or ddf.empty or len(ddf) < min_days:
            continue
        try:
            avg_vol = float(ddf["volume"].mean())
        except Exception:
            continue
        if avg_vol < min_daily_avg_vol:
            continue
        qual.add(sym)
        if len(qual) >= max_symbols:
            logger.warning(
                "setup_universe.below_vwap_volume_revert_long: hit max_symbols=%d cap "
                "(eligible cohort may be larger; raise config.universe_max_symbols)",
                max_symbols,
            )
            break
    logger.info(
        "setup_universe.below_vwap_volume_revert_long: %d qualifying on %s",
        len(qual), session_date,
    )
    return qual


# ---------------------------------------------------------------------------
# panic_crash_revert_long — illiquid intraday capitulation snapback (LONG)
# ---------------------------------------------------------------------------

def _illiquid_cap_mis_universe(
    daily_dict: Dict[str, pd.DataFrame],
    session_date: date,
    config: Dict[str, Any],
    setup_label: str,
) -> Set[str]:
    """Shared static universe for the illiquid over-extension family.

    Cell lock: cap_segment in allowed_cap_segments (small/micro/unknown),
    MIS-eligible, minimum daily average volume, minimum trading-days coverage.
    Forks the below_vwap_volume_revert_long_universe cap=unknown/MIS pattern,
    generalized to a multi-cap allow-set.

    Static universe — no intraday signal required at session start. The
    detector's per-bar filters (r3, vol_burst, turnover, window) decide when
    the signal actually fires.
    """
    from services.symbol_metadata import get_cap_segment, get_mis_info

    allowed_caps = set(config["allowed_cap_segments"])
    min_daily_avg_vol = float(config["min_daily_avg_volume"])
    min_days = int(config["min_trading_days_required"])
    # Coarse daily-avg TURNOVER floor (Rs). The 'unknown' cap bucket is a
    # catch-all default (any symbol absent from the cap snapshot ~= the whole
    # non-index long tail), so cap alone admits ~1,700 names. The detector's
    # per-day Rs 1.5cr turnover floor is the real liquidity selector — this
    # coarse daily-AVG floor pre-trims names that can never reach it, shrinking
    # the live 5m-fetch universe ~3x WITHOUT clipping edge (validated signal
    # symbols have daily-avg turnover >= ~Rs 44L for spike / ~Rs 5L min for
    # panic; floor set well below). Set 0 to disable.
    min_daily_avg_turnover = float(config["min_daily_avg_turnover_inr"])
    max_symbols = int(config.get("universe_max_symbols", 1500))

    qual: Set[str] = set()
    for sym, ddf in daily_dict.items():
        bare = sym.replace("NSE:", "")
        nse_sym = f"NSE:{bare}"
        try:
            if get_cap_segment(nse_sym) not in allowed_caps:
                continue
            if not get_mis_info(nse_sym).get("mis_enabled", False):
                continue
        except Exception:
            continue
        if ddf is None or ddf.empty or len(ddf) < min_days:
            continue
        try:
            avg_vol = float(ddf["volume"].mean())
        except Exception:
            continue
        if avg_vol < min_daily_avg_vol:
            continue
        if min_daily_avg_turnover > 0.0:
            try:
                avg_turnover = float((ddf["close"] * ddf["volume"]).mean())
            except Exception:
                continue
            if not (avg_turnover >= min_daily_avg_turnover):
                continue
        qual.add(sym)
        if len(qual) >= max_symbols:
            logger.warning(
                "setup_universe.%s: hit max_symbols=%d cap "
                "(eligible cohort may be larger; raise config.universe_max_symbols)",
                setup_label, max_symbols,
            )
            break
    logger.info(
        "setup_universe.%s: %d qualifying on %s", setup_label, len(qual), session_date,
    )
    return qual


def panic_crash_revert_long_universe(
    daily_dict: Dict[str, pd.DataFrame],
    session_date: date,
    config: Dict[str, Any],
) -> Set[str]:
    """Static universe for panic_crash_revert_long.

    cap_segment in {small_cap, micro_cap, unknown} + MIS-eligible (LONG MIS,
    no shortability gate needed — see brief Section 5).

    Spec: specs/2026-06-03-brief-panic_crash_revert_long.md
    """
    return _illiquid_cap_mis_universe(
        daily_dict, session_date, config, "panic_crash_revert_long",
    )


# ---------------------------------------------------------------------------
# up_spike_fade_short — illiquid intraday up-spike fade (SHORT)
# ---------------------------------------------------------------------------

def up_spike_fade_short_universe(
    daily_dict: Dict[str, pd.DataFrame],
    session_date: date,
    config: Dict[str, Any],
) -> Set[str]:
    """Static universe for up_spike_fade_short.

    cap_segment in {small_cap, micro_cap, unknown} + MIS-eligible — identical to
    panic_crash_revert_long. SHORTABILITY is enforced by the EXISTING MIS layers,
    NOT a setup-specific gate: the screener filters core_symbols by the live
    Zerodha MIS-allowed list (early_mis_universe_filter ->
    services/state/zerodha_mis_fetcher.py) AND capital_manager.can_enter_position
    blocks non-MIS stocks for BUY *and* SELL (paper); in live the Zerodha broker
    rejects non-MIS shorts directly. Same path gap_fade_short and every other
    live short relies on — no third MIS mechanism.

    Spec: specs/2026-06-03-brief-up_spike_fade_short.md
    """
    return _illiquid_cap_mis_universe(
        daily_dict, session_date, config, "up_spike_fade_short",
    )


# ---------------------------------------------------------------------------
# close_dn_overnight_long — overnight CNC/MTF setup, cell-locked
# ---------------------------------------------------------------------------

def close_dn_overnight_long_universe(
    daily_dict: Dict[str, pd.DataFrame],
    session_date: date,
    config: Dict[str, Any],
) -> Set[str]:
    """Universe builder for close_dn_overnight_long.

    Cap segment: large/mid/small/unknown (NOT micro).
    Eligibility: MTF-approved → MTF at fire time, else CNC fallback.
    ETFs excluded (different mechanism; the equity closing-flush thesis
    doesn't apply to ETF NAV-tracking microstructure).

    Aligned with backtest sanity
    (tools/sub9_research/sanity_close_dn_overnight_multi_exit.py) which
    iterated over all cap-eligible NSE equities with NO MIS gate and NO
    MTF-only restriction. MIS filter + unknown-cap-non-MTF exclusion were
    dropped 2026-06-09 because they silently shrank production's universe
    vs the population PF 2.44 was measured on. Liquidity floor
    (min_daily_avg_volume=50k, min_trading_days_required=16) matches
    sanity's MIN_DAILY_AVG_VOLUME=50_000 + 80%-coverage-of-20-day-rolling.

    Spec: specs/2026-05-21-close_dn_overnight_long-paper-trade-implementation-spec.md
    Cell-lock: tools/sub9_research/close_dn_overnight_long_cell_lock.json
    """
    from services.symbol_metadata import get_cap_segment
    from services.mtf_universe import MtfUniverse

    min_daily_avg_vol = float(config["min_daily_avg_volume"])
    min_days = int(config["min_trading_days_required"])
    max_symbols = int(config.get("universe_max_symbols", 1500))

    mtf_cfg = config.get("mtf", {})
    snapshot_path = Path(str(mtf_cfg["approved_list_snapshot_path"]))
    stale_warn_days = int(mtf_cfg.get("stale_snapshot_warn_days", 7))
    exclude_etf = bool(mtf_cfg.get("exclude_etf", True))

    mtf = MtfUniverse(snapshot_path)
    age = mtf.snapshot_age_days()
    if age > stale_warn_days:
        logger.warning(
            "setup_universe.close_dn_overnight_long: MTF_SNAPSHOT_STALE "
            "(%d days old; refresh tools/scrape_zerodha_mtf.py recommended)",
            age,
        )

    accepted_caps = {"large_cap", "mid_cap", "small_cap", "unknown"}

    qual: Set[str] = set()
    for sym, ddf in daily_dict.items():
        bare = sym.replace("NSE:", "")
        nse_sym = f"NSE:{bare}"
        try:
            cap = get_cap_segment(nse_sym)
            if cap not in accepted_caps:
                continue
        except Exception:
            continue
        if ddf is None or ddf.empty or len(ddf) < min_days:
            continue
        try:
            avg_vol = float(ddf["volume"].mean())
        except Exception:
            continue
        if avg_vol < min_daily_avg_vol:
            continue
        # ETF block — ETFs don't fit the equity closing-flush thesis even
        # when MTF-approved.
        mtf_info = mtf.lookup(bare)
        if exclude_etf and mtf_info is not None and mtf_info.category == "etf":
            continue
        # Accept everything else. MTF-approved gets MTF order at fire
        # time; non-MTF falls back to CNC (no leverage). Backtest sanity
        # included CNC-fallback names across all cap segments — same here.
        qual.add(sym)
        if len(qual) >= max_symbols:
            logger.warning(
                "setup_universe.close_dn_overnight_long: hit max_symbols=%d cap "
                "(raise config.universe_max_symbols if needed)",
                max_symbols,
            )
            break
    logger.info(
        "setup_universe.close_dn_overnight_long: %d qualifying on %s",
        len(qual), session_date,
    )
    return qual


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def _cap_mis_static_universe(
    daily_dict: Dict[str, pd.DataFrame],
    config: Dict[str, Any],
    *,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    max_symbols: Optional[int] = None,
) -> Set[str]:
    """Generic static universe builder: filter by cap_segment + MIS + optional price band.

    Used by setups whose universe needs are purely static at session start
    (no intraday signal required for inclusion). Detector's per-bar filters
    still kick in to determine signal eligibility.

    Returns symbols in the SAME FORMAT as the keys of `daily_dict` (typically
    NSE-prefixed, e.g. "NSE:RELIANCE"). This must match the screener's
    `core_symbols` format - returning bare symbols here causes a silent
    intersection-with-core mismatch that drops the entire universe (bug
    discovered 2026-05-16 smoke test).
    """
    from services.symbol_metadata import get_cap_segment, get_mis_info

    allowed_caps = set(config.get("allowed_cap_segments", ["small_cap", "mid_cap"]))
    cap = int(max_symbols) if max_symbols is not None else MAX_EXTRA_PER_SETUP
    qual: Set[str] = set()
    for sym, ddf in daily_dict.items():
        # symbol may be "NSE:XYZ" or "XYZ" — daily_dict keys vary by source.
        # Preserve the original key format in the returned set so it matches
        # whatever the screener uses as core_symbols.
        bare = sym.replace("NSE:", "")
        nse_sym = f"NSE:{bare}"
        try:
            if get_cap_segment(nse_sym) not in allowed_caps:
                continue
            if not get_mis_info(nse_sym).get("mis_enabled", False):
                continue
        except Exception:
            continue
        # Price band filter using yesterday's close (PDC) as proxy
        if (min_price is not None or max_price is not None) and ddf is not None and not ddf.empty:
            try:
                pdc = float(ddf.iloc[-1]["close"])
                if min_price is not None and pdc < min_price:
                    continue
                if max_price is not None and pdc > max_price:
                    continue
            except Exception:
                continue
        qual.add(sym)  # ← preserve the daily_dict key format (NSE-prefixed)
        if len(qual) >= cap:
            break
    return qual


def or_window_failure_fade_short_universe(daily_dict, session_date, config):
    """C-10 universe: small_cap + MIS-eligible (static).

    Universe is ~500-700 small_cap MIS-eligible symbols. Raise cap above
    default MAX_EXTRA_PER_SETUP=200 to avoid silent truncation.
    """
    qual = _cap_mis_static_universe(daily_dict, config, max_symbols=1000)
    logger.info("setup_universe.or_window_failure_fade_short: %d symbols on %s", len(qual), session_date)
    return qual


def compute_static_universes(
    setups_cfg: Dict[str, Dict[str, Any]],
    daily_dict: Dict[str, pd.DataFrame],
    session_date: date,
) -> Dict[str, Set[str]]:
    """Compute session-start universes (cross-day + static-filter setups).

    Dynamic setups (gap_fade, long_panic_gap_down, circuit_release_fade) are
    lazy-built later — call their specific universe functions separately.

    Returns dict: {setup_name: set_of_symbols}. Setups with no universe
    contribution (e.g. gap_fade pre-09:15) are absent from the result.
    """
    out: Dict[str, Set[str]] = {}
    if not setups_cfg or not daily_dict:
        return out

    # circuit_t1 (cross-day)
    cfg = (setups_cfg.get("circuit_t1_fade_short") or {})
    if cfg.get("enabled"):
        out["circuit_t1_fade_short"] = circuit_t1_universe(daily_dict, session_date, cfg)

    # delivery_pct (cross-day)
    cfg = (setups_cfg.get("delivery_pct_anomaly_short") or {})
    if cfg.get("enabled"):
        out["delivery_pct_anomaly_short"] = delivery_pct_universe(daily_dict, session_date, cfg)

    # round_number_sweep_short: RETIRED 2026-05-19 (see docs/retired_setups.md)

    # or_window_failure_fade_short (C-10, static cap+MIS filter)
    cfg = (setups_cfg.get("or_window_failure_fade_short") or {})
    if cfg.get("enabled"):
        out["or_window_failure_fade_short"] = or_window_failure_fade_short_universe(daily_dict, session_date, cfg)

    # below_vwap_volume_revert_long (3D cell-lock static universe)
    # Was previously orphaned — the universe function existed but no dispatch
    # entry called it, so `_setup_universes` was missing the key entirely.
    # That blocked the cross-day RVOL baseline populate (which unions
    # rvol-dependent universes) from including below_vwap symbols, leaving
    # the detector to silently no-fire with "baseline volume unavailable".
    cfg = (setups_cfg.get("below_vwap_volume_revert_long") or {})
    if cfg.get("enabled"):
        out["below_vwap_volume_revert_long"] = below_vwap_volume_revert_long_universe(daily_dict, session_date, cfg)

    # panic_crash_revert_long (illiquid capitulation snapback, static cap+MIS)
    cfg = (setups_cfg.get("panic_crash_revert_long") or {})
    if cfg.get("enabled"):
        out["panic_crash_revert_long"] = panic_crash_revert_long_universe(daily_dict, session_date, cfg)

    # up_spike_fade_short (illiquid up-spike fade, static cap+MIS + leverage>1 gate)
    cfg = (setups_cfg.get("up_spike_fade_short") or {})
    if cfg.get("enabled"):
        out["up_spike_fade_short"] = up_spike_fade_short_universe(daily_dict, session_date, cfg)

    # mis_unwind_vwap_revert_short: RETIRED 2026-05-19 (see docs/retired_setups.md)
    # circuit_release_fade_short: RETIRED 2026-05-19 (see docs/retired_setups.md)

    return out
