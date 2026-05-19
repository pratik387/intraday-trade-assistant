"""Worker-side entry point. Runs in ProcessPoolExecutor workers.

Module-level detector instance cache keyed by setup name. Cache populated
lazily on first use within the worker process; survives across batches.

dispatch_worker_batch returns list[(sym, GateDecision)] so the existing
downstream reduction loop in screener_live (_run_5m_scan) is UNCHANGED.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from services.dispatch.planner import Batch
from services.dispatch.setup_registry import SetupRegistry, _import_path


# Per-worker-process caches.
_detector_cache: dict = {}
_registry_cache: Optional[SetupRegistry] = None


def init_worker(registry) -> None:
    """Initialize worker-process state. Called on worker spawn or in tests."""
    global _registry_cache
    _registry_cache = registry
    _detector_cache.clear()


def _get_detector(name: str):
    if name in _detector_cache:
        return _detector_cache[name]
    if _registry_cache is None:
        raise RuntimeError("worker not initialized — call init_worker first")
    spec = _registry_cache.get(name)
    cls = _import_path(spec.detector_class_path)
    instance = cls(spec.raw_config)
    _detector_cache[name] = instance
    return instance


def _build_market_context(
    sym: str,
    df5: pd.DataFrame,
    levels: dict,
    *,
    bar_ts=None,
    session_date=None,
    regime: str = "chop",
    cap_segment: str = "unknown",
    df_daily: Optional[pd.DataFrame] = None,
):
    """Build a real MarketContext for the detector.

    Mirrors MainDetector._create_market_context so individual structure
    detectors receive the same context shape as via the old dispatch path.
    Indicators (vol_z, atr) are computed inline if not already in df5.
    """
    from structures.data_models import MarketContext

    try:
        d = df5.copy()

        # Ensure vol_z exists
        if "vol_z" not in d.columns:
            vol_mean = d["volume"].rolling(20, min_periods=10).mean()
            vol_std = d["volume"].rolling(20, min_periods=10).std()
            d["vol_z"] = (d["volume"] - vol_mean) / vol_std.replace(0, np.nan)

        # Ensure atr exists
        if "atr" not in d.columns:
            hl = d["high"] - d["low"]
            hc = np.abs(d["high"] - d["close"].shift())
            lc = np.abs(d["low"] - d["close"].shift())
            tr = np.maximum(hl, np.maximum(hc, lc))
            d["atr"] = tr.rolling(14, min_periods=10).mean()

        indicators = {
            "vol_z": float(d["vol_z"].iloc[-1]) if not pd.isna(d["vol_z"].iloc[-1]) else 0.0,
            "atr": float(d["atr"].iloc[-1]) if not pd.isna(d["atr"].iloc[-1]) else 1.0,
        }

        bar_timestamp = pd.to_datetime(d.index[-1])
        _session_date = session_date or bar_timestamp.date()

        return MarketContext(
            symbol=sym,
            current_price=float(d["close"].iloc[-1]),
            timestamp=bar_timestamp,
            df_5m=d,
            session_date=_session_date,
            df_daily=df_daily,
            orh=levels.get("ORH"),
            orl=levels.get("ORL"),
            pdh=levels.get("PDH"),
            pdl=levels.get("PDL"),
            pdc=levels.get("PDC"),
            regime=regime,
            cap_segment=cap_segment,
            indicators=indicators,
        )
    except Exception:
        # Fallback: pass df5 through as-is — detectors that don't need computed
        # indicators will still work.
        from structures.data_models import MarketContext
        bar_timestamp = pd.to_datetime(df5.index[-1]) if not df5.empty else pd.Timestamp("now")
        _session_date = session_date or bar_timestamp.date()
        return MarketContext(
            symbol=sym,
            current_price=float(df5["close"].iloc[-1]) if not df5.empty else 0.0,
            timestamp=bar_timestamp,
            df_5m=df5,
            session_date=_session_date,
            df_daily=df_daily,
            orh=levels.get("ORH"),
            orl=levels.get("ORL"),
            pdh=levels.get("PDH"),
            pdl=levels.get("PDL"),
            pdc=levels.get("PDC"),
            regime=regime,
            cap_segment=cap_segment,
            indicators={},
        )


# ---------------------------------------------------------------------------
# Structure → SetupType mapping (extracted from MainDetector for standalone use).
# Keep in sync with structures/main_detector.py:_map_structure_to_setup_type.
# ---------------------------------------------------------------------------
_STRUCTURE_TO_SETUP_TYPE: dict[str, str] = {
    # Level breakouts
    "level_breakout_long": "breakout_long",
    "level_breakout_short": "breakout_short",
    "orb_level_breakout_long": "orb_level_breakout_long",
    "orb_level_breakout_short": "orb_level_breakout_short",
    # Failure fades
    "failure_fade_long": "failure_fade_long",
    "failure_fade_short": "failure_fade_short",
    # Squeeze
    "squeeze_release_long": "squeeze_release_long",
    "squeeze_release_short": "squeeze_release_short",
    # Flag
    "flag_continuation_long": "flag_continuation_long",
    "flag_continuation_short": "flag_continuation_short",
    # Momentum
    "momentum_breakout_long": "momentum_breakout_long",
    "momentum_breakout_short": "momentum_breakout_short",
    "momentum_trend_long": "trend_pullback_long",
    "momentum_trend_short": "trend_pullback_short",
    # ICT
    "order_block_long": "order_block_long",
    "order_block_short": "order_block_short",
    "fair_value_gap_long": "fair_value_gap_long",
    "fair_value_gap_short": "fair_value_gap_short",
    "liquidity_sweep_long": "liquidity_sweep_long",
    "liquidity_sweep_short": "liquidity_sweep_short",
    "premium_zone_short": "premium_zone_short",
    "discount_zone_long": "discount_zone_long",
    "break_of_structure_long": "break_of_structure_long",
    "break_of_structure_short": "break_of_structure_short",
    "change_of_character_long": "change_of_character_long",
    "change_of_character_short": "change_of_character_short",
    # Gap
    "gap_fill_long": "gap_fill_long",
    "gap_fill_short": "gap_fill_short",
    "gap_breakout_long": "gap_breakout_long",
    "gap_breakout_short": "gap_breakout_short",
    # ORB
    "orb_breakout_long": "orb_breakout_long",
    "orb_breakout_short": "orb_breakout_short",
    "orb_breakdown_short": "orb_breakdown_short",
    "orb_breakout": "orb_breakout",
    "orb_breakdown": "orb_breakdown",
    "orb_pullback_long": "orb_pullback_long",
    "orb_pullback_short": "orb_pullback_short",
    # VWAP
    "vwap_reclaim_long": "vwap_reclaim_long",
    "vwap_lose_short": "vwap_lose_short",
    "vwap_mean_reversion_long": "vwap_mean_reversion_long",
    "vwap_mean_reversion_short": "vwap_mean_reversion_short",
    # Trend
    "trend_continuation_long": "trend_continuation_long",
    "trend_continuation_short": "trend_continuation_short",
    "trend_pullback_long": "trend_pullback_long",
    "trend_pullback_short": "trend_pullback_short",
    "trend_reversal_long": "trend_reversal_long",
    "trend_reversal_short": "trend_reversal_short",
    # Volume
    "volume_spike_reversal_long": "volume_spike_reversal_long",
    "volume_spike_reversal_short": "volume_spike_reversal_short",
    "volume_breakout_long": "volume_breakout_long",
    "volume_breakout_short": "volume_breakout_short",
    # Range
    "range_bounce_long": "range_bounce_long",
    "range_bounce_short": "range_bounce_short",
    "range_breakdown_short": "range_breakdown_short",
    "range_breakout_long": "range_breakout_long",
    "range_rejection_long": "range_rejection_long",
    "range_rejection_short": "range_rejection_short",
    # Support / Resistance
    "support_bounce_long": "support_bounce_long",
    "resistance_bounce_short": "resistance_bounce_short",
    "support_breakdown_short": "support_breakdown_short",
    "resistance_breakout_long": "resistance_breakout_long",
    # FHM
    "first_hour_momentum_long": "first_hour_momentum_long",
    "first_hour_momentum_short": "first_hour_momentum_short",
    # Sub-7 / Indian native setups
    "gap_fade_short": "gap_fade_short",
    "cpr_mean_revert": "cpr_mean_revert",
    "circuit_t1_fade_short": "circuit_t1_fade_short",
    "delivery_pct_anomaly_short": "delivery_pct_anomaly_short",
    "long_panic_gap_down": "long_panic_gap_down",
    "or_window_failure_fade_short": "or_window_failure_fade_short",
}


def _get_setup_type_mapping(structure_type: str):
    """Map structure_type → setup_type. Falls through to structure_type itself
    for new detectors not yet in the table (safe: downstream ignores unknowns)."""
    return _STRUCTURE_TO_SETUP_TYPE.get(structure_type, structure_type)


def _events_to_gate_decision(sym: str, all_analyses: list, regime: str, regime_diagnostics):
    """Convert a list of (det_name, StructureAnalysis) pairs to a GateDecision.

    Mirrors MainDetector._convert_to_setup_candidates + TradeDecisionGate.evaluate
    so the existing downstream (screener_live ~line 1820) is UNCHANGED.
    """
    from services.gates.trade_decision_gate import GateDecision, SetupCandidate

    # Collect all StructureEvent objects across detectors
    all_events = []
    analysis_by_detector: dict = {}
    for det_name, analysis in all_analyses:
        if analysis is not None and analysis.structure_detected and analysis.events:
            all_events.extend(analysis.events)
            analysis_by_detector[det_name] = analysis

    if not all_events:
        return GateDecision(
            accept=False,
            reasons=["no_structure_event"],
            regime=regime,
            regime_diagnostics=regime_diagnostics,
        )

    # Convert StructureEvent → SetupCandidate (mirrors MainDetector._convert_to_setup_candidates)
    setup_candidates = []
    for event in all_events:
        try:
            detector_name = "unknown"
            reasons = []
            if hasattr(event, "context") and event.context:
                detector_name = event.context.get("detector_name", "unknown")
                reasons.append(f"detector:{detector_name}")
                if "level_name" in event.context:
                    reasons.append(f"level:{event.context['level_name']}")
                if "pattern_type" in event.context:
                    reasons.append(f"pattern:{event.context['pattern_type']}")

            setup_type = _get_setup_type_mapping(event.structure_type)
            if not setup_type:
                continue

            entry_mode = event.context.get("entry_mode") if event.context else None
            retest_zone = event.context.get("retest_zone") if event.context else None

            detected_level = None
            if event.levels:
                if event.side == "long":
                    detected_level = (
                        event.levels.get("support")
                        or event.levels.get("nearest_support")
                        or event.levels.get("broken_level")
                    )
                else:
                    detected_level = (
                        event.levels.get("resistance")
                        or event.levels.get("nearest_resistance")
                        or event.levels.get("broken_level")
                    )

            extras_dict = None
            if hasattr(event, "context") and event.context:
                extras_dict = {
                    k: v for k, v in event.context.items()
                    if isinstance(v, (str, int, float, bool, type(None)))
                } or None

            parent_analysis = analysis_by_detector.get(detector_name)
            parent_quality = float(parent_analysis.quality_score) if parent_analysis is not None else 0.0

            # ORH/ORL from levels dict (passed on the batch item)
            # (event.levels carries detector-specific levels like support/resistance, not ORH/ORL)
            orh = None  # will be filled from MarketContext if needed
            orl = None

            setup_candidates.append(SetupCandidate(
                setup_type=setup_type,
                strength=float(event.confidence),
                reasons=reasons,
                orh=orh,
                orl=orl,
                entry_mode=entry_mode,
                retest_zone=retest_zone,
                cap_segment=None,   # cap_segment is on MarketContext, not SetupCandidate in the new path
                detected_level=detected_level,
                extras=extras_dict,
                structure_event=event,
                quality_score=parent_quality,
            ))
        except Exception:
            pass

    if not setup_candidates:
        return GateDecision(
            accept=False,
            reasons=["no_setup_type_mapping"],
            regime=regime,
            regime_diagnostics=regime_diagnostics,
        )

    setup_candidates.sort(key=lambda s: s.strength, reverse=True)
    best = setup_candidates[0]
    reasons = [f"structure:{r}" for r in best.reasons] + [f"regime:{regime}"]
    strength = float(best.strength)

    return GateDecision(
        accept=True,
        reasons=reasons,
        setup_type=best.setup_type,
        regime=regime,
        regime_conf=0.5,
        size_mult=1.0,
        min_hold_bars=0,
        setup_candidates=setup_candidates,
        regime_diagnostics=regime_diagnostics,
        structure_confidence=strength,
    )


def dispatch_worker_batch(batch: Batch) -> list:
    """Process one batch: for each (sym, df5, levels, tags[, cap_segment]),
    run each tagged detector and return list[(sym, GateDecision)].

    The GateDecision format is identical to what _worker_process_batch
    returned, so screener_live's downstream pipeline is UNCHANGED.
    """
    out: list = []
    regime = getattr(batch, "regime", "chop")
    regime_diagnostics = getattr(batch, "regime_diagnostics", None)
    session_date = getattr(batch, "session_date", None)
    bar_ts = getattr(batch, "bar_ts", None)

    daily_dict = getattr(batch, "daily_dict", None) or {}

    for item in batch.items:
        # Support both old 4-tuple (sym, df5, levels, tags) and
        # new 5-tuple (sym, df5, levels, tags, cap_segment) for test compat.
        if len(item) == 5:
            sym, df5, levels, tags, cap_segment = item
        else:
            sym, df5, levels, tags = item
            cap_segment = "unknown"

        df_daily = daily_dict.get(sym)

        try:
            ctx = _build_market_context(
                sym, df5, levels,
                bar_ts=bar_ts,
                session_date=session_date,
                regime=regime,
                cap_segment=cap_segment,
                df_daily=df_daily,
            )
            all_analyses = []
            for det_name in tags:
                try:
                    det = _get_detector(det_name)
                    analysis = det.detect(ctx)
                    all_analyses.append((det_name, analysis))
                except Exception as e:
                    try:
                        from config.logging_config import get_agent_logger
                        get_agent_logger().exception(
                            "DISPATCH_DET_FAILED | sym=%s det=%s: %s", sym, det_name, e
                        )
                    except Exception:
                        pass
                    all_analyses.append((det_name, None))

            decision = _events_to_gate_decision(sym, all_analyses, regime, regime_diagnostics)
            out.append((sym, decision))
        except Exception as e:
            try:
                from config.logging_config import get_agent_logger
                get_agent_logger().exception("DISPATCH_SYM_FAILED | sym=%s: %s", sym, e)
            except Exception:
                pass
            # Still emit a reject decision so downstream symbol tracking is consistent
            try:
                from services.gates.trade_decision_gate import GateDecision
                out.append((sym, GateDecision(
                    accept=False,
                    reasons=[f"worker_error:{type(e).__name__}"],
                    regime=regime,
                )))
            except Exception:
                pass
    return out
