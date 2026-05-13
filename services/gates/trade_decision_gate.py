from __future__ import annotations
"""
trade_decision_gate.py
----------------------
Lean structure-detection entry point. Computes broad-market regime, calls
the structure detector, returns a GateDecision.

2026-05-13: stripped pre-detector legacy filters (momentum_consolidation,
range_compression, opening_bell whitelist, time_window_block, ORB priority
re-ranking, entry_validation, breakout_quality, news_spike, event_policy,
HCET, setup_sequencing). These predated the cell-mining methodology, were
not part of any sanity validation, and actively anti-selected production
setups (e.g. range_compression mathematically rejects every circuit_t1
candidate since T-1 was a 4.5%+ pump). Each setup's cell-locked filters
(allowed_regimes, allowed_caps, active_window, gap thresholds, ADV bands)
live in the setup config and are enforced inside the detector.

Public API
----------
class TradeDecisionGate:
    def __init__(self, *, structure_detector, regime_gate): ...
    def evaluate(self, *, symbol, now, df5m_tail, index_df5m, levels, ...) -> GateDecision: ...
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple, Protocol, Literal, Dict, Any

import pandas as pd

from structures.data_models import StructureEvent
from config.logging_config import get_agent_logger
from utils.perf_timer import perf

logger = get_agent_logger()

# Per-setup wide_open (2026-05-13): production-ready setups (gap_fade,
# circuit_t1, earnings_day) run with wide_open=false (full filters);
# research setups (delivery_pct) run with wide_open=true (capture mode).
# Falls back to top-level wide_open_mode when setup_type missing or unset.
def _get_wide_open_mode(setup_type: Optional[str] = None) -> bool:
    """Return wide_open flag for a setup_type (or top-level fallback)."""
    try:
        from services.config_loader import is_wide_open_for_setup
        return is_wide_open_for_setup(setup_type)
    except Exception:
        return False


def _get_setup_min_bars(setup_type: str) -> Optional[int]:
    """Return setup-specific min_bars_required from configuration.json.

    Each detector knows its own bar requirement (e.g. gap_fade_short fires
    at 09:30 with only 4 bars, min_bars_required=1; orb_15 needs 6;
    pdh_pdl_reject needs 30). Configuration keys some setups by full name
    (gap_fade_short) and others by base name (orb_15) — try both.
    """
    if not setup_type:
        return None
    try:
        from config.filters_setup import load_filters
        from config.setup_categories import get_base_setup_name
        cfg = load_filters() or {}
        setups_cfg = cfg.get("setups") or {}
        setup_cfg = setups_cfg.get(setup_type) or setups_cfg.get(get_base_setup_name(setup_type)) or {}
        v = setup_cfg.get("min_bars_required")
        return int(v) if v is not None else None
    except Exception:
        return None


SetupType = Literal[
    "breakout_long",
    "breakout_short",
    "vwap_reclaim_long",
    "vwap_lose_short",
    "squeeze_release_long",
    "squeeze_release_short",
    "failure_fade_long",
    "failure_fade_short",
    "gap_fill_long",
    "gap_fill_short",
    "flag_continuation_long",
    "flag_continuation_short",
    "support_bounce_long",
    "resistance_bounce_short",
    "orb_breakout_long",
    "orb_breakout_short",
    "orb_breakdown_long",
    "orb_breakdown_short",
    "orb_pullback_long",
    "orb_pullback_short",
    "vwap_mean_reversion_long",
    "vwap_mean_reversion_short",
    "volume_spike_reversal_long",
    "volume_spike_reversal_short",
    "trend_pullback_long",
    "trend_pullback_short",
    "range_rejection_long",
    "range_rejection_short",
    "momentum_breakout_long",
    "momentum_breakout_short",
    "trend_continuation_long",
    "trend_continuation_short",
    "order_block_long",
    "order_block_short",
    "fair_value_gap_long",
    "fair_value_gap_short",
    "liquidity_sweep_long",
    "liquidity_sweep_short",
    "premium_zone_short",
    "discount_zone_long",
    "equilibrium_breakout_long",
    "equilibrium_breakout_short",
    "break_of_structure_long",
    "break_of_structure_short",
    "change_of_character_long",
    "change_of_character_short",
    "range_deviation_long",
    "range_deviation_short",
    "range_mean_reversion_long",
    "range_mean_reversion_short",
    "first_hour_momentum_long",
    "first_hour_momentum_short",
]


@dataclass(frozen=True)
class SetupCandidate:
    setup_type: SetupType
    strength: float  # arbitrary score from detector (higher = better)
    reasons: List[str]
    orh: Optional[float] = None
    orl: Optional[float] = None
    entry_mode: Optional[str] = None
    retest_zone: Optional[List[float]] = None
    cap_segment: Optional[str] = None
    detected_level: Optional[float] = None
    extras: Optional[Dict[str, Any]] = None

    # Single source of truth for the structure detected by detect(). Carried
    # in-process from MainDetector through the orchestrator into plan_*_strategy
    # so downstream layers NEVER re-call detect().
    structure_event: Optional[StructureEvent] = None

    # StructureAnalysis.quality_score (0-100) for the detector that produced
    # this candidate. Threaded into bar_scheduler priority formula
    # (setup_cfg.priority * quality_score / 100).
    quality_score: float = 0.0


@dataclass(frozen=True)
class GateDecision:
    accept: bool
    reasons: List[str]
    setup_type: Optional[SetupType] = None
    regime: Optional[str] = None
    regime_conf: float = 0.0
    size_mult: float = 1.0
    min_hold_bars: int = 0
    matched_rule: Optional[str] = None
    p_breakout: Optional[float] = None
    setup_candidates: Optional[List[SetupCandidate]] = None
    regime_diagnostics: Optional[dict] = None
    structure_confidence: float = 0.0


# ----------------------------- Component Protocols -----------------------------

class StructureDetector(Protocol):  # pragma: no cover (interface only)
    def detect_setups(self, symbol: str, df5m_tail: pd.DataFrame,
                      levels: dict | None,
                      daily_df: pd.DataFrame | None = None,
                      regime: str | None = None) -> List[SetupCandidate]:
        ...


class RegimeGate(Protocol):  # pragma: no cover (interface only)
    def compute_regime(self, index_df5m: pd.DataFrame) -> Tuple[str, float]:
        ...


def _safe_float(x, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


# ---------------------------- TradeDecisionGate --------------------------------

class TradeDecisionGate:
    """Structure-detection entry point. Computes regime, calls detector, returns decision.

    The pre-detector filter stack was removed 2026-05-13 (see module docstring).
    Each setup's cell-validated filters live in setup config + detector code.
    """

    def __init__(
        self,
        *,
        structure_detector: StructureDetector,
        regime_gate: RegimeGate,
        # Kept for backward-compatible call sites; not used by lean evaluate.
        event_policy_gate: Optional[object] = None,
        news_spike_gate: Optional[object] = None,
        quality_filters: Optional[dict] = None,
    ) -> None:
        self.structure = structure_detector
        self.regime_gate = regime_gate

    def evaluate(
        self,
        *,
        symbol: str,
        now,
        df5m_tail: pd.DataFrame,
        index_df5m: pd.DataFrame,
        levels: Optional[dict],
        plan: Optional[dict] = None,
        daily_df: Optional[pd.DataFrame] = None,
    ) -> GateDecision:
        """Compute regime, run detector, return decision.

        Each setup's cell-locked filters (allowed_caps, active_window, etc.)
        are enforced inside the detector via MarketContext. This gate adds
        only: regime computation (needed by detectors that read ctx.regime)
        and per-setup min_bars_required (from config).
        """
        reasons: List[str] = []

        # ---------------- REGIME (computed BEFORE detect_setups) ---------------
        # Cell-locked detectors read ctx.regime inside detect() and reject when
        # the broad-market regime doesn't match their allowed_regimes.
        df_for_regime = index_df5m if index_df5m is not None and not index_df5m.empty else df5m_tail
        regime_diagnostics = None
        regime = "chop"
        regime_confidence = 0.5
        try:
            with perf("gate", "regime_compute", sym=symbol,
                      multi_tf=(hasattr(self.regime_gate, 'compute_regime_multi_tf') and daily_df is not None)):
                if hasattr(self.regime_gate, 'compute_regime_multi_tf') and daily_df is not None:
                    try:
                        regime, regime_confidence, regime_diagnostics = self.regime_gate.compute_regime_multi_tf(
                            df5=df_for_regime, daily_df=daily_df, symbol=symbol,
                        )
                    except Exception as e:
                        logger.warning(f"Multi-TF regime failed for {symbol}, falling back to 5m-only: {e}")
                        regime, regime_confidence = self.regime_gate.compute_regime(df_for_regime)
                else:
                    regime, regime_confidence = self.regime_gate.compute_regime(df_for_regime)
        except Exception as e:
            logger.warning(f"Regime compute failed for {symbol}: {e} — defaulting to chop")

        # ---------------- STRUCTURE DETECTION ----------------
        with perf("gate", "detect_setups", sym=symbol):
            setups = self.structure.detect_setups(
                symbol, df5m_tail, levels, daily_df=daily_df, regime=regime,
            )

        if not setups:
            return GateDecision(
                accept=False,
                reasons=["no_structure_event"],
                regime=regime,
                regime_conf=regime_confidence,
                regime_diagnostics=regime_diagnostics,
            )

        # ---------------- PICK BEST + MIN BARS ----------------
        setups.sort(key=lambda s: s.strength, reverse=True)
        best = setups[0]
        reasons.extend([f"structure:{r}" for r in best.reasons])
        reasons.append(f"regime:{regime}")

        setup_min_bars = _get_setup_min_bars(best.setup_type)
        if setup_min_bars is not None:
            bars_available = len(df5m_tail) if df5m_tail is not None else 0
            if bars_available < setup_min_bars:
                reasons.append(f"setup_min_bars_fail:{best.setup_type}={setup_min_bars}>{bars_available}")
                return GateDecision(
                    accept=False,
                    reasons=reasons,
                    setup_type=best.setup_type,
                    regime=regime,
                    regime_conf=regime_confidence,
                    regime_diagnostics=regime_diagnostics,
                )
            reasons.append(f"setup_min_bars:{best.setup_type}={setup_min_bars}")

        # ---------------- ACCEPT ----------------
        strength = _safe_float(best.strength, 0.0)
        return GateDecision(
            accept=True,
            reasons=reasons,
            setup_type=best.setup_type,
            regime=regime,
            regime_conf=regime_confidence,
            size_mult=1.0,
            min_hold_bars=0,
            setup_candidates=setups,
            regime_diagnostics=regime_diagnostics,
            structure_confidence=strength,
        )
