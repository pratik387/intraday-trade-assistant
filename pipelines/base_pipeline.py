# pipelines/base_pipeline.py
"""
Base Pipeline - Abstract interface for category-specific pipelines.

Each category (BREAKOUT, LEVEL, REVERSION, MOMENTUM) implements this interface
with specialized logic for screening, quality, ranking, gates, entries, and targets.

IMPORTANT: All configuration is loaded from JSON files in config/pipelines/.
NO DEFAULTS - all values must be explicitly defined in config files.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import numpy as np
import pandas as pd

from config.logging_config import get_agent_logger
from services.indicators.indicators import (
    calculate_atr as _calculate_atr_util,
    calculate_atr_series as _calculate_atr_series_util,
    volume_ratio as _volume_ratio_util
)

logger = get_agent_logger()


def calculate_structure_stop(
    entry_price: float,
    bias: str,
    atr: float,
    sl_atr_mult: float = 2.25
) -> float:
    """
    DEPRECATED: This function will be removed. SL should come from structure's calculate_risk_params().

    Calculate stop loss based on ATR multiplier. The multiplier should come from
    the structure-specific configuration, not be hardcoded here.

    Args:
        entry_price: Entry price for the trade
        bias: "long" or "short"
        atr: ATR value (already adjusted for cap segment if needed)
        sl_atr_mult: ATR multiplier for stop distance (structure-specific)

    NOTE: For immediate entry mode, if the structure's hard_sl is available,
    USE THAT INSTEAD of calling this function.
    """
    if bias == "long":
        return entry_price - (atr * sl_atr_mult)
    else:
        return entry_price + (atr * sl_atr_mult)


class ConfigurationError(Exception):
    """Raised when required configuration is missing."""
    pass


def _normalize_symbol(symbol: str) -> str:
    """Normalize symbol to base name (e.g., 'NSE:RELIANCE' -> 'RELIANCE', 'RELIANCE.NS' -> 'RELIANCE')."""
    # Remove exchange prefix (NSE:, BSE:)
    if ":" in symbol:
        symbol = symbol.split(":")[-1]
    # Remove suffix (.NS, .BO)
    if "." in symbol:
        symbol = symbol.split(".")[0]
    return symbol


# Cache for cap_segment lookups (loaded once per session)
_cap_segment_cache: Dict[str, str] = {}
_cap_segment_loaded: bool = False


def get_cap_segment(symbol: str) -> str:
    """
    Get market cap segment for a symbol from nse_all.json.

    Used for cap-aware sizing (Van Tharp evidence):
    - large_cap: 1.2x size (lower volatility)
    - mid_cap: 1.0x size (baseline)
    - small_cap: 0.6x size + wider stops (higher volatility)

    Transferred from planner_internal.py lines 1074-1087.
    """
    global _cap_segment_cache, _cap_segment_loaded

    # Load cache once per session
    if not _cap_segment_loaded:
        try:
            nse_file = Path(__file__).parent.parent / "nse_all.json"
            if nse_file.exists():
                with nse_file.open() as f:
                    data = json.load(f)
                # Build map with normalized symbol names (strip .NS suffix)
                _cap_segment_cache = {_normalize_symbol(item["symbol"]): item.get("cap_segment", "unknown") for item in data}
                _cap_segment_loaded = True
                logger.debug(f"CAP_SEGMENT: Loaded {len(_cap_segment_cache)} symbols from nse_all.json")
        except Exception as e:
            logger.debug(f"CAP_SIZING: Failed to load cap_segment cache: {e}")
            _cap_segment_loaded = True  # Don't retry on failure

    # Normalize input symbol (strip NSE: prefix)
    normalized = _normalize_symbol(symbol)
    return _cap_segment_cache.get(normalized, "unknown")


@dataclass
class ScreeningResult:
    """Result of screening phase."""
    passed: bool
    reasons: List[str] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityResult:
    """Result of quality calculation phase."""
    structural_rr: float
    quality_status: str  # "excellent", "good", "fair", "poor"
    metrics: Dict[str, Any] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)


@dataclass
class GateResult:
    """Result of gate validation phase."""
    passed: bool
    reasons: List[str] = field(default_factory=list)
    size_mult: float = 1.0
    min_hold_bars: int = 0


@dataclass
class RankingResult:
    """Result of ranking phase."""
    score: float
    components: Dict[str, float] = field(default_factory=dict)
    multipliers: Dict[str, float] = field(default_factory=dict)


@dataclass
class EntryResult:
    """Result of entry calculation phase."""
    entry_zone: Tuple[float, float]
    entry_ref_price: float
    entry_trigger: str
    entry_mode: str  # "immediate", "retest", "pending"


@dataclass
class TargetResult:
    """Result of target calculation phase."""
    targets: List[Dict[str, Any]]  # [{"name": "T1", "level": x, "rr": y}, ...]
    hard_sl: float
    risk_per_share: float
    trail_config: Optional[Dict] = None


_BASE_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def load_base_config() -> Dict[str, Any]:
    """
    Load base configuration (universal settings for all pipelines).

    Returns:
        Base configuration dict

    Raises:
        ConfigurationError: If config file is missing or invalid
    """
    global _BASE_CONFIG_CACHE
    if _BASE_CONFIG_CACHE is not None:
        return _BASE_CONFIG_CACHE

    config_dir = Path(__file__).parent.parent / "config" / "pipelines"
    config_file = config_dir / "base_config.json"

    if not config_file.exists():
        logger.warning(f"Base config not found: {config_file}, using empty base config")
        _BASE_CONFIG_CACHE = {}
        return _BASE_CONFIG_CACHE

    try:
        with open(config_file, 'r') as f:
            _BASE_CONFIG_CACHE = json.load(f)
        logger.debug(f"Loaded base pipeline config: {len(_BASE_CONFIG_CACHE)} keys")
        return _BASE_CONFIG_CACHE
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {config_file}: {e}")
        raise ConfigurationError(f"Invalid JSON in {config_file}: {e}")


def load_pipeline_config(category: str) -> Dict[str, Any]:
    """
    Load configuration for a pipeline category, merged with base config.

    Base config provides universal settings (volatility_sizing, cap_risk_adjustments, etc.)
    Category config provides category-specific settings (screening, gates, targets, etc.)
    Category config overrides base config for any overlapping keys.

    Args:
        category: One of BREAKOUT, LEVEL, REVERSION, MOMENTUM

    Returns:
        Merged configuration dict (base + category)

    Raises:
        ConfigurationError: If config file is missing or invalid
    """
    config_dir = Path(__file__).parent.parent / "config" / "pipelines"
    config_file = config_dir / f"{category.lower()}_config.json"

    if not config_file.exists():
        logger.error(f"Configuration file not found: {config_file}")
        raise ConfigurationError(f"Configuration file not found: {config_file}")

    try:
        with open(config_file, 'r') as f:
            category_config = json.load(f)
        logger.debug(f"Loaded {category} pipeline config: {len(category_config)} keys")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {config_file}: {e}")
        raise ConfigurationError(f"Invalid JSON in {config_file}: {e}")

    # Merge base config with category config (category overrides base)
    base_config = load_base_config()
    merged_config = {**base_config, **category_config}

    logger.debug(f"Merged config for {category}: base={len(base_config)} + category={len(category_config)} = {len(merged_config)} keys")
    return merged_config


def require_config(config: Dict[str, Any], *keys: str) -> Any:
    """
    Navigate nested config and raise error if key is missing.

    Args:
        config: Configuration dict
        keys: Path to required key (e.g., "screening", "time_windows", "morning_start")

    Returns:
        The value at the specified path

    Raises:
        ConfigurationError: If any key in the path is missing
    """
    current = config
    path = []
    for key in keys:
        path.append(key)
        if not isinstance(current, dict) or key not in current:
            raise ConfigurationError(f"Required configuration missing: {'.'.join(path)}")
        current = current[key]
    return current


class BasePipeline(ABC):
    """
    Abstract base class for category-specific pipelines.

    Subclasses must implement all abstract methods with category-specific logic.
    All configuration is loaded from JSON files - NO DEFAULTS.
    """

    def __init__(self):
        """
        Initialize pipeline by loading configuration.

        Raises:
            ConfigurationError: If config file is missing or invalid
        """
        category = self.get_category_name()
        self.cfg = load_pipeline_config(category)
        self._validate_config()

    def _validate_config(self):
        """Validate that required config sections exist."""
        required_sections = ["screening", "quality", "gates", "ranking", "entry", "targets"]
        for section in required_sections:
            if section not in self.cfg:
                raise ConfigurationError(f"Required config section missing: {section}")

    def _get(self, *keys: str) -> Any:
        """
        Get config value. Raises error if missing (no defaults).

        Args:
            keys: Path to config value

        Returns:
            The config value

        Raises:
            ConfigurationError: If value is missing
        """
        return require_config(self.cfg, *keys)

    # ======================== ABSTRACT METHODS ========================

    @abstractmethod
    def get_category_name(self) -> str:
        """Return category name (BREAKOUT, LEVEL, REVERSION, MOMENTUM)."""
        pass

    @abstractmethod
    def get_setup_types(self) -> List[str]:
        """Return list of setup types that belong to this category."""
        pass

    @abstractmethod
    def screen(
        self,
        symbol: str,
        df5m: pd.DataFrame,
        features: Dict[str, Any],
        levels: Dict[str, float],
        now: pd.Timestamp
    ) -> ScreeningResult:
        """Apply category-specific screening filters."""
        pass

    @abstractmethod
    def calculate_quality(
        self,
        symbol: str,
        df5m: pd.DataFrame,
        bias: str,
        levels: Dict[str, float],
        atr: float
    ) -> QualityResult:
        """Calculate category-specific quality metrics."""
        pass

    @abstractmethod
    def validate_gates(
        self,
        symbol: str,
        setup_type: str,
        regime: str,
        df5m: pd.DataFrame,
        df1m: Optional[pd.DataFrame],
        strength: float,
        adx: float,
        vol_mult: float,
        regime_diagnostics: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """Apply category-specific gate validations."""
        pass

    @abstractmethod
    def calculate_rank_score(
        self,
        symbol: str,
        intraday_features: Dict[str, Any],
        regime: str,
        daily_trend: Optional[str] = None,
        htf_context: Optional[Dict] = None
    ) -> RankingResult:
        """Calculate category-specific ranking score."""
        pass

    @abstractmethod
    def calculate_entry(
        self,
        symbol: str,
        df5m: pd.DataFrame,
        bias: str,
        levels: Dict[str, float],
        atr: float,
        setup_type: str
    ) -> EntryResult:
        """Calculate category-specific entry zone and trigger."""
        pass

    @abstractmethod
    def calculate_targets(
        self,
        symbol: str,
        entry_ref_price: float,
        hard_sl: float,
        bias: str,
        atr: float,
        levels: Dict[str, float],
        measured_move: float,
        setup_type: str = ""
    ) -> TargetResult:
        """Calculate category-specific targets and stop loss."""
        pass

    # ======================== HOOK METHODS (override in subclasses) ========================

    def _apply_rsi_penalty(self, rsi_val: float, bias: str) -> tuple:
        """
        Apply category-specific RSI penalty to position sizing.

        Override in subclass to implement category-specific logic.
        Pro Trader RSI Framework (Minervini, Connors, Raschke):
        - BREAKOUT/MOMENTUM: High RSI = momentum confirmation (GOOD)
        - REVERSION: Extreme RSI = setup condition (GOOD)
        - LEVEL: Neutral RSI = ideal, extreme = caution

        Args:
            rsi_val: Current RSI value (0-100)
            bias: Trade direction ("long" or "short")

        Returns:
            Tuple of (multiplier, caution_string or None)
            multiplier: 1.0 = no penalty, 0.9 = 10% reduction, etc.
        """
        return (1.0, None)

    # ======================== COMMON UTILITY METHODS ========================

    def parse_time(self, time_str: str) -> int:
        """Convert HH:MM string to minutes from midnight."""
        parts = time_str.split(':')
        return int(parts[0]) * 60 + int(parts[1])

    def get_time_bucket(self, now: pd.Timestamp) -> str:
        """Get time bucket for time-based filters."""
        md = now.hour * 60 + now.minute
        if md <= 10 * 60 + 30:
            return "early"
        elif md <= 13 * 60 + 30:
            return "mid"
        return "late"

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> Optional[float]:
        """
        Calculate ATR from DataFrame using centralized indicator utility.

        Returns:
            ATR value or None if insufficient data
        """
        if df is None or len(df) < period:
            logger.debug(f"ATR: Insufficient data ({len(df) if df is not None else 0} bars < {period} required)")
            return None

        try:
            result = _calculate_atr_util(df, period=period)
            if pd.isna(result):
                logger.debug("ATR: Calculation resulted in NaN")
                return None
            return float(result)
        except Exception as e:
            logger.debug(f"ATR: Calculation failed: {e}")
            return None

    def get_volume_ratio(self, df: pd.DataFrame, lookback: int = 20) -> float:
        """
        Calculate volume ratio vs recent average using centralized utility.

        Returns:
            Volume ratio (defaults to 1.0 if insufficient data)
        """
        # Use centralized utility from services/indicators/indicators.py
        # It already handles None/empty df and returns 1.0 as default
        return _volume_ratio_util(df, lookback=lookback)

    def _get_strategy_regime_mult(self, setup_type: str, regime: str) -> float:
        """
        Get regime multiplier - strategy-specific if available, else fallback.

        Ported from baseline ranker.py _get_regime_multiplier() function.
        This allows each strategy type to have tuned regime multipliers that
        were calibrated in the baseline configuration.

        Args:
            setup_type: Setup type (e.g., "breakout_long", "vwap_mean_reversion_short")
            regime: Current regime ("trend_up", "trend_down", "chop", "squeeze")

        Returns:
            Regime multiplier for ranking score adjustment
        """
        # Get strategy-specific regime multipliers from config
        strategy_mults = self._get("ranking", "strategy_regime_multipliers")

        # Extract base strategy name (e.g., "breakout" from "breakout_long")
        base_strategy = setup_type.replace("_long", "").replace("_short", "") if setup_type else ""

        if base_strategy in strategy_mults:
            strat_cfg = strategy_mults[base_strategy]
            return strat_cfg.get(regime, strat_cfg["default"])

        # Fallback to generic regime multipliers
        regime_mults = self._get("ranking", "regime_multipliers")
        return regime_mults[regime]

    # ======================== UNIVERSAL RANKING ADJUSTMENTS ========================
    # These adjustments apply to ALL categories (from ranker.py analysis)

    def apply_universal_ranking_adjustments(
        self,
        base_score: float,
        symbol: str,
        strategy_type: str,
        structural_rr: float,
        bias: str,
        current_time: pd.Timestamp,
        regime_diagnostics: Optional[Dict[str, Any]] = None,
        daily_score: float = 0.0,
        htf_context: Optional[Dict[str, Any]] = None,
        daily_trend: Optional[str] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Apply universal ranking adjustments that affect ALL categories.

        From ranker.py analysis - these are ported from OLD baseline:
        1. Time of day multiplier (1.5x/2.5x late-day threshold)
        2. Blacklist penalty (-999 for poor-performing setups)
        3. Unrealistic RR penalty (-0.3 when RR > max)
        4. Multi-TF daily multiplier (±15% at confidence ≥ 0.70)
        5. Multi-TF hourly multiplier (±10% at confidence ≥ 0.60)
        6. HTF 15m multiplier (±12% trend align, +8% volume)
        7. Daily trend multiplier (±25% for trend alignment)
        8. Daily score weighting (w_daily * daily_score + w_intr * intraday_score)

        Args:
            base_score: Score calculated by category-specific ranking
            symbol: Stock symbol
            strategy_type: Setup type (e.g., "orb_breakout_long")
            structural_rr: Calculated structural R:R
            bias: "long" or "short"
            current_time: Current timestamp
            regime_diagnostics: Multi-TF regime info {daily: {regime, confidence}, hourly: {session_bias, confidence}}
            daily_score: Daily timeframe score (optional)
            htf_context: HTF 15m context {trend_aligned: bool, volume_mult_15m: float}
            daily_trend: Daily trend direction ("up", "down", "neutral")

        Returns:
            Tuple of (adjusted_score, adjustment_details)
        """
        score = base_score
        adjustments = {}

        # Get universal ranking config from base config
        base_cfg = load_base_config()
        ranking_cfg = base_cfg["universal_ranking"]

        # 1. TIME OF DAY MULTIPLIER
        # Late-day signals have poor quality (39 signals in 14:00-15:00, only 3 trades = 7.7% conversion)
        time_mult = self._get_time_of_day_multiplier(current_time, ranking_cfg)
        adjustments["time_of_day_mult"] = time_mult
        # Note: We apply this as a threshold multiplier, not a score multiplier
        # The actual filtering happens in the orchestrator when comparing to threshold

        # 2. BLACKLIST PENALTY
        blacklist_penalty = self._get_blacklist_penalty(strategy_type, ranking_cfg)
        score += blacklist_penalty
        adjustments["blacklist_penalty"] = blacklist_penalty

        # 3. UNREALISTIC RR PENALTY
        rr_penalty = self._get_unrealistic_rr_penalty(structural_rr, ranking_cfg)
        score += rr_penalty
        adjustments["rr_penalty"] = rr_penalty

        # 4. MULTI-TF DAILY MULTIPLIER (±15% at confidence ≥ 0.70)
        daily_mult = self._get_multi_tf_daily_multiplier(regime_diagnostics, bias, ranking_cfg)
        score *= daily_mult
        adjustments["multi_tf_daily_mult"] = daily_mult

        # 5. MULTI-TF HOURLY MULTIPLIER (±10% at confidence ≥ 0.60)
        hourly_mult = self._get_multi_tf_hourly_multiplier(regime_diagnostics, bias, ranking_cfg)
        score *= hourly_mult
        adjustments["multi_tf_hourly_mult"] = hourly_mult

        # 6. HTF 15m MULTIPLIER (±12% trend align, +8% volume)
        # From ranker.py _get_htf_15m_multiplier() lines 305-347
        htf_15m_mult = self._get_htf_15m_multiplier(htf_context, bias, ranking_cfg)
        score *= htf_15m_mult
        adjustments["htf_15m_mult"] = htf_15m_mult

        # 7. DAILY TREND MULTIPLIER (±25% for trend alignment)
        # From ranker.py _get_daily_trend_multiplier() lines 269-302
        daily_trend_mult = self._get_daily_trend_multiplier(daily_trend, bias, ranking_cfg)
        score *= daily_trend_mult
        adjustments["daily_trend_mult"] = daily_trend_mult

        # 8. DAILY SCORE WEIGHTING
        if daily_score != 0.0:
            score = self._apply_daily_score_weighting(score, daily_score, ranking_cfg)
            adjustments["daily_score_applied"] = True
        else:
            adjustments["daily_score_applied"] = False

        return score, adjustments

    def _get_time_of_day_multiplier(
        self,
        current_time: pd.Timestamp,
        ranking_cfg: Dict[str, Any]
    ) -> float:
        """
        Dynamic rank threshold multiplier based on time of day.

        From ranker.py lines 569-622:
        - Morning (10:15-12:00): Normal threshold (1.0x)
        - Midday (12:00-14:00): Normal threshold (1.0x)
        - Late afternoon (14:00-14:30): Moderate filter (1.5x)
        - Final hour (14:30-15:15): Aggressive filter (2.5x)

        This multiplier is applied to the threshold, not the score.
        Higher multiplier = harder to qualify.
        """
        time_cfg = ranking_cfg["time_of_day"]
        if not time_cfg["enabled"]:
            return 1.0

        hour = current_time.hour
        minute = current_time.minute

        # Get thresholds from config
        late_afternoon_start = time_cfg["late_afternoon_start_hour"]
        late_afternoon_end_min = time_cfg["late_afternoon_end_minute"]
        late_afternoon_mult = time_cfg["late_afternoon_mult"]
        final_hour_mult = time_cfg["final_hour_mult"]

        # Morning and midday: normal threshold
        if hour < late_afternoon_start:
            return 1.0

        # Late afternoon (14:00-14:30): moderate filter
        if hour == late_afternoon_start and minute < late_afternoon_end_min:
            return late_afternoon_mult

        # Final hour (14:30+): aggressive filter
        return final_hour_mult

    def _get_blacklist_penalty(
        self,
        strategy_type: str,
        ranking_cfg: Dict[str, Any]
    ) -> float:
        """
        Apply penalty for historically poor-performing setups.

        From ranker.py lines 513-517:
        - Blacklisted strategies get a severe penalty (-999 by default)
        - This effectively prevents them from being selected
        """
        blacklist_cfg = ranking_cfg["blacklist"]
        if not blacklist_cfg["enabled"]:
            return 0.0

        blacklisted_setups = blacklist_cfg["strategies"]
        penalty = blacklist_cfg["penalty"]

        if strategy_type in blacklisted_setups:
            logger.debug(f"BLACKLIST_PENALTY: {strategy_type} penalty={penalty}")
            return penalty

        return 0.0

    def _get_unrealistic_rr_penalty(
        self,
        structural_rr: float,
        ranking_cfg: Dict[str, Any]
    ) -> float:
        """
        Apply penalty for unrealistically high R:R ratios.

        From ranker.py lines 519-525:
        - If structural_rr > max_rr, apply penalty (-0.3 by default)
        - Unrealistic R:R often indicates bad level placement or data issues
        """
        rr_cfg = ranking_cfg["unrealistic_rr"]
        if not rr_cfg["enabled"]:
            return 0.0

        max_rr = rr_cfg["max_structural_rr"]
        penalty = rr_cfg["penalty"]

        if structural_rr > max_rr:
            logger.debug(f"HIGH_RR_PENALTY: structural_rr={structural_rr:.1f}>{max_rr} penalty={penalty}")
            return penalty

        return 0.0

    def _get_multi_tf_daily_multiplier(
        self,
        regime_diagnostics: Optional[Dict[str, Any]],
        bias: str,
        ranking_cfg: Dict[str, Any]
    ) -> float:
        """
        Apply daily regime multiplier based on multi-timeframe analysis.

        From ranker.py lines 350-399 (Linda Raschke MTF filtering):
        - ±15% adjustment based on daily regime alignment
        - Only apply if confidence ≥ 0.70
        - Long boost in daily uptrend, short boost in daily downtrend
        """
        mtf_cfg = ranking_cfg["multi_tf_daily"]
        if not mtf_cfg["enabled"]:
            return 1.0

        if not regime_diagnostics or "daily" not in regime_diagnostics:
            return 1.0

        daily = regime_diagnostics["daily"]
        daily_regime = daily.get("regime", "chop")
        daily_confidence = daily.get("confidence", 0.0)

        # Confidence threshold from config
        min_confidence = mtf_cfg["min_confidence"]
        if daily_confidence < min_confidence:
            return 1.0

        aligned_mult = mtf_cfg["aligned_mult"]
        counter_mult = mtf_cfg["counter_mult"]
        squeeze_mult = mtf_cfg["squeeze_mult"]

        is_long = bias == "long"
        is_short = bias == "short"

        # Daily trend_up: Boost longs, penalize shorts
        if daily_regime == "trend_up":
            if is_long:
                return aligned_mult  # +15% boost
            elif is_short:
                return counter_mult  # -15% penalty

        # Daily trend_down: Boost shorts, penalize longs
        elif daily_regime == "trend_down":
            if is_short:
                return aligned_mult  # +15% boost
            elif is_long:
                return counter_mult  # -15% penalty

        # Daily squeeze: Mild penalty for all
        elif daily_regime == "squeeze":
            return squeeze_mult  # -10% penalty

        return 1.0

    def _get_multi_tf_hourly_multiplier(
        self,
        regime_diagnostics: Optional[Dict[str, Any]],
        bias: str,
        ranking_cfg: Dict[str, Any]
    ) -> float:
        """
        Apply hourly session bias multiplier based on multi-timeframe analysis.

        From ranker.py lines 402-447 (Linda Raschke lower TF execution):
        - ±10% adjustment based on hourly session bias
        - Only apply if confidence ≥ 0.60
        - Smaller than daily (it's a lower TF, noisier)
        """
        mtf_cfg = ranking_cfg["multi_tf_hourly"]
        if not mtf_cfg["enabled"]:
            return 1.0

        if not regime_diagnostics or "hourly" not in regime_diagnostics:
            return 1.0

        hourly = regime_diagnostics["hourly"]
        session_bias = hourly.get("session_bias", "neutral")
        hourly_confidence = hourly.get("confidence", 0.0)

        # Confidence threshold from config (lower than daily)
        min_confidence = mtf_cfg["min_confidence"]
        if hourly_confidence < min_confidence:
            return 1.0

        aligned_mult = mtf_cfg["aligned_mult"]
        counter_mult = mtf_cfg["counter_mult"]

        is_long = bias == "long"
        is_short = bias == "short"

        # Hourly bullish: Boost longs, penalize shorts
        if session_bias == "bullish":
            if is_long:
                return aligned_mult  # +10% boost
            elif is_short:
                return counter_mult  # -10% penalty

        # Hourly bearish: Boost shorts, penalize longs
        elif session_bias == "bearish":
            if is_short:
                return aligned_mult  # +10% boost
            elif is_long:
                return counter_mult  # -10% penalty

        return 1.0

    def _get_htf_15m_multiplier(
        self,
        htf_context: Optional[Dict[str, Any]],
        bias: str,
        ranking_cfg: Dict[str, Any]
    ) -> float:
        """
        Apply 15m HTF (Higher TimeFrame) confirmation multiplier.

        From ranker.py lines 305-347 (Intraday Scanner Playbook):
        - Trend align bonus: +12% (5m + 15m same direction)
        - Volume multiplier bonus: +8% (15m volume > 1.3x median)
        - Opposing trend penalty: -10% (5m vs 15m divergence)

        Never blocks entries - only affects ranking scores.

        Args:
            htf_context: Dict with htf_trend, volume_mult_15m, trend_aligned etc.
            bias: "long" or "short"
            ranking_cfg: Universal ranking config section

        Returns:
            Multiplier: 1.0 (neutral) to ~1.21 (aligned + volume) or 0.90 (opposing)
        """
        htf_cfg = ranking_cfg.get("htf_15m")
        if htf_cfg is None or not htf_cfg.get("enabled", True):
            return 1.0

        if not htf_context:
            return 1.0  # No HTF data available, neutral

        is_long = bias == "long"
        is_short = bias == "short"
        multiplier = 1.0

        # Get alignment/penalty values from config
        aligned_bonus = htf_cfg["aligned_bonus"]
        opposing_penalty = htf_cfg["opposing_penalty"]
        volume_bonus = htf_cfg["volume_bonus"]
        volume_threshold = htf_cfg["volume_threshold"]

        # Check 15m trend alignment (screener populates "trend_aligned" as boolean)
        htf_trend_aligned = htf_context.get("trend_aligned", False)

        # Apply alignment bonus or penalty
        if is_long:
            if htf_trend_aligned:  # 15m uptrend aligns with long setup
                multiplier *= aligned_bonus  # +12% bonus
            else:  # 15m downtrend opposes long setup
                multiplier *= opposing_penalty  # -10% penalty
        elif is_short:
            if not htf_trend_aligned:  # 15m downtrend aligns with short setup
                multiplier *= aligned_bonus  # +12% bonus
            else:  # 15m uptrend opposes short setup
                multiplier *= opposing_penalty  # -10% penalty

        # Check 15m volume context
        htf_vol_mult = htf_context.get("volume_mult_15m", 1.0)
        if htf_vol_mult >= volume_threshold:  # 15m volume surge
            multiplier *= volume_bonus  # +8% bonus

        return multiplier

    def _get_daily_trend_multiplier(
        self,
        daily_trend: Optional[str],
        bias: str,
        ranking_cfg: Dict[str, Any]
    ) -> float:
        """
        Apply daily trend alignment multiplier for multi-timeframe confluence.

        From ranker.py lines 269-302:
        - Long setups in daily uptrend: 25% win rate boost
        - Short setups in daily downtrend: 25% win rate boost
        - Counter-trend trades: 25% penalty

        Args:
            daily_trend: "up", "down", or "neutral"
            bias: "long" or "short"
            ranking_cfg: Universal ranking config section

        Returns:
            Multiplier: 1.25 (aligned), 0.75 (counter), 1.0 (neutral)
        """
        daily_cfg = ranking_cfg.get("daily_trend")
        if daily_cfg is None or not daily_cfg.get("enabled", True):
            return 1.0

        if not daily_trend or daily_trend == "neutral":
            return 1.0

        # Get multiplier values from config
        aligned_mult = daily_cfg["aligned_mult"]
        counter_mult = daily_cfg["counter_mult"]

        is_long = bias == "long"
        is_short = bias == "short"

        # Apply trend alignment bonus/penalty
        if daily_trend == "up" and is_long:
            return aligned_mult  # 25% boost for trend-aligned longs
        elif daily_trend == "down" and is_short:
            return aligned_mult  # 25% boost for trend-aligned shorts
        elif daily_trend == "up" and is_short:
            return counter_mult  # 25% penalty for counter-trend shorts
        elif daily_trend == "down" and is_long:
            return counter_mult  # 25% penalty for counter-trend longs

        return 1.0  # Neutral

    def _apply_daily_score_weighting(
        self,
        intraday_score: float,
        daily_score: float,
        ranking_cfg: Dict[str, Any]
    ) -> float:
        """
        Combine daily and intraday scores with configurable weights.

        From ranker.py line 505:
        final_score = w_daily * daily_score + w_intr * intraday_score

        Default: 30% daily, 70% intraday
        """
        weight_cfg = ranking_cfg["daily_score_weighting"]
        if not weight_cfg["enabled"]:
            return intraday_score

        w_daily = weight_cfg["weight_daily"]
        w_intraday = weight_cfg["weight_intraday"]

        return w_daily * daily_score + w_intraday * intraday_score

    # ======================== UNIVERSAL GATES ========================
    # These gates apply BEFORE category-specific gates

    def _check_range_compression(
        self,
        df5m: pd.DataFrame,
        atr: float
    ) -> Tuple[bool, str]:
        """
        Range compression filter - reject during volatility expansion.

        From trade_decision_gate.py analysis:
        - Computes current_atr / avg_atr_20
        - Ratio > 0.8 means volatility is expanding → bad for level-based setups
        - Volatility expansion invalidates support/resistance levels

        Returns:
            Tuple of (passed: bool, reason: str)
        """
        compression_cfg = self.cfg["range_compression"]
        if not compression_cfg["enabled"]:
            return True, ""

        try:
            # Use centralized ATR series calculation
            lookback = compression_cfg["lookback_bars"]
            atr_series = _calculate_atr_series_util(df5m, period=14)

            if len(atr_series) < lookback:
                return True, ""  # Not enough data, allow through

            avg_atr = atr_series.tail(lookback).mean()

            if pd.isna(avg_atr) or avg_atr <= 0:
                return True, ""  # Can't calculate, allow through

            # Current ATR vs average
            compression_ratio = atr / avg_atr

            max_ratio = compression_cfg["max_expansion_ratio"]
            if compression_ratio > max_ratio:
                reason = f"range_compression_fail: ratio={compression_ratio:.2f}>{max_ratio}"
                logger.debug(f"RANGE_COMPRESSION: {reason}")
                return False, reason

            return True, ""

        except Exception as e:
            logger.debug(f"RANGE_COMPRESSION: Error calculating: {e}")
            return True, ""  # Allow through on error

    def _check_cap_strategy_blocking(
        self,
        symbol: str,
        setup_type: str,
        cap_segment: str
    ) -> Tuple[bool, str]:
        """
        Cap-strategy blocking - block certain setups for certain market cap segments.

        From trade_decision_gate.py analysis:
        - Micro-cap stocks shouldn't use momentum_breakout (too volatile)
        - Large-cap stocks prefer level-based setups over reversions

        Returns:
            Tuple of (passed: bool, reason: str)
        """
        blocking_cfg = self.cfg["cap_strategy_blocking"]
        if not blocking_cfg["enabled"]:
            return True, ""

        if cap_segment == "unknown":
            return True, ""  # Can't determine cap, allow through

        # Check if this setup type is blocked for this cap segment
        blocked_setups = blocking_cfg["blocked_setups"]
        blocked_for_cap = blocked_setups.get(cap_segment, [])

        # Check for exact match
        if setup_type in blocked_for_cap:
            reason = f"cap_strategy_blocked: {setup_type} blocked for {cap_segment}"
            logger.debug(f"CAP_STRATEGY_BLOCKING: {symbol} {reason}")
            return False, reason

        # Check for base setup name match (e.g., "momentum_breakout" blocks both _long and _short)
        base_setup = setup_type.rsplit("_", 1)[0] if "_long" in setup_type or "_short" in setup_type else setup_type
        if base_setup in blocked_for_cap:
            reason = f"cap_strategy_blocked: {base_setup} blocked for {cap_segment}"
            logger.debug(f"CAP_STRATEGY_BLOCKING: {symbol} {reason}")
            return False, reason

        return True, ""

    def _is_opening_bell_window(self, now: pd.Timestamp) -> bool:
        """
        Check if we're in the opening bell window (9:15-9:30 AM).

        From trade_decision_gate.py lines 443-463:
        - Opening bell has different volatility/participation patterns
        - Some gates can be bypassed during this window
        """
        ob_cfg = self.cfg["opening_bell_override"]
        if not ob_cfg["enabled"]:
            return False

        try:
            start_hour = ob_cfg["start_hour"]
            start_minute = ob_cfg["start_minute"]
            end_hour = ob_cfg["end_hour"]
            end_minute = ob_cfg["end_minute"]

            current_minutes = now.hour * 60 + now.minute
            start_minutes = start_hour * 60 + start_minute
            end_minutes = end_hour * 60 + end_minute

            return start_minutes <= current_minutes <= end_minutes
        except Exception:
            return False

    def _opening_bell_bypasses(self, gate_name: str) -> bool:
        """
        Check if a specific gate can be bypassed during opening bell.

        From trade_decision_gate.py lines 488-573:
        - Range compression can be bypassed (high volatility expected at open)
        - Momentum consolidation can be bypassed (different patterns at open)
        """
        ob_cfg = self.cfg["opening_bell_override"]
        if not ob_cfg["enabled"]:
            return False

        bypasses = ob_cfg["bypass_gates"]
        return gate_name in bypasses

    def _check_price_action_directionality(
        self,
        df5m: pd.DataFrame,
        bias: str
    ) -> Tuple[bool, str]:
        """
        Price action directionality - validates last 3 bars follow expected direction.

        From trade_decision_gate.py lines 1041-1056:
        - Long: Last 3 bars should NOT be declining (close[-1] >= close[-3])
        - Short: Last 3 bars should NOT be rising (close[-1] <= close[-3])
        - Prevents counter-trend entry fills

        Returns:
            Tuple of (passed: bool, reason: str)
        """
        pa_cfg = self.cfg["price_action_directionality"]
        if not pa_cfg["enabled"]:
            return True, ""

        try:
            if len(df5m) < 4:
                return True, ""  # Not enough data, allow through

            close_now = float(df5m["close"].iloc[-1])
            close_3_ago = float(df5m["close"].iloc[-3])

            if bias == "long":
                # For long: last 3 bars should NOT be declining
                if close_now < close_3_ago:
                    reason = f"price_action_fail: long but declining {close_now:.2f}<{close_3_ago:.2f}"
                    logger.debug(f"PRICE_ACTION_DIRECTIONALITY: {reason}")
                    return False, reason
            else:
                # For short: last 3 bars should NOT be rising
                if close_now > close_3_ago:
                    reason = f"price_action_fail: short but rising {close_now:.2f}>{close_3_ago:.2f}"
                    logger.debug(f"PRICE_ACTION_DIRECTIONALITY: {reason}")
                    return False, reason

            return True, ""

        except Exception as e:
            logger.debug(f"PRICE_ACTION_DIRECTIONALITY: Error calculating: {e}")
            return True, ""  # Allow through on error

    # ======================== PIPELINE EXECUTION ========================

    def run_pipeline(
        self,
        symbol: str,
        setup_type: str,
        df5m: pd.DataFrame,
        df1m: Optional[pd.DataFrame],
        levels: Dict[str, float],
        regime: str,
        now: pd.Timestamp,
        daily_df: Optional[pd.DataFrame] = None,
        htf_context: Optional[Dict[str, Any]] = None,
        regime_diagnostics: Optional[Dict[str, Any]] = None,
        daily_score: float = 0.0,
        cap_segment: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Run complete pipeline for a setup.

        Args:
            htf_context: HTF (15m) context for category-specific ranking adjustments.
                         Contains: htf_trend, htf_volume_surge, htf_momentum, htf_exhaustion
            regime_diagnostics: Multi-TF regime info for universal ranking adjustments.
                         Contains: {daily: {regime, confidence}, hourly: {session_bias, confidence}}
            daily_score: Daily timeframe score for score weighting (optional)

        Returns:
            Complete plan dict or None if rejected
        """
        category = self.get_category_name()
        logger.debug(f"[{category}] Running pipeline for {symbol} {setup_type} in {regime}")

        bias = "long" if "_long" in setup_type else "short"
        atr = self.calculate_atr(df5m)

        # For ORB setups in morning session, use daily ATR if intraday ATR unavailable
        # Professional traders use daily ATR for morning setups since intraday data is insufficient
        is_orb_setup = setup_type.startswith("orb_")
        is_morning_session = now.hour < 11 if now else False  # Before 11:00 AM

        if atr is None and is_orb_setup and is_morning_session and daily_df is not None:
            daily_atr = self.calculate_atr(daily_df, period=14)
            if daily_atr is not None:
                atr = daily_atr
                logger.info(f"[{category}] {symbol} {setup_type}: Using daily ATR {atr:.2f} for morning ORB (intraday ATR unavailable)")

        current_close = float(df5m["close"].iloc[-1])

        # ATR FALLBACK: Use config value when ATR unavailable
        if atr is None:
            atr_fallback_pct = self.cfg.get("atr_fallback_pct")
            atr = current_close * atr_fallback_pct
            logger.debug(f"[{category}] {symbol} {setup_type}: Using fallback ATR {atr:.2f} ({atr_fallback_pct*100}% of price)")

        # Get indicator values
        adx_val = float(df5m["adx"].iloc[-1]) if "adx" in df5m.columns and not pd.isna(df5m["adx"].iloc[-1]) else None
        vwap_val = float(df5m["vwap"].iloc[-1]) if "vwap" in df5m.columns and not pd.isna(df5m["vwap"].iloc[-1]) else None
        rsi_val = float(df5m["rsi"].iloc[-1]) if "rsi" in df5m.columns and not pd.isna(df5m["rsi"].iloc[-1]) else None

        # Volume ratio - utility already defaults to 1.0 if insufficient data
        volume_ratio = self.get_volume_ratio(df5m)

        # Calculate slopes from last few bars if available
        rsi_slope = 0.0
        adx_slope = 0.0
        if len(df5m) >= 5:
            # RSI slope: difference over last 5 bars
            if "rsi" in df5m.columns:
                rsi_series = df5m["rsi"].dropna().tail(5)
                if len(rsi_series) >= 2:
                    rsi_slope = float(rsi_series.iloc[-1] - rsi_series.iloc[0]) / len(rsi_series)
            # ADX slope: difference over last 5 bars
            if "adx" in df5m.columns:
                adx_series = df5m["adx"].dropna().tail(5)
                if len(adx_series) >= 2:
                    adx_slope = float(adx_series.iloc[-1] - adx_series.iloc[0]) / len(adx_series)

        features = {
            "volume_ratio": volume_ratio,
            "adx": adx_val,
            "above_vwap": current_close >= vwap_val if vwap_val else None,
            "current_close": current_close,
            "bias": bias,
            "rsi": rsi_val,
            "rsi_slope": rsi_slope,
            "adx_slope": adx_slope,
            "setup_type": setup_type,  # For strategy-specific regime multipliers
            "atr": atr,  # ATR with fallback already applied - use this instead of recalculating
        }

        # Add avg_daily_volume from daily_df for FHM true RVOL calculation
        # EXPERIMENTAL: Used for First Hour Momentum RVOL calculation.
        # May need to revert if quality degrades - see _check_fhm_conditions.
        if daily_df is not None and len(daily_df) >= 20 and "volume" in daily_df.columns:
            features["avg_daily_volume"] = float(daily_df["volume"].tail(20).mean())
        else:
            features["avg_daily_volume"] = None

        # daily_df now used for daily ATR fallback in morning ORB setups

        # 0. UNIVERSAL GATES (apply BEFORE category-specific gates)
        # Check if we're in opening bell window (9:15-9:30) - may bypass some gates
        opening_bell_active = self._is_opening_bell_window(now)

        # Range compression filter - reject during volatility expansion
        # Can be bypassed during opening bell if configured
        if not (opening_bell_active and self._opening_bell_bypasses("range_compression")):
            range_passed, range_reason = self._check_range_compression(df5m, atr)
            if not range_passed:
                logger.info(f"[{category}] {symbol} {setup_type} rejected at UNIVERSAL_GATES: {range_reason}")
                return {"eligible": False, "reason": "universal_gate_fail", "details": [range_reason]}

        # Cap-strategy blocking - block certain setups for certain market cap segments
        # Use passed cap_segment or fetch if not provided
        if cap_segment is None:
            cap_segment = get_cap_segment(symbol)
        cap_passed, cap_reason = self._check_cap_strategy_blocking(symbol, setup_type, cap_segment)
        if not cap_passed:
            logger.debug(f"[{category}] {symbol} {setup_type} rejected at UNIVERSAL_GATES: {cap_reason}")
            return {"eligible": False, "reason": "universal_gate_fail", "details": [cap_reason]}

        # Price action directionality - validates last 3 bars follow expected direction
        pa_passed, pa_reason = self._check_price_action_directionality(df5m, bias)
        if not pa_passed:
            logger.debug(f"[{category}] {symbol} {setup_type} rejected at UNIVERSAL_GATES: {pa_reason}")
            return {"eligible": False, "reason": "universal_gate_fail", "details": [pa_reason]}

        # 1. SCREENING
        screen_result = self.screen(symbol, df5m, features, levels, now)
        if not screen_result.passed:
            logger.debug(f"[{category}] {symbol} {setup_type} rejected at SCREENING: {screen_result.reasons}")
            return {"eligible": False, "reason": "screening_fail", "details": screen_result.reasons}

        # 2. QUALITY
        quality_result = self.calculate_quality(symbol, df5m, bias, levels, atr)

        # Log quality metrics (like QUALITY_BREAKOUT, QUALITY_LEVEL in planner_internal.py)
        logger.info(f"QUALITY_{category}: {symbol} structural_rr={quality_result.structural_rr:.2f} "
                   f"status={quality_result.quality_status} metrics={quality_result.metrics}")

        # NOTE: Structural R:R min check with strategy-specific overrides is done in _apply_quality_filters()
        # This allows breakouts to use relaxed thresholds while other categories use stricter defaults

        # 3. GATES
        gate_result = self.validate_gates(
            symbol, setup_type, regime, df5m, df1m,
            strength=quality_result.structural_rr,
            adx=features.get("adx") or 0.0,
            vol_mult=features["volume_ratio"],
            regime_diagnostics=regime_diagnostics
        )
        if not gate_result.passed:
            logger.debug(f"[{category}] {symbol} {setup_type} rejected at GATES: {gate_result.reasons}")
            return {"eligible": False, "reason": "gate_fail", "details": gate_result.reasons}

        # 3b. UNIVERSAL RSI DEAD ZONE FILTER (for LONGS only) - CONFIG DRIVEN
        # Data-driven: RSI 35-50 has 43% of ALL hard_sl losses for LONG trades
        # ORB/FHM setups bypass this filter - RSI erratic at market open, structure detector enforces time cutoff
        base_cfg = load_base_config()
        rsi_dead_zone_cfg = base_cfg["rsi_dead_zone"]
        if rsi_dead_zone_cfg["enabled"]:
            is_orb = "orb" in setup_type.lower()
            is_fhm = "first_hour_momentum" in setup_type.lower()
            if is_orb or is_fhm:
                logger.debug(f"[{category}] {symbol} {setup_type} RSI dead zone BYPASSED: early morning setup")
            else:
                rsi_val = features.get("rsi") or features.get("rsi14")
                if rsi_val is None:
                    raise KeyError(f"RSI indicator missing from features for {symbol}")
                rsi_min = rsi_dead_zone_cfg["long_min"]
                rsi_max = rsi_dead_zone_cfg["long_max"]
                if bias == "long" and rsi_min <= rsi_val <= rsi_max:
                    logger.debug(f"[{category}] {symbol} {setup_type} BLOCKED: RSI dead zone ({rsi_val:.1f} in {rsi_min}-{rsi_max} range)")
                    return {"eligible": False, "reason": "rsi_dead_zone", "details": [f"rsi_{rsi_val:.1f}_in_{rsi_min}_{rsi_max}_dead_zone"]}

        # 4. RANKING (with HTF context for category-specific adjustments)
        rank_result = self.calculate_rank_score(symbol, features, regime, htf_context=htf_context)

        # Extract daily_trend from htf_context if available (from screener/planner)
        daily_trend = None
        if htf_context:
            daily_trend = htf_context.get("daily_trend", None)

        # 4b. UNIVERSAL RANKING ADJUSTMENTS (from ranker.py)
        # Apply time-of-day, blacklist, unrealistic RR, multi-TF daily/hourly, HTF 15m, daily trend multipliers
        adjusted_score, universal_adjustments = self.apply_universal_ranking_adjustments(
            base_score=rank_result.score,
            symbol=symbol,
            strategy_type=setup_type,
            structural_rr=quality_result.structural_rr,
            bias=bias,
            current_time=now,
            regime_diagnostics=regime_diagnostics,
            daily_score=daily_score,
            htf_context=htf_context,
            daily_trend=daily_trend
        )

        # 5. ENTRY - Category pipeline handles setup-type-specific entry logic
        orh = levels.get("ORH", current_close)
        orl = levels.get("ORL", current_close)

        # Get entry from category pipeline (includes setup-type-specific logic)
        entry_result = self.calculate_entry(symbol, df5m, bias, levels, atr, setup_type)
        entry_ref_price = entry_result.entry_ref_price

        # CRITICAL: For immediate mode, entry_ref_price is the level (ORH/ORL) but
        # actual fill will be at current_close (price has already broken through).
        # TARGETS: Use effective_entry_price (where we actually enter) for realistic R:R
        # SL: Use entry_ref_price (the level) - if price breaks back below, thesis is invalid
        if entry_result.entry_mode == "immediate":
            effective_entry_price = current_close
            logger.debug(f"[{category}] {symbol} {setup_type} IMMEDIATE mode: effective_entry={effective_entry_price:.2f} (ref={entry_ref_price:.2f})")
        else:
            effective_entry_price = entry_ref_price
            logger.debug(f"[{category}] {symbol} {setup_type} CONDITIONAL mode: entry_ref={entry_ref_price:.2f}")

        # 5b. VOLATILITY-ADJUSTED SIZING - DISABLED for Pro Trader approach
        # Van Tharp CPR: qty = risk_per_trade / rps
        # The ATR-based stop loss (rps) ALREADY incorporates volatility risk.
        # Adding volatility_mult on TOP of this is double-penalizing!
        # High volatility → larger ATR → larger rps → smaller qty (built-in)
        volatility_mult = 1.0  # DISABLED - ATR-based sizing handles volatility
        # volatility_cfg = self.cfg["volatility_sizing"]
        # if volatility_cfg["enabled"]:
        #     price_atr_ratio = (atr / entry_ref_price) * 100 if entry_ref_price > 0 else 1.0
        #     low_vol_threshold = volatility_cfg["low_volatility_threshold"]
        #     high_vol_threshold = volatility_cfg["high_volatility_threshold"]
        #
        #     if price_atr_ratio < low_vol_threshold:
        #         volatility_mult = volatility_cfg["low_volatility_multiplier"]
        #     elif price_atr_ratio > high_vol_threshold:
        #         volatility_mult = volatility_cfg["high_volatility_multiplier"]
        #     else:
        #         volatility_mult = volatility_cfg["normal_volatility_multiplier"]
        #
        #     # Clamp to limits
        #     max_adj = volatility_cfg["max_size_adjustment"]
        #     min_adj = volatility_cfg["min_size_adjustment"]
        #     volatility_mult = max(min_adj, min(max_adj, volatility_mult))
        logger.debug(f"VOLATILITY_SIZING: {symbol} DISABLED - using vol_mult=1.0 (ATR-based sizing handles volatility)")

        # 5c. CAP-AWARE SIZING - DISABLED for Pro Trader approach
        # Van Tharp CPR: qty = risk_per_trade / rps
        # The rps is ATR-based, and ATR naturally varies by cap segment.
        # Small/micro caps have higher ATR% → larger rps → smaller qty (built-in)
        # Adding cap_size_mult on TOP of this is double-penalizing!
        # cap_segment already set earlier in run_pipeline (passed or fetched)
        cap_size_mult = 1.0  # DISABLED - ATR-based sizing handles cap segment risk
        cap_sl_mult = 1.0  # Keep uniform SL multiplier

        # cap_risk_cfg = self.cfg["cap_risk_adjustments"]
        # if cap_risk_cfg["enabled"] and cap_segment != "unknown":
        #     seg_cfg = cap_risk_cfg[cap_segment]
        #     cap_size_mult = seg_cfg["size_multiplier"]
        #     cap_sl_mult = seg_cfg["sl_atr_multiplier"]
        logger.debug(f"CAP_SIZING: {symbol} {cap_segment} DISABLED - using size_mult=1.0 (ATR-based sizing handles cap risk)")

        # 6. STOP LOSS - Structure-based SL with RPS floor protection
        # Ported from OLD planner_internal.py lines 512-537
        # Uses: structure_stop from levels, ATR-based volatility stop, RPS floor

        # Check for setup-specific SL multiplier first (DATA-DRIVEN Dec 2024)
        sl_atr_mult = self.cfg["sl_atr_mult"]
        setup_specific_sl = self.cfg.get("stop_loss", {}).get("setup_specific")
        if setup_specific_sl and setup_type in setup_specific_sl:
            setup_sl_cfg = setup_specific_sl[setup_type]
            if "atr_multiplier" in setup_sl_cfg:
                sl_atr_mult = setup_sl_cfg["atr_multiplier"]
                logger.debug(f"[PLAN] Using setup-specific SL for {setup_type}: atr_mult={sl_atr_mult}")

        adjusted_atr = atr * cap_sl_mult
        sl_below_swing_ticks = self.cfg["sl_below_swing_ticks"]

        # FHM-SPECIFIC SL: Use percentage of price instead of ATR-based (Pro Indian market standard)
        # PRO STANDARD: 0.5-1.5% of price for volatile momentum plays
        # Read from breakout_config.json screening.first_hour_momentum.stop_loss config
        is_fhm = setup_type.startswith("first_hour_momentum")
        if is_fhm:
            # Get FHM SL config from pipeline config (breakout_config.json)
            # Path: screening.first_hour_momentum.stop_loss.pct_of_price
            fhm_sl_pct = self._get("screening", "first_hour_momentum", "stop_loss", "pct_of_price")
            fhm_sl_distance = effective_entry_price * fhm_sl_pct
            if bias == "long":
                hard_sl = effective_entry_price - fhm_sl_distance
            else:
                hard_sl = effective_entry_price + fhm_sl_distance
            rps = fhm_sl_distance
            logger.debug(f"FHM_SL: {symbol} using {fhm_sl_pct*100:.1f}% of price = {fhm_sl_distance:.2f} Rs, hard_sl={hard_sl:.2f}")
        else:
            # Get structure stop from levels if available (swing low for long, swing high for short)
            structure_stop = levels.get("structure_stop")
            if structure_stop is None:
                # Fallback: use ORH/ORL as structure reference
                if bias == "long":
                    structure_stop = levels.get("ORL", entry_ref_price - adjusted_atr)
                else:
                    structure_stop = levels.get("ORH", entry_ref_price + adjusted_atr)

            # Calculate both structure-based and volatility-based stops
            # CRITICAL: Use entry_ref_price (the level), NOT effective_entry_price (fill price)
            # Reason: SL should protect the breakout thesis - if price breaks back below the level,
            # the breakout has failed regardless of where we filled. This matches OLD planner_internal.py.
            # Pro trader approach: For long breakouts, SL at ORL (or below level), not relative to fill.
            vol_stop = entry_ref_price - (sl_atr_mult * adjusted_atr) if bias == "long" else entry_ref_price + (sl_atr_mult * adjusted_atr)

            if bias == "long":
                # For LONG: SL below entry - use HIGHER value (closer to entry) = TIGHTER stop
                structure_sl = structure_stop - sl_below_swing_ticks
                hard_sl = max(structure_sl, vol_stop)  # Takes closer SL to entry
                rps = max(effective_entry_price - hard_sl, 0.0)
            else:
                # For SHORT: SL above entry - use LOWER value (closer to entry) = TIGHTER stop
                structure_sl = structure_stop + sl_below_swing_ticks
                hard_sl = min(structure_sl, vol_stop)  # Takes closer SL to entry
                rps = max(hard_sl - effective_entry_price, 0.0)

        # RPS FLOOR PROTECTION (from planner_internal.py lines 527-537)
        # Prevents too-tight stops that get hit easily
        planner_precision = self.cfg["planner_precision"]
        min_rps_bpct = planner_precision["min_rps_bpct"]
        atr_rps_mult = planner_precision["atr_rps_mult"]
        floor_by_px = effective_entry_price * (min_rps_bpct / 100.0)
        floor_by_atr = adjusted_atr * atr_rps_mult
        rps_floor = max(floor_by_px, floor_by_atr, 0.0)

        if rps < rps_floor:
            # Widen SL to meet floor
            if bias == "long":
                hard_sl = effective_entry_price - rps_floor
            else:
                hard_sl = effective_entry_price + rps_floor
            rps = rps_floor
            logger.debug(f"RPS_FLOOR: {symbol} rps widened to floor={rps_floor:.4f}")

        # 7. TARGETS
        # CRITICAL: Use effective_entry_price for target calculations
        # For immediate mode, targets should be relative to where we actually enter
        measured_move = max(orh - orl, atr)

        target_result = self.calculate_targets(
            symbol, effective_entry_price, hard_sl,
            bias, atr, levels, measured_move, setup_type
        )

        # REJECTION: Low volatility - can't achieve viable targets
        if target_result is None:
            logger.debug(f"[{category}] {symbol} {setup_type} rejected: low volatility - T1 cap below threshold")
            return {"eligible": False, "reason": "low_volatility_target", "details": ["t1_cap_below_0.8R"]}

        # 8. POSITION SIZING - NO SOFT PENALTIES (Pro Trader Approach)
        #
        # REMOVED ALL STACKING PENALTIES (Option A - Hard Gates Only):
        # - RSI penalty: Now hard gate in pipeline.validate_gates()
        # - MACD late penalty: If late, don't trade (handled by timing gates)
        # - Volume penalty: Hard gate in pipeline.screen() or validate_gates()
        # - ADX penalty: Hard gate in pipeline.validate_gates()
        #
        # Pro Trader Formula (Van Tharp CPR):
        #   Position Size = Capital at Risk / Risk per Unit
        # Then apply STRUCTURAL adjustments only:
        #   - volatility_mult: Built into ATR-based sizing
        #   - cap_size_mult: Market cap tier adjustment
        #
        # NO STACKING of soft penalties that crushed position sizes to 22% of intended!
        size_mult = gate_result.size_mult  # 1.0 from gates (no penalties there either)
        cautions = []
        # 8a. APPLY VOLATILITY AND CAP SIZING MULTIPLIERS (structural, not penalties)
        # These adjust position size based on volatility regime and market cap
        size_mult *= volatility_mult * cap_size_mult

        # 8b. CALCULATE POSITION SIZE (qty)
        # Uses risk-based position sizing: qty = risk_per_trade / risk_per_share * multipliers
        # NOTE: Use locally calculated 'rps' (with floor protection) NOT target_result.risk_per_share
        risk_per_trade_rupees = self.cfg["risk_per_trade_rupees"]
        risk_per_share = rps  # Use our calculated rps with floor protection
        if risk_per_share > 0:
            base_qty = int(risk_per_trade_rupees / risk_per_share)
            qty = max(int(base_qty * size_mult), 0)
        else:
            qty = 0
        # Use effective_entry_price for notional (actual entry price for immediate mode)
        notional = qty * effective_entry_price

        # 9. BUILD PLAN
        plan = {
            "symbol": symbol,
            "eligible": True,
            "strategy": setup_type,
            "bias": bias,
            "regime": regime,
            "category": self.get_category_name(),

            "entry_ref_price": round(entry_ref_price, 2),
            "entry": {
                "reference": round(entry_ref_price, 2),
                "zone": [round(entry_result.entry_zone[0], 2), round(entry_result.entry_zone[1], 2)],
                "trigger": entry_result.entry_trigger,
                "mode": entry_result.entry_mode,
            },

            "stop": {
                "hard": round(hard_sl, 2),
                "risk_per_share": round(rps, 2),  # Use locally calculated rps with floor protection
            },

            "targets": target_result.targets,
            "trail": target_result.trail_config,

            "quality": {
                "structural_rr": round(quality_result.structural_rr, 2),
                "status": quality_result.quality_status,
                "metrics": quality_result.metrics,
                "t1_feasible": True,  # Will be validated below
                "t2_feasible": True,
            },

            "ranking": {
                "score": round(adjusted_score, 3),
                "base_score": round(rank_result.score, 3),
                "components": rank_result.components,
                "multipliers": rank_result.multipliers,
                "universal_adjustments": universal_adjustments,
            },

            "sizing": {
                "qty": qty,
                "notional": round(notional, 2),
                "risk_rupees": risk_per_trade_rupees,
                "risk_per_share": round(risk_per_share, 2),
                "size_mult": round(size_mult, 2),
                "base_mult": round(gate_result.size_mult, 2),
                "volatility_mult": round(volatility_mult, 2),
                "cap_size_mult": round(cap_size_mult, 2),
                "cap_segment": cap_segment,
                "cap_sl_mult": round(cap_sl_mult, 2),
                "min_hold_bars": gate_result.min_hold_bars,
            },

            "indicators": {
                "atr": round(atr, 2),
                "adx": round(adx_val, 1) if adx_val else None,
                "rsi": round(rsi_val, 1) if rsi_val else None,
                "vwap": round(vwap_val, 2) if vwap_val else None,
            },

            "levels": levels,
            "pipeline_reasons": screen_result.reasons + gate_result.reasons,
            "cautions": cautions,
            # FHM context for diagnostics logging (RVOL, price move %)
            "fhm_context": {
                "rvol": screen_result.features.get("fhm_rvol", 0.0),
                "price_move_pct": screen_result.features.get("fhm_price_move_pct", 0.0),
                "eligible": screen_result.features.get("fhm_eligible", False),
            } if "first_hour_momentum" in setup_type else None,
        }

        # 10. QUALITY FILTER ENFORCEMENT (transferred from planner_internal.py lines 1358-1451)
        # CRITICAL FIX: Use effective_entry_price for quality filters
        # For immediate mode breakouts, targets are relative to current_close (effective_entry_price)
        # not the level (entry_ref_price). Quality filters must use the same reference.
        plan = self._apply_quality_filters(plan, effective_entry_price, atr, measured_move)

        if plan["eligible"]:
            logger.info(f"[{category}] {symbol} {setup_type} APPROVED: score={adjusted_score:.2f} (base={rank_result.score:.2f}), quality={quality_result.quality_status}, entry={entry_ref_price:.2f}")
        else:
            logger.debug(f"[{category}] {symbol} {setup_type} rejected by quality filters: {plan.get('quality', {}).get('rejection_reason', 'unknown')}")

        return plan

    def _apply_quality_filters(
        self,
        plan: Dict[str, Any],
        entry_ref_price: float,
        atr: float,
        measured_move: float
    ) -> Dict[str, Any]:
        """
        Apply universal quality filter enforcement (from planner_internal.py lines 1358-1451).

        Universal Filters (apply to ALL categories):
        1. T1 feasibility check (cap to reasonable target)
        2. T2 feasibility → T1-only scalp mode (reduce qty instead of reject)
        3. Min T1 R:R filter

        Category-specific filters (handled in category pipelines):
        - ADX for breakouts → BreakoutPipeline.validate_gates()
        """
        if not plan["eligible"]:
            return plan

        # Get quality filter config
        quality_filters = self.cfg["quality_filters"]
        if not quality_filters["enabled"]:
            return plan

        risk_per_share = plan["stop"]["risk_per_share"]

        # Feasibility caps (from planner_precision config)
        t1_max_pct = quality_filters["t1_max_pct"]
        t1_max_mm_frac = quality_filters["t1_max_mm_frac"]
        t2_max_pct = quality_filters["t2_max_pct"]
        t2_max_mm_frac = quality_filters["t2_max_mm_frac"]

        cap1 = min(entry_ref_price * (t1_max_pct / 100.0), measured_move * t1_max_mm_frac)
        cap2 = min(entry_ref_price * (t2_max_pct / 100.0), measured_move * t2_max_mm_frac)

        # Check T1 feasibility
        if plan["targets"]:
            t1_level = plan["targets"][0]["level"]
            if plan["bias"] == "long":
                t1_move = t1_level - entry_ref_price
            else:
                t1_move = entry_ref_price - t1_level

            plan["quality"]["t1_feasible"] = t1_move <= cap1 + 0.01  # Small tolerance

            if not plan["quality"]["t1_feasible"]:
                # Cap T1 to feasible level
                if plan["bias"] == "long":
                    plan["targets"][0]["level"] = round(entry_ref_price + cap1, 2)
                else:
                    plan["targets"][0]["level"] = round(entry_ref_price - cap1, 2)

                # Recalculate R:R
                plan["targets"][0]["rr"] = round(cap1 / max(risk_per_share, 0.01), 2)
                logger.debug(f"T1 capped to feasible level: {plan['targets'][0]['level']}")

        # Check T2 feasibility → T1-only scalp mode
        if len(plan["targets"]) > 1:
            t2_level = plan["targets"][1]["level"]
            if plan["bias"] == "long":
                t2_move = t2_level - entry_ref_price
            else:
                t2_move = entry_ref_price - t2_level

            plan["quality"]["t2_feasible"] = t2_move <= cap2 + 0.01

            if not plan["quality"]["t2_feasible"]:
                # Cap T2 and reduce size for T1-only scalp mode
                if plan["bias"] == "long":
                    plan["targets"][1]["level"] = round(entry_ref_price + cap2, 2)
                else:
                    plan["targets"][1]["level"] = round(entry_ref_price - cap2, 2)

                plan["targets"][1]["rr"] = round(cap2 / max(risk_per_share, 0.01), 2)

                # Reduce size for T1-only mode (from planner_internal.py lines 1398-1418)
                plan["sizing"]["size_mult"] *= 0.5
                plan["quality"]["t2_exit_mode"] = "T1_only_scalp"
                plan["cautions"].append("T2_infeasible_scalp_mode")
                logger.debug(f"T2 infeasible → T1-only scalp mode, size_mult reduced to {plan['sizing']['size_mult']:.2f}")

        # NOTE: ADX filter for breakouts is handled in BreakoutPipeline.validate_gates()
        # (category-specific gate, not a universal quality filter)

        # Structural R:R filter with strategy-specific overrides (from planner_internal.py lines 1365-1396)
        # Breakouts have momentum that blows through resistance, so they can use relaxed thresholds
        strategy_type = plan.get("strategy", "")
        strategy_rr_overrides = quality_filters["strategy_structural_rr_overrides"]
        min_structural_rr = strategy_rr_overrides.get(strategy_type, quality_filters["min_structural_rr"])

        structural_rr_val = plan["quality"].get("structural_rr")
        if plan["eligible"] and structural_rr_val is not None and structural_rr_val < min_structural_rr:
            plan["eligible"] = False
            plan["quality"]["rejection_reason"] = f"structural_rr {structural_rr_val:.2f} < {min_structural_rr:.2f}"
            logger.debug(f"Rejected: structural_rr {structural_rr_val:.2f} < {min_structural_rr:.2f} (strategy={strategy_type})")

        # Min T1 R:R filter - check for bias-specific override
        bias = plan.get("bias", "long")
        if bias == "long" and "min_t1_rr_long" in quality_filters:
            min_t1_rr = quality_filters["min_t1_rr_long"]
        elif bias == "short" and "min_t1_rr_short" in quality_filters:
            min_t1_rr = quality_filters["min_t1_rr_short"]
        else:
            min_t1_rr = quality_filters["min_t1_rr"]

        if plan["eligible"] and plan["targets"]:
            t1_rr = plan["targets"][0].get("rr", 0)
            if t1_rr < min_t1_rr:
                plan["eligible"] = False
                plan["quality"]["rejection_reason"] = f"T1_rr {t1_rr:.2f} < {min_t1_rr}"
                logger.debug(f"Rejected: T1_rr {t1_rr:.2f} < {min_t1_rr} (bias={bias})")

        return plan
