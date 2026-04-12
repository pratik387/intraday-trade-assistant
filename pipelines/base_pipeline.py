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


# ================================================================================
# CENTRALIZED HARD BLOCKS - Single source of truth for regime-based blocking
# ================================================================================
# These blocks CANNOT be bypassed by config or any mechanism.
# Based on backtest analysis showing these setups underperform in specific regimes.
#
# SQUEEZE REGIME ANALYSIS (backtest_20251211):
#   - first_hour_momentum_long: 40 trades, Rs 119 total = Rs 3/trade avg (worthless)
#   - volume_spike_reversal_short: 7 trades, Rs -632 total (negative)
#   - premium_zone_short: Not in allowed list but was trading via HCET bypass
#
# Pro trader approach: AVOID trading during squeeze/consolidation. Wait for breakout.
# ================================================================================
HARD_BLOCKS = {
    # GATES-OFF RUN: All hard blocks disabled for raw detection quality test
    # Re-enable with regime-specific entries after next backtest validates:
    # "squeeze": ["first_hour_momentum_long", "volume_spike_reversal_short"],
}

if not HARD_BLOCKS:
    logger.warning("HARD_BLOCKS is empty — all regime-based hard blocks are DISABLED (gates-off mode)")


def is_hard_blocked(setup_type: str, regime: str) -> bool:
    """
    Check if a setup is HARD BLOCKED for a regime.

    Hard blocks CANNOT be bypassed by config or any mechanism.
    This is the single source of truth for regime-based blocking.

    Moved from regime_gate.py to base_pipeline.py for:
    - Single source of truth
    - Pipeline-level access without gate dependency
    - Cleaner architecture separation

    Args:
        setup_type: The setup type to check
        regime: The current market regime

    Returns:
        True if the setup is hard blocked for this regime
    """
    blocked_setups = HARD_BLOCKS.get(regime, [])
    return setup_type in blocked_setups


def safe_level_get(levels: Dict, key: str, fallback: float) -> float:
    """
    Safely get a level value with proper NaN handling.

    CRITICAL: dict.get(key, fallback) returns the value if key exists, even if value is NaN.
    This function properly returns the fallback when the value is NaN.

    This is important for late server starts when ORH/ORL are NaN.
    """
    val = levels.get(key)
    if val is None or pd.isna(val):
        return fallback
    return val


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


# Cache for MIS info lookups (loaded once per session)
_mis_info_cache: Dict[str, dict] = {}
_mis_info_loaded: bool = False


def get_mis_info(symbol: str) -> dict:
    """
    Get MIS (Margin Intraday Square-off) info for a symbol from nse_all.json.

    Returns dict with:
      - mis_enabled: bool (whether Zerodha allows MIS for this stock)
      - mis_leverage: float or None (typically 5.0 for MIS-eligible stocks)
    """
    global _mis_info_cache, _mis_info_loaded

    if not _mis_info_loaded:
        try:
            nse_file = Path(__file__).parent.parent / "nse_all.json"
            if nse_file.exists():
                with nse_file.open() as f:
                    data = json.load(f)
                _mis_info_cache = {
                    _normalize_symbol(item["symbol"]): {
                        "mis_enabled": item.get("mis_enabled", False),
                        "mis_leverage": item.get("mis_leverage"),
                    }
                    for item in data
                }
                _mis_info_loaded = True
                logger.debug(f"MIS_INFO: Loaded {len(_mis_info_cache)} symbols from nse_all.json")
        except Exception as e:
            logger.debug(f"MIS_INFO: Failed to load cache: {e}")
            _mis_info_loaded = True

    normalized = _normalize_symbol(symbol)
    return _mis_info_cache.get(normalized, {"mis_enabled": False, "mis_leverage": None})


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


def set_base_config_override(key: str, value: Any) -> None:
    """
    Override a value in the cached base config.

    This allows main.py to set runtime values (like risk_per_trade_rupees from
    capital manager) that will be picked up by all pipelines.

    MUST be called AFTER load_base_config() has been called at least once.

    Args:
        key: Config key to override (e.g., "risk_per_trade_rupees")
        value: New value to set
    """
    global _BASE_CONFIG_CACHE
    if _BASE_CONFIG_CACHE is None:
        # Force load first
        load_base_config()
    _BASE_CONFIG_CACHE[key] = value
    logger.info(f"CONFIG_OVERRIDE: Set {key}={value}")


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
        required_sections = ["screening", "quality", "gates", "entry", "targets"]
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
        atr: float,
        setup_type: str = ""
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
        strength: float,
        adx: float,
        vol_mult: float,
        regime_diagnostics: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """Apply category-specific gate validations."""
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

    # ======================== VALIDATED COMBINATION WHITELIST ========================
    # Walk-forward validated filter gate. A trade MUST match at least one
    # validated combination (setup-specific OR cap-segment) to pass.
    # Config: base_config.json -> validated_combinations

    def _check_validated_combinations(
        self, setup_type: str, features: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Whitelist gate: trade must match at least one walk-forward validated
        combination to pass. Returns (passed, reason).

        Rules are OR'd: if ANY setup rule OR ANY cap rule matches, trade passes.
        Within a single rule, all conditions are AND'd.
        """
        base_cfg = load_base_config()
        # WIDE-OPEN MODE: bypass VC whitelist for edge rediscovery
        if base_cfg.get("wide_open_mode", False):
            return True, "wide_open_mode"
        vc_cfg = base_cfg.get("validated_combinations", {})
        if not vc_cfg.get("enabled", False):
            return True, "vc_disabled"

        setup_lower = setup_type.lower()
        cap = features.get("cap_segment", "unknown")

        # Check setup-specific rules (OR logic - ANY match = pass)
        setup_rules = vc_cfg.get("setup_rules", {}).get(setup_lower, [])
        for rule in setup_rules:
            if self._vc_rule_matches(rule, features):
                return True, f"vc_setup:{setup_lower}:{rule.get('name', 'unnamed')}"

        # Check cap-segment rules (OR logic - ANY match = pass)
        cap_rules = vc_cfg.get("cap_rules", {}).get(cap, [])
        for rule in cap_rules:
            if self._vc_rule_matches(rule, features):
                return True, f"vc_cap:{cap}:{rule.get('name', 'unnamed')}"

        # No match found - reject
        return False, f"vc_no_match:{setup_lower}:{cap}"

    def _vc_rule_matches(self, rule: Dict[str, Any], features: Dict[str, Any]) -> bool:
        """Check if a single validated combination rule matches the current features."""
        conditions = rule.get("conditions", [])
        if len(conditions) == 0:
            # Empty conditions = unconditional pass (e.g. gap_fill_short "all" rule)
            return True
        for cond in conditions:
            feat_name = cond["feature"]
            val = features.get(feat_name)
            if val is None:
                return False
            if "min" in cond and val < cond["min"]:
                return False
            if "max" in cond and val > cond["max"]:
                return False
        return True

    # ======================== UNIFIED FILTER SYSTEM ========================
    # All setup-specific filters are applied through this centralized method.
    # Filters are configured in gates.setup_filters.<setup_type> with enabled flags.

    def apply_setup_filters(
        self,
        setup_type: str,
        symbol: str,
        regime: str,
        adx: float,
        rsi: Optional[float],
        volume: float,
        current_hour: int,
        cap_segment: str,
        rank_score: Optional[float] = None,
        structural_rr: Optional[float] = None
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Apply unified setup-specific filters from config.

        All filters are in gates.setup_filters.<setup_type> with:
        - enabled: bool - master toggle for this setup's filters
        - Individual filter params with optional enabled flags

        Args:
            setup_type: Setup type (e.g., "orb_pullback_short")
            symbol: Stock symbol
            regime: Current market regime
            adx: Current ADX value
            rsi: Current RSI value (can be None)
            volume: Current bar volume
            current_hour: Current hour (0-23)
            cap_segment: Market cap segment (large_cap, mid_cap, small_cap, micro_cap)
            rank_score: Rank score (only for post-ranking filters, None otherwise)
            structural_rr: Structural risk-reward ratio (for min_rr/max_rr filters)

        Returns:
            Tuple of (passed, reasons, modifiers):
            - passed: bool - True if all filters pass
            - reasons: List[str] - Filter results for logging
            - modifiers: Dict with sl_mult, t1_mult, t2_mult if configured
        """
        setup_lower = setup_type.lower()
        reasons = []
        modifiers = {"sl_mult": 1.0, "t1_mult": 1.0, "t2_mult": 1.0}

        # WIDE-OPEN MODE: bypass ALL setup_filters (blocked_entirely, blocked_regimes,
        # blocked_caps, blocked_hours, allowed_*, min_adx, max_adx, etc.). Used during
        # 3-year edge rediscovery so raw detector signal flows to trade_report.csv for
        # analysis via deep_edge_analysis.py / edge_optimizer.py / filter_simulation.py.
        try:
            _wide_open = bool(self.cfg.get("wide_open_mode", False))
        except Exception:
            _wide_open = False
        if _wide_open:
            reasons.append(f"wide_open_mode:{setup_lower}")
            return True, reasons, modifiers

        # Get setup-specific filter config - MUST exist in gates.setup_filters
        setup_filters = self._get("gates", "setup_filters") or {}
        filter_cfg = setup_filters.get(setup_lower)

        # If no config for this setup, pass through (no filters defined)
        if not filter_cfg:
            reasons.append(f"no_setup_filter_config:{setup_lower}")
            return True, reasons, modifiers

        # Check if filters are enabled for this setup - REQUIRED field
        if not filter_cfg.get("enabled"):
            reasons.append(f"filters_disabled:{setup_lower}")
            return True, reasons, modifiers

        # 1. BLOCKED_ENTIRELY - setup is completely disabled (only check if key exists)
        if filter_cfg.get("blocked_entirely"):
            reasons.append(f"blocked_entirely:{setup_lower}")
            logger.debug(f"[FILTER] {symbol} {setup_lower} BLOCKED: blocked_entirely=true")
            return False, reasons, modifiers

        # 2. REGIME FILTERS
        blocked_regimes = filter_cfg.get("blocked_regimes") or []
        if regime in blocked_regimes:
            reasons.append(f"regime_blocked:{setup_lower}_{regime}")
            logger.debug(f"[FILTER] {symbol} {setup_lower} BLOCKED: regime {regime} in blocked_regimes")
            return False, reasons, modifiers

        allowed_regimes = filter_cfg.get("allowed_regimes")
        if allowed_regimes is not None and regime not in allowed_regimes:
            reasons.append(f"regime_not_allowed:{setup_lower}_{regime}")
            logger.debug(f"[FILTER] {symbol} {setup_lower} BLOCKED: regime {regime} not in allowed_regimes {allowed_regimes}")
            return False, reasons, modifiers

        # 3. CAP SEGMENT FILTERS
        blocked_caps = filter_cfg.get("blocked_caps") or []
        if cap_segment in blocked_caps:
            reasons.append(f"cap_blocked:{setup_lower}_{cap_segment}")
            logger.debug(f"[FILTER] {symbol} {setup_lower} BLOCKED: cap_segment {cap_segment} in blocked_caps")
            return False, reasons, modifiers

        allowed_caps = filter_cfg.get("allowed_caps")
        if allowed_caps is not None and cap_segment not in allowed_caps:
            reasons.append(f"cap_not_allowed:{setup_lower}_{cap_segment}")
            logger.debug(f"[FILTER] {symbol} {setup_lower} BLOCKED: cap_segment {cap_segment} not in allowed_caps {allowed_caps}")
            return False, reasons, modifiers

        # 4. HOUR FILTERS
        blocked_hours = filter_cfg.get("blocked_hours") or []
        if current_hour in blocked_hours:
            reasons.append(f"hour_blocked:{setup_lower}_{current_hour}")
            logger.debug(f"[FILTER] {symbol} {setup_lower} BLOCKED: hour {current_hour} in blocked_hours {blocked_hours}")
            return False, reasons, modifiers

        allowed_hours = filter_cfg.get("allowed_hours")
        if allowed_hours is not None and current_hour not in allowed_hours:
            reasons.append(f"hour_not_allowed:{setup_lower}_{current_hour}")
            logger.debug(f"[FILTER] {symbol} {setup_lower} BLOCKED: hour {current_hour} not in allowed_hours {allowed_hours}")
            return False, reasons, modifiers

        # 5. ADX FILTERS
        min_adx = filter_cfg.get("min_adx")
        if min_adx is not None and adx < min_adx:
            reasons.append(f"adx_below_min:{setup_lower}_adx{adx:.0f}<{min_adx}")
            logger.debug(f"[FILTER] {symbol} {setup_lower} BLOCKED: ADX {adx:.1f} < min_adx {min_adx}")
            return False, reasons, modifiers

        max_adx = filter_cfg.get("max_adx")
        if max_adx is not None and adx >= max_adx:
            reasons.append(f"adx_above_max:{setup_lower}_adx{adx:.0f}>={max_adx}")
            logger.debug(f"[FILTER] {symbol} {setup_lower} BLOCKED: ADX {adx:.1f} >= max_adx {max_adx}")
            return False, reasons, modifiers

        # 6. RSI FILTERS
        min_rsi = filter_cfg.get("min_rsi")
        if min_rsi is not None:
            if rsi is None:
                logger.debug(f"[FILTER] {symbol} {setup_lower} RSI filter skipped: RSI is None (min_rsi={min_rsi})")
            elif rsi < min_rsi:
                reasons.append(f"rsi_below_min:{setup_lower}_rsi{rsi:.0f}<{min_rsi}")
                logger.debug(f"[FILTER] {symbol} {setup_lower} BLOCKED: RSI {rsi:.1f} < min_rsi {min_rsi}")
                return False, reasons, modifiers

        max_rsi = filter_cfg.get("max_rsi")
        if max_rsi is not None and rsi is not None and rsi > max_rsi:
            reasons.append(f"rsi_above_max:{setup_lower}_rsi{rsi:.0f}>{max_rsi}")
            logger.debug(f"[FILTER] {symbol} {setup_lower} BLOCKED: RSI {rsi:.1f} > max_rsi {max_rsi}")
            return False, reasons, modifiers

        # 7. VOLUME FILTERS
        min_volume = filter_cfg.get("min_volume")
        if min_volume is not None and volume < min_volume:
            reasons.append(f"vol_below_min:{setup_lower}_vol{volume/1000:.0f}k<{min_volume/1000:.0f}k")
            logger.debug(f"[FILTER] {symbol} {setup_lower} BLOCKED: volume {volume/1000:.0f}k < min_volume {min_volume/1000:.0f}k")
            return False, reasons, modifiers

        # 8. STRUCTURAL R:R FILTERS (min_rr, max_rr for quality filtering)
        if structural_rr is not None:
            min_rr = filter_cfg.get("min_rr")
            if min_rr is not None and structural_rr < min_rr:
                reasons.append(f"rr_below_min:{setup_lower}_rr{structural_rr:.2f}<{min_rr}")
                logger.debug(f"[FILTER] {symbol} {setup_lower} BLOCKED: structural_rr {structural_rr:.2f} < min_rr {min_rr}")
                return False, reasons, modifiers

            max_rr = filter_cfg.get("max_rr")
            if max_rr is not None and structural_rr >= max_rr:
                reasons.append(f"rr_above_max:{setup_lower}_rr{structural_rr:.2f}>={max_rr}")
                logger.debug(f"[FILTER] {symbol} {setup_lower} BLOCKED: structural_rr {structural_rr:.2f} >= max_rr {max_rr}")
                return False, reasons, modifiers

        # 9. RANK SCORE FILTERS (only if rank_score provided - post-ranking call)
        if rank_score is not None:
            min_rank_score = filter_cfg.get("min_rank_score")
            if min_rank_score is not None and rank_score < min_rank_score:
                reasons.append(f"rank_below_min:{setup_lower}_rank{rank_score:.2f}<{min_rank_score}")
                logger.debug(f"[FILTER] {symbol} {setup_lower} BLOCKED: rank_score {rank_score:.2f} < min_rank_score {min_rank_score}")
                return False, reasons, modifiers

            max_rank_score = filter_cfg.get("max_rank_score")
            if max_rank_score is not None and rank_score > max_rank_score:
                reasons.append(f"rank_above_max:{setup_lower}_rank{rank_score:.2f}>{max_rank_score}")
                logger.debug(f"[FILTER] {symbol} {setup_lower} BLOCKED: rank_score {rank_score:.2f} > max_rank_score {max_rank_score}")
                return False, reasons, modifiers

        # 9. COLLECT TARGET/SL MODIFIERS (these don't block, just modify)
        if filter_cfg.get("sl_mult"):
            modifiers["sl_mult"] = filter_cfg["sl_mult"]
        if filter_cfg.get("t1_mult"):
            modifiers["t1_mult"] = filter_cfg["t1_mult"]
        if filter_cfg.get("t2_mult"):
            modifiers["t2_mult"] = filter_cfg["t2_mult"]

        reasons.append(f"setup_filters_passed:{setup_lower}")
        return True, reasons, modifiers

    def apply_global_filters(
        self,
        setup_type: str,
        symbol: str,
        bias: str,
        adx: float,
        rsi: Optional[float],
        volume: float
    ) -> Tuple[bool, List[str]]:
        """
        Apply global filters that apply to all setups (e.g., short min ADX, long min volume).

        These are pipeline-agnostic filters configured in gates.global_filters.

        Args:
            setup_type: Setup type for logging
            symbol: Stock symbol
            bias: Trade direction ("long" or "short")
            adx: Current ADX value
            rsi: Current RSI value
            volume: Current bar volume

        Returns:
            Tuple of (passed, reasons)
        """
        reasons = []

        global_filters = self._get("gates", "global_filters") or {}

        # 1. GLOBAL SHORT MIN ADX - MUST be configured in gates.global_filters.short_min_adx
        short_adx_cfg = global_filters.get("short_min_adx")
        if bias == "short" and short_adx_cfg and short_adx_cfg.get("enabled"):
            min_adx = short_adx_cfg["min_adx"]  # REQUIRED - no default
            if adx < min_adx:
                reasons.append(f"global_short_adx_blocked:{adx:.0f}<{min_adx}")
                logger.debug(f"[GLOBAL_FILTER] {symbol} {setup_type} BLOCKED: short ADX {adx:.1f} < global min {min_adx}")
                return False, reasons

        # 2. GLOBAL LONG MIN VOLUME - MUST be configured in gates.global_filters.long_min_volume
        long_vol_cfg = global_filters.get("long_min_volume")
        if bias == "long" and long_vol_cfg and long_vol_cfg.get("enabled"):
            min_volume = long_vol_cfg["min_volume"]  # REQUIRED - no default
            if volume < min_volume:
                reasons.append(f"global_long_vol_blocked:{volume/1000:.0f}k<{min_volume/1000:.0f}k")
                logger.debug(f"[GLOBAL_FILTER] {symbol} {setup_type} BLOCKED: long volume {volume/1000:.0f}k < global min {min_volume/1000:.0f}k")
                return False, reasons

        # 3. RSI DEAD ZONE - MUST be configured in gates.global_filters.rsi_dead_zone
        rsi_dead_zone_cfg = global_filters.get("rsi_dead_zone")
        if rsi_dead_zone_cfg and rsi_dead_zone_cfg.get("enabled") and rsi is not None:
            rsi_min = rsi_dead_zone_cfg["min_rsi"]  # REQUIRED - no default
            rsi_max = rsi_dead_zone_cfg["max_rsi"]  # REQUIRED - no default
            if rsi_min <= rsi <= rsi_max:
                reasons.append(f"global_rsi_dead_zone:{rsi:.0f}_in_{rsi_min}_{rsi_max}")
                logger.debug(f"[GLOBAL_FILTER] {symbol} {setup_type} BLOCKED: RSI {rsi:.1f} in dead zone {rsi_min}-{rsi_max}")
                return False, reasons

        reasons.append("global_filters_passed")
        return True, reasons

    # ======================== REMOVED: RANKING ADJUSTMENTS (Feb 2026) ========================
    # 3yr backtest: rank_score had -0.019 correlation with winning. All ranking computation
    # removed. structural_rr used as ordering metric. See plan: pure-twirling-deer.md

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
                logger.debug(f"[{category}] {symbol} {setup_type}: Using daily ATR {atr:.2f} for morning ORB (intraday ATR unavailable)")

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

        # --- Derived features for validated combination whitelist ---
        last_bar = df5m.iloc[-1]
        bar_range = float(last_bar["high"]) - float(last_bar["low"])
        features["bar_body_ratio"] = abs(float(last_bar["close"]) - float(last_bar["open"])) / bar_range if bar_range > 0 else 0.0

        vwap_for_dist = float(last_bar.get("vwap", 0)) if "vwap" in last_bar.index else 0
        features["vwap_dist_pct"] = abs(current_close - vwap_for_dist) / vwap_for_dist * 100 if vwap_for_dist > 0 else 0.0

        orh_level = levels.get("ORH")
        orl_level = levels.get("ORL")
        if orh_level is not None and orl_level is not None and not pd.isna(orh_level) and not pd.isna(orl_level) and orl_level > 0:
            or_width = orh_level - orl_level
            features["or_range_pct"] = or_width / orl_level * 100
            features["or_position"] = (current_close - orl_level) / or_width if or_width > 0 else 0.5
        else:
            features["or_range_pct"] = None
            features["or_position"] = None

        pdh_level = levels.get("PDH")
        pdl_level = levels.get("PDL")
        if pdh_level is not None and pdl_level is not None and not pd.isna(pdh_level) and not pd.isna(pdl_level) and pdl_level > 0:
            features["pd_range_pct"] = (pdh_level - pdl_level) / pdl_level * 100
        else:
            features["pd_range_pct"] = None

        last_volume = float(last_bar["volume"]) if "volume" in last_bar.index else 0
        or_range_val = features.get("or_range_pct")
        features["vol_or_ratio"] = last_volume / (or_range_val * 10000) if or_range_val and or_range_val > 0 else None
        features["bb_width_proxy"] = float(last_bar.get("bb_width_proxy", 0)) if "bb_width_proxy" in last_bar.index else 0.0
        features["volume5"] = last_volume

        # dist_from_level_bpct: distance from nearest key level in basis points
        _key_levels = [v for k in ("ORH", "ORL", "PDH", "PDL") if (v := levels.get(k)) is not None and not pd.isna(v)]
        if _key_levels and current_close > 0:
            _nearest = min(_key_levels, key=lambda lv: abs(lv - current_close))
            features["dist_from_level_bpct"] = (current_close - _nearest) / current_close * 10000  # signed bps
        else:
            features["dist_from_level_bpct"] = None

        # squeeze_pctile: percentile rank of current BB width within recent 5m bars
        if "bb_width_proxy" in df5m.columns and len(df5m) >= 10:
            _bbw_series = df5m["bb_width_proxy"].dropna()
            if len(_bbw_series) >= 10:
                _cur_bbw = features["bb_width_proxy"]
                features["squeeze_pctile"] = float((_bbw_series < _cur_bbw).sum() / len(_bbw_series) * 100)
            else:
                features["squeeze_pctile"] = None
        else:
            features["squeeze_pctile"] = None

        # cap_segment assignment deferred until after fallback lookup below

        # Approximate sl_atr_mult at gate time: sl_dist / bar_range
        # In step 6 (SL computation), structure_stop falls back to ORL (long) / ORH (short)
        # since levels["structure_stop"] is never populated. We replicate that here.
        if bar_range > 0:
            if bias == "long":
                _sl_ref = levels.get("ORL")
                if _sl_ref is not None and not pd.isna(_sl_ref):
                    features["sl_atr_mult"] = abs(current_close - _sl_ref) / bar_range
                else:
                    features["sl_atr_mult"] = None
            else:
                _sl_ref = levels.get("ORH")
                if _sl_ref is not None and not pd.isna(_sl_ref):
                    features["sl_atr_mult"] = abs(_sl_ref - current_close) / bar_range
                else:
                    features["sl_atr_mult"] = None
        else:
            features["sl_atr_mult"] = None

        # 0. UNIVERSAL GATES (apply BEFORE category-specific gates)
        # Check if we're in opening bell window (9:15-9:30) - may bypass some gates
        opening_bell_active = self._is_opening_bell_window(now)

        # Range compression filter - reject during volatility expansion
        # Can be bypassed during opening bell if configured
        if not (opening_bell_active and self._opening_bell_bypasses("range_compression")):
            range_passed, range_reason = self._check_range_compression(df5m, atr)
            if not range_passed:
                logger.debug(f"[{category}] {symbol} {setup_type} rejected at UNIVERSAL_GATES: {range_reason}")
                return {"eligible": False, "reason": "universal_gate_fail", "details": [range_reason]}

        # Cap-strategy blocking - block certain setups for certain market cap segments
        # Use passed cap_segment or fetch if not provided
        if cap_segment is None:
            cap_segment = get_cap_segment(symbol)
        features["cap_segment"] = cap_segment
        mis_info = get_mis_info(symbol)
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
        quality_result = self.calculate_quality(symbol, df5m, bias, levels, atr, setup_type=setup_type)

        # Reject if quality calculation failed (e.g., missing required levels)
        if quality_result is None:
            logger.debug(f"[{category}] {symbol} {setup_type} rejected at QUALITY: missing required data")
            return {"eligible": False, "reason": "quality_fail", "details": ["missing_required_levels"]}

        # Log quality metrics (DEBUG - detailed data in planning.jsonl)
        clean_metrics = {k: round(float(v), 2) if v is not None else None for k, v in quality_result.metrics.items()}
        logger.debug(f"QUALITY_{category}: {symbol} structural_rr={quality_result.structural_rr:.2f} "
                    f"status={quality_result.quality_status} metrics={clean_metrics}")

        # Hard floor: reject plans with zero structural R:R (no reward)
        # This fires even when quality_filters are disabled — zero-reward trades are never worth taking
        if quality_result.structural_rr <= 0.0:
            logger.warning(f"ZERO_RR: {symbol} {setup_type} structural_rr={quality_result.structural_rr:.2f} — rejected (zero reward)")
            return {"eligible": False, "reason": "zero_structural_rr", "details": [f"structural_rr={quality_result.structural_rr:.2f}"]}

        # NOTE: Structural R:R min check with strategy-specific overrides is done in _apply_quality_filters()
        # This allows breakouts to use relaxed thresholds while other categories use stricter defaults

        # 3. GATES
        gate_result = self.validate_gates(
            symbol, setup_type, regime, df5m,
            strength=quality_result.structural_rr,
            adx=features.get("adx") or 0.0,
            vol_mult=features["volume_ratio"],
            regime_diagnostics=regime_diagnostics
        )
        if not gate_result.passed:
            logger.debug(f"[{category}] {symbol} {setup_type} rejected at GATES: {gate_result.reasons}")
            return {"eligible": False, "reason": "gate_fail", "details": gate_result.reasons}

        # 3b. VALIDATED COMBINATION WHITELIST (walk-forward evidence gate)
        vc_passed, vc_reason = self._check_validated_combinations(setup_type, features)
        vc_audit = {
            "bar_body_ratio": round(features.get("bar_body_ratio", 0), 4),
            "vwap_dist_pct": round(features.get("vwap_dist_pct", 0), 4),
            "bb_width_proxy": round(features.get("bb_width_proxy", 0), 6),
            "volume5": features.get("volume5", 0),
            "vol_or_ratio": round(features["vol_or_ratio"], 4) if features.get("vol_or_ratio") is not None else None,
            "or_range_pct": round(features["or_range_pct"], 4) if features.get("or_range_pct") is not None else None,
            "or_position": round(features["or_position"], 4) if features.get("or_position") is not None else None,
            "pd_range_pct": round(features["pd_range_pct"], 4) if features.get("pd_range_pct") is not None else None,
            "sl_atr_mult": round(features["sl_atr_mult"], 4) if features.get("sl_atr_mult") is not None else None,
            "adx": round(features.get("adx", 0), 1) if features.get("adx") else None,
            "cap": features.get("cap_segment", "unknown"),
        }
        if not vc_passed:
            logger.debug(f"[{category}] {symbol} {setup_type} REJECTED VC: {vc_reason} | {vc_audit}")
            return {
                "eligible": False,
                "reason": "validated_combo_fail",
                "details": [vc_reason],
                "quality": {
                    "structural_rr": quality_result.structural_rr,
                    "status": quality_result.quality_status,
                },
                "indicators": vc_audit,
            }
        logger.debug(f"[{category}] {symbol} {setup_type} PASSED VC: {vc_reason} | {vc_audit}")

        # 3c. UNIVERSAL RSI DEAD ZONE FILTER (for LONGS only) - CONFIG DRIVEN
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

        # Add acceptance_status from quality calculation to features
        features["acceptance_status"] = quality_result.quality_status

        # 3d. QUALITY STATUS GATE (DATA-DRIVEN Feb 2026)
        # 3yr backtest: "fair" = 860 trades, WR 46%, Avg -Rs 30. All other statuses profitable.
        blocked_statuses = self.cfg.get("quality_filters", {}).get("blocked_quality_statuses", [])
        if quality_result.quality_status in blocked_statuses:
            logger.debug(f"[{category}] {symbol} {setup_type} BLOCKED: quality_status '{quality_result.quality_status}' in blocked list")
            return {"eligible": False, "reason": "quality_status_blocked", "details": [f"status_{quality_result.quality_status}_blocked"]}

        # 3e. TRADE QUALITY GATES (DATA-DRIVEN Feb 2026)
        # Pipeline-aware A: per-category srr + t1_rr thresholds
        # 13.9 T/day, WR 58.6%, Avg Rs 299, PF 1.81, all 4 FYs profitable
        tqg = self.cfg.get("trade_quality_gates", {})
        # WIDE-OPEN MODE: bypass entire trade_quality_gates block (blocked_hours, category floors,
        # setup_thresholds floors). Used during 3-year edge rediscovery.
        if self.cfg.get("wide_open_mode", False):
            tqg = {}
        if tqg.get("enabled", False):
            blocked_hours = tqg.get("blocked_hours", [])
            # Setup-specific override: allow a setup to opt out of the global blocked_hours
            # filter. Used by ICT zone setups in wide-open edge discovery mode — every hour
            # flows through so we can see the raw detector edge across the full session.
            setup_thresholds = tqg.get("setup_thresholds", {}).get(setup_type, {})
            bypass_blocked_hours = setup_thresholds.get("bypass_blocked_hours", False)
            if not bypass_blocked_hours and now and now.hour in blocked_hours:
                logger.debug(f"[{category}] {symbol} {setup_type} BLOCKED: hour {now.hour} in blocked_hours {blocked_hours}")
                return {"eligible": False, "reason": "blocked_hour", "details": [f"hour_{now.hour}_blocked"]}

            cat_thresholds = tqg.get("category_thresholds", {}).get(category, {})
            # Setup-specific override: allows per-setup floors, e.g. ICT zones use
            # structural range (tighter) and produce lower srr values than classical levels.
            min_srr = setup_thresholds.get("min_structural_rr", cat_thresholds.get("min_structural_rr", 0))
            if min_srr > 0 and quality_result.structural_rr < min_srr:
                logger.debug(f"[{category}] {symbol} {setup_type} BLOCKED: structural_rr {quality_result.structural_rr:.2f} < {min_srr:.1f}")
                return {"eligible": False, "reason": "structural_rr_below_floor", "details": [f"srr_{quality_result.structural_rr:.2f}_below_{min_srr}"]}

        # 4. ORDERING SCORE — structural R:R is the natural, data-validated metric
        # Ranking system REMOVED (Feb 2026): 3yr backtest proved composite rank_score has -0.019 correlation
        # with winning. rank_position #1-#5 all perform similarly. Pro approach: quality gates filter,
        # structural_rr orders. No composite scoring needed.
        adjusted_score = quality_result.structural_rr

        # NOTE: allowed_hours, min_rsi, max_rsi filters removed - handled pre-ranking in validate_gates()

        # 5. ENTRY - Category pipeline handles setup-type-specific entry logic
        orh = safe_level_get(levels, "ORH", current_close)
        orl = safe_level_get(levels, "ORL", current_close)

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
            if structure_stop is None or pd.isna(structure_stop):
                # Fallback: use ORH/ORL as structure reference
                # CRITICAL: Handle NaN properly - levels.get(key, fallback) returns NaN if key exists but value is NaN
                # This happens when server starts late (after ORB window) - ORH/ORL will be NaN
                if bias == "long":
                    orl_val = levels.get("ORL")
                    structure_stop = orl_val if (orl_val is not None and not pd.isna(orl_val)) else (entry_ref_price - adjusted_atr)
                else:
                    orh_val = levels.get("ORH")
                    structure_stop = orh_val if (orh_val is not None and not pd.isna(orh_val)) else (entry_ref_price + adjusted_atr)

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

            # SAFETY CHECK: Reject if hard_sl or rps is NaN (bad level data)
            if pd.isna(hard_sl) or pd.isna(rps):
                logger.warning(f"[{category}] {symbol} REJECTED: hard_sl or rps is NaN (hard_sl={hard_sl}, rps={rps}, structure_stop={structure_stop})")
                return {"eligible": False, "reason": "nan_stop_loss", "details": [f"hard_sl={hard_sl}", f"rps={rps}"]}

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

        # 6b. SETUP-SPECIFIC SL MULTIPLIER (sl_mult from setup_filters)
        # This WIDENS the stop loss for setups that need more room (e.g., resistance_bounce_short sl_mult=1.5)
        # NOTE: sl_mult defaults to 1.0 (no widening) if not specified - this is a neutral modifier, not a required value
        setup_block_cfg = self.cfg.get("gates", {}).get("setup_filters", {}).get(setup_type.lower(), {})
        sl_mult = setup_block_cfg.get("sl_mult", 1.0)
        if sl_mult != 1.0:
            if bias == "long":
                # For long, SL is below entry - widening moves it lower
                risk_distance = effective_entry_price - hard_sl
                widened_risk = risk_distance * sl_mult
                hard_sl = effective_entry_price - widened_risk
            else:
                # For short, SL is above entry - widening moves it higher
                risk_distance = hard_sl - effective_entry_price
                widened_risk = risk_distance * sl_mult
                hard_sl = effective_entry_price + widened_risk
            rps = widened_risk
            logger.debug(f"SL_MULT: {symbol} {setup_type} sl_mult={sl_mult} -> hard_sl={hard_sl:.2f}, rps={rps:.2f}")

        # 6c. VALIDATE SL IS OUTSIDE ENTRY ZONE - REJECT if not
        # For shorts: SL must be > entry_zone_high (above where we can enter)
        # For longs: SL must be < entry_zone_low (below where we can enter)
        #
        # If SL is inside entry zone, the trade is structurally broken:
        # - We could fill at edge of zone where SL is on WRONG side of entry
        # - E.g., short fills at 228, SL at 227.75 = SL below entry = exits on profit not loss
        #
        # REJECT rather than auto-adjust - don't risk real money on broken setups
        entry_zone = entry_result.entry_zone
        if bias == "long":
            if hard_sl >= entry_zone[0]:
                logger.warning(f"[{category}] {symbol} {setup_type} REJECTED: SL ({hard_sl:.2f}) >= entry_zone_low ({entry_zone[0]:.2f}) - SL inside entry zone")
                return {"eligible": False, "reason": "sl_inside_entry_zone", "details": [f"sl={hard_sl:.2f}", f"zone_low={entry_zone[0]:.2f}"]}
        else:  # short
            if hard_sl <= entry_zone[1]:
                logger.warning(f"[{category}] {symbol} {setup_type} REJECTED: SL ({hard_sl:.2f}) <= entry_zone_high ({entry_zone[1]:.2f}) - SL inside entry zone")
                return {"eligible": False, "reason": "sl_inside_entry_zone", "details": [f"sl={hard_sl:.2f}", f"zone_high={entry_zone[1]:.2f}"]}

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

        # 8a-ii. DIRECTIONAL BIAS MULTIPLIER (Nifty green/red → position size modulation)
        dir_bias_mult = 1.0
        dir_bias_reason = "dir_bias:neutral"
        from services.gates.directional_bias import get_tracker
        db_tracker = get_tracker()
        if db_tracker is not None:
            dir_bias_mult, dir_bias_reason = db_tracker.get_size_mult(bias, category=self.get_category_name())
            size_mult *= dir_bias_mult
            if dir_bias_mult != 1.0:
                cautions.append(dir_bias_reason)

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
        # CRITICAL: For immediate mode, use effective_entry_price (current_close) as the reference
        # since targets are calculated from there. This ensures plan is consistent.
        plan_entry_ref = effective_entry_price if entry_result.entry_mode == "immediate" else entry_ref_price
        plan = {
            "symbol": symbol,
            "eligible": True,
            "strategy": setup_type,
            "bias": bias,
            "regime": regime,
            "category": self.get_category_name(),

            "entry_ref_price": round(plan_entry_ref, 2),
            "entry_zone": [round(entry_result.entry_zone[0], 2), round(entry_result.entry_zone[1], 2)],
            "entry": {
                "reference": round(plan_entry_ref, 2),
                "zone": [round(entry_result.entry_zone[0], 2), round(entry_result.entry_zone[1], 2)],
                "trigger": entry_result.entry_trigger,
                "mode": entry_result.entry_mode,
            },

            "stop": {
                "hard": round(hard_sl, 2),
                "risk_per_share": round(rps, 2),  # Absolute risk for position sizing (entry - SL)
                "target_risk": round(target_result.risk_per_share, 2),  # Risk basis used for target R:R (structure-based for breakout)
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
                "score": round(adjusted_score, 3),  # = structural_rr (ranking removed Feb 2026)
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
                "dir_bias_mult": round(dir_bias_mult, 2),
                "dir_bias_reason": dir_bias_reason,
                "dir_bias_alignment": db_tracker.classify_alignment(bias) if db_tracker else "neutral",
                "cap_segment": cap_segment,
                "cap_sl_mult": round(cap_sl_mult, 2),
                "min_hold_bars": gate_result.min_hold_bars,
                "mis_enabled": mis_info.get("mis_enabled", False),
                "mis_leverage": mis_info.get("mis_leverage") or 1.0,
            },

            "indicators": {
                "atr": round(atr, 2),
                "adx": round(adx_val, 1) if adx_val else None,
                "rsi": round(rsi_val, 1) if rsi_val else None,
                "vwap": round(vwap_val, 2) if vwap_val else None,
            },

            "vc_reason": vc_reason,

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
            logger.info(f"[{category}] {symbol} {setup_type} APPROVED: score={adjusted_score:.2f} (structural_rr), quality={quality_result.quality_status}, entry={plan_entry_ref:.2f}")
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
            # quality_filters disabled, but trade_quality_gates t1_rr still applies
            tqg = self.cfg.get("trade_quality_gates", {})
            # WIDE-OPEN MODE: bypass trade_quality_gates t1_rr check
            if self.cfg.get("wide_open_mode", False):
                return plan
            if tqg.get("enabled", False) and plan["eligible"] and plan["targets"]:
                category = self.get_category_name()
                cat_thresholds = tqg.get("category_thresholds", {}).get(category, {})
                setup_thresholds_t1 = tqg.get("setup_thresholds", {}).get(plan.get("strategy", ""), {})
                tqg_min_t1_rr = setup_thresholds_t1.get("min_t1_rr", cat_thresholds.get("min_t1_rr", 0))
                if tqg_min_t1_rr > 0:
                    t1_rr_val = plan["targets"][0].get("rr", 0)
                    if t1_rr_val < tqg_min_t1_rr:
                        plan["eligible"] = False
                        plan["quality"]["rejection_reason"] = f"T1_rr {t1_rr_val:.2f} < {tqg_min_t1_rr} (trade_quality_gate)"
                        logger.debug(f"Rejected: T1_rr {t1_rr_val:.2f} < {tqg_min_t1_rr} (trade_quality_gate)")
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

        # Trade quality gates — per-category t1_rr floor (DATA-DRIVEN Feb 2026)
        # Applied AFTER quality_filters t1_rr check (trade_quality_gates threshold may be higher)
        tqg = self.cfg.get("trade_quality_gates", {})
        # WIDE-OPEN MODE: bypass second trade_quality_gates t1_rr check
        if self.cfg.get("wide_open_mode", False):
            return plan
        if tqg.get("enabled", False) and plan["eligible"] and plan["targets"]:
            category = self.get_category_name()
            cat_thresholds = tqg.get("category_thresholds", {}).get(category, {})
            setup_thresholds_t1 = tqg.get("setup_thresholds", {}).get(plan.get("strategy", ""), {})
            tqg_min_t1_rr = setup_thresholds_t1.get("min_t1_rr", cat_thresholds.get("min_t1_rr", 0))
            if tqg_min_t1_rr > 0:
                t1_rr_val = plan["targets"][0].get("rr", 0)
                if t1_rr_val < tqg_min_t1_rr:
                    plan["eligible"] = False
                    plan["quality"]["rejection_reason"] = f"T1_rr {t1_rr_val:.2f} < {tqg_min_t1_rr} (trade_quality_gate)"
                    logger.debug(f"Rejected: T1_rr {t1_rr_val:.2f} < {tqg_min_t1_rr} (trade_quality_gate)")

        return plan
