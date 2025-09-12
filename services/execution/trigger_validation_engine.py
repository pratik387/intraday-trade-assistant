# services/execution/trigger_validation_engine.py
"""Fast trigger validation engine optimized for real-time trading performance"""
from __future__ import annotations

import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional
from utils.time_util import _minute_of_day
from config.filters_setup import load_filters
from config.logging_config import get_loggers

logger, _ = get_loggers()

# ======================== CORE DATA STRUCTURES ========================

class TradeState(Enum):
    """Trade execution states for trigger-aware execution"""
    WAITING_TRIGGER = "waiting_trigger"
    TRIGGERED = "triggered"
    EXECUTED = "executed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"

class ConditionType(Enum):
    PRICE_LEVEL = "price_level"
    VWAP_RELATION = "vwap_relation"
    TECHNICAL = "technical"
    VOLUME = "volume"
    TIME_BASED = "time_based"

@dataclass
class TriggerCondition:
    condition_type: ConditionType
    params: Dict[str, Any]
    description: str
    last_result: bool = False
    last_check_ts: Optional[pd.Timestamp] = None
    consecutive_hits: int = 0
    required_consecutive: int = 1

# ======================== FAST TRIGGER FACTORY ========================

class FastTriggerConditionFactory:
    """Fast factory for simple trigger conditions optimized for speed"""
    
    def __init__(self):
        self.cfg = load_filters()
    
    def create_conditions_for_strategy(self, plan: Dict[str, Any]) -> Tuple[List[TriggerCondition], List[TriggerCondition], List[TriggerCondition]]:
        """Create fast trigger conditions based on strategy"""
        strategy = plan.get("strategy", "")
        bias = plan.get("bias", "long")
        
        # All strategies use fast, simple conditions
        primary_triggers = [
            TriggerCondition(
                ConditionType.TECHNICAL,
                {"type": "immediate"},
                "Immediate execution (fast mode)",
                required_consecutive=1
            )
        ]
        
        must_conditions = [
            TriggerCondition(
                ConditionType.TECHNICAL,
                {"type": "entry_zone_check"},
                "Price in entry zone"
            ),
            TriggerCondition(
                ConditionType.TIME_BASED,
                {"type": "market_hours"},
                "Market hours check"
            )
        ]
        
        # Strategy-specific should conditions for basic filtering
        should_conditions = self._create_should_conditions(strategy, bias, plan)
        
        return primary_triggers, must_conditions, should_conditions
    
    def _create_should_conditions(self, strategy: str, bias: str, plan: Dict[str, Any]) -> List[TriggerCondition]:
        """Create basic should conditions for strategy filtering"""
        conditions = []
        
        # Basic VWAP alignment for all strategies
        vwap_direction = "above" if bias == "long" else "below"
        conditions.append(
            TriggerCondition(
                ConditionType.VWAP_RELATION,
                {"direction": vwap_direction, "tolerance_pct": 0.1},
                f"Price {vwap_direction} VWAP"
            )
        )
        
        # Basic volume check
        conditions.append(
            TriggerCondition(
                ConditionType.VOLUME,
                {"min_ratio": 1.1},
                "Above average volume"
            )
        )
        
        return conditions

# ======================== FAST CONDITION VALIDATOR ========================

class FastConditionValidator:
    """Ultra-fast validator optimized for <1ms per symbol validation"""
    
    def __init__(self, get_ltp_ts_func, bar_builder):
        self.get_ltp_ts = get_ltp_ts_func
        self.bar_builder = bar_builder
        self.cfg = load_filters()
        
        # Pre-compiled constants for speed
        self.market_open_minutes = 9 * 60 + 15  # 555
        self.market_close_minutes = 15 * 60 + 30  # 930
        
        # Volume cache for fast lookups
        self._volume_cache = {}  # symbol -> {avg_volume: float, last_update: timestamp}
        
        logger.info("FastConditionValidator initialized for production trading")
    
    def _get_current_time(self, context_symbol: str = None) -> pd.Timestamp:
        """Get current time from tick stream - works for live and backtest"""
        # Try to get time from tick stream first
        if context_symbol:
            try:
                _, ts = self.get_ltp_ts(context_symbol)
                if ts:
                    return pd.Timestamp(ts)
            except:
                pass
        
        # Try any recent symbol from bar builder
        try:
            if hasattr(self.bar_builder, '_ltp') and self.bar_builder._ltp:
                for symbol, tick_data in self.bar_builder._ltp.items():
                    if hasattr(tick_data, 'ts'):
                        return pd.Timestamp(tick_data.ts)
        except:
            pass
        
        # Fallback
        return pd.Timestamp.now()
    
    def validate_condition(self, condition: TriggerCondition, symbol: str, bar_1m: pd.Series, context: Dict) -> bool:
        """Fast validation optimized for <1ms execution"""
        try:
            if condition.condition_type == ConditionType.TECHNICAL:
                return self._validate_technical_fast(condition, symbol, bar_1m, context)
            elif condition.condition_type == ConditionType.TIME_BASED:
                return self._validate_time_fast(condition, bar_1m)
            elif condition.condition_type == ConditionType.VWAP_RELATION:
                return self._validate_vwap_fast(condition, bar_1m, context)
            elif condition.condition_type == ConditionType.VOLUME:
                return self._validate_volume_fast(condition, symbol, bar_1m, context)
            elif condition.condition_type == ConditionType.PRICE_LEVEL:
                return self._validate_price_level_fast(condition, bar_1m, context)
            else:
                return False
                
        except Exception as e:
            logger.exception(f"Fast validation error for {symbol}: {e}")
            return False
    
    def _validate_technical_fast(self, condition: TriggerCondition, symbol: str, bar_1m: pd.Series, context: Dict) -> bool:
        """Fast technical validation using pre-computed values"""
        tech_type = condition.params.get("type", "")
        
        if tech_type == "immediate":
            return True
        
        elif tech_type == "entry_zone_check":
            return self._fast_entry_zone_check(bar_1m, context)
        
        elif tech_type == "momentum_check":
            # Use pre-computed momentum proxy from bar
            momentum = bar_1m.get("momentum_proxy", 0)
            min_momentum = float(condition.params.get("min_momentum", 0))
            return float(momentum) >= min_momentum
        
        elif tech_type == "rsi_check":
            # Use pre-computed RSI proxy from bar
            rsi = bar_1m.get("rsi_proxy", 50)
            min_rsi = float(condition.params.get("min", 30))
            max_rsi = float(condition.params.get("max", 70))
            return min_rsi <= float(rsi) <= max_rsi
        
        else:
            # Unknown technical type - default to true for speed
            return True
    
    def _validate_time_fast(self, condition: TriggerCondition, bar_1m: pd.Series) -> bool:
        """Ultra-fast time validation"""
        params = condition.params
        time_type = params.get("type", "market_hours")
        
        if time_type == "market_hours":
            # Get timestamp from bar or current time
            if hasattr(bar_1m, 'name') and bar_1m.name:
                ts = pd.Timestamp(bar_1m.name)
            else:
                ts = self._get_current_time()
            
            minute_of_day = _minute_of_day(ts)
            return self.market_open_minutes <= minute_of_day <= self.market_close_minutes
        
        return True
    
    def _validate_vwap_fast(self, condition: TriggerCondition, bar_1m: pd.Series, context: Dict) -> bool:
        """Fast VWAP validation using simple comparison"""
        direction = condition.params.get("direction", "above")
        tolerance_pct = float(condition.params.get("tolerance_pct", 0.02))
        
        current_price = float(bar_1m.get("close", 0))
        
        # Get VWAP from context or bar
        current_vwap = context.get("vwap", bar_1m.get("vwap", 0))
        if not current_vwap or current_vwap <= 0:
            return True  # Skip if no VWAP available
        
        current_vwap = float(current_vwap)
        tolerance = current_vwap * (tolerance_pct / 100.0)
        
        if direction == "above":
            return current_price > (current_vwap - tolerance)
        elif direction == "below":
            return current_price < (current_vwap + tolerance)
        
        return False
    
    def _validate_volume_fast(self, condition: TriggerCondition, symbol: str, bar_1m: pd.Series, context: Dict) -> bool:
        """Fast volume validation using cached averages"""
        min_ratio = float(condition.params.get("min_ratio", 1.0))
        current_volume = float(bar_1m.get("volume", 0))
        
        if current_volume <= 0:
            return True  # Skip if no volume data
        
        # Use pre-computed volume ratio from context if available
        vol_ratio = context.get("vol_ratio")
        if vol_ratio is not None:
            return float(vol_ratio) >= min_ratio
        
        # Fast volume ratio calculation using cache
        avg_volume = self._get_cached_avg_volume(symbol, current_volume)
        if avg_volume > 0:
            ratio = current_volume / avg_volume
            return ratio >= min_ratio
        
        return True  # Default to true if no volume history
    
    def _validate_price_level_fast(self, condition: TriggerCondition, bar_1m: pd.Series, context: Dict) -> bool:
        """Fast price level validation"""
        direction = condition.params.get("direction", "above")
        level_value = float(condition.params.get("level_value", 0))
        tolerance_pct = float(condition.params.get("tolerance_pct", 0.1))
        
        if level_value <= 0:
            return True
        
        current_price = float(bar_1m.get("close", 0))
        tolerance = level_value * (tolerance_pct / 100.0)
        
        if direction == "above":
            return current_price > (level_value + tolerance)
        elif direction == "below":
            return current_price < (level_value - tolerance)
        elif direction == "near":
            return abs(current_price - level_value) <= tolerance
        
        return False
    
    def _fast_entry_zone_check(self, bar_1m: pd.Series, context: Dict) -> bool:
        """Fast entry zone validation"""
        try:
            current_price = float(bar_1m.get("close", 0))
            
            # Get entry range from context
            entry_zone = context.get("entry_zone", [])
            if len(entry_zone) == 2:
                entry_min, entry_max = float(entry_zone[0]), float(entry_zone[1])
                return entry_min <= current_price <= entry_max
            
            # Fallback: reasonable range around current price (±0.2%)
            fallback_range = current_price * 0.002
            entry_min = current_price - fallback_range
            entry_max = current_price + fallback_range
            return entry_min <= current_price <= entry_max
            
        except Exception:
            return True  # Default to true for speed
    
    def _get_cached_avg_volume(self, symbol: str, current_volume: float) -> float:
        """Get cached average volume for fast lookups"""
        current_time = self._get_current_time()
        
        # Check cache
        if symbol in self._volume_cache:
            cache_entry = self._volume_cache[symbol]
            last_update = cache_entry.get("last_update")
            
            # Use cached value if recent (within 5 minutes)
            if last_update and (current_time - last_update).total_seconds() < 300:
                return cache_entry.get("avg_volume", 0)
        
        # Calculate new average (fast approximation)
        try:
            # Get recent bars for volume average
            df_recent = self.bar_builder.get_df_1m_tail(symbol, 20)
            if len(df_recent) >= 5:
                avg_volume = float(df_recent["volume"].mean())
                
                # Cache the result
                self._volume_cache[symbol] = {
                    "avg_volume": avg_volume,
                    "last_update": current_time
                }
                
                return avg_volume
        except Exception:
            pass
        
        return 0
    
    def _get_volume_ratio(self, symbol: str, current_volume: float) -> float:
        """Get volume ratio compared to average - fast calculation"""
        try:
            avg_volume = self._get_cached_avg_volume(symbol, current_volume)
            if avg_volume > 0:
                return current_volume / avg_volume
            return 1.0  # Default ratio if no history
        except Exception:
            return 1.0
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get basic performance statistics"""
        return {
            "validator_type": "fast",
            "cached_symbols": len(self._volume_cache),
            "cache_enabled": True
        }
    
    def cleanup_stale_cache(self, max_age_minutes: int = 30) -> None:
        """Clean up stale volume cache entries"""
        current_time = self._get_current_time()
        cutoff_time = current_time - pd.Timedelta(minutes=max_age_minutes)
        
        stale_symbols = []
        for symbol, cache_entry in self._volume_cache.items():
            last_update = cache_entry.get("last_update")
            if last_update and last_update < cutoff_time:
                stale_symbols.append(symbol)
        
        for symbol in stale_symbols:
            del self._volume_cache[symbol]
        
        if stale_symbols:
            logger.info(f"Cleaned up {len(stale_symbols)} stale volume cache entries")