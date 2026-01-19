"""
Index & Sector Crossover Risk Modulator

This module implements dynamic risk modulation based on index and sector VWAP crossovers.
It does NOT generate entry/exit signals - it only adjusts risk (stop loss) for open positions.

Key Concepts:
- Index VWAP Crossover: 2 consecutive 5m closes above/below VWAP indicates institutional bias shift
- Opening Range (OR): First 15-30 min high/low used for session bias
- State Classification: STRONG_BULL, WEAK_BULL, NEUTRAL, WEAK_BEAR, STRONG_BEAR, CHOP

Usage:
    modulator = IndexSectorRiskModulator(config)
    modulator.update_bars(index_symbol, df_5m)  # Called on each 5m bar close
    multiplier = modulator.get_risk_multiplier(symbol, position_side)
    effective_sl = base_sl * multiplier
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Path to symbol-sector mapping file
SECTOR_MAP_PATH = Path(__file__).resolve().parents[2] / "assets" / "symbol_sector_map.json"


class IndexState(Enum):
    """Index/Sector state based on VWAP and OR position."""
    STRONG_BULL = "strong_bull"   # Above VWAP + holding above ORH
    WEAK_BULL = "weak_bull"       # Above VWAP + failed to hold ORH
    NEUTRAL = "neutral"           # At VWAP or insufficient data
    WEAK_BEAR = "weak_bear"       # Below VWAP + failed to hold ORL
    STRONG_BEAR = "strong_bear"   # Below VWAP + holding below ORL
    CHOP = "chop"                 # Repeated VWAP crossings (whipsaw)


@dataclass
class CrossoverState:
    """Tracks crossover state for a single index/sector."""
    symbol: str
    vwap: float = 0.0
    or_high: float = 0.0          # Opening range high
    or_low: float = 0.0           # Opening range low
    or_set: bool = False          # Whether OR has been established

    # VWAP crossover tracking
    closes_above_vwap: int = 0    # Consecutive closes above VWAP
    closes_below_vwap: int = 0    # Consecutive closes below VWAP
    vwap_crossover_count: int = 0 # Total crossovers today (for chop detection)
    last_vwap_side: str = ""      # "above" or "below"

    # Current state
    state: IndexState = IndexState.NEUTRAL
    last_update: Optional[datetime] = None

    # Session VWAP calculation
    cumulative_vwap_num: float = 0.0  # sum(price * volume)
    cumulative_vwap_den: float = 0.0  # sum(volume)

    # Session return tracking (for relative strength calculation)
    session_open: float = 0.0     # First bar open price
    session_return: float = 0.0   # (current_close - session_open) / session_open


def load_sector_map(log_info: bool = True) -> Dict[str, str]:
    """
    Load symbol-to-sector mapping from JSON file.

    Args:
        log_info: Whether to log info message on successful load

    Returns:
        Dict mapping symbol (e.g., "HDFCBANK") to sector index (e.g., "NIFTY FIN SERVICE")
    """
    if not SECTOR_MAP_PATH.exists():
        logger.warning(f"RISK_MOD | Sector map not found: {SECTOR_MAP_PATH}")
        return {}

    try:
        with open(SECTOR_MAP_PATH, "r") as f:
            data = json.load(f)
        mapping = data.get("mapping", {})
        if log_info:
            logger.info(f"RISK_MOD | Loaded {len(mapping)} symbol-sector mappings from {SECTOR_MAP_PATH.name}")
        return mapping
    except Exception as e:
        logger.error(f"RISK_MOD | Failed to load sector map: {e}")
        return {}


# Module-level cache (loaded once at import, can be refreshed via reload)
_SYMBOL_SECTOR_MAP: Dict[str, str] = {}


@dataclass
class RiskModulatorConfig:
    """
    Configuration for risk modulation.

    All values MUST be provided from configuration.json - no defaults.
    """
    enabled: bool

    # Crossover detection
    crossover_confirmation_bars: int  # Bars needed to confirm crossover
    chop_threshold_crossovers: int    # Crossovers/session to classify as CHOP

    # Opening range settings
    or_period_minutes: int            # First N minutes for OR calculation

    # Risk multipliers (applied to stop loss distance)
    # < 1.0 = tighter SL (cut losers fast), > 1.0 = wider SL (let winners run)
    # Trend-aligned trades survive noise better → give more room
    # Counter-trend trades fail faster → cut quickly
    multipliers: Dict[str, float]

    # Primary indices to track
    primary_indices: List[str]


class IndexSectorRiskModulator:
    """
    Manages index/sector state and provides risk multipliers for exit management.

    Thread-safe: All state modifications happen through update_bars() which should
    be called from the main trading loop.
    """

    def __init__(self, config: RiskModulatorConfig):
        self.config = config
        self._states: Dict[str, CrossoverState] = {}
        self._session_date: Optional[datetime] = None

        # Load sector mapping (will be reloaded at session start)
        self._symbol_sector_map: Dict[str, str] = {}
        self._load_sector_map()

        # Track individual stock session returns for relative strength
        # Key: symbol (e.g., "RELIANCE"), Value: (session_open, current_close)
        self._stock_prices: Dict[str, Tuple[float, float]] = {}

        # Initialize states for primary indices
        for idx in self.config.primary_indices:
            self._states[idx] = CrossoverState(symbol=idx)

    def _load_sector_map(self) -> None:
        """Load or reload sector mapping from JSON file."""
        global _SYMBOL_SECTOR_MAP
        _SYMBOL_SECTOR_MAP = load_sector_map(log_info=True)
        self._symbol_sector_map = _SYMBOL_SECTOR_MAP

    def reset_session(self, session_date: datetime) -> None:
        """
        Reset all states for a new trading session.

        This also reloads the sector mapping to pick up any changes
        made to the JSON file (e.g., index rebalancing).
        """
        self._session_date = session_date

        # Reload sector mapping at session start (picks up any updates)
        self._load_sector_map()

        # Reset all index states
        for symbol in list(self._states.keys()):
            self._states[symbol] = CrossoverState(symbol=symbol)

        # Reset stock price tracking
        self._stock_prices.clear()

        logger.info(f"RISK_MOD | Session reset for {session_date.date()}")

    def update_stock_price(self, symbol: str, open_price: float, current_price: float) -> None:
        """
        Update stock price for relative strength calculation.

        Call this when you receive price updates for stocks (from tick data or bar data).

        Args:
            symbol: Stock symbol (e.g., "RELIANCE" or "NSE:RELIANCE")
            open_price: Session open price (first trade of day)
            current_price: Current/latest price
        """
        # Normalize symbol
        base_symbol = symbol
        if base_symbol.startswith("NSE:"):
            base_symbol = base_symbol[4:]
        if base_symbol.endswith(".NS"):
            base_symbol = base_symbol[:-3]

        self._stock_prices[base_symbol] = (open_price, current_price)

    def _get_stock_return(self, symbol: str) -> Optional[float]:
        """Get session return for a stock."""
        base_symbol = symbol
        if base_symbol.startswith("NSE:"):
            base_symbol = base_symbol[4:]
        if base_symbol.endswith(".NS"):
            base_symbol = base_symbol[:-3]

        if base_symbol not in self._stock_prices:
            return None

        open_price, current_price = self._stock_prices[base_symbol]
        if open_price <= 0:
            return None

        return (current_price - open_price) / open_price

    def _get_nifty_return(self) -> float:
        """Get NIFTY 50 session return."""
        nifty_state = self._states.get("NSE:NIFTY 50")
        if nifty_state and nifty_state.session_open > 0:
            return nifty_state.session_return
        return 0.0

    def _calculate_relative_strength_state(
        self,
        symbol: str,
        position_side: str,
        stock_day_open: Optional[float] = None,
        stock_current_price: Optional[float] = None
    ) -> Tuple[float, str]:
        """
        Calculate risk multiplier based on stock vs NIFTY relative strength.

        This is the Tier 2 fallback when no sector mapping exists.

        RS = stock_return - nifty_return
        - RS > 0.5%: Stock outperforming → bullish bias
        - RS < -0.5%: Stock underperforming → bearish bias
        - Otherwise: Neutral

        Args:
            symbol: Stock symbol
            position_side: "BUY" or "SELL"
            stock_day_open: Stock's day open price (passed directly)
            stock_current_price: Stock's current price (passed directly)

        Returns:
            Tuple of (multiplier, reason_string)
        """
        # Calculate stock return from passed prices (preferred) or cached prices
        stock_return = None
        if stock_day_open and stock_current_price and stock_day_open > 0:
            stock_return = (stock_current_price - stock_day_open) / stock_day_open
        else:
            stock_return = self._get_stock_return(symbol)

        nifty_return = self._get_nifty_return()

        # If we don't have stock price data, return neutral
        if stock_return is None:
            return 1.0, "no_price_data"

        # Calculate relative strength
        rs = stock_return - nifty_return
        rs_pct = rs * 100  # Convert to percentage for readability

        # Thresholds (in decimal, so 0.005 = 0.5%)
        STRONG_THRESHOLD = 0.01   # 1% outperformance/underperformance
        WEAK_THRESHOLD = 0.005    # 0.5%

        is_long = position_side.upper() == "BUY"

        # Determine state based on relative strength
        if rs >= STRONG_THRESHOLD:
            # Stock strongly outperforming NIFTY → bullish
            if is_long:
                return self.config.multipliers["strong_bull_long"], f"RS:+{rs_pct:.2f}%_strong_outperform"
            else:
                return self.config.multipliers["strong_bull_short"], f"RS:+{rs_pct:.2f}%_strong_outperform"
        elif rs >= WEAK_THRESHOLD:
            # Stock weakly outperforming
            if is_long:
                return self.config.multipliers["weak_bull_long"], f"RS:+{rs_pct:.2f}%_weak_outperform"
            else:
                return self.config.multipliers["weak_bull_short"], f"RS:+{rs_pct:.2f}%_weak_outperform"
        elif rs <= -STRONG_THRESHOLD:
            # Stock strongly underperforming NIFTY → bearish
            if is_long:
                return self.config.multipliers["strong_bear_long"], f"RS:{rs_pct:.2f}%_strong_underperform"
            else:
                return self.config.multipliers["strong_bear_short"], f"RS:{rs_pct:.2f}%_strong_underperform"
        elif rs <= -WEAK_THRESHOLD:
            # Stock weakly underperforming
            if is_long:
                return self.config.multipliers["weak_bear_long"], f"RS:{rs_pct:.2f}%_weak_underperform"
            else:
                return self.config.multipliers["weak_bear_short"], f"RS:{rs_pct:.2f}%_weak_underperform"
        else:
            # Neutral - stock moving with NIFTY
            return self.config.multipliers["neutral_long"], f"RS:{rs_pct:.2f}%_neutral"

    def update_bars(self, symbol: str, df_5m: pd.DataFrame, current_time: datetime) -> None:
        """
        Update state with new 5m bar data.

        Args:
            symbol: Index/sector symbol (e.g., "NSE:NIFTY 50")
            df_5m: DataFrame with columns [open, high, low, close, volume, vwap]
            current_time: Current bar timestamp
        """
        if not self.config.enabled:
            return

        if symbol not in self._states:
            self._states[symbol] = CrossoverState(symbol=symbol)

        state = self._states[symbol]

        if df_5m.empty or len(df_5m) < 1:
            return

        # Get latest bar
        latest = df_5m.iloc[-1]
        close = float(latest["close"])
        high = float(latest["high"])
        low = float(latest["low"])
        open_price = float(latest["open"])
        volume = float(latest.get("volume", 0))

        # Track session open (first bar) and calculate session return
        if state.session_open == 0:
            state.session_open = open_price
        if state.session_open > 0:
            state.session_return = (close - state.session_open) / state.session_open

        # Calculate typical price if VWAP not available
        bar_vwap = float(latest.get("vwap", 0))
        if bar_vwap <= 0:
            # Use typical price (HLC/3) as proxy for VWAP
            bar_vwap = (high + low + close) / 3.0

        # Update session VWAP (cumulative from market open)
        # For indices with volume=0, use bar count as weight instead
        weight = volume if volume > 0 else 1.0
        state.cumulative_vwap_num += bar_vwap * weight
        state.cumulative_vwap_den += weight
        if state.cumulative_vwap_den > 0:
            state.vwap = state.cumulative_vwap_num / state.cumulative_vwap_den
        else:
            state.vwap = close

        # Establish Opening Range (first N minutes from market open at 9:15)
        if not state.or_set:
            # Calculate OR cutoff time: 9:15 + or_period_minutes
            total_minutes = 9 * 60 + 15 + self.config.or_period_minutes
            or_cutoff = time(total_minutes // 60, total_minutes % 60, 0)
            bar_time = current_time.time()

            # Update OR high/low
            if state.or_high == 0 or high > state.or_high:
                state.or_high = high
            if state.or_low == 0 or low < state.or_low:
                state.or_low = low

            # Check if OR period is complete
            if bar_time >= or_cutoff:
                state.or_set = True
                logger.debug(f"RISK_MOD | {symbol} OR set: H={state.or_high:.2f} L={state.or_low:.2f}")

        # Track VWAP crossovers
        self._update_vwap_crossover(state, close)

        # Classify state
        old_state = state.state
        state.state = self._classify_state(state, close)
        state.last_update = current_time

        # Log state CHANGES at INFO level for backtest analysis
        if state.state != old_state:
            logger.info(
                f"RISK_MOD_STATE_CHANGE | {symbol} | "
                f"{old_state.value} -> {state.state.value} | "
                f"Close={close:.2f} VWAP={state.vwap:.2f} | "
                f"Crossovers={state.vwap_crossover_count}"
            )

        # Per-bar updates at DEBUG level (too noisy for INFO)
        logger.debug(
            f"RISK_MOD | {symbol} | State={state.state.value} | "
            f"Close={close:.2f} VWAP={state.vwap:.2f} | "
            f"OR: H={state.or_high:.2f} L={state.or_low:.2f} | "
            f"Crossovers={state.vwap_crossover_count}"
        )

    def _update_vwap_crossover(self, state: CrossoverState, close: float) -> None:
        """Track VWAP crossover with confirmation bars."""
        if state.vwap <= 0:
            return

        current_side = "above" if close > state.vwap else "below"

        # Check for crossover
        if state.last_vwap_side and current_side != state.last_vwap_side:
            state.vwap_crossover_count += 1
            # Reset counters on crossover
            state.closes_above_vwap = 0
            state.closes_below_vwap = 0

        # Increment appropriate counter
        if current_side == "above":
            state.closes_above_vwap += 1
            state.closes_below_vwap = 0
        else:
            state.closes_below_vwap += 1
            state.closes_above_vwap = 0

        state.last_vwap_side = current_side

    def _classify_state(self, state: CrossoverState, close: float) -> IndexState:
        """Classify index state based on VWAP and OR position."""

        # Check for chop (too many crossovers)
        if state.vwap_crossover_count >= self.config.chop_threshold_crossovers:
            return IndexState.CHOP

        # Need confirmed VWAP position
        confirmed_above = state.closes_above_vwap >= self.config.crossover_confirmation_bars
        confirmed_below = state.closes_below_vwap >= self.config.crossover_confirmation_bars

        if not (confirmed_above or confirmed_below):
            return IndexState.NEUTRAL

        # Check OR position (if OR is set)
        if state.or_set and state.or_high > 0 and state.or_low > 0:
            above_orh = close > state.or_high
            below_orl = close < state.or_low

            if confirmed_above:
                if above_orh:
                    return IndexState.STRONG_BULL
                else:
                    return IndexState.WEAK_BULL
            elif confirmed_below:
                if below_orl:
                    return IndexState.STRONG_BEAR
                else:
                    return IndexState.WEAK_BEAR
        else:
            # OR not set yet, use VWAP only
            if confirmed_above:
                return IndexState.WEAK_BULL
            elif confirmed_below:
                return IndexState.WEAK_BEAR

        return IndexState.NEUTRAL

    def get_index_state(self, index_symbol: str) -> IndexState:
        """Get current state for an index."""
        if index_symbol in self._states:
            return self._states[index_symbol].state
        return IndexState.NEUTRAL

    def get_crossover_state(self, index_symbol: str) -> Optional[CrossoverState]:
        """Get full crossover state for an index (for debugging)."""
        return self._states.get(index_symbol)

    def _get_sector_index(self, symbol: str) -> Tuple[str, bool]:
        """
        Get the sector index for a stock symbol.

        Args:
            symbol: Trading symbol (e.g., "NSE:RELIANCE" or "RELIANCE.NS")

        Returns:
            Tuple of (index_symbol, is_fallback)
            - index_symbol: Sector index (e.g., "NSE:NIFTY FIN SERVICE") or "NSE:NIFTY 50" fallback
            - is_fallback: True if using NIFTY 50 fallback, False if using sector mapping
        """
        # Extract base symbol (remove NSE: prefix and .NS suffix)
        base_symbol = symbol
        if base_symbol.startswith("NSE:"):
            base_symbol = base_symbol[4:]
        if base_symbol.endswith(".NS"):
            base_symbol = base_symbol[:-3]

        # Look up in mapping
        sector = self._symbol_sector_map.get(base_symbol)
        if sector:
            return f"NSE:{sector}", False

        # Fallback to NIFTY 50 for unmapped stocks
        return "NSE:NIFTY 50", True

    def get_risk_multiplier(
        self,
        symbol: str,
        position_side: str,
        index_symbol: Optional[str] = None,
        _stock_day_open: Optional[float] = None,
        _stock_current_price: Optional[float] = None
    ) -> Tuple[float, str, str]:
        """
        Get risk multiplier for a position based on index/sector state.

        Uses sector index mapping with NIFTY 50 as fallback for unmapped stocks.

        Args:
            symbol: Trading symbol (e.g., "NSE:RELIANCE")
            position_side: "BUY" or "SELL"
            index_symbol: Override index to use (default: auto-detect from sector)

        Returns:
            Tuple of (multiplier, reason_string, tier)
            - multiplier: Risk multiplier to apply to stop loss distance
            - reason_string: e.g., "NSE:NIFTY METAL:chop"
            - tier: "sector" if using sector index, "nifty50_fallback" if using NIFTY 50 fallback
        """
        if not self.config.enabled:
            return 1.0, "disabled", "disabled"

        # Get sector index (falls back to NIFTY 50 for unmapped stocks)
        is_fallback = False
        if index_symbol is None:
            index_symbol, is_fallback = self._get_sector_index(symbol)

        # Use sector/index state for risk modulation
        state = self.get_index_state(index_symbol)

        # Determine multiplier key
        side_key = "long" if position_side.upper() == "BUY" else "short"
        multiplier_key = f"{state.value}_{side_key}"

        multiplier = self.config.multipliers[multiplier_key]
        reason = f"{index_symbol}:{state.value}"
        tier = "nifty50_fallback" if is_fallback else "sector"

        return multiplier, reason, tier

    def get_all_states(self) -> Dict[str, Dict]:
        """Get all index states for logging/debugging."""
        result = {}
        for symbol, state in self._states.items():
            result[symbol] = {
                "state": state.state.value,
                "vwap": state.vwap,
                "or_high": state.or_high,
                "or_low": state.or_low,
                "or_set": state.or_set,
                "vwap_crossovers": state.vwap_crossover_count,
                "closes_above_vwap": state.closes_above_vwap,
                "closes_below_vwap": state.closes_below_vwap,
                "last_update": state.last_update.isoformat() if state.last_update else None,
            }
        return result

    def format_status(self) -> str:
        """Format current status for logging."""
        parts = []
        for symbol, state in self._states.items():
            short_name = symbol.replace("NSE:", "").replace(" ", "")[:10]
            parts.append(f"{short_name}:{state.state.value[:4].upper()}")
        return " | ".join(parts) if parts else "NO_DATA"
