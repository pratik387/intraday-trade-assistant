"""
Capital Management Module

Supports 3 modes:
1. DISABLED (current): Unlimited capital, all trades executed (strategy development)
2. ENABLED + NO MIS: Rs.1 lakh capital, 1x leverage (delivery/conservative)
3. ENABLED + MIS: Rs.1 lakh capital, 5-20x leverage (realistic intraday)

Usage:
    # Mode 1: Unlimited (current behavior)
    capital_mgr = CapitalManager(enabled=False)

    # Mode 2: Rs.1L, no leverage
    capital_mgr = CapitalManager(enabled=True, initial_capital=100000, mis_enabled=False)

    # Mode 3: Rs.1L + MIS leverage
    capital_mgr = CapitalManager(enabled=True, initial_capital=100000, mis_enabled=True)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Tuple
from datetime import datetime

try:
    from config.logging_config import get_agent_logger
    logger = get_agent_logger()
    if logger is None:
        raise ValueError("get_agent_logger returned None")
except Exception:
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        logger.addHandler(handler)


class CapitalManager:
    """
    Manages capital allocation, margin requirements, and position limits.

    Three operational modes:
    - DISABLED: No capital tracking (unlimited capital)
    - ENABLED (No MIS): Capital tracking with 1x leverage
    - ENABLED (MIS): Capital tracking with MIS leverage (5-20x)
    """

    def __init__(
        self,
        enabled: bool = False,
        initial_capital: float = 100000.0,
        mis_enabled: bool = False,
        mis_config_path: Optional[str] = None,
        max_positions: int = 25,
        capital_utilization: float = 0.85,
        max_allocation_per_trade: float = 0.20,
        risk_pct_per_trade: float = 0.01
    ):
        """
        Initialize CapitalManager.

        Args:
            enabled: If False, unlimited capital (all trades allowed)
            initial_capital: Starting capital in Rs.
            mis_enabled: If True, use MIS leverage from config
            mis_config_path: Path to MIS margins config (default: config/mis_margins.json)
            max_positions: Maximum concurrent positions
            capital_utilization: Safety buffer on available capital (0.85 = use 85% of available)
            max_allocation_per_trade: Max % of total capital per trade (0.20 = 20% per trade)
                                      Set to 0.20 for 5 concurrent trades, 0.25 for 4, etc.
            risk_pct_per_trade: Risk per trade as % of total capital (0.01 = 1%)
                               Used for dynamic position sizing based on capital
        """
        self.enabled = enabled
        self.total_capital = initial_capital
        self.available_capital = initial_capital
        self.mis_enabled = mis_enabled
        self.max_positions = max_positions
        self.capital_utilization = max(0.5, min(1.0, capital_utilization))  # Clamp to [0.5, 1.0]
        self.max_allocation_per_trade = max(0.05, min(1.0, max_allocation_per_trade))  # Clamp to [5%, 100%]
        self.risk_pct_per_trade = max(0.001, min(0.05, risk_pct_per_trade))  # Clamp to [0.1%, 5%]

        # Position tracking
        self.positions: Dict[str, Dict] = {}  # symbol -> position info

        # Statistics
        self.stats = {
            'trades_attempted': 0,
            'trades_accepted': 0,
            'trades_rejected_capital': 0,
            'trades_rejected_positions': 0,
            'trades_shadow': 0,  # Shadow trades (tracked but no capital)
            'max_concurrent_positions': 0,
            'max_capital_used': 0.0,
            'total_margin_used_sum': 0.0,  # For averaging
            'capital_checks': 0
        }

        # MIS data now comes from nse_all.json (mis_leverage field)
        # No separate config file needed
        self.mis_config = None

        # Log mode
        risk_rupees = self.total_capital * self.risk_pct_per_trade
        if not enabled:
            logger.info("CapitalManager: DISABLED | Mode: Unlimited capital (all trades allowed)")
        elif mis_enabled:
            logger.info(f"CapitalManager: ENABLED + MIS | Capital: Rs.{initial_capital:,} | "
                       f"Risk: {self.risk_pct_per_trade*100:.1f}% (Rs.{risk_rupees:,.0f}) | "
                       f"Max positions: {max_positions}")
        else:
            logger.info(f"CapitalManager: ENABLED (No MIS) | Capital: Rs.{initial_capital:,} | "
                       f"Risk: {self.risk_pct_per_trade*100:.1f}% (Rs.{risk_rupees:,.0f}) | "
                       f"Max positions: {max_positions}")

    def get_risk_per_trade(self, fallback: float = 1000.0) -> float:
        """
        Get risk per trade in Rupees.

        For live/paper trading (enabled=True): Returns capital * risk_pct_per_trade
        For backtests (enabled=False): Returns fallback value from config

        Args:
            fallback: Value to use when capital manager is disabled (backtest mode)

        Returns:
            Risk per trade in Rupees
        """
        if not self.enabled:
            return fallback
        return self.total_capital * self.risk_pct_per_trade

    def _load_mis_config(self, config_path: Optional[str] = None) -> Dict:
        """Load MIS margin configuration."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "mis_margins.json"
        else:
            config_path = Path(config_path)

        try:
            with config_path.open() as f:
                cfg = json.load(f)
            logger.info(f"MIS config loaded from {config_path}")
            return cfg
        except Exception as e:
            logger.error(f"Failed to load MIS config from {config_path}: {e}")
            logger.warning("Falling back to 5x default leverage")
            return {'leverage_by_cap_segment': {'large_cap': {'leverage': 5.0}, 'mid_cap': {'leverage': 5.0}, 'small_cap': {'leverage': 5.0}, 'micro_cap': {'leverage': 5.0}}}

    def _get_leverage(self, symbol: str, cap_segment: str, mis_leverage: Optional[float] = None) -> float:
        """
        Get leverage for a symbol.

        Args:
            symbol: Stock symbol
            cap_segment: Market cap segment
            mis_leverage: Direct MIS leverage from stock data (preferred if provided)

        Returns:
            1.0 if MIS disabled (full capital required)
            mis_leverage if provided (from nse_all.json)
            5-20x if MIS enabled (from config fallback)
        """
        if not self.mis_enabled:
            return 1.0

        # PRIORITY 1: Use direct leverage from stock data (nse_all.json)
        if mis_leverage is not None and mis_leverage > 0:
            return mis_leverage

        # PRIORITY 2: Use old config file (for backward compatibility)
        if self.mis_config:
            # Check stock-specific overrides first
            overrides = self.mis_config.get('stock_specific_overrides', {})
            bare_symbol = symbol.replace('.NS', '').replace('.BO', '')
            if bare_symbol in overrides:
                return overrides[bare_symbol].get('leverage', 5.0)

            # Use cap-segment based leverage
            segment_cfg = self.mis_config.get('leverage_by_cap_segment', {})
            cap_cfg = segment_cfg.get(cap_segment, {})
            return cap_cfg.get('leverage', 5.0)

        # PRIORITY 3: Default fallback - use conservative 5x for MIS
        # Professional algo traders use fixed estimates; actual Zerodha margin may vary
        # but 5x is a safe conservative estimate for most liquid stocks
        return 5.0 if self.mis_enabled else 1.0

    def is_at_capacity(self) -> bool:
        """
        Check if we're at maximum position capacity.

        Used for shadow trade decision - if at capacity, new trades become shadow trades.
        """
        if not self.enabled:
            return False  # Unlimited capacity when disabled
        return len(self.positions) >= self.max_positions

    def can_enter_position(
        self,
        symbol: str,
        qty: int,
        price: float,
        cap_segment: str = "unknown",
        mis_leverage: Optional[float] = None,
        shadow: bool = False
    ) -> Tuple[bool, int, str]:
        """
        Check if we can enter a new position and return adjusted quantity if needed.

        Args:
            symbol: Stock symbol
            qty: Requested quantity
            price: Entry price
            cap_segment: Market cap segment (for MIS leverage lookup)
            mis_leverage: Direct MIS leverage from stock data (from nse_all.json)
            shadow: If True, this is a shadow trade (no capital consumed)

        Returns:
            (can_enter: bool, adjusted_qty: int, reason: str)
            - If capital is sufficient: (True, original_qty, reason)
            - If capital is insufficient: (True, scaled_down_qty, reason)
            - If position limit reached: (False, 0, reason)
            - If shadow trade: (True, original_qty, "shadow_trade")
        """
        self.stats['trades_attempted'] += 1

        # Shadow trades always allowed - they don't consume capital
        if shadow:
            self.stats['trades_shadow'] += 1
            logger.info(f"CAP_SHADOW | {symbol} | Shadow trade (no capital) | Qty: {qty}")
            return True, qty, "shadow_trade"

        # MODE 1: DISABLED - Always allow (unlimited capital)
        if not self.enabled:
            self.stats['trades_accepted'] += 1
            leverage = self._get_leverage(symbol, cap_segment, mis_leverage) if self.mis_enabled else 1.0
            margin = (qty * price) / leverage
            logger.debug(f"CAP_DISABLED | {symbol} | Would need Rs.{margin:,.0f} @ {leverage}x | ALLOWED (unlimited)")
            return True, qty, "unlimited_capital"

        # Check 1: Position limit? (check BEFORE capital scaling)
        if len(self.positions) >= self.max_positions:
            self.stats['trades_rejected_positions'] += 1
            reason = f"max_positions_{len(self.positions)}/{self.max_positions}"
            logger.warning(f"CAP_REJECT | {symbol} | {reason}")
            return False, 0, reason

        # MODE 2 & 3: ENABLED - Check capital constraints and scale if needed
        leverage = self._get_leverage(symbol, cap_segment, mis_leverage)
        notional = qty * price
        margin_required = notional / leverage

        # Check 2: Per-trade allocation limit (e.g., 20% of total capital per trade)
        max_margin_per_trade = self.total_capital * self.max_allocation_per_trade
        if margin_required > max_margin_per_trade:
            # Scale down to fit per-trade limit
            max_notional = max_margin_per_trade * leverage
            adjusted_qty = int(max_notional / price)

            if adjusted_qty < 1:
                self.stats['trades_rejected_capital'] += 1
                reason = f"per_trade_limit_exceeded_need_{margin_required:.0f}_max_{max_margin_per_trade:.0f}"
                logger.warning(f"CAP_REJECT | {symbol} | {reason}")
                return False, 0, reason

            # Update margin for subsequent checks
            margin_required = (adjusted_qty * price) / leverage
            notional = adjusted_qty * price
            qty = adjusted_qty
            logger.info(
                f"CAP_LIMIT | {symbol} | Per-trade limit {int(self.max_allocation_per_trade*100)}% | "
                f"Max margin Rs.{max_margin_per_trade:,.0f} | SCALED to qty={adjusted_qty}"
            )

        # Check 3: Sufficient available capital?
        if self.available_capital < margin_required:
            # Scale down quantity to fit available capital with safety buffer
            usable_capital = self.available_capital * self.capital_utilization
            max_notional = usable_capital * leverage
            adjusted_qty = int(max_notional / price)

            # Reject if scaled quantity is too small (< 1)
            if adjusted_qty < 1:
                self.stats['trades_rejected_capital'] += 1
                reason = f"insufficient_capital_need_{margin_required:.0f}_have_{self.available_capital:.0f}_min_qty_not_met"
                logger.warning(f"CAP_REJECT | {symbol} | {reason} | Rejected: {self.stats['trades_rejected_capital']}")
                return False, 0, reason

            # Accept with scaled quantity
            self.stats['trades_accepted'] += 1
            adjusted_margin = (adjusted_qty * price) / leverage
            reason = f"scaled_qty_{qty}→{adjusted_qty}_margin_{adjusted_margin:.0f}@{leverage}x_avail_{int(self.capital_utilization*100)}%"
            logger.info(
                f"CAP_SCALE | {symbol} | Requested qty={qty} margin={margin_required:.0f} | "
                f"Available={self.available_capital:.0f} ({int(self.capital_utilization*100)}% buffer) | "
                f"SCALED to qty={adjusted_qty} margin={adjusted_margin:.0f}"
            )
            return True, adjusted_qty, reason

        # All checks passed - use (possibly already scaled) quantity
        self.stats['trades_accepted'] += 1
        return True, qty, f"margin_{margin_required:.0f}@{leverage}x"

    def enter_position(
        self,
        symbol: str,
        qty: int,
        price: float,
        cap_segment: str = "unknown",
        timestamp: Optional[datetime] = None,
        mis_leverage: Optional[float] = None,
        shadow: bool = False
    ) -> None:
        """
        Record a new position and allocate capital.

        Args:
            symbol: Stock symbol
            qty: Quantity
            price: Entry price
            cap_segment: Market cap segment
            timestamp: Entry timestamp
            mis_leverage: Direct MIS leverage from stock data (from nse_all.json)
            shadow: If True, this is a shadow trade (no margin allocated)
        """
        if not self.enabled:
            # No tracking in disabled mode
            return

        # Shadow trades: skip margin allocation entirely
        if shadow:
            logger.info(f"CAP_SHADOW_ENTRY | {symbol} | Shadow trade - no margin allocated | Qty: {qty} @ Rs.{price:.2f}")
            return

        leverage = self._get_leverage(symbol, cap_segment, mis_leverage)
        notional = qty * price
        margin_used = notional / leverage

        self.positions[symbol] = {
            'qty': qty,
            'price': price,
            'margin_used': margin_used,
            'leverage': leverage,
            'cap_segment': cap_segment,
            'notional': notional,
            'entry_time': timestamp or datetime.now()
        }

        self.available_capital -= margin_used

        # Update stats
        self.stats['max_concurrent_positions'] = max(self.stats['max_concurrent_positions'], len(self.positions))
        margin_used_total = sum(p['margin_used'] for p in self.positions.values())
        self.stats['max_capital_used'] = max(self.stats['max_capital_used'], margin_used_total)
        self.stats['total_margin_used_sum'] += margin_used
        self.stats['capital_checks'] += 1

        logger.info(f"CAP_ENTRY | {symbol} | Qty: {qty} @ Rs.{price:.2f} | "
                    f"Margin: Rs.{margin_used:,.0f} ({leverage}x) | "
                    f"Available: Rs.{self.available_capital:,.0f}/{self.total_capital:,.0f} | "
                    f"Positions: {len(self.positions)}/{self.max_positions}")

    def exit_position(self, symbol: str, shadow: bool = False) -> None:
        """
        Release capital when a position is closed.

        Args:
            symbol: Stock symbol
            shadow: If True, this is a shadow trade (no margin to release)
        """
        if not self.enabled:
            return

        # Shadow trades: no margin was allocated, nothing to release
        if shadow:
            logger.info(f"CAP_SHADOW_EXIT | {symbol} | Shadow trade - no margin to release")
            return

        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        margin_freed = pos['margin_used']

        self.available_capital += margin_freed
        del self.positions[symbol]

        logger.info(f"CAP_EXIT | {symbol} | Margin freed: Rs.{margin_freed:,.0f} | "
                    f"Available: Rs.{self.available_capital:,.0f}/{self.total_capital:,.0f} | "
                    f"Positions: {len(self.positions)}")

    def get_stats(self) -> Dict:
        """Get current capital and position statistics."""
        margin_used = sum(p['margin_used'] for p in self.positions.values())
        total_exposure = sum(p['notional'] for p in self.positions.values())

        return {
            'enabled': self.enabled,
            'mis_enabled': self.mis_enabled,
            'total_capital': self.total_capital,
            'available_capital': self.available_capital,
            'margin_used': margin_used,
            'capital_utilization_pct': (margin_used / self.total_capital * 100) if self.total_capital > 0 else 0,
            'positions_count': len(self.positions),
            'max_positions': self.max_positions,
            'total_exposure': total_exposure,
            'leverage_ratio': total_exposure / self.total_capital if self.total_capital > 0 else 0,
            'trades_attempted': self.stats['trades_attempted'],
            'trades_accepted': self.stats['trades_accepted'],
            'trades_rejected_capital': self.stats['trades_rejected_capital'],
            'trades_rejected_positions': self.stats['trades_rejected_positions'],
            'trades_shadow': self.stats['trades_shadow'],
            'acceptance_rate_pct': (self.stats['trades_accepted'] / self.stats['trades_attempted'] * 100) if self.stats['trades_attempted'] > 0 else 100
        }

    def get_final_report(self) -> Dict:
        """Generate final report for end-of-session analytics."""
        stats = self.get_stats()

        # Calculate averages
        avg_margin = (self.stats['total_margin_used_sum'] / self.stats['capital_checks']) if self.stats['capital_checks'] > 0 else 0

        return {
            'mode': 'unlimited' if not self.enabled else ('mis' if self.mis_enabled else 'no_leverage'),
            'capital_stats': {
                'initial_capital': self.total_capital,
                'final_available': self.available_capital,
                'max_capital_used': self.stats['max_capital_used'],
                'avg_margin_per_trade': avg_margin,
                'max_utilization_pct': (self.stats['max_capital_used'] / self.total_capital * 100) if self.total_capital > 0 else 0
            },
            'trade_stats': {
                'trades_attempted': self.stats['trades_attempted'],
                'trades_accepted': self.stats['trades_accepted'],
                'trades_shadow': self.stats['trades_shadow'],
                'trades_rejected': {
                    'capital': self.stats['trades_rejected_capital'],
                    'positions': self.stats['trades_rejected_positions'],
                    'total': self.stats['trades_rejected_capital'] + self.stats['trades_rejected_positions']
                },
                'acceptance_rate_pct': stats['acceptance_rate_pct']
            },
            'position_stats': {
                'max_concurrent': self.stats['max_concurrent_positions'],
                'max_positions_limit': self.max_positions
            }
        }
