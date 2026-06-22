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
from typing import Optional, Dict, Tuple, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from services.state.zerodha_mis_fetcher import ZerodhaMISFetcher

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

    Two risk modes:
    - PERCENTAGE: Risk per trade = capital * percentage (e.g., 1% of 5L = Rs.5000)
    - FIXED: Risk per trade = fixed amount in Rs. (e.g., Rs.1000 regardless of capital)
    """

    def __init__(
        self,
        enabled: bool,
        initial_capital: float,
        max_positions: int,
        min_notional_pct: float,
        capital_utilization: float,
        max_allocation_per_trade: float,
        risk_mode: str,
        risk_fixed_amount: float,
        risk_percentage: float,
        mis_enabled: bool = False,
        mis_config_path: Optional[str] = None,
        mis_fetcher: Optional["ZerodhaMISFetcher"] = None,
        # Legacy parameter for backwards compatibility
        risk_pct_per_trade: Optional[float] = None,
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
            risk_mode: "percentage" or "fixed" - determines how risk per trade is calculated
            risk_fixed_amount: Risk per trade in Rs. when mode="fixed"
            risk_percentage: Risk per trade as % of capital when mode="percentage" (0.01 = 1%)
            risk_pct_per_trade: DEPRECATED - use risk_percentage instead (kept for backwards compatibility)
            min_notional_pct: Minimum position size as % of capital (0.04 = 4%)
                             Trades below this become shadow trades (tracked but not executed)
        """
        self.enabled = enabled
        self.total_capital = initial_capital
        self.available_capital = initial_capital

        # Per-setup capital budgets (2026-05-12 architectural refactor). Each
        # setup gets a percentage share of total_capital. Tracking lets us
        # reject new entries when a setup hits its cap, preventing one setup
        # (e.g. gap_fade morning storm) from starving others (e.g. earnings_day
        # at 10:30). Pipeline init wires this from setups_cfg.
        self.setup_budgets_pct: Dict[str, float] = {}
        self.setup_budget_used: Dict[str, float] = {}
        # symbol -> setup_type that owns the position (for reduce_position to know
        # which bucket to decrement when broker callback fires).
        self._position_setup: Dict[str, str] = {}

        self.mis_enabled = mis_enabled
        self.max_positions = max_positions
        self.capital_utilization = max(0.5, min(1.0, capital_utilization))  # Clamp to [0.5, 1.0]
        self.max_allocation_per_trade = max(0.05, min(1.0, max_allocation_per_trade))  # Clamp to [5%, 100%]
        self.min_notional_pct = max(0.0, min(0.20, min_notional_pct))  # Clamp to [0%, 20%]

        # Risk mode configuration
        self.risk_mode = risk_mode.lower() if risk_mode else "percentage"
        self.risk_fixed_amount = max(100.0, risk_fixed_amount)  # Min Rs.100
        # Handle legacy parameter
        if risk_pct_per_trade is not None:
            self.risk_percentage = max(0.001, min(0.05, risk_pct_per_trade))
        else:
            self.risk_percentage = max(0.001, min(0.05, risk_percentage))

        # Position tracking
        self.positions: Dict[str, Dict] = {}  # symbol -> position info

        # Statistics
        self.stats = {
            'trades_attempted': 0,
            'trades_accepted': 0,
            'trades_rejected_capital': 0,
            'trades_rejected_positions': 0,
            'trades_rejected_mis': 0,  # Rejected: stock not MIS-eligible
            'trades_shadow': 0,  # Shadow trades (tracked but no capital)
            'max_concurrent_positions': 0,
            'max_capital_used': 0.0,
            'total_margin_used_sum': 0.0,  # For averaging
            'capital_checks': 0
        }

        # MIS data now comes from nse_all.json (mis_leverage field)
        # No separate config file needed
        self.mis_config = None

        # MIS fetcher for paper trading validation (optional)
        # When set, validates stocks against Zerodha's live MIS list
        self.mis_fetcher = mis_fetcher

        # Log mode with risk configuration
        risk_rupees = self._calculate_risk_amount()
        risk_desc = self._get_risk_description()
        if not enabled:
            logger.info("CapitalManager: DISABLED | Mode: Unlimited capital (all trades allowed)")
        elif mis_enabled:
            logger.info(f"CapitalManager: ENABLED + MIS | Capital: Rs.{initial_capital:,} | "
                       f"Risk: {risk_desc} (Rs.{risk_rupees:,.0f}) | "
                       f"Max positions: {max_positions}")
        else:
            logger.info(f"CapitalManager: ENABLED (No MIS) | Capital: Rs.{initial_capital:,} | "
                       f"Risk: {risk_desc} (Rs.{risk_rupees:,.0f}) | "
                       f"Max positions: {max_positions}")

    def _calculate_risk_amount(self) -> float:
        """Calculate risk per trade based on current mode."""
        if self.risk_mode == "fixed":
            return self.risk_fixed_amount
        else:  # percentage mode
            return self.total_capital * self.risk_percentage

    def _get_risk_description(self) -> str:
        """Get human-readable description of current risk mode."""
        if self.risk_mode == "fixed":
            return f"Fixed Rs.{self.risk_fixed_amount:,.0f}"
        else:
            return f"{self.risk_percentage*100:.1f}%"

    def get_risk_per_trade(self, fallback: float = 1000.0) -> float:
        """
        Get risk per trade in Rupees.

        For live/paper trading (enabled=True):
          - If risk_mode="percentage": Returns capital * risk_percentage
          - If risk_mode="fixed": Returns risk_fixed_amount
        For backtests (enabled=False): Returns fallback value from config

        Args:
            fallback: Value to use when capital manager is disabled (backtest mode)

        Returns:
            Risk per trade in Rupees
        """
        if not self.enabled:
            return fallback
        return self._calculate_risk_amount()

    def get_min_notional(self) -> float:
        """
        Get minimum notional (position size) in Rupees.

        Returns 0 when capital manager is disabled (backtest mode).
        When enabled, returns min_notional_pct * total_capital.

        Trades below this threshold become shadow trades.

        Returns:
            Minimum notional in Rupees (e.g., 20000 for 4% of 5L capital)
        """
        if not self.enabled:
            return 0.0  # No min notional filter when disabled
        return self.total_capital * self.min_notional_pct

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

    def is_mis_allowed(self, symbol: str, side: str = "BUY") -> Tuple[bool, str]:  # noqa: ARG002
        """
        Check if a symbol is allowed for MIS trading.

        Only validates when mis_fetcher is set (paper trading mode).
        Live trading doesn't use this - Zerodha broker handles rejection.

        Matching live behavior (no CNC fallback):
        - Non-MIS stock: BLOCKED for both BUY and SELL
        - This keeps backtest/paper/live results consistent

        Args:
            symbol: Stock symbol (any format: RELIANCE, RELIANCE.NS, NSE:RELIANCE)
            side: "BUY" or "SELL"

        Returns:
            (is_allowed: bool, reason: str)
        """
        # No validation if MIS disabled or fetcher not set
        if not self.mis_enabled or self.mis_fetcher is None:
            return True, "ok"

        # Check against Zerodha's MIS list
        if not self.mis_fetcher.is_mis_allowed(symbol):
            # Block all trades on non-MIS stocks (no CNC fallback)
            return False, f"mis_blocked:{symbol}"

        return True, "ok"

    def can_enter_position(
        self,
        symbol: str,
        qty: int,
        price: float,
        cap_segment: str = "unknown",
        mis_leverage: Optional[float] = None,
        shadow: bool = False,
        side: str = "BUY",
        setup_type: Optional[str] = None,
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
            side: "BUY" or "SELL" - used for MIS validation (SHORT blocked on non-MIS stocks)

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

        # MIS eligibility check (paper trading only - mis_fetcher only set for paper mode)
        # Non-MIS stocks are blocked entirely (no CNC fallback)
        is_allowed, mis_reason = self.is_mis_allowed(symbol, side)
        if not is_allowed:
            self.stats['trades_rejected_mis'] += 1
            logger.warning(f"MIS_REJECT | {symbol} | {mis_reason} | Stock not allowed for intraday trading")
            return False, 0, mis_reason

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
        shadow: bool = False,
        setup_type: Optional[str] = None,
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

        if margin_used > self.available_capital and not shadow:
            logger.error(
                f"CAP_OVERFLOW | {symbol} | margin_used={margin_used:,.0f} > "
                f"available={self.available_capital:,.0f} — blocking entry"
            )
            self.positions.pop(symbol, None)
            return

        self.available_capital -= margin_used

        # Track per-setup budget usage (2026-05-12 refactor).
        if setup_type and setup_type in self.setup_budgets_pct:
            self.setup_budget_used[setup_type] = (
                self.setup_budget_used.get(setup_type, 0.0) + margin_used
            )
            self._position_setup[symbol] = setup_type

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

    def reduce_position(self, symbol: str, qty_exited: int, new_qty: int) -> None:
        """
        Release partial margin when position is reduced (T1 partial exit).

        Args:
            symbol: Stock symbol
            qty_exited: Quantity being exited
            new_qty: Remaining quantity after exit
        """
        if not self.enabled:
            return

        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        original_qty = pos.get('qty', 0)
        if original_qty <= 0:
            return

        # Calculate proportional margin release
        exit_fraction = qty_exited / original_qty
        margin_freed = pos['margin_used'] * exit_fraction

        # Update position tracking
        pos['margin_used'] -= margin_freed
        pos['qty'] = new_qty
        pos['notional'] = pos['notional'] * (new_qty / original_qty)

        self.available_capital += margin_freed

        # Decrement per-setup budget (2026-05-12 refactor). Uses the
        # symbol→setup_type map established at enter_position time.
        setup_type = self._position_setup.get(symbol)
        if setup_type and setup_type in self.setup_budgets_pct:
            self.setup_budget_used[setup_type] = max(
                0.0, self.setup_budget_used.get(setup_type, 0.0) - margin_freed
            )
            # Clean up map on full exit (new_qty == 0)
            if new_qty == 0:
                self._position_setup.pop(symbol, None)

        logger.info(f"CAP_REDUCE | {symbol} | Qty: {original_qty} -> {new_qty} ({qty_exited} exited) | "
                    f"Margin freed: Rs.{margin_freed:,.0f} | "
                    f"Available: Rs.{self.available_capital:,.0f}/{self.total_capital:,.0f}")

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

    def save_final_report(self, log_dir: Path) -> Optional[Path]:
        """Save final report to JSON file in log directory.

        Args:
            log_dir: Directory to save the report

        Returns:
            Path to saved file, or None if save failed
        """
        try:
            report = self.get_final_report()
            file_path = log_dir / "capital_report.json"
            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"CAPITAL | Saved final report to {file_path}")
            return file_path
        except Exception as e:
            logger.warning(f"CAPITAL | Failed to save final report: {e}")
            return None


# ============================================================================
# Overnight Slot Pool (close_dn_overnight_long setup)
# ----------------------------------------------------------------------------
# The overnight setup runs as a cron-triggered short-lived script (NOT a
# long-lived daemon). Slot state must therefore live in a JSON file so each
# cron invocation can load -> mutate -> persist atomically. This block is
# deliberately decoupled from CapitalManager (which tracks MIS intraday budget)
# to avoid entanglement between two unrelated capital cycles.
# ============================================================================

from dataclasses import dataclass, asdict, fields
from datetime import date


@dataclass
class OvernightSlot:
    """One slot in the overnight capital pool.

    Lifecycle: free -> t0_open (BUY filled) -> t1_settling (AMO SELL filled,
    cash pending T+2 settlement) -> free (T+2 settle morning).

    All timestamps are IST-naive ISO 8601 strings.
    All dates are ISO 8601 date strings (YYYY-MM-DD).
    """
    slot_id: int
    status: str = "free"               # 'free' | 't0_open' | 't1_settling'
    symbol: Optional[str] = None       # e.g. "NSE:RELIANCE"
    product: Optional[str] = None      # 'MTF' | 'CNC'
    leverage: float = 1.0              # 1.0 for CNC, 2.0-5.0 for MTF
    margin_inr: float = 0.0
    notional_inr: float = 0.0
    buy_fill_price: Optional[float] = None
    buy_fill_ts: Optional[str] = None        # IST-naive ISO
    buy_order_id: Optional[str] = None
    amo_sell_order_id: Optional[str] = None
    expected_exit_date: Optional[str] = None  # ISO date (next trading day)
    sell_fill_price: Optional[float] = None
    sell_fill_ts: Optional[str] = None
    realized_pnl_inr: Optional[float] = None
    fees_inr: Optional[float] = None
    interest_inr: Optional[float] = None
    reserved_today: Optional[str] = None     # ISO date when reserve() was called
    gtt_id: Optional[str] = None             # broker GTT trigger id for the catastrophe stop

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "OvernightSlot":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


class OvernightSlotPool:
    """4-slot rolling overnight capital pool with JSON-persisted state.

    Each invocation of the entry/verify-exit cron jobs:
      1. Loads state from `state_path` (or creates empty pool if absent)
      2. Mutates state via reserve / attach_buy_fill / attach_amo_sell / settle / release
      3. Persists back to `state_path` before exit

    This avoids any in-memory state machine — the file is the source of truth.
    """

    def __init__(self, state_path: Path, max_slots: int, margin_per_slot: float,
                 max_new_per_day: int):
        if max_slots <= 0:
            raise ValueError(f"max_slots must be positive, got {max_slots}")
        if margin_per_slot <= 0:
            raise ValueError(f"margin_per_slot must be positive, got {margin_per_slot}")
        if max_new_per_day <= 0:
            raise ValueError(f"max_new_per_day must be positive, got {max_new_per_day}")
        self._state_path = Path(state_path)
        self._max_slots = int(max_slots)
        self._margin_per_slot = float(margin_per_slot)
        self._max_new_per_day = int(max_new_per_day)
        self._slots: list[OvernightSlot] = self._load_or_init()

    # ---------- Persistence ----------

    def _load_or_init(self) -> list[OvernightSlot]:
        if not self._state_path.exists():
            return [OvernightSlot(slot_id=i) for i in range(1, self._max_slots + 1)]
        try:
            with open(self._state_path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"OvernightSlotPool: state file {self._state_path} is corrupt "
                f"(invalid JSON): {e}. Fix or delete the file manually."
            )
        if not isinstance(data, dict) or "slots" not in data:
            raise ValueError(
                f"OvernightSlotPool: state file {self._state_path} has unexpected "
                f"shape (expected dict with 'slots' key). Fix or delete manually."
            )
        slots = [OvernightSlot.from_dict(s) for s in data["slots"]]
        # Validate slot count vs config. EXPANDING (state < config) is safe —
        # preserve existing slots and append free slots up to max_slots.
        # SHRINKING (state > config) is NOT safe — would orphan slots that
        # may hold capital/open positions; require manual migration.
        if len(slots) > self._max_slots:
            raise ValueError(
                f"OvernightSlotPool: state file has {len(slots)} slots but config "
                f"specifies max_slots={self._max_slots} (smaller). Shrinking is unsafe "
                f"— may orphan reserved capital. Migrate state manually."
            )
        if len(slots) < self._max_slots:
            # Auto-extend: keep existing slots (with their state), append empty.
            existing_ids = {s.slot_id for s in slots}
            next_id = max(existing_ids) + 1 if existing_ids else 1
            while len(slots) < self._max_slots:
                while next_id in existing_ids:
                    next_id += 1
                slots.append(OvernightSlot(slot_id=next_id))
                existing_ids.add(next_id)
                next_id += 1
        return slots

    def _persist(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "max_slots": self._max_slots,
            "margin_per_slot_inr": self._margin_per_slot,
            "max_new_per_day": self._max_new_per_day,
            "slots": [s.to_dict() for s in self._slots],
        }
        # Write to temp + rename for atomicity (avoids partial writes on crash)
        tmp = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp.replace(self._state_path)

    # ---------- Public lifecycle ----------

    def reserve(self, symbol: str, product: str, leverage: float,
                today: date) -> Optional[OvernightSlot]:
        """Reserve a free slot for a new BUY.

        Returns the slot if (a) a free slot exists and (b) max_new_per_day cap not hit.
        Returns None if no capacity. Caller must check return value.
        Mutated state is NOT persisted automatically — caller must call .persist().
        """
        if self.new_today_count(today) >= self._max_new_per_day:
            return None
        for slot in self._slots:
            if slot.status == "free":
                slot.status = "t0_open"
                slot.symbol = symbol
                slot.product = product
                slot.leverage = float(leverage)
                slot.margin_inr = self._margin_per_slot
                slot.notional_inr = self._margin_per_slot * float(leverage)
                slot.reserved_today = today.isoformat()
                return slot
        return None

    def attach_buy_fill(self, slot_id: int, fill_price: float,
                        fill_ts_iso: str, order_id: str) -> None:
        slot = self._get_slot(slot_id)
        if slot.status != "t0_open":
            raise ValueError(
                f"attach_buy_fill: slot {slot_id} status is {slot.status!r}, "
                f"expected 't0_open'"
            )
        slot.buy_fill_price = float(fill_price)
        slot.buy_fill_ts = fill_ts_iso
        slot.buy_order_id = order_id

    def attach_amo_sell(self, slot_id: int, amo_order_id: str,
                        expected_exit_date: date) -> None:
        slot = self._get_slot(slot_id)
        if slot.status != "t0_open":
            raise ValueError(
                f"attach_amo_sell: slot {slot_id} status is {slot.status!r}, "
                f"expected 't0_open'"
            )
        slot.amo_sell_order_id = amo_order_id
        slot.expected_exit_date = expected_exit_date.isoformat()

    def settle(self, slot_id: int, sell_fill_price: float, sell_fill_ts_iso: str,
               fees_inr: float, interest_inr: float) -> None:
        """T+1 morning: AMO filled. Compute realized PnL, transition to t1_settling."""
        slot = self._get_slot(slot_id)
        if slot.status != "t0_open":
            raise ValueError(
                f"settle: slot {slot_id} status is {slot.status!r}, expected 't0_open'"
            )
        if slot.buy_fill_price is None or slot.notional_inr <= 0:
            raise ValueError(
                f"settle: slot {slot_id} missing buy_fill_price or notional"
            )
        # Compute qty from notional and buy fill price (avoids storing qty separately)
        qty = int(round(slot.notional_inr / slot.buy_fill_price))
        gross_pnl = (float(sell_fill_price) - slot.buy_fill_price) * qty
        net_pnl = gross_pnl - float(fees_inr) - float(interest_inr)
        slot.sell_fill_price = float(sell_fill_price)
        slot.sell_fill_ts = sell_fill_ts_iso
        slot.fees_inr = float(fees_inr)
        slot.interest_inr = float(interest_inr)
        slot.realized_pnl_inr = net_pnl
        slot.status = "t1_settling"

    def release(self, slot_id: int, cash_back_date: date) -> None:
        """T+2 morning: cash settled, slot transitions to free."""
        slot = self._get_slot(slot_id)
        if slot.status != "t1_settling":
            raise ValueError(
                f"release: slot {slot_id} status is {slot.status!r}, "
                f"expected 't1_settling'"
            )
        # Reset slot fields
        slot.status = "free"
        slot.symbol = None
        slot.product = None
        slot.leverage = 1.0
        slot.margin_inr = 0.0
        slot.notional_inr = 0.0
        slot.buy_fill_price = None
        slot.buy_fill_ts = None
        slot.buy_order_id = None
        slot.amo_sell_order_id = None
        slot.expected_exit_date = None
        slot.sell_fill_price = None
        slot.sell_fill_ts = None
        slot.realized_pnl_inr = None
        slot.fees_inr = None
        slot.interest_inr = None
        slot.reserved_today = None

    # ---------- Queries ----------

    def _get_slot(self, slot_id: int) -> OvernightSlot:
        for s in self._slots:
            if s.slot_id == slot_id:
                return s
        raise KeyError(f"slot_id {slot_id} not found")

    def active(self) -> list[OvernightSlot]:
        """All slots with status != 'free'."""
        return [s for s in self._slots if s.status != "free"]

    def free_count(self) -> int:
        return sum(1 for s in self._slots if s.status == "free")

    def settling_count(self) -> int:
        return sum(1 for s in self._slots if s.status == "t1_settling")

    def open_count(self) -> int:
        return sum(1 for s in self._slots if s.status == "t0_open")

    def new_today_count(self, today: date) -> int:
        iso = today.isoformat()
        return sum(
            1 for s in self._slots
            if s.reserved_today == iso and s.status != "free"
        )

    def overnight_capital_committed_inr(self) -> float:
        return sum(s.margin_inr for s in self._slots if s.status != "free")

    def persist(self) -> None:
        """Public persist hook — call after a batch of mutations."""
        self._persist()
