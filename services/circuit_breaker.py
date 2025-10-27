# services/circuit_breaker.py
"""
Circuit Breaker - Emergency stop mechanism for trading system.

Triggers on:
- Consecutive losses (10+)
- Max drawdown (50%)
- Trade rejection rate (>80%)
- API error rate (>50%)

Usage:
    breaker = CircuitBreaker()

    # Check before each trade
    if breaker.is_tripped():
        logger.critical("Circuit breaker tripped - stopping trading")
        return

    # Update with trade results
    breaker.record_trade(pnl=500.0, rejected=False)

    # Update with errors
    breaker.record_error()
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker thresholds"""
    max_consecutive_losses: int = 10
    max_drawdown_pct: float = 50.0
    max_rejection_rate_pct: float = 80.0
    max_error_rate_pct: float = 50.0
    lookback_minutes: int = 60
    cooldown_minutes: int = 30


class CircuitBreaker:
    """
    Emergency stop mechanism for trading system.

    Monitors trading activity and halts execution if dangerous patterns are detected.
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()

        # Trade tracking
        self._trade_history: List[Tuple[datetime, float, bool]] = []  # (timestamp, pnl, rejected)
        self._consecutive_losses = 0
        self._peak_capital = 0.0
        self._current_capital = 0.0

        # Error tracking
        self._error_history: List[Tuple[datetime, str]] = []  # (timestamp, error_type)

        # Circuit breaker state
        self._tripped = False
        self._trip_reason: Optional[str] = None
        self._trip_time: Optional[datetime] = None

        logger.info(f"CircuitBreaker initialized: max_losses={self.config.max_consecutive_losses} "
                   f"max_drawdown={self.config.max_drawdown_pct}% "
                   f"cooldown={self.config.cooldown_minutes}min")

    def is_tripped(self) -> bool:
        """Check if circuit breaker is currently tripped"""
        if not self._tripped:
            return False

        # Check if cooldown period has passed
        if self._trip_time:
            elapsed = (datetime.now() - self._trip_time).total_seconds() / 60
            if elapsed >= self.config.cooldown_minutes:
                logger.warning(f"Circuit breaker cooldown complete ({elapsed:.1f}min) - RESETTING")
                self._reset()
                return False

        return True

    def get_status(self) -> dict:
        """Get current circuit breaker status"""
        return {
            'tripped': self._tripped,
            'trip_reason': self._trip_reason,
            'trip_time': str(self._trip_time) if self._trip_time else None,
            'consecutive_losses': self._consecutive_losses,
            'current_drawdown_pct': self._get_current_drawdown_pct(),
            'rejection_rate_pct': self._get_rejection_rate(),
            'error_rate_pct': self._get_error_rate(),
            'total_trades': len(self._trade_history),
            'total_errors': len(self._error_history)
        }

    def record_trade(self, pnl: float, rejected: bool = False):
        """Record a trade result and check thresholds"""
        now = datetime.now()
        self._trade_history.append((now, pnl, rejected))

        # Update capital tracking
        if not rejected:
            self._current_capital += pnl
            self._peak_capital = max(self._peak_capital, self._current_capital)

            # Update consecutive losses
            if pnl < 0:
                self._consecutive_losses += 1
            else:
                self._consecutive_losses = 0

        # Check thresholds
        self._check_thresholds()

    def record_error(self, error_type: str = "unknown"):
        """Record an error (API failure, exception, etc.)"""
        now = datetime.now()
        self._error_history.append((now, error_type))

        # Check thresholds
        self._check_thresholds()

    def _check_thresholds(self):
        """Check if any circuit breaker threshold is exceeded"""
        if self._tripped:
            return  # Already tripped

        # 1. Consecutive losses
        if self._consecutive_losses >= self.config.max_consecutive_losses:
            self._trip(f"Consecutive losses: {self._consecutive_losses}")
            return

        # 2. Max drawdown
        drawdown = self._get_current_drawdown_pct()
        if drawdown >= self.config.max_drawdown_pct:
            self._trip(f"Max drawdown: {drawdown:.1f}%")
            return

        # 3. Rejection rate (last hour)
        rejection_rate = self._get_rejection_rate()
        if rejection_rate >= self.config.max_rejection_rate_pct:
            self._trip(f"High rejection rate: {rejection_rate:.1f}%")
            return

        # 4. Error rate (last hour)
        error_rate = self._get_error_rate()
        if error_rate >= self.config.max_error_rate_pct:
            self._trip(f"High error rate: {error_rate:.1f}%")
            return

    def _trip(self, reason: str):
        """Trip the circuit breaker"""
        self._tripped = True
        self._trip_reason = reason
        self._trip_time = datetime.now()

        logger.critical(f"🚨 CIRCUIT BREAKER TRIPPED: {reason}")
        logger.critical(f"   Consecutive losses: {self._consecutive_losses}")
        logger.critical(f"   Drawdown: {self._get_current_drawdown_pct():.1f}%")
        logger.critical(f"   Rejection rate: {self._get_rejection_rate():.1f}%")
        logger.critical(f"   Error rate: {self._get_error_rate():.1f}%")
        logger.critical(f"   System will resume after {self.config.cooldown_minutes}min cooldown")

    def _reset(self):
        """Reset circuit breaker after cooldown"""
        logger.info("Circuit breaker RESET - resuming normal operation")
        self._tripped = False
        self._trip_reason = None
        self._trip_time = None
        self._consecutive_losses = 0

    def _get_current_drawdown_pct(self) -> float:
        """Calculate current drawdown percentage"""
        if self._peak_capital <= 0:
            return 0.0

        drawdown = self._peak_capital - self._current_capital
        return (drawdown / self._peak_capital) * 100

    def _get_rejection_rate(self) -> float:
        """Calculate trade rejection rate over lookback period"""
        cutoff = datetime.now() - timedelta(minutes=self.config.lookback_minutes)
        recent = [(ts, pnl, rej) for ts, pnl, rej in self._trade_history if ts >= cutoff]

        if len(recent) == 0:
            return 0.0

        rejected = sum(1 for _, _, rej in recent if rej)
        return (rejected / len(recent)) * 100

    def _get_error_rate(self) -> float:
        """Calculate error rate over lookback period"""
        cutoff = datetime.now() - timedelta(minutes=self.config.lookback_minutes)
        recent_errors = [ts for ts, _ in self._error_history if ts >= cutoff]

        # Calculate as percentage of expected operations
        # Assume 1 operation per second = 3600 ops/hour
        expected_ops = self.config.lookback_minutes * 60

        if expected_ops == 0:
            return 0.0

        return (len(recent_errors) / expected_ops) * 100
