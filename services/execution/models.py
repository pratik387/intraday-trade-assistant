# services/execution/models.py
"""Shared data models for the execution layer."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Position:
    symbol: str
    side: str                 # "BUY" or "SELL"
    qty: int
    avg_price: float
    plan: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskState:
    max_concurrent: int
    open_positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    gross_exposure_rupees: float = 0.0
    sector_exposure: Dict[str, float] = field(default_factory=dict)
    correlation_buckets: Dict[str, List[str]] = field(default_factory=dict)

    def can_open_more(self) -> bool:
        """Check if we can open more positions based on risk limits"""
        return len(self.open_positions) < self.max_concurrent

    def get_position_count(self) -> int:
        """Get current number of open positions"""
        return len(self.open_positions)

    def get_symbol_exposure(self, symbol: str) -> float:
        """Get current exposure for a specific symbol"""
        pos = self.open_positions.get(symbol)
        if pos:
            return pos.get("qty", 0) * pos.get("avg_price", 0)
        return 0.0

    def update_position(self, symbol: str, side: str, qty: int, avg_price: float) -> None:
        """Update position in risk state"""
        self.open_positions[symbol] = {
            "side": side,
            "qty": qty,
            "avg_price": avg_price
        }
        # Recalculate gross exposure
        self.gross_exposure_rupees = sum(
            pos["qty"] * pos["avg_price"]
            for pos in self.open_positions.values()
        )
