# @role: Base broker interface definition
# @used_by: entry_service.py, exit_service.py, kite_broker.py, mock_broker.py, suggestion_logic.py, trade_executor.py
# @filter_type: utility
# @tags: broker, abstract, interface
from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime

class BaseBroker(ABC):
    """
    Abstract base class for all brokers (e.g., KiteBroker, MockBroker).
    This interface ensures that both data access and order execution
    are handled in a consistent, broker-agnostic way.
    """

    # --- Market Data ---

    @abstractmethod
    def get_ltp(self, symbols: List[str]) -> dict:
        """Fetch latest traded price for a list of symbols."""
        pass

    @abstractmethod
    def fetch_candles(self, symbol: str, interval: str, token: Optional[int]) -> list:
        """Fetch OHLC data for the given symbol and interval."""
        pass

    @abstractmethod
    def get_symbols(self, index) -> list:
        """Fetch OHLC data for the given symbol and interval."""
        pass

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        quantity: int,
        action: str,
        price: Optional[float] = None,
        order_type: str = "MARKET",
        timestamp: Optional[datetime] = None
    ) -> dict:
        """Place a buy/sell order."""
        pass