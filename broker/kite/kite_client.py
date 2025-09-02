# kite_client.py
# Zerodha Kite SDK wrapper for managing instrument metadata, token/symbol maps, and equity universe filtering.
# Efficient in-memory maps for symbol/token resolution, with optional CSV fallback.

from typing import Dict, List, Iterable
from dataclasses import dataclass
import logging
from kiteconnect import KiteConnect
from config.env_setup import env
from kiteconnect import KiteTicker
from config.logging_config import get_loggers

logger, _ = get_loggers()

@dataclass(frozen=True)
class _Inst:
    token: int
    tsym: str            # e.g. 'RELIANCE'

class KiteClient:
    def __init__(self):
        self.api_key = env.KITE_API_KEY
        self.access_token = env.KITE_ACCESS_TOKEN
        if not self.api_key or not self.access_token:
            raise RuntimeError("KiteBroker: KITE_API_KEY and KITE_ACCESS_TOKEN are required")
        self._kc = KiteConnect(api_key=self.api_key)
        self._kc.set_access_token(self.access_token)
        self._sym2inst: Dict[str, _Inst] = {}
        self._tok2sym: Dict[int, str] = {}
        self._equity_instruments: List[str] = []

        self._load_instruments()
        
    def make_ticker(self):
        try:
            return KiteTicker(api_key=self.api_key, access_token=self.access_token)
        except Exception as e:
            raise RuntimeError(f"KiteClient: failed to create KiteTicker: {e}")

    def _load_instruments(self) -> None:
        """Load instruments from Kite API or CSV and build internal maps."""
        try:
            raw = self._kc.instruments()
        except Exception as e:
            raise RuntimeError(f"KiteClient: failed to load instruments: {e}")

        self._sym2inst.clear()
        self._tok2sym.clear()
        self._equity_instruments.clear()

        for i in raw:
            exch = i.get("exchange", "")
            seg = i.get("segment", "")
            itype = i.get("instrument_type", "")
            name = i.get("name", "").strip().lower()
            tsym = i.get("tradingsymbol", "")
            token = i.get("instrument_token")
            expiry = i.get("expiry")

            if not exch or not seg or not itype or not tsym or not token:
                continue

            # Safe equity filter
            if (
                exch != "NSE" or
                seg != "NSE" or
                itype != "EQ" or
                expiry or
                not name or len(name) < 3 or
                tsym.upper().startswith(tuple(str(i) for i in range(10))) or  # starts with digit
                any(x in tsym.upper() for x in ["SDL", "NCD", "GSEC", "GOI", "GS", "T-BILL"]) or
                "-" in tsym
            ):
                continue

            inst = _Inst(
                token=token,
                tsym=tsym
            )

            sym_key = f"{exch}:{tsym}"
            self._sym2inst[sym_key] = inst
            self._tok2sym[token] = sym_key
            self._equity_instruments.append(sym_key)

        logger.info(f"KiteClient: loaded {len(self._sym2inst)} total; NSE EQ = {len(self._equity_instruments)}")


    def list_equities(self) -> List[str]:
        """All NSE:EQ symbols as ['NSE:RELIANCE', ...]."""
        return self._equity_instruments

    def list_symbols(self, exchange: str = "NSE", instrument_type: str = "EQ") -> List[str]:
        """Return symbols like 'EXCH:TSYM' filtered by exchange/instrument_type."""
        e = exchange.upper()
        it = (instrument_type or "").upper()
        return [
            f"{i.exch}:{i.tsym}"
            for i in self._sym2inst.values()
            if i.exch == e and (not it or i.instrument_type == it)
        ]

    def list_tokens(self, exchange: str = "NSE", instrument_type: str = "EQ") -> List[int]:
        """Return instrument tokens filtered by exchange/instrument_type."""
        e = exchange.upper()
        it = (instrument_type or "").upper()
        return [
            i.token
            for i in self._sym2inst.values()
            if i.exch == e and (not it or i.instrument_type == it)
        ]

    def get_symbol_map(self) -> Dict[str, int]:
        return {s: i.token for s, i in self._sym2inst.items()}

    def get_token_map(self) -> Dict[int, str]:
        return dict(self._tok2sym)

    def resolve_tokens(self, symbols: Iterable[str]) -> List[int]:
        """Map ['EXCH:TSYM', ...] -> [token, ...]; unknowns skipped."""
        out: List[int] = []
        miss = 0
        for s in symbols:
            inst = self._sym2inst.get(str(s).upper())
            if inst:
                out.append(inst.token)
            else:
                miss += 1
        if miss:
            logger.debug("KiteClient.resolve_tokens: %d symbols not found", miss)
        return out
