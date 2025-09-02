import os
import json
import logging
from typing import List, Dict, Iterable

from broker.mock.feather_tick_loader import FeatherTickLoader
from broker.mock.feather_ticker import FeatherTicker
from config.logging_config import get_loggers

logger, _ = get_loggers()


class _Inst:
    def __init__(self, token: int, exch: str, tsym: str, instrument_type: str):
        self.token = token
        self.exch = exch
        self.tsym = tsym
        self.instrument_type = instrument_type


class MockBroker:
    def __init__(self, path_json: str = "nse_all.json", from_date: str = None, to_date: str = None):
        self._sym2inst: Dict[str, _Inst] = {}
        self._tok2sym: Dict[int, str] = {}
        self._equity_instruments: List[str] = []
        self._from_date = from_date
        self._to_date = to_date

        self._load_instruments(path_json)

    def _load_instruments(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"MockBroker: JSON file not found: {path}")

        with open(path, "r") as f:
            raw = json.load(f)

        for i in raw:
            tsym = i.get("symbol")
            token = i.get("instrument_token")
            if not tsym or not token:
                continue

            exch = "NSE"
            itype = "EQ"

            inst = _Inst(token=token, exch=exch, tsym=tsym.replace(".NS", ""), instrument_type=itype)

            sym_key = f"{exch}:{inst.tsym}"
            self._sym2inst[sym_key] = inst
            self._tok2sym[token] = sym_key

        self._equity_instruments = list(self._sym2inst.keys())
        logger.info(f"MockBroker: loaded {len(self._equity_instruments)} instruments from {path}")

    def list_equities(self) -> List[str]:
        return self._equity_instruments

    def get_symbol_map(self) -> Dict[str, int]:
        return {s: i.token for s, i in self._sym2inst.items()}

    def get_token_map(self) -> Dict[int, str]:
        return dict(self._tok2sym)

    def list_symbols(self, exchange: str = "NSE", instrument_type: str = "EQ") -> List[str]:
        return [
            f"{i.exch}:{i.tsym}"
            for i in self._sym2inst.values()
            if i.exch == exchange and i.instrument_type == instrument_type
        ]

    def list_tokens(self, exchange: str = "NSE", instrument_type: str = "EQ") -> List[int]:
        return [
            i.token
            for i in self._sym2inst.values()
            if i.exch == exchange and i.instrument_type == instrument_type
        ]

    def resolve_tokens(self, symbols: Iterable[str]) -> List[int]:
        out: List[int] = []
        miss = 0
        for s in symbols:
            inst = self._sym2inst.get(str(s).upper())
            if inst:
                out.append(inst.token)
            else:
                miss += 1
        if miss:
            logger.warning(f"MockBroker.resolve_tokens: {miss} symbols not found")
        return out

    def make_ticker(self) -> FeatherTicker:
        if not self._from_date or not self._to_date:
            raise ValueError("MockBroker: from_date and to_date must be set")

        loader = FeatherTickLoader(
            from_date=self._from_date,
            to_date=self._to_date,
            symbols=self._equity_instruments,
        )

        tok2sym = self.get_token_map()
        sym2tok = {v: k for k, v in tok2sym.items()}

        return FeatherTicker(
            loader=loader,
            tok2sym=tok2sym,
            sym2tok=sym2tok,
            replay_sleep=0.01,
            use_close_as_price=True,
        )
