"""Zerodha MTF approved-securities loader.

Provides a clean interface to query MTF eligibility + per-stock leverage
for the close_dn_overnight_long setup (and any future MTF setups).

Snapshot source: data/mtf_universe/approved_mtf_securities_YYYY-MM-DD.json
Refresh tool:    tools/scrape_zerodha_mtf.py
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MtfInfo:
    """One row from the Zerodha MTF approved-securities snapshot."""
    tradingsymbol: str        # NSE bare symbol, e.g. "RELIANCE", "BANKBEES"
    isin: str                 # ISIN code
    category: str             # 'fo' | 'non_fo' | 'non_categorized' | 'etf'
    margin_pct: float         # client margin requirement, 0-100 (e.g. 26.0)
    leverage: float           # 100 / margin_pct (2.0 to 5.0)


class MtfUniverse:
    """Loads + queries the Zerodha MTF approved-securities snapshot."""

    def __init__(self, snapshot_path: Path):
        self._snapshot_path = Path(snapshot_path)
        self._by_symbol: dict[str, MtfInfo] = {}
        self._load()

    def _load(self) -> None:
        if not self._snapshot_path.exists():
            raise FileNotFoundError(
                f"MTF snapshot not found: {self._snapshot_path}. "
                f"Run tools/scrape_zerodha_mtf.py to refresh."
            )
        with open(self._snapshot_path, encoding="utf-8") as f:
            entries = json.load(f)
        if not isinstance(entries, list):
            raise ValueError(
                f"MTF snapshot at {self._snapshot_path} has unexpected shape "
                f"(expected list, got {type(entries).__name__})"
            )
        for e in entries:
            try:
                sym = str(e["tradingsymbol"]).upper().strip()
                if not sym:
                    continue
                self._by_symbol[sym] = MtfInfo(
                    tradingsymbol=sym,
                    isin=str(e.get("isin", "")),
                    category=str(e.get("category", "non_categorized")),
                    margin_pct=float(e["margin"]),
                    leverage=float(e["leverage"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "mtf_universe: skipping malformed entry %r: %s", e, exc
                )
        logger.info(
            "mtf_universe: loaded %d entries from %s",
            len(self._by_symbol), self._snapshot_path.name,
        )

    def lookup(self, bare_symbol: str) -> Optional[MtfInfo]:
        """Return MtfInfo if symbol is in the MTF list, None otherwise.

        Accepts either bare symbol ('RELIANCE') or NSE-prefixed ('NSE:RELIANCE').
        """
        s = bare_symbol.upper().replace("NSE:", "").strip()
        return self._by_symbol.get(s)

    def is_eligible(self, bare_symbol: str, *, exclude_etf: bool = True) -> bool:
        """True if the symbol is in the MTF approved list.

        If exclude_etf=True (default), category='etf' entries return False.
        Our close_dn_overnight_long setup excludes ETFs because their EOD
        microstructure (NAV-vs-market-price arb) differs from the
        equity-mechanism story the setup targets.
        """
        info = self.lookup(bare_symbol)
        if info is None:
            return False
        if exclude_etf and info.category == "etf":
            return False
        return True

    def all_symbols(self) -> set[str]:
        """All bare tradingsymbols in the snapshot (ETFs included).

        Callers wanting the tradeable set should intersect with is_eligible()
        (which applies the ETF exclusion).
        """
        return set(self._by_symbol.keys())

    def snapshot_age_days(self) -> int:
        """Days since snapshot file was last modified (for staleness warnings)."""
        mtime = self._snapshot_path.stat().st_mtime
        age_sec = time.time() - mtime
        return int(age_sec // 86400)

    def __len__(self) -> int:
        return len(self._by_symbol)
