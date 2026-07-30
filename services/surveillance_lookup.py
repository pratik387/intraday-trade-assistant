"""NSE ASM / BSE GSM surveillance membership lookup.

Answers one question: "was this symbol under surveillance on this date?"
Used by `services.setup_universe.earnings_downshock_continuation_short_universe`
to exclude surveillance names, because an ASM listing carries 100% margin —
MIS leverage is revoked, so an MIS short cannot be placed at all.

Same shape as `services/delivery_pct_enrichment.py` and
`services/earnings_reaction_enrichment.py`: load the parquet once, cache a
set for O(1) lookup, never raise into a caller.

Research basis (2026-07-29 gauntlet, brief §7.4): 9.2% of era_A and 6.7% of
era_B qualifying events were ASM-listed on the entry date, and **0.00% were
Stage III/IV trade-for-trade in either era**. Excluding surveillance names
*improved* era_A expectancy (+0.057% -> +0.173%) — the ASM subset was the
loss centre (n=35, -1.09% net).

DATA DEPENDENCY (must exist on the OCI/VM box):
  data/asm_gsm_history/asm_gsm_events.parquet
Materialised 2026-07-28 via tools/asm_gsm_history/fetch_asm_gsm.py
(1,068,662 rows, 2023-01-02 -> 2026-07-24). NSE rows carry
ASM_LONGTERM / ASM_SHORTTERM / ASM_LONGTERM_IBC; NSE GSM circulars are
image PDFs and were never parsed, so GSM membership comes from BSE-exchange
rows resolved by ISIN and is thin (~51 panel symbols).

FAIL-OPEN vs FAIL-CLOSED: a missing parquet returns "not under
surveillance" for every symbol, i.e. it does NOT block trading. That is
deliberate — a missing exclusion file must not silently empty the universe
(the below_vwap silent-zero trap, lesson #16). The caller is expected to
surface `load_error()`, and the data-health monitor in
jobs/check_edge_integrity.py is the real guard.

All dates IST-naive.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional, Set, Tuple

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
_PARQUET = _REPO / "data" / "asm_gsm_history" / "asm_gsm_events.parquet"

_MEMBERSHIP: Optional[Set[Tuple[str, date]]] = None
_LOAD_ATTEMPTED = False
_LOAD_ERROR: Optional[str] = None


def _bare(sym: str) -> str:
    return str(sym).replace("NSE:", "").replace(".NS", "").strip().upper()


def _load() -> Optional[Set[Tuple[str, date]]]:
    """Build the {(bare_symbol, date)} membership set once. None on failure."""
    global _MEMBERSHIP, _LOAD_ATTEMPTED, _LOAD_ERROR
    if _LOAD_ATTEMPTED:
        return _MEMBERSHIP
    _LOAD_ATTEMPTED = True

    if not _PARQUET.exists():
        _LOAD_ERROR = f"missing {_PARQUET}"
        return None
    try:
        df = pd.read_parquet(_PARQUET, columns=["symbol", "date", "transition_type"])
        # Every row is a daily snapshot of "in the list on this date", so any
        # row except an explicit exit means the name was listed that session.
        df = df[df["transition_type"].astype(str) != "exit"]
        sym = df["symbol"].map(_bare)
        dts = pd.to_datetime(df["date"]).dt.date
        _MEMBERSHIP = set(zip(sym, dts))
        return _MEMBERSHIP
    except Exception as exc:  # never raise into the universe path
        _LOAD_ERROR = f"{type(exc).__name__}: {exc}"
        return None


def load_error() -> Optional[str]:
    """Why the lookup is empty, for logging by the caller."""
    return _LOAD_ERROR


def is_under_surveillance(symbol: str, on_date: date) -> bool:
    """True iff `symbol` was in an ASM/GSM list on `on_date`.

    Fails OPEN (returns False) when the parquet is unavailable — see the
    module docstring for why.
    """
    keys = _load()
    if not keys:
        return False
    return (_bare(symbol), on_date) in keys
