"""Taxonomy-pinning tests for earnings_downshock_continuation_short_universe.

2026-08-04 incident: the live cap lookup preferred the NSE-Indices snapshot
(~755 names) while the V2 'small_cap only' filter was mined/pre-registered on
the nse_all.json market-cap bands — every validated candidate read 'unknown'
and the universe was empty on both first paper days (missed SIGNPOST fire).
"""
import sys
from datetime import date
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from services.setup_universe import earnings_downshock_continuation_short_universe  # noqa: E402
from services.symbol_metadata import get_cap_segment_nse_all  # noqa: E402


def test_unknown_taxonomy_fails_fast():
    with pytest.raises(ValueError):
        earnings_downshock_continuation_short_universe(
            {}, date(2026, 8, 4), {"cap_segment_taxonomy": "index_membership"})


def test_missing_taxonomy_key_fails_fast():
    with pytest.raises(KeyError):
        earnings_downshock_continuation_short_universe({}, date(2026, 8, 4), {})


def test_nse_all_taxonomy_classifies_validated_small_cap():
    """SIGNPOST is small_cap in nse_all.json (the validation taxonomy) — the
    Jul-31 −9.09% reaction the snapshot taxonomy silently dropped."""
    assert get_cap_segment_nse_all("NSE:SIGNPOST") == "small_cap"
    # unmapped garbage stays 'unknown' (never raises)
    assert get_cap_segment_nse_all("NSE:__NOPE__") == "unknown"
