"""refresh_mis_short_eligibility.py — daily pre-open MIS-short eligibility refresh.

For the up_spike_fade_short setup (and any future SHORT illiquid-fade setup),
the distinguishing gate is broker intraday SELL/MIS LEVERAGE > 1:
  - leverage 5  -> genuine MIS short (e.g. 5x intraday) -> ELIGIBLE
  - leverage 1  -> 100%-margin surveillance (ASM/GSM, e.g. JINDALPHOT) -> EXCLUDE
  - leverage 0  -> invalid / delisted -> EXCLUDE
The `order_margins` STATUS field is NOT a valid gate — it returned "available"
even for a non-existent symbol (FAKEXYZ123). The leverage field is authoritative.

This job, run pre-open (daily cron), instantiates KiteBroker, builds the
cap in {small,micro,unknown} + MIS-eligible candidate universe, calls
`kc.order_margins([...SELL/MIS...])` per symbol, reads the `leverage` field,
and writes {symbol: leverage} to data/mis_short_eligibility/latest.json. The
up_spike_fade_short universe builder (services/setup_universe.py) reads that
map and admits only leverage > mis_short_min_leverage.

Pattern mirrors tools/sub9_research/_t1_shortability_sweep.py.

Usage (pre-open daily cron):
    python -m jobs.refresh_mis_short_eligibility
    python -m jobs.refresh_mis_short_eligibility --setup up_spike_fade_short

Requires KITE_API_KEY + KITE_ACCESS_TOKEN in the environment (KiteBroker).
All trading-relevant params (cap allow-set, output path, throttle) come from
config/configuration.json — NO hardcoded trading defaults (CLAUDE.md Rule #1).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Repo root on path so `broker`, `services`, `config` import when run as a file.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.logging_config import get_agent_logger  # noqa: E402

logger = get_agent_logger()

CONFIG_PATH = PROJECT_ROOT / "config" / "configuration.json"


def _load_setup_block(setup_name: str) -> Dict[str, Any]:
    """Read setups[setup_name] from config/configuration.json (UTF-8)."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    setups = cfg.get("setups", {})
    if setup_name not in setups:
        raise KeyError(f"setup '{setup_name}' not found under setups in {CONFIG_PATH}")
    return setups[setup_name]


def _candidate_symbols(allowed_caps: set) -> List[str]:
    """Bare symbols whose cap_segment is in allowed_caps AND MIS-enabled.

    Enumerates the FULL tradable universe from nse_all.json (not the
    cap_segments snapshot) so 'unknown'-cap names are included — get_cap_segment
    returns 'unknown' for symbols outside the NSE indices, and the
    up_spike_fade_short cell explicitly includes cap_segment 'unknown'. This
    mirrors how services/setup_universe iterates daily_dict + get_cap_segment.
    """
    from services.symbol_metadata import (
        get_cap_segment,
        get_mis_info,
        _load_nse_all,  # type: ignore[attr-defined]
        _normalize_symbol,  # type: ignore[attr-defined]
    )

    data = _load_nse_all()  # list of {"symbol": ..., "cap_segment": ..., "mis_enabled": ...}
    if not data:
        return []

    out: List[str] = []
    for item in data:
        try:
            # nse_all symbols carry a '.NS' suffix (e.g. 'AAATECH.NS'); strip BOTH
            # the exchange prefix and the suffix so 'NSE:'+bare resolves in kc.ltp /
            # order_margins. Stripping only 'NSE:' left '.NS' on, which made every
            # LTP lookup miss -> price 0 -> leverage 0 -> 0 eligible (caught in the
            # first end-to-end refresher run, 2026-06-03).
            bare = _normalize_symbol(str(item["symbol"]))
        except Exception:
            continue
        nse_sym = f"NSE:{bare}"
        try:
            if get_cap_segment(nse_sym) not in allowed_caps:
                continue
            if not get_mis_info(nse_sym).get("mis_enabled", False):
                continue
        except Exception:
            continue
        out.append(bare)
    return sorted(set(out))


def refresh(setup_name: str = "up_spike_fade_short") -> bool:
    """Build {symbol: leverage} via Kite order_margins and write latest.json."""
    block = _load_setup_block(setup_name)

    allowed_caps = set(block["allowed_cap_segments"])
    out_path = PROJECT_ROOT / Path(str(block["mis_short_eligibility_path"]))
    # Per-call throttle (config-driven; falls back to a conservative live-safe
    # value if not configured — this is an operational rate-limit knob, not a
    # trading parameter, so a runtime default is acceptable).
    throttle_s = float(block.get("mis_short_refresh_throttle_s", 0.12))
    min_lev = float(block["mis_short_min_leverage"])

    symbols = _candidate_symbols(allowed_caps)
    logger.info(
        "refresh_mis_short_eligibility[%s]: %d candidate symbols (caps=%s)",
        setup_name, len(symbols), sorted(allowed_caps),
    )
    if not symbols:
        logger.error("refresh_mis_short_eligibility: no candidate symbols — aborting")
        return False

    from broker.kite.kite_broker import KiteBroker

    broker = KiteBroker(dry_run=True)  # dry_run: no orders sent; order_margins is read-only

    # LTP in batches (order_margins leverage field wants a price).
    nsyms = ["NSE:" + s for s in symbols]
    ltp: Dict[str, float] = {}
    for j in range(0, len(nsyms), 400):
        chunk = nsyms[j:j + 400]
        try:
            r = broker.kc.ltp(chunk)
            for kk, vv in r.items():
                ltp[kk.split(":")[-1]] = vv.get("last_price")
        except Exception as e:
            logger.warning("ltp chunk %d err: %s", j, e)
        time.sleep(0.3)

    lev: Dict[str, Any] = {}
    for i, s in enumerate(symbols):
        px = ltp.get(s)
        try:
            m = broker.kc.order_margins([{
                "exchange": "NSE", "tradingsymbol": s,
                "transaction_type": "SELL", "variety": "regular",
                "product": "MIS", "order_type": "MARKET",
                "quantity": 1, "price": float(px or 0),
            }])
            lev[s] = (m[0].get("leverage") if m else None)
        except Exception:
            lev[s] = None
        time.sleep(throttle_s)
        if (i + 1) % 200 == 0:
            ok = sum(1 for v in lev.values() if v and v > min_lev)
            logger.info("  %d/%d checked, leverage>%.0f so far %d", i + 1, len(symbols), min_lev, ok)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write atomically (temp + replace) so a concurrent universe-builder read
    # never sees a partial file.
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(lev, f, ensure_ascii=False)
    tmp.replace(out_path)

    eligible = sum(1 for v in lev.values() if v and v > min_lev)
    logger.info(
        "refresh_mis_short_eligibility[%s]: wrote %d symbols -> %s (eligible leverage>%.0f: %d)",
        setup_name, len(lev), out_path, min_lev, eligible,
    )
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Refresh MIS-short eligibility (leverage) map.")
    ap.add_argument("--setup", default="up_spike_fade_short",
                    help="Setup whose config block supplies caps + output path.")
    args = ap.parse_args()
    ok = refresh(args.setup)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
