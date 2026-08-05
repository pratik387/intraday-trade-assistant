"""Backfill `market_cap_cr` / `cap_segment` in nse_all.json from BSE.

Why
---
`nse_all.json` carries a `cap_segment` per symbol, derived from `market_cap_cr`
with the bands below. 1,037 of 2,333 symbols (44%) have `market_cap_cr == 0` and
therefore fall through to `cap_segment == "unknown"` — a MISSING-VALUE SENTINEL,
not a size class. Verified 2026-08-05: every single `unknown` symbol has
market_cap_cr exactly 0.0, and no symbol with a real cap is `unknown`.

That silently gates any setup filtering on cap. `earnings_downshock_continuation_short`
(V2, small_cap-only) loses every in-band candidate that happens to be unclassified —
on 2026-08-05 its ONLY qualifying candidate (BUTTERFLY, −9.78%) was dropped purely
because its cap was missing. The `unknown` share of that setup's trades ran
0.0% (2023) → 0.0% (2024) → 2.3% (2025) → 38.7% (2026): a growing data gap, not a
changing market.

Source chain (all public, no auth)
----------------------------------
    NSE symbol  --EQUITY_L.csv-->  ISIN
    ISIN        --bse_scrip_master.json (local, reversed)-->  BSE scripcode
    scripcode   --api.bseindia.com StockTrading-->  MktCapFull  (already Rs CRORE)

Verified against Reliance (500325): MktCapFull 17,38,931 = Rs 17.4 lakh crore. Correct unit.

Bands (reverse-engineered 2026-08-05 from the already-classified symbols; every cut
point falls cleanly inside an observed gap, no overlap):
    micro_cap  <   500
    small_cap  <  5000
    mid_cap    < 20000
    large_cap  >= 20000

Coverage: ~648 of 1,037 unknowns resolve (63%); the remainder are NSE-only
listings (SME / recent) with no BSE counterpart and stay `unknown`.

Provenance / safety
-------------------
This CHANGES THE INPUT to a pre-registered filter, so it is deliberately explicit:
  * the pre-change file is copied to nse_all.json.bak-<date> before writing
  * a dated snapshot of the applied deltas is written to
    data/cap_segments/market_cap_backfill_<date>.json
  * ONLY symbols currently `unknown` are touched — an existing classification is
    never overwritten, so the validated population cannot shift underneath us
  * --dry-run reports the deltas without writing

Usage:
    python tools/backfill_market_caps.py --dry-run
    python tools/backfill_market_caps.py
    python tools/backfill_market_caps.py --from-cache /tmp/mcap_backfill.json
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Dict, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
NSE_ALL = ROOT / "nse_all.json"
BSE_MASTER = ROOT / "data" / "earnings_calendar" / "bse_scrip_master.json"
SNAPSHOT_DIR = ROOT / "data" / "cap_segments"

EQUITY_L_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
BSE_URL = ("https://api.bseindia.com/BseIndiaAPI/api/StockTrading/w"
           "?flag=&quotetype=EQ&scripcode={code}&seriesid=")

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120 Safari/537.36")
BSE_HEADERS = {"User-Agent": _UA, "Accept": "application/json",
               "Referer": "https://www.bseindia.com/",
               "Origin": "https://www.bseindia.com"}

# Cut points in Rs crore. Reverse-engineered, NOT invented — see module docstring.
BANDS = ((500.0, "micro_cap"), (5000.0, "small_cap"), (20000.0, "mid_cap"))


def classify(cap_cr: float) -> str:
    for edge, name in BANDS:
        if cap_cr < edge:
            return name
    return "large_cap"


def load_symbol_to_isin() -> Dict[str, str]:
    r = requests.get(EQUITY_L_URL, headers={"User-Agent": _UA}, timeout=30)
    r.raise_for_status()
    out: Dict[str, str] = {}
    for row in csv.DictReader(io.StringIO(r.text)):
        isin_key = next((c for c in row if "ISIN" in c), None)
        if isin_key:
            out[row["SYMBOL"].strip()] = (row[isin_key] or "").strip()
    return out


def fetch_market_cap(code: str, timeout: int = 20) -> Optional[float]:
    """Return MktCapFull in Rs crore, or None. Never raises."""
    try:
        r = requests.get(BSE_URL.format(code=code), headers=BSE_HEADERS, timeout=timeout)
        if r.status_code != 200:
            return None
        raw = r.json().get("MktCapFull")
        if raw in (None, "", "-"):
            return None
        return float(str(raw).replace(",", ""))
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="report deltas, write nothing")
    ap.add_argument("--sleep-secs", type=float, default=1.2,
                    help="politeness between BSE calls (BSE responds in 1-3.5s)")
    ap.add_argument("--from-cache", type=Path,
                    help="reuse a previously fetched {symbol: cap_cr} JSON instead of refetching")
    args = ap.parse_args()

    nse = json.loads(NSE_ALL.read_text(encoding="utf-8"))
    unknown = [r for r in nse if r.get("cap_segment") == "unknown"]
    print(f"nse_all symbols: {len(nse)} | currently unknown: {len(unknown)}")

    if args.from_cache:
        caps: Dict[str, float] = {k: float(v) for k, v in
                                  json.loads(args.from_cache.read_text()).items()}
        print(f"loaded {len(caps)} cached market caps from {args.from_cache}")
    else:
        sym2isin = load_symbol_to_isin()
        isin2code = {v: k for k, v in json.loads(BSE_MASTER.read_text(encoding="utf-8")).items()}
        todo = []
        for r in unknown:
            s = r["symbol"].replace(".NS", "")
            isin = sym2isin.get(s)
            code = isin2code.get(isin) if isin else None
            if code:
                todo.append((s, code))
        print(f"resolvable to a BSE scripcode: {len(todo)} "
              f"({100*len(todo)/max(len(unknown),1):.0f}%)")
        caps = {}
        for i, (s, code) in enumerate(todo, 1):
            cap = fetch_market_cap(code)
            if cap is not None:
                caps[s] = cap
            if i % 100 == 0:
                print(f"  {i}/{len(todo)} fetched_ok={len(caps)}", flush=True)
            time.sleep(args.sleep_secs)

    # Apply — ONLY to symbols currently unknown.
    deltas, counts = {}, Counter()
    for r in nse:
        if r.get("cap_segment") != "unknown":
            continue
        s = r["symbol"].replace(".NS", "")
        cap = caps.get(s)
        if cap is None or cap <= 0:
            continue
        seg = classify(cap)
        deltas[r["symbol"]] = {"market_cap_cr": cap, "cap_segment": seg}
        counts[seg] += 1
        if not args.dry_run:
            r["market_cap_cr"] = cap
            r["cap_segment"] = seg

    print(f"\nreclassified {len(deltas)} of {len(unknown)} unknown symbols:")
    for seg in ("micro_cap", "small_cap", "mid_cap", "large_cap"):
        print(f"  {seg:<10} {counts.get(seg, 0):4d}")
    print(f"  {'still unknown':<10} {len(unknown) - len(deltas):4d}  (no BSE listing)")

    before = Counter(r.get("cap_segment") for r in json.loads(NSE_ALL.read_text(encoding="utf-8")))
    after = Counter(before)
    for d in deltas.values():
        after["unknown"] -= 1
        after[d["cap_segment"]] += 1
    print("\nuniverse-wide cap_segment, before -> after:")
    for k in ("large_cap", "mid_cap", "small_cap", "micro_cap", "unknown"):
        print(f"  {k:<10} {before.get(k,0):5d} -> {after.get(k,0):5d}")

    if args.dry_run:
        print("\nDRY RUN — nothing written.")
        return 0

    stamp = date.today().isoformat()
    backup = NSE_ALL.with_suffix(f".json.bak-{stamp}")
    shutil.copy2(NSE_ALL, backup)
    NSE_ALL.write_text(json.dumps(nse, indent=2), encoding="utf-8")
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    snap = SNAPSHOT_DIR / f"market_cap_backfill_{stamp}.json"
    snap.write_text(json.dumps(
        {"applied": stamp, "source": "BSE StockTrading MktCapFull via ISIN",
         "bands_cr": {"micro_cap": "<500", "small_cap": "<5000",
                      "mid_cap": "<20000", "large_cap": ">=20000"},
         "deltas": deltas}, indent=2), encoding="utf-8")
    print(f"\nwrote {NSE_ALL}\nbackup {backup}\nsnapshot {snap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
