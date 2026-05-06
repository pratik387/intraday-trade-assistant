"""Build stock-to-sector mapping for sub-9 round-4 sector_rotation_relative_strength sanity.

Reads NSE Indices constituent CSVs from assets/ and builds a JSON map
keyed by NSE-prefixed symbol -> sector index name (matching index_ohlcv directory names).

Priority order (FIRST match wins, deterministic):
    BANK -> IT -> AUTO -> FMCG -> PHARMA -> METAL -> ENERGY ->
    FIN_SERVICE -> REALTY -> PSU_BANK -> NIFTY 50 (fallback)

Output:
    assets/stock_sector_map.json
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "assets"

# (csv_filename, sector_label_for_index_ohlcv)
# Order = priority; first match wins.
SECTOR_FILES = [
    ("ind_niftybanklist.csv",     "NSE_NIFTY_BANK"),
    ("ind_niftyitlist.csv",       "NSE_NIFTY_IT"),
    ("ind_niftyautolist.csv",     "NSE_NIFTY_AUTO"),
    ("ind_niftyfmcglist.csv",     "NSE_NIFTY_FMCG"),
    ("ind_niftypharmalist.csv",   "NSE_NIFTY_PHARMA"),
    ("ind_niftymetallist.csv",    "NSE_NIFTY_METAL"),
    ("ind_niftyenergylist.csv",   "NSE_NIFTY_ENERGY"),
    ("ind_niftyfinancelist.csv",  "NSE_NIFTY_FIN_SERVICE"),  # Nifty Financial Services
    ("ind_niftyrealtylist.csv",   "NSE_NIFTY_REALTY"),
    ("ind_niftypsubanklist.csv",  "NSE_NIFTY_PSU_BANK"),
]
FALLBACK_SECTOR = "NSE_NIFTY_50"
FALLBACK_FILE = "ind_nifty50list.csv"


def load_symbols(csv_path: Path) -> set[str]:
    """Return uppercase set of Symbol column from a NSE constituent CSV."""
    syms: set[str] = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sym = (row.get("Symbol") or "").strip().upper()
            if sym:
                syms.add(sym)
    return syms


def main() -> None:
    # 1. Load every sector constituent set (deterministic ordering).
    sector_members: list[tuple[str, set[str]]] = []
    for fname, label in SECTOR_FILES:
        path = ASSETS / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing sector CSV: {path}")
        sector_members.append((label, load_symbols(path)))

    fallback_path = ASSETS / FALLBACK_FILE
    if not fallback_path.exists():
        raise FileNotFoundError(f"Missing fallback CSV: {fallback_path}")
    fallback_members = load_symbols(fallback_path)

    # 2. Read F&O 200 universe.
    fno_path = ASSETS / "fno_liquid_200.csv"
    fno_symbols: list[str] = []
    with open(fno_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sym = (row.get("symbol") or "").strip()
            if sym:
                fno_symbols.append(sym)

    # 3. Build mapping; record overlaps and fallbacks.
    mapping: dict[str, str] = {}
    overlap: dict[str, list[str]] = defaultdict(list)  # symbol -> all sectors it appears in
    fallback_hits: list[str] = []
    unmatched: list[str] = []

    for nse_sym in fno_symbols:
        # Strip "NSE:" prefix for sector membership check.
        bare = nse_sym.split(":", 1)[1].upper() if ":" in nse_sym else nse_sym.upper()

        # Track ALL sector matches (for overlap reporting).
        all_matches: list[str] = []
        for label, members in sector_members:
            if bare in members:
                all_matches.append(label)
        if bare in fallback_members:
            all_matches.append(FALLBACK_SECTOR)

        if all_matches:
            mapping[nse_sym] = all_matches[0]  # first = highest priority
            if all_matches[0] == FALLBACK_SECTOR and len(all_matches) == 1:
                fallback_hits.append(nse_sym)
            if len(all_matches) > 1:
                overlap[nse_sym] = all_matches
        else:
            mapping[nse_sym] = FALLBACK_SECTOR
            unmatched.append(nse_sym)

    # 4. Write output JSON (sorted keys for deterministic diff).
    out_path = ASSETS / "stock_sector_map.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(mapping.items())), f, indent=2, ensure_ascii=False)

    # 5. Print stats so caller (and validation report) can see them.
    counts = Counter(mapping.values())
    print(f"Wrote: {out_path}")
    print(f"Total F&O 200 symbols mapped: {len(mapping)}")
    print("\nSector counts:")
    for sec, cnt in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {sec}: {cnt}")
    print(f"\nFallback to NSE_NIFTY_50 (no specific sector found): {len(fallback_hits)}")
    for s in fallback_hits:
        print(f"  {s}")
    print(f"\nUnmatched (also fell to NSE_NIFTY_50): {len(unmatched)}")
    for s in unmatched:
        print(f"  {s}")
    print(f"\nMulti-sector overlaps (took first by priority): {len(overlap)}")
    for sym, secs in sorted(overlap.items()):
        print(f"  {sym} -> picked {secs[0]} from {secs}")

    # Stash stats for the validation-report writer in this module's globals.
    main.stats = {  # type: ignore[attr-defined]
        "counts": dict(counts),
        "fallback_hits": fallback_hits,
        "unmatched": unmatched,
        "overlap": dict(overlap),
        "total": len(mapping),
    }


if __name__ == "__main__":
    main()
