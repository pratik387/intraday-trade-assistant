#!/usr/bin/env python
"""
Fast market cap enrichment with incremental saves.

This version:
- Saves every 50 symbols (incremental progress)
- Skips already enriched symbols
- Shows progress in real-time
"""
import json
import yfinance as yf
from pathlib import Path
import time


ROOT = Path(__file__).parent.parent
NSE_FILE = ROOT / "nse_all.json"


def classify_cap(mcap_cr: float) -> str:
    """Classify market cap into segments (in crores)."""
    if mcap_cr >= 20000:
        return "large_cap"
    elif mcap_cr >= 5000:
        return "mid_cap"
    elif mcap_cr >= 500:
        return "small_cap"
    else:
        return "micro_cap"


def fetch_market_cap(symbol: str) -> tuple[float, str]:
    """Fetch market cap for a symbol using yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        mcap_inr = info.get("marketCap", 0)
        mcap_cr = mcap_inr / 1e7
        cap_segment = classify_cap(mcap_cr)
        return mcap_cr, cap_segment
    except Exception as e:
        return 0.0, "unknown"


def main():
    """Main enrichment workflow with incremental saves."""

    # Check if yfinance is installed
    try:
        import yfinance
    except ImportError:
        print("[X] yfinance not installed. Run: pip install yfinance")
        return

    # Load existing data
    if not NSE_FILE.exists():
        print(f"[X] {NSE_FILE} not found")
        return

    print(f"[*] Loading {NSE_FILE}...")
    with NSE_FILE.open() as f:
        data = json.load(f)

    print(f"[+] Loaded {len(data)} symbols")

    # Backup original (only once)
    backup_file = NSE_FILE.with_suffix(".json.bak")
    if not backup_file.exists():
        print(f"[*] Creating backup: {backup_file.name}")
        with backup_file.open("w") as f:
            json.dump(data, f, indent=2)

    # Count already enriched
    enriched_count = sum(1 for item in data if item.get("market_cap_cr", 0) > 0)
    print(f"[+] Already enriched: {enriched_count}/{len(data)} symbols")

    if enriched_count == len(data):
        print("[+] All symbols already enriched!")
        return

    print(f"\n[*] Fetching market cap data for {len(data) - enriched_count} symbols...\n")

    # Enrich each symbol with incremental saves
    modified = False
    for i, item in enumerate(data):
        symbol = item["symbol"]

        # Skip if already enriched
        if item.get("market_cap_cr", 0) > 0:
            continue

        # Fetch market cap
        mcap_cr, cap_segment = fetch_market_cap(symbol)

        # Add to item
        item["market_cap_cr"] = round(mcap_cr, 2)
        item["cap_segment"] = cap_segment
        modified = True

        # Progress logging
        current_enriched = sum(1 for x in data if x.get("market_cap_cr", 0) > 0)
        if (i + 1) % 10 == 0:
            pct = (current_enriched / len(data)) * 100
            print(f"  Progress: {current_enriched}/{len(data)} ({pct:.1f}%) - Latest: {symbol} -> {cap_segment}")

        # Incremental save every 50 symbols
        if modified and (i + 1) % 50 == 0:
            print(f"  [*] Saving progress ({current_enriched}/{len(data)} enriched)...")
            with NSE_FILE.open("w") as f:
                json.dump(data, f, indent=2)
            modified = False

        # Rate limiting
        time.sleep(0.1)  # 100ms delay (faster than original)

    # Final save
    if modified:
        print(f"\n[*] Writing final enriched data...")
        with NSE_FILE.open("w") as f:
            json.dump(data, f, indent=2)

    print(f"[+] Enriched data written\n")

    # Print distribution summary
    segments = {}
    for item in data:
        seg = item.get("cap_segment", "unknown")
        segments[seg] = segments.get(seg, 0) + 1

    print("=" * 50)
    print("MARKET CAP DISTRIBUTION")
    print("=" * 50)
    for seg in ["large_cap", "mid_cap", "small_cap", "micro_cap", "unknown"]:
        count = segments.get(seg, 0)
        pct = (count / len(data) * 100) if len(data) > 0 else 0
        print(f"  {seg:12s}: {count:4d} symbols ({pct:5.1f}%)")
    print("=" * 50)

    print(f"\n[+] ENRICHMENT COMPLETE")


if __name__ == "__main__":
    main()
