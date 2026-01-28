#!/usr/bin/env python
"""
Enrich nse_all.json with market cap data from Yahoo Finance.

Usage:
    python tools/enrich_mcap.py

This script:
1. Loads nse_all.json
2. Fetches market cap for each symbol using yfinance
3. Classifies symbols into cap segments (large/mid/small/micro)
4. Writes enriched data back to nse_all.json (original backed up as .bak)
5. Prints distribution summary

NSE India Market Cap Definitions:
- Large-Cap: ≥ ₹20,000 cr (Nifty 50 + top 100)
- Mid-Cap: ₹5,000 - ₹20,000 cr (Nifty MidCap 100)
- Small-Cap: ₹500 - ₹5,000 cr (Nifty SmallCap 250)
- Micro-Cap: < ₹500 cr (exclude from intraday)
"""
import json
import yfinance as yf
from pathlib import Path
import time
from typing import Dict, List


ROOT = Path(__file__).parent.parent
NSE_FILE = ROOT / "nse_all.json"


def classify_cap(mcap_cr: float) -> str:
    """
    Classify market cap into segments (in crores).

    Args:
        mcap_cr: Market cap in crores (₹)

    Returns:
        Cap segment: "large_cap", "mid_cap", "small_cap", or "micro_cap"
    """
    if mcap_cr >= 20000:
        return "large_cap"
    elif mcap_cr >= 5000:
        return "mid_cap"
    elif mcap_cr >= 500:
        return "small_cap"
    else:
        return "micro_cap"


def fetch_market_cap(symbol: str) -> tuple[float, str]:
    """
    Fetch market cap for a symbol using yfinance.

    Args:
        symbol: NSE symbol (e.g., "AARTIIND.NS")

    Returns:
        Tuple of (market_cap_cr, cap_segment)
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # Get market cap in INR
        mcap_inr = info.get("marketCap", 0)

        # Convert to crores (1 crore = 10 million = 1e7)
        mcap_cr = mcap_inr / 1e7

        # Classify
        cap_segment = classify_cap(mcap_cr)

        return mcap_cr, cap_segment

    except Exception as e:
        print(f"  [!] Failed to fetch {symbol}: {e}")
        return 0.0, "unknown"


def main():
    """Main enrichment workflow."""

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

    print(f"[+] Loaded {len(data)} symbols\n")

    # Backup original
    backup_file = NSE_FILE.with_suffix(".json.bak")
    print(f"[*] Creating backup: {backup_file.name}")
    with backup_file.open("w") as f:
        json.dump(data, f, indent=2)

    # Enrich each symbol
    print(f"\n[*] Fetching market cap data (this may take 5-10 minutes)...\n")

    for i, item in enumerate(data):
        symbol = item["symbol"]

        # Fetch market cap
        mcap_cr, cap_segment = fetch_market_cap(symbol)

        # Add to item
        item["market_cap_cr"] = round(mcap_cr, 2)
        item["cap_segment"] = cap_segment

        # Progress logging
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(data)} symbols...")

        # Rate limiting (avoid hitting API limits)
        time.sleep(0.2)  # 200ms delay between requests

    # Write enriched data
    print(f"\n[*] Writing enriched data to {NSE_FILE.name}...")
    with NSE_FILE.open("w") as f:
        json.dump(data, f, indent=2)

    print(f"[+] Enriched data written\n")

    # Print distribution summary
    segments: Dict[str, int] = {}
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

    # Summary stats
    mcaps = [item.get("market_cap_cr", 0) for item in data if item.get("market_cap_cr", 0) > 0]
    if mcaps:
        avg_mcap = sum(mcaps) / len(mcaps)
        print(f"\nAverage market cap: Rs.{avg_mcap:,.0f} cr")
        print(f"Median market cap: Rs.{sorted(mcaps)[len(mcaps)//2]:,.0f} cr")
        print(f"Max market cap: Rs.{max(mcaps):,.0f} cr")
        print(f"Min market cap: Rs.{min(mcaps):,.0f} cr")

    print(f"\n[+] ENRICHMENT COMPLETE")
    print(f"[+] Backup saved: {backup_file.name}")
    print(f"[+] Enriched file: {NSE_FILE.name}")


if __name__ == "__main__":
    main()
