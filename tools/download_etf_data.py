"""
Download historical data for ETF symbols for verification
"""
import os
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

# Upstox API configuration
UPSTOX_API_BASE = "https://api.upstox.com/v2/historical-candle"

# Symbols to download with their instrument keys
SYMBOLS = {
    "PSUBNKBEES": "NSE_EQ|INF204KB16I7",
    "PSUBANK": "NSE_EQ|INF373I01023"
}

# Date ranges
DATE_RANGES = [
    {"name": "Oct30_PrevDay", "start": "2025-10-30", "end": "2025-10-30"},
    {"name": "Oct31_TradingDay", "start": "2025-10-31", "end": "2025-10-31"}
]

# Timeframes
TIMEFRAMES = [
    {"interval": "1minute"},
    {"interval": "day"}
]

def download_data(symbol, instrument_key, date_range, timeframe):
    """Download data from Upstox API"""
    interval = timeframe["interval"]
    start_date = date_range["start"]
    end_date = date_range["end"]

    # Format dates for API
    from_date = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y-%m-%d")
    to_date = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y-%m-%d")

    # Build API URL
    url = f"{UPSTOX_API_BASE}/{instrument_key}/{interval}/{to_date}/{from_date}"

    print(f"Downloading {symbol} {interval} for {date_range['name']}...")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "success":
            print(f"  FAILED: {data.get('message', 'Unknown error')}")
            return None

        candles = data.get("data", {}).get("candles", [])
        if not candles:
            print(f"  No data returned")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(candles, columns=["date", "open", "high", "low", "close", "volume", "oi"])
        df = df.drop(columns=["oi"])  # Remove OI column
        df["date"] = pd.to_datetime(df["date"])

        print(f"  Downloaded {len(df)} candles")
        return df

    except Exception as e:
        print(f"  ERROR: {e}")
        return None

def main():
    output_dir = Path("cache/verification/oct31")

    for symbol, instrument_key in SYMBOLS.items():
        symbol_dir = output_dir / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)

        for date_range in DATE_RANGES:
            for timeframe in TIMEFRAMES:
                df = download_data(symbol, instrument_key, date_range, timeframe)
                if df is not None:
                    # Save to feather
                    interval = timeframe["interval"]
                    filename = f"{symbol}_{interval}s_{date_range['name']}.feather"
                    filepath = symbol_dir / filename
                    df.to_feather(filepath)
                    print(f"  Saved to {filepath}")

if __name__ == "__main__":
    main()
