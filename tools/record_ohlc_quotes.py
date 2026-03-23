#!/usr/bin/env python3
"""
OHLC Quotes Recorder — Records Upstox REST API I1 candle snapshots to Parquet.

Polls the bulk OHLC quotes endpoint every 60 seconds during market hours.
Stores completed 1m candles (prev_ohlc) for comparison with WebSocket I1
and Historical API data to determine which pipeline the quotes endpoint uses.

Usage:
    python tools/record_ohlc_quotes.py

Output: data/sidecar/ohlc_quotes/ohlc_quotes_{YYYYMMDD}.parquet
Schema: symbol, minute_ts, open, high, low, close, volume

Requires: UPSTOX_ACCESS_TOKEN env var or broker/upstox/token.txt
"""

import sys
import time
import signal
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List
from urllib.parse import quote as url_quote

import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Output directory
DATA_DIR = ROOT / "data" / "sidecar" / "ohlc_quotes"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Market hours (IST)
MARKET_OPEN_H, MARKET_OPEN_M = 9, 15
MARKET_CLOSE_H, MARKET_CLOSE_M = 15, 30

# Batch size: keep URL under ~8KB. Each key ~25 chars + comma.
# Conservative: 250 per request (250 * 26 = 6500 chars for keys alone).
BATCH_SIZE = 250

# Poll interval in seconds
POLL_INTERVAL_SEC = 60

# Periodic flush interval (every N polls, write to disk as backup)
FLUSH_EVERY_N_POLLS = 10


def load_access_token() -> str:
    """Load Upstox access token from env or token.txt."""
    import os
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    token = os.getenv("UPSTOX_ACCESS_TOKEN", "")
    if token:
        return token

    token_file = ROOT / "broker" / "upstox" / "token.txt"
    if token_file.exists():
        token = token_file.read_text().strip()
        if token:
            return token

    raise RuntimeError("No UPSTOX_ACCESS_TOKEN found in env or broker/upstox/token.txt")


def load_instrument_keys() -> Dict[str, str]:
    """Load NSE EQ instrument keys. Returns {symbol: instrument_key}."""
    from broker.upstox.upstox_data_client import UpstoxDataClient
    client = UpstoxDataClient()
    sym_to_key = {}
    for sym, inst in client._sym2inst.items():
        sym_to_key[sym] = inst.instrument_key
    return sym_to_key


def fetch_ohlc_batch(
    keys: List[str], token: str, interval: str = "I1", max_retries: int = 3,
) -> Dict[str, dict]:
    """Fetch OHLC quotes for a batch of instrument keys.

    Returns the 'data' dict from the API response.
    Retries on HTTP 429 (rate limit) with exponential backoff.
    """
    keys_param = ",".join(keys)
    url = f"https://api.upstox.com/v3/market-quote/ohlc?instrument_key={keys_param}&interval={interval}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=15)
        except requests.exceptions.RequestException as e:
            print(f"  WARN: Request failed (attempt {attempt + 1}): {e}")
            time.sleep(2 ** attempt)
            continue

        if resp.status_code == 200:
            return resp.json().get("data", {})
        elif resp.status_code == 429:
            wait = 2 ** (attempt + 1)
            print(f"  WARN: Rate limited (429), waiting {wait}s...")
            time.sleep(wait)
        elif resp.status_code == 401:
            print(f"  ERROR: Token expired/invalid (401). Cannot recover.")
            return {}
        else:
            print(f"  WARN: HTTP {resp.status_code} for batch of {len(keys)} (attempt {attempt + 1})")
            time.sleep(1)

    return {}


class OHLCQuotesRecorder:
    """Records OHLC quotes snapshots to Parquet throughout the trading day."""

    def __init__(self, token: str, sym_to_key: Dict[str, str]):
        self._token = token
        self._sym_to_key = sym_to_key  # "NSE:RELIANCE" -> "NSE_EQ|INE002A01018"

        # Reverse map: instrument_key -> symbol (for API response parsing)
        # API response uses instrument_token field with pipe format: "NSE_EQ|INE002A01018"
        self._key_to_sym = {v: k for k, v in sym_to_key.items()}

        # Completed candles: (symbol, minute_ts_epoch_ms) -> (open, high, low, close, volume)
        self._candles: Dict[Tuple[str, int], Tuple[float, float, float, float, int]] = {}

        self._today = datetime.now().strftime("%Y%m%d")
        self._file_path = DATA_DIR / f"ohlc_quotes_{self._today}.parquet"
        self._poll_count = 0
        self._running = False
        self._token_valid = True

    def _record_ohlc(self, sym: str, ohlc: dict) -> bool:
        """Record a single OHLC entry. Returns True if new candle added."""
        if not ohlc or not ohlc.get("ts"):
            return False
        ts = int(ohlc["ts"])
        key = (sym, ts)
        if key in self._candles:
            return False
        self._candles[key] = (
            float(ohlc["open"]),
            float(ohlc["high"]),
            float(ohlc["low"]),
            float(ohlc["close"]),
            int(ohlc.get("volume", 0)),
        )
        return True

    def poll_once(self) -> int:
        """Fetch OHLC quotes for all symbols. Returns number of new candles recorded."""
        if not self._token_valid:
            return 0

        all_keys = list(self._sym_to_key.values())
        batches = [all_keys[i:i + BATCH_SIZE] for i in range(0, len(all_keys), BATCH_SIZE)]

        new_candles = 0
        for i, batch in enumerate(batches):
            try:
                data = fetch_ohlc_batch(batch, self._token)
            except Exception as e:
                print(f"  WARN: Batch {i+1}/{len(batches)} failed: {e}")
                continue
            if not data:
                # Check if all batches fail (token expired)
                if i == 0:
                    self._token_valid = False
                    print("  ERROR: First batch returned no data, token may be expired")
                continue

            for upstox_sym, quote in data.items():
                # Map instrument_token (pipe format) back to our symbol format
                inst_key = quote.get("instrument_token", "")
                sym = self._key_to_sym.get(inst_key, upstox_sym)

                # Record both prev_ohlc (completed) and live_ohlc (running/final)
                if self._record_ohlc(sym, quote.get("prev_ohlc")):
                    new_candles += 1
                if self._record_ohlc(sym, quote.get("live_ohlc")):
                    new_candles += 1

            # Small delay between batches to be gentle on API
            if i < len(batches) - 1:
                time.sleep(0.2)

        self._poll_count += 1
        return new_candles

    def run(self) -> None:
        """Poll continuously during market hours."""
        self._running = True
        n_batches = (len(self._sym_to_key) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"OHLC_QUOTES_RECORDER | Starting, {len(self._sym_to_key)} symbols, output: {self._file_path}")
        print(f"OHLC_QUOTES_RECORDER | Batch size: {BATCH_SIZE}, batches/poll: {n_batches}, interval: {POLL_INTERVAL_SEC}s")

        while self._running:
            now = datetime.now()

            market_open = now.replace(hour=MARKET_OPEN_H, minute=MARKET_OPEN_M, second=0)
            market_close = now.replace(hour=MARKET_CLOSE_H, minute=MARKET_CLOSE_M, second=0)

            if now < market_open:
                wait = (market_open - now).total_seconds()
                print(f"OHLC_QUOTES_RECORDER | Waiting {wait:.0f}s for market open...")
                time.sleep(min(wait, 60))
                continue

            if now > market_close + timedelta(minutes=2):
                print("OHLC_QUOTES_RECORDER | Market closed, finalizing...")
                break

            if not self._token_valid:
                print("OHLC_QUOTES_RECORDER | Token expired, stopping.")
                break

            # Poll
            t0 = time.perf_counter()
            new = self.poll_once()
            elapsed = time.perf_counter() - t0
            print(
                f"OHLC_QUOTES_RECORDER | Poll #{self._poll_count}: "
                f"+{new} new, {len(self._candles)} total, "
                f"{elapsed:.1f}s"
            )

            # Periodic flush to disk (crash recovery)
            if self._poll_count % FLUSH_EVERY_N_POLLS == 0:
                self.finalize()
                print(f"OHLC_QUOTES_RECORDER | Periodic flush to disk")

            # Sleep until next poll
            sleep_time = max(0, POLL_INTERVAL_SEC - elapsed)
            time.sleep(sleep_time)

        self.finalize()

    def stop(self):
        self._running = False

    def finalize(self) -> Optional[Path]:
        """Write all candles to parquet."""
        if not self._candles:
            print("OHLC_QUOTES_RECORDER | No candles recorded")
            return None

        rows = []
        for (sym, ts_epoch), (o, h, l, c, v) in self._candles.items():
            rows.append({
                "symbol": sym,
                "minute_ts": ts_epoch,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
            })

        df = pd.DataFrame(rows)
        df.to_parquet(self._file_path, compression="snappy", index=False)

        unique_symbols = df["symbol"].nunique()
        size_mb = self._file_path.stat().st_size / (1024 * 1024)

        print(
            f"OHLC_QUOTES_RECORDER | Saved: {len(df):,} candles "
            f"({unique_symbols} symbols) -> {self._file_path.name} ({size_mb:.1f} MB)"
        )
        return self._file_path


def main():
    print("=" * 60)
    print("OHLC QUOTES RECORDER")
    print("Records Upstox REST OHLC quotes for comparison with WebSocket I1")
    print("=" * 60)

    token = load_access_token()
    print(f"Token: {token[:20]}...{token[-10:]}")

    sym_to_key = load_instrument_keys()
    print(f"Loaded {len(sym_to_key)} instrument keys")

    recorder = OHLCQuotesRecorder(token, sym_to_key)

    # Handle Ctrl+C
    def signal_handler(sig, frame):
        print("\nInterrupt received, finalizing...")
        recorder.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    recorder.run()


if __name__ == "__main__":
    main()
