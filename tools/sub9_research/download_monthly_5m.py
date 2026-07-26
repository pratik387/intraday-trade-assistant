"""Download monthly 5m OHLCV feathers for the NSE EQ universe into
backtest-cache-download/monthly/ (base OHLCV columns only — enrichment
columns like vwap/adx/rsi are deliberately absent so consumers that need
them fail loudly instead of reading blanks).

Usage:
    python tools/sub9_research/download_monthly_5m.py 2026-08 [2026-09 ...]

Rate: 8 rps. The Upstox historical endpoint 429-storms at 40 rps for
months-old data (2026-07-26: 12.5k retries, 14% of symbols returned) even
though the intraday endpoint sustains 40 rps — do not raise this.
"""
from __future__ import annotations
import sys
import asyncio
import calendar
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from broker.upstox.upstox_data_client import UpstoxDataClient  # noqa: E402

MONTHLY = ROOT / "backtest-cache-download" / "monthly"
RPS = 8


def month_range(arg: str) -> tuple[str, str]:
    y, m = map(int, arg.split("-"))
    last = min(calendar.monthrange(y, m)[1], 31)
    end = date(y, m, last)
    today = date.today()
    if end > today:
        end = today
    return f"{y:04d}-{m:02d}-01", end.isoformat()


def main() -> int:
    months = sys.argv[1:]
    if not months:
        print(__doc__)
        return 2
    sdk = UpstoxDataClient()
    universe = list(getattr(sdk, "_equity_instruments", []))
    print(f"universe {len(universe)}", flush=True)
    for arg in months:
        f, t = month_range(arg)
        m = asyncio.run(sdk.async_fetch_historical_5m_batch(
            universe, f, t, concurrency=RPS, rps=RPS))
        missing = [s for s in universe if m.get(s) is None or m[s].empty]
        if missing:
            print(f"{arg}: retrying {len(missing)} missing", flush=True)
            m2 = asyncio.run(sdk.async_fetch_historical_5m_batch(
                missing, f, t, concurrency=4, rps=4))
            m.update({k: v for k, v in m2.items() if v is not None and not v.empty})
        frames = []
        for sym, df in m.items():
            if df is None or df.empty:
                continue
            d = df.reset_index()
            d = d.rename(columns={d.columns[0]: "date"})
            d["symbol"] = sym.replace("NSE:", "")
            frames.append(d[["date", "symbol", "open", "high", "low", "close", "volume"]])
        out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["symbol", "date"])
        out["date"] = pd.to_datetime(out["date"])
        if out["date"].dt.tz is not None:
            out["date"] = out["date"].dt.tz_localize(None)
        out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
        p = MONTHLY / f"{arg.replace('-', '_')}_5m_enriched.feather"
        out.to_feather(p)
        print(f"wrote {p.name}: {len(out)} rows, {out['symbol'].nunique()} symbols, "
              f"{str(out['date'].min())[:10]} .. {str(out['date'].max())[:10]}", flush=True)
    print("DOWNLOAD COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
