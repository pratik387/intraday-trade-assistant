"""Download NSE's daily price-band list into one parquet.

NSE publishes ``sec_list_DDMMYYYY.csv`` every trading day (columns Symbol, Series,
Security Name, Band, Remarks). The file dated D carries the bands IN FORCE ON THE
NEXT TRADING DAY: verified 2026-10-29 against eq_band_changes and price action
(VNL band 10->5 effective 10-28: sec_list_27102026 already says 5, and VNL still
moved -10% on 10-27). So a session S uses the latest file dated strictly before S
(services/price_bands.py) -- no holiday calendar needed, and the file is published
the evening before, so live and backtest read the same thing.

Output: data/price_bands/price_bands.parquet with columns
    file_date (datetime64), symbol (str), series (str), band_pct (float, NaN = "No Band")

Usage:
    python tools/fetch_nse_price_bands.py                     # last 10 calendar days
    python tools/fetch_nse_price_bands.py --from 2023-01-01   # backfill
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.time_util import _now_naive_ist  # noqa: E402


def _cfg() -> dict:
    cfg = json.load(open(ROOT / "config" / "configuration.json", encoding="utf-8"))
    return cfg["price_bands"]


def _fetch(day: date, cfg: dict) -> pd.DataFrame | None:
    url = cfg["source_url_template"].format(ddmmyyyy=day.strftime("%d%m%Y"))
    r = requests.get(url, headers={"User-Agent": cfg["http_user_agent"]},
                     timeout=float(cfg["http_timeout_sec"]))
    if r.status_code == 404 or r.text.lstrip().startswith("<"):
        return None  # not a trading day / not published yet (NSE returns an HTML page)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip() for c in df.columns]
    missing = {"Symbol", "Series", "Band"} - set(df.columns)
    if missing:
        raise RuntimeError(f"{url}: unexpected columns {list(df.columns)} (missing {missing})")
    band = pd.to_numeric(df["Band"].astype(str).str.strip(), errors="coerce")
    bad = df.loc[band.isna() & (df["Band"].astype(str).str.strip() != "No Band"), "Band"]
    if len(bad):
        raise RuntimeError(f"{url}: unparseable Band values {sorted(set(bad.astype(str)))[:5]}")
    return pd.DataFrame({
        "file_date": pd.Timestamp(day),
        "symbol": df["Symbol"].astype(str).str.strip(),
        "series": df["Series"].astype(str).str.strip(),
        "band_pct": band,
    })


def _save(out: Path, have: pd.DataFrame, new: list) -> pd.DataFrame:
    allf = pd.concat(([have] if len(have) else []) + new, ignore_index=True)
    allf["file_date"] = pd.to_datetime(allf["file_date"])
    allf = allf.drop_duplicates(["file_date", "symbol", "series"], keep="last")
    allf = allf.sort_values(["file_date", "symbol", "series"]).reset_index(drop=True)
    tmp = out.with_suffix(".tmp")
    allf.to_parquet(tmp, index=False)
    tmp.replace(out)                       # atomic: a reader never sees half a file
    return allf


def main() -> int:
    cfg = _cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="d0")
    ap.add_argument("--to", dest="d1")
    a = ap.parse_args()
    today = _now_naive_ist().date()  # only picks which files to download
    d1 = date.fromisoformat(a.d1) if a.d1 else today
    d0 = date.fromisoformat(a.d0) if a.d0 else d1 - timedelta(days=int(cfg["refresh_lookback_days"]))

    out = ROOT / cfg["parquet_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    have = pd.read_parquet(out) if out.exists() else pd.DataFrame(
        columns=["file_date", "symbol", "series", "band_pct"])
    have_days = set(pd.to_datetime(have["file_date"]).dt.date) if len(have) else set()

    new, d, fetched, added = [], d0, 0, 0
    while d <= d1:
        if d.weekday() < 5 and d not in have_days:
            df = _fetch(d, cfg)
            fetched += 1
            if df is not None:
                added += 1
                new.append(df)
                print(f"{d} {len(df)} rows", flush=True)
                if len(new) >= int(cfg["save_every_files"]):   # a long backfill survives interruption
                    have = _save(out, have, new)
                    have_days |= {x["file_date"].iloc[0].date() for x in new}
                    new = []
            time.sleep(float(cfg["http_pause_sec"]))
        d += timedelta(days=1)

    if new:
        have = _save(out, have, new)
    latest = pd.to_datetime(have["file_date"]).max().date() if len(have) else None
    print(f"requested {fetched} days, added {added} files; latest file_date = {latest}")
    if latest is None:
        print("ERROR: no price-band file available", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
