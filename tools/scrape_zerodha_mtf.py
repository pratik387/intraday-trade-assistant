"""Refresh Zerodha MTF approved-securities list.

Source endpoint (discovered 2026-05-21 by inspecting
https://zerodha.com/static/js/mtf_approved_securities.js):
    https://public.zrd.sh/crux/approved-mtf-securities.json

Returns ~1,489 entries with fields:
    isin, tradingsymbol, category (fo|non_fo|non_categorized|etf),
    margin (% of notional required from client), leverage (1 / margin_pct).

Run periodically (weekly recommended — list updates daily but slowly).
Output: data/mtf_universe/approved_mtf_securities_YYYY-MM-DD.json
"""
from __future__ import annotations

import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OUT_DIR = _REPO_ROOT / "data" / "mtf_universe"
_URL = "https://public.zrd.sh/crux/approved-mtf-securities.json"


def main() -> int:
    today = date.today().isoformat()
    out_path = _OUT_DIR / f"approved_mtf_securities_{today}.json"
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {_URL} ...")
    req = urllib.request.Request(_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)

    if not isinstance(data, list) or not data:
        print(f"ERROR: unexpected payload (type={type(data).__name__}, len={len(data) if hasattr(data, '__len__') else 'n/a'})", file=sys.stderr)
        return 1

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Summary
    by_cat: dict[str, int] = {}
    lev_total = 0.0
    for e in data:
        by_cat[e.get("category", "?")] = by_cat.get(e.get("category", "?"), 0) + 1
        lev_total += float(e.get("leverage", 0.0))

    print(f"Saved: {out_path}  ({len(data):,} entries)")
    print("Categories:")
    for cat, n in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {n}")
    print(f"Mean leverage: {lev_total / len(data):.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
