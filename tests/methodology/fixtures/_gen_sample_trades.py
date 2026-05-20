"""One-time fixture generator. Run with:
    .venv/Scripts/python tests/methodology/fixtures/_gen_sample_trades.py
"""
import csv
from datetime import date, timedelta
import random

random.seed(20260519)

OUT = "tests/methodology/fixtures/sample_trades.csv"
START = date(2023, 1, 1)
END = date(2026, 3, 31)

rows = []
d = START
while d <= END:
    if d.weekday() < 5:
        for _ in range(5):
            if d < date(2025, 1, 1):
                pnl = random.gauss(0.15, 0.40)
            else:
                pnl = random.gauss(-0.05, 0.50)
            rows.append({
                "signal_date": d.isoformat(),
                "symbol": f"SYM{random.randint(1, 50):02d}",
                "pnl_pct": round(pnl, 4),
            })
    d += timedelta(days=1)

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["signal_date", "symbol", "pnl_pct"])
    w.writeheader()
    w.writerows(rows)

print(f"wrote {len(rows)} rows to {OUT}")
