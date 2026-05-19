"""Generate synthetic mini-run for gauntlet tests.

Structure: {fixture_root}/{YYYY-MM-DD}/analytics.jsonl + trade_report.csv
Each session has 20 trades across 2 setup types, mixed outcomes.
"""
import json
import random
from pathlib import Path
import csv


def make_mini_run(root: Path, dates: list, trades_per_session: int = 20, seed: int = 42):
    """Create synthetic session dirs. dates: list of YYYY-MM-DD strings."""
    random.seed(seed)
    root.mkdir(parents=True, exist_ok=True)

    for date in dates:
        sdir = root / date
        sdir.mkdir(exist_ok=True)

        # trade_report.csv with planned trades
        csv_path = sdir / "trade_report.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "run_id", "trade_id", "symbol", "decision_ts", "setup_type",
                "regime", "minute_of_day", "cap_segment", "executed",
            ])
            for i in range(trades_per_session):
                trade_id = f"{date}_t{i}"
                setup = random.choice(["setup_a_long", "setup_b_short"])
                regime = random.choice(["chop", "trend_up", "trend_down", "squeeze"])
                minute = random.choice([555, 600, 720, 840, 900])  # 9:15, 10:00, 12:00, 14:00, 15:00
                cap = random.choice(["large_cap", "mid_cap", "small_cap"])
                w.writerow([
                    "test_run", trade_id, f"NSE:TICKER{i}", f"{date}T09:30:00",
                    setup, regime, minute, cap, True,
                ])

        # analytics.jsonl — one is_final_exit=True entry per trade
        jl_path = sdir / "analytics.jsonl"
        with open(jl_path, "w") as f:
            for i in range(trades_per_session):
                trade_id = f"{date}_t{i}"
                pnl = random.gauss(0, 500)  # zero-mean normal
                entry = {
                    "trade_id": trade_id,
                    "symbol": f"NSE:TICKER{i}",
                    "setup_type": random.choice(["setup_a_long", "setup_b_short"]),
                    "regime": random.choice(["chop", "trend_up", "trend_down", "squeeze"]),
                    "pnl": pnl,
                    "total_trade_pnl": pnl,
                    "is_final_exit": True,
                    "reason": "hard_sl" if pnl < 0 else "target_t1",
                    "actual_entry_price": 100.0,
                    "exit_price": 100.0 + pnl / 10,
                    "qty": 10,
                    "mae": min(pnl, 0) * 1.5,
                    "mfe": max(pnl, 0) * 1.5,
                    "r_multiple": pnl / 500,
                    "bars_held": random.randint(1, 20),
                }
                f.write(json.dumps(entry) + "\n")
                # Also write a partial exit (is_final_exit=False) to test filter
                if i % 3 == 0:
                    partial = dict(entry)
                    partial["trade_id"] = trade_id  # same trade_id
                    partial["is_final_exit"] = False
                    partial["reason"] = "t1_partial"
                    partial["pnl"] = 100
                    partial.pop("total_trade_pnl", None)
                    f.write(json.dumps(partial) + "\n")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/edge_discovery/fixtures/mini_run")
    dates = [f"2023-{m:02d}-{d:02d}" for m in range(1, 7) for d in [1, 15]]
    make_mini_run(out, dates)
    print(f"Fixture written to {out} with {len(dates)} sessions")
