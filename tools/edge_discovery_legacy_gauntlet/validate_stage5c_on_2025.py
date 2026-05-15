"""Validate Stage 5c (F1 RVOL + F2 Crowdedness) on 2025 OOS data using the
Discovery-derived 90 approved rules.

This is the real OOS test of the cross-sectional thresholds. Discovery results
missed spec targets (PF +2.5% vs +5% target, session Sharpe regressed). 2025
data tells us whether the filter generalizes out-of-sample.

Uses:
  - Discovery's 90 approved rules (stage3_survivors.json + fill_narratives.REJECTED_RULES)
  - 2025 trades loaded via data_loader.load_run(20260421-134338_full)
  - 2025 OHLCV from backtest-cache-download/monthly/2025_*.feather + historical
    for RVOL warm-up window
  - Config-driven thresholds from cross_sectional_gate (UNCHANGED from Discovery)
"""
import json
import importlib.util
from pathlib import Path

import pandas as pd

from tools.edge_discovery_legacy_gauntlet.data_loader import load_run
from tools.edge_discovery_legacy_gauntlet.stages.stage5b_ruleset_simulation import apply_filter
from tools.edge_discovery_legacy_gauntlet.stages.stage5c_cross_sectional_simulation import run_stage5c

ROOT = Path(__file__).parent.parent.parent
BACKTEST_DIR = ROOT / "20260421-134338_full"
DISCOVERY_SURVIVORS = ROOT / "analysis" / "edge_discovery_runs" / "2026-04-20" / "stage3_survivors.json"
FILL_SCRIPT = ROOT / "tools" / "edge_discovery" / "fill_narratives.py"
MONTHLY_DIR = ROOT / "backtest-cache-download" / "monthly"
OUT_DIR = ROOT / "docs" / "edge_discovery" / "2026-04-21-validation"


def load_approved_rules():
    spec = importlib.util.spec_from_file_location("fill_narratives", FILL_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rejected = mod.REJECTED_RULES
    survivors = json.loads(DISCOVERY_SURVIVORS.read_text(encoding="utf-8"))
    rules = []
    for cell in survivors["details"]:
        if not cell.get("passed"):
            continue
        rid = f"{cell['setup']}__{cell['conditioner']}={cell['cell_value']}"
        if rid in rejected:
            continue
        rules.append({
            "setup": cell["setup"],
            "conditions": list(zip(cell["conditioner"].split("+"), cell["cell_value"].split("+"))),
        })
    return rules


def main():
    print(f"Loading 2025 backtest: {BACKTEST_DIR}")
    data = load_run(BACKTEST_DIR)
    trades = data.trades
    print(f"Loaded {len(trades):,} trades across {data.sessions_loaded} sessions")

    approved_rules = load_approved_rules()
    print(f"Using {len(approved_rules)} Discovery-approved rules")

    # Add symbol_raw + setup (stage5c expected columns)
    trades["setup"] = trades["setup_type"]
    trades["symbol_raw"] = trades["symbol"].astype(str).str.replace("NSE:", "", regex=False)

    # Apply rule filter (Stage 5b semantic)
    filtered = apply_filter(trades, approved_rules)
    print(f"After rule filter: {len(filtered):,} trades")

    # Derive decision_ts
    if "decision_ts" not in filtered.columns:
        filtered = filtered.copy()
        filtered["decision_ts"] = (
            pd.to_datetime(filtered["session_date_dt"])
            + pd.to_timedelta(filtered["minute_of_day"].astype(int), unit="m")
        ).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Load OHLCV (2024 + 2025 monthly feathers to cover RVOL warm-up + 2025 bars)
    trade_syms = set(filtered["symbol_raw"].unique())
    print(f"Loading OHLCV for {len(trade_syms)} symbols...")
    parts = []
    for f in sorted(MONTHLY_DIR.glob("2024_*_5m_enriched.feather")) + \
             sorted(MONTHLY_DIR.glob("2025_*_5m_enriched.feather")):
        df = pd.read_feather(f, columns=["date", "symbol", "volume"])
        df = df[df["symbol"].isin(trade_syms)]
        parts.append(df)
    ohlcv = pd.concat(parts, ignore_index=True)
    del parts
    if ohlcv["date"].dt.tz is not None:
        ohlcv["ts"] = ohlcv["date"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    else:
        ohlcv["ts"] = ohlcv["date"]
    ohlcv["mod"] = (ohlcv["ts"].dt.hour * 60 + ohlcv["ts"].dt.minute).astype("int16")
    ohlcv["date_only"] = ohlcv["ts"].dt.date
    ohlcv = ohlcv[["symbol", "date_only", "mod", "volume"]]
    print(f"OHLCV rows: {len(ohlcv):,}")

    # Load Stage 5c config
    cfg = json.loads((ROOT / "config" / "configuration.json").read_text(encoding="utf-8"))["cross_sectional_gate"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = run_stage5c(
        trades=filtered,
        ohlcv=ohlcv,
        cfg=cfg,
        report_path=OUT_DIR / "07-cross-sectional-simulation-2025.md",
        summary_json=OUT_DIR / "stage5c_simulation_2025.json",
    )
    print()
    print("=== OOS (2025) Stage 5c results ===")
    print(f"Before: N={result['before']['n_trades']:,} trades/day={result['before']['trades_per_day']}  PF={result['before']['pf']}  Sharpe={result['before']['session_sharpe']}")
    print(f"After:  N={result['after']['n_trades']:,} trades/day={result['after']['trades_per_day']}  PF={result['after']['pf']}  Sharpe={result['after']['session_sharpe']}")
    print(f"Delta: PF={result['delta']['pf_delta']:+.3f}  Sharpe={result['delta']['session_sharpe_delta']:+.3f}  trades/day={result['delta']['trades_per_day_delta']:+.1f}")


if __name__ == "__main__":
    main()
