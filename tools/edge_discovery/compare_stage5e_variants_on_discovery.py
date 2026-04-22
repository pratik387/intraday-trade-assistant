"""In-sample variant comparison: Stage 5e FULL vs Stage 5e SIMPLE on Discovery (2023-24).

Methodology fix: all architecture/variant comparison must happen on in-sample
Discovery data. Only ONCE the winning variant is chosen do we validate on 2025
OOS. Running variants on 2025 would consume OOS signal.

This script runs the full chain 5b -> 5c -> {5d, 5e-full, 5e-simple} on Discovery
and reports the side-by-side comparison.
"""
from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pandas as pd

from tools.edge_discovery.data_loader import load_run
from tools.edge_discovery.stages.stage5b_ruleset_simulation import apply_filter
from tools.edge_discovery.stages.stage5c_cross_sectional_simulation import run_stage5c
from tools.edge_discovery.stages.stage5d_conviction_simulation import run_stage5d
from tools.edge_discovery.stages.stage5e_budgeted_selector import run_stage5e
from tools.conviction.build_training_dataset import load_trade_report_features
from services.conviction.scorer import XGBoostScorer

ROOT = Path(__file__).parent.parent.parent
BACKTEST_DIR = ROOT / "cloud_results" / "20260419_discovery"
DISCOVERY_SURVIVORS = ROOT / "analysis" / "edge_discovery_runs" / "2026-04-20" / "stage3_survivors.json"
FILL_SCRIPT = ROOT / "tools" / "edge_discovery" / "fill_narratives.py"
MONTHLY_DIR = ROOT / "backtest-cache-download" / "monthly"
ADV_PARQUET = ROOT / "models" / "gauntlet" / "stage5e_adv_rupees.parquet"
OUT_DIR = ROOT / "docs" / "edge_discovery" / "2026-04-22-discovery-variants"

CV_MODEL = ROOT / "models" / "conviction" / "2026-04-22-universal-xgboost.json"
CV_FEATURE_SPEC = ROOT / "models" / "conviction" / "2026-04-22-feature-spec.json"


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
    print(f"Loading Discovery backtest: {BACKTEST_DIR}")
    data = load_run(BACKTEST_DIR)
    trades = data.trades
    print(f"  {len(trades):,} trades / {data.sessions_loaded} sessions")

    # Merge trade_report features
    print("Merging trade_report features...")
    tr_feats = load_trade_report_features(BACKTEST_DIR)
    trades = trades.merge(tr_feats, on="trade_id", how="left", suffixes=("", "_tr"))

    # Stage 5b: 90-rule filter
    rules = load_approved_rules()
    trades["setup"] = trades["setup_type"]
    trades["symbol_raw"] = trades["symbol"].astype(str).str.replace("NSE:", "", regex=False)
    stage5b = apply_filter(trades, rules)
    print(f"Stage 5b (90 rules): {len(stage5b):,} trades")

    stage5b = stage5b.copy()
    stage5b["decision_ts"] = (
        pd.to_datetime(stage5b["session_date_dt"])
        + pd.to_timedelta(stage5b["minute_of_day"].astype(int), unit="m")
    ).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Stage 5c
    print("Loading OHLCV (2023+2024 monthly feathers) ...")
    parts = []
    for f in sorted(MONTHLY_DIR.glob("2023_*_5m_enriched.feather")) + sorted(
        MONTHLY_DIR.glob("2024_*_5m_enriched.feather")
    ):
        df = pd.read_feather(f, columns=["date", "symbol", "volume"])
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
    trade_syms = set(stage5b["symbol_raw"].unique())
    ohlcv = ohlcv[ohlcv["symbol"].isin(trade_syms)].reset_index(drop=True)
    print(f"  ohlcv rows (after sym filter): {len(ohlcv):,}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cs_cfg = json.loads((ROOT / "config" / "configuration.json").read_text(encoding="utf-8"))["cross_sectional_gate"]
    print("Running Stage 5c ...")
    s5c = run_stage5c(
        trades=stage5b,
        ohlcv=ohlcv,
        cfg=cs_cfg,
        report_path=OUT_DIR / "07-stage5c-discovery.md",
        summary_json=OUT_DIR / "stage5c_discovery.json",
    )
    stage5c_filtered = s5c["filtered_trades"]
    chain_input = stage5c_filtered[stage5c_filtered["allowed"].astype(bool)].reset_index(drop=True)
    print(f"Stage 5c: {len(chain_input):,} trades pass")

    # Stage 5d: FIFO cap 50
    print("Running Stage 5d (FIFO) ...")
    scorer = XGBoostScorer(CV_MODEL, CV_FEATURE_SPEC)
    cv_cfg = json.loads((ROOT / "config" / "configuration.json").read_text(encoding="utf-8"))["conviction_gate"]
    run_stage5d(
        trades=chain_input,
        scorer=scorer,
        cfg={"enabled": True, "daily_cap": int(cv_cfg["daily_cap"]), "min_predicted_r": float(cv_cfg["min_predicted_r"])},
        report_path=OUT_DIR / "08-stage5d-discovery.md",
        summary_json=OUT_DIR / "stage5d_discovery.json",
    )

    # Stage 5e variants
    adv_df = pd.read_parquet(ADV_PARQUET)
    adv_map = {
        (row.symbol, row.date_only): float(row.adv_rupees_20d)
        for row in adv_df.itertuples()
        if pd.notna(row.adv_rupees_20d)
    }

    print("Running Stage 5e FULL (all 6 constraints) ...")
    run_stage5e(
        trades=chain_input,
        adv_map=adv_map,
        cfg={},
        report_path=OUT_DIR / "09-stage5e-FULL-discovery.md",
        summary_json=OUT_DIR / "stage5e_full_discovery.json",
    )

    print("Running Stage 5e SIMPLE (embargo + time buckets only) ...")
    simple_cfg = {
        "enable_embargo": True,
        "enable_bucket_quota": True,
        "enable_adv_cap": False,
        "enable_bar_cap": False,
        "enable_rate_limit": False,
        "enable_concurrency_cap": False,
    }
    run_stage5e(
        trades=chain_input,
        adv_map=adv_map,
        cfg=simple_cfg,
        report_path=OUT_DIR / "09-stage5e-SIMPLE-discovery.md",
        summary_json=OUT_DIR / "stage5e_simple_discovery.json",
    )

    # Summary
    print()
    print("=" * 80)
    print("DISCOVERY (in-sample 2023-24, 484 sessions): FIFO vs 5e-FULL vs 5e-SIMPLE")
    print("=" * 80)
    d = json.loads((OUT_DIR / "stage5d_discovery.json").read_text())
    ef = json.loads((OUT_DIR / "stage5e_full_discovery.json").read_text())
    es = json.loads((OUT_DIR / "stage5e_simple_discovery.json").read_text())
    fmt = "{:<22} {:>14} {:>14} {:>14}"
    print(fmt.format("Metric", "5d FIFO", "5e FULL", "5e SIMPLE"))
    print("-" * 72)
    for k in ["n_trades", "trades_per_day", "total_pnl", "pf", "wr_pct",
              "session_sharpe", "losing_days_pct"]:
        print(fmt.format(k, d["after"][k], ef["after"][k], es["after"][k]))
    print()
    for label, v in [("5e FULL  ", ef), ("5e SIMPLE", es)]:
        print(f"-- {label} vs 5d FIFO --")
        print(f"   PF delta:     {v['after']['pf'] - d['after']['pf']:+.3f}")
        print(f"   Sharpe delta: {v['after']['session_sharpe'] - d['after']['session_sharpe']:+.3f}")
        print(f"   PnL delta:    Rs {v['after']['total_pnl'] - d['after']['total_pnl']:+,.0f}")


if __name__ == "__main__":
    main()
