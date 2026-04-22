"""OOS validate Stage 5e (Budgeted Selector) on 2025 data.

Runs the full chain 5b -> 5c -> {5d, 5e} on 2025 (20260421-134338_full),
comparing FIFO cap-50 (Stage 5d) head-to-head with the illiquid-aware
BudgetedSelector (Stage 5e) that beat 5d on 2023-24 Discovery.

If Stage 5e's Discovery-period wins (+2.1% PF, +20% Sharpe, +7% gross PnL)
hold up on 2025 OOS, the live gate should adopt its logic.

Reuses existing stage modules (stage5b ruleset filter, stage5c simulate_filter,
stage5d ConvictionGate, stage5e BudgetedSelector).
"""
from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from datetime import date

import pandas as pd

from tools.edge_discovery.data_loader import load_run
from tools.edge_discovery.stages.stage5b_ruleset_simulation import apply_filter
from tools.edge_discovery.stages.stage5c_cross_sectional_simulation import run_stage5c
from tools.edge_discovery.stages.stage5d_conviction_simulation import run_stage5d
from tools.edge_discovery.stages.stage5e_budgeted_selector import run_stage5e
from tools.conviction.build_training_dataset import load_trade_report_features
from services.conviction.scorer import XGBoostScorer

ROOT = Path(__file__).parent.parent.parent
BACKTEST_DIR = ROOT / "20260421-134338_full"
DISCOVERY_SURVIVORS = ROOT / "analysis" / "edge_discovery_runs" / "2026-04-20" / "stage3_survivors.json"
FILL_SCRIPT = ROOT / "tools" / "edge_discovery" / "fill_narratives.py"
MONTHLY_DIR = ROOT / "backtest-cache-download" / "monthly"
ADV_PARQUET = ROOT / "models" / "gauntlet" / "stage5e_adv_rupees.parquet"
OUT_DIR = ROOT / "docs" / "edge_discovery" / "2026-04-22-validation"

# Conviction model artifacts (for Stage 5d)
CV_MODEL = ROOT / "models" / "conviction" / "2026-04-22-universal-xgboost.json"
CV_FEATURE_SPEC = ROOT / "models" / "conviction" / "2026-04-22-feature-spec.json"

VAL_START = date(2025, 1, 1)
VAL_END = date(2025, 9, 30)


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


def load_stage5c_config():
    cfg_full = json.loads((ROOT / "config" / "configuration.json").read_text(encoding="utf-8"))
    return cfg_full["cross_sectional_gate"]


def main():
    print(f"Loading 2025 backtest: {BACKTEST_DIR}")
    data = load_run(BACKTEST_DIR)
    trades = data.trades
    print(f"  loaded {len(trades):,} trades / {data.sessions_loaded} sessions")

    # Restrict to 2025 validation window
    trades = trades[
        (trades["session_date_dt"] >= VAL_START)
        & (trades["session_date_dt"] <= VAL_END)
    ].copy()
    print(f"  filtered to {VAL_START}..{VAL_END}: {len(trades):,} trades")

    # Merge trade_report features (volume5, close5, plan_notional, last_exit_ts)
    print("Merging trade_report features...")
    tr_feats = load_trade_report_features(BACKTEST_DIR)
    trades = trades.merge(tr_feats, on="trade_id", how="left", suffixes=("", "_tr"))

    # Stage 5b: 90-rule filter
    rules = load_approved_rules()
    trades["setup"] = trades["setup_type"]
    trades["symbol_raw"] = trades["symbol"].astype(str).str.replace("NSE:", "", regex=False)
    stage5b_out = apply_filter(trades, rules)
    print(f"Stage 5b (90 rules): {len(stage5b_out):,} trades")

    # Synthesize decision_ts for downstream stages
    stage5b_out = stage5b_out.copy()
    stage5b_out["decision_ts"] = (
        pd.to_datetime(stage5b_out["session_date_dt"])
        + pd.to_timedelta(stage5b_out["minute_of_day"].astype(int), unit="m")
    ).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Stage 5c: cross-sectional F1+F2 on 2025 ohlcv
    print("Loading 2025 OHLCV (monthly feathers)...")
    parts = []
    for f in sorted(MONTHLY_DIR.glob("2024_*_5m_enriched.feather")) + sorted(
        MONTHLY_DIR.glob("2025_*_5m_enriched.feather")
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
    # Restrict to symbols we have trades for, to reduce memory
    trade_syms = set(stage5b_out["symbol_raw"].unique())
    ohlcv = ohlcv[ohlcv["symbol"].isin(trade_syms)].reset_index(drop=True)
    print(f"  OHLCV rows after symbol filter: {len(ohlcv):,}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cs_cfg = load_stage5c_config()
    print("Running Stage 5c (cross-sectional F1+F2)...")
    stage5c_result = run_stage5c(
        trades=stage5b_out,
        ohlcv=ohlcv,
        cfg=cs_cfg,
        report_path=OUT_DIR / "07-cross-sectional-simulation-2025.md",
        summary_json=OUT_DIR / "stage5c_simulation_2025.json",
    )
    stage5c_filtered = stage5c_result["filtered_trades"]
    allowed_mask = stage5c_filtered["allowed"].astype(bool)
    chain_input = stage5c_filtered[allowed_mask].reset_index(drop=True)
    print(f"Stage 5c: {len(chain_input):,} trades pass F1+F2")

    # Stage 5d: ConvictionGate (FIFO cap 50)
    print("Running Stage 5d (ConvictionGate FIFO cap 50)...")
    scorer = XGBoostScorer(CV_MODEL, CV_FEATURE_SPEC)
    cv_cfg_full = json.loads((ROOT / "config" / "configuration.json").read_text(encoding="utf-8"))["conviction_gate"]
    gate_cfg = {
        "enabled": True,
        "daily_cap": int(cv_cfg_full["daily_cap"]),
        "min_predicted_r": float(cv_cfg_full["min_predicted_r"]),
    }
    run_stage5d(
        trades=chain_input,
        scorer=scorer,
        cfg=gate_cfg,
        report_path=OUT_DIR / "08-conviction-simulation-2025.md",
        summary_json=OUT_DIR / "stage5d_simulation_2025.json",
    )

    # Stage 5e variants — full (all 6 constraints) + simple (embargo + buckets only)
    adv_df = pd.read_parquet(ADV_PARQUET)
    adv_map = {
        (row.symbol, row.date_only): float(row.adv_rupees_20d)
        for row in adv_df.itertuples()
        if pd.notna(row.adv_rupees_20d)
    }

    print("Running Stage 5e (FULL — all 6 constraints)...")
    run_stage5e(
        trades=chain_input,
        adv_map=adv_map,
        cfg={},
        report_path=OUT_DIR / "09-budgeted-selector-FULL-2025.md",
        summary_json=OUT_DIR / "stage5e_full_2025.json",
    )

    print("Running Stage 5e (SIMPLE — embargo + time buckets only)...")
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
        report_path=OUT_DIR / "09-budgeted-selector-SIMPLE-2025.md",
        summary_json=OUT_DIR / "stage5e_simple_2025.json",
    )

    # Side-by-side summary
    print()
    print("=" * 80)
    print("2025 OOS VALIDATION: Stage 5d (FIFO) vs Stage 5e FULL vs Stage 5e SIMPLE")
    print("=" * 80)
    d = json.loads((OUT_DIR / "stage5d_simulation_2025.json").read_text())
    ef = json.loads((OUT_DIR / "stage5e_full_2025.json").read_text())
    es = json.loads((OUT_DIR / "stage5e_simple_2025.json").read_text())
    fmt = "{:<22} {:>14} {:>14} {:>14}"
    print(fmt.format("Metric", "Stage 5d", "5e FULL", "5e SIMPLE"))
    print("-" * 72)
    for k in ["n_trades", "trades_per_day", "total_pnl", "pf", "wr_pct",
              "session_sharpe", "losing_days_pct"]:
        print(fmt.format(k, d["after"][k], ef["after"][k], es["after"][k]))
    print()
    for label, v in [("5e FULL", ef), ("5e SIMPLE", es)]:
        print(f"-- {label} vs 5d --")
        print(f"   PF delta:     {v['after']['pf'] - d['after']['pf']:+.3f}")
        print(f"   Sharpe delta: {v['after']['session_sharpe'] - d['after']['session_sharpe']:+.3f}")
        print(f"   PnL delta:    Rs {v['after']['total_pnl'] - d['after']['total_pnl']:+,.0f}")


if __name__ == "__main__":
    main()
