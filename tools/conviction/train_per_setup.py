"""Per-setup model comparison for sub-project #2.

Trains separate XGBoost regressors for premium_zone_short + range_bounce_short
(the 2 setups with enough data after 74-rule survivor filter), compares OOS
RMSE + top-50 PF against universal model predictions on the same setups.

Matches universal T7 training config (pseudo-Huber, target clip at R=5) for
apples-to-apples architecture comparison. OOS evaluation applies the same
survivor-rule filter + trade_report.csv feature merge as T8 validation.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from services.conviction.scorer import XGBoostScorer
from services.conviction.feature_spec import extract_features
from tools.edge_discovery.data_loader import load_run
from tools.conviction.build_training_dataset import load_trade_report_features

ROOT = Path(__file__).parent.parent.parent
PARQUET = ROOT / "models" / "conviction" / "2026-04-22-training-dataset.parquet"
UNIVERSAL_MODEL = ROOT / "models" / "conviction" / "2026-04-22-universal-xgboost.json"
FEATURE_SPEC = ROOT / "models" / "conviction" / "2026-04-22-feature-spec.json"
PER_SETUP_DIR = ROOT / "models" / "conviction" / "per_setup"
BACKTEST_2025 = ROOT / "20260421-134338_full"
SURVIVORS = ROOT / "analysis" / "edge_discovery_runs" / "2026-04-22-validation-gate" / "stage6_validation_survivors.json"
OUT_REPORT = ROOT / "models" / "conviction" / "2026-04-22-architecture-decision.md"
OUT_JSON = ROOT / "models" / "conviction" / "2026-04-22-architecture-decision.json"

TARGET_SETUPS = ["premium_zone_short", "range_bounce_short"]
R_CLIP_MAX = 5.0

VAL_START = date(2025, 1, 1)
VAL_END = date(2025, 9, 30)


def _load_survivor_rules():
    survs = json.loads(SURVIVORS.read_text(encoding="utf-8"))
    rules = []
    for s in survs["survivors"]:
        rule_id = s["rule_id"]
        setup, cond_part = rule_id.split("__", 1)
        cond_key_part, cond_val_part = cond_part.split("=", 1)
        conds = list(zip(cond_key_part.split("+"), cond_val_part.split("+")))
        rules.append({"setup": setup, "conditions": conds})
    return rules


def _matches_any_rule(row, rules):
    for r in rules:
        if row["setup_type"] != r["setup"]:
            continue
        if all(row.get(k) == v for k, v in r["conditions"]):
            return True
    return False


def train_single_setup(setup, train_df, feature_cols):
    setup_col = f"setup_type_{setup}"
    if setup_col not in train_df.columns:
        raise ValueError(f"Setup one-hot missing: {setup_col}")
    sub = train_df[train_df[setup_col] == 1].copy()
    sub = sub.sort_values("_session_date_dt").reset_index(drop=True)
    split = int(0.8 * len(sub))
    tr, vl = sub.iloc[:split], sub.iloc[split:]

    y_tr = np.clip(tr["_label_r_multiple"].values, a_min=-1.0, a_max=R_CLIP_MAX)
    y_vl = np.clip(vl["_label_r_multiple"].values, a_min=-1.0, a_max=R_CLIP_MAX)

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        early_stopping_rounds=30,
        objective="reg:pseudohubererror",
        huber_slope=1.0,
        eval_metric="mae",
        random_state=42,
    )
    model.fit(
        tr[feature_cols].values, y_tr,
        eval_set=[(vl[feature_cols].values, y_vl)],
        verbose=False,
    )
    return model, tr, vl


def _build_2025_setup_frame(setup, rules):
    """Load 2025 validation window, merge trade_report features, filter to
    survivor rules AND the target setup."""
    data = load_run(BACKTEST_2025)
    t = data.trades
    t = t[(t["session_date_dt"] >= VAL_START) & (t["session_date_dt"] <= VAL_END)].copy()
    features_df = load_trade_report_features(BACKTEST_2025)
    t = t.merge(features_df, on="trade_id", how="left", suffixes=("", "_tr"))
    t = t[t["setup_type"] == setup].copy()
    t = t[t.apply(lambda r: _matches_any_rule(r, rules), axis=1)].copy()
    t["day_of_week"] = pd.to_datetime(t["session_date_dt"]).dt.day_name()
    return t


def _eval(model, feature_cols, trades):
    feat_rows = [extract_features(r.to_dict()) for _, r in trades.iterrows()]
    X = pd.DataFrame(feat_rows)
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0.0
    X = X[feature_cols].values
    y = trades["r_multiple"].astype(float).fillna(-1.0).values
    pred = model.predict(X)
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    pearson = float(np.corrcoef(pred, y)[0, 1]) if len(pred) > 1 else 0.0
    df = pd.DataFrame({"pred": pred, "realized": y, "sess": trades["session_date_dt"].values})
    pf_top50 = _top_n_pf(df, n=50)
    return {"rmse": rmse, "pearson": pearson, "pf_top50": pf_top50, "n": len(trades)}


def _top_n_pf(df, n=50):
    out = []
    for _, g in df.groupby("sess"):
        out.append(g.sort_values("pred", ascending=False).head(n))
    top = pd.concat(out, ignore_index=True)
    wins = top[top["realized"] > 0]["realized"]
    losses = top[top["realized"] < 0]["realized"]
    return float(wins.sum() / abs(losses.sum())) if losses.sum() < 0 else float("inf")


def main():
    df = pd.read_parquet(PARQUET)
    feature_cols = [c for c in df.columns if not c.startswith("_")]
    rules = _load_survivor_rules()

    universal_scorer = XGBoostScorer(UNIVERSAL_MODEL, FEATURE_SPEC)

    PER_SETUP_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Conviction architecture decision",
        "",
        "Comparison: per-setup XGBoost (trained only on the setup's trades) vs",
        "the universal model (trained on all 74-survivor trades with setup_type",
        "one-hot).  Both use pseudo-Huber loss + R<=5 Winsorization.  OOS eval",
        "applies the 74-survivor rule filter AND restricts to the target setup.",
        "",
        "| Setup | N (OOS) | Universal RMSE | Per-setup RMSE | Universal PF@50 | Per-setup PF@50 | Winner |",
        "|---|---|---|---|---|---|---|",
    ]
    decisions = []
    for setup in TARGET_SETUPS:
        print(f"Training per-setup model for {setup}...")
        setup_model, _, _ = train_single_setup(setup, df, feature_cols)
        ps_path = PER_SETUP_DIR / f"{setup}-xgboost.json"
        setup_model.save_model(str(ps_path))

        print(f"Building 2025 OOS frame for {setup}...")
        oos_trades = _build_2025_setup_frame(setup, rules)
        print(f"  OOS rows: {len(oos_trades)}")

        per_setup_eval = _eval(setup_model, feature_cols, oos_trades)
        universal_eval = _eval(universal_scorer.model, feature_cols, oos_trades)

        pf_delta_pct = 100 * (per_setup_eval["pf_top50"] - universal_eval["pf_top50"]) / universal_eval["pf_top50"]
        winner = "per_setup" if pf_delta_pct > 5 else "universal"

        decisions.append({
            "setup": setup,
            "universal": universal_eval,
            "per_setup": per_setup_eval,
            "pf_delta_pct": pf_delta_pct,
            "winner": winner,
        })
        lines.append(
            f"| {setup} | {per_setup_eval['n']} | {universal_eval['rmse']:.3f} | "
            f"{per_setup_eval['rmse']:.3f} | {universal_eval['pf_top50']:.3f} | "
            f"{per_setup_eval['pf_top50']:.3f} | {winner} |"
        )

    ship_per_setup = [d["setup"] for d in decisions if d["winner"] == "per_setup"]
    lines.append("")
    if ship_per_setup:
        lines.append(f"## Decision: Ship per-setup for {', '.join(ship_per_setup)}, universal for the rest")
    else:
        lines.append("## Decision: Ship universal only (per-setup does not beat universal by >5% PF)")

    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps(decisions, indent=2), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nReport: {OUT_REPORT}")


if __name__ == "__main__":
    main()
