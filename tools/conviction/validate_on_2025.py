"""Run 4 model validation tests on 2025 OOS data for sub-project #2.

Tests:
1. OOS PF lift: top-50 per session by predicted_R vs random 50 → PF delta
2. Calibration monotonicity: decile curve on 2025 predictions
3. SHAP stability: top features Discovery ↔ 2025 must overlap
4. Per-session Spearman: median rank correlation pred vs realized

Output: models/conviction/2026-04-22-validation-report.md + JSON
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, binomtest

from tools.edge_discovery.data_loader import load_run
from tools.conviction.build_training_dataset import load_trade_report_features
from services.conviction.feature_spec import extract_features, audit_leakage
from services.conviction.calibration import build_decile_calibration
from services.conviction.scorer import XGBoostScorer

ROOT = Path(__file__).parent.parent.parent
BACKTEST_DIR = ROOT / "20260421-134338_full"
MODEL_PATH = ROOT / "models" / "conviction" / "2026-04-22-universal-xgboost.json"
FEATURE_SPEC_PATH = ROOT / "models" / "conviction" / "2026-04-22-feature-spec.json"
PARQUET_TRAIN = ROOT / "models" / "conviction" / "2026-04-22-training-dataset.parquet"
SURVIVORS = (
    ROOT
    / "analysis"
    / "edge_discovery_runs"
    / "2026-04-22-validation-gate"
    / "stage6_validation_survivors.json"
)

REPORT_OUT = ROOT / "models" / "conviction" / "2026-04-22-validation-report.md"
JSON_OUT = ROOT / "models" / "conviction" / "2026-04-22-validation-report.json"

VALIDATION_START = date(2025, 1, 1)
VALIDATION_END = date(2025, 9, 30)


def build_2025_feature_frame() -> pd.DataFrame:
    """Load 2025 OOS trades, enrich with trade_report.csv features, filter to survivors."""
    print(f"Loading trades from {BACKTEST_DIR} ...")
    data = load_run(BACKTEST_DIR)
    t = data.trades
    print(f"Total trades loaded: {len(t):,}")

    t = t[
        (t["session_date_dt"] >= VALIDATION_START) & (t["session_date_dt"] <= VALIDATION_END)
    ].copy()
    print(f"2025 OOS window trades: {len(t):,}")

    # CRITICAL: merge rich features from per-session trade_report.csv files.
    # data_loader.load_run only extracts outcomes; the trained model expects
    # pdz_*/ob_*/resistance_*/bb_width_proxy etc. which live in trade_report.csv.
    print("Loading rich detector features from trade_report.csv files...")
    features_df = load_trade_report_features(BACKTEST_DIR)
    pre_merge = len(t)
    t = t.merge(features_df, on="trade_id", how="left", suffixes=("", "_tr"))
    assert len(t) == pre_merge, (
        f"Merge changed row count: {pre_merge} -> {len(t)} (duplicate trade_ids?)"
    )

    # Verify enrichment worked — log pdz_confluence_count non-zero count
    if "pdz_confluence_count" in t.columns:
        n_nonzero = int((t["pdz_confluence_count"].fillna(0) != 0).sum())
        print(f"  pdz_confluence_count non-zero: {n_nonzero:,} / {len(t):,} rows")
    else:
        print("  WARNING: pdz_confluence_count column absent — feature enrichment may have failed")

    # Filter to survivor rules
    survs = json.loads(SURVIVORS.read_text(encoding="utf-8"))
    rules = []
    for s in survs["survivors"]:
        rule_id = s["rule_id"]
        setup, cond_part = rule_id.split("__", 1)
        cond_key_part, cond_val_part = cond_part.split("=", 1)
        conds = list(zip(cond_key_part.split("+"), cond_val_part.split("+")))
        rules.append({"setup": setup, "conditions": conds})

    print(f"Applying {len(rules)} survivor rules ...")

    def matches(row):
        for r in rules:
            if row["setup_type"] != r["setup"]:
                continue
            if all(row.get(k) == v for k, v in r["conditions"]):
                return True
        return False

    t = t[t.apply(matches, axis=1)].copy()
    print(f"Survivor-matching 2025 trades: {len(t):,}")

    t["day_of_week"] = pd.to_datetime(t["session_date_dt"]).dt.day_name()
    return t


def main():
    print("=" * 60)
    print("Conviction model validation — 2025 OOS data")
    print("=" * 60)

    print("\nLoading model...")
    scorer = XGBoostScorer(MODEL_PATH, FEATURE_SPEC_PATH)
    feature_list = scorer.features
    print(f"Features in model: {len(feature_list)}")

    print("\nBuilding 2025 feature frame...")
    trades_2025 = build_2025_feature_frame()
    print(f"Trade rows available: {len(trades_2025):,}")

    print("\nExtracting features...")
    feat_rows = [extract_features(r.to_dict()) for _, r in trades_2025.iterrows()]
    X_2025 = pd.DataFrame(feat_rows)
    # Ensure all model features present; fill missing with 0
    for c in feature_list:
        if c not in X_2025.columns:
            X_2025[c] = 0.0
    X_2025 = X_2025[feature_list]
    audit_leakage(X_2025)

    preds = scorer.model.predict(X_2025.values)
    realized = trades_2025["r_multiple"].astype(float).fillna(-1.0).values
    trades_2025 = trades_2025.copy()
    trades_2025["predicted_r"] = preds
    trades_2025["realized_r"] = realized

    print(f"Predictions range: [{preds.min():.3f}, {preds.max():.3f}]")
    print(f"Realized R range:  [{realized.min():.3f}, {realized.max():.3f}]")

    # =========================================================
    # Test 1: OOS PF lift (top-50 per session vs random 50)
    # =========================================================
    print("\n--- Test 1: OOS PF lift ---")

    def top_n_per_session(df, score_col, n=50, random=False):
        out = []
        for _, g in df.groupby("session_date_dt"):
            if random:
                sample = g.sample(n=min(n, len(g)), random_state=42)
            else:
                sample = g.sort_values(score_col, ascending=False).head(n)
            out.append(sample)
        return pd.concat(out, ignore_index=True)

    top_ml = top_n_per_session(trades_2025, "predicted_r")
    top_random = top_n_per_session(trades_2025, "predicted_r", random=True)

    def _pf_from_r(rs):
        rs = np.asarray(rs, dtype=float)
        wins = rs[rs > 0]
        losses = rs[rs < 0]
        if losses.sum() == 0:
            return float("inf")
        return float(wins.sum() / abs(losses.sum()))

    pf_ml = _pf_from_r(top_ml["realized_r"].values)
    pf_random = _pf_from_r(top_random["realized_r"].values)
    test1_pass = pf_ml > pf_random
    print(f"  PF (ML top-50):    {pf_ml:.4f}")
    print(f"  PF (random 50):    {pf_random:.4f}")
    print(f"  PASS: {test1_pass}")

    # =========================================================
    # Test 2: Calibration monotonicity on 2025
    # =========================================================
    print("\n--- Test 2: Calibration monotonicity ---")
    curve = build_decile_calibration(
        pd.Series(trades_2025["predicted_r"].values),
        pd.Series(trades_2025["realized_r"].values),
    )
    curve_sorted = curve.sort_values("decile")
    deltas = curve_sorted["realized_median"].diff().dropna().values
    n_non_decreasing = int(sum(d >= -0.1 for d in deltas))
    test2_pass = n_non_decreasing >= 7
    print(f"  Non-decreasing transitions: {n_non_decreasing}/9")
    print(f"  Decile realized medians: {curve_sorted['realized_median'].tolist()}")
    print(f"  PASS: {test2_pass}")

    # =========================================================
    # Test 3: SHAP stability Discovery <-> 2025
    # =========================================================
    print("\n--- Test 3: SHAP feature stability ---")
    import shap  # noqa: PLC0415

    parquet_df = pd.read_parquet(PARQUET_TRAIN)
    # parquet may have more rows than feature_list; subset to model features only
    available_train_feats = [f for f in feature_list if f in parquet_df.columns]
    X_train_df = parquet_df[available_train_feats].sample(
        n=min(5000, len(parquet_df)), random_state=42
    )
    # Fill any missing feature columns with 0
    for c in feature_list:
        if c not in X_train_df.columns:
            X_train_df[c] = 0.0
    X_train_df = X_train_df[feature_list]
    X_train = X_train_df.values

    X_val = X_2025.sample(n=min(5000, len(X_2025)), random_state=42).values

    explainer = shap.TreeExplainer(scorer.model)
    shap_train = explainer.shap_values(X_train)
    shap_val = explainer.shap_values(X_val)

    top_train_idx = set(np.argsort(np.abs(shap_train).mean(axis=0))[-10:].tolist())
    top_val_idx = set(np.argsort(np.abs(shap_val).mean(axis=0))[-10:].tolist())
    overlap = len(top_train_idx & top_val_idx)
    test3_pass = overlap >= 7

    top_train_names = [feature_list[i] for i in sorted(top_train_idx)]
    top_val_names = [feature_list[i] for i in sorted(top_val_idx)]
    print(f"  Top-10 Discovery features: {top_train_names}")
    print(f"  Top-10 2025 features:      {top_val_names}")
    print(f"  Overlap: {overlap}/10")
    print(f"  PASS: {test3_pass}")

    # =========================================================
    # Test 4: Per-session Spearman
    # =========================================================
    print("\n--- Test 4: Per-session Spearman rank correlation ---")
    per_session_rhos = []
    for _, g in trades_2025.groupby("session_date_dt"):
        if len(g) < 5:
            continue
        rho, _ = spearmanr(g["predicted_r"].values, g["realized_r"].values)
        if not np.isnan(rho):
            per_session_rhos.append(rho)

    if per_session_rhos:
        median_rho = float(np.median(per_session_rhos))
        n_positive = sum(1 for r in per_session_rhos if r > 0)
        p_sign = float(
            binomtest(n_positive, len(per_session_rhos), p=0.5, alternative="greater").pvalue
        )
    else:
        median_rho = 0.0
        n_positive = 0
        p_sign = 1.0

    test4_pass = median_rho > 0.05 and p_sign < 0.05
    print(f"  Sessions with >=5 trades: {len(per_session_rhos)}")
    print(f"  Median rho:  {median_rho:.4f}")
    print(f"  N positive:  {n_positive}")
    print(f"  Sign-test p: {p_sign:.4f}")
    print(f"  PASS: {test4_pass}")

    # =========================================================
    # Results
    # =========================================================
    results = {
        "model": MODEL_PATH.name,
        "oos_period": f"{VALIDATION_START} to {VALIDATION_END}",
        "n_trades_evaluated": len(trades_2025),
        "test1_oos_pf_lift": {
            "pf_ml": pf_ml,
            "pf_random": pf_random,
            "passed": bool(test1_pass),
        },
        "test2_calibration_monotonicity": {
            "n_non_decreasing_transitions": n_non_decreasing,
            "required_min": 7,
            "passed": bool(test2_pass),
        },
        "test3_shap_stability": {
            "overlap_top10": overlap,
            "required_min": 7,
            "top_train_features": top_train_names,
            "top_val_features": top_val_names,
            "passed": bool(test3_pass),
        },
        "test4_per_session_spearman": {
            "median_rho": median_rho,
            "n_sessions": len(per_session_rhos),
            "n_positive": n_positive,
            "sign_test_p": p_sign,
            "passed": bool(test4_pass),
        },
        "all_passed": bool(test1_pass and test2_pass and test3_pass and test4_pass),
    }
    JSON_OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nJSON written: {JSON_OUT}")

    all_ok = results["all_passed"]
    lines = [
        "# Conviction model validation report (2025 OOS)",
        "",
        f"Model: `{MODEL_PATH.name}`",
        f"OOS period: {VALIDATION_START} to {VALIDATION_END}",
        f"Trades evaluated: {len(trades_2025):,}",
        "",
        "## Results",
        "",
        "| Test | Required | Actual | Passed |",
        "|---|---|---|---|",
        (
            f"| 1. OOS PF lift (top-50 vs random) | PF_ml > PF_random"
            f" | PF_ml={pf_ml:.3f}, PF_random={pf_random:.3f}"
            f" | {'PASS' if test1_pass else 'FAIL'} |"
        ),
        (
            f"| 2. Calibration monotonicity | >=7/9 transitions non-decreasing"
            f" | {n_non_decreasing}/9"
            f" | {'PASS' if test2_pass else 'FAIL'} |"
        ),
        (
            f"| 3. SHAP stability | >=7/10 features overlap"
            f" | {overlap}/10"
            f" | {'PASS' if test3_pass else 'FAIL'} |"
        ),
        (
            f"| 4. Per-session Spearman | median rho > 0.05, p < 0.05"
            f" | rho={median_rho:.3f}, p={p_sign:.4f}"
            f" | {'PASS' if test4_pass else 'FAIL'} |"
        ),
        "",
        f"**All passed: {'YES' if all_ok else 'NO'}**",
    ]
    report_text = "\n".join(lines) + "\n"
    REPORT_OUT.write_text(report_text, encoding="utf-8")
    print("\n" + report_text)
    print(f"Report: {REPORT_OUT}")


if __name__ == "__main__":
    main()
