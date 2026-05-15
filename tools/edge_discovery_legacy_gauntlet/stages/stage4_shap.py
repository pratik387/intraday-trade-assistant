"""Stage 4: ML feature importance via SHAP. Interpretation aid per design §3.3.

Trains an XGBoost classifier per surviving setup on Discovery period with
`realized_pnl > 0` as the binary label. Time-series CV prevents leakage.
SHAP values surface which features drive the win/loss split — used only to
check whether Stage 3 missed a structural conditioner.

We do NOT deploy XGBoost. Output is markdown for human inspection.

Per spec Section 3.3 Stage 4: "Mostly confirms Stage 3; occasionally surfaces
a missed structural driver. If SHAP reveals strong unaccounted structural
driver → add to Stage 3 conditioner list, re-run Stage 3. Only structural /
interpretable drivers added."
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repo root is on sys.path when run as a script
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from tools.edge_discovery_legacy_gauntlet.periods import DiscoveryConfig


# Conditioners that are already in Stage 3 — included in features as a sanity check
_CONDITIONERS = ("regime", "cap_segment", "hour_bucket", "volatility_regime")

# Numeric feature columns to include if present in the trade rows
_NUMERIC_FEATURE_CANDIDATES = (
    "vol_z", "atr", "rank_score", "minute_of_day", "rsi", "adx",
    "bb_width_proxy", "vwap_distance_pct", "gap_pct", "wick_ratio",
    "upper_wick_ratio", "body_size_pct", "vol_ratio", "momentum_1bar_pct",
    "momentum_3bar_pct", "volume5",
)


def _build_feature_matrix(trades: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode conditioners + select numeric features. Returns float matrix."""
    parts: List[pd.DataFrame] = []
    # One-hot the structural conditioners present
    for c in _CONDITIONERS:
        if c in trades.columns:
            d = pd.get_dummies(trades[c].fillna("unknown"), prefix=c, dtype=float)
            parts.append(d)
    # Numeric features
    num_cols = [c for c in _NUMERIC_FEATURE_CANDIDATES if c in trades.columns]
    if num_cols:
        num = trades[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        parts.append(num)
    if not parts:
        return pd.DataFrame(index=trades.index)
    return pd.concat(parts, axis=1)


def _train_and_shap(
    features: pd.DataFrame,
    labels: pd.Series,
    n_splits: int = 5,
) -> Dict[str, float]:
    """Time-series CV XGBoost + return mean |SHAP| per feature."""
    import xgboost as xgb
    import shap
    from sklearn.model_selection import TimeSeriesSplit

    if len(features) < 200 or labels.nunique() < 2:
        return {}

    # Features must be ordered chronologically by the caller (we sort by ts upstream).
    tss = TimeSeriesSplit(n_splits=min(n_splits, max(2, len(features) // 200)))
    fold_shap_means: List[pd.Series] = []
    for train_idx, test_idx in tss.split(features):
        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue
        clf = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42,
            n_jobs=4, verbosity=0,
        )
        clf.fit(X_train, y_train)

        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):  # multi-class wrap
            shap_values = shap_values[1]
        mean_abs = pd.Series(
            np.abs(shap_values).mean(axis=0),
            index=features.columns,
        )
        fold_shap_means.append(mean_abs)

    if not fold_shap_means:
        return {}
    avg = pd.concat(fold_shap_means, axis=1).mean(axis=1)
    return avg.sort_values(ascending=False).to_dict()


def run_stage4(
    trades: pd.DataFrame,
    cfg: DiscoveryConfig,
    survivors_input: List[str],
    report_path: Path,
    top_k: int = 15,
) -> Dict[str, Dict[str, float]]:
    """Run Stage 4 SHAP per setup. Returns {setup: {feature: mean_abs_shap}}.

    `survivors_input` is the list of Stage 3-surviving setup names (deduped).
    Trades should already be filtered to Discovery period; we filter again
    defensively in case caller passes the full frame.
    """
    if "session_date_dt" in trades.columns:
        sd = pd.to_datetime(trades["session_date_dt"], errors="coerce")
        discovery_mask = (
            (sd >= pd.Timestamp(cfg.discovery_start)) &
            (sd <= pd.Timestamp(cfg.discovery_end))
        )
        df = trades[discovery_mask].copy()
        df["session_date_dt"] = sd[discovery_mask].values
    else:
        df = trades.copy()

    pnl_col = "realized_pnl" if "realized_pnl" in df.columns else (
        "net_pnl" if "net_pnl" in df.columns else "total_trade_pnl"
    )
    if pnl_col not in df.columns:
        raise KeyError(f"No PnL column found in trades (expected one of: realized_pnl, net_pnl, total_trade_pnl)")

    # Sort chronologically for time-series CV
    if "session_date_dt" in df.columns:
        df = df.sort_values("session_date_dt").reset_index(drop=True)

    out: Dict[str, Dict[str, float]] = {}
    md_lines: List[str] = ["# Stage 4: SHAP feature importance (interpretation aid)\n",
                           "Per design §3.3: identifies missed structural drivers; does NOT kill rules.\n",
                           f"Discovery period: {cfg.discovery_start} → {cfg.discovery_end}\n"]

    for setup in survivors_input:
        sub = df[df["setup_type"] == setup].copy()
        if len(sub) < 200:
            md_lines.append(f"\n## {setup}\n\nN={len(sub)} too small (<200) — skipped.\n")
            out[setup] = {}
            continue

        labels = (pd.to_numeric(sub[pnl_col], errors="coerce") > 0).astype(int)
        features = _build_feature_matrix(sub)

        if features.empty or features.shape[1] == 0:
            md_lines.append(f"\n## {setup}\n\nNo features available — skipped.\n")
            out[setup] = {}
            continue

        importances = _train_and_shap(features, labels)
        out[setup] = importances

        md_lines.append(f"\n## {setup}\n")
        md_lines.append(f"N={len(sub):,} | win_rate={labels.mean()*100:.1f}% | n_features={features.shape[1]}\n")
        if not importances:
            md_lines.append("\n_SHAP unavailable (insufficient class balance or features)._\n")
            continue

        md_lines.append("\nTop features by mean |SHAP|:\n")
        md_lines.append("\n| Rank | Feature | Mean abs SHAP |")
        md_lines.append("\n|---:|:---|---:|")
        for i, (feat, val) in enumerate(list(importances.items())[:top_k], 1):
            md_lines.append(f"\n| {i} | `{feat}` | {val:.4f} |")
        md_lines.append("\n")

        # Auto-flag if a non-conditioner feature outranks all conditioners
        conditioner_features = {f for f in importances.keys()
                                if any(f.startswith(c + "_") for c in _CONDITIONERS)}
        top_features = list(importances.keys())[:5]
        non_cond_in_top5 = [f for f in top_features if f not in conditioner_features]
        if non_cond_in_top5:
            md_lines.append(
                f"\n**Note**: top-5 includes non-conditioner feature(s) "
                f"{non_cond_in_top5}. Consider whether any are STRUCTURAL "
                f"(stable across regimes, interpretable) and could be added to Stage 3 conditioners.\n"
            )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("".join(md_lines), encoding="utf-8")
    return out


def main():
    """CLI entry point."""
    import argparse, json
    from datetime import date
    from tools.edge_discovery_legacy_gauntlet.data_loader import load_run

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backtest-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--survivors-json", required=True, help="Stage 3 survivors JSON")
    p.add_argument("--discovery-start", default="2023-01-01")
    p.add_argument("--discovery-end", default="2024-12-31")
    args = p.parse_args()

    survivors = json.loads(Path(args.survivors_json).read_text())
    setups = sorted({r["setup"] for r in survivors.get("survivors", [])})
    print(f"[stage4] Surviving setups: {setups}")

    data = load_run(Path(args.backtest_dir))
    print(f"[stage4] Loaded {len(data.trades):,} trades")

    cfg = DiscoveryConfig(
        discovery_start=date.fromisoformat(args.discovery_start),
        discovery_end=date.fromisoformat(args.discovery_end),
        validation_start=date(2025, 1, 1), validation_end=date(2025, 9, 30),
        holdout_start=date(2025, 10, 1), holdout_end=date(2026, 3, 31),
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    importances = run_stage4(data.trades, cfg, setups, out_dir / "04-feature-importance.md")
    (out_dir / "04-shap.json").write_text(json.dumps(importances, indent=2, default=str))
    print(f"[stage4] Wrote {out_dir / '04-feature-importance.md'}")


if __name__ == "__main__":
    main()
