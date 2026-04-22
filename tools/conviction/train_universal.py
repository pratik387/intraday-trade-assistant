"""Train universal XGBoost model for sub-project #2.

Loads training dataset (from build_training_dataset.py), chronologically splits
80/20 (early-stop validation within Discovery), trains XGBoost regressor on
r_multiple target.

Output: models/conviction/2026-04-22-universal-xgboost.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).parent.parent.parent
PARQUET = ROOT / "models" / "conviction" / "2026-04-22-training-dataset.parquet"
MODEL_OUT = ROOT / "models" / "conviction" / "2026-04-22-universal-xgboost.json"
METRICS_OUT = ROOT / "models" / "conviction" / "2026-04-22-training-metrics.json"


def main():
    print(f"Loading training dataset: {PARQUET}")
    df = pd.read_parquet(PARQUET).copy()
    print(f"Rows: {len(df):,}, columns: {len(df.columns)}")

    # Chronological split 80/20
    df = df.sort_values("_session_date_dt").reset_index(drop=True)
    split_idx = int(0.8 * len(df))
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    print(f"Train: {len(train_df):,} rows ({train_df['_session_date_dt'].min()} to {train_df['_session_date_dt'].max()})")
    print(f"Val:   {len(val_df):,} rows ({val_df['_session_date_dt'].min()} to {val_df['_session_date_dt'].max()})")

    feature_cols = [c for c in df.columns if not c.startswith("_")]
    X_train = train_df[feature_cols].values
    y_train = train_df["_label_r_multiple"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["_label_r_multiple"].values

    # XGBoost hyperparameters — modest, regularized
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        early_stopping_rounds=30,
        eval_metric="rmse",
        random_state=42,
    )
    print("Training...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    best_iter = model.best_iteration
    print(f"Best iteration: {best_iter}")

    # Evaluate on val fold
    y_val_pred = model.predict(X_val)
    rmse_val = float(np.sqrt(np.mean((y_val_pred - y_val) ** 2)))
    corr_val = float(np.corrcoef(y_val_pred, y_val)[0, 1])
    print(f"Validation RMSE: {rmse_val:.4f}")
    print(f"Validation Pearson: {corr_val:.4f}")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_OUT))
    print(f"Saved model: {MODEL_OUT}")

    # Training metrics artifact
    METRICS_OUT.write_text(json.dumps({
        "model_path": str(MODEL_OUT.relative_to(ROOT)),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "best_iteration": best_iter,
        "rmse_val": rmse_val,
        "pearson_val": corr_val,
        "features": feature_cols,
        "hyperparameters": {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "early_stopping_rounds": 30,
        },
    }, indent=2), encoding="utf-8")
    print(f"Saved metrics: {METRICS_OUT}")


if __name__ == "__main__":
    main()
