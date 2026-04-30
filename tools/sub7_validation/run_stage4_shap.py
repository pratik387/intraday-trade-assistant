"""Stage 4 SHAP feature importance for sub7+sub8 Phase-1 rescue cells.

INPUT: per-setup parquets enriched with trade_report.csv features.
OUTPUT: Stage 4 SHAP report per setup — top features by mean |SHAP|, with
auto-flag if a non-conditioner feature outranks the cell conditioners.

Per master plan §3.3 Stage 4:
  - We do NOT deploy XGBoost.
  - SHAP answers: "what features drive the win/loss split, beyond Stage 3
    conditioners?"
  - If SHAP reveals a strong unaccounted STRUCTURAL driver (interpretable,
    stable across regimes), add it to Stage 3 conditioners and re-run
    Stage 3.
  - Spec expectation: "mostly confirms Stage 3; occasionally surfaces a
    missed driver."

Methodology
-----------
For each setup that passed Stage 3 (with a rescue cell):
  1. Read every session's trade_report.csv from the OCI run.
  2. Filter to the setup, executed=True, intended universe + caps.
  3. Compute fee + net_pnl via the Indian intraday fee model.
  4. Build a feature matrix from the rich trade_report columns:
       - One-hot-encoded conditioners: regime, cap_segment, hour_bucket,
         side
       - Numeric features: atr, bb_width_proxy, body_pct, gap_pct,
         range_pct, vol_x_median, vol_x_recent, adx5, minute_of_day,
         first_bar_volume_ratio, daily_trend_distance_pct, rank_score
  5. Label: realized_pnl > 0 (gross win/loss; SHAP isn't fee-sensitive at
     interpretation time).
  6. Train XGBoost classifier with TimeSeriesSplit CV (no leakage).
  7. Mean |SHAP| across folds → ranked feature list.

CLI
---
    python tools/sub7_validation/run_stage4_shap.py \\
        --oci-dir 20260430-232414_full \\
        --setup orb_15 --setup pdh_pdl_reject \\
        --output-dir docs/edge_discovery/2026-05-01-sub8-stage4-shap

Why a per-setup CLI: each setup has a different intended-universe filter,
and SHAP runs are slow enough that batching adds little value. Single-
setup invocations parallelise easily on a laptop.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from services.symbol_metadata import in_universe   # noqa: E402

from tools.sub7_validation.build_per_setup_pnl import (   # noqa: E402
    calc_fee, _drop_bad_priced_trades, INTRADAY_PRICE_RATIO_LO,
    INTRADAY_PRICE_RATIO_HI, MAX_PNL_R_MULTIPLE, RISK_PER_TRADE_RUPEES,
)


# Feature columns we'll one-hot or directly use.
# Conditioners — one-hot encoded
_CONDITIONER_COLS = ("side", "regime", "cap_segment")
# Numeric features (subset of trade_report.csv columns; only kept if present)
_NUMERIC_FEATURE_COLS = (
    "atr", "bb_width_proxy", "body_pct", "gap_pct", "range_pct",
    "vol_x_median", "vol_x_recent", "adx5", "minute_of_day",
    "first_bar_volume_ratio", "daily_trend_distance_pct", "rank_score",
    "day_of_week",
)


def _hour_bucket_from_minute(m: float) -> Optional[str]:
    if pd.isna(m):
        return None
    m = int(m)
    if m < 555:
        return None
    if m < 600:
        return "opening"
    if m < 720:
        return "morning"
    if m < 780:
        return "lunch"
    if m < 870:
        return "afternoon"
    return "late"


def load_setup_trades(
    oci_dir: Path,
    setup: str,
    universe_key: Optional[str],
    allowed_caps: Optional[set],
) -> pd.DataFrame:
    """Walk OCI dir, load all trade_reports, filter to one setup + executed +
    intended universe + cap. Compute net_pnl. Returns one row per trade."""
    parts: List[pd.DataFrame] = []
    sessions = sorted(oci_dir.glob("*/trade_report.csv"))
    if not sessions:
        raise SystemExit(f"No trade_report.csv under {oci_dir}/*")

    for f in sessions:
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception as e:
            print(f"  skip {f.parent.name}: {e}", file=sys.stderr)
            continue
        if df.empty or "setup_type" not in df.columns:
            continue
        sub = df[(df["setup_type"] == setup) & (df.get("executed", False) == True)].copy()
        if sub.empty:
            continue
        sub["session_date"] = f.parent.name
        parts.append(sub)

    if not parts:
        return pd.DataFrame()

    big = pd.concat(parts, ignore_index=True)
    big, n_ratio, n_pnl = _drop_bad_priced_trades(big)
    if n_ratio + n_pnl > 0:
        print(f"  sanity-clean dropped {n_ratio + n_pnl} rows")

    # Apply intended-universe + cap filter
    if universe_key:
        try:
            mask_u = big["symbol"].apply(lambda s: in_universe(s, universe_key))
            big = big[mask_u].copy()
        except KeyError:
            print(f"  warn: unknown universe_key={universe_key}; skipping universe filter")
    if allowed_caps and "cap_segment" in big.columns:
        big = big[big["cap_segment"].isin(allowed_caps)].copy()

    if big.empty:
        return big

    # Compute fees + net_pnl using the same model as build_per_setup_pnl
    big["fee"] = big.apply(
        lambda r: calc_fee(r.get("entry_price"), r.get("e1_price"),
                           int(r.get("qty", 0) or 0), r.get("side", "")),
        axis=1,
    )
    big["net_pnl"] = big["realized_pnl"].astype(float) - big["fee"]

    # Add hour_bucket from minute_of_day for one-hot conditioner
    if "minute_of_day" in big.columns:
        big["hour_bucket"] = big["minute_of_day"].map(_hour_bucket_from_minute)

    # Sort chronologically for time-series CV
    big["_ts"] = pd.to_datetime(big["session_date"]) + pd.to_timedelta(
        big.get("minute_of_day", 0).fillna(0), unit="m"
    )
    big = big.sort_values("_ts").reset_index(drop=True)
    return big


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot conditioners + numeric features, all float, no NaN."""
    parts: List[pd.DataFrame] = []
    for c in _CONDITIONER_COLS + ("hour_bucket",):
        if c in df.columns and df[c].notna().any():
            d = pd.get_dummies(df[c].fillna("unknown"), prefix=c, dtype=float)
            parts.append(d)
    nums = [c for c in _NUMERIC_FEATURE_COLS if c in df.columns]
    if nums:
        n = df[nums].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        parts.append(n)
    if not parts:
        return pd.DataFrame(index=df.index)
    return pd.concat(parts, axis=1)


def train_and_shap(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> Dict[str, float]:
    """TimeSeriesSplit XGBoost + mean |SHAP| per feature."""
    import xgboost as xgb
    import shap
    from sklearn.model_selection import TimeSeriesSplit

    if len(X) < 200 or y.nunique() < 2:
        return {}

    n_splits = min(n_splits, max(2, len(X) // 200))
    tss = TimeSeriesSplit(n_splits=n_splits)
    fold_means: List[pd.Series] = []
    for fold, (tr, te) in enumerate(tss.split(X)):
        Xt, Xv = X.iloc[tr], X.iloc[te]
        yt, yv = y.iloc[tr], y.iloc[te]
        if yt.nunique() < 2 or yv.nunique() < 2:
            continue
        clf = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42,
            n_jobs=4, verbosity=0,
        )
        clf.fit(Xt, yt)
        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(Xv)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        fold_means.append(pd.Series(np.abs(shap_vals).mean(axis=0), index=X.columns))
        print(f"    fold {fold + 1}/{n_splits} done (train={len(Xt):,} test={len(Xv):,})")
    if not fold_means:
        return {}
    avg = pd.concat(fold_means, axis=1).mean(axis=1)
    return avg.sort_values(ascending=False).to_dict()


def render_setup_report(
    setup: str,
    df: pd.DataFrame,
    importances: Dict[str, float],
    out_path: Path,
    rescue_cell: Optional[str] = None,
    top_k: int = 15,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    md: List[str] = [
        f"# Stage 4 SHAP — {setup}\n",
        f"_Per master plan §3.3 Stage 4: interpretation aid only. "
        f"Does NOT kill rules; surfaces missed structural drivers._\n",
    ]
    if rescue_cell:
        md.append(f"**Rescue cell from Stage 3:** `{rescue_cell}`\n")

    n = len(df)
    wr = (df["realized_pnl"] > 0).mean() * 100
    pf_wins = df.loc[df["net_pnl"] > 0, "net_pnl"].sum()
    pf_losses = df.loc[df["net_pnl"] < 0, "net_pnl"].abs().sum()
    pf = round(pf_wins / pf_losses, 3) if pf_losses else float("inf")
    md.append(
        f"\n**Sample (intended universe):** n={n:,} | gross WR={wr:.1f}% "
        f"| NET PF={pf}\n"
    )

    if not importances:
        md.append("\n_SHAP unavailable (n<200 or single-class label)._\n")
        out_path.write_text("".join(md), encoding="utf-8")
        return

    md.append(f"\n## Top {top_k} features by mean |SHAP|\n")
    md.append("\n| Rank | Feature | Mean abs SHAP |")
    md.append("\n|---:|:---|---:|")
    for i, (feat, val) in enumerate(list(importances.items())[:top_k], 1):
        md.append(f"\n| {i} | `{feat}` | {val:.4f} |")
    md.append("\n")

    # Auto-flag non-conditioner features in top-5
    cond_prefixes = tuple(f"{c}_" for c in (_CONDITIONER_COLS + ("hour_bucket",)))
    top5 = list(importances.keys())[:5]
    non_cond = [f for f in top5 if not f.startswith(cond_prefixes)]
    if non_cond:
        md.append(
            f"\n**Note**: top-5 includes non-conditioner feature(s) "
            f"`{non_cond}`. If any are STRUCTURAL (interpretable, stable "
            f"across regimes) consider adding to Stage 3 conditioners and "
            f"re-running Stage 3 (master plan §3.3).\n"
        )
    else:
        md.append(
            "\n_All top-5 features are conditioners — Stage 3 already "
            "captures the dominant structural drivers (master plan "
            "expectation: 'mostly confirms Stage 3')._\n"
        )

    out_path.write_text("".join(md), encoding="utf-8")


def _load_intended_filter(setup: str, config_path: Path) -> Tuple[Optional[str], Optional[set]]:
    cfg = json.loads(config_path.read_text())
    sc = (cfg.get("setups") or {}).get(setup) or {}
    universe_key = sc.get("universe_key")
    caps = sc.get("allowed_cap_segments")
    return universe_key, set(caps) if isinstance(caps, list) else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oci-dir", required=True)
    p.add_argument("--setup", action="append", required=True,
                   help="Setup to analyze; pass multiple --setup")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--config-path", default="config/configuration.json")
    p.add_argument("--top-k", type=int, default=15)
    args = p.parse_args()

    oci_dir = Path(args.oci_dir)
    out_dir = Path(args.output_dir)
    cfg_path = Path(args.config_path)

    # Pre-known rescue cells from Stage 3 (just for report context)
    rescue_cells = {
        "orb_15": "side=SELL × cap_segment=mid_cap (n=154 PF=1.60)",
        "pdh_pdl_reject": "cap_segment=mid_cap (n=128 PF=1.46)",
    }

    summary: Dict[str, Any] = {}
    for setup in args.setup:
        print(f"\n=== {setup} ===")
        universe_key, caps = _load_intended_filter(setup, cfg_path)
        print(f"  intended: universe={universe_key} caps={sorted(caps) if caps else None}")
        df = load_setup_trades(oci_dir, setup, universe_key, caps)
        if df.empty:
            print(f"  no trades after filter; skipping")
            continue
        print(f"  n_trades after filter: {len(df):,}")

        X = build_feature_matrix(df)
        y = (df["realized_pnl"].astype(float) > 0).astype(int)
        print(f"  features: {X.shape[1]} | win_rate: {y.mean()*100:.1f}%")
        print(f"  training XGBoost + SHAP...")
        importances = train_and_shap(X, y)

        out_path = out_dir / f"{setup}.md"
        render_setup_report(
            setup, df, importances, out_path,
            rescue_cell=rescue_cells.get(setup),
            top_k=args.top_k,
        )
        summary[setup] = {
            "n_trades": len(df),
            "win_rate_pct": round(float(y.mean() * 100), 1),
            "top_features": list(importances.items())[:args.top_k],
        }
        print(f"  --> {out_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "00-summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(f"\nSummary --> {out_dir}/00-summary.json")


if __name__ == "__main__":
    main()
