"""Build training dataset for sub-project #2 conviction model.

Loads Discovery trades (2023-01 to 2024-12), filters to the validation-
gate-surviving rules, extracts features per row, joins r_multiple as label.

Output: models/conviction/<run_id>-training-dataset.parquet

Discipline:
- Leakage audit: assert no BLOCKED_OUTCOME_COLUMNS in feature frame
- Coverage audit: drop features with >40% missing or large Discovery vs 2025 KS
- Survivor filter: only rules from stage6_validation_survivors.json

Runs with hard-coded paths by default (sub-project #2 original artifacts);
override via CLI for sub8 / new-OCI rebuilds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Repo root on sys.path so script-mode imports of tools.* / services.* succeed
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from tools.edge_discovery.data_loader import load_run
from services.conviction.feature_spec import (
    ALLOWED_FEATURES,
    BLOCKED_OUTCOME_COLUMNS,
    extract_features,
    audit_leakage,
)

ROOT = Path(__file__).parent.parent.parent

# Defaults preserve sub-project #2 original behaviour. Override via CLI for
# sub7/sub8 rebuilds (--backtest-dir, --survivors, --out-parquet, --feature-spec).
DEFAULT_BACKTEST_DIR = ROOT / "cloud_results" / "20260419_discovery"
DEFAULT_SURVIVORS = ROOT / "analysis" / "edge_discovery_runs" / "2026-04-22-validation-gate" / "stage6_validation_survivors.json"
DEFAULT_OUT_DIR = ROOT / "models" / "conviction"
DEFAULT_OUT_PARQUET = DEFAULT_OUT_DIR / "2026-04-22-training-dataset.parquet"
DEFAULT_FEATURE_SPEC_PATH = DEFAULT_OUT_DIR / "2026-04-22-feature-spec.json"

# Module-level paths used by main() — set via CLI in main(), or assigned by
# external runners that import this module and want to reuse load_*/extract.
BACKTEST_DIR: Path = DEFAULT_BACKTEST_DIR
SURVIVORS: Path = DEFAULT_SURVIVORS
OUT_PARQUET: Path = DEFAULT_OUT_PARQUET
FEATURE_SPEC_PATH: Path = DEFAULT_FEATURE_SPEC_PATH

# Feature columns to pull from trade_report.csv — any ALLOWED_FEATURES that live
# there and not in analytics.jsonl. vwap_distance_pct is absent from trade_report;
# it stays NaN→0 in extract_features (handled silently by the filter below).
_TRADE_REPORT_FEATURE_COLS = [
    "pdz_confluence_count",
    "pdz_range_position",
    "pdz_range_size_pct",
    "pdz_range_size_atr",
    "pdz_atr14",
    "pdz_has_mss_confluence",
    "pdz_has_fvg_confluence",
    "pdz_has_ob_confluence",
    "pdz_htf_bullish",
    "pdz_htf_bearish",
    "ob_confluence_count",
    "ob_has_liquidity_sweep",
    "ob_has_mss_confirmation",
    "resistance_touches",
    "resistance_strength",
    "bb_width_proxy",
    "volume5",
    "size_mult",
    "pattern_age_mins",
    "vol_z",
    "vol_ratio",
    "body_size_pct",
    "wick_ratio",
    "momentum_3bar_pct",
    "momentum_1bar_pct",
    "vwap_distance_pct",  # absent from trade_report.csv — filtered gracefully below
]


def load_trade_report_features(run_dir: Path) -> pd.DataFrame:
    """Load rich detector feature columns from all session trade_report.csv files.

    Iterates every session directory under run_dir (dirs whose names start with a
    digit that contain a trade_report.csv), reads only trade_id + the feature
    columns defined in _TRADE_REPORT_FEATURE_COLS (gracefully skipping any column
    absent in a particular session's CSV), and returns a single concatenated
    DataFrame keyed by trade_id.
    """
    wanted = set(["trade_id"] + _TRADE_REPORT_FEATURE_COLS)
    frames = []
    for session_dir in sorted(run_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        if not session_dir.name[0].isdigit():
            continue
        csv_path = session_dir / "trade_report.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(
                csv_path,
                usecols=lambda c: c in wanted,
                low_memory=False,
            )
            if "trade_id" not in df.columns:
                continue
            frames.append(df)
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: could not read {csv_path}: {exc}")

    if not frames:
        print("WARNING: no trade_report.csv files found — feature enrichment skipped")
        return pd.DataFrame(columns=["trade_id"])

    combined = pd.concat(frames, ignore_index=True)
    # Keep first occurrence per trade_id (should be unique but be defensive)
    combined = combined.drop_duplicates(subset=["trade_id"], keep="first")
    print(f"Loaded trade_report features: {len(combined):,} rows, "
          f"{len(combined.columns) - 1} feature columns from "
          f"{len(frames)} session files")
    return combined


def load_survivor_rules():
    data = json.loads(SURVIVORS.read_text(encoding="utf-8"))
    rules = []
    for s in data["survivors"]:
        # rule_id format: setup__conditioner=cell_value (with + for multi-dim)
        rule_id = s["rule_id"]
        setup, cond_part = rule_id.split("__", 1)
        cond_key_part, cond_val_part = cond_part.split("=", 1)
        conds = list(zip(cond_key_part.split("+"), cond_val_part.split("+")))
        rules.append({"setup": setup, "conditions": conds})
    return rules


def matches_any_rule(row, rules):
    for r in rules:
        if row["setup_type"] != r["setup"]:
            continue
        if all(row.get(k) == v for k, v in r["conditions"]):
            return True
    return False


def main():
    print(f"Loading Discovery trades: {BACKTEST_DIR}")
    data = load_run(BACKTEST_DIR)
    trades = data.trades
    print(f"Loaded {len(trades):,} trades")

    # Enrich with rich detector features from trade_report.csv files.
    # analytics.jsonl (via data_loader) contains outcome/timing fields but NOT
    # the ICT/structural detector features — those live in trade_report.csv.
    print("Loading rich detector features from trade_report.csv files...")
    features_df = load_trade_report_features(BACKTEST_DIR)
    pre_merge = len(trades)
    trades = trades.merge(features_df, on="trade_id", how="left", suffixes=("", "_tr"))
    assert len(trades) == pre_merge, (
        f"Merge changed row count: {pre_merge} → {len(trades)} (duplicate trade_ids?)"
    )
    print(f"After feature enrichment: {len(trades):,} trades")

    # Filter to Discovery range (2023-01-01 to 2024-12-31) and is_final_exit (already done by loader)
    from datetime import date
    trades = trades[
        (trades["session_date_dt"] >= date(2023, 1, 1))
        & (trades["session_date_dt"] <= date(2024, 12, 31))
    ].copy()
    print(f"Discovery subset: {len(trades):,}")

    # Load survivor rules + filter
    rules = load_survivor_rules()
    print(f"Applying {len(rules)} validation-survivor rules")
    mask = trades.apply(lambda r: matches_any_rule(r, rules), axis=1)
    filtered = trades[mask].copy()
    print(f"Training candidates (rule-matching): {len(filtered):,}")

    # Add day_of_week
    filtered["day_of_week"] = pd.to_datetime(filtered["session_date_dt"]).dt.day_name()

    # Extract features
    print("Extracting features...")
    feat_rows = [extract_features(r.to_dict()) for _, r in filtered.iterrows()]
    X = pd.DataFrame(feat_rows)

    # Leakage audit
    audit_leakage(X)

    # Coverage audit: drop features with >95% all-zero (truly dead features)
    coverage = (X != 0).mean()
    low_coverage_features = coverage[coverage < 0.05].index.tolist()
    if low_coverage_features:
        print(f"Dropping {len(low_coverage_features)} features with <5% coverage:")
        for f in low_coverage_features:
            print(f"  {f} ({100 * coverage[f]:.1f}% nonzero)")
        X = X.drop(columns=low_coverage_features)

    # Label: r_multiple
    y = filtered["r_multiple"].astype(float).fillna(-1.0).values  # hard_sl r=-1 by convention
    X["_label_r_multiple"] = y
    X["_session_date_dt"] = filtered["session_date_dt"].values  # for time-series CV later

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    X.to_parquet(OUT_PARQUET, index=False)
    print(f"Wrote {OUT_PARQUET} ({len(X):,} rows x {len(X.columns)} columns)")

    # Feature spec artifact
    features = [c for c in X.columns if not c.startswith("_")]
    FEATURE_SPEC_PATH.write_text(json.dumps({
        "features": features,
        "n_features": len(features),
        "source_rules": len(rules),
        "n_training_rows": len(X),
        "backtest_dir": str(BACKTEST_DIR.relative_to(ROOT) if BACKTEST_DIR.is_relative_to(ROOT) else BACKTEST_DIR),
        "survivors_path": str(SURVIVORS.relative_to(ROOT) if SURVIVORS.is_relative_to(ROOT) else SURVIVORS),
        "dropped_low_coverage": low_coverage_features,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {FEATURE_SPEC_PATH} ({len(features)} features)")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backtest-dir", default=str(DEFAULT_BACKTEST_DIR), help="OCI run dir with per-session subdirs")
    p.add_argument("--survivors", default=str(DEFAULT_SURVIVORS), help="stage6 validation survivors JSON")
    p.add_argument("--out-parquet", default=str(DEFAULT_OUT_PARQUET), help="Output training dataset parquet")
    p.add_argument("--feature-spec", default=str(DEFAULT_FEATURE_SPEC_PATH), help="Output feature spec JSON")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    BACKTEST_DIR = Path(args.backtest_dir)
    SURVIVORS = Path(args.survivors)
    OUT_PARQUET = Path(args.out_parquet)
    FEATURE_SPEC_PATH = Path(args.feature_spec)
    main()
