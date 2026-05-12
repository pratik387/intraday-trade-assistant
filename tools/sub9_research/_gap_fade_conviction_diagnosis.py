"""Phase B Step 1: Diagnose which gate is anti-selecting gap_fade.

Leading hypothesis: ConvictionGate (XGBoost min_predicted_r=0.3) rejects
~67% of gap_fade fires that go on to be profitable.

Approach (Discovery 2023-2024 only):
  1. Read gate_input.jsonl files from the wide_open OCI run 20260509-122929_full
  2. Filter to setup_type == "gap_fade_short"
  3. Score each candidate with the production XGBoost model
  4. Check predicted_r distribution
  5. Count: how many would pass conviction (predicted_r >= 0.3)?
  6. Cross-reference with sub7_validation parquet (the kept 3,762 trades)

Pass criterion for "conviction is the bottleneck":
  >=50% of the ~8,041 sub7-filtered-OUT trades have predicted_r < 0.3

Usage:
    python tools/sub9_research/_gap_fade_conviction_diagnosis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from services.conviction.feature_spec import extract_features  # noqa: E402
from services.conviction.scorer import XGBoostScorer            # noqa: E402

MODEL_PATH = _REPO / "models" / "conviction" / "2026-04-28-sub8-C-universal-xgboost.json"
FEATURE_SPEC = _REPO / "models" / "conviction" / "2026-04-28-sub8-C-feature-spec.json"
OCI_RUN_DIR = _REPO / "20260509-122929_full"
SUB7_PARQUET = _REPO / "reports" / "sub7_validation" / "gap_fade_short.parquet"

MIN_PREDICTED_R = 0.3   # from production config


def load_candidates() -> pd.DataFrame:
    """Load all gap_fade candidates from Discovery 2023-2024 gate_input.jsonl."""
    rows = []
    session_dirs = sorted([d for d in OCI_RUN_DIR.iterdir() if d.is_dir() and d.name.startswith("202")])
    print(f"  found {len(session_dirs)} session dirs in {OCI_RUN_DIR.name}")
    for sdir in session_dirs:
        fp = sdir / "gate_input.jsonl"
        if not fp.exists():
            continue
        try:
            with open(fp, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    # Patch NaN/Infinity which json strict mode rejects
                    safe = line.replace("NaN", "null").replace("Infinity", "1e30").replace("-1e30", "-1e30")
                    try:
                        obj = json.loads(safe)
                    except Exception:
                        continue
                    for c in obj.get("candidates", []):
                        if c.get("setup_type") == "gap_fade_short":
                            rows.append(c)
        except Exception as e:
            print(f"    skip {sdir.name}: {e}")
    print(f"  loaded {len(rows)} gap_fade candidates")
    return pd.DataFrame(rows)


def score_candidates(df: pd.DataFrame) -> np.ndarray:
    print(f"  loading model: {MODEL_PATH.name}")
    scorer = XGBoostScorer(MODEL_PATH, FEATURE_SPEC)
    print("  extracting features ...")
    feat_list = [extract_features(r.to_dict()) for _, r in df.iterrows()]
    print(f"  scoring {len(feat_list)} candidates ...")
    preds = scorer.predict_batch(feat_list)
    return preds


def main():
    print("=== Phase B Step 1: gap_fade conviction diagnosis ===\n")
    cands = load_candidates()
    if len(cands) == 0:
        print("No gap_fade candidates found in gate_input.jsonl — aborting")
        return

    # Filter to Discovery 2023-2024 only
    cands["session_date"] = pd.to_datetime(cands["session_date_dt"]).dt.date
    from datetime import date
    cands = cands[(cands["session_date"] >= date(2023, 1, 1)) & (cands["session_date"] <= date(2024, 12, 31))]
    print(f"  filtered to Discovery 2023-2024: {len(cands)} candidates\n")

    preds = score_candidates(cands)
    cands["predicted_r"] = preds

    print("\n=== predicted_r DISTRIBUTION (gap_fade Discovery) ===")
    print(f"  count:  {len(cands)}")
    print(f"  min:    {cands['predicted_r'].min():.3f}")
    print(f"  p10:    {cands['predicted_r'].quantile(0.10):.3f}")
    print(f"  p25:    {cands['predicted_r'].quantile(0.25):.3f}")
    print(f"  p50:    {cands['predicted_r'].quantile(0.50):.3f}")
    print(f"  p75:    {cands['predicted_r'].quantile(0.75):.3f}")
    print(f"  p90:    {cands['predicted_r'].quantile(0.90):.3f}")
    print(f"  max:    {cands['predicted_r'].max():.3f}")
    print()
    pct_below = (cands["predicted_r"] < MIN_PREDICTED_R).mean() * 100
    print(f"  pct with predicted_r < {MIN_PREDICTED_R}: {pct_below:.1f}%  ({(cands['predicted_r'] < MIN_PREDICTED_R).sum()}/{len(cands)})")
    print()

    # Compare with sub7-admitted set
    print("=== Cross-reference with sub7-admitted (small_cap only) ===")
    sub7 = pd.read_parquet(SUB7_PARQUET)
    sub7_sm = sub7[sub7["cap_segment"] == "small_cap"].copy()
    sub7_sm["session_date"] = pd.to_datetime(sub7_sm["session_date"]).dt.date
    sub7_keys = set(zip(sub7_sm["symbol"], sub7_sm["session_date"]))
    print(f"  sub7 admitted (small_cap): {len(sub7_sm)} trades")

    cands["in_sub7"] = cands.apply(lambda r: (r["symbol"], r["session_date"]) in sub7_keys, axis=1)
    cands_sm = cands[cands["cap_segment"] == "small_cap"]
    print(f"  candidates small_cap: {len(cands_sm)}")
    print(f"  cands_sm in sub7:      {cands_sm['in_sub7'].sum()}")
    print(f"  cands_sm NOT in sub7:  {(~cands_sm['in_sub7']).sum()}")
    print()

    # The hypothesis: if conviction is the bottleneck, NOT-in-sub7 should have low predicted_r
    in_sub7 = cands_sm[cands_sm["in_sub7"]]
    out_sub7 = cands_sm[~cands_sm["in_sub7"]]
    print(f"  IN sub7  predicted_r:  median={in_sub7['predicted_r'].median():.3f}  mean={in_sub7['predicted_r'].mean():.3f}  pct>=0.3={100*(in_sub7['predicted_r']>=MIN_PREDICTED_R).mean():.1f}%")
    print(f"  OUT sub7 predicted_r:  median={out_sub7['predicted_r'].median():.3f}  mean={out_sub7['predicted_r'].mean():.3f}  pct>=0.3={100*(out_sub7['predicted_r']>=MIN_PREDICTED_R).mean():.1f}%")
    print()

    # Verdict
    out_below = (out_sub7["predicted_r"] < MIN_PREDICTED_R).mean() * 100
    print("=== VERDICT ===")
    if out_below >= 50:
        print(f"  CONVICTION IS THE BOTTLENECK")
        print(f"  {out_below:.1f}% of filtered-OUT candidates have predicted_r < {MIN_PREDICTED_R}")
        print(f"  Recommendation: bypass ConvictionGate for gap_fade_short")
    else:
        print(f"  Conviction is NOT the dominant filter")
        print(f"  Only {out_below:.1f}% of filtered-OUT candidates fail the predicted_r threshold")
        print(f"  Other gates (CrossSectional, Dedup) are doing the rejection")
        print(f"  Need to expand to full per-gate replay (Phase B Option B)")

    out_csv = _REPO / "analysis" / "_gap_fade_conviction_diagnosis_discovery.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cands.to_csv(out_csv, index=False)
    print(f"\nFull output: {out_csv}")


if __name__ == "__main__":
    main()
