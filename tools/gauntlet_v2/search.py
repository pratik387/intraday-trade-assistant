"""Optuna Bayesian search over the 8 gauntlet_v2 gate-config params.

Sub-project #5 Phase 2. Objective: Sharpe maximisation. Constraint:
n_trades >= min_n_trades over Discovery (default 500, tunable via CLI).

Search space (mirror of design spec §6.2):
    conviction_gate.daily_cap                         int [20, 150]
    conviction_gate.min_predicted_r                   float [-0.5, 1.5]
    dedup_gate.cooloff_bars                           int [0, 12]
    dedup_gate.require_setup_change                   categorical [True, False]
    cross_sectional_gate.f1_rvol_threshold_pct        float [50, 95]
    cross_sectional_gate.f2_crowdedness_threshold     int [10, 80]
    cross_sectional_gate.f2_crowdedness_window_min    int [3, 15]
    rank_pctl_min                                     float [0.4, 0.9]

Outputs (under <output_dir>):
    study.db              Optuna SQLite — resumable
    best_config.json      Merged cfg overrides for the best trial
    trials.csv            All completed trial params + metrics
    06-search-report.md   Human-readable top-10 + best config
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import optuna

from tools.gauntlet_v2.trial import run_trial


def _suggest_overrides(trial: optuna.Trial) -> Dict[str, Any]:
    """Sample one config-override dict from the search space."""
    return {
        "conviction_gate": {
            "daily_cap": trial.suggest_int("daily_cap", 20, 150),
            "min_predicted_r": trial.suggest_float("min_predicted_r", -0.5, 1.5),
        },
        "dedup_gate": {
            "cooloff_bars": trial.suggest_int("cooloff_bars", 0, 12),
            "require_setup_change": trial.suggest_categorical("require_setup_change", [True, False]),
        },
        "cross_sectional_gate": {
            "f1_rvol_threshold_pct": trial.suggest_float("f1_rvol_threshold_pct", 50.0, 95.0),
            "f2_crowdedness_threshold": trial.suggest_int("f2_crowdedness_threshold", 10, 80),
            "f2_crowdedness_window_min": trial.suggest_int("f2_crowdedness_window_min", 3, 15),
        },
        "rank_pctl_min": trial.suggest_float("rank_pctl_min", 0.4, 0.9),
    }


def _params_to_overrides(flat: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Optuna's flat trial.params dict back to nested config overrides."""
    return {
        "conviction_gate": {
            "daily_cap": int(flat["daily_cap"]),
            "min_predicted_r": float(flat["min_predicted_r"]),
        },
        "dedup_gate": {
            "cooloff_bars": int(flat["cooloff_bars"]),
            "require_setup_change": bool(flat["require_setup_change"]),
        },
        "cross_sectional_gate": {
            "f1_rvol_threshold_pct": float(flat["f1_rvol_threshold_pct"]),
            "f2_crowdedness_threshold": int(flat["f2_crowdedness_threshold"]),
            "f2_crowdedness_window_min": int(flat["f2_crowdedness_window_min"]),
        },
        "rank_pctl_min": float(flat["rank_pctl_min"]),
    }


def run_search(
    base_cfg: Dict[str, Any],
    gate_input_dir: Path,
    pnl_index: pd.DataFrame,
    output_dir: Path,
    n_trials: int,
    n_jobs: int = 1,
    min_n_trades: int = 500,
) -> None:
    """Run an Optuna study and persist all artefacts to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="maximize",
        storage=f"sqlite:///{output_dir / 'study.db'}",
        study_name="gauntlet_v2",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    def objective(trial: optuna.Trial) -> float:
        overrides = _suggest_overrides(trial)
        try:
            metrics = run_trial(overrides, gate_input_dir, pnl_index, base_cfg)
        except SystemExit as e:
            trial.set_user_attr("error", str(e))
            return float("-inf")
        trial.set_user_attr("metrics", metrics)
        if metrics["n_trades"] < min_n_trades:
            return float("-inf")
        return metrics["sharpe"]

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    # ---- Persist artefacts ----
    best = study.best_trial
    best_overrides = _params_to_overrides(best.params)
    (output_dir / "best_config.json").write_text(json.dumps(best_overrides, indent=2))

    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = dict(t.params)
        row["value"] = t.value
        row["number"] = t.number
        m = t.user_attrs.get("metrics") or {}
        for k, v in m.items():
            row[f"m_{k}"] = v
        rows.append(row)
    trials_df = pd.DataFrame(rows)
    trials_df.to_csv(output_dir / "trials.csv", index=False)

    n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    report = [
        "# Gauntlet v2 — Search Report (sub5-T7)",
        "",
        f"- **Study:** {study.study_name}",
        f"- **n_trials_requested:** {n_trials}",
        f"- **n_trials_completed:** {n_completed}",
        f"- **best_sharpe:** {best.value:.3f}" if best.value is not None and best.value != float("-inf") else "- **best_sharpe:** -inf (no qualifying trial)",
        "",
        "## Best config overrides",
        "",
        "```json",
        json.dumps(best_overrides, indent=2),
        "```",
        "",
        "## Top 10 by sharpe",
        "",
    ]
    if not trials_df.empty:
        top10 = trials_df.sort_values("value", ascending=False).head(10)
        report.append(top10.to_markdown(index=False))
    else:
        report.append("_(no completed trials)_")
    (output_dir / "06-search-report.md").write_text("\n".join(report), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-cfg", required=True)
    p.add_argument("--gate-input-dir", required=True)
    p.add_argument("--pnl-index", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-trials", type=int, default=500)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--min-n-trades", type=int, default=500)
    args = p.parse_args()

    base_cfg = json.loads(Path(args.base_cfg).read_text(encoding="utf-8"))
    pnl_df = pd.read_parquet(args.pnl_index)
    run_search(
        base_cfg=base_cfg,
        gate_input_dir=Path(args.gate_input_dir),
        pnl_index=pnl_df,
        output_dir=Path(args.output_dir),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        min_n_trades=args.min_n_trades,
    )


if __name__ == "__main__":
    main()
