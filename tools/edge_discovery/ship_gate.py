"""Two-tier ship gate: standalone setup vs ensemble feature."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ShipVerdict:
    shipped: bool
    tier: str  # "standalone" | "ensemble_feature" | "rejected"
    reasons: List[str] = field(default_factory=list)


def evaluate_standalone(stats: Dict, config: Dict) -> ShipVerdict:
    """Standalone setup ship gate. Returns ShipVerdict with reasons if any gate fails."""
    reasons: List[str] = []
    if stats["n_per_year"] < config["n_per_year_min"]:
        reasons.append(f"n_per_year={stats['n_per_year']} < min={config['n_per_year_min']}")
    if stats["pf_discovery"] < config["pf_discovery_min"]:
        reasons.append(f"pf_discovery={stats['pf_discovery']:.2f} < min={config['pf_discovery_min']}")
    if stats["pf_oos"] < config["pf_oos_min"]:
        reasons.append(f"pf_oos={stats['pf_oos']:.2f} < min={config['pf_oos_min']}")
    if stats["pf_holdout"] < config["pf_holdout_min"]:
        reasons.append(f"pf_holdout={stats['pf_holdout']:.2f} < min={config['pf_holdout_min']}")
    if stats["walk_forward_stability"] < config["walk_forward_stability_min"]:
        reasons.append(
            f"walk_forward_stability={stats['walk_forward_stability']:.2f} "
            f"< min={config['walk_forward_stability_min']}"
        )
    if stats["win_months_pct"] < config["win_months_pct_min"]:
        reasons.append(f"win_months_pct={stats['win_months_pct']} < min={config['win_months_pct_min']}")
    if stats["top_month_concentration_pct"] > config["top_month_concentration_max_pct"]:
        reasons.append(
            f"top_month_concentration_pct={stats['top_month_concentration_pct']} "
            f"> max={config['top_month_concentration_max_pct']}"
        )
    if not stats.get("rule_orthogonal", False):
        reasons.append("not rule_orthogonal and no hedging story attached")
    return ShipVerdict(
        shipped=(len(reasons) == 0),
        tier=("standalone" if not reasons else "rejected"),
        reasons=reasons,
    )


def evaluate_ensemble_feature(stats: Dict, config: Dict) -> ShipVerdict:
    """Ensemble-feature gate. Setup-too-small but the feature lifts live setup PF."""
    reasons: List[str] = []
    if stats["n_per_year"] < config["n_per_year_min"]:
        reasons.append(f"n_per_year={stats['n_per_year']} < min={config['n_per_year_min']}")
    if stats["n_per_year"] > config["n_per_year_max"]:
        reasons.append(
            f"n_per_year={stats['n_per_year']} > max={config['n_per_year_max']} "
            f"(setup is large enough for standalone tier)"
        )
    if stats["effect_size_sigma"] < config["effect_size_min_sigma"]:
        reasons.append(
            f"effect_size_sigma={stats['effect_size_sigma']:.2f} "
            f"< min={config['effect_size_min_sigma']}"
        )
    if stats["walk_forward_stability"] < config["walk_forward_stability_min"]:
        reasons.append(
            f"walk_forward_stability={stats['walk_forward_stability']:.2f} "
            f"< min={config['walk_forward_stability_min']}"
        )
    if stats["live_setup_pf_lift"] < config["live_setup_pf_lift_min"]:
        reasons.append(
            f"live_setup_pf_lift={stats['live_setup_pf_lift']:.2f} "
            f"< min={config['live_setup_pf_lift_min']}"
        )
    return ShipVerdict(
        shipped=(len(reasons) == 0),
        tier=("ensemble_feature" if not reasons else "rejected"),
        reasons=reasons,
    )
