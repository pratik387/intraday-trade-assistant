import pandas as pd
from tools.edge_discovery.ship_gate import evaluate_standalone, evaluate_ensemble_feature


def _ship_config_standalone() -> dict:
    return {
        "n_per_year_min": 300,
        "pf_discovery_min": 1.30,
        "pf_oos_min": 1.20,
        "pf_holdout_min": 1.15,
        "walk_forward_stability_min": 0.5,
        "win_months_pct_min": 55,
        "top_month_concentration_max_pct": 40,
    }


def test_standalone_pass_strict_thresholds():
    stats = {
        "n_per_year": 400,
        "pf_discovery": 1.40,
        "pf_oos": 1.25,
        "pf_holdout": 1.18,
        "walk_forward_stability": 0.7,
        "win_months_pct": 60,
        "top_month_concentration_pct": 30,
        "rule_orthogonal": True,
    }
    verdict = evaluate_standalone(stats, _ship_config_standalone())
    assert verdict.shipped is True
    assert verdict.reasons == []


def test_standalone_fail_pf_oos():
    stats = {
        "n_per_year": 400, "pf_discovery": 1.40, "pf_oos": 1.10,
        "pf_holdout": 1.18, "walk_forward_stability": 0.7,
        "win_months_pct": 60, "top_month_concentration_pct": 30,
        "rule_orthogonal": True,
    }
    verdict = evaluate_standalone(stats, _ship_config_standalone())
    assert verdict.shipped is False
    assert any("pf_oos" in r for r in verdict.reasons)


def test_ensemble_feature_pass():
    cfg = {
        "n_per_year_min": 50, "n_per_year_max": 299,
        "effect_size_min_sigma": 0.4,
        "walk_forward_stability_min": 0.5,
        "live_setup_pf_lift_min": 0.15,
    }
    stats = {
        "n_per_year": 120,
        "effect_size_sigma": 0.6,
        "walk_forward_stability": 0.65,
        "live_setup_pf_lift": 0.22,
    }
    verdict = evaluate_ensemble_feature(stats, cfg)
    assert verdict.shipped is True
