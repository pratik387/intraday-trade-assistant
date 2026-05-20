"""Portfolio-level summary view: all setups side-by-side.

For a researcher scanning across the portfolio, this surfaces:
- Aggregate PF CI per setup
- Which regimes are weak per setup
- Selection-bias-adjusted Sharpe per setup
- Compared against retired-vs-active label for sanity check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from tools.methodology.confidence.bootstrap_ci import compute_aggregate_ci
from tools.methodology.confidence.regime_breakdown import compute_per_regime_stats
from tools.methodology.confidence.selection_bias import analyze_setups_selection_bias


# Known labels — used for visual sanity check, NOT as input to the framework
RETIRED_SETUPS = {
    # OCI-data retired
    "circuit_release_fade_short",
    "mis_unwind_vwap_revert_short",
    "round_number_sweep_short",
    # Sanity-data retired
    "capitulation_long_v2",
    "pre_results_t1_fade_v2",
    "block_deal_accumulation_long",
    "volume_spike_reversal_midsession",
    "expiry_pin_strike_reversal",
}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--oci-canonical-dir", type=Path,
                   default=_REPO_ROOT / "reports" / "oci_canonical_v2")
    p.add_argument("--sanity-canonical-dir", type=Path,
                   default=_REPO_ROOT / "reports" / "sanity_canonical")
    p.add_argument("--include-sanity", action="store_true")
    p.add_argument("--out", type=Path,
                   default=_REPO_ROOT / "reports" / "confidence_cards" / "_portfolio_summary.md")
    args = p.parse_args(argv)

    setups_trades: Dict[str, pd.DataFrame] = {}
    setup_sources: Dict[str, str] = {}
    for csv_path in sorted(args.oci_canonical_dir.glob("*_oci_canonical.csv")):
        name = csv_path.name.replace("_oci_canonical.csv", "")
        setups_trades[name] = pd.read_csv(csv_path)
        setup_sources[name] = "oci"

    if args.include_sanity and args.sanity_canonical_dir.exists():
        for csv_path in sorted(args.sanity_canonical_dir.glob("*_sanity_canonical.csv")):
            name = csv_path.name.replace("_sanity_canonical.csv", "")
            if name in setups_trades:
                continue
            setups_trades[name] = pd.read_csv(csv_path)
            setup_sources[name] = "sanity"

    if not setups_trades:
        print("No setups", file=sys.stderr)
        return 2

    # Component 3 first — applies across all setups
    print(f"Computing selection-bias correction across {len(setups_trades)} setups...")
    haircut_results, effective_N, cluster_map = analyze_setups_selection_bias(setups_trades)

    rows = []
    for name, df in setups_trades.items():
        agg = compute_aggregate_ci(df, n_resamples=2000)  # smaller for speed
        regime_stats = compute_per_regime_stats(df, n_resamples=1500)
        haircut = haircut_results[name]

        # Count regimes where CI lower < 1.0 (edge not statistically real)
        weak_regimes = 0
        regimes_with_trades = 0
        for s in regime_stats:
            if s.n_trades < 30:
                continue
            regimes_with_trades += 1
            if s.pf_ci.ci_lower < 1.0:
                weak_regimes += 1

        rows.append({
            "setup": name,
            "source": setup_sources.get(name, "oci").upper(),
            "label": "RETIRED" if name in RETIRED_SETUPS else "ACTIVE",
            "n_trades": len(df),
            "net_rs": float(df["net_pnl_inr"].sum()),
            "pf_point": agg["pf"].point_estimate,
            "pf_ci_lo": agg["pf"].ci_lower,
            "pf_ci_hi": agg["pf"].ci_upper,
            "wr_point": agg["win_rate"].point_estimate,
            "raw_sharpe": haircut.raw_sharpe,
            "adj_sharpe": haircut.adjusted_sharpe,
            "haircut_pct": haircut.haircut_pct,
            "regimes_n>=30": regimes_with_trades,
            "regimes_weak_CI": weak_regimes,
        })

    df = pd.DataFrame(rows).sort_values("net_rs", ascending=False)

    # Render markdown table
    n_oci = sum(1 for s in setup_sources.values() if s == "oci")
    n_sanity = sum(1 for s in setup_sources.values() if s == "sanity")
    md = ["# Portfolio Confidence Summary",
          "",
          f"**Setups analyzed:** {len(df)} (OCI={n_oci}, SANITY={n_sanity})  |  **Effective N (ONC):** {effective_N}  |  **Haircut method:** Bonferroni",
          "",
          "**SANITY rows carry Lesson #13 inflation caveat:** RED verdict confirms retirement; GREEN verdict does NOT permit revival.",
          "",
          "## Aggregate metrics (BCa 95% CI on PF)",
          "",
          "| Src | Label | Setup | n | Net Rs | PF point | PF [low, high] | WR | Raw SR | Adj SR | regimes (>=30) weak |",
          "|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|"]
    for _, r in df.iterrows():
        md.append(
            f"| {r['source']} | {r['label']} | {r['setup']} | {r['n_trades']:,} | "
            f"{r['net_rs']:+,.0f} | {r['pf_point']:.2f} | "
            f"[{r['pf_ci_lo']:.2f}, {r['pf_ci_hi']:.2f}] | "
            f"{r['wr_point']:.2f} | "
            f"{r['raw_sharpe']:+.2f} | {r['adj_sharpe']:+.2f} | "
            f"{r['regimes_weak_CI']}/{r['regimes_n>=30']} |"
        )

    md += [
        "",
        "## Reading the table",
        "",
        "- **PF [low, high]:** if **low > 1.0**, aggregate edge is statistically distinguishable from break-even. If low < 1.0, observed PF could be noise.",
        "- **Adj SR:** Bonferroni haircut for testing M=" + f"{effective_N}" + " effective setups. Same percentage haircut applies to all (function of M, not individual setup).",
        "- **Regimes weak CI:** number of regimes (with n>=30) where the per-regime PF CI lower bound is below 1.0 — flags regime-conditional fragility.",
        "",
        "## Researcher interpretation (per setup)",
        "",
    ]

    for _, r in df.iterrows():
        verdict_signals = []
        if r["pf_ci_lo"] > 1.0:
            verdict_signals.append("**aggregate edge real** (PF CI lower > 1.0)")
        else:
            verdict_signals.append("**aggregate edge uncertain** (PF CI crosses 1.0)")
        if r["adj_sharpe"] > 0:
            verdict_signals.append(f"adj Sharpe positive ({r['adj_sharpe']:.2f})")
        else:
            verdict_signals.append(f"adj Sharpe NEGATIVE ({r['adj_sharpe']:.2f})")
        if r["regimes_weak_CI"] > 0 and r["regimes_n>=30"] > 0:
            verdict_signals.append(f"weak in {r['regimes_weak_CI']}/{r['regimes_n>=30']} regimes")

        md.append(f"- **{r['setup']}** [{r['source']}/{r['label']}]: " + "; ".join(verdict_signals))

    md += [
        "",
        "## What the framework does NOT decide",
        "",
        "- No ship/no-ship binary verdict. Researcher reads intervals and judges.",
        "- No threshold on Adj Sharpe — \"positive\" is necessary but not sufficient evidence.",
        "- No threshold on PF — interval lower bound > 1.0 means edge is real, but doesn't tell you if it's worth the operational cost.",
        "- Regime-conditional fragility: count of weak regimes is informational. López de Prado tactical paradigm: a setup weak in 2 regimes might still ship with regime gates.",
        "",
        "## Calibration sanity check",
        "",
        "Compare RETIRED vs ACTIVE rows in the table above. If the framework is well-calibrated:",
        "- Retired setups should generally have weaker indicators (lower PF, negative or low adj SR, more weak regimes)",
        "- Active setups should generally have stronger indicators",
        "- Both sets should overlap in the marginal middle — there's no clean threshold",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
