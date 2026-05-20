"""Confidence card: ties all 3 framework components together per setup.

For each setup with OCI production trades, produces a "confidence card" with:
  - Aggregate BCa CI on PF / expectancy / win rate (Component 1)
  - Per-regime PF + CI table (Component 2)
  - Raw Sharpe + selection-bias-haircut Sharpe (Component 3)

Output is INTERVALS, not binary verdicts. Researcher applies judgment.

The framework does NOT have ship/no-ship thresholds. Per
`reports/sub9_sanity/_per_trade_validation_research.md`, the literature is
genuinely silent on per-trade ship thresholds. Any threshold would be
folklore. The card surfaces the evidence; the researcher decides.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from tools.methodology.confidence.bootstrap_ci import compute_aggregate_ci
from tools.methodology.confidence.regime_breakdown import (
    compute_per_regime_stats, format_regime_table,
)
from tools.methodology.confidence.selection_bias import (
    analyze_setups_selection_bias,
)


def render_card(
    setup_name: str,
    trades_df: pd.DataFrame,
    *,
    selection_bias_haircut=None,
    effective_N: Optional[int] = None,
    source: str = "oci",
) -> str:
    """Render a single setup's confidence card as markdown text."""
    n = len(trades_df)
    if n == 0:
        return f"# {setup_name}\n\nNo trades. Skip.\n"

    # Date range
    dates = pd.to_datetime(trades_df["signal_date"]).dt.date
    period = f"{dates.min()} to {dates.max()}"

    # Net total
    net_total = float(trades_df["net_pnl_inr"].sum())

    # Component 1: aggregate CIs
    agg = compute_aggregate_ci(trades_df)
    pf = agg["pf"]
    exp = agg["expectancy"]
    wr = agg["win_rate"]

    # Component 2: per-regime
    regime_stats = compute_per_regime_stats(trades_df)
    regime_table = format_regime_table(regime_stats)

    source_label = source.upper()
    lines = [
        f"# CONFIDENCE CARD: {setup_name}  [source: {source_label}]",
        "",
        f"**Period:** {period}",
        f"**Trades:** {n:,}",
        f"**Net Rs:** {net_total:+,.0f}  ({'production-traded sizes' if source == 'oci' else 'sanity NAKED qty (un-leveraged)'})",
        "",
    ]
    if source == "sanity":
        lines += [
            "> **INFLATION CAVEAT (Lesson #13):** Sanity Mode B systematically",
            "> OVER-ESTIMATES production PF, especially for SHORT setups using the",
            "> entry_zone retest filter. Read this card as a NEGATIVE filter:",
            "> a sanity-RED verdict confirms retirement, a sanity-GREEN verdict",
            "> does NOT permit revival without an OCI structure-code run to verify.",
            "",
        ]
    lines += [
        "## Aggregate (BCa 95% CI, B=5000)",
        "",
        f"- **Profit Factor:** {pf.point_estimate:.3f}  CI [{pf.ci_lower:.3f}, {pf.ci_upper:.3f}]",
        f"- **Expectancy (Rs/trade):** {exp.point_estimate:+,.2f}  CI [{exp.ci_lower:+,.2f}, {exp.ci_upper:+,.2f}]",
        f"- **Win rate:** {wr.point_estimate:.3f}  CI [{wr.ci_lower:.3f}, {wr.ci_upper:.3f}]",
        "",
        "## Per-regime breakdown (BCa 95% CI on PF per regime)",
        "",
        "```",
        regime_table,
        "```",
        "",
    ]

    if selection_bias_haircut is not None:
        haircut = selection_bias_haircut
        lines += [
            "## Selection-bias correction (Harvey-Liu haircut)",
            "",
            f"- **Raw daily Sharpe (annualized):** {haircut.raw_sharpe:.3f}",
            f"- **Effective N (ONC clustering):** {haircut.effective_M}  "
            f"(out of {effective_N or 'N/A'} total setups clustered)",
            f"- **Method:** {haircut.method}",
            f"- **Adjusted Sharpe:** {haircut.adjusted_sharpe:.3f}",
            f"- **Haircut:** {haircut.haircut_pct:.1f}%",
            "",
        ]

    lines += [
        "## Researcher interpretation guide",
        "",
        "The framework produces INTERVALS, not binary ship/no-ship verdicts.",
        "Per `reports/sub9_sanity/_per_trade_validation_research.md`, the literature",
        "is genuinely silent on per-trade ship thresholds. Any threshold would be",
        "folklore.",
        "",
        "**Things to consider when interpreting:**",
        "",
        "1. **Aggregate PF CI lower bound:** if well above 1.0, edge is statistically",
        "   distinguishable from break-even. Below 1.0 means observed PF could be noise.",
        "2. **Per-regime breakdown:** if PF is positive in MOST regimes (especially",
        "   the LARGEST regimes by trade count), edge appears robust to regime shifts.",
        "   Lopez de Prado tactical paradigm: if positive only in specific regimes,",
        "   consider gating to those regimes.",
        "3. **Selection-bias-adjusted Sharpe:** if positive after haircut, edge is",
        "   not just selection bias. Negative adjusted Sharpe = the survivor was",
        "   probably the best of N coin flips.",
        "4. **R3 (post_election_consolidation) is the WEAKEST regime by evidence**",
        "   (boundary interpolated). Wide CIs in R3 are expected; don't over-interpret.",
        "5. **R2 and R3 are short (~30-50 days each)** — low-frequency setups will",
        "   have very wide CIs in these regimes. Honest reporting.",
        "",
        "---",
        "",
        "_Generated by `tools/methodology/confidence/confidence_card.py`._",
        "_Framework: bootstrap BCa (Efron-Tibshirani 1993) + regime decomposition",
        "(Lopez de Prado tactical 2019) + Harvey-Liu haircut (Harvey-Liu 2015)._",
    ]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--oci-canonical-dir", type=Path,
                   default=_REPO_ROOT / "reports" / "oci_canonical_v2",
                   help="Directory containing <setup>_oci_canonical.csv files")
    p.add_argument("--sanity-canonical-dir", type=Path,
                   default=_REPO_ROOT / "reports" / "sanity_canonical",
                   help="Directory containing <setup>_sanity_canonical.csv files (retired setups without OCI data)")
    p.add_argument("--include-sanity", action="store_true",
                   help="Include sanity-canonical setups in the corpus (raises effective M for haircut)")
    p.add_argument("--out-dir", type=Path,
                   default=_REPO_ROOT / "reports" / "confidence_cards",
                   help="Directory to write confidence cards")
    p.add_argument("--setups", nargs="*", default=None,
                   help="Optional list of setup names to process; default = all")
    p.add_argument("--haircut-method", choices=["Bonferroni", "Holm", "BHY"],
                   default="Bonferroni")
    args = p.parse_args(argv)

    # Load all setup CSVs and track source
    setups_trades: Dict[str, pd.DataFrame] = {}
    setup_sources: Dict[str, str] = {}

    for csv_path in sorted(args.oci_canonical_dir.glob("*_oci_canonical.csv")):
        setup_name = csv_path.name.replace("_oci_canonical.csv", "")
        if args.setups and setup_name not in args.setups:
            continue
        df = pd.read_csv(csv_path)
        setups_trades[setup_name] = df
        setup_sources[setup_name] = "oci"
        print(f"  [OCI] loaded {setup_name}: {len(df)} trades")

    if args.include_sanity and args.sanity_canonical_dir.exists():
        for csv_path in sorted(args.sanity_canonical_dir.glob("*_sanity_canonical.csv")):
            setup_name = csv_path.name.replace("_sanity_canonical.csv", "")
            if args.setups and setup_name not in args.setups:
                continue
            if setup_name in setups_trades:
                print(f"  [SANITY] skip {setup_name} (already loaded from OCI)")
                continue
            df = pd.read_csv(csv_path)
            setups_trades[setup_name] = df
            setup_sources[setup_name] = "sanity"
            print(f"  [SANITY] loaded {setup_name}: {len(df)} trades")

    if not setups_trades:
        print("No setups found", file=sys.stderr)
        return 2

    # Component 3: selection-bias analysis across ALL loaded setups
    print(f"\n[Selection-bias] Computing ONC + Harvey-Liu across {len(setups_trades)} setups...")
    haircut_results, effective_N, cluster_map = analyze_setups_selection_bias(
        setups_trades, haircut_method=args.haircut_method,
    )
    print(f"  Effective N (ONC): {effective_N}")
    print(f"  Cluster assignments: {cluster_map}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for setup_name, trades in setups_trades.items():
        haircut = haircut_results.get(setup_name)
        card_text = render_card(
            setup_name, trades,
            selection_bias_haircut=haircut,
            effective_N=effective_N,
            source=setup_sources[setup_name],
        )
        out = args.out_dir / f"{setup_name}_confidence_card.md"
        out.write_text(card_text, encoding="utf-8")
        print(f"  wrote {out.name}  ({setup_sources[setup_name]})")

    print(f"\nDone. Cards in {args.out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
