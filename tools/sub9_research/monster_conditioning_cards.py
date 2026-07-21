"""Confidence cards for the monster-conditioning RULE-T baskets.

Renders per-setup + pooled cards from the CSVs written by
monster_conditioning_validation.py, using the real confidence framework
(BCa CIs, per-regime breakdown, Harvey-Liu haircut).

Harvey-Liu M=64: the 2-feature rule came from a 17-feature x 4-setup screen
(rounded up); charging the full screen is deliberate per Lesson #15/#16.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.methodology.confidence.bootstrap_ci import compute_aggregate_ci
from tools.methodology.confidence.regime_breakdown import (
    compute_per_regime_stats, format_regime_table,
)
from tools.methodology.confidence.selection_bias import (
    build_daily_equity_curve, daily_sharpe, harvey_liu_haircut,
)

IN_DIR = ROOT / "reports" / "sub9_sanity"
OUT_DIR = ROOT / "reports" / "confidence_cards"
M_TESTS = 64


def card(tag: str, df: pd.DataFrame) -> str:
    lines = [f"# monster-conditioning RULE-T — {tag}", ""]
    dates = pd.to_datetime(df["signal_date"]).dt.date
    lines.append(f"Period: {dates.min()} .. {dates.max()}  n={len(df)}  "
                 f"net=Rs{df.net_pnl_inr.sum():,.0f}")
    agg = compute_aggregate_ci(df)
    for k in ("pf", "expectancy", "win_rate"):
        r = agg[k]
        lines.append(f"- {k}: {r.point_estimate:.3f}  "
                     f"CI95 [{r.ci_lower:.3f}, {r.ci_upper:.3f}]")
    lines.append("")
    lines.append("## Per-regime")
    stats = compute_per_regime_stats(df)
    lines.append(format_regime_table(stats))
    lines.append("")
    curve = build_daily_equity_curve(df)
    raw = daily_sharpe(curve)
    hc = harvey_liu_haircut(tag, curve, M=M_TESTS)
    lines.append(f"## Sharpe: raw={raw:.2f}  Harvey-Liu(M={M_TESTS}) "
                 f"adj={hc.adjusted_sharpe:.2f} (haircut {hc.haircut_pct:.0f}%)")
    lines.append("")
    for per in ("Disc23-24", "OOS25", "Rec26"):
        sub = df[df["per"] == per]
        if not len(sub):
            continue
        x = sub.net_pnl_inr.values
        w = x[x > 0].sum(); l = -x[x < 0].sum()
        lines.append(f"- {per}: n={len(sub)} net=Rs{x.sum():,.0f} "
                     f"PF={w/l if l else np.inf:.2f}")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(IN_DIR.glob("_monster_cond_*_rule_t.csv"))
    for f in files:
        tag = f.stem.replace("_monster_cond_", "").replace("_rule_t", "")
        df = pd.read_csv(f)
        text = card(tag, df)
        out = OUT_DIR / f"monster_cond_rule_t_{tag}.md"
        out.write_text(text, encoding="utf-8")
        print(f"--- {tag} -> {out.name}")
        print(text)


if __name__ == "__main__":
    main()
