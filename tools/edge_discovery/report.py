"""Aggregate framework outputs into a single decision report.

Reads JSON outputs from parity gates and discovery targets, emits a
markdown decision report. Setup names + report paths come from
config/pipelines/base_config.json -> edge_discovery.report_generator.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List

from services.config_loader import load_base_config


_REPO = Path(__file__).resolve().parents[2]
_REPORT_DIR = _REPO / "reports" / "edge_discovery"


def _safe_load_json(path: Path) -> dict:
    if not path.exists():
        return {"_error": f"missing: {path.name}"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"_error": f"parse error: {e}"}


def _fmt(v, kind: str = "pct") -> str:
    """Format a maybe-missing numeric. Returns 'NA' if not numeric."""
    if not isinstance(v, (int, float)):
        return "NA"
    if kind == "pct":
        return f"{v:.1f}%"
    if kind == "pp":
        return f"{v:.1f}pp"
    if kind == "ret":
        return f"{v:.4f}"
    if kind == "t":
        return f"{v:.2f}"
    return f"{v}"


def build_report() -> str:
    cfg = load_base_config()
    rcfg = cfg["edge_discovery"]["report_generator"]
    parity_setups: List[str] = list(rcfg["parity_setups"])
    discovery_targets: List[str] = list(rcfg["discovery_targets"])
    ensemble_summary_filename: str = str(rcfg["ensemble_summary_filename"])
    decision_report_filename: str = str(rcfg["decision_report_filename"])
    top_n_target_regions: int = int(rcfg["top_n_target_regions_in_report"])

    lines: List[str] = []
    lines.append("# Edge-First Discovery Framework — Decision Report")
    lines.append(f"_Generated: {datetime.now().isoformat()}_\n")

    lines.append("## Parity Gate")
    for setup in parity_setups:
        data = _safe_load_json(_REPORT_DIR / f"parity_{setup}.json")
        if "_error" in data:
            lines.append(f"- **{setup}**: NO-DATA ({data['_error']})")
            continue
        v = data.get("verdict") or {}
        passed = v.get("passed", False)
        lines.append(
            f"- **{setup}**: {'PASS' if passed else 'FAIL'} "
            f"(pf_delta={_fmt(v.get('pf_delta_pct'), 'pct')}, "
            f"wr_delta={_fmt(v.get('wr_delta_pp'), 'pp')}, "
            f"n_delta={_fmt(v.get('n_delta_pct'), 'pct')})"
        )
        if not passed:
            failures = v.get("failures") or []
            if failures:
                lines.append(f"  - failures: {failures}")
    lines.append("")

    for target in discovery_targets:
        regions_path = _REPORT_DIR / f"{target}_regions.json"
        data = _safe_load_json(regions_path)
        lines.append(f"## Discovery Target: `{target}`")
        if isinstance(data, dict) and "_error" in data:
            lines.append(f"- NO-DATA ({data['_error']})")
            lines.append("")
            continue
        if isinstance(data, list):
            lines.append(f"- regions found: {len(data)}")
            lines.append(f"- top {top_n_target_regions} by t_proxy:")
            for r in data[:top_n_target_regions]:
                cut = r.get("feature_cut", {})
                lines.append(
                    f"  - cut={cut} n={r.get('n')} "
                    f"mean_ret={_fmt(r.get('mean_return'), 'ret')} "
                    f"t_proxy={_fmt(r.get('t_proxy'), 't')}"
                )
        lines.append("")

    lines.append("## Target 3: Ensemble feature mining (live setups)")
    summary = _safe_load_json(_REPORT_DIR / ensemble_summary_filename)
    if isinstance(summary, dict) and "_error" in summary:
        lines.append(f"- NO-DATA ({summary['_error']})")
    elif isinstance(summary, dict):
        for setup, info in summary.items():
            lines.append(
                f"- **{setup}**: n_trades={info.get('n_trades', 'NA')}, "
                f"pnl_col={info.get('pnl_col_used', 'NA')}"
            )
            for r in info.get("top_3", []) or []:
                cut = r.get("feature_cut", {})
                lines.append(
                    f"  - cut={cut} n={r.get('n')} "
                    f"mean={_fmt(r.get('mean_return'), 'ret')} "
                    f"t_proxy={_fmt(r.get('t_proxy'), 't')}"
                )
    lines.append("")

    md = "\n".join(lines)
    out_path = _REPORT_DIR / decision_report_filename
    out_path.write_text(md, encoding="utf-8")
    return md


if __name__ == "__main__":
    print(build_report())
