"""Stage 5: Human narrative gate template generator.

Generates one markdown template per Stage 3 survivor. Human fills in the
WHY section; unfilled templates mean the rule has not passed Stage 5.

Per spec Section 3.3 Stage 5.
"""
from pathlib import Path
from typing import Any, Dict, List


TEMPLATE = """# Narrative Gate \u2014 {rule_id}

## Setup
`{setup}`

## Conditional rule
{conditioner_desc}

## Discovery stats
| Metric | Value |
|--------|-------|
| N | {n} |
| PF (full) | {pf} |
| PF (h1) | {pf_h1} |
| PF (h2) | {pf_h2} |
| Win rate | {wr_pct}% |
| Avg PnL (raw, Rs) | {avg_pnl} |

## Auto-generated context
### Canonical pro definition
(Paste from `docs/edge_discovery/audit/{setup_audit_link}` \u2014 Item 1)

### Stage 4 top features
(If Stage 4 was run, paste SHAP top features here; else leave blank.)

### Suggested microstructure rationale
This setup passes statistical gates in the cell `{conditioner_desc}`. Candidate
mechanisms to consider:
- Retail long-bias: do losers in the opposite direction cluster under this condition?
- Institutional flow: does this regime correlate with measurable FII/DII activity?
- Microstructure: is price action at this hour bucket dominated by MIS unwinding,
  opening auction noise, or expiry gamma flow?

## Human narrative (REQUIRED \u2014 unfilled = auto-REJECT)

### WHY does this work? What market participant behavior creates this edge?
_(Human-written. Reference a specific participant, a specific behavior, and a
specific structural reason the edge persists. LLM-plausible is insufficient \u2014
write only what you would defend to another trader.)_

PARTICIPANT:

BEHAVIOR:

STRUCTURAL REASON IT PERSISTS:

## Pass/fail decision

- [ ] APPROVED \u2014 narrative plausible and grounded in market reality
- [ ] REJECTED \u2014 cannot articulate why this would persist

**Signed:**
**Date:**
"""


def generate_narrative_templates(
    stage3_survivors: List[Dict[str, Any]],
    stage3_details: List[Dict[str, Any]],
    out_dir: Path,
) -> List[Path]:
    """Generate one template file per Stage 3 survivor rule.

    Returns list of paths created.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Index details by (setup, rule)
    by_rule: Dict[str, Dict[str, Any]] = {}
    for d in stage3_details:
        if not d.get("passed"):
            continue
        rule_id = _rule_id(d)
        by_rule[rule_id] = d

    paths: List[Path] = []
    for surv in stage3_survivors:
        rule_id = surv["rule"]
        cell = by_rule.get(rule_id)
        if cell is None:
            continue
        cond_desc = f"{cell['conditioner']} = {cell['cell_value']}"
        body = TEMPLATE.format(
            rule_id=rule_id,
            setup=cell["setup"],
            conditioner_desc=cond_desc,
            n=cell["n"],
            pf=cell["pf"],
            pf_h1=cell["pf_h1"],
            pf_h2=cell["pf_h2"],
            wr_pct=cell["wr_pct"],
            avg_pnl=cell["avg_pnl"],
            setup_audit_link=f"{cell['setup']}.md",
        )
        safe_name = rule_id.replace("=", "-").replace("+", "_and_")
        path = out_dir / f"{safe_name}.md"
        path.write_text(body, encoding="utf-8")
        paths.append(path)

    # Index file
    index = out_dir / "00-index.md"
    lines = ["# Narrative gate templates", ""]
    for p in paths:
        lines.append(f"- [ ] `{p.name}` \u2014 awaiting human narrative")
    index.write_text("\n".join(lines), encoding="utf-8")

    return paths


def _rule_id(cell: Dict[str, Any]) -> str:
    return f"{cell['setup']}__{cell['conditioner']}={cell['cell_value']}"
