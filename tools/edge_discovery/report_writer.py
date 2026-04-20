"""Markdown + JSON report helpers used by every stage.

Markdown convention: every stage file starts with a summary table of setups +
pass/fail status, then drills into detail per setup. JSON artifacts are
written alongside for machine readability (e.g., stage1_survivors.json).
"""
import json
from pathlib import Path
from typing import Any, Dict, List


def write_stage_report(
    path: Path,
    stage_name: str,
    criteria: str,
    summary_rows: List[Dict[str, Any]],
) -> None:
    """Create a stage report file with header + summary table.

    summary_rows: list of dicts; each must contain 'setup' and 'passed' keys.
    All other keys become columns in the summary table.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append(f"# {stage_name}\n")
    lines.append(f"**Criteria:** {criteria}\n")
    lines.append("")
    if not summary_rows:
        lines.append("_No setups processed._\n")
        path.write_text("\n".join(lines), encoding="utf-8")
        return
    # Column order: setup, status, then remaining keys
    extra_keys = sorted(
        k for k in summary_rows[0].keys() if k not in ("setup", "passed")
    )
    header = ["setup", "status"] + extra_keys
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for row in summary_rows:
        status = "PASS" if row.get("passed") else "FAIL"
        values = [str(row.get("setup", ""))] + [status] + [
            _fmt(row.get(k)) for k in extra_keys
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def append_section(path: Path, heading: str, body: str) -> None:
    """Append a markdown section to an existing report file. Creates the
    parent directory if it doesn't exist (defensive — matches write_stage_report)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write(heading + "\n\n")
        f.write(body + "\n")


def write_json_artifact(path: Path, data: Dict[str, Any]) -> None:
    """Write a JSON artifact. pandas NaN + numpy scalars are sanitized."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    # Check bool BEFORE float/int: Python bool is a subclass of int,
    # and numpy.bool_ is neither (but has .__bool__). Handle both paths.
    if v is True or v is False:
        return "YES" if v else "NO"
    # numpy.bool_ check (without forcing numpy import)
    if type(v).__name__ == "bool_":
        return "YES" if bool(v) else "NO"
    if isinstance(v, float):
        if v != v:  # NaN
            return "—"
        if abs(v) >= 1e6 or (abs(v) < 0.01 and v != 0):
            return f"{v:.2e}"
        return f"{v:.2f}"
    return str(v)


def _json_default(obj: Any) -> Any:
    """Serializer for pandas / numpy values."""
    import numpy as np
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
