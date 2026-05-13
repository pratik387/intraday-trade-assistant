"""Static guard: every detector emits a target_anchor_type the dispatcher
knows about.

Background — 2026-05-13
-----------------------
Three detectors (delivery_pct_anomaly_short, options_vol_iv_rank_revert,
capitulation_long_morning) shipped with `target_anchor_type="arithmetic"`.
That key isn't in `services/target_recalc.py`'s dispatch table, so the
dispatcher logged a warning and silently fell through to
`_recalc_structural` — which keeps detect-time T1/T2 unchanged. On
Discovery 2yr, delivery_pct lost ~Rs.70K (PF 0.987 -> 0.903) purely
from this anchor mismatch.

This test scans every `structures/*_structure.py` for `target_anchor_type=...`
literals and fails if any are outside the dispatcher's valid set.

It runs statically (no detector imports / instantiation needed), so it's
cheap and stays green even if a detector has missing config keys.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from services.target_recalc import _VALID_ANCHOR_TYPES


_REPO = pathlib.Path(__file__).resolve().parents[2]
_STRUCT_DIR = _REPO / "structures"


def _scan_for_anchor_assignments():
    """Return list of (file_path, lineno, value) for every literal assignment
    `target_anchor_type=<str>` found in structures/*_structure.py."""
    hits = []
    for fp in sorted(_STRUCT_DIR.glob("*_structure.py")):
        try:
            tree = ast.parse(fp.read_text(encoding="utf-8"))
        except SyntaxError as e:
            pytest.fail(f"{fp.name} has a syntax error: {e}")
        for node in ast.walk(tree):
            # Captures both kwarg form `target_anchor_type="r_multiple"`
            # and attribute assignment `self.target_anchor_type = "..."`
            if isinstance(node, ast.keyword) and node.arg == "target_anchor_type":
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    hits.append((fp.name, node.value.lineno, node.value.value))
            elif isinstance(node, ast.Assign):
                for tgt in node.targets:
                    name = None
                    if isinstance(tgt, ast.Name):
                        name = tgt.id
                    elif isinstance(tgt, ast.Attribute):
                        name = tgt.attr
                    if name == "target_anchor_type" and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        hits.append((fp.name, node.value.lineno, node.value.value))
    return hits


def test_every_detector_uses_a_recognized_anchor_type():
    hits = _scan_for_anchor_assignments()
    assert hits, "No `target_anchor_type=...` assignments found in structures/ — scan is broken"

    bad = [(f, ln, v) for (f, ln, v) in hits if v not in _VALID_ANCHOR_TYPES]
    if bad:
        msg_lines = [
            f"Found {len(bad)} detector(s) with unrecognized target_anchor_type. "
            f"Valid set: {_VALID_ANCHOR_TYPES}.",
            "",
            "Offenders (file:line value):",
        ]
        msg_lines.extend(f"  {f}:{ln}  {v!r}" for f, ln, v in bad)
        pytest.fail("\n".join(msg_lines))


def test_dispatcher_handles_every_anchor_in_use():
    """Belt-and-braces: every anchor literal we ship must round-trip through
    the dispatcher without raising UnknownAnchorTypeError. We don't run the
    full plan recalc — just confirm the anchor string is in the valid set."""
    used_anchors = {v for (_, _, v) in _scan_for_anchor_assignments()}
    unknown = used_anchors - set(_VALID_ANCHOR_TYPES)
    assert not unknown, (
        f"Anchors used by detectors but not in dispatcher: {unknown}. "
        f"Add them to target_recalc.py OR fix the detector typo."
    )
