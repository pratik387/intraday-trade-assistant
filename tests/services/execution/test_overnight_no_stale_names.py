"""Catch NameErrors that only surface at runtime in cron-only branches.

2026-08-14..08-18 incident: bbfd3eb renamed `paper_enabled_setups` ->
`managed_setups` inside run_verify_exit with a BOUNDED line-range replace
(range(1263, 1500)). The function extends past 1500, so one reference
survived — inside the best-effort baseline-build block, whose
`except Exception` swallowed the NameError. Therefore:

  - exits kept settling and the run reported success,
  - candidates_latest.json silently stopped updating after 2026-08-13,
  - the NEXT day's 15:26 entry cron fell back to a ~232s full universe build
    and blew the 15:29:20 placement guard.

The live overnight book placed NOTHING for three sessions — 6, 1 and 3 ranked
signals forfeited — and the only visible symptom was a cutoff warning in a
different log the following day.

No linter is installed in this venv, and no unit test reaches that branch (it
needs a broker with a live _data_sdk), so these are static guards.
"""
import ast
import builtins
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[3] / "services" / "execution" / "overnight_handlers.py"
TEXT = SRC.read_text(encoding="utf-8")
TREE = ast.parse(TEXT)

# TOP-LEVEL functions only. Nested defs are analysed as part of their parent's
# scope — checking a closure standalone would flag every enclosing variable it
# legitimately reads.
TOP_FUNCS = {n.name: n for n in TREE.body if isinstance(n, ast.FunctionDef)}


def _module_scope() -> set:
    # module dunders are always bound at runtime
    names = set(dir(builtins)) | {"__file__", "__name__", "__doc__",
                                  "__package__", "__spec__", "__loader__"}
    for node in TREE.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Try):
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    for a in sub.names:
                        names.add((a.asname or a.name).split(".")[0])
                elif isinstance(sub, ast.Assign):
                    names.update(t.id for t in sub.targets if isinstance(t, ast.Name))
    return names


MODULE = _module_scope()


def _undefined(fn: ast.FunctionDef) -> set:
    bound, loaded = set(), set()
    for n in ast.walk(fn):
        if isinstance(n, ast.arg):
            bound.add(n.arg)
        elif isinstance(n, ast.Name):
            (bound if isinstance(n.ctx, (ast.Store, ast.Del)) else loaded).add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                bound.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            bound.update(n.names)
    return loaded - bound - MODULE


@pytest.mark.parametrize("name", sorted(TOP_FUNCS))
def test_no_undefined_names(name):
    """A NameError in a cron-only branch costs a whole session before anyone sees it."""
    missing = _undefined(TOP_FUNCS[name])
    assert not missing, f"{name}() references undefined name(s): {sorted(missing)}"


def test_run_verify_exit_does_not_reference_the_entry_legs_variable():
    """The exact regression, checked as CODE — a comment mentioning the old
    name is fine, an ast.Name load of it is not."""
    fn = TOP_FUNCS["run_verify_exit"]
    hits = [n.id for n in ast.walk(fn) if isinstance(n, ast.Name)
            and n.id == "paper_enabled_setups"]
    assert not hits, "run_verify_exit still reads run_entry's variable — incomplete rename"


def test_baseline_build_failure_logs_critical():
    """Silent and delayed: exits still succeed, but the next session's entry
    cron takes the slow path and misses its placement window."""
    i = TEXT.index("baseline build FAILED")
    assert "logger.critical" in TEXT[max(0, i - 500):i], \
        "a failed baseline build must log CRITICAL — it costs the next session's signals"
