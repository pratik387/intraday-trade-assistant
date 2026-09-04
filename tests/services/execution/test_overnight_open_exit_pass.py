"""The 09:16 open-exit pass, and the diagnostic that explains a dropped AMO.

2026-09-04: all four overnight AMO SELLs were silently dropped — accepted at
16:05 the previous evening (order ids returned, GTTs placed alongside) and then
absent from the next morning's order book entirely, neither filled nor visibly
rejected. `run_verify_exit` already knows how to recover from this: it adopts an
out-of-band sell, or failsafes the unsold remainder, or re-prices a still-OPEN
AMO to marketable. But it only ran at 09:30, so the positions sat live for
fourteen minutes after the open and the operator sold them by hand at 09:16.

Two changes, both covered here:

  1. `exits_only` lets a 09:16 cron do the exit work and skip the ~4-minute
     close_dn baseline+candidate build. Running that heavy fetch twice would
     double the API load for nothing; only the 09:30 pass needs it, and it has
     until 15:25 to produce candidates.

  2. An unfilled AMO now logs its broker status BEFORE anything else. Kite's
     order_history covers only the current day, so an AMO placed at 16:05 on T
     is diagnosable ONLY on T+1 morning — and the out-of-band branch used to
     adopt the manual sell and return without ever asking why. That is exactly
     how the 2026-09-04 root cause became unrecoverable.
"""
import ast
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

import services.execution.overnight_handlers as oh

REPO = Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------
# 1. exits_only
# --------------------------------------------------------------------------

def test_run_verify_exit_accepts_exits_only_defaulting_to_false():
    """Default False: every existing caller keeps building the baseline."""
    p = inspect.signature(oh.run_verify_exit).parameters
    assert "exits_only" in p, "run_verify_exit must accept exits_only"
    assert p["exits_only"].default is False, "must default to False"
    assert p["exits_only"].kind is inspect.Parameter.KEYWORD_ONLY


def test_exits_only_short_circuits_the_baseline_build():
    src = inspect.getsource(oh.run_verify_exit)
    i_flag = src.index("if exits_only:")
    i_base = src.index('elif getattr(broker, "_data_sdk", None) is not None:')
    assert i_flag < i_base, "the exits_only check must precede the baseline gate"
    # It must be the SAME if/elif chain, not a second independent branch that
    # would let both run.
    assert "baseline_skipped" in src[i_flag:i_base]


def test_baseline_gate_still_keyed_on_data_sdk():
    """The pre-existing gate must survive: exits_only is additive, not a rewrite."""
    src = inspect.getsource(oh.run_verify_exit)
    assert 'getattr(broker, "_data_sdk", None) is not None' in src


def test_exits_only_does_not_touch_the_settlement_path():
    """The flag must gate ONLY the baseline build — never the exit work.

    A flag that skipped settlement would turn the safety net into the bug it
    exists to prevent.
    """
    src = inspect.getsource(oh.run_verify_exit)
    tree = ast.parse(src.lstrip())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    guarded = []
    for node in ast.walk(fn):
        if isinstance(node, ast.If) and any(
            isinstance(d, ast.Name) and d.id == "exits_only"
            for d in ast.walk(node.test)
        ):
            guarded.append(ast.dump(node))
    assert guarded, "expected an `if exits_only:` branch"
    body = "\n".join(guarded)
    for forbidden in ("pool.settle", "_place_failsafe_sell", "cancel_gtt"):
        assert forbidden not in body, (
            f"exits_only must not gate {forbidden} — it gates only the "
            "baseline build")


# --------------------------------------------------------------------------
# 2. AMO diagnostic
# --------------------------------------------------------------------------

def test_unfilled_amo_status_is_captured_before_out_of_band_adoption():
    """Order why-it-failed BEFORE the branch that returns early.

    The out-of-band adoption path exits the block with `sell_price` set, so a
    diagnostic placed after it never runs on the very days it is needed.
    """
    src = inspect.getsource(oh.run_verify_exit)
    i_check = src.index("sell_price = _live_check_amo_fill(")
    i_diag = src.index("AMO_UNFILLED")
    i_oob = src.index("oob = _find_out_of_band_sell(")
    assert i_check < i_diag < i_oob, (
        "the AMO status capture must sit between the fill check and the "
        "out-of-band adoption")


def test_diagnostic_records_the_fields_that_identify_the_cause():
    src = inspect.getsource(oh.run_verify_exit)
    blk = src[src.index("AMO_UNFILLED"):src.index("oob = _find_out_of_band_sell(")]
    for field in ("status", "filled_quantity", "product", "variety"):
        assert field in blk, f"diagnostic must record {field}"
    assert "status_message" in blk, "the rejection reason is the whole point"


def test_diagnostic_cannot_break_the_exit_path():
    """A logging failure must never stop a position from being sold."""
    src = inspect.getsource(oh.run_verify_exit)
    blk = src[src.index("AMO_UNFILLED") - 400:src.index("oob = _find_out_of_band_sell(")]
    assert "try:" in blk and "except Exception" in blk, (
        "the status fetch must be wrapped — a broker hiccup here would "
        "otherwise abort the settlement it was added to explain")


# --------------------------------------------------------------------------
# 3. CLI wiring
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mode,action", [
    ("intraday", "run"),
    ("multi_day", "entry"),
    ("multi_day", "exit"),
    ("overnight", "entry"),
    ("overnight", "place-exit"),
])
def test_exits_only_rejected_outside_overnight_verify_exit(mode, action):
    """Must print and exit(2) like every sibling guard.

    The first version of this guard set the message but forgot the print/exit,
    so `--mode intraday --action run --exits-only` fell through and launched the
    LIVE intraday daemon instead of erroring.
    """
    r = subprocess.run(
        [sys.executable, "main.py", "--mode", mode, "--action", action, "--exits-only"],
        cwd=REPO, capture_output=True, text=True, timeout=180,
    )
    assert r.returncode == 2, (
        f"--mode={mode} --action={action} --exits-only must exit 2, "
        f"got {r.returncode}\nstdout={r.stdout[-400:]}\nstderr={r.stderr[-400:]}")
    assert "--exits-only is only valid" in r.stderr


def test_every_validation_block_prints_and_exits():
    """Structural guard against repeating the fall-through bug."""
    src = (REPO / "main.py").read_text(encoding="utf-8").splitlines()
    missing = [
        i + 1 for i, line in enumerate(src)
        if "parser_error = " in line
        and "sys.exit(2)" not in "\n".join(src[i:i + 6])
    ]
    assert not missing, f"validation blocks without sys.exit(2) at lines {missing}"


def test_main_forwards_the_flag():
    src = (REPO / "main.py").read_text(encoding="utf-8")
    assert "exits_only=bool(args.exits_only)" in src


# --------------------------------------------------------------------------
# 4. cron script
# --------------------------------------------------------------------------

def test_cron_script_passes_the_flag_only_when_asked():
    sh = (REPO / "scripts" / "cron-verify-exit.sh").read_text(encoding="utf-8")
    assert '"${EXITS_ONLY:-0}" == "1"' in sh, (
        "must default to OFF when unset — the 09:30 and 10:30 passes still "
        "need the baseline build")
    assert "--exits-only" in sh
    i_flag = sh.index("EXITS_FLAG=")
    i_run = sh.index("main.py --mode overnight --action verify-exit")
    assert i_flag < i_run, "EXITS_FLAG must be set before the invocation"
    assert "$EXITS_FLAG" in sh[i_run:], "the invocation must actually use it"
