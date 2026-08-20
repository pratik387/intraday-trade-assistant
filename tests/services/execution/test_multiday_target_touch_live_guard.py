"""A touched target must never book a fill that no order could have produced.

The multi-day exit leg contains a PAPER modelling assumption: if the day's HIGH
touches `target_px`, it records a sale AT the target. That mirrors the
2026-07-10 study geometry ("sell the snapback climax instead of the faded
close") and is correct in paper.

It is not correct in live, because no resting limit/GTT is ever placed at
`target_px` — the code only *checks* whether the high touched it. A live run
would therefore settle the position in our ledger and drop it from the store
while the real holding stayed open at the broker: a phantom fill.

The stakes are concrete. Of 220 settled multi-day positions, the 30 that exited
via `target_touch` returned +Rs364,583 at a 100% win rate (100% by construction
— the target sits above entry), while the 190 `kday_close_moc` exits returned
-Rs507,535. Every rupee of the book's paper profit comes from the mechanism
whose live path does not exist.

Faithful degradation is to do nothing: with no order resting, the exchange sold
nothing, so the position continues to its scheduled exit. `target_touched` is
only ever set while `today < exit_on`, so skipping cannot strand it.
"""
import ast
import inspect
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[3] / "services" / "execution" / "mtf_capitulation_handlers.py"
TEXT = SRC.read_text(encoding="utf-8")


def test_live_guard_precedes_the_touch_fill():
    """Ordering is the whole defect: the paper fill must not be reachable live."""
    guard = TEXT.index("if target_touched and not paper_mode:")
    fill = TEXT.index("sell_price = max(day_open, tpx)")
    assert guard < fill, "the live guard must come BEFORE the touch-fill branch"


def test_live_guard_skips_rather_than_booking_a_price():
    """Between the guard and the fill there must be a `continue` and no
    sell_price assignment — inventing a price is the failure mode."""
    guard = TEXT.index("if target_touched and not paper_mode:")
    fill = TEXT.index("sell_price = max(day_open, tpx)")
    body = TEXT[guard:fill]
    assert "continue" in body, "live guard must skip the position, not fall through"
    assert "sell_price" not in body, "live guard must not assign a sell price"


def test_live_guard_is_loud():
    guard = TEXT.index("if target_touched and not paper_mode:")
    fill = TEXT.index("sell_price = max(day_open, tpx)")
    body = TEXT[guard:fill]
    assert "logger.critical" in body, (
        "a missing live exit path must log CRITICAL — it silently costs the "
        "book's entire paper profit")
    assert "target_touch_live_unsupported" in body, "must surface in the run summary"


def test_touch_only_fires_before_the_scheduled_exit():
    """Guarantees skipping is safe: a touched position is never already due,
    so `continue` leaves it to exit normally rather than stranding it."""
    tree = ast.parse(TEXT)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_run_exits")
    src = ast.get_source_segment(TEXT, fn)
    # target_touched is set True only inside the `today < exit_on` branch
    i = src.index("target_touched = True")
    guard_block = src[:i]
    assert "today < exit_on" in guard_block, (
        "target_touched must only be settable while the position is NOT yet due")


def test_paper_behaviour_is_unchanged():
    """The guard must be live-only — paper is the validated simulation."""
    guard_line = "if target_touched and not paper_mode:"
    assert guard_line in TEXT
    # the paper path still reaches the touch fill
    assert "sell_price = max(day_open, tpx) if day_open is not None else tpx" in TEXT


def test_module_imports_and_run_exits_is_callable():
    from services.execution import mtf_capitulation_handlers as H
    assert callable(H._run_exits)
    assert "paper_mode" in inspect.signature(H._run_exits).parameters
