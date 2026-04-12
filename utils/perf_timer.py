"""Structured performance timer — writes to timing.jsonl via buffered JSONL logger.

Usage:
    from utils.perf_timer import perf, mark

    with perf("scan", "stage0_execute", shortlist_count=len(shortlist)):
        stage0_future.result()

    mark("scan", "bar_start", bar_ts=now.isoformat())

Design:
- **Disabled by default.** Enable via env var TRADING_PERF_TIMER=1.
  OCI 3-year production runs: leave unset -> zero cost, no timing.jsonl produced.
  Dev measurement runs: set to 1 -> full instrumentation.
- Zero-overhead disabled path: shared _NoOpTimer singleton, no allocations per call.
- Enabled path uses slotted _RealTimer class (no dict overhead).
- Writes through JSONLLogger.log_event() -> buffered file (~6us per write).
- Never raises in the hot path — exceptions in timing swallowed silently.
- The _ENABLED check happens ONCE at module import time; the exported perf/mark
  function references point to the right implementation from the start.
"""
from __future__ import annotations
import os
import time

_ENABLED = os.environ.get("TRADING_PERF_TIMER", "0") == "1"


class _NoOpTimer:
    """Shared singleton context manager used when instrumentation is disabled.

    __slots__ = () eliminates per-instance dict. Using one shared instance
    means perf() returns the same object every call — no allocation.
    """
    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


_NOOP = _NoOpTimer()


class _RealTimer:
    """Real timing context manager — records duration_ms to timing.jsonl on exit."""
    __slots__ = ("_stage", "_substage", "_meta", "_t0")

    def __init__(self, stage: str, substage: str, meta: dict):
        self._stage = stage
        self._substage = substage
        self._meta = meta
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        duration_ms = (time.perf_counter() - self._t0) * 1000.0
        try:
            from config.logging_config import get_timing_logger
            lg = get_timing_logger()
            if lg is not None:
                lg.log_event(
                    ts=time.time(),
                    pid=os.getpid(),
                    stage=self._stage,
                    substage=self._substage,
                    duration_ms=round(duration_ms, 3),
                    **self._meta,
                )
        except Exception:
            pass  # never let timing break the caller
        return False


if _ENABLED:
    def perf(stage: str, substage: str = "", **meta):
        """Context manager that records the duration of the block to timing.jsonl."""
        return _RealTimer(stage, substage, meta)

    def mark(stage: str, substage: str = "", **meta):
        """Emit a point-in-time marker (no duration) — useful for phase boundaries."""
        try:
            from config.logging_config import get_timing_logger
            lg = get_timing_logger()
            if lg is not None:
                lg.log_event(
                    ts=time.time(),
                    pid=os.getpid(),
                    stage=stage,
                    substage=substage,
                    event="mark",
                    **meta,
                )
        except Exception:
            pass
else:
    def perf(stage: str = "", substage: str = "", **meta):
        return _NOOP

    def mark(stage: str = "", substage: str = "", **meta):
        return None


def is_enabled() -> bool:
    """Return True if TRADING_PERF_TIMER=1 (checked once at import time)."""
    return _ENABLED
