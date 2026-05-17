"""Worker-side entry point. Runs in ProcessPoolExecutor workers.

Module-level detector instance cache keyed by setup name. Cache populated
lazily on first use within the worker process; survives across batches.
"""
from __future__ import annotations

from services.dispatch.planner import Batch
from services.dispatch.setup_registry import SetupRegistry, _import_path


_detector_cache: dict = {}
_registry_cache = None


def init_worker(registry) -> None:
    """Initialize worker-process state. Called on worker spawn or in tests."""
    global _registry_cache
    _registry_cache = registry
    _detector_cache.clear()


def _get_detector(name: str):
    if name in _detector_cache:
        return _detector_cache[name]
    if _registry_cache is None:
        raise RuntimeError("worker not initialized — call init_worker first")
    spec = _registry_cache.get(name)
    cls = _import_path(spec.detector_class_path)
    instance = cls(spec.raw_config)
    _detector_cache[name] = instance
    return instance


def _build_market_context(sym: str, df5, levels: dict):
    """Build the MarketContext object detectors expect.

    Minimal stub for now — Task 10 will plumb full context fields from caller.
    Returns a simple namespace so the worker compiles and tests that mock
    detect() can run without hitting MarketContext's required-field validation.
    """
    import types
    ctx = types.SimpleNamespace(
        symbol=sym,
        df_5m=df5,
        levels=levels,
    )
    return ctx


def dispatch_worker_batch(batch: Batch) -> list:
    """Process one batch: for each (sym, df5, levels, tags), run each tagged detector."""
    out: list = []
    for sym, df5, levels, tags in batch.items:
        ctx = _build_market_context(sym, df5, levels)
        for det_name in tags:
            det = _get_detector(det_name)
            analysis = det.detect(ctx)
            out.extend(analysis.events)
    return out
