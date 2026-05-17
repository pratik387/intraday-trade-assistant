import pandas as pd
import pytest
from unittest.mock import MagicMock
from services.dispatch.planner import Batch
from services.dispatch.worker import dispatch_worker_batch, _detector_cache, init_worker


@pytest.fixture(autouse=True)
def clear_caches():
    _detector_cache.clear()
    yield
    _detector_cache.clear()


def test_worker_returns_empty_for_empty_batch():
    init_worker(registry=MagicMock())
    out = dispatch_worker_batch(Batch(items=[]))
    assert out == []


def test_worker_returns_sym_decision_tuples(monkeypatch):
    """dispatch_worker_batch returns list[(sym, GateDecision)] — not raw events."""
    from structures.data_models import StructureAnalysis, StructureEvent
    import pandas as _pd

    fake_event = MagicMock(spec=StructureEvent)
    fake_event.structure_type = "gap_fade_short"
    fake_event.side = "short"
    fake_event.confidence = 0.8
    fake_event.levels = {}
    fake_event.context = {"detector_name": "gap_fade_short"}

    fake_analysis = MagicMock(spec=StructureAnalysis)
    fake_analysis.structure_detected = True
    fake_analysis.events = [fake_event]
    fake_analysis.quality_score = 75.0

    fake_spec = MagicMock()
    fake_detector_instance = MagicMock()
    fake_detector_instance.detect.return_value = fake_analysis
    fake_spec.detector_class_path = "fake.path"
    fake_spec.raw_config = {}
    fake_registry = MagicMock()
    fake_registry.get.return_value = fake_spec

    def fake_import_path(path):
        return MagicMock(return_value=fake_detector_instance)

    monkeypatch.setattr("services.dispatch.worker._import_path", fake_import_path)
    init_worker(registry=fake_registry)

    df = pd.DataFrame({
        "open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0], "volume": [10000.0]
    }, index=[_pd.Timestamp("2024-05-03 09:15:00")])
    levels = {"PDC": 100.0, "ORH": 102.0, "ORL": 98.0}
    batch = Batch(
        items=[("NSE:A", df, levels, {"gap_fade_short"})],
        regime="chop",
        session_date=None,
    )
    results = dispatch_worker_batch(batch)

    # Should return 1 tuple (sym, decision)
    assert len(results) == 1
    sym, decision = results[0]
    assert sym == "NSE:A"
    # GateDecision should have accept=True since we returned a real event
    assert decision.accept is True
    assert decision.setup_candidates is not None
    assert len(decision.setup_candidates) >= 1


def test_worker_reject_when_no_events(monkeypatch):
    """If all detectors produce no events, decision is accept=False."""
    from structures.data_models import StructureAnalysis
    import pandas as _pd

    fake_analysis = MagicMock(spec=StructureAnalysis)
    fake_analysis.structure_detected = False
    fake_analysis.events = []
    fake_analysis.quality_score = 0.0

    fake_spec = MagicMock()
    fake_detector_instance = MagicMock()
    fake_detector_instance.detect.return_value = fake_analysis
    fake_spec.detector_class_path = "fake.path"
    fake_spec.raw_config = {}
    fake_registry = MagicMock()
    fake_registry.get.return_value = fake_spec

    def fake_import_path(path):
        return MagicMock(return_value=fake_detector_instance)

    monkeypatch.setattr("services.dispatch.worker._import_path", fake_import_path)
    init_worker(registry=fake_registry)

    df = pd.DataFrame({
        "open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0], "volume": [10000.0]
    }, index=[_pd.Timestamp("2024-05-03 09:15:00")])
    levels = {"PDC": 100.0}
    batch = Batch(items=[("NSE:A", df, levels, {"gap_fade_short"})])
    results = dispatch_worker_batch(batch)

    assert len(results) == 1
    sym, decision = results[0]
    assert sym == "NSE:A"
    assert decision.accept is False


def test_worker_detector_cached_after_first_use(monkeypatch):
    """Detector instance is cached per worker process."""
    from structures.data_models import StructureAnalysis
    import pandas as _pd

    fake_analysis = MagicMock(spec=StructureAnalysis)
    fake_analysis.structure_detected = False
    fake_analysis.events = []
    fake_analysis.quality_score = 0.0

    fake_spec = MagicMock()
    fake_detector_instance = MagicMock()
    fake_detector_instance.detect.return_value = fake_analysis
    fake_spec.detector_class_path = "fake.path"
    fake_spec.raw_config = {}
    fake_registry = MagicMock()
    fake_registry.get.return_value = fake_spec

    def fake_import_path(path):
        return MagicMock(return_value=fake_detector_instance)

    monkeypatch.setattr("services.dispatch.worker._import_path", fake_import_path)
    init_worker(registry=fake_registry)

    df = pd.DataFrame({
        "open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0], "volume": [10000.0]
    }, index=[_pd.Timestamp("2024-05-03 09:15:00")])
    levels = {"PDC": 100.0}
    batch = Batch(items=[("NSE:A", df, levels, {"gap_fade_short", "mis_unwind"})])
    dispatch_worker_batch(batch)
    # Both detectors should now be cached
    assert "gap_fade_short" in _detector_cache
    assert "mis_unwind" in _detector_cache
