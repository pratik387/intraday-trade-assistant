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


def test_worker_calls_detector_for_each_tag(monkeypatch):
    fake_spec = MagicMock()
    fake_detector_instance = MagicMock()
    fake_detector_instance.detect.return_value = MagicMock(events=["evt1"])
    fake_spec.detector_class_path = "fake.path"
    fake_spec.raw_config = {}
    fake_registry = MagicMock()
    fake_registry.get.return_value = fake_spec

    def fake_import_path(path):
        return MagicMock(return_value=fake_detector_instance)

    monkeypatch.setattr("services.dispatch.worker._import_path", fake_import_path)
    init_worker(registry=fake_registry)

    df = pd.DataFrame({"close": [100.0]})
    levels = {"PDC": 100.0}
    batch = Batch(items=[("NSE:A", df, levels, {"gap_fade_short", "mis_unwind"})])
    events = dispatch_worker_batch(batch)
    assert len(events) == 2
    assert "gap_fade_short" in _detector_cache
    assert "mis_unwind" in _detector_cache
