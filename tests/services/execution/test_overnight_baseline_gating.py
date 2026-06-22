import inspect
import services.execution.overnight_handlers as oh

def test_baseline_build_gated_on_data_sdk_not_paper_mode():
    src = inspect.getsource(oh.run_verify_exit)
    # The baseline build must trigger on data_sdk presence, not paper_mode.
    assert 'getattr(broker, "_data_sdk", None) is not None' in src
    # Guard against regression to the paper-only gate wrapping the baseline build.
    assert "if paper_mode:\n        try:\n            data_sdk = getattr(broker" not in src
