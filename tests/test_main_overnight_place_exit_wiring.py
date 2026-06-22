import inspect
import main


def test_place_exit_is_a_valid_overnight_action():
    src = inspect.getsource(main)
    assert '"place-exit"' in src  # added to --action choices / routing
    assert "run_place_exit" in src


def test_live_overnight_uses_hybrid_broker():
    src = inspect.getsource(main)
    assert "LiveOvernightBroker" in src
