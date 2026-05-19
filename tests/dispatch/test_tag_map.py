import pytest
from services.dispatch.tag_map import TagMap


def test_empty_tag_map():
    tm = TagMap()
    assert tm.active_symbols() == set()
    assert tm.active_tags("NSE:RML") == set()


def test_add_universe_then_open_window():
    tm = TagMap()
    tm.add_universe("gap_fade_short", {"NSE:A", "NSE:B", "NSE:C"})
    assert tm.active_symbols() == set()  # tagged but not active yet
    tm.open_window("gap_fade_short")
    assert tm.active_symbols() == {"NSE:A", "NSE:B", "NSE:C"}
    assert tm.active_tags("NSE:A") == {"gap_fade_short"}


def test_close_window_drops_from_active():
    tm = TagMap()
    tm.add_universe("gap_fade_short", {"NSE:A"})
    tm.open_window("gap_fade_short")
    tm.close_window("gap_fade_short")
    assert tm.active_symbols() == set()
    assert tm.active_tags("NSE:A") == set()


def test_multiple_setups_same_symbol():
    tm = TagMap()
    tm.add_universe("gap_fade_short", {"NSE:RML"})
    tm.add_universe("mis_unwind_vwap_revert_short", {"NSE:RML", "NSE:OTHER"})
    tm.open_window("gap_fade_short")
    tm.open_window("mis_unwind_vwap_revert_short")
    assert tm.active_tags("NSE:RML") == {"gap_fade_short", "mis_unwind_vwap_revert_short"}
    tm.close_window("gap_fade_short")
    assert tm.active_tags("NSE:RML") == {"mis_unwind_vwap_revert_short"}
    assert "NSE:RML" in tm.active_symbols()


def test_reopen_window_after_close():
    tm = TagMap()
    tm.add_universe("gap_fade_short", {"NSE:A"})
    tm.open_window("gap_fade_short")
    tm.close_window("gap_fade_short")
    tm.open_window("gap_fade_short")
    assert tm.active_symbols() == {"NSE:A"}


def test_close_unopened_window_is_noop():
    tm = TagMap()
    tm.add_universe("gap_fade_short", {"NSE:A"})
    tm.close_window("gap_fade_short")
    assert tm.active_symbols() == set()
