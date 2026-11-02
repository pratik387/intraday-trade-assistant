"""Tick recorder part files across restarts (2026-11-02).

Parts were numbered from 0 by every process and merged in lexicographic order, so
a same-day restart overwrote the earlier parts and part10 merged before part2.
"""
from datetime import datetime, timedelta

import pandas as pd

from market_data.tick_recorder import TickRecorder

T0 = datetime(2026, 11, 2, 9, 15)


def _feed(rec, start, n):
    for i in range(start, start + n):
        rec.on_tick("NSE:X", 100.0 + i, 1, T0 + timedelta(seconds=i), i)


def test_a_restart_continues_the_part_numbers_and_keeps_every_tick(tmp_path):
    a = TickRecorder(buffer_size=2, output_dir=tmp_path)
    _feed(a, 0, 4)                                   # parts 0, 1
    b = TickRecorder(buffer_size=2, output_dir=tmp_path)
    assert b._part_num == 2
    _feed(b, 4, 4)                                   # parts 2, 3
    out = pd.read_parquet(b.finalize())
    assert list(out["volume"]) == list(range(8))


def test_parts_merge_in_numeric_order(tmp_path):
    rec = TickRecorder(buffer_size=1, output_dir=tmp_path)
    _feed(rec, 0, 12)                                # part0 .. part11
    out = pd.read_parquet(rec.finalize())
    assert list(out["volume"]) == list(range(12))


def test_a_second_finalize_the_same_day_keeps_the_first_sessions_ticks(tmp_path):
    a = TickRecorder(buffer_size=10, output_dir=tmp_path)
    _feed(a, 0, 3)
    a.finalize()
    b = TickRecorder(buffer_size=10, output_dir=tmp_path)
    _feed(b, 3, 2)
    out = pd.read_parquet(b.finalize())
    assert list(out["volume"]) == list(range(5))
    assert not list(tmp_path.glob("*.part*.parquet"))
