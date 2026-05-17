import pandas as pd
from datetime import datetime, date
from services.dispatch.tag_map import TagMap
from services.dispatch.planner import Batch, DispatchPlanner


def _df(): return pd.DataFrame({"close": [100.0]})
def _lvl(): return {"PDC": 100.0, "ORH": 102.0, "ORL": 98.0}


def test_empty_plan_when_no_active_tags():
    tm = TagMap()
    planner = DispatchPlanner(batch_size=50)
    plan = planner.plan(datetime(2024, 5, 3, 10, 30), tm, df5_by_symbol={}, levels_by_symbol={})
    assert plan == []


def test_plan_chunks_at_batch_size():
    tm = TagMap()
    syms = {f"NSE:S{i}" for i in range(120)}
    tm.add_universe("gap_fade_short", syms)
    tm.open_window("gap_fade_short")
    df5 = {s: _df() for s in syms}
    levels = {s: _lvl() for s in syms}
    planner = DispatchPlanner(batch_size=50)
    plan = planner.plan(datetime(2024, 5, 3, 9, 20), tm, df5_by_symbol=df5, levels_by_symbol=levels)
    # 120 syms / 50 = 3 batches (50 + 50 + 20)
    assert len(plan) == 3
    assert sum(len(b.items) for b in plan) == 120


def test_plan_items_carry_tag_set():
    tm = TagMap()
    tm.add_universe("gap_fade_short", {"NSE:A"})
    tm.add_universe("mis_unwind_vwap_revert_short", {"NSE:A"})
    tm.open_window("gap_fade_short")
    tm.open_window("mis_unwind_vwap_revert_short")
    planner = DispatchPlanner(batch_size=50)
    plan = planner.plan(datetime(2024, 5, 3, 14, 30), tm,
                        df5_by_symbol={"NSE:A": _df()},
                        levels_by_symbol={"NSE:A": _lvl()})
    assert len(plan) == 1
    sym, df5, levels, tags, cap_seg = plan[0].items[0]
    assert sym == "NSE:A"
    assert tags == {"gap_fade_short", "mis_unwind_vwap_revert_short"}


def test_plan_items_carry_cap_segment():
    tm = TagMap()
    tm.add_universe("gap_fade_short", {"NSE:A"})
    tm.open_window("gap_fade_short")
    planner = DispatchPlanner(batch_size=50)
    cap_map = {"NSE:A": "small_cap"}
    plan = planner.plan(datetime(2024, 5, 3, 9, 20), tm,
                        df5_by_symbol={"NSE:A": _df()},
                        levels_by_symbol={"NSE:A": _lvl()},
                        cap_segment_map=cap_map)
    assert len(plan) == 1
    sym, df5, levels, tags, cap_seg = plan[0].items[0]
    assert cap_seg == "small_cap"


def test_plan_carries_bar_metadata():
    tm = TagMap()
    tm.add_universe("gap_fade_short", {"NSE:A"})
    tm.open_window("gap_fade_short")
    bar_ts = datetime(2024, 5, 3, 9, 20)
    sess = date(2024, 5, 3)
    planner = DispatchPlanner(batch_size=50)
    plan = planner.plan(bar_ts, tm,
                        df5_by_symbol={"NSE:A": _df()},
                        levels_by_symbol={"NSE:A": _lvl()},
                        session_date=sess,
                        regime="trend_up",
                        regime_diagnostics={"detail": "x"})
    assert len(plan) == 1
    b = plan[0]
    assert b.bar_ts == bar_ts
    assert b.session_date == sess
    assert b.regime == "trend_up"
    assert b.regime_diagnostics == {"detail": "x"}


def test_plan_skips_symbols_without_df5():
    tm = TagMap()
    tm.add_universe("gap_fade_short", {"NSE:A", "NSE:B"})
    tm.open_window("gap_fade_short")
    planner = DispatchPlanner(batch_size=50)
    plan = planner.plan(datetime(2024, 5, 3, 9, 20), tm,
                        df5_by_symbol={"NSE:A": _df()},  # B missing
                        levels_by_symbol={"NSE:A": _lvl()})
    syms_in_plan = [item[0] for b in plan for item in b.items]
    assert syms_in_plan == ["NSE:A"]
