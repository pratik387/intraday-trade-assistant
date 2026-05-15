import pandas as pd
import numpy as np

from tools.edge_discovery.types import Event, ConditionalOutcomeTable
from tools.edge_discovery.explorer import Explorer
from tools.edge_discovery.outcomes.returns import ForwardReturns
from tools.edge_discovery.features.symbol_features import SymbolFeaturesTierA


def _bars_for_symbol(symbol: str) -> pd.DataFrame:
    idx = pd.date_range("2024-06-15 09:15:00", periods=75, freq="5min")
    return pd.DataFrame({
        "symbol": symbol,
        "date": idx,
        "open": np.linspace(100.0, 110.0, 75),
        "high": np.linspace(100.5, 110.5, 75),
        "low": np.linspace(99.5, 109.5, 75),
        "close": np.linspace(100.2, 110.2, 75),
        "volume": np.full(75, 5000),
    })


def test_explorer_returns_conditional_outcome_table():
    bars_x = _bars_for_symbol("X")
    bars_y = _bars_for_symbol("Y")
    events = [
        Event(symbol="X", event_time=bars_x["date"].iloc[10], metadata={"direction": "long"}),
        Event(symbol="Y", event_time=bars_y["date"].iloc[10], metadata={"direction": "long"}),
    ]
    bar_data = {"X": bars_x, "Y": bars_y}
    sym_meta = {
        "X": {"cap_segment": "small_cap", "mis_leverage": 5.0},
        "Y": {"cap_segment": "mid_cap", "mis_leverage": 5.0},
    }
    explorer = Explorer(
        features=[SymbolFeaturesTierA()],
        outcomes=[ForwardReturns(horizons_minutes=[5, 30], eod=False)],
    )
    table = explorer.run(events, bar_data=bar_data, symbol_meta=sym_meta,
                          pdh_pdl_close_by_event=None, adv_by_symbol={"X": 200_000, "Y": 1_500_000})
    assert isinstance(table, ConditionalOutcomeTable)
    assert len(table.rows) == 2
    assert "cap_segment" in table.rows.columns
    assert "ret_5m" in table.rows.columns
    assert "ret_30m" in table.rows.columns
    assert set(table.rows["cap_segment"]) == {"small_cap", "mid_cap"}
