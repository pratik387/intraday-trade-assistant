"""Forward returns + MFE/MAE outcome module."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from tools.edge_discovery.types import Event


class ForwardReturns:
    """Computes forward returns at fixed horizons + MFE/MAE per horizon.

    Direction semantics:
      direction == "long":  ret = (future_close - entry_close) / entry_close
      direction == "short": ret = (entry_close - future_close) / entry_close
      mfe (max favorable) and mae (max adverse) reported in direction's frame.
    """

    name = "forward_returns"

    def __init__(self, horizons_minutes: List[int], eod: bool = True) -> None:
        self.horizons_minutes = list(horizons_minutes)
        self.eod = eod

    def compute(self, event: Event, bars: pd.DataFrame) -> Dict[str, float]:
        direction = event.metadata.get("direction", "long")
        sign = 1.0 if direction == "long" else -1.0

        # Locate entry bar: the bar whose `date` equals event_time
        entry_idx = bars.index[bars["date"] == event.event_time]
        if len(entry_idx) == 0:
            # event_time isn't on a bar boundary; pick first bar at-or-after
            entry_idx = bars.index[bars["date"] >= event.event_time]
            if len(entry_idx) == 0:
                return self._empty()
        entry_i = int(entry_idx[0])
        entry_close = float(bars.at[entry_i, "close"])

        out: Dict[str, float] = {}
        for h_min in self.horizons_minutes:
            n_bars = h_min // 5
            future_i = entry_i + n_bars
            if future_i >= len(bars):
                out[f"ret_{h_min}m"] = np.nan
                out[f"mfe_{h_min}m"] = np.nan
                out[f"mae_{h_min}m"] = np.nan
                continue
            future_close = float(bars.at[future_i, "close"])
            out[f"ret_{h_min}m"] = sign * (future_close - entry_close) / entry_close

            window = bars.iloc[entry_i + 1 : future_i + 1]
            if direction == "long":
                mfe_price = window["high"].max()
                mae_price = window["low"].min()
                out[f"mfe_{h_min}m"] = (mfe_price - entry_close) / entry_close
                out[f"mae_{h_min}m"] = (mae_price - entry_close) / entry_close
            else:
                mfe_price = window["low"].min()
                mae_price = window["high"].max()
                out[f"mfe_{h_min}m"] = (entry_close - mfe_price) / entry_close
                out[f"mae_{h_min}m"] = (entry_close - mae_price) / entry_close

        if self.eod:
            day_floor = event.event_time.floor("D")
            same_day = bars[bars["date"].dt.floor("D") == day_floor]
            eod_idx = same_day.index.max()
            if eod_idx > entry_i:
                eod_close = float(bars.at[eod_idx, "close"])
                out["ret_eod"] = sign * (eod_close - entry_close) / entry_close
            else:
                out["ret_eod"] = np.nan

        return out

    def _empty(self) -> Dict[str, float]:
        out = {}
        for h in self.horizons_minutes:
            out[f"ret_{h}m"] = np.nan
            out[f"mfe_{h}m"] = np.nan
            out[f"mae_{h}m"] = np.nan
        if self.eod:
            out["ret_eod"] = np.nan
        return out
