"""Target 2: LONG-side panic-gap-down catch in small/mid-cap.

Event population: all small/mid-cap MIS-eligible names with gap-down >=
`gap_down_pct_min` on the first 5m bar (broader than the candidate trigger;
explorer will narrow).

All thresholds / scan parameters / cost horizon / cap-segment filter / adv
midpoints are sourced from
config/pipelines/base_config.json -> edge_discovery.target_long_panic_gap_down.
No hardcoded defaults (CLAUDE.md Rule 1).
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from services.config_loader import load_base_config
from tools.edge_discovery.data_loader import load_5m_period
from tools.edge_discovery.explorer import Explorer
from tools.edge_discovery.features.symbol_features import SymbolFeaturesTierA
from tools.edge_discovery.features.event_features import EventCalendarFeatures
from tools.edge_discovery.outcomes.returns import ForwardReturns
from tools.edge_discovery.outcomes.costs import ExecutionCosts
from tools.edge_discovery.types import Event
from tools.edge_discovery.universe import load_nse_all, mis_eligible_universe


_REPO = Path(__file__).resolve().parents[3]
_REPORT_DIR = _REPO / "reports" / "edge_discovery"


def _build_events_for_window(
    start: date,
    end: date,
    universe: set,
    meta: dict,
    gap_down_pct_min: float,
    cap_segments: set,
) -> tuple:
    """Build events for the window. Returns (events, bar_data, symbol_meta, pdh_pdl_close, adv_by_sym)."""
    bars = load_5m_period(start, end, symbols=universe)
    if bars.empty:
        return [], {}, {}, {}, {}
    bars["_day"] = bars["date"].dt.floor("D")
    bars = bars.sort_values(["symbol", "_day", "date"], kind="mergesort").reset_index(drop=True)

    # First bar per (symbol, day)
    first_mask = bars.groupby(["symbol", "_day"]).cumcount() == 0
    firsts = bars[first_mask][["symbol", "_day", "date", "open", "high", "low", "close"]]

    # Prior session close per (symbol, day)
    daily = bars.groupby(["symbol", "_day"]).agg(
        day_high=("high", "max"), day_low=("low", "min"), day_close=("close", "last"),
    ).reset_index()
    daily = daily.sort_values(["symbol", "_day"])
    daily["prior_high"] = daily.groupby("symbol")["day_high"].shift(1)
    daily["prior_low"] = daily.groupby("symbol")["day_low"].shift(1)
    daily["prior_close"] = daily.groupby("symbol")["day_close"].shift(1)

    firsts = firsts.merge(
        daily[["symbol", "_day", "prior_high", "prior_low", "prior_close"]],
        on=["symbol", "_day"], how="left",
    )
    firsts["gap_pct"] = (firsts["open"] - firsts["prior_close"]) / firsts["prior_close"]
    triggers = firsts[firsts["gap_pct"] <= -gap_down_pct_min]
    # Cap-segment filter (small/mid by default — driven by config)
    triggers = triggers[triggers["symbol"].apply(
        lambda s: meta.get(s, {}).get("cap_segment") in cap_segments
    )]

    events: List[Event] = []
    pdh_pdl_close: Dict[int, Dict[str, float]] = {}
    bar_data: Dict[str, pd.DataFrame] = {}
    symbol_meta_used: Dict[str, dict] = {}
    adv_by_sym: Dict[str, float] = {}

    # Compute 20d ADV per symbol from daily volume
    daily_vol = bars.groupby(["symbol", "_day"])["volume"].sum().reset_index()
    daily_vol = daily_vol.sort_values(["symbol", "_day"])
    daily_vol["adv20"] = daily_vol.groupby("symbol")["volume"].transform(
        lambda s: s.rolling(20, min_periods=10).mean().shift(1)
    )
    adv_lookup = daily_vol.set_index(["symbol", "_day"])["adv20"].to_dict()

    # Pre-group bars by symbol once, avoiding O(N_triggers × N_bars) re-scans.
    # reset_index per symbol because downstream features use .iloc[i] with i sourced
    # from bars.index[mask] — that pattern requires a 0..N RangeIndex.
    bars_by_sym = {
        sym: g.reset_index(drop=True)
        for sym, g in bars.groupby("symbol", sort=False)
    }

    # NOTE: pandas itertuples renames leading-underscore columns positionally
    # (e.g. `_day` -> `_2`), so iterate the dict directly to avoid attribute access.
    triggers_records = triggers.to_dict(orient="records")
    for i, t in enumerate(triggers_records):
        sym, day, ts = t["symbol"], t["_day"], t["date"]
        if sym not in bar_data:
            bar_data[sym] = bars_by_sym.get(sym, bars.iloc[0:0])
            symbol_meta_used[sym] = meta.get(sym, {})
        events.append(Event(symbol=sym, event_time=ts,
                            metadata={"direction": "long", "gap_pct": float(t["gap_pct"])}))
        pdh_pdl_close[i] = {
            "pdh": float(t["prior_high"]) if t["prior_high"] is not None and not pd.isna(t["prior_high"]) else None,
            "pdl": float(t["prior_low"]) if t["prior_low"] is not None and not pd.isna(t["prior_low"]) else None,
            "prior_close": float(t["prior_close"]) if t["prior_close"] is not None and not pd.isna(t["prior_close"]) else None,
        }
        adv_val = adv_lookup.get((sym, day))
        adv_by_sym[sym] = float(adv_val) if adv_val is not None and not pd.isna(adv_val) else 0.0

    return events, bar_data, symbol_meta_used, pdh_pdl_close, adv_by_sym


def _apply_costs_to_outcomes(
    rows: pd.DataFrame,
    cost_block: dict,
    horizon: int,
    adv_bucket_midpoints: dict,
    order_size_pct_of_adv: float,
) -> pd.DataFrame:
    """Add post-cost return columns for given horizon."""
    costs = ExecutionCosts(cost_block)
    pre_col = f"ret_{horizon}m"
    post_col = f"ret_{horizon}m_post_cost"

    fallback_adv = float(adv_bucket_midpoints["_fallback"])

    def _row_cost(r) -> float:
        cap = r["cap_segment"]
        bucket = r["adv_bucket"]
        # Map adv_bucket back to a representative ADV midpoint (configured)
        adv = float(adv_bucket_midpoints.get(bucket, fallback_adv))
        gross = float(r.get(pre_col, np.nan))
        if pd.isna(gross):
            return np.nan
        return costs.apply_round_trip(
            gross_return_pct=gross, cap_segment=cap,
            adv_shares=adv, order_shares=adv * order_size_pct_of_adv,
            sl_hit=False, sl_bar_range_pct=None,
        )

    rows[post_col] = rows.apply(_row_cost, axis=1)
    return rows


def run_target_long_panic_gap_down() -> dict:
    cfg = load_base_config()
    edisc = cfg["edge_discovery"]
    periods = edisc["periods"]
    discovery_start = date.fromisoformat(periods["discovery_start"])
    discovery_end = date.fromisoformat(periods["discovery_end"])
    cost_block = edisc["cost_model"]
    tcfg = edisc["target_long_panic_gap_down"]

    gap_down_pct_min = float(tcfg["gap_down_pct_min"])
    order_size_pct_of_adv = float(tcfg["order_size_pct_of_adv"])
    cost_horizon_minutes = int(tcfg["cost_horizon_minutes"])
    scan_min_n = int(tcfg["scan_min_n"])
    scan_top_n = int(tcfg["scan_top_n"])
    scan_max_dims = int(tcfg["scan_max_dims"])
    scan_outcome = str(tcfg["scan_outcome"])
    scan_feature_names = list(tcfg["scan_feature_names"])
    forward_horizons_minutes = list(tcfg["forward_horizons_minutes"])
    forward_eod = bool(tcfg["forward_eod"])
    cap_segments = set(tcfg["cap_segments"])
    adv_bucket_midpoints = dict(tcfg["adv_bucket_midpoints"])

    meta = load_nse_all()
    universe = mis_eligible_universe(meta) & {
        s for s, m in meta.items() if m.get("cap_segment") in cap_segments
    }
    print(f"[target2] universe size: {len(universe):,}")

    events, bar_data, sym_meta, pdh_pdl, adv_by_sym = _build_events_for_window(
        discovery_start, discovery_end, universe, meta,
        gap_down_pct_min=gap_down_pct_min,
        cap_segments=cap_segments,
    )
    print(f"[target2] events: {len(events):,}")

    explorer = Explorer(
        features=[SymbolFeaturesTierA(), EventCalendarFeatures()],
        outcomes=[ForwardReturns(horizons_minutes=forward_horizons_minutes, eod=forward_eod)],
    )
    table = explorer.run(
        events, bar_data=bar_data, symbol_meta=sym_meta,
        pdh_pdl_close_by_event=pdh_pdl, adv_by_symbol=adv_by_sym,
    )
    table.rows = _apply_costs_to_outcomes(
        table.rows, cost_block,
        horizon=cost_horizon_minutes,
        adv_bucket_midpoints=adv_bucket_midpoints,
        order_size_pct_of_adv=order_size_pct_of_adv,
    )

    regions = table.top_edge_regions(
        outcome=scan_outcome,
        feature_names=scan_feature_names,
        min_n=scan_min_n,
        top_n=scan_top_n,
        max_dims=scan_max_dims,
    )
    out_path = _REPORT_DIR / "target_long_panic_gap_down.csv"
    table.rows.to_csv(out_path, index=False)
    regions_path = _REPORT_DIR / "target_long_panic_gap_down_regions.json"
    with open(regions_path, "w", encoding="utf-8") as f:
        json.dump(regions, f, indent=2, default=str)
    print(f"[target2] wrote {out_path} ({len(table.rows):,} rows)")
    print(f"[target2] top edge regions written to {regions_path}")
    return {"n_events": len(events), "n_rows": len(table.rows), "regions": regions[:5]}


if __name__ == "__main__":
    run_target_long_panic_gap_down()
