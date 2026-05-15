"""Target 3: Ensemble feature mining on the 3 live setups.

Approach: load each live setup's existing trade parquet (Discovery window),
attach Tier-A + event-calendar features per trade, and run edge-region
detection. The goal is to find context-conditional sub-regions where the
post-cost PF lifts above the setup's baseline.

All parameters are sourced from
config/pipelines/base_config.json -> edge_discovery.target_ensemble_live_setups.
No hardcoded defaults (CLAUDE.md Rule 1).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from services.config_loader import load_base_config
from tools.edge_discovery.types import ConditionalOutcomeTable, Event
from tools.edge_discovery.features.event_features import EventCalendarFeatures


_REPO = Path(__file__).resolve().parents[3]
_REPORT_DIR = _REPO / "reports" / "edge_discovery"


def _load_setup_trades(setup_name: str, parquet_dir: str) -> pd.DataFrame:
    """Load Discovery trades for one live setup from the configured parquet dir."""
    pq = _REPO / parquet_dir / f"{setup_name}.parquet"
    if not pq.exists():
        raise FileNotFoundError(f"missing: {pq}")
    return pd.read_parquet(pq)


def _resolve_entry_time_column(df: pd.DataFrame, aliases: list) -> pd.DataFrame:
    """Rename whichever alias is present to 'entry_time'. Raise if none found."""
    for col in aliases:
        if col in df.columns:
            if col != "entry_time":
                df = df.rename(columns={col: "entry_time"})
            return df
    raise KeyError(
        f"No entry-time column found. Searched aliases: {aliases}. "
        f"Columns available: {list(df.columns)}"
    )


def _attach_event_features(df: pd.DataFrame, aliases: list) -> pd.DataFrame:
    """Attach calendar features per trade.

    Timestamps stay IST-naive (no tzinfo). pd.to_datetime on a string column
    of the form '2023-01-02 09:30:00' produces tz-naive output — verified.
    """
    df = _resolve_entry_time_column(df, aliases)
    df["entry_time"] = pd.to_datetime(df["entry_time"])

    # Guard: reject tz-aware timestamps (CLAUDE.md Rule 2)
    sample_ts = df["entry_time"].iloc[0]
    if sample_ts.tzinfo is not None:
        raise ValueError(
            f"entry_time column has tzinfo={sample_ts.tzinfo}. "
            "Use utils.time_util.to_naive_ist() to strip tz before passing to Event."
        )

    ef = EventCalendarFeatures()
    rows = []
    for ts in df["entry_time"]:
        e = Event(symbol="DUMMY", event_time=ts, metadata={})
        rows.append(ef.compute(e, bars=pd.DataFrame()))
    feat_df = pd.DataFrame(rows)
    return pd.concat([df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)


def run_target_ensemble_live_setups() -> dict:
    cfg = load_base_config()
    edisc = cfg["edge_discovery"]
    tcfg = edisc["target_ensemble_live_setups"]

    live_setups = list(tcfg["live_setups"])
    scan_min_n = int(tcfg["scan_min_n"])
    scan_top_n = int(tcfg["scan_top_n"])
    scan_max_dims = int(tcfg["scan_max_dims"])
    scan_outcome = str(tcfg["scan_outcome"])
    scan_feature_names = list(tcfg["scan_feature_names"])
    entry_time_aliases = list(tcfg["entry_time_column_aliases"])
    pnl_aliases = list(tcfg["pnl_column_aliases"])
    parquet_dir = str(tcfg["trade_parquet_dir"])

    _REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for setup in live_setups:
        try:
            trades = _load_setup_trades(setup, parquet_dir)
        except FileNotFoundError as e:
            print(f"[ensemble] SKIP {setup}: {e}")
            continue

        # Identify the PnL / return column
        pnl_col = None
        for c in pnl_aliases:
            if c in trades.columns:
                pnl_col = c
                break
        if pnl_col is None:
            print(f"[ensemble] SKIP {setup}: no return column (searched {pnl_aliases})")
            continue

        try:
            trades = _attach_event_features(trades, entry_time_aliases)
        except (KeyError, ValueError) as e:
            print(f"[ensemble] SKIP {setup}: event-feature attachment failed — {e}")
            continue

        # Ensure cap_segment present (parquet for gap_fade_short already has it)
        if "cap_segment" not in trades.columns:
            trades["cap_segment"] = "unknown"

        # Build outcome column: raw PnL value (positive = win)
        trades["outcome"] = trades[pnl_col]

        table = ConditionalOutcomeTable(rows=trades)

        # Use only features that actually exist in the DataFrame
        existing = [f for f in scan_feature_names if f in trades.columns]
        missing = [f for f in scan_feature_names if f not in trades.columns]
        if missing:
            print(f"[ensemble] {setup}: feature(s) not in DataFrame, skipping in scan: {missing}")

        regions = table.top_edge_regions(
            outcome=scan_outcome,
            feature_names=existing,
            min_n=scan_min_n,
            top_n=scan_top_n,
            max_dims=scan_max_dims,
        )

        out_path = _REPORT_DIR / f"ensemble_{setup}_regions.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(regions, f, indent=2, default=str)
        print(
            f"[ensemble] {setup}: {len(trades):,} trades, "
            f"{len(existing)} features, {len(regions)} regions -> {out_path}"
        )
        results[setup] = {
            "n_trades": int(len(trades)),
            "pnl_col_used": pnl_col,
            "features_scanned": existing,
            "regions_path": str(out_path),
            "top_3": regions[:3],
        }

    summary_path = _REPORT_DIR / "target_ensemble_live_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[ensemble] summary written to {summary_path}")
    return results


if __name__ == "__main__":
    run_target_ensemble_live_setups()
