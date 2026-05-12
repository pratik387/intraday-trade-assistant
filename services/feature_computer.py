"""Bar feature compute — extracted from EnergyScanner.compute_features.

2026-05-12 architectural refactor: features are now computed only for the
universe-union shortlist (~200 symbols/bar), not the full eligible pool
(1500). Detector access to feature scores moves through plan["model_features"]
the same way as before; this utility just narrows the input domain.

Pure data-driven — no scanner instance state, no top-K filtering. The
shortlist (universe) is the input domain.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def _vwap(df: pd.DataFrame) -> float:
    """Volume-weighted average over the bars in df."""
    v = df["volume"].astype(float)
    p = ((df["high"] + df["low"] + df["close"]) / 3.0).astype(float)
    tot_v = v.sum()
    return float((p * v).sum() / tot_v) if tot_v > 0 else float(df["close"].iloc[-1])


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return float(a / b) if b not in (0, 0.0) else default


def compute_bar_features(
    df5_by_symbol: Dict[str, pd.DataFrame],
    universe: Iterable[str],
    now: pd.Timestamp,
    levels_by_symbol: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.DataFrame:
    """Compute per-symbol bar features over the universe.

    Args:
      df5_by_symbol: full 5m DataFrame per symbol (must have open/high/low/close/volume,
        DatetimeIndex). Length should be >= 2 for full features.
      universe: set/list of symbols to compute for. Symbols outside are skipped.
      now: bar timestamp; rows are computed using bars at <= now.
      levels_by_symbol: optional {sym: {PDH, PDL, PDC, ORH, ORL}} for distance fields.

    Returns:
      pd.DataFrame with columns:
        symbol, close, vwap, volume, ret_1, ret_3, atr_pct14, vol_z20,
        dist_to_vwap, bb_width_proxy, dist_to_PDH, dist_to_PDL,
        turnover, vol_ratio, rank_score
      One row per symbol that had >= 2 bars in df.
    """
    universe_set = set(universe)
    rows = []
    levels_by_symbol = levels_by_symbol or {}

    for sym, dft in df5_by_symbol.items():
        if sym not in universe_set:
            continue
        if dft is None or dft.empty or len(dft) < 2:
            continue

        close_series = dft["close"].astype(float)
        volume_series = dft["volume"].astype(float)
        close = float(close_series.iloc[-1])
        vwap = _vwap(dft.tail(20))

        # ret_1 = current vs prior bar
        prev = float(close_series.iloc[-2])
        ret_1 = _safe_div(close - prev, prev)
        ret_3 = (
            _safe_div(close - float(close_series.iloc[-4]), float(close_series.iloc[-4]))
            if len(close_series) >= 4 else 0.0
        )

        # ATR pct over 14 bars (high-low/close)
        if len(dft) >= 14:
            tr = (dft["high"] - dft["low"]).tail(14).astype(float)
            atr_pct14 = float(tr.mean() / close) if close > 0 else 0.0
        else:
            atr_pct14 = 0.0

        # vol_z20 = z-score of current volume vs trailing 20-bar mean
        vol_tail = volume_series.tail(20)
        if len(vol_tail) >= 2:
            mu = float(vol_tail.mean())
            sigma = float(vol_tail.std(ddof=0))
            vol_z20 = _safe_div(float(volume_series.iloc[-1]) - mu, sigma)
            vol_ratio = _safe_div(float(volume_series.iloc[-1]), mu)
        else:
            vol_z20 = 0.0
            vol_ratio = 1.0

        # Distance to VWAP (signed, as fraction)
        dist_to_vwap = _safe_div(close - vwap, vwap)

        # Bollinger-width proxy: 2*std / mean over trailing 20 close
        if len(close_series) >= 20:
            mu_c = float(close_series.tail(20).mean())
            sd_c = float(close_series.tail(20).std(ddof=0))
            bb_width_proxy = _safe_div(2 * sd_c, mu_c)
        else:
            bb_width_proxy = 0.0

        # Distance to PDH/PDL if levels provided
        lvls = levels_by_symbol.get(sym) or {}
        pdh = float(lvls.get("PDH") or 0.0)
        pdl = float(lvls.get("PDL") or 0.0)
        dist_pdh = _safe_div(close - pdh, pdh) if pdh else 0.0
        dist_pdl = _safe_div(close - pdl, pdl) if pdl else 0.0

        turnover = close * float(volume_series.iloc[-1])

        # rank_score: lightweight momentum+volume composite (replaces Stage-0
        # base_score). Kept for analytics + plan logging. Not used to filter.
        rank_score = float(
            (abs(ret_1) * 100.0) + (vol_z20 * 0.5) + (abs(dist_to_vwap) * 50.0)
        )

        rows.append({
            "symbol": sym,
            "close": close,
            "vwap": vwap,
            "volume": float(volume_series.iloc[-1]),
            "ret_1": ret_1,
            "ret_3": ret_3,
            "atr_pct14": atr_pct14,
            "vol_z20": vol_z20,
            "dist_to_vwap": dist_to_vwap,
            "bb_width_proxy": bb_width_proxy,
            "dist_to_PDH": dist_pdh,
            "dist_to_PDL": dist_pdl,
            "turnover": turnover,
            "vol_ratio": vol_ratio,
            "rank_score": rank_score,
        })

    if not rows:
        return pd.DataFrame(columns=[
            "symbol", "close", "vwap", "volume", "ret_1", "ret_3",
            "atr_pct14", "vol_z20", "dist_to_vwap", "bb_width_proxy",
            "dist_to_PDH", "dist_to_PDL", "turnover", "vol_ratio", "rank_score",
        ])
    return pd.DataFrame(rows)
