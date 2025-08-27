from __future__ import annotations
"""
regime_gate.py
---------------
Classifies the market regime from 5‑minute index bars and exposes simple
permissions for setup types. No config reads and no hidden defaults in the
caller path: you pass the index DataFrame, it returns a regime label.

Expected input DataFrame (IST‑aligned, 5‑minute close index):
  columns: [open, high, low, close, volume, vwap] (bb_width_proxy optional)

Public API
----------
MarketRegimeGate.compute_regime(df5) -> (regime: str, metrics: dict)
MarketRegimeGate.allow_setup(setup_type: str, regime: str) -> bool
MarketRegimeGate.size_multiplier(regime: str, counter_trend: bool=False) -> float

Regime labels
-------------
  "trend_up" | "trend_down" | "chop" | "squeeze"

Setup types (consistent with our structure detectors):
  "breakout_long", "breakout_short",
  "vwap_reclaim_long", "vwap_lose_short",
  "squeeze_release_long", "squeeze_release_short",
  "failure_fade_long", "failure_fade_short"
"""
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeMetrics:
    ema_fast: float
    ema_slow: float
    slope_fast: float
    width_proxy: float
    vol_z: float


class MarketRegimeGate:
    """Computes regime using only the provided 5‑minute index bars.

    Heuristics (simple and robust):
      • Trend up   → EMA(20) > EMA(50) and recent slope of EMA(20) > 0
      • Trend down → EMA(20) < EMA(50) and recent slope of EMA(20) < 0
      • Squeeze    → Bollinger‑width proxy in bottom ~30% of last 2 hours
      • Else       → Chop

    We keep constants internal to avoid config sprawl. No external reads.
    """

    def compute_regime(self, df5: pd.DataFrame) -> Tuple[str, Dict[str, float]]:
        if df5 is None or df5.empty:
            # No data → assume chop; callers can decide to be conservative
            return "chop", {}

        c = df5["close"].astype(float).copy()
        # guard for short warmup
        if len(c) < 55:
            ema20 = c.ewm(span=20, adjust=False, min_periods=20).mean()
            ema50 = c.ewm(span=50, adjust=False, min_periods=50).mean()
        else:
            ema20 = c.ewm(span=20, adjust=False).mean()
            ema50 = c.ewm(span=50, adjust=False).mean()

        # slope over the last 6 bars (~30 minutes)
        window = 6 if len(ema20) >= 6 else max(2, len(ema20))
        slope_fast = float((ema20.diff().tail(window)).mean())

        # width proxy: std(close,20)/vwap if available, else std/mean
        if "bb_width_proxy" in df5.columns:
            width_proxy = float(df5["bb_width_proxy"].tail(20).mean())
        else:
            std20 = float(c.tail(20).std(ddof=0)) if len(c) >= 5 else 0.0
            mean20 = float(c.tail(20).mean()) if len(c) >= 1 else 1.0
            width_proxy = (std20 / mean20) if mean20 else 0.0

        # intraday volume z against recent 24 bars (~2 hours)
        if "volume" in df5.columns and len(df5) >= 10:
            vol = df5["volume"].astype(float)
            recent = vol.tail(min(24, len(vol)))
            mu, sd = float(recent.mean()), float(recent.std(ddof=0))
            vol_z = 0.0 if sd == 0.0 else float((recent.iloc[-1] - mu) / sd)
        else:
            vol_z = 0.0

        m = RegimeMetrics(
            ema_fast=float(ema20.iloc[-1]),
            ema_slow=float(ema50.iloc[-1]) if not np.isnan(ema50.iloc[-1]) else float(ema20.iloc[-1]),
            slope_fast=slope_fast,
            width_proxy=width_proxy,
            vol_z=vol_z,
        )

        # squeeze if width is very low vs its recent range
        width_series = (df5["bb_width_proxy"] if "bb_width_proxy" in df5.columns else c.pct_change().abs())
        recent_w = width_series.tail(min(24, len(width_series)))
        if len(recent_w) >= 8:
            q30 = float(recent_w.quantile(0.30))
            is_squeeze = m.width_proxy <= q30
        else:
            is_squeeze = False

        if m.ema_fast > m.ema_slow and m.slope_fast > 0 and not is_squeeze:
            regime = "trend_up"
        elif m.ema_fast < m.ema_slow and m.slope_fast < 0 and not is_squeeze:
            regime = "trend_down"
        elif is_squeeze:
            regime = "squeeze"
        else:
            regime = "chop"

        return regime, {
            "ema_fast": m.ema_fast,
            "ema_slow": m.ema_slow,
            "slope_fast": m.slope_fast,
            "width_proxy": m.width_proxy,
            "vol_z": m.vol_z,
        }

    # ---------------- Permissions & sizing ----------------
    def allow_setup(self, setup_type: str, regime: str) -> bool:
        """Simple allow matrix by regime.

        • Trend up: allow long breakouts/reclaims; disallow fresh short breakouts
        • Trend down: mirror of trend up
        • Chop: disable breakouts; allow failure fades only
        • Squeeze: allow squeeze releases both sides
        """
        if regime == "trend_up":
            return setup_type in {"breakout_long", "vwap_reclaim_long", "failure_fade_short"}
        if regime == "trend_down":
            return setup_type in {"breakout_short", "vwap_lose_short", "failure_fade_long"}
        if regime == "chop":
            return setup_type in {"failure_fade_long", "failure_fade_short"}
        if regime == "squeeze":
            return setup_type in {"squeeze_release_long", "squeeze_release_short"}
        return False

    def size_multiplier(self, regime: str, *, counter_trend: bool = False) -> float:
        """Return a simple sizing bias. Caller can multiply their base size.

        We keep the values conservative and deterministic.
        """
        if regime == "trend_up" or regime == "trend_down":
            return 0.7 if counter_trend else 1.15
        if regime == "squeeze":
            return 0.9  # uncertainty until release confirms
        return 0.85  # chop
