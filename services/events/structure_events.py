# services/events/structure_events.py
"""
Detects structural intraday events driven by price/volume/VWAP/levels.

Events:
- breakout_long / breakout_short         : PDH/PDL/ORH/ORL break + hold + volume
- vwap_reclaim_long / vwap_lose_short    : VWAP cross + hold + volume
- squeeze_release_long / squeeze_release_short : recent vol expansion vs prior regime
- failure_fade_long / failure_fade_short : failed breakout at PDH/PDL

Strict config (NO in-code defaults):
  entry_config.json must define:
    - k_atr              (e.g., 0.25)
    - hold_bars          (e.g., 1)
    - vol_z_required     (e.g., 1.5)
    - width_window       (e.g., 40)
    - expansion_ratio    (e.g., 1.5)

Assumptions:
- DataFrames have naive IST DatetimeIndex (use utils.time_util.ensure_naive_ist_index).
- Columns expected where applicable: ["open","high","low","close","volume","vwap?","ts?"].
  If "ts" column is absent, the index is used as timestamp.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from config.logging_config import get_loggers
from config.filters_setup import load_filters
from utils.time_util import ensure_naive_ist_index

logger, _ = get_loggers()


@dataclass
class StructureEvent:
    setup: str
    ts: pd.Timestamp
    level_name: str
    strength: float


class StructureEventDetector:
    """Structure-based event detector (config-driven; no code defaults)."""

    def __init__(self) -> None:
        # Fail fast if required keys are missing (filters_setup enforces this)
        self.cfg = load_filters()

    # ---- internal helpers -------------------------------------------------

    @staticmethod
    def _safe_ts(d: pd.DataFrame) -> pd.Timestamp:
        """Return last timestamp from 'ts' column if present, else from index."""
        if "ts" in d.columns:
            return pd.to_datetime(d["ts"].iloc[-1])
        return pd.to_datetime(d.index[-1])

    @staticmethod
    def _vol_z(d: pd.DataFrame, win: int = 30, minp: int = 10) -> pd.Series:
        """
        Volume Z-score ~ (vol - mean) / std using a rolling window.
        Safe against zero std (returns 0 where std==0).
        """
        mu = d["volume"].rolling(win, min_periods=minp).mean()
        sd = d["volume"].rolling(win, min_periods=minp).std(ddof=0)
        z = (d["volume"] - mu) / sd.replace(0, np.nan)
        return z.fillna(0)

    # ---- detectors --------------------------------------------------------

    def detect_level_breakouts(self, df: pd.DataFrame, levels: Dict[str, float]) -> List[StructureEvent]:
        """
        Breakouts across PDH/PDL/ORH/ORL with hold & volume confirmation.

        Requires in entry_config.json:
          k_atr, hold_bars, vol_z_required
        """
        try:
            if df is None or df.empty or len(df) < 5:
                return []

            d = ensure_naive_ist_index(df.copy())
            d["vol_z"] = self._vol_z(d)

            # Simple ATR proxy using abs returns mean over 14 bars
            atr = float(
                max(
                    1e-9,
                    d["close"].pct_change().abs().rolling(14, min_periods=5).mean().iloc[-1],
                )
            )
            last = d.iloc[-1]

            k_atr = float(self.cfg["k_atr"])
            hold_bars = int(self.cfg["hold_bars"])
            vol_z_required = float(self.cfg["vol_z_required"])

            out: List[StructureEvent] = []
            for name, lvl in (levels or {}).items():
                if lvl is None or not np.isfinite(lvl):
                    continue

                # Long breakout above PDH/ORH
                if name in ("PDH", "ORH"):
                    cond = (
                        (last["close"] > lvl + k_atr * atr)
                        and (d["close"] > lvl).tail(hold_bars).all()
                        and d["vol_z"].iloc[-1] >= vol_z_required
                    )
                    if cond:
                        ts = self._safe_ts(d)
                        evt = StructureEvent("breakout_long", ts, name, float(d["vol_z"].iloc[-1]))
                        out.append(evt)
                        logger.info(f"structure_event: {evt}")

                # Short breakdown below PDL/ORL
                if name in ("PDL", "ORL"):
                    cond = (
                        (last["close"] < lvl - k_atr * atr)
                        and (d["close"] < lvl).tail(hold_bars).all()
                        and d["vol_z"].iloc[-1] >= vol_z_required
                    )
                    if cond:
                        ts = self._safe_ts(d)
                        evt = StructureEvent("breakout_short", ts, name, float(d["vol_z"].iloc[-1]))
                        out.append(evt)
                        logger.info(f"structure_event: {evt}")

            return out

        except Exception as e:
            logger.exception(f"structure_event: detect_level_breakouts error: {e}")
            return []

    def detect_vwap_cross_and_hold(self, df: pd.DataFrame) -> List[StructureEvent]:
        """
        VWAP reclaim (long) / lose (short) with volume confirmation.

        Requires in entry_config.json:
          hold_bars, vol_z_required
        """
        try:
            if df is None or df.empty or "vwap" not in df.columns:
                return []

            d = ensure_naive_ist_index(df.copy())
            d["vol_z"] = self._vol_z(d)

            hold_bars = int(self.cfg["hold_bars"])
            vol_z_required = float(self.cfg["vol_z_required"])
            out: List[StructureEvent] = []

            # Reclaim (close > vwap for hold_bars) + volume
            if (d["close"] > d["vwap"]).tail(hold_bars).all() and d["vol_z"].iloc[-1] >= vol_z_required:
                ts = self._safe_ts(d)
                evt = StructureEvent("vwap_reclaim_long", ts, "VWAP", float(d["vol_z"].iloc[-1]))
                out.append(evt)
                logger.info(f"structure_event: {evt}")

            # Lose (close < vwap for hold_bars) + volume
            if (d["close"] < d["vwap"]).tail(hold_bars).all() and d["vol_z"].iloc[-1] >= vol_z_required:
                ts = self._safe_ts(d)
                evt = StructureEvent("vwap_lose_short", ts, "VWAP", float(d["vol_z"].iloc[-1]))
                out.append(evt)
                logger.info(f"structure_event: {evt}")

            return out

        except Exception as e:
            logger.exception(f"structure_event: detect_vwap_cross_and_hold error: {e}")
            return []

    def detect_squeeze_release(self, df: pd.DataFrame) -> List[StructureEvent]:
        """
        Volatility regime expansion:
          recent width (σ/μ over last 5 bars) vs prior average width over a window.

        Requires in entry_config.json:
          width_window, expansion_ratio
        """
        try:
            width_window = int(self.cfg["width_window"])
            expansion_ratio = float(self.cfg["expansion_ratio"])

            if df is None or df.empty or len(df) < width_window + 5:
                return []

            d = ensure_naive_ist_index(df.copy())

            # Width = std/mean over 20 bars (min_periods=20) as a simple squeeze proxy
            std20 = d["close"].rolling(20, min_periods=20).std(ddof=0)
            mean20 = d["close"].rolling(20, min_periods=20).mean().replace(0, np.nan)
            width = (std20 / mean20).replace([np.inf, -np.inf], np.nan)

            recent = width.iloc[-5:].mean()
            prior = width.iloc[-(width_window + 5) : -5].mean()

            if pd.isna(prior) or prior == 0 or pd.isna(recent):
                return []

            if recent > expansion_ratio * prior:
                r3 = float(d["close"].pct_change(3).iloc[-1])
                setup = "squeeze_release_long" if r3 > 0 else "squeeze_release_short"
                ts = self._safe_ts(d)
                evt = StructureEvent(setup, ts, "SQUEEZE", abs(r3))
                logger.info(f"structure_event: {evt}")
                return [evt]

            return []

        except Exception as e:
            logger.exception(f"structure_event: detect_squeeze_release error: {e}")
            return []

    def detect_level_failure_fade(self, df: pd.DataFrame, levels: Dict[str, float]) -> List[StructureEvent]:
        """
        Failure fade: previous bar pierced level, next bar closed back within it.
        PDH -> short fade; PDL -> long fade.
        """
        try:
            if df is None or df.empty or len(df) < 2:
                return []

            d = ensure_naive_ist_index(df.copy())
            last, prev = d.iloc[-1], d.iloc[-2]

            out: List[StructureEvent] = []

            PDH = (levels or {}).get("PDH")
            PDL = (levels or {}).get("PDL")

            if PDH is not None and np.isfinite(PDH):
                if (prev["high"] > PDH) and (last["close"] < PDH):
                    ts = self._safe_ts(d)
                    evt = StructureEvent("failure_fade_short", ts, "PDH", 1.0)
                    out.append(evt)
                    logger.info(f"structure_event: {evt}")

            if PDL is not None and np.isfinite(PDL):
                if (prev["low"] < PDL) and (last["close"] > PDL):
                    ts = self._safe_ts(d)
                    evt = StructureEvent("failure_fade_long", ts, "PDL", 1.0)
                    out.append(evt)
                    logger.info(f"structure_event: {evt}")

            return out

        except Exception as e:
            logger.exception(f"structure_event: detect_level_failure_fade error: {e}")
            return []
        
    # put this inside class StructureEventDetector
    def detect_setups(self, symbol: str, df5m_tail: pd.DataFrame, levels: Dict[str, float] | None):
        """
        Adapter for TradeDecisionGate:
        - runs our specific detectors
        - maps StructureEvent -> SetupCandidate
        - returns List[SetupCandidate] sorted by strength desc
        """
        # local import avoids any chance of circular import at module load time
        from services.gates.trade_decision_gate import SetupCandidate

        d = ensure_naive_ist_index(df5m_tail.copy())

        evts = []
        evts += self.detect_level_breakouts(d, levels or {})
        evts += self.detect_vwap_cross_and_hold(d)
        evts += self.detect_squeeze_release(d)
        evts += self.detect_level_failure_fade(d, levels or {})

        setups = []
        for e in evts or []:
            # e.setup should be one of:
            #   'breakout_long','breakout_short','vwap_reclaim_long','vwap_lose_short',
            #   'squeeze_release_long','squeeze_release_short','failure_fade_long','failure_fade_short'
            reasons = []
            lvl = getattr(e, "level_name", None)
            if lvl:
                reasons.append(f"level:{lvl}")
            strength = float(getattr(e, "strength", 1.0))
            setups.append(SetupCandidate(setup_type=e.setup, strength=strength, reasons=reasons))

        # strongest first (gate will pick the max anyway)
        setups.sort(key=lambda s: s.strength, reverse=True)
        return setups
