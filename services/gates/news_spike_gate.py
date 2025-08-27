from __future__ import annotations
"""
news_spike_gate.py
------------------
Detects very-short-horizon anomalies that are usually caused by news/announcements
or sudden order-flow imbalances. It does **not** fetch any data or read config;
callers must pass a 1‑minute bar tail (today) and explicit thresholds.

API
---
- NewsSpikeGate(window_bars, vol_z_thresh, ret_z_thresh, body_atr_ratio_thresh)
- has_symbol_spike(df1m_tail) -> tuple[bool, NewsSignal]
- adjustment_for(signal) -> Adjustment

Typical use in engine
---------------------
    gate = NewsSpikeGate(window_bars=30,
                         vol_z_thresh=3.0,
                         ret_z_thresh=2.0,
                         body_atr_ratio_thresh=2.0)
    spike, sig = gate.has_symbol_spike(df1m_tail)
    adj = gate.adjustment_for(sig)
    # Enforce in the decision layer (e.g., TradeDecisionGate):
    #   min_hold_bars += adj.require_hold_bars
    #   size_mult     *= adj.size_mult

Notes
-----
• We evaluate the **last closed 1m bar** against the prior `window_bars` minutes
  to avoid look-ahead. If not enough data, we return "no spike".
• ATR is a short rolling mean (by default computed on the same window) and we
  measure candle body / ATR to catch extraordinary single-bar impulses.
"""
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NewsSignal:
    spike: bool
    reasons: List[str]
    vol_z: float
    ret_z: float
    body_atr_ratio: float


@dataclass(frozen=True)
class Adjustment:
    """How decision policy should adapt when a news spike is detected."""
    require_hold_bars: int  # extra confirmation bars to require
    size_mult: float        # multiply base size by this factor


class NewsSpikeGate:
    """Lightweight 1m anomaly detector.

    Parameters
    ----------
    window_bars : int
        Number of **prior** 1m bars used to compute baselines (>= 10 recommended).
    vol_z_thresh : float
        Z-score threshold for volume spike.
    ret_z_thresh : float
        Z-score threshold for absolute 1m return.
    body_atr_ratio_thresh : float
        If |close-open| / ATR_1m_mean >= this threshold, mark as spike.
    """

    def __init__(
        self,
        *,
        window_bars: int,
        vol_z_thresh: float,
        ret_z_thresh: float,
        body_atr_ratio_thresh: float,
    ) -> None:
        if window_bars < 5:
            raise ValueError("window_bars must be >= 5")
        self.window = int(window_bars)
        self.vol_z_thresh = float(vol_z_thresh)
        self.ret_z_thresh = float(ret_z_thresh)
        self.body_atr_ratio_thresh = float(body_atr_ratio_thresh)

    # ------------------------------ Public API ------------------------------
    def has_symbol_spike(self, df1m_tail: pd.DataFrame) -> Tuple[bool, NewsSignal]:
        """Return (is_spike, NewsSignal) for the **last closed bar** in df1m_tail.

        Expects df1m_tail with columns: [open, high, low, close, volume].
        Index should be the 1m close timestamps; only the last row is evaluated.
        """
        if df1m_tail is None or df1m_tail.empty:
            return False, NewsSignal(False, ["no_data"], 0.0, 0.0, 0.0)
        if len(df1m_tail) < self.window + 1:
            return False, NewsSignal(False, ["insufficient_history"], 0.0, 0.0, 0.0)

        # Split history vs. last closed bar (no look-ahead)
        hist = df1m_tail.iloc[-(self.window + 1):-1]
        last = df1m_tail.iloc[-1]

        # Volume z-score of last bar vs prior baseline
        vol = hist["volume"].astype(float)
        vol_mu = float(vol.mean()) if len(vol) else 0.0
        vol_sd = float(vol.std(ddof=0)) if len(vol) else 0.0
        vol_z = (float(last["volume"]) - vol_mu) / vol_sd if vol_sd > 0 else 0.0

        # 1m return absolute z-score vs prior returns distribution
        ret = hist["close"].pct_change().dropna()
        ret_mu = float(ret.mean()) if len(ret) else 0.0
        ret_sd = float(ret.std(ddof=0)) if len(ret) else 0.0
        ret_last = 0.0
        if (c := float(last.get("close", np.nan))) and (p := float(hist.iloc[-1].get("close", np.nan))):
            if not np.isnan(c) and not np.isnan(p) and p != 0.0:
                ret_last = (c - p) / p
        ret_z = (abs(ret_last) - abs(ret_mu)) / ret_sd if ret_sd > 0 else 0.0

        # Candle body vs ATR_1m (short mean)
        body = abs(float(last["close"]) - float(last["open"]))
        atr_1m = _atr_1m_mean(hist)
        body_atr_ratio = (body / atr_1m) if atr_1m > 0 else 0.0

        reasons: List[str] = []
        if vol_z >= self.vol_z_thresh:
            reasons.append(f"vol_z={vol_z:.2f}>={self.vol_z_thresh}")
        if ret_z >= self.ret_z_thresh:
            reasons.append(f"ret_z={ret_z:.2f}>={self.ret_z_thresh}")
        if body_atr_ratio >= self.body_atr_ratio_thresh:
            reasons.append(f"body/atr={body_atr_ratio:.2f}>={self.body_atr_ratio_thresh}")

        spike = len(reasons) > 0
        sig = NewsSignal(spike=spike, reasons=reasons, vol_z=vol_z, ret_z=ret_z, body_atr_ratio=body_atr_ratio)
        return spike, sig

    def adjustment_for(self, signal: NewsSignal) -> Adjustment:
        """Map a signal to conservative policy adjustments.

        Default mapping (tuned later via data):
          • any spike → require 2 confirmation bars and reduce size by 10%.
          • very strong spike (vol_z>=5 or body/atr>=3.5) → require 3 bars, size 0.8.
        """
        if not signal.spike:
            return Adjustment(require_hold_bars=0, size_mult=1.0)

        strong = (signal.vol_z >= 5.0) or (signal.body_atr_ratio >= 3.5)
        if strong:
            return Adjustment(require_hold_bars=3, size_mult=0.8)
        return Adjustment(require_hold_bars=2, size_mult=0.9)


# ------------------------------- Internals ----------------------------------

def _atr_1m_mean(hist_1m: pd.DataFrame) -> float:
    """Compute a short ATR mean from 1m history (no look-ahead)."""
    if hist_1m is None or len(hist_1m) < 3:
        return 0.0
    h = hist_1m["high"].astype(float)
    l = hist_1m["low"].astype(float)
    c = hist_1m["close"].astype(float)
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (pc - l).abs()], axis=1).max(axis=1)
    return float(tr.rolling(14, min_periods=5).mean().iloc[-1]) if len(tr) >= 5 else float(tr.mean())
