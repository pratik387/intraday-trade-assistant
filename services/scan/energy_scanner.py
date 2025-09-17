from __future__ import annotations
"""
EnergyScanner — ultra‑cheap, vectorized shortlist builder (Tier‑0).

This does NOT read config or make any network calls. Give it the latest
5‑minute bars per symbol (from BarBuilder), and it returns a balanced
shortlist of symbols for deeper checks.

Inputs expected per symbol DataFrame (tail of today only):
  index: 5‑minute close timestamps (IST‑naive)
  columns: [open, high, low, close, volume, vwap, bb_width_proxy]

Public API:
  - compute_features(df5_by_symbol: dict[str, pd.DataFrame], lookback_bars: int) -> pd.DataFrame
  - rank_candidates(features_df: pd.DataFrame) -> pd.DataFrame
  - select_shortlist(features_df: pd.DataFrame) -> dict[str, list[str]] with keys 'long','short'

Notes:
  • No defaults from config; constructor requires top_k_long and top_k_short.
  • If you have PDH/PDL/ORH/ORL distances, you can pass them in via the optional
    `levels_by_symbol` map to slightly improve ranking. Otherwise they are ignored.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class EnergyScanner:
    top_k_long: int
    top_k_short: int

    # ------------------------------ Core API ------------------------------ #
    def compute_features(
        self,
        df5_by_symbol: Dict[str, pd.DataFrame],
        *,
        lookback_bars: int,
        levels_by_symbol: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> pd.DataFrame:
        """Build one feature row per symbol using ONLY the last closed 5m bar.

        Parameters
        ----------
        df5_by_symbol : dict[str, DataFrame]
            Each DataFrame is the *today* 5m history for a symbol (at least a few rows).
        lookback_bars : int
            Number of recent bars to consider for rolling stats/z‑scores (e.g., 18).
        levels_by_symbol : optional map with keys like 'PDH','PDL','ORH','ORL'.

        Returns
        -------
        pd.DataFrame with columns:
          symbol, ts, close, vwap, volume,
          ret_1, ret_3, atr_pct14, vol_z20, dist_to_vwap,
          bb_width_proxy,
          (optional) dist_to_PDH, dist_to_PDL, dist_to_ORH, dist_to_ORL,
          score_long, score_short
        """
        rows: List[dict] = []
        for sym, df in df5_by_symbol.items():
            if df is None or df.empty or len(df) < 3:
                continue
            dft = df.tail(max(lookback_bars, 20)).copy()
            dft = dft.dropna(subset=["close"])  # be strict
            if dft.empty:
                continue

            #  Pre-compute series once (symbol-safe)
            close_series = dft["close"].astype(float)
            volume_series = dft["volume"].astype(float)

            #  Single vectorized operations per symbol
            returns = close_series.pct_change()
            ret_1 = float(returns.iloc[-1]) if len(dft) >= 2 else 0.0
            ret_3 = float(close_series.pct_change(3).iloc[-1]) if len(dft) >= 4 else 0.0

            #  ATR calculation (symbol-safe)
            if len(dft) >= 5:
                atr_pct14 = float(returns.abs().rolling(14, min_periods=5).mean().iloc[-1] or 0.0)
            else:
                atr_pct14 = 0.0

            #  Volume statistics (symbol-safe)
            if len(dft) >= 6:
                vol_tail20 = volume_series.tail(20)
                vol_mean = float(vol_tail20.mean())
                vol_std = float(vol_tail20.std(ddof=0) or 0.0)
                vol_z20 = (float(volume_series.iloc[-1]) - vol_mean) / vol_std if vol_std > 0 else 0.0
            else:
                vol_z20 = 0.0

            #  Direct value extraction
            close = float(close_series.iloc[-1])
            vwap = float(dft["vwap"].iloc[-1]) if "vwap" in dft.columns else close
            dist_to_vwap = (close - vwap) / vwap if vwap else 0.0

            #  Compression proxy
            bb_width_proxy = float(dft["bb_width_proxy"].iloc[-1]) if "bb_width_proxy" in dft.columns else 0.0

            # Optional: distances to structural levels to nudge rank
            lvs = (levels_by_symbol or {}).get(sym, {})
            dist_pdh = ((close - float(lvs.get("PDH", np.nan))) / close) if "PDH" in lvs else np.nan
            dist_pdl = ((close - float(lvs.get("PDL", np.nan))) / close) if "PDL" in lvs else np.nan
            dist_orh = ((close - float(lvs.get("ORH", np.nan))) / close) if "ORH" in lvs else np.nan
            dist_orl = ((close - float(lvs.get("ORL", np.nan))) / close) if "ORL" in lvs else np.nan
            # PROFESSIONAL: Winning-focused energy scores
            # Focus on extreme volume + directional bias + gap momentum
            
            # Calculate vol_ratio first
            if len(dft) >= 6:
                vol_ratio = float(volume_series.iloc[-1]) / float(vol_mean) if vol_mean > 0 else 1.0
            else:
                vol_ratio = 1.0

            # PROFESSIONAL STAGE-0: Capture quality opportunities, not just extremes

            # 1. PARTICIPATION FILTER (not extremes)
            vol_participation = min(vol_z20 * 1.0, 2.0) if vol_z20 > 0.8 else 0.0  # Above average volume
            vol_ratio_factor = min((vol_ratio - 1.5) * 0.8, 2.0) if vol_ratio > 1.5 else 0.0  # 1.5x+ volume

            # 2. MOVEMENT FILTER (not momentum)
            movement_factor = 0.0
            if abs(ret_1) > 0.003:  # 0.3%+ movement
                movement_factor = min(abs(ret_1) * 100, 2.0)  # Scale movement

            # 3. VOLATILITY AVAILABILITY
            atr_availability = min(atr_pct14 * 50, 1.5) if atr_pct14 > 0.008 else 0.0  # Some volatility

            # 4. STRUCTURE PROXIMITY (fast check)
            structure_factor = 0.0
            if abs(dist_to_vwap) < 0.02:  # Within 2% of VWAP
                structure_factor = 1.0

            # 5. TIME-DECAY URGENCY
            current_hour = dft.index[-1].hour
            current_minute = dft.index[-1].minute
            time_multiplier = 1.0

            if current_hour < 10 or (current_hour == 10 and current_minute < 30):
                # Early session: more lenient
                time_multiplier = 1.2
            elif current_hour >= 14:
                # Late session: require more movement
                time_multiplier = 0.8 if movement_factor < 1.0 else 1.1

            # 6. PROFESSIONAL SCORING (capture more quality candidates)
            base_score = (
                0.30 * vol_participation +        # Volume participation
                0.25 * movement_factor +          # Price movement
                0.20 * vol_ratio_factor +         # Volume ratio
                0.15 * atr_availability +         # Volatility available
                0.10 * structure_factor           # Near key levels
            ) * time_multiplier

            # Directional bias (simple)
            mom = ret_1 + 0.5 * ret_3
            directional_bias = 1.0
            if abs(mom) > 0.005:  # 0.5%+ directional movement
                directional_bias = 1.2

            score_long = base_score * directional_bias if mom >= 0 else base_score * 0.8
            score_short = base_score * directional_bias if mom <= 0 else base_score * 0.8

            # Calculate gap for logging (removed from scoring)
            if len(close_series) >= 2:
                gap_pct = abs(close - close_series.iloc[-2]) / close_series.iloc[-2]
            else:
                gap_pct = 0.0

            #  Volume persistence (symbol-safe)
            if len(dft) >= 4:
                vol_tail4 = volume_series.tail(4)
                vol_mean_3 = vol_tail4.tail(3).mean()
                vol_persist = int((vol_tail4.iloc[-1] >= vol_mean_3) and (vol_tail4.iloc[-2] >= vol_mean_3))
            else:
                vol_persist = 0

            turnover = close * float(volume_series.iloc[-1])
            rows.append(
                {
                    "symbol": sym,
                    "ts": dft.index[-1],
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
                    "dist_to_ORH": dist_orh,
                    "dist_to_ORL": dist_orl,
                    "turnover": turnover,    
                    "vol_ratio": vol_ratio,
                    "vol_persist_ok": int(vol_persist),
                    "gap_pct": gap_pct,
                    "score_long": float(score_long),
                    "score_short": float(score_short),
                }
            )

        return pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)

    def rank_candidates(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Attach rank columns and return a copy sorted by best long/short first.

        Output columns added: rank_long (1=best), rank_short (1=best).
        """
        if features_df is None or features_df.empty:
            return features_df.copy()
        df = features_df.copy()
        df = df.assign(
            rank_long=df["score_long"].rank(ascending=False, method="first").astype(int),
            rank_short=df["score_short"].rank(ascending=False, method="first").astype(int),
        )
        # Just return sorted by long then short; caller can slice either way
        return df.sort_values(["rank_long", "rank_short"])  # no reset; keep ts order

    def select_shortlist(self, features_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Return balanced shortlist {'long': [...], 'short': [...]}.
        Uses the rank columns if present, otherwise scores directly.
        """
        if features_df is None or features_df.empty:
            return {"long": [], "short": []}
        df = features_df.copy()
        if "rank_long" not in df.columns or "rank_short" not in df.columns:
            df = self.rank_candidates(df)
        long_syms = (
            df.sort_values("rank_long")["symbol"].head(self.top_k_long).tolist()
            if self.top_k_long > 0 else []
        )
        short_syms = (
            df.sort_values("rank_short")["symbol"].head(self.top_k_short).tolist()
            if self.top_k_short > 0 else []
        )
        return {"long": long_syms, "short": short_syms}
