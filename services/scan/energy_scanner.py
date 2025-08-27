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

            # Basic returns
            dft["ret"] = dft["close"].pct_change()
            ret_1 = float(dft["ret"].iloc[-1]) if len(dft) >= 2 else 0.0
            ret_3 = float(dft["close"].pct_change(3).iloc[-1]) if len(dft) >= 4 else 0.0

            # ATR% proxy (mean absolute return over 14 bars)
            if len(dft) >= 5:
                atr_pct14 = float(dft["ret"].abs().rolling(14, min_periods=5).mean().iloc[-1] or 0.0)
            else:
                atr_pct14 = 0.0

            # Volume z‑score (intraday baseline over last 20 bars)
            if len(dft) >= 6:
                vol = dft["volume"].astype(float)
                mu = float(vol.tail(20).mean())
                sd = float(vol.tail(20).std(ddof=0) or 0.0)
                vol_z20 = (float(vol.iloc[-1]) - mu) / sd if sd > 0 else 0.0
            else:
                vol_z20 = 0.0

            # Distance to VWAP (dimensionless)
            close = float(dft["close"].iloc[-1])
            vwap = float(dft.get("vwap", dft["close"]).iloc[-1])
            dist_to_vwap = (close - vwap) / vwap if vwap else 0.0

            # Compression proxy (if present)
            bb_width_proxy = float(dft.get("bb_width_proxy", pd.Series([0.0])).iloc[-1] or 0.0)

            # Optional: distances to structural levels to nudge rank
            lvs = (levels_by_symbol or {}).get(sym, {})
            dist_pdh = ((close - float(lvs.get("PDH", np.nan))) / close) if "PDH" in lvs else np.nan
            dist_pdl = ((close - float(lvs.get("PDL", np.nan))) / close) if "PDL" in lvs else np.nan
            dist_orh = ((close - float(lvs.get("ORH", np.nan))) / close) if "ORH" in lvs else np.nan
            dist_orl = ((close - float(lvs.get("ORL", np.nan))) / close) if "ORL" in lvs else np.nan

            # Energy scores (dimensionless). Keep it cheap & stable.
            # Long likes: +volume, +momentum, near levels above, small compression, near VWAP or above
            mom = ret_1 + 0.5 * ret_3
            atr_nz = atr_pct14 if atr_pct14 > 1e-9 else 1e-9
            mom_norm = mom / atr_nz
            score_long = (
                0.40 * vol_z20 +
                0.35 * mom_norm +
                0.15 * (-abs(dist_to_vwap)) +
                0.10 * (0.0 if np.isnan(bb_width_proxy) else -bb_width_proxy)
            )
            # Nudge if near PDH/ORH (potential breakout magnet)
            if not np.isnan(dist_pdh):
                score_long += 0.10 * (-abs(dist_pdh))
            if not np.isnan(dist_orh):
                score_long += 0.05 * (-abs(dist_orh))

            # Short likes: mirror signals
            score_short = (
                0.40 * vol_z20 +
                0.35 * (-mom_norm) +
                0.15 * (-abs(dist_to_vwap)) +
                0.10 * (0.0 if np.isnan(bb_width_proxy) else -bb_width_proxy)
            )
            if not np.isnan(dist_pdl):
                score_short += 0.10 * (-abs(dist_pdl))
            if not np.isnan(dist_orl):
                score_short += 0.05 * (-abs(dist_orl))

            rows.append(
                {
                    "symbol": sym,
                    "ts": dft.index[-1],
                    "close": close,
                    "vwap": vwap,
                    "volume": float(dft["volume"].iloc[-1]),
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
