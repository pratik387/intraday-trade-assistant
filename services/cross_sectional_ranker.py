"""Cross-sectional EOD ranker for multi-day CNC/MTF factor setups.

The one genuinely-new component of the multi-day-hold architecture. Given a
clean daily OHLCV panel (CA-adjusted) and a session date, it forms the target
basket for T+1 entry by cross-sectionally ranking the MTF-eligible universe.

Generic by construction: `mtf_capitulation_revert_long` plugs in the
loser-decile + turnover-shock selection; future CNC factors (PEAD, momentum)
swap the selection while reusing entry/exit/position machinery downstream.

Live/backtest-symmetric: identical logic whether `daily_panel` is the backtest
feather or the live rolling EOD store. IST-naive throughout. NO hardcoded
defaults — every parameter is read from the setup config (CLAUDE.md rule 1).
"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from config.logging_config import get_agent_logger

logger = get_agent_logger()


class CrossSectionalRanker:
    """Forms the daily target basket for a multi-day CNC/MTF setup."""

    # Selection modes the ranker supports. Each plugs a different "which names
    # capitulated" rule into the SAME universe/shock/tier/CA machinery.
    _MODES = ("trailing_loser_decile", "near_period_low", "zscore_oversold")

    def __init__(self, config: Dict[str, Any]):
        # Fail-fast: every key required, no silent defaults (CLAUDE.md rule 1).
        self.selection_mode = str(config["selection_mode"])
        if self.selection_mode not in self._MODES:
            raise ValueError(
                f"selection_mode {self.selection_mode!r} not in {self._MODES}"
            )
        # Mode-specific keys (read only the ones the chosen mode needs).
        if self.selection_mode == "trailing_loser_decile":
            self.lookback_days = int(config["lookback_days"])      # trailing-return window
            self.loser_pct = float(config["loser_pct"])            # bottom cross-sectional cut
        elif self.selection_mode == "near_period_low":
            self.low_lookback_days = int(config["low_lookback_days"])  # e.g. 252d low
            self.dist_low_max = float(config["dist_low_max"])          # close within X of the low
        else:  # zscore_oversold
            self.zscore_lookback_days = int(config["zscore_lookback_days"])  # mean/std window (e.g. 20)
            self.zscore_max = float(config["zscore_max"])                    # select close <= this many sd below mean (negative)
        # Common keys.
        self.adv_tier = int(config["adv_tier"])
        self.adv_tier_count = int(config["adv_tier_count"])
        self.turnover_shock_min = float(config["turnover_shock_min"])
        self.shock_lookback_days = int(config["shock_lookback_days"])
        self.adv_floor_inr = float(config["adv_floor_inr"])
        self.min_price = float(config["min_price"])
        self.min_universe = int(config["min_universe_symbols_per_day"])
        self.hold_days = int(config["hold_days"])
        self.exclude_ca_in_hold_window = bool(config["exclude_ca_in_hold_window"])

    def rank(
        self,
        daily_panel: pd.DataFrame,
        session_date: date,
        mtf_eligible: Set[str],
        ca_ex_dates: Optional[Dict[str, List[date]]] = None,
    ) -> List[Dict[str, Any]]:
        """Return the selected basket (list of dicts) for T+1 entry.

        Args:
            daily_panel: clean daily OHLCV; columns date, symbol, open, high,
                low, close, volume. `symbol` is the bare ticker. Must contain
                history through session_date.
            session_date: the signal date (T); entry is T+1 open downstream.
            mtf_eligible: bare symbols that are MTF-approved (leverageable).
            ca_ex_dates: optional {bare_symbol: [ex_date,...]} to exclude names
                with a corporate action inside the K-day hold window.

        Returns: [{symbol, trail_ret, tshock, adv_tier, rank_pct, close}], or []
            if the cross-section is too thin to rank reliably.
        """
        sd = pd.Timestamp(session_date).normalize()
        df = daily_panel.copy()
        df["date"] = pd.to_datetime(df["date"])
        if getattr(df["date"].dt, "tz", None) is not None:
            df["date"] = df["date"].dt.tz_localize(None)
        df = df[df["date"].dt.normalize() <= sd].sort_values(["symbol", "date"])
        if df.empty:
            return []

        # Bound compute: only the trailing window each symbol needs (mode-aware).
        if self.selection_mode == "trailing_loser_decile":
            need = max(self.lookback_days, self.shock_lookback_days) + 2
        elif self.selection_mode == "near_period_low":
            need = max(self.low_lookback_days, self.shock_lookback_days) + 2
        else:  # zscore_oversold
            need = max(self.zscore_lookback_days, self.shock_lookback_days) + 2
        df = df.groupby("symbol", sort=False).tail(need)

        g = df.groupby("symbol", sort=False)
        df["turnover"] = df["close"] * df["volume"]
        df["adv"] = g["turnover"].transform(lambda s: s.rolling(self.shock_lookback_days).mean())
        df["adv_prior"] = g["turnover"].transform(
            lambda s: s.shift(1).rolling(self.shock_lookback_days).mean()
        )
        df["tshock"] = df["turnover"] / df["adv_prior"]
        # Mode-specific signal. `signal` is the value the selection thresholds on;
        # both modes keep the same output schema (trail_ret carries the signal).
        if self.selection_mode == "trailing_loser_decile":
            df["signal"] = g["close"].transform(lambda s: s / s.shift(self.lookback_days) - 1.0)
        elif self.selection_mode == "near_period_low":  # how far above the trailing low (0 = at the low)
            low = g["low"].transform(
                lambda s: s.rolling(self.low_lookback_days, min_periods=max(20, self.shock_lookback_days)).min()
            )
            df["signal"] = df["close"] / low - 1.0
        else:  # zscore_oversold: standardized deviation below the rolling mean (negative = oversold)
            mp = max(20, self.shock_lookback_days)
            mean = g["close"].transform(lambda s: s.rolling(self.zscore_lookback_days, min_periods=mp).mean())
            std = g["close"].transform(lambda s: s.rolling(self.zscore_lookback_days, min_periods=mp).std())
            df["signal"] = (df["close"] - mean) / std

        # The session-date cross-section.
        today = df[df["date"].dt.normalize() == sd].copy()
        today = today[
            today["symbol"].isin(mtf_eligible)
            & today["signal"].notna()
            & np.isfinite(today["signal"])  # guard inf from a degenerate low=0
            & today["tshock"].notna()
            & (today["close"] >= self.min_price)
            & (today["adv"] >= self.adv_floor_inr)
        ].copy()  # own frame for the adv_tier/rank_pct/cap_score assignments below
        if len(today) < self.min_universe:
            logger.info(
                "x_sectional_ranker: %s only %d qualifying symbols (< min %d) -> no basket",
                sd.date(), len(today), self.min_universe,
            )
            return []

        try:
            today["adv_tier"] = pd.qcut(
                today["adv"], self.adv_tier_count,
                labels=list(range(1, self.adv_tier_count + 1)), duplicates="drop",
            ).astype("Int64")
        except ValueError:
            logger.warning("x_sectional_ranker: %s qcut failed (degenerate adv); no basket", sd.date())
            return []
        today["rank_pct"] = today["signal"].rank(pct=True)

        # Capitulation magnitude, oriented so MORE capitulated = larger, then
        # standardized cross-sectionally over the qualifying universe and
        # lower-clipped at 0 (tail only). The composite selector blends this
        # across setups; the UPPER clip (family-level cap_score_clip) is applied
        # there, so this stays a pure per-setup normalization (CLAUDE.md rule 1:
        # the clip value is not read here).
        mag = -today["signal"]
        mu = float(mag.mean())
        sd_mag = float(mag.std())
        if np.isfinite(sd_mag) and sd_mag > 0.0:
            today["cap_score"] = ((mag - mu) / sd_mag).clip(lower=0.0)
        else:
            today["cap_score"] = 0.0

        # Mode-specific selection rule (universe/tier/shock/CA gates are shared).
        if self.selection_mode == "trailing_loser_decile":
            sig_sel = today["rank_pct"] <= self.loser_pct      # bottom cross-sectional cut
        elif self.selection_mode == "near_period_low":         # absolute proximity to the trailing low
            sig_sel = today["signal"] <= self.dist_low_max
        else:  # zscore_oversold: at/below the negative z threshold
            sig_sel = today["signal"] <= self.zscore_max
        sel = today[
            sig_sel
            & (today["adv_tier"] == self.adv_tier)
            & (today["tshock"] >= self.turnover_shock_min)
        ]

        if self.exclude_ca_in_hold_window and ca_ex_dates:
            window = self._hold_window_dates(sd)
            sel = sel[~sel["symbol"].apply(
                lambda s: any(ex in window for ex in ca_ex_dates.get(s, []))
            )]

        out = [
            {
                "symbol": r["symbol"],
                "trail_ret": float(r["signal"]),  # signal value (trailing-ret or dist-from-low)
                "tshock": float(r["tshock"]),
                "adv_tier": int(r["adv_tier"]),
                "rank_pct": float(r["rank_pct"]),
                "close": float(r["close"]),
                "cap_score": float(r["cap_score"]),
            }
            for _, r in sel.iterrows()
        ]
        logger.info("x_sectional_ranker: %s selected %d names for T+1 entry", sd.date(), len(out))
        return out

    def _hold_window_dates(self, sd: pd.Timestamp) -> Set[date]:
        """Calendar-date window the position would be held (entry T+1 .. exit).

        Trading-day arithmetic is handled by the executor; here we use a
        conservative calendar span (hold_days + weekend buffer) to flag CA
        ex-dates that could fall inside the hold.
        """
        span = self.hold_days + 4  # buffer for weekends/holidays inside the hold
        return {(sd + pd.Timedelta(days=i)).date() for i in range(1, span + 1)}
