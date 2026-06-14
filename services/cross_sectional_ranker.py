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

import pandas as pd

from config.logging_config import get_agent_logger

logger = get_agent_logger()


class CrossSectionalRanker:
    """Forms the daily target basket for a multi-day CNC/MTF setup."""

    def __init__(self, config: Dict[str, Any]):
        # Fail-fast: every key required, no silent defaults.
        self.lookback_days = int(config["lookback_days"])
        self.loser_pct = float(config["loser_pct"])
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

        # Bound compute: only the trailing window each symbol needs.
        need = max(self.lookback_days, self.shock_lookback_days) + 2
        df = df.groupby("symbol", sort=False).tail(need)

        g = df.groupby("symbol", sort=False)
        df["trail_ret"] = g["close"].transform(lambda s: s / s.shift(self.lookback_days) - 1.0)
        df["turnover"] = df["close"] * df["volume"]
        df["adv"] = g["turnover"].transform(lambda s: s.rolling(self.shock_lookback_days).mean())
        df["adv_prior"] = g["turnover"].transform(
            lambda s: s.shift(1).rolling(self.shock_lookback_days).mean()
        )
        df["tshock"] = df["turnover"] / df["adv_prior"]

        # The session-date cross-section.
        today = df[df["date"].dt.normalize() == sd].copy()
        today = today[
            today["symbol"].isin(mtf_eligible)
            & today["trail_ret"].notna()
            & today["tshock"].notna()
            & (today["close"] >= self.min_price)
            & (today["adv"] >= self.adv_floor_inr)
        ]
        if len(today) < self.min_universe:
            logger.info(
                "x_sectional_ranker: %s only %d qualifying symbols (< min %d) -> no basket",
                sd.date(), len(today), self.min_universe,
            )
            return []

        # Cross-sectional tier + loser rank.
        try:
            today["adv_tier"] = pd.qcut(
                today["adv"], self.adv_tier_count,
                labels=list(range(1, self.adv_tier_count + 1)), duplicates="drop",
            ).astype("Int64")
        except ValueError:
            logger.warning("x_sectional_ranker: %s qcut failed (degenerate adv); no basket", sd.date())
            return []
        today["rank_pct"] = today["trail_ret"].rank(pct=True)

        sel = today[
            (today["rank_pct"] <= self.loser_pct)
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
                "trail_ret": float(r["trail_ret"]),
                "tshock": float(r["tshock"]),
                "adv_tier": int(r["adv_tier"]),
                "rank_pct": float(r["rank_pct"]),
                "close": float(r["close"]),
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
