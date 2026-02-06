# services/gates/directional_bias.py
"""
Directional Bias Tracker - Nifty Green/Red Position Size Modulation

Tracks whether Nifty is green (above prev close) or red (below prev close)
and returns a position size multiplier based on trade side alignment.

V2: Hysteresis thresholds + category-specific multipliers.
  - entry_threshold (0.3%): How far Nifty must move to establish bias
  - exit_threshold (0.1%): How far Nifty must fall back to clear bias
  - Prevents flip-flopping on sub-0.3% oscillations
  - Category overrides: breakout/momentum get aggressive treatment,
    reversion barely affected (mean reversion trades are the point)

Evidence (3-year backtest, 3,493 trades):
  - WITH trend:    63.9% WR, Rs 656 avg PnL
  - AGAINST trend: 56.7% WR, Rs 469 avg PnL
  - Edge: +7.2pp win rate, scales to +11.3pp on >1% Nifty days
  - First hour has strongest edge (15.4pp vs 7pp midday)
  - Entry-only: existing positions never touched
  - Chop detection: 4+ flips = neutral (no bias)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from config.logging_config import get_agent_logger

logger = get_agent_logger()

INDEX_OHLCV_DIR = Path(__file__).resolve().parents[2] / "backtest-cache-download" / "index_ohlcv"


class DirectionalBiasTracker:
    """
    Tracks Nifty direction (green/red/neutral/chop) and returns
    a position size multiplier based on trade side alignment.

    Uses hysteresis thresholds to prevent flip-flopping:
      - entry_threshold: How far Nifty must move from prev_close to establish bias
      - exit_threshold:  How far Nifty must fall back to clear an established bias

    States:
      green   - current > prev_close + entry_threshold  (Nifty is up)
      red     - current < prev_close - entry_threshold  (Nifty is down)
      neutral - within threshold band, or cleared via exit_threshold
      chop    - direction flipped > max_flips times today
    """

    def __init__(self, cfg: dict):
        db_cfg = cfg["directional_bias"]
        self.enabled: bool = db_cfg["enabled"]
        self._index_symbol: str = db_cfg["index_symbol"]
        self.entry_threshold_pct: float = db_cfg["entry_threshold_pct"]
        self.exit_threshold_pct: float = db_cfg["exit_threshold_pct"]
        self.with_mult: float = db_cfg["with_trend_mult"]
        self.against_mult: float = db_cfg["against_trend_mult"]
        self.chop_max_flips: int = db_cfg["chop_max_flips"]
        self._category_overrides: dict = db_cfg.get("category_overrides", {})

        self.prev_close: Optional[float] = None
        self.current_price: Optional[float] = None
        self.direction: str = "neutral"
        self._prev_direction: Optional[str] = None
        self._flip_count: int = 0
        self._change_pct: float = 0.0

        # Cache for backtest intraday 5m bars (Nifty close at each 5m bar)
        self._intraday_5m: Optional[pd.Series] = None

        if self.enabled:
            logger.info(
                f"DIR_BIAS | Initialized: entry_threshold={self.entry_threshold_pct}%, "
                f"exit_threshold={self.exit_threshold_pct}%, "
                f"with={self.with_mult}x, against={self.against_mult}x, "
                f"chop_flips={self.chop_max_flips}, "
                f"category_overrides={list(self._category_overrides.keys())}"
            )

    # ------------------------------------------------------------------ #
    #  Session lifecycle                                                   #
    # ------------------------------------------------------------------ #

    def set_prev_close(self, price: float) -> None:
        """Called once at session start with previous day's Nifty close."""
        self.prev_close = price
        logger.info(f"DIR_BIAS | prev_close set to {price:.2f}")

    def set_prev_close_for_date(self, session_date) -> bool:
        """
        Set prev_close from the 1m feather for backtest mode.
        Derives daily close from the same intraday data used for price tracking.
        No separate CSV needed — single data source for both daily and intraday.
        """
        # Ensure intraday cache is loaded (same feather used for price tracking)
        if self._intraday_5m is None:
            self._intraday_5m = self._load_intraday_5m()
        if self._intraday_5m is None or self._intraday_5m.empty:
            return False

        session_dt = pd.Timestamp(session_date).normalize()
        # Get last close from the previous trading day
        prior = self._intraday_5m[self._intraday_5m.index < session_dt]
        if prior.empty:
            return False

        self.prev_close = float(prior.iloc[-1])
        prev_date = prior.index[-1].date()
        logger.info(
            f"DIR_BIAS | Backtest prev_close for {session_dt.date()}: "
            f"{self.prev_close:.2f} (from {prev_date})"
        )
        return True

    def get_backtest_price_at(self, now: pd.Timestamp) -> Optional[float]:
        """
        Get Nifty close price at `now` from cached 5m intraday bars.

        Loads the 1m feather once, resamples to 5m, and returns the latest
        5m close <= now. This matches live behavior where direction updates
        on every 5m scan.
        """
        if self._intraday_5m is None:
            self._intraday_5m = self._load_intraday_5m()
        if self._intraday_5m is None or self._intraday_5m.empty:
            return None

        # Find latest 5m bar close <= now
        valid = self._intraday_5m[self._intraday_5m.index <= now]
        if valid.empty:
            logger.info(
                f"DIR_BIAS | get_backtest_price_at: no bars <= {now} "
                f"(cache range: {self._intraday_5m.index[0]} to {self._intraday_5m.index[-1]})"
            )
            return None
        price = float(valid.iloc[-1])
        if self.current_price is None:  # First call — log for diagnostics
            logger.info(f"DIR_BIAS | First backtest price at {now}: {price:.2f} (prev_close={self.prev_close:.2f})")
        return price

    def reset_session(self) -> None:
        """Reset for new trading day."""
        self.prev_close = None
        self.current_price = None
        self.direction = "neutral"
        self._prev_direction = None
        self._flip_count = 0
        self._change_pct = 0.0

    # ------------------------------------------------------------------ #
    #  Price updates                                                       #
    # ------------------------------------------------------------------ #

    def update_price(self, price: float) -> None:
        """Called on each index 5m bar close (live/paper) or per-bar (backtest).

        Uses hysteresis to prevent flip-flopping:
          - To establish green: change_pct must exceed +entry_threshold_pct
          - To clear green back to neutral: change_pct must drop below +exit_threshold_pct
          - Symmetric for red (negative thresholds)
        """
        if not self.enabled or self.prev_close is None or self.prev_close <= 0:
            return

        self.current_price = price
        self._change_pct = (price - self.prev_close) / self.prev_close * 100

        # Classify direction with hysteresis
        new_dir = self._classify_direction_hysteresis()

        # Track flips (only between green and red, not neutral transitions)
        if (self._prev_direction in ("green", "red")
                and new_dir in ("green", "red")
                and new_dir != self._prev_direction):
            self._flip_count += 1
            logger.info(
                f"DIR_BIAS | FLIP #{self._flip_count}: "
                f"{self._prev_direction} -> {new_dir} | "
                f"Nifty change={self._change_pct:+.2f}%"
            )

        # Log direction established (neutral/chop → green/red)
        if (new_dir in ("green", "red")
                and self.direction not in ("green", "red")):
            logger.info(
                f"DIR_BIAS | Direction ESTABLISHED: {new_dir} | "
                f"Nifty change={self._change_pct:+.2f}% "
                f"(crossed {self.entry_threshold_pct}% entry threshold)"
            )

        # Log direction cleared (green/red → neutral)
        if (new_dir == "neutral"
                and self.direction in ("green", "red")):
            logger.info(
                f"DIR_BIAS | Direction CLEARED: {self.direction} -> neutral | "
                f"Nifty change={self._change_pct:+.2f}% "
                f"(dropped below {self.exit_threshold_pct}% exit threshold)"
            )

        if new_dir in ("green", "red"):
            self._prev_direction = new_dir

        # Chop: too many flips today
        if self._flip_count >= self.chop_max_flips:
            self.direction = "chop"
        else:
            self.direction = new_dir

    def _classify_direction_hysteresis(self) -> str:
        """Apply hysteresis to classify Nifty direction.

        State transitions:
          neutral -> green:  change_pct > +entry_threshold
          neutral -> red:    change_pct < -entry_threshold
          green -> neutral:  change_pct < +exit_threshold
          red -> neutral:    change_pct > -exit_threshold
          green -> red:      change_pct < -entry_threshold  (cross through)
          red -> green:      change_pct > +entry_threshold  (cross through)
        """
        # Use underlying direction even if chop is set (for state continuity)
        current = self.direction if self.direction != "chop" else (self._prev_direction or "neutral")

        if current == "green":
            if self._change_pct < -self.entry_threshold_pct:
                return "red"        # Crossed all the way to red
            elif self._change_pct < self.exit_threshold_pct:
                return "neutral"    # Dropped below exit band
            return "green"          # Still in green zone

        elif current == "red":
            if self._change_pct > self.entry_threshold_pct:
                return "green"      # Crossed all the way to green
            elif self._change_pct > -self.exit_threshold_pct:
                return "neutral"    # Rose above exit band
            return "red"            # Still in red zone

        else:  # neutral
            if self._change_pct > self.entry_threshold_pct:
                return "green"
            elif self._change_pct < -self.entry_threshold_pct:
                return "red"
            return "neutral"

    # ------------------------------------------------------------------ #
    #  Multiplier query                                                    #
    # ------------------------------------------------------------------ #

    def get_size_mult(self, side: str, category: str = None) -> Tuple[float, str]:
        """
        Get position size multiplier based on trade side vs market direction.

        Uses category-specific overrides when available:
          BREAKOUT/MOMENTUM: Aggressive (1.25x with / 0.65x against)
          REVERSION: Mild (1.05x with / 0.95x against)
          LEVEL: Moderate (1.15x with / 0.85x against)
        Falls back to global defaults if no category override.

        Args:
            side: "BUY"/"SELL" or "long"/"short"
            category: Optional trade category ("BREAKOUT", "MOMENTUM", etc.)

        Returns:
            (multiplier, reason_string)
        """
        if not self.enabled or self.direction in ("neutral", "chop") or self.prev_close is None:
            return 1.0, f"dir_bias:{self.direction}"

        # Resolve multipliers: category-specific or global
        cat_upper = category.upper() if category else None
        if cat_upper and cat_upper in self._category_overrides:
            override = self._category_overrides[cat_upper]
            w_mult = override["with_trend_mult"]
            a_mult = override["against_trend_mult"]
        else:
            w_mult = self.with_mult
            a_mult = self.against_mult

        side_upper = side.upper()
        is_with = (
            (side_upper in ("BUY", "LONG") and self.direction == "green")
            or (side_upper in ("SELL", "SHORT") and self.direction == "red")
        )

        cat_tag = f"_{cat_upper}" if cat_upper else ""
        if is_with:
            mult, reason = w_mult, f"dir_bias:with_{self.direction}{cat_tag}"
        else:
            mult, reason = a_mult, f"dir_bias:against_{self.direction}{cat_tag}"

        logger.info(
            f"DIR_BIAS | APPLIED {mult}x ({reason}) for {side_upper} | "
            f"Nifty {self.direction} ({self._change_pct:+.2f}%)"
        )
        return mult, reason

    # ------------------------------------------------------------------ #
    #  Status / debugging                                                  #
    # ------------------------------------------------------------------ #

    def classify_alignment(self, side: str) -> str:
        """
        Classify whether a trade side aligns with current Nifty direction.

        Returns: "with", "against", or "neutral"
        """
        if not self.enabled or self.direction in ("neutral", "chop") or self.prev_close is None:
            return "neutral"

        side_upper = side.upper()
        is_with = (
            (side_upper in ("BUY", "LONG") and self.direction == "green")
            or (side_upper in ("SELL", "SHORT") and self.direction == "red")
        )
        return "with" if is_with else "against"

    def format_status(self) -> str:
        """Format current state for logging / API."""
        if not self.enabled:
            return "DIR_BIAS:disabled"
        return (
            f"DIR_BIAS:{self.direction} "
            f"({self._change_pct:+.2f}%) "
            f"flips={self._flip_count}"
        )

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _load_intraday_5m(self) -> Optional[pd.Series]:
        """
        Load Nifty 1m feather from backtest cache, resample to 5m closes.

        Returns a Series indexed by 5m bar timestamp with close prices.
        """
        index_sym = self._index_symbol
        # NSE:NIFTY 50 → NSE_NIFTY_50
        safe_name = index_sym.replace(":", "_").replace(" ", "_")
        feather_path = INDEX_OHLCV_DIR / safe_name / f"{safe_name}_1minutes.feather"

        if not feather_path.exists():
            logger.warning(f"DIR_BIAS | Intraday feather not found: {feather_path}")
            return None

        try:
            df = pd.read_feather(feather_path)
            # Normalize date column to IST-naive
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                if df["date"].dt.tz is not None:
                    df["date"] = df["date"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
                df = df.set_index("date").sort_index()
            # Resample 1m → 5m (close only)
            close_5m = df["close"].resample("5min").last().dropna()
            logger.info(f"DIR_BIAS | Loaded intraday 5m cache: {len(close_5m)} bars from {feather_path.name}")
            return close_5m
        except Exception as e:
            logger.error(f"DIR_BIAS | Failed to load intraday feather: {e}")
            return None


# ------------------------------------------------------------------ #
#  Module-level singleton (same pattern as get_orchestrator)           #
# ------------------------------------------------------------------ #

_instance: Optional[DirectionalBiasTracker] = None


def set_tracker(tracker: DirectionalBiasTracker) -> None:
    """Register the singleton tracker instance (called from screener_live)."""
    global _instance
    _instance = tracker


def get_tracker() -> Optional[DirectionalBiasTracker]:
    """Get the singleton tracker instance (called from base_pipeline)."""
    return _instance
