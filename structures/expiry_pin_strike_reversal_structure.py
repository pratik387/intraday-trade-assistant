"""Expiry-Pin Strike Reversal detector — sub-project #8 (Phase 0 2026-04-29).

Calendar+OI driven pin-magnet reversal per
specs/2026-04-29-research-new-indian-setup-candidates.md (§ Candidate 8).

Indian sources cited:
  - Wright Research — "expiry pinning is real on NSE; market-makers reduce
    gamma exposure by hedging the highest-OI strike, which magnets spot toward
    that strike intraday on expiry sessions"
  - ICFM Indore — Indian options trading academy on weekly-expiry pin strikes
  - OptionX — retail Indian options analytics on max-OI / max-pain
  - PL Capital — institutional desk research on expiry-day gamma effects

Mechanic (short; mirror long):
  On Indian F&O weekly+monthly expiry sessions (NIFTY weekly Thursday pre-
  2025-09 / Tuesday post-2025-09; monthly = last weekly of month):
    Identify pin_strike = argmax(CE_OI + PE_OI) on the relevant NIFTY expiry,
    looked up via services.option_chain_loader.find_max_oi_strike (settlement
    OI from prior session).
    After 13:30 IST (market-makers actively hedge during the post-noon decay
    window), if NIFTY spot is ≥0.3% ABOVE pin_strike (room for the magnet
    pull), and RSI(14) on the symbol's 5m chart shows an overbought→neutral
    decay (prior_rsi > 70 AND current_rsi <= 70), fire SHORT toward the
    constituent's price implied by spot dropping to pin_strike.
  Mirror long: spot ≥0.3% BELOW pin → RSI oversold→neutral decay
  (prior < 30 AND current >= 30) → fire LONG.
  Stop: 0.5 * ATR (5m) beyond entry. Tiered T1 50% at half-distance to pin;
  T2 100% at the pin-implied price.
  First-trigger latch keyed on (symbol, side, session_date_iso).

Universe:
  Top-10 NIFTY heavyweights only (HDFCBANK, RELIANCE, ICICIBANK, INFY, TCS,
  BHARTIARTL, LT, ITC, AXISBANK, KOTAKBANK). Pin gravity is mediated through
  index-basket constituents; random F&O symbols don't share the gamma pull.
  Loaded once at __init__ from assets/nifty_heavyweights.csv.

Wide-open mode bypass:
  - Bypassed (design-inferred): RSI-decay confirmation
  - Always enforced (mechanical / mandatory): expiry-day check, active window
    13:30-15:15, min spot distance 0.3%, heavyweight universe, min_bars
"""
from __future__ import annotations

from datetime import time
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import pandas as pd

from config.logging_config import get_agent_logger
from .base_structure import BaseStructure
from .data_models import (
    ExitLevels,
    MarketContext,
    RiskParams,
    StructureAnalysis,
    StructureEvent,
    TradePlan,
)

logger = get_agent_logger()


def _is_wide_open() -> bool:
    """Read top-level wide_open_mode flag from base config."""
    try:
        from pipelines.base_pipeline import load_base_config
        return bool(load_base_config().get("wide_open_mode", False))
    except Exception:
        return False


def _compute_rsi(closes: pd.Series, period: int) -> pd.Series:
    """Wilder-smoothed RSI(period). Returns NaN until enough bars."""
    delta = closes.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, pd.NA)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _load_heavyweights(csv_path: Path) -> Dict[str, float]:
    """Load NIFTY heavyweights CSV → {symbol: weight_pct} with NSE: prefix.

    The CSV has columns symbol,weight_pct (no exchange prefix). We prefix
    "NSE:" to match the project's symbol format used everywhere else.
    """
    df = pd.read_csv(csv_path)
    out: Dict[str, float] = {}
    for _, row in df.iterrows():
        sym = str(row["symbol"]).strip()
        if not sym:
            continue
        if not sym.startswith("NSE:"):
            sym = f"NSE:{sym}"
        out[sym] = float(row["weight_pct"])
    return out


class ExpiryPinStrikeReversalStructure(BaseStructure):
    """Calendar+OI pin-magnet mean-revert on NIFTY heavyweights.

    Fires only on NSE F&O expiry sessions, only after 13:30 IST, only when the
    NIFTY spot is far enough from the highest-OI strike to allow a meaningful
    magnet pull. Side selected by spot-vs-pin direction, gated on an RSI
    decay (overbought→neutral for short; oversold→neutral for long).

    OI lookups go through an injectable `oi_loader` so unit tests can stub the
    parquet reads and the expiry calendar.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        oi_loader: Optional[Any] = None,
    ):
        super().__init__(config)
        self.structure_type = "expiry_pin_strike_reversal"
        self.configured_setup_type = config.get("_setup_name")

        # Per CLAUDE.md rule 1: every parameter from config (KeyError on missing).
        self.active_start = self._parse_time(config["active_window_start"])
        self.active_end = self._parse_time(config["active_window_end"])
        self.min_spot_dist_pct = float(config["min_spot_distance_to_pin_pct"]) / 100.0
        self.pin_index = str(config["pin_index"])
        self.expiry_types: Set[str] = set(config["expiry_types"])
        self.heavyweights_csv = str(config["nifty_heavyweights_csv"])
        self.rsi_period = int(config["rsi_period"])
        self.rsi_overbought = float(config["rsi_overbought"])
        self.rsi_oversold = float(config["rsi_oversold"])
        self.stop_atr_mult = float(config["stop_atr_multiplier"])
        self.t1_qty_pct = float(config["t1_qty_pct"])
        self.t1_target_frac = float(config["t1_target_frac"])
        self.allowed_sides: Set[str] = set(config["allowed_sides"])
        self.allowed_caps: Set[str] = set(config["allowed_cap_segments"])
        self.min_bars_required = int(config["min_bars_required"])
        self.min_stop_distance_pct = float(config["min_stop_distance_pct"]) / 100.0

        # Inject oi_loader for test stubbing; default to the real module.
        if oi_loader is None:
            from services import option_chain_loader as _ocl
            oi_loader = _ocl
        self.oi_loader = oi_loader

        # Resolve heavyweights path relative to the repo root.
        repo_root = Path(__file__).resolve().parents[1]
        csv_path = (repo_root / self.heavyweights_csv).resolve()
        self._heavyweights: Dict[str, float] = _load_heavyweights(csv_path)

        # First-trigger latch keyed by (symbol, side, session_date_iso).
        self._fired_today: Set[Tuple[str, str, str]] = set()
        self._latch_session_date: Optional[str] = None

    @staticmethod
    def _parse_time(s: str) -> time:
        h, m = s.split(":")
        return time(int(h), int(m))

    def _get_atr(self, ctx: MarketContext) -> float:
        if ctx.indicators and "atr" in ctx.indicators:
            try:
                v = float(ctx.indicators["atr"])
                if pd.notna(v) and v > 0:
                    return v
            except (TypeError, ValueError):
                pass
        if ctx.df_5m is not None and len(ctx.df_5m) >= 14:
            df = ctx.df_5m
            return float((df["high"] - df["low"]).tail(14).mean())
        return ctx.current_price * 0.01

    def _get_rsi_pair(self, ctx: MarketContext) -> Optional[Tuple[float, float]]:
        """Return (prior_rsi, current_rsi) on the 5m chart, or None if NA."""
        if ctx.indicators and "rsi" in ctx.indicators and "rsi_prior" in ctx.indicators:
            try:
                cur = float(ctx.indicators["rsi"])
                prior = float(ctx.indicators["rsi_prior"])
                if pd.notna(cur) and pd.notna(prior):
                    return prior, cur
            except (TypeError, ValueError):
                pass
        if ctx.df_5m is None or len(ctx.df_5m) < self.rsi_period + 2:
            return None
        if "rsi" in ctx.df_5m.columns:
            ser = ctx.df_5m["rsi"].dropna()
            if len(ser) < 2:
                return None
            return float(ser.iloc[-2]), float(ser.iloc[-1])
        rsi_ser = _compute_rsi(ctx.df_5m["close"], self.rsi_period).dropna()
        if len(rsi_ser) < 2:
            return None
        return float(rsi_ser.iloc[-2]), float(rsi_ser.iloc[-1])

    def _get_nifty_spot(self, ctx: MarketContext) -> Optional[float]:
        """NIFTY spot read from indicators dict (set by orchestrator at runtime)."""
        if ctx.indicators and "nifty_spot" in ctx.indicators:
            try:
                v = float(ctx.indicators["nifty_spot"])
                if pd.notna(v) and v > 0:
                    return v
            except (TypeError, ValueError):
                pass
        return None

    def _resolve_pin_strike(
        self,
        session_date,
    ) -> Optional[float]:
        """Look up the pin strike via the OI loader.

        Tries each expiry mode in `self.expiry_types` in order; returns the
        first non-None result. None on any miss (so detect can reject cleanly).
        """
        for expiry_mode in ("weekly", "monthly"):
            if expiry_mode not in self.expiry_types:
                continue
            try:
                strike = self.oi_loader.find_max_oi_strike(
                    session_date, symbol=self.pin_index, expiry=expiry_mode,
                )
                return float(strike)
            except FileNotFoundError:
                # Snapshot missing → skip mode; might succeed under another
                continue
            except ValueError:
                # No contracts at the chosen expiry; try next mode
                continue
        return None

    def _maybe_reset_state(self, session_date_iso: str) -> None:
        if session_date_iso != self._latch_session_date:
            self._fired_today.clear()
            self._latch_session_date = session_date_iso

    def detect(self, ctx: MarketContext) -> StructureAnalysis:
        def _empty(reason: str = "") -> StructureAnalysis:
            return StructureAnalysis(
                structure_detected=False,
                events=[],
                quality_score=0.0,
                rejection_reason=reason or None,
            )

        _wide_open = _is_wide_open()

        # ---- Heavyweight universe (mandatory; not bypassed by wide_open) ----
        if ctx.symbol not in self._heavyweights:
            return _empty(f"symbol {ctx.symbol} not in NIFTY heavyweights")

        # ---- Cap segment guard (design-inferred; bypassed under wide_open) ----
        if not _wide_open and ctx.cap_segment not in self.allowed_caps:
            return _empty(f"cap_segment {ctx.cap_segment!r} not in allowed set")

        df = ctx.df_5m
        if df is None or len(df) < self.min_bars_required:
            return _empty("Insufficient bars")

        last_ts = df.index[-1]
        cur_t = last_ts.time() if hasattr(last_ts, "time") else last_ts
        if not (self.active_start <= cur_t <= self.active_end):
            return _empty(f"Outside active window: {cur_t}")

        # ---- Expiry-day check (mandatory; not bypassed by wide_open) ----
        # Calendar gate: per the strategy thesis, the gamma pin-magnet effect
        # only exists on F&O expiry sessions. Wide-open MUST NOT bypass this.
        if not self.oi_loader.is_expiry_day(ctx.session_date):
            return _empty("not an F&O expiry day")

        # ---- Pin-strike lookup ----
        pin_strike = self._resolve_pin_strike(ctx.session_date)
        if pin_strike is None or pin_strike <= 0:
            return _empty("pin strike unavailable from OI snapshot")

        # ---- NIFTY spot distance to pin ----
        nifty_spot = self._get_nifty_spot(ctx)
        if nifty_spot is None:
            return _empty("NIFTY spot unavailable")
        spot_distance = (nifty_spot - pin_strike) / pin_strike   # signed

        # ---- Min distance gate (mandatory) ----
        if abs(spot_distance) < self.min_spot_dist_pct:
            return _empty(
                f"spot too close to pin: |{spot_distance*100:.3f}%| < "
                f"{self.min_spot_dist_pct*100:.2f}%"
            )

        # ---- Side selection ----
        side = "short" if spot_distance > 0 else "long"
        if side not in self.allowed_sides:
            return _empty(f"side {side!r} not in allowed_sides")

        # ---- Latch reset + check ----
        session_date_iso = pd.Timestamp(ctx.session_date).strftime("%Y-%m-%d")
        self._maybe_reset_state(session_date_iso)
        latch_key = (ctx.symbol, side, session_date_iso)
        if latch_key in self._fired_today:
            return _empty("already fired this side today (latch)")

        # ---- RSI decay confirmation (design-inferred; bypassed under wide_open) ----
        if not _wide_open:
            rsi_pair = self._get_rsi_pair(ctx)
            if rsi_pair is None:
                return _empty("RSI(14) unavailable")
            prior_rsi, current_rsi = rsi_pair
            if side == "short":
                # Overbought decay: prior > overbought AND current <= overbought
                if not (prior_rsi > self.rsi_overbought
                        and current_rsi <= self.rsi_overbought):
                    return _empty(
                        f"RSI no overbought decay (prior={prior_rsi:.1f}, "
                        f"current={current_rsi:.1f})"
                    )
            else:   # long
                # Oversold decay: prior < oversold AND current >= oversold
                if not (prior_rsi < self.rsi_oversold
                        and current_rsi >= self.rsi_oversold):
                    return _empty(
                        f"RSI no oversold decay (prior={prior_rsi:.1f}, "
                        f"current={current_rsi:.1f})"
                    )

        # ---- Build event + latch ----
        self._fired_today.add(latch_key)
        evt = self._build_event(
            ctx, side, pin_strike, nifty_spot, spot_distance, session_date_iso,
        )
        return StructureAnalysis(
            structure_detected=True,
            events=[evt],
            quality_score=evt.confidence * 100.0,
        )

    def _build_event(
        self,
        ctx: MarketContext,
        side: str,
        pin_strike: float,
        nifty_spot: float,
        spot_distance: float,
        session_date_iso: str,
    ) -> StructureEvent:
        last_ts = ctx.df_5m.index[-1]
        atr = self._get_atr(ctx)
        close = float(ctx.df_5m["close"].iloc[-1])
        # Implied target = constituent price scaled by NIFTY's projected move
        # to pin. spot_distance is (NIFTY - pin) / pin; so the constituent's
        # implied % move toward pin equals -spot_distance applied to its
        # current price.
        target_constituent_price = close * (1.0 - spot_distance)

        # Confidence proxy: |spot_distance| as a fraction of the min gate, capped.
        # 1.0 means we're 1× the min, scaling up linearly to ~3×.
        confidence = min(1.0, max(0.0, abs(spot_distance) / max(self.min_spot_dist_pct * 3.0, 1e-6)))

        return StructureEvent(
            symbol=ctx.symbol,
            timestamp=last_ts,
            structure_type=self.structure_type,
            side=side,
            confidence=confidence,
            levels={
                "pin_strike": float(pin_strike),
                "nifty_spot": float(nifty_spot),
                "spot_distance_pct": float(spot_distance * 100.0),
                "target_constituent_price": float(target_constituent_price),
                "close": float(close),
            },
            context={
                "session_date_iso": session_date_iso,
                "atr": atr,
                "pin_index": self.pin_index,
            },
            price=close,
        )

    def _build_plan(self, ctx: MarketContext, side: str) -> Optional[TradePlan]:
        analysis = self.detect(ctx)
        if not analysis.structure_detected or not analysis.events:
            return None
        evt = analysis.events[0]
        if evt.side != side:
            return None

        levels = evt.levels
        target_price = float(levels["target_constituent_price"])
        close = float(ctx.df_5m["close"].iloc[-1])
        atr = self._get_atr(ctx)

        if side == "long":
            hard_sl = close - atr * self.stop_atr_mult
            risk = max(close - hard_sl, atr * 0.1)
            min_risk = close * self.min_stop_distance_pct
            if risk < min_risk:
                risk = min_risk
                hard_sl = close - risk
            # T1 partial at fraction of the way to target; T2 at target.
            if target_price > close:
                t2_level = target_price
                t1_level = close + (target_price - close) * self.t1_target_frac
            else:
                # Target below entry (bad direction) → fall back to risk-multiple
                t2_level = close + 2.0 * risk
                t1_level = close + risk
        else:   # short
            hard_sl = close + atr * self.stop_atr_mult
            risk = max(hard_sl - close, atr * 0.1)
            min_risk = close * self.min_stop_distance_pct
            if risk < min_risk:
                risk = min_risk
                hard_sl = close + risk
            if target_price < close:
                t2_level = target_price
                t1_level = close - (close - target_price) * self.t1_target_frac
            else:
                t2_level = close - 2.0 * risk
                t1_level = close - risk

        rr_t1 = abs(close - t1_level) / max(risk, 1e-6)
        rr_t2 = abs(close - t2_level) / max(risk, 1e-6)
        targets = [
            {
                "name": "T1", "level": t1_level, "rr": rr_t1,
                "qty_pct": self.t1_qty_pct, "action": "partial_exit",
            },
            {
                "name": "T2", "level": t2_level, "rr": rr_t2,
                "qty_pct": round(1.0 - self.t1_qty_pct, 4), "action": "exit_full",
            },
        ]
        risk_params = RiskParams(hard_sl=hard_sl, risk_per_share=risk, atr=atr)
        exit_levels = ExitLevels(hard_sl=hard_sl, targets=targets)
        return TradePlan(
            symbol=ctx.symbol,
            side=side,
            structure_type=evt.structure_type,
            entry_price=close,
            risk_params=risk_params,
            exit_levels=exit_levels,
            qty=0,
            notional=0.0,
            confidence=evt.confidence,
            notes=evt.context,
            trade_id=evt.trade_id,
        )

    def plan_long_strategy(
        self,
        ctx: MarketContext,
        event: Optional[StructureEvent] = None,
    ) -> Optional[TradePlan]:
        if "long" not in self.allowed_sides:
            return None
        return self._build_plan(ctx, "long")

    def plan_short_strategy(
        self,
        ctx: MarketContext,
        event: Optional[StructureEvent] = None,
    ) -> Optional[TradePlan]:
        if "short" not in self.allowed_sides:
            return None
        return self._build_plan(ctx, "short")

    def calculate_risk_params(
        self,
        entry_price: float,
        market_context: MarketContext,
    ) -> RiskParams:
        """Placeholder using ATR-based stop (real risk computed inside
        _build_plan from spot-distance-implied target + ATR stop)."""
        atr = self._get_atr(market_context)
        stop_distance = max(atr * self.stop_atr_mult,
                            entry_price * self.min_stop_distance_pct)
        return RiskParams(
            hard_sl=entry_price - stop_distance,
            risk_per_share=stop_distance,
            atr=atr,
        )

    def get_exit_levels(self, trade_plan: TradePlan) -> ExitLevels:
        return trade_plan.exit_levels

    def rank_setup_quality(
        self,
        ctx: MarketContext,
        event: Optional[StructureEvent] = None,
    ) -> float:
        return self.detect(ctx).quality_score

    def validate_timing(self, current_time: pd.Timestamp) -> bool:
        t = current_time.time() if hasattr(current_time, "time") else current_time
        return self.active_start <= t <= self.active_end
