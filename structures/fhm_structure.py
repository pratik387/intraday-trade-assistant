# structures/fhm_structure.py
"""
First Hour Momentum (FHM) Structure Implementation.

PRO TRADER RULES (from Warrior Trading + NSE India research):
1. DETECTION: Opening drive detected (RVOL >= 2x + price move >= 1.5%)
2. ENTRY: DON'T chase the spike - WAIT for pullback to VWAP
3. ENTRY ZONE: VWAP +/- 0.15 ATR (tight zone around VWAP)
4. STOP: Just below VWAP for longs (above for shorts) - not arbitrary ATR
5. TARGETS: Handled by pipeline's calculate_targets() - R:R based from config

Key conditions:
- Time window: 09:15 - 10:30 (first 75 min including pullback time)
- RVOL >= 2.0x (unusual institutional activity)
- Price move >= 1.5% from open (opening drive detected)
- Volume >= 100k (sufficient liquidity)
- VWAP position confirms direction (long above VWAP, short below VWAP)

Structure's job: DETECT and provide essential levels (vwap, entry_price, stop_loss)
Pipeline's job: Calculate targets, ATR already available in features dict

Sources:
- Warrior Trading: "Instead of trying to pursue the first spike, professionals seek conviction"
- VWAP Pullback Entry: "Wait for pullback to VWAP, enter on retest"
- NSE India: "09:15-10:30 IST = peak momentum hours"
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import pandas as pd
from datetime import time

from config.logging_config import get_agent_logger
from .base_structure import BaseStructure
from .data_models import (
    StructureEvent,
    TradePlan,
    RiskParams,
    ExitLevels,
    MarketContext,
    StructureAnalysis
)

logger = get_agent_logger()


class FHMStructure(BaseStructure):
    """
    First Hour Momentum structure implementation.

    Detects high-momentum setups in the first hour of trading based on
    RVOL, price movement from open, and VWAP position confirmation.

    NOTE: This structure only handles DETECTION. ATR and targets are
    calculated by the pipeline (base_pipeline.py handles ATR calculation,
    momentum_pipeline.py handles target calculation).
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize FHM structure with configuration."""
        super().__init__(config)

        # Time window config - KeyError if missing (no defaults)
        time_window = config["time_window"]
        self.start_time = self._parse_time(time_window["start"])
        self.end_time = self._parse_time(time_window["end"])

        # Trigger thresholds - KeyError if missing (no defaults)
        triggers = config["triggers"]
        self.min_rvol = triggers["min_rvol"]
        self.min_price_move_pct = triggers["min_price_move_pct"]
        self.min_volume = triggers["min_volume"]

        # VWAP position config - KeyError if missing (no defaults)
        vwap_cfg = config["vwap_position"]
        self.long_requires_above_vwap = vwap_cfg["long_requires_above_vwap"]
        self.short_requires_below_vwap = vwap_cfg["short_requires_below_vwap"]

        # PRO TRADER ENTRY CONFIG - VWAP pullback entry
        entry_cfg = config["entry"]
        self.entry_mode = entry_cfg["mode"]  # "pullback"
        self.entry_reference = entry_cfg["reference"]  # "vwap"
        self.entry_zone_atr_mult = entry_cfg["zone_atr_mult"]  # 0.15

        # PRO TRADER STOP LOSS CONFIG - VWAP-based stops
        # NOTE: Current stops are ~1-1.5 ATR from VWAP (wider than PRO TRADER's 0.2 ATR rule).
        # This gives more room to avoid noise-induced stops. If 6-month backtest shows
        # poor results (e.g., too many full SL hits), consider tightening to 0.2-0.5 ATR.
        sl_cfg = config["stop_loss"]
        self.stop_reference = sl_cfg["reference"]  # "vwap"
        self.stop_buffer_atr_mult = sl_cfg["buffer_atr_mult"]  # 0.2

        # Confidence levels - KeyError if missing (no defaults)
        self.confidence_high_rvol = config["confidence_high_rvol"]
        self.confidence_base = config["confidence_base"]
        self.high_rvol_threshold = config["high_rvol_threshold"]

        # Minimum bars required
        self.min_bars_required = config["min_bars_required"]

        logger.debug(f"FHM: Initialized - Time: {self.start_time}-{self.end_time}, "
                    f"RVOL>={self.min_rvol}x, Move>={self.min_price_move_pct}%, "
                    f"Entry={self.entry_mode}@{self.entry_reference}")

    def _parse_time(self, time_str: str) -> time:
        """Parse time string HH:MM to time object."""
        parts = time_str.split(":")
        return time(int(parts[0]), int(parts[1]))

    def detect(self, market_context: MarketContext) -> StructureAnalysis:
        """
        Detect FHM patterns in market data.

        Args:
            market_context: Market data and context

        Returns:
            StructureAnalysis with detected FHM events
        """
        symbol = market_context.symbol

        try:
            df = market_context.df_5m

            if df is None or len(df) < self.min_bars_required:
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason=f"Insufficient data"
                )

            events = []
            current_time = market_context.timestamp
            current_price = market_context.current_price

            # Check time window
            current_time_of_day = current_time.time()
            in_window = self.start_time <= current_time_of_day <= self.end_time

            if not in_window:
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason=f"Outside FHM time window ({self.start_time}-{self.end_time})"
                )

            # Calculate RVOL
            rvol = self._calculate_rvol(df)

            # Calculate price move from open
            open_price = df['open'].iloc[0]
            price_move_pct = ((current_price - open_price) / open_price) * 100
            abs_price_move = abs(price_move_pct)

            # Check volume
            current_volume = df['volume'].iloc[-1]

            # Log near-misses at INFO level (symbols with at least one strong signal)
            is_near_miss = (rvol >= self.min_rvol * 0.8 or abs_price_move >= self.min_price_move_pct * 0.8)

            if rvol < self.min_rvol:
                if is_near_miss:
                    logger.debug(f"FHM_REJECT: {symbol} | RVOL={rvol:.2f}x < {self.min_rvol}x | "
                                f"Move={abs_price_move:.2f}% | Vol={current_volume:.0f}")
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason=f"RVOL too low: {rvol:.2f}x < {self.min_rvol}x"
                )

            if abs_price_move < self.min_price_move_pct:
                if is_near_miss:
                    logger.debug(f"FHM_REJECT: {symbol} | RVOL={rvol:.2f}x | "
                                f"Move={abs_price_move:.2f}% < {self.min_price_move_pct}% | Vol={current_volume:.0f}")
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason=f"Price move too small: {abs_price_move:.2f}% < {self.min_price_move_pct}%"
                )

            if current_volume < self.min_volume:
                logger.debug(f"FHM_REJECT: {symbol} | RVOL={rvol:.2f}x | Move={abs_price_move:.2f}% | "
                            f"Vol={current_volume:.0f} < {self.min_volume}")
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason=f"Volume too low: {current_volume:.0f} < {self.min_volume}"
                )

            # Get VWAP
            vwap = df['vwap'].iloc[-1] if 'vwap' in df.columns else None

            # Get ATR from dataframe (already calculated by pipeline) or fallback
            if 'atr' in df.columns and not pd.isna(df['atr'].iloc[-1]):
                atr = float(df['atr'].iloc[-1])
            else:
                # Fallback: estimate ATR as 1% of current price
                atr = current_price * 0.01

            if price_move_pct > 0:
                # Long candidate
                vwap_check = not self.long_requires_above_vwap or vwap is None or current_price > vwap

                if not vwap_check:
                    logger.debug(f"FHM_REJECT: {symbol} | RVOL={rvol:.2f}x | Move={abs_price_move:.2f}% | "
                                f"VWAP position fail (price={current_price:.2f} <= VWAP={vwap:.2f})")
                    return StructureAnalysis(
                        structure_detected=False,
                        events=[],
                        quality_score=0.0,
                        rejection_reason=f"Long requires price above VWAP"
                    )

                confidence = self._calculate_confidence(rvol, abs_price_move, current_volume)

                # PRO TRADER: Entry at VWAP (pullback), not current price
                entry_price = vwap if vwap else current_price
                entry_zone_buffer = atr * self.entry_zone_atr_mult

                # PRO TRADER: Stop just below VWAP
                stop_loss = (vwap - atr * self.stop_buffer_atr_mult) if vwap else (entry_price - atr * self.stop_buffer_atr_mult)

                logger.debug(f"FHM_DETECT: {symbol} LONG | RVOL={rvol:.2f}x | Move={price_move_pct:+.2f}% | "
                            f"Entry@VWAP={entry_price:.2f} | SL={stop_loss:.2f} | Vol={current_volume:.0f}")

                event = StructureEvent(
                    symbol=symbol,
                    timestamp=current_time,
                    structure_type="first_hour_momentum_long",
                    side="long",
                    confidence=confidence,
                    levels={
                        "open": open_price,
                        "current": current_price,
                        "vwap": vwap,
                        # PRO TRADER: Entry zone around VWAP
                        "entry_price": entry_price,
                        "entry_zone_low": entry_price - entry_zone_buffer,
                        "entry_zone_high": entry_price + entry_zone_buffer,
                        # PRO TRADER: Stop below VWAP
                        "stop_loss": stop_loss,
                        "atr": atr
                    },
                    context={
                        "rvol": rvol,
                        "price_move_pct": price_move_pct,
                        "volume": current_volume,
                        "fhm_type": "momentum_long",
                        "entry_mode": self.entry_mode,
                        "entry_reference": self.entry_reference
                    },
                    price=entry_price  # Entry at VWAP, not current price
                )
                events.append(event)

            else:
                # Short candidate
                vwap_check = not self.short_requires_below_vwap or vwap is None or current_price < vwap

                if not vwap_check:
                    logger.debug(f"FHM_REJECT: {symbol} | RVOL={rvol:.2f}x | Move={abs_price_move:.2f}% | "
                                f"VWAP position fail (price={current_price:.2f} >= VWAP={vwap:.2f})")
                    return StructureAnalysis(
                        structure_detected=False,
                        events=[],
                        quality_score=0.0,
                        rejection_reason=f"Short requires price below VWAP"
                    )

                confidence = self._calculate_confidence(rvol, abs_price_move, current_volume)

                # PRO TRADER: Entry at VWAP (pullback), not current price
                entry_price = vwap if vwap else current_price
                entry_zone_buffer = atr * self.entry_zone_atr_mult

                # PRO TRADER: Stop just above VWAP for shorts
                stop_loss = (vwap + atr * self.stop_buffer_atr_mult) if vwap else (entry_price + atr * self.stop_buffer_atr_mult)

                logger.debug(f"FHM_DETECT: {symbol} SHORT | RVOL={rvol:.2f}x | Move={price_move_pct:+.2f}% | "
                            f"Entry@VWAP={entry_price:.2f} | SL={stop_loss:.2f} | Vol={current_volume:.0f}")

                event = StructureEvent(
                    symbol=symbol,
                    timestamp=current_time,
                    structure_type="first_hour_momentum_short",
                    side="short",
                    confidence=confidence,
                    levels={
                        "open": open_price,
                        "current": current_price,
                        "vwap": vwap,
                        # PRO TRADER: Entry zone around VWAP
                        "entry_price": entry_price,
                        "entry_zone_low": entry_price - entry_zone_buffer,
                        "entry_zone_high": entry_price + entry_zone_buffer,
                        # PRO TRADER: Stop above VWAP for shorts
                        "stop_loss": stop_loss,
                        "atr": atr
                    },
                    context={
                        "rvol": rvol,
                        "price_move_pct": price_move_pct,
                        "volume": current_volume,
                        "fhm_type": "momentum_short",
                        "entry_mode": self.entry_mode,
                        "entry_reference": self.entry_reference
                    },
                    price=entry_price  # Entry at VWAP, not current price
                )
                events.append(event)

            structure_detected = len(events) > 0
            quality_score = self._calculate_quality_score(rvol, abs_price_move, current_volume) if structure_detected else 0.0

            return StructureAnalysis(
                structure_detected=structure_detected,
                events=events,
                quality_score=quality_score,
                analysis_notes={
                    "rvol": rvol,
                    "price_move_pct": price_move_pct,
                    "volume": current_volume,
                    "timing_valid": True
                }
            )

        except Exception as e:
            logger.exception(f"FHM detection failed for {symbol}: {e}")
            return StructureAnalysis(
                structure_detected=False,
                events=[],
                quality_score=0.0,
                rejection_reason=f"Detection error: {e}"
            )

    def _calculate_rvol(self, df: pd.DataFrame) -> float:
        """Calculate Relative Volume (RVOL) - current volume vs 20-bar average."""
        try:
            if len(df) < 5:
                return 1.0

            current_volume = df['volume'].iloc[-1]
            lookback = min(20, len(df) - 1)
            avg_volume = df['volume'].iloc[-lookback-1:-1].mean()

            if pd.isna(avg_volume) or avg_volume <= 0:
                return 1.0

            return current_volume / avg_volume
        except Exception:
            return 1.0

    def _calculate_confidence(self, rvol: float, price_move_pct: float, volume: float) -> float:
        """Calculate confidence based on FHM signals."""
        # Base confidence
        if rvol >= self.high_rvol_threshold:
            confidence = self.confidence_high_rvol
        else:
            confidence = self.confidence_base

        # Bonus for strong RVOL
        if rvol >= 3.0:
            confidence += 0.3
        elif rvol >= 2.5:
            confidence += 0.2

        # Bonus for strong price move
        if price_move_pct >= 2.5:
            confidence += 0.2
        elif price_move_pct >= 2.0:
            confidence += 0.1

        # Bonus for high volume
        if volume >= 500000:
            confidence += 0.2
        elif volume >= 200000:
            confidence += 0.1

        return min(3.0, confidence)  # Cap at 3.0

    def _calculate_quality_score(self, rvol: float, price_move_pct: float, volume: float) -> float:
        """Calculate overall quality score (0-100)."""
        score = 0.0

        # RVOL component (up to 40 points)
        if rvol >= 4.0:
            score += 40.0
        elif rvol >= 3.0:
            score += 30.0
        elif rvol >= 2.0:
            score += 20.0

        # Price move component (up to 30 points)
        if price_move_pct >= 3.0:
            score += 30.0
        elif price_move_pct >= 2.0:
            score += 20.0
        elif price_move_pct >= 1.5:
            score += 10.0

        # Volume component (up to 30 points)
        if volume >= 500000:
            score += 30.0
        elif volume >= 200000:
            score += 20.0
        elif volume >= 100000:
            score += 10.0

        return min(100.0, score)

    def should_detect_at_time(self, current_time: pd.Timestamp) -> bool:
        """
        Override: FHM detection should only happen within the FHM time window.
        """
        try:
            current_time_of_day = current_time.time()
            return self.start_time <= current_time_of_day <= self.end_time
        except Exception:
            return False

    def validate_timing(self, current_time: pd.Timestamp) -> bool:
        """Check if timing is appropriate for FHM strategies."""
        return self.should_detect_at_time(current_time)

    def rank_setup_quality(self, market_context: MarketContext) -> float:
        """Score FHM setup quality (0-100)."""
        try:
            df = market_context.df_5m
            if df is None or len(df) < self.min_bars_required:
                return 0.0

            rvol = self._calculate_rvol(df)
            open_price = df['open'].iloc[0]
            price_move_pct = abs((market_context.current_price - open_price) / open_price) * 100
            volume = df['volume'].iloc[-1]

            return self._calculate_quality_score(rvol, price_move_pct, volume)

        except Exception as e:
            logger.exception(f"FHM ranking failed for {market_context.symbol}: {e}")
            return 0.0

    def plan_long_strategy(self, market_context: MarketContext) -> Optional[TradePlan]:
        """
        Plan long FHM strategy using PRO TRADER rules.

        PRO RULE: Entry at VWAP pullback, stop below VWAP, R:R based targets.
        """
        try:
            df = market_context.df_5m
            if df is None or len(df) < self.min_bars_required:
                return None

            # PRO TRADER: Entry at VWAP (pullback entry)
            vwap = float(df['vwap'].iloc[-1]) if 'vwap' in df.columns else market_context.current_price
            entry_price = vwap

            risk_params = self.calculate_risk_params(entry_price, market_context)
            exit_levels = self._calculate_exit_levels(entry_price, risk_params, "long")

            return TradePlan(
                symbol=market_context.symbol,
                side="long",
                structure_type="first_hour_momentum_long",
                entry_price=entry_price,
                risk_params=risk_params,
                exit_levels=exit_levels,
                qty=0,  # Pipeline calculates sizing
                notional=0.0,
                confidence=self._calculate_confidence(
                    self._calculate_rvol(df),
                    abs((market_context.current_price - df['open'].iloc[0]) / df['open'].iloc[0]) * 100,
                    df['volume'].iloc[-1]
                ),
                timestamp=market_context.timestamp,
                market_context=market_context
            )
        except Exception as e:
            logger.exception(f"FHM: plan_long_strategy failed: {e}")
            return None

    def plan_short_strategy(self, market_context: MarketContext) -> Optional[TradePlan]:
        """
        Plan short FHM strategy using PRO TRADER rules.

        PRO RULE: Entry at VWAP pullback, stop above VWAP, R:R based targets.
        """
        try:
            df = market_context.df_5m
            if df is None or len(df) < self.min_bars_required:
                return None

            # PRO TRADER: Entry at VWAP (pullback entry)
            vwap = float(df['vwap'].iloc[-1]) if 'vwap' in df.columns else market_context.current_price
            entry_price = vwap

            risk_params = self.calculate_risk_params(entry_price, market_context)
            exit_levels = self._calculate_exit_levels(entry_price, risk_params, "short")

            return TradePlan(
                symbol=market_context.symbol,
                side="short",
                structure_type="first_hour_momentum_short",
                entry_price=entry_price,
                risk_params=risk_params,
                exit_levels=exit_levels,
                qty=0,  # Pipeline calculates sizing
                notional=0.0,
                confidence=self._calculate_confidence(
                    self._calculate_rvol(df),
                    abs((market_context.current_price - df['open'].iloc[0]) / df['open'].iloc[0]) * 100,
                    df['volume'].iloc[-1]
                ),
                timestamp=market_context.timestamp,
                market_context=market_context
            )
        except Exception as e:
            logger.exception(f"FHM: plan_short_strategy failed: {e}")
            return None

    def calculate_risk_params(self, entry_price: float, market_context: MarketContext) -> RiskParams:
        """
        Calculate FHM risk parameters using PRO TRADER rules.

        PRO RULE: Stop just below VWAP for longs, above VWAP for shorts.
        This gives tight stops based on VWAP as the invalidation level.
        """
        try:
            df = market_context.df_5m

            # Get ATR from dataframe or fallback
            if df is not None and 'atr' in df.columns and not pd.isna(df['atr'].iloc[-1]):
                atr = float(df['atr'].iloc[-1])
            else:
                atr = entry_price * 0.01  # 1% fallback

            # Get VWAP for stop calculation
            vwap = float(df['vwap'].iloc[-1]) if df is not None and 'vwap' in df.columns else entry_price

            # Determine direction from price vs open
            if df is not None and len(df) > 0:
                open_price = df['open'].iloc[0]
                is_long = market_context.current_price > open_price
            else:
                is_long = True

            # PRO TRADER: Stop just below/above VWAP + buffer
            if is_long:
                hard_sl = vwap - (atr * self.stop_buffer_atr_mult)
            else:
                hard_sl = vwap + (atr * self.stop_buffer_atr_mult)

            risk_per_share = abs(entry_price - hard_sl)

            # Enforce minimum stop distance
            min_stop = entry_price * 0.005  # 0.5% minimum
            if risk_per_share < min_stop:
                risk_per_share = min_stop
                if is_long:
                    hard_sl = entry_price - min_stop
                else:
                    hard_sl = entry_price + min_stop

            return RiskParams(
                hard_sl=hard_sl,
                risk_per_share=risk_per_share,
                atr=atr,
                volatility_adj=1.0
            )
        except Exception as e:
            logger.exception(f"FHM: calculate_risk_params failed: {e}")
            # Safe fallback
            atr = entry_price * 0.01
            return RiskParams(
                hard_sl=entry_price - atr * 2,
                risk_per_share=atr * 2,
                atr=atr,
                volatility_adj=1.0
            )

    def get_exit_levels(self, trade_plan: TradePlan) -> ExitLevels:
        """Get FHM exit levels from trade plan."""
        return trade_plan.exit_levels

    def _calculate_exit_levels(self, entry_price: float, risk_params: RiskParams, side: str) -> ExitLevels:
        """
        Calculate FHM exit levels using PRO TRADER R:R based targets.

        PRO RULE: Targets at 1R, 2R, 3R with 60/40/0 split.
        """
        risk = risk_params.risk_per_share

        # PRO TRADER: 60/40 split at 1R and 2R targets
        if side == "long":
            t1 = entry_price + (risk * 1.0)  # 1R
            t2 = entry_price + (risk * 2.0)  # 2R
        else:
            t1 = entry_price - (risk * 1.0)  # 1R
            t2 = entry_price - (risk * 2.0)  # 2R

        return ExitLevels(
            targets=[
                {"level": t1, "qty_pct": 60, "rr": 1.0},
                {"level": t2, "qty_pct": 40, "rr": 2.0},
            ],
            hard_sl=risk_params.hard_sl,
            trail_to="t1",
            structure_exit={
                "vwap_cross": True,  # Exit if price crosses VWAP against us
                "time_exit": str(self.end_time)  # Exit by end of FHM window
            }
        )
