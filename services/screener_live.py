from __future__ import annotations
"""
ScreenerLive — live 5m close driver.

Responsibilities
- Maintain WebSocket stream → BarBuilder (1m → 5m closed bars).
- Keep a small in-memory store of recent 5m bars per symbol (via BarBuilder APIs).
- On each 5m close: run Stage-0 shortlist, gates, ranker, planner.
- Enqueue orders (or log) via OrderQueue.

Notes
- No REST market-data calls during market hours. All data comes from WS ticks.
- Config: we intentionally do **not** set defaults. Missing keys raise immediately.
- WSClient is created with `on_tick=bar_builder.on_tick` (adapter calls ws.on_message)
- SubscriptionManager handles batching/debouncing subscriptions.

Integration points you likely already have:
- services/ingest/stream_client.WSClient (constructor: WSClient(on_tick, adapter=None))
- services/ingest/subscription_manager.SubscriptionManager
- services/ingest/bar_builder.BarBuilder (constructor requires on_1m_close, on_5m_close)
- services/scan/energy_scanner.EnergyScanner (optional; we fall back if not present)
- services/gates/{regime_gate,event_policy_gate,news_spike_gate,trade_decision_gate}
- services/intraday/{levels,metrics_intraday,planner_internal,ranker}
- services/orders/order_queue.OrderQueue

Flow per 5m close:
  1) Stage-0 (EnergyScanner): compute_features → _filter_stage0 → shortlist
  2) Gate: TradeDecisionGate (structure + regime + events + news)
  3) Rank: PipelineOrchestrator → category-based ranking with regime budget allocation
  4) De-dupe: block quick re-entries unless (cooloff over) AND (setup changed if required) AND (second entry score ≥ stricter bar)
  5) Plan & enqueue

Notes:
- No defaults here; everything tunable is read via config (filters_setup.load_filters).
- Time is naive IST via BarBuilder timestamps.
- We intentionally keep networking out; data comes from BarBuilder.
"""

from dataclasses import dataclass
from datetime import datetime, time as dtime
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp

import pandas as pd

from config.filters_setup import load_filters
from config.logging_config import get_agent_logger, get_screener_logger, get_ranking_logger, get_events_decision_logger
from utils.level_utils import get_previous_day_levels
from utils.dataframe_utils import validate_df, has_column, safe_get_last

# ingest / streaming
from services.ingest.stream_client import WSClient
from services.ingest.subscription_manager import SubscriptionManager
from services.ingest.bar_builder import BarBuilder
from services.ingest.tick_router import TickRouter

# gates
from services.gates.regime_gate import MarketRegimeGate
from services.gates.event_policy_gate import EventPolicyGate
from services.gates.news_spike_gate import NewsSpikeGate
from services.gates.market_sentiment_gate import MarketSentimentGate
from services.gates.trade_decision_gate import TradeDecisionGate, GateDecision as Decision

# planning & ranking
from services import levels
from services import metrics_intraday as mi
from structures.main_detector import MainDetector

# Category-based pipeline orchestrator (replaces services.ranker)
from pipelines import process_setup_candidates

# orders & execution
from services.orders.order_queue import OrderQueue
from services.scan.energy_scanner import EnergyScanner
from diagnostics.diag_event_log import diag_event_log, mint_trade_id
import uuid


logger = get_agent_logger()

# ---------------------------------------------------------------------
# Worker Pool State (initialized once per worker process)
# ---------------------------------------------------------------------
_worker_decision_gate = None

def _init_worker(config_dict):
    """
    Initialize heavy objects in worker process (called once per worker).
    This prevents recreating MainDetector + 40 structures on every task.
    """
    global _worker_decision_gate
    try:
        # Initialize logging for worker FIRST - set to INFO level so cache messages appear
        from config.logging_config import get_agent_logger
        import logging
        worker_logger = get_agent_logger()
        if worker_logger and worker_logger.level == logging.WARNING:
            worker_logger.setLevel(logging.INFO)  # Enable INFO messages in worker

        # Apply structure caching in worker process (if enabled via config flag)
        if config_dict.get("_enable_structure_cache", False):
            import tools.cached_engine_structures  # noqa: F401
            if worker_logger:
                worker_logger.info("[CACHE] Worker process: Structure caching enabled")
            else:
                print("[CACHE] Worker process: Structure caching enabled")

        from services.gates.trade_decision_gate import TradeDecisionGate
        from services.gates.regime_gate import MarketRegimeGate
        from services.gates.event_policy_gate import EventPolicyGate
        from services.gates.news_spike_gate import NewsSpikeGate
        from services.gates.market_sentiment_gate import MarketSentimentGate
        from structures.main_detector import MainDetector
        from config.logging_config import get_agent_logger

        news_cfg = config_dict.get("news_gate", {})

        regime_gate = MarketRegimeGate(cfg=config_dict)
        event_gate = EventPolicyGate(cfg=config_dict)
        news_gate = NewsSpikeGate(
            window_bars=news_cfg.get("window_bars"),
            vol_z_thresh=news_cfg.get("vol_z_thresh"),
            ret_z_thresh=news_cfg.get("ret_z_thresh"),
            body_atr_ratio_thresh=news_cfg.get("body_atr_ratio_thresh"),
        )
        sentiment_gate = MarketSentimentGate(cfg=config_dict, log=get_agent_logger())
        structure_detector = MainDetector(config_dict)

        _worker_decision_gate = TradeDecisionGate(
            structure_detector=structure_detector,
            regime_gate=regime_gate,
            event_policy_gate=event_gate,
            news_spike_gate=news_gate,
            market_sentiment_gate=sentiment_gate,
            quality_filters=config_dict.get('quality_filters', {}),
            regime_allowed_setups=config_dict.get('regime_allowed_setups', {})
        )
    except Exception as e:
        get_agent_logger().exception(f"Worker init failed: {e}")
        raise

def _worker_process_symbol(symbol, df5_data, df1m_data, index_df5_data, levels, now, daily_df=None):
    """
    Process single symbol using pre-initialized decision gate.
    This function is called once per task and reuses the gate.

    Phase 2: Added daily_df parameter for multi-timeframe regime detection.
    """
    global _worker_decision_gate
    if _worker_decision_gate is None:
        return (symbol, None)

    try:
        decision = _worker_decision_gate.evaluate(
            symbol=symbol,
            now=now,
            df1m_tail=df1m_data,
            df5m_tail=df5_data,
            index_df5m=index_df5_data,
            levels=levels,
            daily_df=daily_df,  # Phase 2: Multi-TF regime (210 days)
        )
        return (symbol, decision)
    except Exception as e:
        from config.logging_config import get_agent_logger
        get_agent_logger().exception(f"Worker task failed for {symbol}: {e}")
        return (symbol, None)

@dataclass
class ScreenerConfig:
    """Strict config contract for ScreenerLive."""
    screener_store_5m_max: int
    rank_exec_threshold: float
    rank_pctl_min: float               # <- percentile gate (0..1)
    producer_min_interval_sec: int
    intraday_cutoff_hhmm: str          # "HH:MM" (used as EOD square-off here)


class ScreenerLive:
    """Live orchestrator. Construct once and call start()."""

    def __init__(self, *, sdk, order_queue: OrderQueue) -> None:
        self.sdk = sdk
        self.oq = order_queue

        raw = load_filters()
        self.raw_cfg = raw  # Store raw config for other methods
        try:
            self.cfg = ScreenerConfig(
                screener_store_5m_max=int(raw["screener_store_5m_max"]),
                rank_exec_threshold=float(raw["rank_exec_threshold"]),
                rank_pctl_min=float(raw.get("rank_pctl_min", 0.80)),
                producer_min_interval_sec=int(raw["producer_min_interval_sec"]),
                intraday_cutoff_hhmm=str(raw["eod_squareoff_hhmm"]),
            )
        except KeyError as e:
            raise KeyError(f"ScreenerLive: missing config key {e!s}") from e

        # Timestamp tracking for logging throttle
        self._last_logged_timestamp = None

        # Core ingest / aggregation
        self.agg = BarBuilder(
            bar_5m_span_minutes=5,
            on_1m_close=self._on_1m_close,
            on_5m_close=self._on_5m_close,
            on_15m_close=self._on_15m_close,
            index_symbols=self._index_symbols(),
        )
        self.ws = WSClient(sdk=sdk, on_tick=self.agg.on_tick)
        self.router = TickRouter(on_tick=self.agg.on_tick, token_to_symbol=self._load_core_universe())
        self.ws.on_message(self.router.handle_raw)
        # Register on_close to trigger clean exit when replay ends (don't use datetime.now())
        self.ws.on_close(lambda: self._handle_eod())  # No timestamp - just trigger shutdown
        self.subs = SubscriptionManager(self.ws)

        # Gates - Use MainDetector directly for structure detection
        self.detector = MainDetector(raw)
        news_cfg = raw.get("news_gate")
        self.regime_gate = MarketRegimeGate(cfg=raw)
        # Phase 4: Pass config to EventPolicyGate for session/event policies
        self.event_gate = EventPolicyGate(cfg=raw)
        self.news_gate = NewsSpikeGate(
            window_bars=news_cfg.get("window_bars"),
            vol_z_thresh=news_cfg.get("vol_z_thresh"),
            ret_z_thresh=news_cfg.get("ret_z_thresh"),
            body_atr_ratio_thresh=news_cfg.get("body_atr_ratio_thresh"),
        )
        self.sentiment_gate = MarketSentimentGate(cfg=raw, log=logger)
        self.decision_gate = TradeDecisionGate(
            structure_detector=self.detector,
            regime_gate=self.regime_gate,
            event_policy_gate=self.event_gate,
            news_spike_gate=self.news_gate,
            market_sentiment_gate=self.sentiment_gate,
            quality_filters=raw.get("quality_filters", {}),
            regime_allowed_setups=raw.get("regime_allowed_setups", {}),
        )

        # Stage-0 scanner
        scanner_cfg = raw.get("energy_scanner")
        self.scanner = EnergyScanner(
            top_k_long=scanner_cfg["top_k_long"],
            top_k_short=scanner_cfg["top_k_short"],
        )

        # Trigger-aware executor for live trade execution
        # Note: This needs proper risk state and position management in production
        # For now, creating minimal placeholder objects
        # TriggerAwareExecutor is managed by main.py, not by screener

        # State
        self._last_produced_at: Optional[datetime] = None
        self._levels_cache: Dict[tuple, Dict[str, float]] = {}

        # ORB levels cache: computed once per day at 09:35 and reused for entire day
        # Key: date, Value: Dict[symbol, Dict[str, float]] containing PDH/PDL/PDC/ORH/ORL
        self._orb_levels_cache: Dict = {}

        # NEW: de-dupe memory (per symbol last accepted)
        # stores: {symbol: {"ts": pd.Timestamp, "setup": str|None, "score": float}}
        self._last_entry: Dict[str, Dict[str, object]] = {}

        self._eod_done: bool = False
        self._request_exit: bool = False
        
        self._opening_block = (
            str(raw.get("opening_block_start_hhmm", "")) or None,
            str(raw.get("opening_block_end_hhmm", "")) or None,
        )

        # Create persistent worker pool for structure detection (avoid 3-5s overhead every 5m)
        self._executor = ProcessPoolExecutor(
            max_workers=2,
            initializer=_init_worker,
            initargs=(self.raw_cfg,)
        )
        logger.info("ScreenerLive: Persistent worker pool created (2 workers)")

        # Pre-warm daily data cache for all symbols (avoid 6s per bar disk I/O)
        if hasattr(self.sdk, 'prewarm_daily_cache'):
            self.sdk.prewarm_daily_cache(self.core_symbols, days=210)
            logger.info("ScreenerLive: Daily cache pre-warmed for all symbols")
        else:
            logger.info("ScreenerLive: SDK doesn't support cache pre-warming (live mode)")

        logger.debug(
            "ScreenerLive init: universe=%d symbols, store5m=%d",
            len(self.core_symbols),
            self.cfg.screener_store_5m_max,
        )

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def start(self) -> None:
        """Connect WS and subscribe the core universe."""
        self.subs.set_core(self.token_map)
        self.subs.start()
        self.ws.start()

        # Trigger executor is managed by main.py

        logger.info("WS connected; core subscriptions scheduled: %d symbols", len(self.core_symbols))

    def stop(self) -> None:
        try: self.subs.stop()
        except Exception: pass
        try: self.ws.stop()
        except Exception: pass

        # Shutdown persistent worker pool
        try:
            if hasattr(self, '_executor') and self._executor:
                self._executor.shutdown(wait=True, cancel_futures=True)
                logger.info("ScreenerLive: Worker pool shut down")
        except Exception as e:
            logger.warning(f"ScreenerLive: Worker pool shutdown error: {e}")

        # Trigger executor is managed by main.py

        logger.info("ScreenerLive stopped")

    # ---------------------------------------------------------------------
    # BarBuilder callbacks
    # ---------------------------------------------------------------------
    def _on_1m_close(self, symbol: str, bar_1m: pd.Series) -> None:
        """Hook 1-minute bar closes - TriggerAwareExecutor handles this automatically"""
        # Note: TriggerAwareExecutor automatically hooks into BarBuilder's 1m callback
        # No manual forwarding needed - it replaces this method during initialization
        pass

    def _on_15m_close(self, symbol: str, bar_15m: pd.Series) -> None:
        """
        HTF (15m) bar close handler - triggers re-ranking of candidates.

        Per playbook: 15m bars used for HTF confirmation but NEVER block entries.
        Only affects ranking scores via HTF bonuses/penalties.
        """
        # TODO Phase 1.2: Implement HTF rank update logic
        # For now, just log that 15m bars are being captured
        ts = bar_15m.name if hasattr(bar_15m, "name") else datetime.now()
        logger.debug(f"HTF: 15m bar closed for {symbol} at {ts}")
        # HTF context is now passed to pipeline orchestrator via htf_context parameter

    def _enhance_candidates_with_htf(self, symbol: str, candidates: List) -> List:
        """
        Phase 1.4: Enhance setup candidate strength with 15m HTF confirmation.

        Applies confidence boost/penalty based on 15m trend and volume alignment:
        - +15% boost if 15m trend aligned with setup direction
        - -10% penalty if 15m trend opposes setup direction
        - +5% boost if 15m volume surge (>1.3x median)
        """
        from dataclasses import dataclass, replace
        from services.gates.trade_decision_gate import SetupCandidate

        df15 = self.agg.get_df_15m_tail(symbol, 10)
        if df15 is None or df15.empty or len(df15) < 2:
            return candidates  # No HTF data, return as-is

        last_15m = df15.iloc[-1]
        prev_15m = df15.iloc[-2]

        # 15m trend detection
        htf_trend_up = float(last_15m.get("close", 0.0)) > float(prev_15m.get("close", 0.0))

        # 15m volume surge detection
        htf_volume_surge = False
        if "volume" in df15.columns and len(df15) >= 6:
            recent_vol_15m = df15["volume"].tail(6).median()
            current_vol_15m = float(last_15m.get("volume", 0.0) or 0.0)
            htf_volume_surge = (current_vol_15m / recent_vol_15m) >= 1.3 if recent_vol_15m > 0 else False

        # Adjust each candidate based on setup direction vs HTF trend
        enhanced = []
        for candidate in candidates:
            setup_type = candidate.setup_type if hasattr(candidate, 'setup_type') else None
            if not setup_type:
                enhanced.append(candidate)
                continue

            # Determine setup direction (long vs short)
            is_long_setup = any(kw in setup_type.lower() for kw in ["long", "bull", "buy"])
            is_short_setup = any(kw in setup_type.lower() for kw in ["short", "bear", "sell"])

            # Base strength
            base_strength = candidate.strength if hasattr(candidate, 'strength') else 0.5
            adjusted_strength = base_strength

            # Apply HTF trend alignment adjustment
            if is_long_setup and htf_trend_up:
                adjusted_strength *= 1.15  # +15% for aligned long
            elif is_short_setup and not htf_trend_up:
                adjusted_strength *= 1.15  # +15% for aligned short
            elif is_long_setup and not htf_trend_up:
                adjusted_strength *= 0.90  # -10% for opposing long
            elif is_short_setup and htf_trend_up:
                adjusted_strength *= 0.90  # -10% for opposing short

            # Apply volume surge bonus (additive, regardless of direction)
            if htf_volume_surge:
                adjusted_strength *= 1.05  # +5% for volume confirmation

            # Phase 2.1: Determine trading lane (Precision vs Fast Scalp)
            # Precision Lane: HTF trend aligned + volume surge -> full size + T1+T2
            # Fast Scalp Lane: HTF not aligned or weak -> 50% size + T1 only
            htf_aligned = (is_long_setup and htf_trend_up) or (is_short_setup and not htf_trend_up)
            lane_type = "precision_lane" if (htf_aligned and htf_volume_surge) else "fast_scalp_lane"

            # Add lane info to reasons
            updated_reasons = list(candidate.reasons if hasattr(candidate, 'reasons') else [])
            updated_reasons.append(f"lane:{lane_type}")
            if htf_aligned:
                updated_reasons.append("htf:aligned")
            if htf_volume_surge:
                updated_reasons.append("htf:volume_surge")

            # Create adjusted candidate (immutable dataclass, so use replace if available)
            try:
                adjusted_candidate = replace(candidate, strength=adjusted_strength, reasons=updated_reasons)
            except:
                # Fallback if not a dataclass - create new SetupCandidate
                adjusted_candidate = SetupCandidate(
                    setup_type=candidate.setup_type,
                    strength=adjusted_strength,
                    reasons=updated_reasons,
                    orh=getattr(candidate, 'orh', None),
                    orl=getattr(candidate, 'orl', None)
                )

            enhanced.append(adjusted_candidate)

        return enhanced

    def _build_htf_context(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Build HTF (15m) context for pipeline ranking adjustments.

        Returns dict with:
        - htf_trend: "up", "down", or "neutral"
        - htf_volume_surge: True if 15m volume > 1.3x median
        - htf_momentum: normalized momentum score (-1 to 1)
        - htf_exhaustion: True if signs of trend exhaustion on 15m

        Each category pipeline uses these differently in calculate_rank_score().
        """
        df15 = self.agg.get_df_15m_tail(symbol, 10)
        if df15 is None or df15.empty or len(df15) < 2:
            return None

        last_15m = df15.iloc[-1]
        prev_15m = df15.iloc[-2]

        # Trend detection
        close_now = float(last_15m.get("close", 0.0))
        close_prev = float(prev_15m.get("close", 0.0))
        high_now = float(last_15m.get("high", 0.0))
        low_now = float(last_15m.get("low", 0.0))
        high_prev = float(prev_15m.get("high", 0.0))
        low_prev = float(prev_15m.get("low", 0.0))

        # Higher highs and higher lows = up, lower highs and lower lows = down
        hh = high_now > high_prev
        hl = low_now > low_prev
        lh = high_now < high_prev
        ll = low_now < low_prev

        if hh and hl:
            htf_trend = "up"
        elif lh and ll:
            htf_trend = "down"
        else:
            htf_trend = "neutral"

        # Volume surge detection
        htf_volume_surge = False
        if "volume" in df15.columns and len(df15) >= 6:
            recent_vol_15m = df15["volume"].tail(6).median()
            current_vol_15m = float(last_15m.get("volume", 0.0) or 0.0)
            htf_volume_surge = (current_vol_15m / recent_vol_15m) >= 1.3 if recent_vol_15m > 0 else False

        # Momentum score: normalized price change vs ATR
        htf_momentum = 0.0
        if len(df15) >= 3:
            atr_proxy = (df15["high"] - df15["low"]).tail(5).mean()
            if atr_proxy > 0:
                price_change = close_now - close_prev
                htf_momentum = max(-1.0, min(1.0, price_change / atr_proxy))

        # Exhaustion detection: long upper/lower wick relative to body
        body = abs(close_now - float(last_15m.get("open", close_now)))
        upper_wick = high_now - max(close_now, float(last_15m.get("open", close_now)))
        lower_wick = min(close_now, float(last_15m.get("open", close_now))) - low_now
        total_range = high_now - low_now

        htf_exhaustion = False
        if total_range > 0 and body > 0:
            # Exhaustion: wick > 2x body in direction of move
            if htf_trend == "up" and upper_wick > 2 * body:
                htf_exhaustion = True
            elif htf_trend == "down" and lower_wick > 2 * body:
                htf_exhaustion = True

        return {
            "htf_trend": htf_trend,
            "htf_volume_surge": htf_volume_surge,
            "htf_momentum": round(htf_momentum, 3),
            "htf_exhaustion": htf_exhaustion,
        }

    def _on_5m_close(self, symbol: str, bar_5m: pd.Series) -> None:
        """Main driver: invoked for each CLOSED 5m bar of any symbol."""
        import time
        _t_bar_start = time.perf_counter()

        now = bar_5m.name if hasattr(bar_5m, "name") else datetime.now()

        # EOD square-off guard (using configured HH:MM). Leave parser as-is per your preference.
        if self._is_after_cutoff(now):
            if not getattr(self, "_eod_done", False):
                self._handle_eod(now)
            return

        # Throttle production if last run was too recent (config-driven).
        if not self._should_produce(now):
            return
        self._last_produced_at = now

        # ENHANCED LOGGING: Progress tracking (only log once per unique timestamp)
        current_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        if self._last_logged_timestamp != current_time_str:
            logger.info("SCANNER_PROGRESS | Bar: %s | Stage-0 starting", current_time_str)
            self._last_logged_timestamp = current_time_str

        # ---------- Stage-0: EnergyScanner (single unified path) ----------
        shortlist: List[str] = []
        feats_df = None
        levels_by_symbol = None  # Initialize so it's available to structure detection phase

        # OPENING BELL FIX: Determine minimum bars needed - 1 during opening bell (09:20-09:30), 3 normally
        current_time = now.time() if hasattr(now, 'time') else now
        from datetime import time as dtime
        in_opening_bell = dtime(9, 20) <= current_time < dtime(9, 30)
        min_bars_for_processing = 1 if in_opening_bell else 3

        try:
            df5_by_symbol: Dict[str, pd.DataFrame] = {}
            for s in self.core_symbols:
                df5 = self.agg.get_df_5m_tail(s, self.cfg.screener_store_5m_max)
                if validate_df(df5, min_rows=min_bars_for_processing):
                    df5_by_symbol[s] = df5
            if df5_by_symbol:
                # PERFORMANCE FIX: Compute ORB levels once at 09:40 and cache for entire day
                # - ORH/ORL values finalized at 09:30 (end of opening range)
                # - Computing for all 1992 symbols takes ~54s in OCI (one-time cost)
                # - Returns cached values on all subsequent bars (fast)
                # - Before 09:40: returns None (ORB priority scanner won't have dist_to_ORH/ORL columns)
                # Impact: ONE-TIME 54s cost at 09:40 instead of 30min spread across multiple bars
                levels_by_symbol = self._compute_orb_levels_once(now, df5_by_symbol)

                feats_df = self.scanner.compute_features(df5_by_symbol, lookback_bars=20, levels_by_symbol=levels_by_symbol, allow_early_scan=in_opening_bell)
                feats_df = self._filter_stage0(feats_df, now, skip_vol_persist=in_opening_bell)  # liquidity + vwap proximity + momentum + (opt) vol persistence
                # Use scanner's select_shortlist for proper structured logging
                shortlist_dict = self.scanner.select_shortlist(feats_df)
                shortlist = shortlist_dict.get("long", []) + shortlist_dict.get("short", [])
        except Exception as e:
            logger.exception("EnergyScanner failed; fallback shortlist: %s", e)
            shortlist = self._fallback_shortlist()

        # ENHANCED LOGGING: Stage-0 completion with accurate counts
        _t_scanner_end = time.perf_counter()
        eligible_symbols = len(df5_by_symbol) if df5_by_symbol else 0
        total_symbols = len(self.core_symbols)
        shortlist_count = len(shortlist)

        logger.info("SCANNER_COMPLETE | Processed %d eligible of %d total symbols → %d shortlisted (%.1f%%) | Stage-0→Gates | TIME: %.2fs",
                   eligible_symbols, total_symbols, shortlist_count,
                   (shortlist_count/max(eligible_symbols,1))*100, _t_scanner_end - _t_bar_start)
        if not shortlist:
            return

        # ---------- Gate per candidate (structure + regime + events + news) ----------
        index_df5 = self._index_df5()
        decisions: List[Tuple[str, Decision]] = []
        screener_logger = get_screener_logger()

        # PARALLEL STRUCTURE DETECTION (Phase 1 Optimization)
        # Use ProcessPoolExecutor to parallelize structure detection across symbols
        # This reduces 50s bottleneck to ~15s (3.3x speedup)

        # Prepare data for parallel processing
        # Phase 2: Include daily_df for multi-timeframe regime detection
        symbol_data_map = {}
        for sym in shortlist:
            df5 = self.agg.get_df_5m_tail(sym, self.cfg.screener_store_5m_max)
            # OPENING BELL FIX: Use same min_bars as scanner (1 during opening bell, 5 normally)
            if not validate_df(df5, min_rows=min_bars_for_processing):
                continue
            df1m = self.agg.get_df_1m_tail(sym, 60)

            # PERFORMANCE FIX: Use cached ORB levels if available
            if levels_by_symbol and sym in levels_by_symbol:
                # Symbol was in cache - use cached levels
                lvl = levels_by_symbol[sym]
            elif levels_by_symbol is not None:
                # Cache was computed but symbol not in it (started trading late)
                # Use empty levels - don't try to compute ORB from incomplete data
                lvl = {"PDH": float("nan"), "PDL": float("nan"), "PDC": float("nan"),
                       "ORH": float("nan"), "ORL": float("nan")}
            else:
                # OPENING BELL FIX: Before 09:40 - cache not ready yet
                # During opening bell (09:20-09:30) with <3 bars, use empty levels to avoid warnings
                if in_opening_bell and len(df5) < 3:
                    lvl = {"PDH": float("nan"), "PDL": float("nan"), "PDC": float("nan"),
                           "ORH": float("nan"), "ORL": float("nan")}
                else:
                    # Enough bars to compute levels normally
                    lvl = self._levels_for(sym, df5, now)

            # Phase 2: Fetch daily data (210 days for EMA200, uses cache)
            daily_df = self.sdk.get_daily(sym, days=210)
            symbol_data_map[sym] = (df5, df1m, lvl, daily_df)

        if not symbol_data_map:
            logger.info("GATES_COMPLETE | No symbols with sufficient data")
            return

        _t_data_prep_end = time.perf_counter()
        logger.info("DATA_PREP_COMPLETE | Prepared %d symbols | TIME: %.2fs",
                   len(symbol_data_map), _t_data_prep_end - _t_scanner_end)

        # Use persistent worker pool (created once in __init__)
        # This saves 3-5 seconds per 5m bar by avoiding worker recreation
        # Phase 2: Unpack 4-tuple (df5, df1m, lvl, daily_df) for multi-TF regime
        futures = {}
        for sym, (df5, df1m, lvl, daily_df) in symbol_data_map.items():
            future = self._executor.submit(
                _worker_process_symbol,  # Use function that reuses initialized gate
                sym,
                df5,
                df1m,
                index_df5,
                lvl,
                now,
                daily_df  # Phase 2: Pass daily_df for multi-timeframe regime
            )
            futures[future] = sym

        # Collect results as they complete
        try:
            for future in as_completed(futures):
                expected_sym = futures[future]
                try:
                    returned_sym, decision = future.result()

                    # DATA INTEGRITY CHECK: Verify symbol matches
                    assert returned_sym == expected_sym, \
                        f"Symbol mismatch! Expected {expected_sym}, got {returned_sym}"

                    if decision is None:
                        logger.debug(f"Structure detection returned None for {returned_sym}")
                        continue

                    sym = returned_sym  # Use verified symbol
                    df5 = symbol_data_map[sym][0]  # Get original df5 for logging

                except Exception as e:
                    logger.exception(f"Failed to process {expected_sym}: {e}")
                    continue

                if not decision.accept:
                    top_reason = next((r for r in decision.reasons if r.startswith("regime_block:")), None) or \
                                 (decision.reasons[0] if decision.reasons else "reject")
                    logger.debug(
                        "DECISION:REJECT sym=%s setup=%s regime=%s reason=%s | all=%s",
                        sym, decision.setup_type, decision.regime, top_reason, ";".join(decision.reasons),
                    )

                    # Log screener rejection with detailed context
                    # Phase 2: Include multi-TF regime diagnostics
                    screener_logger.log_reject(
                        sym,
                        top_reason,
                        timestamp=now.isoformat(),
                        setup_type=decision.setup_type or "unknown",
                        regime=decision.regime or "unknown",
                        all_reasons=decision.reasons,
                        structure_confidence=getattr(decision, 'structure_confidence', 0),
                        current_price=df5['close'].iloc[-1] if not df5.empty else 0,
                        regime_diagnostics=getattr(decision, 'regime_diagnostics', None)  # Phase 2: Multi-TF regime
                    )
                    continue

                logger.info(
                    "DECISION:ACCEPT sym=%s setup=%s regime=%s size_mult=%.2f hold_bars=%d | %s",
                    sym, decision.setup_type, decision.regime, decision.size_mult, decision.min_hold_bars,
                    ";".join(decision.reasons),
                )

                # Log screener acceptance with detailed context
                # Phase 2: Include multi-TF regime diagnostics
                screener_logger.log_accept(
                    sym,
                    timestamp=now.isoformat(),
                    setup_type=decision.setup_type or "unknown",
                    regime=decision.regime or "unknown",
                    size_mult=decision.size_mult,
                    min_hold_bars=decision.min_hold_bars,
                    all_reasons=decision.reasons,
                    structure_confidence=getattr(decision, 'structure_confidence', 0),
                    current_price=df5['close'].iloc[-1] if not df5.empty else 0,
                    vwap=df5.get('vwap', pd.Series([0])).iloc[-1] if not df5.empty else 0,
                    regime_diagnostics=getattr(decision, 'regime_diagnostics', None)  # Phase 2: Multi-TF regime
                )
                decisions.append((sym, decision))
        except Exception as e:
            logger.exception(f"Worker pool processing failed: {e}")

        # DATA INTEGRITY CHECK: Verify all accepted symbols are in original shortlist
        accepted_symbols = {sym for sym, _ in decisions}
        assert accepted_symbols.issubset(set(shortlist)), \
            f"Symbol integrity violation! Accepted symbols not in shortlist: {accepted_symbols - set(shortlist)}"

        _t_structure_end = time.perf_counter()
        logger.info(f"PARALLEL_STRUCTURE_COMPLETE | Processed {len(symbol_data_map)} symbols, {len(decisions)} accepted | TIME: %.2fs", _t_structure_end - _t_data_prep_end)

        # EOD check AFTER structure detection (which can take 20+ minutes)
        # Prevents hanging when structure completes after market close
        if self._is_after_cutoff(now):
            if not getattr(self, "_eod_done", False):
                logger.warning("EOD reached during structure detection at %s — stopping", now)
                self._handle_eod(now)
            return

        gate_accept_count = len(decisions)
        logger.info("GATES_COMPLETE | %d→%d symbols (%.1f%%) | Gates→Ranking", shortlist_count, gate_accept_count, (gate_accept_count/max(shortlist_count,1))*100)
        if not decisions:
            return

        dec_map = {s: d for (s, d) in decisions}

        # ---------- Pipeline Orchestrator: Ranking + Planning ----------
        # Orchestrator handles: screening + gates + quality + ranking + entry + targets
        # Returns plans already sorted by ranking score
        logger.info("ORCHESTRATOR | Processing %d symbols via pipeline orchestrator", len(decisions))

        max_trades_per_cycle = self.raw_cfg.get("max_trades_per_cycle", 10)
        trades_planned = 0
        ranking_logger = get_ranking_logger()
        events_logger = get_events_decision_logger()

        eligible_plans: List[Tuple[str, Dict, float]] = []  # (symbol, plan, score)

        for sym, decision in decisions:
            df5 = symbol_data_map.get(sym, (None,))[0]
            df1m = symbol_data_map.get(sym, (None, None))[1]
            lvl = symbol_data_map.get(sym, (None, None, {}))[2]
            daily_df = symbol_data_map.get(sym, (None, None, None, None))[3] if len(symbol_data_map.get(sym, ())) > 3 else None

            if df5 is None:
                continue

            setup_candidates = getattr(decision, 'setup_candidates', None)
            if not setup_candidates:
                continue

            # Build HTF context from 15m data for category-specific ranking adjustments
            htf_context = self._build_htf_context(sym)

            try:
                plan = process_setup_candidates(
                    symbol=sym,
                    df5m=df5,
                    df1m=df1m,
                    levels=lvl,
                    regime=decision.regime,
                    now=now,
                    candidates=setup_candidates,
                    daily_df=daily_df,
                    htf_context=htf_context
                )
            except Exception as e:
                logger.exception("orchestrator failed for %s: %s", sym, e)
                continue

            if plan and plan.get("eligible", False):
                score = plan.get("ranking", {}).get("score", 0.0)
                eligible_plans.append((sym, plan, score, decision))
                logger.debug(f"ORCHESTRATOR:ELIGIBLE {sym} score={score:.3f}")
            else:
                reason = plan.get("reason", "no_plan") if plan else "no_plan"
                logger.debug(f"ORCHESTRATOR:REJECT {sym} reason={reason}")

        # Sort by score descending
        eligible_plans.sort(key=lambda x: x[2], reverse=True)

        # Compute percentile for logging
        if eligible_plans:
            pctl_score = self._compute_percentile_cut([(s, sc) for s, _, sc, _ in eligible_plans], self.cfg.rank_pctl_min)
        else:
            pctl_score = 0.0

        logger.info("ORCHESTRATOR_COMPLETE | %d eligible plans from %d decisions", len(eligible_plans), len(decisions))

        # ---------- Process eligible plans → Execution ----------
        for i, (sym, plan, score, decision) in enumerate(eligible_plans):
            strategy_type = plan.get("strategy", "unknown")
            df5 = symbol_data_map.get(sym, (None,))[0]

            # Log ranking acceptance
            ranking_logger.log_accept(
                sym,
                timestamp=now.isoformat(),
                rank_score=score,
                threshold=0.0,  # Orchestrator already applied thresholds
                percentile_score=pctl_score,
                strategy_type=strategy_type,
                rank_position=i + 1,
                total_candidates=len(eligible_plans),
                regime_diagnostics=getattr(decision, 'regime_diagnostics', None) if decision else None
            )

            # 1) Eligibility (already checked, but keep for compatibility)
            if not plan.get("eligible", False):
                # Try quality.rejection_reason first, then fall back to top-level reason
                rejection_reason = (plan.get("quality") or {}).get("rejection_reason") or plan.get("reason", "unknown")
                cautions = ";".join((plan.get("notes") or {}).get("cautions", []))

                # Enhanced logging for unknown rejections
                if rejection_reason == "unknown":
                    logger.debug("PLAN_INELIGIBLE_UNKNOWN: %s | strategy=%s | eligible=%s | reason=%s | quality=%s",
                                sym, plan.get("strategy", "unknown"), plan.get("eligible"),
                                plan.get("reason"), plan.get("quality"))

                logger.info("SKIP %s: ineligible plan rejection_reason=%s cautions=%s",
                            sym, rejection_reason, cautions)
                events_logger.log_reject(
                    sym,
                    "plan_ineligible",
                    timestamp=now.isoformat(),
                    rejection_reason=rejection_reason,
                    cautions=cautions,
                    strategy_type=strategy_type or "unknown"
                )
                continue

            # 2) Qty
            qty = int((plan.get("sizing") or {}).get("qty") or 0)
            if qty <= 0:
                logger.info("SKIP %s: qty<=0", sym)
                events_logger.log_reject(sym, "zero_quantity", timestamp=now.isoformat(), qty=qty, strategy_type=strategy_type or "unknown")
                continue

            # 3) Bias → side
            bias = str(plan.get("bias", "")).lower()
            if bias not in ("long", "short"):
                logger.info("SKIP %s: bad bias=%r", sym, bias)
                events_logger.log_reject(sym, "invalid_bias", timestamp=now.isoformat(), bias=bias, strategy_type=strategy_type or "unknown")
                continue

            # --- DECISION: canonical payload (no fallbacks) ---

            # 1) stable id
            if "trade_id" not in plan:
                plan["trade_id"] = mint_trade_id(sym, token=uuid.uuid4().hex[:8])

            # --- de-dupe check (cooloff, setup change, second-entry strength) ---
            decision_obj = dec_map.get(sym)
            setup_type = getattr(decision_obj, "setup_type", None) if decision_obj is not None else None
            if not self._dedupe_ok(sym=sym, now_ts=now, setup_type=setup_type, score=score, pctl_score=pctl_score):
                logger.info("DEDUPE:SKIP sym=%s reason=cooloff/setup_not_stronger", sym)
                events_logger.log_reject(
                    sym,
                    "deduplication_block",
                    timestamp=now.isoformat(),
                    strategy_type=strategy_type or "unknown",
                    score=score,
                    pctl_score=pctl_score
                )
                continue

            # Minimal bar5/features snapshot for diagnostics
            last5 = safe_get_last(df5, "close") if validate_df(df5) else None
            if last5 is not None and validate_df(df5):
                last5 = df5.iloc[-1]  # Get full row for feature extraction
            else:
                last5 = None
            bar5 = {}
            if last5 is not None:
                for k in ("open", "high", "low", "close", "volume", "vwap", "adx", "bb_width_proxy"):
                    if k in last5.index:
                        bar5[k] = float(last5.get(k, 0.0))

            # Build ranker dict with rank_score and FHM context if available
            ranker_dict = {"rank_score": float(score)}
            fhm_ctx = plan.get("fhm_context")
            if fhm_ctx:
                ranker_dict["fhm_rvol"] = fhm_ctx.get("rvol", 0.0)
                ranker_dict["fhm_price_move_pct"] = fhm_ctx.get("price_move_pct", 0.0)

            features = {
                "bar5": bar5,
                "ranker": ranker_dict,
                "time": {"minute_of_day": now.hour * 60 + now.minute, "day_of_week": now.weekday()},
            }

            if plan.get("price") is None:
                plan["price"] = (plan.get("entry") or {}).get("reference")
            plan["decision_ts"] = str(now)

            # Flatten decision to serializable dict for diag log
            # Phase 2: Include multi-TF regime diagnostics
            reasons_str = None
            if decision_obj is not None:
                r = getattr(decision_obj, "reasons", None)
                if isinstance(r, (list, tuple)): reasons_str = ";".join(str(x) for x in r)
                elif r is not None: reasons_str = str(r)
            decision_dict = {
                "setup_type": plan.get("strategy"),  # Use plan's strategy (from pipeline), not deprecated decision_obj.setup_type
                "regime": getattr(decision_obj, "regime", None) if decision_obj is not None else None,
                "reasons": reasons_str,
                "size_mult": getattr(decision_obj, "size_mult", None) if decision_obj is not None else None,
                "min_hold_bars": getattr(decision_obj, "min_hold_bars", None) if decision_obj is not None else None,
                "regime_diagnostics": getattr(decision_obj, "regime_diagnostics", None) if decision_obj is not None else None,  # Phase 2: Multi-TF regime
            }
            diag_event_log.log_decision(symbol=plan["symbol"], now=now, plan=plan, features=features, decision=decision_dict)

            exec_item = {
                "symbol": plan["symbol"],
                "plan": {
                    "side": "BUY" if plan["bias"] == "long" else "SELL",
                    "qty": int(plan["sizing"]["qty"]),
                    "entry_zone": (plan["entry"] or {}).get("zone"),
                    "price": (plan["entry"] or {}).get("reference"),
                    "entry_ref_price": (plan["entry"] or {}).get("reference"),
                    "stop": plan.get("stop"),  # Full stop dict: {"hard": x, "risk_per_share": y}
                    "hard_sl": (plan.get("stop") or {}).get("hard"),
                    "targets": plan.get("targets"),
                    "trail": plan.get("trail"),
                    "trade_id": plan["trade_id"],
                    "orh": (plan.get("levels") or {}).get("ORH"),
                    "orl": (plan.get("levels") or {}).get("ORL"),
                    "decision_ts": plan["decision_ts"],
                    "strategy": plan.get("strategy", ""),
                },
                "meta": plan,
            }

            # Check trades per cycle limit (only for live trading)
            if trades_planned >= max_trades_per_cycle:
                logger.info("CYCLE:LIMIT_REACHED %d/%d trades - skipping %s", trades_planned, max_trades_per_cycle, sym)
                events_logger.log_reject(
                    sym,
                    "cycle_limit_reached",
                    timestamp=now.isoformat(),
                    trades_planned=trades_planned,
                    max_trades_per_cycle=max_trades_per_cycle,
                    strategy_type=strategy_type or "unknown"
                )
                break

            # Log final events decision acceptance
            events_logger.log_accept(
                sym,
                timestamp=now.isoformat(),
                strategy_type=strategy_type or "unknown",
                side=plan["bias"],
                entry_price=plan.get("price"),
                qty=qty,
                trade_id=plan["trade_id"],
                score=score,
                trades_planned=trades_planned + 1,
                max_trades_per_cycle=max_trades_per_cycle
            )

            # Update de-dupe memory only when we actually enqueue
            self._last_entry[sym] = {"ts": now, "setup": setup_type, "score": float(score)}
            self.oq.enqueue(exec_item)
            trades_planned += 1
            logger.info("ENQUEUE %s score=%.3f count=%d/%d reasons=%s", sym, score, trades_planned, max_trades_per_cycle, ";".join(self._reasons_for(sym, decisions)))

        # Final timing for entire bar processing
        _t_bar_end = time.perf_counter()
        logger.info("BAR_COMPLETE | Total bar processing time: %.2fs (Scanner: %.2fs, DataPrep: %.2fs, Structure: %.2fs, Ranking+Planning: %.2fs)",
                   _t_bar_end - _t_bar_start,
                   _t_scanner_end - _t_bar_start,
                   _t_data_prep_end - _t_scanner_end,
                   _t_structure_end - _t_data_prep_end,
                   _t_bar_end - _t_structure_end)

    # ---------- EOD handler ----------
    def _handle_eod(self, now: datetime = None) -> None:
        try:
            if now:
                logger.warning("EOD: cutoff reached at %s — stopping for the day", now)
            else:
                logger.warning("EOD: replay ended — stopping for the day")
        except Exception:
            pass
        self._eod_done = True
        try: self.subs.stop()
        except Exception: pass
        try: self.ws.stop()
        except Exception: pass
        self._request_exit = True

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _load_core_universe(self):
        try:
            self.symbol_map = self.sdk.get_symbol_map()
            self.token_map = self.sdk.get_token_map()
            self.core_symbols = list(self.symbol_map.keys())
        except Exception as e:
            raise RuntimeError(f"ScreenerLive: sdk.list_equities() failed: {e}")
        return self.token_map

    def _index_symbols(self) -> List[str]:
        return []

    def _index_df5(self) -> pd.DataFrame:
        idx = self.agg.index_df_5m()
        if isinstance(idx, dict):
            for _, df in idx.items():
                return df
            return pd.DataFrame()
        return idx if isinstance(idx, pd.DataFrame) else pd.DataFrame()

    def _compute_orb_levels_once(self, now, df5_by_symbol: Dict[str, pd.DataFrame]) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Compute ORH/ORL/PDH/PDL/PDC for all symbols once at 09:35 and cache for the day.

        This is a critical performance optimization:
        - ORB (Opening Range Breakout) window is 09:15-09:30
        - ORH/ORL values are finalized at 09:30 and don't change for rest of day
        - Computing levels for all 1992 symbols takes ~54s in OCI (2 CPUs)
        - By computing once at 09:35 instead of on every bar, we save ~30 minutes per day

        Returns:
            Dict[symbol, Dict[str, float]] if computed/cached, None if before 09:35
        """
        if not now:
            return None

        session_date = now.date()
        current_time = now.time()

        # Check if already computed for today
        if session_date in self._orb_levels_cache:
            return self._orb_levels_cache[session_date]

        # Only compute at or after 09:40 (ensures ORB data 09:15-09:30 is complete)
        # By 09:40, more symbols have sufficient data for reliable ORH/ORL calculation
        if current_time < dtime(9, 40):
            return None

        import time as time_module
        start_time = time_module.perf_counter()
        logger.info(f"ORB_CACHE | Computing ORH/ORL/PDH/PDL/PDC for all symbols once at {current_time} (session_date={session_date})")

        levels_by_symbol = {}
        success_count = 0
        fail_count = 0

        for sym, df5 in df5_by_symbol.items():
            try:
                lvl = self._levels_for(sym, df5, now)
                # Only count as success if we got valid ORH/ORL (not NaN)
                orh = lvl.get("ORH", float("nan"))
                orl = lvl.get("ORL", float("nan"))
                if not (pd.isna(orh) or pd.isna(orl)):
                    levels_by_symbol[sym] = lvl
                    success_count += 1
                else:
                    # ORH/ORL are NaN - not enough data yet
                    levels_by_symbol[sym] = {}
                    fail_count += 1
            except Exception as e:
                logger.warning(f"ORB_CACHE | Failed to compute levels for {sym}: {e}")
                levels_by_symbol[sym] = {}
                fail_count += 1

        # Cache for the entire day
        self._orb_levels_cache[session_date] = levels_by_symbol

        elapsed = time_module.perf_counter() - start_time
        logger.info(
            f"ORB_CACHE | Cached levels for {success_count} symbols (failed: {fail_count}) | "
            f"Session: {session_date} | Time: {elapsed:.2f}s | "
            f"This is a ONE-TIME cost - all subsequent bars will use cached values"
        )

        return levels_by_symbol

    def _levels_for(self, symbol: str, df5: pd.DataFrame, now) -> Dict[str, float]:
        """Prev-day PDH/PDL/PDC and today ORH/ORL (cached per (symbol, session_date))."""
        try:
            session_date = df5.index[-1].date() if validate_df(df5) else None
        except Exception:
            session_date = None
        key = (symbol, session_date)
        cached = self._levels_cache.get(key)
        if cached:
            return cached

        # Use centralized level calculation to avoid code duplication
        try:
            logger.debug(f"LEVELS: Getting daily data for {symbol}, session_date={session_date}")
            daily = self.sdk.get_daily(symbol, days=210)
            logger.debug(f"LEVELS: Daily data shape: {daily.shape if daily is not None else None}")

            # Use centralized get_previous_day_levels with fallback for backtests
            level_dict = get_previous_day_levels(
                daily_df=daily,
                session_date=session_date,
                fallback_df=df5,
                enable_fallback=True  # Enable fallback for backtests
            )
            pdh = level_dict.get("PDH", float("nan"))
            pdl = level_dict.get("PDL", float("nan"))
            pdc = level_dict.get("PDC", float("nan"))

        except Exception as e:
            # CRITICAL FIX: Log previous day level computation failures
            import traceback
            logger.error(f"LEVELS: Failed to compute PDH/PDL/PDC for {symbol}: {e}")
            logger.error(f"LEVELS: Traceback: {traceback.format_exc()}")
            pdh = pdl = pdc = float("nan")

        try:
            logger.debug(f"LEVELS: Computing opening range for {symbol}, df5 shape: {df5.shape if df5 is not None else None}")
            orh, orl = levels.opening_range(df5, symbol=symbol)
            orh = float(orh); orl = float(orl)
            logger.debug(f"LEVELS: Computed ORH={orh}, ORL={orl}")
        except Exception as e:
            # CRITICAL FIX: Log opening range computation failures
            import traceback
            logger.error(f"LEVELS: Failed to compute ORH/ORL for {symbol}: {e}")
            logger.error(f"LEVELS: Traceback: {traceback.format_exc()}")
            orh = orl = float("nan")

            # Fallback: compute simple opening range from first few bars
            if validate_df(df5, min_rows=3):
                opening_bars = df5.iloc[:min(3, len(df5))]  # First 15 minutes (3 x 5min bars)
                orh = float(opening_bars['high'].max())
                orl = float(opening_bars['low'].min())
                logger.info(f"LEVELS: Using fallback ORH/ORL for {symbol}: ORH={orh:.2f}, ORL={orl:.2f}")

        out = {"PDH": pdh, "PDL": pdl, "PDC": pdc, "ORH": orh, "ORL": orl}

        # Log levels to screener structured logging for analysis
        valid_levels = {k: v for k, v in out.items() if not pd.isna(v)}
        screener_logger = get_screener_logger()

        if valid_levels and screener_logger:
            screener_logger.log_accept(
                symbol,
                timestamp=now.isoformat(),
                action_type="levels_computed",
                levels_count=len(valid_levels),
                **valid_levels  # PDH, PDL, PDC, ORH, ORL as separate fields
            )
            logger.debug(f"LEVELS: Computed levels for {symbol}: {valid_levels} (total: {len(valid_levels)}/5)")
        elif not valid_levels and screener_logger:
            screener_logger.log_reject(
                symbol,
                "no_valid_levels_computed",
                timestamp=now.isoformat(),
                action_type="levels_computation_failed"
            )
            logger.warning(f"LEVELS: No valid levels computed for {symbol} - structure detection will be skipped")

        self._levels_cache.clear()
        self._levels_cache[key] = out
        return out

    def _fallback_shortlist(self) -> List[str]:
        """Safety net if Stage-0 fails — cheap breakout proxy over last ~2h."""
        out: List[str] = []
        for sym in self.core_symbols:
            df5 = self.agg.get_df_5m_tail(sym, 25)
            if df5 is None or df5.empty or len(df5) < 10:
                continue
            try:
                enriched_df = mi.compute_intraday_breakout_score(df5)
            except Exception as e:
                # CRITICAL FIX: Log breakout score computation failures
                logger.error(f"SCREENER: Failed to compute breakout score for {sym}: {e}")
                continue
            # If enrichment succeeded and DataFrame is not empty, include in fallback
            if not enriched_df.empty:
                out.append(sym)
        return out[:60]

    def _reasons_for(self, sym: str, decisions: List[Tuple[str, Decision]]) -> List[str]:
        for s, d in decisions:
            if s == sym:
                return d.reasons
        return []

    def _compute_percentile_cut(self, ranked: List[Tuple[str, float]], pctl: float) -> float:
        """Return score threshold at the given percentile over the ranked batch."""
        try:
            scores = pd.Series([sc for _, sc in ranked], dtype=float)
            return float(scores.quantile(max(0.0, min(1.0, pctl))))
        except Exception:
            return float("-inf")

    def _should_produce(self, now: datetime) -> bool:
        lp = self._last_produced_at
        if lp is None:
            return True
        return (now - lp).total_seconds() >= self.cfg.producer_min_interval_sec

    def _is_after_cutoff(self, now: datetime) -> bool:
        # Intentional: keep simple "HH:MM" split as you requested
        hh, mm = self.cfg.intraday_cutoff_hhmm.split(":")
        cutoff = dtime(hour=int(hh), minute=int(mm))
        return now.time() >= cutoff

    # ---------- Stage-0 helpers ----------
    def _time_bucket(self, now_ts: pd.Timestamp) -> str:
        md = now_ts.hour * 60 + now_ts.minute
        if md <= 10 * 60 + 30: return "early"
        if md <= 13 * 60 + 30: return "mid"
        return "late"

    def _load_cap_mapping(self) -> dict:
        """
        Load market cap data from nse_all.json. Cached for performance.

        Returns:
            Dict mapping symbol → {"market_cap_cr": float, "cap_segment": str}

        Market Cap Segments (NSE India standards):
        - large_cap: >= Rs.20,000 cr
        - mid_cap: Rs.5,000 - 20,000 cr
        - small_cap: Rs.500 - 5,000 cr
        - micro_cap: < Rs.500 cr (excluded from intraday)
        """
        if hasattr(self, "_cap_map_cache"):
            return self._cap_map_cache

        try:
            import json
            from pathlib import Path
            nse_file = Path(__file__).parent.parent / "nse_all.json"

            with nse_file.open() as f:
                data = json.load(f)

            # Build symbol → cap_data mapping
            cap_map = {}
            for item in data:
                sym = item["symbol"]
                cap_map[sym] = {
                    "market_cap_cr": item.get("market_cap_cr", 0),
                    "cap_segment": item.get("cap_segment", "unknown")
                }

            self._cap_map_cache = cap_map
            logger.info(f"CAP_MAPPING | Loaded market cap data for {len(cap_map)} symbols")
            return cap_map

        except Exception as e:
            logger.warning(f"CAP_MAPPING | Failed to load market cap data: {e}")
            return {}

    def _filter_stage0(self, feats: pd.DataFrame, now_ts, skip_vol_persist: bool = False) -> pd.DataFrame:
        """
        Stage-0 shortlist filter: liquidity + VWAP proximity (time-aware) + momentum clamp.
        Optional: volume persistency & vol ratio if columns are present.

        Args:
            skip_vol_persist: If True, skip volume persistence filter (used during opening bell with <3 bars)
        """
        cfg = load_filters()
        if feats is None or feats.empty:
            return feats
        bkt = self._time_bucket(now_ts)

        vmin = int(cfg.get("scanner_min_bar5_volume", 0))
        vwap_caps = cfg.get("scanner_vwap_bps_caps", {"early": 50, "mid": 60, "late": 100})
        ret1_max = float(cfg.get("scanner_ret1_max", 1.0))
        vp_bars = int(cfg.get("scanner_vol_persist_bars", 2))
        vr_min = float(cfg.get("scanner_vol_ratio_min", 1.2))

        df = feats.copy()
        if "volume" in df.columns and vmin > 0:
            df = df[df["volume"] >= vmin]

        if "dist_to_vwap" in df.columns:
            cap_bps = float(vwap_caps.get(bkt, 100))
            df = df[(df["dist_to_vwap"].abs() * 10000.0) <= cap_bps]

        if "ret_1" in df.columns:
            df = df[df["ret_1"].abs() <= ret1_max]

        # OPENING BELL FIX: Skip volume persistence filter during opening bell (09:20-09:30)
        # With only 1-2 bars, vol_persist_ok would reject all symbols
        if not skip_vol_persist:
            if "vol_persist_ok" in df.columns:
                df = df[df["vol_persist_ok"] >= (1 if vp_bars >= 2 else 0)]
            if "vol_ratio" in df.columns:
                df = df[df["vol_ratio"] >= vr_min]

        # ========== CAP-AWARE LIQUIDITY GATES (Priority 1) ==========
        liq_cfg = cfg.get("liquidity_gates", {})
        if liq_cfg.get("enabled", False):
            # Load market cap mapping
            cap_map = self._load_cap_mapping()

            # Build pass/fail mask for cap-specific volume requirements
            passes_liquidity = []
            for sym in df.index:
                cap_data = cap_map.get(sym, {})
                cap_segment = cap_data.get("cap_segment", "unknown")

                # Exclude micro-caps if configured (institutional standard)
                if liq_cfg.get("exclude_micro_caps", True) and cap_segment == "micro_cap":
                    passes_liquidity.append(False)
                    continue

                # Get segment-specific requirements
                seg_cfg = liq_cfg.get(cap_segment, {})
                if not seg_cfg:
                    passes_liquidity.append(True)  # No config = allow
                    continue

                # Check volume surge requirement (cap-specific)
                # Small-caps need 3x surge, mid-caps 2x, large-caps 1.3x
                vol_mult = df.loc[sym, "vol_ratio"] if "vol_ratio" in df.columns else 1.0
                min_surge = seg_cfg.get("volume_surge_min", 1.0)

                passes = vol_mult >= min_surge
                passes_liquidity.append(passes)

            # Apply filter
            before_count = len(df)
            df = df[passes_liquidity]
            after_count = len(df)

            if before_count > after_count:
                logger.info(f"LIQUIDITY_GATE | Filtered {before_count}→{after_count} symbols "
                           f"({before_count - after_count} failed cap-specific volume requirements)")

        return df

    # ---------- De-dupe ----------
    def _bars_since(self, older: pd.Timestamp, newer: pd.Timestamp) -> int:
        try:
            return int(max(0, (newer - older).total_seconds() // 300))
        except Exception:
            return 9999

    def _dedupe_ok(self, *, sym: str, now_ts: pd.Timestamp, setup_type: Optional[str],
                   score: float, pctl_score: float) -> bool:
        """
        Allow re-entry only if:
          • >= dedupe_cooloff_bars have passed since last acceptance, AND
          • (if required) setup_type changed, AND
          • current score >= max(pctl_score, last_score)  (second entry must be stronger than both the day’s cut and last attempt)
        """
        cfg = load_filters()
        cool = int(cfg.get("dedupe_cooloff_bars", 6))
        need_change = bool(cfg.get("dedupe_require_setup_change", True))
        last = self._last_entry.get(sym)
        if not last:
            return True  # no prior accept → allow

        bars_gap = self._bars_since(last["ts"], now_ts) if isinstance(last.get("ts"), pd.Timestamp) else 9999
        if bars_gap < cool:
            return False

        if need_change and (setup_type is not None) and (last.get("setup") == setup_type):
            return False

        last_score = float(last.get("score") or float("-inf"))
        required = max(pctl_score, last_score)
        return float(score) >= required
    
    def _blocked_by_time_policy(self, now: datetime) -> bool:
        hhmm = f"{now.hour:02d}:{now.minute:02d}"

        # Strict lunch pause from config (already present in your configs)
        if bool(self.raw_cfg.get("enable_lunch_pause")):
            ls = str(self.raw_cfg.get("lunch_start", "12:15"))
            le = str(self.raw_cfg.get("lunch_end", "13:15"))
            if ls <= hhmm <= le:
                return True

        # Opening noise block (configurable window)
        s, e = self._opening_block
        if s and e and s <= hhmm <= e:
            return True

        # Late entry cutoff for NEW entries (keep EOD square-off separate)
        entry_cut = str(self.raw_cfg.get("entry_cutoff_hhmm", "")).strip()
        if entry_cut and hhmm >= entry_cut:
            return True

        return False
