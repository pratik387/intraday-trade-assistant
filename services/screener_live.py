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
from datetime import datetime, time as dtime, timedelta
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import threading

import pandas as pd

from config.filters_setup import load_filters
from config.logging_config import get_agent_logger, get_screener_logger, get_events_decision_logger
from config.env_setup import env
from utils.level_utils import get_previous_day_levels
from utils.dataframe_utils import validate_df, safe_get_last

# ingest / streaming
from services.ingest.stream_client import WSClient
from services.ingest.subscription_manager import SubscriptionManager
from services.ingest.live_tick_handler import LiveTickHandler
from services.ingest.tick_router import TickRouter

# market data bus (shared tick/bar distribution for multi-instance trading)
from market_data.shared_ltp import SharedLTPCache

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
from services.state.orb_cache_persistence import ORBCachePersistence
import uuid


logger = get_agent_logger()

# ---------------------------------------------------------------------
# Worker Pool State (initialized once per worker process)
# ---------------------------------------------------------------------
_worker_decision_gate = None
_worker_daily_cache = {}

def _seed_worker_daily_cache(daily_data_dict):
    """Called once per worker to pre-load daily data cache (avoids re-pickling per bar)."""
    global _worker_daily_cache
    _worker_daily_cache = daily_data_dict

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
        )
    except Exception as e:
        get_agent_logger().exception(f"Worker init failed: {e}")
        raise

def _worker_process_symbol(symbol, df5_data, index_df5_data, levels, now, daily_df=None):
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

def _worker_process_batch(batch_items, index_df5_data, now):
    """Process a batch of symbols. Returns list of (symbol, decision) tuples.

    Data integrity: symbol is carried through each tuple — no positional dependency.
    Per-symbol try/except ensures one failure doesn't kill the entire batch.
    """
    import time as _time
    _t0 = _time.perf_counter()
    global _worker_decision_gate, _worker_daily_cache
    results = []
    for (symbol, df5_data, levels) in batch_items:
        if _worker_decision_gate is None:
            results.append((symbol, None))
            continue
        daily_df = _worker_daily_cache.get(symbol)
        try:
            decision = _worker_decision_gate.evaluate(
                symbol=symbol, now=now,
                df5m_tail=df5_data,
                index_df5m=index_df5_data,
                levels=levels, daily_df=daily_df,
            )
            results.append((symbol, decision))
        except Exception as e:
            from config.logging_config import get_agent_logger
            get_agent_logger().exception(f"Worker batch task failed for {symbol}: {e}")
            results.append((symbol, None))
    _elapsed = _time.perf_counter() - _t0
    from config.logging_config import get_agent_logger
    get_agent_logger().debug("WORKER_BATCH_DONE | %d symbols | %.2fs (%.0fms/sym)",
                           len(batch_items), _elapsed, (_elapsed / max(len(batch_items), 1)) * 1000)
    return results

# ---------------------------------------------------------------------
# Stage-0 Worker Pool State (initialized once per Stage-0 worker process)
# Runs EnergyScanner.compute_features + filter + shortlist in a separate
# process to avoid GIL contention with the MarketDataBus subscriber thread.
# ---------------------------------------------------------------------
_stage0_scanner = None
_stage0_config = None
_stage0_cap_map = None


def _init_stage0_worker(config_dict):
    """
    Initialize Stage-0 worker process (called once per worker).
    Pre-loads EnergyScanner + cap mapping to avoid per-task overhead.
    """
    global _stage0_scanner, _stage0_config, _stage0_cap_map
    try:
        # Suppress console output from worker process — keep only JSONL file loggers.
        import logging
        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(logging.NullHandler())

        # Initialize scanner JSONL logger in subprocess so scanning.jsonl gets written
        log_dir = config_dict.get("_log_dir")
        if log_dir:
            from pathlib import Path
            from config.logging_config import JSONLLogger
            import config.logging_config as _lc
            _lc._scanner_logger = JSONLLogger(Path(log_dir) / "scanning.jsonl", "scanner")

        from services.scan.energy_scanner import EnergyScanner

        _stage0_config = config_dict

        scanner_cfg = config_dict["energy_scanner"]
        _stage0_scanner = EnergyScanner(
            top_k_long=scanner_cfg["top_k_long"],
            top_k_short=scanner_cfg["top_k_short"],
        )

        # Pre-load cap mapping (same logic as ScreenerLive._load_cap_mapping)
        import json
        from pathlib import Path
        nse_file = Path(__file__).parent.parent / "nse_all.json"
        if nse_file.exists():
            with nse_file.open() as f:
                data = json.load(f)
            cap_map = {}
            for item in data:
                raw_sym = item["symbol"]
                sym = f"NSE:{raw_sym[:-3]}" if raw_sym.endswith(".NS") else raw_sym
                cap_map[sym] = {
                    "market_cap_cr": item.get("market_cap_cr", 0),
                    "cap_segment": item.get("cap_segment", "unknown"),
                    "mis_enabled": item.get("mis_enabled", False),
                    "mis_leverage": item.get("mis_leverage"),
                }
            _stage0_cap_map = cap_map
        else:
            _stage0_cap_map = {}

    except Exception as e:
        # Re-enable console logging for error reporting if init fails
        import sys
        logging.getLogger().handlers.clear()
        logging.getLogger().addHandler(logging.StreamHandler(sys.stderr))
        logging.getLogger().exception(f"Stage-0 worker init failed: {e}")
        raise


def _filter_stage0_standalone(feats, now_ts, *, skip_vol_persist, config, cap_map, is_dry_run):
    """
    Stage-0 filter logic extracted for cross-process execution.
    Mirrors ScreenerLive._filter_stage0 exactly, but takes all
    dependencies as explicit parameters instead of reading from self.
    """
    import logging
    _logger = logging.getLogger(__name__)

    if feats is None or feats.empty:
        return feats

    # Inline _time_bucket (avoids self reference)
    md = now_ts.hour * 60 + now_ts.minute
    if md <= 10 * 60 + 30:
        bkt = "early"
    elif md <= 13 * 60 + 30:
        bkt = "mid"
    else:
        bkt = "late"

    vmin = int(config["scanner_min_bar5_volume"])
    vwap_caps = config["scanner_vwap_bps_caps"]
    ret1_max = float(config["scanner_ret1_max"])
    vp_bars = int(config["scanner_vol_persist_bars"])
    vr_min = float(config["scanner_vol_ratio_min"])

    df = feats.copy()
    initial_count = len(df)

    if "volume" in df.columns and vmin > 0:
        df = df[df["volume"] >= vmin]
        if len(df) < initial_count:
            _logger.debug(f"STAGE0_DEBUG | volume filter: {initial_count}→{len(df)} (vmin={vmin})")

    after_vol = len(df)
    if "dist_to_vwap" in df.columns:
        cap_bps = float(vwap_caps[bkt])
        df = df[(df["dist_to_vwap"].abs() * 10000.0) <= cap_bps]
        if len(df) < after_vol:
            _logger.debug(f"STAGE0_DEBUG | vwap filter: {after_vol}→{len(df)} (cap_bps={cap_bps}, bkt={bkt})")

    after_vwap = len(df)
    if "ret_1" in df.columns:
        df = df[df["ret_1"].abs() <= ret1_max]
        if len(df) < after_vwap:
            _logger.debug(f"STAGE0_DEBUG | ret_1 filter: {after_vwap}→{len(df)} (max={ret1_max})")

    if not skip_vol_persist:
        if "vol_persist_ok" in df.columns:
            before_vp = len(df)
            df = df[df["vol_persist_ok"] >= (1 if vp_bars >= 2 else 0)]
            if len(df) < before_vp:
                _logger.debug(f"STAGE0_DEBUG | vol_persist filter: {before_vp}→{len(df)} (vp_bars={vp_bars})")
        if "vol_ratio" in df.columns:
            before_vr = len(df)
            df = df[df["vol_ratio"] >= vr_min]
            if len(df) < before_vr:
                _logger.debug(f"STAGE0_DEBUG | vol_ratio filter: {before_vr}→{len(df)} (vr_min={vr_min})")

    # Log if all symbols were filtered
    if len(df) == 0 and initial_count > 0:
        sample_feats = feats.head(3)
        sample_info = ""
        if not sample_feats.empty:
            for col in ["volume", "dist_to_vwap", "ret_1", "vol_persist_ok", "vol_ratio"]:
                if col in sample_feats.columns:
                    vals = sample_feats[col].tolist()
                    sample_info += f"{col}={vals[:3]} "
        _logger.warning(f"STAGE0_DEBUG | ALL symbols filtered! initial={initial_count} skip_vol_persist={skip_vol_persist} sample: {sample_info}")

    # Cap-aware liquidity gates
    liq_cfg = config["liquidity_gates"]
    if liq_cfg["enabled"] and len(df) > 0:
        passes_liquidity = []
        for _, row in df.iterrows():
            sym = row["symbol"]
            cap_data = cap_map.get(sym, {})
            cap_segment = cap_data.get("cap_segment", "unknown")

            if liq_cfg["exclude_micro_caps"] and cap_segment == "micro_cap":
                passes_liquidity.append(False)
                continue

            seg_cfg = liq_cfg.get(cap_segment, {})
            if not seg_cfg:
                passes_liquidity.append(True)
                continue

            vol_mult = row["vol_ratio"] if "vol_ratio" in df.columns else 1.0
            min_surge = seg_cfg["volume_surge_min"]
            passes = vol_mult >= min_surge
            passes_liquidity.append(passes)

        before_count = len(df)
        df = df[passes_liquidity]
        after_count = len(df)
        if before_count > after_count:
            _logger.info(f"LIQUIDITY_GATE | Filtered {before_count}→{after_count} symbols "
                        f"({before_count - after_count} failed cap-specific volume requirements)")

    # MIS filter (backtest only)
    mis_cfg = config["backtest_mis_filtering"]
    if mis_cfg["enabled"] and is_dry_run and len(df) > 0:
        before_mis = len(df)
        mis_mask = [cap_map.get(sym, {}).get("mis_enabled", False) for sym in df["symbol"]]
        df = df[mis_mask]
        rejected = before_mis - len(df)
        if rejected > 0:
            _logger.info(f"MIS_FILTER | Filtered {before_mis}→{len(df)} ({rejected} non-MIS rejected)")

    return df


def _run_stage0_in_process(df5_tails, levels_by_symbol, now_ts, in_opening_bell, is_dry_run):
    """
    Execute Stage-0 pipeline in a separate process (GIL-free).
    Runs compute_features + filter + select_shortlist.

    Returns (feats_df, shortlist_dict).
    """
    global _stage0_scanner, _stage0_cap_map, _stage0_config

    if _stage0_scanner is None:
        raise RuntimeError("Stage-0 worker not initialized")

    import time
    t0 = time.perf_counter()

    feats_df = _stage0_scanner.compute_features(
        df5_tails,
        lookback_bars=20,
        levels_by_symbol=levels_by_symbol,
        allow_early_scan=in_opening_bell,
    )
    t1 = time.perf_counter()

    feats_df = _filter_stage0_standalone(
        feats_df, now_ts,
        skip_vol_persist=in_opening_bell,
        config=_stage0_config,
        cap_map=_stage0_cap_map,
        is_dry_run=is_dry_run,
    )
    t2 = time.perf_counter()

    shortlist_dict = _stage0_scanner.select_shortlist(feats_df)
    t3 = time.perf_counter()

    timing = {
        "compute": t1 - t0, "filter": t2 - t1, "shortlist": t3 - t2, "total": t3 - t0,
        "filtered": len(feats_df) if feats_df is not None and not feats_df.empty else 0,
        "long": len(shortlist_dict.get("long", [])),
        "short": len(shortlist_dict.get("short", [])),
    }

    return (feats_df, shortlist_dict, timing)


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

    def __init__(self, *, sdk, order_queue: OrderQueue, mis_fetcher=None) -> None:
        self.sdk = sdk
        self.oq = order_queue
        self._mis_fetcher = mis_fetcher

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

        # Core tick handler + LTP cache (standalone — MDS removed)
        self.agg = LiveTickHandler(
            bar_5m_span_minutes=raw.get("bar_5m_span_minutes", 5),
            on_1m_close=self._on_1m_close,
            on_5m_close=self._on_5m_close,
            on_15m_close=self._on_15m_close,
            index_symbols=self._index_symbols(),
        )
        self._shared_ltp_cache = SharedLTPCache(mode="standalone")

        # Shared 5m bar cache (Redis) — prevents duplicate API fetches across instances
        from market_data.shared_5m_cache import Shared5mCache
        redis_url = raw.get("market_data_bus", {}).get("redis_url", "redis://localhost:6379/0")
        self._shared_5m_cache = Shared5mCache(redis_url=redis_url) if not env.DRY_RUN else None

        # WebSocket and tick routing
        self.ws = WSClient(sdk=sdk, on_tick=self.agg.on_tick)
        self.router = TickRouter(on_tick=self.agg.on_tick, token_to_symbol=self._load_core_universe())
        self.ws.on_message(self.router.handle_raw)
        if env.DRY_RUN:
            self.ws.on_close(lambda: self._handle_eod())  # Replay ended = EOD shutdown
        else:
            self.ws.on_close(lambda: logger.warning("WebSocket closed - Kite SDK will auto-reconnect"))
        self.subs = SubscriptionManager(self.ws)

        # Precomputed enriched 5m bars for backtest (loaded from feather cache)
        # In DRY_RUN: loaded at init, served time-filtered during scans
        # In paper/live: empty (API 5m bars enriched at runtime instead)
        self._precomputed_5m: Dict[str, pd.DataFrame] = {}
        if env.DRY_RUN:
            self._load_precomputed_5m()

        # Wire I1 (broker 1m) candles to LiveTickHandler for live signal generation.
        # In backtest, no I1 candles arrive so bars are still built from ticks.
        self.ws.set_i1_candle_listener(self.agg.on_i1_candle)

        # In backtest with precomputed 5m: FeatherTicker fires scan directly via enriched 5m
        # replay, bypassing LiveTickHandler's 1m→5m aggregation.
        if env.DRY_RUN and self._precomputed_5m:
            self.agg._on_5m_close = lambda sym, bar: None  # disable LiveTickHandler scan trigger
            self.ws.set_5m_enriched_listener(self._on_5m_close)  # enriched replay triggers scan

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
        )

        # Directional bias tracker (Nifty green/red → position size modulation)
        from services.gates.directional_bias import DirectionalBiasTracker, set_tracker
        self.directional_bias = DirectionalBiasTracker(raw)
        set_tracker(self.directional_bias)  # Module-level singleton for pipeline access

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
        self._levels_cache_lock = threading.Lock()

        # ORB levels cache: computed once per day at 09:35 and reused for entire day
        # Key: date, Value: Dict[symbol, Dict[str, float]] containing PDH/PDL/PDC/ORH/ORL
        self._orb_levels_cache: Dict = {}
        self._orb_cache_lock = threading.Lock()
        self._orb_recovery_in_progress = False
        self._orb_recovery_thread: Optional[threading.Thread] = None

        # ORB cache persistence for restart recovery
        self._orb_cache_persistence = ORBCachePersistence()
        self._load_orb_cache_from_disk()

        # NEW: de-dupe memory (per symbol last accepted)
        # stores: {symbol: {"ts": pd.Timestamp, "setup": str|None, "score": float}}
        self._last_entry: Dict[str, Dict[str, object]] = {}

        self._eod_done: bool = False
        self._request_exit: bool = False

        # Warmup cache: previous day's raw 5m bars per symbol for indicator stabilization.
        # Loaded once at first API fetch. Ensures parity between paper (API enrichment)
        # and backtest (precomputed enrichment which uses 30-bar warmup).
        self._api_warmup_cache: Dict[str, pd.DataFrame] = {}
        self._api_warmup_loaded: bool = False

        # (precomputed 5m init moved earlier — before WebSocket wiring)
        
        self._opening_block = (
            str(raw.get("opening_block_start_hhmm", "")) or None,
            str(raw.get("opening_block_end_hhmm", "")) or None,
        )

        # Create persistent worker pool for structure detection (avoid 3-5s overhead every 5m)
        # Worker count configurable: default 2 (safe for OCI 1.5 OCPU pods), increase locally for faster runs
        structure_workers = int(raw["structure_detection_workers"])
        local_override = raw.get("structure_detection_workers_local")
        if local_override is not None and not os.environ.get("OCI_RESOURCE_PRINCIPAL_VERSION"):
            structure_workers = int(local_override)
        self._structure_workers = structure_workers
        self._executor = ProcessPoolExecutor(
            max_workers=structure_workers,
            initializer=_init_worker,
            initargs=(self.raw_cfg,)
        )
        self._daily_cache_seeded = False
        logger.info("ScreenerLive: Persistent worker pool created (%d workers)", structure_workers)

        # Create Stage-0 worker pool (1 worker — GIL-free compute_features + filter + shortlist)
        # Pass log directory so scanner logger can be initialized in subprocess
        stage0_cfg = dict(self.raw_cfg)
        from config.logging_config import get_log_directory
        log_dir = get_log_directory()
        if log_dir:
            stage0_cfg["_log_dir"] = str(log_dir)
        self._stage0_executor = ProcessPoolExecutor(
            max_workers=1,
            initializer=_init_stage0_worker,
            initargs=(stage0_cfg,)
        )
        logger.info("ScreenerLive: Stage-0 worker pool created (1 worker)")

        # Timer scan dispatch (paper/live only — backtest fires scans synchronously)
        self._scan_thread: Optional[threading.Thread] = None
        self._scan_running = False

        logger.debug(
            "ScreenerLive init: universe=%d symbols, store5m=%d",
            len(self.core_symbols),
            self.cfg.screener_store_5m_max,
        )

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def start(self) -> None:
        """Connect WS and subscribe index symbols only (scan trigger + directional bias).
        Active trade symbols are subscribed dynamically via add_hot at enqueue time."""
        # Paper/live: lean WebSocket — no core subscriptions, trades added dynamically
        # Backtest: subscribe all symbols (FeatherTicker emulates ticks)
        if not env.DRY_RUN:
            self.subs.set_core(set())
            logger.info("WS_LEAN | Zero core subscriptions. Trade symbols added via add_hot.")
        else:
            self.subs.set_core(self.token_map)

        self.subs.start()
        self.ws.start()

        # Prewarm API warmup cache (paper/live only) before first scan
        # Fetches yesterday's 5m bars from Historical API for all core_symbols
        if not env.DRY_RUN and not self._api_warmup_loaded:
            self._load_api_warmup_cache()

        # Start async scan dispatch worker (live/paper only)
        self._start_scan_worker()

        logger.info("WS connected; core subscriptions scheduled: %d symbols",
                    len(self.subs._core))

    def stop(self) -> None:
        # Stop WebSocket/subscriptions (if in publisher/standalone mode)
        if self.subs:
            try: self.subs.stop()
            except Exception: pass
        if self.ws:
            try: self.ws.stop()
            except Exception: pass

        if self._shared_ltp_cache:
            try: self._shared_ltp_cache.shutdown()
            except Exception: pass

        # Stop async scan dispatch worker
        self._stop_scan_worker()

        # Shutdown Stage-0 worker pool
        try:
            if hasattr(self, '_stage0_executor') and self._stage0_executor:
                self._stage0_executor.shutdown(wait=True, cancel_futures=True)
                logger.info("ScreenerLive: Stage-0 worker pool shut down")
        except Exception as e:
            logger.warning(f"ScreenerLive: Stage-0 worker pool shutdown error: {e}")

        # Shutdown persistent worker pool (structure detection)
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
        DEPRECATED: HTF enhancement is now handled via htf_context parameter
        passed to PipelineOrchestrator.process_setup_candidates().
        The orchestrator applies HTF 15m multipliers in apply_universal_ranking_adjustments().
        This method is kept for reference but is not called.
        """
        logger.debug(f"_enhance_candidates_with_htf called for {symbol} but is DEPRECATED — HTF context flows via orchestrator")
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
                    orl=getattr(candidate, 'orl', None),
                    detected_level=getattr(candidate, 'detected_level', None)
                )

            enhanced.append(adjusted_candidate)

        return enhanced

    def _build_htf_context(self, symbol: str, df5: Optional[pd.DataFrame] = None) -> Optional[Dict[str, Any]]:
        """
        Build HTF (15m) context for pipeline ranking adjustments.

        Returns dict with:
        - htf_trend: "up", "down", or "neutral"
        - htf_volume_surge: True if 15m volume > 1.3x median
        - htf_momentum: normalized momentum score (-1 to 1)
        - htf_exhaustion: True if signs of trend exhaustion on 15m

        If df5 is provided, aggregates 5m → 15m (paper/backtest path).
        Otherwise falls back to BarBuilder's 15m bars (legacy).
        """
        if df5 is not None and not df5.empty and len(df5) >= 6:
            # Aggregate 5m → 15m on the fly (3 bars per 15m)
            df_5m = df5.copy()
            df_5m["bar_15m"] = df_5m.index.floor("15min")
            df15 = df_5m.groupby("bar_15m").agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=("volume", "sum"),
            )
        else:
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

    # ----- Async scan dispatch (live/paper) ---------------------------------
    def _start_scan_worker(self) -> None:
        """Start timer-based scan thread (paper/live only)."""
        if env.DRY_RUN:
            return  # Backtest scan triggered by FeatherTicker enriched 5m callback
        self._scan_running = True
        self._scan_thread = threading.Thread(
            target=self._timer_scan_loop,
            name="ScanTimer",
            daemon=True,
        )
        self._scan_thread.start()
        logger.info("SCAN_DISPATCH | Timer-based scan worker started (5m interval)")

    def _timer_scan_loop(self) -> None:
        """Timer-based scan: fire every 5 minutes aligned to IST bar boundaries.
        Schedule: 09:20 .. last_scan_hhmm IST (bar 09:15 closes at 09:20, etc.)
        Each scan fires at bar_close + min_delay_after_bar_close_sec.
        After the last scan slot, loop only waits for the EOD cutoff (eod_squareoff_hhmm),
        then fires _handle_eod directly. No WebSocket dependency."""
        import time
        from utils.time_util import _now_naive_ist
        from datetime import time as dtime

        min_delay = float(self.raw_cfg.get("api_5m_bars", {}).get("min_delay_after_bar_close_sec", 60))
        market_open = dtime(9, 20)   # First bar 09:15 closes at 09:20

        # Last scan cutoff (fail-fast: required config)
        _lsh, _lsm = str(self.raw_cfg["last_scan_hhmm"]).split(":")
        last_scan_time = dtime(int(_lsh), int(_lsm))

        # EOD squareoff cutoff — fired directly from timer (independent of scan loop)
        _eh, _em = str(self.raw_cfg["eod_squareoff_hhmm"]).split(":")
        eod_time = dtime(int(_eh), int(_em))

        # Pre-compute all scan slots up to last_scan_hhmm (inclusive)
        scan_times = []
        t = datetime(2000, 1, 1, 9, 20)  # First slot at 09:20 (09:15 bar close)
        while t.time() <= last_scan_time:
            scan_times.append(t.time())
            t += timedelta(minutes=5)

        logger.info("SCAN_TIMER | %d scan slots: %s ... %s IST | last_scan=%s eod=%s | delay=%.0fs",
                    len(scan_times), scan_times[0], scan_times[-1],
                    last_scan_time, eod_time, min_delay)

        # Mark all past slots as fired so we only scan FUTURE bars
        now = _now_naive_ist()
        fired_today = set()
        for slot in scan_times:
            slot_dt = now.replace(hour=slot.hour, minute=slot.minute, second=0, microsecond=0)
            target = slot_dt + timedelta(seconds=min_delay)
            if now >= target:
                fired_today.add(slot)

        if fired_today:
            logger.info("SCAN_TIMER | Skipped %d past slots, waiting for next bar", len(fired_today))

        while self._scan_running:
            try:
                now = _now_naive_ist()

                # Outside market hours — sleep and re-check
                if now.time() < market_open or now.time() > dtime(15, 30):
                    time.sleep(10)
                    fired_today.clear()  # Reset for next day
                    continue

                # EOD squareoff trigger — fired directly from timer, independent of scans.
                # Guarantees _handle_eod runs at eod_squareoff_hhmm even if the last scan
                # is still in-flight or slots have drifted.
                if not getattr(self, "_eod_done", False) and now.time() >= eod_time:
                    logger.warning("SCAN_TIMER | EOD cutoff %s reached at wallclock %s — triggering EOD",
                                   eod_time, now.strftime("%H:%M:%S"))
                    self._handle_eod(now)
                    break  # exit timer loop; main loop will tear down via _request_exit

                # Find the next slot that's due
                fired_this_loop = False
                for slot in scan_times:
                    if slot in fired_today:
                        continue
                    # Target fire time = slot + min_delay
                    slot_dt = now.replace(hour=slot.hour, minute=slot.minute, second=0, microsecond=0)
                    target = slot_dt + timedelta(seconds=min_delay)

                    if now >= target:
                        bar_start = slot_dt - timedelta(minutes=5)
                        dummy_bar = pd.Series(
                            {"open": 0, "high": 0, "low": 0, "close": 0, "volume": 0},
                            name=bar_start,
                        )

                        logger.info("SCAN_TIMER | Firing scan for bar %s | wallclock: %s IST",
                                   bar_start.strftime("%H:%M"), now.strftime("%H:%M:%S"))

                        self._run_5m_scan("TIMER", dummy_bar)
                        fired_today.add(slot)
                        fired_this_loop = True
                        break

                if not fired_this_loop:
                    time.sleep(2)

            except Exception as e:
                logger.exception("SCAN_TIMER | Scan failed: %s", e)
                time.sleep(30)

    def _stop_scan_worker(self) -> None:
        """Stop timer scan thread."""
        self._scan_running = False
        if self._scan_thread and self._scan_thread.is_alive():
            self._scan_thread.join(timeout=5.0)
        logger.info("SCAN_DISPATCH | Timer scan worker stopped")

    # ----- 5m scan dispatch (backtest only) --------------------------------
    def _on_5m_close(self, symbol: str, bar_5m: pd.Series) -> None:
        """Backtest-only scan dispatch. Paper uses timer thread."""
        if env.DRY_RUN:
            self._run_5m_scan(symbol, bar_5m)

    def _run_5m_scan(self, symbol: str, bar_5m: pd.Series) -> None:
        """Main scan driver: runs full pipeline for a 5m bar close."""
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

        # ENHANCED LOGGING: Progress tracking with wall-clock for inter-bar gap analysis
        current_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        if self._last_logged_timestamp != current_time_str:
            logger.info("SCANNER_PROGRESS | Bar: %s | wallclock: %s | Stage-0 starting",
                       current_time_str, datetime.now().strftime("%H:%M:%S"))
            self._last_logged_timestamp = current_time_str

        # ---------- Seed daily_df cache in workers (once per session) ----------
        if not self._daily_cache_seeded:
            _t_seed_start = time.perf_counter()
            daily_dict = {}
            for sym in self.core_symbols:
                dd = self.sdk.get_daily(sym, days=210)
                if dd is not None and not dd.empty:
                    daily_dict[sym] = dd
            _t_seed_fetch = time.perf_counter()
            seed_futures = []
            for _ in range(self._structure_workers):
                seed_futures.append(self._executor.submit(_seed_worker_daily_cache, daily_dict))
            for f in seed_futures:
                f.result(timeout=60)
            self._daily_cache_seeded = True
            _t_seed_done = time.perf_counter()
            logger.info("DAILY_CACHE_SEEDED | %d symbols to %d workers | fetch=%.2fs send=%.2fs total=%.2fs",
                       len(daily_dict), self._structure_workers,
                       _t_seed_fetch - _t_seed_start, _t_seed_done - _t_seed_fetch,
                       _t_seed_done - _t_seed_start)

        # ---------- Stage-0: EnergyScanner (single unified path) ----------
        shortlist: List[str] = []
        levels_by_symbol = None  # Initialize so it's available to structure detection phase

        # OPENING BELL FIX: Determine minimum bars needed - 1 during opening bell (09:20-09:30), 3 normally
        current_time = now.time() if hasattr(now, 'time') else now
        from datetime import time as dtime
        in_opening_bell = dtime(9, 20) <= current_time < dtime(9, 30)
        min_bars_for_processing = 1 if in_opening_bell else 3

        # ---------- 5m bars: Paper fetches from V3 Intraday API ----------
        # Paper: fetch native 5m → enrich with warmup → api_df5_cache
        # Backtest: uses _precomputed_5m path (served in Stage-0 data source loop)
        api_df5_cache: Dict[str, pd.DataFrame] = {}
        api_5m_cfg = self.raw_cfg.get("api_5m_bars", {})

        if not env.DRY_RUN:
            if hasattr(self.sdk, "async_fetch_intraday_5m_batch"):
                # Wait for API data availability
                min_delay = float(api_5m_cfg["min_delay_after_bar_close_sec"])
                bar_close_time = now + timedelta(minutes=5)
                from utils.time_util import _now_naive_ist
                wall_now = _now_naive_ist()
                elapsed_since_close = (wall_now - bar_close_time).total_seconds()
                remaining_wait = min_delay - elapsed_since_close
                if remaining_wait > 0:
                    logger.info("API_5M_FETCH | Waiting %.1fs for API data availability", remaining_wait)
                    time.sleep(remaining_wait)

                _t_api_start = time.perf_counter()
                fetch_symbols = list(self.core_symbols)
                bar_ts = now.isoformat()

                # Check shared Redis cache first
                cached = None
                if self._shared_5m_cache and self._shared_5m_cache.enabled:
                    cached = self._shared_5m_cache.get_cached_bars(bar_ts)

                if cached is not None:
                    api_ok = 0
                    for sym in fetch_symbols:
                        if sym in cached and len(cached[sym]) >= min_bars_for_processing:
                            api_df5_cache[sym] = cached[sym]
                            api_ok += 1
                    _t_api_elapsed = time.perf_counter() - _t_api_start
                    logger.info(
                        "API_5M_FETCH | %d ok of %d symbols | %.1fs (redis cache)",
                        api_ok, len(fetch_symbols), _t_api_elapsed,
                    )
                else:
                    rps = float(api_5m_cfg["rps"])
                    concurrency = int(api_5m_cfg["concurrency"])
                    try:
                        import asyncio
                        raw_api = asyncio.run(
                            self.sdk.async_fetch_intraday_5m_batch(
                                fetch_symbols, concurrency=concurrency, rps=rps
                            )
                        )
                    except Exception as e:
                        logger.warning("API_5M_FETCH | async batch failed: %s", e)
                        raw_api = {}

                    from services.indicators.bar_enrichment import enrich_5m_bars

                    if not self._api_warmup_loaded:
                        self._load_api_warmup_cache()

                    today_ts = pd.Timestamp(now).normalize() if now else None
                    api_ok, api_fail, warmup_used = 0, 0, 0

                    for sym, df_api in raw_api.items():
                        if df_api is not None and len(df_api) >= min_bars_for_processing:
                            warmup = self._api_warmup_cache.get(sym)
                            if warmup is not None and not warmup.empty:
                                combined = pd.concat([warmup, df_api])
                                df_api = enrich_5m_bars(combined, session_date=today_ts)
                                warmup_used += 1
                            else:
                                df_api = enrich_5m_bars(df_api)
                            api_df5_cache[sym] = df_api
                            api_ok += 1
                        else:
                            api_fail += 1

                    if self._shared_5m_cache and api_df5_cache:
                        self._shared_5m_cache.store_bars(bar_ts, api_df5_cache)

                    _t_api_elapsed = time.perf_counter() - _t_api_start
                    # Surface 429 count from the batch fetcher for monitoring
                    throttle_count = getattr(self.sdk, '_last_batch_429s', 0)
                    throttle_info = f" | 429s: {throttle_count}" if throttle_count > 0 else ""
                    logger.info(
                        "API_5M_FETCH | %d ok, %d failed of %d symbols | %.1fs (async, %s RPS) | warmup: %d/%d%s",
                        api_ok, api_fail, len(fetch_symbols), _t_api_elapsed,
                        rps, warmup_used, api_ok, throttle_info,
                    )

        try:
            # Build df5_by_symbol from enriched data (API for paper, precomputed for backtest)
            df5_by_symbol: Dict[str, pd.DataFrame] = {}
            for s in self.core_symbols:
                # Data source:
                #  Paper/live: api_df5_cache (V3 Intraday API + enrichment)
                #  Backtest:   _precomputed_5m (enriched feather cache)
                if s in api_df5_cache:
                    df5 = api_df5_cache[s]
                elif self._precomputed_5m and s in self._precomputed_5m:
                    df5 = self._get_precomputed_5m(s, now, self.cfg.screener_store_5m_max)
                else:
                    continue  # No data source available — skip symbol

                if validate_df(df5, min_rows=min_bars_for_processing):
                    df5_by_symbol[s] = df5
            if df5_by_symbol:
                # Compute ORB levels once at 09:40 and cache for entire day
                levels_by_symbol = self._compute_orb_levels_once(now, df5_by_symbol)

                # GIL FIX: Trim to 20-bar tails before pickling (55MB → 2.2MB)
                # compute_features only needs lookback_bars=20, not the full 500
                df5_tails = {
                    sym: df.tail(20).copy()
                    for sym, df in df5_by_symbol.items()
                }

                # Submit Stage-0 to separate process (GIL released during wait)
                # future.result() is a C-level wait on pipe — subscriber thread runs freely
                stage0_future = self._stage0_executor.submit(
                    _run_stage0_in_process,
                    df5_tails,
                    levels_by_symbol,
                    now,
                    in_opening_bell,
                    bool(env.DRY_RUN),
                )
                _, shortlist_dict, stage0_timing = stage0_future.result(timeout=60)
                shortlist = shortlist_dict.get("long", []) + shortlist_dict.get("short", [])
                logger.info(
                    "STAGE0_PROCESS | compute=%.2fs filter=%.2fs shortlist=%.2fs total=%.2fs | "
                    "filtered=%d long=%d short=%d",
                    stage0_timing["compute"], stage0_timing["filter"],
                    stage0_timing["shortlist"], stage0_timing["total"],
                    stage0_timing["filtered"], stage0_timing["long"], stage0_timing["short"],
                )
        except Exception as e:
            logger.exception("Stage-0 process failed; fallback shortlist: %s", e)
            shortlist = self._fallback_shortlist()

        # ENHANCED LOGGING: Stage-0 completion with accurate counts
        _t_scanner_end = time.perf_counter()
        eligible_symbols = len(df5_by_symbol) if df5_by_symbol else 0
        total_symbols = len(self.core_symbols)
        shortlist_count = len(shortlist)

        api_sourced = sum(1 for s in df5_by_symbol if s in api_df5_cache)
        precomputed_sourced = sum(1 for s in df5_by_symbol if s not in api_df5_cache and self._precomputed_5m and s in self._precomputed_5m)
        bb_sourced = len(df5_by_symbol) - api_sourced - precomputed_sourced
        logger.info("SCANNER_COMPLETE | Processed %d eligible of %d total symbols → %d shortlisted (%.1f%%) | "
                   "data_source: api=%d precomputed=%d barbuilder=%d | Stage-0→Gates | TIME: %.2fs",
                   eligible_symbols, total_symbols, shortlist_count,
                   (shortlist_count/max(eligible_symbols,1))*100,
                   api_sourced, precomputed_sourced, bb_sourced, _t_scanner_end - _t_bar_start)
        if not shortlist:
            return

        # ---------- Directional bias: init prev_close once + update every scan ----------
        if self.directional_bias.enabled:
            index_sym = self.raw_cfg["directional_bias"]["index_symbol"]

            # Set prev_close once (first scan only)
            if self.directional_bias.prev_close is None:
                if env.DRY_RUN:
                    self.directional_bias.set_prev_close_for_date(now)
                else:
                    try:
                        pdc = self.sdk.get_prevday_levels(index_sym).get("PDC", float("nan"))
                        if pdc and not pd.isna(pdc):
                            self.directional_bias.set_prev_close(pdc)
                        else:
                            logger.warning(f"DIR_BIAS | No prev close available for {index_sym}")
                    except Exception as e:
                        logger.error(f"DIR_BIAS | Failed to fetch prev close for {index_sym}: {e}")

            # Update direction every scan (same logic for live and backtest)
            if self.directional_bias.prev_close is not None:
                if env.DRY_RUN:
                    index_price = self.directional_bias.get_backtest_price_at(now)
                else:
                    index_price = self._shared_ltp_cache.get_ltp(index_sym)
                if index_price is not None:
                    self.directional_bias.update_price(index_price)

        # ---------- Gate per candidate (structure + regime + events + news) ----------
        index_df5 = self._index_df5()
        decisions: List[Tuple[str, Decision]] = []
        screener_logger = get_screener_logger()

        # PARALLEL STRUCTURE DETECTION (Phase 1 Optimization)
        # Use ProcessPoolExecutor to parallelize structure detection across symbols
        # This reduces 50s bottleneck to ~15s (3.3x speedup)

        # Prepare data for parallel processing
        # daily_df is cached in worker processes (seeded once at session start)
        symbol_data_map = {}
        for sym in shortlist:
            # Data source:
            #  Paper/live: api_df5_cache
            #  Backtest:   _precomputed_5m
            if sym in api_df5_cache:
                df5 = api_df5_cache[sym]
            elif self._precomputed_5m and sym in self._precomputed_5m:
                df5 = self._get_precomputed_5m(sym, now, self.cfg.screener_store_5m_max)
            else:
                continue  # No enriched 5m data available
            if not validate_df(df5, min_rows=min_bars_for_processing):
                continue
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

            symbol_data_map[sym] = (df5, lvl)

        if not symbol_data_map:
            logger.info("GATES_COMPLETE | No symbols with sufficient data")
            return

        _t_data_prep_end = time.perf_counter()
        logger.info("DATA_PREP_COMPLETE | Prepared %d symbols | TIME: %.2fs",
                   len(symbol_data_map), _t_data_prep_end - _t_scanner_end)

        # BATCH SUBMISSION: Submit symbols in batches of ~50 to reduce IPC overhead
        # index_df5 pickled once per batch (not per symbol), daily_df read from worker cache
        BATCH_SIZE = 50
        all_items = [
            (sym, df5, lvl)
            for sym, (df5, lvl) in symbol_data_map.items()
        ]
        batches = [all_items[i:i + BATCH_SIZE] for i in range(0, len(all_items), BATCH_SIZE)]
        _t_submit_start = time.perf_counter()
        futures = []
        for batch in batches:
            future = self._executor.submit(_worker_process_batch, batch, index_df5, now)
            futures.append((future, {s for s, _, _ in batch}))
        _t_submit_end = time.perf_counter()
        logger.info("BATCH_SUBMIT | %d batches (%d symbols, batch_size=%d) | submit_time=%.2fs",
                   len(batches), len(all_items), BATCH_SIZE, _t_submit_end - _t_submit_start)

        # Collect results from batch futures
        try:
            for future, expected_syms in futures:
                try:
                    batch_results = future.result()
                except Exception as e:
                    logger.exception(f"Batch processing failed: {e}")
                    continue

                # DATA INTEGRITY CHECK: Verify batch completeness and symbol set
                returned_syms = {s for s, _ in batch_results}
                assert returned_syms == expected_syms, \
                    f"Batch symbol mismatch! Expected {expected_syms}, got {returned_syms}"

                for sym, decision in batch_results:
                    if decision is None:
                        logger.debug(f"Structure detection returned None for {sym}")
                        continue

                    df5 = symbol_data_map[sym][0]  # Get original df5 for logging

                    if not decision.accept:
                        top_reason = next((r for r in decision.reasons if r.startswith("regime_block:")), None) or \
                                     (decision.reasons[0] if decision.reasons else "reject")
                        logger.debug(
                            "DECISION:REJECT sym=%s setup=%s regime=%s reason=%s | all=%s",
                            sym, decision.setup_type, decision.regime, top_reason, ";".join(decision.reasons),
                        )

                        # Log screener rejection with detailed context
                        if screener_logger:
                            screener_logger.log_reject(
                                sym,
                                top_reason,
                                timestamp=now.isoformat(),
                                setup_type=decision.setup_type or "unknown",
                                regime=decision.regime or "unknown",
                                all_reasons=decision.reasons,
                                structure_confidence=getattr(decision, 'structure_confidence', 0),
                                current_price=df5['close'].iloc[-1] if not df5.empty else 0,
                                regime_diagnostics=getattr(decision, 'regime_diagnostics', None),
                            )
                        continue

                    logger.debug(
                        "DECISION:ACCEPT sym=%s setup=%s regime=%s size_mult=%.2f hold_bars=%d | %s",
                        sym, decision.setup_type, decision.regime, decision.size_mult, decision.min_hold_bars,
                        ";".join(decision.reasons),
                    )

                    # Log screener acceptance with detailed context
                    if screener_logger:
                        screener_logger.log_accept(
                            sym,
                            timestamp=now.isoformat(),
                            setup_type=decision.setup_type or "unknown",
                            regime=decision.regime or "unknown",
                            size_mult=decision.size_mult,
                            min_hold_bars=decision.min_hold_bars,
                            all_reasons=decision.reasons,
                            structure_confidence=getattr(decision, 'structure_confidence', 0),
                            current_price=float(df5['close'].iloc[-1]) if not df5.empty else 0,
                            vwap=float(df5.get('vwap', pd.Series([0])).iloc[-1]) if not df5.empty else 0,
                            regime_diagnostics=getattr(decision, 'regime_diagnostics', None),
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
        logger.info("GATES_COMPLETE | %d→%d symbols (%.1f%%) | Gates→Orchestrator", shortlist_count, gate_accept_count, (gate_accept_count/max(shortlist_count,1))*100)
        if not decisions:
            return

        dec_map = {s: d for (s, d) in decisions}

        # ---------- Pipeline Orchestrator: Ranking + Planning ----------
        # Orchestrator handles: screening + gates + quality + ranking + entry + targets
        # Returns plans already sorted by ranking score
        _t_orch_start = time.perf_counter()
        logger.info("ORCHESTRATOR | Processing %d symbols via pipeline orchestrator", len(decisions))

        max_trades_per_cycle = int(self.raw_cfg["max_trades_per_cycle"])
        trades_planned = 0
        events_logger = get_events_decision_logger()

        eligible_plans: List[Tuple[str, Dict, float]] = []  # (symbol, plan, score)

        for sym, decision in decisions:
            df5 = symbol_data_map.get(sym, (None,))[0]
            lvl = symbol_data_map.get(sym, (None, {}))[1]
            # daily_df only needed for ~5-20 accepted symbols (not 800), fetch from sdk cache
            daily_df = self.sdk.get_daily(sym, days=210)

            if df5 is None:
                continue

            setup_candidates = getattr(decision, 'setup_candidates', None)
            if not setup_candidates:
                continue

            # Build HTF context from 15m data (aggregated from 5m enriched bars)
            htf_context = self._build_htf_context(sym, df5=df5)

            # Compute daily_score from daily_df for daily/intraday score weighting
            daily_score = 0.0
            if daily_df is not None and len(daily_df) >= 20:
                try:
                    _d_close = float(daily_df["close"].iloc[-1])
                    _d_sma20 = float(daily_df["close"].tail(20).mean())
                    _d_atr = float((daily_df["high"] - daily_df["low"]).tail(14).mean())
                    if _d_atr > 0:
                        daily_score = max(-1.0, min(1.0, (_d_close - _d_sma20) / _d_atr))
                except Exception:
                    daily_score = 0.0

            try:
                plan = process_setup_candidates(
                    symbol=sym,
                    df5m=df5,
                    levels=lvl,
                    regime=decision.regime,
                    now=now,
                    candidates=setup_candidates,
                    daily_df=daily_df,
                    htf_context=htf_context,
                    regime_diagnostics=getattr(decision, 'regime_diagnostics', None),
                    daily_score=daily_score
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

        _t_orch_end = time.perf_counter()
        logger.info("ORCHESTRATOR_COMPLETE | %d eligible plans from %d decisions | TIME: %.2fs",
                   len(eligible_plans), len(decisions), _t_orch_end - _t_orch_start)

        # ---------- Process eligible plans → Execution ----------
        for i, (sym, plan, score, decision) in enumerate(eligible_plans):
            strategy_type = plan.get("strategy", "unknown")
            df5 = symbol_data_map.get(sym, (None,))[0]

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
                if events_logger is not None:
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
                if events_logger is not None:
                    events_logger.log_reject(sym, "zero_quantity", timestamp=now.isoformat(), qty=qty, strategy_type=strategy_type or "unknown")
                continue

            # 3) Bias → side
            bias = str(plan.get("bias", "")).lower()
            if bias not in ("long", "short"):
                logger.info("SKIP %s: bad bias=%r", sym, bias)
                if events_logger is not None:
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
                if events_logger is not None:
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
            try:
                diag_event_log.log_decision(symbol=plan["symbol"], now=now, plan=plan, features=features, decision=decision_dict)
            except Exception as _diag_err:
                logger.warning("diag_event_log.log_decision failed for %s: %s", plan["symbol"], _diag_err)

            exec_item = {
                "symbol": plan["symbol"],
                "plan": {
                    "symbol": plan["symbol"],
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
                    "setup_type": plan.get("strategy", ""),  # Alias: exit_executor reads setup_type
                    "regime": plan.get("regime", ""),
                    "quality": plan.get("quality"),
                    "levels": plan.get("levels"),
                    "category": plan.get("category", ""),
                    "cap_segment": plan.get("sizing", {}).get("cap_segment", ""),
                    "mis_leverage": plan.get("sizing", {}).get("mis_leverage", 1.0),
                    "bias": plan.get("bias"),
                    "indicators": plan.get("indicators"),
                },
                "meta": plan,
            }

            # Check trades per cycle limit (only for live trading)
            if trades_planned >= max_trades_per_cycle:
                logger.info("CYCLE:LIMIT_REACHED %d/%d trades - skipping %s", trades_planned, max_trades_per_cycle, sym)
                if events_logger is not None:
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
            if events_logger is not None:
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

            # Subscribe to this symbol's ticks for execution layer (entry zone, SL, targets)
            sym_token = self.symbol_map.get(sym)
            if sym_token is not None and self.subs:
                self.subs.add_hot([int(sym_token)])

            trades_planned += 1
            logger.info("ENQUEUE %s score=%.3f count=%d/%d reasons=%s", sym, score, trades_planned, max_trades_per_cycle, ";".join(self._reasons_for(sym, decisions)))

        # Final timing for entire bar processing
        _t_bar_end = time.perf_counter()
        logger.info("BAR_COMPLETE | Total: %.2fs | Scanner: %.2fs | DataPrep: %.2fs | Structure: %.2fs | Orchestrator: %.2fs | Execution: %.2fs | wallclock: %s",
                   _t_bar_end - _t_bar_start,
                   _t_scanner_end - _t_bar_start,
                   _t_data_prep_end - _t_scanner_end,
                   _t_structure_end - _t_data_prep_end,
                   _t_orch_end - _t_structure_end,
                   _t_bar_end - _t_orch_end,
                   datetime.now().strftime("%H:%M:%S"))

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
            # Filter out ETFs/index funds — covers *ETF, *BEES, *IETF, LIQUID*
            all_symbols = list(self.symbol_map.keys())
            def _is_etf(sym: str) -> bool:
                bare = sym.replace("NSE:", "").upper()
                return (bare.endswith("ETF") or bare.endswith("BEES")
                        or bare.endswith("IETF") or bare.startswith("LIQUID"))
            self.core_symbols = [
                sym for sym in all_symbols if not _is_etf(sym)
            ]
            etf_count = len(all_symbols) - len(self.core_symbols)
            if etf_count > 0:
                logger.info(f"ETF_FILTER | Excluded {etf_count} ETF symbols from trading universe")

            # Early MIS filter — reduces WS subscriptions, Stage-0, daily cache, ORB
            mis_filter_cfg = self.raw_cfg.get("early_mis_universe_filter", {})
            if mis_filter_cfg.get("enabled", False) and self._mis_fetcher and self._mis_fetcher.is_loaded():
                before = len(self.core_symbols)
                self.core_symbols = [s for s in self.core_symbols if self._mis_fetcher.is_mis_allowed(s)]
                filtered_set = set(self.core_symbols)
                self.token_map = {tok: sym for tok, sym in self.token_map.items() if sym in filtered_set}
                self.symbol_map = {sym: tok for sym, tok in self.symbol_map.items() if sym in filtered_set}
                removed = before - len(self.core_symbols)
                logger.info(f"MIS_UNIVERSE | core_symbols: {before} -> {len(self.core_symbols)} ({removed} non-MIS removed)")
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

    def _load_orb_cache_from_disk(self) -> None:
        """
        Load ORB levels cache from disk on startup.

        If the server restarts after 09:40, the opening range bars (09:15-09:30)
        may not be available from the broker's intraday API. This loads the
        previously computed ORB levels from disk.
        """
        try:
            cached_levels = self._orb_cache_persistence.load_today()
            if cached_levels:
                today = datetime.now().date()
                with self._orb_cache_lock:
                    self._orb_levels_cache[today] = cached_levels
                logger.info(
                    f"ORB_CACHE | Loaded {len(cached_levels)} symbols from disk cache on startup"
                )
        except Exception as e:
            logger.warning(f"ORB_CACHE | Failed to load from disk on startup: {e}")

    def _save_orb_cache_to_disk(self, session_date, levels_by_symbol: Dict[str, Dict[str, float]]) -> None:
        """Save ORB levels cache to disk for restart recovery."""
        try:
            self._orb_cache_persistence.save(session_date, levels_by_symbol)
        except Exception as e:
            logger.warning(f"ORB_CACHE | Failed to save to disk: {e}")

    def _compute_orb_levels_once(self, now, df5_by_symbol: Dict[str, pd.DataFrame]) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Compute ORH/ORL/PDH/PDL/PDC for all symbols once at 09:35 and cache for the day.

        This is a critical performance optimization:
        - ORB (Opening Range Breakout) window is 09:15-09:30
        - ORH/ORL values are finalized at 09:30 and don't change for rest of day
        - Computing levels for all 1992 symbols takes ~54s in OCI (2 CPUs)
        - By computing once at 09:35 instead of on every bar, we save ~30 minutes per day

        Late Start Recovery:
        - If server starts after 09:30 (e.g., 10:00 AM), 5m bars for opening range are missing
        - Detect this by checking if we have bars before 09:35 in df5_by_symbol
        - Trigger recovery via _recover_orb_levels_from_historical() which fetches 1m data from broker

        Returns:
            Dict[symbol, Dict[str, float]] if computed/cached, None if before 09:35
        """
        if not now:
            return None

        from datetime import time as dtime  # Import at function scope for time comparisons

        session_date = now.date()
        current_time = now.time()
        session_date_str = session_date.isoformat() if hasattr(session_date, 'isoformat') else str(session_date)

        # Check if already computed for today
        if session_date in self._orb_levels_cache:
            return self._orb_levels_cache[session_date]

        # Only compute at or after 09:40 (ensures ORB data 09:15-09:30 is complete)
        # By 09:40, more symbols have sufficient data for reliable ORH/ORL calculation
        if current_time < dtime(9, 40):
            return None

        import time as time_module
        start_time = time_module.perf_counter()

        # LATE START DETECTION: Check if we have opening range bars (09:15-09:30)
        # If not, we likely started late and need to recover from historical data
        has_opening_range_bars = False
        orb_end_time = dtime(9, 30)

        for sym, df5 in df5_by_symbol.items():
            if df5 is not None and len(df5) >= 3:
                # Check if any bar is from before 09:30 (indicating we have opening range data)
                earliest_bar_time = df5.index[0].time() if hasattr(df5.index[0], 'time') else None
                if earliest_bar_time and earliest_bar_time < orb_end_time:
                    has_opening_range_bars = True
                    break

        if not has_opening_range_bars:
            # Late start detected - check if recovery is even worth it
            # ORB setups are disabled after 10:30 (orb_structure.py:86), so skip recovery if too late
            ORB_CUTOFF = dtime(10, 30)

            if current_time >= ORB_CUTOFF:
                logger.info(
                    f"ORB_CACHE | Late start at {current_time} is AFTER ORB cutoff ({ORB_CUTOFF}). "
                    f"Computing PDH/PDL/PDC only (ORH/ORL will be NaN - ORB setups disabled anyway)."
                )
                # Still compute PDH/PDL/PDC from pre-warmed daily data for level-based setups
                # Only ORH/ORL will be NaN (which is fine since ORB setups are disabled after 10:30)
                # IMPORTANT: Iterate over ALL core_symbols, not just df5_by_symbol
                # PDH/PDL/PDC come from daily cache, don't need 5m bars
                levels_by_symbol = {}
                for sym in self.core_symbols:
                    try:
                        df5 = df5_by_symbol.get(sym)  # May be None, that's OK for PDH/PDL/PDC
                        lvl = self._levels_for(sym, df5, now)
                        # Keep PDH/PDL/PDC even if ORH/ORL are NaN
                        levels_by_symbol[sym] = lvl
                    except Exception as e:
                        logger.debug(f"ORB_CACHE | Late start: Failed to compute levels for {sym}: {e}")
                        levels_by_symbol[sym] = {}

                # Cache for the day
                self._orb_levels_cache[session_date] = levels_by_symbol
                self._save_orb_cache_to_disk(session_date, levels_by_symbol)

                # (MDS publisher mode removed — ORB levels computed locally)

                valid_pdh = sum(1 for v in levels_by_symbol.values() if not pd.isna(v.get("PDH")))
                logger.info(f"ORB_CACHE | Late start: Computed PDH/PDL/PDC for {valid_pdh}/{len(levels_by_symbol)} symbols")
                return levels_by_symbol

            # Still within ORB window - spawn background thread for recovery
            # This allows the app to continue processing non-ORB setups immediately
            with self._orb_cache_lock:
                if self._orb_recovery_in_progress:
                    logger.debug("ORB_CACHE | Background recovery already in progress, skipping")
                    return {}

                if session_date in self._orb_levels_cache:
                    # Recovery completed by background thread
                    return self._orb_levels_cache[session_date]

                # Start background recovery
                self._orb_recovery_in_progress = True
                logger.warning(
                    f"ORB_CACHE | Late start at {current_time} but BEFORE ORB cutoff ({ORB_CUTOFF}). "
                    f"Starting BACKGROUND recovery (app continues running)."
                )

                self._orb_recovery_thread = threading.Thread(
                    target=self._background_orb_recovery,
                    args=(session_date,),
                    name="ORB-Recovery",
                    daemon=True
                )
                self._orb_recovery_thread.start()

            # Return empty cache so app continues - ORB setups won't work until recovery completes
            return {}

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

        # Persist to disk for restart recovery
        self._save_orb_cache_to_disk(session_date, levels_by_symbol)

        # (MDS publisher mode removed — ORB levels computed locally)

        elapsed = time_module.perf_counter() - start_time
        logger.info(
            f"ORB_CACHE | Cached levels for {success_count} symbols (failed: {fail_count}) | "
            f"Session: {session_date} | Time: {elapsed:.2f}s | "
            f"This is a ONE-TIME cost - all subsequent bars will use cached values"
        )

        return levels_by_symbol

    def _recover_orb_levels_from_historical(self, session_date) -> Dict[str, Dict[str, float]]:
        """
        Recover ORH/ORL from historical 1-minute data when server starts late.

        This handles the case when the server starts after 09:30 (e.g., 10:00 AM)
        and misses the opening range bars. We fetch historical 1m data from the
        broker API for the 09:15-09:30 window and compute ORH/ORL directly.

        PDH/PDL/PDC are computed normally from daily data (unaffected by late start).

        Returns:
            Dict[symbol, Dict[str, float]] with ORH, ORL, PDH, PDL, PDC
        """
        import time as time_module
        from datetime import datetime as dt, time as dtime

        # Check if SDK supports historical 1m fetch
        if not hasattr(self.sdk, 'get_historical_1m'):
            logger.warning("ORB_RECOVERY | SDK doesn't support get_historical_1m. ORB levels unavailable.")
            return {}

        start_time = time_module.perf_counter()
        logger.info(f"ORB_RECOVERY | Late start detected. Fetching historical 1m data for ORB window (09:15-09:30)")

        orb_start = dt.combine(session_date, dtime(9, 15))
        orb_end = dt.combine(session_date, dtime(9, 30))

        levels_by_symbol = {}
        success_count = 0
        fail_count = 0

        for sym in self.core_symbols:
            try:
                # Fetch 1m historical data for opening range window
                df_1m = self.sdk.get_historical_1m(sym, orb_start, orb_end)

                if df_1m is None or len(df_1m) < 10:  # Need at least 10 bars for reliable ORH/ORL
                    logger.debug(f"ORB_RECOVERY | {sym}: Insufficient 1m data ({len(df_1m) if df_1m is not None else 0} bars)")
                    orh, orl = float("nan"), float("nan")
                else:
                    orh = float(df_1m["high"].max())
                    orl = float(df_1m["low"].min())

                # PDH/PDL/PDC from daily data (works normally)
                daily = self.sdk.get_daily(sym, days=210)
                level_dict = get_previous_day_levels(
                    daily_df=daily,
                    session_date=session_date,
                    fallback_df=None,
                    enable_fallback=False
                )
                pdh = level_dict.get("PDH", float("nan"))
                pdl = level_dict.get("PDL", float("nan"))
                pdc = level_dict.get("PDC", float("nan"))

                levels_by_symbol[sym] = {
                    "ORH": orh,
                    "ORL": orl,
                    "PDH": pdh,
                    "PDL": pdl,
                    "PDC": pdc,
                }

                if not (pd.isna(orh) or pd.isna(orl)):
                    success_count += 1
                else:
                    fail_count += 1

            except Exception as e:
                logger.warning(f"ORB_RECOVERY | {sym}: Failed - {e}")
                levels_by_symbol[sym] = {
                    "ORH": float("nan"), "ORL": float("nan"),
                    "PDH": float("nan"), "PDL": float("nan"), "PDC": float("nan")
                }
                fail_count += 1

        elapsed = time_module.perf_counter() - start_time
        logger.info(
            f"ORB_RECOVERY | Complete. Recovered ORH/ORL for {success_count}/{len(self.core_symbols)} symbols | "
            f"Failed: {fail_count} | Time: {elapsed:.2f}s"
        )

        return levels_by_symbol

    def _background_orb_recovery(self, session_date) -> None:
        """
        Background thread wrapper for ORB level recovery.

        This runs in a daemon thread so the main app can continue processing
        non-ORB setups while we fetch historical 1m data from the broker.

        Based on code analysis:
        - Only ORBStructure REQUIRES ORH/ORL (returns rejection without them)
        - All other structures (support_bounce, resistance_bounce, failure_fade,
          level_breakout, vwap, momentum, etc.) work fine without ORH/ORL

        So this allows:
        1. Non-ORB setups to work immediately with PDH/PDL/PDC
        2. ORB setups to enable once recovery completes
        """
        try:
            logger.info(f"ORB_BACKGROUND_RECOVERY | Starting for {session_date} | core_symbols: {len(self.core_symbols)}")

            # Do the actual recovery (fetches 1m data, computes ORH/ORL)
            levels_by_symbol = self._recover_orb_levels_from_historical(session_date)

            # Update cache atomically
            with self._orb_cache_lock:
                self._orb_levels_cache[session_date] = levels_by_symbol
                self._orb_recovery_in_progress = False

            # Persist to disk for restart recovery
            self._save_orb_cache_to_disk(session_date, levels_by_symbol)

            # (MDS publisher mode removed — ORB levels computed locally)

            orb_count = sum(1 for v in levels_by_symbol.values()
                           if not (pd.isna(v.get("ORH")) or pd.isna(v.get("ORL"))))
            logger.info(
                f"ORB_BACKGROUND_RECOVERY | Complete | {orb_count}/{len(levels_by_symbol)} symbols with valid ORH/ORL | "
                f"ORB setups now ENABLED"
            )

        except Exception as e:
            logger.exception(f"ORB_BACKGROUND_RECOVERY | Failed: {e}")
            with self._orb_cache_lock:
                self._orb_recovery_in_progress = False
                # Cache empty dict so we don't retry
                self._orb_levels_cache[session_date] = {}

    def _levels_for(self, symbol: str, df5: pd.DataFrame, now) -> Dict[str, float]:
        """Prev-day PDH/PDL/PDC and today ORH/ORL (cached per (symbol, session_date))."""
        try:
            session_date = df5.index[-1].date() if validate_df(df5) else None
        except Exception:
            session_date = None
        key = (symbol, session_date)
        with self._levels_cache_lock:
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

        # Only cache if we have at least PDH+PDL (essential for structure detection).
        # If daily data wasn't available yet (SDK warming up), don't cache NaN —
        # allow retry next 5m cycle when data may be ready.
        # ORH/ORL being NaN is acceptable to cache (opening range is fixed from first bars).
        has_prev_day = not (pd.isna(pdh) or pd.isna(pdl))
        if has_prev_day:
            with self._levels_cache_lock:
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

    def _load_api_warmup_cache(self) -> None:
        """
        Fetch previous trading day's 5m bars from V3 Historical API for all core_symbols.
        Used to prepend warmup bars before enrich_5m_bars() at runtime, ensuring indicator
        stabilization (ADX/RSI/BB_width) for the first scan of the day.

        Self-contained: no filesystem dependency, no cross-dependency with backtest cache.
        Paper is fully self-sufficient.
        """
        import time as time_module
        import asyncio
        from utils.time_util import _now_naive_ist

        _t0 = time_module.perf_counter()
        warmup_bars = 30  # Number of bars to retain per symbol

        # Find previous trading day (skip weekends, limited backward)
        today_dt = _now_naive_ist().date()
        from datetime import timedelta as _td
        prev_day = today_dt - _td(days=1)
        # Skip weekends
        while prev_day.weekday() >= 5:  # Saturday=5, Sunday=6
            prev_day -= _td(days=1)

        from_date = prev_day.isoformat()
        to_date = prev_day.isoformat()

        logger.info("API_WARMUP_CACHE | Fetching yesterday's 5m bars (%s) from Historical API", from_date)

        if not hasattr(self.sdk, "async_fetch_historical_5m_batch"):
            logger.warning("API_WARMUP_CACHE | SDK has no async_fetch_historical_5m_batch — skipping warmup")
            self._api_warmup_loaded = True
            return

        api_5m_cfg = self.raw_cfg.get("api_5m_bars", {})
        rps = float(api_5m_cfg.get("rps", 15))
        concurrency = int(api_5m_cfg.get("concurrency", 30))

        try:
            raw = asyncio.run(
                self.sdk.async_fetch_historical_5m_batch(
                    list(self.core_symbols), from_date, to_date,
                    concurrency=concurrency, rps=rps,
                )
            )
        except Exception as e:
            logger.exception("API_WARMUP_CACHE | async batch failed: %s", e)
            raw = {}

        loaded = 0
        for sym, df in raw.items():
            if df is not None and len(df) > 0:
                self._api_warmup_cache[sym] = df.tail(warmup_bars)
                loaded += 1

        self._api_warmup_loaded = True
        _elapsed = time_module.perf_counter() - _t0
        logger.info("API_WARMUP_CACHE | Loaded %d/%d symbols (%.1fs) | %d warmup bars each",
                   loaded, len(self.core_symbols), _elapsed, warmup_bars)

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

    # ---------- Precomputed enriched 5m bars (backtest) ----------
    def _load_precomputed_5m(self) -> None:
        """Load precomputed enriched 5m bars from feather cache (backtest only)."""
        from pathlib import Path
        cache_dir = Path("cache/ohlcv_archive")
        loaded = 0
        single_day_count = 0
        for sym in self.core_symbols:
            tsym = sym.split(":", 1)[-1].strip().upper()
            for suffix in [f"{tsym}.NS", tsym]:
                path = cache_dir / suffix / f"{suffix}_5minutes_enriched.feather"
                if path.exists():
                    try:
                        df = pd.read_feather(path)
                        df["date"] = pd.to_datetime(df["date"])
                        if getattr(df["date"].dt, "tz", None) is not None:
                            df["date"] = df["date"].dt.tz_localize(None)
                        df = df.set_index("date").sort_index()
                        self._precomputed_5m[sym] = df
                        loaded += 1
                        if len(set(df.index.date)) < 2:
                            single_day_count += 1
                    except Exception:
                        pass
                    break
        logger.info("PRECOMPUTED_5M | Loaded %d/%d symbols from enriched feather cache | single_day_only=%d (no warmup)",
                    loaded, len(self.core_symbols), single_day_count)

    def _get_precomputed_5m(self, symbol: str, up_to: datetime, n: int = 120) -> pd.DataFrame:
        """Get precomputed enriched 5m bars up to the given timestamp."""
        df = self._precomputed_5m.get(symbol)
        if df is None or df.empty:
            return pd.DataFrame()
        # Filter to bars <= current simulation time
        mask = df.index <= pd.Timestamp(up_to)
        return df[mask].tail(n)

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
                raw_sym = item["symbol"]
                # Convert "AARTIIND.NS" → "NSE:AARTIIND" to match screener df.index format
                sym = f"NSE:{raw_sym[:-3]}" if raw_sym.endswith(".NS") else raw_sym
                cap_map[sym] = {
                    "market_cap_cr": item.get("market_cap_cr", 0),
                    "cap_segment": item.get("cap_segment", "unknown"),
                    "mis_enabled": item.get("mis_enabled", False),
                    "mis_leverage": item.get("mis_leverage"),
                }

            self._cap_map_cache = cap_map
            mis_count = sum(1 for v in cap_map.values() if v.get("mis_enabled"))
            logger.info(f"CAP_MAPPING | Loaded market cap data for {len(cap_map)} symbols ({mis_count} MIS-enabled)")
            return cap_map

        except Exception as e:
            logger.warning(f"CAP_MAPPING | Failed to load market cap data: {e}")
            return {}

    def _filter_stage0(self, feats: pd.DataFrame, now_ts, skip_vol_persist: bool = False) -> pd.DataFrame:
        """
        Stage-0 shortlist filter — delegates to standalone function.
        Kept as instance method for fallback path compatibility.
        """
        return _filter_stage0_standalone(
            feats, now_ts,
            skip_vol_persist=skip_vol_persist,
            config=load_filters(),
            cap_map=self._load_cap_mapping(),
            is_dry_run=bool(env.DRY_RUN),
        )

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
