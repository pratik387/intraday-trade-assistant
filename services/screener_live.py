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
from typing import Dict, List, Optional, Set, Tuple
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
from services.gates.multi_timeframe_regime import DailyRegimeDetector
from services.gates.trade_decision_gate import TradeDecisionGate, GateDecision as Decision

# planning & ranking
from services import levels
from services import metrics_intraday as mi
# MainDetector removed (Task 12): dispatch path uses services.dispatch.worker directly.

# Plan orchestrator (sub7/sub8 fast path; Phase C 2026-04-30)
from services.plan_orchestrator import process_setup_candidates

# orders & execution
from services.orders.order_queue import OrderQueue
from services.scan.energy_scanner import EnergyScanner
from diagnostics.diag_event_log import diag_event_log, mint_trade_id
from utils.perf_timer import perf, mark
from services.state.orb_cache_persistence import ORBCachePersistence
import uuid

# 2026-05-12 architectural refactor — Phase 7
from services.bar_scheduler import schedule_admits
from services.setup_risk import SetupRiskTracker
from services.feature_computer import compute_bar_features

# 2026-05-17 dispatch refactor — Phase 1 (Task 10)
from services.dispatch.setup_registry import SetupRegistry, _import_path as _dispatch_import_path
from services.dispatch.transition_calendar import TransitionCalendar
from services.dispatch.tag_map import TagMap
from services.dispatch.fetch_scope import FetchScopeManager
from services.dispatch.planner import DispatchPlanner
from services.dispatch.worker import dispatch_worker_batch, init_worker

# Sentinel: "before market open" — used as the initial _last_dispatch_ts value
# so the very first calendar walk (after=00:00, until=bar_t) fires all events
# with at <= bar_t including the 09:15 build_universe events.
_TIME_BEFORE_OPEN = __import__("datetime").time(0, 0)

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

        # Initialize timing logger in worker process so per-symbol gate timings
        # (detect_setups, regime_compute, hcet_features, allow_setup) are emitted
        # to the parent's timing.jsonl file. Required on Windows (spawn mode) where
        # workers don't inherit the parent's _timing_logger singleton via fork.
        # On Linux/OCI (fork mode) inheritance works automatically, but this code
        # is harmless there too — it just rebinds the singleton to a fresh handle.
        _log_dir = config_dict.get("_log_dir")
        if _log_dir:
            try:
                from pathlib import Path
                from config.logging_config import JSONLLogger, _NoopLogger, _diag_logs_disabled
                import config.logging_config as _lc
                _lc._timing_logger = JSONLLogger(Path(_log_dir) / "timing.jsonl", "timing")
                # Also wire screening.jsonl logger so worker-side decisions write through
                # via the buffered fork-safe path (consistent with other JSONL output).
                # Honors BACKTEST_NO_DIAG_LOGS=1 to substitute a noop (matches parent init).
                _diag_off = _diag_logs_disabled()
                _lc._screener_logger = (
                    _NoopLogger(Path(_log_dir) / "screening.jsonl", "screener")
                    if _diag_off
                    else JSONLLogger(Path(_log_dir) / "screening.jsonl", "screener")
                )
                # Per-event detector accept/reject loggers (audit/14 + audit/15 logging
                # infra). MainDetector runs inside this worker; without explicit wiring
                # on Windows spawn mode, get_detector_rejections_logger() returns None
                # and the per-event JSONL is silently empty. Same buffered fork-safe
                # path as screener/timing loggers above.
                _lc._detector_rejections_logger = (
                    _NoopLogger(Path(_log_dir) / "detector_rejections.jsonl", "detector_reject")
                    if _diag_off
                    else JSONLLogger(Path(_log_dir) / "detector_rejections.jsonl", "detector_reject")
                )
                _lc._detector_accepts_logger = JSONLLogger(
                    Path(_log_dir) / "detector_accepts.jsonl", "detector_accept"
                )
            except Exception:
                pass  # never let logger init break worker startup

        # Apply structure caching in worker process (if enabled via config flag)
        if config_dict.get("_enable_structure_cache", False):
            if worker_logger:
                worker_logger.info("[CACHE] Worker process: Structure caching enabled")
            else:
                print("[CACHE] Worker process: Structure caching enabled")

        # MainDetector removed (Task 12): the _worker_decision_gate / _worker_process_batch
        # path is dead code (replaced by dispatch_worker_batch). _init_worker is still
        # called by _init_worker_combined for logging setup; no gate needed.
        pass
    except Exception as e:
        get_agent_logger().exception(f"Worker init failed: {e}")
        raise

def _init_worker_combined(config_dict, registry):
    """Combined worker initializer: runs the existing _init_worker (logging + TradeDecisionGate)
    AND the new dispatch init_worker (SetupRegistry → detector cache).

    Called once per worker process at spawn time.
    """
    # 1) Existing init: sets up logging, TradeDecisionGate, etc.
    try:
        _init_worker(config_dict)
    except Exception as e:
        get_agent_logger().exception("_init_worker_combined: existing init failed: %s", e)
        raise

    # 2) New dispatch init: registers registry in the worker so _get_detector() works.
    try:
        init_worker(registry)
    except Exception as e:
        get_agent_logger().exception("_init_worker_combined: dispatch init_worker failed: %s", e)
        # Non-fatal: dispatch path will error gracefully per-symbol if registry is None


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

        # scanning.jsonl disabled 2026-05-12 (architectural refactor):
        # Stage-0 shortlisting was replaced by universe-driven scanning,
        # so per-symbol stage-0 logs are no longer meaningful.

        from services.scan.energy_scanner import EnergyScanner

        _stage0_config = config_dict

        scanner_cfg = config_dict["energy_scanner"]
        # IV-rank prepend logic removed 2026-05-14 with options_vol_iv_rank_revert
        # retire (see docs/retired_setups.md).
        _stage0_scanner = EnergyScanner(
            top_k_long=scanner_cfg["top_k_long"],
            top_k_short=scanner_cfg["top_k_short"],
            wide_open=bool(config_dict.get("wide_open_mode", False)),
        )

        # Pre-load cap mapping (same logic as ScreenerLive._load_cap_mapping).
        # Cap segment now comes from data/cap_segments/cap_segments_latest.json
        # (refreshed weekly from niftyindices.com). MIS info still rides on
        # nse_all.json here because the live MIS_FETCHER lives in the parent
        # process and isn't accessible from this worker.
        from services.symbol_metadata import get_all_cap_segments
        cap_segments = get_all_cap_segments()
        import json
        from pathlib import Path
        nse_file = Path(__file__).parent.parent / "nse_all.json"
        mis_info: dict = {}
        if nse_file.exists():
            with nse_file.open() as f:
                data = json.load(f)
            for item in data:
                raw_sym = item["symbol"]
                sym = f"NSE:{raw_sym[:-3]}" if raw_sym.endswith(".NS") else raw_sym
                mis_info[sym] = {
                    "mis_enabled": item.get("mis_enabled", False),
                    "mis_leverage": item.get("mis_leverage"),
                }
        # Union of symbols across both sources (so we don't drop classifications
        # that exist in one but not the other).
        all_syms = set(cap_segments.keys()) | set(mis_info.keys())
        cap_map = {}
        for sym in all_syms:
            mis = mis_info.get(sym, {})
            cap_map[sym] = {
                "cap_segment": cap_segments.get(sym, "unknown"),
                "mis_enabled": mis.get("mis_enabled", False),
                "mis_leverage": mis.get("mis_leverage"),
            }
        _stage0_cap_map = cap_map

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

    Under wide_open_mode the entire filter is bypassed: vol/vwap/ret_1/
    liquidity caps are momentum/mean-rev biased and silently drop
    candidates for setups like circuit_t1_fade_short whose entry bar
    sits OUTSIDE the standard intraday-momentum profile. Wide-open's
    contract is "every detector evaluates every candidate" — the
    detectors do the filtering downstream.
    """
    import logging
    _logger = logging.getLogger(__name__)

    if feats is None or feats.empty:
        return feats

    if bool(config.get("wide_open_mode", False)):
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


def _compute_build_df5_narrow_set(active_symbols: set, core_symbols) -> set:
    """Compute the subset of core_symbols to iterate in _run_5m_scan's
    build_df5_map loop.

    Returns active_symbols intersected with core_symbols if active_symbols is
    non-empty; otherwise returns the full core_symbols set as a fallback for
    pre-09:15 bars (before any universe builders have populated tag_map).
    The fallback preserves correctness for 5-arg universe builders that
    iterate df5_by_symbol — they need the full universe at their bar:HH:MM
    trigger.

    See: docs/superpowers/specs/2026-05-21-backtest-bar-fetch-narrowing-design.md
    """
    if not active_symbols:
        return set(core_symbols)
    return active_symbols & set(core_symbols)


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

        # Shared 5m bar cache (Redis) — prevents duplicate API fetches across instances.
        # Only useful in multi-instance ("subscriber") mode. In single-process
        # ("standalone") mode there is no other writer/reader, so every scan tick
        # is a guaranteed miss → fetch → store cycle that just adds a Redis
        # roundtrip per tick. Disable.
        self._shared_5m_cache = None
        mdb_cfg = raw.get("market_data_bus", {})
        if not env.DRY_RUN and mdb_cfg.get("mode") != "standalone":
            from market_data.shared_5m_cache import Shared5mCache
            redis_url = mdb_cfg.get("redis_url", "redis://localhost:6379/0")
            self._shared_5m_cache = Shared5mCache(redis_url=redis_url)

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

        # Gates - MainDetector removed (Task 12): dispatch uses services.dispatch.worker.
        self.regime_gate = MarketRegimeGate(cfg=raw)

        # Directional bias tracker (Nifty green/red → position size modulation)
        from services.gates.directional_bias import DirectionalBiasTracker, set_tracker
        self.directional_bias = DirectionalBiasTracker(raw)
        set_tracker(self.directional_bias)  # Module-level singleton for pipeline access

        # Stage-0 scanner — iv_rank short-threshold path removed 2026-05-14
        # with options_vol_iv_rank_revert retire (see docs/retired_setups.md).
        scanner_cfg = raw.get("energy_scanner")
        self.scanner = EnergyScanner(
            top_k_long=scanner_cfg["top_k_long"],
            top_k_short=scanner_cfg["top_k_short"],
            wide_open=bool(raw.get("wide_open_mode", False)),
        )

        # CapitalManager for bar-level admission (2026-05-12 architectural refactor).
        # Used by schedule_admits to enforce portfolio + per-setup capital budgets.
        from services.capital_manager import CapitalManager
        _cap_cfg = raw.get("capital_management") or {}
        _risk_mode = str(_cap_cfg.get("risk_mode", "fixed"))
        _risk_fixed = float(_cap_cfg.get("risk_fixed_amount", 0) or 0)
        _risk_pct = float(_cap_cfg.get("risk_percentage", 0) or 0)
        self.capital_manager = CapitalManager(
            enabled=bool(_cap_cfg.get("enabled", False)),
            initial_capital=float(
                _cap_cfg.get("paper_initial_capital", 0)
                if env.DRY_RUN
                else _cap_cfg.get("initial_capital", 0)
            ),
            max_positions=int(_cap_cfg.get("max_concurrent_positions") or raw.get("max_concurrent_positions") or 1),
            min_notional_pct=float(_cap_cfg.get("min_notional_pct", 0) or 0),
            capital_utilization=float(_cap_cfg.get("capital_utilization", 0.85) or 0.85),
            max_allocation_per_trade=float(_cap_cfg.get("max_allocation_per_trade", 0.20) or 0.20),
            risk_mode=_risk_mode,
            risk_fixed_amount=_risk_fixed,
            risk_percentage=_risk_pct,
            mis_enabled=bool(_cap_cfg.get("mis_enabled", False)),
            mis_fetcher=mis_fetcher,
        )

        # PositionStore for SetupRiskTracker concurrency lookups.
        # Screener-local store tracks admitted-but-not-yet-triggered plans
        # within the same bar-scheduling context.
        from services.state.position_store import PositionStore
        self.position_store = PositionStore()

        # Per-setup risk tracker (2026-05-12 refactor — replaces deleted
        # CrossSectionalGate + DedupGate). Reads max_concurrent_positions,
        # per_symbol_cooloff_min, max_fires_per_5min from each setup config.
        self.setup_risk = SetupRiskTracker(
            self.raw_cfg.get("setups") or {},
            self.position_store,
        )

        # Per-setup capital budgets (2026-05-12 refactor). Wires from each
        # setup config's capital_budget_pct into CapitalManager so it can
        # block setups that monopolize total capital.
        _setups_cfg_init = self.raw_cfg.get("setups") or {}
        self.capital_manager.setup_budgets_pct = {
            name: float(cfg.get("capital_budget_pct", 0))
            for name, cfg in _setups_cfg_init.items()
            if cfg.get("enabled") and float(cfg.get("capital_budget_pct", 0) or 0) > 0
        }
        self.capital_manager.setup_budget_used = {
            name: 0.0 for name in self.capital_manager.setup_budgets_pct
        }

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

        # Pass log_dir into worker init so structure workers can write to timing.jsonl
        # (Required on Windows where ProcessPoolExecutor uses spawn, not fork — workers
        # don't inherit the parent's logger singletons.)
        worker_cfg = dict(self.raw_cfg)
        from config.logging_config import get_log_directory
        _wd = get_log_directory()
        if _wd:
            worker_cfg["_log_dir"] = str(_wd)

        # Build SetupRegistry BEFORE creating the executor so we can pass it as
        # the initializer argument to each worker process.
        self._dispatch_registry = SetupRegistry.load_from_config(self.raw_cfg)
        try:
            self._dispatch_registry.validate()
        except Exception as _reg_e:
            logger.warning("SetupRegistry validation warning (non-fatal): %s", _reg_e)

        self._executor = ProcessPoolExecutor(
            max_workers=structure_workers,
            initializer=_init_worker_combined,
            initargs=(worker_cfg, self._dispatch_registry)
        )
        self._daily_cache_seeded = False
        # Per-setup qualifying-universe contributions (computed once at session
        # seed). 2026-05-12 architectural fix: cross-day setups (circuit_t1,
        # earnings_day, delivery_pct) have signals invisible to Stage-0's
        # intraday momentum ranking. Their qualifying symbols are union'd with
        # the Stage-0 shortlist per-bar so they're never silently dropped.
        # See services/setup_universe.py.
        self._setup_universes: Dict[str, Set[str]] = {}
        self._daily_dict_cache: Dict[str, "pd.DataFrame"] = {}

        # Last-scan cutoff (parsed once). Backtest and paper both honor this — skips scans
        # after this time, leaving exit_executor running for the remaining ticks until EOD.
        self._last_scan_time = None
        try:
            from datetime import time as _dt_time
            _ls_str = str(self.raw_cfg["last_scan_hhmm"])
            _lh, _lm = _ls_str.split(":")
            self._last_scan_time = _dt_time(int(_lh), int(_lm))
        except Exception as _e:
            logger.warning("ScreenerLive: failed to parse last_scan_hhmm; backtest scans will run until EOD: %s", _e)

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

        # State for per-bar volume accumulation (fed to RVOL on bar transition)
        self._pending_bar_ts = None
        self._pending_bar_vols: Dict[str, int] = {}

        # ------------------------------------------------------------------ #
        # Dispatch refactor state (Phase 1 — Task 10, 2026-05-17)            #
        # ------------------------------------------------------------------ #
        # TransitionCalendar / TagMap / FetchScopeManager / DispatchPlanner  #
        # drive the per-bar calendar-walk → tag dispatch path.               #
        # Hardcoded lazy-build if-blocks deleted by Task 13.                 #
        self._transition_calendar = TransitionCalendar.from_registry(self._dispatch_registry)
        self._tag_map = TagMap()
        self._fetch_scope = FetchScopeManager()
        self._dispatch_planner = DispatchPlanner(batch_size=int(self.raw_cfg.get("dispatch_batch_size", 50)))
        # Tracks which time slice the calendar has been walked to; reset each session.
        self._last_dispatch_ts = None   # set to _TIME_BEFORE_OPEN on first scan

        logger.debug(
            "ScreenerLive init: universe=%d symbols, store5m=%d",
            len(self.core_symbols),
            self.cfg.screener_store_5m_max,
        )

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def set_position_store(self, store) -> None:
        """Replace screener-local PositionStore with the shared one from main.py.

        Without this, SetupRiskTracker queries an empty store and
        max_concurrent_positions never fires. Call right after main.py
        constructs the authoritative PositionStore.
        """
        self.position_store = store
        self.setup_risk._positions = store

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
                    detected_level=getattr(candidate, 'detected_level', None),
                    extras=getattr(candidate, 'extras', None),  # preserve detector context
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
        # Accumulate per-symbol bar volumes for future RVOL state (bar_scheduler
        # integration lands in Phase 7).
        try:
            bar_ts = bar_5m.name if hasattr(bar_5m, "name") else None
            vol = float(bar_5m.get("volume", 0)) if hasattr(bar_5m, "get") else 0
            if bar_ts is not None:
                if self._pending_bar_ts is not None and bar_ts != self._pending_bar_ts:
                    self._pending_bar_vols = {}
                self._pending_bar_ts = bar_ts
                self._pending_bar_vols[symbol] = int(vol)
        except Exception as e:
            logger.warning("BAR_VOL_ACCUMULATOR | error: %s", e)

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

        # Last-scan cutoff: skip new scans after last_scan_hhmm (e.g., 14:50).
        # This was originally only enforced in the paper timer loop. Backtest scans were
        # firing all the way to 15:15 EOD which wastes work and corrupts analysis (some
        # scans drift past entry_cutoff_hhmm anyway). Exit executor still receives ticks
        # for bars between last_scan_hhmm and eod_squareoff_hhmm — only the SCAN is skipped.
        try:
            now_t = now.time() if hasattr(now, 'time') else now
            if self._last_scan_time is not None and now_t > self._last_scan_time:
                return
        except Exception:
            pass

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
        # FAIL-FAST: flag is set BEFORE broadcast so a BrokenProcessPool on the
        # broadcast doesn't cause every subsequent bar to retry forever on a
        # dead pool.  If the broadcast fails, the flag still flips, workers
        # proceed without daily cache (degraded but responsive), and the run
        # exits gracefully at EOD instead of generating hundreds of identical
        # error tracebacks.
        if not self._daily_cache_seeded:
            self._daily_cache_seeded = True  # set BEFORE work so retries are bounded
            with perf("scan", "daily_seed_total", n_symbols=len(self.core_symbols)):
                _t_seed_start = time.perf_counter()
                daily_dict = {}
                with perf("scan", "daily_seed_fetch", n_symbols=len(self.core_symbols)):
                    for sym in self.core_symbols:
                        dd = self.sdk.get_daily(sym, days=210)
                        if dd is not None and not dd.empty:
                            daily_dict[sym] = dd
                # Enrich daily_df with delivery_pct (NSE bhavcopy archive).
                # No-op if data/delivery_pct/delivery_history.parquet is missing.
                # Required by `delivery_pct_anomaly_short` detector.
                try:
                    from services.delivery_pct_enrichment import enrich_dict as _enrich_delivery
                    daily_dict = _enrich_delivery(daily_dict)
                except Exception as _e:
                    logger.warning("delivery_pct enrichment failed: %s", _e)

                # ---------- Per-setup qualifying universes (static, session-start) ----------
                # 2026-05-12 architectural fix: setups whose qualifying signals
                # are invisible to Stage-0's intraday momentum ranking
                # (circuit_t1, earnings_day, delivery_pct) declare their
                # universe here. Screener unions per-bar with Stage-0 shortlist.
                # Also cache daily_dict on self so the gap_fade universe (which
                # needs PDC and runs lazy at the 09:15 bar) can re-use it.
                self._daily_dict_cache = daily_dict
                try:
                    from services.setup_universe import compute_static_universes
                    session_date_obj = now.date() if hasattr(now, "date") else now
                    setups_cfg = (self.raw_cfg.get("setups") or {})
                    self._setup_universes = compute_static_universes(
                        setups_cfg, daily_dict, session_date_obj,
                    )
                    total_extra = sum(len(s) for s in self._setup_universes.values())
                    logger.info(
                        "SETUP_UNIVERSES_LOADED | total=%d %s",
                        total_extra,
                        ", ".join(f"{k}={len(v)}" for k, v in self._setup_universes.items()),
                    )
                except Exception as _e:
                    logger.warning("setup_universe computation failed: %s", _e)
                    self._setup_universes = {}

                # Cross-day RVOL baseline (live/paper only).
                # Backtest skips this and falls through to the parquet path in
                # services/cross_day_rvol_enrichment.py (preserves existing
                # behavior for `--dry-run`).
                #
                # Consumers (must all be present in the baseline OR they silently
                # no-fire on "baseline volume unavailable"):
                #   - delivery_pct_anomaly_short  hhmm 0930-1000
                #   - below_vwap_volume_revert_long  hhmm 1300-1455
                #
                # Previous bug: only delivery_pct's universe was populated, and only
                # for the 0930-1000 hhmm window. below_vwap detector at 1305 looked
                # up (sym, 1305) → empty → "baseline volume unavailable" 459 times
                # in one session. Now: union of universes from all rvol-dependent
                # enabled setups, with broad hhmm window covering both consumers.
                if not env.DRY_RUN:
                    rvol_dependent_setups = (
                        "delivery_pct_anomaly_short",
                        "below_vwap_volume_revert_long",
                    )
                    rvol_universe: set = set()
                    rvol_setups_used: list = []
                    for _name in rvol_dependent_setups:
                        _u = self._setup_universes.get(_name, set())
                        if _u and (setups_cfg.get(_name) or {}).get("enabled", False):
                            rvol_universe |= _u
                            rvol_setups_used.append(f"{_name}={len(_u)}")
                    if rvol_universe:
                        try:
                            import asyncio
                            from services.runtime_rvol_baseline import RuntimeRvolBaseline
                            from services import cross_day_rvol_enrichment as _rvol
                            rt = RuntimeRvolBaseline()
                            stats = asyncio.run(rt.populate(
                                self.sdk, sorted(rvol_universe), session_date_obj,
                                hhmm_window=(930, 1455), rolling_days=20,
                            ))
                            _rvol.set_runtime_baseline(rt)
                            logger.info(
                                "RUNTIME_RVOL_INSTALLED | union=%d (%s) | %s",
                                len(rvol_universe), ", ".join(rvol_setups_used), stats,
                            )
                        except Exception as _e:
                            logger.warning(
                                "RUNTIME_RVOL | populate failed: %s — "
                                "rvol-dependent detectors will silently no-fire", _e,
                            )
                    else:
                        logger.info(
                            "RUNTIME_RVOL | no enabled rvol-dependent setup produced a "
                            "non-empty universe — skipping populate"
                        )
                _t_seed_fetch = time.perf_counter()
                try:
                    with perf("scan", "daily_seed_broadcast", n_workers=self._structure_workers,
                              n_symbols=len(daily_dict)):
                        seed_futures = []
                        for _ in range(self._structure_workers):
                            seed_futures.append(self._executor.submit(_seed_worker_daily_cache, daily_dict))
                        for f in seed_futures:
                            f.result(timeout=120)
                    _t_seed_done = time.perf_counter()
                    logger.info("DAILY_CACHE_SEEDED | %d symbols to %d workers | fetch=%.2fs send=%.2fs total=%.2fs",
                               len(daily_dict), self._structure_workers,
                               _t_seed_fetch - _t_seed_start, _t_seed_done - _t_seed_fetch,
                               _t_seed_done - _t_seed_start)
                except Exception as seed_exc:
                    logger.error(
                        "DAILY_CACHE_SEED_FAILED | %s | continuing without worker daily cache "
                        "(workers will operate with reduced context). This is typically "
                        "BrokenProcessPool from pickle-IPC of a large daily_dict; fix by "
                        "reducing structure_detection_workers_local to 1 in config or "
                        "switching to file-based sharing.",
                        seed_exc,
                    )

        # ---------- Stage-0: EnergyScanner (single unified path) ----------
        shortlist: List[str] = []
        levels_by_symbol = None  # Initialize so it's available to structure detection phase

        # OPENING BELL FIX: Determine minimum bars needed - 1 during opening
        # bell (09:15-09:30), 3 normally.
        #
        # Window starts at 09:15 (not 09:20) because the FIRST scan of the
        # day is for bar 09:15 (now=09:15), and at that point only 1-2 5m
        # bars exist in today's data (the just-closed 09:15 bar + maybe the
        # in-progress 09:20 bar). With min_bars=3, the entire universe gets
        # silently filtered out — empirical 2026-06-04 09:21 scan: SDK
        # returned 1525 dfs (1522 with len=2, 3 with len=1), screener
        # treated all 1525 as "failed" because none met len>=3, leaving
        # gap_fade_short and long_panic_gap_down (both 09:15-window setups)
        # running on a tag_map-only subset of ~9-50 instead of the full
        # universe. Extending the relaxation back to 09:15 closes the gap.
        current_time = now.time() if hasattr(now, 'time') else now
        from datetime import time as dtime
        in_opening_bell = dtime(9, 15) <= current_time < dtime(9, 30)
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
                # Narrow to tag_map active symbols (~99) instead of all core_symbols
                # (~1500). Falls back to full universe if tag_map is empty (pre-09:15).
                _univ = self._tag_map.active_symbols()
                fetch_symbols = sorted(_univ & set(self.core_symbols)) if _univ else list(self.core_symbols)
                # Piggyback the regime index on the same batch — one extra HTTP
                # call per dispatch, no separate fallback path. WebSocket never
                # subscribes the index token, so this is the only way the
                # regime gate sees real NIFTY bars in paper/live.
                _idx_sym_for_fetch = (self.raw_cfg.get("directional_bias", {}) or {}).get("index_symbol")
                if _idx_sym_for_fetch and _idx_sym_for_fetch not in fetch_symbols:
                    fetch_symbols = list(fetch_symbols) + [_idx_sym_for_fetch]
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

                    # Ground-truth diagnostic: what did the SDK actually return?
                    # Speculation about silent-failure paths in async_fetch_intraday_5m_batch
                    # keeps missing — log the shape distribution directly.
                    _shape_counts: Dict[str, int] = {}
                    _sample_short: List[str] = []
                    for _sym, _v in raw_api.items():
                        if _v is None:
                            _shape_counts["None"] = _shape_counts.get("None", 0) + 1
                        elif hasattr(_v, "__len__"):
                            _n = len(_v)
                            _key = f"df_len={_n}"
                            _shape_counts[_key] = _shape_counts.get(_key, 0) + 1
                            if _n < 3 and len(_sample_short) < 3:
                                _sample_short.append(f"{_sym}:n={_n}")
                        else:
                            _shape_counts[type(_v).__name__] = _shape_counts.get(type(_v).__name__, 0) + 1
                    logger.info(
                        "API_5M_RAW | raw_api_len=%d req=%d | shape_dist=%s | short_samples=%s",
                        len(raw_api), len(fetch_symbols), _shape_counts, _sample_short,
                    )

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
                    # Surface 429 count + failure-mode breakdown from the batch
                    # fetcher for monitoring (silent-fail diagnosis).
                    throttle_count = getattr(self.sdk, '_last_batch_429s', 0)
                    throttle_info = f" | 429s: {throttle_count}" if throttle_count > 0 else ""
                    diag = getattr(self.sdk, '_last_batch_diag', None)
                    diag_info = ""
                    if diag and api_fail > 0:
                        diag_info = (
                            f" | 400={diag.get('n_400', 0)} 5xx={diag.get('n_5xx', 0)} "
                            f"other_http={diag.get('n_other_http', 0)} "
                            f"empty={diag.get('n_empty_candles', 0)} "
                            f"timeout={diag.get('n_timeout', 0)} "
                            f"conn_err={diag.get('n_conn_err', 0)} "
                            f"other_exc={diag.get('n_other_exc', 0)} "
                            f"429_exhausted={diag.get('n_429_exhausted', 0)} "
                            f"UNACCOUNTED={diag.get('n_unaccounted', 0)}"
                        )
                        if diag.get("sample_400_url"):
                            diag_info += f" | sample_400={diag['sample_400_url']}"
                        if diag.get("sample_5xx"):
                            diag_info += f" | sample_5xx={diag['sample_5xx']}"
                        if diag.get("sample_exc"):
                            diag_info += f" | sample_exc={diag['sample_exc']}"
                        if diag.get("sample_429_url"):
                            diag_info += f" | sample_429={diag['sample_429_url']}"
                    logger.info(
                        "API_5M_FETCH | %d ok, %d failed of %d symbols | %.1fs (async, %s RPS) | warmup: %d/%d%s%s",
                        api_ok, api_fail, len(fetch_symbols), _t_api_elapsed,
                        rps, warmup_used, api_ok, throttle_info, diag_info,
                    )

                # Capture the regime index df out of the batch result. Pop so the
                # dispatcher doesn't waste cycles running detectors on the index.
                if _idx_sym_for_fetch and _idx_sym_for_fetch in api_df5_cache:
                    idx_df = api_df5_cache.pop(_idx_sym_for_fetch)
                    if isinstance(idx_df, pd.DataFrame) and not idx_df.empty:
                        self._latest_index_df5 = idx_df

        try:
            # Build df5_by_symbol from enriched data (API for paper, precomputed for backtest).
            # Narrow to active universes when tag_map is populated (mirrors live behavior at
            # services/screener_live.py:1357). Falls back to full core_symbols when tag_map
            # is empty (pre-09:15) to preserve correctness for 5-arg universe builders that
            # iterate df5_by_symbol at the bar:09:15 trigger.
            # Spec: docs/superpowers/specs/2026-05-21-backtest-bar-fetch-narrowing-design.md
            narrow_set = _compute_build_df5_narrow_set(
                self._tag_map.active_symbols(), self.core_symbols
            )
            with perf("scan", "build_df5_map", n_core=len(narrow_set)):
                df5_by_symbol: Dict[str, pd.DataFrame] = {}
                for s in narrow_set:
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
            if not df5_by_symbol:
                # Early-session symbols may not yet have min_bars_for_processing.
                # Skip silently rather than falling through to the dispatch path,
                # which would raise UnboundLocalError on _dp_decisions below.
                return
            if df5_by_symbol:
                # Compute ORB levels once at 09:40 and cache for entire day
                with perf("scan", "compute_orb_levels", n=len(df5_by_symbol)):
                    levels_by_symbol = self._compute_orb_levels_once(now, df5_by_symbol)

                # ---- DISPATCH PATH (Task 10, 2026-05-17) ----
                # Calendar-driven tag dispatch replaces universe-union + Stage-0.
                # Returns None → SCAN_SKIPPED (no active detectors this bar).
                # Returns dict  → decisions + symbol_data_map already collected.
                _dispatch_result = self._run_dispatch_path(
                    now, df5_by_symbol, levels_by_symbol, api_df5_cache,
                    min_bars_for_processing=min_bars_for_processing,
                )
                if _dispatch_result is None:
                    # No active detectors — skip the rest of the scan entirely.
                    return
                _dp_decisions = _dispatch_result["decisions"]
                _dp_sdmap     = _dispatch_result["symbol_data_map"]
                _dp_sc        = _dispatch_result["shortlist_count"]
                _dp_sl        = _dispatch_result["shortlist"]
                _dp_slog      = _dispatch_result["screener_logger"]

        except Exception as e:
            logger.exception("Universe scan failed; dispatch path error at this bar: %s", e)
            # In dispatch path, a top-level exception is fatal for this bar
            if "_dp_decisions" not in dir():
                return

        # ---- DISPATCH PATH: alias results into downstream variable names ----
        decisions: List[Tuple[str, Decision]] = _dp_decisions
        symbol_data_map: Dict[str, tuple] = _dp_sdmap
        shortlist: List[str] = _dp_sl
        shortlist_count: int = _dp_sc
        screener_logger = _dp_slog
        _t_scanner_end = time.perf_counter()
        _t_data_prep_end = _t_scanner_end  # No separate prep phase in dispatch path
        _t_structure_end = _t_scanner_end
        logger.info(
            "SCANNER_COMPLETE | data_loaded=%d/%d dispatch_active=%d shortlist=%d | TIME: %.2fs",
            len(df5_by_symbol) if df5_by_symbol else 0, len(self.core_symbols),
            shortlist_count, shortlist_count, _t_scanner_end - _t_bar_start,
        )
        logger.info(
            "PARALLEL_STRUCTURE_COMPLETE | Processed %d symbols, %d accepted (dispatch path)",
            shortlist_count, len(decisions),
        )

        # ---- DEAD CODE START: Old universe-union + structure-prep + batch-submit ----
        # Replaced by calendar-driven dispatch path (_run_dispatch_path).
        # Tasks 12-13 delete this section.
        # See git history for the full code that was here.
        pass  # dispatch path bridge above already set decisions/symbol_data_map/etc.
        # ---- DEAD CODE END ----

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

        with perf("scan", "orchestrator_loop", n_decisions=len(decisions)):
            for sym, decision in decisions:
                df5 = symbol_data_map.get(sym, (None,))[0]
                lvl = symbol_data_map.get(sym, (None, {}))[1]
                # daily_df only needed for ~5-20 accepted symbols (not 800), fetch from sdk cache
                with perf("orch", "sdk_get_daily", sym=sym):
                    daily_df = self.sdk.get_daily(sym, days=210)

                if df5 is None:
                    continue

                setup_candidates = getattr(decision, 'setup_candidates', None)
                if not setup_candidates:
                    continue

                # Build HTF context from 15m data (aggregated from 5m enriched bars)
                with perf("orch", "build_htf_context", sym=sym):
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
                    with perf("orch", "process_setup_candidates", sym=sym,
                              n_candidates=len(setup_candidates)):
                        # return_all_eligible=True: get every eligible plan
                        # (one per setup) instead of the orchestrator's
                        # per-symbol-category-best dedupe. Matches gauntlet's
                        # no-dedupe behavior (every executed trade is a separate
                        # row in analytics.jsonl). Without this, range_bounce/
                        # resistance_bounce candidates that share a symbol with a
                        # higher-RR premium_zone_short never reach execution
                        # (live-vs-gauntlet parity bug 2026-04-23).
                        plans = process_setup_candidates(
                            symbol=sym,
                            df5m=df5,
                            levels=lvl,
                            regime=decision.regime,
                            now=now,
                            candidates=setup_candidates,
                            daily_df=daily_df,
                            htf_context=htf_context,
                            regime_diagnostics=getattr(decision, 'regime_diagnostics', None),
                            daily_score=daily_score,
                            return_all_eligible=True,
                        )
                except Exception as e:
                    logger.exception("orchestrator failed for %s: %s", sym, e)
                    continue

                # `plans` is now a List[Dict] — one entry per eligible plan
                # across all categories for this symbol-bar.
                if not plans:
                    logger.debug(f"ORCHESTRATOR:REJECT {sym} reason=no_eligible_plans")
                    continue
                for plan in plans:
                    if plan and plan.get("eligible", False):
                        score = plan.get("ranking", {}).get("score", 0.0)
                        eligible_plans.append((sym, plan, score, decision))
                        logger.debug(
                            f"ORCHESTRATOR:ELIGIBLE {sym} {plan.get('strategy', '?')} score={score:.3f}"
                        )

        _t_orch_end = time.perf_counter()
        logger.info("ORCHESTRATOR_COMPLETE | %d eligible plans from %d decisions | TIME: %.2fs",
                   len(eligible_plans), len(decisions), _t_orch_end - _t_orch_start)

        # ---------- Gate-input capture writer (offline parity replay) ----------
        # gate_input_logging.enabled writes one JSONL row per bar with candidate
        # dicts for the parity_simulator. Gate chain removed in Phase 6.
        _gate_input_on = (
            self.raw_cfg.get("gate_input_logging", {}).get("enabled", False)
            and bool(eligible_plans)
        )
        if _gate_input_on:
            _mod = now.hour * 60 + now.minute  # minute-of-day
            if _mod < 600:
                _hour_bucket = "opening"
            elif _mod < 720:
                _hour_bucket = "morning"
            elif _mod < 780:
                _hour_bucket = "lunch"
            elif _mod < 870:
                _hour_bucket = "afternoon"
            else:
                _hour_bucket = "late"

            _day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

            _cand_dicts = []
            for _ep in eligible_plans:
                _sym, _plan, _score, _dec = _ep
                _extras = _plan.get("extras") or {}
                _model_features = _plan.get("model_features") or {}
                _cand = {
                    "symbol": _sym,
                    "setup_type": _plan.get("strategy", ""),
                    "regime": _plan.get("regime", ""),
                    "cap_segment": (_plan.get("sizing") or {}).get("cap_segment", "unknown"),
                    "hour_bucket": _hour_bucket,
                    "decision_ts": now,
                    "session_date_dt": now.date(),
                    "minute_of_day": _mod,
                    "day_of_week": _day_names[now.weekday()] if now.weekday() < 7 else "Monday",
                    "size_mult": (_plan.get("sizing") or {}).get("size_mult", 1.0),
                    "rank_score": float(_score),
                    **{k: v for k, v in _extras.items()
                       if isinstance(v, (int, float, bool, str)) or v is None},
                    **{k: v for k, v in _model_features.items()
                       if isinstance(v, (int, float, bool))},
                }
                _cand_dicts.append(_cand)

            from config.logging_config import get_gate_input_logger
            _gi_logger = get_gate_input_logger()
            _cap_map = self._load_cap_mapping()
            _bar_vols = dict(self._pending_bar_vols) if self._pending_bar_vols else {}
            _sym_caps = {s: _cap_map.get(s, {}).get("cap_segment", "unknown")
                         for s in _bar_vols.keys()}
            _serializable_cands = []
            for _c in _cand_dicts:
                _safe = {}
                for _k, _v in _c.items():
                    if hasattr(_v, "isoformat"):
                        _safe[_k] = _v.isoformat()
                    else:
                        _safe[_k] = _v
                _serializable_cands.append(_safe)
            _gi_logger.log_event(
                ts=now.isoformat(),
                session_date=str(now.date()),
                candidates=_serializable_cands,
                bar_volumes=_bar_vols,
                symbol_caps=_sym_caps,
            )

        # Bar-level scheduling: priority-sorted capital admission.
        # 2026-05-12 architectural refactor — replaces the gate_chain
        # invocation. setup_risk + capital_manager carry all gate-equivalent
        # logic. Plans sorted by plan["priority"] desc; admitted in order.
        # eligible_plans is List[Tuple[sym, plan, score, decision]] — extract
        # the plan dicts for schedule_admits, then rejoin with original tuples.
        _ep_plan_dicts = [plan for (_, plan, _, _) in eligible_plans]
        _admitted_dicts_set = set(
            id(p) for p in schedule_admits(
                _ep_plan_dicts, self.capital_manager, self.setup_risk, ts=now,
            )
        )
        admitted_plans = [
            (sym, plan, score, decision)
            for (sym, plan, score, decision) in eligible_plans
            if id(plan) in _admitted_dicts_set
        ]

        # Sort admitted plans by rank_score desc for execution priority.
        admitted_plans.sort(key=lambda x: x[2], reverse=True)

        # ---------- Process admitted plans → Execution ----------
        for i, (sym, plan, score, decision) in enumerate(admitted_plans):
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

            # 1) stable id — should already be set by orchestrator/detector via
            # StructureEvent auto-mint. Falling back here means a code path
            # bypassed that mint; log so we can find it. Safety mint stays so
            # DECISION still gets a usable id rather than crashing.
            if "trade_id" not in plan:
                logger.warning(
                    "TRADE_ID_LATE_MINT | %s | strategy=%s — plan reached screener_live "
                    "without trade_id; expected detection-time mint",
                    sym, plan.get("strategy", "unknown"),
                )
                plan["trade_id"] = mint_trade_id(sym, token=uuid.uuid4().hex[:8])

            decision_obj = dec_map.get(sym)

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
                    # 2026-05-13 fix: plan-as-source-of-truth fields must propagate
                    # through to the executor. exits.time_stop_hhmm is read by
                    # exit_executor._effective_eod_md to cap EOD per-position;
                    # priority is logged for bar_scheduler audit.
                    "exits": plan.get("exits"),
                    "priority": plan.get("priority"),
                    # target_anchor_type drives services/target_recalc.py at fill
                    # time. Missing from this re-build prior to 2026-05-13 caused
                    # every plan to default to "structural" in the executor —
                    # silently nullifying r_multiple recalc for delivery_pct +
                    # options_vol_iv_rank_revert. Verified via run_0d03a7a6_*
                    # smoke: DECISION event had r_multiple, but agent.log showed
                    # STRUCTURAL_TARGET_PRESERVED for every trade.
                    "target_anchor_type": plan.get("target_anchor_type"),
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

            # Early MIS filter — reduces WS subscriptions, Stage-0, daily cache, ORB.
            # SKIPPED in backtest mode: the live Zerodha MIS sheet reflects today's
            # eligibility, not the historical session date's. Applying it to a
            # multi-year backtest strips symbols that WERE MIS-eligible at signal
            # time but were removed from Zerodha's live list later — caused
            # circuit_release_fade_short to miss 58/256 sanity-only signals.
            # Universe builders still check per-symbol MIS via `nse_all.json`
            # (date-anchored snapshot) at universe-build time, so eligibility
            # filtering still happens correctly downstream.
            mis_filter_cfg = self.raw_cfg.get("early_mis_universe_filter", {})
            if env.DRY_RUN:
                logger.info("MIS_UNIVERSE | DRY_RUN: skipping live Zerodha MIS filter (use nse_all.json downstream)")
            elif mis_filter_cfg.get("enabled", False) and self._mis_fetcher and self._mis_fetcher.is_loaded():
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
        # Configured index drives market-wide regime classification.
        idx_sym = (self.raw_cfg.get("directional_bias", {}) or {}).get("index_symbol")
        return [idx_sym] if idx_sym else []

    def _index_df5(self) -> pd.DataFrame:
        # Paper/live path: the per-bar 5m batch fetch (see API_5M_FETCH) stashes
        # the configured index symbol's df here. LiveTickHandler's aggregator
        # stays empty because SubscriptionManager doesn't subscribe the index
        # token to WS — so without this attribute, regime silently defaults to
        # "chop". Prefer the freshest captured batch result.
        cached_idx = getattr(self, "_latest_index_df5", None)
        if isinstance(cached_idx, pd.DataFrame) and not cached_idx.empty:
            return cached_idx
        idx = self.agg.index_df_5m()
        if isinstance(idx, dict):
            for _, df in idx.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df
            idx = pd.DataFrame()
        if isinstance(idx, pd.DataFrame) and not idx.empty:
            return idx
        # Backtest fallback: the aggregator never receives index ticks in DRY_RUN
        # because there's no websocket subscription. Synthesize a market-wide
        # 5m DataFrame by aggregating across the precomputed universe so the
        # regime classifier has actual price action to work with rather than
        # short-circuiting to "chop". Without this, every backtest bar produces
        # regime=chop regardless of actual market state.
        if env.DRY_RUN and self._precomputed_5m:
            try:
                return self._synthesize_market_df5()
            except Exception as e:
                logger.debug("INDEX_DF5_SYNTH_FAILED | %s", e)
        return pd.DataFrame()

    def _synthesize_market_df5(self) -> pd.DataFrame:
        """Build a market-wide 5m DF by equal-weighted aggregation across the
        precomputed universe. Used as a NIFTY-50 proxy in backtest mode where
        no index symbol is available in the cache."""
        # Cache the synthesized DF per-session to avoid rebuilding every bar.
        cache_key = getattr(self, "_synth_market_session_date", None)
        cached_df = getattr(self, "_synth_market_df5", None)
        try:
            session_date = getattr(self.sdk, "_dry_session_date", None) or self.raw_cfg.get("session_date")
            session_date = pd.to_datetime(session_date).date() if session_date else None
        except Exception:
            session_date = None
        if cached_df is not None and cache_key == session_date:
            return cached_df

        # Pick up to N most-liquid symbols by mean volume in the cache to keep
        # the synthesis cheap and representative of broad-market state.
        max_syms = 50
        candidates = []
        for sym, df in self._precomputed_5m.items():
            if df is None or df.empty:
                continue
            try:
                v = float(df["volume"].mean())
                candidates.append((v, sym))
            except Exception:
                continue
        candidates.sort(reverse=True)
        selected = [s for _, s in candidates[:max_syms]]
        if not selected:
            return pd.DataFrame()

        frames = []
        for s in selected:
            df = self._precomputed_5m.get(s)
            if df is None or df.empty:
                continue
            # Normalize prices so cross-symbol aggregation is meaningful
            base = float(df["close"].iloc[0]) if len(df) > 0 else None
            if not base or base <= 0:
                continue
            d = df[["open", "high", "low", "close", "volume"]].copy()
            d["open"] /= base
            d["high"] /= base
            d["low"] /= base
            d["close"] /= base
            frames.append(d)
        if not frames:
            return pd.DataFrame()

        # Index-aligned mean across symbols, treating missing bars as NaN.
        combined = pd.concat(frames).groupby(level=0).mean()
        combined = combined.sort_index()
        self._synth_market_df5 = combined
        self._synth_market_session_date = session_date
        return combined

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

    # ------------------------------------------------------------------
    # Calendar-driven dispatch path (Task 10, 2026-05-17)
    # ------------------------------------------------------------------

    def _run_dispatch_path(
        self,
        now,
        df5_by_symbol: Dict[str, pd.DataFrame],
        levels_by_symbol,
        api_df5_cache: Dict[str, pd.DataFrame],
        min_bars_for_processing: int = 3,
    ):
        """Per-bar dispatch pipeline: calendar walk → tagmap → fetch-scope → planner → executor.

        Returns a dict with keys (decisions, symbol_data_map, shortlist_count, screener_logger)
        or None if no active detectors at this bar (SCAN_SKIPPED).

        This method is the entry point for the dispatch refactor (Phase 1 Task 10).
        The OLD scan path (universe-union + _worker_process_batch) is DEAD CODE after
        this method exists and is called; Tasks 12-13 delete it.
        """
        import time as _time_mod

        now_t = now.time() if hasattr(now, "time") else now
        session_date_obj = now.date() if hasattr(now, "date") else now

        # ---- 1. Walk calendar → mutate TagMap ----
        # Phase 1 (pre-dispatch): apply build_universe + open_window for all events in
        # (last_t, now_t].  Apply close_window only for events STRICTLY BEFORE now_t
        # (windows that closed in a prior bar).  Defer close_window events whose
        # ev.at == now_t to Phase 2 (post-dispatch) so that one-shot setups whose
        # active_window=[T, T] get their dispatch before the window is closed.
        last_t = (
            self._last_dispatch_ts.time()
            if self._last_dispatch_ts is not None
            else _TIME_BEFORE_OPEN
        )
        post_close_events = []
        for ev in self._transition_calendar.events_in(after=last_t, until=now_t):
            if ev.kind == "build_universe":
                try:
                    spec = self._dispatch_registry.get(ev.setup)
                    builder_fn = _dispatch_import_path(spec.universe_builder_path)
                    # Build cap_map {sym: cap_segment_str} for universe builders
                    full_cap_map = self._load_cap_mapping()
                    cap_map_str = {s: full_cap_map.get(s, {}).get("cap_segment", "unknown")
                                   for s in df5_by_symbol.keys()}
                    daily_dict = getattr(self, "_daily_dict_cache", {}) or {}
                    # Universe builder signatures are not uniform:
                    #   5-arg: (df5_today_by_symbol, daily_dict, session_date, config, cap_map)
                    #     — gap_fade_universe, long_panic_gap_down_universe,
                    #       circuit_release_fade_short_universe
                    #   3-arg: (daily_dict, session_date, config)
                    #     — or_window_failure_fade_short_universe
                    # Detect by inspecting the first parameter name.
                    import inspect as _inspect
                    _sig_params = list(_inspect.signature(builder_fn).parameters.keys())
                    if _sig_params and _sig_params[0] in ("df5_today_by_symbol", "df5_by_symbol"):
                        syms = builder_fn(
                            df5_by_symbol,
                            daily_dict,
                            session_date_obj,
                            spec.raw_config,
                            cap_map_str,
                        )
                    else:
                        syms = builder_fn(
                            daily_dict,
                            session_date_obj,
                            spec.raw_config,
                        )
                    self._tag_map.add_universe(ev.setup, set(syms or []))
                    logger.info("DISPATCH_BUILD_UNIVERSE | %s | %d symbols", ev.setup, len(syms or []))
                except Exception as build_e:
                    logger.warning("UNIVERSE_BUILD_FAILED | %s | %s", ev.setup, build_e)
                    self._tag_map.add_universe(ev.setup, set())
            elif ev.kind == "open_window":
                self._tag_map.open_window(ev.setup)
                logger.info("DISPATCH_OPEN_WINDOW | %s", ev.setup)
            elif ev.kind == "close_window":
                if ev.at < now_t:
                    # Window closed in a prior bar — apply immediately.
                    self._tag_map.close_window(ev.setup)
                    logger.info("DISPATCH_CLOSE_WINDOW | %s", ev.setup)
                else:
                    # Window ends exactly at this bar — defer until after dispatch so
                    # the setup fires its last (or only) trade before the window closes.
                    post_close_events.append(ev)
                    logger.info("DISPATCH_CLOSE_WINDOW_DEFERRED | %s (will close post-dispatch)", ev.setup)

        # ---- 2. Skip if no active detectors ----
        active_syms = self._tag_map.active_symbols()
        if not active_syms:
            logger.info("SCAN_SKIPPED | no active detectors at bar %s", now)
            # Still flush any deferred close_window events so TagMap stays consistent.
            for ev in post_close_events:
                self._tag_map.close_window(ev.setup)
                logger.info("DISPATCH_CLOSE_WINDOW_POST | %s (closed after skip at bar %s)", ev.setup, now_t)
            self._last_dispatch_ts = now
            return None

        # ---- 3. Restrict active syms to what has df5 data ----
        active_syms_with_data = active_syms & set(df5_by_symbol.keys())
        if not active_syms_with_data:
            logger.info("SCAN_SKIPPED | active detectors exist but no df5 data for active syms | bar %s", now)
            # Still flush any deferred close_window events so TagMap stays consistent.
            for ev in post_close_events:
                self._tag_map.close_window(ev.setup)
                logger.info("DISPATCH_CLOSE_WINDOW_POST | %s (closed after skip at bar %s)", ev.setup, now_t)
            self._last_dispatch_ts = now
            return None

        # ---- 4. Compute regime once (shared across all symbols in this bar) ----
        index_df5 = self._index_df5()
        # Filter synthesized/index DF to history up to current bar so the regime
        # is computed from data actually available at this point in the session
        # (no look-ahead in backtest where the synthesized DF spans the full day).
        try:
            if isinstance(index_df5, pd.DataFrame) and not index_df5.empty and now is not None:
                index_df5 = index_df5.loc[index_df5.index <= now]
        except Exception:
            pass
        regime = "chop"
        regime_conf = 0.5
        regime_diagnostics = None
        try:
            # days MUST cover the daily regime detector's needs (EMA200 + BB-width
            # window); 30 starved it -> it returned chop/insufficient every cycle,
            # leaving the daily squeeze/trend layer inert for the broad-market gate.
            daily_idx = self.sdk.get_daily(
                self.raw_cfg.get("directional_bias", {}).get("index_symbol", "NSE:NIFTY 50"),
                days=DailyRegimeDetector.MIN_BARS_REQUIRED,
            ) if not env.DRY_RUN else None
            if hasattr(self.regime_gate, "compute_regime_multi_tf") and daily_idx is not None:
                try:
                    regime, regime_conf, regime_diagnostics = self.regime_gate.compute_regime_multi_tf(
                        df5=index_df5, daily_df=daily_idx, symbol="INDEX",
                    )
                except Exception:
                    regime, regime_conf = self.regime_gate.compute_regime(index_df5)
            else:
                regime, regime_conf = self.regime_gate.compute_regime(index_df5)
        except Exception as regime_e:
            logger.warning("DISPATCH_REGIME_FAILED | %s — defaulting to chop", regime_e)

        # ---- 5. Directional bias update (mirrors old path) ----
        if self.directional_bias.enabled:
            index_sym = self.raw_cfg["directional_bias"]["index_symbol"]
            if self.directional_bias.prev_close is None:
                if env.DRY_RUN:
                    self.directional_bias.set_prev_close_for_date(now)
                else:
                    try:
                        pdc = self.sdk.get_prevday_levels(index_sym).get("PDC", float("nan"))
                        if pdc and not pd.isna(pdc):
                            self.directional_bias.set_prev_close(pdc)
                    except Exception:
                        pass
            if self.directional_bias.prev_close is not None:
                if env.DRY_RUN:
                    index_price = self.directional_bias.get_backtest_price_at(now)
                else:
                    index_price = self._shared_ltp_cache.get_ltp(index_sym)
                if index_price is not None:
                    self.directional_bias.update_price(index_price)

        # ---- 6. Build cap_segment_map for batch metadata ----
        full_cap_map = self._load_cap_mapping()
        cap_segment_map = {s: full_cap_map.get(s, {}).get("cap_segment", "unknown")
                           for s in active_syms_with_data}

        # ---- 7. Build symbol_data_map (mirrors old structure_prep block) ----
        _t_data_prep_start = _time_mod.perf_counter()
        symbol_data_map: Dict[str, tuple] = {}
        screener_logger = get_screener_logger()
        for sym in active_syms_with_data:
            if sym in api_df5_cache:
                df5 = api_df5_cache[sym]
            elif self._precomputed_5m and sym in self._precomputed_5m:
                df5 = self._get_precomputed_5m(sym, now, self.cfg.screener_store_5m_max)
            elif sym in df5_by_symbol:
                df5 = df5_by_symbol[sym]
            else:
                continue
            if not validate_df(df5, min_rows=min_bars_for_processing):
                continue
            # Resolve levels
            if levels_by_symbol and sym in levels_by_symbol and levels_by_symbol[sym]:
                lvl = levels_by_symbol[sym]
            elif levels_by_symbol is not None:
                try:
                    lvl = self._levels_for(sym, df5, now)
                except Exception:
                    lvl = {"PDH": float("nan"), "PDL": float("nan"),
                           "PDC": float("nan"), "ORH": float("nan"), "ORL": float("nan")}
            else:
                lvl = self._levels_for(sym, df5, now)
            symbol_data_map[sym] = (df5, lvl)

        if not symbol_data_map:
            logger.info("DISPATCH_DATA_EMPTY | No symbols with sufficient data at bar %s", now)
            # Still flush any deferred close_window events so TagMap stays consistent.
            for ev in post_close_events:
                self._tag_map.close_window(ev.setup)
                logger.info("DISPATCH_CLOSE_WINDOW_POST | %s (closed after data-empty at bar %s)", ev.setup, now_t)
            self._last_dispatch_ts = now
            return None

        _t_data_prep_end = _time_mod.perf_counter()
        shortlist = sorted(symbol_data_map.keys())
        shortlist_count = len(shortlist)
        logger.info(
            "DISPATCH_SCAN | active_syms=%d with_data=%d regime=%s | bar %s",
            len(active_syms), shortlist_count, regime, now,
        )

        # ---- 8. Plan batches ----
        levels_for_plan = {sym: lvl for sym, (_, lvl) in symbol_data_map.items()}
        df5_for_plan = {sym: df5 for sym, (df5, _) in symbol_data_map.items()}
        plan_batches = self._dispatch_planner.plan(
            now,
            self._tag_map,
            df5_for_plan,
            levels_for_plan,
            session_date=session_date_obj,
            regime=regime,
            cap_segment_map=cap_segment_map,
            regime_diagnostics=regime_diagnostics,
            daily_dict=getattr(self, "_daily_dict_cache", None) or {},
        )
        if not plan_batches:
            logger.info("PLAN_EMPTY | bar %s", now)
            # Still flush any deferred close_window events so TagMap stays consistent.
            for ev in post_close_events:
                self._tag_map.close_window(ev.setup)
                logger.info("DISPATCH_CLOSE_WINDOW_POST | %s (closed after plan-empty at bar %s)", ev.setup, now_t)
            self._last_dispatch_ts = now
            return {"decisions": [], "symbol_data_map": symbol_data_map,
                    "shortlist_count": shortlist_count, "screener_logger": screener_logger,
                    "shortlist": shortlist}

        # ---- 9. Submit to ProcessPoolExecutor ----
        _t_submit_start = _time_mod.perf_counter()
        futures = []
        with perf("scan", "dispatch_submit", n_batches=len(plan_batches), n_symbols=shortlist_count):
            for batch in plan_batches:
                fut = self._executor.submit(dispatch_worker_batch, batch)
                futures.append((fut, {item[0] for item in batch.items}))
        _t_submit_end = _time_mod.perf_counter()
        logger.info(
            "DISPATCH_BATCH_SUBMIT | %d batches (%d symbols) | submit_time=%.2fs",
            len(futures), shortlist_count, _t_submit_end - _t_submit_start,
        )

        # ---- 10. Collect results ----
        _t_collect_start = _time_mod.perf_counter()
        all_sym_decisions: List[tuple] = []
        for fut, _expected_syms in futures:
            try:
                batch_results = fut.result(timeout=60)
                all_sym_decisions.extend(batch_results)
            except Exception as fut_e:
                logger.exception("DISPATCH_BATCH_FAILED | %s", fut_e)

        _t_collect_end = _time_mod.perf_counter()
        decisions_accept = [(sym, d) for sym, d in all_sym_decisions if d.accept]
        decisions_reject = [(sym, d) for sym, d in all_sym_decisions if not d.accept]
        logger.info(
            "DISPATCH_COMPLETE | %d total → %d accept + %d reject | collect=%.2fs | bar %s",
            len(all_sym_decisions), len(decisions_accept), len(decisions_reject),
            _t_collect_end - _t_collect_start, now,
        )

        # Log screener accept/reject events (mirrors old path)
        for sym, decision in all_sym_decisions:
            df5 = symbol_data_map.get(sym, (None,))[0]
            if decision.accept:
                if screener_logger and df5 is not None and not df5.empty:
                    try:
                        screener_logger.log_accept(
                            sym,
                            timestamp=now.isoformat(),
                            setup_type=decision.setup_type or "unknown",
                            regime=decision.regime or "unknown",
                            size_mult=decision.size_mult,
                            min_hold_bars=decision.min_hold_bars,
                            all_reasons=decision.reasons,
                            structure_confidence=getattr(decision, "structure_confidence", 0),
                            current_price=float(df5["close"].iloc[-1]) if not df5.empty else 0,
                            vwap=float(df5.get("vwap", pd.Series([0])).iloc[-1]) if not df5.empty else 0,
                            regime_diagnostics=getattr(decision, "regime_diagnostics", None),
                        )
                    except Exception:
                        pass
            else:
                if screener_logger:
                    top_reason = (
                        next((r for r in decision.reasons if r.startswith("regime_block:")), None)
                        or (decision.reasons[0] if decision.reasons else "reject")
                    )
                    try:
                        screener_logger.log_reject(
                            sym,
                            top_reason,
                            timestamp=now.isoformat(),
                            setup_type=decision.setup_type or "unknown",
                            regime=decision.regime or "unknown",
                            all_reasons=decision.reasons,
                            structure_confidence=getattr(decision, "structure_confidence", 0),
                            current_price=float(df5["close"].iloc[-1]) if df5 is not None and not df5.empty else 0,
                            regime_diagnostics=getattr(decision, "regime_diagnostics", None),
                        )
                    except Exception:
                        pass

        # Emit structure_collect timing event
        try:
            from config.logging_config import get_timing_logger
            _lg = get_timing_logger()
            if _lg is not None:
                from utils.perf_timer import is_enabled as _perf_enabled
                if _perf_enabled():
                    _lg.log_event(
                        ts=_time_mod.time(), pid=os.getpid(),
                        stage="scan", substage="dispatch_collect",
                        duration_ms=round((_t_collect_end - _t_collect_start) * 1000.0, 3),
                        n_batches=len(futures), n_symbols=shortlist_count,
                        n_accepted=len(decisions_accept),
                    )
        except Exception:
            pass

        # ---- Phase 2: close windows that ended at the just-dispatched bar ----
        # These were deferred so dispatch could fire with the window still open.
        for ev in post_close_events:
            self._tag_map.close_window(ev.setup)
            logger.info("DISPATCH_CLOSE_WINDOW_POST | %s (closed after dispatch at bar %s)", ev.setup, now_t)

        self._last_dispatch_ts = now

        return {
            "decisions": decisions_accept,
            "symbol_data_map": symbol_data_map,
            "shortlist_count": shortlist_count,
            "screener_logger": screener_logger,
            "shortlist": shortlist,
        }

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
        # In backtest, df5 is multi-day; df5.index[0] is yesterday's bar, so we must
        # filter to today's bars before checking the earliest bar's time.
        has_opening_range_bars = False
        orb_end_time = dtime(9, 30)

        for sym, df5 in df5_by_symbol.items():
            if df5 is None or len(df5) < 1:
                continue
            try:
                today_bars = df5[df5.index.date == session_date]
            except Exception:
                today_bars = df5
            if len(today_bars) < 1:
                continue
            earliest_bar_time = today_bars.index[0].time() if hasattr(today_bars.index[0], 'time') else None
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
                # Narrowed to tag_map active symbols — downstream lookup only queries
                # shortlist symbols, so PDH/PDL/PDC for non-universe symbols is waste.
                _univ = self._tag_map.active_symbols()
                _scan_syms = _univ if _univ else set(self.core_symbols)
                levels_by_symbol = {}
                for sym in _scan_syms:
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

        # Narrow ORB compute to universe-union — universe is finalized by 09:30
        # (cross-day at startup + gap_fade lazy at 09:15-09:30), ORB fires at
        # 09:40, downstream lookup at line 1532 only queries symbols in
        # shortlist symbols — computing levels for non-universe symbols is pure waste.
        _univ = self._tag_map.active_symbols()
        _scan_syms = (set(df5_by_symbol.keys()) & _univ) if _univ else set(df5_by_symbol.keys())
        logger.info(
            f"ORB_CACHE | Computing ORH/ORL/PDH/PDL/PDC for {len(_scan_syms)} universe symbols at {current_time} (session_date={session_date})"
        )

        levels_by_symbol = {}
        success_count = 0
        fail_count = 0

        for sym in _scan_syms:
            df5 = df5_by_symbol.get(sym)
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

    def _recover_orb_levels_from_precomputed_5m(self, session_date) -> Dict[str, Dict[str, float]]:
        """Backtest fallback: compute ORB from precomputed 5m feathers (no broker API)."""
        import time as time_module
        from datetime import time as dtime

        start_time = time_module.perf_counter()
        precomputed = getattr(self, "_precomputed_5m", None) or {}
        orb_start_t = dtime(9, 15)
        orb_end_t = dtime(9, 30)

        _univ = self._tag_map.active_symbols()
        _recover_syms = _univ if _univ else self.core_symbols
        logger.info(
            f"ORB_RECOVERY_PRECOMPUTED | Recovering ORB for {len(_recover_syms)} "
            f"symbols from precomputed 5m bars."
        )

        levels_by_symbol = {}
        success_count = 0
        fail_count = 0
        for sym in _recover_syms:
            try:
                df5_full = precomputed.get(sym)
                if df5_full is None or len(df5_full) < 1:
                    orh = orl = float("nan")
                else:
                    today_bars = df5_full[df5_full.index.date == session_date]
                    orb_window = today_bars[
                        (today_bars.index.time >= orb_start_t)
                        & (today_bars.index.time < orb_end_t)
                    ]
                    if len(orb_window) >= 1:
                        orh = float(orb_window["high"].max())
                        orl = float(orb_window["low"].min())
                    else:
                        orh = orl = float("nan")

                daily = self.sdk.get_daily(sym, days=210)
                level_dict = get_previous_day_levels(
                    daily_df=daily,
                    session_date=session_date,
                    fallback_df=None,
                    enable_fallback=False,
                )
                pdh = level_dict.get("PDH", float("nan"))
                pdl = level_dict.get("PDL", float("nan"))
                pdc = level_dict.get("PDC", float("nan"))

                levels_by_symbol[sym] = {
                    "ORH": orh, "ORL": orl,
                    "PDH": pdh, "PDL": pdl, "PDC": pdc,
                }
                if not (pd.isna(orh) or pd.isna(orl)):
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                logger.warning(f"ORB_RECOVERY_PRECOMPUTED | {sym}: Failed - {e}")
                levels_by_symbol[sym] = {
                    "ORH": float("nan"), "ORL": float("nan"),
                    "PDH": float("nan"), "PDL": float("nan"), "PDC": float("nan"),
                }
                fail_count += 1

        elapsed = time_module.perf_counter() - start_time
        logger.info(
            f"ORB_RECOVERY_PRECOMPUTED | Done. Recovered ORH/ORL for "
            f"{success_count}/{len(_recover_syms)} symbols | Failed: {fail_count} | "
            f"Time: {elapsed:.2f}s"
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

        # Check if SDK supports historical 1m fetch.
        # In backtest, MockBroker lacks this — fall back to the precomputed 5m
        # feathers we already loaded for today's 09:15-09:30 window.
        if not hasattr(self.sdk, 'get_historical_1m'):
            precomputed = getattr(self, "_precomputed_5m", None)
            if precomputed:
                logger.info(
                    "ORB_RECOVERY | SDK lacks get_historical_1m — falling back to "
                    "precomputed 5m bars for ORB recovery (backtest mode)."
                )
                return self._recover_orb_levels_from_precomputed_5m(session_date)
            logger.warning("ORB_RECOVERY | SDK doesn't support get_historical_1m. ORB levels unavailable.")
            return {}

        start_time = time_module.perf_counter()
        logger.info(f"ORB_RECOVERY | Late start detected. Fetching historical 1m data for ORB window (09:15-09:30)")

        orb_start = dt.combine(session_date, dtime(9, 15))
        orb_end = dt.combine(session_date, dtime(9, 30))

        levels_by_symbol = {}
        success_count = 0
        fail_count = 0

        # Narrow recovery to universe-union — ORB levels are only queried for
        # symbols in the scan shortlist (= universe-union). Fetching 1m
        # historical for non-universe symbols just burns broker rate-limits.
        _univ = self._tag_map.active_symbols()
        _recover_syms = _univ if _univ else self.core_symbols
        logger.info(f"ORB_RECOVERY | Recovering levels for {len(_recover_syms)} universe symbols")
        for sym in _recover_syms:
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
        # ORH/ORL: only cache once ORB period (09:30) has finalized. Caching NaN
        # ORH/ORL pre-09:30 poisons the cache for the rest of the session — every
        # afternoon trade on the symbol then sees NaN even though the ORB bars
        # arrive later. Force recomputation until 09:30 has passed.
        has_prev_day = not (pd.isna(pdh) or pd.isna(pdl))
        has_orb = not (pd.isna(orh) or pd.isna(orl))
        try:
            now_time = now.time() if hasattr(now, "time") else None
        except Exception:
            now_time = None
        orb_finalized = (now_time is not None and now_time >= dtime(9, 30))
        if has_prev_day and (has_orb or orb_finalized):
            with self._levels_cache_lock:
                self._levels_cache[key] = out
        return out

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

        # Find previous trading day (skip weekends AND NSE holidays)
        today_dt = _now_naive_ist().date()
        from datetime import timedelta as _td
        from utils.util import is_trading_day
        prev_day = today_dt - _td(days=1)
        # Walk backwards through weekends + holidays. Cap at 10 attempts to
        # avoid an infinite loop if the holiday calendar is corrupted.
        for _ in range(10):
            if is_trading_day(prev_day):
                break
            prev_day -= _td(days=1)
        else:
            logger.warning(
                "API_WARMUP_CACHE | Could not find a trading day in the last 10 days "
                "from %s — skipping warmup", today_dt
            )
            self._api_warmup_loaded = True
            return

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
        """Load precomputed enriched 5m bars from feather cache (backtest only).

        Two data sources (tried in order):
        1. Monthly consolidated file: backtest-cache-download/monthly/{year_month}_5m_enriched.feather
           (OCI pods only have this — fast path, one file for all symbols)
        2. Per-symbol files: cache/ohlcv_archive/{SYMBOL}/{SYMBOL}_5minutes_enriched.feather
           (Local dev has these — slower, one file per symbol)
        """
        from pathlib import Path
        loaded = 0
        single_day_count = 0

        # Path 1: Try monthly consolidated file (OCI fast path)
        # Same file MockBroker._load_enriched_5m uses (backtest-cache-download/monthly/)
        # Also check /app/backtest-cache-download/monthly/ for OCI absolute path
        monthly_dir = Path("backtest-cache-download/monthly")
        if not monthly_dir.exists():
            # OCI pods run from /app — try absolute path
            monthly_dir = Path("/app/backtest-cache-download/monthly")
        if not monthly_dir.exists():
            # Also try relative to cache/ (some setups)
            monthly_dir = Path("cache/preaggregate")
        logger.info("PRECOMPUTED_5M | monthly_dir=%s exists=%s cwd=%s",
                     monthly_dir, monthly_dir.exists(), Path.cwd())
        if monthly_dir.exists():
            try:
                # Get session date from MockBroker (set by --session-date CLI arg)
                _session_date = getattr(self.sdk, '_dry_session_date', None)
                if _session_date is None:
                    _session_date = self.raw_cfg.get("session_date")
                from_dt = pd.to_datetime(_session_date) if _session_date else None
                if from_dt is None:
                    raise ValueError("No session_date for monthly file lookup")
                year_month = f"{from_dt.year}_{from_dt.month:02d}"
                monthly_file = monthly_dir / f"{year_month}_5m_enriched.feather"
                logger.info("PRECOMPUTED_5M | Looking for %s exists=%s", monthly_file, monthly_file.exists())
                if monthly_file.exists():
                    df_all = pd.read_feather(monthly_file)
                    df_all["date"] = pd.to_datetime(df_all["date"])
                    if df_all["date"].dt.tz is not None:
                        df_all["date"] = df_all["date"].dt.tz_localize(None)
                    for sym_raw, group in df_all.groupby("symbol"):
                        sym = f"NSE:{sym_raw}"
                        df_sym = group.drop(columns=["symbol"]).set_index("date").sort_index()
                        if sym in self.core_symbols or f"NSE:{sym_raw}" in self.core_symbols:
                            self._precomputed_5m[sym] = df_sym
                            loaded += 1
                            if len(set(df_sym.index.date)) < 2:
                                single_day_count += 1
                    if loaded > 0:
                        logger.info("PRECOMPUTED_5M | Loaded %d symbols from monthly file %s",
                                    loaded, monthly_file.name)
            except Exception as e:
                logger.warning("PRECOMPUTED_5M | Monthly file load failed: %s", e)

        # Path 2: Per-symbol files (local dev fallback)
        if loaded == 0:
            cache_dir = Path("cache/ohlcv_archive")
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
        Build the symbol-to-cap_data map used by Stage-0 and plan augmentation.

        Returns:
            Dict mapping 'NSE:SYM' -> {
                "cap_segment": large_cap / mid_cap / small_cap / micro_cap / unknown,
                "mis_enabled": bool,
                "mis_leverage": float | None,
            }

        Sources:
          - cap_segment: data/cap_segments/cap_segments_latest.json
            (refreshed weekly via scripts/refresh-cap-segments.sh from
            niftyindices.com — NIFTY 100 / Midcap 150 / Smallcap 250 /
            Microcap 250 constituents)
          - mis_enabled / mis_leverage: nse_all.json (kept as the source for
            MIS data within the cap_map for now; the parent screener also
            uses the live MIS_FETCHER for is_mis_allowed checks on the
            universe, which is the actual gate)

        market_cap_cr was previously included but never consumed — removed.

        Market Cap Segments (NSE India standards):
        - large_cap: top 100 (NIFTY 100)
        - mid_cap: 101-250 (NIFTY Midcap 150)
        - small_cap: 251-500 (NIFTY Smallcap 250)
        - micro_cap: 501-750 (NIFTY Microcap 250)
        - unknown: outside top-750 (very illiquid / new listings)
        """
        if hasattr(self, "_cap_map_cache"):
            return self._cap_map_cache

        try:
            from services.symbol_metadata import get_all_cap_segments
            cap_segments = get_all_cap_segments()

            # MIS info still rides on nse_all.json; the live MIS_FETCHER is
            # the actual gate at universe-build time (see is_mis_allowed
            # check around line 1991).
            import json
            from pathlib import Path
            nse_file = Path(__file__).parent.parent / "nse_all.json"
            mis_info: dict = {}
            if nse_file.exists():
                with nse_file.open() as f:
                    data = json.load(f)
                for item in data:
                    raw_sym = item["symbol"]
                    sym = f"NSE:{raw_sym[:-3]}" if raw_sym.endswith(".NS") else raw_sym
                    mis_info[sym] = {
                        "mis_enabled": item.get("mis_enabled", False),
                        "mis_leverage": item.get("mis_leverage"),
                    }

            all_syms = set(cap_segments.keys()) | set(mis_info.keys())
            cap_map = {}
            for sym in all_syms:
                mis = mis_info.get(sym, {})
                cap_map[sym] = {
                    "cap_segment": cap_segments.get(sym, "unknown"),
                    "mis_enabled": mis.get("mis_enabled", False),
                    "mis_leverage": mis.get("mis_leverage"),
                }

            self._cap_map_cache = cap_map
            mis_count = sum(1 for v in cap_map.values() if v.get("mis_enabled"))
            classified = sum(1 for v in cap_map.values() if v.get("cap_segment") != "unknown")
            logger.info(
                f"CAP_MAPPING | Loaded {len(cap_map)} symbols "
                f"({classified} classified by cap_segment, {mis_count} MIS-enabled)"
            )
            return cap_map

        except Exception as e:
            logger.warning(f"CAP_MAPPING | Failed to load cap mapping: {e}")
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
