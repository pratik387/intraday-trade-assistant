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
  3) Rank: rank_candidates → keep ≥ rank_exec_threshold AND ≥ rank_pctl_min percentile
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

import pandas as pd

from config.filters_setup import load_filters
from config.logging_config import get_agent_logger, get_screener_logger, get_ranking_logger, get_events_decision_logger

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
from services.planner_internal import generate_trade_plan
from structures.main_detector import MainDetector
from services.ranker import rank_candidates, get_strategy_threshold

# orders & execution
from services.orders.order_queue import OrderQueue
from services.scan.energy_scanner import EnergyScanner
from diagnostics.diag_event_log import diag_event_log, mint_trade_id
import uuid


logger = get_agent_logger()

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

        # NEW: de-dupe memory (per symbol last accepted)
        # stores: {symbol: {"ts": pd.Timestamp, "setup": str|None, "score": float}}
        self._last_entry: Dict[str, Dict[str, object]] = {}

        self._eod_done: bool = False
        self._request_exit: bool = False
        
        self._opening_block = (
            str(raw.get("opening_block_start_hhmm", "")) or None,
            str(raw.get("opening_block_end_hhmm", "")) or None,
        )

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
        # Future: Trigger rank_candidates() update with HTF context

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
                    reasons=updated_reasons
                )

            enhanced.append(adjusted_candidate)

        return enhanced

    def _on_5m_close(self, symbol: str, bar_5m: pd.Series) -> None:
        """Main driver: invoked for each CLOSED 5m bar of any symbol."""
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
        try:
            df5_by_symbol: Dict[str, pd.DataFrame] = {}
            for s in self.core_symbols:
                df5 = self.agg.get_df_5m_tail(s, self.cfg.screener_store_5m_max)
                if df5 is not None and not df5.empty and len(df5) >= 5:
                    df5_by_symbol[s] = df5
            if df5_by_symbol:
                feats_df = self.scanner.compute_features(df5_by_symbol, lookback_bars=20)
                feats_df = self._filter_stage0(feats_df, now)  # liquidity + vwap proximity + momentum + (opt) vol persistence
                # Use scanner's select_shortlist for proper structured logging
                shortlist_dict = self.scanner.select_shortlist(feats_df)
                shortlist = shortlist_dict.get("long", []) + shortlist_dict.get("short", [])
        except Exception as e:
            logger.exception("EnergyScanner failed; fallback shortlist: %s", e)
            shortlist = self._fallback_shortlist()

        # ENHANCED LOGGING: Stage-0 completion with accurate counts
        eligible_symbols = len(df5_by_symbol) if df5_by_symbol else 0
        total_symbols = len(self.core_symbols)
        shortlist_count = len(shortlist)

        logger.info("SCANNER_COMPLETE | Processed %d eligible of %d total symbols → %d shortlisted (%.1f%%) | Stage-0→Gates",
                   eligible_symbols, total_symbols, shortlist_count,
                   (shortlist_count/max(eligible_symbols,1))*100)
        if not shortlist:
            return

        # ---------- Gate per candidate (structure + regime + events + news) ----------
        index_df5 = self._index_df5()
        decisions: List[Tuple[str, Decision]] = []
        screener_logger = get_screener_logger()
        for sym in shortlist:
            df5 = self.agg.get_df_5m_tail(sym, self.cfg.screener_store_5m_max)
            if df5 is None or df5.empty or len(df5) < 5:
                continue

            lvl = self._levels_for(sym, df5, now)
            try:
                decision = self.decision_gate.evaluate(
                    symbol=sym,
                    now=now,
                    df1m_tail=self.agg.get_df_1m_tail(sym, 60),
                    df5m_tail=df5,
                    index_df5m=index_df5,
                    levels=lvl,
                )
            except Exception as e:
                logger.exception("decision_gate failed for %s: %s", sym, e)
                continue

            if not decision.accept:
                top_reason = next((r for r in decision.reasons if r.startswith("regime_block:")), None) or \
                             (decision.reasons[0] if decision.reasons else "reject")
                logger.debug(
                    "DECISION:REJECT sym=%s setup=%s regime=%s reason=%s | all=%s",
                    sym, decision.setup_type, decision.regime, top_reason, ";".join(decision.reasons),
                )

                # Log screener rejection with detailed context
                screener_logger.log_reject(
                    sym,
                    top_reason,
                    timestamp=now.isoformat(),
                    setup_type=decision.setup_type or "unknown",
                    regime=decision.regime or "unknown",
                    all_reasons=decision.reasons,
                    structure_confidence=getattr(decision, 'structure_confidence', 0),
                    current_price=df5['close'].iloc[-1] if not df5.empty else 0
                )
                continue

            logger.info(
                "DECISION:ACCEPT sym=%s setup=%s regime=%s size_mult=%.2f hold_bars=%d | %s",
                sym, decision.setup_type, decision.regime, decision.size_mult, decision.min_hold_bars,
                ";".join(decision.reasons),
            )

            # Log screener acceptance with detailed context
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
                vwap=df5.get('vwap', pd.Series([0])).iloc[-1] if not df5.empty else 0
            )
            decisions.append((sym, decision))

        gate_accept_count = len(decisions)
        logger.info("GATES_COMPLETE | %d→%d symbols (%.1f%%) | Gates→Ranking", shortlist_count, gate_accept_count, (gate_accept_count/max(shortlist_count,1))*100)
        if not decisions:
            return

        dec_map = {s: d for (s, d) in decisions}

        # ---------- Rank (distribution clamp) ----------
        ranked: List[Tuple[str, float]] = self._rank_by_intraday_edge([s for s, _ in decisions], decisions)
        if not ranked:
            return

        # percentile gate: compute once for this batch
        pctl_score = self._compute_percentile_cut(ranked, self.cfg.rank_pctl_min)
        max_trades_per_cycle = self.raw_cfg.get("max_trades_per_cycle", 10)
        trades_planned = 0

        # ---------- Plan & de-dupe & enqueue ----------
        ranking_logger = get_ranking_logger()
        events_logger = get_events_decision_logger()
        for i, (sym, score) in enumerate(ranked):
            # Get strategy-specific threshold
            decision = dec_map.get(sym)
            strategy_type = getattr(decision, "setup_type", None) if decision else None
            threshold = get_strategy_threshold(strategy_type) if strategy_type else self.cfg.rank_exec_threshold

            # absolute threshold gate
            if score < threshold:
                logger.info("RANK:REJECT sym=%s score=%.3f < threshold=%.3f (strategy=%s)",
                           sym, score, threshold, strategy_type)

                # Log ranking rejection
                ranking_logger.log_reject(
                    sym,
                    "score_below_threshold",
                    timestamp=now.isoformat(),
                    rank_score=score,
                    threshold=threshold,
                    strategy_type=strategy_type or "unknown",
                    rank_position=i + 1,
                    total_candidates=len(ranked),
                    percentile_score=pctl_score
                )
                continue
            # percentile gate
            if score < pctl_score:
                # Log percentile rejection
                ranking_logger.log_reject(
                    sym,
                    "score_below_percentile",
                    timestamp=now.isoformat(),
                    rank_score=score,
                    percentile_score=pctl_score,
                    strategy_type=strategy_type or "unknown",
                    rank_position=i + 1,
                    total_candidates=len(ranked)
                )
                continue

            # Log ranking acceptance (passed both threshold and percentile gates)
            ranking_logger.log_accept(
                sym,
                timestamp=now.isoformat(),
                rank_score=score,
                threshold=threshold,
                percentile_score=pctl_score,
                strategy_type=strategy_type or "unknown",
                rank_position=i + 1,
                total_candidates=len(ranked)
            )

            df5 = self.agg.get_df_5m_tail(sym, self.cfg.screener_store_5m_max)
            daily_df = self.sdk.get_daily(sym, days=90)

            try:
                decision = dec_map[sym]
                # Use new structure system approach with setup_candidates
                setup_candidates = getattr(decision, 'setup_candidates', None)
                if setup_candidates:
                    # Phase 1.4: Enhance candidate strength with HTF 15m confirmation
                    setup_candidates = self._enhance_candidates_with_htf(sym, setup_candidates)
                    plan = generate_trade_plan(df=df5, symbol=sym, daily_df=daily_df, setup_candidates=setup_candidates)
                else:
                    # Fallback for compatibility during transition
                    plan = generate_trade_plan(df=df5, symbol=sym, daily_df=daily_df, setup_type=decision.setup_type)
            except Exception as e:
                logger.exception("planner failed for %s: %s", sym, e)
                continue
            if not plan:
                logger.info("SKIP %s: empty plan (no_setup)", sym)
                events_logger.log_reject(sym, "empty_plan", timestamp=now.isoformat(), strategy_type=strategy_type or "unknown")
                continue

            # 1) Eligibility
            if not plan.get("eligible", False):
                rejection_reason = (plan.get("quality") or {}).get("rejection_reason", "unknown")
                cautions = ";".join((plan.get("notes") or {}).get("cautions", []))
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
            last5 = df5.iloc[-1] if (df5 is not None and not df5.empty) else None
            bar5 = {}
            if last5 is not None:
                for k in ("open", "high", "low", "close", "volume", "vwap", "adx", "bb_width_proxy"):
                    if k in last5.index:
                        bar5[k] = float(last5.get(k, 0.0))

            features = {
                "bar5": bar5,
                "ranker": {"rank_score": float(score)},
                "time": {"minute_of_day": now.hour * 60 + now.minute, "day_of_week": now.weekday()},
            }

            if plan.get("price") is None:
                plan["price"] = (plan.get("entry") or {}).get("reference")
            plan["decision_ts"] = str(now)

            # Flatten decision to serializable dict for diag log
            reasons_str = None
            if decision_obj is not None:
                r = getattr(decision_obj, "reasons", None)
                if isinstance(r, (list, tuple)): reasons_str = ";".join(str(x) for x in r)
                elif r is not None: reasons_str = str(r)
            decision_dict = {
                "setup_type": getattr(decision_obj, "setup_type", None) if decision_obj is not None else None,
                "regime": getattr(decision_obj, "regime", None) if decision_obj is not None else None,
                "reasons": reasons_str,
                "size_mult": getattr(decision_obj, "size_mult", None) if decision_obj is not None else None,
                "min_hold_bars": getattr(decision_obj, "min_hold_bars", None) if decision_obj is not None else None,
            }
            diag_event_log.log_decision(symbol=plan["symbol"], now=now, plan=plan, features=features, decision=decision_dict)

            exec_item = {
                "symbol": plan["symbol"],
                "plan": {
                    "side": "BUY" if plan["bias"] == "long" else "SELL",
                    "qty": int(plan["sizing"]["qty"]),
                    "entry_zone": (plan["entry"] or {}).get("zone"),
                    "price": (plan["entry"] or {}).get("reference"),
                    "hard_sl": (plan.get("stop") or {}).get("hard"),
                    "targets": plan.get("targets"),
                    "trail": plan.get("trail"),
                    "trade_id": plan["trade_id"],
                    "orh": (plan.get("levels") or {}).get("ORH"),
                    "orl": (plan.get("levels") or {}).get("ORL"),
                    "decision_ts": plan["decision_ts"],
                    "strategy": plan.get("strategy", ""),  # Add missing strategy field
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

    # ---------- EOD handler ----------
    def _handle_eod(self, now: datetime) -> None:
        try:
            logger.warning("EOD: cutoff reached at %s — stopping for the day", now)
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

    def _levels_for(self, symbol: str, df5: pd.DataFrame, now) -> Dict[str, float]:
        """Prev-day PDH/PDL/PDC and today ORH/ORL (cached per (symbol, session_date))."""
        try:
            session_date = df5.index[-1].date() if (df5 is not None and not df5.empty) else None
        except Exception:
            session_date = None
        key = (symbol, session_date)
        cached = self._levels_cache.get(key)
        if cached:
            return cached

        pdh = pdl = pdc = float("nan")
        try:
            logger.debug(f"LEVELS: Getting daily data for {symbol}, session_date={session_date}")
            daily = self.sdk.get_daily(symbol, days=12)
            logger.debug(f"LEVELS: Daily data shape: {daily.shape if daily is not None else None}")

            if daily is not None and not daily.empty:
                d = daily.copy()
                logger.debug(f"LEVELS: Daily data columns: {list(d.columns)}")
                logger.debug(f"LEVELS: Daily data index type: {type(d.index)}")

                if "date" in d.columns:
                    d["date"] = pd.to_datetime(d["date"]); d = d.sort_values("date").set_index("date")
                else:
                    d.index = pd.to_datetime(d.index); d = d.sort_index()

                logger.debug(f"LEVELS: After date processing, shape: {d.shape}")

                if session_date is not None:
                    pre_filter_size = len(d)
                    d = d[d.index.date < session_date]
                    logger.debug(f"LEVELS: After session_date filter ({session_date}): {len(d)}/{pre_filter_size}")

                if "volume" in d.columns:
                    pre_vol_size = len(d)
                    d = d[d["volume"].fillna(0) > 0]
                    logger.debug(f"LEVELS: After volume filter: {len(d)}/{pre_vol_size}")

                pre_clean_size = len(d)
                d = d[d["high"].notna() & d["low"].notna()]
                logger.debug(f"LEVELS: After high/low filter: {len(d)}/{pre_clean_size}")

                if not d.empty:
                    prev = d.iloc[-1]
                    pdh = float(prev["high"]); pdl = float(prev["low"]); pdc = float(prev.get("close", float("nan")))
                    logger.debug(f"LEVELS: Computed PDH={pdh}, PDL={pdl}, PDC={pdc} from date {prev.name}")
                else:
                    logger.warning(f"LEVELS: No valid previous day data for {symbol} after filtering")
                    # For backtests, we might not have previous day data - try to estimate from current session
                    if df5 is not None and not df5.empty and len(df5) > 10:
                        # Use early session data as rough estimates
                        early_bars = df5.iloc[:min(10, len(df5))]
                        pdh_est = float(early_bars['high'].max() * 1.02)  # 2% above early high
                        pdl_est = float(early_bars['low'].min() * 0.98)   # 2% below early low
                        pdc_est = float(early_bars['close'].iloc[-1])
                        logger.info(f"LEVELS: Using estimated levels for {symbol}: PDH={pdh_est:.2f}, PDL={pdl_est:.2f}, PDC={pdc_est:.2f}")
                        pdh, pdl, pdc = pdh_est, pdl_est, pdc_est
            else:
                logger.warning(f"LEVELS: No daily data available for {symbol}")
                # Fallback for backtests - estimate from current data if available
                if df5 is not None and not df5.empty and len(df5) > 10:
                    early_bars = df5.iloc[:min(10, len(df5))]
                    pdh = float(early_bars['high'].max() * 1.02)
                    pdl = float(early_bars['low'].min() * 0.98)
                    pdc = float(early_bars['close'].iloc[-1])
                    logger.info(f"LEVELS: Using fallback estimated levels for {symbol}: PDH={pdh:.2f}, PDL={pdl:.2f}, PDC={pdc:.2f}")
        except Exception as e:
            # CRITICAL FIX: Log previous day level computation failures
            import traceback
            logger.error(f"LEVELS: Failed to compute PDH/PDL/PDC for {symbol}: {e}")
            logger.error(f"LEVELS: Traceback: {traceback.format_exc()}")
            pass

        try:
            logger.debug(f"LEVELS: Computing opening range for {symbol}, df5 shape: {df5.shape if df5 is not None else None}")
            orh, orl = levels.opening_range(df5)
            orh = float(orh); orl = float(orl)
            logger.debug(f"LEVELS: Computed ORH={orh}, ORL={orl}")
        except Exception as e:
            # CRITICAL FIX: Log opening range computation failures
            import traceback
            logger.error(f"LEVELS: Failed to compute ORH/ORL for {symbol}: {e}")
            logger.error(f"LEVELS: Traceback: {traceback.format_exc()}")
            orh = orl = float("nan")

            # Fallback: compute simple opening range from first few bars
            if df5 is not None and not df5.empty and len(df5) >= 3:
                opening_bars = df5.iloc[:min(3, len(df5))]  # First 15 minutes (3 x 5min bars)
                orh = float(opening_bars['high'].max())
                orl = float(opening_bars['low'].min())
                logger.info(f"LEVELS: Using fallback ORH/ORL for {symbol}: ORH={orh:.2f}, ORL={orl:.2f}")

        out = {"PDH": pdh, "PDL": pdl, "PDC": pdc, "ORH": orh, "ORL": orl}

        # Log levels to screener structured logging for analysis
        valid_levels = {k: v for k, v in out.items() if not pd.isna(v)}
        screener_logger = get_screener_logger()

        if valid_levels:
            screener_logger.log_accept(
                symbol,
                timestamp=now.isoformat(),
                action_type="levels_computed",
                levels_count=len(valid_levels),
                **valid_levels  # PDH, PDL, PDC, ORH, ORL as separate fields
            )
            logger.debug(f"LEVELS: Computed levels for {symbol}: {valid_levels} (total: {len(valid_levels)}/5)")
        else:
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
                score = mi.compute_intraday_breakout_score(df5)
            except Exception as e:
                # CRITICAL FIX: Log breakout score computation failures
                logger.error(f"SCREENER: Failed to compute breakout score for {sym}: {e}")
                continue
            if score >= 0.0:
                out.append(sym)
        return out[:60]

    def _rank_by_intraday_edge(self, symbols: List[str], decisions: List[Tuple[str, Decision]]) -> List[Tuple[str, float]]:
        """Map symbols → rank_scores via ranker (kept identical to your current wiring)."""
        rows_for_ranker: List[Dict] = []
        for sym in symbols:
            df5 = self.agg.get_df_5m_tail(sym, self.cfg.screener_store_5m_max)
            if df5 is None or df5.empty:
                continue

            last = df5.iloc[-1]
            # simple features for ranker
            if "volume" in df5.columns:
                recent_vol = df5["volume"].tail(24)
                med_vol = float(recent_vol[recent_vol > 0].median() or 1.0)
                volume_ratio = float(last.get("volume", 0.0) or 0.0) / (med_vol or 1.0)
            else:
                volume_ratio = 1.0

            adx = float(last.get("adx", 0.0) or 0.0)
            vwap = float(last.get("vwap", last.get("close", 0.0)) or 0.0)
            close = float(last.get("close", 0.0) or 0.0)
            above_vwap = bool(close >= vwap) if vwap else False

            sq_pct = None
            if "bb_width_proxy" in df5.columns:
                recent_bw = df5["bb_width_proxy"].tail(24).dropna()
                if len(recent_bw) > 1:
                    cur_bw = float(last.get("bb_width_proxy", 0.0) or 0.0)
                    sq_pct = float((recent_bw <= cur_bw).mean() * 100.0)

            # Get strategy type and regime from decisions
            strategy_type = None
            regime_context = None
            for s, d in decisions:
                if s == sym:
                    strategy_type = getattr(d, "setup_type", None)
                    regime_context = getattr(d, "regime", None)
                    break

            # Extract HTF 15m context for ranking multipliers (Phase 1.3)
            htf_15m_context = {}
            df15 = self.agg.get_df_15m_tail(sym, 10)  # Last 10 x 15m bars
            if df15 is not None and not df15.empty and len(df15) >= 2:
                last_15m = df15.iloc[-1]
                prev_15m = df15.iloc[-2]

                # 15m trend direction (price and ADX)
                htf_15m_context["trend_aligned"] = float(last_15m.get("close", 0.0)) > float(prev_15m.get("close", 0.0))
                htf_15m_context["adx_15m"] = float(last_15m.get("adx", 0.0) or 0.0)

                # 15m volume multiplier (relative to 15m average)
                if "volume" in df15.columns and len(df15) >= 3:
                    recent_vol_15m = df15["volume"].tail(6).median()
                    current_vol_15m = float(last_15m.get("volume", 0.0) or 0.0)
                    htf_15m_context["volume_mult_15m"] = (current_vol_15m / recent_vol_15m) if recent_vol_15m > 0 else 1.0
                else:
                    htf_15m_context["volume_mult_15m"] = 1.0

            # Strategy type detection (debug logging removed for cleaner output)

            rows_for_ranker.append({
                "symbol": sym,
                "strategy_type": strategy_type,
                "regime": regime_context,
                "daily_score": 0.0,
                "intraday": {
                    "volume_ratio": volume_ratio,
                    "adx": adx,
                    "above_vwap": above_vwap,
                    "squeeze_pctile": sq_pct,
                },
                "htf_15m": htf_15m_context,
            })

        if not rows_for_ranker:
            return []

        # Log strategy distribution
        strategy_counts = {}
        for row in rows_for_ranker:
            strategy = row.get("strategy_type") or "unknown"
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        strategy_summary = ", ".join(f"{k}:{v}" for k, v in strategy_counts.items())
        logger.info(f"RANKER_INPUT | {len(rows_for_ranker)} symbols by strategy: {strategy_summary}")

        # Determine most common regime for ranking context (diagnostic report insight)
        regime_counts = {}
        for _, d in decisions:
            regime = getattr(d, "regime", None)
            if regime:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1

        # Use most common regime as context for ranking
        common_regime = max(regime_counts.keys(), key=lambda k: regime_counts[k]) if regime_counts else None
        if common_regime:
            logger.info(f"REGIME_CONTEXT | Using {common_regime} for ranking (regimes: {regime_counts})")

        # Apply rank_top_n cap from config
        rank_top_n = self.raw_cfg.get("rank_top_n", 100)
        ranked_rows = rank_candidates(rows_for_ranker, top_n=rank_top_n, regime_context=common_regime)

        # ENHANCED FILTERING: Apply rank floors and per-cycle caps (from plan document)
        if ranked_rows:
            initial_count = len(ranked_rows)

            # Apply rank score floor and percentile floor
            min_rank_score = float(self.raw_cfg.get("rank_exec_threshold", 1.0))
            min_percentile = float(self.raw_cfg.get("rank_pctl_min", 0.60))

            # Filter by rank score and percentile
            filtered_rows = []
            for row in ranked_rows:
                rank_score = float(row.get("rank_score", 0.0))
                rank_percentile = float(row.get("rank_percentile", 0.7))

                if rank_score >= min_rank_score and rank_percentile >= min_percentile:
                    filtered_rows.append(row)

            # Apply max per cycle cap
            max_per_cycle = int(self.raw_cfg.get("max_per_cycle", 100))
            if len(filtered_rows) > max_per_cycle:
                # Sort by rank_score descending and take top N
                filtered_rows.sort(key=lambda x: float(x.get("rank_score", 0.0)), reverse=True)
                filtered_rows = filtered_rows[:max_per_cycle]

            ranked_rows = filtered_rows
            logger.info(f"ENHANCED_FILTERING | {initial_count} -> {len(ranked_rows)} symbols after rank floor {min_rank_score}, percentile {min_percentile}, max_per_cycle {max_per_cycle}")

        if len(rows_for_ranker) > rank_top_n:
            logger.info(f"RANKER_OUTPUT | Capped to top {len(ranked_rows)}/{len(rows_for_ranker)} symbols (rank_top_n={rank_top_n})")
        else:
            logger.info(f"RANKER_OUTPUT | All {len(ranked_rows)} symbols ranked")
        return [(r.get("symbol", "?"), float(r.get("rank_score", 0.0))) for r in ranked_rows]

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

    def _filter_stage0(self, feats: pd.DataFrame, now_ts) -> pd.DataFrame:
        """
        Stage-0 shortlist filter: liquidity + VWAP proximity (time-aware) + momentum clamp.
        Optional: volume persistency & vol ratio if columns are present.
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

        if "vol_persist_ok" in df.columns:
            df = df[df["vol_persist_ok"] >= (1 if vp_bars >= 2 else 0)]
        if "vol_ratio" in df.columns:
            df = df[df["vol_ratio"] >= vr_min]

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
