from __future__ import annotations

"""
ScreenerLive — orchestrates the live intraday pipeline.

Responsibilities
- Maintain WebSocket stream → BarBuilder (1m → 5m closed bars).
- Keep a small in-memory store of recent 5m bars per symbol (via BarBuilder APIs).
- On each 5m close: run Stage‑0 shortlist, gates, ranker, planner.
- Enqueue orders (or log) via OrderQueue.

Notes
- No REST market‑data calls during market hours. All data comes from WS ticks.
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
"""

from dataclasses import dataclass
from datetime import datetime, time as dtime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config.filters_setup import load_filters  # strict loader, no defaults
from config.logging_config import get_loggers

# ingest / streaming
from services.ingest.stream_client import WSClient
from services.ingest.subscription_manager import SubscriptionManager
from services.ingest.bar_builder import BarBuilder
from services.ingest.tick_router import TickRouter

# gates
from services.gates.regime_gate import MarketRegimeGate
from services.gates.event_policy_gate import EventPolicyGate
from services.gates.news_spike_gate import NewsSpikeGate
from services.gates.trade_decision_gate import TradeDecisionGate, GateDecision as Decision

# planning & ranking
from services import levels
from services import metrics_intraday as mi
from services.planner_internal import generate_trade_plan
from services.events.structure_events import StructureEventDetector
from services.ranker import rank_candidates

# orders
from services.orders.order_queue import OrderQueue
from services.scan.energy_scanner import EnergyScanner


logger, trade_logger = get_loggers()

@dataclass
class ScreenerConfig:
    """Minimal Screener config — no defaults. All keys must be provided in filters.json."""
    screener_store_5m_max: int
    rank_exec_threshold: float
    producer_min_interval_sec: int
    intraday_cutoff_hhmm: str  # "HH:MM"


class ScreenerLive:
    """Live orchestrator. Construct once and call start()."""

    def __init__(self, *, sdk, order_queue: OrderQueue) -> None:
        self.sdk = sdk
        self.oq = order_queue

        # Load config (raises if any key is missing)
        raw = load_filters()
        try:
            self.cfg = ScreenerConfig(
                screener_store_5m_max=int(raw["screener_store_5m_max"]),
                rank_exec_threshold=float(raw["rank_exec_threshold"]),
                producer_min_interval_sec=int(raw["producer_min_interval_sec"]),
                intraday_cutoff_hhmm=str(raw["intraday_cutoff_hhmm"]),
            )
        except KeyError as e:  # re-raise with context
            raise KeyError(f"ScreenerLive: missing config key {e!s}") from e

        # ---------- Build core components ----------
        # Bar aggregator (1m builder → 5m rollup)
        self.agg = BarBuilder(
            bar_5m_span_minutes=5,
            on_1m_close=self._on_1m_close,
            on_5m_close=self._on_5m_close,
            index_symbols=self._index_symbols(),
        )

        # WS client forwards parsed ticks → BarBuilder.on_tick
        self.ws = WSClient(sdk=sdk, on_tick=self.agg.on_tick)  # adapter set elsewhere
        
        self.detector = StructureEventDetector()
        news_cfg = raw.get("news_gate")
        # Gates (pure, stateless)
        self.regime_gate = MarketRegimeGate(cfg=raw)
        self.event_gate = EventPolicyGate()
        self.news_gate = NewsSpikeGate(
            window_bars=news_cfg.get("window_bars"),
            vol_z_thresh=news_cfg.get("vol_z_thresh"),
            ret_z_thresh=news_cfg.get("ret_z_thresh"),
            body_atr_ratio_thresh=news_cfg.get("body_atr_ratio_thresh")
        )
        self.decision_gate = TradeDecisionGate(
            structure_detector=self.detector,
            regime_gate=self.regime_gate,
            event_policy_gate=self.event_gate,
            news_spike_gate=self.news_gate,
        )

        # Optional Stage‑0 energy scanner
        scanner_cfg = raw.get("energy_scanner")
        self.scanner = EnergyScanner(
            top_k_long=scanner_cfg["top_k_long"],
            top_k_short=scanner_cfg["top_k_short"]
        )

        # Caches
        self._last_produced_at: Optional[datetime] = None
        self._levels_cache: Dict[str, Dict[str, float]] = {}

        # Universe (core set we keep subscribed all day)
        self._load_core_universe()
        self.router = TickRouter(
            on_tick=self.agg.on_tick,
            token_to_symbol=self.token_map,   # built in _load_core_universe()
        )
        self.ws.on_message(self.router.handle_raw)

        self.subs = SubscriptionManager(self.ws)

        # EOD state flags (set at cutoff; main.py can exit when _request_exit flips)
        self._eod_done: bool = False
        self._request_exit: bool = False

        logger.info(
            "ScreenerLive init: universe=%d symbols, store5m=%d",
            len(self.core_symbols),
            self.cfg.screener_store_5m_max,
        )

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def start(self) -> None:
        """Connect WS and subscribe the core universe. Non-blocking."""
        # Subscribe first to avoid missing ticks immediately after connect
        self.subs.set_core(self.token_map)
        self.subs.start()
        self.ws.start()
        logger.info("WS connected; core subscriptions scheduled: %d symbols", len(self.core_symbols))

    def stop(self) -> None:
        try:
            self.subs.stop()
        except Exception:
            pass
        try:
            self.ws.stop()
        except Exception:
            pass
        logger.info("ScreenerLive stopped")

    # ---------------------------------------------------------------------
    # Callbacks from BarBuilder
    # ---------------------------------------------------------------------
    def _on_1m_close(self, symbol: str, bar_1m: pd.Series) -> None:
        # No heavy work here; NewsSpike gate reads recent 1m from BarBuilder when needed
        return

    def _on_5m_close(self, symbol: str, bar_5m: pd.Series) -> None:
        """Main driver: called on every CLOSED 5m bar per symbol."""
        # Keep store size bounded (BarBuilder retains full day; we just read tails when needed)
        # Guard cutoff time for new entries
        now = bar_5m.name if hasattr(bar_5m, "name") else datetime.now()
        if self._is_after_cutoff(now):
            if not getattr(self, "_eod_done", False):
                self._handle_eod(now)
            return

        # Pace Stage‑0 and planning; only run every producer_min_interval_sec
        if not self._should_produce(now):
            return
        self._last_produced_at = now

        # ---------- Stage‑0 shortlist (EnergyScanner) ----------
        shortlist: List[str] = []
        if self.scanner is not None:
            try:
                # Build 5m tails for the whole core universe (scanner operates on features_df)
                df5_by_symbol: Dict[str, pd.DataFrame] = {}
                for s in self.core_symbols:
                    df5 = self.agg.get_df_5m_tail(s, self.cfg.screener_store_5m_max)
                    if df5 is not None and not df5.empty and len(df5) >= 3:
                        df5_by_symbol[s] = df5
                if df5_by_symbol:
                    feats = self.scanner.compute_features(df5_by_symbol, lookback_bars=20)
                    picks = self.scanner.select_shortlist(feats)
                    shortlist = (picks.get("long", []) + picks.get("short", []))
            except Exception as e:
                logger.exception("EnergyScanner failed; falling back to naive shortlist: %s", e)
                shortlist = self._fallback_shortlist()
        else:
            shortlist = self._fallback_shortlist()

        if not shortlist:
            return

        # ---------- Per-candidate evaluation ----------
        index_df5 = self._index_df5()
        decisions: List[Tuple[str, Decision]] = []

        for sym in shortlist:
            df5 = self.agg.get_df_5m_tail(sym, self.cfg.screener_store_5m_max)
            if df5 is None or df5.empty or len(df5) < 5:
                continue

            # Compute/cached levels used by structure detectors
            lvl = self._levels_for(sym, df5)

            # Evaluate gates + structure to get a trade decision
            try:
                decision = self.decision_gate.evaluate(
                    symbol=sym,
                    now=now,
                    df1m_tail=self.agg.get_df_1m_tail(sym, 60),
                    df5m_tail=df5,
                    index_df5m=index_df5,
                    levels=lvl,
                )
            except Exception as e:  # gate failures should not break the producer
                logger.exception("decision_gate failed for %s: %s", sym, e)
                continue

            if not decision.accept:
                rb = next((r for r in decision.reasons if r.startswith("regime_block:")), None)
                top_reason = rb or (decision.reasons[0] if decision.reasons else "reject")

                # Log at INFO with full context (symbol, setup, regime, regime_conf)
                logger.info(
                    "DECISION:REJECT sym=%s setup=%s regime=%s reason=%s | all=%s",
                    sym,
                    decision.setup_type,
                    decision.regime,
                    top_reason,
                    ";".join(decision.reasons),
                )
                continue
            
            logger.info(
                "DECISION:ACCEPT sym=%s setup=%s regime=%s size_mult=%.2f hold_bars=%d | %s",
                sym,
                decision.setup_type,
                decision.regime,
                decision.size_mult,
                decision.min_hold_bars,
                ";".join(decision.reasons),
            )

            decisions.append((sym, decision))

        if not decisions:
            return

        # ---------- Rank & Plan ----------
        ranked: List[Tuple[str, float]] = self._rank_by_intraday_edge([s for s, _ in decisions])
        for sym, score in ranked:
            if score < self.cfg.rank_exec_threshold:
                continue

            df5 = self.agg.get_df_5m_tail(sym, self.cfg.screener_store_5m_max)
            daily_df = self.sdk.get_daily(sym, days=90)

            try:
                plan = generate_trade_plan(df=df5, symbol=sym, daily_df=daily_df)
            except Exception as e:
                logger.exception("planner failed for %s: %s", sym, e)
                continue
            
            if not plan:
                logger.info("SKIP %s: empty plan (no_setup)", sym)
                continue

            # 1) Eligibility
            if not plan.get("eligible", False):
                logger.info("SKIP %s: ineligible plan reasons=%s",
                            sym, ";".join((plan.get("notes") or {}).get("cautions", [])))
                continue

            # 2) Qty
            qty = int((plan.get("sizing") or {}).get("qty") or 0)
            if qty <= 0:
                logger.info("SKIP %s: qty<=0", sym)
                continue

            # 3) Bias → side
            bias = str(plan.get("bias", "")).lower()
            if bias not in ("long", "short"):
                logger.info("SKIP %s: bad bias=%r", sym, bias)
                continue
            
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
                },
                "meta": plan,
            }

            self.oq.enqueue(exec_item)
            logger.info("ENQUEUE %s score=%.3f reasons=%s", sym, score, ";".join(self._reasons_for(sym, decisions)))
        # ---------- EOD handler ----------
    def _handle_eod(self, now: datetime) -> None:
        """Stop subscriptions and stream at cutoff and request clean exit."""
        try:
            logger.warning("EOD: cutoff reached at %s — stopping for the day", now)
        except Exception:
            pass
        self._eod_done = True
        try:
            self.subs.stop()
        except Exception:
            pass
        try:
            self.ws.stop()
        except Exception:
            pass
        self._request_exit = True

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _load_core_universe(self):
        """Ask the SDK for equities once. Keep it simple and deterministic."""
        try:
            self.symbol_map = self.sdk.get_symbol_map()
            self.token_map = self.sdk.get_token_map()
            self.core_symbols = list(self.symbol_map.keys())
        except Exception as e:
            raise RuntimeError(f"ScreenerLive: sdk.list_equities() failed: {e}")

    def _index_symbols(self) -> List[str]:
        """Return index symbols you stream for regime (override if needed)."""
        # If your WS adapter uses different identifiers, adapt here.
        # Keeping empty by default; TradeDecisionGate will skip regime when absent.
        return []

    def _index_df5(self) -> pd.DataFrame:
        idx = self.agg.index_df_5m()
        if isinstance(idx, dict):  # multiple
            # pick the primary if multiple present
            for _, df in idx.items():
                return df
            return pd.DataFrame()
        return idx if isinstance(idx, pd.DataFrame) else pd.DataFrame()

    def _levels_for(self, symbol: str, df5: pd.DataFrame) -> Dict[str, float]:
        cache = self._levels_cache.get(symbol)
        if cache:
            return cache
        try:
            pdh, pdl, pdc = levels.yesterday_levels(df5)
            orh, orl = levels.opening_range(df5)
        except Exception:
            pdh = pdl = pdc = orh = orl = float("nan")
        cache = {"PDH": pdh, "PDL": pdl, "PDC": pdc, "ORH": orh, "ORL": orl}
        self._levels_cache[symbol] = cache
        return cache

    def _fallback_shortlist(self) -> List[str]:
        """Very cheap shortlist: symbols with newest 5m bar volume z-score and range expansion.
        Used only if EnergyScanner isn't installed yet.
        """
        out: List[str] = []
        for sym in self.core_symbols:
            df5 = self.agg.get_df_5m_tail(sym, 25)
            if df5 is None or df5.empty or len(df5) < 10:
                continue
            try:
                score = mi.compute_intraday_breakout_score(df5)
            except Exception:
                continue
            if score >= 0.0:
                out.append(sym)
        return out[:60]  # keep it bounded

    def _rank_by_intraday_edge(self, symbols: List[str]) -> List[Tuple[str, float]]:
        rows_for_ranker: List[Dict] = []

        for sym in symbols:
            df5 = self.agg.get_df_5m_tail(sym, self.cfg.screener_store_5m_max)
            if df5 is None or df5.empty:
                continue

            last = df5.iloc[-1]

            # --- minimal, robust intraday feature pack for ranker ---
            # volume ratio = last5m / median of recent non-zero vols (~2h)
            if "volume" in df5.columns:
                recent_vol = df5["volume"].tail(24)
                med_vol = float(recent_vol[recent_vol > 0].median() or 1.0)
                volume_ratio = float(last.get("volume", 0.0) or 0.0) / (med_vol or 1.0)
            else:
                volume_ratio = 1.0

            # ADX from 5m bar (BarBuilder now sets this)
            adx = float(last.get("adx", 0.0) or 0.0)

            # above/below VWAP
            vwap = float(last.get("vwap", last.get("close", 0.0)) or 0.0)
            close = float(last.get("close", 0.0) or 0.0)
            above_vwap = bool(close >= vwap) if vwap else False

            # squeeze percentile from bb_width_proxy (optional)
            sq_pct = None
            if "bb_width_proxy" in df5.columns:
                recent_bw = df5["bb_width_proxy"].tail(24).dropna()
                if len(recent_bw) > 1:
                    cur_bw = float(last.get("bb_width_proxy", 0.0) or 0.0)
                    sq_pct = float((recent_bw <= cur_bw).mean() * 100.0)

            rows_for_ranker.append({
                "symbol": sym,
                "daily_score": 0.0,  # keep neutral unless you have a daily model
                "intraday": {
                    "volume_ratio": volume_ratio,
                    "adx": adx,
                    "above_vwap": above_vwap,
                    "squeeze_pctile": sq_pct,
                    # other keys (rsi, slopes, dist_from_level_bpct, acceptance_ok, bias) are optional
                },
            })

        if not rows_for_ranker:
            return []

        # Ask ranker to score and sort; keep all returned rows
        ranked_rows = rank_candidates(rows_for_ranker, top_n=len(rows_for_ranker))
        return [(r.get("symbol", "?"), float(r.get("rank_score", 0.0))) for r in ranked_rows]


    def _reasons_for(self, sym: str, decisions: List[Tuple[str, Decision]]) -> List[str]:
        for s, d in decisions:
            if s == sym:
                return d.reasons
        return []

    def _should_produce(self, now: datetime) -> bool:
        lp = self._last_produced_at
        if lp is None:
            return True
        delta = (now - lp).total_seconds()
        return delta >= self.cfg.producer_min_interval_sec

    def _is_after_cutoff(self, now: datetime) -> bool:
        hh, mm = self.cfg.intraday_cutoff_hhmm.split(":")
        cutoff = dtime(hour=int(hh), minute=int(mm))
        return now.time() >= cutoff
