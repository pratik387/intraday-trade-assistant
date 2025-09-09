from __future__ import annotations

"""
ScreenerLive — orchestrates the live intraday pipeline.

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
from diagnostics.diag_event_log import diag_event_log, mint_trade_id
import uuid


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

        raw = load_filters()
        try:
            self.cfg = ScreenerConfig(
                screener_store_5m_max=int(raw["screener_store_5m_max"]),
                rank_exec_threshold=float(raw["rank_exec_threshold"]),
                producer_min_interval_sec=int(raw["producer_min_interval_sec"]),
                intraday_cutoff_hhmm=str(raw["eod_squareoff_hhmm"]),
            )
        except KeyError as e:
            raise KeyError(f"ScreenerLive: missing config key {e!s}") from e

        self.agg = BarBuilder(
            bar_5m_span_minutes=5,
            on_1m_close=self._on_1m_close,
            on_5m_close=self._on_5m_close,
            index_symbols=self._index_symbols(),
        )

        self.ws = WSClient(sdk=sdk, on_tick=self.agg.on_tick)
        
        self.detector = StructureEventDetector()
        news_cfg = raw.get("news_gate")
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

        scanner_cfg = raw.get("energy_scanner")
        self.scanner = EnergyScanner(
            top_k_long=scanner_cfg["top_k_long"],
            top_k_short=scanner_cfg["top_k_short"]
        )

        self._last_produced_at: Optional[datetime] = None
        self._levels_cache: Dict[str, Dict[str, float]] = {}

        self._load_core_universe()
        self.router = TickRouter(
            on_tick=self.agg.on_tick,
            token_to_symbol=self.token_map,
        )
        self.ws.on_message(self.router.handle_raw)

        self.subs = SubscriptionManager(self.ws)

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

            lvl = self._levels_for(sym, df5)

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
                rb = next((r for r in decision.reasons if r.startswith("regime_block:")), None)
                top_reason = rb or (decision.reasons[0] if decision.reasons else "reject")
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

        dec_map = {s: d for (s, d) in decisions}

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

            # --- DECISION: canonical payload (no fallbacks) ---

            # 1) stable id
            if "trade_id" not in plan:
                plan["trade_id"] = mint_trade_id(sym, token=uuid.uuid4().hex[:8])

            # 2) bar5 snapshot (only these keys)
            df5 = self.agg.get_df_5m_tail(sym, self.cfg.screener_store_5m_max)
            last5 = df5.iloc[-1] if (df5 is not None and not df5.empty) else None
            bar5 = {}
            if last5 is not None:
                if "open"  in last5.index: bar5["open"]  = float(last5.get("open", 0.0))
                if "high"  in last5.index: bar5["high"]  = float(last5.get("high", 0.0))
                if "low"   in last5.index: bar5["low"]   = float(last5.get("low", 0.0))
                if "close" in last5.index: bar5["close"] = float(last5.get("close", 0.0))
                if "volume" in last5.index:         bar5["volume"] = float(last5.get("volume", 0.0))
                if "vwap" in last5.index:           bar5["vwap"] = float(last5.get("vwap", bar5.get("close", 0.0)))
                if "adx" in last5.index:            bar5["adx"] = float(last5.get("adx", 0.0))
                if "bb_width_proxy" in last5.index: bar5["bb_width_proxy"] = float(last5.get("bb_width_proxy", 0.0))

            # 3) canonical features layout (rank_score lives here)
            features = {
                "bar5": bar5,
                "ranker": {"rank_score": float(score)},
                "time": {"minute_of_day": now.hour * 60 + now.minute, "day_of_week": now.weekday()},
            }

            # 5) canonical plan reference price (report reads plan.price)
            if plan.get("price") is None:
                plan["price"] = (plan.get("entry") or {}).get("reference")
                
            plan["decision_ts"] = str(now)

            # 6) canonical decision dict (reasons = string)
            decision_obj = dec_map.get(sym)
            reasons_str = None
            if decision_obj is not None:
                r = getattr(decision_obj, "reasons", None)
                if isinstance(r, (list, tuple)):
                    reasons_str = ";".join(str(x) for x in r)
                elif r is not None:
                    reasons_str = str(r)
            decision_dict = {
                "setup_type": getattr(decision_obj, "setup_type", None) if decision_obj is not None else None,
                "regime": getattr(decision_obj, "regime", None) if decision_obj is not None else None,
                "reasons": reasons_str,
                "size_mult": getattr(decision_obj, "size_mult", None) if decision_obj is not None else None,
                "min_hold_bars": getattr(decision_obj, "min_hold_bars", None) if decision_obj is not None else None,
            }

            # 7) emit (time-naive `now` from pipeline)
            diag_event_log.log_decision(
                symbol=plan["symbol"],
                now=now,
                plan=plan,
                features=features,
                decision=decision_dict,
            )

            
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
        try:
            self.symbol_map = self.sdk.get_symbol_map()
            self.token_map = self.sdk.get_token_map()
            self.core_symbols = list(self.symbol_map.keys())
        except Exception as e:
            raise RuntimeError(f"ScreenerLive: sdk.list_equities() failed: {e}")

    def _index_symbols(self) -> List[str]:
        return []

    def _index_df5(self) -> pd.DataFrame:
        idx = self.agg.index_df_5m()
        if isinstance(idx, dict):
            for _, df in idx.items():
                return df
            return pd.DataFrame()
        return idx if isinstance(idx, pd.DataFrame) else pd.DataFrame()

    def _levels_for(self, symbol: str, df5: pd.DataFrame) -> Dict[str, float]:
        # cache key per session date
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
            daily = self.sdk.get_daily(symbol, days=12)
            if daily is not None and not daily.empty:
                d = daily.copy()
                if "date" in d.columns:
                    d["date"] = pd.to_datetime(d["date"])
                    d = d.sort_values("date").set_index("date")
                else:
                    d.index = pd.to_datetime(d.index)
                    d = d.sort_index()
                if session_date is not None:
                    d = d[d.index.date < session_date]
                if "volume" in d.columns:
                    d = d[d["volume"].fillna(0) > 0]
                d = d[d["high"].notna() & d["low"].notna()]
                if not d.empty:
                    prev = d.iloc[-1]
                    pdh = float(prev["high"]); pdl = float(prev["low"]); pdc = float(prev.get("close", float("nan")))
        except Exception:
            pass

        try:
            orh, orl = levels.opening_range(df5)
            orh = float(orh); orl = float(orl)
        except Exception:
            orh = orl = float("nan")

        out = {"PDH": pdh, "PDL": pdl, "PDC": pdc, "ORH": orh, "ORL": orl}
        self._levels_cache.clear()
        self._levels_cache[key] = out
        return out



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
        return out[:60]

    def _rank_by_intraday_edge(self, symbols: List[str]) -> List[Tuple[str, float]]:
        rows_for_ranker: List[Dict] = []

        for sym in symbols:
            df5 = self.agg.get_df_5m_tail(sym, self.cfg.screener_store_5m_max)
            if df5 is None or df5.empty:
                continue

            last = df5.iloc[-1]

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

            rows_for_ranker.append({
                "symbol": sym,
                "daily_score": 0.0,
                "intraday": {
                    "volume_ratio": volume_ratio,
                    "adx": adx,
                    "above_vwap": above_vwap,
                    "squeeze_pctile": sq_pct,
                },
            })

        if not rows_for_ranker:
            return []

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
