"""Cron-triggered handlers for overnight setups (close_dn_overnight_long).

Short-lived entry points called by cron once per trading day:
  - run_entry(): 15:25 IST. Compute signal, place MOC BUY only.
  - run_place_exit(): ~16:05 IST. Place exit AMO SELL + GTT catastrophe stop
    (AMO window opens at 16:00, so the exit can't be placed in run_entry).
  - run_verify_exit(): 09:30 IST next day. Verify fills, settle, release.

Both are idempotent — safe to re-run on missed cron fires.

Spec: specs/2026-05-21-close_dn_overnight_long-paper-trade-implementation-spec.md
"""
from __future__ import annotations

import logging
import time as _time_mod
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Use the agent logger so warnings/errors from the cron-triggered handlers
# actually surface in the cron log file. The default `logging.getLogger(...)`
# named logger has no handler in this entrypoint and was swallowing all
# diagnostics (including the "0 daily DataFrames" ERROR we added).
try:
    from config.logging_config import get_agent_logger
    logger = get_agent_logger()
except Exception:
    logger = logging.getLogger(__name__)

from utils.price_utils import clamp_round_limit, round_to_tick


def _safe_quote(broker, symbol):
    """Best-effort (last_price, upper_circuit, lower_circuit) for LIMIT pricing.

    Prefers broker.get_quote (circuit bands); falls back to get_ltp (no bands)
    and finally (None, None, None). Never raises — a quote failure must not block
    order placement (the caller still tick-rounds and has a plan-price fallback).
    """
    get_quote = getattr(broker, "get_quote", None)
    if get_quote is not None:
        try:
            q = get_quote(symbol)
            lp = float(q.get("last_price") or 0.0) or None
            uc = float(q.get("upper_circuit_limit") or 0.0) or None
            lc = float(q.get("lower_circuit_limit") or 0.0) or None
            return lp, uc, lc
        except Exception as e:
            logger.debug("overnight: get_quote failed for %s (%s); falling back to LTP", symbol, e)
    try:
        lp = float(broker.get_ltp(symbol)) or None
        return lp, None, None
    except Exception:
        return None, None, None


_NSE_HOLIDAYS = None


def _nse_holidays() -> set:
    """Set of NSE trading holidays (IST dates) from assets/nse_holidays.json.

    Cached. Empty set on load failure -> graceful degradation to weekend-only
    skipping (the prior behavior) WITH a logged warning, so a missing/corrupt
    calendar fails loud, not silent. Calendar coverage must extend past the
    dates being traded (refresh assets/nse_holidays.json before each new FY).
    """
    global _NSE_HOLIDAYS
    if _NSE_HOLIDAYS is None:
        import json
        from datetime import datetime as _dt
        path = Path(__file__).resolve().parents[2] / "assets" / "nse_holidays.json"
        out = set()
        try:
            for r in json.load(open(path, encoding="utf-8")):
                ds = r.get("tradingDate") if isinstance(r, dict) else None
                if ds:
                    try:
                        out.add(_dt.strptime(ds, "%d-%b-%Y").date())
                    except (ValueError, TypeError):
                        pass
            if not out:
                logger.warning("_nse_holidays: %s parsed to 0 holidays — trading-day math is weekend-only", path)
        except Exception as e:
            logger.warning("_nse_holidays: could not load %s (%s) — trading-day math is weekend-only", path, e)
        _NSE_HOLIDAYS = out
    return _NSE_HOLIDAYS


def _is_trading_day(d: date) -> bool:
    """True iff d is an NSE trading day (not a weekend or exchange holiday).

    The overnight crons fire on weekdays (Mon-Fri) regardless of holidays, so the
    handlers must self-guard: running a LIVE entry on a holiday would compute
    signals on stale/closed-market data and could place real orders. Fail safe.
    """
    return d.weekday() < 5 and d not in _nse_holidays()


def _next_trading_day(d: date) -> date:
    """Next NSE trading day after `d`: skips weekends AND exchange holidays.

    Holiday-aware (assets/nse_holidays.json) — critical for multi-day setups:
    an exit/entry date that lands on a holiday would otherwise be missed by the
    EOD cron (the session never happens), orphaning the position (backtest_findings #0).
    """
    hols = _nse_holidays()
    cur = d + timedelta(days=1)
    while cur.weekday() >= 5 or cur in hols:
        cur += timedelta(days=1)
    return cur


def _select_overnight_setups(config: dict, *, paper_mode: bool) -> list:
    """Return SetupSpec list for setups with mode='overnight' that are eligible.

    In paper_mode, a setup is eligible if its raw_config has paper_enabled=True.
    In live mode, a setup is eligible if it is enabled=True.

    Note: get_active_setups() filters strictly by enabled=True, which doesn't
    work for paper-only setups (enabled=False, paper_enabled=True). We filter
    the full spec list directly so paper-mode can run setups that aren't yet
    live-active.
    """
    from services.dispatch.setup_registry import SetupRegistry
    registry = SetupRegistry.load_from_config(config)
    out: list = []
    for spec in registry._specs.values():  # noqa: SLF001
        if spec.mode != "overnight":
            continue
        if paper_mode:
            if bool(spec.raw_config.get("paper_enabled", False)):
                out.append(spec)
        else:
            if bool(spec.enabled):
                out.append(spec)
    return out


def run_entry(
    config: dict,
    broker,
    *,
    now_ist: Optional[pd.Timestamp] = None,
    paper_mode: bool = True,
) -> dict:
    """At 15:25 IST: compute signal, place orders, persist state.

    Returns a summary dict for logging/testing.
    """
    from utils.time_util import _now_naive_ist
    from services.capital_manager import OvernightSlotPool
    from services.dispatch.setup_registry import _import_path
    from services.setup_universe import close_dn_overnight_long_universe

    now = pd.Timestamp(now_ist) if now_ist is not None else _now_naive_ist()
    today = now.date()

    summary: dict = {
        "now_ist": str(now), "today": str(today),
        "paper_mode": paper_mode,
        "fired_count": 0, "skipped_count": 0, "rejected_count": 0,
        "events": [],
    }

    # Non-trading-day guard: the cron fires Mon-Fri regardless of holidays, but
    # a LIVE entry on a holiday would compute signals on stale/closed-market data
    # and could place real orders. Bail before any data fetch / order placement.
    if not _is_trading_day(today):
        logger.info("run_entry: %s is not an NSE trading day (weekend/holiday); skipping", today)
        summary["skipped_non_trading_day"] = True
        return summary

    paper_enabled_setups = _select_overnight_setups(config, paper_mode=paper_mode)
    if not paper_enabled_setups:
        logger.info("run_entry: no overnight setups active (paper_mode=%s); exit", paper_mode)
        return summary

    # Pool (shared across overnight setups — currently only close_dn_overnight_long)
    slot_cfg = paper_enabled_setups[0].raw_config["capital_allocation"]
    state_path = Path(slot_cfg["state_file"])
    pool = OvernightSlotPool(
        state_path,
        max_slots=int(slot_cfg["max_concurrent_slots"]),
        margin_per_slot=float(slot_cfg["margin_per_slot_inr"]),
        max_new_per_day=int(slot_cfg["max_new_positions_per_day"]),
    )

    # Decay tripwire check — skip dispatch for setups whose forward edge has
    # decayed (Task 7). The tripwire pauses dispatch when rolling N-trade PF
    # drops below floor for sustained_weeks. Manual unpause: delete state file
    # or call DecayTripwire.reset().
    from services.risk.decay_tripwire import DecayTripwire
    paused_names: List[str] = []
    for spec in paper_enabled_setups:
        tw_cfg = spec.raw_config.get("decay_tripwire")
        if tw_cfg is None:
            continue
        tw = DecayTripwire(
            setup_name=spec.name,
            state_path=Path(tw_cfg["state_file"]),
            window_trades=int(tw_cfg["window_trades"]),
            pf_floor=float(tw_cfg["pf_floor"]),
            sustained_weeks=int(tw_cfg["sustained_weeks"]),
        )
        if tw.is_paused():
            logger.warning(
                "run_entry: setup %s is PAUSED by decay tripwire (since %s); skipping dispatch",
                spec.name, tw._paused_since,  # noqa: SLF001
            )
            paused_names.append(spec.name)
    if paused_names:
        summary["paused_setups"] = paused_names
    paper_enabled_setups = [s for s in paper_enabled_setups if s.name not in set(paused_names)]
    if not paper_enabled_setups:
        pool.persist()
        logger.info("run_entry: all overnight setups paused by decay tripwire; exit")
        return summary

    # Iterate setups (currently only close_dn_overnight_long; loop is generic)
    for spec in paper_enabled_setups:
        detector_cls = _import_path(spec.detector_class_path)
        # Detector reads its params from the raw_config block; pass setup name via private key.
        detector = detector_cls({**spec.raw_config, "_setup_name": spec.name})

        # Read pre-filtered candidate list from data/close_dn_baseline/
        # (built at 09:30 IST by run_verify_exit). Each candidate carries
        # its pre-computed prior_close + prev_prior_close so we can
        # construct a 2-row df_daily without re-fetching daily history —
        # skips the _gather_daily_dict 50s cost entirely.
        candidates_path = (
            Path(__file__).resolve().parents[2]
            / "data" / "close_dn_baseline" / "candidates_latest.json"
        )
        prior_returns_by_symbol: Dict[str, Dict[str, float]] = {}
        if candidates_path.exists():
            try:
                import json as _json
                payload = _json.loads(candidates_path.read_text())
                expected_date = today.isoformat()
                actual_date = payload.get("session_date")
                if actual_date != expected_date:
                    logger.warning(
                        "run_entry: candidates_latest.json session_date=%s != today=%s; "
                        "stale snapshot — falling back to full universe",
                        actual_date, expected_date,
                    )
                else:
                    for entry in payload.get("candidates") or []:
                        sym = entry.get("symbol")
                        if sym:
                            prior_returns_by_symbol[sym] = entry
                    logger.info(
                        "run_entry: loaded %d pre-filtered candidates from %s",
                        len(prior_returns_by_symbol), candidates_path.name,
                    )
            except Exception as e:
                logger.warning("run_entry: failed to load candidates_latest.json: %s", e)

        if prior_returns_by_symbol:
            universe = set(prior_returns_by_symbol.keys())
            daily_dict: Dict[str, pd.DataFrame] = {}  # skip the expensive gather
        else:
            # Fallback path — fresh VM with no baseline yet, or stale snapshot.
            # Build the universe from scratch and gather daily history per symbol.
            logger.warning(
                "run_entry: no candidate file — falling back to full universe build "
                "(slow path: gathers daily_dict for ~2344 symbols)"
            )
            daily_dict = _gather_daily_dict(broker, spec.raw_config)
            try:
                universe = close_dn_overnight_long_universe(daily_dict, today, spec.raw_config)
            except Exception as e:
                logger.exception("run_entry: universe builder failed for %s: %s", spec.name, e)
                continue
        logger.info("run_entry: universe size for %s = %d", spec.name, len(universe))

        # Batch-fetch today's 5m bars for the (now small) candidate universe.
        # rps=20 / concurrency=30 — the memo's "40 RPS safe" data point
        # assumed a single consumer; during 15:25 the long-running paper-trade
        # is also hitting Upstox from the same IP, so half-rate coexists cleanly.
        intraday_5m_by_symbol: Dict[str, pd.DataFrame] = {}
        try:
            data_sdk = getattr(broker, "_data_sdk", None)
            if data_sdk is not None and hasattr(data_sdk, "async_fetch_intraday_5m_batch"):
                import asyncio as _asyncio
                import time as _time_perf
                _t0 = _time_perf.perf_counter()
                intraday_5m_by_symbol = _asyncio.run(
                    data_sdk.async_fetch_intraday_5m_batch(
                        list(universe), concurrency=30, rps=20.0,
                    )
                )
                logger.info(
                    "run_entry: intraday 5m batch fetched %d/%d symbols in %.1fs",
                    len(intraday_5m_by_symbol), len(universe),
                    _time_perf.perf_counter() - _t0,
                )
            else:
                logger.warning(
                    "run_entry: data_sdk has no async_fetch_intraday_5m_batch — "
                    "falling back to per-symbol broker.get_intraday_5m (slow)"
                )
        except Exception as e:
            logger.exception("run_entry: intraday batch fetch failed: %s", e)
            intraday_5m_by_symbol = {}

        # Aggregate per-symbol rejection_reason buckets so the cron log shows
        # WHY all N symbols rejected instead of just N. Same diagnostic that
        # the screener (services/dispatch/worker.py:248-254) carries through
        # screening.jsonl; the overnight cron had no equivalent surface.
        from collections import Counter as _Counter
        reject_reasons: _Counter = _Counter()

        # Iterate symbols
        for symbol in universe:
            # Per-symbol df_daily: prefer pre-computed candidate entry's
            # 2 closes (cheapest path); fall back to daily_dict from the
            # gather (only populated on the slow fallback above).
            cand = prior_returns_by_symbol.get(symbol)
            df_daily = None
            if cand is not None:
                df_daily = _mini_df_daily_from_candidate(cand, today)
            elif daily_dict:
                df_daily = daily_dict.get(symbol)

            ctx = _build_market_context(
                broker, symbol, now, today, spec.raw_config,
                intraday_5m=intraday_5m_by_symbol.get(symbol),
                df_daily=df_daily,
            )
            if ctx is None:
                summary["skipped_count"] += 1
                continue
            try:
                analysis = detector.detect(ctx)
            except Exception as e:
                logger.warning("run_entry: detector.detect failed for %s: %s", symbol, e)
                summary["rejected_count"] += 1
                reject_reasons["detector.detect EXCEPTION: " + type(e).__name__] += 1
                continue
            if not analysis.structure_detected or not analysis.events:
                summary["rejected_count"] += 1
                _r = getattr(analysis, "rejection_reason", None) or "no_event (no rejection_reason set)"
                reject_reasons[str(_r)[:120]] += 1
                continue
            evt = analysis.events[0]
            try:
                plan = detector.plan_long_strategy(ctx, evt)
            except Exception as e:
                logger.warning("run_entry: plan_long_strategy failed for %s: %s", symbol, e)
                plan = None
            if plan is None:
                summary["rejected_count"] += 1
                reject_reasons["plan_long_strategy returned None"] += 1
                continue

            # Reserve slot (fails if capacity or per-day cap hit)
            slot = pool.reserve(
                symbol=symbol,
                product=evt.context["product"],
                leverage=float(evt.context["leverage"]),
                today=today,
            )
            if slot is None:
                logger.info(
                    "run_entry: slot capacity hit (free=%d, new_today=%d); skipping %s",
                    pool.free_count(), pool.new_today_count(today), symbol,
                )
                summary["skipped_count"] += 1
                continue

            # Place marketable-LIMIT BUY. Kite bars plain MARKET orders on the
            # illiquid/trade-to-trade universe, so cross the spread with a LIMIT
            # at ref*(1+buffer). Ref = live LTP when available (the actual price
            # at 15:26), else the plan's 15:25-close entry.
            entry_buf_pct = float(spec.raw_config["entry_limit_buffer_pct"])
            ref_px, upper_circuit, _lc = _safe_quote(broker, symbol)
            if ref_px is None or ref_px <= 0:
                # Expected in paper/dry-run (no live LTP) — price off the plan's
                # 15:25-close entry. Logged so the live-vs-plan basis is observable.
                logger.debug("run_entry: live LTP/quote unavailable for %s (paper_mode=%s); "
                             "pricing BUY limit off plan.entry_price", symbol, paper_mode)
                ref_px = float(plan.entry_price)
            # Tick-valid AND <= upper circuit (Kite rejects non-tick prices and
            # any BUY above the circuit band — see 2026-06-29 DOLLAR/TIRUPATIFL).
            buy_limit = clamp_round_limit(
                ref_px * (1.0 + entry_buf_pct / 100.0), "BUY",
                upper_circuit=upper_circuit,
            )
            try:
                buy_order_id = _place_buy(
                    broker, symbol=symbol, qty=plan.qty,
                    product=evt.context["product"],
                    paper_mode=paper_mode,
                    trade_id=f"OVERNIGHT_{today.isoformat()}_{slot.slot_id}",
                    limit_price=buy_limit,
                )
            except Exception as e:
                logger.error("run_entry: BUY failed for %s: %s; releasing reservation", symbol, e)
                # Roll back the reservation by directly resetting (free) — slot
                # never transitioned past t0_open, so cleanest is to release.
                # Since release() requires t1_settling, we directly reset fields:
                slot.status = "free"
                slot.symbol = None
                slot.product = None
                slot.leverage = 1.0
                slot.margin_inr = 0.0
                slot.notional_inr = 0.0
                slot.reserved_today = None
                summary["skipped_count"] += 1
                continue

            # Determine fill price
            if paper_mode:
                fill_price = _paper_fill_price_entry(broker, symbol, today)
                if fill_price is None:
                    logger.warning(
                        "run_entry: paper fill price unavailable for %s; using plan.entry_price",
                        symbol,
                    )
                    fill_price = plan.entry_price
            else:
                fill_price = _live_poll_fill(broker, buy_order_id, timeout_sec=int(spec.raw_config["fill_poll_timeout_sec"]))
                if fill_price is None:
                    logger.warning(
                        "run_entry: %s BUY order %s did not fill within timeout",
                        symbol, buy_order_id,
                    )
                    # Leave the slot in t0_open with no buy_fill — verify-exit
                    # will detect the orphan and decide.
                    summary["skipped_count"] += 1
                    pool.persist()
                    continue

            pool.attach_buy_fill(
                slot.slot_id,
                fill_price=float(fill_price),
                fill_ts_iso=now.isoformat(),
                order_id=str(buy_order_id),
                qty=int(plan.qty),
            )

            summary["fired_count"] += 1
            summary["events"].append({
                "symbol": symbol, "qty": plan.qty,
                "product": evt.context["product"],
                "buy_fill_price": float(fill_price),
            })

    # Persist
    pool.persist()
    # Surface the top-K rejection reasons so an all-reject day is diagnosable
    # without re-running on a research workstation. The Counter is scoped per
    # setup but logged once at the end (currently only close_dn_overnight_long
    # is wired; if a second overnight setup ships, scope this per-setup).
    try:
        if reject_reasons:
            top = reject_reasons.most_common(8)
            summary["reject_reason_top"] = top
            logger.info("run_entry: top reject reasons:")
            for reason, n in top:
                logger.info("  %5d  %s", n, reason)
    except NameError:
        # reject_reasons only exists if at least one setup ran (paused setups skip the loop).
        pass
    logger.info(
        "run_entry: complete | fired=%d skipped=%d rejected=%d",
        summary["fired_count"], summary["skipped_count"], summary["rejected_count"],
    )
    return summary


def run_place_exit(
    config: dict,
    broker,
    *,
    now_ist: Optional[pd.Timestamp] = None,
    paper_mode: bool = True,
) -> dict:
    """At ~16:05 IST (T0): for each t0_open slot place the exit AMO SELL + a
    GTT catastrophe stop. Idempotent — slots already carrying an
    amo_sell_order_id are skipped, so a re-run (or the morning fallback)
    is safe. Refuses to run before the 16:00 AMO window opens.
    """
    from utils.time_util import _now_naive_ist
    from services.capital_manager import OvernightSlotPool

    now = pd.Timestamp(now_ist) if now_ist is not None else _now_naive_ist()
    summary: dict = {"now_ist": str(now), "paper_mode": paper_mode,
                     "placed_count": 0, "gtt_failed_count": 0, "events": []}

    if (now.hour, now.minute) < (16, 0):
        logger.warning("run_place_exit: before 16:00 AMO window (now=%s); refusing", now)
        summary["refused_amo_window"] = True
        return summary

    if not _is_trading_day(now.date()):
        logger.info("run_place_exit: %s is not an NSE trading day; skipping", now.date())
        summary["skipped_non_trading_day"] = True
        return summary

    setups = _select_overnight_setups(config, paper_mode=paper_mode)
    if not setups:
        logger.info("run_place_exit: no overnight setups active; exit")
        return summary
    sc = setups[0].raw_config
    slot_cfg = sc["capital_allocation"]
    state_path = Path(slot_cfg["state_file"])
    if not state_path.exists():
        logger.info("run_place_exit: no state file; nothing to place")
        return summary
    pool = OvernightSlotPool(
        state_path,
        max_slots=int(slot_cfg["max_concurrent_slots"]),
        margin_per_slot=float(slot_cfg["margin_per_slot_inr"]),
        max_new_per_day=int(slot_cfg["max_new_positions_per_day"]),
    )
    catastrophe_pct = float(sc["catastrophe_stop_pct"])
    gtt_buffer_pct = float(sc["gtt_limit_buffer_pct"])

    for slot in list(pool.active()):
        if slot.status != "t0_open":
            continue
        if slot.amo_sell_order_id is not None:
            continue
        if slot.buy_fill_price is None or slot.buy_fill_price <= 0 or slot.notional_inr <= 0:
            logger.warning("run_place_exit: slot %d has no buy fill; skipping", slot.slot_id)
            continue
        if slot.reserved_today is None:
            logger.warning("run_place_exit: slot %d has no reserved_today; skipping", slot.slot_id)
            continue
        # Capture the idealized paper entry reference (15:25 bar CLOSE, now
        # complete post-15:30) for exact slippage measurement later. Best-effort:
        # never let this block exit placement.
        try:
            ie = _paper_fill_price_entry(broker, slot.symbol, date.fromisoformat(slot.reserved_today))
            if ie is not None:
                slot.idealized_entry_price = float(ie)
        except Exception as e:
            logger.warning("run_place_exit: idealized-entry capture failed for %s: %s", slot.symbol, e)

        # Exit sells the ACTUAL filled qty (not a notional/fill re-estimate that
        # over/under-sold on 2026-07-01).
        qty = slot.exit_qty()
        next_day = _next_trading_day(date.fromisoformat(slot.reserved_today))
        # Per-slot isolation: a single AMO rejection (e.g. WELENT "insufficient
        # holding" on 2026-07-01) MUST NOT abort exit placement for the remaining
        # slots. Wrap the whole placement; persist in `finally` so an order that
        # DID place is never lost from state (the earlier end-only persist orphaned
        # 3 placed orders when the loop crashed).
        try:
            # Lower circuit for the SELL clamp (a catastrophe-floored sell could
            # fall below the band on a name near its lower circuit).
            _lp, _uc, lower_circuit = _safe_quote(broker, slot.symbol)
            # AMO LIMIT floor at the catastrophe level: fills at the pre-open
            # auction for any open >= -catastrophe (limit sells fill at the better
            # price); a gap below the floor is left to the GTT / morning failsafe.
            # Tick-valid AND >= lower circuit (Kite rejects non-tick/sub-circuit).
            amo_limit = clamp_round_limit(
                slot.buy_fill_price * (1.0 - catastrophe_pct / 100.0), "SELL",
                lower_circuit=lower_circuit,
            )
            amo_id = _place_amo_sell(
                broker, symbol=slot.symbol, qty=qty,
                product=slot.product or "CNC", paper_mode=paper_mode,
                trade_id=f"OVERNIGHT_AMO_{slot.reserved_today}_{slot.slot_id}",
                limit_price=amo_limit,
            )
            pool.attach_amo_sell(slot.slot_id, str(amo_id), next_day)
            # GTT trigger/limit must also be tick-valid (Kite rejects non-tick GTTs).
            trigger = round_to_tick(slot.buy_fill_price * (1.0 - catastrophe_pct / 100.0))
            limit = round_to_tick(trigger * (1.0 - gtt_buffer_pct / 100.0))
            try:
                gid = broker.place_gtt_stop(
                    symbol=slot.symbol, qty=qty,
                    trigger_price=trigger, limit_price=limit,
                    product=slot.product or "CNC",
                )
                slot.gtt_id = str(gid)
            except Exception as e:
                logger.error("run_place_exit: GTT place failed for %s: %s "
                             "(AMO still queued; morning failsafe covers it)", slot.symbol, e)
                summary["gtt_failed_count"] += 1
            summary["placed_count"] += 1
            summary["events"].append({"slot_id": slot.slot_id, "symbol": slot.symbol,
                                      "amo_sell_order_id": str(amo_id), "gtt_id": slot.gtt_id,
                                      "expected_exit_date": next_day.isoformat()})
        except Exception as e:
            logger.error("run_place_exit: exit placement FAILED for slot %d %s: %s "
                         "(continuing; 09:30 verify-exit will place a failsafe SELL)",
                         slot.slot_id, slot.symbol, e)
            summary["exit_failed_count"] = summary.get("exit_failed_count", 0) + 1
        finally:
            # Persist after EVERY slot so a later failure never orphans an
            # already-placed AMO/GTT from state.
            pool.persist()

    pool.persist()
    logger.info("run_place_exit: complete | placed=%d gtt_failed=%d",
                summary["placed_count"], summary["gtt_failed_count"])
    return summary


def run_verify_exit(
    config: dict,
    broker,
    *,
    now_ist: Optional[pd.Timestamp] = None,
    paper_mode: bool = True,
) -> dict:
    """At 09:30 IST: verify AMO fills, settle, release.

    Idempotent — safe to re-run after a missed cron fire.
    """
    from utils.time_util import _now_naive_ist
    from services.capital_manager import OvernightSlotPool
    from tools.sub7_validation.build_per_setup_pnl import (
        calc_fee_cnc, calc_fee_mtf, MTF_INTEREST_RATE_PER_DAY,
    )

    now = pd.Timestamp(now_ist) if now_ist is not None else _now_naive_ist()
    today = now.date()

    summary: dict = {
        "now_ist": str(now), "today": str(today),
        "paper_mode": paper_mode,
        "settled_count": 0, "released_count": 0,
        "orphan_t0_count": 0, "events": [],
    }

    # Non-trading-day guard: exit dates are always trading days (via
    # _next_trading_day), so a holiday morning has nothing legitimate to settle;
    # any stale orphan defers safely to the next trading day's verify-exit.
    if not _is_trading_day(today):
        logger.info("run_verify_exit: %s is not an NSE trading day; skipping", today)
        summary["skipped_non_trading_day"] = True
        return summary

    paper_enabled_setups = _select_overnight_setups(config, paper_mode=paper_mode)
    if not paper_enabled_setups:
        logger.info("run_verify_exit: no overnight setups active; exit")
        return summary

    slot_cfg = paper_enabled_setups[0].raw_config["capital_allocation"]
    state_path = Path(slot_cfg["state_file"])
    if not state_path.exists():
        logger.info("run_verify_exit: no state file at %s; nothing to verify", state_path)
        return summary

    pool = OvernightSlotPool(
        state_path,
        max_slots=int(slot_cfg["max_concurrent_slots"]),
        margin_per_slot=float(slot_cfg["margin_per_slot_inr"]),
        max_new_per_day=int(slot_cfg["max_new_positions_per_day"]),
    )

    # Phase 1: settle T0 slots whose AMO has executed (expected_exit_date <= today)
    for slot in list(pool.active()):
        if slot.status != "t0_open":
            continue
        if slot.expected_exit_date is None:
            logger.warning(
                "run_verify_exit: slot %d in t0_open without expected_exit_date -- orphan",
                slot.slot_id,
            )
            summary["orphan_t0_count"] += 1
            continue
        expected_exit = date.fromisoformat(slot.expected_exit_date)
        if today < expected_exit:
            logger.info(
                "run_verify_exit: slot %d not yet eligible (expected_exit=%s, today=%s)",
                slot.slot_id, expected_exit, today,
            )
            continue
        if today > expected_exit:
            # Stale slot: expected_exit was a PRIOR session that this cron
            # missed (e.g. paper-mode bug, holiday gap, cron-skip). Settling
            # here would use TODAY's 09:15 fill price instead of the actual
            # exit day's 09:15 — silently mis-prices the trade. Flag as orphan
            # and skip so it's surfaced for manual resolution / historical-
            # fetch rebuild (see Task #73).
            logger.warning(
                "run_verify_exit: slot %d STALE (expected_exit=%s < today=%s) -- "
                "orphan, requires manual settlement at correct historical price",
                slot.slot_id, expected_exit, today,
            )
            # Cancel any live GTT on this stale slot — the AMO almost certainly
            # already filled at the correct open, so a surviving GTT could trigger
            # on a now-flat position and open a NAKED SHORT.
            if slot.gtt_id and hasattr(broker, "cancel_gtt"):
                if not broker.cancel_gtt(slot.gtt_id):
                    logger.warning("run_verify_exit: stale-slot GTT %s cancel returned False "
                                   "for slot %d — verify manually (naked-short risk)",
                                   slot.gtt_id, slot.slot_id)
                slot.gtt_id = None
            summary["orphan_t0_count"] += 1
            continue
        if slot.buy_fill_price is None or slot.notional_inr <= 0:
            logger.warning(
                "run_verify_exit: slot %d missing buy_fill_price/notional -- orphan",
                slot.slot_id,
            )
            summary["orphan_t0_count"] += 1
            continue

        # Determine sell fill price
        if paper_mode:
            sell_price = _paper_fill_price_exit(broker, slot.symbol, expected_exit)
            if sell_price is None:
                logger.warning(
                    "run_verify_exit: paper fill unavailable for %s on %s",
                    slot.symbol, expected_exit,
                )
                continue
        else:
            sell_price = _live_check_amo_fill(broker, slot.amo_sell_order_id)
            if sell_price is None:
                logger.warning(
                    "run_verify_exit: live AMO %s did not fill; placing failsafe SELL",
                    slot.amo_sell_order_id,
                )
                # Failsafe: marketable-LIMIT SELL just below live LTP (Kite bars
                # plain MARKET on this universe). Fetch LTP first to price it.
                qty_for_failsafe = slot.exit_qty()
                failsafe_buf = float(paper_enabled_setups[0].raw_config["gtt_limit_buffer_pct"])
                try:
                    ltp_q, _uc, lower_circuit = _safe_quote(broker, slot.symbol)
                    ltp = ltp_q if (ltp_q is not None and ltp_q > 0) else float(broker.get_ltp(slot.symbol))
                    # Tick-valid AND >= lower circuit (Kite rejects non-tick /
                    # sub-circuit SELL prices).
                    failsafe_limit = clamp_round_limit(
                        ltp * (1.0 - failsafe_buf / 100.0), "SELL",
                        lower_circuit=lower_circuit,
                    )
                    _place_failsafe_sell(
                        broker,
                        symbol=slot.symbol, qty=qty_for_failsafe,
                        product=slot.product or "CNC",
                        limit_price=failsafe_limit,
                    )
                    sell_price = ltp
                except Exception as e:
                    logger.error(
                        "run_verify_exit: failsafe SELL failed for slot %d: %s",
                        slot.slot_id, e,
                    )
                    continue

        # Compute fees + interest — on the ACTUAL filled qty.
        qty = slot.exit_qty()
        buy_value = slot.buy_fill_price * qty
        sell_value = float(sell_price) * qty
        if (slot.product or "").upper() == "MTF":
            buy_day = (
                date.fromisoformat(slot.reserved_today)
                if slot.reserved_today is not None else expected_exit
            )
            hold_days = max(1, (expected_exit - buy_day).days)
            fees_total = calc_fee_mtf(buy_value, sell_value, slot.margin_inr, hold_days)
            borrowed = max(0.0, buy_value - slot.margin_inr)
            interest = borrowed * MTF_INTEREST_RATE_PER_DAY * hold_days
            fees_only = max(0.0, fees_total - interest)
        else:
            fees_only = calc_fee_cnc(buy_value, sell_value)
            interest = 0.0

        pool.settle(
            slot_id=slot.slot_id,
            sell_fill_price=float(sell_price),
            sell_fill_ts_iso=now.isoformat(),
            fees_inr=float(fees_only),
            interest_inr=float(interest),
        )

        # Idealized exit reference (09:15 open) for slippage — distinct from the
        # real live AMO fill. In paper mode this equals sell_price; in live it
        # differs and is what we measure slippage against. Best-effort.
        idealized_exit = None
        try:
            idealized_exit = _paper_fill_price_exit(broker, slot.symbol, expected_exit)
        except Exception as e:
            logger.warning("run_verify_exit: idealized-exit capture failed for %s: %s", slot.symbol, e)

        # Cancel the dangling catastrophe GTT now that the AMO has filled and
        # the slot is flat — a later GTT trigger would open a NAKED SHORT.
        if slot.gtt_id and hasattr(broker, "cancel_gtt"):
            ok = broker.cancel_gtt(slot.gtt_id)
            if not ok:
                logger.warning("run_verify_exit: GTT %s cancel returned False for slot %d "
                               "(verify manually — risk of naked short)", slot.gtt_id, slot.slot_id)
            slot.gtt_id = None

        summary["settled_count"] += 1
        settled = pool._get_slot(slot.slot_id)  # noqa: SLF001
        summary["events"].append({
            "slot_id": slot.slot_id,
            "symbol": slot.symbol,
            "buy_price": slot.buy_fill_price,
            "sell_price": float(sell_price),
            "realized_pnl": settled.realized_pnl_inr,
        })

        # Record settled trade in decay tripwire (Task 7). Re-instantiates per
        # settle; cheap because settles are O(slots) per day and small N. The
        # tripwire's rolling-PF check is what gates next-day dispatch.
        tw_cfg = paper_enabled_setups[0].raw_config.get("decay_tripwire")
        if tw_cfg is not None:
            from services.risk.decay_tripwire import DecayTripwire
            tw = DecayTripwire(
                setup_name=paper_enabled_setups[0].name,
                state_path=Path(tw_cfg["state_file"]),
                window_trades=int(tw_cfg["window_trades"]),
                pf_floor=float(tw_cfg["pf_floor"]),
                sustained_weeks=int(tw_cfg["sustained_weeks"]),
            )
            # Only attach the cost breakdown when we have a real settled net PnL;
            # otherwise leave fees/gross unset (legacy net-only record) rather than
            # fabricate a gross == fees row.
            if settled.realized_pnl_inr is not None:
                net_pnl = float(settled.realized_pnl_inr)
                total_cost = float(fees_only) + float(interest)
                tw.record_trade(
                    net_pnl_inr=net_pnl, ts_iso=now.isoformat(),
                    fees_inr=total_cost, gross_pnl_inr=net_pnl + total_cost,
                    symbol=slot.symbol, entry_price=slot.buy_fill_price,
                    exit_price=float(sell_price), exit_reason="t1_settle", qty=qty,
                    idealized_entry=slot.idealized_entry_price,
                    idealized_exit=(float(idealized_exit) if idealized_exit is not None else None),
                )
            else:
                tw.record_trade(net_pnl_inr=0.0, ts_iso=now.isoformat())

    # Phase 2: release T1 slots whose T+2 cash settle date has arrived
    for slot in list(pool.active()):
        if slot.status != "t1_settling":
            continue
        if slot.expected_exit_date is None:
            continue
        expected_exit = date.fromisoformat(slot.expected_exit_date)
        t2_settle_day = _next_trading_day(expected_exit)
        if today >= t2_settle_day:
            pool.release(slot.slot_id, today)
            summary["released_count"] += 1

    pool.persist()
    logger.info(
        "run_verify_exit: complete | settled=%d released=%d orphan_t0=%d",
        summary["settled_count"], summary["released_count"], summary["orphan_t0_count"],
    )

    # Daily baseline + candidate pre-filter for TODAY's 15:25 entry cron.
    # Runs after the critical path (settle + failsafe SELLs are done), so
    # any failure here doesn't block AMO settlement. The 09:30 timing
    # gives ~5h45m of headroom before 15:25; ~4 min wall-clock here is
    # cheap pre-market work.
    #
    # The 15:25 entry cron reads the resulting candidates_latest.json
    # (~50 symbols on a typical day) instead of building the 1118-symbol
    # universe from scratch — drops cron wall-clock from minutes to seconds.
    if getattr(broker, "_data_sdk", None) is not None:
        try:
            data_sdk = broker._data_sdk
            if data_sdk is not None:
                from services.execution.close_dn_baseline_build import (
                    build_baseline_and_candidates,
                )
                close_dn_cfg = (
                    next(
                        (s.raw_config for s in paper_enabled_setups
                         if s.name == "close_dn_overnight_long"),
                        None,
                    )
                    or {}
                )
                cell_min = float(close_dn_cfg["cell_prior_day_return_pct_min"])
                rolling_days = int(close_dn_cfg["baseline_rolling_days"])
                stats = build_baseline_and_candidates(
                    data_sdk, today,
                    rolling_days=rolling_days,
                    cell_min_prior_ret_pct=cell_min,
                )
                summary["baseline_build"] = stats
                logger.info(
                    "run_verify_exit: baseline+candidates built | %s",
                    stats,
                )
            else:
                logger.warning(
                    "run_verify_exit: broker has no _data_sdk — skipping baseline build"
                )
        except Exception as e:
            logger.exception("run_verify_exit: baseline build failed: %s", e)

    return summary


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mini_df_daily_from_candidate(cand: Dict[str, Any], today: date) -> pd.DataFrame:
    """Build a 2-row daily DataFrame from a pre-computed candidate entry.

    The candidate file carries prior_close + prev_prior_close per symbol so
    run_entry doesn't need to re-fetch daily history at 15:25. The detector's
    _prior_day_return_pct reads ctx.df_daily; we hand it a minimal frame
    with the two values it actually consults.

    Index is the two prior session dates (using calendar-1 / calendar-2 as a
    proxy — the detector's `before = ddf[ddf.index.date < session_date]`
    just needs them strictly before today, exact dates don't matter to the
    return calculation).
    """
    prev_close = float(cand["prior_close"])
    prev_prev_close = float(cand["prev_prior_close"])
    idx = pd.DatetimeIndex([
        pd.Timestamp(today) - pd.Timedelta(days=2),
        pd.Timestamp(today) - pd.Timedelta(days=1),
    ])
    return pd.DataFrame(
        {"close": [prev_prev_close, prev_close]},
        index=idx,
    )


def _gather_daily_dict(broker, setup_cfg: dict) -> Dict[str, pd.DataFrame]:
    """Build a daily_dict for the universe builder.

    In paper/live mode the broker exposes get_daily() (per-symbol cache) plus
    list_symbols(). The overnight universe builder consults nse_all.json
    (cap_segment + MIS) and daily history (volume + coverage). We pre-fetch
    daily history for ALL listed NSE EQ symbols here.
    """
    if not hasattr(broker, "list_symbols"):
        logger.warning("_gather_daily_dict: broker has no list_symbols(); returning empty dict")
        return {}
    symbols = broker.list_symbols(exchange="NSE", instrument_type="EQ")
    days = int(setup_cfg.get("min_trading_days_required", 30)) + 5
    daily_dict: Dict[str, pd.DataFrame] = {}
    n_empty = 0
    err_counts: Dict[str, int] = {}
    first_err_per_type: Dict[str, str] = {}
    for sym in symbols:
        nse_sym = sym if str(sym).startswith("NSE:") else f"NSE:{sym}"
        try:
            ddf = broker.get_daily(nse_sym, days=days)
            if ddf is not None and not ddf.empty:
                daily_dict[nse_sym] = ddf
            else:
                n_empty += 1
        except Exception as e:
            # Tally by exception type so a systemic failure (e.g. all symbols
            # raise the same AttributeError because session_date is None)
            # surfaces in the log instead of being silently swallowed.
            etype = type(e).__name__
            err_counts[etype] = err_counts.get(etype, 0) + 1
            first_err_per_type.setdefault(etype, f"{nse_sym}: {e}")

    total = len(symbols)
    n_ok = len(daily_dict)
    if err_counts:
        logger.warning(
            "_gather_daily_dict: %d/%d symbols raised | error counts: %s | first samples: %s",
            sum(err_counts.values()), total, err_counts, first_err_per_type,
        )
    if n_ok == 0:
        logger.error(
            "_gather_daily_dict: got 0 daily DataFrames out of %d symbols "
            "(empty=%d, errors=%d). Universe builder will return empty and "
            "the entry handler will report fired=0 even though no signal was "
            "actually evaluated. Check broker.set_session_date() / data archive.",
            total, n_empty, sum(err_counts.values()),
        )
    else:
        logger.info(
            "_gather_daily_dict: %d/%d symbols loaded (empty=%d, errors=%d)",
            n_ok, total, n_empty, sum(err_counts.values()),
        )
    return daily_dict


def _build_market_context(broker, symbol: str, now: pd.Timestamp, today: date,
                          setup_cfg: dict,
                          intraday_5m: Optional[pd.DataFrame] = None,
                          df_daily: Optional[pd.DataFrame] = None):
    """Build a MarketContext for a single symbol at the current bar timestamp.

    `intraday_5m` (when provided by the caller) is today's 5m bars
    pre-fetched in a single async batch — replaces the per-symbol
    broker.get_intraday_5m API call (which dominated cron wall-clock at
    ~100ms × 1118 symbols ≈ 110s sequential).

    `df_daily` (when provided) is the symbol's daily OHLCV frame —
    detectors that compute prior-day return (close_dn_overnight_long's
    `_prior_day_return_pct`) read ctx.df_daily first; without it they
    fall through to a df_5m-prior-session derivation that returns None
    in live/paper mode because df_5m is today-only.
    """
    from structures.data_models import MarketContext

    # 5m bars — priority order:
    # 0. intraday_5m kwarg — caller pre-fetched in async batch (paper/live).
    # 1. broker.get_intraday_5m(symbol) — live API per-symbol (paper mode via
    #    data_sdk, or live Upstox/Kite). Used when batch fetch wasn't done.
    # 2. broker._load_enriched_5m() — backtest archive. ONLY in DRY_RUN.
    # 3. broker.fetch_candles(...) — Kite live fallback.
    is_dry_run = getattr(broker, "_dry_session_date", None) is not None
    df_5m: Optional[pd.DataFrame] = None
    if intraday_5m is not None and not intraday_5m.empty:
        df_5m = intraday_5m
    elif hasattr(broker, "get_intraday_5m"):
        try:
            df_5m = broker.get_intraday_5m(symbol)
        except Exception as e:
            logger.warning("_build_market_context: get_intraday_5m failed for %s: %s", symbol, e)
            df_5m = None
    if is_dry_run and (df_5m is None or df_5m.empty) and hasattr(broker, "_load_enriched_5m"):
        try:
            all_enriched = broker._load_enriched_5m()
            bare = symbol.replace("NSE:", "")
            df_5m = all_enriched.get(bare)
            if df_5m is None:
                df_5m = all_enriched.get(symbol)
        except Exception as e:
            logger.warning("_build_market_context: 5m load failed for %s: %s", symbol, e)
            df_5m = None
    if df_5m is None or df_5m.empty:
        df_5m = _get_5m_for_symbol_live(broker, symbol, today, setup_cfg)

    if df_5m is None or df_5m.empty:
        return None
    # Filter to bars on or before `now`
    df_5m = df_5m[df_5m.index <= now]
    if df_5m.empty:
        return None

    # cap_segment (optional; detector tolerates None)
    cap_seg = None
    try:
        from services.symbol_metadata import get_cap_segment
        cap_seg = get_cap_segment(symbol)
    except Exception:
        cap_seg = None

    return MarketContext(
        symbol=symbol,
        current_price=float(df_5m["close"].iloc[-1]),
        timestamp=df_5m.index[-1],
        df_5m=df_5m,
        df_daily=df_daily,
        session_date=pd.Timestamp(today),
        cap_segment=cap_seg,
    )


def _get_5m_for_symbol_live(broker, symbol: str, today: date, setup_cfg: dict) -> Optional[pd.DataFrame]:
    """Fetch today's 5m bars from a live broker (Kite API)."""
    if not hasattr(broker, "fetch_candles"):
        return None
    try:
        candles = broker.fetch_candles(symbol, interval="5minute", token=None)
        df = pd.DataFrame(candles, columns=["date", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        logger.warning("_get_5m_for_symbol_live: fetch_candles failed for %s: %s", symbol, e)
        return None


def _place_buy(broker, *, symbol: str, qty: int, product: str, paper_mode: bool,
               trade_id: str, limit_price: float) -> str:
    """Place a marketable-LIMIT BUY (variety=regular, product=MTF or CNC).

    Kite bars plain MARKET orders on close_dn's illiquid/trade-to-trade
    universe, so the BUY crosses the spread via a LIMIT at ref*(1+buffer).
    """
    order_id = broker.place_order(
        symbol=symbol, side="BUY", qty=qty,
        order_type="LIMIT", price=round_to_tick(float(limit_price)),
        product=product, variety="regular",
        trade_id=trade_id, check_margins=(product == "MIS"),
    )
    return str(order_id)


def _place_amo_sell(broker, *, symbol: str, qty: int, product: str, paper_mode: bool,
                    trade_id: str, limit_price: float) -> str:
    """Place an AMO LIMIT SELL for next-day pre-open execution.

    Floor is set at the catastrophe level so it fills at the pre-open auction
    for any non-catastrophic open (limit sells fill at the better auction
    price); a gap below the floor is left to the GTT / morning failsafe.
    """
    order_id = broker.place_order(
        symbol=symbol, side="SELL", qty=qty,
        order_type="LIMIT", price=round_to_tick(float(limit_price)),
        product=product, variety="amo",
        trade_id=trade_id, check_margins=False,
    )
    return str(order_id)


def _place_failsafe_sell(broker, *, symbol: str, qty: int, product: str,
                         limit_price: float) -> str:
    """Failsafe regular LIMIT SELL (marketable, below live LTP) when AMO did not execute."""
    order_id = broker.place_order(
        symbol=symbol, side="SELL", qty=qty,
        order_type="LIMIT", price=round_to_tick(float(limit_price)),
        product=product, variety="regular",
        check_margins=False,
    )
    return str(order_id)


def _today_5m(broker, symbol: str) -> Optional[pd.DataFrame]:
    """Return today's 5m bars from the broker's live API (paper-live mode).

    Falls back to the archive only when running under DRY_RUN (true backtest).
    Returns None on any error / missing data — callers handle the None case.

    NOTE on the is_dry_run check: paper-trading cron passes
    `--session-date $(date +%F)` to anchor "today" without wall-clock
    dependency, which sets MockBroker._dry_session_date. So we MUST NOT
    use that attribute as the dry-run proxy — env.DRY_RUN is the
    canonical signal for true backtest mode. The earlier implementation
    used the attribute and silently broke paper-mode settlement (LAL +
    PCJEWELLER stuck on 2026-06-02 verify-exit — paper fill unavailable
    despite the data being available via Upstox).
    """
    try:
        from config.env_setup import env
        is_dry_run = bool(getattr(env, "DRY_RUN", False))
    except Exception:
        # If env import fails, fall back to the (less accurate) attribute
        # check rather than blocking the live path.
        is_dry_run = getattr(broker, "_dry_session_date", None) is not None

    if hasattr(broker, "get_intraday_5m") and not is_dry_run:
        try:
            df = broker.get_intraday_5m(symbol)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            logger.warning("_today_5m: get_intraday_5m failed for %s: %s", symbol, e)
    if is_dry_run and hasattr(broker, "_load_enriched_5m"):
        try:
            bare = symbol.replace("NSE:", "")
            all_enriched = broker._load_enriched_5m()
            # NOTE: do NOT use `get(bare) or get(symbol)` — `bool(DataFrame)`
            # raises "truth value ambiguous" on a non-empty frame, silently
            # losing the archive fill (settlement then reports "fill unavailable").
            df = all_enriched.get(bare)
            if df is None:
                df = all_enriched.get(symbol)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            logger.warning("_today_5m: archive load failed for %s: %s", symbol, e)
    return None


def _paper_fill_price_entry(broker, symbol: str, today: date) -> Optional[float]:
    """In paper mode, the 15:25 bar's CLOSE is the MOC fill price (= 15:30 IST close)."""
    df = _today_5m(broker, symbol)
    if df is None or df.empty:
        return None
    try:
        target_ts = pd.Timestamp.combine(today, time(15, 25))
        if target_ts in df.index:
            return float(df.loc[target_ts, "close"])
        before = df[df.index <= target_ts]
        if before.empty:
            return None
        return float(before["close"].iloc[-1])
    except Exception as e:
        logger.warning("_paper_fill_price_entry failed for %s: %s", symbol, e)
        return None


def _paper_fill_price_exit(broker, symbol: str, exit_date: date) -> Optional[float]:
    """In paper mode, AMO SELL fills at next-day 09:15 bar's OPEN."""
    df = _today_5m(broker, symbol)
    if df is None or df.empty:
        return None
    try:
        target_ts = pd.Timestamp.combine(exit_date, time(9, 15))
        if target_ts in df.index:
            return float(df.loc[target_ts, "open"])
        after = df[df.index >= target_ts]
        if after.empty:
            return None
        return float(after["open"].iloc[0])
    except Exception as e:
        logger.warning("_paper_fill_price_exit failed for %s: %s", symbol, e)
        return None


def _live_poll_fill(broker, order_id: str, timeout_sec: int = 60) -> Optional[float]:
    """Poll broker for order status until filled or timeout. Returns avg fill price."""
    if not hasattr(broker, "get_order_status"):
        logger.warning("_live_poll_fill: broker has no get_order_status; returning None")
        return None
    deadline = _time_mod.time() + timeout_sec
    while _time_mod.time() < deadline:
        try:
            status = broker.get_order_status(order_id)
            if status and status.get("status") == "COMPLETE":
                return float(status.get("average_price", 0.0))
        except Exception as e:
            logger.warning("_live_poll_fill: status check failed: %s", e)
        _time_mod.sleep(2)
    return None


def _live_check_amo_fill(broker, amo_order_id: str) -> Optional[float]:
    """Check whether an AMO order has filled. Returns avg fill price or None."""
    if not hasattr(broker, "get_order_status"):
        return None
    try:
        status = broker.get_order_status(amo_order_id)
        if status and status.get("status") == "COMPLETE":
            return float(status.get("average_price", 0.0))
    except Exception as e:
        logger.warning("_live_check_amo_fill failed: %s", e)
    return None
