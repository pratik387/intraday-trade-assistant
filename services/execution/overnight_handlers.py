"""Cron-triggered handlers for overnight setups (close_dn_overnight_long).

Two short-lived entry points called by cron once per trading day:
  - run_entry(): 15:25 IST. Compute signal, place MOC BUY + AMO SELL.
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


def _next_trading_day(d: date) -> date:
    """Naive next-trading-day: Mon-Thu -> +1 day, Fri -> +3 days (skip weekend).

    Public holidays NOT handled — broker AMO accepts the order regardless;
    the actual fill happens on the next exchange session. Caller must
    update via NSE calendar lookup if precision needed.
    """
    weekday = d.weekday()  # Mon=0, Fri=4, Sat=5, Sun=6
    if weekday == 4:       # Friday -> Monday
        return d + timedelta(days=3)
    if weekday == 5:       # Saturday -> Monday
        return d + timedelta(days=2)
    if weekday == 6:       # Sunday -> Monday
        return d + timedelta(days=1)
    return d + timedelta(days=1)


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

        # Universe (currently hardcoded for close_dn_overnight_long; this is the only
        # registered overnight setup. Future overnight setups will need a dispatch on
        # spec.universe_builder_path here.)
        daily_dict = _gather_daily_dict(broker, spec.raw_config)
        try:
            universe = close_dn_overnight_long_universe(daily_dict, today, spec.raw_config)
        except Exception as e:
            logger.exception("run_entry: universe builder failed for %s: %s", spec.name, e)
            continue
        logger.info("run_entry: universe size for %s = %d", spec.name, len(universe))

        # Pre-fetch prior-N-day 5m bars for the universe so detectors that
        # need historical 5m context (close_dn_overnight_long's
        # _closing_baseline reads df_5m for prior 15:00-15:20 bars) work in
        # paper/live mode. broker.get_intraday_5m only returns today's bars
        # — without this batch fetch, the baseline would be empty and the
        # detector would reject every signal-eligible symbol with
        # "closing-25m baseline unavailable" (as observed today on the
        # 2026-05-29 cron: 466 rejects).
        prior_5m_by_symbol: Dict[str, pd.DataFrame] = {}
        try:
            data_sdk = getattr(broker, "_data_sdk", None)
            if data_sdk is not None and hasattr(data_sdk, "async_fetch_historical_5m_batch"):
                import asyncio as _asyncio
                from datetime import timedelta as _td
                rolling_days = int(spec.raw_config.get("baseline_rolling_days", 20))
                # 30 calendar days ~= 21 trading days; sufficient for rolling_days=20
                # rolling. async_fetch_historical_5m_batch chunks internally to stay
                # under Upstox's 1-month per-request limit.
                from_d = (today - _td(days=30)).isoformat()
                to_d = (today - _td(days=1)).isoformat()
                logger.info(
                    "run_entry: pre-fetching prior 5m bars for %d symbols (%s -> %s)",
                    len(universe), from_d, to_d,
                )
                prior_5m_by_symbol = _asyncio.run(
                    data_sdk.async_fetch_historical_5m_batch(
                        list(universe), from_d, to_d,
                        concurrency=30, rps=20.0,
                    )
                )
                logger.info(
                    "run_entry: prior 5m fetched for %d/%d universe symbols",
                    len(prior_5m_by_symbol), len(universe),
                )
            else:
                logger.warning(
                    "run_entry: data_sdk has no async_fetch_historical_5m_batch — "
                    "detectors needing prior 5m history will reject all symbols"
                )
        except Exception as e:
            logger.exception("run_entry: prior 5m batch fetch failed: %s", e)
            prior_5m_by_symbol = {}

        # Aggregate per-symbol rejection_reason buckets so the cron log shows
        # WHY all N symbols rejected instead of just N. Same diagnostic that
        # the screener (services/dispatch/worker.py:248-254) carries through
        # screening.jsonl; the overnight cron had no equivalent surface.
        from collections import Counter as _Counter
        reject_reasons: _Counter = _Counter()

        # Iterate symbols
        for symbol in universe:
            ctx = _build_market_context(
                broker, symbol, now, today, spec.raw_config,
                prior_5m=prior_5m_by_symbol.get(symbol),
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

            # Place MOC BUY
            try:
                buy_order_id = _place_buy(
                    broker, symbol=symbol, qty=plan.qty,
                    product=evt.context["product"],
                    paper_mode=paper_mode,
                    trade_id=f"OVERNIGHT_{today.isoformat()}_{slot.slot_id}",
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
                fill_price = _live_poll_fill(broker, buy_order_id, timeout_sec=60)
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
            )

            # Place AMO SELL for next trading day
            next_day = _next_trading_day(today)
            try:
                amo_order_id = _place_amo_sell(
                    broker, symbol=symbol, qty=plan.qty,
                    product=evt.context["product"],
                    paper_mode=paper_mode,
                    trade_id=f"OVERNIGHT_AMO_{today.isoformat()}_{slot.slot_id}",
                )
            except Exception as e:
                logger.error(
                    "run_entry: AMO SELL failed for %s: %s; slot left at t0_open without amo_order_id",
                    symbol, e,
                )
                # Persist what we have so far so the orphan is recoverable on
                # next cron fire (verify-exit reads the state).
                pool.persist()
                summary["skipped_count"] += 1
                continue
            pool.attach_amo_sell(slot.slot_id, str(amo_order_id), next_day)

            summary["fired_count"] += 1
            summary["events"].append({
                "symbol": symbol, "qty": plan.qty,
                "product": evt.context["product"],
                "buy_fill_price": float(fill_price),
                "amo_sell_order_id": str(amo_order_id),
                "expected_exit_date": next_day.isoformat(),
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
                # Failsafe: place a regular market SELL at current LTP
                qty_for_failsafe = int(round(slot.notional_inr / slot.buy_fill_price))
                try:
                    _place_failsafe_sell(
                        broker,
                        symbol=slot.symbol, qty=qty_for_failsafe,
                        product=slot.product or "CNC",
                    )
                    sell_price = float(broker.get_ltp(slot.symbol))
                except Exception as e:
                    logger.error(
                        "run_verify_exit: failsafe SELL failed for slot %d: %s",
                        slot.slot_id, e,
                    )
                    continue

        # Compute fees + interest
        qty = int(round(slot.notional_inr / slot.buy_fill_price))
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
            net_pnl = settled.realized_pnl_inr if settled.realized_pnl_inr is not None else 0.0
            tw.record_trade(net_pnl_inr=float(net_pnl), ts_iso=now.isoformat())

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
    return summary


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
                          prior_5m: Optional[pd.DataFrame] = None):
    """Build a MarketContext for a single symbol at the current bar timestamp.

    `prior_5m` (when provided) is pre-fetched historical 5m bars covering
    sessions strictly before `today` — used by detectors like
    close_dn_overnight_long whose baseline reads multi-day history out of
    ctx.df_5m. In live/paper mode the broker's intraday endpoint only
    returns today's bars; without prior_5m the baseline is empty and
    detectors silently no-fire.
    """
    from structures.data_models import MarketContext

    # 5m bars — priority order:
    # 1. broker.get_intraday_5m(symbol) — live API (paper mode via data_sdk,
    #    or live Upstox/Kite). This is what we want for today's session.
    # 2. broker._load_enriched_5m() — backtest archive. ONLY in DRY_RUN.
    #    In paper/live mode there is no archive on disk, so falling through
    #    here just produces "Loaded 0 symbols from individual files" noise.
    # 3. broker.fetch_candles(...) — Kite live fallback.
    is_dry_run = getattr(broker, "_dry_session_date", None) is not None
    df_5m: Optional[pd.DataFrame] = None
    if hasattr(broker, "get_intraday_5m"):
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

    # Prepend prior-day 5m history (paper/live path). Backtest archive
    # already carries multi-day df_5m natively; skip the concat in dry-run
    # to avoid duplicate rows.
    if not is_dry_run and prior_5m is not None and not prior_5m.empty:
        try:
            today_ts = pd.Timestamp(today)
            prior_only = prior_5m[prior_5m.index < today_ts]
            if not prior_only.empty:
                df_5m = pd.concat([prior_only, df_5m]).sort_index()
                df_5m = df_5m[~df_5m.index.duplicated(keep="last")]
        except Exception as e:
            logger.warning(
                "_build_market_context: prior_5m concat failed for %s: %s",
                symbol, e,
            )

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
               trade_id: str) -> str:
    """Place a MOC BUY order (variety=regular, product=MTF or CNC)."""
    order_id = broker.place_order(
        symbol=symbol, side="BUY", qty=qty,
        order_type="MARKET", product=product, variety="regular",
        trade_id=trade_id, check_margins=(product == "MIS"),
    )
    return str(order_id)


def _place_amo_sell(broker, *, symbol: str, qty: int, product: str, paper_mode: bool,
                    trade_id: str) -> str:
    """Place an AMO SELL for next-day pre-open execution."""
    order_id = broker.place_order(
        symbol=symbol, side="SELL", qty=qty,
        order_type="MARKET", product=product, variety="amo",
        trade_id=trade_id, check_margins=False,
    )
    return str(order_id)


def _place_failsafe_sell(broker, *, symbol: str, qty: int, product: str) -> str:
    """Failsafe regular market SELL when AMO did not execute."""
    order_id = broker.place_order(
        symbol=symbol, side="SELL", qty=qty,
        order_type="MARKET", product=product, variety="regular",
        check_margins=False,
    )
    return str(order_id)


def _today_5m(broker, symbol: str) -> Optional[pd.DataFrame]:
    """Return today's 5m bars from the broker's live API (paper-live mode).

    Falls back to the archive only when running under DRY_RUN. Returns None
    on any error / missing data — callers handle the None case.
    """
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
            df = all_enriched.get(bare) or all_enriched.get(symbol)
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
