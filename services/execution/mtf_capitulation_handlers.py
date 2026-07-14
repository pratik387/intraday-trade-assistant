"""Cron-triggered executor for the multi-day CNC/MTF capitulation setup.

`mtf_capitulation_revert_long` is a daily cross-sectional mean-reversion with a
K-day delivery hold (NOT intraday-MIS). Its execution lifecycle is fundamentally
different from `close_dn_overnight_long` (which is a 1-night MOC-buy / AMO-sell-
next-open), so it lives in its own module and reuses the genuinely-shared pieces
(CrossSectionalRanker, DailyPanelProvider, MtfUniverse, PositionPersistence,
the MTF/CNC fee functions) rather than overloading close_dn's slot pool — whose
field names (`amo_sell_order_id`, `expected_exit_date`) encode the 1-night
semantics and would mean a second, confusing code path here.

Lifecycle (all IST-naive, both entry points idempotent):

  run_eod(T close, ~15:25):
    A. EXITS: square positions whose exit_on_date == today via MOC SELL at the
       T-close; book net PnL (fees + MTF interest over the actual hold); feed the
       decay tripwire; drop the position.
    B. ENTRIES: rank the MTF universe on the day-T panel, then place AMO BUY for
       the next session's open for each NEW basket name (under concurrency /
       new-per-day caps); persist each as a pending multi-day hold.

  run_verify_entries(E open, ~09:30):
    Confirm the AMO BUYs filled at the entry day's open; record the fill price;
    clear the pending flag. Unfilled live AMO -> failsafe market BUY (or drop).

Signal known at day-t close => entry is T+1 open; held `hold_days` trading days;
exit at the K-day close. Every parameter is read from the setup config block
(CLAUDE.md rule 1 — no hardcoded trading defaults).

Spec: specs/2026-06-14-brief-mtf_capitulation_revert_long.md
"""
from __future__ import annotations

from datetime import date, time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from config.logging_config import get_agent_logger
    logger = get_agent_logger()
except Exception:  # pragma: no cover - logging fallback
    import logging
    logger = logging.getLogger(__name__)

from diagnostics.diag_event_log import diag_event_log
from services.cross_sectional_ranker import CrossSectionalRanker
from services.daily_panel_provider import make_provider
from services.execution.overnight_handlers import _next_trading_day
from services.mtf_universe import MtfUniverse
from services.state.position_persistence import PositionPersistence

# Infra tuning for the daily-history prewarm (NOT trading thresholds). The Upstox
# daily endpoint rate-limits at the default 50 RPS; these issue gently to avoid
# 429 backoff. Mirrors the per-family rate literals in overnight_handlers.
_DAILY_PREWARM_RPS = 15.0
_DAILY_PREWARM_WORKERS = 6


def _add_trading_days(d: date, n: int) -> date:
    """Add `n` trading days (Mon-Fri, holidays not modelled — see _next_trading_day)."""
    cur = d
    for _ in range(int(n)):
        cur = _next_trading_day(cur)
    return cur


def _eligible_multiday_setups(config: dict, *, paper_mode: bool):
    """All eligible multi-day (horizon='multi_day') CNC/MTF setups.

    Generic by construction: each setup plugs its own selection_mode into the
    shared ranker + entry/exit/position machinery (e.g. mtf_capitulation_revert_long
    = trailing-loser, low52_capitulation_revert_long = near-period-low). Paper mode
    gates on paper_enabled; live on enabled (mirrors close_dn).
    Returns list of (name, raw_cfg).
    """
    out = []
    for name, raw in (config.get("setups") or {}).items():
        if str(raw.get("horizon")) != "multi_day":
            continue
        ok = bool(raw.get("paper_enabled", False)) if paper_mode else bool(raw.get("enabled", False))
        if ok:
            out.append((name, raw))
    return out


def _position_state_dir(raw: dict) -> Path:
    """Dedicated dir for this setup's PositionPersistence snapshot.

    Derived deterministically from the configured slot state_file so it stays
    under the same `state/` tree without a magic literal and without colliding
    with other setups' snapshots (PositionPersistence's filename is fixed).
    """
    state_file = Path(str(raw["capital_allocation"]["state_file"]))
    return state_file.parent / f"{state_file.stem}_positions"


# ---------------------------------------------------------------------------
# Public cron entry points
# ---------------------------------------------------------------------------

def run_eod(
    config: dict,
    broker,
    *,
    now_ist: Optional[pd.Timestamp] = None,
    paper_mode: bool = True,
    ca_ex_dates: Optional[Dict[str, List[date]]] = None,
    repo_root: Optional[Path] = None,
    phase: str = "both",
) -> dict:
    """EOD exits + T+1 entry basket.

    `phase` splits the two legs so LIVE can reproduce the backtest, which they
    can't if combined: the EXIT must fire BEFORE the 15:30 close (sell at the
    close), but the ENTRY signal needs day-T's COMPLETE bar (close + full volume),
    only available AFTER the close. So live/paper run two crons:
      phase='exits'   — pre-close (~15:28): square off positions due today.
      phase='entries' — post-close (~15:35): rank on the full day-T bar, AMO buy.
    phase='both' (default) keeps the combined path for the dry-run replay harness
    (the feather already has the complete day-T bar, so timing collapses).

    `ca_ex_dates` ({bare_symbol: [ex_date,...]}) excludes names with a corporate
    action inside the hold window; if None it is loaded from the configured
    ca_events_path. Returns a summary dict for logging/testing.
    """
    if phase not in ("both", "exits", "entries"):
        raise ValueError(f"phase must be both|exits|entries, got {phase!r}")
    from utils.time_util import _now_naive_ist

    now = pd.Timestamp(now_ist) if now_ist is not None else _now_naive_ist()
    today = now.date()
    summary: dict = {
        "now_ist": str(now), "today": str(today), "paper_mode": paper_mode,
        "exited_count": 0, "entered_count": 0, "skipped_count": 0,
        "rejected_count": 0, "events": [],
    }

    setups = _eligible_multiday_setups(config, paper_mode=paper_mode)
    if not setups:
        logger.info("mtf_capitulation.run_eod: no eligible multi-day setups (paper=%s); exit", paper_mode)
        return summary

    # Concurrent daily-cache prewarm at MAX depth, once, before any ranking
    # (perf + deep-lookback correctness — see _prewarm_daily_universe).
    if phase in ("both", "entries"):
        _prewarm_daily_universe(setups, broker)

    summary["by_setup"] = {}
    persistences = {name: PositionPersistence(_position_state_dir(raw)) for name, raw in setups}
    setups_by_name = {n: r for n, r in setups}
    _warn_mtf_delisted(setups, persistences, summary)
    # ---- Phase A: exits due today (per-setup, pre-close) ----
    if phase in ("both", "exits"):
        for name, raw in setups:
            _run_exits(name, raw, broker, persistences[name], today, now, paper_mode, summary,
                       setups_by_name=setups_by_name)
    # ---- Phase B: rank + AMO BUY across the whole family (post-close) ----
    if phase in ("both", "entries"):
        _run_entries_composite(setups, broker, persistences, today, now, paper_mode, summary,
                               ca_ex_dates=ca_ex_dates, repo_root=repo_root, config=config)

    logger.info(
        "mtf_capitulation.run_eod: complete | setups=%d exited=%d entered=%d skipped=%d rejected=%d",
        len(setups), summary["exited_count"], summary["entered_count"],
        summary["skipped_count"], summary["rejected_count"],
    )
    return summary


def run_verify_entries(
    config: dict,
    broker,
    *,
    now_ist: Optional[pd.Timestamp] = None,
    paper_mode: bool = True,
) -> dict:
    """At entry-day open (~09:30): confirm AMO BUY fills, record entry price.

    Idempotent — a position whose pending flag is already cleared is skipped.
    """
    from utils.time_util import _now_naive_ist

    now = pd.Timestamp(now_ist) if now_ist is not None else _now_naive_ist()
    today = now.date()
    summary: dict = {
        "now_ist": str(now), "today": str(today), "paper_mode": paper_mode,
        "filled_count": 0, "unfilled_count": 0, "events": [],
    }

    setups = _eligible_multiday_setups(config, paper_mode=paper_mode)
    if not setups:
        logger.info("mtf_capitulation.run_verify_entries: no eligible multi-day setups; exit")
        return summary

    for name, raw in setups:
        persistence = PositionPersistence(_position_state_dir(raw))
        for symbol, pos in list(persistence.load_snapshot().items()):
            if not pos.state.get("pending_entry_fill"):
                continue
            entry_date = _parse_iso_date(pos.entry_date)
            if entry_date is None or entry_date > today:
                continue  # not its entry day yet

            qty = int(pos.state.get("qty", 0))
            if paper_mode:
                fill = _paper_open_price(broker, symbol, entry_date)
            else:
                fill = _live_poll_fill(broker, pos.order_id, timeout_sec=60)
                if fill is None:
                    fill = _failsafe_market_buy(broker, symbol, qty, pos.product or "CNC")

            if fill is None:
                logger.warning(
                    "mtf_capitulation.run_verify_entries[%s]: no fill for %s on %s; dropping",
                    name, symbol, entry_date,
                )
                persistence.remove_position(symbol)
                summary["unfilled_count"] += 1
                continue

            # avg_price carries the realized entry price for exit PnL — persisted
            # atomically (under the store's lock) alongside the state update.
            state_updates = {"pending_entry_fill": False, "entry_fill_price": float(fill)}
            # Price the optional vol-scaled target off the REAL fill (the study
            # geometry: target = entry * (1 + k*sigma20)). Only when the entry
            # persisted target_k/sigma (config-gated + sigma available).
            tk = pos.state.get("target_k")
            tsig = pos.state.get("target_sigma20_pct")
            if tk is not None and tsig is not None:
                state_updates["target_px"] = round(float(fill) * (1.0 + float(tk) * float(tsig)), 2)
            persistence.update_position(
                symbol,
                avg_price=float(fill),
                state_updates=state_updates,
            )
            # Record the ENTRY fill on the single-writer events.jsonl surface so the
            # paper trade flows into analytics.jsonl + the session report (the cron
            # families previously had no such surface). Never let logging break the run.
            if not pos.trade_id:
                # An empty trade_id would make the ENTRY and EXIT events mint
                # different time-based IDs and never pair in analytics.jsonl.
                logger.warning("mtf_capitulation[%s]: %s has empty trade_id; ENTRY/EXIT may not pair", name, symbol)
            try:
                diag_event_log.log_entry_fill(
                    symbol=symbol, plan={"trade_id": pos.trade_id, "setup": name},
                    side="BUY", qty=qty, price=float(fill), entry_ts=now,
                    order_meta={"product": pos.product or "CNC", "variety": "amo", "setup": name},
                )
            except Exception as _diag_err:  # pragma: no cover - logging must not break cron
                logger.warning("mtf_capitulation: diag_event_log.log_entry_fill failed for %s: %s", symbol, _diag_err)
            summary["filled_count"] += 1
            summary["events"].append({"setup": name, "symbol": symbol, "entry_fill": float(fill), "qty": qty})

    logger.info(
        "mtf_capitulation.run_verify_entries: complete | filled=%d unfilled=%d",
        summary["filled_count"], summary["unfilled_count"],
    )
    return summary


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------

def _run_exits(name, raw, broker, persistence, today, now, paper_mode, summary,
               setups_by_name=None) -> None:
    from tools.sub7_validation.build_per_setup_pnl import (
        calc_fee_cnc, calc_fee_mtf, MTF_INTEREST_RATE_PER_DAY,
    )

    for symbol, pos in list(persistence.load_snapshot().items()):
        exit_on = _parse_iso_date(pos.exit_on_date)
        target_touched = False
        if exit_on is None or today < exit_on:
            # Not due yet — but an enabled vol-scaled target that today's HIGH
            # has touched exits NOW (study 2026-07-10: sell the snapback climax
            # instead of the faded close; checked EVERY hold day incl. entry day).
            tpx = pos.state.get("target_px")
            entry_d = _parse_iso_date(pos.entry_date)
            if (tpx is None or pos.state.get("pending_entry_fill")
                    or entry_d is None or today < entry_d):
                continue
            day_open, day_high = _today_open_high(broker, symbol, today)
            if day_high is None or day_high < float(tpx):
                continue
            target_touched = True
        if pos.state.get("pending_entry_fill"):
            # Entry never confirmed (AMO BUY didn't fill / verify never ran).
            # Nothing to exit; drop so it doesn't linger.
            logger.warning(
                "mtf_capitulation: %s due to exit but entry unfilled; dropping", symbol,
            )
            persistence.remove_position(symbol)
            continue
        if today > exit_on:
            # Cron-missed exit (rare now that trading-day math is holiday-aware).
            # SELF-HEAL: settle at today's close rather than orphan the position
            # (the prior 'flag + leave' path leaked positions in the store —
            # backtest_findings #0). A 1-session-late delivery exit is a minor
            # mis-price; an orphaned leveraged position is not. Count for monitoring.
            logger.warning(
                "mtf_capitulation: %s late exit (exit_on=%s < today=%s); settling at today's close",
                symbol, exit_on, today,
            )
            summary["stale_exit_count"] = summary.get("stale_exit_count", 0) + 1
            # fall through to settle at today's close (no continue, no leak)

        qty = int(pos.state.get("qty", 0))
        entry_price = float(pos.avg_price or pos.state.get("entry_fill_price") or 0.0)
        if qty <= 0 or entry_price <= 0:
            logger.warning("mtf_capitulation: %s missing qty/entry; dropping", symbol)
            persistence.remove_position(symbol)
            continue

        if target_touched:
            # Vol-scaled target touched today: fill AT target (gap-open above
            # target fills at the open) — the study's touch geometry. Paper-only
            # family today; a live path would place a limit/GTT instead.
            day_open, _dh = _today_open_high(broker, symbol, today)
            tpx = float(pos.state.get("target_px"))
            sell_price = max(day_open, tpx) if day_open is not None else tpx
            exit_on = today  # actual exit date for hold-days/interest math
        elif paper_mode:
            sell_price = _paper_close_price(broker, symbol, today)
        else:
            sell_price = _place_moc_sell_and_fill(broker, symbol, qty, pos.product or "CNC")
        if sell_price is None:
            logger.warning("mtf_capitulation: exit price unavailable for %s; skip", symbol)
            continue

        buy_value = entry_price * qty
        sell_value = float(sell_price) * qty
        leverage = float(pos.state.get("leverage", 1.0))
        entry_date = _parse_iso_date(pos.entry_date) or exit_on
        # CALENDAR days, intentionally: MTF interest accrues every calendar day
        # the position is borrowed (weekends included) — same semantics as
        # close_dn (Fri entry -> Mon exit = 3 days of interest). Do NOT "fix"
        # this to trading days.
        hold_days = max(1, (exit_on - entry_date).days)
        if (pos.product or "").upper() == "MTF":
            margin_inr = buy_value / leverage if leverage > 0 else buy_value
            borrowed = max(0.0, buy_value - margin_inr)
            # Interest rate is config-driven (CLAUDE.md rule 1). calc_fee_mtf
            # bundles base+pledge+unpledge but also adds interest at its own
            # module constant; strip that and re-add the config-rated interest so
            # config remains the single source of truth for the MTF rate.
            interest_rate = float(raw["mtf"]["interest_pct_per_day"])
            fees_total = calc_fee_mtf(buy_value, sell_value, margin_inr, hold_days)
            const_interest = borrowed * MTF_INTEREST_RATE_PER_DAY * hold_days
            fees_only = max(0.0, fees_total - const_interest)  # base + pledge + unpledge
            interest = borrowed * interest_rate * hold_days
        else:
            fees_only = calc_fee_cnc(buy_value, sell_value)
            interest = 0.0

        gross = (float(sell_price) - entry_price) * qty
        net = gross - fees_only - interest

        persistence.remove_position(symbol)
        # Record the EXIT (with realized net PnL) on the single-writer events.jsonl
        # surface — feeds analytics.jsonl + the session report. Never break the cron.
        try:
            diag_event_log.log_exit(
                symbol=symbol, plan={"trade_id": pos.trade_id, "setup": name},
                reason=("target_touch" if target_touched else "kday_close_moc"),
                exit_price=float(sell_price), exit_qty=qty,
                ts=now, pnl=float(net),
                diagnostics={
                    "hold_days": hold_days, "entry_date": pos.entry_date,
                    "exit_date": exit_on.isoformat(), "entry_price": entry_price,
                    "product": pos.product or "CNC", "setup": name,
                },
            )
        except Exception as _diag_err:  # pragma: no cover - logging must not break cron
            logger.warning("mtf_capitulation: diag_event_log.log_exit failed for %s: %s", symbol, _diag_err)
        summary["exited_count"] += 1
        summary["events"].append({
            "setup": name, "symbol": symbol, "qty": qty, "entry": entry_price,
            "exit": float(sell_price), "net_pnl": net, "hold_days": hold_days,
            "entry_date": pos.entry_date, "exit_date": exit_on.isoformat(),
        })

        # Feed the realized trade to EVERY contributing setup's decay tripwire,
        # not just the owner's — the book holds the name once (owner store) but
        # each setup that flagged it must see the outcome for standalone-edge
        # measurement (spec section 5). Falls back to the owner when the position
        # predates contributor tagging.
        contributors = pos.state.get("contributors") or [name]
        lookup = setups_by_name or {name: raw}
        for cname in contributors:
            craw = lookup.get(cname)
            if craw is None:
                continue
            tw_cfg = craw.get("decay_tripwire")
            if tw_cfg is None:
                continue
            from services.risk.decay_tripwire import DecayTripwire
            DecayTripwire(
                setup_name=cname,
                state_path=Path(tw_cfg["state_file"]),
                window_trades=int(tw_cfg["window_trades"]),
                pf_floor=float(tw_cfg["pf_floor"]),
                sustained_weeks=int(tw_cfg["sustained_weeks"]),
            ).record_trade(
                net_pnl_inr=float(net), ts_iso=now.isoformat(),
                fees_inr=float(fees_only) + float(interest), gross_pnl_inr=float(gross),
                symbol=symbol, entry_price=float(entry_price),
                exit_price=float(sell_price),
                exit_reason=("target_touch" if target_touched else "kday_close_moc"),
                qty=int(qty),
                # Mirror rows (non-owner contributors) are tagged so pooled/
                # portfolio views can exclude them — one book position must
                # count once. The owner's row stays untagged.
                attributed=(cname != name),
                entry_date=pos.entry_date,
            )


def _rank_basket_for_setup(name, raw, broker, today, ca_ex_dates, repo_root):
    """Build one setup's ranked basket (cap_score-bearing) for `today`.

    Returns [] when the setup has no MTF universe, no panel, or an empty basket.
    The MTF prefetch + panel build are the same as the legacy per-setup path.
    """
    mtf_cfg = raw["mtf"]
    mtf = MtfUniverse(Path(str(mtf_cfg["approved_list_snapshot_path"])))
    exclude_etf = bool(mtf_cfg["exclude_etf"])
    eligible = {s for s in mtf.all_symbols() if mtf.is_eligible(s, exclude_etf=exclude_etf)}
    if not eligible:
        logger.warning("mtf_capitulation[%s]: empty MTF eligible set; no basket", name)
        return []
    if ca_ex_dates is None and bool(raw.get("exclude_ca_in_hold_window")):
        ca_ex_dates = _load_ca_ex_dates(raw, repo_root)

    if not _is_dry_run(broker) and hasattr(broker, "set_intraday_5m_prefetch"):
        sdk = getattr(broker, "_data_sdk", None)
        if sdk is not None and hasattr(sdk, "async_fetch_intraday_5m_batch"):
            existing = getattr(broker, "_intraday_5m_prefetch", {}) or {}
            need = [f"NSE:{s}" for s in eligible if f"NSE:{s}" not in existing and s not in existing]
            if need:
                import asyncio
                try:
                    fetched = asyncio.run(sdk.async_fetch_intraday_5m_batch(need, concurrency=30, rps=20.0))
                    merged = dict(existing); merged.update(fetched or {})
                    broker.set_intraday_5m_prefetch(merged)
                except Exception as e:
                    logger.exception("mtf_capitulation[%s]: 5m batch prewarm failed: %s", name, e)

    provider = make_provider(raw, dry_run=_is_dry_run(broker),
                             fetch_fn=getattr(broker, "fetch_daily_window", None),
                             mtf_symbols=eligible, repo_root=repo_root)
    panel = provider.get_panel(today)
    if panel is None or panel.empty:
        logger.warning("mtf_capitulation[%s]: empty daily panel for %s; no basket", name, today)
        return []
    basket = CrossSectionalRanker(raw).rank(panel, today, eligible, ca_ex_dates=ca_ex_dates)
    return basket or []


def _target_exit_state(raw: dict, sigma20_pct) -> dict:
    """State fields for the optional vol-scaled target exit. Empty dict when the
    setup has no enabled target_exit config or sigma is unavailable (early-history
    names) — position then behaves exactly as hold-to-close."""
    te = raw.get("target_exit")
    if not te or not bool(te.get("enabled")):
        return {}
    if str(te["mode"]) != "vol_scaled_k":
        raise ValueError(f"unsupported target_exit.mode {te.get('mode')!r}")
    if sigma20_pct is None or not (float(sigma20_pct) > 0):
        return {}
    return {"target_k": float(te["k"]), "target_sigma20_pct": float(sigma20_pct)}


def _today_open_high(broker, symbol: str, day) -> tuple:
    """(day open, day high-so-far) from today's 5m bars; (None, None) if absent."""
    df = _today_5m(broker, symbol)
    if df is None or df.empty:
        return None, None
    try:
        dd = df[df.index.normalize() == pd.Timestamp(day)]
        if dd.empty:
            return None, None
        return float(dd["open"].iloc[0]), float(dd["high"].max())
    except Exception as e:
        logger.warning("_today_open_high failed for %s: %s", symbol, e)
        return None, None


def _warn_mtf_delisted(setups, persistences, summary) -> None:
    """Surface held names that are NO LONGER on Zerodha's MTF approved list.

    2026-07-14: refreshing an 8-week-stale snapshot revealed 3 of 11 held paper
    positions had been dropped from the list mid-hold. Live, Zerodha can
    force-convert or square off MTF positions on delisted names — this check
    makes that visible in the daily cron log THE DAY the (daily-refreshed)
    snapshot drops a held name, instead of at a broker rejection. Never raises.
    """
    try:
        flagged = []
        for name, raw in setups:
            mtf = MtfUniverse(Path(str(raw["mtf"]["approved_list_snapshot_path"])))
            approved = mtf.all_symbols()
            for sym in persistences[name].load_snapshot().keys():
                if sym.replace("NSE:", "").upper() not in approved:
                    flagged.append(f"{name}:{sym}")
        if flagged:
            summary["mtf_delisted_held"] = flagged
            logger.warning(
                "mtf_capitulation: %d HELD position(s) no longer on the MTF "
                "approved list (broker may force-convert/square off if live): %s",
                len(flagged), ", ".join(flagged),
            )
    except Exception as e:  # pragma: no cover - sentinel must not break the cron
        logger.warning("mtf_capitulation: MTF-delisting check failed: %s", e)


def _held_snapshots(setups, persistences):
    """Single snapshot read per setup → (held bare-symbol set, total open count).

    `held` is the cross-day dedupe set (a name already open in ANY store is not
    re-entered); `total_held` is the book-slot count (sum of open positions) used
    for the family concurrency cap. Read once to avoid two snapshot passes.
    """
    held, total_held = set(), 0
    for name, _raw in setups:
        snap = persistences[name].load_snapshot()
        total_held += len(snap)
        for sym in snap.keys():
            held.add(str(sym).replace("NSE:", "").upper())
    return held, total_held


def _log_selection_diagnostics(baskets, chosen, today, log_path_str):
    """One jsonl row per (setup, symbol, day): cap_score + composite/owner/
    contributors/consensus + chosen flag. Feeds the section 6.1 IC analysis.
    Logs EVERY flagged name (not only chosen), so forward-return IC can be
    computed for picks that were capped out. Never breaks the cron.
    """
    import json as _json
    try:
        # consensus + composite views keyed by bare symbol
        consensus, by_sym = {}, {}
        for _setup, cands in baskets.items():
            for c in cands:
                b = str(c["symbol"]).replace("NSE:", "").upper()
                consensus[b] = consensus.get(b, 0) + 1
        for c in chosen:
            by_sym[c["bare"]] = c
        path = Path(log_path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for setup_name, cands in baskets.items():
                for c in cands:
                    b = str(c["symbol"]).replace("NSE:", "").upper()
                    ch = by_sym.get(b)
                    f.write(_json.dumps({
                        "session_date": today.isoformat(),
                        "setup": setup_name, "symbol": b,
                        "cap_score": float(c["cap_score"]), "tshock": float(c["tshock"]),
                        "trail_ret": float(c["trail_ret"]), "rank_pct": float(c["rank_pct"]),
                        "consensus_count": int(consensus[b]),
                        "chosen": ch is not None,
                        "composite": (float(ch["composite"]) if ch else None),
                        "owner": (ch["owner"] if ch else None),
                        "contributors": (ch["contributors"] if ch else None),
                    }) + "\n")
    except Exception as e:  # pragma: no cover - diagnostics must not break the cron
        logger.warning("mtf_capitulation: selection diagnostics log failed: %s", e)


def _run_entries_composite(setups, broker, persistences, today, now, paper_mode,
                           summary, *, ca_ex_dates, repo_root, config):
    from services.multiday_composite_selector import MultiDayCompositeSelector

    active = [(name, raw) for name, raw in setups if not _decay_paused(name, raw)]
    for name, _raw in setups:
        if (name, _raw) not in active:
            summary.setdefault("by_setup", {}).setdefault(name, {})["decay_paused"] = True
    if not active:
        logger.info("mtf_capitulation: all multi-day setups decay-paused; no entries")
        return

    baskets = {}
    for name, raw in active:
        baskets[name] = _rank_basket_for_setup(name, raw, broker, today, ca_ex_dates, repo_root)
    if not any(baskets.values()):
        logger.info("mtf_capitulation: all baskets empty for %s; no entries", today)
        return

    fam = config["multi_day_portfolio"]
    selector = MultiDayCompositeSelector(fam)
    weights = {name: float(raw["composite_weight"]) for name, raw in active}
    held, total_held = _held_snapshots(setups, persistences)
    limit = min(int(fam["max_new_per_day"]),
                max(0, int(fam["max_concurrent"]) - total_held))
    chosen = selector.select(baskets, held_symbols=held, weights=weights, limit=limit)
    _log_selection_diagnostics(baskets, chosen, today, fam["selection_log_path"])
    if not chosen:
        return

    active_by_name = dict(active)
    entry_date = _next_trading_day(today)
    for c in chosen:
        owner = c["owner"]
        raw = active_by_name[owner]
        exit_on_date = _add_trading_days(entry_date, int(raw["hold_days"]))
        bare = c["bare"]; symbol = c["symbol"]
        mtf = MtfUniverse(Path(str(raw["mtf"]["approved_list_snapshot_path"])))
        info = mtf.lookup(bare)
        if info is not None:
            product, leverage = "MTF", float(info.leverage)
        elif bool(raw["mtf"]["fallback_to_cnc_if_not_mtf"]):
            product, leverage = "CNC", 1.0
        else:
            summary["rejected_count"] += 1
            continue
        margin_per_slot = float(raw["capital_allocation"]["margin_per_slot_inr"])
        qty = int((margin_per_slot * leverage) // float(c["close"]))
        if qty <= 0:
            summary["rejected_count"] += 1
            continue
        trade_id = f"{owner}_{today.isoformat()}_{bare}"
        try:
            order_id = _place_amo_buy(broker, symbol, qty, product, trade_id)
        except Exception as e:
            logger.error("mtf_capitulation[%s]: AMO BUY failed for %s: %s", owner, symbol, e)
            summary["skipped_count"] += 1
            continue
        persistences[owner].save_position(
            symbol=symbol, side="BUY", qty=qty, avg_price=0.0, trade_id=trade_id,
            order_id=str(order_id), order_tag=trade_id,
            plan={"setup": owner, "trail_ret": c["trail_ret"], "tshock": c["tshock"],
                  "composite": c["composite"]},
            state={"pending_entry_fill": True, "qty": qty, "leverage": leverage,
                   "signal_close": float(c["close"]), "signal_date": today.isoformat(),
                   "contributors": c["contributors"],
                   "per_setup_cap_score": c["per_setup_cap_score"],
                   # Optional vol-scaled target exit (config-gated per setup;
                   # study 2026-07-10: k~2.0 sigma beats hold-to-close in all 4
                   # setups by selling the snapback climax instead of the faded
                   # close). target_px is priced at fill-confirm = fill*(1+k*sigma).
                   **_target_exit_state(raw, c.get("sigma20_pct"))},
            entry_date=entry_date.isoformat(), exit_on_date=exit_on_date.isoformat(),
            product=product)
        summary["entered_count"] += 1
        summary.setdefault("by_setup", {}).setdefault(owner, {"entered": 0})
        summary["by_setup"][owner]["entered"] = summary["by_setup"][owner].get("entered", 0) + 1
        summary["events"].append({
            "setup": owner, "symbol": symbol, "qty": qty, "product": product,
            "leverage": leverage, "entry_date": entry_date.isoformat(),
            "exit_on_date": exit_on_date.isoformat(), "amo_buy_order_id": str(order_id),
            "composite": c["composite"], "contributors": c["contributors"],
        })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prewarm_daily_universe(setups, broker) -> None:
    """Concurrently warm the daily cache for the UNION MTF universe at the DEEPEST
    window any setup needs, BEFORE per-setup ranking. Two reasons:
      - perf: turns ~1300 serial get_daily calls (minutes) into one concurrent
        batch (seconds);
      - correctness: the daily cache is first-depth-wins (it does not re-fetch to
        deepen), so if a shallow-window setup (A2 ~90d) cached a symbol first, a
        deep-window setup (low52's 252-day low) would silently rank on ~90 bars.
        Prewarming at the max depth first guarantees every setup sees full history.
    Paper/live only (DRY_RUN uses the feather provider). Never raises.
    """
    if _is_dry_run(broker):
        return
    sdk = getattr(broker, "_data_sdk", None)
    if sdk is None or not hasattr(sdk, "prewarm_daily_concurrent"):
        logger.warning("mtf_capitulation: no concurrent daily prewarm available; "
                       "panel fetch will be per-symbol (slow)")
        return
    from services.daily_panel_provider import _window_calendar_days
    eligible: set = set()
    max_days = 0
    for _name, raw in setups:
        mtf_cfg = raw["mtf"]
        mtf = MtfUniverse(Path(str(mtf_cfg["approved_list_snapshot_path"])))
        eligible |= {s for s in mtf.all_symbols()
                     if mtf.is_eligible(s, exclude_etf=bool(mtf_cfg["exclude_etf"]))}
        max_days = max(max_days, _window_calendar_days(raw) + 5)
    if not eligible or max_days <= 0:
        return
    nse_syms = [f"NSE:{s}" for s in sorted(eligible)]
    # The Upstox DAILY endpoint rate-limits hard at the default 50 RPS (429
    # storms + backoff, serial or concurrent). Issue gently so we stay under its
    # limit and avoid backoff entirely; modest concurrency pipelines the gentle
    # stream. Tunable via config if the safe rate is later pinned.
    try:
        if hasattr(sdk, "set_hist_rate_limit"):
            sdk.set_hist_rate_limit(_DAILY_PREWARM_RPS)
        n = sdk.prewarm_daily_concurrent(nse_syms, days=max_days,
                                         max_workers=_DAILY_PREWARM_WORKERS)
        logger.info("mtf_capitulation: daily prewarm %d/%d symbols @%dd depth "
                    "(%.0f RPS, %d workers)", n, len(nse_syms), max_days,
                    _DAILY_PREWARM_RPS, _DAILY_PREWARM_WORKERS)
    except Exception as e:  # pragma: no cover - prewarm must not break the cron
        logger.exception("mtf_capitulation: daily prewarm failed: %s", e)


def _decay_paused(name: str, raw: dict) -> bool:
    tw_cfg = raw.get("decay_tripwire")
    if tw_cfg is None:
        return False
    from services.risk.decay_tripwire import DecayTripwire
    return DecayTripwire(
        setup_name=name,
        state_path=Path(tw_cfg["state_file"]),
        window_trades=int(tw_cfg["window_trades"]),
        pf_floor=float(tw_cfg["pf_floor"]),
        sustained_weeks=int(tw_cfg["sustained_weeks"]),
    ).is_paused()


def _is_dry_run(broker) -> bool:
    try:
        from config.env_setup import env
        return bool(getattr(env, "DRY_RUN", False))
    except Exception:
        return getattr(broker, "_dry_session_date", None) is not None


def _parse_iso_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(str(s)[:10])
    except (ValueError, TypeError):
        return None


def _load_ca_ex_dates(raw: dict, repo_root: Optional[Path]) -> Dict[str, List[date]]:
    """Load {bare_symbol: [ex_date,...]} from the configured CA events parquet.

    Missing file => warn and return empty (CA exclusion silently inactive). The
    PRICE panel is already CA-adjusted (the killer risk #1); this exclusion is a
    belt-and-braces filter against holding through an ex-date, not the primary
    contamination guard.
    """
    path = raw.get("ca_events_path")
    if not path:
        return {}
    root = repo_root or Path(__file__).resolve().parents[2]
    p = (root / path) if not Path(path).is_absolute() else Path(path)
    if not p.exists():
        logger.warning("mtf_capitulation: CA events file missing (%s); CA exclusion inactive", p)
        return {}
    try:
        df = pd.read_parquet(p)
        sym_col = next((c for c in ("symbol", "tradingsymbol", "nse_symbol") if c in df.columns), None)
        date_col = next((c for c in ("ex_date", "exDate", "date") if c in df.columns), None)
        if sym_col is None or date_col is None:
            logger.warning("mtf_capitulation: CA parquet missing symbol/ex_date columns; skipping")
            return {}
        out: Dict[str, List[date]] = {}
        for _, r in df.iterrows():
            sym = str(r[sym_col]).replace("NSE:", "").upper()
            d = pd.to_datetime(r[date_col]).date()
            out.setdefault(sym, []).append(d)
        return out
    except Exception as e:
        logger.warning("mtf_capitulation: failed to load CA events: %s", e)
        return {}


def _today_5m(broker, symbol: str) -> Optional[pd.DataFrame]:
    """Today's 5m bars: live API in paper/live, archive under DRY_RUN."""
    is_dry = _is_dry_run(broker)
    if hasattr(broker, "get_intraday_5m") and not is_dry:
        try:
            df = broker.get_intraday_5m(symbol)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            logger.warning("mtf_capitulation._today_5m: get_intraday_5m failed for %s: %s", symbol, e)
    if hasattr(broker, "_load_enriched_5m"):
        try:
            bare = symbol.replace("NSE:", "")
            alld = broker._load_enriched_5m()  # noqa: SLF001
            df = alld.get(bare)
            if df is None:
                df = alld.get(symbol)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            logger.warning("mtf_capitulation._today_5m: archive load failed for %s: %s", symbol, e)
    return None


def _paper_open_price(broker, symbol: str, day: date) -> Optional[float]:
    """AMO BUY fills at the entry day's 09:15 open bar."""
    df = _today_5m(broker, symbol)
    if df is None or df.empty:
        return None
    try:
        target = pd.Timestamp.combine(day, time(9, 15))
        if target in df.index:
            return float(df.loc[target, "open"])
        after = df[df.index >= target]
        if after.empty:
            return None
        return float(after["open"].iloc[0])
    except Exception as e:
        logger.warning("mtf_capitulation._paper_open_price failed for %s: %s", symbol, e)
        return None


def _paper_close_price(broker, symbol: str, day: date) -> Optional[float]:
    """MOC SELL fills at the exit day's 15:25 close bar (= 15:30 session close)."""
    df = _today_5m(broker, symbol)
    if df is None or df.empty:
        return None
    try:
        target = pd.Timestamp.combine(day, time(15, 25))
        if target in df.index:
            return float(df.loc[target, "close"])
        before = df[df.index <= target]
        if before.empty:
            return None
        return float(before["close"].iloc[-1])
    except Exception as e:
        logger.warning("mtf_capitulation._paper_close_price failed for %s: %s", symbol, e)
        return None


def _place_amo_buy(broker, symbol: str, qty: int, product: str, trade_id: str) -> str:
    order_id = broker.place_order(
        symbol=symbol, side="BUY", qty=qty,
        order_type="MARKET", product=product, variety="amo",
        trade_id=trade_id, check_margins=False,
    )
    return str(order_id)


def _place_moc_sell_and_fill(broker, symbol: str, qty: int, product: str) -> Optional[float]:
    """Live: place a market SELL near the close and return the fill price."""
    try:
        broker.place_order(
            symbol=symbol, side="SELL", qty=qty,
            order_type="MARKET", product=product, variety="regular",
            check_margins=False,
        )
        return float(broker.get_ltp(symbol))
    except Exception as e:
        logger.error("mtf_capitulation: live MOC SELL failed for %s: %s", symbol, e)
        return None


def _failsafe_market_buy(broker, symbol: str, qty: int, product: str) -> Optional[float]:
    try:
        broker.place_order(
            symbol=symbol, side="BUY", qty=qty,
            order_type="MARKET", product=product, variety="regular",
            check_margins=False,
        )
        return float(broker.get_ltp(symbol))
    except Exception as e:
        logger.error("mtf_capitulation: failsafe BUY failed for %s: %s", symbol, e)
        return None


def _live_poll_fill(broker, order_id: Optional[str], timeout_sec: int = 60) -> Optional[float]:
    import time as _t
    if not order_id or not hasattr(broker, "get_order_status"):
        return None
    deadline = _t.time() + timeout_sec
    while _t.time() < deadline:
        try:
            status = broker.get_order_status(order_id)
            if status and status.get("status") == "COMPLETE":
                return float(status.get("average_price", 0.0))
        except Exception as e:
            logger.warning("mtf_capitulation._live_poll_fill: %s", e)
        _t.sleep(2)
    return None
