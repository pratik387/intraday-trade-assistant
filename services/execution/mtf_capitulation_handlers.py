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

from services.cross_sectional_ranker import CrossSectionalRanker
from services.daily_panel_provider import make_provider
from services.execution.overnight_handlers import _next_trading_day
from services.mtf_universe import MtfUniverse
from services.state.position_persistence import PositionPersistence

_SETUP_NAME = "mtf_capitulation_revert_long"


def _add_trading_days(d: date, n: int) -> date:
    """Add `n` trading days (Mon-Fri, holidays not modelled — see _next_trading_day)."""
    cur = d
    for _ in range(int(n)):
        cur = _next_trading_day(cur)
    return cur


def _setup_cfg(config: dict) -> Optional[dict]:
    """Return the setup's raw config block, or None if absent."""
    return (config.get("setups") or {}).get(_SETUP_NAME)


def _is_eligible(raw: dict, *, paper_mode: bool) -> bool:
    """Paper mode: paper_enabled. Live mode: enabled. Mirrors close_dn gating."""
    if paper_mode:
        return bool(raw.get("paper_enabled", False))
    return bool(raw.get("enabled", False))


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
) -> dict:
    """At T close (~15:25): exit positions due today, then enter T+1 basket.

    `ca_ex_dates` ({bare_symbol: [ex_date,...]}) excludes names with a corporate
    action inside the hold window; if None it is loaded from the configured
    ca_events_path. Returns a summary dict for logging/testing.
    """
    from utils.time_util import _now_naive_ist

    now = pd.Timestamp(now_ist) if now_ist is not None else _now_naive_ist()
    today = now.date()
    summary: dict = {
        "now_ist": str(now), "today": str(today), "paper_mode": paper_mode,
        "exited_count": 0, "entered_count": 0, "skipped_count": 0,
        "rejected_count": 0, "events": [],
    }

    raw = _setup_cfg(config)
    if raw is None or not _is_eligible(raw, paper_mode=paper_mode):
        logger.info("mtf_capitulation.run_eod: setup not eligible (paper=%s); exit", paper_mode)
        return summary

    persistence = PositionPersistence(_position_state_dir(raw))

    # ---- Phase A: exits due today --------------------------------------
    _run_exits(raw, broker, persistence, today, now, paper_mode, summary)

    # ---- Phase B: rank + place AMO BUYs for next session ----------------
    if _decay_paused(raw):
        logger.warning("mtf_capitulation.run_eod: decay tripwire PAUSED; skipping entries")
        summary["decay_paused"] = True
        persistence.load_snapshot()  # no-op; state already persisted by exits
        return summary

    _run_entries(
        raw, broker, persistence, today, now, paper_mode, summary,
        ca_ex_dates=ca_ex_dates, repo_root=repo_root,
    )

    logger.info(
        "mtf_capitulation.run_eod: complete | exited=%d entered=%d skipped=%d rejected=%d",
        summary["exited_count"], summary["entered_count"],
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

    raw = _setup_cfg(config)
    if raw is None or not _is_eligible(raw, paper_mode=paper_mode):
        logger.info("mtf_capitulation.run_verify_entries: setup not eligible; exit")
        return summary

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
                "mtf_capitulation.run_verify_entries: no fill for %s on %s; dropping",
                symbol, entry_date,
            )
            persistence.remove_position(symbol)
            summary["unfilled_count"] += 1
            continue

        # avg_price carries the realized entry price for exit PnL — persisted
        # atomically (under the store's lock) alongside the state update.
        persistence.update_position(
            symbol,
            avg_price=float(fill),
            state_updates={"pending_entry_fill": False, "entry_fill_price": float(fill)},
        )
        summary["filled_count"] += 1
        summary["events"].append({"symbol": symbol, "entry_fill": float(fill), "qty": qty})

    logger.info(
        "mtf_capitulation.run_verify_entries: complete | filled=%d unfilled=%d",
        summary["filled_count"], summary["unfilled_count"],
    )
    return summary


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------

def _run_exits(raw, broker, persistence, today, now, paper_mode, summary) -> None:
    from tools.sub7_validation.build_per_setup_pnl import (
        calc_fee_cnc, calc_fee_mtf, MTF_INTEREST_RATE_PER_DAY,
    )

    for symbol, pos in list(persistence.load_snapshot().items()):
        exit_on = _parse_iso_date(pos.exit_on_date)
        if exit_on is None or today < exit_on:
            continue
        if pos.state.get("pending_entry_fill"):
            # Entry never confirmed (AMO BUY didn't fill / verify never ran).
            # Nothing to exit; drop so it doesn't linger.
            logger.warning(
                "mtf_capitulation: %s due to exit but entry unfilled; dropping", symbol,
            )
            persistence.remove_position(symbol)
            continue
        if today > exit_on:
            # Missed the exact exit-day close (cron skip/holiday). Selling at a
            # later close mis-prices the trade; flag for manual settlement once
            # (don't re-warn every subsequent cron fire).
            if not pos.state.get("manual_settle_required"):
                logger.warning(
                    "mtf_capitulation: %s STALE exit (exit_on=%s < today=%s); "
                    "flagged for manual settle", symbol, exit_on, today,
                )
                persistence.update_position(
                    symbol, state_updates={"manual_settle_required": True},
                )
            summary["stale_exit_count"] = summary.get("stale_exit_count", 0) + 1
            continue

        qty = int(pos.state.get("qty", 0))
        entry_price = float(pos.avg_price or pos.state.get("entry_fill_price") or 0.0)
        if qty <= 0 or entry_price <= 0:
            logger.warning("mtf_capitulation: %s missing qty/entry; dropping", symbol)
            persistence.remove_position(symbol)
            continue

        if paper_mode:
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
        summary["exited_count"] += 1
        summary["events"].append({
            "symbol": symbol, "qty": qty, "entry": entry_price,
            "exit": float(sell_price), "net_pnl": net, "hold_days": hold_days,
        })

        tw_cfg = raw.get("decay_tripwire")
        if tw_cfg is not None:
            from services.risk.decay_tripwire import DecayTripwire
            DecayTripwire(
                setup_name=_SETUP_NAME,
                state_path=Path(tw_cfg["state_file"]),
                window_trades=int(tw_cfg["window_trades"]),
                pf_floor=float(tw_cfg["pf_floor"]),
                sustained_weeks=int(tw_cfg["sustained_weeks"]),
            ).record_trade(net_pnl_inr=float(net), ts_iso=now.isoformat())


def _run_entries(raw, broker, persistence, today, now, paper_mode, summary,
                 *, ca_ex_dates, repo_root) -> None:
    cap = raw["capital_allocation"]
    max_concurrent = int(cap["max_concurrent_slots"])
    max_new_per_day = int(cap["max_new_positions_per_day"])
    margin_per_slot = float(cap["margin_per_slot_inr"])

    held = persistence.load_snapshot()
    if len(held) >= max_concurrent:
        logger.info("mtf_capitulation: at concurrency cap (%d); no new entries", max_concurrent)
        return

    # MTF universe (eligibility + per-symbol leverage).
    mtf_cfg = raw["mtf"]
    mtf = MtfUniverse(Path(str(mtf_cfg["approved_list_snapshot_path"])))
    exclude_etf = bool(mtf_cfg["exclude_etf"])
    fallback_cnc = bool(mtf_cfg["fallback_to_cnc_if_not_mtf"])
    eligible = {s for s in mtf.all_symbols() if mtf.is_eligible(s, exclude_etf=exclude_etf)}
    if not eligible:
        logger.warning("mtf_capitulation: empty MTF eligible set; no entries")
        return

    # CA ex-dates (load from configured parquet if not injected).
    if ca_ex_dates is None and bool(raw.get("exclude_ca_in_hold_window")):
        ca_ex_dates = _load_ca_ex_dates(raw, repo_root)

    # Daily panel + rank. Backtest (DRY_RUN) slices the clean feather; live/paper
    # fetches trailing adjusted daily bars via the broker's daily fetcher. If
    # that fetcher isn't wired in live/paper, make_provider fails fast — an
    # honest gate (you cannot run this setup without the daily panel source).
    provider = make_provider(
        raw, dry_run=_is_dry_run(broker),
        fetch_fn=getattr(broker, "fetch_daily_window", None), mtf_symbols=eligible,
        repo_root=repo_root,
    )
    panel = provider.get_panel(today)
    if panel is None or panel.empty:
        logger.warning("mtf_capitulation: empty daily panel for %s; no entries", today)
        return

    ranker = CrossSectionalRanker(raw)
    basket = ranker.rank(panel, today, eligible, ca_ex_dates=ca_ex_dates)
    if not basket:
        logger.info("mtf_capitulation: ranker returned empty basket for %s", today)
        return

    entry_date = _next_trading_day(today)
    exit_on_date = _add_trading_days(entry_date, int(raw["hold_days"]))
    new_today = 0
    for cand in basket:
        if len(persistence.load_snapshot()) >= max_concurrent:
            break
        if new_today >= max_new_per_day:
            break
        bare = str(cand["symbol"]).replace("NSE:", "").upper()
        symbol = f"NSE:{bare}"
        if persistence.get_position(symbol) is not None:
            continue  # already held — basket is "diff vs target"

        info = mtf.lookup(bare)
        if info is not None:
            product, leverage = "MTF", float(info.leverage)
        elif fallback_cnc:
            product, leverage = "CNC", 1.0
        else:
            summary["rejected_count"] += 1
            continue

        signal_close = float(cand["close"])
        qty = int((margin_per_slot * leverage) // signal_close)
        if qty <= 0:
            summary["rejected_count"] += 1
            continue

        trade_id = f"MTFCAP_{today.isoformat()}_{bare}"
        try:
            order_id = _place_amo_buy(broker, symbol, qty, product, trade_id)
        except Exception as e:
            logger.error("mtf_capitulation: AMO BUY failed for %s: %s", symbol, e)
            summary["skipped_count"] += 1
            continue

        persistence.save_position(
            symbol=symbol, side="BUY", qty=qty, avg_price=0.0, trade_id=trade_id,
            order_id=str(order_id), order_tag=trade_id,
            plan={"setup": _SETUP_NAME, "trail_ret": cand["trail_ret"], "tshock": cand["tshock"]},
            state={
                "pending_entry_fill": True, "qty": qty, "leverage": leverage,
                "signal_close": signal_close, "signal_date": today.isoformat(),
            },
            entry_date=entry_date.isoformat(),
            exit_on_date=exit_on_date.isoformat(),
            product=product,
        )
        new_today += 1
        summary["entered_count"] += 1
        summary["events"].append({
            "symbol": symbol, "qty": qty, "product": product, "leverage": leverage,
            "entry_date": entry_date.isoformat(), "exit_on_date": exit_on_date.isoformat(),
            "amo_buy_order_id": str(order_id),
        })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decay_paused(raw: dict) -> bool:
    tw_cfg = raw.get("decay_tripwire")
    if tw_cfg is None:
        return False
    from services.risk.decay_tripwire import DecayTripwire
    return DecayTripwire(
        setup_name=_SETUP_NAME,
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
