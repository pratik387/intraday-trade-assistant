"""Post-EOD paper-reconstruction + slippage tool for close_dn_overnight_long.

READ-ONLY analysis tool. Does NOT place orders.

The setup now trades LIVE at Rs 10k/slot. We keep a separate PAPER record at
Rs 1,00,000/slot (idealized fills) for research continuity and to measure
slippage (real vs idealized). For a given ENTRY date this script:

  1. Parses every FIRED signal from logs/overnight_entry_<DATE>.log (traded +
     live-rejected alike).
  2. Fetches idealized 5m prices via the Upstox SDK (entry = 15:25 bar CLOSE on
     the entry date; exit = 09:15 bar OPEN on the next trading day).
  3. Reconstructs the idealized-Rs1L paper-equivalent for EVERY fired signal and
     appends it (idempotently) to the PAPER ledger
     state/decay_tripwire_close_dn_overnight_long.json.
  4. For signals that ALSO appear in the LIVE ledger
     (state/..._live.json), computes real-vs-idealized slippage and writes a
     report to reports/overnight_slippage_<DATE>.json.

CLI:
    python tools/overnight_paper_slippage.py --date YYYY-MM-DD

The data fetch needs live Upstox creds, so the fetch path is integration-only
(runs on the VM); unit tests inject a fake `fetch_fn`.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from datetime import date, time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

# tools/ is not a package and this is run directly (python tools/overnight_paper_slippage.py),
# so sys.path[0] is tools/, not the repo root. Put the repo root on the path BEFORE importing
# config/services/broker, otherwise those imports raise ModuleNotFoundError at runtime.
import sys
ROOT = Path(__file__).resolve().parents[1]  # this file lives in tools/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from config.logging_config import get_agent_logger
    logger = get_agent_logger()
except Exception:  # pragma: no cover - fallback when agent logger unavailable
    logger = logging.getLogger(__name__)

# Documented PAPER sizing constant (the historical Rs 1,00,000 margin/slot the
# Phase-5 confidence card was computed under). NOT a trading threshold — it is
# the fixed paper-equivalent margin, so it is allowed as a constant here.
PAPER_MARGIN_INR = 100000.0

# Idealized fill anchors (mirror services/execution/overnight_handlers.py).
ENTRY_BAR_HHMM = "15:25"   # MOC paper fill = 15:25 bar CLOSE
EXIT_BAR_HHMM = "09:15"    # AMO paper fill = next-day 09:15 bar OPEN

# Fired-line regex. Example line:
#   close_dn_overnight_long fired | symbol=NSE:PACEDIGITK svr=-1.000 vol_z=24.95 prior_ret=11.23% product=MTF lev=3.34
_FIRED_RE = re.compile(
    r"close_dn_overnight_long fired \|.*?"
    r"symbol=(?P<symbol>\S+).*?"
    r"product=(?P<product>MTF|CNC).*?"
    r"lev=(?P<lev>[-+]?\d+(?:\.\d+)?)"
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_fired_signals(log_text: str) -> List[Tuple[str, str, float]]:
    """Parse FIRED signal lines from an overnight_entry log.

    Returns a list of (symbol, product, lev) tuples in file order. Each fired
    line is a signal the detector produced — traded live OR rejected by slot
    caps; we reconstruct the paper equivalent for all of them.
    """
    out: List[Tuple[str, str, float]] = []
    for line in log_text.splitlines():
        m = _FIRED_RE.search(line)
        if m is None:
            continue
        out.append((m.group("symbol"), m.group("product"), float(m.group("lev"))))
    return out


def load_fired_signals(log_path: Path) -> List[Tuple[str, str, float]]:
    """Read + parse the fired signals from a log file path. Empty if missing."""
    if not log_path.exists():
        logger.warning("load_fired_signals: log not found at %s", log_path)
        return []
    return parse_fired_signals(log_path.read_text(encoding="utf-8", errors="replace"))


# ---------------------------------------------------------------------------
# Idealized price extraction
# ---------------------------------------------------------------------------

def extract_idealized_prices(
    df: Optional[pd.DataFrame], entry_dt: date, exit_dt: date,
) -> Tuple[Optional[float], Optional[float]]:
    """Idealized (entry_close@15:25 on entry_dt, exit_open@09:15 on exit_dt).

    Mirrors the paper fill anchors used live (overnight_handlers
    _paper_fill_price_entry/_exit): entry is the 15:25 bar CLOSE, exit is the
    next-day 09:15 bar OPEN. If the exact bar is missing, fall back to the last
    bar <= 15:25 on the entry date (entry) / the first bar >= 09:15 on the exit
    date (exit). Returns (None, None) components when unavailable.
    """
    if df is None or df.empty:
        return None, None

    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        return None, None

    # Entry: 15:25 bar CLOSE on entry_dt.
    entry_day = df[idx.normalize() == pd.Timestamp(entry_dt)]
    idealized_entry: Optional[float] = None
    if not entry_day.empty:
        exact = entry_day[entry_day.index.strftime("%H:%M") == ENTRY_BAR_HHMM]
        if not exact.empty:
            idealized_entry = float(exact["close"].iloc[0])
        else:
            target = pd.Timestamp.combine(entry_dt, time(15, 25))
            before = entry_day[entry_day.index <= target]
            if not before.empty:
                idealized_entry = float(before["close"].iloc[-1])

    # Exit: 09:15 bar OPEN on exit_dt.
    exit_day = df[idx.normalize() == pd.Timestamp(exit_dt)]
    idealized_exit: Optional[float] = None
    if not exit_day.empty:
        exact = exit_day[exit_day.index.strftime("%H:%M") == EXIT_BAR_HHMM]
        if not exact.empty:
            idealized_exit = float(exact["open"].iloc[0])
        else:
            target = pd.Timestamp.combine(exit_dt, time(9, 15))
            after = exit_day[exit_day.index >= target]
            if not after.empty:
                idealized_exit = float(after["open"].iloc[0])

    return idealized_entry, idealized_exit


# ---------------------------------------------------------------------------
# Reconstruction (idealized Rs 1L paper-equivalent)
# ---------------------------------------------------------------------------

def reconstruct_paper_trade(
    *, symbol: str, product: str, lev: float,
    idealized_entry: float, idealized_exit: float, hold_days: int,
    ts_iso: Optional[str] = None,
) -> Dict:
    """Build the idealized Rs 1,00,000-margin paper trade record for one signal.

    Sizing: notional = PAPER_MARGIN_INR * lev; qty = int(notional / entry) (>=1).
    Fees via the real fee model (CNC: calc_fee_cnc; MTF: calc_fee_mtf with
    margin=PAPER_MARGIN_INR, hold_days).
    """
    from tools.sub7_validation.build_per_setup_pnl import calc_fee_cnc, calc_fee_mtf

    notional_1L = PAPER_MARGIN_INR * float(lev)
    qty_1L = max(1, int(notional_1L / float(idealized_entry)))
    buy_value = float(idealized_entry) * qty_1L
    sell_value = float(idealized_exit) * qty_1L
    gross_1L = (float(idealized_exit) - float(idealized_entry)) * qty_1L

    prod = product.upper()
    if prod == "CNC":
        fees_1L = calc_fee_cnc(buy_value, sell_value)
    elif prod == "MTF":
        fees_1L = calc_fee_mtf(buy_value, sell_value, PAPER_MARGIN_INR, hold_days)
    else:
        raise ValueError(f"reconstruct_paper_trade: unsupported product {product!r}")

    net_1L = gross_1L - fees_1L

    return {
        "symbol": symbol,
        "entry_price": float(idealized_entry),
        "exit_price": float(idealized_exit),
        "qty": qty_1L,
        "product": prod,
        "leverage": float(lev),
        "gross_pnl_inr": gross_1L,
        "fees_inr": fees_1L,
        "net_pnl_inr": net_1L,
        "ts_iso": ts_iso,
        "exit_reason": "reconstructed_paper",
        "source": "reconstructed",
    }


# ---------------------------------------------------------------------------
# Slippage
# ---------------------------------------------------------------------------

def compute_slippage(
    *, idealized_entry: float, real_entry: float,
    idealized_exit: float, real_exit: float,
) -> Dict[str, float]:
    """Real-vs-idealized slippage for one LONG round-trip.

    entry_slip = real_entry - idealized_entry  (paying MORE than the idealized
        close on a long = adverse, positive).
    exit_slip  = idealized_exit - real_exit    (selling for LESS than the
        idealized open = adverse, positive).
    bps versions normalize by the idealized price.
    """
    entry_slip = float(real_entry) - float(idealized_entry)
    exit_slip = float(idealized_exit) - float(real_exit)
    return {
        "entry_slip": entry_slip,
        "exit_slip": exit_slip,
        "entry_slip_bps": entry_slip / float(idealized_entry) * 1e4,
        "exit_slip_bps": exit_slip / float(idealized_exit) * 1e4,
    }


# ---------------------------------------------------------------------------
# Ledger I/O
# ---------------------------------------------------------------------------

def _load_ledger(path: Path) -> Dict:
    """Load a tripwire-style ledger ({"trades": [...]}). Empty shell if absent."""
    if not path.exists():
        return {"trades": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("_load_ledger: failed to parse %s (%s); treating as empty", path, e)
        return {"trades": []}
    if not isinstance(data, dict):
        return {"trades": []}
    data.setdefault("trades", [])
    return data


def _ts_date(ts_iso: Optional[str]) -> Optional[str]:
    """Extract the YYYY-MM-DD date portion from an ISO timestamp string."""
    if not ts_iso:
        return None
    return str(ts_iso)[:10]


def index_live_trades_by_symbol(
    live_ledger: Dict, exit_date: date,
) -> Dict[str, Dict]:
    """Map symbol -> live trade for trades whose ts_iso date == exit_date.

    Live trades are recorded at SETTLE (next-day 09:30), so the live trade for
    an entry on date D has ts_iso on the exit date (next trading day). We match
    a fired signal to its live fill by (symbol, ts_date == exit_date).
    """
    out: Dict[str, Dict] = {}
    target = exit_date.isoformat()
    for t in live_ledger.get("trades", []):
        if not isinstance(t, dict):
            continue
        if _ts_date(t.get("ts_iso")) != target:
            continue
        sym = t.get("symbol")
        if sym:
            out[sym] = t
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _next_trading_day(d: date) -> date:
    """Holiday-aware next trading day (reuses the production helper)."""
    from services.execution.overnight_handlers import _next_trading_day as _ntd
    return _ntd(d)


def _default_fetch_fn(symbols, from_date_iso, to_date_iso):  # pragma: no cover - network
    """Real Upstox SDK fetch (HISTORICAL path; needs live creds).

    The historical endpoint does NOT contain the CURRENT day's bars (they only
    land there after EOD processing). Use this for past dates only; for today's
    bars use `_default_intraday_fetch_fn`.
    """
    import asyncio
    from broker.upstox.upstox_data_client import UpstoxDataClient

    sdk = UpstoxDataClient()
    return asyncio.run(
        sdk.async_fetch_historical_5m_batch(
            symbols, from_date_iso, to_date_iso, concurrency=30, rps=20.0,
        )
    )


def _default_intraday_fetch_fn(symbols):  # pragma: no cover - network
    """Real Upstox SDK fetch for TODAY's 5m bars (intraday endpoint).

    Returns {symbol: df_5m} of today's bars. Mirrors the call
    services/execution/overnight_handlers.py run_entry uses.
    """
    import asyncio
    from broker.upstox.upstox_data_client import UpstoxDataClient

    sdk = UpstoxDataClient()
    return asyncio.run(
        sdk.async_fetch_intraday_5m_batch(symbols, concurrency=30, rps=20.0)
    )


def _today_ist() -> date:
    """Today's date in IST (the VM runs IST; helper keeps it explicit)."""
    try:
        from utils.time_util import _now_naive_ist
        return _now_naive_ist().date()
    except Exception:  # pragma: no cover - fallback when helper unavailable
        from datetime import datetime
        return datetime.now().date()


def reconstruct_for_date(
    *,
    entry_date: date,
    log_path: Path,
    paper_ledger_path: Path,
    live_ledger_path: Path,
    reports_dir: Path,
    fetch_fn: Callable[[List[str], str, str], Dict[str, pd.DataFrame]],
    intraday_fetch_fn: Optional[Callable[[List[str]], Dict[str, pd.DataFrame]]] = None,
) -> Dict:
    """End-to-end reconstruction + slippage for one entry date.

    Returns a summary dict. Writes:
      - reconstructed Rs1L trades into `paper_ledger_path` (idempotent),
      - a slippage report into `reports_dir/overnight_slippage_<DATE>.json`.

    Price fetch is SPLIT by date because the Upstox historical endpoint does
    NOT contain the current day's bars:
      - entry price (15:25 close on the entry date, always past): `fetch_fn`
        (historical).
      - exit price (09:15 open on the exit date): if the exit date is today,
        `intraday_fetch_fn` (today's intraday bars); else `fetch_fn`.
    `intraday_fetch_fn` defaults to the real SDK intraday wrapper; tests inject
    a fake.
    """
    if intraday_fetch_fn is None:
        intraday_fetch_fn = _default_intraday_fetch_fn

    exit_date = _next_trading_day(entry_date)

    fired = load_fired_signals(log_path)
    symbols = sorted({s for s, _, _ in fired})
    logger.info(
        "reconstruct_for_date: entry=%s exit=%s | %d fired signals (%d unique symbols)",
        entry_date, exit_date, len(fired), len(symbols),
    )

    # --- Entry-day bars: 15:25 close on the (always-past) entry date -> historical.
    entry_bars: Dict[str, pd.DataFrame] = {}
    # --- Exit-day bars: 09:15 open on the exit date. If exit is today, the
    # historical endpoint lacks today's bars -> use the intraday endpoint.
    exit_bars: Dict[str, pd.DataFrame] = {}
    exit_is_today = exit_date == _today_ist()
    if symbols:
        entry_iso = entry_date.isoformat()
        entry_bars = fetch_fn(symbols, entry_iso, entry_iso) or {}
        if exit_is_today:
            logger.info(
                "reconstruct_for_date: exit_date %s == today -> intraday endpoint for exit price",
                exit_date,
            )
            exit_bars = intraday_fetch_fn(symbols) or {}
        else:
            exit_iso = exit_date.isoformat()
            exit_bars = fetch_fn(symbols, exit_iso, exit_iso) or {}

    # hold_days: trading days between entry and exit (normally 1).
    hold_days = max(1, (exit_date - entry_date).days)

    live_ledger = _load_ledger(live_ledger_path)
    live_by_symbol = index_live_trades_by_symbol(live_ledger, exit_date)

    recon_ts_iso = f"{exit_date.isoformat()}T09:30:00"

    reconstructed: List[Dict] = []
    per_trade_report: List[Dict] = []
    n_missing_price = 0

    for symbol, product, lev in fired:
        # Entry price comes from the entry-day (historical) bars; exit price
        # from the exit-day bars (intraday if today, else historical).
        ie, _ = extract_idealized_prices(entry_bars.get(symbol), entry_date, exit_date)
        _, ix = extract_idealized_prices(exit_bars.get(symbol), entry_date, exit_date)
        if ie is None or ix is None:
            n_missing_price += 1
            logger.warning(
                "reconstruct_for_date: missing idealized price for %s "
                "(entry=%s exit=%s); skipping reconstruction",
                symbol, ie, ix,
            )
            continue

        rec = reconstruct_paper_trade(
            symbol=symbol, product=product, lev=lev,
            idealized_entry=ie, idealized_exit=ix, hold_days=hold_days,
            ts_iso=recon_ts_iso,
        )
        reconstructed.append(rec)

        # Slippage row (real fills present only if the signal traded live).
        live = live_by_symbol.get(symbol)
        row: Dict = {
            "symbol": symbol,
            "product": product.upper(),
            "idealized_entry": ie,
            "real_entry": None,
            "entry_slip": None,
            "entry_slip_bps": None,
            "idealized_exit": ix,
            "real_exit": None,
            "exit_slip": None,
            "exit_slip_bps": None,
            "paper_net_1L": rec["net_pnl_inr"],
            "live_net_10k": None,
        }
        if live is not None:
            real_entry = float(live.get("entry_price")) if live.get("entry_price") is not None else None
            real_exit = float(live.get("exit_price")) if live.get("exit_price") is not None else None
            if real_entry is not None and real_exit is not None:
                # Prefer the idealized refs PERSISTED on the live trade record at
                # trade time — these are exact (no post-hoc re-fetch drift). Fall
                # back to the fetched ie/ix only when a leg is absent (legacy
                # ledgers predate the persisted refs).
                persisted_ie = live.get("idealized_entry")
                persisted_ix = live.get("idealized_exit")
                slip_ie = float(persisted_ie) if persisted_ie is not None else ie
                slip_ix = float(persisted_ix) if persisted_ix is not None else ix
                slip = compute_slippage(
                    idealized_entry=slip_ie, real_entry=real_entry,
                    idealized_exit=slip_ix, real_exit=real_exit,
                )
                row["idealized_entry"] = slip_ie
                row["idealized_exit"] = slip_ix
                row["real_entry"] = real_entry
                row["real_exit"] = real_exit
                row["entry_slip"] = slip["entry_slip"]
                row["entry_slip_bps"] = slip["entry_slip_bps"]
                row["exit_slip"] = slip["exit_slip"]
                row["exit_slip_bps"] = slip["exit_slip_bps"]
            if live.get("net_pnl_inr") is not None:
                row["live_net_10k"] = float(live.get("net_pnl_inr"))
        per_trade_report.append(row)

    # --- Write the paper ledger (idempotent) ---
    _append_reconstructed(paper_ledger_path, reconstructed, exit_date, entry_date)

    # --- Aggregate slippage ---
    matched = [r for r in per_trade_report if r["real_entry"] is not None]
    n_traded_live = len(matched)

    def _mean(vals: List[float]) -> Optional[float]:
        vals = [v for v in vals if v is not None]
        return (sum(vals) / len(vals)) if vals else None

    # Per-share slip * live qty (when the live qty is recorded) -> rupee slippage
    # actually borne on the live (Rs10k) book.
    total_slip_inr_on_live = 0.0
    for r in matched:
        live = live_by_symbol.get(r["symbol"])
        qty = live.get("qty") if live else None
        if qty and r["entry_slip"] is not None and r["exit_slip"] is not None:
            total_slip_inr_on_live += (r["entry_slip"] + r["exit_slip"]) * float(qty)

    aggregate = {
        "total_entry_slip_bps_mean": _mean([r["entry_slip_bps"] for r in matched]),
        "total_exit_slip_bps_mean": _mean([r["exit_slip_bps"] for r in matched]),
        "total_slip_bps_mean": _mean(
            [(r["entry_slip_bps"] or 0.0) + (r["exit_slip_bps"] or 0.0) for r in matched]
        ),
        "total_slip_inr_on_live": total_slip_inr_on_live,
        "paper_net_1L_sum": sum(r["paper_net_1L"] for r in per_trade_report),
        "live_net_10k_sum": sum(
            r["live_net_10k"] for r in matched if r["live_net_10k"] is not None
        ),
    }

    report = {
        "date": entry_date.isoformat(),
        "exit_date": exit_date.isoformat(),
        "n_fired": len(fired),
        "n_reconstructed": len(reconstructed),
        "n_missing_price": n_missing_price,
        "n_traded_live": n_traded_live,
        "per_trade": per_trade_report,
        "aggregate": aggregate,
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"overnight_slippage_{entry_date.isoformat()}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("reconstruct_for_date: wrote slippage report -> %s", report_path)

    summary = {
        "date": entry_date.isoformat(),
        "exit_date": exit_date.isoformat(),
        "n_fired": len(fired),
        "n_reconstructed": len(reconstructed),
        "n_missing_price": n_missing_price,
        "n_traded_live": n_traded_live,
        "report_path": str(report_path),
        "aggregate": aggregate,
    }
    return summary


def _append_reconstructed(
    paper_ledger_path: Path, reconstructed: List[Dict],
    exit_date: date, entry_date: date,
) -> None:
    """Idempotently replace this exit-date's reconstructed entries in the PAPER ledger.

    Backs up the file first, then removes any existing entries with
    source=="reconstructed" whose ts_iso date == exit_date, and appends the new
    set. Non-reconstructed (real paper-run) trades are never touched.
    """
    ledger = _load_ledger(paper_ledger_path)
    trades = ledger.get("trades", [])
    exit_iso = exit_date.isoformat()

    # Back up before any mutation (only if the file already exists).
    paper_ledger_path.parent.mkdir(parents=True, exist_ok=True)
    if paper_ledger_path.exists():
        backup = paper_ledger_path.with_name(paper_ledger_path.name + f".bak-{entry_date.isoformat()}")
        shutil.copy2(paper_ledger_path, backup)
        logger.info("_append_reconstructed: backed up paper ledger -> %s", backup)

    # Drop prior reconstructed entries for THIS exit date only.
    kept = [
        t for t in trades
        if not (
            isinstance(t, dict)
            and t.get("source") == "reconstructed"
            and _ts_date(t.get("ts_iso")) == exit_iso
        )
    ]
    n_removed = len(trades) - len(kept)
    kept.extend(reconstructed)
    ledger["trades"] = kept

    paper_ledger_path.write_text(json.dumps(ledger, indent=2, default=str), encoding="utf-8")
    logger.info(
        "_append_reconstructed: paper ledger now %d trades "
        "(removed %d stale reconstructed for %s, added %d)",
        len(kept), n_removed, exit_iso, len(reconstructed),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date", required=True, help="ENTRY date (YYYY-MM-DD)")
    p.add_argument(
        "--log-dir", default=str(ROOT / "logs"),
        help="Directory holding overnight_entry_<DATE>.log",
    )
    p.add_argument(
        "--paper-ledger",
        default=str(ROOT / "state" / "decay_tripwire_close_dn_overnight_long.json"),
    )
    p.add_argument(
        "--live-ledger",
        default=str(ROOT / "state" / "decay_tripwire_close_dn_overnight_long_live.json"),
    )
    p.add_argument("--reports-dir", default=str(ROOT / "reports"))
    args = p.parse_args(argv)

    entry_date = date.fromisoformat(args.date)
    log_path = Path(args.log_dir) / f"overnight_entry_{args.date}.log"

    summary = reconstruct_for_date(
        entry_date=entry_date,
        log_path=log_path,
        paper_ledger_path=Path(args.paper_ledger),
        live_ledger_path=Path(args.live_ledger),
        reports_dir=Path(args.reports_dir),
        fetch_fn=_default_fetch_fn,
    )

    agg = summary["aggregate"]
    print(f"=== overnight paper-reconstruction + slippage | {summary['date']} "
          f"(exit {summary['exit_date']}) ===")
    print(f"  fired signals        : {summary['n_fired']}")
    print(f"  reconstructed (Rs1L) : {summary['n_reconstructed']}"
          f"  (missing-price skipped: {summary['n_missing_price']})")
    print(f"  traded live          : {summary['n_traded_live']}")
    print(f"  entry slip bps (mean): {agg['total_entry_slip_bps_mean']}")
    print(f"  exit  slip bps (mean): {agg['total_exit_slip_bps_mean']}")
    print(f"  total slip bps (mean): {agg['total_slip_bps_mean']}")
    print(f"  total slip INR(live) : {agg['total_slip_inr_on_live']}")
    print(f"  report               : {summary['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
