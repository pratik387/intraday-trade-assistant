"""Live execution shortfall ledger (Risk Charter, execution rules).

Every live intraday fill is recorded against the price the decision assumed: the
entry against the trigger price, every exit leg against the exit price the exit
logic used. One JSON line per fill in `shortfall_log.ledger_path`, so the record
survives restarts and spans sessions. Shortfall is signed ADVERSE: positive bps =
the fill was worse than the decision.

After every exit leg the setup's last `rolling_trades` trades (entry + qty-weighted
exits, each capped at `per_trade_cap_bps` so one circuit-locked cover cannot carry
the mean) are averaged; above `alarm_frac_of_backtest_edge` x the setup's
`backtest_edge_bps` the alarm is logged CRITICAL. The alarm only reports: what to
change (order type, size, the setup) is a decision, not an automatic action.

Live only. Paper and backtest fills are simulated, so their shortfall is not a
measurement.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional

from config.logging_config import get_execution_loggers
from utils.time_util import _now_naive_ist

logger, _ = get_execution_loggers()

_lock = threading.Lock()


def _cfg() -> dict:
    from services.plan_orchestrator import _load_root_config
    return _load_root_config()["shortfall_log"]


def _ledger_path() -> Path:
    p = Path(_cfg()["ledger_path"])
    return p if p.is_absolute() else Path(__file__).resolve().parents[2] / p


def adverse_bps(side: str, decision_px: float, fill_px: float) -> float:
    """bps the fill is worse than the decision for an order on `side` (BUY/SELL)."""
    if not decision_px or decision_px <= 0:
        raise ValueError(f"decision price must be > 0, got {decision_px!r}")
    d = (fill_px - decision_px) if side.upper() == "BUY" else (decision_px - fill_px)
    return 1e4 * d / decision_px


def validate(root: dict, live_setups: List[str]) -> None:
    """Live startup: the config is usable and the ledger is writable (else ValueError).
    Without this a bad value would only surface on the first fill, as a swallowed error."""
    cfg = root.get("shortfall_log")
    keys = ("ledger_path", "rolling_trades", "alarm_frac_of_backtest_edge", "per_trade_cap_bps")
    if not isinstance(cfg, dict) or not all(k in cfg for k in keys):
        raise ValueError(f"shortfall_log.{{{', '.join(keys)}}} missing (live fills are measured against it)")
    if not (isinstance(cfg["rolling_trades"], int) and cfg["rolling_trades"] >= 1):
        raise ValueError(f"shortfall_log.rolling_trades must be an integer >= 1, got {cfg['rolling_trades']!r}")
    for k in ("alarm_frac_of_backtest_edge", "per_trade_cap_bps"):
        if not (isinstance(cfg[k], (int, float)) and cfg[k] > 0):
            raise ValueError(f"shortfall_log.{k} must be > 0, got {cfg[k]!r}")
    setups = root.get("setups") or {}
    bad = [n for n in live_setups
           if not (isinstance(setups[n].get("backtest_edge_bps"), (int, float)) and setups[n]["backtest_edge_bps"] > 0)]
    if bad:
        raise ValueError(f"setups.<name>.backtest_edge_bps (> 0) missing on live setups {bad} "
                         f"(the shortfall alarm reads it)")
    if live_setups:
        p = Path(cfg["ledger_path"])
        p = p if p.is_absolute() else Path(__file__).resolve().parents[2] / p
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8"):
                pass
        except OSError as e:
            raise ValueError(f"shortfall ledger {p} not writable: {e}") from e


def _append(row: dict) -> None:
    path = _ledger_path()
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def record_entry(*, trade_id: str, setup: str, symbol: str, side: str, qty: int,
                 decision_px: float, fill_px: float) -> None:
    if not trade_id or not setup:
        raise ValueError(f"trade_id and setup are required (got {trade_id!r}, {setup!r})")
    bps = adverse_bps(side, decision_px, fill_px)
    _append({"ts": _now_naive_ist().isoformat(timespec="seconds"), "leg": "entry",
             "trade_id": trade_id, "setup": setup, "symbol": symbol, "side": side.upper(),
             "qty": int(qty), "decision_px": float(decision_px), "fill_px": float(fill_px),
             "bps": round(bps, 2), "inr": round(bps / 1e4 * decision_px * int(qty), 2)})


def record_exit(*, trade_id: str, setup: str, symbol: str, side: str, qty: int,
                decision_px: float, fill_px: float, reason: str) -> None:
    """`side` is the exit order's side. Records the leg, then checks the alarm."""
    if not trade_id or not setup:
        raise ValueError(f"trade_id and setup are required (got {trade_id!r}, {setup!r})")
    bps = adverse_bps(side, decision_px, fill_px)
    _append({"ts": _now_naive_ist().isoformat(timespec="seconds"), "leg": "exit",
             "trade_id": trade_id, "setup": setup, "symbol": symbol, "side": side.upper(),
             "qty": int(qty), "decision_px": float(decision_px), "fill_px": float(fill_px),
             "bps": round(bps, 2), "inr": round(bps / 1e4 * decision_px * int(qty), 2),
             "reason": reason})
    try:
        check_alarm(setup)
    except Exception as e:      # the leg is recorded; only the check failed
        logger.error(f"SHORTFALL | {setup} | alarm check failed: {e}")


def trade_shortfalls(rows: List[dict], setup: str) -> List[dict]:
    """Per trade of `setup` that has an exit leg, in exit order: entry bps + the
    qty-weighted exit bps, and the rupee total."""
    by: Dict[str, dict] = {}
    for r in rows:
        if r.get("setup") != setup:
            continue
        t = by.setdefault(r["trade_id"], {"trade_id": r["trade_id"], "entry_bps": None,
                                          "x_qty": 0, "x_bps_qty": 0.0, "inr": 0.0, "last": None})
        t["inr"] += float(r["inr"])
        if r["leg"] == "entry":
            t["entry_bps"] = float(r["bps"])
        else:
            t["x_qty"] += int(r["qty"])
            t["x_bps_qty"] += float(r["bps"]) * int(r["qty"])
            t["last"] = r["ts"]
    out = []
    for t in by.values():
        if t["entry_bps"] is None or not t["x_qty"]:
            continue                       # entry before the ledger existed, or still open
        x = t["x_bps_qty"] / t["x_qty"]
        out.append({"trade_id": t["trade_id"], "entry_bps": t["entry_bps"], "exit_bps": x,
                    "total_bps": t["entry_bps"] + x, "inr": t["inr"], "last": t["last"]})
    return sorted(out, key=lambda t: t["last"])


def check_alarm(setup: str) -> Optional[float]:
    """Mean total shortfall (bps) of the setup's last rolling_trades trades, or None
    while fewer exist. Logs SHORTFALL_ALARM CRITICAL above the threshold."""
    from services.plan_orchestrator import _load_root_config
    root = _load_root_config()
    cfg = root["shortfall_log"]
    n = int(cfg["rolling_trades"])
    edge = float(root["setups"][setup]["backtest_edge_bps"])
    path = _ledger_path()
    with _lock:
        lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    rows = []
    for ln in lines:
        try:
            r = json.loads(ln)
        except ValueError:
            r = None
        if isinstance(r, dict):
            rows.append(r)
        elif ln.strip():
            logger.error(f"SHORTFALL | unreadable ledger line skipped: {ln[:120]!r}")
    trades = trade_shortfalls(rows, setup)[-n:]
    if len(trades) < n:
        return None
    cap = float(cfg["per_trade_cap_bps"])
    mean = sum(min(t["total_bps"], cap) for t in trades) / n
    capped = sum(t["total_bps"] > cap for t in trades)
    limit = float(cfg["alarm_frac_of_backtest_edge"]) * edge
    if mean > limit:
        logger.critical(f"SHORTFALL_ALARM | {setup} | last {n} live trades lose {mean:.1f} bps to execution "
                        f"(entry {sum(t['entry_bps'] for t in trades)/n:.1f}, exit "
                        f"{sum(t['exit_bps'] for t in trades)/n:.1f} uncapped; {capped} trade(s) capped at "
                        f"{cap:.0f}) > {limit:.1f} bps "
                        f"({cfg['alarm_frac_of_backtest_edge']:.0%} of the {edge:.0f} bps backtest edge)")
    return mean
