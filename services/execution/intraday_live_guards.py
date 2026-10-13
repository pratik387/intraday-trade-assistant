"""Funds gate for the intraday book when it shares a real account.

The intraday daemon and the overnight crons trade the SAME Zerodha account. The
overnight book needs cash at its 15:26 entry and the intraday book borrows what
is idle before that. `CapitalManager` budgets against a static config number;
it does not know what the account actually holds, and the overnight book can
use the whole account on a busy day (free balance 2026-10-13: median Rs31k,
low Rs-16k). So a LIVE entry is taken only if

    free_balance - margin_for_this_order >= overnight_reserve_inr

Losses lower the free balance, so the same rule also stops new entries after a
bad morning: the reserve is the cash the overnight entry is left with, give or
take the losses on positions already open.

Fails CLOSED: no balance or no margin figure means no real order. A blocked
entry continues as a shadow trade (the executor's existing mechanism) so what
the gate cost stays measurable.

Paper (`broker.dry_run`): not evaluated - paper money has no account balance,
and the paper book is the validation stream. No broker call is made.

Deliberately NOT here: any circuit-proximity rule. How each setup treats
circuits was settled per setup in backtest; a blanket live rule would change
which trades are taken without that evidence (user, 2026-10-13).

No trading decision reads the wall clock; `time.monotonic()` only bounds how
often the balance endpoint is hit.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

# Same pattern as capital_manager / overnight_handlers: the agent logger is what
# reaches the session log file; fall back to a module logger outside a session.
try:
    from config.logging_config import get_agent_logger
    logger = get_agent_logger()
    if logger is None:
        raise ValueError("get_agent_logger returned None")
except Exception:
    logger = logging.getLogger(__name__)

CONFIG_KEY = "intraday_live_guards"


class IntradayLiveGuardsConfigError(Exception):
    """A required intraday_live_guards key is missing or invalid."""


def validate_cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Fail fast on a missing/invalid block. Returns the block unchanged."""
    if not isinstance(cfg, dict):
        raise IntradayLiveGuardsConfigError(f"{CONFIG_KEY} missing from configuration.json")
    sub = cfg.get("funds_gate")
    if not isinstance(sub, dict):
        raise IntradayLiveGuardsConfigError(f"{CONFIG_KEY}.funds_gate missing")
    for k in ("enabled", "overnight_reserve_inr", "balance_cache_secs"):
        if k not in sub:
            raise IntradayLiveGuardsConfigError(f"{CONFIG_KEY}.funds_gate.{k} missing")
    if float(sub["overnight_reserve_inr"]) < 0:
        raise IntradayLiveGuardsConfigError(f"{CONFIG_KEY}.funds_gate.overnight_reserve_inr must be >= 0")
    if float(sub["balance_cache_secs"]) < 0:
        raise IntradayLiveGuardsConfigError(f"{CONFIG_KEY}.funds_gate.balance_cache_secs must be >= 0")
    return cfg


def funds_verdict(free_balance_inr: float, margin_required_inr: float, reserve_inr: float) -> Tuple[bool, str]:
    """Pure rule: does this order leave the overnight reserve intact?"""
    after = float(free_balance_inr) - float(margin_required_inr)
    if after >= float(reserve_inr):
        return True, f"free={free_balance_inr:.0f} margin={margin_required_inr:.0f} after={after:.0f} reserve={reserve_inr:.0f}"
    return False, f"free={free_balance_inr:.0f} margin={margin_required_inr:.0f} after={after:.0f} < reserve={reserve_inr:.0f}"


class IntradayLiveGuards:
    """Evaluates the funds gate for one prospective entry against the real broker."""

    def __init__(self, cfg: Dict[str, Any], broker) -> None:
        self.cfg = validate_cfg(cfg)
        self.broker = broker
        self.live = not bool(getattr(broker, "dry_run", True))
        self._bal: Optional[float] = None
        self._bal_at: float = 0.0
        fg = self.cfg["funds_gate"]
        logger.info(
            "LIVE_GUARDS | mode=%s | funds_gate enabled=%s reserve=Rs%.0f",
            "LIVE" if self.live else "PAPER (not evaluated)",
            fg["enabled"], float(fg["overnight_reserve_inr"]),
        )

    def _free_balance(self) -> float:
        ttl = float(self.cfg["funds_gate"]["balance_cache_secs"])
        now = time.monotonic()
        if self._bal is not None and (now - self._bal_at) <= ttl:
            return self._bal
        self._bal = float(self.broker.get_free_balance())
        self._bal_at = now
        return self._bal

    def note_entry(self) -> None:
        """An order went out: the cached balance is stale, drop it."""
        self._bal = None

    def check_entry(self, symbol: str, side: str, qty: int, price: float) -> Tuple[bool, str]:  # noqa: ARG002
        """(allowed, reason). Paper: always allowed, no broker call."""
        if not self.live:
            return True, "paper_not_evaluated"
        fg = self.cfg["funds_gate"]
        if not fg["enabled"]:
            return True, "funds_gate_disabled"
        try:
            free = self._free_balance()
            margin = float(self.broker.get_order_margin(symbol, side, int(qty), product="MIS"))
        except Exception as e:  # fail closed: no figure, no real order
            self._bal = None
            logger.warning("FUNDS_GATE | SKIP | %s | %s qty=%d | funds_unavailable:%s", symbol, side, int(qty), e)
            return False, f"funds_gate:funds_unavailable:{e}"
        if margin <= 0:
            logger.warning("FUNDS_GATE | SKIP | %s | %s qty=%d | margin_unavailable", symbol, side, int(qty))
            return False, "funds_gate:margin_unavailable"
        ok, reason = funds_verdict(free, margin, float(fg["overnight_reserve_inr"]))
        if not ok:
            logger.warning("FUNDS_GATE | SKIP | %s | %s qty=%d | %s", symbol, side, int(qty), reason)
            return False, f"funds_gate:{reason}"
        logger.info("FUNDS_GATE | OK | %s | %s qty=%d | %s", symbol, side, int(qty), reason)
        return True, "ok"
