from __future__ import annotations
"""
event_policy_gate.py
--------------------
Lightweight event/expiry gate that adjusts permissions and sizing during
known risk windows. No implicit config reads: the orchestrator (or caller)
registers macro windows and symbol events explicitly.

Policy semantics
----------------
Policy(allow_breakout, allow_fade, size_mult, min_hold_bars)
  • allow_breakout: permit fresh breakout-style entries now
  • allow_fade:     permit failure/fade/mean-revert entries now
  • size_mult:      multiplicative sizing bias to apply to base size
  • min_hold_bars:  require holding at least N closed bars (confirmation)

Typical usage in engine
-----------------------
  gate = EventPolicyGate()
  gate.register_macro_window(start, end, name="RBI_MPC", severity=2)
  gate.register_symbol_event("NSE:TCS", date(2025, 8, 25), name="earnings")
  policy, ctx = gate.decide_policy(now, symbol)
  if setup_type.startswith("breakout") and not policy.allow_breakout: block
  if setup_type.endswith("fade") and not policy.allow_fade: block
  size *= policy.size_mult
  requirements.min_hold_bars = max(requirements.min_hold_bars, policy.min_hold_bars)

Notes
-----
• Times are treated as IST-naive datetimes (engine already runs in IST).
• If no registered windows match, the gate returns a neutral Policy.
"""
from dataclasses import dataclass
from datetime import datetime, date, time as dtime
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Policy:
    allow_breakout: bool
    allow_fade: bool
    size_mult: float = 1.0
    min_hold_bars: int = 0


class EventPolicyGate:
    """Event/expiry/earnings rules with explicit registration.

    The gate does not read any external files. Callers must register windows
    (e.g., RBI/CPI/Budget) and symbol events if they want behavior changes.
    In absence of registrations, the gate is neutral (no restrictions).
    """

    def __init__(self) -> None:
        # Macro windows: list of (start_dt, end_dt, name, severity)
        self._macro: List[Tuple[datetime, datetime, str, int]] = []
        # Symbol events: symbol -> list[(date, name)]
        self._symbol_events: Dict[str, List[Tuple[date, str]]] = {}

    # ---------------- Registration API ----------------
    def register_macro_window(self, start: datetime, end: datetime, *, name: str, severity: int = 1) -> None:
        if end <= start:
            raise ValueError("macro window end must be after start")
        self._macro.append((start, end, name, severity))

    def register_symbol_event(self, symbol: str, on: date, *, name: str) -> None:
        self._symbol_events.setdefault(symbol, []).append((on, name))

    # ---------------- Query helpers ----------------
    def get_active_events(self, now: datetime, symbol: Optional[str] = None) -> List[str]:
        names: List[str] = []
        # Macro
        for s, e, n, _sev in self._macro:
            if s <= now <= e:
                names.append(n)
        # Weekly/monthly expiry heuristic (India): Thursday 13:30–15:00
        if now.weekday() == 3:  # Monday=0 ... Thursday=3
            if dtime(13, 30) <= now.time() <= dtime(15, 0):
                names.append("expiry_window")
        # Symbol
        if symbol is not None:
            for d, n in self._symbol_events.get(symbol, []):
                if d == now.date():
                    names.append(f"symbol:{n}")
        return names

    # ---------------- Core decision ----------------
    def decide_policy(self, now: datetime, symbol: Optional[str] = None) -> Tuple[Policy, Dict[str, object]]:
        """Return a Policy and a small context dict for logging.

        Rules (conservative defaults; adjust by registration):
          • During macro window: block new entries ±20m around window edges by
            requiring min_hold_bars=2 (confirmation), slight size reduction.
          • Expiry Thu 13:30–15:00: disable fresh breakouts, allow fades only, size 0.7.
          • Symbol earnings day: before 13:00 block entries (min_hold_bars=9 ≈45m),
            after 13:00 require confirmation and reduce size.
        """
        active = self.get_active_events(now, symbol)
        ctx = {"active": active}

        # Neutral baseline
        policy = Policy(allow_breakout=True, allow_fade=True, size_mult=1.0, min_hold_bars=0)

        # Expiry window — fade-friendly, avoid initiating breakouts
        if "expiry_window" in active:
            return Policy(allow_breakout=False, allow_fade=True, size_mult=0.7, min_hold_bars=0), ctx

        # Macro windows — require confirmation, trim size a bit
        if any(n for n in active if not n.startswith("symbol:")):
            return Policy(allow_breakout=True, allow_fade=True, size_mult=0.9, min_hold_bars=2), ctx

        # Symbol earnings policy — stricter before/after 13:00
        if any(n for n in active if n.startswith("symbol:")):
            if now.time() < dtime(13, 0):
                return Policy(allow_breakout=False, allow_fade=False, size_mult=0.8, min_hold_bars=9), ctx
            else:
                return Policy(allow_breakout=True, allow_fade=True, size_mult=0.9, min_hold_bars=2), ctx

        return policy, ctx
