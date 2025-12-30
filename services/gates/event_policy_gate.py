from __future__ import annotations
"""
event_policy_gate.py
--------------------
Comprehensive event/expiry/session gate that adjusts permissions, sizing, and thresholds
during known risk windows and time-of-day sessions.

Phase 4 Integration:
- Time-of-day sessions (opening drive, lunch drift, power hour)
- Special event days (expiry, RBI policy, budget, earnings)
- Session-specific threshold adjustments (ADX, volume, strength)
- Event-specific threshold adjustments

Policy semantics
----------------
Policy(allow_breakout, allow_fade, size_mult, min_hold_bars,
       min_adx, min_volume_mult, min_strength, allow_fast_scalp)
  • allow_breakout:   permit fresh breakout-style entries now
  • allow_fade:       permit failure/fade/mean-revert entries now
  • size_mult:        multiplicative sizing bias to apply to base size
  • min_hold_bars:    require holding at least N closed bars (confirmation)
  • min_adx:          minimum ADX threshold for trend strength (Phase 4)
  • min_volume_mult:  minimum volume multiplier requirement (Phase 4)
  • min_strength:     minimum structure strength requirement (Phase 4)
  • allow_fast_scalp: permit fast scalp lane (weak HTF) entries (Phase 4)

Typical usage in engine
-----------------------
  gate = EventPolicyGate(cfg)
  gate.register_macro_window(start, end, name="RBI_MPC", severity=2)
  gate.register_symbol_event("NSE:TCS", date(2025, 8, 25), name="earnings")

  policy, ctx = gate.decide_policy(
      now=now,
      symbol=symbol,
      adx_5m=adx_5m,
      vol_mult_5m=vol_mult_5m,
      strength=strength,
      lane_type=lane_type
  )

  # Check policy thresholds
  if policy.min_adx and adx_5m < policy.min_adx: block
  if policy.min_volume_mult and vol_mult_5m < policy.min_volume_mult: block
  if policy.min_strength and strength < policy.min_strength: block
  if lane_type == "fast_scalp_lane" and not policy.allow_fast_scalp: block

  size *= policy.size_mult

Notes
-----
• Times are treated as IST-naive datetimes (engine already runs in IST).
• If no registered windows or sessions match, returns neutral Policy.
• Phase 4 session/event thresholds can be customized via config.
"""
from dataclasses import dataclass
from datetime import datetime, date, time as dtime
from typing import Dict, List, Optional, Tuple
from enum import Enum

# Import EventsLoader for accurate event detection
try:
    from services.events.events_loader import EventsLoader
    EVENTS_LOADER_AVAILABLE = True
except ImportError:
    EVENTS_LOADER_AVAILABLE = False


class SessionType(Enum):
    """Time-of-day trading sessions with different characteristics."""
    OPENING_DRIVE = "opening_drive"  # 09:15-09:30: Volatile, fast moves
    LUNCH_DRIFT = "lunch_drift"      # 12:00-13:30: Low volume, choppy
    POWER_HOUR = "power_hour"        # 14:30-15:10: Final push, directional
    NORMAL = "normal"                # Regular trading hours


class EventType(Enum):
    """Special event days requiring threshold adjustments."""
    MONTHLY_EXPIRY = "monthly_expiry"
    RBI_POLICY = "rbi_policy"
    FED_POLICY = "fed_policy"
    BUDGET_DAY = "budget_day"
    ELECTION_RESULTS = "election_results"
    EARNINGS_SEASON = "earnings_season"


@dataclass(frozen=True)
class Policy:
    """Trading policy with permissions and thresholds."""
    allow_breakout: bool
    allow_fade: bool
    size_mult: float = 1.0
    min_hold_bars: int = 0

    # Phase 4 additions: session/event-specific thresholds
    min_adx: Optional[float] = None
    min_volume_mult: Optional[float] = None
    min_strength: Optional[float] = None
    allow_fast_scalp: bool = True

    # Context tracking
    session_type: Optional[str] = None
    event_type: Optional[str] = None


class EventPolicyGate:
    """Event/expiry/earnings/session rules with explicit registration.

    Phase 4 Integration:
    - Time-of-day session detection and policies
    - Special event day detection and threshold adjustments
    - Configurable session/event policies

    The gate can read session/event policies from config or use sensible defaults.
    Callers can still register macro windows and symbol events explicitly.
    """

    def __init__(self, cfg: Optional[Dict] = None) -> None:
        # Macro windows: list of (start_dt, end_dt, name, severity)
        self._macro: List[Tuple[datetime, datetime, str, int]] = []
        # Symbol events: symbol -> list[(date, name)]
        self._symbol_events: Dict[str, List[Tuple[date, str]]] = {}

        # Phase 4: Config for session/event policies
        self.cfg = cfg or {}
        self._session_policies = self._load_session_policies()
        self._event_policies = self._load_event_policies()

        # Initialize EventsLoader for accurate event detection
        self._events_loader: Optional["EventsLoader"] = None
        if EVENTS_LOADER_AVAILABLE:
            try:
                self._events_loader = EventsLoader()
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Failed to initialize EventsLoader: {e}")

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
        # Weekly/monthly expiry heuristic (India): 13:30–15:00 on expiry day
        # NSE moved weekly expiry from Thursday to Tuesday starting Sept 2025
        cutoff_date = date(2025, 9, 1)
        expiry_weekday = 1 if now.date() >= cutoff_date else 3  # Tuesday=1, Thursday=3
        if now.weekday() == expiry_weekday:
            if dtime(13, 30) <= now.time() <= dtime(15, 0):
                names.append("expiry_window")
        # Symbol
        if symbol is not None:
            for d, n in self._symbol_events.get(symbol, []):
                if d == now.date():
                    names.append(f"symbol:{n}")
        return names

    # ---------------- Phase 4: Session Detection ----------------
    def get_session_type(self, now: datetime) -> SessionType:
        """
        Determine current session type from timestamp.

        Session types and characteristics:
        - opening_drive (09:15-09:30): Volatile, fast moves, fast scalp allowed
        - lunch_drift (12:00-13:30): Low volume, choppy, stricter filters
        - power_hour (14:30-15:10): Final push, precision-only, no fast scalps
        - normal: Regular trading hours
        """
        minute_of_day = now.hour * 60 + now.minute

        # 09:15 = 555, 09:30 = 570
        if 555 <= minute_of_day <= 570:
            return SessionType.OPENING_DRIVE

        # 12:00 = 720, 13:30 = 810
        elif 720 <= minute_of_day <= 810:
            return SessionType.LUNCH_DRIFT

        # 14:30 = 870, 15:10 = 910
        elif 870 <= minute_of_day <= 910:
            return SessionType.POWER_HOUR

        else:
            return SessionType.NORMAL

    # ---------------- Phase 4: Special Event Detection ----------------
    def detect_special_event(self, now: datetime, symbol: Optional[str] = None) -> Optional[EventType]:
        """
        Detect special event days requiring threshold adjustments.

        Uses EventsLoader for accurate detection of:
        - Monthly F&O expiry (last Thursday of month - calculated properly)
        - RBI MPC policy announcements (from static JSON)
        - Union Budget day (from static JSON)
        - Stock-specific earnings (from dynamic cache)

        Falls back to heuristic if EventsLoader unavailable.

        Returns EventType if special day, None otherwise.
        """
        check_date = now.date() if isinstance(now, datetime) else now

        # Use EventsLoader for accurate detection if available
        if self._events_loader:
            # Check stock-specific earnings (highest priority)
            if symbol and self._events_loader.has_earnings_today(symbol, check_date):
                return EventType.EARNINGS_SEASON

            # Check macro events
            if self._events_loader.is_budget_day(check_date):
                return EventType.BUDGET_DAY

            if self._events_loader.is_rbi_day(check_date):
                return EventType.RBI_POLICY

            if self._events_loader.is_expiry_day(check_date):
                return EventType.MONTHLY_EXPIRY

            return None

        # Fallback to original heuristic-based detection
        import pandas as pd

        try:
            ts = pd.to_datetime(now)
            day_of_week = ts.dayofweek  # Monday=0, Thursday=3
            day_of_month = ts.day

            # Check if it's monthly/quarterly expiry (last Thursday of month)
            if day_of_month >= 24 and day_of_week == 3:
                return EventType.MONTHLY_EXPIRY

            # Check configured special dates
            special_dates = self.cfg.get('special_event_dates', {})
            date_str = ts.strftime('%Y-%m-%d')

            if date_str in special_dates:
                event_str = special_dates[date_str]
                try:
                    return EventType(event_str)
                except ValueError:
                    # Unknown event type string, ignore
                    pass

        except Exception:
            pass

        return None

    # ---------------- Core decision (enhanced for Phase 4) ----------------
    def decide_policy(
        self,
        now: datetime,
        symbol: Optional[str] = None,
        adx_5m: float = 0.0,
        vol_mult_5m: float = 1.0,
        strength: float = 0.0,
        lane_type: Optional[str] = None
    ) -> Tuple[Policy, Dict[str, object]]:
        """
        Return a Policy and context dict based on events, sessions, and thresholds.

        Phase 4 Integration:
        - Detects time-of-day session (opening/lunch/power hour)
        - Detects special event days (expiry, RBI, budget, etc.)
        - Applies session-specific threshold adjustments
        - Applies event-specific threshold adjustments (overrides session if present)
        - Returns policy with min_adx, min_volume_mult, min_strength, allow_fast_scalp

        Args:
            now: Current timestamp (IST-naive)
            symbol: Symbol being evaluated
            adx_5m: Current 5m ADX value (for threshold checks)
            vol_mult_5m: Current volume multiplier (for threshold checks)
            strength: Structure strength (for threshold checks)
            lane_type: Trading lane ("precision_lane" or "fast_scalp_lane")

        Returns:
            (policy, context): Policy with thresholds + context dict for logging
        """
        ctx = {}

        # 1. Get base policy from original logic (expiry window, macro, symbol events)
        base_policy, base_ctx = self._decide_base_policy(now, symbol)
        ctx.update(base_ctx)

        # 2. Detect session type (Phase 4)
        session_type = self.get_session_type(now)
        ctx["session_type"] = session_type.value

        # 3. Detect special event (Phase 4) - now includes symbol for earnings
        event_type = self.detect_special_event(now, symbol=symbol)
        if event_type:
            ctx["event_type"] = event_type.value

        # 4. Apply session-specific adjustments
        policy = self._apply_session_policy(
            base_policy, session_type, adx_5m, vol_mult_5m, lane_type
        )

        # 5. Apply event-specific adjustments (overrides session if present)
        if event_type:
            policy = self._apply_event_policy(
                policy, event_type, adx_5m, vol_mult_5m, strength
            )

        return policy, ctx

    def _decide_base_policy(self, now: datetime, symbol: Optional[str]) -> Tuple[Policy, Dict]:
        """
        Original policy logic for expiry window, macro events, symbol events.

        This is the pre-Phase 4 logic preserved as baseline.
        """
        active = self.get_active_events(now, symbol)
        ctx = {"active": active}

        # Neutral baseline
        policy = Policy(
            allow_breakout=True,
            allow_fade=True,
            size_mult=1.0,
            min_hold_bars=0,
            allow_fast_scalp=True
        )

        # Expiry window — fade-friendly, avoid initiating breakouts
        if "expiry_window" in active:
            return Policy(
                allow_breakout=False,
                allow_fade=True,
                size_mult=0.7,
                min_hold_bars=0,
                allow_fast_scalp=True
            ), ctx

        # Macro windows — require confirmation, trim size a bit
        if any(n for n in active if not n.startswith("symbol:")):
            return Policy(
                allow_breakout=True,
                allow_fade=True,
                size_mult=0.9,
                min_hold_bars=2,
                allow_fast_scalp=True
            ), ctx

        # Symbol earnings policy — stricter before/after 13:00
        if any(n for n in active if n.startswith("symbol:")):
            if now.time() < dtime(13, 0):
                return Policy(
                    allow_breakout=False,
                    allow_fade=False,
                    size_mult=0.8,
                    min_hold_bars=9,
                    allow_fast_scalp=False
                ), ctx
            else:
                return Policy(
                    allow_breakout=True,
                    allow_fade=True,
                    size_mult=0.9,
                    min_hold_bars=2,
                    allow_fast_scalp=True
                ), ctx

        return policy, ctx

    # ---------------- Phase 4: Session Policy Application ----------------
    def _apply_session_policy(
        self,
        base_policy: Policy,
        session: SessionType,
        adx_5m: float,
        vol_mult_5m: float,
        lane_type: Optional[str]
    ) -> Policy:
        """
        Apply session-specific threshold adjustments to base policy.

        Session policies:
        - opening_drive: Relaxed volume (1.2x), allow fast scalp
        - lunch_drift: Strict volume (2.0x), higher ADX (22)
        - power_hour: Precision only (no fast scalp), high ADX (25)
        - normal: No adjustments
        """
        if session == SessionType.NORMAL:
            return base_policy

        cfg = self._session_policies.get(session, {})

        return Policy(
            allow_breakout=base_policy.allow_breakout and cfg.get("allow_breakout", True),
            allow_fade=base_policy.allow_fade and cfg.get("allow_fade", True),
            size_mult=base_policy.size_mult * cfg.get("size_mult", 1.0),
            min_hold_bars=max(base_policy.min_hold_bars, cfg.get("min_hold_bars", 0)),
            min_adx=cfg.get("min_adx"),
            min_volume_mult=cfg.get("min_volume_mult"),
            min_strength=cfg.get("min_strength"),
            allow_fast_scalp=cfg.get("allow_fast_scalp", True),
            session_type=session.value,
            event_type=None
        )

    # ---------------- Phase 4: Event Policy Application ----------------
    def _apply_event_policy(
        self,
        base_policy: Policy,
        event: EventType,
        adx_5m: float,
        vol_mult_5m: float,
        strength: float
    ) -> Policy:
        """
        Apply event-specific threshold adjustments to base policy.

        Event policies override session policies for stronger controls.

        Event characteristics:
        - monthly_expiry: Higher noise, require stronger confirmation
        - rbi_policy/fed_policy: High volatility, allow lower ADX but require volume
        - earnings_season: Sector-wide moves, require precision
        - budget_day/election_results: Extreme volatility, very selective
        """
        cfg = self._event_policies.get(event, {})

        return Policy(
            allow_breakout=base_policy.allow_breakout and cfg.get("allow_breakout", True),
            allow_fade=base_policy.allow_fade and cfg.get("allow_fade", True),
            size_mult=base_policy.size_mult * cfg.get("size_mult", 1.0),
            min_hold_bars=max(base_policy.min_hold_bars, cfg.get("min_hold_bars", 0)),
            min_adx=cfg.get("min_adx"),
            min_volume_mult=cfg.get("min_volume_mult"),
            min_strength=cfg.get("min_strength"),
            allow_fast_scalp=cfg.get("allow_fast_scalp", True),
            session_type=None,
            event_type=event.value
        )

    # ---------------- Phase 4: Policy Configuration ----------------
    def _load_session_policies(self) -> Dict[SessionType, dict]:
        """
        Load session-specific policy configurations.

        Can be overridden via self.cfg['session_policies'] if provided.
        """
        default_policies = {
            SessionType.OPENING_DRIVE: {
                "allow_breakout": True,
                "allow_fade": True,
                "size_mult": 1.0,
                "min_volume_mult": 1.2,  # Relaxed for volatile opening
                "min_adx": 18,           # Relaxed
                "allow_fast_scalp": True
            },
            SessionType.LUNCH_DRIFT: {
                "allow_breakout": True,
                "allow_fade": True,
                "size_mult": 0.9,        # Reduce size in choppy session
                "min_volume_mult": 2.0,  # Strict volume requirement
                "min_adx": 22,           # Higher ADX for trend confirmation
                "allow_fast_scalp": True
            },
            SessionType.POWER_HOUR: {
                "allow_breakout": True,
                "allow_fade": True,
                "size_mult": 1.0,
                "min_volume_mult": 1.5,
                "min_adx": 25,           # Require strong trend
                "allow_fast_scalp": False  # Precision only in final hour
            },
            SessionType.NORMAL: {
                "allow_breakout": True,
                "allow_fade": True,
                "size_mult": 1.0,
            }
        }

        # Allow override from config
        return self.cfg.get('session_policies', default_policies)

    def _load_event_policies(self) -> Dict[EventType, dict]:
        """
        Load event-specific policy configurations.

        Can be overridden via self.cfg['event_policies'] if provided.
        """
        default_policies = {
            EventType.MONTHLY_EXPIRY: {
                "allow_breakout": True,
                "allow_fade": True,
                "size_mult": 0.8,        # Reduce size on noisy expiry day
                "min_adx": 24,           # Higher than normal
                "min_strength": 0.65,    # Require strong structures
                "allow_fast_scalp": True
            },
            EventType.RBI_POLICY: {
                "allow_breakout": True,
                "allow_fade": True,
                "size_mult": 0.7,        # Significant size reduction
                "min_adx": 15,           # Relaxed (high volatility creates trends)
                "min_volume_mult": 2.5,  # Very strict volume requirement
                "allow_fast_scalp": True
            },
            EventType.FED_POLICY: {
                "allow_breakout": True,
                "allow_fade": True,
                "size_mult": 0.7,
                "min_adx": 15,
                "min_volume_mult": 2.5,
                "allow_fast_scalp": True
            },
            EventType.EARNINGS_SEASON: {
                "allow_breakout": True,
                "allow_fade": True,
                "size_mult": 0.9,
                "min_adx": 23,
                "min_volume_mult": 1.8,
                "allow_fast_scalp": True
            },
            EventType.BUDGET_DAY: {
                "allow_breakout": True,
                "allow_fade": True,
                "size_mult": 0.6,        # Major size reduction for extreme volatility
                "min_adx": 25,           # Strong trend required
                "min_volume_mult": 3.0,  # Exceptional volume
                "min_strength": 0.70,    # Very high confidence
                "allow_fast_scalp": False  # Precision only
            },
            EventType.ELECTION_RESULTS: {
                "allow_breakout": True,
                "allow_fade": True,
                "size_mult": 0.6,
                "min_adx": 25,
                "min_volume_mult": 3.0,
                "min_strength": 0.70,
                "allow_fast_scalp": False
            }
        }

        # Allow override from config
        return self.cfg.get('event_policies', default_policies)

    # ---------------- Convenience Methods ----------------
    def load_stock_events(self, cache_date: Optional[date] = None) -> None:
        """
        Load stock-specific events from dynamic cache.

        Call this at market open to load earnings/corporate actions.
        """
        if self._events_loader:
            self._events_loader.load_stock_events(cache_date)

    def get_trading_multiplier(self, check_date: date, symbol: Optional[str] = None) -> Tuple[float, Optional[str]]:
        """
        Get trading size multiplier based on events.

        Convenience method that wraps EventsLoader.get_trading_multiplier.

        Returns:
            (multiplier, reason): e.g., (0.5, "rbi_policy_day") or (1.0, None)
        """
        if self._events_loader:
            return self._events_loader.get_trading_multiplier(check_date, symbol)

        # Fallback if loader not available - use detect_special_event
        from datetime import datetime
        now = datetime.combine(check_date, dtime(10, 0))
        event_type = self.detect_special_event(now, symbol)

        if event_type == EventType.BUDGET_DAY:
            return (0.0, "budget_day")
        elif event_type == EventType.RBI_POLICY:
            return (0.5, "rbi_policy_day")
        elif event_type == EventType.MONTHLY_EXPIRY:
            return (0.75, "monthly_expiry")
        elif event_type == EventType.EARNINGS_SEASON:
            return (0.0, "earnings_day")

        return (1.0, None)

    def is_high_impact_day(self, check_date: date) -> bool:
        """Check if date has high-impact macro events (Budget or RBI)."""
        if self._events_loader:
            return self._events_loader.is_high_impact_day(check_date)

        # Fallback
        from datetime import datetime
        now = datetime.combine(check_date, dtime(10, 0))
        event_type = self.detect_special_event(now)
        return event_type in (EventType.BUDGET_DAY, EventType.RBI_POLICY)

    def is_expiry_day(self, check_date: date) -> bool:
        """Check if date is monthly F&O expiry day."""
        if self._events_loader:
            return self._events_loader.is_expiry_day(check_date)

        # Fallback
        from datetime import datetime
        now = datetime.combine(check_date, dtime(10, 0))
        return self.detect_special_event(now) == EventType.MONTHLY_EXPIRY

    @property
    def events_loader(self) -> Optional["EventsLoader"]:
        """Access the underlying EventsLoader if available."""
        return self._events_loader
