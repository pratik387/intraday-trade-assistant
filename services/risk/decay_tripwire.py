"""Rolling-PF tripwire to pause a setup whose forward edge has decayed.

State persisted to JSON between cron runs. The handler (Task 6) calls
`record_trade()` after each settle and `is_paused()` before dispatching
the detector.

For close_dn_overnight_long, configured thresholds (from cell_lock JSON
decay_warning.tripwire):
  window_trades: 30      (rolling window length)
  pf_floor: 1.20         (pause if PF below this for sustained period)
  sustained_weeks: 6     (must stay below floor for this long)

Reset: delete the state file, or call DecayTripwire.reset() programmatically.

Spec: specs/2026-05-21-close_dn_overnight_long-paper-trade-implementation-spec.md
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class _TradeRecord:
    net_pnl_inr: float
    ts_iso: str  # IST-naive ISO 8601
    # Optional cost breakdown + per-trade detail. The rolling-PF gate uses net
    # only; these exist so downstream readers (e.g. the swing dashboard trades
    # tab) can show real gross/fees and per-symbol entry/exit. Legacy ledgers
    # predate them -> None (omitted from persisted JSON).
    fees_inr: Optional[float] = None
    gross_pnl_inr: Optional[float] = None
    symbol: Optional[str] = None
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    qty: Optional[int] = None
    # Idealized paper-fill references captured at trade time (15:25 close /
    # 09:15 open). Persisted so the slippage tool measures real-vs-idealized
    # without post-hoc re-fetch drift. Legacy ledgers predate them -> None.
    idealized_entry: Optional[float] = None
    idealized_exit: Optional[float] = None
    # Multi-day composite attribution: True when this row MIRRORS a trade whose
    # book position is OWNED by another setup (PnL fed to every contributor so
    # each setup's standalone edge stays measurable). Pooled/portfolio views
    # MUST exclude attributed rows or one position double-counts. None/False =
    # this setup's own (owner) position. Legacy ledgers predate this -> None.
    attributed: Optional[bool] = None
    # Entry date (YYYY-MM-DD) of the position this settle belongs to. Needed by
    # the target-exit A/B report to derive the SCHEDULED hold-to-close exit day
    # (entry + K trading days) for counterfactuals. Legacy rows -> None.
    entry_date: Optional[str] = None


class DecayTripwire:
    """Rolling-window PF guard. Pauses a setup whose recent edge has decayed."""

    def __init__(
        self,
        setup_name: str,
        state_path: Path,
        window_trades: int,
        pf_floor: float,
        sustained_weeks: int,
    ):
        if window_trades < 5:
            raise ValueError(f"window_trades must be >= 5, got {window_trades}")
        if pf_floor <= 0:
            raise ValueError(f"pf_floor must be > 0, got {pf_floor}")
        if sustained_weeks < 1:
            raise ValueError(f"sustained_weeks must be >= 1, got {sustained_weeks}")
        self._setup_name = setup_name
        self._state_path = Path(state_path)
        self._window_trades = int(window_trades)
        self._pf_floor = float(pf_floor)
        self._sustained_weeks = int(sustained_weeks)
        self._trades: List[_TradeRecord] = []
        self._first_below_floor_ts: Optional[str] = None
        self._paused_since: Optional[str] = None
        self._load()

    def _load(self) -> None:
        if not self._state_path.exists():
            return
        try:
            with open(self._state_path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"DecayTripwire: state file {self._state_path} corrupt "
                f"(invalid JSON): {e}"
            )
        if not isinstance(data, dict):
            raise ValueError(
                f"DecayTripwire: state file {self._state_path} unexpected shape"
            )
        if data.get("setup_name") != self._setup_name:
            raise ValueError(
                f"DecayTripwire: state file is for setup "
                f"{data.get('setup_name')!r}, expected {self._setup_name!r}"
            )
        trades = data.get("trades", [])
        self._trades = [
            _TradeRecord(
                net_pnl_inr=float(t["net_pnl_inr"]),
                ts_iso=str(t["ts_iso"]),
                fees_inr=(float(t["fees_inr"]) if t.get("fees_inr") is not None else None),
                gross_pnl_inr=(float(t["gross_pnl_inr"]) if t.get("gross_pnl_inr") is not None else None),
                symbol=(str(t["symbol"]) if t.get("symbol") is not None else None),
                entry_price=(float(t["entry_price"]) if t.get("entry_price") is not None else None),
                exit_price=(float(t["exit_price"]) if t.get("exit_price") is not None else None),
                exit_reason=(str(t["exit_reason"]) if t.get("exit_reason") is not None else None),
                qty=(int(t["qty"]) if t.get("qty") is not None else None),
                idealized_entry=(float(t["idealized_entry"]) if t.get("idealized_entry") is not None else None),
                idealized_exit=(float(t["idealized_exit"]) if t.get("idealized_exit") is not None else None),
                attributed=(bool(t["attributed"]) if t.get("attributed") is not None else None),
                entry_date=(str(t["entry_date"]) if t.get("entry_date") is not None else None),
            )
            for t in trades
        ]
        self._first_below_floor_ts = data.get("first_below_floor_ts")
        self._paused_since = data.get("paused_since")

    def _persist(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "setup_name": self._setup_name,
            "window_trades": self._window_trades,
            "pf_floor": self._pf_floor,
            "sustained_weeks": self._sustained_weeks,
            "first_below_floor_ts": self._first_below_floor_ts,
            "paused_since": self._paused_since,
            "trades": [
                # Drop None cost fields so legacy/net-only records stay compact.
                {k: v for k, v in asdict(t).items() if v is not None}
                for t in self._trades
            ],
        }
        tmp = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp.replace(self._state_path)

    def record_trade(self, net_pnl_inr: float, ts_iso: str,
                     fees_inr: Optional[float] = None,
                     gross_pnl_inr: Optional[float] = None,
                     symbol: Optional[str] = None,
                     entry_price: Optional[float] = None,
                     exit_price: Optional[float] = None,
                     exit_reason: Optional[str] = None,
                     qty: Optional[int] = None,
                     idealized_entry: Optional[float] = None,
                     idealized_exit: Optional[float] = None,
                     attributed: Optional[bool] = None,
                     entry_date: Optional[str] = None) -> None:
        """Append a settled trade and re-evaluate paused status.

        Call from run_verify_exit after pool.settle() completes.
        ts_iso is the IST-naive ISO 8601 settlement timestamp.
        fees_inr / gross_pnl_inr and the symbol/entry/exit/reason/qty detail are
        optional metadata for downstream readers (the swing dashboard); they do
        NOT affect the rolling-PF gate (which uses net only).
        """
        self._trades.append(_TradeRecord(
            net_pnl_inr=float(net_pnl_inr), ts_iso=str(ts_iso),
            fees_inr=(float(fees_inr) if fees_inr is not None else None),
            gross_pnl_inr=(float(gross_pnl_inr) if gross_pnl_inr is not None else None),
            symbol=(str(symbol) if symbol is not None else None),
            entry_price=(float(entry_price) if entry_price is not None else None),
            exit_price=(float(exit_price) if exit_price is not None else None),
            exit_reason=(str(exit_reason) if exit_reason is not None else None),
            qty=(int(qty) if qty is not None else None),
            idealized_entry=(float(idealized_entry) if idealized_entry is not None else None),
            idealized_exit=(float(idealized_exit) if idealized_exit is not None else None),
            attributed=(bool(attributed) if attributed is not None else None),
            entry_date=(str(entry_date) if entry_date is not None else None),
        ))
        # Keep buffer larger than window (we trim to last 200 to bound disk size)
        if len(self._trades) > max(self._window_trades * 5, 200):
            self._trades = self._trades[-(self._window_trades * 5):]
        self._reevaluate(now_iso=ts_iso)
        self._persist()

    def _rolling_pf(self) -> Optional[float]:
        """PF over the last `window_trades` trades. Returns None if too few trades."""
        if len(self._trades) < self._window_trades:
            return None
        recent = self._trades[-self._window_trades:]
        gains = sum(t.net_pnl_inr for t in recent if t.net_pnl_inr > 0)
        losses = -sum(t.net_pnl_inr for t in recent if t.net_pnl_inr < 0)
        if losses <= 0:
            return float("inf") if gains > 0 else 1.0
        return gains / losses

    def _reevaluate(self, now_iso: str) -> None:
        pf = self._rolling_pf()
        if pf is None:
            return  # not enough trades yet
        now_dt = datetime.fromisoformat(now_iso)
        if pf < self._pf_floor:
            # Floor breach
            if self._first_below_floor_ts is None:
                self._first_below_floor_ts = now_iso
                logger.info(
                    "DecayTripwire[%s]: rolling PF=%.3f < floor=%.3f - starting sustained-weeks watch",
                    self._setup_name, pf, self._pf_floor,
                )
            elif self._paused_since is None:
                first_dt = datetime.fromisoformat(self._first_below_floor_ts)
                age = now_dt - first_dt
                if age >= timedelta(weeks=self._sustained_weeks):
                    self._paused_since = now_iso
                    logger.warning(
                        "DecayTripwire[%s]: PAUSING - rolling PF=%.3f below floor=%.3f "
                        "for %s (>= %d weeks)",
                        self._setup_name, pf, self._pf_floor, age, self._sustained_weeks,
                    )
        else:
            # PF above floor - clear breach tracking unless we're already paused
            if self._first_below_floor_ts is not None and self._paused_since is None:
                logger.info(
                    "DecayTripwire[%s]: rolling PF=%.3f recovered above floor=%.3f - clearing watch",
                    self._setup_name, pf, self._pf_floor,
                )
                self._first_below_floor_ts = None

    def is_paused(self) -> bool:
        return self._paused_since is not None

    def state_summary(self) -> dict:
        return {
            "setup_name": self._setup_name,
            "trade_count": len(self._trades),
            "rolling_window": self._window_trades,
            "pf_floor": self._pf_floor,
            "sustained_weeks": self._sustained_weeks,
            "current_rolling_pf": self._rolling_pf(),
            "first_below_floor_ts": self._first_below_floor_ts,
            "paused_since": self._paused_since,
        }

    def reset(self) -> None:
        """Manual unpause - clears pause state and floor-breach watch.

        Trade history is preserved so the rolling PF continues from the same
        baseline. To start over completely, delete the state file.
        """
        self._first_below_floor_ts = None
        self._paused_since = None
        self._persist()
        logger.info("DecayTripwire[%s]: manually reset", self._setup_name)
