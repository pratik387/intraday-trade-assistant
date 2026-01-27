# services/diagnostics/diag_event_log.py
from __future__ import annotations

import os
import json
import threading
import atexit
import queue
import time
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np
import portalocker

SCHEMA_VERSION = "trade.v1"
from pathlib import Path
try:
    from config.logging_config import get_log_directory
except Exception:
    # fallback if logging_config isn't available during tests
    def get_log_directory() -> str:
        return str((Path.cwd() / "logs").absolute())


# ---------------------- helpers ----------------------
def _iso(ts: Any) -> Optional[str]:
    """
    Return a NAIVE timestamp string (no timezone, no suffix),
    e.g. '2025-09-07 10:12:13.123456'. If input is tz-aware, drop tz.
    """
    if ts is None:
        return None
    try:
        t = pd.Timestamp(ts)
    except Exception:
        # fall back to plain string; caller ensured it's serializable
        return str(ts)

    # drop timezone if present
    if getattr(t, "tz", None) is not None:
        try:
            t = t.tz_convert(None)
        except Exception:
            # if it's naive already or not convertible, localize unset
            t = t.tz_localize(None)

    # format without trailing microsecond zeros
    s = t.strftime("%Y-%m-%d %H:%M:%S.%f").rstrip("0").rstrip(".")
    return s


def _np_item(x: Any) -> Any:
    """Convert numpy scalars to native Python."""
    try:
        if isinstance(x, np.generic):
            return x.item()
    except Exception:
        pass
    return x


def _json_coerce(x: Any) -> Any:
    """
    Recursively coerce common non-JSON types (pandas/NumPy/Timestamps) to JSON-safe values.
    We avoid lossy stringification for timestamps by routing through _iso.
    """
    if x is None:
        return None
    if isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, pd.Timestamp):
        return _iso(x)
    x = _np_item(x)
    if isinstance(x, (bool, int, float, str)) or x is None:
        return x
    if isinstance(x, dict):
        return {str(k): _json_coerce(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_json_coerce(v) for v in x]
    if isinstance(x, pd.Series):
        try:
            return _json_coerce(x.to_dict())
        except Exception:
            return _json_coerce(x.tolist())
    if isinstance(x, pd.DataFrame):
        try:
            return _json_coerce(x.to_dict(orient="records"))
        except Exception:
            return _json_coerce(x.values.tolist())
    try:
        return str(x)
    except Exception:
        return None


def mint_trade_id(symbol: str, token: Optional[str] = None) -> str:
    """
    Convenience helper for upstream code (e.g., Screener) to mint an immutable trade_id once.
    Keep it simple: SYMBOL_<8-char-token>. If token not provided, use a short time-based token.
    """
    if token is None:
        token = pd.Timestamp.utcnow().strftime("%H%M%S%f")[:8]
    return f"{symbol}_{token}"


def _trade_id(symbol: str, plan: Dict[str, Any]) -> str:
    """
    Stable trade ID resolution:
    1) If plan.trade_id exists, ALWAYS use it (recommended).
    2) Else fall back to entry_epoch_ms or entry_ts if present.
    3) Else mint a time-based id (rare if you mint upfront).
    """
    if plan and plan.get("trade_id"):
        return str(plan["trade_id"])
    ep = (plan or {}).get("entry_epoch_ms")
    if ep is not None:
        try:
            return f"{symbol}_{int(ep)}"
        except Exception:
            pass
    ets = (plan or {}).get("entry_ts")
    if ets:
        return f"{symbol}_{str(ets).replace(':', '').replace('-', '').replace(' ', '_')}"
    # last resort (avoid where possible; mint upstream instead)
    return f"{symbol}_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}"


# ---------------------- event writer ----------------------
class _EventWriter:
    """
    Append-only JSONL event log for decisions, entries, and exits.
    - Pure pass-through: no indicator math, no recompute.
    - Versioned schema (SCHEMA_VERSION).
    - Thread-safe; optional async mode for hot paths.
    """

    def __init__(self):
        self.run_id: Optional[str] = None
        self.dir = get_log_directory()
        self.path: Optional[str] = None
        self._fh = None

        # hardening for live
        self._lock = threading.Lock()
        self._async = False
        self._q: Optional[queue.Queue] = None
        self._worker: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._flush_every_n = 1
        self._write_count = 0
        self._with_file_lock = False
        self._portalocker = None  # set when with_file_lock=True and portalocker is available

    # -------- lifecycle --------
    def set_output(
        self,
        out_dir: str,
        run_id: Optional[str],
        async_mode: bool = False,
        flush_every_n: int = 1,
        with_file_lock: bool = False,
    ):
        """
        Configure the log file and (optionally) enable async mode.
        - async_mode=True: trading threads enqueue to an in-memory queue; a daemon thread writes to disk.
        - flush_every_n: flush every N writes (sync mode). In async mode, a background flusher runs ~2x/sec.
        - with_file_lock=True: use OS file locks (requires `portalocker`) if multiple *processes* write same file.
        """
        self.dir = out_dir or self.dir
        self.run_id = run_id
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, "events.jsonl")
        self._fh = open(self.path, "a", encoding="utf-8")
        self._async = bool(async_mode)
        self._flush_every_n = max(1, int(flush_every_n))
        self._with_file_lock = bool(with_file_lock)

        if self._with_file_lock:
            try:
                self._portalocker = portalocker
            except Exception:
                self._portalocker = None

        if self._async and self._worker is None:
            self._q = queue.Queue(maxsize=10000)
            self._stop.clear()
            self._worker = threading.Thread(target=self._drain, name="diag-writer", daemon=True)
            self._worker.start()
            atexit.register(self.close)

    def close(self):
        # stop async worker first
        if self._worker:
            self._stop.set()
            try:
                if self._q:
                    self._q.put_nowait(None)
            except Exception:
                pass
            self._worker.join(timeout=2.0)
            self._worker = None

        # close file
        if self._fh:
            try:
                self._fh.flush()
                self._fh.close()
            finally:
                self._fh = None

    def reset(self):
        """Close current handle (if any). Caller can set_output() again after this."""
        self.close()
        self.path = None

    # -------- internals --------
    def _ensure_open(self):
        if not self._fh:
            os.makedirs(self.dir, exist_ok=True)
            self.path = os.path.join(self.dir, "events.jsonl")
            self._fh = open(self.path, "a", encoding="utf-8")

    def _emit(self, obj: Dict[str, Any]):
        if self._async and self._q is not None:
            # non-blocking hot path; if queue is full, fallback to sync to avoid data loss
            try:
                self._q.put_nowait(_json_coerce(obj))
            except queue.Full:
                self._write_sync(_json_coerce(obj))
            return
        self._write_sync(_json_coerce(obj))

    def _write_sync(self, safe_obj: Dict[str, Any]):
        self._ensure_open()
        line = json.dumps(safe_obj, ensure_ascii=False) + "\n"
        with self._lock:
            if self._portalocker and self._with_file_lock:
                self._portalocker.lock(self._fh, self._portalocker.LOCK_EX)
            try:
                self._fh.write(line)
                self._write_count += 1
                if (self._write_count % self._flush_every_n) == 0:
                    self._fh.flush()
            finally:
                if self._portalocker and self._with_file_lock:
                    self._portalocker.unlock(self._fh)

    def _drain(self):
        # background writer loop
        last_flush = time.time()
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.2)
            except queue.Empty:
                item = None
            if item is None:
                # periodic flush
                if self._fh and (time.time() - last_flush) > 0.5:
                    with self._lock:
                        self._fh.flush()
                    last_flush = time.time()
                if self._stop.is_set():
                    break
                continue
            self._write_sync(item)

    # -------- public API: events --------
    def log_decision(
        self,
        *,
        symbol: str,
        now: Any,
        plan: Dict[str, Any],
        features: Dict[str, Any],
        decision: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        DECISION event
        - features: precomputed inputs grouped by namespaces:
            features["bar5"]   -> OHLCV/VWAP/ADX/bb_width_proxy (5m snapshot)
            features["ranker"] -> dict fed to your Ranker (volume_ratio, rsi, adx, etc.)
            features["index"]  -> optional index context you already computed
            features["news"]   -> optional news spike metrics already computed
            features["time"]   -> optional time-of-day context already computed
          (No computation here—pure pass-through.)
        - decision: setup_type, regime, reasons (string or list), size_mult, min_hold_bars
        - levels: PDH/PDL/PDC/ORH/ORL dictionary
        - plan: full planner snapshot as-is (bias/strategy/entry/stop/targets/trail/sizing/indicators/quality)
        """
        tid = _trade_id(symbol, plan)
        # (Optionally normalize decision.reasons to string upstream)
        ev = {
            "schema_version": SCHEMA_VERSION,
            "type": "DECISION",
            "run_id": self.run_id,
            "trade_id": tid,
            "symbol": symbol,
            "ts": _iso(now),
            "decision": decision or {},
            "plan": plan or {},
            "bar5": (features or {}).get("bar5", {}),
            "features": (features or {}).get("ranker", {}),
            "index": (features or {}).get("index", {}),
            "news": (features or {}).get("news", {}),
            "timectx": (features or {}).get("time", {}),
            "meta": meta or {},
        }
        self._emit(ev)
        return tid

    def log_entry_fill(
        self,
        *,
        symbol: str,
        plan: Dict[str, Any],
        side: str,
        qty: int,
        price: float,
        entry_ts: Any = None,
        order_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        ENTRY event (actual execution fill).
        - plan MUST carry the same trade_id minted at decision time; logger will use it.
        - order_meta: optional dict for broker/product/variety, etc. (pure pass-through).
        """
        tid = _trade_id(symbol, plan)
        ev = {
            "schema_version": SCHEMA_VERSION,
            "type": "ENTRY",
            "run_id": self.run_id,
            "trade_id": tid,
            "symbol": symbol,
            "ts": _iso(entry_ts),
            "entry": {
                "side": str(side).upper(),
                "qty": int(qty),
                "price": float(price),
            },
            "order": order_meta or {},
        }
        self._emit(ev)
        return tid

    def log_exit(
        self,
        *,
        symbol: str,
        plan: Dict[str, Any],
        reason: str,
        exit_price: float,
        exit_qty: int,
        ts: Any = None,
    ) -> str:
        """
        EXIT event (partial or full).
        - reason should be canonicalized upstream (e.g., target_t1, target_t2, hard_sl, trail_stop, eod_squareoff).
        """
        tid = _trade_id(symbol, plan)
        ev = {
            "schema_version": SCHEMA_VERSION,
            "type": "EXIT",
            "run_id": self.run_id,
            "trade_id": tid,
            "symbol": symbol,
            "ts": _iso(ts),
            "exit": {
                "reason": str(reason).lower(),
                "qty": int(exit_qty),
                "price": float(exit_price),
            },
        }
        self._emit(ev)
        return tid

    def log_thesis_exit(
        self,
        *,
        symbol: str,
        trade_id: str,
        ts: Any = None,
        setup_type: str,
        category: str,
        combined_score: float,
        threshold: float,
        momentum_score: float,
        volume_score: float,
        structure_score: float,
        target_score: float,
        entry_indicators: Dict[str, Any],
        current_indicators: Dict[str, Any],
        primary_factors: list,
        exit_reason: str,
    ) -> str:
        """
        THESIS_EXIT event - logged when position thesis monitoring triggers an exit.
        Captures all indicator comparisons for analysis.
        """
        ev = {
            "schema_version": SCHEMA_VERSION,
            "type": "THESIS_EXIT",
            "run_id": self.run_id,
            "trade_id": trade_id,
            "symbol": symbol,
            "ts": _iso(ts),
            "thesis": {
                "setup_type": setup_type,
                "category": category,
                "combined_score": round(combined_score, 3),
                "threshold": threshold,
                "scores": {
                    "momentum": round(momentum_score, 3),
                    "volume": round(volume_score, 3),
                    "structure": round(structure_score, 3),
                    "target": round(target_score, 3),
                },
                "entry_indicators": _json_coerce(entry_indicators),
                "current_indicators": _json_coerce(current_indicators),
                "primary_factors": primary_factors,
                "exit_reason": exit_reason,
            },
        }
        self._emit(ev)
        return trade_id


# ---------------------- thread-local proxy ----------------------
_tls = threading.local()


def _get_writer() -> _EventWriter:
    if not hasattr(_tls, "writer"):
        _tls.writer = _EventWriter()
    return _tls.writer


class EventLogProxy:
    """
    Thread-local proxy, so you can:
        from services.diagnostics.diag_event_log import diag_event_log, mint_trade_id
        diag_event_log.set_output(...); diag_event_log.log_decision(...); ...
    """
    def __getattr__(self, name):
        return getattr(_get_writer(), name)


diag_event_log = EventLogProxy()
