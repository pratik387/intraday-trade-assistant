# diagnostics_report_builder.py
import json, math
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

try:
    from config.logging_config import get_log_directory
    LOG_DIR = Path(get_log_directory())
except Exception:
    LOG_DIR = Path("logs")

EVENTS_NAME = "events.jsonl"
CSV_NAME = "trade_report.csv"

def _g(d, *p, default=None):
    cur = d
    for x in p:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(x, default)
    return cur

def _ts(x):
    try: return pd.Timestamp(x)
    except Exception: return pd.NaT

def _ok(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v))

def _levels_from_plan(ev):
    plv = _g(ev, "plan", "levels") or {}
    out = {}
    for k in ("PDH","PDL","PDC","ORH","ORL"):
        v = plv.get(k)
        if _ok(v): out[k] = float(v)
    return out

def _load_events(path: Path) -> List[Dict[str, Any]]:
    out = []
    if not path.exists(): return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                ev = json.loads(line)
                if isinstance(ev, dict): out.append(ev)
            except Exception:
                continue
    return out

def build_csv_from_events(log_dir: Path | None = None,
                          jsonl_name: str = EVENTS_NAME,
                          out_name: str = CSV_NAME) -> Path:
    base = Path(log_dir) if log_dir else LOG_DIR
    events_path = base / jsonl_name
    events = _load_events(events_path)

    by_tid: Dict[str, List[Dict[str, Any]]] = {}
    for ev in events:
        tid = ev.get("trade_id") or _g(ev, "plan", "trade_id")
        if not tid: continue
        by_tid.setdefault(tid, []).append(ev)

    rows: List[Dict[str, Any]] = []
    MAX_EXITS = 12

    for tid, evs in by_tid.items():
        evs.sort(key=lambda e: (_ts(e.get("ts")), {"DECISION":0,"ENTRY":1,"EXIT":2}.get(e.get("type"), 9)))
        first = evs[0]
        dec = next((e for e in evs if e.get("type") == "DECISION"), None)
        entries = [e for e in evs if e.get("type") == "ENTRY"]
        exits   = [e for e in evs if e.get("type") == "EXIT"]

        row: Dict[str, Any] = {}
        row["run_id"]   = first.get("run_id")
        row["trade_id"] = tid
        row["symbol"]   = first.get("symbol")

        if dec:
            plan = dec.get("plan") or {}
            decision = dec.get("decision") or {}
            feat = dec.get("features") or {}
            bar5 = dec.get("bar5") or _g(feat, "bar5") or {}
            timectx = _g(feat, "time") or dec.get("timectx") or {}

            row["decision_ts"]  = dec.get("ts")
            row["setup_type"]   = decision.get("setup_type")
            row["regime"]       = decision.get("regime")
            row["reasons"]      = decision.get("reasons")
            row["size_mult"]    = decision.get("size_mult")
            row["min_hold_bars"]= decision.get("min_hold_bars")
            row["rank_score"]   = _g(feat, "ranker", "rank_score", default=feat.get("rank_score"))

            for k in ("open","high","low","close","volume","vwap","adx","bb_width_proxy"):
                row[(k+"5") if k in ("open","high","low","close","volume","vwap","adx") else k] = bar5.get(k)
            row["minute_of_day"]= timectx.get("minute_of_day")
            row["day_of_week"]  = timectx.get("day_of_week")

            row["plan_bias"]    = plan.get("bias")
            row["plan_strategy"]= plan.get("strategy")
            row["plan_price_ref"]= plan.get("price", _g(plan, "entry", "reference"))
            ez = _g(plan, "entry", "zone")
            row["entry_zone_low"]  = ez[0] if isinstance(ez, (list,tuple)) and len(ez)>0 else None
            row["entry_zone_high"] = ez[1] if isinstance(ez, (list,tuple)) and len(ez)>1 else None
            row["hard_sl"] = plan.get("hard_sl", _g(plan, "stop", "hard"))
            tgts = plan.get("targets") or []
            if len(tgts)>0: row["t1"], row["t1_rr"], row["t1_action"] = _g(tgts[0],"level"), _g(tgts[0],"rr"), _g(tgts[0],"action")
            if len(tgts)>1: row["t2"], row["t2_rr"], row["t2_action"] = _g(tgts[1],"level"), _g(tgts[1],"rr"), _g(tgts[1],"action")
            sz = plan.get("sizing") or {}
            row["plan_qty"] = sz.get("qty")
            row["plan_notional"] = sz.get("notional")
            row["risk_per_share"] = sz.get("risk_per_share")
            row["risk_rupees"]    = sz.get("risk_rupees")

            lv = _levels_from_plan(dec)
            row["PDH"], row["PDL"], row["PDC"], row["ORH"], row["ORL"] = lv.get("PDH"), lv.get("PDL"), lv.get("PDC"), lv.get("ORH"), lv.get("ORL")

        total_entry_qty = 0
        entry_prices: List[Any] = []
        if entries:
            e0 = entries[0]; ent0 = e0.get("entry") or {}
            row["entry_ts"]    = e0.get("ts")
            row["side"]        = ent0.get("side")
            row["entry_price"] = ent0.get("price")
            for en in entries:
                enq = int((_g(en,"entry","qty") or 0))
                total_entry_qty += enq
                ep = _g(en,"entry","price")
                if ep is not None: entry_prices.append((enq, float(ep)))
            meta = entries[-1].get("order") or {}
            row["pos_after_qty_entry"] = meta.get("pos_after_qty")
            row["pos_after_avg_entry"] = meta.get("pos_after_avg")
            row["scaled_in"]           = meta.get("scaled_in")
        else:
            row["entry_ts"] = None; row["side"] = None; row["entry_price"] = None
        row["qty"] = total_entry_qty

        gross_exit_qty = 0
        for i, ex in enumerate(exits[:MAX_EXITS], start=1):
            exd = ex.get("exit") or {}
            row[f"e{i}_ts"]     = ex.get("ts")
            row[f"e{i}_reason"] = exd.get("reason")
            row[f"e{i}_qty"]    = exd.get("qty")
            row[f"e{i}_price"]  = exd.get("price")
        for ex in exits:
            exd = ex.get("exit") or {}
            gross_exit_qty += int(exd.get("qty") or 0)

        row["label_hit_t1"] = any((_g(ex,"exit","reason") in ("t1_partial",)) for ex in exits)
        row["label_hit_t2"] = any((_g(ex,"exit","reason") in ("target_t2",)) for ex in exits)

        realized = None
        if total_entry_qty > 0 and entry_prices and exits:
            wae = sum(q*px for q,px in entry_prices) / max(1, sum(q for q,_ in entry_prices))
            side0 = (row.get("side") or "").upper()
            sign = 1.0 if side0 == "BUY" else -1.0
            realized = 0.0
            for ex in exits:
                exd = ex.get("exit") or {}
                q = int(exd.get("qty") or 0); px = exd.get("price")
                if px is None or q == 0: continue
                realized += (float(px) - float(wae)) * float(q) * sign
        row["realized_pnl"] = realized

        row["gross_exit_qty"] = gross_exit_qty
        row["executed"]       = total_entry_qty > 0
        row["position_closed"]= bool(total_entry_qty > 0 and gross_exit_qty >= total_entry_qty)
        row["last_exit_ts"]   = exits[-1].get("ts") if exits else None
        row["last_exit_reason"]= (_g(exits[-1],"exit","reason") if exits else None)

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=[c for c in ["decision_ts","entry_ts","symbol","trade_id"] if c in df.columns],
                            na_position="last")
    out_path = base / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV with explicit flush and error handling
    try:
        df.to_csv(out_path, index=False)

        # Force flush to disk
        import os
        with open(out_path, 'r+b') as f:
            f.flush()
            os.fsync(f.fileno())

    except Exception as e:
        print(f"ERROR: Failed to write CSV to {out_path}: {e}")
        raise

    return out_path

def build_combined_csv_from_current_run(logs_root_dir: Path | None = None,
                                        out_name: str = "combined_trade_report.csv") -> Path:
    """
    Generate a combined CSV report from sessions in the CURRENT TRADING RUN only.

    Args:
        logs_root_dir: Root logs directory (defaults to logs/)
        out_name: Output CSV filename

    Returns:
        Path to the generated combined CSV
    """
    if logs_root_dir is None:
        logs_root_dir = Path(__file__).resolve().parents[1] / "logs"

    logs_root = Path(logs_root_dir)
    all_events = []
    session_count = 0

    # Get sessions from current run
    try:
        from services.logging.run_manager import RunManager
        run_manager = RunManager(logs_root)
        current_run_sessions = run_manager.get_current_run_sessions()
        run_info = run_manager.get_current_run_info()

        if not current_run_sessions:
            print("No current run found or no sessions in current run")
            empty_path = logs_root / out_name
            pd.DataFrame().to_csv(empty_path, index=False)
            return empty_path

        print(f"Current run: {run_info.get('run_id', 'unknown')} with {len(current_run_sessions)} sessions")

    except ImportError:
        print("Run manager not available, falling back to all recent sessions")
        current_run_sessions = [d.name for d in sorted(logs_root.glob("*/")) if d.is_dir() and d.name.startswith("20")]

    # Collect events from current run sessions only
    for session_id in current_run_sessions:
        session_dir = logs_root / session_id
        if not session_dir.is_dir():
            continue

        events_file = session_dir / EVENTS_NAME
        if events_file.exists():
            session_events = _load_events(events_file)
            if session_events:  # Only count sessions with actual events
                # Add session metadata to each event
                for event in session_events:
                    event['session_id'] = session_dir.name
                all_events.extend(session_events)
                session_count += 1

    if not all_events:
        print("No events found across all sessions")
        empty_path = logs_root / out_name
        pd.DataFrame().to_csv(empty_path, index=False)
        return empty_path

    print(f"Found {len(all_events)} events across {session_count} sessions from current run")

    # Process events using the same logic as single-session CSV
    by_tid: Dict[str, List[Dict[str, Any]]] = {}
    for ev in all_events:
        tid = ev.get("trade_id") or _g(ev, "plan", "trade_id")
        if not tid:
            continue
        by_tid.setdefault(tid, []).append(ev)

    rows: List[Dict[str, Any]] = []
    MAX_EXITS = 12

    # Process each trade (same logic as build_csv_from_events)
    for tid, evs in by_tid.items():
        evs.sort(key=lambda e: (_ts(e.get("ts")), {"DECISION":0,"ENTRY":1,"EXIT":2}.get(e.get("type"), 9)))
        first = evs[0]
        dec = next((e for e in evs if e.get("type") == "DECISION"), None)
        exits = [e for e in evs if e.get("type") == "EXIT"]

        if not dec:
            continue

        row = {
            "trade_id": tid,
            "session_id": first.get("session_id", "unknown"),  # Add session tracking
            "symbol": _g(first, "symbol") or "",
            "decision_ts": _g(dec, "ts"),
            "setup_type": _g(dec, "decision", "setup_type"),
            "regime": _g(dec, "decision", "regime"),
            "strategy": _g(dec, "plan", "strategy"),
            "bias": _g(dec, "plan", "bias"),
            "entry_ref": _g(dec, "plan", "entry", "reference"),
            "hard_sl": _g(dec, "plan", "stop", "hard"),
            "qty": _g(dec, "plan", "sizing", "qty"),
            "rank_score": _g(dec, "features", "rank_score"),

            # Add key levels for analysis
            "PDH": _g(dec, "plan", "levels", "PDH"),
            "PDL": _g(dec, "plan", "levels", "PDL"),
            "ORH": _g(dec, "plan", "levels", "ORH"),
            "ORL": _g(dec, "plan", "levels", "ORL"),
        }

        # Add exit data
        if exits:
            exit_data = exits[0].get("exit", {})
            row.update({
                "exit_ts": _g(exits[0], "ts"),
                "exit_reason": exit_data.get("reason"),
                "exit_price": exit_data.get("price"),
                "exit_qty": exit_data.get("qty"),
            })

            # Calculate PnL if we have the data
            entry_price = row["entry_ref"]
            exit_price = row["exit_price"]
            qty = row["qty"]
            bias = row["bias"]

            if all(_ok(x) for x in [entry_price, exit_price, qty]) and bias in ["long", "short"]:
                if bias == "long":
                    pnl = (exit_price - entry_price) * qty
                else:
                    pnl = (entry_price - exit_price) * qty
                row["pnl"] = round(pnl, 2)

        rows.append(row)

    # Create DataFrame and sort by timestamp
    df = pd.DataFrame(rows)
    if not df.empty and "decision_ts" in df.columns:
        df = df.sort_values(by=["decision_ts", "symbol", "trade_id"], na_position="last")

    # Write combined CSV to logs root directory
    out_path = logs_root / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Combined CSV written: {out_path}")
    print(f"Total trades: {len(df)} (from {session_count} sessions in current run)")
    return out_path

if __name__ == "__main__":
    # Generate both single session and combined reports
    p = build_csv_from_events()
    print(f"Single session CSV: {p}")

    combined = build_combined_csv_from_current_run()
    print(f"Combined CSV: {combined}")
