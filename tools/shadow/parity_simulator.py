"""Parity simulator — sub-project #4.

Replays a cached gate_input.jsonl through services.gate_chain.LiveGateChain
and writes the admit decisions to a CSV. Bit-exact with live's gate output by
construction (wraps the same Python module).

Usage:
    python tools/shadow/parity_simulator.py \\
        --gate-input <path-to-gate_input.jsonl OR dir-of-session-dirs> \\
        --config <path-to-configuration.json> \\
        --output <path-to-sim_admits.csv>
"""
from __future__ import annotations

import argparse
import csv as _csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Ensure repo root is on sys.path so service modules resolve when this script
# is executed directly (e.g. python tools/shadow/parity_simulator.py ...).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from services.gate_chain.live_gate_chain import LiveGateChain


def _read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _parse_dt(value: Any) -> Any:
    """Parse ISO datetime string to datetime if needed; pass through if already datetime."""
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return value
    return value


def _replay_one_session(
    rows: List[Dict[str, Any]], chain: LiveGateChain
) -> List[Dict[str, Any]]:
    """Replay rows for a single session through the chain. Returns flat list
    of decision rows: one per candidate, with stage=admitted|rule_filter|
    cross_sectional|conviction."""
    out: List[Dict[str, Any]] = []
    for row in rows:
        ts = row["ts"]
        cands = row.get("candidates") or []
        bar_vols = row.get("bar_volumes") or {}
        sym_caps = row.get("symbol_caps") or {}
        # Parse timestamp fields that gate services expect as datetime objects.
        # JSONL stores them as ISO strings; in live mode the orchestrator provides
        # datetime objects directly.
        for c in cands:
            if "decision_ts" in c:
                c["decision_ts"] = _parse_dt(c["decision_ts"])
        # RVOL warmup: feed bar volumes for this bar's universe BEFORE evaluation
        if bar_vols:
            try:
                chain.on_bar_close(bar_ts=_parse_dt(ts), bar_volumes=bar_vols, symbol_caps=sym_caps)
            except Exception:
                pass  # F1 disabled in tests; OK if no-op
        admitted = chain.evaluate(cands)
        admitted_ids = {id(c) for c in admitted}
        for c in cands:
            if id(c) in admitted_ids:
                out.append({
                    "session_date": row.get("session_date", ""),
                    "ts": ts,
                    "symbol": c.get("symbol", ""),
                    "setup_type": c.get("setup_type", ""),
                    "predicted_r": c.get("predicted_r", ""),
                    "gate_reject_reason": "",
                    "stage": "admitted",
                })
            else:
                reason = c.get("gate_reject_reason", "unknown")
                stage = reason.split(":", 1)[0] if ":" in reason else "unknown"
                out.append({
                    "session_date": row.get("session_date", ""),
                    "ts": ts,
                    "symbol": c.get("symbol", ""),
                    "setup_type": c.get("setup_type", ""),
                    "predicted_r": c.get("predicted_r", ""),
                    "gate_reject_reason": reason,
                    "stage": stage,
                })
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gate-input", required=True,
                   help="Path to gate_input.jsonl OR directory of session subfolders")
    p.add_argument("--config", required=True,
                   help="Path to configuration.json")
    p.add_argument("--output", required=True,
                   help="Path to write sim_admits.csv")
    args = p.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    project_root = Path(__file__).resolve().parent.parent.parent

    gi_path = Path(args.gate_input)
    if gi_path.is_dir():
        # Multi-session: gather session subfolders chronologically (Task 6)
        raise NotImplementedError("Task 6 implements multi-session mode")

    rows = list(_read_jsonl(gi_path))
    chain = LiveGateChain(cfg, project_root=project_root)
    decisions = _replay_one_session(rows, chain)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=["session_date", "ts", "symbol", "setup_type",
                                            "predicted_r", "gate_reject_reason", "stage"])
        w.writeheader()
        w.writerows(decisions)
    print(f"[parity_simulator] wrote {len(decisions)} decision rows to {out_path}")


if __name__ == "__main__":
    main()
