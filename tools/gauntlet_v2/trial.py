"""Single-config gauntlet_v2 trial evaluator.

Sub-project #5 Step 3 / Phase 1. Reusable as library:
    run_trial(cfg_overrides, gate_input_dir, pnl_index, base_cfg) -> metrics

Also runnable as CLI for Phase 1 sanity + manual A/B tests.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from services.gate_chain.live_gate_chain import LiveGateChain
from tools.shadow.parity_simulator import _read_jsonl, _replay_one_session


def _merge_cfg(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge `overrides` into `base`. Returns a new dict; neither input mutated."""
    out = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_cfg(out[k], v)
        else:
            out[k] = v
    return out


def _collect_admits(gate_input_dir: Path, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Run parity_simulator over every <date>/gate_input.jsonl under gate_input_dir.
    Returns DataFrame of admits only (stage == "admitted")."""
    session_files = sorted(gate_input_dir.glob("*/gate_input.jsonl"))
    if not session_files:
        single = gate_input_dir / "gate_input.jsonl"
        if single.exists():
            session_files = [single]
        else:
            raise SystemExit(f"[trial] no gate_input.jsonl under {gate_input_dir}")
    chain = LiveGateChain(cfg, project_root=_REPO_ROOT)
    rows = []
    prev_session = None
    for sess_file in session_files:
        session_rows = list(_read_jsonl(sess_file))
        if not session_rows:
            continue
        this_session = session_rows[0].get("session_date")
        if prev_session is not None and this_session <= prev_session:
            raise SystemExit(f"[trial] session out of order: {this_session} after {prev_session}")
        decisions = _replay_one_session(session_rows, chain)
        for d in decisions:
            if d.get("stage") == "admitted":
                rows.append({
                    "session_date": d.get("session_date", ""),
                    "ts": d.get("ts", ""),
                    "symbol": d.get("symbol", ""),
                    "setup_type": d.get("setup_type", ""),
                })
        prev_session = this_session
    return pd.DataFrame(rows)


def _compute_metrics(labeled: pd.DataFrame) -> Dict[str, Any]:
    """labeled: DataFrame with at least total_trade_pnl, session_date per admitted row."""
    if labeled.empty:
        return {"n_trades": 0, "n_sessions": 0, "total_pnl": 0.0, "pf": 0.0,
                "sharpe": 0.0, "wr": 0.0, "trades_per_day": 0.0, "losing_days_pct": 0.0}
    pnl = labeled["total_trade_pnl"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0].abs()
    pf = float(wins.sum() / losses.sum()) if losses.sum() > 0 else float("inf")
    daily = labeled.groupby("session_date")["total_trade_pnl"].sum()
    sharpe = float(daily.mean() / daily.std()) if daily.std() > 0 else 0.0
    losing = int((daily < 0).sum())
    return {
        "n_trades": int(len(labeled)),
        "n_sessions": int(daily.size),
        "total_pnl": float(pnl.sum()),
        "pf": round(pf, 3) if pf != float("inf") else 999.0,
        "sharpe": round(sharpe, 3),
        "wr": round(float((pnl > 0).mean()), 3),
        "trades_per_day": round(len(labeled) / daily.size, 2) if daily.size else 0.0,
        "losing_days_pct": round(100 * losing / daily.size, 1) if daily.size else 0.0,
    }


def run_trial(
    cfg_overrides: Dict[str, Any],
    gate_input_dir: Path,
    pnl_index: pd.DataFrame,
    base_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate ONE config against Discovery period gate_input + cached PnL.

    Returns: {pf, sharpe, wr, trades_per_day, losing_days_pct, total_pnl,
              n_trades, n_sessions}
    """
    cfg = _merge_cfg(base_cfg, cfg_overrides)
    admits = _collect_admits(gate_input_dir, cfg)
    if admits.empty:
        return _compute_metrics(admits)

    KEY = ["session_date", "ts", "symbol", "setup_type"]
    labeled = admits.merge(
        pnl_index[KEY + ["total_trade_pnl", "r_multiple"]],
        on=KEY, how="left",
    )
    miss_rate = labeled["total_trade_pnl"].isna().mean()
    if miss_rate > 0.001:
        raise SystemExit(
            f"[trial] {miss_rate*100:.2f}% of sim admits have no PnL match. "
            f"Expected <0.1%. Check the OCI run is wide_open and pnl_index is complete."
        )
    labeled = labeled.dropna(subset=["total_trade_pnl"])
    return _compute_metrics(labeled)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cfg-overrides", default="{}", help="JSON string of config overrides")
    p.add_argument("--base-cfg", required=True)
    p.add_argument("--gate-input-dir", required=True)
    p.add_argument("--pnl-index", required=True)
    args = p.parse_args()

    base = json.loads(Path(args.base_cfg).read_text(encoding="utf-8"))
    overrides = json.loads(args.cfg_overrides)
    pnl_df = pd.read_parquet(args.pnl_index)

    metrics = run_trial(overrides, Path(args.gate_input_dir), pnl_df, base)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
