"""Universe loader — thin wrapper over nse_all.json scaffolding."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Set


_REPO = Path(__file__).resolve().parents[2]


def load_nse_all() -> Dict[str, dict]:
    """Symbol → metadata dict from nse_all.json."""
    path = _REPO / "nse_all.json"
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[str, dict] = {}
    for row in raw:
        sym = str(row.get("symbol", ""))
        if sym.endswith(".NS"):
            sym = sym[:-3]
        if not sym:
            continue
        out[sym] = {
            "mis_leverage": float(row.get("mis_leverage", 0.0) or 0.0),
            "mis_enabled": bool(row.get("mis_enabled", False)),
            "cap_segment": str(row.get("cap_segment", "unknown")),
        }
    return out


def mis_eligible_universe(meta: Dict[str, dict]) -> Set[str]:
    return {
        sym for sym, m in meta.items()
        if m.get("mis_enabled") and m.get("mis_leverage", 0.0) >= 1.0
        and m.get("cap_segment") in {"small_cap", "mid_cap", "large_cap"}
    }
