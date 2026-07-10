"""Cross-setup integrated-composite selector for the multi-day CNC/MTF family.

Pure function of (baskets, held, weights, config): blends each name's per-setup
`cap_score` into a single composite score (weighted SUM across the setups that
selected it — so multi-setup agreement raises rank), dedupes to one row per
symbol, drops names already held by any setup, and returns the top-`limit`
composite-ranked candidates. No IO, no broker, no clock (live/backtest-identical,
IST-naive by construction). NO hardcoded defaults (CLAUDE.md rule 1).

Spec: specs/2026-06-29-multiday-composite-selection-design.md
"""
from __future__ import annotations

from typing import Any, Dict, List, Set

from config.logging_config import get_agent_logger

logger = get_agent_logger()


def _bare(symbol: str) -> str:
    """Canonical bare ticker for cross-setup dedupe (strip NSE:, upper)."""
    return str(symbol).replace("NSE:", "").upper()


class MultiDayCompositeSelector:
    """Blend per-setup baskets into one deduped, consensus-ranked basket."""

    def __init__(self, config: Dict[str, Any]):
        # Fail-fast on every key (no silent defaults). max_new_per_day /
        # max_concurrent are validated here but ENFORCED BY THE CALLER, which
        # computes `limit` from them + the current book size and passes it to
        # select(); they are intentionally not re-read inside select().
        self.max_new_per_day = int(config["max_new_per_day"])
        self.max_concurrent = int(config["max_concurrent"])
        self.cap_score_clip = float(config["cap_score_clip"])
        self.tiebreaker = str(config["tiebreaker"])
        if self.tiebreaker != "tshock":
            raise ValueError(f"unsupported tiebreaker {self.tiebreaker!r} (v1: 'tshock')")

    def select(
        self,
        baskets: Dict[str, List[Dict[str, Any]]],
        held_symbols: Set[str],
        weights: Dict[str, float],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Return the deduped, composite-ranked basket (≤ `limit` rows).

        Args:
            baskets: {setup_name: [ranker cand dict, ...]} — each cand carries
                `symbol`, `cap_score`, `tshock`, `close`, `trail_ret`.
            held_symbols: bare or NSE: symbols already held by ANY setup.
            weights: {setup_name: composite_weight}.
            limit: max names to return (caller computes from combined caps).

        Returns: [{symbol(NSE:), bare, composite, tshock, owner, contributors,
            per_setup_cap_score, close, trail_ret}], composite-desc.
        """
        held = {_bare(s) for s in held_symbols}
        agg: Dict[str, Dict[str, Any]] = {}
        for setup_name, cands in baskets.items():
            w = float(weights[setup_name])
            for cand in cands:
                bare = _bare(cand["symbol"])
                if bare in held:
                    continue
                contrib = w * min(float(cand["cap_score"]), self.cap_score_clip)
                a = agg.get(bare)
                if a is None:
                    a = {
                        "bare": bare, "composite": 0.0, "tshock": 0.0,
                        "contributors": [], "per_setup_cap_score": {},
                        "_owner_weighted": -1.0, "owner": None,
                        "close": float(cand["close"]),
                        "trail_ret": float(cand["trail_ret"]),
                        "sigma20_pct": cand.get("sigma20_pct"),
                    }
                    agg[bare] = a
                a["composite"] += contrib
                a["tshock"] = max(a["tshock"], float(cand["tshock"]))
                a["contributors"].append(setup_name)
                a["per_setup_cap_score"][setup_name] = float(cand["cap_score"])
                if contrib > a["_owner_weighted"]:
                    a["_owner_weighted"] = contrib
                    a["owner"] = setup_name
                    a["close"] = float(cand["close"])
                    a["trail_ret"] = float(cand["trail_ret"])
                    a["sigma20_pct"] = cand.get("sigma20_pct")

        rows = sorted(
            agg.values(),
            key=lambda a: (-a["composite"], -a["tshock"], a["bare"]),
        )
        capped = rows[: max(0, int(limit))]
        out: List[Dict[str, Any]] = []
        for a in capped:
            a.pop("_owner_weighted", None)
            a["symbol"] = f"NSE:{a['bare']}"
            a["contributors"] = sorted(set(a["contributors"]))
            out.append(a)
        logger.info(
            "composite_selector: %d unique candidates -> %d chosen (limit=%d, %d held excluded)",
            len(agg), len(out), limit, len(held),
        )
        return out
