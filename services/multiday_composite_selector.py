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
        # No silent default: an unknown/missing mode must fail at construction.
        self.slot_ranking_mode = str(config["slot_ranking_mode"])
        if self.slot_ranking_mode not in ("unbiased_hash", "composite"):
            raise ValueError(
                f"unknown slot_ranking_mode {self.slot_ranking_mode!r} "
                "(expected 'unbiased_hash' or 'composite')"
            )

    def _rank(self, rows, session_date):
        """Order the deduped candidates for slot allocation.

        mode == 'unbiased_hash' (production since 2026-08-12):
            Deterministic date-salted hash — NO view on which candidate is
            better. `composite` is still computed and returned so it can be
            scored against this baseline, it just does not decide anything.

            Why: composite exists to promote CONSENSUS names (it sums
            weight * cap_score across contributing setups). Measured on 121
            deduped book positions, consensus does not predict — 1 contributor
            -0.049%, 2 contributors -1.033%, 3 contributors +1.204% (n=8);
            pooled consensus -0.448% vs solo -0.049%, t=-0.52, permutation
            p=0.69. Non-monotonic and statistically nothing.

            This is the same failure shape as the overnight book's 'conviction'
            ranker, which was validated on Disc/OOS/Holdout and then measured
            ANTI-predictive forward at p=0.0001 (0th percentile of 3,000 draws)
            and had to be replaced by exactly this hash ordering.

            Ordering became load-bearing on 2026-08-12: with cluster caps in
            place, 40 of 121 historical positions get dropped by a cap, so the
            order now decides which trades the book actually gets. Previously
            caps never bound and the ordering was inert.

        mode == 'composite': the legacy consensus ordering, kept for research
            comparison against the random baseline.
        """
        if self.slot_ranking_mode == "composite":
            return sorted(rows, key=lambda a: (-a["composite"], -a["tshock"], a["bare"]))
        import hashlib
        salt = session_date.isoformat() if hasattr(session_date, "isoformat") else str(session_date)
        return sorted(
            rows,
            key=lambda a: hashlib.sha1(("%s|%s" % (salt, a["bare"])).encode("utf-8")).hexdigest(),
        )

    def select(
        self,
        baskets: Dict[str, List[Dict[str, Any]]],
        held_symbols: Set[str],
        weights: Dict[str, float],
        limit: int,
        session_date=None,
    ) -> List[Dict[str, Any]]:
        """Return the deduped, RANKED basket (≤ `limit` rows).

        `session_date` salts the unbiased-hash order: reproducible within a
        session (idempotent re-runs pick the same names) and reshuffled across
        days (no symbol is permanently favoured). Required when
        slot_ranking_mode == 'unbiased_hash'.

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

        if self.slot_ranking_mode == "unbiased_hash" and session_date is None:
            raise ValueError("select(session_date=...) is required for unbiased_hash ordering")
        rows = self._rank(agg.values(), session_date)
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
