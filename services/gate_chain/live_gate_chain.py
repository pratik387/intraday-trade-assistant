"""Live composed gate chain — RuleFilter -> CrossSectional -> Conviction.

Owns the lifecycle of all three sub-gates and their state. Singleton per
ScreenerLive process. Inserted in pipelines/orchestrator.py after existing
ranking, before order enqueue.

When live_gate_chain.enabled=false, evaluate() is a passthrough — the chain
makes no decisions and incurs no model-load cost. This lets us ship the wiring
behind a flag and flip it on per-config.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from services.cross_sectional.crowdedness_counter import CrowdednessCounter
from services.cross_sectional.gate import Candidate as CSCand, CrossSectionalGate
from services.cross_sectional.universe_rvol import UniverseRVOLState
from services.conviction.feature_spec import extract_features
from services.conviction.gate import ConvictionGate
from services.conviction.scorer import XGBoostScorer
from services.gate_chain.dedup_gate import DedupGate
from services.gate_chain.rule_filter import RuleFilterGate

from config.logging_config import get_agent_logger
log = get_agent_logger()


class LiveGateChain:
    """Composed three-stage gate. Stateful (RVOL + crowdedness counters)."""

    def __init__(self, full_cfg: Dict[str, Any], project_root: Path):
        # Keep a reference to the *full* config so evaluate() can read top-level
        # keys like `rank_pctl_min` at bar-time (DedupGate pctl computation).
        self.full_cfg = full_cfg
        self.cfg = full_cfg.get("live_gate_chain", {})
        self.enabled = bool(self.cfg.get("enabled", False))
        # NOTE: dedup_drop inserted BEFORE admitted — admitted is kept last for
        # readability of the running chain counters.
        self._stats = {"in": 0, "rule_drop": 0, "cs_drop": 0, "conv_drop": 0, "dedup_drop": 0, "admitted": 0}

        if not self.enabled:
            log.info("LiveGateChain disabled in config — evaluate() is passthrough")
            return

        # Sub-project #5 (gauntlet v2): when wide_open_mode is set, evaluate()
        # short-circuits to passthrough (T3). Skip ALL gate dependency loads
        # (XGBoost model, survivors json, RVOL state) so OCI wide-open captures
        # don't need those artifacts in the code tarball — they're produced by
        # Sub-project #1 outputs that are gitignored and not packaged.
        if full_cfg.get("wide_open_mode"):
            log.info("LiveGateChain init skipped — wide_open_mode forces passthrough")
            return

        # [A] Rule filter
        rf_cfg = full_cfg.get("rule_filter_gate", {})
        survivors_path = Path(project_root) / rf_cfg.get(
            "survivors_path",
            "analysis/edge_discovery_runs/2026-04-22-validation-gate/stage6_validation_survivors.json",
        )
        self.rule_filter = RuleFilterGate(survivors_path)

        # [B] Cross-sectional
        cs_cfg = full_cfg["cross_sectional_gate"]
        self.rvol = UniverseRVOLState(
            rolling_sessions=int(cs_cfg["f1_rolling_window_sessions"]),
            min_sessions=int(cs_cfg["f1_min_history_sessions"]),
        )
        self.crowdedness = CrowdednessCounter(
            window_min=int(cs_cfg["f2_crowdedness_window_min"]),
        )
        self.cross_sectional = CrossSectionalGate(
            cs_cfg, rvol=self.rvol, crowdedness=self.crowdedness,
        )

        # [C] Conviction
        cv_cfg = full_cfg["conviction_gate"]
        self.scorer = XGBoostScorer(
            Path(project_root) / cv_cfg["model_artifact"],
            Path(project_root) / cv_cfg["feature_spec_path"],
        )
        self.conviction = ConvictionGate({
            "enabled": True,
            "daily_cap": int(cv_cfg["daily_cap"]),
            "min_predicted_r": float(cv_cfg["min_predicted_r"]),
        })

        # [D] Dedup — sub-project #4 gate chain stage D. Replaces the historical
        # ScreenerLive._dedupe_ok. Operates on candidates that passed conviction.
        self.dedup = DedupGate(full_cfg["dedup_gate"])

    def on_bar_close(self, bar_ts, bar_volumes: Dict[str, int], symbol_caps: Dict[str, str]) -> None:
        """Forward to UniverseRVOLState; called by ScreenerLive on each 5-min close."""
        if not self.enabled:
            return
        self.rvol.on_bar_close(ts=bar_ts, bar_volumes=bar_volumes, symbol_caps=symbol_caps)

    def evaluate(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter candidates through 3-stage chain. Returns admitted subset.

        Each candidate dict must contain at minimum: symbol, setup_type, regime,
        cap_segment, hour_bucket, decision_ts, session_date_dt, plus the feature
        keys consumed by feature_spec.extract_features.

        Stages annotate REJECTED candidates in-place with `gate_reject_reason`
        (caller can inspect for logging). Admitted candidates also get
        `predicted_r` annotated.
        """
        # Sub-project #5 (gauntlet v2): wide_open_mode is the master kill-switch.
        # When set at top-level config, force passthrough even if the chain is
        # enabled. Used for the OCI wide-open capture so gate_input.jsonl carries
        # the maximal candidate pool for offline config iteration.
        if self.full_cfg.get("wide_open_mode"):
            return candidates

        if not self.enabled:
            return candidates
        if not candidates:
            return []

        self._stats["in"] += len(candidates)

        # [A] Rule filter
        passed_a: List[Dict[str, Any]] = []
        for c in candidates:
            ok, reason = self.rule_filter.evaluate(c)
            if ok:
                passed_a.append(c)
            else:
                c["gate_reject_reason"] = f"rule_filter:{reason}"
                self._stats["rule_drop"] += 1

        # [B] Cross-sectional
        passed_b: List[Dict[str, Any]] = []
        for c in passed_a:
            cs_cand = CSCand(
                symbol=str(c["symbol"]).replace("NSE:", ""),
                setup_type=c["setup_type"],
                cap_segment=c["cap_segment"],
                hour_bucket=c["hour_bucket"],
                decision_ts=c["decision_ts"],
            )
            ok, reason = self.cross_sectional.evaluate(cs_cand)
            # F2 parity with Stage 5c: record EVERY rule-filter-surviving
            # candidate regardless of accept/reject. Without this, the
            # crowdedness sliding window stays empty in live and F2 never
            # binds — causing 250+ same-setup candidates per bar to flood
            # ConvictionGate (gauntlet sim records at stage5c line 149).
            self.crowdedness.record(c["setup_type"], c["decision_ts"])
            if ok:
                passed_b.append(c)
            else:
                c["gate_reject_reason"] = f"cross_sectional:{reason}"
                self._stats["cs_drop"] += 1

        # [C] Conviction — BATCHED scoring then per-cand gate eval
        admitted: List[Dict[str, Any]] = []
        if passed_b:
            feat_list = [extract_features(c) for c in passed_b]
            preds = self.scorer.predict_batch(feat_list)
            for c, pred in zip(passed_b, preds):
                cand_for_gate = {
                    "symbol": c["symbol"],
                    "decision_ts": c["decision_ts"],
                    "session_date": c["session_date_dt"],
                }
                ok, reason = self.conviction.evaluate(cand_for_gate, float(pred))
                c["predicted_r"] = float(pred)
                if ok:
                    admitted.append(c)
                    self._stats["admitted"] += 1
                else:
                    c["gate_reject_reason"] = f"conviction:{reason}"
                    self._stats["conv_drop"] += 1

        # [D] Dedup — final per-symbol cooloff / setup-change / score-strength
        # check. Replaces the historical screener_live._dedupe_ok. Dedup processes
        # admitted candidates in rank_score-desc order so higher-score admit wins
        # the cooloff slot when multiple same-symbol candidates pass earlier stages.
        if not admitted:
            return admitted
        # Sort admitted by rank_score desc (previously done post-chain in screener)
        admitted.sort(key=lambda c: float(c.get("rank_score", 0.0)), reverse=True)
        # Compute pctl_score from the ORIGINAL chain input pool (cands entering
        # the chain before any drops). Matches screener_live._compute_percentile_cut.
        pctl = float(self.full_cfg["rank_pctl_min"])
        _scores = [float(c.get("rank_score", 0.0)) for c in candidates]
        try:
            import pandas as pd
            bar_pctl = float(pd.Series(_scores, dtype=float).quantile(max(0.0, min(1.0, pctl))))
        except Exception:
            bar_pctl = float("-inf")
        final: List[Dict[str, Any]] = []
        for c in admitted:
            ok, reason = self.dedup.evaluate(
                sym=str(c["symbol"]).replace("NSE:", ""),
                now_ts=c["decision_ts"],
                setup_type=c.get("setup_type"),
                score=float(c.get("rank_score", 0.0)),
                pctl_score=bar_pctl,
                session_date=c.get("session_date_dt"),
            )
            if ok:
                final.append(c)
            else:
                c["gate_reject_reason"] = f"dedup:{reason}"
                self._stats["dedup_drop"] += 1
                # Correct the admitted counter — conviction bumped it on pass,
                # but dedup has now rejected. Keep "admitted" meaning "fully
                # admitted through all chain stages" in stats().
                self._stats["admitted"] -= 1
        return final

    def stats(self) -> Dict[str, int]:
        return dict(self._stats)
