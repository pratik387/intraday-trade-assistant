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
from services.gate_chain.rule_filter import RuleFilterGate

log = logging.getLogger(__name__)


class LiveGateChain:
    """Composed three-stage gate. Stateful (RVOL + crowdedness counters)."""

    def __init__(self, full_cfg: Dict[str, Any], project_root: Path):
        self.cfg = full_cfg.get("live_gate_chain", {})
        self.enabled = bool(self.cfg.get("enabled", False))
        self._stats = {"in": 0, "rule_drop": 0, "cs_drop": 0, "conv_drop": 0, "admitted": 0}

        if not self.enabled:
            log.info("LiveGateChain disabled in config — evaluate() is passthrough")
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

        return admitted

    def stats(self) -> Dict[str, int]:
        return dict(self._stats)
