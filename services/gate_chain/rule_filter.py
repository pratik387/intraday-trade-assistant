"""Rule filter — Stage 5b live equivalent.

Loads the validation-gate-surviving rules from JSON at startup. Each rule
specifies (setup_type, conditioner_keys, conditioner_values) constraints. A
candidate is admitted iff its tuple matches at least one rule.

The 74 surviving rules are the trained-distribution input to the conviction
model. Trades not matching any rule are setups the conviction model has
never seen scored data for — drop them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Set, Tuple


class RuleFilterGate:
    """Stateless filter on (setup_type, conditioner_keys, conditioner_values)."""

    def __init__(self, survivors_json_path: Path):
        self._survivor_set: Set[Tuple] = self._load(Path(survivors_json_path))

    @staticmethod
    def _load(path: Path) -> Set[Tuple]:
        data = json.loads(path.read_text(encoding="utf-8"))
        rules: Set[Tuple] = set()
        for s in data["survivors"]:
            rule_id = s["rule_id"]
            setup, cond_part = rule_id.split("__", 1)
            cond_key_part, cond_val_part = cond_part.split("=", 1)
            keys = tuple(cond_key_part.split("+"))
            vals = tuple(cond_val_part.split("+"))
            rules.add((setup, keys, vals))
        return rules

    def evaluate(self, cand: Dict[str, Any]) -> Tuple[bool, str]:
        """Return (allow, reason).

        cand must contain `setup_type` and the conditioner keys referenced by
        the survivor rules (cap_segment, regime, hour_bucket).
        """
        setup = cand.get("setup_type")
        for r_setup, r_keys, r_vals in self._survivor_set:
            if setup != r_setup:
                continue
            if all(cand.get(k) == v for k, v in zip(r_keys, r_vals)):
                return True, "matched_rule"
        return False, "no_matching_survivor_rule"
