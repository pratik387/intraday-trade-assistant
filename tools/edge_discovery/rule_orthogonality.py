"""Rule-orthogonality classification + policy-date PF break detection."""
from __future__ import annotations

from datetime import date
from typing import Dict, List

import pandas as pd


# Known SEBI / STT rule-change dates. Sources: SEBI press releases, NSE circulars.
KNOWN_POLICY_DATES: List[date] = [
    date(2024, 10, 1),   # STT hike: futures 0.0125%→0.02%; options on premium 0.0625%→0.1%
    date(2025, 2, 1),    # SEBI: full option premium upfront, no leverage on long options
    date(2025, 10, 1),   # SEBI: MWPL formula tightened, single-stock position limits cut
    date(2026, 4, 1),    # Anticipated STT changes per 2025 Union Budget
]


# Tokens that mark rule-dependent edge sources.
RULE_DEPENDENT_TOKENS = (
    "stt rate", "stt differential", "stt arbitrage",
    "mis leverage cap", "mwpl formula", "position limit",
    "option premium margin", "option leverage",
    "futures basis arbitrage", "f&o margin",
)


def classify_candidate(name: str, edge_source: str, depends_on: List[str]) -> str:
    """Classify a candidate setup by rule sensitivity.

    rule_orthogonal: edge from structural microstructure (retail flow, institutional
        rebalancing, auction effects).
    rule_dependent: edge requires a specific regulatory parameter to hold.
    """
    text = edge_source.lower()
    for token in RULE_DEPENDENT_TOKENS:
        if token in text:
            return "rule_dependent"
    rule_deps = {"stt_rate", "mwpl_formula", "f_and_o_margin", "option_leverage"}
    if any(d in rule_deps for d in depends_on):
        return "rule_dependent"
    return "rule_orthogonal"


def check_against_policy_dates(
    pf_series: pd.Series,
    drop_threshold_pct: float = 50.0,
) -> List[Dict]:
    """For each known policy date, check if PF dropped by drop_threshold_pct around the date.

    Looks at average of 3 PF values before and 3 after the policy date.
    Returns list of detected breaks with date + magnitude.
    """
    breaks: List[Dict] = []
    pf_sorted = pf_series.sort_index()
    for policy_date in KNOWN_POLICY_DATES:
        before_mask = pf_sorted.index <= pd.Timestamp(policy_date)
        after_mask = pf_sorted.index > pd.Timestamp(policy_date)
        if not before_mask.any() or not after_mask.any():
            continue
        pf_before = pf_sorted.loc[before_mask].iloc[-3:].mean() if before_mask.sum() >= 1 else None
        pf_after = pf_sorted.loc[after_mask].iloc[:3].mean() if after_mask.sum() >= 1 else None
        if pf_before is None or pf_after is None or pf_before <= 0:
            continue
        drop_pct = (pf_before - pf_after) / pf_before * 100.0
        if drop_pct >= drop_threshold_pct:
            breaks.append({
                "policy_date": policy_date,
                "pf_before": float(pf_before),
                "pf_after": float(pf_after),
                "drop_pct": float(drop_pct),
            })
    return breaks
