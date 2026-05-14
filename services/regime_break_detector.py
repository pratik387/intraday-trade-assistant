"""Regime-break detector for the gauntlet.

Reads `data/sebi_calendar/rule_changes.csv` and checks whether a validation
window (Discovery / OOS / Holdout) spans any rule change that affects the
strategy's declared `depends_on` tags.

Why this exists
---------------
Our 2025 gauntlet validated `delivery_pct_anomaly_short` with a Holdout that
straddled SEBI's Oct 1, 2025 F&O rules (MWPL formula + position limit cuts).
The Holdout reported a confused PF 1.13 — average of (pre-rule clean PF ~0.95)
+ (war-period PF ~1.0). We shipped a setup whose mechanism was actually broken.
This detector exists so the gauntlet refuses to run if a window spans a known
high/critical rule change for any of the strategy's mechanism dependencies.

Usage
-----
    from services.regime_break_detector import check_window, GauntletRegimeBreak

    try:
        check_window(
            strategy_name="delivery_pct_anomaly_short",
            depends_on=["MWPL", "single_stock_FO_speculator_unwind"],
            window_label="Holdout",
            start=date(2025, 10, 1),
            end=date(2026, 4, 30),
        )
    except GauntletRegimeBreak as e:
        # Split the window or re-attribute. Don't proceed silently.
        print(f"REGIME BREAK: {e}")
        raise
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, List, Optional, Set

_CALENDAR = Path(__file__).resolve().parents[1] / "data" / "sebi_calendar" / "rule_changes.csv"


@dataclass(frozen=True)
class RuleChange:
    effective_date: date
    announced_date: Optional[date]
    category: str
    severity: str          # low / medium / high / critical
    affects: Set[str]      # dependency tags
    description: str
    circular_ref: str
    source_url: str


class GauntletRegimeBreak(RuntimeError):
    """Raised when a validation window contains a high/critical rule change."""


def _parse_date(s: str) -> Optional[date]:
    s = (s or "").strip()
    if not s:
        return None
    return datetime.strptime(s, "%Y-%m-%d").date()


def load_rule_changes(path: Path = _CALENDAR) -> List[RuleChange]:
    """Load + parse the CSV. Returns sorted by effective_date asc."""
    if not path.exists():
        return []
    rows: List[RuleChange] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            try:
                rows.append(RuleChange(
                    effective_date=_parse_date(raw["effective_date"]) or date.min,
                    announced_date=_parse_date(raw.get("announced_date") or ""),
                    category=(raw.get("category") or "").strip().lower(),
                    severity=(raw.get("severity") or "low").strip().lower(),
                    affects=set(
                        t.strip() for t in (raw.get("affects") or "").split(";")
                        if t.strip()
                    ),
                    description=(raw.get("description") or "").strip(),
                    circular_ref=(raw.get("circular_ref") or "").strip(),
                    source_url=(raw.get("source_url") or "").strip(),
                ))
            except (KeyError, ValueError):
                continue
    rows.sort(key=lambda r: r.effective_date)
    return rows


def find_relevant_changes(
    depends_on: Iterable[str],
    start: date,
    end: date,
    *,
    min_severity: str = "high",
    rules: Optional[List[RuleChange]] = None,
) -> List[RuleChange]:
    """Return all rule changes in [start, end] that affect any of the strategy's deps.

    `min_severity` filters out low/medium changes by default — those are
    re-tunable, not regime breaks. Pass "low" to surface all.
    """
    severity_rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    threshold = severity_rank.get(min_severity, 2)
    deps = set(depends_on)
    rows = rules if rules is not None else load_rule_changes()
    out: List[RuleChange] = []
    for r in rows:
        if not (deps & r.affects):
            continue
        if severity_rank.get(r.severity, 0) < threshold:
            continue
        if start <= r.effective_date <= end:
            out.append(r)
    return out


def check_window(
    strategy_name: str,
    depends_on: Iterable[str],
    window_label: str,
    start: date,
    end: date,
    *,
    min_severity: str = "high",
    raise_on_break: bool = True,
) -> List[RuleChange]:
    """Check a single window. Returns relevant rule-change rows; raises if any.

    Set `raise_on_break=False` to get the list without raising (useful for
    pre-flight reporting in a research notebook).
    """
    hits = find_relevant_changes(
        depends_on, start, end, min_severity=min_severity,
    )
    if hits and raise_on_break:
        msgs = [
            f"  - {r.effective_date} [{r.severity.upper()}] {r.description} "
            f"(affects {sorted(set(depends_on) & r.affects)})"
            for r in hits
        ]
        raise GauntletRegimeBreak(
            f"{strategy_name} {window_label} window {start}..{end} "
            f"contains {len(hits)} rule change(s):\n" + "\n".join(msgs)
        )
    return hits
