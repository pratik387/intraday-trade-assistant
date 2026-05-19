"""Parity gate — compares framework-reproduced statistics to live setup baselines."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class ParityTolerance:
    pf_pct: float  # +/- this % around live PF
    wr_pp: float   # +/- this percentage-points around live WR
    n_pct: float   # +/- this % around live N


@dataclass
class ParityVerdict:
    passed: bool
    failures: List[str] = field(default_factory=list)
    pf_delta_pct: float = 0.0
    wr_delta_pp: float = 0.0
    n_delta_pct: float = 0.0


def compare_parity(framework: Dict, live: Dict, tol: ParityTolerance) -> ParityVerdict:
    """Compare framework stats to live baselines under given tolerance."""
    failures: List[str] = []
    pf_live = float(live["pf"])
    pf_fw = float(framework["pf"])
    pf_delta_pct = abs(pf_fw - pf_live) / pf_live * 100.0 if pf_live > 0 else float("inf")
    if pf_delta_pct > tol.pf_pct:
        failures.append(f"pf_delta_pct={pf_delta_pct:.1f}% > tol={tol.pf_pct}%")

    wr_live = float(live["wr"])
    wr_fw = float(framework["wr"])
    wr_delta_pp = abs(wr_fw - wr_live) * 100.0
    if wr_delta_pp > tol.wr_pp:
        failures.append(f"wr_delta_pp={wr_delta_pp:.1f}pp > tol={tol.wr_pp}pp")

    n_live = float(live["n"])
    n_fw = float(framework["n"])
    n_delta_pct = abs(n_fw - n_live) / n_live * 100.0 if n_live > 0 else float("inf")
    if n_delta_pct > tol.n_pct:
        failures.append(f"n_delta_pct={n_delta_pct:.1f}% > tol={tol.n_pct}%")

    return ParityVerdict(
        passed=(len(failures) == 0),
        failures=failures,
        pf_delta_pct=pf_delta_pct,
        wr_delta_pp=wr_delta_pp,
        n_delta_pct=n_delta_pct,
    )
