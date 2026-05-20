"""Per-regime decomposition of strategy trades.

Bucket per-trade outcomes by Indian-market regime (as defined in
assets/regime_schema.yaml), then compute BCa CI per regime.

Per Lopez de Prado *Tactical Investment Algorithms* (2019):
"A given investment algorithm should NOT be deployed throughout all market
regimes... identifying investment algorithms that are optimal for specific
market regimes."

This module enables that paradigm: if a setup works in 5 of 7 regimes but
fails in 2, the researcher can decide whether to regime-gate it or retire it.
The framework does NOT make that decision — it produces the data.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import yaml

from tools.methodology.confidence.bootstrap_ci import (
    bootstrap_ci, stat_pf, stat_expectancy, stat_win_rate, CIResult,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_SCHEMA = _REPO_ROOT / "assets" / "regime_schema.yaml"


@dataclass(frozen=True)
class Regime:
    id: str
    name: str
    start: date
    end: date
    days: str
    evidence_class: str
    mechanism: str


@dataclass(frozen=True)
class RegimeStats:
    regime: Regime
    n_trades: int
    net_pnl: float
    pf_ci: CIResult
    expectancy_ci: CIResult
    win_rate_ci: CIResult


def load_regime_schema(schema_path: Optional[Path] = None) -> List[Regime]:
    """Parse the regime schema YAML into ordered Regime objects."""
    schema_path = schema_path or _DEFAULT_SCHEMA
    with open(schema_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    regimes = []
    for r in data["regimes"]:
        regimes.append(Regime(
            id=r["id"],
            name=r["name"],
            start=r["start"] if isinstance(r["start"], date) else datetime.strptime(r["start"], "%Y-%m-%d").date(),
            end=r["end"] if isinstance(r["end"], date) else datetime.strptime(r["end"], "%Y-%m-%d").date(),
            days=str(r.get("days", "")),
            evidence_class=r["evidence_class"],
            mechanism=r["mechanism"].strip(),
        ))
    return regimes


def assign_regime(signal_date: date, regimes: List[Regime]) -> Optional[Regime]:
    """Find the regime containing signal_date. Returns None if outside all."""
    for r in regimes:
        if r.start <= signal_date <= r.end:
            return r
    return None


def compute_per_regime_stats(
    trades_df: pd.DataFrame,
    *,
    schema_path: Optional[Path] = None,
    pnl_column: str = "net_pnl_inr",
    date_column: str = "signal_date",
    n_resamples: int = 5000,
    seed: int = 20260520,
) -> List[RegimeStats]:
    """Bucket trades by regime and compute per-regime CIs.

    Args:
        trades_df: must have date_column and pnl_column
        schema_path: path to regime_schema.yaml (default: assets/regime_schema.yaml)
        pnl_column: column to use for PnL (default 'net_pnl_inr')
        date_column: column to use for regime assignment (default 'signal_date')
        n_resamples: bootstrap iterations per regime
        seed: RNG seed (incremented per regime for independent samples)

    Returns:
        List of RegimeStats in regime order. Regimes with n < 10 trades will
        have degenerate CIResults (method='insufficient_data').
    """
    regimes = load_regime_schema(schema_path)
    df = trades_df.copy()
    df[date_column] = pd.to_datetime(df[date_column]).dt.date
    df["_regime"] = df[date_column].apply(lambda d: assign_regime(d, regimes))

    results = []
    for i, r in enumerate(regimes):
        sub = df[df["_regime"] == r]
        n = len(sub)
        if n == 0:
            # No trades in this regime — emit empty stats
            empty_ci = CIResult(
                point_estimate=float("nan"), ci_lower=float("nan"),
                ci_upper=float("nan"), n=0, n_resamples=0, method="no_trades",
            )
            results.append(RegimeStats(
                regime=r, n_trades=0, net_pnl=0.0,
                pf_ci=empty_ci, expectancy_ci=empty_ci, win_rate_ci=empty_ci,
            ))
            continue

        pnls = sub[pnl_column].to_numpy()
        net = float(pnls.sum())
        # Different seed per regime for independent bootstrap samples
        regime_seed = seed + i * 100
        pf_ci = bootstrap_ci(pnls, stat_pf, n_resamples=n_resamples, seed=regime_seed)
        exp_ci = bootstrap_ci(pnls, stat_expectancy, n_resamples=n_resamples, seed=regime_seed + 1)
        wr_ci = bootstrap_ci(pnls, stat_win_rate, n_resamples=n_resamples, seed=regime_seed + 2)

        results.append(RegimeStats(
            regime=r, n_trades=n, net_pnl=net,
            pf_ci=pf_ci, expectancy_ci=exp_ci, win_rate_ci=wr_ci,
        ))
    return results


def format_regime_table(stats: List[RegimeStats]) -> str:
    """Format regime stats as a readable text table."""
    lines = [
        f"{'Regime':<35} {'n':>5} {'PF point':>10} {'PF [low, high]':>20} {'Net Rs':>12}",
        "-" * 90,
    ]
    for s in stats:
        if s.n_trades == 0:
            lines.append(f"{s.regime.name:<35} {0:>5} {'--':>10} {'--':>20} {'--':>12}")
            continue
        pf = s.pf_ci
        if pf.method == "insufficient_data":
            ci_str = "n<10 underpowered"
        else:
            ci_str = f"[{pf.ci_lower:.2f}, {pf.ci_upper:.2f}]"
        lines.append(
            f"{s.regime.name:<35} {s.n_trades:>5} "
            f"{pf.point_estimate:>10.3f} {ci_str:>20} {s.net_pnl:>+12,.0f}"
        )
    return "\n".join(lines)
