"""Shared cell + parameter sweep utility for Phase 5 (setup_lifecycle.md Stage 5).

Replaces the ad-hoc per-setup scripts in tools/sub9_research/*_sweep_cellmine.py
that each re-implemented the same pattern with slightly different bugs.

The helper takes a candidates DataFrame whose rows represent SIGNAL-level events
with bars-walk metadata already pre-computed (mfe_r, mae_r, close_at_<HHMM>
plus mode-specific extras), sweeps a parameter grid that includes targets,
partial-booking mode, and time-stop, mines filter cells, and locks the winning
joint (filter_cell × grid_entry) into a JSON contract.

# Target modes (audit 2026-05-20)

A survey of 25 production-relevant sanity/sweep scripts found 3 target paradigms:

  - R-multiple: 14/25 (56%). T1/T2 expressed as R-multiples; SL=1R.
  - Hybrid structural+R: 6/25 (24%). T1/T2 are fixed prices per row
    (PDC, t1_open, etc.); SL is a derived structural+ATR distance from
    which R is computed.
  - Percentage: 2/25 (8%). T1/T2/SL all in % of entry.
  - Pure ATR-based targets: 0 (gap_fade uses ATR for SL only; targets remain
    structural or R-derived).

This module unifies all three modes by converting per-row to R-units at the
top of simulate_exit. The exit semantics (resolution order, same-bar pessimism,
fee model) are identical across modes.

# Partial-booking modes

Sweep parameter `partial_mode` ∈ {"all_in", "partial_50_no_trail",
"partial_50_be_trail"} reflects what production setups actually do:

  - all_in: full qty, exit at first of {SL, T2, time-stop}
  - partial_50_no_trail: 50% at T1, 50% to T2/time-stop. SL stays at -1R for
    both legs until first hit.
  - partial_50_be_trail: 50% at T1; after T1 hit, SL moves to entry (BE) for
    the remaining 50%. CONSERVATIVE APPROXIMATION: with only summary MFE/MAE
    columns (no pre/post-T1 split), we assume BE was tagged whenever
    `mfe_r >= T1_R AND mae_r >= 1.0`. This overcounts BE exits (some of those
    paths may have stopped pre-T1 instead) but is structurally safe — it
    under-reports PF on BE-trail combos, not over-reports.

# Anti-bias guards enforced here

  - Lookahead dim block: filter dims with names `day_high`, `day_low`,
    `day_vwap`, `day_close`, `close_off_high*`, `EOD_*`, `eod_*`,
    `session_close_*` are rejected by validate_candidates_schema. The earlier
    over-broad `day_` prefix rule (v1) incorrectly rejected legitimate columns
    like `day_gain_bucket` where the underlying field was computed from
    `session_high_so_far` at signal time. v2 uses a precise blocklist.
  - Same-bar pessimism: simulate_exit picks SL when mae_r >= 1.0 even if
    mfe_r >= T2_R (Lesson #5 failure mode #4).
  - Side-aware PnL: SHORT uses -(exit - entry); LONG uses +(exit - entry).
  - Post-hoc lock blocked: lock_cell refuses overwrite without force=True.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.sub7_validation.build_per_setup_pnl import calc_fee


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_SIDES = ("LONG", "SHORT")
ALLOWED_TARGET_UNITS = ("R", "pct", "structural")
ALLOWED_PARTIAL_MODES = ("all_in", "partial_50_no_trail", "partial_50_be_trail")

# Precise blocklist for look-ahead-prone columns (Lesson #5 failure mode #1).
# v2: more specific than v1 — allows legitimate columns like day_gain_bucket
# whose underlying value can be computed from session_high_so_far at signal.
FORBIDDEN_DIM_EXACT = frozenset({
    "day_high", "day_low", "day_vwap", "day_close",
    "day_volume", "day_range", "day_atr",
})
FORBIDDEN_DIM_PREFIXES = (
    "close_off_high",   # circuit_release_fade removed this 2026-05-16 (uses EOD close)
    "EOD_", "eod_",
    "session_close_",   # session_close is end-of-session, not known at signal
)

# Required candidate columns per target_unit mode
_REQ_COLS_BY_UNIT: Dict[str, Tuple[str, ...]] = {
    "R": ("entry_ts", "entry_price", "qty", "mfe_r", "mae_r", "R_per_share"),
    "pct": ("entry_ts", "entry_price", "qty", "mfe_pct", "mae_pct"),
    "structural": ("entry_ts", "entry_price", "qty", "mfe_r", "mae_r",
                   "R_per_share", "t1_price", "t2_price"),
}


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SchemaIssue:
    severity: str          # "error" | "warn"
    code: str
    message: str


@dataclass(frozen=True)
class SchemaValidation:
    is_valid: bool
    issues: Tuple[SchemaIssue, ...]


def _dim_is_lookahead(dim: str) -> bool:
    if dim in FORBIDDEN_DIM_EXACT:
        return True
    return any(dim.startswith(p) for p in FORBIDDEN_DIM_PREFIXES)


def validate_candidates_schema(
    df: pd.DataFrame,
    *,
    target_unit: str,
    dim_pool: Sequence[str],
    ts_hhmms: Sequence[int],
) -> SchemaValidation:
    """Check candidates df against the schema contract for a given target_unit.

    Catches recurring sanity-script bugs BEFORE any sweep work happens.
    """
    issues: List[SchemaIssue] = []

    if target_unit not in ALLOWED_TARGET_UNITS:
        issues.append(SchemaIssue(
            "error", "bad_target_unit",
            f"target_unit must be one of {ALLOWED_TARGET_UNITS}; got {target_unit!r}",
        ))
        return SchemaValidation(is_valid=False, issues=tuple(issues))

    required = _REQ_COLS_BY_UNIT[target_unit]
    missing = [c for c in required if c not in df.columns]
    if missing:
        issues.append(SchemaIssue(
            "error", "missing_required",
            f"target_unit={target_unit!r} requires columns {list(required)}; "
            f"missing: {missing}",
        ))

    for dim in dim_pool:
        if _dim_is_lookahead(dim):
            issues.append(SchemaIssue(
                "error", "lookahead_dim",
                f"dim '{dim}' is look-ahead (Lesson #5 FM #1). Forbidden exacts: "
                f"{sorted(FORBIDDEN_DIM_EXACT)}; forbidden prefixes: "
                f"{FORBIDDEN_DIM_PREFIXES}",
            ))
        if dim not in df.columns:
            issues.append(SchemaIssue(
                "error", "dim_missing",
                f"dim '{dim}' declared in dim_pool but not in candidates df",
            ))

    for ts in sorted(set(ts_hhmms)):
        col = f"close_at_{int(ts)}"
        if col not in df.columns:
            issues.append(SchemaIssue(
                "error", "close_col_missing",
                f"r_grid uses time-stop {ts} but column '{col}' missing",
            ))

    # Sign-convention checks
    if target_unit in ("R", "structural"):
        if "mfe_r" in df.columns and (df["mfe_r"].dropna() < 0).any():
            issues.append(SchemaIssue(
                "error", "negative_mfe",
                "mfe_r must be UNSIGNED favorable excursion (>= 0)",
            ))
        if "mae_r" in df.columns and (df["mae_r"].dropna() < 0).any():
            issues.append(SchemaIssue(
                "error", "negative_mae",
                "mae_r must be UNSIGNED adverse excursion (>= 0)",
            ))
    if target_unit == "pct":
        if "mfe_pct" in df.columns and (df["mfe_pct"].dropna() < 0).any():
            issues.append(SchemaIssue(
                "error", "negative_mfe",
                "mfe_pct must be UNSIGNED favorable excursion in % (>= 0)",
            ))
        if "mae_pct" in df.columns and (df["mae_pct"].dropna() < 0).any():
            issues.append(SchemaIssue(
                "error", "negative_mae",
                "mae_pct must be UNSIGNED adverse excursion in % (>= 0)",
            ))

    is_valid = not any(i.severity == "error" for i in issues)
    return SchemaValidation(is_valid=is_valid, issues=tuple(issues))


# ---------------------------------------------------------------------------
# Grid entries
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GridEntry:
    """One parameter combination to sweep.

    Fields used depend on target_unit (see ``CellSweepConfig.target_unit``):
      - R mode:          t1, t2 are R-multiples; sl defaults to 1.0R.
      - pct mode:        t1, t2 are favorable %; sl is the stop % (positive).
      - structural mode: t1/t2/sl are None — targets come from per-row
                         t1_price/t2_price columns; SL distance is derived
                         from R_per_share.

    ts_hhmm and partial_mode are swept across all modes.
    """
    label: str
    ts_hhmm: int
    partial_mode: str = "partial_50_no_trail"
    t1: Optional[float] = None
    t2: Optional[float] = None
    sl: Optional[float] = None    # 1.0 for R; required for pct; None for structural

    def __post_init__(self):
        if self.partial_mode not in ALLOWED_PARTIAL_MODES:
            raise ValueError(
                f"partial_mode must be one of {ALLOWED_PARTIAL_MODES}; "
                f"got {self.partial_mode!r}"
            )


# ---------------------------------------------------------------------------
# Per-trade exit simulation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExitOutcome:
    exit_price: float
    exit_reason: str       # "sl" | "t2_full" | "t1_partial" | "t1_be_trail" | "time_stop"
    net_pnl_inr: float


def _to_r_units(
    *,
    target_unit: str,
    grid: GridEntry,
    entry_price: float,
    R_per_share: Optional[float],
    t1_price: Optional[float],
    t2_price: Optional[float],
    mfe_in_unit: float,
    mae_in_unit: float,
    side_sign: float,    # +1 LONG, -1 SHORT
) -> Tuple[float, float, float, float, float]:
    """Convert per-row + grid params to R-unit equivalents.

    Returns (T1_R, T2_R, SL_R, mfe_R, mae_R). All non-negative.
    For pct mode: R-equivalent = pct / sl_pct.
    For structural: R-equivalent T1/T2 = |t_price - entry| / R_per_share.
    """
    if target_unit == "R":
        return float(grid.t1), float(grid.t2), float(grid.sl or 1.0), mfe_in_unit, mae_in_unit

    if target_unit == "pct":
        sl_pct = float(grid.sl)
        if sl_pct <= 0:
            raise ValueError("pct mode: grid.sl (stop %) must be > 0")
        t1_R = float(grid.t1) / sl_pct
        t2_R = float(grid.t2) / sl_pct
        mfe_R = mfe_in_unit / sl_pct
        mae_R = mae_in_unit / sl_pct
        return t1_R, t2_R, 1.0, mfe_R, mae_R

    if target_unit == "structural":
        if R_per_share is None or R_per_share <= 0:
            raise ValueError("structural mode requires R_per_share > 0")
        if t1_price is None or t2_price is None:
            raise ValueError("structural mode requires t1_price + t2_price columns")
        # Favorable direction: LONG = t1>entry, SHORT = t1<entry
        t1_R = (t1_price - entry_price) * side_sign / R_per_share
        t2_R = (t2_price - entry_price) * side_sign / R_per_share
        # Reject negative — structural target on the wrong side of entry
        if t1_R <= 0 or t2_R <= 0:
            raise ValueError(
                f"structural t1/t2 are on adverse side of entry "
                f"(t1_R={t1_R}, t2_R={t2_R}); check side or price columns"
            )
        return t1_R, t2_R, 1.0, mfe_in_unit, mae_in_unit

    raise ValueError(f"unknown target_unit: {target_unit}")


def simulate_exit(
    *,
    target_unit: str,
    side: str,
    grid: GridEntry,
    entry_price: float,
    qty: int,
    close_at_ts: float,
    mfe: float,
    mae: float,
    R_per_share: Optional[float] = None,
    t1_price: Optional[float] = None,
    t2_price: Optional[float] = None,
    fee_fn: Callable = calc_fee,
) -> Optional[ExitOutcome]:
    """Compute net PnL for one candidate under a given grid entry.

    Resolution order (deterministic, conservative):

      1. If `mae_R >= 1.0`: stop fires at -1R. (Failure mode #4: same-bar
         pessimism — stop wins when both stop and target hit on one bar.)
      2. Else if `mfe_R >= T2_R`: depends on partial_mode:
         - all_in: full qty exits at T2
         - partial_50_*: 50% at T1, 50% at T2
      3. Else if `mfe_R >= T1_R`: depends on partial_mode:
         - all_in: full qty exits at time-stop close (T1 alone doesn't fire
           in all_in mode — only T2 matters)
         - partial_50_no_trail: 50% at T1, 50% at time-stop close
         - partial_50_be_trail: 50% at T1; remaining 50% — see BE-trail rule
      4. Else: full qty at time-stop close.

    BE-trail rule (partial_50_be_trail, conservative approximation):
      If T1 hit AND mae_R >= 1.0, assume the post-T1 path retraced to entry
      and the trail caught it. This OVER-counts BE exits when stop was
      actually hit BEFORE T1 (but in that path, rule 1 already fired). After
      rule 1, mae_R < 1.0 by construction, so this rule is structurally
      safe for the all-paths interpretation.

      Actually with the resolution-order ordering: by the time we're here,
      mae_R < 1.0 strictly, so this rule reduces to: if T1 hit and mae was
      anywhere close to 1R, conservatively assume BE trailed in. Use
      threshold `mae_R >= 0.75` to flag likely retrace post-T1. Documented
      as a CONSERVATIVE approximation.

    Returns ExitOutcome or None on unusable inputs.
    """
    if side not in ALLOWED_SIDES:
        raise ValueError(f"side must be LONG/SHORT, got {side!r}")
    if pd.isna(entry_price) or pd.isna(close_at_ts):
        return None
    if qty <= 0:
        return None

    side_sign = 1.0 if side == "LONG" else -1.0

    try:
        T1_R, T2_R, SL_R, mfe_R, mae_R = _to_r_units(
            target_unit=target_unit, grid=grid,
            entry_price=entry_price, R_per_share=R_per_share,
            t1_price=t1_price, t2_price=t2_price,
            mfe_in_unit=mfe, mae_in_unit=mae, side_sign=side_sign,
        )
    except ValueError:
        return None

    # For exit price computation we need an R-Rs conversion. In structural
    # mode we use absolute prices directly when available.
    if target_unit == "structural" and t1_price is not None and t2_price is not None:
        t1_exit = float(t1_price)
        t2_exit = float(t2_price)
        sl_exit = entry_price - side_sign * (R_per_share or 0.0)
    elif target_unit == "R":
        rps = float(R_per_share or 0.0)
        if rps <= 0:
            return None
        t1_exit = entry_price + side_sign * T1_R * rps
        t2_exit = entry_price + side_sign * T2_R * rps
        sl_exit = entry_price - side_sign * rps
    else:  # pct
        sl_pct = float(grid.sl or 0.0)
        t1_pct = float(grid.t1 or 0.0)
        t2_pct = float(grid.t2 or 0.0)
        if sl_pct <= 0:
            return None
        t1_exit = entry_price * (1.0 + side_sign * t1_pct / 100.0)
        t2_exit = entry_price * (1.0 + side_sign * t2_pct / 100.0)
        sl_exit = entry_price * (1.0 - side_sign * sl_pct / 100.0)

    fee_side = "BUY" if side == "LONG" else "SELL"

    def _pnl(exit_price: float, q: int) -> float:
        return side_sign * (exit_price - entry_price) * q

    def _fee(exit_price: float, q: int) -> float:
        return fee_fn(entry_price, exit_price, q, fee_side)

    # Rule 1 — same-bar SL pessimism
    if mae_R >= 1.0:
        return ExitOutcome(sl_exit, "sl", _pnl(sl_exit, qty) - _fee(sl_exit, qty))

    partial_mode = grid.partial_mode
    is_partial = partial_mode != "all_in"
    partial_q = max(int(qty * 0.5), 1) if is_partial else qty
    remain_q = qty - partial_q if is_partial else 0

    # Rule 2 — T2 hit
    if mfe_R >= T2_R:
        if is_partial:
            gross = _pnl(t1_exit, partial_q) + _pnl(t2_exit, remain_q)
            fee = _fee(t1_exit, partial_q) + _fee(t2_exit, remain_q)
            return ExitOutcome(t2_exit, "t2_full", gross - fee)
        gross = _pnl(t2_exit, qty)
        return ExitOutcome(t2_exit, "t2_full", gross - _fee(t2_exit, qty))

    # Rule 3 — T1 hit, T2 not hit
    if mfe_R >= T1_R:
        if partial_mode == "all_in":
            # T1 doesn't trigger anything in all_in mode; full qty -> TS close.
            gross = _pnl(close_at_ts, qty)
            return ExitOutcome(close_at_ts, "time_stop", gross - _fee(close_at_ts, qty))

        if partial_mode == "partial_50_be_trail" and mae_R >= 0.75:
            # CONSERVATIVE BE trail: T1 partial booked, remaining 50% exits at
            # entry (breakeven) — assume post-T1 retrace hit BE trail.
            gross = _pnl(t1_exit, partial_q) + _pnl(entry_price, remain_q)
            fee = _fee(t1_exit, partial_q) + _fee(entry_price, remain_q)
            return ExitOutcome(entry_price, "t1_be_trail", gross - fee)

        # partial_50_no_trail OR (be_trail with mae_R < 0.75): T1 partial,
        # remainder exits at TS close.
        gross = _pnl(t1_exit, partial_q) + _pnl(close_at_ts, remain_q)
        fee = _fee(t1_exit, partial_q) + _fee(close_at_ts, remain_q)
        return ExitOutcome(close_at_ts, "t1_partial", gross - fee)

    # Rule 4 — nothing hit, full qty exits at TS close
    gross = _pnl(close_at_ts, qty)
    return ExitOutcome(close_at_ts, "time_stop", gross - _fee(close_at_ts, qty))


# ---------------------------------------------------------------------------
# Cell sweep
# ---------------------------------------------------------------------------

@dataclass
class CellSweepConfig:
    side: str                                            # LONG | SHORT
    target_unit: str                                     # R | pct | structural
    grid: List[GridEntry]
    dim_pool: List[str]
    k_max: int = 2
    n_min_floor: int = 100
    pf_min_floor: float = 1.10
    n_min_ship: int = 200
    pf_min_ship: float = 1.30

    def __post_init__(self):
        if self.side not in ALLOWED_SIDES:
            raise ValueError(f"side must be LONG/SHORT, got {self.side}")
        if self.target_unit not in ALLOWED_TARGET_UNITS:
            raise ValueError(
                f"target_unit must be {ALLOWED_TARGET_UNITS}; got {self.target_unit}")
        if not self.grid:
            raise ValueError("grid must contain at least one GridEntry")
        if self.k_max < 1 or self.k_max > 3:
            raise ValueError(f"k_max must be 1..3; got {self.k_max}")
        # Per-mode grid validation
        for ge in self.grid:
            if self.target_unit == "R":
                if ge.t1 is None or ge.t2 is None:
                    raise ValueError(
                        f"R mode requires GridEntry.t1 and t2 (R-multiples); got {ge}")
            elif self.target_unit == "pct":
                if ge.t1 is None or ge.t2 is None or ge.sl is None:
                    raise ValueError(
                        f"pct mode requires GridEntry.t1, t2, and sl (all in %); got {ge}")
            # structural mode: t1/t2/sl come from row columns; grid only carries
            # ts_hhmm + partial_mode + label.


@dataclass(frozen=True)
class CellResult:
    dims: Tuple[str, ...]
    cell_label: str
    grid_label: str
    ts_hhmm: int
    partial_mode: str
    t1: Optional[float]
    t2: Optional[float]
    sl: Optional[float]
    n: int
    pf: float
    wr_pct: float
    net_pnl_inr: float
    expectancy_inr: float

    def to_dict(self) -> dict:
        return {**asdict(self), "dims": list(self.dims)}


def _profit_factor(pnls: pd.Series) -> float:
    g = float(pnls[pnls > 0].sum())
    l = float(-pnls[pnls < 0].sum())
    if l <= 0:
        return float("inf") if g > 0 else 1.0
    return g / l


def _row_simulate(row: pd.Series, cfg: CellSweepConfig, grid: GridEntry,
                   close_col: str, fee_fn: Callable) -> float:
    if cfg.target_unit == "R":
        mfe = float(row["mfe_r"]); mae = float(row["mae_r"])
        rps = float(row["R_per_share"])
        out = simulate_exit(
            target_unit="R", side=cfg.side, grid=grid,
            entry_price=float(row["entry_price"]), qty=int(row["qty"]),
            close_at_ts=float(row[close_col]),
            mfe=mfe, mae=mae, R_per_share=rps, fee_fn=fee_fn,
        )
    elif cfg.target_unit == "pct":
        out = simulate_exit(
            target_unit="pct", side=cfg.side, grid=grid,
            entry_price=float(row["entry_price"]), qty=int(row["qty"]),
            close_at_ts=float(row[close_col]),
            mfe=float(row["mfe_pct"]), mae=float(row["mae_pct"]),
            fee_fn=fee_fn,
        )
    else:  # structural
        out = simulate_exit(
            target_unit="structural", side=cfg.side, grid=grid,
            entry_price=float(row["entry_price"]), qty=int(row["qty"]),
            close_at_ts=float(row[close_col]),
            mfe=float(row["mfe_r"]), mae=float(row["mae_r"]),
            R_per_share=float(row["R_per_share"]),
            t1_price=float(row["t1_price"]), t2_price=float(row["t2_price"]),
            fee_fn=fee_fn,
        )
    return float("nan") if out is None else out.net_pnl_inr


def run_cell_sweep(
    candidates_df: pd.DataFrame,
    cfg: CellSweepConfig,
    *,
    fee_fn: Callable = calc_fee,
) -> pd.DataFrame:
    """Sweep (grid × filter cells) on the candidates df.

    Returns a DataFrame of CellResult rows sorted by PF desc then n desc.
    Raises ValueError if schema validation fails — refuses to score against
    a malformed candidates df.
    """
    ts_hhmms = [g.ts_hhmm for g in cfg.grid]
    validation = validate_candidates_schema(
        candidates_df, target_unit=cfg.target_unit,
        dim_pool=cfg.dim_pool, ts_hhmms=ts_hhmms,
    )
    if not validation.is_valid:
        errs = "\n".join(f"  [{i.severity}] {i.code}: {i.message}"
                         for i in validation.issues)
        raise ValueError(f"candidates_df failed schema validation:\n{errs}")

    df = candidates_df.copy()
    rows: List[CellResult] = []

    for grid_entry in cfg.grid:
        close_col = f"close_at_{int(grid_entry.ts_hhmm)}"

        df["_hyp_pnl"] = df.apply(
            lambda r: _row_simulate(r, cfg, grid_entry, close_col, fee_fn),
            axis=1,
        )
        usable = df.dropna(subset=["_hyp_pnl"])
        if usable.empty:
            continue

        for k in range(1, cfg.k_max + 1):
            for combo in combinations(cfg.dim_pool, k):
                sub = usable[list(combo) + ["_hyp_pnl"]].dropna()
                if sub.empty:
                    continue
                agg = sub.groupby(list(combo), observed=True)["_hyp_pnl"].agg(
                    n="count", net="sum",
                    pf=_profit_factor,
                    wr=lambda s: 100.0 * (s > 0).mean(),
                ).reset_index()
                for _, r in agg.iterrows():
                    n_val = int(r["n"])
                    pf_val = float(r["pf"])
                    if n_val < cfg.n_min_floor or pf_val < cfg.pf_min_floor:
                        continue
                    cell_label = " | ".join(f"{c}={r[c]}" for c in combo)
                    rows.append(CellResult(
                        dims=tuple(combo),
                        cell_label=cell_label,
                        grid_label=grid_entry.label,
                        ts_hhmm=int(grid_entry.ts_hhmm),
                        partial_mode=grid_entry.partial_mode,
                        t1=grid_entry.t1, t2=grid_entry.t2, sl=grid_entry.sl,
                        n=n_val, pf=pf_val,
                        wr_pct=float(r["wr"]),
                        net_pnl_inr=float(r["net"]),
                        expectancy_inr=float(r["net"]) / n_val,
                    ))

    if not rows:
        return pd.DataFrame(columns=[
            "dims", "cell_label", "grid_label", "ts_hhmm", "partial_mode",
            "t1", "t2", "sl", "n", "pf", "wr_pct", "net_pnl_inr", "expectancy_inr",
        ])
    out = pd.DataFrame([r.to_dict() for r in rows])
    out = out.sort_values(["pf", "n"], ascending=[False, False]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Selection + lock
# ---------------------------------------------------------------------------

def select_best_cell(
    sweep_results: pd.DataFrame,
    cfg: CellSweepConfig,
    *,
    require_ship_eligible: bool = True,
) -> Optional[dict]:
    """Pick best cell from sweep results. None = kill signal."""
    if sweep_results.empty:
        return None
    if require_ship_eligible:
        eligible = sweep_results[
            (sweep_results["n"] >= cfg.n_min_ship) &
            (sweep_results["pf"] >= cfg.pf_min_ship)
        ]
    else:
        eligible = sweep_results
    if eligible.empty:
        return None
    return eligible.iloc[0].to_dict()


def lock_cell(
    selected: dict,
    *,
    setup_name: str,
    window_label: str,
    output_path: Path,
    force: bool = False,
    extra_metadata: Optional[dict] = None,
) -> Path:
    """Write locked cell JSON. Refuses overwrite without force=True."""
    if selected is None:
        raise ValueError("selected is None — no cell to lock (likely a kill signal)")
    output_path = Path(output_path)
    if output_path.exists() and not force:
        raise FileExistsError(
            f"lock already exists at {output_path}; pass force=True to overwrite "
            f"(this means you accept that you are RE-SELECTING after seeing data)"
        )
    payload = {
        "setup_name": setup_name,
        "window_label": window_label,
        "locked_at": datetime.utcnow().isoformat() + "Z",
        "selected_cell": selected,
        "metadata": extra_metadata or {},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return output_path
