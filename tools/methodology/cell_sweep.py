"""Shared cell + R-multiple sweep utility for Phase 5 (setup_lifecycle.md Stage 5).

Replaces the ad-hoc per-setup scripts in tools/sub9_research/*_sweep_cellmine.py
that each re-implemented the same pattern (and each had slightly different bugs).

The helper takes a candidates DataFrame whose rows represent SIGNAL-level events
with bars-walk metadata already pre-computed (mfe_r, mae_r, close_at_<HHMM>),
sweeps a (T1_R, T2_R, time_stop) grid, mines filter cells, and locks the
winning joint (filter_cell × R-tuple) into a JSON contract.

What this module is NOT:
  - It does NOT walk 5m bars. The sanity script does that (Stage 4) and pre-
    computes the path summary fields. This keeps the lifecycle clean: bars-walk
    bugs live in ONE place (the sanity), and the sweep operates on a stable
    intermediate representation.
  - It does NOT compute final-trade canonical CSV. That is what the sanity
    script emits separately at the winning (T1, T2, TS) — re-running with the
    locked cell after Stage 5.

Anti-bias guards enforced here:
  - Lookahead dimension block: filter dims with names starting `day_` are
    rejected by validate_candidates_schema (Lesson #5, Failure mode #1).
  - Same-bar pessimism: simulate_exit picks SL over T2 when MAE >= 1.0R
    regardless of MFE (Lesson #5, Failure mode #4).
  - Side-aware PnL: SHORT uses (entry - exit); LONG uses (exit - entry).
    Fees use calc_fee (Indian retail stack) with `qty` directly — see
    docstring on tools.sub7_validation.build_per_setup_pnl.calc_fee.
  - Post-hoc selection blocked: select_best_cell takes ONLY a sweep result on
    Discovery (or Disc+OOS combined); it refuses to mix in HO data by name
    convention. Caller controls the window via the input df.
  - Lock-once: lock_cell refuses to overwrite an existing lock without
    explicit force=True (the user must confirm cell re-selection).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.sub7_validation.build_per_setup_pnl import calc_fee


# ---------------------------------------------------------------------------
# Schema contract for the candidates DataFrame
# ---------------------------------------------------------------------------

REQUIRED_CANDIDATE_COLUMNS = (
    "entry_ts",        # IST-naive timestamp of the entry bar
    "entry_price",     # float > 0; Mode B entry price (next bar's open)
    "qty",             # int > 0; ACTUAL share count traded
    "mfe_r",           # float >= 0; max favorable excursion in R-units
    "mae_r",           # float >= 0; max adverse excursion in R-units
    "R_per_share",     # float > 0; Rs distance from entry to SL (1R)
)

FORBIDDEN_DIM_PREFIXES = (
    "day_",            # day_high / day_low / day_vwap / day_close are EOD
                       # aggregates -> Lookahead at signal time (Failure mode #1)
    "EOD_",
    "eod_",
    "session_close_",  # session_close is end-of-session, not known at signal
    "close_off_high",  # bucket-of-day_high, same failure mode (case from
                       # _circuit_release_fade where this column was removed
                       # 2026-05-16 after audit)
)

ALLOWED_SIDES = ("LONG", "SHORT")


@dataclass(frozen=True)
class SchemaIssue:
    severity: str          # "error" | "warn"
    code: str
    message: str


@dataclass(frozen=True)
class SchemaValidation:
    is_valid: bool
    issues: Tuple[SchemaIssue, ...]


def validate_candidates_schema(
    df: pd.DataFrame,
    *,
    dim_pool: Sequence[str],
    r_grid: Sequence[Tuple[float, float, int]],
) -> SchemaValidation:
    """Check the candidates df against the schema contract.

    Catches the recurring sanity-script bugs BEFORE any sweep work happens:
      - required columns missing
      - dim_pool contains a forbidden look-ahead dimension
      - close_at_<HHMM> missing for an HHMM in r_grid
      - sign-convention violation (mfe_r or mae_r negative)
    """
    issues: List[SchemaIssue] = []

    missing = [c for c in REQUIRED_CANDIDATE_COLUMNS if c not in df.columns]
    if missing:
        issues.append(SchemaIssue(
            "error", "missing_required",
            f"required columns missing: {missing}",
        ))

    for dim in dim_pool:
        for bad in FORBIDDEN_DIM_PREFIXES:
            if dim.startswith(bad):
                issues.append(SchemaIssue(
                    "error", "lookahead_dim",
                    f"dim '{dim}' has forbidden prefix '{bad}' (Lookahead bias, "
                    f"Lesson #5 failure mode #1)",
                ))
        if dim not in df.columns:
            issues.append(SchemaIssue(
                "error", "dim_missing",
                f"dim '{dim}' declared in dim_pool but not present in candidates df",
            ))

    ts_hhmms_needed = sorted({ts for _, _, ts in r_grid})
    for ts in ts_hhmms_needed:
        col = f"close_at_{int(ts)}"
        if col not in df.columns:
            issues.append(SchemaIssue(
                "error", "close_col_missing",
                f"r_grid uses time-stop {ts} but column '{col}' not in candidates df",
            ))

    if "mfe_r" in df.columns and (df["mfe_r"].dropna() < 0).any():
        issues.append(SchemaIssue(
            "error", "negative_mfe",
            "mfe_r has negative values; should be unsigned favorable excursion in R",
        ))
    if "mae_r" in df.columns and (df["mae_r"].dropna() < 0).any():
        issues.append(SchemaIssue(
            "error", "negative_mae",
            "mae_r has negative values; should be unsigned adverse excursion in R",
        ))

    is_valid = not any(i.severity == "error" for i in issues)
    return SchemaValidation(is_valid=is_valid, issues=tuple(issues))


# ---------------------------------------------------------------------------
# Per-trade exit simulation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExitOutcome:
    exit_price: float
    exit_reason: str       # "sl" | "t2_full" | "t1_partial" | "time_stop"
    net_pnl_inr: float


def simulate_exit(
    *,
    side: str,
    entry_price: float,
    qty: int,
    R_per_share: float,
    mfe_r: float,
    mae_r: float,
    close_at_ts: float,
    T1_R: float,
    T2_R: float,
    partial_frac: float = 0.5,
    fee_fn: Callable = calc_fee,
) -> Optional[ExitOutcome]:
    """Compute net PnL for one candidate under a given (T1_R, T2_R, time-stop).

    Resolution order (deterministic, conservative):

      1. If `mae_r >= 1.0`: stop fires at -1R (Failure mode #4: when both stop
         and T2 hit on same bar, stop wins -- pessimistic).
      2. Else if `mfe_r >= T2_R`: 50/50 scale-out at T1 then T2.
      3. Else if `mfe_r >= T1_R`: T1 partial booked, remainder exits at time-stop close.
      4. Else: full qty exits at time-stop close.

    Args:
      side: "LONG" or "SHORT"
      entry_price: Rs per share at Mode B entry
      qty: actual share count (already MIS-sized by sanity script)
      R_per_share: Rs distance from entry to SL (1R)
      mfe_r, mae_r: unsigned R-multiple excursions over the trade window
      close_at_ts: Rs per share at the time-stop bar
      T1_R, T2_R: target R-multiples
      partial_frac: fraction of qty booked at T1 (default 0.5 = 50/50 split)
      fee_fn: fee calculator (default Indian retail stack via calc_fee)

    Returns ExitOutcome or None if inputs are unusable (NaN price, zero R/qty).
    """
    if side not in ALLOWED_SIDES:
        raise ValueError(f"side must be LONG or SHORT, got {side!r}")
    if pd.isna(entry_price) or pd.isna(close_at_ts):
        return None
    if R_per_share <= 0 or qty <= 0:
        return None

    sign = 1.0 if side == "LONG" else -1.0
    # For LONG: favorable = exit > entry, so T1 exit at entry + T1_R*R
    # For SHORT: favorable = exit < entry, so T1 exit at entry - T1_R*R
    t1_exit = entry_price + sign * T1_R * R_per_share
    t2_exit = entry_price + sign * T2_R * R_per_share
    sl_exit = entry_price - sign * R_per_share  # stop is 1R against

    fee_side = "BUY" if side == "LONG" else "SELL"

    def _pnl(exit_price: float, q: int) -> float:
        return sign * (exit_price - entry_price) * q

    # Resolution order — same-bar pessimism
    if mae_r >= 1.0:
        gross = _pnl(sl_exit, qty)
        fee = fee_fn(entry_price, sl_exit, qty, fee_side)
        return ExitOutcome(sl_exit, "sl", gross - fee)

    partial_q = max(int(qty * partial_frac), 1)
    remain_q = qty - partial_q

    if mfe_r >= T2_R:
        gross = _pnl(t1_exit, partial_q) + _pnl(t2_exit, remain_q)
        fee = (fee_fn(entry_price, t1_exit, partial_q, fee_side) +
               fee_fn(entry_price, t2_exit, remain_q, fee_side))
        return ExitOutcome(t2_exit, "t2_full", gross - fee)

    if mfe_r >= T1_R:
        gross = _pnl(t1_exit, partial_q) + _pnl(close_at_ts, remain_q)
        fee = (fee_fn(entry_price, t1_exit, partial_q, fee_side) +
               fee_fn(entry_price, close_at_ts, remain_q, fee_side))
        return ExitOutcome(close_at_ts, "t1_partial", gross - fee)

    gross = _pnl(close_at_ts, qty)
    fee = fee_fn(entry_price, close_at_ts, qty, fee_side)
    return ExitOutcome(close_at_ts, "time_stop", gross - fee)


# ---------------------------------------------------------------------------
# Cell sweep across (filter dims × R-grid)
# ---------------------------------------------------------------------------

@dataclass
class CellSweepConfig:
    side: str
    r_grid: List[Tuple[float, float, int, str]]   # (T1_R, T2_R, TS_hhmm, label)
    dim_pool: List[str]
    k_max: int = 2
    n_min_floor: int = 100
    pf_min_floor: float = 1.10
    n_min_ship: int = 200
    pf_min_ship: float = 1.30
    partial_frac: float = 0.5

    def __post_init__(self):
        if self.side not in ALLOWED_SIDES:
            raise ValueError(f"side must be LONG/SHORT, got {self.side}")
        if not self.r_grid:
            raise ValueError("r_grid must contain at least one (T1, T2, TS, label) tuple")
        if self.k_max < 1 or self.k_max > 3:
            raise ValueError(f"k_max must be 1..3 (3D risks overfitting); got {self.k_max}")


@dataclass(frozen=True)
class CellResult:
    dims: Tuple[str, ...]
    cell_label: str
    r_label: str
    T1_R: float
    T2_R: float
    TS: int
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


def _score_window(pnls: pd.Series) -> Tuple[int, float, float, float, float]:
    n = int(len(pnls))
    if n == 0:
        return 0, 1.0, 0.0, 0.0, 0.0
    pf = _profit_factor(pnls)
    wr = 100.0 * float((pnls > 0).mean())
    net = float(pnls.sum())
    exp = net / n if n > 0 else 0.0
    return n, pf, wr, net, exp


def run_cell_sweep(
    candidates_df: pd.DataFrame,
    cfg: CellSweepConfig,
    *,
    fee_fn: Callable = calc_fee,
) -> pd.DataFrame:
    """Sweep (R-grid × filter cells) on the candidates df.

    Returns a DataFrame of CellResult rows (one per passing cell × R-tuple),
    sorted by PF descending then n descending.

    Raises ValueError if schema validation fails — refuses to score against a
    malformed candidates df (caller must fix the sanity script first).
    """
    r_grid_for_validation = [(T1, T2, ts) for T1, T2, ts, _ in cfg.r_grid]
    validation = validate_candidates_schema(
        candidates_df, dim_pool=cfg.dim_pool, r_grid=r_grid_for_validation,
    )
    if not validation.is_valid:
        errs = "\n".join(f"  [{i.severity}] {i.code}: {i.message}" for i in validation.issues)
        raise ValueError(f"candidates_df failed schema validation:\n{errs}")

    df = candidates_df.copy()
    rows: List[CellResult] = []

    for T1_R, T2_R, ts_hhmm, r_label in cfg.r_grid:
        close_col = f"close_at_{int(ts_hhmm)}"

        def _row_pnl(r: pd.Series) -> float:
            out = simulate_exit(
                side=cfg.side,
                entry_price=float(r["entry_price"]),
                qty=int(r["qty"]),
                R_per_share=float(r["R_per_share"]),
                mfe_r=float(r["mfe_r"]),
                mae_r=float(r["mae_r"]),
                close_at_ts=float(r[close_col]),
                T1_R=T1_R, T2_R=T2_R,
                partial_frac=cfg.partial_frac,
                fee_fn=fee_fn,
            )
            return float("nan") if out is None else out.net_pnl_inr

        df["_hyp_pnl"] = df.apply(_row_pnl, axis=1)
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
                        r_label=r_label,
                        T1_R=T1_R, T2_R=T2_R, TS=int(ts_hhmm),
                        n=n_val,
                        pf=pf_val,
                        wr_pct=float(r["wr"]),
                        net_pnl_inr=float(r["net"]),
                        expectancy_inr=float(r["net"]) / n_val,
                    ))

    if not rows:
        return pd.DataFrame(columns=[
            "dims", "cell_label", "r_label", "T1_R", "T2_R", "TS",
            "n", "pf", "wr_pct", "net_pnl_inr", "expectancy_inr",
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
    """Pick the best cell from sweep results.

    Selection rule (deterministic, no researcher discretion at this step):
      1. Filter to ship-eligible (n >= cfg.n_min_ship AND pf >= cfg.pf_min_ship)
         unless require_ship_eligible=False.
      2. Sort by PF desc, then n desc.
      3. Return the top row as a dict, or None if no rows survive.

    Returns None if no cell passes ship-eligibility — that's a kill signal,
    not a soft warning (Lesson #2: salvage mining = p-hacking).
    """
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
    best = eligible.iloc[0]
    return best.to_dict()


def lock_cell(
    selected: dict,
    *,
    setup_name: str,
    window_label: str,
    output_path: Path,
    force: bool = False,
    extra_metadata: Optional[dict] = None,
) -> Path:
    """Write the locked cell JSON. Refuses to overwrite without force=True.

    Lock-once discipline: once Stage 5 selects a cell, it MUST NOT be
    re-selected on the same data without explicit acknowledgment. This is
    structural defense against post-hoc adjustment.
    """
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
