"""Canonical schema for sanity trade CSVs consumed by walk-forward.

Every sanity script must emit (or be normalized into) this schema before
walk-forward will accept it. The validator catches the five bugs that
would silently invert walk-forward verdicts:

  1. Sign-convention inversion (LONG vs SHORT pnl_pct wrong)
  2. Double-counted fees/leverage in pnl_pct
  3. IST/UTC date drift on signal_date
  4. Raw-candidates vs filtered-trades layer mismatch
  5. NaN/inf in pnl_pct silently skewing PF

Per spec docs/superpowers/specs/2026-05-19-walk-forward-methodology-design.md
+ schema discussion 2026-05-20.

THE CONTRACT
------------
Walk-forward consumes pnl_pct = RAW per-share % return, BEFORE fees,
BEFORE MIS leverage. The engine applies its own fee+leverage stack
(`_compute_per_trade_net_pnl`). Sanity scripts MUST emit raw % returns.

The validator computes pnl_pct from entry_price + exit_price + side
(the only trustworthy fields across schemas) and compares to the stored
pnl_pct. If they disagree by more than the float-rounding tolerance,
validation fails loudly — this catches accidentally-leveraged or
accidentally-fee-discounted pnl_pct.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = (
    "signal_date",      # date — YYYY-MM-DD, IST-naive
    "symbol",           # str — "NSE:XYZ" format
    "side",             # str — exactly "LONG" or "SHORT"
    "entry_price",      # float > 0
    "exit_price",       # float > 0
    "qty",              # int > 0
    "pnl_pct",          # float — raw per-share % return, side-aware,
                        #         NO fees, NO leverage applied
    "exit_reason",      # str — one of EXIT_REASONS
    "same_bar",         # bool — True iff SL hit in entry bar (i+1 under Mode B)
)

# Optional columns are passed through if present.
OPTIONAL_COLUMNS = (
    "cap_segment",            # large_cap | mid_cap | small_cap | micro_cap | unknown
    "signal_ts",              # full IST-naive timestamp
    "entry_ts",
    "exit_ts",
    "realized_pnl_inr",       # rupee gross (per-trade); cross-check only
    "fee_inr",                # rupee fees (per-trade); cross-check only
    "net_pnl_inr",            # rupee net = realized - fee; cross-check only
    "r_multiple",             # float — units of R risk per trade
    "t1_target",
    "t2_target",
    "hard_sl",
    "t1_partial_booked",      # bool — True iff a T1 partial profit was booked before
                              # the final exit. When True, pnl_pct is BLENDED across
                              # both legs (T1 partial qty + final exit qty) and CANNOT
                              # be derived from a single entry/exit pair. Validator
                              # skips the sign cross-check when this is True.
    # ---- Cell-sweep helper fields (BE-trail v3 enablement, 2026-05-20) ----
    # When present, tools/methodology/cell_sweep.py can compute exact BE-trail
    # exit reasons instead of the v2 conservative approximation (which assumes
    # mae_r >= 0.75 post-T1 triggers BE). Sanity scripts that track these
    # should populate them; otherwise leave absent and the helper falls back
    # to the conservative path.
    "mfe_r",                  # float >= 0 — max favorable excursion in R across full trade
    "mae_r",                  # float >= 0 — max adverse excursion in R across full trade
    "mfe_r_pre_t1",           # float >= 0 — MFE in R BEFORE T1 was hit (NaN if T1 never hit)
    "mae_r_post_t1",          # float >= 0 — MAE in R AFTER T1 was hit (NaN if T1 never hit).
                              # When mae_r_post_t1 >= 1.0 AND t1_partial_booked, BE trail
                              # tripped the post-T1 leg in production.
    "R_per_share",            # float > 0 — Rs distance from entry to SL (1R in Rs)
    # ---- Look-ahead-prone EOD columns (metadata only) ----
    # These are present in some sanity outputs for AUDIT purposes only. They
    # must NEVER be used as cell-mining filter dimensions. cell_sweep.py's
    # FORBIDDEN_DIM_PREFIXES/EXACT blocks the obvious ones; the registry at
    # assets/setup_dimension_registry.yaml carries the per-setup forbidden list.
    "day_high",               # EOD session high — for cross-check only
    "day_low",                # EOD session low — for cross-check only
    "day_close",              # EOD close — for cross-check only
    "day_gain_pct_eod_metadata",  # EOD day_gain_pct — metadata only, DO NOT FILTER ON THIS
    # Safe signal-time equivalents (preferred for filtering):
    "session_high_at_signal", # max(bar.high for bar in bars[:signal_bar]) — safe
    "day_gain_at_signal_pct", # (session_high_at_signal / pdc - 1) * 100 — safe
)

ALLOWED_SIDES = ("LONG", "SHORT")

EXIT_REASONS = (
    "sl",              # stop-loss hit (multi-bar path)
    "same_bar_sl",     # stop-loss hit in entry bar itself
    "breakeven_stop",  # SL moved to breakeven after T1 partial; final leg exited at BE.
                       # Carries non-zero blended pnl_pct (T1 partial profit on half, 0 on remainder).
                       # When this exit_reason fires, t1_partial_booked MUST be True.
    "t1",              # T1 target hit (partial exit)
    "t2",              # T2 target hit (full exit if no partial; second half if partial)
    "time_stop",       # time stop fired
    "eod",             # forced EOD exit (15:20 MIS auto-square)
    "manual",          # forced exit (rare; manual intervention modeled in sanity)
)

# Date range we expect — walk-forward is built on Jan 2023 - Apr 2026
EXPECTED_DATE_MIN = date(2022, 1, 1)
EXPECTED_DATE_MAX = date(2026, 12, 31)

# Tolerance for pnl_pct vs (exit-entry)/entry computation cross-check
# 10 basis points = 0.1pp = 0.001 in fractional terms.
PNL_PCT_TOLERANCE_PP = 0.10

# Required metadata (passed alongside the DataFrame, not as columns)
REQUIRED_METADATA = ("_setup_name", "_layer")

ALLOWED_LAYERS = ("filtered_trades",)
# "raw_candidates" is INTENTIONALLY not in ALLOWED_LAYERS — walk-forward
# only accepts post-filter trades. Sanity scripts that emit raw signal
# candidates must filter to the final cell before saving for walk-forward.


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidationIssue:
    severity: str          # "error" | "warn"
    code: str              # short stable identifier
    message: str           # human-readable description
    row_indices: Optional[List[int]] = None   # if row-specific


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    issues: List[ValidationIssue]
    n_rows: int

    @property
    def n_errors(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def n_warnings(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warn")

    def summary(self) -> str:
        if self.is_valid and not self.issues:
            return f"OK ({self.n_rows} rows)"
        lines = [f"VALIDATION RESULT ({self.n_rows} rows): "
                 f"valid={self.is_valid}, "
                 f"errors={self.n_errors}, warnings={self.n_warnings}"]
        for issue in self.issues:
            row_str = ""
            if issue.row_indices is not None:
                if len(issue.row_indices) > 10:
                    row_str = (f" [{len(issue.row_indices)} rows; "
                               f"first 5: {issue.row_indices[:5]}]")
                else:
                    row_str = f" [rows: {issue.row_indices}]"
            lines.append(f"  [{issue.severity}/{issue.code}] {issue.message}{row_str}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

def validate(
    df: pd.DataFrame,
    *,
    setup_name: str,
    layer: str,
) -> ValidationResult:
    """Validate that `df` conforms to the canonical sanity CSV schema.

    Args:
        df: trade DataFrame to validate
        setup_name: setup identifier (informational; appears in issue messages)
        layer: must be "filtered_trades" (raw candidates not accepted)

    Returns:
        ValidationResult; check `is_valid`. If False, see `issues` for details.

    The validator does NOT mutate `df`. It also does NOT cast types —
    if dtype is wrong, that's an error condition the caller must fix
    in their adapter, not silently mask.
    """
    issues: List[ValidationIssue] = []

    # Metadata check
    if layer not in ALLOWED_LAYERS:
        issues.append(ValidationIssue(
            severity="error",
            code="metadata.layer.invalid",
            message=(
                f"layer={layer!r} not in {ALLOWED_LAYERS}. "
                f"Walk-forward requires post-filter trades, not raw candidates. "
                f"Filter your sanity output to the final cell before saving."
            ),
        ))

    # Required column presence
    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        issues.append(ValidationIssue(
            severity="error",
            code="columns.missing_required",
            message=(
                f"Setup {setup_name!r} CSV missing required columns: "
                f"{missing_required}. Required schema: {list(REQUIRED_COLUMNS)}"
            ),
        ))
        # Can't validate row contents if required columns are missing —
        # return early so we don't cascade-fail on missing-key errors.
        return ValidationResult(is_valid=False, issues=issues, n_rows=len(df))

    n = len(df)
    if n == 0:
        issues.append(ValidationIssue(
            severity="error",
            code="rows.empty",
            message=f"Setup {setup_name!r} CSV has zero rows.",
        ))
        return ValidationResult(is_valid=False, issues=issues, n_rows=0)

    # ---- signal_date ----
    try:
        # Convert to date — works for date objects, strings, Timestamps
        sd = pd.to_datetime(df["signal_date"], errors="raise")
        # Reject if any value has tz info
        if sd.dt.tz is not None:
            issues.append(ValidationIssue(
                severity="error",
                code="signal_date.has_tz",
                message=(
                    "signal_date has timezone info. IST/UTC drift risk — "
                    "walk-forward requires IST-naive dates (no tzinfo)."
                ),
            ))
        sd_dates = sd.dt.date
    except (ValueError, TypeError) as e:
        issues.append(ValidationIssue(
            severity="error",
            code="signal_date.unparseable",
            message=f"signal_date column cannot be parsed as date: {e}",
        ))
        return ValidationResult(is_valid=False, issues=issues, n_rows=n)

    out_of_range = sd_dates[(sd_dates < EXPECTED_DATE_MIN) | (sd_dates > EXPECTED_DATE_MAX)]
    if len(out_of_range) > 0:
        issues.append(ValidationIssue(
            severity="error",
            code="signal_date.out_of_range",
            message=(
                f"{len(out_of_range)} signal_date values outside "
                f"[{EXPECTED_DATE_MIN}, {EXPECTED_DATE_MAX}]"
            ),
            row_indices=out_of_range.index.tolist(),
        ))

    # Weekend / holiday check: warn-only (not all sanity scripts honor holidays)
    weekend_mask = sd.dt.dayofweek >= 5
    if weekend_mask.any():
        issues.append(ValidationIssue(
            severity="warn",
            code="signal_date.weekend",
            message=(
                f"{int(weekend_mask.sum())} signal_date values fall on a weekend. "
                f"Check sanity script's holiday/weekend filter."
            ),
            row_indices=df.index[weekend_mask].tolist(),
        ))

    # ---- side ----
    side = df["side"].astype(str)
    bad_side = ~side.isin(ALLOWED_SIDES)
    if bad_side.any():
        unique_bad = sorted(side[bad_side].unique().tolist())[:10]
        issues.append(ValidationIssue(
            severity="error",
            code="side.invalid_value",
            message=(
                f"side column must contain exactly {ALLOWED_SIDES}. "
                f"Found invalid values (first 10): {unique_bad}"
            ),
            row_indices=df.index[bad_side].tolist(),
        ))

    # ---- prices + qty: numeric, positive ----
    for col in ("entry_price", "exit_price"):
        try:
            vals = pd.to_numeric(df[col], errors="raise")
        except (ValueError, TypeError):
            issues.append(ValidationIssue(
                severity="error",
                code=f"{col}.non_numeric",
                message=f"{col} contains non-numeric values",
            ))
            continue
        bad = (vals <= 0) | vals.isna() | ~np.isfinite(vals)
        if bad.any():
            issues.append(ValidationIssue(
                severity="error",
                code=f"{col}.non_positive",
                message=f"{col} has {int(bad.sum())} non-positive/NaN/inf rows",
                row_indices=df.index[bad].tolist(),
            ))

    try:
        qty = pd.to_numeric(df["qty"], errors="raise")
        bad_qty = (qty <= 0) | qty.isna() | ~np.isfinite(qty)
        if bad_qty.any():
            issues.append(ValidationIssue(
                severity="error",
                code="qty.non_positive",
                message=f"qty has {int(bad_qty.sum())} non-positive/NaN/inf rows",
                row_indices=df.index[bad_qty].tolist(),
            ))
    except (ValueError, TypeError):
        issues.append(ValidationIssue(
            severity="error",
            code="qty.non_numeric",
            message="qty contains non-numeric values",
        ))

    # ---- pnl_pct: numeric, finite, sign-consistent with prices+side ----
    try:
        pp = pd.to_numeric(df["pnl_pct"], errors="raise")
    except (ValueError, TypeError):
        issues.append(ValidationIssue(
            severity="error",
            code="pnl_pct.non_numeric",
            message="pnl_pct contains non-numeric values",
        ))
        pp = None

    if pp is not None:
        bad_pp = pp.isna() | ~np.isfinite(pp)
        if bad_pp.any():
            issues.append(ValidationIssue(
                severity="error",
                code="pnl_pct.nan_or_inf",
                message=(
                    f"pnl_pct has {int(bad_pp.sum())} NaN/inf rows. "
                    f"Walk-forward PF computation will be biased — fix before continuing."
                ),
                row_indices=df.index[bad_pp].tolist(),
            ))

        # Sign-convention cross-check: compute pnl_pct from prices+side,
        # compare to stored. This catches LONG/SHORT inversion AND any
        # contamination from fees/leverage.
        #
        # SKIP this check for rows where t1_partial_booked=True — those rows
        # have a BLENDED pnl_pct (T1 partial profit qty + final exit qty)
        # which cannot be derived from a single entry/exit pair. The adapter
        # is responsible for computing the blended pnl_pct correctly from
        # realized_pnl_inr; we trust the adapter for these rows.
        if "entry_price" in df.columns and "exit_price" in df.columns and "side" in df.columns:
            try:
                ep = pd.to_numeric(df["entry_price"], errors="coerce")
                xp = pd.to_numeric(df["exit_price"], errors="coerce")
                s = df["side"].astype(str)
                # t1_partial_booked: skip cross-check on those rows
                if "t1_partial_booked" in df.columns:
                    t1pb = df["t1_partial_booked"].astype(bool)
                else:
                    t1pb = pd.Series(False, index=df.index)
                # Only validate single-leg rows with valid prices + side
                ok = ((ep > 0) & xp.notna() & np.isfinite(ep) & np.isfinite(xp)
                      & s.isin(ALLOWED_SIDES) & ~t1pb)
                if ok.any():
                    long_mask = ok & (s == "LONG")
                    short_mask = ok & (s == "SHORT")
                    expected = pd.Series(np.nan, index=df.index, dtype=float)
                    expected.loc[long_mask] = (xp[long_mask] - ep[long_mask]) / ep[long_mask] * 100.0
                    expected.loc[short_mask] = (ep[short_mask] - xp[short_mask]) / ep[short_mask] * 100.0
                    diff = (pp - expected).abs()
                    # Use percentage-points tolerance (PNL_PCT_TOLERANCE_PP = 0.10pp)
                    mismatched = diff > PNL_PCT_TOLERANCE_PP
                    mismatched = mismatched & ok  # only flag rows where we computed expected
                    if mismatched.any():
                        # Show first few mismatches for diagnostic
                        sample_rows = df.index[mismatched].tolist()[:5]
                        sample_details = []
                        for r in sample_rows:
                            sample_details.append(
                                f"row {r}: stored={pp.loc[r]:.4f} "
                                f"expected={expected.loc[r]:.4f} "
                                f"side={s.loc[r]} entry={ep.loc[r]:.2f} exit={xp.loc[r]:.2f}"
                            )
                        issues.append(ValidationIssue(
                            severity="error",
                            code="pnl_pct.sign_or_magnitude_mismatch",
                            message=(
                                f"pnl_pct does not match (exit-entry)/entry*100 by side "
                                f"within {PNL_PCT_TOLERANCE_PP}pp tolerance on "
                                f"{int(mismatched.sum())} rows. This indicates EITHER "
                                f"wrong LONG/SHORT side OR fees/leverage already applied "
                                f"to pnl_pct (which is forbidden — walk-forward applies its own). "
                                f"Sample: {sample_details}"
                            ),
                            row_indices=df.index[mismatched].tolist(),
                        ))
            except Exception as e:
                issues.append(ValidationIssue(
                    severity="warn",
                    code="pnl_pct.cross_check_failed",
                    message=f"Could not cross-check pnl_pct vs prices+side: {e}",
                ))

    # ---- symbol format ----
    sym = df["symbol"].astype(str)
    bad_sym = ~sym.str.match(r"^NSE:[A-Z0-9_&.-]+$")
    if bad_sym.any():
        unique_bad = sorted(sym[bad_sym].unique().tolist())[:5]
        issues.append(ValidationIssue(
            severity="warn",
            code="symbol.unexpected_format",
            message=(
                f"{int(bad_sym.sum())} symbols don't match 'NSE:XYZ' format. "
                f"Sample: {unique_bad}. Walk-forward doesn't use symbol directly "
                f"but downstream per-symbol analyses may break."
            ),
            row_indices=df.index[bad_sym].tolist(),
        ))

    # ---- exit_reason ----
    if "exit_reason" in df.columns:
        er = df["exit_reason"].astype(str)
        bad_er = ~er.isin(EXIT_REASONS)
        if bad_er.any():
            unique_bad = sorted(er[bad_er].unique().tolist())[:5]
            issues.append(ValidationIssue(
                severity="warn",
                code="exit_reason.unknown_value",
                message=(
                    f"{int(bad_er.sum())} exit_reason values not in canonical set "
                    f"{EXIT_REASONS}. Sample: {unique_bad}. Adapter should normalize."
                ),
                row_indices=df.index[bad_er].tolist(),
            ))

    # ---- exit_reason=breakeven_stop requires t1_partial_booked=True ----
    if "exit_reason" in df.columns:
        er = df["exit_reason"].astype(str)
        bes_mask = er == "breakeven_stop"
        if bes_mask.any():
            if "t1_partial_booked" not in df.columns:
                issues.append(ValidationIssue(
                    severity="error",
                    code="breakeven_stop.missing_t1_partial_booked",
                    message=(
                        f"{int(bes_mask.sum())} rows have exit_reason='breakeven_stop' "
                        f"but t1_partial_booked column is missing. breakeven_stop is "
                        f"only valid when a T1 partial was booked first."
                    ),
                    row_indices=df.index[bes_mask].tolist(),
                ))
            else:
                t1pb = df["t1_partial_booked"].astype(bool)
                bad = bes_mask & ~t1pb
                if bad.any():
                    issues.append(ValidationIssue(
                        severity="error",
                        code="breakeven_stop.t1_partial_booked_false",
                        message=(
                            f"{int(bad.sum())} rows have exit_reason='breakeven_stop' "
                            f"but t1_partial_booked=False. Inconsistent — adapter bug."
                        ),
                        row_indices=df.index[bad].tolist(),
                    ))

    # ---- same_bar: bool ----
    if "same_bar" in df.columns:
        sb = df["same_bar"]
        # Accept bool or 0/1 ints
        bad_sb = ~sb.isin([True, False, 0, 1])
        if bad_sb.any():
            issues.append(ValidationIssue(
                severity="error",
                code="same_bar.non_boolean",
                message=f"same_bar has {int(bad_sb.sum())} non-boolean values",
                row_indices=df.index[bad_sb].tolist(),
            ))

    # ---- Optional rupee cross-check ----
    # If realized_pnl_inr + fee_inr + net_pnl_inr all present: net == realized - fee
    rupee_cols = {"realized_pnl_inr", "fee_inr", "net_pnl_inr"}
    if rupee_cols.issubset(df.columns):
        try:
            r = pd.to_numeric(df["realized_pnl_inr"], errors="coerce")
            f_ = pd.to_numeric(df["fee_inr"], errors="coerce")
            n_ = pd.to_numeric(df["net_pnl_inr"], errors="coerce")
            ok = r.notna() & f_.notna() & n_.notna()
            if ok.any():
                # Tolerance: 1 rupee or 1% of |realized|, whichever is larger
                tol = np.maximum(1.0, r.abs() * 0.01)
                bad = (n_ - (r - f_)).abs() > tol
                bad = bad & ok
                if bad.any():
                    issues.append(ValidationIssue(
                        severity="warn",
                        code="rupee_pnl.cross_check_failed",
                        message=(
                            f"{int(bad.sum())} rows where net_pnl_inr != "
                            f"realized_pnl_inr - fee_inr (within tolerance). "
                            f"Adapter may have leverage/fee model inconsistency."
                        ),
                        row_indices=df.index[bad].tolist(),
                    ))
        except Exception as e:
            issues.append(ValidationIssue(
                severity="warn",
                code="rupee_pnl.cross_check_skipped",
                message=f"Rupee cross-check skipped: {e}",
            ))

    n_errors = sum(1 for i in issues if i.severity == "error")
    return ValidationResult(
        is_valid=(n_errors == 0),
        issues=issues,
        n_rows=n,
    )


# ---------------------------------------------------------------------------
# Canonical column ordering (for tidy CSV output)
# ---------------------------------------------------------------------------

def canonical_column_order(present_optional: List[str] = None) -> List[str]:
    """Return columns in canonical order: required first, then any present optional."""
    present_optional = present_optional or []
    optional_in_order = [c for c in OPTIONAL_COLUMNS if c in present_optional]
    return list(REQUIRED_COLUMNS) + optional_in_order
