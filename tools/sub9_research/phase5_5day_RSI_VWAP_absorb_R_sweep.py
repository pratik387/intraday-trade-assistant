"""Phase 5 R-multiple sweep for 5day_RSI_VWAP_absorb_continuation_long.

Brief: specs/2026-05-22-brief-5day_RSI_VWAP_absorb_continuation_long.md
Predecessor Phase 4 sanity: tools/sub9_research/sanity_5day_RSI_VWAP_absorb_continuation_long.py

Operates on Phase 4 sanity output CSVs. Sweeps T1_R and T2_R (LOWER than the
locked Phase 4 values of 1.0R / 2.0R) for the small_cap-only cohort, since
Phase 4 showed all the edge sits in small_cap.

# Why this sweep is bounded to T1_new <= 1.0R, T2_new <= 2.0R

The Phase 4 sanity walks STOP at the first hit (T1, T2, SL, or time_stop). The
recorded mfe_r is capped at that exit point. For T1_new > 1.0R or T2_new > 2.0R
we would need a full-session re-walk (not implemented here — would cost a
fresh sanity-run pass). For T1_new <= current T1 and T2_new <= current T2,
the existing mfe_r is sufficient: any trade that hit current T1=1.0R also hit
T1_new (where T1_new < 1.0R) at an earlier bar, so the lower-T1 cell can be
derived from existing data.

# Same-bar / cross-bar SL ambiguity (Lesson #5 #4 pessimism)

When mae_r >= 1.0 (SL hit at SOME bar during the walk) AND mfe_r >= T_new,
we don't know chronologically whether SL or T_new hit first. Per the brief's
pessimistic policy (same Lesson #5 #4 rule extended across bars), SL wins.

# Cell-lock pre-registered

Lock to cap_segment = 'small_cap'. Phase 4 showed mid_cap drags aggregate
net PF below 1.0 in BOTH Discovery and OOS; small_cap alone produced
Discovery net PF 1.16 / OOS 1.11.

Acceptance gates (per setup_lifecycle Stage 5):
  - Discovery net PF >= 1.20 with n >= 200
  - OOS net PF >= 1.10 with n >= 30
  - Holdout: n >= 30 required for validation (Phase 4 showed n=8 — likely
    BLOCKS ship regardless of R-sweep outcome; sweep proceeds anyway as a
    necessary diagnostic).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
from tools.sub7_validation.build_per_setup_pnl import calc_fee  # noqa: E402


# ---------------------------------------------------------------------------
# Sweep grid (pre-registered: T1_R <= 1.0, T2_R <= 2.0 — see header)
# ---------------------------------------------------------------------------

T1_GRID = [0.3, 0.5, 0.75, 1.0]
T2_GRID = [1.0, 1.25, 1.5, 1.75, 2.0]

# Locked filters (carry-over from Phase 4 brief)
CAP_LOCK = "small_cap"
RISK_PER_TRADE_RUPEES = 1000  # for qty-based fee recomputation if R changes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simulate_exit(
    mfe_r: float, mae_r: float,
    close_at_target: float, entry_price: float, R_per_share: float,
    t1_r_new: float, t2_r_new: float,
) -> Tuple[str, float]:
    """Simulate exit reason + R-multiple under new (T1_R, T2_R).

    Pessimism rule: when SL hit (mae_r >= 1.0) AND target also hit, SL wins.
    """
    sl_hit = mae_r >= 1.0
    t2_hit = mfe_r >= t2_r_new
    t1_hit = mfe_r >= t1_r_new

    if sl_hit:
        # Pessimistic: SL wins regardless of T1/T2 hits in MFE
        return ("sl", -1.0)
    if t2_hit:
        return ("t2", t2_r_new)
    if t1_hit:
        return ("t1", t1_r_new)
    # Time stop — compute from close_at_target
    if R_per_share <= 0:
        return ("invalid", 0.0)
    pnl_per_share = close_at_target - entry_price
    return ("time_stop", pnl_per_share / R_per_share)


def _sweep_cell(
    df: pd.DataFrame, t1_r_new: float, t2_r_new: float,
) -> Dict[str, float]:
    """Apply (t1_r_new, t2_r_new) to all rows; return aggregate metrics."""
    n = len(df)
    if n == 0:
        return {"n": 0, "net_pf": 0.0, "gross_pf": 0.0, "mean_R": 0.0,
                "win_rate": 0.0, "net_inr": 0.0, "gross_inr": 0.0}

    exit_reasons = []
    r_mults = []
    new_exit_prices = []
    for r in df.itertuples():
        reason, r_mult = _simulate_exit(
            float(r.mfe_r), float(r.mae_r),
            float(r.close_at_1325), float(r.entry_price), float(r.R_per_share),
            t1_r_new, t2_r_new,
        )
        exit_reasons.append(reason)
        r_mults.append(r_mult)
        new_exit_prices.append(float(r.entry_price) + r_mult * float(r.R_per_share))

    df2 = df.copy()
    df2["new_r_mult"] = r_mults
    df2["new_exit_reason"] = exit_reasons
    df2["new_exit_price"] = new_exit_prices
    df2["new_gross_inr"] = (df2["new_exit_price"] - df2["entry_price"]) * df2["qty"]
    # Recompute fees with new exit (qty unchanged since R unchanged here)
    df2["new_fee_inr"] = df2.apply(
        lambda r: calc_fee(r["entry_price"], r["new_exit_price"], int(r["qty"]), "BUY"),
        axis=1,
    )
    df2["new_net_inr"] = df2["new_gross_inr"] - df2["new_fee_inr"]

    gross_win = float(df2.loc[df2["new_gross_inr"] > 0, "new_gross_inr"].sum())
    gross_loss = abs(float(df2.loc[df2["new_gross_inr"] < 0, "new_gross_inr"].sum()))
    net_win = float(df2.loc[df2["new_net_inr"] > 0, "new_net_inr"].sum())
    net_loss = abs(float(df2.loc[df2["new_net_inr"] < 0, "new_net_inr"].sum()))

    return {
        "n": n,
        "gross_pf": gross_win / max(gross_loss, 1e-9),
        "net_pf": net_win / max(net_loss, 1e-9),
        "mean_R": float(np.mean(r_mults)),
        "win_rate": float((np.array(r_mults) > 0).mean()),
        "gross_inr": float(df2["new_gross_inr"].sum()),
        "net_inr": float(df2["new_net_inr"].sum()),
        "exit_mix": df2["new_exit_reason"].value_counts(normalize=True).round(3).to_dict(),
    }


def _load_window(window: str) -> pd.DataFrame:
    p = _REPO_ROOT / "reports" / "sub9_sanity" / (
        f"_5day_RSI_VWAP_absorb_continuation_long_trades_{window}.csv"
    )
    df = pd.read_csv(p)
    # Cell-lock: small_cap only
    df = df[df["cap_segment"] == CAP_LOCK].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    print(f"Phase 5 R-sweep for 5day_RSI_VWAP_absorb_continuation_long")
    print(f"  Cell-lock: cap_segment = {CAP_LOCK}")
    print(f"  T1_R grid: {T1_GRID}")
    print(f"  T2_R grid: {T2_GRID}")
    print(f"  Constraint: T2_R > T1_R")
    print()

    windows = {}
    for w in ("discovery", "oos", "holdout"):
        df = _load_window(w)
        windows[w] = df
        print(f"  Loaded {w}: n={len(df)} (small_cap-only)")
    print()

    # Header: Discovery PF / OOS PF / HO PF / Discovery N / OOS N / HO N
    rows = []
    for t1 in T1_GRID:
        for t2 in T2_GRID:
            if t2 <= t1:
                continue
            cell = {"T1_R": t1, "T2_R": t2}
            for w in ("discovery", "oos", "holdout"):
                m = _sweep_cell(windows[w], t1, t2)
                cell[f"{w}_n"] = m["n"]
                cell[f"{w}_net_pf"] = m["net_pf"]
                cell[f"{w}_gross_pf"] = m["gross_pf"]
                cell[f"{w}_mean_R"] = m["mean_R"]
                cell[f"{w}_wr"] = m["win_rate"]
                cell[f"{w}_net_inr"] = m["net_inr"]
            rows.append(cell)

    results = pd.DataFrame(rows)
    out_path = _REPO_ROOT / "reports" / "sub9_sanity" / (
        "_phase5_5day_RSI_VWAP_absorb_continuation_long_R_sweep.csv"
    )
    results.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    print()

    # Print compact summary
    cols = ["T1_R", "T2_R", "discovery_n", "discovery_net_pf", "oos_n",
            "oos_net_pf", "holdout_n", "holdout_net_pf",
            "discovery_net_inr", "oos_net_inr", "holdout_net_inr"]
    print("R-sweep results (sorted by discovery_net_pf desc):")
    print("=" * 110)
    sorted_results = results.sort_values("discovery_net_pf", ascending=False)
    with pd.option_context("display.max_rows", None, "display.width", 200,
                            "display.float_format", "{:.3f}".format):
        print(sorted_results[cols].to_string(index=False))
    print()

    # Acceptance check
    print("Acceptance gates (Stage 5):")
    print("  Discovery: net_pf >= 1.20 AND n >= 200")
    print("  OOS:       net_pf >= 1.10 AND n >= 30")
    print("  Holdout:   n >= 30 (validation floor)")
    print()
    passes = results[
        (results["discovery_net_pf"] >= 1.20)
        & (results["discovery_n"] >= 200)
        & (results["oos_net_pf"] >= 1.10)
        & (results["oos_n"] >= 30)
    ]
    if passes.empty:
        print("  NO cell passes Discovery + OOS gate. Phase 5 KILL signal.")
    else:
        print(f"  {len(passes)} cell(s) pass Discovery + OOS gate:")
        print(passes[cols].to_string(index=False))
        ho_validated = passes[passes["holdout_n"] >= 30]
        if ho_validated.empty:
            print()
            print("  WARNING: 0 cells have HO n >= 30 (Lesson #5 #5: HO not validated).")
            print("  Best Discovery+OOS cell is PARK candidate (forward-validate paper).")
        else:
            print(f"  {len(ho_validated)} cell(s) ALSO pass HO n>=30:")
            print(ho_validated[cols].to_string(index=False))


if __name__ == "__main__":
    main()
