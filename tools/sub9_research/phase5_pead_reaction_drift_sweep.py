"""Phase 5 cell sweep: pead_reaction_drift — LONG leg only.

Brief:     specs/2026-07-27-brief-pead_reaction_drift.md  Section 9b (pre-registered
           sweep, BINDING — no dimensions added or changed after seeing results).
Sanity:    tools/sub9_research/sanity_pead_reaction_drift.py (construction REUSED via
           build_window_ledger(); this driver adds NO signal logic of its own).
Lifecycle: docs/setup_lifecycle.md Stage 5 + 2026-07-27 amendments A1/A2.

PATH TAKEN (tooling decision, stated per Stage-5 instruction):
    tools/methodology/cell_sweep.py supports exactly three target paradigms — R,
    pct, structural — all built around intraday exits with `close_at_<HHMM>`
    time-stop grids and SL/T1/T2 resolution. A multiday CNC hold-to-close
    candidate has none of those (no intraday SL, exit = close of the H-th
    in-market session). This driver therefore follows the MULTIDAY sweep
    precedent instead:
      - filter-dimension cell sweep pattern: run_cell_sweep_close_dn_overnight.py
      - vol-scaled target touch-sim:        multiday_target_exit_study.py
        (V-family: target = entry * (1 + k * std20/close), touch day j fills at
        open if gap-opened above target else at target; a touch on the final
        hold day fills at target instead of that day's close).

PRE-REGISTERED GRID (§9b — EXACTLY these dimensions, nothing else):
    threshold    {pct5 (causal 95th), pct3 (causal 97th)}          [causal, trailing 60cal-day xsec]
    hold         {5, 7, 10} sessions in-market
    cap_segment  {all, large+mid, mid+small+unknown}
    ADV tier     {all, tiers 1-4 (drop most-liquid tier 5)}
    exit         {hold-to-H close (baseline); 2-sigma vol-scaled target w/ H time-stop}
    -> 2 x 3 x 3 x 2 x 2 = 72 cells.
    Floors: n >= 100 (Discovery); ship-eligible PF_net >= 1.20 (Discovery).
    Stopped variants: §9b allows ONE diagnostic measurement (2-sigma vol-scaled
    stop, hold-to-close) — reported as diagnostic rows, NEVER selectable
    (Stage-4 MAE mean -6.3% says stops clip the drift).

COSTS (mtf_capitulation CNC cost model, per §9b / _tmp_b1_pead_killtest.py):
    round-trip fixed = delivery STT 0.20% + brokerage 0.06% + charges 0.047%
                     = 0.307% of notional
    slippage         = 20 bp round-trip
    total            = 0.507% subtracted from the raw per-trade return.
    Equal Rs 1L notional per trade (fractional qty) -> PF/WR/expectancy are
    per-trade-return statistics; no MIS/MTF leverage (CNC delivery, no interest).

WINDOW DISCIPLINE (amendment A1):
    --window discovery : reaction dates 2023-01-01 .. 2024-12-31. Sweep + lock.
    --window demoted   : reaction dates 2025-01-01 .. 2026-04-30 (demoted
                         development data). Runs the LOCKED cell ONLY, ONCE,
                         unchanged, from the lock JSON. No iteration.
    Reaction dates >= 2026-05-01 (fresh decisive pool) are unreachable — the
    ledger builder raises on any window touching them.

Run:
    .venv/Scripts/python tools/sub9_research/phase5_pead_reaction_drift_sweep.py --window discovery
    .venv/Scripts/python tools/sub9_research/phase5_pead_reaction_drift_sweep.py --window demoted
Output:
    reports/sub9_sanity/_pead_reaction_drift_phase5_cells_discovery.csv (full 72-cell table)
    tools/sub9_research/pead_reaction_drift_cell_lock.json (written manually after
    reading the table — selection is stability-first, documented in the JSON)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.sub9_research.sanity_pead_reaction_drift import (  # noqa: E402
    H_MAX, LOCKED_ENTRY_LAG, build_window_ledger,
)

# ---------------------------------------------------------------------------
# PRE-REGISTERED constants (§9b). DO NOT EDIT after the first Discovery run.
# ---------------------------------------------------------------------------

DISCOVERY = (pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31"))
DEMOTED = (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-04-30"))

THRESHOLDS = ("pct5", "pct3")               # pct5 = causal 95th (Stage-4 base); pct3 = causal 97th
HOLDS = (5, 7, 10)                          # sessions in-market
CAP_SEGMENTS = {
    "all": None,
    "large_mid": {"large_cap", "mid_cap"},
    "mid_small_unknown": {"mid_cap", "small_cap", "unknown"},
}
ADV_TIERS = {"all": None, "tiers_1_4": {1, 2, 3, 4}}
EXIT_MODES = ("hold_close", "target_2sigma")
TARGET_SIGMA_K = 2.0                        # 2-sigma vol-scaled target (V-family, k=2)

COST_RT_FIXED = 0.0020 + 0.0006 + 0.00047   # CNC STT + brokerage + charges (round-trip)
COST_SLIPPAGE = 0.0020                      # 20bp round-trip
COST_TOTAL = COST_RT_FIXED + COST_SLIPPAGE  # 0.00507
NOTIONAL_INR = 100_000.0                    # equal Rs 1L per trade, fractional qty

N_MIN_FLOOR = 100                           # Discovery per-cell floor (§9b)
PF_MIN_SHIP = 1.20                          # Discovery ship-eligibility floor

CELL_LOCK_JSON = REPO_ROOT / "tools" / "sub9_research" / "pead_reaction_drift_cell_lock.json"
OUT_CELLS_CSV = REPO_ROOT / "reports" / "sub9_sanity" / "_pead_reaction_drift_phase5_cells_discovery.csv"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Exit simulation (multiday_target_exit_study.py V-family conventions)
# ---------------------------------------------------------------------------

def simulate_exits(sub: pd.DataFrame, hold: int, exit_mode: str,
                   stop_diagnostic: bool = False) -> np.ndarray:
    """Net per-trade return (fraction) for the LONG leg, Rs-1L-equal-notional.

    hold_close    : exit at close of the hold-th in-market session (close_p{hold}).
    target_2sigma : target = entry * (1 + 2 * sigma20_frac), computed AT ENTRY from
                    signal-date data. Touch scan over in-market sessions p1..p{hold}
                    (p1 = entry day; entry is at its open, so its post-open range
                    counts). Fill = open if the day gap-opens above target, else
                    target. A touch on the final day fills at target instead of the
                    close. No touch (or sigma undefined) -> hold-to-close exit.
    stop_diagnostic (NEVER selectable): stop = entry * (1 - 2 * sigma20_frac);
                    scan checks the STOP FIRST each day (pessimistic, Lesson #5
                    failure mode 4), fill = open if gap-opened below stop else stop.
    """
    entry = sub["entry_open"].to_numpy(dtype=float, copy=True)
    sigma = sub["sigma20_frac"].to_numpy(dtype=float, copy=True)
    # copy=True is LOAD-BEARING: without it exit_px can be a view into the
    # DataFrame block and the touch-fill writes below would corrupt close_p*
    # for every later cell (caught in the first run: identical cell gave
    # PF 1.7254 in-loop vs 1.4656 post-loop).
    exit_px = sub[f"close_p{LOCKED_ENTRY_LAG + hold - 1}"].to_numpy(dtype=float, copy=True)

    use_target = exit_mode == "target_2sigma"
    tgt = entry * (1.0 + TARGET_SIGMA_K * sigma) if use_target else None
    stp = entry * (1.0 - TARGET_SIGMA_K * sigma) if stop_diagnostic else None

    done = np.zeros(len(sub), bool)
    for j in range(1, hold + 1):
        pos = LOCKED_ENTRY_LAG + j - 1
        hi = sub[f"high_p{pos}"].to_numpy(float)
        lo = sub[f"low_p{pos}"].to_numpy(float)
        op = sub[f"open_p{pos}"].to_numpy(float)
        if stop_diagnostic:
            s_touch = ~done & np.isfinite(stp) & (stp < entry) & np.isfinite(lo) & (lo <= stp)
            s_fill = np.where(np.isfinite(op) & (op <= stp), op, stp)
            exit_px[s_touch] = s_fill[s_touch]
            done |= s_touch
        if use_target:
            t_touch = ~done & np.isfinite(tgt) & (tgt > entry) & np.isfinite(hi) & (hi >= tgt)
            t_fill = np.where(np.isfinite(op) & (op >= tgt), op, tgt)
            exit_px[t_touch] = t_fill[t_touch]
            done |= t_touch
    raw_ret = exit_px / entry - 1.0
    return raw_ret - COST_TOTAL, float(done.mean()) if len(done) else 0.0


def cell_stats(net_ret: np.ndarray, years: np.ndarray) -> dict:
    pnl = net_ret * NOTIONAL_INR
    g = float(pnl[pnl > 0].sum())
    l = float(-pnl[pnl < 0].sum())
    pf = (g / l) if l > 0 else float("inf")
    out = {
        "n": int(len(pnl)),
        "pf_net": round(pf, 4),
        "wr_pct": round(float((pnl > 0).mean() * 100.0), 2),
        "exp_net_pct": round(float(net_ret.mean() * 100.0), 4),
        "exp_net_inr": round(float(pnl.mean()), 1),
        "net_inr": round(float(pnl.sum()), 0),
    }
    for y in np.unique(years):
        m = years == y
        gy = float(pnl[m & (pnl > 0)].sum())
        ly = float(-pnl[m & (pnl < 0)].sum())
        out[f"pf_{y}"] = round((gy / ly) if ly > 0 else float("inf"), 4)
        out[f"n_{y}"] = int(m.sum())
    return out


def apply_cell(led: pd.DataFrame, threshold: str, cap_key: str, tier_key: str) -> pd.DataFrame:
    sub = led
    if threshold == "pct3":
        sub = sub[sub["reaction_ret_pct"] >= sub["thr_long_p97_pct"]]
    caps = CAP_SEGMENTS[cap_key]
    if caps is not None:
        sub = sub[sub["cap_segment"].isin(caps)]
    tiers = ADV_TIERS[tier_key]
    if tiers is not None:
        sub = sub[sub["adv_tier"].isin(tiers)]
    return sub


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_discovery() -> None:
    log("=" * 78)
    log("PHASE 5 DISCOVERY SWEEP: pead_reaction_drift LONG "
        f"({DISCOVERY[0].date()} -> {DISCOVERY[1].date()})")
    log("=" * 78)
    led = build_window_ledger(*DISCOVERY)
    led = led[led["leg"] == "long"].reset_index(drop=True)
    log(f"LONG-leg ledger: n={len(led)}")
    years = pd.to_datetime(led["reaction_date"]).dt.year.to_numpy()
    led = led.assign(_year=years)

    rows = []
    for thr in THRESHOLDS:
        for cap_key in CAP_SEGMENTS:
            for tier_key in ADV_TIERS:
                sub = apply_cell(led, thr, cap_key, tier_key)
                for hold in HOLDS:
                    for ex in EXIT_MODES:
                        net, touched = simulate_exits(sub, hold, ex)
                        st = cell_stats(net, sub["_year"].to_numpy())
                        rows.append({
                            "threshold": thr, "cap_segment": cap_key,
                            "adv_tier": tier_key, "hold": hold, "exit": ex,
                            **st, "touch_pct": round(touched * 100.0, 1),
                            "floor_n": st["n"] >= N_MIN_FLOOR,
                            "ship_eligible": (st["n"] >= N_MIN_FLOOR
                                              and st["pf_net"] >= PF_MIN_SHIP),
                        })
    cells = pd.DataFrame(rows)
    # min-year PF as the within-Discovery stability diagnostic (2023 vs 2024).
    cells["min_year_pf"] = cells[["pf_2023", "pf_2024"]].min(axis=1)
    cells = cells.sort_values(["ship_eligible", "min_year_pf", "n"],
                              ascending=[False, False, False]).reset_index(drop=True)
    OUT_CELLS_CSV.parent.mkdir(parents=True, exist_ok=True)
    cells.to_csv(OUT_CELLS_CSV, index=False)
    log(f"wrote {len(cells)} cells -> {OUT_CELLS_CSV}")

    elig = cells[cells["ship_eligible"]]
    log(f"cells passing n>={N_MIN_FLOOR} floor: {int(cells['floor_n'].sum())}/{len(cells)}; "
        f"ship-eligible (PF_net>={PF_MIN_SHIP}): {len(elig)}")
    show = ["threshold", "cap_segment", "adv_tier", "hold", "exit", "n", "pf_net",
            "wr_pct", "exp_net_pct", "touch_pct", "pf_2023", "pf_2024", "min_year_pf"]
    with pd.option_context("display.width", 250):
        if len(elig):
            log("SHIP-ELIGIBLE cells (stability-ordered):")
            print(elig[show].to_string(index=False))
        log("Top 10 ineligible cells by pf_net:")
        print(cells[~cells["ship_eligible"]].sort_values("pf_net", ascending=False)
              .head(10)[show].to_string(index=False))

    # §9b stopped-variant DIAGNOSTIC (never selectable): 2-sigma vol-scaled stop,
    # hold-to-close, on the base pct5/all/all cell at each hold.
    log("DIAGNOSTIC (stopped variant, NEVER selectable): 2-sigma vol-scaled stop + hold-to-close, pct5/all/all")
    base = apply_cell(led, "pct5", "all", "all")
    for hold in HOLDS:
        net, stopped = simulate_exits(base, hold, "hold_close", stop_diagnostic=True)
        st = cell_stats(net, base["_year"].to_numpy())
        base_net, _ = simulate_exits(base, hold, "hold_close")
        base_pf = cell_stats(base_net, base["_year"].to_numpy())["pf_net"]
        log(f"  hold={hold}: n={st['n']} PF_net={st['pf_net']} WR={st['wr_pct']}% "
            f"exp={st['exp_net_pct']}% stopped={stopped*100:.1f}%  "
            f"(baseline hold_close PF_net={base_pf})")


def run_demoted() -> None:
    if not CELL_LOCK_JSON.exists():
        raise SystemExit("No cell lock JSON — run/lock Discovery first. "
                         "If Discovery produced no eligible cell, that is a KILL; do not run demoted.")
    lock = json.loads(CELL_LOCK_JSON.read_text(encoding="utf-8"))
    lc = lock["locked_cell"]
    log("=" * 78)
    log(f"PHASE 5 DEMOTED-WINDOW ONE-SHOT: locked cell {lc} "
        f"({DEMOTED[0].date()} -> {DEMOTED[1].date()})  [amendment A1 — ONE run, no iteration]")
    log("=" * 78)
    led = build_window_ledger(*DEMOTED)
    led = led[led["leg"] == "long"].reset_index(drop=True)
    log(f"LONG-leg ledger: n={len(led)}")
    sub = apply_cell(led, lc["threshold"], lc["cap_segment"], lc["adv_tier"])
    years = pd.to_datetime(sub["reaction_date"]).dt.year.to_numpy()
    net, touched = simulate_exits(sub, lc["hold"], lc["exit"])
    st = cell_stats(net, years)
    st["touch_pct"] = round(touched * 100.0, 1)
    log(f"DEMOTED result: {st}")
    d_pf = lock["discovery_stats"]["pf_net"]
    gap = abs(d_pf - st["pf_net"])
    log(f"|PF_disc - PF_demoted| = |{d_pf} - {st['pf_net']}| = {gap:.4f} "
        f"(lifecycle overfit gate: > 0.30 = overfit)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", choices=["discovery", "demoted"], required=True)
    args = ap.parse_args()
    if args.window == "discovery":
        run_discovery()
    else:
        run_demoted()


if __name__ == "__main__":
    main()
