"""phase5_sweep_earnings_downshock_continuation_short.py -- lifecycle Stage 5.

Cell-lock + geometry sweep, then ONE demoted-window check, for
`earnings_downshock_continuation_short`.

THE GRID IS PRE-REGISTERED AND COMMITTED (0c69ed5) in
`specs/2026-07-29-brief-earnings_downshock_continuation_short.md` SS9c.  It is BINDING.
No dimension may be added, removed, reweighted or reinterpreted after seeing any result.
This file encodes SS9c verbatim in `PREREG` below and refuses to run on anything else.

FIXED, NOT SWEPT (SS9c)
-----------------------
  signal <= -8%, de-dup (symbol, reaction_date), 09:20 entry, 15:15 EXIT (broker
  mechanics -- Zerodha auto-squares MIS from ~15:20), ProductionUniverseGate,
  ASM/GSM + circuit exclusions, real Zerodha MIS-short fees, measured per-trade
  slippage.  All of these come from `sanity_earnings_downshock_continuation_short.
  build_cohort()` -- this driver does NOT re-implement the construction.

SWEPT (SS9c, and nothing else)
------------------------------
  cells    : ADV tier {low, mid, both}
             x shock depth {-8%, -10%, -12%}
             x announce class {AMC, intraday, all}                        = 27 cells
  geometry : stop   in {none, 2%, 3%, 4%}
             x target in {none, 1.5%, 2.5%}                               = 12 geoms
  `none/none` (the Stage-4 construction) is the INCUMBENT and WINS TIES.  A geometry
  displaces it ONLY on a strictly better result in BOTH eras -- never on pooled numbers.
  Applied mechanically in `incumbent_test()`.

LOCKABLE-CELL RULE (amendments A5 + A5-b), applied mechanically
--------------------------------------------------------------
  net expectancy > 0 on the ABSOLUTE statistic (net %/trade -- this is a SINGLE-LEG
  cash SHORT, so A5-b forbids a relative statistic) in BOTH eras at CONSERVATIVE
  measured slippage, n >= 100 per era, pooled PF >= 1.20.
  Discovery alone is era_A, so the within-development era split used HERE is
  2023 vs 2024.  *** The true era_B test is Step 3 (the demoted window). ***
  Selection is STABILITY-FIRST (smallest year-gap at comparable PF), never top-PF-only
  (memory: feedback_cell_sweep_stability_over_top_pf).
  If NO cell is eligible that is a KILL -- no salvage, no relaxed thresholds.

STEP 3 -- demoted-window check, ONE SHOT, NO ITERATION
------------------------------------------------------
  The locked cell UNCHANGED on reaction dates 2025-01-01 .. 2026-04-30 (amendment A1:
  demoted development data, but it still burns).  |PF_disc - PF_demoted| > 0.30 is
  called overfit by the lifecycle (Stage 5 Step 3).
  "UNCHANGED" is taken literally: the ADV tercile CUT POINTS are FROZEN from Discovery
  and imposed on the demoted window rather than re-fitted there (the within-window
  re-fit is emitted alongside as a robustness line from the SAME single run).
  Reaction dates >= 2026-05-01 are NEVER touched -- `build_cohort` asserts on it.

GEOMETRY MECHANICS (stated once; identical in both windows)
-----------------------------------------------------------
  Entry = close of the 09:15-09:20 bar (the 09:20 print).  Path walk starts at bar 1
  (Mode-B off-by-one; bar 0's range is already spent at entry).
  SHORT: stop_px = entry * (1 + stop%/100)   -> triggered when bar HIGH >= stop_px
         tgt_px  = entry * (1 - target%/100) -> triggered when bar LOW  <= tgt_px
  SAME-BAR PESSIMISM (lifecycle failure mode #4): if a bar touches BOTH, the STOP wins.
  GAP-THROUGH: if the bar OPENS beyond the level, the fill is the OPEN, not the level.
  Untriggered trades exit at the close of the 15:10-15:15 bar (the 15:15 print),
  exit_reason "eod" -- i.e. the none/none incumbent's exit.

SLIPPAGE (measured, per trade, NOT assumed)
-------------------------------------------
  `measure_slippage_earnings_downshock.py` triangulated this exact cohort on 1m data at
  the FEASIBLE 15:15 exit.  Per trade:
      CENTRAL      = perside_blend_bps   (mean 18.1 bp/side; adv_low 22.3 / adv_mid 13.8)
      CONSERVATIVE = perside_cons_bps    (mean 26.6 bp/side; adv_low 33.8 / adv_mid 19.5)
  DISCLOSED APPROXIMATION: the measurement is anchored on the 09:20 entry minute and the
  15:14 exit minute.  A geometry that stops out at, say, 11:05 exits into an unmeasured
  minute.  The 15:15 exit-leg estimate is applied to it as the best available number.
  Intraday stop-outs are into a fast, adverse tape and would realistically cost MORE --
  so this approximation FLATTERS the geometries and never the none/none incumbent.
  That bias points away from displacing the incumbent, i.e. in the safe direction.
  A flat-rate cross-check (CENTRAL 18.7 / CONSERVATIVE 27.5 bp/side, the brief SS9b
  headline rates) is emitted alongside every cell.

OUTPUTS
-------
  reports/sub9_sanity/_earnings_downshock_phase5_cells.csv          (ALL 324 rows)
  reports/sub9_sanity/_earnings_downshock_phase5_trades_<window>.csv (per-trade x geom)
  reports/sub9_sanity/_earnings_downshock_continuation_short_trades_demoted_exit1510.csv
  tools/sub9_research/earnings_downshock_continuation_short_cell_lock.json  (if eligible)

USAGE
-----
  # STEP 1+2 -- Discovery sweep + cell lock
  .venv/Scripts/python tools/sub9_research/phase5_sweep_earnings_downshock_continuation_short.py --mode sweep

  # STEP 3 -- ONE-SHOT demoted-window check on the locked cell
  .venv/Scripts/python tools/sub9_research/phase5_sweep_earnings_downshock_continuation_short.py --mode demoted-check
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "tools" / "sub9_research"))

import sanity_earnings_downshock_continuation_short as SAN          # noqa: E402
from tools.sub7_validation.build_per_setup_pnl import calc_fee      # noqa: E402


# =============================================================================
# PRE-REGISTERED GRID -- brief SS9c, commit 0c69ed5.  BINDING.  DO NOT EDIT.
# =============================================================================
PREREG = {
    "adv_tier":      ("low", "mid", "both"),
    "shock_depth":   (-8.0, -10.0, -12.0),
    "announce":      ("AMC", "intraday", "all"),
    "stop_pct":      (None, 2.0, 3.0, 4.0),
    "target_pct":    (None, 1.5, 2.5),
    "incumbent":     {"stop_pct": None, "target_pct": None},
    # lockable-cell rule (A5 + A5-b)
    "n_min_per_era":     100,
    "pooled_pf_min":     1.20,
    "era_expectancy_gt": 0.0,
    "gate_slippage":     "conservative",
    # Stage 5 Step 3
    "overfit_pf_gap":    0.30,
}
assert len(PREREG["adv_tier"]) * len(PREREG["shock_depth"]) * len(PREREG["announce"]) == 27
assert len(PREREG["stop_pct"]) * len(PREREG["target_pct"]) == 12

# FIXED -- broker mechanics, not a swept dimension (brief SS4, lesson #31).
EXIT_BAR_START = "15:10"          # -> the 15:15 print

# flat-rate cross-check (brief SS9b headline measured rates)
FLAT_BPS = {"central": 18.7, "conservative": 27.5}

REPORTS = _REPO / "reports" / "sub9_sanity"
CELLS_CSV = REPORTS / "_earnings_downshock_phase5_cells.csv"
LOCK_JSON = (_REPO / "tools" / "sub9_research"
             / "earnings_downshock_continuation_short_cell_lock.json")
SLIP_DISCOVERY = REPORTS / "_earnings_downshock_slippage_exit1515.csv"
SLIP_DEMOTED = REPORTS / "_earnings_downshock_slippage_demoted_exit1515.csv"
COHORT_DEMOTED = (REPORTS
                  / "_earnings_downshock_continuation_short_trades_demoted_exit1510.csv")

SETUP = "earnings_downshock_continuation_short"


# =============================================================================
# geometry
# =============================================================================
def geometries() -> list[tuple]:
    return [(s, t) for s in PREREG["stop_pct"] for t in PREREG["target_pct"]]


def geom_label(stop, target) -> str:
    s = "none" if stop is None else f"sl{stop:g}"
    t = "none" if target is None else f"tp{target:g}"
    return f"{s}/{t}"


def resolve_geometry(bars: pd.DataFrame, entry_idx: int, exit_idx: int,
                     entry_price: float, stop: float | None, target: float | None):
    """SHORT path walk over bars[entry_idx+1 .. exit_idx].  Returns (px, reason, idx).

    Same-bar pessimism: a bar that touches BOTH levels resolves as the STOP.
    Gap-through: opening beyond a level fills at the OPEN, not the level.
    """
    if stop is None and target is None:
        return float(bars["close"].iloc[exit_idx]), "eod", exit_idx

    stop_px = entry_price * (1.0 + stop / 100.0) if stop is not None else None
    tgt_px = entry_price * (1.0 - target / 100.0) if target is not None else None

    o = bars["open"].values
    h = bars["high"].values
    lo = bars["low"].values
    for i in range(entry_idx + 1, exit_idx + 1):
        hit_stop = stop_px is not None and h[i] >= stop_px
        hit_tgt = tgt_px is not None and lo[i] <= tgt_px
        if hit_stop:                                    # PESSIMISM: stop wins the tie
            return (float(o[i]) if o[i] >= stop_px else float(stop_px)), "stop", i
        if hit_tgt:
            return (float(o[i]) if o[i] <= tgt_px else float(tgt_px)), "target", i
    return float(bars["close"].iloc[exit_idx]), "eod", exit_idx


# =============================================================================
# simulation
# =============================================================================
def simulate(ev: pd.DataFrame, g: dict) -> pd.DataFrame:
    """One row per (trade, geometry).  Construction is 100% from the locked cohort."""
    ENTRY_T = pd.Timestamp("1900-01-01 09:15").time()
    EXIT_T = pd.Timestamp(f"1900-01-01 {EXIT_BAR_START}").time()
    NOTIONAL = SAN.LOCKED_FILTERS["notional_inr"]
    geoms = geometries()

    rows, drops = [], {"nobars": 0, "short": 0, "badstart": 0, "circuit": 0,
                       "noexit": 0, "badqty": 0}
    for r in ev.itertuples(index=False):
        b = g.get((r.symbol, r.entry_date))
        if b is None or len(b) == 0:
            drops["nobars"] += 1
            continue
        b = b.sort_values("date").reset_index(drop=True)
        if len(b) < SAN.LOCKED_FILTERS["min_bars_in_session"]:
            drops["short"] += 1
            continue
        if b["date"].iloc[0].time() != ENTRY_T:
            drops["badstart"] += 1
            continue
        cf = SAN.circuit_blocked_strict(b, float(r.react_close))
        if cf["circuit_blocked"]:
            drops["circuit"] += 1
            continue

        entry_price = float(b["close"].iloc[0])
        hit = np.flatnonzero(b["date"].dt.time.values == EXIT_T)
        if hit.size == 0:
            drops["noexit"] += 1
            continue
        exit_idx = int(hit[0])
        if exit_idx <= 0:
            drops["noexit"] += 1
            continue
        if not (entry_price > 0):
            drops["badqty"] += 1
            continue
        qty = int(NOTIONAL // entry_price)
        if qty < 1:
            drops["badqty"] += 1
            continue

        base = dict(
            signal_date=r.signal_date.date(), symbol=f"NSE:{r.symbol}",
            reaction_date=r.signal_date.date(), entry_date=r.entry_date.date(),
            side="SHORT", entry_price=round(entry_price, 4), qty=qty,
            react_move_pct=round(float(r.react_move), 4),
            react_close=round(float(r.react_close), 4),
            adv20=float(r.adv20) if pd.notna(r.adv20) else np.nan,
            adv_tier=r.adv_tier, adv_tier_win=r.adv_tier_win,
            cap_segment=r.cap_segment, announce_time_class=r.announce_time_class,
            year=int(r.signal_date.year), n_bars=len(b),
            day_high=round(float(b["high"].max()), 4),
            day_low=round(float(b["low"].min()), 4),
            day_close=round(float(b["close"].iloc[-1]), 4),
            open_gap_pct=round(cf["open_gap_pct"], 4),
            entry_bar_volume=cf["entry_bar_volume"],
        )
        for stop, target in geoms:
            xpx, reason, xidx = resolve_geometry(b, 0, exit_idx, entry_price, stop, target)
            rec = dict(base)
            rec.update(
                geom=geom_label(stop, target),
                stop_pct=np.nan if stop is None else stop,
                target_pct=np.nan if target is None else target,
                exit_price=round(xpx, 4), exit_reason=reason,
                exit_ts=b["date"].iloc[xidx], n_bars_held=xidx,
                same_bar=False,
                pnl_pct=(entry_price - xpx) / entry_price * 100.0,
            )
            rows.append(rec)
    tr = pd.DataFrame(rows)
    print(f"  simulated {tr['symbol'].nunique() if len(tr) else 0} symbols; "
          f"{len(tr)//max(len(geoms),1)} trades x {len(geoms)} geometries = {len(tr)} rows")
    print(f"  drops: {drops}")
    return tr


def price_trades(tr: pd.DataFrame, slip: pd.DataFrame | None) -> pd.DataFrame:
    """Attach real Zerodha MIS-short fees + measured/flat slippage nets."""
    tr = tr.copy()
    if slip is not None and len(slip):
        s = slip[["signal_date", "symbol", "perside_blend_bps", "perside_cons_bps"]].copy()
        s["signal_date"] = pd.to_datetime(s["signal_date"]).dt.date
        s = s.drop_duplicates(["signal_date", "symbol"])
        tr["signal_date"] = pd.to_datetime(tr["signal_date"]).dt.date
        tr = tr.merge(s, on=["signal_date", "symbol"], how="left")
    else:
        tr["perside_blend_bps"] = np.nan
        tr["perside_cons_bps"] = np.nan

    # fall back to the tier mean of the MEASURED distribution where a trade is unmeasured
    for col in ("perside_blend_bps", "perside_cons_bps"):
        med = tr.groupby("adv_tier")[col].transform("median")
        tr[col] = tr[col].fillna(med).fillna(tr[col].median())

    for tag, bps_series in (
        ("central",      tr["perside_blend_bps"]),
        ("conservative", tr["perside_cons_bps"]),
        ("centralFlat",      pd.Series(FLAT_BPS["central"], index=tr.index)),
        ("conservativeFlat", pd.Series(FLAT_BPS["conservative"], index=tr.index)),
    ):
        sl = bps_series.astype(float) / 10_000.0
        e = tr["entry_price"] * (1.0 - sl)     # short sells LOWER
        x = tr["exit_price"] * (1.0 + sl)      # covers HIGHER
        gross = (e - x) * tr["qty"]
        fee = [calc_fee(float(a), float(bb), int(qq), "SELL", 1.0)
               for a, bb, qq in zip(e, x, tr["qty"])]
        net = gross - np.asarray(fee)
        tr[f"slip_bps_{tag}"] = bps_series.astype(float)
        tr[f"fees_{tag}"] = np.round(fee, 2)
        tr[f"net_pnl_{tag}"] = np.round(net, 2)
        tr[f"net_pct_{tag}"] = net / (tr["entry_price"] * tr["qty"]) * 100.0
    return tr


# =============================================================================
# stats
# =============================================================================
def pf_of(s) -> float:
    s = np.asarray(s, dtype=float)
    gp = s[s > 0].sum()
    ls = -s[s < 0].sum()
    if ls <= 0:
        return float("inf") if gp > 0 else float("nan")
    return float(gp / ls)


def stat(sub: pd.DataFrame, tag: str) -> dict:
    n = len(sub)
    if n == 0:
        return dict(n=0, exp_pct=np.nan, pf=np.nan, win=np.nan, t=np.nan, net_inr=np.nan)
    q = sub[f"net_pct_{tag}"]
    p = sub[f"net_pnl_{tag}"]
    sd = float(q.std(ddof=1)) if n > 1 else np.nan
    return dict(
        n=n, exp_pct=float(q.mean()), pf=pf_of(p),
        win=float((p > 0).mean()), net_inr=float(p.sum()),
        t=(float(q.mean()) / (sd / np.sqrt(n))) if (n > 1 and sd and sd > 0) else np.nan,
    )


def cell_mask(tr: pd.DataFrame, adv: str, shock: float, ann: str) -> pd.Series:
    m = tr["react_move_pct"] <= shock
    if adv == "low":
        m &= tr["adv_tier"] == "adv_low"
    elif adv == "mid":
        m &= tr["adv_tier"] == "adv_mid"
    if ann != "all":
        m &= tr["announce_time_class"] == ann
    return m


def build_cell_table(tr: pd.DataFrame, era_col: str, eras: tuple) -> pd.DataFrame:
    out = []
    for adv in PREREG["adv_tier"]:
        for shock in PREREG["shock_depth"]:
            for ann in PREREG["announce"]:
                base_m = cell_mask(tr, adv, shock, ann)
                for stop, target in geometries():
                    lbl = geom_label(stop, target)
                    sub = tr[base_m & (tr["geom"] == lbl)]
                    row = dict(adv_tier=adv, shock_depth=shock, announce=ann,
                               geom=lbl, stop_pct=stop, target_pct=target,
                               cell=f"{adv}/{shock:g}/{ann}")
                    for tag in ("central", "conservative",
                                "centralFlat", "conservativeFlat"):
                        s = stat(sub, tag)
                        for k, v in s.items():
                            row[f"{k}_{tag}" if k != "n" else "n"] = v
                    for e in eras:
                        se = sub[sub[era_col] == e]
                        s = stat(se, "conservative")
                        row[f"n_{e}"] = s["n"]
                        row[f"exp_{e}_cons"] = s["exp_pct"]
                        row[f"pf_{e}_cons"] = s["pf"]
                        sc = stat(se, "central")
                        row[f"exp_{e}_central"] = sc["exp_pct"]
                        row[f"pf_{e}_central"] = sc["pf"]
                    if len(sub):
                        row["exit_mix"] = "|".join(
                            f"{k}:{v}" for k, v in
                            sub["exit_reason"].value_counts().sort_index().items())
                    else:
                        row["exit_mix"] = ""
                    out.append(row)
    return pd.DataFrame(out)


def apply_lockable_rule(cells: pd.DataFrame, eras: tuple) -> pd.DataFrame:
    """A5 + A5-b, applied MECHANICALLY.  No relaxation, no salvage."""
    c = cells.copy()
    ok_n = np.ones(len(c), bool)
    ok_exp = np.ones(len(c), bool)
    for e in eras:
        ok_n &= (c[f"n_{e}"] >= PREREG["n_min_per_era"]).values
        ok_exp &= (c[f"exp_{e}_cons"] > PREREG["era_expectancy_gt"]).values
    c["gate_n"] = ok_n
    c["gate_era_expectancy"] = ok_exp
    c["gate_pooled_pf"] = (c["pf_conservative"] >= PREREG["pooled_pf_min"]).values
    c["eligible"] = c["gate_n"] & c["gate_era_expectancy"] & c["gate_pooled_pf"]
    # informational only -- the pre-registered gate slippage is CONSERVATIVE
    c["eligible_at_central"] = (
        ok_n
        & np.all([(c[f"exp_{e}_central"] > 0).values for e in eras], axis=0)
        & (c["pf_central"] >= PREREG["pooled_pf_min"]).values)
    c["era_gap_exp"] = (c[[f"exp_{e}_cons" for e in eras]].max(axis=1)
                        - c[[f"exp_{e}_cons" for e in eras]].min(axis=1))
    c["era_gap_pf"] = (c[[f"pf_{e}_cons" for e in eras]].max(axis=1)
                       - c[[f"pf_{e}_cons" for e in eras]].min(axis=1))
    return c


def incumbent_test(cells: pd.DataFrame, cell_id: str, eras: tuple) -> pd.DataFrame:
    """Strict SS9c displacement rule, applied mechanically.

    A geometry displaces `none/none` ONLY if it is STRICTLY better in BOTH eras.
    Ties -> incumbent.  Pooled numbers are IRRELEVANT to this test by construction.
    """
    sub = cells[cells["cell"] == cell_id].set_index("geom")
    inc = sub.loc["none/none"]
    rows = []
    for gl, r in sub.iterrows():
        better = all(r[f"exp_{e}_cons"] > inc[f"exp_{e}_cons"] for e in eras)
        better_pf = all(r[f"pf_{e}_cons"] > inc[f"pf_{e}_cons"] for e in eras)
        rows.append(dict(
            geom=gl, is_incumbent=(gl == "none/none"),
            **{f"exp_{e}_cons": r[f"exp_{e}_cons"] for e in eras},
            **{f"pf_{e}_cons": r[f"pf_{e}_cons"] for e in eras},
            pooled_exp_cons=r["exp_pct_conservative"], pooled_pf_cons=r["pf_conservative"],
            pooled_exp_central=r["exp_pct_central"], pooled_pf_central=r["pf_central"],
            exit_mix=r["exit_mix"],
            displaces_incumbent_expectancy=better and gl != "none/none",
            displaces_incumbent_pf=better_pf and gl != "none/none",
        ))
    return pd.DataFrame(rows)


# =============================================================================
def print_cells(cells: pd.DataFrame, eras: tuple, only_incumbent_geom: bool = False) -> None:
    c = cells[cells["geom"] == "none/none"] if only_incumbent_geom else cells
    hdr = (f"  {'cell':<18}{'geom':<12}{'n':>5}"
           + "".join(f"{'n_'+str(e):>7}" for e in eras)
           + f"{'expC':>9}{'PFc':>7}{'winC':>7}{'tC':>7}"
           + f"{'expK':>9}{'PFk':>7}{'winK':>7}{'tK':>7}"
           + "".join(f"{'exp'+str(e)+'K':>10}" for e in eras) + "  elig")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for _, r in c.iterrows():
        print(f"  {r['cell']:<18}{r['geom']:<12}{int(r['n']):>5}"
              + "".join(f"{int(r['n_'+str(e)]):>7}" for e in eras)
              + f"{r['exp_pct_central']:>9.3f}{r['pf_central']:>7.2f}"
              f"{r['win_central']:>7.3f}{r['t_central']:>7.2f}"
              + f"{r['exp_pct_conservative']:>9.3f}{r['pf_conservative']:>7.2f}"
              f"{r['win_conservative']:>7.3f}{r['t_conservative']:>7.2f}"
              + "".join(f"{r['exp_'+str(e)+'_cons']:>10.3f}" for e in eras)
              + f"  {'YES' if r['eligible'] else '.'}")


# =============================================================================
def run_sweep() -> None:
    print("=" * 110)
    print(f"STAGE 5 STEP 1+2 -- pre-registered cell + geometry sweep  [{SETUP}]")
    print("  grid = brief SS9c (commit 0c69ed5), BINDING.  27 cells x 12 geometries.")
    print("=" * 110)

    px, ev, g, cuts = SAN.build_cohort("discovery")
    tr = simulate(ev, g)
    assert pd.to_datetime(tr["entry_date"]).max() < SAN.FRESH_POOL_START

    slip = pd.read_csv(SLIP_DISCOVERY) if SLIP_DISCOVERY.exists() else None
    if slip is None:
        raise RuntimeError(f"missing measured slippage: {SLIP_DISCOVERY}")
    tr = price_trades(tr, slip)
    cov = tr.groupby("geom")["perside_blend_bps"].apply(lambda s: s.notna().mean()).min()
    print(f"  measured-slippage coverage: {cov:.3f}   "
          f"CENTRAL mean {tr['slip_bps_central'].mean():.1f} bp/side, "
          f"CONSERVATIVE mean {tr['slip_bps_conservative'].mean():.1f} bp/side")

    out_tr = REPORTS / "_earnings_downshock_phase5_trades_discovery.csv"
    tr.to_csv(out_tr, index=False)
    print(f"  wrote {out_tr}  rows={len(tr)}")

    eras = tuple(sorted(tr["year"].unique()))
    print(f"\n  WITHIN-DEVELOPMENT era split for the A5 gate: {eras}")
    print("  (Discovery alone IS era_A; the true era_B test is Step 3 -- demoted window.)")

    cells = build_cell_table(tr, "year", eras)
    cells = apply_lockable_rule(cells, eras)
    cells.insert(0, "window", "discovery_2023_24")
    CELLS_CSV.parent.mkdir(parents=True, exist_ok=True)
    cells.to_csv(CELLS_CSV, index=False)
    print(f"  wrote {CELLS_CSV}  rows={len(cells)} (ALL cells, dead ones included)")

    print("\n" + "=" * 110)
    print("ALL 27 CELLS at the INCUMBENT geometry none/none "
          "(C = CENTRAL measured slip, K = CONSERVATIVE measured slip)")
    print("=" * 110)
    print_cells(cells, eras, only_incumbent_geom=True)

    print("\n" + "=" * 110)
    print("LOCKABLE-CELL RULE (A5 + A5-b) -- applied mechanically, no relaxation")
    print(f"  net expectancy > 0 in BOTH of {eras} at CONSERVATIVE slippage")
    print(f"  AND n >= {PREREG['n_min_per_era']} per era  AND pooled PF >= "
          f"{PREREG['pooled_pf_min']} (at CONSERVATIVE)")
    print("=" * 110)
    el = cells[cells["eligible"]]
    print(f"  eligible (cell x geometry) combinations: {len(el)} / {len(cells)}")
    if len(el) == 0:
        print("\n  *** NO ELIGIBLE CELL -> KILL (Stage 5 anti-salvage). ***")
        near = cells.copy()
        near["_gates"] = (near["gate_n"].astype(int) + near["gate_era_expectancy"].astype(int)
                          + near["gate_pooled_pf"].astype(int))
        near = near.sort_values(["_gates", "pf_conservative"], ascending=[False, False])
        print("\n  TOP-10 NEAREST MISS:")
        print_cells(near.head(10), eras)
        return
    print_cells(el, eras)

    # ---- STABILITY-FIRST selection among eligible CELLS (not top-PF) ----
    print("\n" + "-" * 110)
    print("CELL SELECTION -- stability-first (smallest era gap at comparable PF),")
    print("NEVER top-PF-only (memory: feedback_cell_sweep_stability_over_top_pf)")
    print("-" * 110)
    elig_cells = sorted(el["cell"].unique())
    print(f"  cells with >=1 eligible geometry: {elig_cells}")
    summ = (el.groupby("cell")
              .agg(n=("n", "max"), best_pf=("pf_conservative", "max"),
                   min_era_gap=("era_gap_exp", "min"))
              .reset_index().sort_values(["min_era_gap", "best_pf"],
                                         ascending=[True, False]))
    print(summ.to_string(index=False))
    locked_cell = str(summ.iloc[0]["cell"])
    print(f"\n  -> LOCKED CELL (smallest era gap): {locked_cell}")

    # ---- GEOMETRY: incumbent none/none wins ties; strict both-eras displacement ----
    print("\n" + "-" * 110)
    print(f"GEOMETRY vs the none/none INCUMBENT inside cell {locked_cell}")
    print("  displacement requires STRICTLY better in BOTH eras (pooled is irrelevant)")
    print("-" * 110)
    it = incumbent_test(cells, locked_cell, eras)
    cols = (["geom", "is_incumbent"] + [f"exp_{e}_cons" for e in eras]
            + [f"pf_{e}_cons" for e in eras]
            + ["pooled_exp_cons", "pooled_pf_cons", "pooled_exp_central",
               "pooled_pf_central", "displaces_incumbent_expectancy",
               "displaces_incumbent_pf", "exit_mix"])
    print(it[cols].to_string(index=False))
    disp = it[it["displaces_incumbent_expectancy"]]
    if len(disp) == 0:
        locked_geom = "none/none"
        print("\n  -> NO geometry is strictly better in BOTH eras. "
              "INCUMBENT none/none HOLDS (SS9c tie rule).")
    else:
        # among strict displacers, take the stability-first one (smallest era gap)
        gs = cells[(cells["cell"] == locked_cell)
                   & cells["geom"].isin(disp["geom"])].sort_values("era_gap_exp")
        locked_geom = str(gs.iloc[0]["geom"])
        print(f"\n  -> {len(disp)} geometry(ies) strictly better in BOTH eras: "
              f"{list(disp['geom'])}")
        print(f"  -> DISPLACES the incumbent; locked geometry = {locked_geom} "
              f"(stability-first among displacers)")

    lock_row = cells[(cells["cell"] == locked_cell) & (cells["geom"] == locked_geom)].iloc[0]
    print("\n" + "=" * 110)
    print("LOCKED (cell x geometry) -- DISCOVERY STATS")
    print("=" * 110)
    for tag, lbl in (("central", "CENTRAL measured (per-trade)"),
                     ("conservative", "CONSERVATIVE measured (per-trade)"),
                     ("centralFlat", f"CENTRAL flat {FLAT_BPS['central']}bp"),
                     ("conservativeFlat", f"CONSERV flat {FLAT_BPS['conservative']}bp")):
        print(f"  {lbl:<34s} n={int(lock_row['n']):4d}  exp={lock_row['exp_pct_'+tag]:+.4f}%  "
              f"PF={lock_row['pf_'+tag]:.4f}  win={lock_row['win_'+tag]:.3f}  "
              f"t={lock_row['t_'+tag]:+.2f}  net=Rs{lock_row['net_inr_'+tag]:,.0f}")
    for e in eras:
        print(f"  {e}: n={int(lock_row['n_'+str(e)]):4d}  "
              f"expK={lock_row['exp_'+str(e)+'_cons']:+.4f}%  PFk={lock_row['pf_'+str(e)+'_cons']:.4f}  "
              f"| expC={lock_row['exp_'+str(e)+'_central']:+.4f}%  PFc={lock_row['pf_'+str(e)+'_central']:.4f}")

    payload = {
        "setup": SETUP,
        "stage": "phase5_cell_lock",
        "locked_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prereg_source": "specs/2026-07-29-brief-earnings_downshock_continuation_short.md "
                         "SS9c (commit 0c69ed5)",
        "window": "discovery_2023_24",
        "within_development_era_split": [int(e) for e in eras],
        "true_era_B_test": "step3_demoted_window_2025_01_to_2026_04",
        "locked_cell": {
            "adv_tier": str(lock_row["adv_tier"]),
            "shock_depth_pct": float(lock_row["shock_depth"]),
            "announce_class": str(lock_row["announce"]),
        },
        "locked_geometry": {
            "label": locked_geom,
            "stop_pct": None if pd.isna(lock_row["stop_pct"]) else float(lock_row["stop_pct"]),
            "target_pct": None if pd.isna(lock_row["target_pct"]) else float(lock_row["target_pct"]),
            "incumbent_held": bool(locked_geom == "none/none"),
            "displacement_rule": "strictly better net expectancy in BOTH eras at "
                                 "CONSERVATIVE slippage; ties -> incumbent",
        },
        "fixed_not_swept": {
            "signal_pct": SAN.LOCKED_FILTERS["shock_threshold_pct"],
            "dedupe_key": list(SAN.LOCKED_FILTERS["dedupe_key"]),
            "entry": "close of the 09:15-09:20 5m bar (the 09:20 print)",
            "exit": "close of the 15:10-15:15 5m bar (the 15:15 print) -- BROKER MECHANICS",
            "universe": "low+mid ADV tercile + ProductionUniverseGate(entry_date)",
            "exclusions": "NSE ASM any stage / BSE GSM on entry date; circuit-blocked sessions",
            "fees": "real Zerodha MIS short round trip (calc_fee, SELL, lev=1.0)",
            "slippage": "measured per-trade (measure_slippage_earnings_downshock.py, "
                        "15:15 exit): CENTRAL perside_blend_bps, CONSERVATIVE perside_cons_bps",
            "notional_inr": SAN.LOCKED_FILTERS["notional_inr"],
        },
        "adv_tercile_cut_points_rs_turnover": {
            "q33": float(cuts[0]), "q66": float(cuts[1]),
            "note": "FROZEN. Step 3 imposes these on the demoted window rather than "
                    "re-fitting terciles there -- 'unchanged' taken literally.",
        },
        "lockable_rule": {
            "statistic": "net %/trade (ABSOLUTE) -- single-leg cash SHORT, A5-b",
            "era_expectancy_gt": PREREG["era_expectancy_gt"],
            "n_min_per_era": PREREG["n_min_per_era"],
            "pooled_pf_min": PREREG["pooled_pf_min"],
            "gate_slippage": PREREG["gate_slippage"],
            "selection": "stability-first (smallest era expectancy gap), never top-PF-only",
        },
        "discovery_stats": {
            tag: dict(n=int(lock_row["n"]), exp_pct=float(lock_row[f"exp_pct_{tag}"]),
                      pf=float(lock_row[f"pf_{tag}"]), win=float(lock_row[f"win_{tag}"]),
                      t=float(lock_row[f"t_{tag}"]), net_inr=float(lock_row[f"net_inr_{tag}"]))
            for tag in ("central", "conservative", "centralFlat", "conservativeFlat")
        },
        "discovery_per_year": {
            str(e): dict(n=int(lock_row[f"n_{e}"]),
                         exp_pct_conservative=float(lock_row[f"exp_{e}_cons"]),
                         pf_conservative=float(lock_row[f"pf_{e}_cons"]),
                         exp_pct_central=float(lock_row[f"exp_{e}_central"]),
                         pf_central=float(lock_row[f"pf_{e}_central"]))
            for e in eras
        },
        "eligible_cell_table": el[
            ["cell", "geom", "n"] + [f"n_{e}" for e in eras]
            + [f"exp_{e}_cons" for e in eras] + [f"pf_{e}_cons" for e in eras]
            + ["exp_pct_central", "pf_central", "win_central", "t_central",
               "exp_pct_conservative", "pf_conservative", "win_conservative",
               "t_conservative", "era_gap_exp", "exit_mix"]
        ].to_dict("records"),
        "n_cells_evaluated": int(len(cells)),
        "n_eligible": int(len(el)),
        "evidence": str(CELLS_CSV.relative_to(_REPO)).replace("\\", "/"),
    }
    LOCK_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n  wrote cell lock -> {LOCK_JSON}")


# =============================================================================
def run_demoted_check() -> None:
    print("=" * 110)
    print(f"STAGE 5 STEP 3 -- ONE-SHOT demoted-window check  [{SETUP}]")
    print("  window: reaction dates 2025-01-01 .. 2026-04-30 (amendment A1: demoted")
    print("  development data -- it still BURNS.  ONE run, no iteration, no re-sweep.)")
    print("  Reaction dates >= 2026-05-01 are NEVER touched.")
    print("=" * 110)
    if not LOCK_JSON.exists():
        raise RuntimeError("no cell lock -- run --mode sweep first (or it was a KILL)")
    lock = json.loads(LOCK_JSON.read_text(encoding="utf-8"))
    lc = lock["locked_cell"]
    lg = lock["locked_geometry"]
    cuts = (lock["adv_tercile_cut_points_rs_turnover"]["q33"],
            lock["adv_tercile_cut_points_rs_turnover"]["q66"])
    print(f"\n  LOCKED CELL      : {lc}")
    print(f"  LOCKED GEOMETRY  : {lg['label']}")
    print(f"  FROZEN ADV cuts  : q33={cuts[0]:,.0f}  q66={cuts[1]:,.0f}")

    px, ev, g, _ = SAN.build_cohort("demoted", adv_cuts=cuts)
    tr = simulate(ev, g)
    assert pd.to_datetime(tr["entry_date"]).max() < SAN.FRESH_POOL_START, "FRESH POOL"

    # ---- emit the none/none base cohort in sanity schema so the slippage measurement
    #      script can be reused UNCHANGED on this window ----
    base = tr[tr["geom"] == "none/none"].copy()
    base["fees_0bp"] = [calc_fee(float(a), float(b), int(q), "SELL", 1.0)
                        for a, b, q in zip(base["entry_price"], base["exit_price"], base["qty"])]
    base.to_csv(COHORT_DEMOTED, index=False)
    print(f"\n  wrote demoted base cohort -> {COHORT_DEMOTED}  n={len(base)}")

    if not SLIP_DEMOTED.exists():
        print("\n--- measuring per-trade slippage on the DEMOTED cohort "
              "(measure_slippage_earnings_downshock.py, 15:15 exit) ---", flush=True)
        cmd = [sys.executable,
               str(_REPO / "tools" / "sub9_research" / "measure_slippage_earnings_downshock.py"),
               "--cohort-csv", str(COHORT_DEMOTED.relative_to(_REPO)).replace("\\", "/"),
               "--out-csv", str(SLIP_DEMOTED.relative_to(_REPO)).replace("\\", "/"),
               "--exit-fill-minute", "15:14",
               "--exit-5m-window", "15:10,15:14",
               "--exit-window", "15:05,15:14",
               "--exit-roll-window", "14:15,15:14"]
        r = subprocess.run(cmd, cwd=str(_REPO), capture_output=True, text=True)
        tail = "\n".join(r.stdout.splitlines()[-4:])
        print(f"  slippage measurement rc={r.returncode}\n{tail}")
        if r.returncode != 0:
            print(r.stderr[-2000:])
            raise RuntimeError("slippage measurement failed")

    slip = pd.read_csv(SLIP_DEMOTED)
    tr = price_trades(tr, slip)
    print(f"  measured slippage: CENTRAL mean {tr['slip_bps_central'].mean():.1f} bp/side, "
          f"CONSERVATIVE mean {tr['slip_bps_conservative'].mean():.1f} bp/side")
    tr.to_csv(REPORTS / "_earnings_downshock_phase5_trades_demoted.csv", index=False)

    # ---- the LOCKED cell, UNCHANGED ----
    adv = lc["adv_tier"]
    m = cell_mask(tr, adv, lc["shock_depth_pct"], lc["announce_class"])
    sub = tr[m & (tr["geom"] == lg["label"])].copy()
    print(f"\n  demoted trades in the locked cell x geometry: n={len(sub)}  "
          f"symbols={sub['symbol'].nunique()}  sessions={sub['entry_date'].nunique()}")

    print("\n" + "=" * 110)
    print("DEMOTED-WINDOW RESULT (locked cell, unchanged)")
    print("=" * 110)
    res = {}
    for tag, lbl in (("central", "CENTRAL measured (per-trade)"),
                     ("conservative", "CONSERVATIVE measured (per-trade)"),
                     ("centralFlat", f"CENTRAL flat {FLAT_BPS['central']}bp"),
                     ("conservativeFlat", f"CONSERV flat {FLAT_BPS['conservative']}bp")):
        s = stat(sub, tag)
        res[tag] = s
        print(f"  {lbl:<34s} n={s['n']:4d}  exp={s['exp_pct']:+.4f}%  PF={s['pf']:.4f}  "
              f"win={s['win']:.3f}  t={s['t']:+.2f}  net=Rs{s['net_inr']:,.0f}")

    print("\n  PER YEAR:")
    per_year = {}
    for yr, sy in sub.groupby("year"):
        row = {}
        for tag in ("central", "conservative"):
            s = stat(sy, tag)
            row[tag] = s
            print(f"    {yr} {tag:<14s} n={s['n']:4d}  exp={s['exp_pct']:+.4f}%  "
                  f"PF={s['pf']:.4f}  win={s['win']:.3f}  t={s['t']:+.2f}")
        per_year[str(yr)] = row

    print("\n  EXIT MIX:", dict(sub["exit_reason"].value_counts()))

    # ---- robustness (same single run): within-window ADV terciles ----
    if adv == "both":
        sub_w = tr[(tr["react_move_pct"] <= lc["shock_depth_pct"])
                   & (tr["geom"] == lg["label"])
                   & (tr["adv_tier_win"].isin(["adv_low", "adv_mid"]))]
    else:
        want = "adv_low" if adv == "low" else "adv_mid"
        sub_w = tr[(tr["react_move_pct"] <= lc["shock_depth_pct"])
                   & (tr["geom"] == lg["label"]) & (tr["adv_tier_win"] == want)]
    if lc["announce_class"] != "all":
        sub_w = sub_w[sub_w["announce_time_class"] == lc["announce_class"]]
    sw = stat(sub_w, "conservative")
    print(f"\n  ROBUSTNESS (not the verdict): within-window ADV terciles instead of the "
          f"frozen Discovery cuts -> n={sw['n']}  exp={sw['exp_pct']:+.4f}%  PF={sw['pf']:.4f}")

    # ---- overfit gate ----
    print("\n" + "=" * 110)
    print("OVERFIT GATE (lifecycle Stage 5 Step 3): |PF_disc - PF_demoted| > "
          f"{PREREG['overfit_pf_gap']:.2f} == OVERFIT")
    print("=" * 110)
    gaps = {}
    for tag in ("central", "conservative"):
        pd_ = float(lock["discovery_stats"][tag]["pf"])
        pm = res[tag]["pf"]
        gaps[tag] = abs(pd_ - pm)
        print(f"  {tag:<14s} PF_disc {pd_:.4f}  PF_demoted {pm:.4f}  "
              f"|gap| {gaps[tag]:.4f}  -> "
              f"{'OVERFIT' if gaps[tag] > PREREG['overfit_pf_gap'] else 'within tolerance'}")

    out = {
        "setup": SETUP, "stage": "phase5_demoted_check",
        "run_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "locked_cell": lc, "locked_geometry": lg,
        "window": "2025-01-01..2026-04-30 (demoted development, A1)",
        "n": int(len(sub)),
        "stats": {k: {kk: (None if pd.isna(vv) else float(vv)) for kk, vv in v.items()}
                  for k, v in res.items()},
        "per_year": {y: {k: {kk: (None if pd.isna(vv) else float(vv))
                             for kk, vv in v.items()} for k, v in r.items()}
                     for y, r in per_year.items()},
        "pf_gap_vs_discovery": gaps,
        "overfit_threshold": PREREG["overfit_pf_gap"],
        "exit_mix": {str(k): int(v) for k, v in sub["exit_reason"].value_counts().items()},
    }
    p = REPORTS / "_earnings_downshock_phase5_demoted_result.json"
    p.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\n  wrote {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=("sweep", "demoted-check"))
    a = ap.parse_args()
    if a.mode == "sweep":
        run_sweep()
    else:
        run_demoted_check()
