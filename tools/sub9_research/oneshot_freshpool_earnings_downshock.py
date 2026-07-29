"""oneshot_freshpool_earnings_downshock.py -- the A1 DECISIVE fresh-pool one-shot.

Candidate : `earnings_downshock_continuation_short`
Brief     : specs/2026-07-29-brief-earnings_downshock_continuation_short.md
            SS9d = the FREEZE + the PRE-REGISTERED decision rule (BINDING).
Cell lock : tools/sub9_research/earnings_downshock_continuation_short_cell_lock.json
Freeze    : commit 33d2046 (declared 2026-07-29, BEFORE this run existed).

WHAT THIS IS
------------
ONE run over signals dated 2026-05-01 -> the latest session on disk.  No iteration, no
threshold reinterpretation, no re-sweep.  The construction is taken verbatim from the
freeze; the verdict bands were fixed in SS9d before any fresh-pool number was computed.

MANDATORY SEQUENCING (this is why the script has TWO steps)
------------------------------------------------------------
  --step count   FIRE COUNT ONLY.  Runs the full frozen funnel and reports n.
                 Computes NO returns, NO PnL, NO expectancy.  The output CSV carries
                 identifiers only -- deliberately no entry/exit price, so a
                 power-blocked run cannot leak an outcome statistic.
  --step full    The one-shot P&L.  REFUSES to run when n < 40 (the pre-registered
                 power gate).  You cannot skip `count`; `full` re-derives n and asserts
                 the gate itself.

The power gate exists so that a thin fresh pool does NOT burn the verdict: n<40 is
POWER-BLOCKED (report counts, project the month n reaches 40, stop), not a KILL.

PRE-REGISTERED VERDICT BANDS (SS9d -- applied to the CONSERVATIVE number, NOT CENTRAL,
because the demoted window showed 2026 Jan-Apr decaying to net-negative at conservative
slippage):
    net >= +0.15%/trade -> PASS      (detector build + paper)
    0 to +0.15%         -> MARGINAL  (hold, re-shoot at larger n, no detector work)
    < 0                 -> KILL

FROZEN CONSTRUCTION (nothing here is a choice made today)
----------------------------------------------------------
  signal    : earnings reaction-day close move <= -8.0%
  dedupe    : (symbol, reaction_date)
  entry     : T+1, close of the 09:15-09:20 5m bar (the 09:20 print), SHORT
  exit      : close of the 15:10-15:15 5m bar (the 15:15 print) -- BROKER MECHANICS
  geometry  : none/none (the incumbent; nothing displaced it at Phase 5)
  universe  : ADV tercile low+mid using the FROZEN Discovery cut points
              (q33 Rs 7,89,04,012 / q66 Rs 25,84,13,059) -- terciles are NOT re-fitted
              on the fresh window; the tier definition is part of the lock
              + ProductionUniverseGate(entry_date) + no NSE ASM / BSE GSM on entry
              + not circuit-blocked + 5m entry session present
  fees      : real Zerodha MIS short round trip (calc_fee, SELL, lev=1.0), Rs 1L notional
  slippage  : RE-MEASURED on this cohort (measure_slippage_earnings_downshock.py).
              The demoted window measured WIDER than Discovery (19.3/29.5 vs 18.1/26.6),
              so it is measured, never assumed.

THE FRESH-POOL GUARD
--------------------
`sanity_earnings_downshock_continuation_short.build_cohort` refuses any window touching
2026-05-01 by default.  That guard is INTACT.  This script bypasses it with an explicit
`allow_fresh_pool=True` plus a mandatory `--i-am-burning-the-fresh-pool` CLI flag, and
the bypass prints a logged banner naming the freeze.  Deliberate and auditable, not deleted.

Output:
  reports/sub9_sanity/_earnings_downshock_freshpool_oneshot.csv
  reports/sub9_sanity/_earnings_downshock_freshpool_result.json
  reports/sub9_sanity/_earnings_downshock_freshpool_cohort.csv   (full step only)

Run:
  .venv/Scripts/python tools/sub9_research/oneshot_freshpool_earnings_downshock.py \
      --step count --i-am-burning-the-fresh-pool
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

import sanity_earnings_downshock_continuation_short as SAN            # noqa: E402
import phase5_sweep_earnings_downshock_continuation_short as P5       # noqa: E402
from tools.sub7_validation.build_per_setup_pnl import calc_fee        # noqa: E402

SETUP = "earnings_downshock_continuation_short"
FREEZE_COMMIT = "33d2046"

# ---- PRE-REGISTERED, brief SS9d.  BINDING.  DO NOT EDIT. ----
PREREG = {
    "window_start": "2026-05-01",
    "power_gate_n_min": 40,
    "verdict_basis": "conservative",     # NOT central -- SS9d is explicit
    "pass_at_or_above": 0.15,            # net %/trade
    "kill_below": 0.0,
}

REPORTS = _REPO / "reports" / "sub9_sanity"
OUT_CSV = REPORTS / "_earnings_downshock_freshpool_oneshot.csv"
RESULT_JSON = REPORTS / "_earnings_downshock_freshpool_result.json"
COHORT_CSV = REPORTS / "_earnings_downshock_freshpool_cohort.csv"
SLIP_CSV = REPORTS / "_earnings_downshock_slippage_freshpool_exit1515.csv"
LOCK_JSON = (_REPO / "tools" / "sub9_research"
             / "earnings_downshock_continuation_short_cell_lock.json")
LEDGER = _REPO / "docs" / "experiment_ledger.jsonl"

REASON = (f"brief SS9d PRE-REGISTERED fresh-pool one-shot; candidate FROZEN at commit "
          f"{FREEZE_COMMIT}; decision rule fixed before this run")


# =============================================================================
def load_lock() -> dict:
    lock = json.loads(LOCK_JSON.read_text(encoding="utf-8"))
    lc, lg = lock["locked_cell"], lock["locked_geometry"]
    cuts = (lock["adv_tercile_cut_points_rs_turnover"]["q33"],
            lock["adv_tercile_cut_points_rs_turnover"]["q66"])
    print(f"  LOCKED CELL      : {lc}")
    print(f"  LOCKED GEOMETRY  : {lg['label']}")
    print(f"  FROZEN ADV cuts  : q33=Rs {cuts[0]:,.0f}   q66=Rs {cuts[1]:,.0f}")
    print("                     (IMPOSED, not re-fitted on the fresh window)")
    return lock


def admit(ev: pd.DataFrame, g: dict) -> tuple[pd.DataFrame, dict]:
    """Bar-level admission -- IDENTICAL checks to P5.simulate, but counts only.

    Returns (admitted keys, drop counts).  Deliberately emits NO price and NO return:
    a power-blocked run must not be able to leak an outcome statistic.
    """
    ENTRY_T = pd.Timestamp("1900-01-01 09:15").time()
    EXIT_T = pd.Timestamp(f"1900-01-01 {P5.EXIT_BAR_START}").time()
    NOTIONAL = SAN.LOCKED_FILTERS["notional_inr"]

    keep, drops = [], {"nobars": 0, "short": 0, "badstart": 0, "circuit": 0,
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
        if hit.size == 0 or int(hit[0]) <= 0:
            drops["noexit"] += 1
            continue
        if not (entry_price > 0) or int(NOTIONAL // entry_price) < 1:
            drops["badqty"] += 1
            continue
        keep.append(dict(
            symbol=r.symbol, reaction_date=r.signal_date.date(),
            entry_date=r.entry_date.date(),
            react_move_pct=round(float(r.react_move), 4),
            adv_tier=r.adv_tier, adv20=float(r.adv20) if pd.notna(r.adv20) else np.nan,
            cap_segment=r.cap_segment, announce_time_class=r.announce_time_class,
            n_bars=len(b), month=str(pd.Timestamp(r.signal_date).to_period("M")),
        ))
    return pd.DataFrame(keep), drops


def build_fresh(lock: dict):
    cuts = (lock["adv_tercile_cut_points_rs_turnover"]["q33"],
            lock["adv_tercile_cut_points_rs_turnover"]["q66"])
    px, ev, g, cuts_used = SAN.build_cohort(
        "freshpool", adv_cuts=cuts, allow_fresh_pool=True, fresh_pool_reason=REASON)
    assert abs(cuts_used[0] - cuts[0]) < 1e-6 and abs(cuts_used[1] - cuts[1]) < 1e-6, \
        "ADV cut points were NOT the frozen ones"
    return px, ev, g


# =============================================================================
# SLIPPAGE ON THE FRESH COHORT -- 5m PROXY, CALIBRATED TO THE 1m MEASUREMENT
# =============================================================================
# `measure_slippage_earnings_downshock.py` is a 1-MINUTE method (Roll spread, tick gcd,
# adjacent-print gaps, minute participation).  `backtest-cache-download/monthly/*_1m.feather`
# STOPS AT 2026-04, so the frozen measurement CANNOT be run on a 2026-05..07 cohort.
#
# Assuming the demoted-window rate would violate the run's own instruction (the demoted
# cohort already measured WIDER than Discovery: 19.5/29.8 vs 18.1/26.6 bp/side), so
# instead the fresh cohort is MEASURED on the inputs that DO exist -- 5-minute bars --
# and that 5m proxy is CALIBRATED against the same 5m proxy computed on the demoted
# cohort, whose true 1m per-side rates are known.  Formally:
#
#     k        = mean(measured_1m_perside on demoted) / mean(proxy_5m_perside on demoted)
#     rate_i   = k * proxy_5m_perside_i          for each fresh-pool trade i
#
# So the LEVEL is anchored on a real 1m measurement and the CROSS-COHORT MOVEMENT and
# per-trade dispersion are measured on this cohort's own tape.  If the fresh pool trades
# in wider/thinner names than the demoted window, k * proxy says so.
#
# The proxy mirrors the frozen method's structure (spread + impact) on 5m inputs:
#   spread proxy : half-range of the fill 5m bar          (M1 analogue)
#   impact       : max(M6a daily sqrt-law, M6b 5m bar-sweep)   -- identical formulas
#   conservative : same, with the Y=1.0 stress impact     (CONSERVATIVE analogue)
# It is a PROXY and is labelled as one everywhere it is reported.
PARKINSON_K = 2.0 * np.sqrt(np.log(2.0))
NOTIONAL = 100_000.0


def _proxy_5m(cohort: pd.DataFrame) -> pd.DataFrame:
    """Per-trade 5m spread+impact proxy (bp/side) for a cohort CSV in sanity schema."""
    c = cohort.copy()
    c["symbol_raw"] = c["symbol"].astype(str).str.replace("NSE:", "", regex=False)
    c["entry_date_ts"] = pd.to_datetime(c["entry_date"])
    ev = c[["symbol_raw", "entry_date_ts"]].rename(
        columns={"symbol_raw": "symbol", "entry_date_ts": "entry_date"})
    bars = SAN.load_entry_bars(ev)
    g = {k: v.sort_values("date").reset_index(drop=True)
         for k, v in bars.groupby(["symbol", "session"], sort=False)}

    ENTRY_T = pd.Timestamp("1900-01-01 09:15").time()
    EXIT_T = pd.Timestamp(f"1900-01-01 {P5.EXIT_BAR_START}").time()
    out = []
    for r in c.itertuples(index=False):
        b = g.get((r.symbol_raw, r.entry_date_ts))
        rec = dict(signal_date=r.signal_date, symbol=r.symbol,
                   perside_blend_bps=np.nan, perside_cons_bps=np.nan)
        if b is None or len(b) == 0:
            out.append(rec)
            continue
        # M6a -- daily square-root law (leg-independent), identical formula to the 1m script
        m6a = m6a_s = np.nan
        dh, dl, dc, adv = (float(r.day_high), float(r.day_low),
                           float(r.day_close), float(r.adv20))
        if dc > 0 and adv > 0 and dh > dl > 0:
            sigma_bps = (dh - dl) / dc * 10_000.0 / PARKINSON_K
            pa = NOTIONAL / adv
            m6a, m6a_s = 0.5 * sigma_bps * np.sqrt(pa), 1.0 * sigma_bps * np.sqrt(pa)
        legs, legs_c = [], []
        for tsel in (ENTRY_T, EXIT_T):
            hit = np.flatnonzero(b["date"].dt.time.values == tsel)
            if hit.size == 0:
                continue
            bar = b.iloc[int(hit[0])]
            px = float(bar["close"])
            if not (px > 0):
                continue
            half = (float(bar["high"]) - float(bar["low"])) / px * 10_000.0 / 2.0
            typ = (float(bar["high"]) + float(bar["low"]) + px) / 3.0
            to5 = typ * float(bar["volume"])
            m6b = (min(1.0, NOTIONAL / to5) * half) if to5 > 0 else half
            legs.append(half + np.nanmax([m6a, m6b]))
            legs_c.append(half + np.nanmax([m6a_s, m6b]))
        if len(legs) == 2:
            rec["perside_blend_bps"] = float(np.mean(legs))
            rec["perside_cons_bps"] = float(np.mean(legs_c))
        out.append(rec)
    return pd.DataFrame(out)


def slippage_fresh_calibrated(fresh_cohort: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    dem_cohort = pd.read_csv(P5.COHORT_DEMOTED)
    dem_meas = pd.read_csv(P5.SLIP_DEMOTED)[
        ["signal_date", "symbol", "perside_blend_bps", "perside_cons_bps"]]
    dem_meas.columns = ["signal_date", "symbol", "meas_blend", "meas_cons"]

    print("\n--- 5m slippage proxy on the DEMOTED cohort (calibration anchor) ---",
          flush=True)
    dem_proxy = _proxy_5m(dem_cohort)
    j = dem_proxy.merge(dem_meas, on=["signal_date", "symbol"], how="inner").dropna(
        subset=["perside_blend_bps", "meas_blend"])
    k_c = float(j["meas_blend"].mean() / j["perside_blend_bps"].mean())
    k_x = float(j["meas_cons"].mean() / j["perside_cons_bps"].mean())
    print(f"  calibration on n={len(j)} demoted trades with BOTH a 1m measurement and a "
          f"5m proxy:")
    print(f"    CENTRAL      1m measured {j['meas_blend'].mean():.2f} bp/side  vs  "
          f"5m proxy {j['perside_blend_bps'].mean():.2f}  ->  k={k_c:.4f}")
    print(f"    CONSERVATIVE 1m measured {j['meas_cons'].mean():.2f} bp/side  vs  "
          f"5m proxy {j['perside_cons_bps'].mean():.2f}  ->  k={k_x:.4f}")
    print(f"    proxy/measured per-trade correlation: "
          f"{j['perside_blend_bps'].corr(j['meas_blend']):.3f} (central), "
          f"{j['perside_cons_bps'].corr(j['meas_cons']):.3f} (cons)")

    print("\n--- 5m slippage proxy on the FRESH-POOL cohort ---", flush=True)
    fr = _proxy_5m(fresh_cohort)
    print(f"  fresh 5m proxy: CENTRAL {fr['perside_blend_bps'].mean():.2f} bp/side  "
          f"CONSERVATIVE {fr['perside_cons_bps'].mean():.2f}  (n={fr['perside_blend_bps'].notna().sum()})")
    rel_c = float(fr["perside_blend_bps"].mean() / dem_proxy["perside_blend_bps"].mean())
    rel_x = float(fr["perside_cons_bps"].mean() / dem_proxy["perside_cons_bps"].mean())
    print(f"  fresh-vs-demoted proxy ratio: {rel_c:.3f} (central), {rel_x:.3f} (cons) "
          f"-> the fresh cohort is {'WIDER' if rel_c > 1 else 'tighter'} on identical inputs")
    fr["perside_blend_bps"] = fr["perside_blend_bps"] * k_c
    fr["perside_cons_bps"] = fr["perside_cons_bps"] * k_x
    print(f"  CALIBRATED fresh rates: CENTRAL {fr['perside_blend_bps'].mean():.2f} bp/side  "
          f"CONSERVATIVE {fr['perside_cons_bps'].mean():.2f}")
    info = dict(method="5m_proxy_calibrated_to_1m_demoted_measurement",
                reason="1m archive ends 2026-04; frozen 1m measurement not runnable on 2026-05+",
                k_central=k_c, k_conservative=k_x, n_calibration=int(len(j)),
                demoted_measured_central=float(j["meas_blend"].mean()),
                demoted_measured_cons=float(j["meas_cons"].mean()),
                fresh_vs_demoted_proxy_ratio_central=rel_c,
                fresh_vs_demoted_proxy_ratio_cons=rel_x,
                fresh_calibrated_central=float(fr["perside_blend_bps"].mean()),
                fresh_calibrated_cons=float(fr["perside_cons_bps"].mean()))
    return fr, info


def project_month_to_40(monthly: pd.Series, n_now: int, n_min: int) -> str:
    """Straight-line projection at the observed fire rate.  Stated as arithmetic only."""
    live = monthly[monthly.index != monthly.index.max()] if len(monthly) > 1 else monthly
    rate = float(monthly.sum()) / max(len(monthly), 1)
    if rate <= 0:
        return "n/a (zero fires observed)"
    need = max(0, n_min - n_now)
    months = int(np.ceil(need / rate))
    last = pd.Period(monthly.index.max(), freq="M")
    return (f"{(last + months)} (at the observed {rate:.1f} fires/month, "
            f"{need} more needed ~ {months} month(s))")


# =============================================================================
def step_count(lock: dict) -> dict:
    print("=" * 110)
    print(f"FRESH-POOL ONE-SHOT  STEP 1 -- FIRE COUNT ONLY  [{SETUP}]")
    print("  NO returns, NO PnL, NO expectancy are computed in this step.")
    print("=" * 110)
    _px, ev, g = build_fresh(lock)

    print("\n--- bar-level admission (same checks as the frozen simulator) ---")
    adm, drops = admit(ev, g)
    n_ev = len(ev)
    SAN.funnel("5m entry session present", n_ev - drops["nobars"],
               f"{drops['nobars']} missing")
    SAN.funnel("session >=2 bars and starts 09:15",
               n_ev - drops["nobars"] - drops["short"] - drops["badstart"],
               f"{drops['short']} too short, {drops['badstart']} bad start bar")
    SAN.funnel("not circuit-blocked (strict)",
               n_ev - drops["nobars"] - drops["short"] - drops["badstart"] - drops["circuit"],
               f"{drops['circuit']} blocked")
    SAN.funnel("15:15 exit bar present",
               n_ev - drops["nobars"] - drops["short"] - drops["badstart"]
               - drops["circuit"] - drops["noexit"],
               f"{drops['noexit']} missing the exit bar")
    SAN.funnel("FIRES (frozen funnel, fresh pool)", len(adm),
               f"{drops['badqty']} dropped on price/qty")

    print("\n  FUNNEL SUMMARY")
    for step, n, note in SAN.FUNNEL:
        print(f"    {step:<48s} {n:6d}   {note}")

    n = len(adm)
    monthly = (adm.groupby("month").size() if n else pd.Series(dtype=int))
    print(f"\n  FINAL n = {n}")
    if n:
        print("\n  monthly fire rate:")
        for mo, c in monthly.items():
            print(f"    {mo}   {c:4d}")
        print(f"    symbols={adm['symbol'].nunique()}  sessions={adm['entry_date'].nunique()}  "
              f"adv_tier={dict(adm['adv_tier'].value_counts())}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    adm.to_csv(OUT_CSV, index=False)
    print(f"\n  wrote {OUT_CSV}  rows={len(adm)}  (identifiers only -- no prices, no returns)")

    gate_ok = n >= PREREG["power_gate_n_min"]
    print("\n" + "=" * 110)
    print(f"STEP 2 -- POWER GATE (pre-registered, SS9d): n >= {PREREG['power_gate_n_min']}")
    print("=" * 110)
    if gate_ok:
        print(f"  n={n} >= {PREREG['power_gate_n_min']}  ->  PROCEED to the one-shot.")
    else:
        proj = project_month_to_40(monthly, n, PREREG["power_gate_n_min"]) if n else "n/a"
        print(f"  n={n} < {PREREG['power_gate_n_min']}  ->  *** POWER-BLOCKED ***")
        print("  Report counts only.  Compute NO verdict.  Do NOT peek at outcomes.")
        print(f"  Re-shoot projected at: {proj}")
    return dict(n=n, gate_ok=bool(gate_ok),
                monthly={str(k): int(v) for k, v in monthly.items()},
                drops=drops, n_events_pre_bar=int(n_ev),
                projection=(project_month_to_40(monthly, n, PREREG["power_gate_n_min"])
                            if (n and not gate_ok) else None),
                symbols=int(adm["symbol"].nunique()) if n else 0,
                sessions=int(adm["entry_date"].nunique()) if n else 0)


# =============================================================================
def step_full(lock: dict, count_res: dict) -> dict:
    if not count_res["gate_ok"]:
        raise RuntimeError(
            f"POWER-BLOCKED: n={count_res['n']} < {PREREG['power_gate_n_min']}. "
            "The pre-registered rule forbids computing any outcome statistic. STOP.")
    lc, lg = lock["locked_cell"], lock["locked_geometry"]

    print("\n" + "=" * 110)
    print("STEP 3 -- THE ONE-SHOT (frozen construction, unchanged)")
    print("=" * 110)
    _px, ev, g = build_fresh(lock)
    tr = P5.simulate(ev, g)
    base = tr[tr["geom"] == "none/none"].copy()
    assert len(base) == count_res["n"], \
        f"simulator n={len(base)} != counted n={count_res['n']} -- funnels diverged"

    base["fees_0bp"] = [calc_fee(float(a), float(b), int(q), "SELL", 1.0)
                        for a, b, q in zip(base["entry_price"], base["exit_price"],
                                           base["qty"])]
    base.to_csv(COHORT_CSV, index=False)
    print(f"  wrote fresh-pool base cohort -> {COHORT_CSV}  n={len(base)}")

    # ---- slippage RE-MEASURED on THIS cohort (never assumed) ----
    # The frozen 1m measurement is attempted first and is used verbatim if the 1m archive
    # covers the cohort.  It does not (it ends 2026-04), so the calibrated 5m proxy runs.
    ym_needed = sorted(pd.to_datetime(base["entry_date"]).dt.strftime("%Y_%m").unique())
    have_1m = [ym for ym in ym_needed
               if (_REPO / "backtest-cache-download" / "monthly" / f"{ym}_1m.feather").exists()]
    print(f"\n  1m archive coverage for this cohort: {len(have_1m)}/{len(ym_needed)} months "
          f"({ym_needed} needed)")
    if len(have_1m) == len(ym_needed):
        if not SLIP_CSV.exists():
            cmd = [sys.executable,
                   str(_REPO / "tools" / "sub9_research" / "measure_slippage_earnings_downshock.py"),
                   "--cohort-csv", str(COHORT_CSV.relative_to(_REPO)).replace("\\", "/"),
                   "--out-csv", str(SLIP_CSV.relative_to(_REPO)).replace("\\", "/"),
                   "--exit-fill-minute", "15:14",
                   "--exit-5m-window", "15:10,15:14",
                   "--exit-window", "15:05,15:14",
                   "--exit-roll-window", "14:15,15:14",
                   "--allow-fresh-pool"]
            r = subprocess.run(cmd, cwd=str(_REPO), capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stderr[-3000:])
                raise RuntimeError("slippage measurement failed")
        slip = pd.read_csv(SLIP_CSV)
        slip_info = dict(method="frozen_1m_measurement")
    else:
        print("  *** the frozen 1m slippage measurement CANNOT be run on this cohort. ***")
        print("  Falling back to a 5m proxy CALIBRATED against the 1m measurement of the")
        print("  demoted cohort -- measured on this cohort's own tape, not assumed.")
        slip, slip_info = slippage_fresh_calibrated(base)
        slip.to_csv(SLIP_CSV, index=False)

    tr = P5.price_trades(tr, slip)
    m = P5.cell_mask(tr, lc["adv_tier"], lc["shock_depth_pct"], lc["announce_class"])
    sub = tr[m & (tr["geom"] == lg["label"])].copy()

    print(f"\n  measured slippage on this cohort: "
          f"CENTRAL {sub['slip_bps_central'].mean():.1f} bp/side  "
          f"CONSERVATIVE {sub['slip_bps_conservative'].mean():.1f} bp/side")

    # gross (no fees, no slippage)
    gross_pct = float(sub["pnl_pct"].mean())
    gross_pf = P5.pf_of((sub["pnl_pct"] / 100.0 * sub["entry_price"] * sub["qty"]).values)

    print("\n" + "=" * 110)
    print("FRESH-POOL RESULT (locked cell x locked geometry, ONE SHOT)")
    print("=" * 110)
    print(f"  n={len(sub)}  symbols={sub['symbol'].nunique()}  "
          f"sessions={sub['entry_date'].nunique()}")
    print(f"  {'GROSS (no fees, no slippage)':<34s} exp={gross_pct:+.4f}%  PF={gross_pf:.4f}  "
          f"win={float((sub['pnl_pct']>0).mean()):.3f}")
    res = {}
    for tag, lbl in (("central", "CENTRAL measured (per-trade)"),
                     ("conservative", "CONSERVATIVE measured (per-trade)"),
                     ("centralFlat", f"CENTRAL flat {P5.FLAT_BPS['central']}bp"),
                     ("conservativeFlat", f"CONSERV flat {P5.FLAT_BPS['conservative']}bp")):
        s = P5.stat(sub, tag)
        res[tag] = s
        print(f"  {lbl:<34s} n={s['n']:4d}  exp={s['exp_pct']:+.4f}%  PF={s['pf']:.4f}  "
              f"win={s['win']:.3f}  t={s['t']:+.2f}  net=Rs{s['net_inr']:,.0f}")

    print("\n  PER MONTH:")
    sub["month"] = pd.to_datetime(sub["signal_date"]).dt.to_period("M").astype(str)
    per_month = {}
    for mo, sm in sub.groupby("month"):
        row = {}
        for tag in ("central", "conservative"):
            s = P5.stat(sm, tag)
            row[tag] = s
            print(f"    {mo} {tag:<14s} n={s['n']:4d}  exp={s['exp_pct']:+.4f}%  "
                  f"PF={s['pf']:.4f}  win={s['win']:.3f}")
        per_month[mo] = row

    print("\n  EXIT MIX:", dict(sub["exit_reason"].value_counts()))
    for tier, st in sub.groupby("adv_tier"):
        s = P5.stat(st, "conservative")
        print(f"    {tier:<9} n={s['n']:4d}  CONSERVATIVE exp={s['exp_pct']:+.4f}%  "
              f"PF={s['pf']:.4f}")

    # ---- STEP 4: mechanical verdict on the CONSERVATIVE number ----
    v = float(res[PREREG["verdict_basis"]]["exp_pct"])
    if v >= PREREG["pass_at_or_above"]:
        verdict, band = "pass", f">= +{PREREG['pass_at_or_above']}%"
    elif v >= PREREG["kill_below"]:
        verdict, band = "marginal", f"0 to +{PREREG['pass_at_or_above']}%"
    else:
        verdict, band = "kill", "< 0"
    print("\n" + "=" * 110)
    print("STEP 4 -- VERDICT (pre-registered bands, applied mechanically to CONSERVATIVE)")
    print("=" * 110)
    print(f"  CONSERVATIVE net {v:+.4f}%/trade   falls in band [{band}]   -> {verdict.upper()}")

    out = dict(setup=SETUP, stage="fresh_pool_oneshot", freeze_commit=FREEZE_COMMIT,
               run_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
               n=int(len(sub)), gross_pct=gross_pct, gross_pf=gross_pf,
               slip_central_bps=float(sub["slip_bps_central"].mean()),
               slip_cons_bps=float(sub["slip_bps_conservative"].mean()),
               stats={k: {kk: (None if pd.isna(vv) else float(vv))
                          for kk, vv in s.items()} for k, s in res.items()},
               per_month={mo: {k: {kk: (None if pd.isna(vv) else float(vv))
                                   for kk, vv in s.items()} for k, s in r.items()}
                          for mo, r in per_month.items()},
               verdict=verdict, verdict_band=band, count=count_res,
               slippage_method=slip_info)
    sub.to_csv(OUT_CSV, index=False)
    print(f"\n  wrote {OUT_CSV}  rows={len(sub)}  (full one-shot ledger)")
    return out


# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step", required=True, choices=("count", "full"))
    ap.add_argument("--i-am-burning-the-fresh-pool", action="store_true",
                    dest="optin", required=True,
                    help="MANDATORY explicit opt-in.  The fresh-pool guard in "
                         "sanity_...build_cohort stays intact; this flag makes "
                         "bypassing it a deliberate, logged act (lifecycle A1).")
    a = ap.parse_args()
    if not a.optin:
        raise SystemExit("refused: --i-am-burning-the-fresh-pool not given")

    print("=" * 110)
    print(f"A1 FRESH-POOL ONE-SHOT  [{SETUP}]   freeze commit {FREEZE_COMMIT}")
    print(f"  window   : signals {PREREG['window_start']} .. latest session on disk")
    print(f"  bands    : PASS >= +{PREREG['pass_at_or_above']}% | "
          f"MARGINAL 0..+{PREREG['pass_at_or_above']}% | KILL < 0  "
          f"(on {PREREG['verdict_basis'].upper()} slippage)")
    print(f"  power    : n < {PREREG['power_gate_n_min']} -> POWER-BLOCKED, no verdict")
    print("=" * 110)
    lock = load_lock()

    count_res = step_count(lock)
    out = dict(setup=SETUP, stage="fresh_pool_oneshot", freeze_commit=FREEZE_COMMIT,
               run_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
               count=count_res,
               verdict=("power_blocked" if not count_res["gate_ok"] else "pending"))
    if a.step == "full":
        if not count_res["gate_ok"]:
            print("\n*** REFUSING --step full: the power gate blocks it. ***")
        else:
            out = step_full(lock, count_res)
    RESULT_JSON.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\n  wrote {RESULT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
