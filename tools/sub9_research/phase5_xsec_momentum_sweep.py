"""Phase 5 era-split cell sweep: `xsec_momentum_demeaned` (lifecycle Stage 5).

Grid PRE-REGISTERED BLIND in specs/2026-07-27-brief-xsec_momentum_demeaned.md
section 9b (commit 40be9f1, before data/asm_gsm_history/asm_gsm_events.parquet
existed). BINDING — no dimensions added/removed/reweighted after results.

Machinery reused from tools/sub9_research/sanity_xsec_momentum_demeaned.py
(Stage 4): panel construction, ProductionUniverseGate universe, shifted ADV20
floor, Mode-B next-real-open entries, per-rebalance per-universe nanmean
demeaning, weekly rebalance = every 5th trading day of the development window.

Grid (72 selectable cells):
  formation {60, 120} skip-2
  x hold {20, 40, 60} sessions
  x cohort {winner decile, top-5%}
  x universe {all, MTF-eligible, ASM-clean-at-entry}
  x exit {hold_close, asm_exit (symbol ENTERS any ASM/GSM list mid-hold ->
          sell next session real open, else hold to H)}
  Rebalance weekly. DIAGNOSTIC-ONLY (measured, never selectable): 3-sigma
  vol-scaled target; 2-sigma vol-scaled stop (PEAD Phase-5 precedent) —
  emitted as exit_mode rows flagged diagnostic=True.

Windows (amendments A1 + A5):
  Development pool rebalances 2023-01-01 .. 2026-04-30 ONLY (RAISES on
  violation). era_A = rebalance dates 2023-01..2024-12, era_B = 2025-01..
  2026-04. Forward holds of Apr-2026 rebalances extend past 2026-05-01 as
  EXITS ONLY (precedent); NO rebalance on/after 2026-05-01 — the fresh pool
  stays untouched. Rebalances whose H-exit falls beyond the panel end are
  DROPPED for that hold (counted n_dropped_eow), never shortened.

Decisive statistic: DEMEANED alpha = position return minus the SAME universe
variant's nanmean forward return over the same window, per rebalance. For
asm_exit cells the benchmark is UNCHANGED (full-H universe mean) so the
alpha differential vs hold_close on the same cohort reads out falsifier #3
directly (does ASM entry consume the winner tail?).

Costs: CNC round-trip 0.307% + 20bp slippage = 0.507% flat per position
(equal notional, A2-family cost model). PF_net over per-position net returns.

LOCKABLE CELL RULE (section 9b): demeaned alpha > 0 in BOTH eras with
n >= 100 per era AND pooled PF_net >= 1.20. Selection among eligible =
SMALLEST |era_A - era_B| alpha gap (stability over top-PF,
feedback_cell_sweep_stability_over_top_pf). No eligible cell = KILL,
no salvage, no relaxed thresholds.

ASM/GSM membership data notes (documented before first run):
  - PRIMARY: exchange=='NSE' rows (ASM_LONGTERM / ASM_SHORTTERM /
    ASM_LONGTERM_IBC daily snapshots from surveillance circulars).
  - FALLBACK (noted per handoff): NSE GSM circulars are image PDFs and were
    never parsed — GSM membership comes from exchange=='BSE',
    surveillance_program=='GSM' rows whose symbol was ISIN-resolved to an
    NSE ticker present in the price panel (51 panel symbols, 70 entry
    events — small vs 4,918 in-panel NSE ASM entry events).
  - Membership on date d = snapshot row (transition_type != 'exit') exists
    within the LAST 5 SESSIONS ending at d ("member_recent"): 37/877 NSE
    dates have sparse snapshots (fetch gaps), so a strict same-day test
    would leak list members into the clean universe on gap dates. The
    5-session lookback is conservative (excludes recently-exited names for
    <=1 week; never admits a current member because of a gap).
  - Causality: an NSE snapshot dated T is published ~17:00 IST on T and
    effective T+1. ASM-clean-at-entry therefore tests membership at the
    REBALANCE date t (known before the t+1 open entry). asm_exit triggers
    on transition_type=='entry' events dated e in [entry session, exit-1];
    sale at the first REAL open >= e+1 (capped at the H-exit session; if no
    real open prints in between, falls back to the H close valuation).

Outputs:
  reports/sub9_sanity/_xsec_momentum_phase5_cells.csv  (ALL cells, per-era +
    pooled + diagnostics)
  tools/sub9_research/xsec_momentum_demeaned_cell_lock.json (ONLY if a cell
    is eligible)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from tools.sub9_research.production_universe import ProductionUniverseGate  # noqa: E402

# ----------------------------------------------------------------------------
# LOCKED_GRID — transcribed from brief section 9b BEFORE first run. DO NOT EDIT.
# ----------------------------------------------------------------------------
LOCKED_GRID = dict(
    DATA_FILE=_REPO_ROOT / "cache/preaggregate/clean_daily_from5m.feather",
    ASM_FILE=_REPO_ROOT / "data/asm_gsm_history/asm_gsm_events.parquet",
    DEV_START=pd.Timestamp("2023-01-01"),
    DEV_END=pd.Timestamp("2026-04-30"),          # last allowed rebalance date
    ERA_A_END=pd.Timestamp("2024-12-31"),        # era_A = <= this; era_B after
    SKIP=2,
    FORMATIONS=(60, 120),
    HOLDS=(20, 40, 60),
    COHORTS=(("decile", 0.10), ("top5pct", 0.05)),
    UNIVERSES=("all", "mtf", "asm_clean"),
    EXITS=("hold_close", "asm_exit"),
    REBALANCE_EVERY=5,                           # weekly (Stage-4 locked)
    ADV_FLOOR=2_000_000.0,                       # Rs 20L
    ADV_WINDOW=20,
    MIN_NAMES=50,
    FFILL_LIMIT=5,
    ASM_MEMBER_LOOKBACK=5,                       # sessions (snapshot-gap guard)
    COST_RT=0.00307 + 0.0020,                    # CNC round trip + 20bp slip
    MTF_SNAPSHOT=_REPO_ROOT / "data/mtf_universe/approved_mtf_securities_latest.json",
    GATE_ACCEPTED_CAPS={"large_cap", "mid_cap", "small_cap", "micro_cap", "unknown"},
    GATE_REQUIRE_MIS=False,
    GATE_REQUIRE_MTF=False,
    # Diagnostic-only geometry (measured on universe='all' cells, never selectable)
    DIAG_TARGET_SIGMA=3.0,
    DIAG_STOP_SIGMA=2.0,
    SIGMA_WINDOW=20,
    # Lockable-cell rule
    N_MIN_PER_ERA=100,
    PF_MIN_POOLED=1.20,
)
OUT_CSV = _REPO_ROOT / "reports/sub9_sanity/_xsec_momentum_phase5_cells.csv"
LOCK_JSON = _REPO_ROOT / "tools/sub9_research/xsec_momentum_demeaned_cell_lock.json"


def load_panels():
    g = LOCKED_GRID
    dd = pd.read_feather(g["DATA_FILE"])
    dd["date"] = pd.to_datetime(dd["date"]).dt.normalize()
    dd["symbol"] = (
        dd["symbol"].astype(str).str.replace("NSE:", "", regex=False).str.upper()
    )
    print(f"[data] {g['DATA_FILE'].name}: shape={dd.shape} "
          f"dates {dd['date'].min().date()} -> {dd['date'].max().date()} "
          f"symbols={dd['symbol'].nunique()}")
    panels = {}
    for col in ("open", "high", "low", "close", "volume"):
        panels[col] = dd.pivot_table(index="date", columns="symbol",
                                     values=col, aggfunc="last")
    C = panels["close"].sort_index()
    for k in panels:
        panels[k] = panels[k].reindex(index=C.index, columns=C.columns)
    return panels


def load_asm(cal, symbols):
    """Membership + entry-event boolean panels on the trading calendar."""
    g = LOCKED_GRID
    a = pd.read_parquet(g["ASM_FILE"])
    a["date"] = pd.to_datetime(a["date"])
    a["symbol"] = a["symbol"].astype(str).str.upper()
    print(f"[asm] {g['ASM_FILE'].name}: shape={a.shape} "
          f"dates {a['date'].min().date()} -> {a['date'].max().date()} "
          f"exchanges={a['exchange'].value_counts().to_dict()}")

    sym_set = set(symbols)
    nse = a[a["exchange"] == "NSE"]
    bse_gsm = a[(a["exchange"] == "BSE") & (a["surveillance_program"] == "GSM")
                & (a["symbol"].isin(sym_set))]
    src = pd.concat([nse, bse_gsm], ignore_index=True)
    print(f"[asm] PRIMARY NSE rows={len(nse)} (in-panel symbols "
          f"{nse['symbol'].isin(sym_set).sum()} rows) | FALLBACK BSE-GSM "
          f"panel-resolved rows={len(bse_gsm)} "
          f"({bse_gsm['symbol'].nunique()} symbols) — NSE GSM circulars are "
          f"image PDFs, never parsed (fetcher limitation #1)")

    sym_idx = {s: i for i, s in enumerate(symbols)}
    date_idx = {d: i for i, d in enumerate(cal)}
    n_days, n_sym = len(cal), len(symbols)
    member = np.zeros((n_days, n_sym), dtype=bool)
    event = np.zeros((n_days, n_sym), dtype=bool)

    mem_rows = src[src["transition_type"] != "exit"]
    ev_rows = src[src["transition_type"] == "entry"]
    n_skip_sym = n_skip_date = 0
    for df, panel in ((mem_rows, member), (ev_rows, event)):
        si = df["symbol"].map(sym_idx)
        di = df["date"].map(date_idx)
        ok = si.notna() & di.notna()
        n_skip_sym += int(si.isna().sum())
        n_skip_date += int((si.notna() & di.isna()).sum())
        panel[di[ok].astype(int).to_numpy(), si[ok].astype(int).to_numpy()] = True
    print(f"[asm] rows skipped: not-in-panel-symbol={n_skip_sym} "
          f"date-off-calendar={n_skip_date}")

    # member_recent[t] = member on any of the last ASM_MEMBER_LOOKBACK sessions
    k = g["ASM_MEMBER_LOOKBACK"]
    mr = member.copy()
    for lag in range(1, k):
        mr[lag:] |= member[:-lag]
    print(f"[asm] membership panel: {member.sum()} member-day flags, "
          f"{event.sum()} entry events on calendar; member_recent lookback={k}")
    return mr, event


def pf_net(x):
    pos = x[x > 0].sum()
    neg = -x[x < 0].sum()
    if neg == 0:
        return float("inf") if pos > 0 else np.nan
    return float(pos / neg)


def main():
    g = LOCKED_GRID
    panels = load_panels()
    C, O, H, L, V = (panels[k] for k in ("close", "open", "high", "low", "volume"))
    cal = C.index
    symbols = C.columns.to_numpy()
    n_days = len(cal)

    ADV = (C * V).rolling(g["ADV_WINDOW"], min_periods=g["ADV_WINDOW"]).mean().shift(1)
    C_ff = C.ffill(limit=g["FFILL_LIMIT"])
    C_np, Cff_np, O_np = C.to_numpy(), C_ff.to_numpy(), O.to_numpy()
    H_np, L_np, ADV_np = H.to_numpy(), L.to_numpy(), ADV.to_numpy()

    # Daily-return sigma for DIAGNOSTIC geometry (causal: value at t uses
    # closes up to t; entry is at t+1 open).
    SIG = (pd.DataFrame(Cff_np, index=cal).pct_change()
           .rolling(g["SIGMA_WINDOW"], min_periods=g["SIGMA_WINDOW"]).std()
           .to_numpy())

    FORM = {}
    for Lf in g["FORMATIONS"]:
        f = np.full_like(C_np, np.nan)
        lo, hi = g["SKIP"] + Lf, g["SKIP"]
        f[lo:, :] = Cff_np[lo - hi: n_days - hi, :] / Cff_np[: n_days - lo, :] - 1.0
        FORM[Lf] = f

    ENTRY = np.full_like(O_np, np.nan)
    ENTRY[: n_days - 1, :] = O_np[1:, :]
    FWD = {}
    for Hh in g["HOLDS"]:
        f = np.full_like(C_np, np.nan)
        f[: n_days - 1 - Hh, :] = Cff_np[1 + Hh:, :] / ENTRY[: n_days - 1 - Hh, :] - 1.0
        FWD[Hh] = f

    member_recent, event = load_asm(cal, symbols)
    # Per-symbol sorted event calendar positions (for mid-hold trigger search).
    ev_pos = [np.flatnonzero(event[:, j]) for j in range(len(symbols))]

    # Development-pool weekly rebalances (A1: HARD raise on violation).
    dev_mask = (cal >= g["DEV_START"]) & (cal <= g["DEV_END"])
    rebal_pos = np.flatnonzero(dev_mask)[:: g["REBALANCE_EVERY"]]
    rd = cal[rebal_pos]
    if not ((rd >= g["DEV_START"]) & (rd <= g["DEV_END"])).all():
        raise AssertionError("A1 VIOLATION: rebalance outside development pool "
                             f"{g['DEV_START'].date()}..{g['DEV_END'].date()}")
    if (rd >= pd.Timestamp("2026-05-01")).any():
        raise AssertionError("A1 VIOLATION: rebalance touches the fresh pool")
    era = np.where(rd <= g["ERA_A_END"], "era_A", "era_B")
    print(f"[rebalance] weekly: {len(rebal_pos)} dates "
          f"({rd[0].date()} -> {rd[-1].date()}); era_A={int((era=='era_A').sum())} "
          f"era_B={int((era=='era_B').sum())}; max hold exit date = "
          f"{cal[min(n_days-1, int(rebal_pos.max())+1+max(g['HOLDS']))].date()} "
          f"(exits-only past 2026-05-01 per precedent; beyond-panel drops counted)")

    gate = ProductionUniverseGate(
        accepted_caps=g["GATE_ACCEPTED_CAPS"],
        require_mis=g["GATE_REQUIRE_MIS"],
        require_mtf=g["GATE_REQUIRE_MTF"],
        min_trading_days_required=0,
        min_daily_avg_volume=0,
        mtf_snapshot_path=g["MTF_SNAPSHOT"],
        exclude_etf=False,
    )
    with open(g["MTF_SNAPSHOT"], encoding="utf-8") as fh:
        mtf_set = {str(e.get("tradingsymbol", "")).upper() for e in json.load(fh)}
    mtf_arr = np.isin(symbols, list(mtf_set))
    print(f"[mtf] {int(mtf_arr.sum())}/{len(symbols)} panel symbols MTF-eligible "
          f"(2026-era snapshot — ANACHRONISTIC for 2023-25, Lesson #27; "
          f"capturability dimension, paper is the production gate)")

    h_max = max(g["HOLDS"])
    rows = []          # position ledger
    n_skipped_universe = {u: 0 for u in g["UNIVERSES"]}

    for p_i, p in enumerate(rebal_pos):
        p = int(p)
        reb_date = cal[p]
        gate_ok = np.array([gate.is_eligible(s, session_date=reb_date.date())
                            for s in symbols])
        base_valid = (
            gate_ok
            & np.isfinite(C_np[p])
            & np.isfinite(ADV_np[p]) & (ADV_np[p] >= g["ADV_FLOOR"])
            & np.isfinite(ENTRY[p]) & (ENTRY[p] > 0)
        )
        for Lf in g["FORMATIONS"]:
            valid_all = base_valid & np.isfinite(FORM[Lf][p])
            u_idx = {
                "all": np.flatnonzero(valid_all),
                "mtf": np.flatnonzero(valid_all & mtf_arr),
                "asm_clean": np.flatnonzero(valid_all & ~member_recent[p]),
            }
            for uni, vidx in u_idx.items():
                if len(vidx) < g["MIN_NAMES"]:
                    n_skipped_universe[uni] += 1
                    continue
                univ_mean = {Hh: float(np.nanmean(FWD[Hh][p][vidx]))
                             for Hh in g["HOLDS"]}
                form_v = FORM[Lf][p][vidx]
                order = np.argsort(-form_v)          # descending formation
                k10 = max(1, int(np.ceil(0.10 * len(vidx))))
                k5 = max(1, int(np.ceil(0.05 * len(vidx))))
                sel = vidx[order[:k10]]
                in_top5 = np.zeros(len(sel), dtype=bool)
                in_top5[:k5] = True

                for rank_i, ci in enumerate(sel):
                    ci = int(ci)
                    entry_open = float(ENTRY[p, ci])
                    # first ASM/GSM entry event at/after the entry session
                    ev = ev_pos[ci]
                    j = int(np.searchsorted(ev, p + 1))
                    e_first = int(ev[j]) if j < len(ev) else -1
                    row = dict(
                        rebalance_date=str(reb_date.date()),
                        era=era[p_i], universe=uni, formation=Lf,
                        symbol=symbols[ci], in_top5=bool(in_top5[rank_i]),
                        entry_open=entry_open,
                    )
                    sig = SIG[p, ci]
                    for Hh in g["HOLDS"]:
                        fw = FWD[Hh][p, ci]
                        exit_pos = p + 1 + Hh
                        if not np.isfinite(fw) or exit_pos >= n_days:
                            row[f"raw_h{Hh}"] = np.nan
                            row[f"alpha_h{Hh}"] = np.nan
                            row[f"asm_raw_h{Hh}"] = np.nan
                            row[f"asm_alpha_h{Hh}"] = np.nan
                            row[f"asm_trunc_h{Hh}"] = np.nan
                            row[f"diag_tgt_h{Hh}"] = np.nan
                            row[f"diag_stp_h{Hh}"] = np.nan
                            continue
                        um = univ_mean[Hh]
                        row[f"raw_h{Hh}"] = fw
                        row[f"alpha_h{Hh}"] = fw - um
                        # ---- asm_exit mode ----
                        trunc = 0.0
                        asm_ret = fw
                        if 0 <= e_first <= p + Hh:       # event before exit session
                            trunc = 1.0
                            sell = np.nan
                            for jj in range(e_first + 1, exit_pos + 1):
                                if np.isfinite(O_np[jj, ci]):
                                    sell = O_np[jj, ci]
                                    break
                            if not np.isfinite(sell):
                                sell = Cff_np[exit_pos, ci]  # fallback: H close
                            asm_ret = float(sell) / entry_open - 1.0
                        row[f"asm_raw_h{Hh}"] = asm_ret
                        row[f"asm_alpha_h{Hh}"] = asm_ret - um
                        row[f"asm_trunc_h{Hh}"] = trunc
                        # ---- diagnostics (universe='all' only; never selectable)
                        if uni == "all" and np.isfinite(sig) and sig > 0:
                            hor = np.sqrt(Hh)
                            tgt = entry_open * (1.0 + g["DIAG_TARGET_SIGMA"] * sig * hor)
                            stp = entry_open * (1.0 - g["DIAG_STOP_SIGMA"] * sig * hor)
                            t_ret, s_ret = fw, fw
                            for jj in range(p + 1, exit_pos + 1):
                                hi = H_np[jj, ci]
                                if np.isfinite(hi) and hi >= tgt:
                                    op = O_np[jj, ci]
                                    fill = op if (np.isfinite(op) and op > tgt) else tgt
                                    t_ret = fill / entry_open - 1.0
                                    break
                            for jj in range(p + 1, exit_pos + 1):
                                lo = L_np[jj, ci]
                                if np.isfinite(lo) and lo <= stp:
                                    op = O_np[jj, ci]
                                    fill = op if (np.isfinite(op) and op < stp) else stp
                                    s_ret = fill / entry_open - 1.0
                                    break
                            row[f"diag_tgt_h{Hh}"] = t_ret
                            row[f"diag_stp_h{Hh}"] = s_ret
                        else:
                            row[f"diag_tgt_h{Hh}"] = np.nan
                            row[f"diag_stp_h{Hh}"] = np.nan
                    rows.append(row)

    led = pd.DataFrame(rows)
    print(f"\n[ledger] {len(led)} position rows; universe-skip counts "
          f"(rebalance x formation with <{g['MIN_NAMES']} names): "
          f"{n_skipped_universe}")

    # ---------------- cell aggregation ----------------
    cost = g["COST_RT"]
    cells = []
    for uni in g["UNIVERSES"]:
        for Lf in g["FORMATIONS"]:
            for coh, _q in g["COHORTS"]:
                sub_base = led[(led.universe == uni) & (led.formation == Lf)]
                if coh == "top5pct":
                    sub_base = sub_base[sub_base.in_top5]
                for Hh in g["HOLDS"]:
                    modes = [
                        ("hold_close", f"raw_h{Hh}", f"alpha_h{Hh}", False),
                        ("asm_exit", f"asm_raw_h{Hh}", f"asm_alpha_h{Hh}", False),
                    ]
                    if uni == "all":
                        modes += [
                            ("DIAG_target_3sigma", f"diag_tgt_h{Hh}", None, True),
                            ("DIAG_stop_2sigma", f"diag_stp_h{Hh}", None, True),
                        ]
                    for mode, rcol, acol, diag in modes:
                        for era_lbl in ("era_A", "era_B", "pooled"):
                            s = (sub_base if era_lbl == "pooled"
                                 else sub_base[sub_base.era == era_lbl])
                            r = s[rcol].dropna()
                            n = len(r)
                            net = r - cost
                            rec = dict(
                                universe=uni, formation=Lf, cohort=coh,
                                hold=Hh, exit_mode=mode, era=era_lbl,
                                diagnostic=diag, n=n,
                                n_nan=int(s[rcol].isna().sum()),
                                pf_net=round(pf_net(net.to_numpy()), 4) if n else np.nan,
                                wr_net=round(float((net > 0).mean()), 4) if n else np.nan,
                                mean_net_pct=round(float(net.mean()) * 100, 4) if n else np.nan,
                            )
                            if acol is not None:
                                a = s.loc[r.index, acol]
                                rec["alpha_pct"] = round(float(a.mean()) * 100, 4) if n else np.nan
                                rec["alpha_median_pct"] = round(float(a.median()) * 100, 4) if n else np.nan
                            else:
                                # diagnostics: alpha vs same universe mean not
                                # defined (exit path differs) — report demeaned
                                # vs hold_close raw instead
                                base_r = s.loc[r.index, f"raw_h{Hh}"]
                                rec["alpha_pct"] = np.nan
                                rec["alpha_median_pct"] = np.nan
                                rec["diag_delta_vs_hold_pct"] = (
                                    round(float((r - base_r).mean()) * 100, 4) if n else np.nan)
                            if mode == "asm_exit" and n:
                                tr = s.loc[r.index, f"asm_trunc_h{Hh}"]
                                rec["pct_truncated"] = round(float(tr.mean()) * 100, 2)
                                hc_a = s.loc[r.index, f"alpha_h{Hh}"]
                                asm_a = s.loc[r.index, f"asm_alpha_h{Hh}"]
                                rec["alpha_diff_vs_holdclose_pct"] = round(
                                    float((asm_a - hc_a).mean()) * 100, 4)
                            cells.append(rec)
    cdf = pd.DataFrame(cells)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cdf.to_csv(OUT_CSV, index=False)
    print(f"[out] {len(cdf)} cell-era rows -> {OUT_CSV}")

    # ---------------- eligibility (section 9b rule) ----------------
    sel_rows = cdf[~cdf.diagnostic]
    piv = sel_rows.pivot_table(
        index=["universe", "formation", "cohort", "hold", "exit_mode"],
        columns="era",
        values=["n", "alpha_pct", "pf_net", "wr_net", "mean_net_pct"],
        aggfunc="first",
    )
    flat = pd.DataFrame({
        "n_A": piv[("n", "era_A")], "n_B": piv[("n", "era_B")],
        "alpha_A": piv[("alpha_pct", "era_A")],
        "alpha_B": piv[("alpha_pct", "era_B")],
        "pf_pooled": piv[("pf_net", "pooled")],
        "wr_pooled": piv[("wr_net", "pooled")],
        "mean_net_pooled": piv[("mean_net_pct", "pooled")],
    }).reset_index()
    flat["alpha_gap"] = (flat["alpha_A"] - flat["alpha_B"]).abs()
    flat["eligible"] = (
        (flat.alpha_A > 0) & (flat.alpha_B > 0)
        & (flat.n_A >= g["N_MIN_PER_ERA"]) & (flat.n_B >= g["N_MIN_PER_ERA"])
        & (flat.pf_pooled >= g["PF_MIN_POOLED"])
    )
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)

    elig = flat[flat.eligible].sort_values("alpha_gap")
    print(f"\n=== ELIGIBLE CELLS (alpha>0 both eras, n>={g['N_MIN_PER_ERA']}/era, "
          f"pooled PF_net>={g['PF_MIN_POOLED']}): {len(elig)} of {len(flat)} ===")
    if len(elig):
        print(elig.to_string(index=False))
        best = elig.iloc[0]
        lock = dict(
            setup="xsec_momentum_demeaned",
            locked_at="2026-07-28",
            stage="phase5_erasplit_sweep",
            selection_rule=("most stable eligible cell = smallest |era_A-era_B| "
                            "demeaned-alpha gap (feedback_cell_sweep_stability_"
                            "over_top_pf); eligibility = alpha>0 both eras, "
                            "n>=100/era, pooled PF_net>=1.20"),
            cell=dict(universe=str(best.universe), formation=int(best.formation),
                      cohort=str(best.cohort), hold=int(best.hold),
                      exit_mode=str(best.exit_mode), rebalance="weekly",
                      skip=2, adv_floor_rs=2_000_000.0,
                      cost_round_trip=g["COST_RT"]),
            stats=dict(
                n_era_A=int(best.n_A), n_era_B=int(best.n_B),
                alpha_era_A_pct=float(best.alpha_A),
                alpha_era_B_pct=float(best.alpha_B),
                alpha_gap_pct=float(best.alpha_gap),
                pf_net_pooled=float(best.pf_pooled),
                wr_net_pooled=float(best.wr_pooled),
                mean_net_pooled_pct=float(best.mean_net_pooled),
            ),
            evidence=str(OUT_CSV.relative_to(_REPO_ROOT)),
            next_gate=("freeze commit, then ONE shot on fresh pool "
                       "2026-05-01+ (amendment A1) + paper"),
        )
        with open(LOCK_JSON, "w", encoding="utf-8") as fh:
            json.dump(lock, fh, indent=2)
        print(f"\n[lock] -> {LOCK_JSON}")
    else:
        near = flat.copy()
        near["min_alpha"] = near[["alpha_A", "alpha_B"]].min(axis=1)
        near = near.sort_values(["min_alpha", "pf_pooled"], ascending=False).head(10)
        print("\n=== KILL: no eligible cell. Top-10 nearest-miss (by min era "
              "alpha, then pooled PF) ===")
        print(near.to_string(index=False))

    # ---------------- console readouts ----------------
    core = flat[(flat.exit_mode == "hold_close") & (flat.cohort == "decile")
                & (flat.universe == "all")]
    print("\n=== A5 readout: era_A vs era_B demeaned alpha (pct) — "
          "universe=all / decile / hold_close ===")
    print(core[["formation", "hold", "n_A", "n_B", "alpha_A", "alpha_B",
                "pf_pooled"]].to_string(index=False))

    asm = cdf[(~cdf.diagnostic) & (cdf.exit_mode == "asm_exit")
              & (cdf.era == "pooled")]
    print("\n=== Falsifier #3 readout: asm_exit vs hold_close (pooled) ===")
    print(asm[["universe", "formation", "cohort", "hold", "n", "pct_truncated",
               "alpha_diff_vs_holdclose_pct", "alpha_pct", "pf_net"]]
          .to_string(index=False))

    diag = cdf[cdf.diagnostic & (cdf.era == "pooled")]
    print("\n=== DIAGNOSTIC-ONLY (never selectable): vol-scaled target/stop, "
          "universe=all ===")
    print(diag[["formation", "cohort", "hold", "exit_mode", "n", "pf_net",
                "diag_delta_vs_hold_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
