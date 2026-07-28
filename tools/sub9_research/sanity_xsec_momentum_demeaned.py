"""Stage 4 sanity: `xsec_momentum_demeaned` — Discovery-only position ledger.

Ranker-style weekly cross-sectional selection (A2 / mtf_capitulation family:
`services/cross_sectional_ranker.py` conventions — one ledger row per selected
position per rebalance, RAW forward returns, no fees / no exits geometry).
Brief: specs/2026-07-27-brief-xsec_momentum_demeaned.md
Phase 2: tools/sub9_research/phase2_xsec_momentum_signature.py (panel
construction reused verbatim; this script narrows to the Phase-2 winning
cells and adds the production-accessor universe gate + per-position ledger).

Decisive statistic: DEMEANED alpha = position fwd return minus the UNIVERSE
MEAN forward return over the same window (per rebalance). Raw returns carry
no evidential weight (lessons.md #28 — that was A4's mistake).

--------------------------------------------------------------------------
6-failure-mode checklist (docs/setup_lifecycle.md Stage 4, Lesson #5) —
mapped to this daily-panel ranker context:

  #1 Intraday-aggregate look-ahead: N/A by construction — daily panel only.
     Formation return = close[t-SKIP-L] -> close[t-SKIP] (SKIP=2); nothing
     dated after the rebalance date t enters selection.
  #2 Baseline includes current bar: ADV20 = 20d rolling mean turnover
     SHIFTED BY 1 DAY (uses t-20..t-1 only; day t never enters its own
     universe filter).
  #3 Mode B entry walk: entry = rebalance+1 session's REAL open (no ffill
     — a genuine print is required or the position is not formed). Forward
     valuation and the MFE/MAE walk start at the entry session t+1: under
     Mode-B-open entry the entry day's full high/low range occurs AFTER the
     open, so t+1 is the correct first walk bar (lifecycle #3 semantics).
  #4 Same-bar SL/T2 ambiguity: N/A — Stage 4 for this ranker candidate has
     NO stop/target geometry; only raw H-session forward returns and
     mfe/mae diagnostics are recorded. Documented as N/A, not skipped.
  #5 Cell-locked filters at signal time: LOCKED_FILTERS constant below,
     written before the first run, from the Phase-2 winning cells. No
     tuning after first run. Universe gate = ProductionUniverseGate
     (production accessor), cap_segment = services.symbol_metadata
     .get_cap_segment (production accessor, lesson #26).
  #6 Reproducibility floor: selection is reproducible by
     services/cross_sectional_ranker.py-style machinery (same clean panel,
     same causal shifts). sanity_csv_schema validation is N/A for this
     ranker-style forward-return ledger (no per-trade exit price yet —
     exit geometry is a later stage); the ledger schema follows the
     multiday A2-family convention instead and is documented below.

Additional anti-bias guards (documented per Stage-4 handoff):
  - Discovery ONLY: every rebalance date asserted inside
    2023-01-01..2024-12-31; RAISES on violation. Forward legs extend into
    2025 as EXITS ONLY; the max valuation date is asserted <= 2025-12-31
    (it lands ~2025-03 for the last Dec-2024 rebalance + 60 sessions).
  - Universe membership uses ONLY date-t-available data (formation, shifted
    ADV, gate, a real close at t) plus the Mode-B entry-feasibility
    requirement (a real open at t+1 — the entry mechanic itself). Universe
    membership is NOT conditioned on forward survival: a name that stops
    printing >FFILL_LIMIT sessions before an H-valuation date yields NaN
    fwd at that H and is nanmean-excluded (counts reported), rather than
    being silently dropped from selection (no survival look-ahead).
  - MTF-eligible flag: data/mtf_universe/approved_mtf_securities_latest.json
    is a 2026-era snapshot applied to 2023-24 rebalances — ANACHRONISTIC
    (Lesson #27: no historical MTF lists exist). The flag is a capturability
    PREVIEW column only, never a filter; paper is the production gate.
  - cap_segment: production accessor resolves from the 2026 snapshot (no
    dated 2023-24 snapshots on disk) — as-of-latest, same caveat as Phase 2;
    ~72% of symbols resolve 'unknown' (the thin tail is its own segment).
  - Overlapping holds: weekly rebalance with 20-60-session horizons means
    positions overlap across rebalances; ledger rows are autocorrelated.
    Stats here are per-position diagnostics, not independent trades.

Output ledger (one row per selected position):
  symbol, rebalance_date, entry_date, entry_open, formation_variant,
  formation_ret_pct, adv_tier (1=lowest ADV quintile of the qualifying
  universe at that rebalance), cap_segment, mtf_eligible,
  fwd_h{20,40,60}_pct (RAW %, no fees), universe_h{20,40,60}_pct (universe
  nanmean over the same window, per rebalance — demeaned alpha per row =
  fwd - universe), alpha_h{20,40,60}_pct (precomputed convenience),
  mfe_pct / mae_pct (daily highs/lows over the h60 window starting at the
  entry session, vs entry_open), n_universe.

-> reports/sub9_sanity/_xsec_momentum_demeaned_trades_discovery.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from services.symbol_metadata import get_cap_segment  # noqa: E402  (lesson #26)
from tools.sub9_research.production_universe import ProductionUniverseGate  # noqa: E402

# ----------------------------------------------------------------------------
# LOCKED_FILTERS — from the Phase-2 winning cells
# (phase2_xsec_momentum_signature.py: weekly / decile / no-size-control,
#  formations 60 & 120). DO NOT EDIT after the first Discovery run.
# ----------------------------------------------------------------------------
LOCKED_FILTERS = dict(
    DATA_FILE=_REPO_ROOT / "cache/preaggregate/clean_daily_from5m.feather",
    DISCOVERY_START=pd.Timestamp("2023-01-01"),
    DISCOVERY_END=pd.Timestamp("2024-12-31"),   # last allowed rebalance date
    EXIT_HARD_STOP=pd.Timestamp("2025-12-31"),  # forward legs: 2025 exits only
    SKIP=2,
    FORMATIONS=(60, 120),                       # two variants (variant column)
    HORIZONS=(20, 40, 60),                      # forward-return sessions
    COHORT_PCT=0.10,                            # winner decile
    REBALANCE_EVERY=5,                          # weekly = every 5th trading day
    ADV_FLOOR=2_000_000.0,                      # Rs 20L
    ADV_WINDOW=20,
    N_ADV_TIERS=5,
    MIN_NAMES=50,                               # min valid cross-section
    FFILL_LIMIT=5,                              # close ffill for valuation only
    MTF_SNAPSHOT=_REPO_ROOT / "data/mtf_universe/approved_mtf_securities_latest.json",
    # ProductionUniverseGate config: Stage-4 pre-cell-lock — all cap segments
    # admitted (cap_segment is a ledger DIMENSION here, cell-lock comes later);
    # CNC/MTF delivery long => no MIS requirement; MTF is a reported flag,
    # never a filter (capturability falsifier #2 is measured, not imposed).
    GATE_ACCEPTED_CAPS={"large_cap", "mid_cap", "small_cap", "micro_cap", "unknown"},
    GATE_REQUIRE_MIS=False,
    GATE_REQUIRE_MTF=False,
)
OUT_CSV = _REPO_ROOT / "reports/sub9_sanity/_xsec_momentum_demeaned_trades_discovery.csv"


def load_panels():
    """Clean daily feather -> date x symbol panels (Phase-2 construction)."""
    g = LOCKED_FILTERS
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


def main():
    g = LOCKED_FILTERS
    panels = load_panels()
    C, O, H, L, V = (panels[k] for k in ("close", "open", "high", "low", "volume"))
    cal = C.index
    symbols = C.columns.to_numpy()
    n_days = len(cal)

    # Guard #2: ADV20 shifted — day t never enters its own filter.
    ADV = (C * V).rolling(g["ADV_WINDOW"], min_periods=g["ADV_WINDOW"]).mean().shift(1)
    # Valuation closes: limited ffill bridges short suspensions ONLY for
    # formation/forward valuation. Entry requires a REAL open (guard #3).
    C_ff = C.ffill(limit=g["FFILL_LIMIT"])

    C_np, Cff_np, O_np = C.to_numpy(), C_ff.to_numpy(), O.to_numpy()
    H_np, L_np, ADV_np = H.to_numpy(), L.to_numpy(), ADV.to_numpy()

    # Formation panels: close[t-SKIP-L] -> close[t-SKIP]  (guard #1: causal).
    FORM = {}
    for Lf in g["FORMATIONS"]:
        f = np.full_like(C_np, np.nan)
        lo, hi = g["SKIP"] + Lf, g["SKIP"]
        f[lo:, :] = Cff_np[lo - hi: n_days - hi, :] / Cff_np[: n_days - lo, :] - 1.0
        FORM[Lf] = f

    # Forward panels: entry = REAL open[t+1]; exit valuation = close_ff[t+1+Hh].
    ENTRY = np.full_like(O_np, np.nan)
    ENTRY[: n_days - 1, :] = O_np[1:, :]
    FWD = {}
    for Hh in g["HORIZONS"]:
        f = np.full_like(C_np, np.nan)
        f[: n_days - 1 - Hh, :] = Cff_np[1 + Hh:, :] / ENTRY[: n_days - 1 - Hh, :] - 1.0
        FWD[Hh] = f

    # Weekly rebalance dates inside Discovery (guard: assertion + raise).
    disc_mask = (cal >= g["DISCOVERY_START"]) & (cal <= g["DISCOVERY_END"])
    rebal_pos = np.flatnonzero(disc_mask)[:: g["REBALANCE_EVERY"]]
    rd = cal[rebal_pos]
    if not ((rd >= g["DISCOVERY_START"]) & (rd <= g["DISCOVERY_END"])).all():
        raise AssertionError("DISCOVERY VIOLATION: rebalance date outside "
                             f"{g['DISCOVERY_START'].date()}..{g['DISCOVERY_END'].date()}")
    h_max = max(g["HORIZONS"])
    last_val_pos = int(rebal_pos.max()) + 1 + h_max
    if last_val_pos >= n_days or cal[last_val_pos] > g["EXIT_HARD_STOP"]:
        raise AssertionError("EXIT-WINDOW VIOLATION: forward valuation extends "
                             "past 2025 (exits-only constraint)")
    print(f"[rebalance] weekly: {len(rebal_pos)} dates "
          f"({rd[0].date()} -> {rd[-1].date()}); "
          f"max forward valuation date = {cal[last_val_pos].date()} (<= 2025 OK)")

    # Production accessors (lesson #26): cap segment + universe gate + MTF flag.
    seg_arr = np.array([get_cap_segment(s) for s in symbols])
    print(f"[segments] {dict(pd.Series(seg_arr).value_counts())}")
    gate = ProductionUniverseGate(
        accepted_caps=g["GATE_ACCEPTED_CAPS"],
        require_mis=g["GATE_REQUIRE_MIS"],
        require_mtf=g["GATE_REQUIRE_MTF"],
        min_trading_days_required=0,   # zeroed per Lesson #17
        min_daily_avg_volume=0,        # zeroed per Lesson #17
        mtf_snapshot_path=g["MTF_SNAPSHOT"],
        exclude_etf=False,
    )
    # Gate is config-static per symbol under this Stage-4 config (no daily
    # history thresholds), but evaluated through the production accessor per
    # rebalance date for interface fidelity.
    import json
    with open(g["MTF_SNAPSHOT"], encoding="utf-8") as fh:
        mtf_set = {str(e.get("tradingsymbol", "")).upper() for e in json.load(fh)}
    mtf_arr = np.isin(symbols, list(mtf_set))
    print(f"[mtf] snapshot {g['MTF_SNAPSHOT'].name}: {int(mtf_arr.sum())}/{len(symbols)} "
          f"panel symbols MTF-eligible (2026-era snapshot — ANACHRONISTIC for "
          f"2023-24, preview column only)")

    rows = []
    nan_fwd_counts = {Hh: 0 for Hh in g["HORIZONS"]}

    for p in rebal_pos:
        p = int(p)
        reb_date = cal[p]
        gate_ok = np.array([gate.is_eligible(s, session_date=reb_date.date())
                            for s in symbols])
        # Universe: date-t data + Mode-B entry feasibility ONLY (no forward
        # survival conditioning — see header).
        base_valid = (
            gate_ok
            & np.isfinite(C_np[p])
            & np.isfinite(ADV_np[p]) & (ADV_np[p] >= g["ADV_FLOOR"])
            & np.isfinite(ENTRY[p]) & (ENTRY[p] > 0)
        )
        for Lf in g["FORMATIONS"]:
            valid = base_valid & np.isfinite(FORM[Lf][p])
            vidx = np.flatnonzero(valid)
            if len(vidx) < g["MIN_NAMES"]:
                continue
            # ADV quintile tiers of the qualifying universe (tier1 = lowest).
            order = vidx[np.argsort(ADV_np[p][vidx])]
            tier_of = {}
            for t_i, chunk in enumerate(np.array_split(order, g["N_ADV_TIERS"]), start=1):
                for ci in chunk:
                    tier_of[int(ci)] = t_i
            # Universe mean forward return per horizon (nanmean; NaN = name
            # stopped printing before valuation — counted, not dropped).
            univ_mean = {}
            for Hh in g["HORIZONS"]:
                fw = FWD[Hh][p][vidx]
                nan_fwd_counts[Hh] += int(np.isnan(fw).sum())
                univ_mean[Hh] = float(np.nanmean(fw))
            # Winner decile of the formation cross-section.
            form_v = FORM[Lf][p][vidx]
            k = max(1, int(np.ceil(g["COHORT_PCT"] * len(vidx))))
            widx = vidx[np.argpartition(form_v, -k)[-k:]]
            # MFE/MAE window: entry session p+1 .. p+1+h_max (guard #3).
            w0, w1 = p + 1, p + 1 + h_max + 1
            for ci in widx:
                ci = int(ci)
                entry_open = float(ENTRY[p, ci])
                hi_win = H_np[w0:w1, ci]
                lo_win = L_np[w0:w1, ci]
                mfe = (float(np.nanmax(hi_win)) / entry_open - 1.0) * 100.0 \
                    if np.isfinite(hi_win).any() else np.nan
                mae = (float(np.nanmin(lo_win)) / entry_open - 1.0) * 100.0 \
                    if np.isfinite(lo_win).any() else np.nan
                row = dict(
                    symbol=symbols[ci],
                    rebalance_date=str(reb_date.date()),
                    entry_date=str(cal[p + 1].date()),
                    entry_open=round(entry_open, 4),
                    formation_variant=f"f{Lf}",
                    formation_ret_pct=round(float(FORM[Lf][p, ci]) * 100.0, 4),
                    adv_tier=tier_of[ci],
                    cap_segment=seg_arr[ci],
                    mtf_eligible=bool(mtf_arr[ci]),
                    n_universe=len(vidx),
                )
                for Hh in g["HORIZONS"]:
                    fw = FWD[Hh][p, ci]
                    fw_pct = round(float(fw) * 100.0, 4) if np.isfinite(fw) else np.nan
                    um_pct = round(univ_mean[Hh] * 100.0, 4)
                    row[f"fwd_h{Hh}_pct"] = fw_pct
                    row[f"universe_h{Hh}_pct"] = um_pct
                    row[f"alpha_h{Hh}_pct"] = (round(fw_pct - um_pct, 4)
                                               if np.isfinite(fw) else np.nan)
                row["mfe_pct"] = round(mfe, 4) if np.isfinite(mfe) else np.nan
                row["mae_pct"] = round(mae, 4) if np.isfinite(mae) else np.nan
                rows.append(row)

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\n[out] {len(out)} position rows -> {OUT_CSV}")
    print(f"[fwd-NaN] universe fwd NaN counts (suspension/delist inside window): "
          f"{nan_fwd_counts}")

    # ------------------- console summary (Stage-4 read) -------------------
    pd.set_option("display.width", 160)
    for var in sorted(out["formation_variant"].unique()):
        d = out[out.formation_variant == var]
        print(f"\n=== variant {var}: n={len(d)} positions, "
              f"{d.rebalance_date.nunique()} rebalances, "
              f"{d.symbol.nunique()} distinct symbols ===")
        for Hh in g["HORIZONS"]:
            a = d[f"alpha_h{Hh}_pct"]
            print(f"  H{Hh}: mean demeaned alpha {a.mean():+.3f}% | "
                  f"median {a.median():+.3f}% | hit-rate vs universe "
                  f"{(a > 0).mean():.3f} | n_valid {a.notna().sum()}")
        print(f"  mfe mean {d.mfe_pct.mean():+.2f}% / mae mean {d.mae_pct.mean():+.2f}%")
        print("  per-ADV-tier alpha_h60 (mean % / n):")
        print("   ", d.groupby("adv_tier")["alpha_h60_pct"]
                       .agg(["mean", "count"]).round(3).to_dict("index"))
        print("  per-cap-segment alpha_h60 (mean % / n):")
        print("   ", d.groupby("cap_segment")["alpha_h60_pct"]
                       .agg(["mean", "count"]).round(3).to_dict("index"))
        m = d[d.mtf_eligible]
        print(f"  MTF-eligible subset: n={len(m)} "
              f"({len(m) / max(len(d), 1):.1%}) | alpha h20/h40/h60 = "
              f"{m.alpha_h20_pct.mean():+.3f}/{m.alpha_h40_pct.mean():+.3f}/"
              f"{m.alpha_h60_pct.mean():+.3f}%  (full: "
              f"{d.alpha_h20_pct.mean():+.3f}/{d.alpha_h40_pct.mean():+.3f}/"
              f"{d.alpha_h60_pct.mean():+.3f}%)")

    # Overlap between the two formation variants (same rebalance dates).
    s60 = out[out.formation_variant == "f60"].groupby("rebalance_date")["symbol"].apply(set)
    s120 = out[out.formation_variant == "f120"].groupby("rebalance_date")["symbol"].apply(set)
    common = sorted(set(s60.index) & set(s120.index))
    jac, share120in60 = [], []
    for rdte in common:
        a, b = s60[rdte], s120[rdte]
        jac.append(len(a & b) / len(a | b))
        share120in60.append(len(a & b) / len(b))
    if common:
        print(f"\n=== f60 vs f120 selection overlap ({len(common)} shared rebalances) ===")
        print(f"  mean Jaccard {np.mean(jac):.3f} | "
              f"mean share of f120 picks also in f60 {np.mean(share120in60):.3f}")


if __name__ == "__main__":
    main()
