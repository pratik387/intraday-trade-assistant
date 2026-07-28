"""Phase 2 empirical signature: `xsec_momentum_demeaned` (lifecycle Stage 2).

Disciplined redo of the A4 weak kill (lessons.md #28): the original
`_tmp_a4_momentum_killtest.py` measured RAW winner-decile returns only and
asserted "survivorship beta" without ever demeaning the cross-section. This
script runs the pre-registered grid of
`specs/2026-07-27-brief-xsec_momentum_demeaned.md` section 5 and makes the
DEMEANED winner-cohort alpha the decisive statistic per cell:

    demeaned_alpha = mean(fwd_ret | winner cohort) - mean(fwd_ret | universe)

Phase 2 = raw footprint ONLY. No fees, no slippage, no exits, no leverage.

Causality guards
----------------
- Formation return = close[t-SKIP-L] -> close[t-SKIP], SKIP=2 (skips the most
  recent 2 days to avoid short-term-reversal contamination).
- ADV20 = 20d rolling mean turnover SHIFTED BY 1 DAY (uses t-20..t-1 only;
  the current day never enters universe selection).
- Entry = next-day open (t+1), exit = close at t+1+H. No look-ahead anywhere.

Discovery-window hard constraints (binding, per Stage-2 handoff)
----------------------------------------------------------------
- Rebalance dates: 2023-01-01 .. 2024-12-31 ONLY. Data extends to 2026-07-24
  but nothing after 2024-12-31 is ever used as a rebalance date.
- Forward holds may not extend past 2025-03-31: any rebalance whose exit close
  date falls after FWD_HARD_STOP is DROPPED for that hold cell and counted in
  `n_truncated` (constant-hold semantics preserved; nothing silently shortened).
- 250d formations only become feasible ~Jan-2024 (panel starts 2023-01-02);
  each cell reports its `effective_start` = first contributing rebalance date.

Grid representation
-------------------
- Return construction {raw, xsec-demeaned, market-adjusted} is emitted as three
  stat COLUMNS on every row (raw_winner_ret_pct / demeaned_alpha_pct /
  mktadj_alpha_pct) rather than as separate rows - all three are computed for
  every cell, nothing pruned.
- Market proxy = NIFTYBEES daily closes from the same clean panel (no Nifty 50
  index daily series covers 2023-24 on disk; the ETF tracks the index to bp).
- Size control `within_cap`: formation ranked WITHIN each cap segment
  (production accessor `services.symbol_metadata.get_cap_segment` - lesson #26)
  and demeaned against the segment's own mean forward return. NOTE: only the
  2026 snapshot exists on disk (no dated 2023-24 snapshots) so segments are
  as-of-latest; 1923/2673 symbols resolve `unknown` (they form their own
  segment = the thin tail).
- ADV 5-tier splits are emitted for every formation x hold combo at the
  weekly/decile/no-size-control corner (tier1 = lowest ADV quintile of the
  qualifying universe, tier5 = highest); winners ranked within tier, demeaned
  against tier mean.
- Loser cohort (mirror threshold) is reported as columns on every row.

Output: reports/sub9_sanity/_xsec_momentum_phase2_discovery.csv
        one row per cell, ALL cells including dead ones.

Kill criterion (lifecycle Stage 2): demeaned alpha < 0.1% per hold everywhere
=> the raw effect was beta/survivorship; kills this candidate AND permanently
parks `illiquid_momentum_long` (brief section 9).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from services.symbol_metadata import get_cap_segment  # noqa: E402  (lesson #26)

# ----------------------------------------------------------------------------
# LOCKED_GRID - pre-registered in specs/2026-07-27-brief-xsec_momentum_demeaned.md
# Do not edit after first full run.
# ----------------------------------------------------------------------------
LOCKED_GRID = dict(
    DATA_FILE=_REPO_ROOT / "cache/preaggregate/clean_daily_from5m.feather",
    DISCOVERY_START=pd.Timestamp("2023-01-01"),
    DISCOVERY_END=pd.Timestamp("2024-12-31"),      # last allowed rebalance date
    FWD_HARD_STOP=pd.Timestamp("2025-03-31"),      # exit close date must be <= this
    SKIP=2,
    FORMATIONS=(20, 60, 120, 250),                 # trading days
    HOLDS=(10, 20, 40, 60),                        # trading days
    COHORTS=(("decile", 0.10), ("quintile", 0.20), ("top5pct", 0.05)),
    REBALANCES=("weekly", "monthly"),              # weekly = every 5th trading day
    SIZE_CONTROLS=("none", "within_cap"),
    ADV_FLOOR=2_000_000.0,                         # Rs 20L
    ADV_WINDOW=20,
    N_ADV_TIERS=5,
    MIN_NAMES=50,                                  # min valid cross-section per rebalance
    MIN_SEG_NAMES=20,                              # min names for a segment to rank within-cap
    MKT_PROXY="NIFTYBEES",
    FFILL_LIMIT=5,                                 # close ffill for formation/exit valuation
)
OUT_CSV = _REPO_ROOT / "reports/sub9_sanity/_xsec_momentum_phase2_discovery.csv"


def load_panels():
    """Load the clean daily feather and pivot to date x symbol panels."""
    g = LOCKED_GRID
    dd = pd.read_feather(g["DATA_FILE"])
    dd["date"] = pd.to_datetime(dd["date"]).dt.normalize()
    dd["symbol"] = (
        dd["symbol"].astype(str).str.replace("NSE:", "", regex=False).str.upper()
    )
    print(f"[data] {g['DATA_FILE'].name}: shape={dd.shape} "
          f"dates {dd['date'].min().date()} -> {dd['date'].max().date()} "
          f"symbols={dd['symbol'].nunique()}")

    C = dd.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    O = dd.pivot_table(index="date", columns="symbol", values="open", aggfunc="last")
    V = dd.pivot_table(index="date", columns="symbol", values="volume", aggfunc="last")
    C = C.sort_index()
    O = O.reindex(C.index)
    V = V.reindex(C.index)

    cal = C.index.to_numpy()
    n_disc = ((C.index >= g["DISCOVERY_START"]) & (C.index <= g["DISCOVERY_END"])).sum()
    print(f"[data] union calendar {len(cal)} trading days; "
          f"{n_disc} inside Discovery rebalance window "
          f"{g['DISCOVERY_START'].date()}..{g['DISCOVERY_END'].date()}")

    # forward/formation valuation closes: limited ffill bridges per-symbol
    # no-trade gaps (suspensions); entry always requires a REAL next-day open.
    C_ff = C.ffill(limit=g["FFILL_LIMIT"])
    TURN = C * V
    ADV = TURN.rolling(g["ADV_WINDOW"], min_periods=g["ADV_WINDOW"]).mean().shift(1)
    return C, C_ff, O, ADV


def rebalance_positions(index, kind):
    """Positional indices (into the full union calendar) of rebalance dates
    inside the Discovery window."""
    g = LOCKED_GRID
    disc_mask = (index >= g["DISCOVERY_START"]) & (index <= g["DISCOVERY_END"])
    disc_pos = np.flatnonzero(disc_mask)
    if kind == "weekly":
        return disc_pos[::5]
    if kind == "monthly":
        months = index[disc_pos].to_period("M")
        first = pd.Series(disc_pos).groupby(months).first().to_numpy()
        return first
    raise ValueError(kind)


def cohort_indices(form, k, largest):
    """Indices of the k largest (winners) / smallest (losers) formation returns."""
    if largest:
        return np.argpartition(form, -k)[-k:]
    return np.argpartition(form, k - 1)[:k]


def main():
    g = LOCKED_GRID
    C, C_ff, O, ADV = load_panels()
    cal = C.index
    symbols = C.columns.to_numpy()
    n_days = len(cal)

    # cap segments via the production accessor (latest snapshot; documented above)
    seg_arr = np.array([get_cap_segment(s) for s in symbols])
    seg_names = sorted(set(seg_arr))
    print(f"[segments] {dict(pd.Series(seg_arr).value_counts())}")

    C_np, Cff_np, O_np, ADV_np = (df.to_numpy() for df in (C, C_ff, O, ADV))

    # precompute formation panels (positional shifts on the union calendar)
    FORM = {}
    for L in g["FORMATIONS"]:
        f = np.full_like(C_np, np.nan)
        lo, hi = g["SKIP"] + L, g["SKIP"]
        f[lo:, :] = Cff_np[lo - hi: n_days - hi, :] / Cff_np[: n_days - lo, :] - 1.0
        FORM[L] = f
        first_ok = np.flatnonzero(np.isfinite(f).sum(axis=1) >= g["MIN_NAMES"])
        print(f"[formation] L={L:3d}: first date with >={g['MIN_NAMES']} names = "
              f"{cal[first_ok[0]].date() if len(first_ok) else 'never'}")

    # forward-return panels: entry = open[t+1], exit = close_ff[t+1+H]
    ENTRY = np.full_like(O_np, np.nan)
    ENTRY[: n_days - 1, :] = O_np[1:, :]
    FWD = {}
    for H in g["HOLDS"]:
        f = np.full_like(C_np, np.nan)
        f[: n_days - 1 - H, :] = Cff_np[1 + H:, :] / ENTRY[: n_days - 1 - H, :] - 1.0
        FWD[H] = f

    mkt_col = int(np.flatnonzero(symbols == g["MKT_PROXY"])[0])

    rebal_pos = {k: rebalance_positions(cal, k) for k in g["REBALANCES"]}
    for k, pos in rebal_pos.items():
        print(f"[rebalance] {k}: {len(pos)} candidate dates "
              f"({cal[pos[0]].date()} -> {cal[pos[-1]].date()})")

    rows = []

    def run_cell(size_control, rebal, cohort_name, q, L, H, adv_tier="all"):
        """Evaluate one grid cell; returns the output row dict."""
        pos_all = rebal_pos[rebal]
        # hard forward-window constraint: exit close date must be <= FWD_HARD_STOP
        exit_pos = pos_all + 1 + H
        in_range = exit_pos < n_days
        exit_ok = np.zeros(len(pos_all), dtype=bool)
        exit_ok[in_range] = cal[exit_pos[in_range]] <= g["FWD_HARD_STOP"]
        n_truncated = int((~exit_ok).sum())
        pos = pos_all[exit_ok]

        per_reb = []  # (date, n_valid, raw_w, univ_mean, dm_alpha, mktadj, l_raw, l_dm)
        for p in pos:
            form = FORM[L][p]
            fwd = FWD[H][p]
            valid = (
                np.isfinite(form) & np.isfinite(fwd)
                & np.isfinite(ENTRY[p]) & (ENTRY[p] > 0)
                & np.isfinite(ADV_np[p]) & (ADV_np[p] >= g["ADV_FLOOR"])
                & np.isfinite(C_np[p])
            )
            vidx = np.flatnonzero(valid)
            if adv_tier != "all":
                # tier1 = lowest-ADV quintile of the qualifying universe
                order = vidx[np.argsort(ADV_np[p][vidx])]
                tiers = np.array_split(order, g["N_ADV_TIERS"])
                vidx = tiers[int(adv_tier[-1]) - 1]
            if len(vidx) < g["MIN_NAMES"] and adv_tier == "all":
                continue
            if len(vidx) < max(10, int(1 / q)) and adv_tier != "all":
                continue
            fv, rv = form[vidx], fwd[vidx]
            univ_mean = float(rv.mean())
            mkt_fwd = FWD[H][p, mkt_col]

            if size_control == "none" or adv_tier != "all":
                k = max(1, int(np.ceil(q * len(vidx))))
                wi = cohort_indices(fv, k, largest=True)
                li = cohort_indices(fv, k, largest=False)
                raw_w, raw_l = float(rv[wi].mean()), float(rv[li].mean())
                dm_w, dm_l = raw_w - univ_mean, raw_l - univ_mean
            else:  # within_cap: rank + demean inside each cap segment
                w_parts, l_parts, dmw_parts, dml_parts = [], [], [], []
                for sname in seg_names:
                    sidx = vidx[seg_arr[vidx] == sname]
                    if len(sidx) < g["MIN_SEG_NAMES"]:
                        continue
                    sf, sr = form[sidx], fwd[sidx]
                    smean = sr.mean()
                    k = max(1, int(np.ceil(q * len(sidx))))
                    swi = cohort_indices(sf, k, largest=True)
                    sli = cohort_indices(sf, k, largest=False)
                    w_parts.append(sr[swi]); l_parts.append(sr[sli])
                    dmw_parts.append(sr[swi] - smean); dml_parts.append(sr[sli] - smean)
                if not w_parts:
                    continue
                raw_w = float(np.concatenate(w_parts).mean())
                raw_l = float(np.concatenate(l_parts).mean())
                dm_w = float(np.concatenate(dmw_parts).mean())
                dm_l = float(np.concatenate(dml_parts).mean())

            mktadj = raw_w - float(mkt_fwd) if np.isfinite(mkt_fwd) else np.nan
            per_reb.append((cal[p], len(vidx), raw_w, univ_mean, dm_w, mktadj,
                            raw_l, dm_l))

        base = dict(size_control=size_control, rebalance=rebal, cohort=cohort_name,
                    formation_d=L, hold_d=H, adv_tier=adv_tier,
                    n_rebalances=len(per_reb), n_truncated=n_truncated)
        if not per_reb:
            base.update(effective_start="", n_names_avg=0.0,
                        raw_winner_ret_pct=np.nan, universe_mean_pct=np.nan,
                        demeaned_alpha_pct=np.nan, mktadj_alpha_pct=np.nan,
                        loser_raw_ret_pct=np.nan, loser_demeaned_alpha_pct=np.nan,
                        hit_rate=np.nan, t_stat_naive=np.nan)
            return base
        R = pd.DataFrame(per_reb, columns=["date", "n", "raw_w", "univ", "dm_w",
                                           "mktadj", "raw_l", "dm_l"])
        dm = R["dm_w"].to_numpy()
        t_naive = (dm.mean() / dm.std(ddof=1) * np.sqrt(len(dm))
                   if len(dm) > 2 and dm.std(ddof=1) > 0 else np.nan)
        base.update(
            effective_start=str(R["date"].iloc[0].date()),
            n_names_avg=round(float(R["n"].mean()), 1),
            raw_winner_ret_pct=round(float(R["raw_w"].mean()) * 100, 4),
            universe_mean_pct=round(float(R["univ"].mean()) * 100, 4),
            demeaned_alpha_pct=round(float(R["dm_w"].mean()) * 100, 4),
            mktadj_alpha_pct=round(float(R["mktadj"].mean()) * 100, 4),
            loser_raw_ret_pct=round(float(R["raw_l"].mean()) * 100, 4),
            loser_demeaned_alpha_pct=round(float(R["dm_l"].mean()) * 100, 4),
            hit_rate=round(float((dm > 0).mean()), 4),
            t_stat_naive=round(float(t_naive), 2) if np.isfinite(t_naive) else np.nan,
        )
        return base

    # ---- full grid: size_control x rebalance x cohort x formation x hold ----
    for size_control in g["SIZE_CONTROLS"]:
        for rebal in g["REBALANCES"]:
            for cohort_name, q in g["COHORTS"]:
                for L in g["FORMATIONS"]:
                    for H in g["HOLDS"]:
                        rows.append(run_cell(size_control, rebal, cohort_name, q, L, H))

    # ---- ADV 5-tier splits: weekly / decile / no-size-control corner --------
    for L in g["FORMATIONS"]:
        for H in g["HOLDS"]:
            for t in range(1, g["N_ADV_TIERS"] + 1):
                rows.append(run_cell("none", "weekly", "decile", 0.10, L, H,
                                     adv_tier=f"tier{t}"))

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\n[out] {len(out)} cells -> {OUT_CSV}")

    # ---- console summary: the decisive table --------------------------------
    core = out[(out.size_control == "none") & (out.rebalance == "weekly")
               & (out.cohort == "decile") & (out.adv_tier == "all")]
    print("\n=== DECISIVE: demeaned winner-decile alpha (pct per hold) — "
          "weekly / decile / no size control ===")
    print(core.pivot(index="formation_d", columns="hold_d",
                     values="demeaned_alpha_pct").to_string())
    print("\n=== raw winner-decile return (pct, NO evidential weight) ===")
    print(core.pivot(index="formation_d", columns="hold_d",
                     values="raw_winner_ret_pct").to_string())
    print("\n=== loser-decile demeaned alpha (pct) ===")
    print(core.pivot(index="formation_d", columns="hold_d",
                     values="loser_demeaned_alpha_pct").to_string())
    alive = out[(out.adv_tier == "all")
                & (out.demeaned_alpha_pct >= 0.1)].shape[0]
    total = out[out.adv_tier == "all"].shape[0]
    print(f"\n[floor] cells with demeaned alpha >= 0.10% per hold: "
          f"{alive}/{total} (excl. tier rows)")


if __name__ == "__main__":
    main()
