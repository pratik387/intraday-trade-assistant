"""DIAGNOSTIC: where does expectancy live NOW? (universe re-calibration by era)

NOT a candidate search. This re-reads the per-trade ledgers of setups we ALREADY
OWN (live + retired) and asks, per mechanism:

    in which cap-segment / ADV tier does net expectancy live in era_B
    (2025-01-01 .. 2026-04-30), and how does that differ from era_A
    (2023-01-01 .. 2024-12-31)?

Motivation (measured 2026-07-28): the illiquidity premium flipped sign.
Universe 10-session drift ran +1.1%..+3.3%/qtr through 2023Q2-2024Q2 and is
~0/negative from 2024Q4; the illiquid-minus-liquid quintile spread went from
+1.9%/10d to -0.5..-0.8% from 2025Q4. The system's founding belief ("our edge
is the illiquid tail") was measured INSIDE that tailwind and every universe
builder / cap filter / ADV band inherits it untested on era_B alone.

HONESTY CONSTRAINTS (also printed to stdout):
  * This is a diagnostic over ALREADY-BURNED windows (Discovery + OOS + Holdout
    are all inside 2023-01..2026-04). Nothing here is a validation.
  * Any universe re-cut it suggests is a NEW cell selection and must go through
    its own freeze + fresh-pool one-shot (docs/setup_lifecycle.md amendment A1).
    Do NOT adopt these numbers as evidence for the re-cut.
  * No signal dated >= 2026-05-01 is touched (fresh pool is not burned here).
  * Fees are NEVER re-derived; each ledger's own net-PnL column is used as-is.

Outputs
  reports/sub9_sanity/_universe_expectancy_by_era.csv   (tidy long format)
  console: source table + all matrices + answers to (a)/(b)/(c)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from services.symbol_metadata import get_cap_segment  # noqa: E402

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 400)

REPORTS = ROOT / "reports"
SANITY = REPORTS / "sub9_sanity"
CLEAN_DAILY = ROOT / "cache" / "preaggregate" / "clean_daily_from5m.feather"
OUT_CSV = SANITY / "_universe_expectancy_by_era.csv"

# ---------------------------------------------------------------- eras
ERA_A = ("2023-01-01", "2024-12-31")
ERA_B = ("2025-01-01", "2026-04-30")
FRESH_POOL_START = pd.Timestamp("2026-05-01")   # hard wall — never crossed

# ---------------------------------------------------------------- ADV tiers
# FIXED absolute bands (not per-era quantiles) so era_A and era_B are directly
# comparable. Per-era quintiles would be relative and would hide a level shift
# in the universe itself, which is precisely what we are testing for.
# Share bands are anchored on the system's own language ("<500K avg vol" =
# the illiquid tail that supposedly carries 78% of trades / 88% of PnL).
ADV_SHARE_BINS = [0, 100_000, 500_000, 2_000_000, 10_000_000, np.inf]
ADV_SHARE_LABELS = [
    "S1_<100K", "S2_100K-500K", "S3_500K-2M", "S4_2M-10M", "S5_>=10M",
]
# Rupee-turnover bands (crore/day). Robustness axis: NSE prices rose materially
# 2023->2026, so a fixed SHARE band slowly reclassifies names; turnover does not.
ADV_TURNOVER_BINS = [0, 1e7, 5e7, 2.5e8, 1e9, np.inf]   # 1cr / 5cr / 25cr / 100cr
ADV_TURNOVER_LABELS = [
    "V1_<1cr", "V2_1-5cr", "V3_5-25cr", "V4_25-100cr", "V5_>=100cr",
]

# ---------------------------------------------------------------- ledgers
# cluster: factor-cluster membership from the 2026-07-27 factor-concentration
# study. A2/C1/C4/C6 + panic_crash are ONE capitulation factor (r 0.55-0.84,
# PC1 38%) and MUST NOT be counted as independent confirmations of each other.
OCI_COLS = dict(date="signal_date", sym="symbol", net="net_pnl_inr")

LEDGERS = [
    # --- intraday, OCI-canonical v2 (production-parity gated ledgers) --------
    ("gap_fade_short",               "oci_canonical_v2/gap_fade_short_oci_canonical.csv",               "oci", "short_open_fade",   "RETIRED 2026-07-27"),
    ("or_window_failure_fade_short", "oci_canonical_v2/or_window_failure_fade_short_oci_canonical.csv", "oci", "short_open_fade",   "retired"),
    ("mis_unwind_vwap_revert_short", "oci_canonical_v2/mis_unwind_vwap_revert_short_oci_canonical.csv", "oci", "short_vwap_fade",   "retired"),
    ("round_number_sweep_short",     "oci_canonical_v2/round_number_sweep_short_oci_canonical.csv",     "oci", "short_vwap_fade",   "retired"),
    ("circuit_release_fade_short",   "oci_canonical_v2/circuit_release_fade_short_oci_canonical.csv",   "oci", "short_circuit",     "retired"),
    ("circuit_t1_fade_short",        "oci_canonical_v2/circuit_t1_fade_short_oci_canonical.csv",        "oci", "short_circuit",     "retired"),
    ("delivery_pct_anomaly_short",   "oci_canonical_v2/delivery_pct_anomaly_short_oci_canonical.csv",   "oci", "short_delivery",    "retired"),
    ("below_vwap_volume_revert_long","oci_canonical_v2/below_vwap_volume_revert_long_oci_canonical.csv","oci", "below_vwap_long",   "candidate (2026+ data only)"),
    ("long_panic_gap_down",          "oci_canonical_v2/long_panic_gap_down_oci_canonical.csv",          "oci", "capitulation_long", "retired"),
    # --- intraday, OCI-canonical v1 (setups with no v2 rerun) ---------------
    ("panic_crash_revert_long",      "oci_canonical/panic_crash_revert_long_oci_canonical.csv",         "oci", "capitulation_long", "candidate"),
    ("up_spike_fade_short",          "oci_canonical/up_spike_fade_short_oci_canonical.csv",             "oci", "short_open_fade",   "retired"),
    # --- multiday capitulation basket (monster-conditioning BASELINE arm) ---
    ("mtf_capitulation_revert_long",   "sub9_sanity/_monster_cond_mtf_capitulation_revert_long_baseline.csv",   "monster", "capitulation_long", "candidate (multiday)"),
    ("low52_capitulation_revert_long", "sub9_sanity/_monster_cond_low52_capitulation_revert_long_baseline.csv", "monster", "capitulation_long", "candidate (multiday)"),
    ("zscore_oversold_revert_long",    "sub9_sanity/_monster_cond_zscore_oversold_revert_long_baseline.csv",    "monster", "capitulation_long", "candidate (multiday)"),
    ("crash2d_revert_long",            "sub9_sanity/_monster_cond_crash2d_revert_long_baseline.csv",            "monster", "capitulation_long", "candidate (multiday)"),
    # --- overnight ----------------------------------------------------------
    ("close_dn_overnight_long",      "sub9_sanity/_close_dn_overnight_long_5bar_trades_*.csv",           "close_dn", "close_dn_overnight", "LIVE (cell #5)"),
]

MONSTER_NOTIONAL = 100_000.0 * 2.79   # MARGIN * LEV, from multiday_target_exit_study


def _norm_symbol(s: str) -> str:
    s = str(s).strip().upper()
    return s.split(":", 1)[1] if ":" in s else s


# ---------------------------------------------------------------- loaders
def load_ledger(name: str, rel: str, kind: str) -> pd.DataFrame:
    """Return tidy per-trade frame: setup, signal_date, symbol, net_pnl_inr,
    notional_inr, net_bps. `notional_inr` is POSITION notional (not margin) so
    net_bps is comparable across setups with different sizing plans."""
    if kind == "oci":
        df = pd.read_csv(REPORTS / rel, parse_dates=["signal_date"])
        out = pd.DataFrame({
            "signal_date": df["signal_date"],
            "symbol": df["symbol"].map(_norm_symbol),
            "net_pnl_inr": df["net_pnl_inr"].astype(float),
            "notional_inr": (df["entry_price"].astype(float) * df["qty"].astype(float)),
        })
        net_desc = "net_pnl_inr (OCI canonical; realized - fee_inr, real Zerodha MIS model)"

    elif kind == "monster":
        df = pd.read_csv(REPORTS / rel, parse_dates=["signal_date"])
        out = pd.DataFrame({
            "signal_date": df["signal_date"],
            "symbol": df["symbol"].map(_norm_symbol),
            "net_pnl_inr": df["net_pnl_inr"].astype(float),
            "notional_inr": MONSTER_NOTIONAL,
        })
        net_desc = "net_pnl_inr (monster-cond BASELINE arm; real MTF fee model @ Rs1L x 2.79 lev)"

    elif kind == "close_dn":
        parts = []
        for p in sorted((REPORTS.parent / "reports" / "sub9_sanity").glob(Path(rel).name)):
            parts.append(pd.read_csv(p, parse_dates=["signal_date"]))
        df = pd.concat(parts, ignore_index=True)
        # locked Cell #5 per tools/sub9_research/close_dn_overnight_long_cell_lock.json
        m = (df["closing_30m_volume_z_bin"] == "extreme") & (df["prior_day_return_bin"] == "up_gt_3pct")
        df = df.loc[m]
        out = pd.DataFrame({
            "signal_date": df["signal_date"],
            "symbol": df["symbol"].map(_norm_symbol),
            "net_pnl_inr": df["net_pnl_inr"].astype(float),
            "notional_inr": (df["entry_price"].astype(float) * df["qty"].astype(float)),
        })
        net_desc = "net_pnl_inr (5-bar realistic-live sanity, CNC fee model; cell #5 filter applied here)"
    else:
        raise ValueError(kind)

    out["setup"] = name
    out["_net_desc"] = net_desc
    return out.reset_index(drop=True)


# ---------------------------------------------------------------- ADV panel
def build_adv_panel() -> pd.DataFrame:
    """Causal ADV20 from clean_daily_from5m.feather.

    WHY this panel and not consolidated_daily.feather:
      * clean_daily_from5m is CA-adjusted and bad-print-cleaned (lessons #24);
        consolidated_daily is UNADJUSTED and carries bad prints, which would
        corrupt turnover bands directly.
      * it is derived from the SAME 5m archive the backtests replayed, so the
        liquidity a trade actually saw and the tier we assign it agree.
      * it extends to 2026-07 (consolidated stops 2026-04-10) and covers 2673
        symbols vs 2206.

    Causality: adv20 at date D uses volume over the 20 sessions ENDING AT D-1
    (rolling(20).mean().shift(1)) — nothing from the signal day itself.
    """
    d = pd.read_feather(CLEAN_DAILY)
    d["symbol"] = d["symbol"].map(_norm_symbol)
    d = d.sort_values(["symbol", "date"]).reset_index(drop=True)
    d["turnover"] = d["volume"] * d["close"]
    g = d.groupby("symbol", sort=False)
    d["adv20_shares"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=15).mean()).groupby(d["symbol"]).shift(1)
    d["adv20_turnover"] = g["turnover"].transform(lambda s: s.rolling(20, min_periods=15).mean()).groupby(d["symbol"]).shift(1)
    return d[["symbol", "date", "adv20_shares", "adv20_turnover"]]


def attach_adv(trades: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """merge_asof backward per symbol: take the most recent panel row at or
    before signal_date (handles symbols missing the exact session)."""
    t = trades.sort_values("signal_date").reset_index(drop=True)
    p = panel.dropna(subset=["adv20_shares"]).sort_values("date").reset_index(drop=True)
    merged = pd.merge_asof(
        t, p,
        left_on="signal_date", right_on="date", by="symbol",
        direction="backward", tolerance=pd.Timedelta(days=10),
    )
    return merged


# ---------------------------------------------------------------- metrics
def agg(df: pd.DataFrame) -> pd.Series:
    net = df["net_pnl_inr"]
    w, l = net[net > 0].sum(), -net[net < 0].sum()
    pf = (w / l) if l > 0 else (np.inf if w > 0 else np.nan)
    return pd.Series({
        "n": int(len(df)),
        "net_pf": round(float(pf), 3) if np.isfinite(pf) else np.nan,
        "exp_inr": round(float(net.mean()), 1),
        "exp_bps": round(float(df["net_bps"].mean()), 2),
        "wr_pct": round(float((net > 0).mean() * 100), 1),
        "net_inr_total": round(float(net.sum())),
    })


def matrix(df: pd.DataFrame, dim: str, min_n: int = 25) -> pd.DataFrame:
    rows = df.groupby(["setup", dim, "era"], observed=True).apply(agg, include_groups=False).reset_index()
    rows["low_n"] = rows["n"] < min_n
    return rows


def pivot_exp(rows: pd.DataFrame, dim: str, value: str = "exp_bps", min_n: int = 25) -> pd.DataFrame:
    r = rows.copy()
    r.loc[r["n"] < min_n, value] = np.nan
    pv = r.pivot_table(index="setup", columns=[dim, "era"], values=value, observed=True)
    return pv


# ---------------------------------------------------------------- main
def main() -> None:
    print("=" * 118)
    print("DIAGNOSTIC — universe expectancy by era  (development window only; fresh pool NOT touched)")
    print(f"  era_A = {ERA_A[0]} .. {ERA_A[1]}      era_B = {ERA_B[0]} .. {ERA_B[1]}")
    print("=" * 118)

    # ---- STEP 1: source table
    frames, src_rows = [], []
    for name, rel, kind, cluster, status in LEDGERS:
        try:
            df = load_ledger(name, rel, kind)
        except Exception as e:                                    # noqa: BLE001
            print(f"  !! SKIP {name}: {e}")
            continue
        pre = len(df)
        df = df[df["signal_date"] < FRESH_POOL_START]             # hard wall
        df["cluster"] = cluster
        frames.append(df)
        src_rows.append({
            "setup": name, "cluster": cluster, "status": status,
            "file": rel, "n": len(df), "dropped_fresh_pool": pre - len(df),
            "from": df["signal_date"].min().date(), "to": df["signal_date"].max().date(),
            "net_col": df["_net_desc"].iloc[0],
        })

    src = pd.DataFrame(src_rows)
    print("\n### STEP 1 — SOURCE TABLE (per-trade ledgers on disk)")
    print(src[["setup", "cluster", "status", "n", "from", "to", "file"]].to_string(index=False))
    print("\n  net-PnL column provenance (fees NOT re-derived anywhere):")
    for d in src["net_col"].unique():
        print(f"    - {d}")
    print(f"\n  TOTAL trades: {int(src['n'].sum())} across {len(src)} setups; "
          f"{int(src['dropped_fresh_pool'].sum())} rows dropped for signal_date >= 2026-05-01.")

    trades = pd.concat(frames, ignore_index=True)

    # ---- STEP 2a: cap_segment via the PRODUCTION accessor, point-in-time
    print("\n### STEP 2a — cap_segment via services.symbol_metadata.get_cap_segment(symbol, session_date)")
    uniq = trades[["symbol", "signal_date"]].drop_duplicates()
    uniq["_d"] = uniq["signal_date"].dt.date
    pairs = uniq[["symbol", "_d"]].drop_duplicates()
    cap_map = {(s, d): get_cap_segment(s, d) for s, d in pairs.itertuples(index=False)}
    trades["cap_segment"] = [
        cap_map[(s, d.date())] for s, d in trades[["symbol", "signal_date"]].itertuples(index=False)
    ]
    snaps = sorted(p.name for p in (ROOT / "data" / "cap_segments").glob("cap_segments_*.json"))
    print(f"    snapshots on disk: {snaps}")
    print("    CAVEAT (C-09 finding): the only dated snapshot is 2026-05-27, which POST-DATES every")
    print("    signal in this study. _pick_snapshot_path() therefore finds no snapshot <= session_date")
    print("    for any dev-window date and degrades to the legacy nse_all.json map — i.e. cap_segment")
    print("    here is TODAY's classification applied to history. It is a structural size proxy, not a")
    print("    point-in-time one, and it carries survivorship (names that grew into mid/large).")
    print("    The ADV20 tier below is the fully causal axis and is the one to trust.")
    print(trades["cap_segment"].value_counts().to_string())

    # ---- STEP 2b: ADV20 tiers (causal)
    print("\n### STEP 2b — ADV20 tiers from cache/preaggregate/clean_daily_from5m.feather (causal, shift(1))")
    panel = build_adv_panel()
    trades = attach_adv(trades, panel)
    trades["adv_tier"] = pd.cut(trades["adv20_shares"], ADV_SHARE_BINS, labels=ADV_SHARE_LABELS, right=False)
    trades["turnover_tier"] = pd.cut(trades["adv20_turnover"], ADV_TURNOVER_BINS, labels=ADV_TURNOVER_LABELS, right=False)
    trades["net_bps"] = trades["net_pnl_inr"] / trades["notional_inr"] * 10_000.0

    trades["era"] = np.where(trades["signal_date"] <= pd.Timestamp(ERA_A[1]), "era_A",
                     np.where(trades["signal_date"] >= pd.Timestamp(ERA_B[0]), "era_B", "gap"))
    trades = trades[trades["era"].isin(["era_A", "era_B"])].copy()

    # ---- coverage caveat, quantified
    print("\n### COVERAGE CAVEAT — daily-panel ADV20 hit rate by era (the 5m-archive asymmetry)")
    cov = trades.assign(hit=trades["adv20_shares"].notna()).groupby("era")["hit"].agg(["size", "mean"])
    cov.columns = ["trades", "adv20_hit_rate"]
    print((cov.assign(adv20_hit_rate=lambda d: (d.adv20_hit_rate * 100).round(1))).to_string())
    covs = (trades.assign(hit=trades["adv20_shares"].notna())
            .groupby(["setup", "era"])["hit"].mean().unstack() * 100).round(1)
    print("\n  per-setup ADV20 hit rate (%):")
    print(covs.to_string())

    pan = panel.dropna(subset=["adv20_shares"]).copy()
    pan["q"] = pan["date"].dt.to_period("Q")
    pan["thin"] = pan["adv20_shares"] < 500_000
    qcov = pan.groupby("q").agg(symbols=("symbol", "nunique"),
                                thin_symbols=("thin", "sum"))
    qcov["thin_symbols"] = pan[pan["thin"]].groupby("q")["symbol"].nunique()
    qcov["thin_share_pct"] = (qcov["thin_symbols"] / qcov["symbols"] * 100).round(1)
    print("\n  clean_daily_from5m panel breadth by quarter (symbols with a live ADV20):")
    print(qcov.to_string())

    # ---- STEP 3: matrices
    print("\n" + "=" * 118)
    print("### STEP 3 — expectancy matrices  (exp_bps = net PnL / position notional, x1e4;"
          " cells with n<25 blanked)")
    print("=" * 118)

    cap_rows = matrix(trades, "cap_segment")
    adv_rows = matrix(trades.dropna(subset=["adv_tier"]), "adv_tier")
    tov_rows = matrix(trades.dropna(subset=["turnover_tier"]), "turnover_tier")

    for label, dim, rows in [("CAP SEGMENT", "cap_segment", cap_rows),
                             ("ADV20 TIER (shares)", "adv_tier", adv_rows),
                             ("ADV20 TIER (turnover)", "turnover_tier", tov_rows)]:
        print(f"\n--- {label} x ERA : expectancy (bps of notional) ---")
        print(pivot_exp(rows, dim).round(1).to_string())
        print(f"\n--- {label} x ERA : n ---")
        print(rows.pivot_table(index="setup", columns=[dim, "era"], values="n", observed=True)
              .fillna(0).astype(int).to_string())

    # ---- cross-setup rollup
    print("\n" + "=" * 118)
    print("### STEP 3b — CROSS-SETUP ROLLUP per segment/tier x era")
    print("  setups_pos  = # setups with exp_bps > 0 and n>=25")
    print("  clusters_pos= # DISTINCT factor clusters represented among those setups")
    print("                (A2/C1/C4/C6 + panic_crash = ONE capitulation factor — not 5 votes)")
    print("=" * 118)
    cl = trades[["setup", "cluster"]].drop_duplicates().set_index("setup")["cluster"]
    for label, dim, rows in [("CAP SEGMENT", "cap_segment", cap_rows),
                             ("ADV20 TIER (shares)", "adv_tier", adv_rows),
                             ("ADV20 TIER (turnover)", "turnover_tier", tov_rows)]:
        r = rows[rows["n"] >= 25].copy()
        r["cluster"] = r["setup"].map(cl)
        roll = r.groupby([dim, "era"], observed=True).apply(
            lambda g: pd.Series({
                "setups_n": g["setup"].nunique(),
                "setups_pos": int((g["exp_bps"] > 0).sum()),
                "clusters_pos": g.loc[g["exp_bps"] > 0, "cluster"].nunique(),
                "clusters_neg": g.loc[g["exp_bps"] <= 0, "cluster"].nunique(),
                "trades": int(g["n"].sum()),
                "pooled_exp_bps": round(float(np.average(g["exp_bps"], weights=g["n"])), 2),
                "median_setup_exp_bps": round(float(g["exp_bps"].median()), 2),
            }), include_groups=False).reset_index()
        print(f"\n--- {label} ---")
        print(roll.to_string(index=False))

    # ---- STEP 3c: pooled trade-level PF per tier (bps can be inflated by the
    # higher volatility of thin names; PF is scale-free and is the cross-check)
    print("\n" + "=" * 118)
    print("### STEP 3c — POOLED TRADE-LEVEL PF per tier x era (scale-free cross-check on exp_bps)")
    print("     'pf_ex_cap' re-pools after dropping the capitulation cluster, which is one factor")
    print("     with 5 member ledgers and would otherwise dominate the pooled number.")
    print("=" * 118)
    for label, dim in [("ADV20 TIER (shares)", "adv_tier"), ("ADV20 TIER (turnover)", "turnover_tier"),
                       ("CAP SEGMENT", "cap_segment")]:
        sub = trades.dropna(subset=[dim])
        rows = []
        for (v, era), g in sub.groupby([dim, "era"], observed=True):
            gx = g[g["cluster"] != "capitulation_long"]

            def _pf(x):
                w, l = x[x > 0].sum(), -x[x < 0].sum()
                return round(float(w / l), 3) if l > 0 else np.nan
            rows.append({dim: v, "era": era, "n": len(g), "pooled_pf": _pf(g["net_pnl_inr"]),
                         "share_of_era_net_pct": np.nan,
                         "n_ex_cap": len(gx), "pf_ex_cap": _pf(gx["net_pnl_inr"]),
                         "exp_bps_ex_cap": round(float(gx["net_bps"].mean()), 2) if len(gx) else np.nan})
        rr = pd.DataFrame(rows)
        for era in ("era_A", "era_B"):
            tot = sub.loc[sub["era"] == era, "net_pnl_inr"].sum()
            idx = rr["era"] == era
            rr.loc[idx, "share_of_era_net_pct"] = [
                round(float(sub[(sub["era"] == era) & (sub[dim] == v)]["net_pnl_inr"].sum() / tot * 100), 1)
                for v in rr.loc[idx, dim]]
        print(f"\n--- {label} ---")
        print(rr.sort_values([dim, "era"]).to_string(index=False))

    print("\n### STEP 3d — WHICH setups/clusters are net-positive in era_B, by tier (n>=25)")
    for label, dim, rows in [("ADV20 TIER (shares)", "adv_tier", adv_rows),
                             ("ADV20 TIER (turnover)", "turnover_tier", tov_rows)]:
        print(f"\n--- {label} ---")
        r = rows[(rows["era"] == "era_B") & (rows["n"] >= 25)].copy()
        r["cluster"] = r["setup"].map(cl)
        for v, g in r.groupby(dim, observed=True):
            pos = g[g["exp_bps"] > 0].sort_values("exp_bps", ascending=False)
            if not len(pos):
                print(f"  {v}: (none)")
                continue
            print(f"  {v}:  clusters={sorted(pos['cluster'].unique())}")
            for _, x in pos.iterrows():
                print(f"        +{x['exp_bps']:>7.1f} bps  pf={x['net_pf']}  n={int(x['n']):>5}  "
                      f"{x['setup']}  [{x['cluster']}]")

    # ---- STEP 4a: illiquid vs liquid, per setup, per era
    print("\n" + "=" * 118)
    print("### STEP 4(a) — DOES THE ILLIQUID TILT STILL PAY?  ILLIQUID = ADV20 < 500K shares"
          "  (the system's own '<500K' line)")
    print("=" * 118)
    t = trades.dropna(subset=["adv_tier"]).copy()
    t["liq"] = np.where(t["adv20_shares"] < 500_000, "illiquid", "liquid")
    a = t.groupby(["setup", "liq", "era"], observed=True).apply(agg, include_groups=False).reset_index()
    piv = a.pivot_table(index="setup", columns=["liq", "era"], values=["exp_bps", "n"], observed=True)
    print(piv.round(1).to_string())

    delta = a.pivot_table(index=["setup", "liq"], columns="era", values="exp_bps", observed=True)
    delta["delta_B_minus_A"] = (delta.get("era_B") - delta.get("era_A")).round(1)
    nn = a.pivot_table(index=["setup", "liq"], columns="era", values="n", observed=True)
    delta = delta.join(nn, rsuffix="_n")
    print("\n  era_B minus era_A expectancy (bps), by setup x liquidity half:")
    print(delta.round(1).to_string())

    # ---- STEP 4c: counterfactual
    print("\n" + "=" * 118)
    print("### STEP 4(c) — COUNTERFACTUAL: era_A expectancy restricted to segments that are era_B-POSITIVE")
    print("     (per setup; era_B-positive = exp_bps>0 with n>=25 in era_B on that dimension)")
    print("=" * 118)
    cf_rows = []
    for dim, rows in [("cap_segment", cap_rows), ("adv_tier", adv_rows), ("turnover_tier", tov_rows)]:
        good = rows[(rows["era"] == "era_B") & (rows["n"] >= 25) & (rows["exp_bps"] > 0)]
        keep = {(s, v) for s, v in good[["setup", dim]].itertuples(index=False)}
        sub = trades.dropna(subset=[dim]) if dim != "cap_segment" else trades
        for setup, g in sub.groupby("setup"):
            ga = g[g["era"] == "era_A"]
            if len(ga) == 0:
                continue
            mask = [(setup, v) in keep for v in ga[dim]]
            gk = ga[mask]
            cf_rows.append({
                "dim": dim, "setup": setup,
                "eraA_n": len(ga), "eraA_exp_bps": round(ga["net_bps"].mean(), 1),
                "eraA_net_inr": round(ga["net_pnl_inr"].sum()),
                "kept_n": len(gk),
                "kept_exp_bps": round(gk["net_bps"].mean(), 1) if len(gk) else np.nan,
                "kept_net_inr": round(gk["net_pnl_inr"].sum()) if len(gk) else 0,
                "pct_eraA_net_from_dead_segments": (
                    round((1 - gk["net_pnl_inr"].sum() / ga["net_pnl_inr"].sum()) * 100, 1)
                    if ga["net_pnl_inr"].sum() != 0 else np.nan),
            })
    cf = pd.DataFrame(cf_rows)
    for dim in cf["dim"].unique():
        print(f"\n--- restricted to era_B-positive {dim} ---")
        print(cf[cf["dim"] == dim].drop(columns="dim").to_string(index=False))

    # ---- STEP 4d: survivorship / coverage-cohort test
    print("\n" + "=" * 118)
    print("### STEP 4(d) — SURVIVORSHIP TEST: is the thin-band edge in ANY thin name, or only in the")
    print("     names the 5m archive has TRACKED SINCE 2023?  (lessons #16/#18 — the archive widened")
    print("     sharply in 2025Q4, so era_B's thin tier contains a cohort era_A never had.)")
    print("=" * 118)
    first_seen = panel.dropna(subset=["adv20_shares"]).groupby("symbol")["date"].min()
    trades["sym_first_seen"] = trades["symbol"].map(first_seen)
    trades["archive_cohort"] = np.where(
        trades["sym_first_seen"] <= pd.Timestamp("2023-06-30"), "tracked_since_2023", "added_later")
    npanel = pd.Series(np.where(first_seen <= pd.Timestamp("2023-06-30"),
                                "tracked_since_2023", "added_later")).value_counts()
    print(f"  panel symbols by cohort: {npanel.to_dict()}")

    def _pf(x):
        w, l = x[x > 0].sum(), -x[x < 0].sum()
        return round(float(w / l), 3) if l > 0 else np.nan

    thin = trades[trades["turnover_tier"].isin(["V1_<1cr", "V2_1-5cr"])]
    for label, sub in [("ALL setups", thin),
                       ("EX-capitulation cluster", thin[thin["cluster"] != "capitulation_long"])]:
        g = sub.groupby(["era", "archive_cohort"], observed=True).agg(
            n=("net_pnl_inr", "size"), pf=("net_pnl_inr", _pf), exp_bps=("net_bps", "mean")).round(2)
        print(f"\n  thin band (<Rs5cr/day turnover) — {label}:")
        print(g.to_string())

    print("\n### STEP 4(e) — is the surviving thin-band edge STABLE INSIDE era_B?")
    print("     B1 = 2025-01..2025-09 (pre archive-widening), B2 = 2025-10..2026-04 (post)")
    b = trades[trades["era"] == "era_B"].copy()
    b["sub"] = np.where(b["signal_date"] < pd.Timestamp("2025-10-01"), "B1_pre_widening", "B2_post")
    for label, sub in [("ALL setups", b), ("EX-capitulation cluster", b[b["cluster"] != "capitulation_long"])]:
        g = sub.dropna(subset=["turnover_tier"]).groupby(["turnover_tier", "sub"], observed=True).agg(
            n=("net_pnl_inr", "size"), pf=("net_pnl_inr", _pf), exp_bps=("net_bps", "mean")).round(2)
        print(f"\n  --- {label} ---")
        print(g.to_string())

    # ---- persist tidy output
    out = []
    for dim, rows in [("cap_segment", cap_rows), ("adv_tier", adv_rows), ("turnover_tier", tov_rows)]:
        r = rows.rename(columns={dim: "segment"}).copy()
        r.insert(0, "dimension", dim)
        r["cluster"] = r["setup"].map(cl)
        out.append(r)
    liq = a.rename(columns={"liq": "segment"}).copy()
    liq.insert(0, "dimension", "liquidity_half")
    liq["cluster"] = liq["setup"].map(cl)
    liq["low_n"] = liq["n"] < 25
    out.append(liq)
    tidy = pd.concat(out, ignore_index=True)
    cfl = cf.copy()
    cfl.insert(0, "dimension", "counterfactual")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(OUT_CSV, index=False)
    cfl.to_csv(OUT_CSV.with_name(OUT_CSV.stem + "_counterfactual.csv"), index=False)
    print(f"\nWROTE {OUT_CSV}  ({len(tidy)} rows)")
    print(f"WROTE {OUT_CSV.with_name(OUT_CSV.stem + '_counterfactual.csv')}  ({len(cfl)} rows)")

    print("\n" + "=" * 118)
    print("### STEP 5 — HONESTY CONSTRAINTS")
    print("  1. DIAGNOSTIC ONLY. Every window here (Discovery, OOS, Holdout) is already burned.")
    print("     No p-value, PF or expectancy in this file is evidence for shipping anything.")
    print("  2. Any universe re-cut suggested by these numbers is a NEW CELL SELECTION and must go")
    print("     through its own pre-registered freeze + fresh-pool one-shot (lifecycle amendment A1).")
    print("     It must NOT be adopted from this study's numbers.")
    print("  3. Fresh pool (signal_date >= 2026-05-01) is excluded by a hard filter — not burned.")
    print("  4. cap_segment is NOT point-in-time (no dated snapshot precedes the dev window) — it is")
    print("     today's classification applied to history. Read the ADV20 tiers, not cap_segment.")
    print("  5. Fees are each ledger's own; nothing re-derived. Cross-setup comparison uses bps of")
    print("     POSITION notional, which ignores that MIS/MTF leverage differs by setup.")
    print("  6. SURVIVORSHIP: the 5m archive tracked ~1230 symbols from 2023 and ~1353 more only")
    print("     later (breadth 1225 symbols in 2023Q1 -> 1505 in 2025Q3 -> 2171 in 2025Q4). era_A's")
    print("     'illiquid edge' is measured on the long-tracked cohort ONLY; see STEP 4(d) for how")
    print("     much of the thin-band edge disappears in names the archive added later.")
    print("  7. Tier assignment is confounded with setup by construction (capitulation setups almost")
    print("     never fire above Rs5cr/day; delivery_pct_anomaly fires only above Rs100cr/day). The")
    print("     ex-capitulation re-pool and the WITHIN-setup era_A-vs-era_B deltas are the honest")
    print("     reads; the pooled cross-setup column is descriptive only.")
    print("=" * 118)


if __name__ == "__main__":
    main()
