# -*- coding: utf-8 -*-
"""
Phase-2 empirical signature: bulk_deal_directional_entity_drift
===============================================================
Lifecycle Stage 2 (docs/setup_lifecycle.md) for the candidate described in
specs/2026-07-28-brief-bulk_deal_directional_entity_drift.md.

Measures the mechanism's RAW footprint (no fees, no leverage, no exits):
do bulk/block deals by DIRECTIONAL entities (non-churn footprint) show a
T+1 -> T+5 drift in the deal's direction that churn-entity deals do not?

PRE-REGISTERED (binding, from brief sections 4/5/6):
  - Entity classification is CAUSAL: per entity per event date, trailing
    both-sided share of symbol-days computed ONLY from deals strictly before
    the event date, min 10 prior symbol-day appearances.
      churn        : both-sided share > 0.80  -> CONTROL cohort
      directional  : both-sided share < 0.30  -> SIGNAL cohort
      ambiguous    : 0.30-0.80, or <10 priors -> EXCLUDED (no salvage-mining)
  - Event = (entity, symbol, date, source); same-day BUY+SELL by one entity is
    NETTED; if |net qty| < 20% of gross qty the event is DROPPED (pre-registered
    choice: drop, not net-side). Rule applied uniformly to BOTH cohorts.
  - Entry T+1 open; drift = signed %% return in deal direction to close of
    T+1 / T+3 / T+5 (holds 1/3/5). RAW, no fees.
  - Baseline = same-universe (event symbols) unconditional same-horizon return,
    per era, side-adjusted for SELL cells.
  - Grid: cohort x side {BUY,SELL} x hold {1,3,5} x entity class
    {institution, individual} x deal-size-vs-ADV20 tercile x era {A,B}
    x source {bulk, block}. Dead cells included in the CSV.
  - Eras (A5): era_A = 2023-01 .. 2024-12, era_B = 2025-01 .. 2026-04.
    era_A carries the coverage caveat (brief 6.3): clean_daily era_A coverage
    of bulk symbols is expected <45%; era_A demoted to sign-only evidence.
  - Kill threshold (Stage-2 table): signal-vs-baseline delta >= 0.1% drift.

ENTITY-CLASS NAME RULE (documented per brief 5): normalized entity name
containing any INSTITUTION_TOKENS member as a whole token -> institution;
otherwise -> individual. Tokens are fund/capital/securities/asset/mgmt/
FPI-style per the brief; legal suffixes (LTD/PVT/LLP/...) are stripped by
normalization BEFORE this test.

Window: deal dates 2023-01-01 .. 2026-04-30 (both parquets end there).
Prices: cache/preaggregate/clean_daily_from5m.feather (A3 hardened set).

Output: reports/sub9_sanity/_bulk_entity_drift_phase2.csv  (full grid,
per-era, control cohort included, dead cells included) + console readouts
for the three falsifiers.

Run:  .venv/Scripts/python tools/sub9_research/phase2_bulk_entity_drift_signature.py
"""
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ----------------------------------------------------------------------------
# Locked constants (pre-registered)
# ----------------------------------------------------------------------------
BULK_PARQUET = ROOT / "data/bulk_deals_cache/nse_bulk_deals_2023_2026.parquet"
BLOCK_PARQUET = ROOT / "data/block_deals/block_deals_events.parquet"
DAILY_FEATHER = ROOT / "cache/preaggregate/clean_daily_from5m.feather"
OUT_CSV = ROOT / "reports/sub9_sanity/_bulk_entity_drift_phase2.csv"

WINDOW_START = pd.Timestamp("2023-01-01")
WINDOW_END = pd.Timestamp("2026-04-30")
ERA_SPLIT = pd.Timestamp("2025-01-01")   # era_A < split <= era_B

CHURN_MIN_SHARE = 0.80        # both-sided share > 0.80 -> churn (control)
DIRECTIONAL_MAX_SHARE = 0.30  # both-sided share < 0.30 -> directional (signal)
MIN_PRIOR_SYMDAYS = 10        # causal history floor
NET_DROP_FRAC = 0.20          # drop event if |net| < 20% of gross
HOLDS = (1, 3, 5)
ADV_WINDOW = 20               # sessions, strictly before deal date
ADV_MIN_PERIODS = 10
DELTA_FLOOR_PCT = 0.1         # Stage-2 cheap-kill threshold (percent)
MIN_CELL_N = 30               # n floor for the "meaningful cell" floor count

# LOCKED alias table (brief section 4: "small manual alias table").
# Derived from the 2026-07-28 recon probes (probe2/probe5/probe6): reordered
# individual names, mid-name spelling variants the suffix-stripper cannot fix,
# ODI/sub-account routing variants of one decision-maker, and the MANSI
# sibling handoff (SHARE AND STOCK ADVISORS / BROKING / SHARES STOCK ADVISORS
# are one operator per the recon). Applied AFTER normalize().
ALIAS_TABLE = {
    # reordered individual names
    "RIMPY MITTAL": "MITTAL RIMPY",
    "KUMAR VINOD": "VINOD KUMAR",
    "POOJA JAIN": "JAIN POOJA",
    "HARSHAWARDHAN HANMANT SABALE": "SABALE HARSHAWARDHAN HANMANT",
    # MANSI sibling handoff + spelling variants
    "MANSI SHARES STOCK ADVISORS": "MANSI SHARE AND STOCK ADVISORS",
    "MANSI SHARE STOCK ADVISORS": "MANSI SHARE AND STOCK ADVISORS",
    "MANSI SHARE AND STOCK BROKING": "MANSI SHARE AND STOCK ADVISORS",
    # typo'd legal suffix the stripper misses
    "ARIHANT CAPITAL MARKETS LIMTED": "ARIHANT CAPITAL MARKETS",
    # ODI / routing variants of one decision-maker
    "GOLDMAN SACHS SINGAPORE PTE ODI": "GOLDMAN SACHS SINGAPORE PTE",
    "GOLDMAN SACHS BANK EUROPE SE ODI": "GOLDMAN SACHS BANK EUROPE SE",
    "GOLDMAN SACHS BANK EUROPESE ODI": "GOLDMAN SACHS BANK EUROPE SE",
    "MORGAN STANLEY ASIA SINGAPORE PTE ODI": "MORGAN STANLEY ASIA SINGAPORE PTE",
    "SOCIETE GENERALE ODI": "SOCIETE GENERALE",
    # spelling/spacing variants
    "SMALL CAP WORLD FUND": "SMALLCAP WORLD FUND",
    "MARSHALL WACE INVESTMENT STRATEGIESGTOPS FUND": "MARSHALL WACE INVESTMENT STRATEGIES TOPS FUND",
    "KAUSHIK MAHESHBHAI WAGHELA": "KAUSHIK MAHESH WAGHELA",
    "SHREE NAMAN SECURITIES FINANCE": "NAMAN SECURITIES FINANCE",
    # same-human tax vehicle
    "AMIT KUMAR JAIN HUF": "AMIT KUMAR JAIN",
    # sub-account variants of one pension manager (GPIF via MTBJ trust)
    "THE MTBJ LTD AS TRST FOR GOVERNMENT PENSION INVESTMENT FUND MTBJ400045849":
        "GPIF MTBJ PENSION FUND",
    "THE MTBJ LTD AS TRST FOR GOVRNMNT PENSION INVSTMNT FUND MTBJ400045828":
        "GPIF MTBJ PENSION FUND",
    # AIA Singapore participating-fund share classes (one manager; includes a
    # PARTCIPATING typo variant)
    "AIA INVESTMENT TRUST AIA SINGAPORE PARTICIPATING FUND SGD":
        "AIA SINGAPORE PARTICIPATING FUND",
    "AIA INVESTMENT TRUST AIA SINGAPORE PARTICIPATING FUND USD":
        "AIA SINGAPORE PARTICIPATING FUND",
    "AIA INVESTMENT TRUST AIA SINGAPORE PARTICIPATING FUND HERITAGE USD":
        "AIA SINGAPORE PARTICIPATING FUND",
    "AIA INVESTMENT TRUST AIA SINGAPORE PARTCIPATING FUND HERITAGE SGD":
        "AIA SINGAPORE PARTICIPATING FUND",
}

# Entity-class rule (brief section 5). Whole-token match on normalized name.
INSTITUTION_TOKENS = {
    "FUND", "FUNDS", "CAPITAL", "SECURITIES", "ASSET", "ASSETS",
    "MANAGEMENT", "MGMT", "INVESTMENT", "INVESTMENTS", "INVEST",
    "PORTFOLIO", "PARTNERS", "ADVISORS", "ADVISOR", "ADVISORY",
    "BROKING", "BROKERS", "FINANCE", "FINANCIAL", "FINVEST",
    "MUTUAL", "INSURANCE", "BANK", "PENSION", "TRUST",
    "AIF", "FPI", "PMS", "ODI", "PTE", "LLC",
    "MARKETS", "RESEARCH", "TRADING", "FINTECH", "EMERGING",
    "VENTURES", "HOLDINGS", "ENTERPRISES", "ENTERPRISE",
    "INSTITUTIONAL", "CONSULTANCY", "COMMODITIES",
    # corporate-style tokens no Indian individual name carries (added after
    # eyeball check flagged SOCIETE GENERALE / BNP PARIBAS ARBITRAGE / CINCO
    # STOCK VISION / VINEY EQUITY MARKET / TRADE CORNER as "individual")
    "ARBITRAGE", "GENERALE", "STOCK", "STOCKS", "EQUITY", "EQUITIES",
    "SHARES", "TRADE", "SHARE",
}

STRIP_SUFFIXES = ("LIMITED", "LTD", "PRIVATE", "PVT", "LLP", "LLC", "PLC",
                  "INC", "CO", "COMPANY", "CORP", "CORPORATION")


def normalize(name: str) -> str:
    """Recon-validated cheap normalization: uppercase, strip punctuation,
    collapse whitespace, strip trailing legal suffixes, then alias table."""
    s = re.sub(r"[^A-Z0-9 ]", " ", str(name).upper())
    s = re.sub(r"\s+", " ", s).strip()
    toks = s.split(" ")
    while toks and toks[-1] in STRIP_SUFFIXES:
        toks = toks[:-1]
    s = " ".join(toks)
    return ALIAS_TABLE.get(s, s)


def classify_entity_name(entity: str) -> str:
    toks = set(entity.split(" "))
    return "institution" if toks & INSTITUTION_TOKENS else "individual"


def main() -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)

    # ------------------------------------------------------------------
    # 1. Load deal feeds (parsing shim for bulk: Indian commas, %d-%b-%Y)
    # ------------------------------------------------------------------
    bulk = pd.read_parquet(BULK_PARQUET)
    block = pd.read_parquet(BLOCK_PARQUET)
    print(f"[load] bulk parquet shape={bulk.shape}  block parquet shape={block.shape}")

    b = pd.DataFrame({
        "date": pd.to_datetime(bulk["Date"], format="%d-%b-%Y"),
        "symbol": bulk["Symbol"].astype(str).str.strip().str.upper(),
        "client": bulk["ClientName"],
        "side": bulk["Buy/Sell"].astype(str).str.strip().str.upper(),
        "qty": pd.to_numeric(bulk["QuantityTraded"].astype(str).str.replace(",", ""),
                             errors="coerce"),
        "source": "bulk",
    })
    k = pd.DataFrame({
        "date": pd.to_datetime(block["trade_date"]),
        "symbol": block["raw_symbol"].astype(str).str.strip().str.upper(),
        "client": block["client_name"],
        "side": block["buy_or_sell"].astype(str).str.strip().str.upper(),
        "qty": pd.to_numeric(block["qty"], errors="coerce"),
        "source": "block",
    })
    print(f"[parse] bulk qty parse failures: {int(b['qty'].isna().sum())}")
    print(f"[parse] bulk side values: {b['side'].value_counts().to_dict()}")
    print(f"[parse] block side values: {k['side'].value_counts().to_dict()}")
    print(f"[parse] block exchange mix: {block['exchange'].value_counts().to_dict()}")

    deals = pd.concat([b, k], ignore_index=True)
    n0 = len(deals)
    deals = deals.drop_duplicates()
    print(f"[dedup] exact duplicate rows dropped: {n0 - len(deals)}")
    deals = deals[deals["side"].isin(["BUY", "SELL"]) & deals["qty"].notna()]
    deals = deals[(deals["date"] >= WINDOW_START) & (deals["date"] <= WINDOW_END)]
    print(f"[window] union rows in {WINDOW_START.date()}..{WINDOW_END.date()}: "
          f"{len(deals)}  date range {deals['date'].min().date()}..{deals['date'].max().date()}")
    print(f"[window] per-source rows: {deals['source'].value_counts().to_dict()}")

    deals["entity"] = deals["client"].map(normalize)
    deals["era"] = np.where(deals["date"] < ERA_SPLIT, "A", "B")
    print(f"[entities] distinct normalized+aliased entities: {deals['entity'].nunique()}")
    print(f"[eras] rows per era: {deals['era'].value_counts().to_dict()}")

    # ------------------------------------------------------------------
    # 2. Causal entity classification (trailing both-sided symbol-day share)
    #    Uses the FULL union feed (both sources) as the entity's footprint.
    # ------------------------------------------------------------------
    sd = (deals.groupby(["entity", "symbol", "date"])["side"]
          .agg(lambda s: 1 if {"BUY", "SELL"} <= set(s) else 0)
          .rename("both").reset_index())
    ed = (sd.groupby(["entity", "date"])
          .agg(cnt=("both", "size"), both_cnt=("both", "sum"))
          .reset_index()
          .sort_values(["entity", "date"]))
    g = ed.groupby("entity")
    ed["prior_n"] = g["cnt"].cumsum() - ed["cnt"]
    ed["prior_both"] = g["both_cnt"].cumsum() - ed["both_cnt"]
    ed["prior_share"] = np.where(ed["prior_n"] > 0, ed["prior_both"] / ed["prior_n"], np.nan)
    conds = [
        ed["prior_n"] < MIN_PRIOR_SYMDAYS,
        ed["prior_share"] > CHURN_MIN_SHARE,
        ed["prior_share"] < DIRECTIONAL_MAX_SHARE,
    ]
    ed["cohort"] = np.select(conds, ["excluded_insufficient", "churn", "directional"],
                             default="excluded_ambiguous")
    print("\n[classify] entity-date cohort funnel (entity-date units):")
    print(ed["cohort"].value_counts().to_string())

    # ------------------------------------------------------------------
    # 3. Events: net same-day same-source buy+sell per (entity, symbol, date)
    # ------------------------------------------------------------------
    piv = (deals.pivot_table(index=["entity", "symbol", "date", "source"],
                             columns="side", values="qty", aggfunc="sum")
           .reindex(columns=["BUY", "SELL"]).fillna(0.0).reset_index())
    piv.columns.name = None
    piv["gross"] = piv["BUY"] + piv["SELL"]
    piv["net"] = piv["BUY"] - piv["SELL"]
    both_mask = (piv["BUY"] > 0) & (piv["SELL"] > 0)
    drop_mask = both_mask & (piv["net"].abs() < NET_DROP_FRAC * piv["gross"])
    piv["evt_side"] = np.where(piv["net"] > 0, "BUY", "SELL")
    ev = piv[~drop_mask].copy()
    print(f"\n[events] (entity,symbol,date,source) units: {len(piv)}; "
          f"both-sided that day: {int(both_mask.sum())}; "
          f"dropped |net|<{NET_DROP_FRAC:.0%} gross: {int(drop_mask.sum())}; kept: {len(ev)}")

    ev = ev.merge(ed[["entity", "date", "cohort", "prior_n", "prior_share"]],
                  on=["entity", "date"], how="left")
    drop_stats = ev["cohort"].value_counts()
    print(f"[events] cohort mix of kept events:\n{drop_stats.to_string()}")
    ev = ev[ev["cohort"].isin(["directional", "churn"])].copy()
    ev["era"] = np.where(ev["date"] < ERA_SPLIT, "A", "B")
    ev["entity_class"] = ev["entity"].map(classify_entity_name)
    print(f"[events] scored-cohort events: {len(ev)}  "
          f"(directional={int((ev['cohort']=='directional').sum())}, "
          f"churn={int((ev['cohort']=='churn').sum())})")
    print("[events] entity_class mix (directional cohort): "
          f"{ev.loc[ev['cohort']=='directional','entity_class'].value_counts().to_dict()}")
    for coh in ("directional", "churn"):
        for cls in ("institution", "individual"):
            top = (ev[(ev["cohort"] == coh) & (ev["entity_class"] == cls)]["entity"]
                   .value_counts().head(8))
            print(f"  top {coh}/{cls}: {list(top.index)}")

    # ------------------------------------------------------------------
    # 4. Prices: clean_daily pivots, entry T+1 open, exits close T+1/3/5, ADV20
    # ------------------------------------------------------------------
    daily = pd.read_feather(DAILY_FEATHER)
    daily["date"] = pd.to_datetime(daily["date"])
    daily["symbol"] = daily["symbol"].astype(str).str.upper().str.replace(".NS", "", regex=False)
    print(f"\n[prices] clean_daily shape={daily.shape}  "
          f"range {daily['date'].min().date()}..{daily['date'].max().date()}  "
          f"symbols={daily['symbol'].nunique()}")

    evsyms = set(ev["symbol"].unique())
    daily = daily[daily["symbol"].isin(evsyms)]
    open_p = daily.pivot_table(index="date", columns="symbol", values="open").sort_index()
    close_p = daily.pivot_table(index="date", columns="symbol", values="close").sort_index()
    vol_p = daily.pivot_table(index="date", columns="symbol", values="volume").sort_index()
    sessions = open_p.index.to_numpy()
    sym_cols = {s: i for i, s in enumerate(open_p.columns)}
    print(f"[prices] pivot: {len(sessions)} sessions x {len(sym_cols)} event-universe symbols "
          f"(event symbols with any price data: {len(sym_cols)}/{len(evsyms)})")

    adv = vol_p.rolling(ADV_WINDOW, min_periods=ADV_MIN_PERIODS).mean().shift(1)

    ev["sym_idx"] = ev["symbol"].map(sym_cols)
    ev["entry_idx"] = np.searchsorted(sessions, ev["date"].to_numpy(), side="right")

    ov = open_p.to_numpy()
    cv = close_p.to_numpy()
    av = adv.to_numpy()
    n_sess = len(sessions)

    def take(mat, ridx, cidx):
        out = np.full(len(ridx), np.nan)
        ok = (~np.isnan(cidx)) & (ridx >= 0) & (ridx < n_sess)
        r = ridx[ok].astype(int)
        c = cidx[ok].astype(int)
        out[ok] = mat[r, c]
        return out

    ridx = ev["entry_idx"].to_numpy()
    cidx = ev["sym_idx"].to_numpy(dtype=float)
    ev["entry_open"] = take(ov, ridx, cidx)
    # ADV measured at the deal-date row (rolling ... shift(1) => strictly
    # before the deal date; deal date row = entry_idx - 1 when T is a session).
    ev["adv20"] = take(av, ridx - 1, cidx)
    for h in HOLDS:
        ev[f"exit_close_{h}"] = take(cv, ridx + h - 1, cidx)

    sign = np.where(ev["evt_side"] == "BUY", 1.0, -1.0)
    valid_entry = ev["entry_open"].notna() & (ev["entry_open"] > 0)
    for h in HOLDS:
        raw = (ev[f"exit_close_{h}"] / ev["entry_open"] - 1.0) * 100.0
        ev[f"raw_ret_{h}"] = np.where(valid_entry, raw, np.nan)
        ev[f"drift_{h}"] = sign * ev[f"raw_ret_{h}"]

    # Coverage falsifier (brief 6.3) — report BOTH eras, era_A is the caveat.
    print("\n[coverage] entry-price availability (T+1 open on clean_daily):")
    for era in ("A", "B"):
        m = ev["era"] == era
        evrate = float(valid_entry[m].mean()) * 100 if m.any() else float("nan")
        syms = ev.loc[m, "symbol"]
        symrate = float(syms.isin(sym_cols.keys()).groupby(syms).first().mean()) * 100
        dm = m & (ev["cohort"] == "directional")
        drate = float(valid_entry[dm].mean()) * 100 if dm.any() else float("nan")
        print(f"  era_{era}: event match {evrate:5.1f}%  (directional cohort {drate:5.1f}%)  "
              f"unique-symbol match {symrate:5.1f}%  events={int(m.sum())}")

    # ------------------------------------------------------------------
    # 5. Deal-size-vs-ADV terciles (within era, pooled cohorts+sources)
    # ------------------------------------------------------------------
    ev["size_ratio"] = np.where(ev["adv20"] > 0, ev["gross"] / ev["adv20"], np.nan)
    ev["size_tercile"] = "na"
    for era in ("A", "B"):
        m = (ev["era"] == era) & ev["size_ratio"].notna() & valid_entry
        if m.sum() >= 9:
            ev.loc[m, "size_tercile"] = pd.qcut(
                ev.loc[m, "size_ratio"].rank(method="first"), 3,
                labels=["t1_small", "t2_mid", "t3_large"]).astype(str)
        q = ev.loc[m, "size_ratio"].quantile([1 / 3, 2 / 3]).round(3).to_dict()
        print(f"[size] era_{era} deal-size/ADV20 tercile breaks: {q}  (n={int(m.sum())})")

    # ------------------------------------------------------------------
    # 6. Baseline: same-universe unconditional same-horizon return, per era
    # ------------------------------------------------------------------
    baselines = {}
    entry_next = open_p.shift(-1)
    era_of_session = pd.Series(np.where(open_p.index < ERA_SPLIT, "A", "B"),
                               index=open_p.index)
    in_window = (open_p.index >= WINDOW_START) & (open_p.index <= WINDOW_END)
    for h in HOLDS:
        rr = (close_p.shift(-h) / entry_next - 1.0) * 100.0
        for era in ("A", "B"):
            m = (era_of_session == era).to_numpy() & in_window
            vals = rr.loc[m].to_numpy().ravel()
            vals = vals[np.isfinite(vals)]
            baselines[(era, h)] = float(np.mean(vals))
    print("\n[baseline] unconditional same-universe long return (%, entry next open -> close T+h):")
    for era in ("A", "B"):
        print("  era_" + era + "  " +
              "  ".join(f"h={h}: {baselines[(era, h)]:+.3f}%" for h in HOLDS))

    # ------------------------------------------------------------------
    # 7. Full grid CSV (dead cells included)
    # ------------------------------------------------------------------
    scored = ev[valid_entry].copy()
    print(f"\n[scored] events with valid entry: {len(scored)}  shape={scored.shape}")
    conc = scored["symbol"].value_counts()
    print(f"[scored] symbol concentration: top-5 {conc.head(5).to_dict()}  "
          f"unique symbols={len(conc)}")

    dims = {
        "cohort": ["directional", "churn"],
        "era": ["A", "B"],
        "source": ["bulk", "block"],
        "side": ["BUY", "SELL"],
        "hold": list(HOLDS),
        "entity_class": ["institution", "individual"],
        "size_tercile": ["t1_small", "t2_mid", "t3_large", "na"],
    }
    grid = pd.MultiIndex.from_product(dims.values(), names=dims.keys()).to_frame(index=False)

    rows = []
    for h in HOLDS:
        d = scored[["cohort", "era", "source", "evt_side", "entity_class",
                    "size_tercile", f"drift_{h}"]].rename(
            columns={"evt_side": "side", f"drift_{h}": "drift"})
        d = d[d["drift"].notna()]
        d["hold"] = h
        rows.append(d)
    long = pd.concat(rows, ignore_index=True)

    agg = (long.groupby(["cohort", "era", "source", "side", "hold",
                         "entity_class", "size_tercile"])["drift"]
           .agg(n="size", mean_drift_pct="mean", median_drift_pct="median",
                std_pct="std", hit_rate=lambda s: float((s > 0).mean()))
           .reset_index())
    cells = grid.merge(agg, how="left",
                       on=["cohort", "era", "source", "side", "hold",
                           "entity_class", "size_tercile"])
    cells["n"] = cells["n"].fillna(0).astype(int)
    cells["baseline_pct"] = [
        baselines[(e, h)] * (1.0 if s == "BUY" else -1.0)
        for e, h, s in zip(cells["era"], cells["hold"], cells["side"])]
    cells["delta_pct"] = cells["mean_drift_pct"] - cells["baseline_pct"]
    for c in ("mean_drift_pct", "median_drift_pct", "std_pct", "hit_rate",
              "baseline_pct", "delta_pct"):
        cells[c] = cells[c].round(4)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cells.to_csv(OUT_CSV, index=False)
    print(f"[csv] wrote {len(cells)} cells ({int((cells['n'] > 0).sum())} live, "
          f"{int((cells['n'] == 0).sum())} dead) -> {OUT_CSV}")

    # ------------------------------------------------------------------
    # 8. Falsifier readouts
    # ------------------------------------------------------------------
    def cohort_table(df, by):
        t = (df.groupby(by)["drift"]
             .agg(n="size", mean="mean", hit=lambda s: float((s > 0).mean()))
             .reset_index())
        t["mean"] = t["mean"].round(3)
        t["hit"] = (t["hit"] * 100).round(1)
        return t

    print("\n" + "=" * 78)
    print("FALSIFIER 1 — directional vs churn drift separation (LOAD-BEARING)")
    print("=" * 78)
    sep = cohort_table(long, ["era", "hold", "cohort"])
    piv1 = sep.pivot_table(index=["era", "hold"], columns="cohort",
                           values=["n", "mean", "hit"], aggfunc="first")
    piv1[("separation", "dir-churn")] = (piv1[("mean", "directional")]
                                         - piv1[("mean", "churn")]).round(3)
    print(piv1.to_string())

    print("\nPer-side per-era drift (signal cohort = directional), baseline-adjusted:")
    dsig = long[long["cohort"] == "directional"]
    t2 = cohort_table(dsig, ["era", "side", "hold"])
    t2["baseline"] = [round(baselines[(e, h)] * (1 if s == "BUY" else -1), 3)
                      for e, s, h in zip(t2["era"], t2["side"], t2["hold"])]
    t2["delta"] = (t2["mean"] - t2["baseline"]).round(3)
    print(t2.to_string(index=False))
    print("\nSame table, control cohort (churn):")
    dchn = long[long["cohort"] == "churn"]
    t2c = cohort_table(dchn, ["era", "side", "hold"])
    t2c["baseline"] = [round(baselines[(e, h)] * (1 if s == "BUY" else -1), 3)
                       for e, s, h in zip(t2c["era"], t2c["side"], t2c["hold"])]
    t2c["delta"] = (t2c["mean"] - t2c["baseline"]).round(3)
    print(t2c.to_string(index=False))

    print("\n" + "=" * 78)
    print("FALSIFIER 2 — salience: size-vs-ADV terciles + institution vs individual")
    print("=" * 78)
    t3 = cohort_table(dsig[dsig["size_tercile"] != "na"],
                      ["era", "hold", "size_tercile"])
    print(t3.pivot_table(index=["era", "hold"], columns="size_tercile",
                         values=["n", "mean"], aggfunc="first").to_string())
    t4 = cohort_table(dsig, ["era", "hold", "entity_class"])
    print("\n" + t4.pivot_table(index=["era", "hold"], columns="entity_class",
                                values=["n", "mean"], aggfunc="first").to_string())

    print("\n" + "=" * 78)
    print("FALSIFIER 3 — era_A coverage (above) + era-sign agreement, signal cohort")
    print("=" * 78)
    m = (t2.pivot_table(index=["side", "hold"], columns="era",
                        values=["mean", "delta", "n"], aggfunc="first"))
    m[("agree", "mean_sign")] = (np.sign(m[("mean", "A")]) == np.sign(m[("mean", "B")]))
    m[("agree", "delta_sign")] = (np.sign(m[("delta", "A")]) == np.sign(m[("delta", "B")]))
    print(m.to_string())
    n_agree = int(m[("agree", "delta_sign")].sum())
    print(f"\nera_A(covered) vs era_B delta-sign agreement: {n_agree}/{len(m)} side x hold marginals")

    print("\n" + "=" * 78)
    print(f"FLOOR COUNT — signal-cohort cells with delta >= {DELTA_FLOOR_PCT}% (Stage-2 kill line)")
    print("=" * 78)
    sig_cells = cells[(cells["cohort"] == "directional") & (cells["n"] > 0)]
    for era in ("A", "B"):
        ec = sig_cells[sig_cells["era"] == era]
        n_all = int((ec["delta_pct"] >= DELTA_FLOOR_PCT).sum())
        n_meaning = int(((ec["delta_pct"] >= DELTA_FLOOR_PCT) & (ec["n"] >= MIN_CELL_N)).sum())
        tot_meaning = int((ec["n"] >= MIN_CELL_N).sum())
        print(f"  era_{era}: {n_all}/{len(ec)} live cells >= floor; "
              f"{n_meaning}/{tot_meaning} among n>={MIN_CELL_N} cells")
    print("\ndone.")


if __name__ == "__main__":
    sys.exit(main())
