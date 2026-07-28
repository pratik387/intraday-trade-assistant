"""Phase 2 empirical signature: `pead_reaction_drift` — post-earnings reaction-day-surprise drift.

Brief: specs/2026-07-27-brief-pead_reaction_drift.md (pre-registered grid, Section 5 — BINDING).
Lifecycle: docs/setup_lifecycle.md Stage 2. Disciplined redo of the B1 weak kill (lessons #28).

WINDOW DISCIPLINE (hard constraint)
    Discovery ONLY: announcement dates 2023-01-01 -> 2024-12-31. No statistic is computed
    on 2025+ events. Forward-return exit legs of Dec-2024 events may extend into early 2025
    (exit legs only). Development-pool exploration -> exempt from experiment-ledger logging.

WHAT IS MEASURED (Phase 2 = raw mechanism footprint)
    Mean signed forward drift (in %) of each signal cohort vs a matched baseline.
    NO fees, NO leverage, NO exits, NO PF. Baseline = SAME-UNIVERSE UNCONDITIONAL
    SAME-HORIZON return: for horizon h, the mean of (close[d+h-1]/open[d] - 1) over ALL
    symbol-days d in the discovery window that pass the same ADV20 floor (Rs 20L),
    i.e. entry at any eligible day's open, exit at close h trading sessions later.
    delta_pct is signed in TRADE direction: long = sig - base; short = base - sig.

CAUSALITY (lesson #25)
    - ADV20 / AvgVol20 exclude the current day (rolling(20).mean().shift(1)).
    - Reaction-day assignment from announce_time_class (same rule as _tmp_b1_pead_killtest):
        intraday / BMO                    -> reaction day = announce day
        scheduled with hour < 15 (or NaT) -> announce day
        AMC / scheduled hour >= 15        -> next trading day
    - For the 2-day reaction window the signal is only known at the CLOSE of the second
      reaction day, so entry lags are anchored to the reaction-window END:
      entry = open of (reaction_end + lag), lag in {1, 2}. For the 1-day window this is
      literally the brief's "reaction+1 open / reaction+2 open".
    - Hold h = h trading sessions in-market: exit at close of (entry_pos + h - 1).
    - Decile / 5% thresholds are pooled over the discovery cohort per (proxy, window)
      — a footprint-measurement simplification, acceptable at Phase 2 (no trading rule
      is being validated); the fixed |reaction| >= 3% threshold has no such pooling.

MARKET ADJUSTMENT
    Nifty 50 daily closes from backtest-cache-download/index_ohlcv/NSE_NIFTY_50
    (coverage starts 2023-01-19; first usable 1d return 2023-01-20). Reaction dates
    before Nifty coverage fall back to the UNIVERSE-MEAN same-window return
    (equal-weight mean over ADV-eligible symbols) — the fallback count is printed.

OUTPUTS
    reports/sub9_sanity/_pead_reaction_drift_phase2_discovery.csv   (all 288 cells, dead ones too)
    reports/sub9_sanity/_pead_reaction_drift_phase2_adv_tiers.csv   (5-tier split, top cells by |delta|)
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(r"E:\Codebase\intraday-trade-assistant")

# ---------------------------------------------------------------------------
# LOCKED_GRID — brief Section 5, binding. Do not extend without a new brief rev.
# ---------------------------------------------------------------------------
LOCKED_GRID = {
    "surprise_proxy": ["raw", "mktadj", "volconf"],   # raw reaction ret; minus Nifty; raw + reaction vol >= 2x AvgVol20
    "reaction_window": [1, 2],                         # trading days (announce reaction day [+ next])
    "leg": ["long", "short"],                          # long on positive surprise; short on negative
    "threshold": ["decile", "pct5", "abs3"],           # top/bottom 10%; top/bottom 5%; |reaction| >= 3%
    "entry_lag": [1, 2],                               # open of reaction_END + lag  (causal anchor, see docstring)
    "hold": [3, 5, 7, 10],                             # trading sessions in-market, exit close
}
ABS3_THRESHOLD = 0.03
VOLCONF_MULT = 2.0            # reaction-window mean volume >= 2x AvgVol20 (share volume)
ADV_FLOOR = 2_000_000         # Rs 20L turnover ADV20 floor, as in B1
N_TIERS = 5                   # ADV tiers, tier1 = most illiquid, as in B1
N_TOP_CELLS_FOR_TIERS = 15    # per-tier split reported for top cells by |delta|

DISCOVERY_START = pd.Timestamp("2023-01-01")
DISCOVERY_END = pd.Timestamp("2024-12-31")

DAILY_FEATHER = REPO_ROOT / "cache" / "preaggregate" / "clean_daily_from5m.feather"
EVENTS_PARQUET = REPO_ROOT / "data" / "earnings_calendar" / "earnings_events.parquet"
NIFTY_FEATHER = (REPO_ROOT / "backtest-cache-download" / "index_ohlcv" / "NSE_NIFTY_50"
                 / "NSE_NIFTY_50_1days.feather")
OUT_CELLS_CSV = REPO_ROOT / "reports" / "sub9_sanity" / "_pead_reaction_drift_phase2_discovery.csv"
OUT_TIERS_CSV = REPO_ROOT / "reports" / "sub9_sanity" / "_pead_reaction_drift_phase2_adv_tiers.csv"

# Offsets needed from the REACTION day position:
#   events:   entry_off = (window - 1) + lag in {1, 2, 3}; exit_off = entry_off + hold - 1 in {3 .. 12}
#   baseline: exit_off = hold - 1 in {2, 4, 6, 9} (entry day = offset 0)
ENTRY_OFFS = [1, 2, 3]
EXIT_OFFS = list(range(2, 13))
REACTION_TOLERANCE_DAYS = 5   # next-day reaction must trade within 5 calendar days of announce


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data loading (with mandatory shape/coverage pre-check prints)
# ---------------------------------------------------------------------------

def load_daily() -> pd.DataFrame:
    dd = pd.read_feather(DAILY_FEATHER)
    dd["date"] = pd.to_datetime(dd["date"]).dt.normalize()
    dd["symbol"] = dd["symbol"].astype(str).str.replace("NSE:", "", regex=False).str.upper()
    dd = dd.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    log(f"daily bars: shape={dd.shape}  dates {dd['date'].min().date()} -> {dd['date'].max().date()}"
        f"  symbols={dd['symbol'].nunique()}")

    grp = dd.groupby("symbol", sort=False)
    dd["turnover"] = dd["close"] * dd["volume"]
    # ADV20 / AvgVol20 EXCLUDE the current day (lesson #25).
    dd["adv20"] = grp["turnover"].transform(lambda s: s.rolling(20).mean().shift(1))
    dd["avgvol20"] = grp["volume"].transform(lambda s: s.rolling(20).mean().shift(1))

    # Vectorized within-symbol shifts: rows are symbol-contiguous, so a plain shift is
    # valid wherever the shifted row's symbol matches — mask the boundary crossings.
    sym = dd["symbol"].to_numpy()

    def shifted(col: str, k: int) -> np.ndarray:
        v = dd[col].to_numpy(dtype=float)
        out = np.full(len(v), np.nan)
        if k > 0:      # value from k rows AHEAD (future within symbol)
            out[:-k] = v[k:]
            ok = sym[:-k] == sym[k:]
            out[:-k][~ok] = np.nan
        elif k < 0:    # value from |k| rows BEFORE
            kk = -k
            out[kk:] = v[:-kk]
            ok = sym[kk:] == sym[:-kk]
            out[kk:][~ok] = np.nan
        else:
            out = v
        return out

    dd["prev_close"] = shifted("close", -1)
    dd["next_close"] = shifted("close", 1)     # for 2d reaction window
    dd["next_volume"] = shifted("volume", 1)
    dd["next_date_ok"] = ~np.isnan(dd["next_close"].to_numpy())
    for k in ENTRY_OFFS:
        dd[f"open_p{k}"] = shifted("open", k)
    for k in EXIT_OFFS:
        dd[f"close_p{k}"] = shifted("close", k)
    return dd


def load_events() -> pd.DataFrame:
    ev = pd.read_parquet(EVENTS_PARQUET)
    log(f"earnings events: shape={ev.shape}  announce {pd.to_datetime(ev['announce_date']).min().date()}"
        f" -> {pd.to_datetime(ev['announce_date']).max().date()}")
    ev = ev[ev["result_type"] == "quarterly"].copy()
    ev["symbol"] = ev["symbol"].astype(str).str.replace("NSE:", "", regex=False).str.upper()
    ev["announce_date"] = pd.to_datetime(ev["announce_date"]).dt.normalize()
    ev = ev[(ev["announce_date"] >= DISCOVERY_START) & (ev["announce_date"] <= DISCOVERY_END)]
    log(f"  quarterly + discovery window ({DISCOVERY_START.date()} -> {DISCOVERY_END.date()}): n={len(ev)}")
    hour = pd.to_datetime(ev["announce_time"], errors="coerce").dt.hour
    cls = ev["announce_time_class"]
    # Reaction-day rule (identical to _tmp_b1_pead_killtest.py):
    ev["same_day"] = (cls.isin(["intraday", "BMO"])
                      | ((cls == "scheduled") & ~(hour.notna() & (hour >= 15))))
    log(f"  same-day reaction: {int(ev['same_day'].sum())}  next-day reaction: {int((~ev['same_day']).sum())}")
    return ev


def load_nifty() -> pd.DataFrame:
    nx = pd.read_feather(NIFTY_FEATHER)
    nx["date"] = pd.to_datetime(nx["date"]).dt.tz_localize(None).dt.normalize()
    nx = nx.sort_values("date").reset_index(drop=True)
    log(f"nifty daily: shape={nx.shape}  dates {nx['date'].min().date()} -> {nx['date'].max().date()}")
    nx["nret_1d"] = nx["close"] / nx["close"].shift(1) - 1.0          # keyed at reaction day d
    nx["nret_2d"] = nx["close"].shift(-1) / nx["close"].shift(1) - 1.0  # d-1 close -> d+1 close
    return nx[["date", "nret_1d", "nret_2d"]]


# ---------------------------------------------------------------------------
# Event -> reaction-day row mapping (vectorized merges, no per-event loop)
# ---------------------------------------------------------------------------

def map_reaction_rows(ev: pd.DataFrame, dd: pd.DataFrame) -> pd.DataFrame:
    keep = ["symbol", "date", "open", "close", "volume", "prev_close", "next_close",
            "next_volume", "next_date_ok", "adv20", "avgvol20"] \
        + [f"open_p{k}" for k in ENTRY_OFFS] + [f"close_p{k}" for k in EXIT_OFFS]
    bars = dd[keep]

    dd_syms = set(dd["symbol"].unique())
    n_sym_miss = int((~ev["symbol"].isin(dd_syms)).sum())
    log(f"events with symbol absent from daily data: {n_sym_miss} / {len(ev)}")

    # Same-day events: exact (symbol, announce_date) match; unmatched dropped
    # (announce day not a trading row — same behavior as the killtest).
    same = ev[ev["same_day"]].merge(
        bars, left_on=["symbol", "announce_date"], right_on=["symbol", "date"], how="inner")

    # Next-day events: first trading day STRICTLY AFTER announce_date, per symbol.
    nxt = ev[~ev["same_day"]].copy()
    nxt["target"] = nxt["announce_date"] + pd.Timedelta(days=1)
    nxt = nxt.sort_values("target", kind="mergesort")
    nxt = pd.merge_asof(nxt, bars.sort_values("date", kind="mergesort"),
                        left_on="target", right_on="date", by="symbol",
                        direction="forward",
                        tolerance=pd.Timedelta(days=REACTION_TOLERANCE_DAYS))
    nxt = nxt[nxt["date"].notna()].drop(columns=["target"])

    evr = pd.concat([same, nxt], ignore_index=True)
    evr = evr.rename(columns={"date": "reaction_date"})
    # Dedupe: multiple filings can map to one (symbol, reaction_date) — keep one.
    evr = evr.drop_duplicates(subset=["symbol", "reaction_date"]).reset_index(drop=True)
    log(f"events mapped to reaction rows: n={len(evr)} "
        f"(reaction dates {evr['reaction_date'].min().date()} -> {evr['reaction_date'].max().date()})")
    return evr


# ---------------------------------------------------------------------------
# Signal features + forward returns on the event frame
# ---------------------------------------------------------------------------

def build_features(evr: pd.DataFrame, nifty: pd.DataFrame, dd: pd.DataFrame) -> pd.DataFrame:
    e = evr.copy()
    e["react_ret_1"] = e["close"] / e["prev_close"] - 1.0
    e["react_ret_2"] = e["next_close"] / e["prev_close"] - 1.0

    # --- market adjustment: Nifty same-window return, universe-mean fallback pre-coverage
    e = e.merge(nifty, left_on="reaction_date", right_on="date", how="left").drop(columns=["date"])
    elig_dd = dd[dd["adv20"] >= ADV_FLOOR]
    uret1 = (elig_dd["close"] / elig_dd["prev_close"] - 1.0).groupby(elig_dd["date"]).mean()
    uret2 = (elig_dd["next_close"] / elig_dd["prev_close"] - 1.0).groupby(elig_dd["date"]).mean()
    fb1 = e["nret_1d"].isna()
    fb2 = e["nret_2d"].isna()
    e.loc[fb1, "nret_1d"] = e.loc[fb1, "reaction_date"].map(uret1)
    e.loc[fb2, "nret_2d"] = e.loc[fb2, "reaction_date"].map(uret2)
    log(f"market adjustment: universe-mean fallback used for {int(fb1.sum())} (1d) / "
        f"{int(fb2.sum())} (2d) events (Nifty coverage starts {nifty['date'].min().date()})")

    e["mktadj_ret_1"] = e["react_ret_1"] - e["nret_1d"]
    e["mktadj_ret_2"] = e["react_ret_2"] - e["nret_2d"]

    # --- volume confirmation: reaction-window mean share volume >= 2x AvgVol20
    e["volconf_1"] = e["volume"] >= VOLCONF_MULT * e["avgvol20"]
    e["volconf_2"] = ((e["volume"] + e["next_volume"]) / 2.0) >= VOLCONF_MULT * e["avgvol20"]

    # --- forward returns: fwd_{entry_off}_{hold} = close[pos+entry_off+hold-1]/open[pos+entry_off]-1
    for eo in ENTRY_OFFS:
        eo_open = e[f"open_p{eo}"].where(e[f"open_p{eo}"] > 0)
        for h in LOCKED_GRID["hold"]:
            e[f"fwd_{eo}_{h}"] = e[f"close_p{eo + h - 1}"] / eo_open - 1.0

    # --- eligibility: ADV floor + finite reaction returns
    n0 = len(e)
    e = e[(e["adv20"] >= ADV_FLOOR) & np.isfinite(e["react_ret_1"])].copy()
    log(f"eligible events after ADV floor (>= Rs {ADV_FLOOR:,.0f}) + finite 1d reaction: "
        f"n={len(e)} (dropped {n0 - len(e)})")
    # 2d-window cells additionally require a valid next trading day
    log(f"  of which valid for 2d reaction window: n={int((e['next_date_ok'] & np.isfinite(e['react_ret_2'])).sum())}")

    # ADV tiers (cross-sectional within scored set, tier1 = most illiquid, as in B1)
    e["adv_tier"], tier_edges = pd.qcut(e["adv20"], N_TIERS, labels=False, retbins=True)
    e["adv_tier"] = e["adv_tier"].astype(int) + 1
    e.attrs["tier_edges"] = tier_edges
    return e


# ---------------------------------------------------------------------------
# Baselines: unconditional same-universe same-horizon (per hold, per tier)
# ---------------------------------------------------------------------------

def build_baselines(dd: pd.DataFrame, tier_edges: np.ndarray) -> tuple[dict, dict]:
    u = dd[(dd["adv20"] >= ADV_FLOOR)
           & (dd["date"] >= DISCOVERY_START) & (dd["date"] <= DISCOVERY_END)
           & (dd["open"] > 0)].copy()
    log(f"baseline universe (ADV-eligible discovery symbol-days): n={len(u):,}")
    edges = tier_edges.copy()
    edges[0], edges[-1] = -np.inf, np.inf   # clip outliers into end tiers
    u["adv_tier"] = pd.cut(u["adv20"], bins=edges, labels=False) + 1

    base, base_tier = {}, {}
    for h in LOCKED_GRID["hold"]:
        fwd = u[f"close_p{h - 1}"] / u["open"] - 1.0
        base[h] = float(fwd.mean())
        base_tier[h] = fwd.groupby(u["adv_tier"]).mean().to_dict()
        log(f"  baseline h={h:2d}: {base[h]:+.4%} (n={int(fwd.notna().sum()):,})")
    return base, base_tier


# ---------------------------------------------------------------------------
# Grid evaluation
# ---------------------------------------------------------------------------

def cohort_for(e: pd.DataFrame, proxy: str, window: int, leg: str, threshold: str) -> tuple[pd.DataFrame, float]:
    """Return (signal cohort, threshold value used). Thresholds pooled over discovery cohort."""
    pool = e
    if window == 2:
        pool = pool[pool["next_date_ok"] & np.isfinite(pool["react_ret_2"])]
    if proxy == "volconf":
        pool = pool[pool[f"volconf_{window}"].fillna(False)]
        sig_col = f"react_ret_{window}"
    elif proxy == "mktadj":
        sig_col = f"mktadj_ret_{window}"
        pool = pool[np.isfinite(pool[sig_col])]
    else:
        sig_col = f"react_ret_{window}"
    s = pool[sig_col]
    if threshold == "decile":
        thr = s.quantile(0.90) if leg == "long" else s.quantile(0.10)
    elif threshold == "pct5":
        thr = s.quantile(0.95) if leg == "long" else s.quantile(0.05)
    else:  # abs3
        thr = ABS3_THRESHOLD if leg == "long" else -ABS3_THRESHOLD
    cohort = pool[s >= thr] if leg == "long" else pool[s <= thr]
    return cohort, float(thr)


def run_grid(e: pd.DataFrame, base: dict) -> pd.DataFrame:
    rows = []
    for proxy in LOCKED_GRID["surprise_proxy"]:
        for window in LOCKED_GRID["reaction_window"]:
            for leg in LOCKED_GRID["leg"]:
                for threshold in LOCKED_GRID["threshold"]:
                    cohort, thr = cohort_for(e, proxy, window, leg, threshold)
                    for lag in LOCKED_GRID["entry_lag"]:
                        eo = (window - 1) + lag
                        for h in LOCKED_GRID["hold"]:
                            fwd = cohort[f"fwd_{eo}_{h}"].dropna()
                            n = len(fwd)
                            sig = float(fwd.mean()) if n else np.nan
                            b = base[h]
                            delta = (sig - b) if leg == "long" else (b - sig)
                            hit = float((fwd > 0).mean() if leg == "long" else (fwd < 0).mean()) if n else np.nan
                            rows.append({
                                "proxy": proxy, "reaction_window": window, "leg": leg,
                                "threshold": threshold, "entry_lag": lag, "hold": h,
                                "threshold_value": round(thr, 6), "n_events": n,
                                "signal_mean_drift_pct": round(sig * 100, 4) if n else np.nan,
                                "baseline_mean_pct": round(b * 100, 4),
                                "delta_pct": round(delta * 100, 4) if n else np.nan,
                                "hit_rate": round(hit, 4) if n else np.nan,
                            })
    return pd.DataFrame(rows)


def tier_split_top_cells(e: pd.DataFrame, cells: pd.DataFrame, base_tier: dict) -> pd.DataFrame:
    ok = cells.dropna(subset=["delta_pct"])
    top_abs = ok.reindex(ok["delta_pct"].abs().sort_values(ascending=False).index).head(N_TOP_CELLS_FOR_TIERS)
    # Each leg must pass falsifier #1 independently -> also include top-5 per leg by delta.
    per_leg = pd.concat([ok[ok["leg"] == leg].nlargest(5, "delta_pct") for leg in LOCKED_GRID["leg"]])
    top = pd.concat([top_abs, per_leg]).drop_duplicates(
        subset=["proxy", "reaction_window", "leg", "threshold", "entry_lag", "hold"])
    rows = []
    for _, c in top.iterrows():
        cohort, _ = cohort_for(e, c["proxy"], c["reaction_window"], c["leg"], c["threshold"])
        eo = (c["reaction_window"] - 1) + c["entry_lag"]
        col = f"fwd_{eo}_{c['hold']}"
        for t, g in cohort.groupby("adv_tier"):
            fwd = g[col].dropna()
            if not len(fwd):
                continue
            sig = float(fwd.mean())
            b = base_tier[c["hold"]].get(t, np.nan)
            delta = (sig - b) if c["leg"] == "long" else (b - sig)
            rows.append({
                "proxy": c["proxy"], "reaction_window": c["reaction_window"], "leg": c["leg"],
                "threshold": c["threshold"], "entry_lag": c["entry_lag"], "hold": c["hold"],
                "adv_tier": t, "n_events": len(fwd),
                "signal_mean_drift_pct": round(sig * 100, 4),
                "tier_baseline_mean_pct": round(b * 100, 4),
                "delta_pct": round(delta * 100, 4),
            })
    return pd.DataFrame(rows)


def main() -> None:
    log("=" * 78)
    log("PHASE 2 SIGNATURE: pead_reaction_drift  (discovery 2023-01-01 -> 2024-12-31 ONLY)")
    log("=" * 78)
    dd = load_daily()
    ev = load_events()
    nifty = load_nifty()
    evr = map_reaction_rows(ev, dd)
    e = build_features(evr, nifty, dd)
    base, base_tier = build_baselines(dd, e.attrs["tier_edges"])

    cells = run_grid(e, base)
    OUT_CELLS_CSV.parent.mkdir(parents=True, exist_ok=True)
    cells.to_csv(OUT_CELLS_CSV, index=False)
    log(f"wrote {len(cells)} grid cells -> {OUT_CELLS_CSV}")

    tiers = tier_split_top_cells(e, cells, base_tier)
    tiers.to_csv(OUT_TIERS_CSV, index=False)
    log(f"wrote ADV-tier split for top {N_TOP_CELLS_FOR_TIERS} cells -> {OUT_TIERS_CSV}")

    ok = cells.dropna(subset=["delta_pct"])
    log("\n=== TOP 15 CELLS by |delta_pct| ===")
    top15 = ok.reindex(ok["delta_pct"].abs().sort_values(ascending=False).index).head(15)
    print(top15.to_string(index=False))
    log("\n=== BOTTOM 5 CELLS by |delta_pct| ===")
    print(ok.reindex(ok["delta_pct"].abs().sort_values(ascending=True).index).head(5).to_string(index=False))
    for leg in ["long", "short"]:
        sub = ok[ok["leg"] == leg]
        n_clear = int((sub["delta_pct"] >= 0.1).sum())
        log(f"cells clearing +0.1% delta floor [{leg}]: {n_clear} / {len(sub)}")
    log("\n=== ADV-TIER SPLIT (top cells) ===")
    if len(tiers):
        piv = tiers.pivot_table(index=["proxy", "reaction_window", "leg", "threshold", "entry_lag", "hold"],
                                columns="adv_tier", values="delta_pct")
        print(piv.to_string())


if __name__ == "__main__":
    main()
