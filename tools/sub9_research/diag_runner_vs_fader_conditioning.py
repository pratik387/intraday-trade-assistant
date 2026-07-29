"""DIAGNOSTIC: runner vs fader -- which SINGLE conditioning variable separates
forward winners from losers inside the momentum screen?

NOT a candidate. NOT a validation. Phase-1/2 exploration on DEVELOPMENT windows
only. No ledger line, no cell selection, no freeze, no commit. Nothing here may
be cited as evidence for a setup; anything it suggests must go through its own
freeze + fresh-pool one-shot (docs/setup_lifecycle.md).

WHY
  `diag_big_mover_anatomy.py` proved the naive momentum screen (move >= 3% from
  open, cumvol >= 3x same-time-of-day baseline, price on the right side of VWAP)
  CANNOT separate runners from faders: ~19 names/day fire, ~24% become +8%
  movers, mean forward return is ~0 in era_A and negative in every era_B cell,
  and the only profitable slices were circuit-LOCKED, i.e. unbuyable.

  `screen_event_classes_cost_clearing.py` showed exactly ONE conditioning
  variable that did work: an earnings down-shock. Knowing WHY a stock is moving
  separated continuation from noise. That is the proof of concept.

  This study asks: applied to the SAME screened population, which OTHER single
  conditioning variables separate winners from losers, with a mechanistic
  reason, in BOTH eras?

METHOD (anti-overfitting guards are binding)
  * Population = the anatomy screen's firing set, rebuilt identically from the
    same cached checkpoint panel. Checkpoints 11:00 and 13:00; UP and DOWN sides
    measured separately. Entry = OPEN of the checkpoint bar (next-bar-open
    fill), exit = session close.
  * CIRCUIT-LOCKED names are EXCLUDED (>=3 zero-range bars at the day's extreme
    on the relevant side). The anatomy study proved that slice is the fake edge.
  * Returns are signed by side (UP -> long, DOWN -> short). Gross, plus a
    net-of-0.31% round-trip column.
  * EVERY FEATURE IS TESTED ALONE. No combinations, no interactions, no ML.
    A feature that only works in combination is a degree of freedom, not
    evidence.
  * Bin edges (terciles unless stated) are computed on the POOLED both-era
    screened population for that side x checkpoint, then applied unchanged to
    each era. Look-ahead therefore touches the THRESHOLD only, never the
    returns, and both eras are scored on the same cut -- which is the whole
    point of an era split.
  * PASS requires ALL of:
        - same-sign top-minus-bottom spread in BOTH eras
        - min(|spread_A|, |spread_B|) >= 0.30pp
        - monotone-ish ordering across bins in BOTH eras
        - n >= MIN_BIN_N in the top and bottom bin of BOTH eras
  * Era split is mandatory: era_A 2023-01..2024-12, era_B 2025-01..2026-04.
  * No signal dated >= 2026-05-01 is touched. Fresh pool stays clean.

CAUSALITY
  Every feature is computable at the checkpoint. Intraday structure uses bars
  STRICTLY BEFORE the checkpoint. Daily context uses shift(1) / rolling windows
  ending at D-1. Per-feature provenance is printed and written to the CSV in the
  `causal_note` column. Two same-day deal features are deliberately included as
  NOT-TRADEABLE mechanistic references and are labelled `NONCAUSAL_*`.

OUTPUTS
  reports/sub9_sanity/_runner_vs_fader_conditioning.csv
  console: per-feature bin tables + PASS/FAIL verdict table
"""
from __future__ import annotations

import glob
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 80)
pd.set_option("display.max_rows", 800)

MONTHLY = ROOT / "backtest-cache-download" / "monthly"
CLEAN_DAILY = ROOT / "cache" / "preaggregate" / "clean_daily_from5m.feather"
SANITY = ROOT / "reports" / "sub9_sanity"
OUT_CSV = SANITY / "_runner_vs_fader_conditioning.csv"

CACHE_DIR = Path(
    r"C:\Users\PRATIK\AppData\Local\Temp\claude"
    r"\E--Codebase-intraday-trade-assistant"
    r"\d9c67968-368c-45aa-a25e-bd4d1cfb4906\scratchpad"
)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
STRUCT_CACHE = CACHE_DIR / "runner_fader_struct_panel.parquet"
INDEX_CACHE = CACHE_DIR / "runner_fader_index_panel.parquet"

ANATOMY = ROOT / "tools" / "sub9_research" / "diag_big_mover_anatomy.py"

FRESH_POOL_START = pd.Timestamp("2026-05-01")
ERA_A_END = pd.Timestamp("2024-12-31")
ERA_B_END = pd.Timestamp("2026-04-30")

CHECKPOINTS = ["11:00", "13:00"]
CP_MIN = {c: int(c[:2]) * 60 + int(c[3:]) for c in CHECKPOINTS}
ACCEL_WINDOW_MIN = 60           # "last hour" for the volume-acceleration feature

MOVE_THR = 0.03                 # identical to the anatomy screen
VOLX_THR = 3.0
COST_PCT = 0.31                 # round-trip, pct points

MIN_BIN_N = 150                 # per bin, per era, for a PASS
SPREAD_PP = 0.30                # minimum |spread| in percentage points
MONO_TOL = 0.15                 # pp of allowed reversal in the middle bin

INDEX_PROXY = "NIFTYBEES"       # Nifty-50 ETF; `data/index_ohlcv` does not exist
NSE_BANDS = np.array([2.0, 5.0, 10.0, 20.0])

_ROWS: list[dict] = []


def emit(table: str, **kw) -> None:
    _ROWS.append({"table": table, **kw})


def _norm(s) -> str:
    s = str(s).strip().upper()
    return s.split(":", 1)[1] if ":" in s else s


def _nseries(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.strip().str.upper()
            .str.replace(r"^(NSE|BSE)\s*:\s*", "", regex=True))


# ============================================================== BASE POPULATION
def load_anatomy_module():
    spec = importlib.util.spec_from_file_location("_anatomy", ANATOMY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_universe() -> pd.DataFrame:
    """Rebuild the anatomy screen's universe by calling the anatomy module's own
    functions -- guarantees the population is identical, not merely similar."""
    an = load_anatomy_module()
    panel = an.build_panel()                       # cached parquet, reused
    p = an.enrich(panel)
    u = an.apply_universe(p)
    return u


# ======================================================== INTRADAY STRUCTURE
def reduce_struct_month(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (symbol, session) MOVE-STRUCTURE features, from bars strictly before
    each checkpoint. Also returns the index-proxy checkpoint row for that month.
    """
    df = pd.read_feather(
        path, columns=["date", "symbol", "open", "high", "low", "close",
                       "volume", "vwap"])
    df["symbol"] = df["symbol"].map(_norm)
    df["day"] = pd.to_datetime(df["date"].values.astype("datetime64[D]"))
    df["tmin"] = df["date"].dt.hour * 60 + df["date"].dt.minute
    df = df.sort_values(["symbol", "date"], kind="stable")

    key = ["symbol", "day"]

    # ---- index proxy (NIFTYBEES): open -> checkpoint return, causal ----------
    ix = df[df["symbol"] == INDEX_PROXY]
    irows = []
    if len(ix):
        for (sym, day), g in ix.groupby(key, sort=False):
            r = {"day": day}
            r["ix_open"] = float(g["open"].iloc[0])
            for c in CHECKPOINTS:
                pre = g[g["tmin"] < CP_MIN[c]]
                tag = c.replace(":", "")
                r[f"ix_move_{tag}"] = (float(pre["close"].iloc[-1]) / r["ix_open"] - 1.0
                                       if len(pre) else np.nan)
            irows.append(r)
    idx_df = pd.DataFrame(irows)

    out = df.groupby(key, sort=False).size().rename("_n").to_frame()

    for c in CHECKPOINTS:
        t = CP_MIN[c]
        tag = c.replace(":", "")
        pre = df[df["tmin"] < t]
        if not len(pre):
            continue
        g = pre.groupby(key, sort=False)

        body = (pre["close"] - pre["open"]).abs()
        pre = pre.assign(_body=body,
                         _up=(pre["close"] > pre["open"]).astype(np.int32),
                         _dn=(pre["close"] < pre["open"]).astype(np.int32))
        gg = pre.groupby(key, sort=False)
        out[f"n_pre_{tag}"] = gg["_body"].size()
        out[f"up_bars_{tag}"] = gg["_up"].sum()
        out[f"dn_bars_{tag}"] = gg["_dn"].sum()
        out[f"max_body_{tag}"] = gg["_body"].max()

        # deepest pullback so far, both directions (running extreme -> excursion)
        cmax = gg["high"].cummax()
        cmin = gg["low"].cummin()
        dd_up = (cmax.to_numpy() - pre["low"].to_numpy()) / np.maximum(cmax.to_numpy(), 1e-9)
        dd_dn = (pre["high"].to_numpy() - cmin.to_numpy()) / np.maximum(cmin.to_numpy(), 1e-9)
        tmp = pd.DataFrame({"symbol": pre["symbol"].to_numpy(),
                            "day": pre["day"].to_numpy(),
                            "dd_up": dd_up, "dd_dn": dd_dn})
        dd = tmp.groupby(["symbol", "day"], sort=False)[["dd_up", "dd_dn"]].max()
        out[f"pullback_up_{tag}"] = dd["dd_up"]
        out[f"pullback_dn_{tag}"] = dd["dd_dn"]

        # volume acceleration: per-bar volume in the last hour vs everything before
        late = pre[pre["tmin"] >= t - ACCEL_WINDOW_MIN]
        early = pre[pre["tmin"] < t - ACCEL_WINDOW_MIN]
        gl = late.groupby(key, sort=False)["volume"]
        ge = early.groupby(key, sort=False)["volume"]
        out[f"vol_late_{tag}"] = gl.sum()
        out[f"bars_late_{tag}"] = gl.size()
        out[f"vol_early_{tag}"] = ge.sum()
        out[f"bars_early_{tag}"] = ge.size()

        # VWAP one hour before the checkpoint -> VWAP slope over the last hour
        out[f"vwap_lag_{tag}"] = early.groupby(key, sort=False)["vwap"].last()

    return out.reset_index().drop(columns=["_n"]), idx_df


def build_struct(months: list[str], force: bool = False):
    if STRUCT_CACHE.exists() and INDEX_CACHE.exists() and not force:
        return pd.read_parquet(STRUCT_CACHE), pd.read_parquet(INDEX_CACHE)
    parts, iparts = [], []
    for m in months:
        p = MONTHLY / f"{m}_5m_enriched.feather"
        if not p.exists():
            print(f"  !! missing {p.name}")
            continue
        t0 = time.time()
        r, ix = reduce_struct_month(p)
        parts.append(r)
        if len(ix):
            iparts.append(ix)
        print(f"  struct {m}: {len(r):>7,} symbol-days ({time.time()-t0:.1f}s)", flush=True)
    s = pd.concat(parts, ignore_index=True)
    ixp = pd.concat(iparts, ignore_index=True).drop_duplicates("day")
    s.to_parquet(STRUCT_CACHE, index=False)
    ixp.to_parquet(INDEX_CACHE, index=False)
    return s, ixp


# ============================================================ DAILY CONTEXT
def build_daily_context(syms: set[str]) -> pd.DataFrame:
    """All strictly-prior-session daily context. Every column here is shift(1)
    or a rolling window ENDING at D-1; nothing from session D is used."""
    d = pd.read_feather(CLEAN_DAILY)
    d["symbol"] = _nseries(d["symbol"])
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["symbol"].isin(syms)].sort_values(["symbol", "date"]).reset_index(drop=True)
    g = d.groupby("symbol", sort=False)

    d["prev_close"] = g["close"].shift(1)
    tr = np.maximum.reduce([
        (d["high"] - d["low"]).to_numpy(float),
        (d["high"] - d["prev_close"]).abs().to_numpy(float),
        (d["low"] - d["prev_close"]).abs().to_numpy(float)])
    d["_tr"] = tr
    d["atr14_prev"] = g["_tr"].transform(
        lambda s: s.rolling(14, min_periods=8).mean().shift(1))

    # 52-week extremes as of D-1
    d["hi52_prev"] = g["high"].transform(
        lambda s: s.rolling(250, min_periods=60).max().shift(1))
    d["lo52_prev"] = g["low"].transform(
        lambda s: s.rolling(250, min_periods=60).min().shift(1))

    # consecutive up-closes ending D-1
    up = (d["close"] > d["prev_close"]).astype(int)
    d["_up"] = up
    grp = (up != up.groupby(d["symbol"]).shift(1)).cumsum()
    d["_streak"] = up.groupby([d["symbol"], grp]).cumcount() + 1
    d["_streak"] = np.where(up == 1, d["_streak"], 0)
    d["consec_up_prev"] = g["_streak"].shift(1)

    # was the name a big mover in the prior 5 sessions?
    d["_bigmove"] = ((d["close"] / d["open"] - 1.0).abs() >= 0.08).astype(float)
    d["repeat_mover_prev5"] = g["_bigmove"].transform(
        lambda s: s.rolling(5, min_periods=1).max().shift(1))

    # circuit-band PROXY: smallest NSE band that contains the trailing-250d max
    # excursion vs prior close. No band table exists on disk; this is structural.
    exc = np.maximum((d["high"] / d["prev_close"] - 1.0).abs(),
                     (d["low"] / d["prev_close"] - 1.0).abs()) * 100.0
    d["_exc"] = exc
    mx = g["_exc"].transform(lambda s: s.rolling(250, min_periods=60).max().shift(1))
    idx = np.searchsorted(NSE_BANDS, mx.to_numpy(float) - 1e-9)
    idx = np.clip(idx, 0, len(NSE_BANDS) - 1)
    d["band_est_pct"] = np.where(np.isfinite(mx), NSE_BANDS[idx], np.nan)

    keep = ["symbol", "date", "prev_close", "atr14_prev", "hi52_prev", "lo52_prev",
            "consec_up_prev", "repeat_mover_prev5", "band_est_pct"]
    return d[keep].rename(columns={"date": "day"})


def build_delivery(syms: set[str]) -> pd.DataFrame:
    """PRIOR-session delivery%, published in the D-1 bhavcopy after that close,
    hence fully known at any checkpoint on D."""
    dl = pd.read_parquet(ROOT / "data" / "delivery_pct" / "delivery_history.parquet",
                         columns=["symbol", "date", "series", "delivery_pct"])
    dl = dl[dl["series"] == "EQ"].copy()
    dl["symbol"] = _nseries(dl["symbol"])
    dl["date"] = pd.to_datetime(dl["date"])
    dl = dl[dl["symbol"].isin(syms)].sort_values(["symbol", "date"])
    g = dl.groupby("symbol", sort=False)["delivery_pct"]
    dl["base"] = g.transform(lambda s: s.rolling(20, min_periods=10).median().shift(1))
    dl["ratio"] = dl["delivery_pct"] / dl["base"].replace(0, np.nan)
    dl = dl.rename(columns={"date": "_dl_date"})
    return dl[["symbol", "_dl_date", "delivery_pct", "ratio"]]


# ================================================================== EVENTS
def build_event_flags(u: pd.DataFrame) -> pd.DataFrame:
    """Per (symbol, day) event flags. Causality is stated per flag.

    CAUSAL at the checkpoint
      ev_earnings   : result announced on D-1 (any time) or on D pre-open (BMO).
                      An AMC/intraday D announcement is NOT counted -- it is not
                      knowable at 11:00.
      ev_ca_ex      : split / bonus / dividend ex-date on D (scheduled ahead).
      ev_asm        : name is on the NSE ASM/GSM list on D (list published the
                      previous evening, effective D).
      ev_asm_change : ASM/GSM stage_up / stage_down / entry effective D.
      ev_recon      : index-reconstitution effective date within [D, D+10] or
                      announcement dated <= D-1 and effective >= D.
      ev_deal_prev  : block or bulk deal reported for D-1 (reported post-close
                      D-1, so known before D's open).
      no_event      : none of the causal flags above.

    NOT CAUSAL (reference only, never tradeable)
      NONCAUSAL_deal_today : block/bulk deal executed on D. Reported only after
                      the close, so a live system cannot see it at 11:00. Kept
                      because the mechanism question ("is the move a real
                      institutional print?") is worth measuring even when the
                      answer is unusable.
    """
    idx = u[["symbol", "day"]].drop_duplicates().reset_index(drop=True)
    syms = set(idx["symbol"])
    flags = idx.copy()

    def _mark(pairs: set, name: str) -> None:
        flags[name] = [(s, d) in pairs for s, d in zip(flags["symbol"], flags["day"])]

    # ---- trading-session calendar, to shift "D-1" correctly -----------------
    sessions = np.sort(u["day"].unique())
    pos = {d: i for i, d in enumerate(sessions)}

    def next_session(ts: pd.Timestamp):
        i = np.searchsorted(sessions, np.datetime64(ts), side="left")
        return sessions[i] if i < len(sessions) else None

    def session_after(ts: pd.Timestamp):
        i = np.searchsorted(sessions, np.datetime64(ts), side="right")
        return sessions[i] if i < len(sessions) else None

    # ---- earnings -----------------------------------------------------------
    e = pd.read_parquet(ROOT / "data" / "earnings_calendar" / "earnings_events.parquet",
                        columns=["symbol", "announce_date", "announce_time",
                                 "announce_time_class"])
    e["symbol"] = _nseries(e["symbol"])
    e = e[e["symbol"].isin(syms)]
    e["announce_date"] = pd.to_datetime(e["announce_date"])
    hour = e["announce_time"].dt.hour.fillna(18)
    pre_open = e["announce_time_class"].eq("BMO") & (hour < 9)
    pairs = set()
    for sym, ad, po in zip(e["symbol"], e["announce_date"], pre_open):
        if not pd.notna(ad):
            continue
        # BMO on D -> actionable on D itself; anything else -> next session
        d0 = next_session(ad) if po else session_after(ad)
        if d0 is not None:
            pairs.add((sym, pd.Timestamp(d0)))
    _mark(pairs, "ev_earnings")

    # ---- corporate actions: split / bonus / dividend ex-date on D ------------
    ca = pd.read_parquet(ROOT / "data" / "corporate_actions" / "split_bonus_events.parquet",
                         columns=["symbol", "ex_date"])
    ca["symbol"] = _nseries(ca["symbol"])
    ca["ex_date"] = pd.to_datetime(ca["ex_date"])
    dv = pd.read_parquet(ROOT / "data" / "dividend_ex_date" / "dividend_events.parquet",
                         columns=["symbol", "ex_date"])
    dv["symbol"] = _nseries(dv["symbol"])
    dv["ex_date"] = pd.to_datetime(dv["ex_date"])
    capairs = set(zip(ca["symbol"], ca["ex_date"])) | set(zip(dv["symbol"], dv["ex_date"]))
    _mark(capairs, "ev_ca_ex")

    # ---- ASM / GSM ----------------------------------------------------------
    a = pd.read_parquet(ROOT / "data" / "asm_gsm_history" / "asm_gsm_events.parquet",
                        columns=["symbol", "exchange", "date", "transition_type"])
    a = a[a["exchange"] == "NSE"].copy()
    a["symbol"] = _nseries(a["symbol"])
    a = a[a["symbol"].isin(syms)]
    a["date"] = pd.to_datetime(a["date"])
    _mark(set(zip(a["symbol"], a["date"])), "ev_asm")
    ch = a[a["transition_type"].isin(["entry", "stage_up", "stage_down"])]
    _mark(set(zip(ch["symbol"], ch["date"])), "ev_asm_change")

    # ---- index reconstitution ----------------------------------------------
    r = pd.read_parquet(ROOT / "data" / "index_reconstitution" / "events.parquet")
    r["symbol"] = _nseries(r["symbol"])
    r["announcement_date"] = pd.to_datetime(r["announcement_date"])
    r["effective_date"] = pd.to_datetime(r["effective_date"])
    r = r[r["symbol"].isin(syms)]
    rpairs = set()
    for sym, ann, eff in zip(r["symbol"], r["announcement_date"], r["effective_date"]):
        if not pd.notna(eff):
            continue
        lo = max(ann + pd.Timedelta(days=1), eff - pd.Timedelta(days=10)) \
            if pd.notna(ann) else eff - pd.Timedelta(days=10)
        win = sessions[(sessions >= np.datetime64(lo)) &
                       (sessions <= np.datetime64(eff + pd.Timedelta(days=1)))]
        for d0 in win:
            rpairs.add((sym, pd.Timestamp(d0)))
    _mark(rpairs, "ev_recon")

    # ---- block / bulk deals -------------------------------------------------
    bl = pd.read_parquet(ROOT / "data" / "block_deals" / "block_deals_events.parquet",
                         columns=["symbol", "trade_date"])
    bl["symbol"] = _nseries(bl["symbol"])
    bl["trade_date"] = pd.to_datetime(bl["trade_date"])
    bu = pd.read_parquet(ROOT / "data" / "bulk_deals_cache" / "nse_bulk_deals_2023_2026.parquet",
                         columns=["Symbol", "Date"])
    bu["symbol"] = _nseries(bu["Symbol"])
    bu["trade_date"] = pd.to_datetime(bu["Date"], format="%d-%b-%Y", errors="coerce")
    deals = pd.concat([bl[["symbol", "trade_date"]], bu[["symbol", "trade_date"]]],
                      ignore_index=True).dropna()
    deals = deals[deals["symbol"].isin(syms)].drop_duplicates()
    _mark(set(zip(deals["symbol"], deals["trade_date"])), "NONCAUSAL_deal_today")
    dprev = set()
    for sym, td in zip(deals["symbol"], deals["trade_date"]):
        d0 = session_after(td)
        if d0 is not None:
            dprev.add((sym, pd.Timestamp(d0)))
    _mark(dprev, "ev_deal_prev")

    causal = ["ev_earnings", "ev_ca_ex", "ev_asm_change", "ev_recon", "ev_deal_prev"]
    flags["no_event"] = ~flags[causal].any(axis=1)
    flags["ev_any"] = flags[causal].any(axis=1)
    return flags


# ================================================================== SECTORS
def build_sector_map() -> dict:
    m: dict[str, str] = {}
    try:
        j = json.loads((ROOT / "assets" / "stock_sector_map.json").read_text())
        for k, v in j.items():
            m[_norm(k)] = str(v)
    except Exception:
        pass
    for f in glob.glob(str(ROOT / "assets" / "ind_nifty*list.csv")):
        try:
            d = pd.read_csv(f)
            for s, ind in zip(d["Symbol"], d["Industry"]):
                m.setdefault(_norm(s), str(ind))
        except Exception:
            continue
    return m


# ========================================================== FEATURE ASSEMBLY
def assemble(u: pd.DataFrame, struct: pd.DataFrame, idxp: pd.DataFrame,
             daily: pd.DataFrame, deliv: pd.DataFrame, flags: pd.DataFrame,
             secmap: dict, mcap: dict) -> pd.DataFrame:
    df = u.merge(struct, on=["symbol", "day"], how="left")
    df = df.merge(daily, on=["symbol", "day"], how="left", suffixes=("", "_d"))
    df = df.merge(flags, on=["symbol", "day"], how="left")
    df = df.merge(idxp, on="day", how="left")

    # prior-session delivery: last delivery row STRICTLY BEFORE day D
    df = df.sort_values("day", kind="stable")
    dl = deliv.sort_values("_dl_date")
    df = pd.merge_asof(df, dl, left_on="day", right_on="_dl_date", by="symbol",
                       direction="backward", allow_exact_matches=False,
                       tolerance=pd.Timedelta(days=7))
    df = df.rename(columns={"delivery_pct": "prev_deliv_pct",
                            "ratio": "prev_deliv_ratio"})

    df["sector"] = df["symbol"].map(secmap)
    df["mcap_cr"] = df["symbol"].map(mcap)
    df["gap_pct"] = (df["day_open"] / df["prev_close"] - 1.0) * 100.0
    df["atr_pct_prev"] = df["atr14_prev"] / df["prev_close"] * 100.0

    for c in CHECKPOINTS:
        tag = c.replace(":", "")
        # ---- move structure -------------------------------------------------
        nl = df[f"bars_late_{tag}"].replace(0, np.nan)
        ne = df[f"bars_early_{tag}"].replace(0, np.nan)
        df[f"vol_accel_{tag}"] = ((df[f"vol_late_{tag}"] / nl) /
                                  (df[f"vol_early_{tag}"] / ne).replace(0, np.nan))
        mv_px = (df[f"px_dec_{tag}"] - df["day_open"]).abs().replace(0, np.nan)
        df[f"max_bar_share_{tag}"] = df[f"max_body_{tag}"] / mv_px
        nb = (df[f"up_bars_{tag}"] + df[f"dn_bars_{tag}"]).replace(0, np.nan)
        df[f"upbar_frac_{tag}"] = df[f"up_bars_{tag}"] / nb
        atr_px = df["atr14_prev"].replace(0, np.nan)
        df[f"vwap_dist_atr_{tag}"] = (df[f"px_dec_{tag}"] - df[f"vwap_{tag}"]) / atr_px
        df[f"vwap_slope_{tag}"] = (df[f"vwap_{tag}"] / df[f"vwap_lag_{tag}"] - 1.0) * 100.0
        # ---- room to run ----------------------------------------------------
        mv_prev = (df[f"px_dec_{tag}"] / df["prev_close"] - 1.0) * 100.0
        df[f"band_room_up_{tag}"] = df["band_est_pct"] - mv_prev
        df[f"band_room_dn_{tag}"] = df["band_est_pct"] + mv_prev
        df[f"room_52w_up_{tag}"] = (df["hi52_prev"] / df[f"px_dec_{tag}"] - 1.0) * 100.0
        df[f"room_52w_dn_{tag}"] = (df[f"px_dec_{tag}"] / df["lo52_prev"] - 1.0) * 100.0
        # ---- sector co-movement (leave-one-out peer mean move) --------------
        mv = df[f"px_dec_{tag}"] / df["day_open"] - 1.0
        df[f"_mv_{tag}"] = mv
        ok = df["sector"].notna() & mv.notna()
        sub = df[ok]
        gs = sub.groupby(["day", "sector"], sort=False)[f"_mv_{tag}"]
        ssum, scnt = gs.transform("sum"), gs.transform("size")
        loo = (ssum - sub[f"_mv_{tag}"]) / (scnt - 1).replace(0, np.nan)
        df[f"peer_move_{tag}"] = np.nan
        df.loc[sub.index, f"peer_move_{tag}"] = loo * 100.0
        df.loc[sub.index, f"peer_n_{tag}"] = scnt
    return df


# ============================================================ SCREEN + SCORE
def screened(df: pd.DataFrame, side: str, cp: str) -> pd.DataFrame:
    tag = cp.replace(":", "")
    sgn = 1.0 if side == "UP" else -1.0
    mv = df[f"move_{tag}"] * sgn
    vx = df[f"volx_{tag}"]
    vw = (df[f"px_dec_{tag}"] > df[f"vwap_{tag}"]) if side == "UP" \
        else (df[f"px_dec_{tag}"] < df[f"vwap_{tag}"])
    lock = (df["zr_at_high"] >= 3) if side == "UP" else (df["zr_at_low"] >= 3)
    fwd = df[f"fwd_{tag}"] * sgn
    sel = (mv.notna() & vx.notna() & fwd.notna()
           & (mv >= MOVE_THR) & (vx >= VOLX_THR) & vw & ~lock
           & df["era"].isin(["era_A", "era_B"]))
    s = df[sel].copy()
    s["fwd_gross_pct"] = fwd[sel] * 100.0
    s["fwd_net_pct"] = s["fwd_gross_pct"] - COST_PCT
    s["side"] = side
    s["cp"] = cp
    return s


# feature spec: (key, group, kind, getter(df, side, tag) -> Series, note)
def feature_specs() -> list[dict]:
    def col(name):
        return lambda d, side, tag: d[f"{name}_{tag}"]

    def sidecol(up, dn):
        return lambda d, side, tag: d[f"{(up if side == 'UP' else dn)}_{tag}"]

    def plain(name):
        return lambda d, side, tag: d[name]

    def signed(name):
        return lambda d, side, tag: d[name] * (1.0 if side == "UP" else -1.0)

    S = []
    add = S.append

    # ---- A. WHY is it moving -------------------------------------------------
    for k, note in [
        ("ev_earnings", "CAUSAL: result announced D-1 any time, or D pre-open (BMO). AMC/intraday-D excluded."),
        ("ev_ca_ex", "CAUSAL: split/bonus/dividend ex-date on D (scheduled in advance)."),
        ("ev_asm", "CAUSAL: on NSE ASM/GSM list on D (published previous evening)."),
        ("ev_asm_change", "CAUSAL: ASM/GSM entry or stage change effective D."),
        ("ev_recon", "CAUSAL: index-recon window (announced <= D-1, effective within +10cd)."),
        ("ev_deal_prev", "CAUSAL: block/bulk deal reported for D-1 (post-close D-1)."),
        ("ev_any", "CAUSAL: any of the causal event flags."),
        ("no_event", "CAUSAL: residual bucket -- none of the causal event flags fire."),
        ("NONCAUSAL_deal_today", "NOT CAUSAL: block/bulk deal executed on D, reported only post-close."),
    ]:
        add(dict(key=k, group="A_event", kind="bool", get=plain(k), note=note))

    # ---- B. Move structure ---------------------------------------------------
    add(dict(key="vol_accel", group="B_structure", kind="num", bins=3,
             get=col("vol_accel"),
             note="CAUSAL: per-bar volume last 60m before cp / per-bar volume before that."))
    add(dict(key="max_bar_share", group="B_structure", kind="num", bins=3,
             get=col("max_bar_share"),
             note="CAUSAL: largest single pre-cp bar body / |decision px - day open|."))
    add(dict(key="upbar_frac", group="B_structure", kind="num", bins=3,
             get=col("upbar_frac"),
             note="CAUSAL: up-bars / (up+down bars) among bars strictly before cp."))
    add(dict(key="pullback_depth", group="B_structure", kind="num", bins=3,
             get=sidecol("pullback_up", "pullback_dn"),
             note="CAUSAL: deepest excursion off the running extreme, pre-cp bars only."))
    add(dict(key="vwap_dist_atr", group="B_structure", kind="num", bins=3,
             get=lambda d, side, tag: d[f"vwap_dist_atr_{tag}"] * (1.0 if side == "UP" else -1.0),
             note="CAUSAL: (decision px - session VWAP) / ATR14 as of D-1, signed by side."))
    add(dict(key="vwap_slope", group="B_structure", kind="num", bins=3,
             get=lambda d, side, tag: d[f"vwap_slope_{tag}"] * (1.0 if side == "UP" else -1.0),
             note="CAUSAL: session VWAP now vs 60m earlier, signed by side."))

    # ---- C. Room to run ------------------------------------------------------
    add(dict(key="band_room", group="C_room", kind="num", bins=3,
             get=sidecol("band_room_up", "band_room_dn"),
             note="PROXY+CAUSAL: est. circuit band (smallest NSE band containing trailing-250d "
                  "max excursion, shift(1)) minus move-from-prev-close at cp. No band table on disk."))
    add(dict(key="room_52w", group="C_room", kind="num", bins=3,
             get=sidecol("room_52w_up", "room_52w_dn"),
             note="CAUSAL: headroom to the 52w extreme as of D-1 vs the decision price."))
    add(dict(key="gap_pct", group="C_room", kind="num", bins=3, get=signed("gap_pct"),
             note="CAUSAL: session open vs D-1 close, signed by side. Known at the open."))

    # ---- D. Context ----------------------------------------------------------
    add(dict(key="prev_deliv_pct", group="D_context", kind="num", bins=3,
             get=plain("prev_deliv_pct"),
             note="CAUSAL: delivery% of the PRIOR session (D-1 bhavcopy, published post-close D-1)."))
    add(dict(key="prev_deliv_ratio", group="D_context", kind="num", bins=3,
             get=plain("prev_deliv_ratio"),
             note="CAUSAL: D-1 delivery% / its own prior 20d median (baseline shift(1))."))
    add(dict(key="index_dir", group="D_context", kind="num", bins=3,
             get=lambda d, side, tag: d[f"ix_move_{tag}"] * (1.0 if side == "UP" else -1.0) * 100.0,
             note=f"CAUSAL: {INDEX_PROXY} (Nifty-50 ETF) open->cp return, signed by side. "
                  "data/index_ohlcv does not exist; ETF is the on-disk index proxy."))
    add(dict(key="peer_move", group="D_context", kind="num", bins=3,
             get=lambda d, side, tag: d[f"peer_move_{tag}"] * (1.0 if side == "UP" else -1.0),
             note="CAUSAL: leave-one-out mean open->cp move of same-sector universe names. "
                  "Sector map covers ~180 large/mid names only -- thin on this population."))
    add(dict(key="consec_up_prev", group="D_context", kind="num", bins=3,
             get=plain("consec_up_prev"),
             note="CAUSAL: consecutive up-closes ending D-1."))
    add(dict(key="repeat_mover_prev5", group="D_context", kind="bool",
             get=lambda d, side, tag: d["repeat_mover_prev5"] > 0,
             note="CAUSAL: |open->close| >= 8% on any of D-5..D-1."))

    # ---- E. Liquidity --------------------------------------------------------
    add(dict(key="adv_tier", group="E_liquidity", kind="cat",
             get=plain("adv_tier"),
             order=["V1_<1cr", "V2_1-5cr", "V3_5-25cr", "V4_25-100cr", "V5_>=100cr"],
             note="CAUSAL: ADV20 turnover, rolling-20 mean shift(1)."))
    add(dict(key="mcap_cr", group="E_liquidity", kind="num", bins=3,
             get=plain("mcap_cr"),
             note="STATIC SNAPSHOT (mild look-ahead on the classifier only): market_cap_cr from "
                  "nse_all.json -- the only free-float/size proxy on disk. Zeros dropped."))
    return S


def bin_series(x: pd.Series, spec: dict):
    """Pooled-era bin edges; returns (labelled series, ordered bin labels)."""
    kind = spec["kind"]
    if kind == "bool":
        b = x.astype("boolean").map({False: "no", True: "yes"}).astype(object)
        return b, ["no", "yes"]
    if kind == "cat":
        order = [o for o in spec["order"] if (x.astype(str) == o).sum() > 0]
        return x.astype(str), order
    v = pd.to_numeric(x, errors="coerce")
    if spec["key"] == "mcap_cr":
        v = v.replace(0, np.nan)
    nb = spec.get("bins", 3)
    valid = v.dropna()
    if valid.nunique() < nb:
        return pd.Series(np.nan, index=x.index), []
    try:
        q = valid.quantile(np.linspace(0, 1, nb + 1)).to_numpy()
        q[0], q[-1] = -np.inf, np.inf
        q = np.unique(q)
        if len(q) - 1 < 2:
            return pd.Series(np.nan, index=x.index), []
        labels = [f"Q{i+1}" for i in range(len(q) - 1)]
        return pd.cut(v, q, labels=labels, include_lowest=True).astype(object), labels
    except Exception:
        return pd.Series(np.nan, index=x.index), []


def monotone_ok(means: list[float]) -> bool:
    if len(means) < 2:
        return False
    if len(means) == 2:
        return True
    d = np.diff(means)
    total = means[-1] - means[0]
    if total == 0:
        return False
    sgn = np.sign(total)
    # allow one small reversal (<= MONO_TOL) but no large counter-move
    bad = d[np.sign(d) != sgn]
    return bool(len(bad) == 0 or (np.abs(bad).max() <= MONO_TOL and len(bad) <= 1))


def evaluate(pop: pd.DataFrame, specs: list[dict], side: str, cp: str) -> list[dict]:
    tag = cp.replace(":", "")
    verdicts = []
    for spec in specs:
        try:
            raw = spec["get"](pop, side, tag)
        except KeyError:
            continue
        b, labels = bin_series(raw, spec)
        if not labels:
            continue
        d = pop.assign(_bin=b)
        stats = {}
        for era in ("era_A", "era_B"):
            e = d[(d["era"] == era) & d["_bin"].notna()]
            per = {}
            for lab in labels:
                x = e.loc[e["_bin"] == lab, "fwd_gross_pct"]
                if len(x) == 0:
                    continue
                per[lab] = dict(n=int(len(x)), mean=float(x.mean()),
                                med=float(x.median()),
                                win=float(100 * (x > 0).mean()),
                                net=float(x.mean() - COST_PCT))
                emit("bin_stats", feature=spec["key"], group=spec["group"], side=side,
                     checkpoint=cp, era=era, bin=lab, n=per[lab]["n"],
                     mean_gross_pct=round(per[lab]["mean"], 4),
                     med_gross_pct=round(per[lab]["med"], 4),
                     win_pct=round(per[lab]["win"], 2),
                     mean_net_pct=round(per[lab]["net"], 4),
                     causal_note=spec["note"])
            stats[era] = per

        present = [l for l in labels if l in stats["era_A"] and l in stats["era_B"]]
        if len(present) < 2:
            continue
        lo, hi = present[0], present[-1]
        sA = stats["era_A"][hi]["mean"] - stats["era_A"][lo]["mean"]
        sB = stats["era_B"][hi]["mean"] - stats["era_B"][lo]["mean"]
        mA = monotone_ok([stats["era_A"][l]["mean"] for l in present])
        mB = monotone_ok([stats["era_B"][l]["mean"] for l in present])
        nmin = min(stats["era_A"][lo]["n"], stats["era_A"][hi]["n"],
                   stats["era_B"][lo]["n"], stats["era_B"][hi]["n"])
        same_sign = (sA > 0) == (sB > 0)
        big = min(abs(sA), abs(sB)) >= SPREAD_PP
        enough = nmin >= MIN_BIN_N
        noncausal = spec["key"].startswith("NONCAUSAL")
        passed = bool(same_sign and big and mA and mB and enough and not noncausal)
        reasons = []
        if not same_sign:
            reasons.append("sign_flip")
        if not big:
            reasons.append("spread<0.30pp")
        if not (mA and mB):
            reasons.append("non_monotone")
        if not enough:
            reasons.append(f"thin_n={nmin}")
        if noncausal:
            reasons.append("NOT_CAUSAL")

        # best bin by the weaker era, reported net of cost
        best_lab = max(present, key=lambda l: min(stats["era_A"][l]["mean"],
                                                  stats["era_B"][l]["mean"]))
        v = dict(feature=spec["key"], group=spec["group"], side=side, checkpoint=cp,
                 bins="|".join(present), n_min_extreme=nmin,
                 n_A=sum(stats["era_A"][l]["n"] for l in present),
                 n_B=sum(stats["era_B"][l]["n"] for l in present),
                 spread_A_pp=round(sA, 3), spread_B_pp=round(sB, 3),
                 min_abs_spread_pp=round(min(abs(sA), abs(sB)), 3),
                 mono_A=mA, mono_B=mB, same_sign=same_sign,
                 best_bin=best_lab,
                 best_bin_net_A=round(stats["era_A"][best_lab]["net"], 3),
                 best_bin_net_B=round(stats["era_B"][best_lab]["net"], 3),
                 best_bin_n_A=stats["era_A"][best_lab]["n"],
                 best_bin_n_B=stats["era_B"][best_lab]["n"],
                 verdict="PASS" if passed else "FAIL",
                 fail_reason=",".join(reasons), causal_note=spec["note"])
        emit("verdict", **v)
        verdicts.append(v)
    return verdicts


# ===================================================================== MAIN
def main() -> None:
    print("=" * 130)
    print("DIAGNOSTIC -- RUNNER vs FADER CONDITIONING  (Phase-1/2 exploration; NOT a candidate)")
    print(f"  fresh-pool wall: no signal >= {FRESH_POOL_START.date()}   "
          f"era_A <= {ERA_A_END.date()}   era_B <= {ERA_B_END.date()}")
    print(f"  population = anatomy screen: move >= {MOVE_THR:.0%} from open, cumvol >= {VOLX_THR}x "
          f"same-time-of-day 20d baseline, price on the VWAP side, CIRCUIT-LOCKED EXCLUDED")
    print(f"  entry = OPEN of the checkpoint bar, exit = session close, cost ref = {COST_PCT}% round trip")
    print(f"  PASS = same-sign spread BOTH eras AND min|spread| >= {SPREAD_PP}pp AND monotone-ish "
          f"BOTH eras AND n >= {MIN_BIN_N} per extreme bin per era")
    print("=" * 130)

    print("\n### 1. rebuild the anatomy universe (same module, same cached panel)")
    u = build_universe()
    print(f"  universe {u.shape}  symbols={u['symbol'].nunique()}  sessions={u['day'].nunique()}"
          f"  {u['day'].min().date()}..{u['day'].max().date()}")
    assert u["day"].max() < FRESH_POOL_START, "fresh-pool wall breached"

    months = sorted({d.strftime("%Y_%m") for d in u["day"].unique()})
    print(f"\n### 2. intraday move-structure panel ({len(months)} months)")
    struct, idxp = build_struct(months)
    print(f"  struct {struct.shape}   index-proxy sessions={len(idxp)} ({INDEX_PROXY})")

    print("\n### 3. daily context (all shift(1) / rolling-to-D-1)")
    syms = set(u["symbol"].unique())
    daily = build_daily_context(syms)
    deliv = build_delivery(syms)
    print(f"  daily {daily.shape}   delivery {deliv.shape}")

    print("\n### 4. event flags")
    flags = build_event_flags(u)
    cols = [c for c in flags.columns if c.startswith(("ev_", "no_event", "NONCAUSAL"))]
    print("  " + flags[cols].mean(numeric_only=False).apply(
        lambda x: f"{100*float(x):.2f}%").to_string().replace("\n", "\n  "))

    secmap = build_sector_map()
    nse = json.loads((ROOT / "nse_all.json").read_text())
    mcap = {_norm(r["symbol"]): float(r.get("market_cap_cr") or 0) for r in nse}
    print(f"  sector map: {len(secmap)} symbols   mcap map: {len(mcap)} symbols")

    print("\n### 5. assemble feature frame")
    df = assemble(u, struct, idxp, daily, deliv, flags, secmap, mcap)
    print(f"  assembled {df.shape}")

    specs = feature_specs()
    all_v = []
    for cp in CHECKPOINTS:
        for side in ("UP", "DN"):
            pop = screened(df, side, cp)
            nd = pop["day"].nunique()
            print("\n" + "=" * 130)
            print(f"POPULATION  side={side}  checkpoint={cp}   n={len(pop):,}  "
                  f"sessions={nd}  ({len(pop)/max(nd,1):.1f} names/day)")
            for era in ("era_A", "era_B"):
                e = pop[pop["era"] == era]
                if not len(e):
                    continue
                print(f"   {era}: n={len(e):,}  gross mean={e['fwd_gross_pct'].mean():+.3f}%  "
                      f"med={e['fwd_gross_pct'].median():+.3f}%  "
                      f"win={100*(e['fwd_gross_pct']>0).mean():.1f}%  "
                      f"net mean={e['fwd_net_pct'].mean():+.3f}%")
                emit("population", side=side, checkpoint=cp, era=era, n=len(e),
                     gross_mean_pct=round(float(e["fwd_gross_pct"].mean()), 4),
                     gross_med_pct=round(float(e["fwd_gross_pct"].median()), 4),
                     win_pct=round(100 * float((e["fwd_gross_pct"] > 0).mean()), 2),
                     net_mean_pct=round(float(e["fwd_net_pct"].mean()), 4))
            if len(pop) < 500:
                print("   (too thin to condition -- skipped)")
                continue
            v = evaluate(pop, specs, side, cp)
            all_v.extend(v)
            t = pd.DataFrame(v)
            show = ["group", "feature", "n_min_extreme", "spread_A_pp", "spread_B_pp",
                    "min_abs_spread_pp", "mono_A", "mono_B", "verdict", "fail_reason"]
            print(t[show].sort_values(["group", "feature"]).to_string(index=False))

    print("\n" + "=" * 130)
    print("PASS SHORTLIST (ranked by minimum-era spread)")
    print("=" * 130)
    V = pd.DataFrame(all_v)
    p = V[V["verdict"] == "PASS"].sort_values("min_abs_spread_pp", ascending=False)
    if len(p):
        print(p[["feature", "group", "side", "checkpoint", "min_abs_spread_pp",
                 "spread_A_pp", "spread_B_pp", "best_bin", "best_bin_net_A",
                 "best_bin_net_B", "best_bin_n_A", "best_bin_n_B"]].to_string(index=False))
    else:
        print("(NONE -- no single conditioning variable clears the bar in both eras)")

    print("\n  -- residual 'no identifiable event' bucket --")
    r = pd.DataFrame([x for x in _ROWS
                      if x.get("table") == "bin_stats" and x.get("feature") == "no_event"])
    if len(r):
        print("  " + r[["side", "checkpoint", "era", "bin", "n", "mean_gross_pct",
                        "med_gross_pct", "win_pct", "mean_net_pct"]]
              .to_string(index=False).replace("\n", "\n  "))

    SANITY.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_ROWS).to_csv(OUT_CSV, index=False)
    print(f"\n  wrote {OUT_CSV}  ({len(_ROWS)} rows)")


if __name__ == "__main__":
    main()
