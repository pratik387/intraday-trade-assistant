"""phase2_asm_gsm_stage_transition_signature.py — Stage 2 (Phase-2 empirical
signature) for `asm_gsm_stage_transition`
(specs/2026-05-08-sub-project-9-brief-asm_gsm_stage_transition.md).

PRE-REGISTERED CONSTRUCTION (brief §6 — implemented verbatim, NOT re-invented):
  Signal day D  = NSE/BSE surveillance snapshot date on which the stage
                  transition first appears (post-market circular, 17:30-18:30).
  Entry day     = T+1 = next trading day with a 5m bar for that symbol.
  Side          = SHORT (brief §6: both promotion and demotion converge on
                  T+1 fade-short).
  Entry price   = CLOSE of the 5m bar labelled 09:25 (i.e. the 09:25-09:30
                  candle, price stamped 09:30) — brief §6 step 3.
  Gates (brief §6 steps 2-3), applied as an explicit grid dimension:
    gate=none  : no gap / no confirmation filter (raw event drift)
    gate=gap   : direction-conditional 09:15 gap window
                   promotion side (entry, stage_up)  : -5% < gap < +1%
                   demotion  side (exit,  stage_down): -1% < gap < +5%
    gate=full  : gate=gap AND bearish confirmation candle
                   (bar0925.close < bar0925.open AND bar0925.close < bar0915.low)
  Exits         = fixed clock horizons. 10:30 is the brief's time stop
                  (bar 10:25 close); 10:00 / 11:30 / EOD are the swept
                  extensions (the brief leaves no other horizon open).
  Stage filter  = ASM I-III + GSM I-III (brief §6 step 1; Stage IV excluded —
                  0%/2% band, no intraday move; GSM stage VI and stage 0/IBC
                  likewise excluded from the primary population).
  Liquidity     = 20-day MEDIAN daily volume >= 50,000 shares (brief §3),
                  computed on data through D (no lookahead).

DIMENSIONS SWEPT THAT THE BRIEF LEAVES OPEN (declared):
  - exit horizon beyond the 10:30 time stop (10:00 / 11:30 / EOD)
  - ADV tier (brief pre-locks no cap segment; tier = within-date quintile of
    shifted 20d median turnover over the full clean_daily universe)
  - gate on/off decomposition (brief fixes the gates but does not report the
    raw un-gated drift, which Phase-2 needs to separate event from momentum)
  Everything else (side, entry clock, stage filter, liquidity floor,
  direction-conditional gap bands) is taken verbatim from the brief.

PHASE-2 STATISTIC (amendment A5-b, BINDING):
  This is a SINGLE-LEG CASH candidate. The era-consistency verdict rides on the
  ABSOLUTE statistic: mean NET return per event and NET PF. The relative
  statistic (delta vs matched baseline) is reported alongside as a DIAGNOSTIC
  only. RAW (no-fee) drift is reported too, per Phase-2 convention.

MATCHED BASELINE (two forms, both reported):
  base_uncond : same symbol pool, same era, same ADV tier, NON-event days,
                identical clock window, no gate.
  base_gated  : identical, but with the SAME gate condition applied on the
                non-event day. This is the decisive control — the bearish
                confirmation candle is itself a momentum filter, so an edge
                that survives only vs base_uncond is a momentum edge, not a
                surveillance edge.

CAUSALITY GUARDS:
  1. Entry is T+1 relative to the snapshot date D. Empirically verified in the
     Gate-B screen that NSE (circular date) and BSE (daily list) agree on D for
     97% of matched transitions, so D is the day the change becomes KNOWN
     (post-market). Entering at T+1 09:30 therefore cannot see the future.
     An `entry_offset=0` diagnostic (entering on D itself) is NOT run — it
     would be lookahead if D is the announcement date.
  2. ADV tier uses rolling(20).median().shift(1) turnover — through D-1 only.
  3. Liquidity gate uses rolling(20).median().shift(0) volume as of D (the
     signal day, already closed when the circular is published).
  4. No post-entry data in any filter. Exits are fixed clock closes.
  5. Signals HARD-capped at 2026-04-30. The fresh pool (2026-05-01+) is never
     used as a signal.
  6. Era split per amendment A5 is mandatory; era_B additionally sub-split
     pre/post 2025-10-01 (SEBI F&O regime break).

Data:
  Events   : data/asm_gsm_history/asm_gsm_events.parquet
  Daily    : cache/preaggregate/clean_daily_from5m.feather (CA-adjusted)
  Intraday : backtest-cache-download/monthly/YYYY_MM_5m_enriched.feather
  Tradabil.: nse_all.json (mis_enabled / mis_leverage / cap_segment — CURRENT
             snapshot, not point-in-time; labelled as such in the output)

Output: reports/sub9_sanity/_asm_gsm_stage_transition_phase2.csv
        (all grid cells, per-era, dead cells included)

Usage: .venv/Scripts/python tools/sub9_research/phase2_asm_gsm_stage_transition_signature.py
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# ---------------- LOCKED GRID (pre-registered — DO NOT EDIT after first run) ----
LOCKED_GRID = {
    "transition_types": ["entry", "exit", "stage_up", "stage_down"],
    "transition_groups": ["stage_move", "promotion", "demotion", "all"],
    "programs": ["ASM_LONGTERM", "ASM_SHORTTERM", "GSM", "ASM_LONGTERM_IBC", "ALL"],
    "gates": ["none", "gap", "full"],
    "horizons": ["h1000", "h1030", "h1130", "hEOD"],   # h1030 = brief time stop
    "adv_tiers": [1, 2, 3, 4, 5],
    "stages": ["I", "II", "III"],
    "eras": ["era_A", "era_B", "era_B1_pre_oct25", "era_B2_post_oct25", "pooled"],
}

SIGNAL_START = dt.date(2023, 1, 1)
SIGNAL_END = dt.date(2026, 4, 30)          # HARD cap — fresh pool untouched
ERA_A_END = dt.date(2024, 12, 31)
ERA_B_SPLIT = dt.date(2025, 10, 1)         # SEBI F&O regime break

STAGE_OK = ("I", "II", "III")              # brief §6 step 1
MIN_MEDIAN_VOL = 50_000                    # brief §3

# brief §6 step 2 — direction-conditional 09:15 gap windows
GAP_PROMO = (-5.0, 1.0)
GAP_DEMO = (-1.0, 5.0)

# Entry / exit clock (5m bars are labelled by bar START)
BAR_0915, BAR_0925 = "09:15", "09:25"
HORIZON_BARS = {"h1000": "09:55", "h1030": "10:25", "h1130": "11:25", "hEOD": None}
KEEP_TIMES = {"09:15", "09:20", "09:25", "09:55", "10:25", "11:25",
              "15:05", "15:10", "15:15", "15:20", "15:25"}

# ---- cost model: Zerodha MIS intraday equity, from tools/report_utils.py ----
BROKERAGE_RATE, BROKERAGE_CAP = 0.0003, 20.0
STT_RATE = 0.00025
EXCHANGE_RATE_NSE = 0.0000307
SEBI_RATE = IPFT_RATE = 0.000001
STAMP_DUTY_RATE = 0.00003
GST_RATE = 0.18
SLIPPAGE_PCT_ROUNDTRIP = 0.20              # 10bp/side, illiquid ASM names

_fee_frac = (
    2 * BROKERAGE_RATE                      # both legs (uncapped at slot size)
    + STT_RATE                              # sell leg only
    + 2 * EXCHANGE_RATE_NSE
    + 2 * (SEBI_RATE + IPFT_RATE)
    + STAMP_DUTY_RATE                       # buy leg only
)
_fee_frac += GST_RATE * (2 * BROKERAGE_RATE + 2 * EXCHANGE_RATE_NSE
                         + 2 * (SEBI_RATE + IPFT_RATE))
COST_PCT_ROUNDTRIP = _fee_frac * 100.0 + SLIPPAGE_PCT_ROUNDTRIP

EV_PARQUET = _REPO / "data" / "asm_gsm_history" / "asm_gsm_events.parquet"
CLEAN_DAILY = _REPO / "cache" / "preaggregate" / "clean_daily_from5m.feather"
MONTHLY = _REPO / "backtest-cache-download" / "monthly"
NSE_ALL = _REPO / "nse_all.json"
FNO_200 = _REPO / "assets" / "fno_liquid_200.csv"
OUT_CSV = _REPO / "reports" / "sub9_sanity" / "_asm_gsm_stage_transition_phase2.csv"
# intermediate (rebuildable) 5m morning-bar panel — not a deliverable
PANEL_CACHE = MONTHLY.parent / "_asm_gsm_phase2_morning_panel.parquet"


# =============================================================================
# helpers
# =============================================================================
def era_label(d: pd.Series) -> pd.Series:
    out = pd.Series("other", index=d.index, dtype=object)
    out[d.dt.date <= ERA_A_END] = "era_A"
    out[d.dt.date > ERA_A_END] = "era_B"
    return out


def era_b_sub(d: pd.Series) -> pd.Series:
    out = pd.Series("", index=d.index, dtype=object)
    m = d.dt.date > ERA_A_END
    out[m & (d.dt.date < ERA_B_SPLIT)] = "era_B1_pre_oct25"
    out[m & (d.dt.date >= ERA_B_SPLIT)] = "era_B2_post_oct25"
    return out


def net_pf(x: np.ndarray) -> float:
    pos = x[x > 0].sum()
    neg = -x[x < 0].sum()
    if neg <= 0:
        return float("inf") if pos > 0 else float("nan")
    return float(pos / neg)


# =============================================================================
# 1. Event construction (brief §6 step 1 + §3 universe)
# =============================================================================
def build_events(dly: pd.DataFrame) -> pd.DataFrame:
    ev = pd.read_parquet(EV_PARQUET)
    ev["date"] = pd.to_datetime(ev["date"])
    print(f"[events] raw parquet {ev.shape}  {ev.date.min().date()} -> {ev.date.max().date()}")

    ev = ev[ev.transition_type != "no_change"].copy()
    print(f"[events] transitions only: {len(ev)}")

    # effective stage: new stage for entry/stage moves, prior stage for exits
    ev["eff_stage"] = ev["stage"].where(ev.transition_type != "exit", ev["prev_stage"])
    ev = ev[ev.eff_stage.isin(STAGE_OK)]
    print(f"[events] stage filter I-III: {len(ev)}")

    ev = ev[(ev.date.dt.date >= SIGNAL_START) & (ev.date.dt.date <= SIGNAL_END)]
    print(f"[events] signal window {SIGNAL_START}..{SIGNAL_END}: {len(ev)}")

    nse_syms = set(dly.symbol.unique())
    ev = ev[ev.symbol.isin(nse_syms)]
    print(f"[events] symbol resolves to NSE ticker with 5m-derived daily: {len(ev)}")

    # dedup per (symbol, signal date): keep most severe transition
    sev = {"stage_up": 0, "stage_down": 1, "entry": 2, "exit": 3}
    ev["_sev"] = ev.transition_type.map(sev)
    ev = (ev.sort_values(["symbol", "date", "_sev"])
            .drop_duplicates(["symbol", "date"], keep="first"))
    print(f"[events] dedup (symbol, signal_date): {len(ev)}")
    return ev


def attach_entry_day(ev: pd.DataFrame, dly: pd.DataFrame) -> pd.DataFrame:
    """Map signal date D -> the symbol's next trading bar (T+1)."""
    base = dly[["symbol", "date", "bar_i", "medvol20", "advtier", "close"]].copy()
    base = base.rename(columns={"date": "sig_bar_date", "close": "sig_close"})
    j = pd.merge_asof(
        ev.sort_values("date"),
        base.sort_values("sig_bar_date"),
        left_on="date", right_on="sig_bar_date", by="symbol",
        direction="backward", tolerance=pd.Timedelta("7D"))
    j = j[j.bar_i.notna()].copy()
    j["entry_bar_i"] = j["bar_i"] + 1
    nxt = dly[["symbol", "bar_i", "date"]].rename(
        columns={"bar_i": "entry_bar_i", "date": "entry_date"})
    j = j.merge(nxt, on=["symbol", "entry_bar_i"], how="left")
    j = j[j.entry_date.notna()]
    print(f"[events] T+1 entry bar exists: {len(j)}")
    j = j[j.medvol20 >= MIN_MEDIAN_VOL]
    print(f"[events] 20d median vol >= {MIN_MEDIAN_VOL:,}: {len(j)}")
    j = j.drop_duplicates(["symbol", "entry_date"])
    print(f"[events] dedup (symbol, entry_date)  << TRADEABLE: {len(j)}")
    return j


# =============================================================================
# 2. Intraday panel
# =============================================================================
def load_intraday(symbols: set[str]) -> pd.DataFrame:
    """Load only the clock bars we need. Vectorised HHMM int (strftime is 100x
    slower); cached to scratch so re-runs of the grid are cheap."""
    keep_int = {int(t.replace(":", "")) for t in KEEP_TIMES}
    files = sorted(MONTHLY.glob("*_5m_enriched.feather"))
    parts = []
    for f in files:
        ym = f.name[:7]
        if ym < "2023_01" or ym > "2026_04":
            continue
        df = pd.read_feather(f, columns=["date", "symbol", "open", "high", "low", "close"])
        df = df[df.symbol.isin(symbols)]
        if df.empty:
            continue
        d = pd.to_datetime(df["date"])
        if getattr(d.dt, "tz", None) is not None:
            d = d.dt.tz_localize(None)
        hm = (d.dt.hour * 100 + d.dt.minute).astype("int32")
        m = hm.isin(keep_int)
        df = df[m].copy()
        df["hm"] = hm[m].values
        df["d"] = d[m].dt.normalize().values
        parts.append(df[["symbol", "d", "hm", "open", "high", "low", "close"]])
        print(f"  [5m] {f.name}: kept {len(parts[-1]):,}", flush=True)
    return pd.concat(parts, ignore_index=True)


def build_day_panel(intr: pd.DataFrame, dly: pd.DataFrame) -> pd.DataFrame:
    """One row per (symbol, day) with entry/exit prices + gate inputs."""
    intr = intr.drop_duplicates(["symbol", "d", "hm"])
    piv = intr.set_index(["symbol", "d", "hm"])[["open", "high", "low", "close"]].unstack("hm")
    piv.columns = [f"{a}_{b:04d}" for a, b in piv.columns]
    piv = piv.reset_index()

    p = pd.DataFrame({"symbol": piv.symbol, "d": piv.d})
    p["o0915"] = piv.get("open_0915")
    p["l0915"] = piv.get("low_0915")
    p["o0925"] = piv.get("open_0925")
    p["entry"] = piv.get("close_0925")
    p["x_h1000"] = piv.get("close_0955")
    p["x_h1030"] = piv.get("close_1025")
    p["x_h1130"] = piv.get("close_1125")
    eod = None
    for t in ["1525", "1520", "1515", "1510", "1505"]:
        c = piv.get(f"close_{t}")
        eod = c if eod is None else (eod if c is None else eod.fillna(c))
    p["x_hEOD"] = eod

    # prior-day close (pdc) from CA-adjusted daily — brief §6 step 2
    pdc = dly[["symbol", "date", "close"]].sort_values(["symbol", "date"]).copy()
    pdc["pdc"] = pdc.groupby("symbol")["close"].shift(1)
    p = p.merge(pdc[["symbol", "date", "pdc"]].rename(columns={"date": "d"}),
                on=["symbol", "d"], how="left")

    p["gap_pct"] = (p.o0915 - p.pdc) / p.pdc * 100.0
    p["bearish"] = (p.entry < p.o0925) & (p.entry < p.l0915)

    for h in HORIZON_BARS:
        p[f"ret_{h}"] = (p["entry"] - p[f"x_{h}"]) / p["entry"] * 100.0  # SHORT, %
    return p


def gate_mask(p: pd.DataFrame, gate: str, promo: pd.Series) -> pd.Series:
    if gate == "none":
        return pd.Series(True, index=p.index)
    lo = np.where(promo, GAP_PROMO[0], GAP_DEMO[0])
    hi = np.where(promo, GAP_PROMO[1], GAP_DEMO[1])
    m = (p.gap_pct > lo) & (p.gap_pct < hi)
    if gate == "gap":
        return m.fillna(False)
    return (m & p.bearish).fillna(False)


# =============================================================================
# 3. main
# =============================================================================
def main() -> None:
    print("=" * 100)
    print("PHASE-2 SIGNATURE — asm_gsm_stage_transition")
    print(f"cost model: round-trip fees {(_fee_frac*100):.4f}% + slippage "
          f"{SLIPPAGE_PCT_ROUNDTRIP:.2f}% = {COST_PCT_ROUNDTRIP:.4f}%")
    print("=" * 100)

    dly = pd.read_feather(CLEAN_DAILY)
    dly["date"] = pd.to_datetime(dly["date"])
    dly = dly.sort_values(["symbol", "date"]).reset_index(drop=True)
    dly["bar_i"] = dly.groupby("symbol").cumcount()
    dly["medvol20"] = dly.groupby("symbol")["volume"].transform(
        lambda s: s.rolling(20, min_periods=10).median())
    turn = dly["close"] * dly["volume"]
    dly["advturn20"] = turn.groupby(dly["symbol"]).transform(
        lambda s: s.rolling(20, min_periods=10).median().shift(1))
    dly["advtier"] = (dly.groupby("date")["advturn20"]
                        .transform(lambda s: pd.qcut(s.rank(method="first"), 5,
                                                     labels=[1, 2, 3, 4, 5])
                                   if s.notna().sum() >= 20 else np.nan))
    dly["advtier"] = pd.to_numeric(dly["advtier"], errors="coerce")

    ev = build_events(dly)
    ev = attach_entry_day(ev, dly)

    ev["era"] = era_label(ev["date"])
    ev["era_sub"] = era_b_sub(ev["date"])
    print("\n[events] per era:", ev.era.value_counts().to_dict())
    print("[events] per era_sub:", ev[ev.era_sub != ""].era_sub.value_counts().to_dict())
    print("[events] per transition_type:", ev.transition_type.value_counts().to_dict())
    print("[events] per program:", ev.surveillance_program.value_counts().to_dict())

    syms = set(ev.symbol.unique())
    print(f"\n[5m] loading intraday for {len(syms)} symbols ...", flush=True)
    if PANEL_CACHE.exists():
        panel = pd.read_parquet(PANEL_CACHE)
        print(f"[panel] loaded from cache {PANEL_CACHE}", flush=True)
        if not syms.issubset(set(panel.symbol.unique())):
            print("[panel] cache symbol set stale -> rebuilding", flush=True)
            panel = build_day_panel(load_intraday(syms), dly)
            PANEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
            panel.to_parquet(PANEL_CACHE, index=False)
    else:
        panel = build_day_panel(load_intraday(syms), dly)
        PANEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(PANEL_CACHE, index=False)
    print(f"[panel] (symbol, day) rows: {len(panel):,}", flush=True)

    # attach panel to events
    e = ev.merge(panel, left_on=["symbol", "entry_date"], right_on=["symbol", "d"],
                 how="left", suffixes=("", "_p"))
    n0 = len(e)
    e = e[e.entry.notna() & e.x_h1030.notna()]
    print(f"[panel] events with a usable 09:25 entry bar + 10:25 exit bar: {len(e)}/{n0}")

    e["promo"] = e.transition_type.isin(["entry", "stage_up"])
    e["is_event"] = True

    # -------- baseline panel: same symbols, NON-event days --------
    ev_keys = set(zip(e.symbol, e.entry_date))
    panel["is_event"] = [(s, d) in ev_keys for s, d in zip(panel.symbol, panel.d)]
    base = panel[~panel.is_event].copy()
    base = base[base.entry.notna() & base.x_h1030.notna()]
    base = base.merge(dly[["symbol", "date", "advtier", "medvol20"]]
                      .rename(columns={"date": "d"}), on=["symbol", "d"], how="left")
    base = base[base.medvol20 >= MIN_MEDIAN_VOL]
    base = base[(base.d.dt.date >= SIGNAL_START) & (base.d.dt.date <= SIGNAL_END)]
    base["era"] = era_label(base["d"])
    print(f"[baseline] non-event (symbol, day) rows: {len(base):,}  "
          f"{base.era.value_counts().to_dict()}")

    rows = []

    def emit(dim_kind, ttype, prog, gate, hz, tier, stage, era, sub, sel_ev, sel_bs):
        if len(sel_ev) == 0:
            rows.append(dict(dim_kind=dim_kind, transition=ttype, program=prog,
                             gate=gate, horizon=hz, adv_tier=tier, stage=stage,
                             era=era, n=0))
            return
        r = sel_ev[f"ret_{hz}"].dropna().values
        if len(r) == 0:
            rows.append(dict(dim_kind=dim_kind, transition=ttype, program=prog,
                             gate=gate, horizon=hz, adv_tier=tier, stage=stage,
                             era=era, n=0))
            return
        net = r - COST_PCT_ROUNDTRIP
        b_un = sel_bs[f"ret_{hz}"].dropna().values
        rows.append(dict(
            dim_kind=dim_kind, transition=ttype, program=prog, gate=gate,
            horizon=hz, adv_tier=tier, stage=stage, era=era,
            n=len(r), n_sym=sel_ev.symbol.nunique(),
            # ---- ABSOLUTE statistic (A5-b: DECISIVE) ----
            raw_mean_pct=float(np.mean(r)),
            raw_median_pct=float(np.median(r)),
            net_mean_pct=float(np.mean(net)),
            net_pf=net_pf(net),
            net_wr=float((net > 0).mean()),
            net_sum_pct=float(np.sum(net)),
            t_stat=float(np.mean(r) / (np.std(r, ddof=1) / np.sqrt(len(r))))
            if len(r) > 1 and np.std(r, ddof=1) > 0 else np.nan,
            max_abs_contrib=float(np.max(np.abs(net)) / np.sum(np.abs(net)))
            if np.sum(np.abs(net)) > 0 else np.nan,
            # ---- RELATIVE statistic (DIAGNOSTIC only) ----
            base_uncond_mean_pct=float(np.mean(b_un)) if len(b_un) else np.nan,
            base_uncond_n=len(b_un),
            delta_vs_uncond=float(np.mean(r) - np.mean(b_un)) if len(b_un) else np.nan,
        ))

    _base_cache: dict = {}

    def slice_baseline(gate, era, tier):
        key = (gate, era, tier)
        if key in _base_cache:
            return _base_cache[key]
        b = base
        if era in ("era_A", "era_B"):
            b = b[b.era == era]
        elif era == "era_B1_pre_oct25":
            b = b[(b.era == "era_B") & (b.d.dt.date < ERA_B_SPLIT)]
        elif era == "era_B2_post_oct25":
            b = b[(b.era == "era_B") & (b.d.dt.date >= ERA_B_SPLIT)]
        if tier != "ALL":
            b = b[b.advtier == tier]
        if gate != "none":
            # gate-matched baseline uses the DEMOTION band as the neutral case
            # plus the same bearish confirmation; reported as base_gated below
            m = (b.gap_pct > GAP_DEMO[0]) & (b.gap_pct < GAP_DEMO[1])
            if gate == "full":
                m = m & b.bearish
            b = b[m.fillna(False)]
        _base_cache[key] = b
        return b

    def slice_events(ttype, prog, gate, tier, stage, era):
        s = e
        if ttype in LOCKED_GRID["transition_types"]:
            s = s[s.transition_type == ttype]
        elif ttype == "stage_move":
            s = s[s.transition_type.isin(["stage_up", "stage_down"])]
        elif ttype == "promotion":
            s = s[s.promo]
        elif ttype == "demotion":
            s = s[~s.promo]
        if prog != "ALL":
            s = s[s.surveillance_program == prog]
        if tier != "ALL":
            s = s[s.advtier == tier]
        if stage != "ALL":
            s = s[s.eff_stage == stage]
        if era in ("era_A", "era_B"):
            s = s[s.era == era]
        elif era in ("era_B1_pre_oct25", "era_B2_post_oct25"):
            s = s[s.era_sub == era]
        if gate != "none":
            s = s[gate_mask(s, gate, s.promo)]
        return s

    eras = LOCKED_GRID["eras"]
    # ---- core grid: transition x program x gate x horizon x era ----
    for ttype in LOCKED_GRID["transition_types"] + LOCKED_GRID["transition_groups"]:
        for prog in LOCKED_GRID["programs"]:
            for gate in LOCKED_GRID["gates"]:
                for hz in LOCKED_GRID["horizons"]:
                    for era in eras:
                        emit("core", ttype, prog, gate, hz, "ALL", "ALL", era, "",
                             slice_events(ttype, prog, gate, "ALL", "ALL", era),
                             slice_baseline(gate, era, "ALL"))
    # ---- ADV tier slice (program=ALL) ----
    for ttype in LOCKED_GRID["transition_types"] + LOCKED_GRID["transition_groups"]:
        for gate in LOCKED_GRID["gates"]:
            for hz in LOCKED_GRID["horizons"]:
                for tier in LOCKED_GRID["adv_tiers"]:
                    for era in eras:
                        emit("adv_tier", ttype, "ALL", gate, hz, tier, "ALL", era, "",
                             slice_events(ttype, "ALL", gate, tier, "ALL", era),
                             slice_baseline(gate, era, tier))
    # ---- salience (stage severity) slice ----
    for ttype in LOCKED_GRID["transition_types"] + LOCKED_GRID["transition_groups"]:
        for gate in LOCKED_GRID["gates"]:
            for hz in LOCKED_GRID["horizons"]:
                for stage in LOCKED_GRID["stages"]:
                    for era in eras:
                        emit("stage", ttype, "ALL", gate, hz, "ALL", stage, era, "",
                             slice_events(ttype, "ALL", gate, "ALL", stage, era),
                             slice_baseline(gate, era, "ALL"))

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\n[out] {len(out)} cells -> {OUT_CSV}")

    # =========================================================================
    # console report
    # =========================================================================
    def show(df, title, cols=None):
        cols = cols or ["transition", "program", "gate", "horizon", "era", "n",
                        "raw_mean_pct", "net_mean_pct", "net_pf", "net_wr",
                        "delta_vs_uncond", "t_stat"]
        print(f"\n--- {title} ---")
        if df.empty:
            print("  (empty)")
            return
        with pd.option_context("display.width", 200, "display.max_columns", 40):
            print(df[cols].round(4).to_string(index=False))

    core = out[out.dim_kind == "core"]
    print("\n" + "=" * 100)
    print("A. BRIEF-FAITHFUL CELL (program=ALL, horizon=h1030 time stop) — all gates")
    print("=" * 100)
    show(core[(core.program == "ALL") & (core.horizon == "h1030")
              & (core.transition.isin(["promotion", "demotion", "stage_move", "all"]))
              & (core.era.isin(["era_A", "era_B", "pooled"]))]
         .sort_values(["transition", "gate", "era"]),
         "brief time-stop 10:30")

    print("\n" + "=" * 100)
    print("B. ERA CONSISTENCY on the ABSOLUTE statistic (A5-b) — gate=full, h1030")
    print("=" * 100)
    show(core[(core.gate == "full") & (core.horizon == "h1030")
              & (core.program.isin(["ALL", "ASM_LONGTERM", "ASM_SHORTTERM", "GSM"]))]
         .sort_values(["transition", "program", "era"]),
         "gate=full / h1030 / per era + Oct-25 sub-split")

    print("\n" + "=" * 100)
    print("C. HORIZON SWEEP (transition=all, program=ALL)")
    print("=" * 100)
    show(core[(core.program == "ALL") & (core.transition == "all")]
         .sort_values(["gate", "horizon", "era"]), "horizon sweep")

    # ---- falsifier readouts ----
    print("\n" + "=" * 100)
    print("FALSIFIER READOUTS (brief §9)")
    print("=" * 100)

    # F1 — demotion inverts to LONG
    dm = core[(core.transition == "demotion") & (core.program == "ALL")
              & (core.horizon == "h1030")]
    print("\nF1. Demotion side inverts (LONG bounce)?  "
          "LONG net PF = 1/short-PF proxy; report short net mean per era:")
    print(dm[["gate", "era", "n", "raw_mean_pct", "net_mean_pct", "net_pf"]]
          .round(4).to_string(index=False))

    # F2 — overlap with circuit_t1_fade_short (proxy: signal-day near-circuit move)
    d2 = dly[["symbol", "date", "close"]].sort_values(["symbol", "date"]).copy()
    d2["ret_d"] = d2.groupby("symbol")["close"].pct_change() * 100
    ov = e.merge(d2.rename(columns={"date": "date_sig"}),
                 left_on=["symbol", "date"], right_on=["symbol", "date_sig"], how="left")
    for band in (4.5, 9.5, 19.5):
        frac = (ov.ret_d.abs() >= band).mean()
        print(f"F2. signal-day |close-to-close| >= {band}% (circuit-hit proxy): {frac:.1%}")

    # F3 — circular feed reliability
    raw = pd.read_parquet(EV_PARQUET, columns=["date", "exchange"])
    raw["date"] = pd.to_datetime(raw["date"])
    tdays = dly.date.nunique()
    for exch in ("NSE", "BSE"):
        nd = raw[raw.exchange == exch].date.nunique()
        print(f"F3. {exch} snapshot days {nd} vs {tdays} trading days in daily panel "
              f"-> missing {100*(1-nd/tdays):.1f}%")

    # F4/F5 — decay: per-year absolute stat
    e["yr"] = e.entry_date.dt.year
    g = gate_mask(e, "full", e.promo)
    for label, sub in (("gate=none", e), ("gate=full", e[g])):
        t = sub.groupby("yr")["ret_h1030"].agg(["count", "mean"])
        t["net_mean"] = t["mean"] - COST_PCT_ROUNDTRIP
        t["net_pf"] = sub.groupby("yr")["ret_h1030"].apply(
            lambda s: net_pf(s.dropna().values - COST_PCT_ROUNDTRIP))
        print(f"\nF4/F5. per-year absolute stat, h1030, {label}:")
        print(t.round(4).to_string())

    # ---- SHORT-LEG TRADABILITY (lesson #28 rule 3) ----
    print("\n" + "=" * 100)
    print("SHORT-LEG TRADABILITY")
    print("=" * 100)
    nse_all = json.load(open(NSE_ALL))
    mis = {r["symbol"].replace(".NS", ""): r for r in nse_all}
    e["mis_enabled"] = e.symbol.map(lambda s: mis.get(s, {}).get("mis_enabled"))
    e["mis_leverage"] = e.symbol.map(lambda s: mis.get(s, {}).get("mis_leverage"))
    e["cap_segment"] = e.symbol.map(lambda s: mis.get(s, {}).get("cap_segment"))
    print(f"symbols present in nse_all.json snapshot: "
          f"{e.symbol.isin(mis).mean():.1%} of events")
    print("mis_enabled (CURRENT snapshot, not point-in-time):",
          e.mis_enabled.value_counts(dropna=False).to_dict())
    print("mis_leverage:", e.mis_leverage.value_counts(dropna=False).to_dict())
    print("cap_segment:", e.cap_segment.value_counts(dropna=False).to_dict())

    # rule-based settlement classification (NSE/BSE surveillance framework)
    #   GSM Stage >= II          -> trade-for-trade, NO intraday square-off
    #   ASM_LONGTERM Stage III   -> trade-for-trade in practice
    #   ASM stages I/II          -> rolling settlement, 100% margin, 1x only
    def settle(row):
        if row.surveillance_program == "GSM":
            return "T2T_no_intraday" if row.eff_stage in ("II", "III") else "rolling_100pct_margin"
        if row.surveillance_program.startswith("ASM") and row.eff_stage == "III":
            return "T2T_no_intraday"
        return "rolling_100pct_margin"

    e["settlement"] = e.apply(settle, axis=1)
    print("\nrule-based settlement class (NSE/BSE surveillance framework):")
    print(e.settlement.value_counts().to_dict())
    shortable = e[(e.settlement == "rolling_100pct_margin") & (e.mis_enabled == True)]
    print(f"SHORTABLE subset (rolling settlement AND mis_enabled snapshot): "
          f"{len(shortable)}/{len(e)} = {len(shortable)/len(e):.1%}")
    print("  per era:", shortable.era.value_counts().to_dict())

    if FNO_200.exists():
        fno = set(pd.read_csv(FNO_200).iloc[:, 0].astype(str).str.replace(".NS", "", regex=False))
        print(f"F&O-present (CURRENT fno_liquid_200 snapshot): "
              f"{e.symbol.isin(fno).mean():.1%} of events")

    print("\nSHORTABLE-subset absolute stat, h1030 (per era, per gate):")
    for gate in ("none", "gap", "full"):
        s = shortable[gate_mask(shortable, gate, shortable.promo)]
        for era in ("era_A", "era_B", "pooled"):
            ss = s if era == "pooled" else s[s.era == era]
            r = ss["ret_h1030"].dropna().values
            if len(r) == 0:
                print(f"  gate={gate:<5s} {era:<7s} n=0")
                continue
            net = r - COST_PCT_ROUNDTRIP
            print(f"  gate={gate:<5s} {era:<7s} n={len(r):<5d} raw={np.mean(r):+.4f}% "
                  f"net={np.mean(net):+.4f}% net_pf={net_pf(net):.4f} "
                  f"wr={100*(net>0).mean():.1f}%")

    e.to_csv(OUT_CSV.with_name("_asm_gsm_stage_transition_phase2_trades.csv"), index=False)
    print(f"\n[out] per-event rows -> "
          f"{OUT_CSV.with_name('_asm_gsm_stage_transition_phase2_trades.csv')}")


if __name__ == "__main__":
    main()
