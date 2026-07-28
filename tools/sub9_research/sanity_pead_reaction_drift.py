"""Stage 4 sanity: pead_reaction_drift — post-earnings reaction-day-surprise drift (both legs).

Brief:        specs/2026-07-27-brief-pead_reaction_drift.md (grid pre-registered, Section 5)
Phase 2:      tools/sub9_research/phase2_pead_reaction_drift_signature.py (winning family:
              volconf proxy x 1-day reaction window x pct5 threshold x entry lag 1)
Lifecycle:    docs/setup_lifecycle.md Stage 4 (6-failure-mode checklist below)
Precedent:    tools/sub9_research/sanity_close_dn_overnight_long.py (multiday CNC per-event
              ledger conventions: one row per event, Rs 1L notional qty, extra research
              columns carried alongside the canonical schema columns)

WINDOW DISCIPLINE (hard constraint, asserted at runtime)
    Discovery ONLY: REACTION dates 2023-01-01 -> 2024-12-31. The script RAISES if any
    emitted candidate's reaction_date falls outside this window. Forward exit legs of
    late-Dec-2024 events extend into early 2025 (exits only — allowed).

ANTI-BIAS GUARDS (Lesson #5 6-failure-mode checklist — each documented explicitly)

  1. Intraday-aggregate look-ahead: N/A by construction — the signal uses only DAILY
     closes/volumes of sessions <= the reaction day. No day_high/day_low/day_vwap is used
     as a filter at signal time. Daily highs/lows appear ONLY in mfe_pct/mae_pct, which
     are descriptive excursion columns computed on bars in the post-entry window
     (never used to select or filter events).
  2. Volume baseline excludes current day: ADV20 (rupee) and AvgVol20 (shares) are
     rolling(20).mean().shift(1) per symbol — the reaction day is NOT in its own baseline.
  3. Mode B entry walk: entry is the NEXT session's OPEN (reaction+1 open). The
     MFE/MAE walk starts at that same entry bar — its full intra-bar range happens
     AFTER the open entry (lifecycle Stage 4 rule 3, open-entry case). No same-day fills.
  4. Same-bar exit ambiguity: no SL/target is modeled at Stage 4 (pure time-stop hold),
     so no favorable-outcome selection is possible. SAME-BAR-PESSIMISM NOTE: mfe_pct /
     mae_pct include the entry bar's high/low; because entry is at that bar's open, the
     range is genuinely post-entry, but intra-bar ORDERING of high vs low is unknown —
     any later stage that turns MAE into an SL rule must re-derive stop-hits
     pessimistically from intraday bars, not from these daily columns.
  5. Cell-locked filters at signal time: every filter is a LOCKED_* constant below,
     written BEFORE the first Discovery run. No post-hoc tuning.
  6. Output validated against tools/methodology/sanity_csv_schema.validate() —
     see SCHEMA NOTE below. Reproducibility floor (sanity vs OCI +/- Rs 10) becomes
     auditable once OCI structure code exists (Stage 7).
  7. No full-window aggregates at signal time: the percentile threshold is CAUSAL —
     for an event with reaction date d it is computed ONLY from volume-confirmed events
     with reaction_date in [d - 60 calendar days, d), i.e. STRICTLY BEFORE d. This
     replaces Phase 2's pooled-cohort quantile, which would be look-ahead in a trading
     rule (Phase 2 explicitly flagged that simplification). Events with fewer than
     LOCKED_MIN_TRAILING_EVENTS trailing events are skipped, not scored.
     (The only pooled-over-cohort column is adv_tier — a DESCRIPTIVE reporting bucket
     per B1/Phase-2 convention, never used to select events.)

SCHEMA NOTE (sanity_csv_schema is intraday-single-exit — saying so, per Stage 4 rule)
    tools/methodology/sanity_csv_schema.py models exactly one exit per row
    (entry_price/exit_price/pnl_pct/exit_reason/same_bar). It has NO multiday
    multi-horizon layer. Following the close_dn_overnight_long precedent, this ledger
    fills the canonical columns with a CANONICAL H=10 exit (exit_price = close of the
    10th in-market session, exit_reason='time_stop', same_bar=False, side-aware
    pnl_pct) so the validator's five bug-checks all run, and carries the multiday
    research columns (fwd_h3/5/7/10, mfe/mae, tradability flags, ...) as extra
    columns, which the validator passes through. Rows lacking a full 10-session
    forward path (delisting/suspension near data edge) are DROPPED and counted —
    a small survivorship caveat, printed at runtime.

UNIVERSE GATE (LOCKED, applied on the ENTRY date)
    ProductionUniverseGate(accepted_caps={large_cap, mid_cap, small_cap, unknown},
    require_mis=False, min_trading_days_required=0, min_daily_avg_volume=0).
    Rationale: CNC multiday setup — no MIS requirement (close_dn CNC precedent);
    micro_cap excluded exactly as in the close_dn production builder; legacy
    coverage/volume filters zeroed per Lesson #17 (liquidity is enforced by the
    LOCKED ADV20 >= Rs 20L floor instead).

TRADABILITY SOURCES (best available on disk — both are CURRENT snapshots, i.e.
    2026-07 membership applied to 2023-24 events; NOT point-in-time. Stated plainly:
    F&O/MTF membership drift over 2023-2026 makes these flags approximate.)
      tradable_fo  = symbol in assets/fno_liquid_200.csv (production
                     services.symbol_metadata.in_fno_liquid_200 list, 153 symbols)
                     OR category == 'fo' in the MTF snapshot (210 symbols)
      tradable_mtf = symbol in data/mtf_universe/approved_mtf_securities_latest.json
                     with category != 'etf'

FORWARD-RETURN CONVENTION (matches Phase 2)
    H sessions IN-MARKET: fwd_hH_pct = (close[entry_pos + H - 1] / entry_open - 1) * 100.
    RAW price %, LONG convention for BOTH legs (short-leg drift is favorable when
    negative), NO fees, NO slippage. mfe_pct / mae_pct are LEG-AWARE over the h10
    window (mfe_pct >= 0 favorable-in-trade-direction, mae_pct <= 0 adverse).

Run:
    .venv/Scripts/python tools/sub9_research/sanity_pead_reaction_drift.py
Output:
    reports/sub9_sanity/_pead_reaction_drift_trades_discovery.csv (both legs, `leg` col)
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# LOCKED FILTERS — written before the first Discovery run. DO NOT TUNE.
# (From the Phase-2 winning family: volconf x window=1 x pct5 x lag=1.)
# ---------------------------------------------------------------------------

LOCKED_SURPRISE_PROXY = "volconf"     # reaction-day share volume >= mult x AvgVol20
LOCKED_VOLCONF_MULT = 2.0             # x trailing AvgVol20 (shifted, excludes reaction day)
LOCKED_REACTION_WINDOW = 1            # trading day (reaction day only)
LOCKED_PCT_LONG = 0.95                # LONG: reaction ret >= causal 95th pct of trailing xsec
LOCKED_PCT_SHORT = 0.05               # SHORT: reaction ret <= causal 5th pct of trailing xsec
LOCKED_TRAILING_CAL_DAYS = 60         # trailing event cross-section window (calendar days)
LOCKED_MIN_TRAILING_EVENTS = 30       # min trailing events to define the causal threshold
LOCKED_ENTRY_LAG = 1                  # Mode B: entry at open of reaction+1 session
LOCKED_ADV_FLOOR_INR = 2_000_000      # ADV20 >= Rs 20L (rupee turnover, shifted)
LOCKED_HOLDS = (3, 5, 7, 10)          # sessions in-market; canonical schema exit = H10
LOCKED_UNIVERSE_CAPS = frozenset({"large_cap", "mid_cap", "small_cap", "unknown"})
LOCKED_REQUIRE_MIS = False            # CNC delivery — no MIS gate (close_dn precedent)

DISCOVERY_START = pd.Timestamp("2023-01-01")
DISCOVERY_END = pd.Timestamp("2024-12-31")

POSITION_NOTIONAL_INR = 100_000       # Rs 1L per trade (close_dn ledger convention)
N_ADV_TIERS = 5                       # descriptive tiers, tier1 = most illiquid (B1 convention)
REACTION_TOLERANCE_DAYS = 5           # next-day reaction must trade within 5 cal days
EVENT_LOAD_PAD_DAYS = 12              # load announce dates slightly before window start
H_MAX = max(LOCKED_HOLDS)

DAILY_FEATHER = REPO_ROOT / "cache" / "preaggregate" / "clean_daily_from5m.feather"
EVENTS_PARQUET = REPO_ROOT / "data" / "earnings_calendar" / "earnings_events.parquet"
MTF_SNAPSHOT = REPO_ROOT / "data" / "mtf_universe" / "approved_mtf_securities_latest.json"
OUT_CSV = REPO_ROOT / "reports" / "sub9_sanity" / "_pead_reaction_drift_trades_discovery.csv"

SETUP_NAME = "pead_reaction_drift"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data loading (shape/coverage pre-check prints; reuses Phase-2 conventions)
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
    # GUARD 2: baselines EXCLUDE the current (reaction) day.
    dd["adv20"] = grp["turnover"].transform(lambda s: s.rolling(20).mean().shift(1))
    dd["avgvol20"] = grp["volume"].transform(lambda s: s.rolling(20).mean().shift(1))

    # Vectorized within-symbol forward shifts (rows are symbol-contiguous;
    # cross-symbol boundaries masked to NaN) — same helper as Phase 2.
    sym = dd["symbol"].to_numpy()

    def shifted(values: np.ndarray, k: int) -> np.ndarray:
        v = values.astype(float)
        out = np.full(len(v), np.nan)
        if k > 0:
            out[:-k] = v[k:]
            ok = sym[:-k] == sym[k:]
            out[:-k][~ok] = np.nan
        elif k < 0:
            kk = -k
            out[kk:] = v[:-kk]
            ok = sym[kk:] == sym[:-kk]
            out[kk:][~ok] = np.nan
        else:
            out = v
        return out

    dd["prev_close"] = shifted(dd["close"].to_numpy(), -1)
    # Entry (Mode B): open of reaction+1 session; entry date carried for the gate.
    dd["entry_open"] = shifted(dd["open"].to_numpy(), LOCKED_ENTRY_LAG)
    date_ns = dd["date"].to_numpy(dtype="datetime64[ns]").astype("int64")
    ed = shifted(date_ns.astype(float), LOCKED_ENTRY_LAG)
    dd["entry_date"] = pd.to_datetime(pd.Series(ed).astype("Int64"), unit="ns")
    # Exit closes at H sessions in-market: close[reaction_pos + H] (entry_pos = +1).
    for h in LOCKED_HOLDS:
        dd[f"exit_close_h{h}"] = shifted(dd["close"].to_numpy(), LOCKED_ENTRY_LAG + h - 1)
    # Post-entry highs/lows for the h10 excursion window (positions +1 .. +10).
    for k in range(LOCKED_ENTRY_LAG, LOCKED_ENTRY_LAG + H_MAX):
        dd[f"high_p{k}"] = shifted(dd["high"].to_numpy(), k)
        dd[f"low_p{k}"] = shifted(dd["low"].to_numpy(), k)
    return dd


def load_events() -> pd.DataFrame:
    ev = pd.read_parquet(EVENTS_PARQUET)
    log(f"earnings events: shape={ev.shape}  announce {pd.to_datetime(ev['announce_date']).min().date()}"
        f" -> {pd.to_datetime(ev['announce_date']).max().date()}")
    ev = ev[ev["result_type"] == "quarterly"].copy()
    ev["symbol"] = ev["symbol"].astype(str).str.replace("NSE:", "", regex=False).str.upper()
    ev["announce_date"] = pd.to_datetime(ev["announce_date"]).dt.normalize()
    # Load a small pad before window start so a Dec-31 AMC announcement can map to a
    # Jan reaction day; the DISCOVERY assertion below is on REACTION dates.
    lo = DISCOVERY_START - pd.Timedelta(days=EVENT_LOAD_PAD_DAYS)
    ev = ev[(ev["announce_date"] >= lo) & (ev["announce_date"] <= DISCOVERY_END)]
    log(f"  quarterly, announce in [{lo.date()} .. {DISCOVERY_END.date()}]: n={len(ev)}")
    hour = pd.to_datetime(ev["announce_time"], errors="coerce").dt.hour
    cls = ev["announce_time_class"]
    # Reaction-day rule — identical to Phase 2 / _tmp_b1_pead_killtest.
    ev["same_day"] = (cls.isin(["intraday", "BMO"])
                      | ((cls == "scheduled") & ~(hour.notna() & (hour >= 15))))
    log(f"  same-day reaction: {int(ev['same_day'].sum())}  next-day: {int((~ev['same_day']).sum())}")
    return ev


def map_reaction_rows(ev: pd.DataFrame, dd: pd.DataFrame) -> pd.DataFrame:
    keep = (["symbol", "date", "open", "close", "volume", "prev_close", "adv20", "avgvol20",
             "entry_open", "entry_date"]
            + [f"exit_close_h{h}" for h in LOCKED_HOLDS]
            + [f"high_p{k}" for k in range(LOCKED_ENTRY_LAG, LOCKED_ENTRY_LAG + H_MAX)]
            + [f"low_p{k}" for k in range(LOCKED_ENTRY_LAG, LOCKED_ENTRY_LAG + H_MAX)])
    bars = dd[keep]

    dd_syms = set(dd["symbol"].unique())
    log(f"events with symbol absent from daily data: {int((~ev['symbol'].isin(dd_syms)).sum())} / {len(ev)}")

    same = ev[ev["same_day"]].merge(
        bars, left_on=["symbol", "announce_date"], right_on=["symbol", "date"], how="inner")

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
    evr = evr.drop_duplicates(subset=["symbol", "reaction_date"]).reset_index(drop=True)
    # Reaction dates must be inside Discovery (announce-pad can push next-day maps in;
    # nothing can leave the window because announce_date <= DISCOVERY_END and later
    # reaction days are filtered here).
    evr = evr[(evr["reaction_date"] >= DISCOVERY_START) & (evr["reaction_date"] <= DISCOVERY_END)]
    log(f"events mapped to reaction rows in Discovery: n={len(evr)} "
        f"({evr['reaction_date'].min().date()} -> {evr['reaction_date'].max().date()})")
    return evr


# ---------------------------------------------------------------------------
# Signal pool + CAUSAL percentile thresholds (GUARD 7)
# ---------------------------------------------------------------------------

def build_pool(evr: pd.DataFrame) -> pd.DataFrame:
    e = evr.copy()
    e["react_ret"] = e["close"] / e["prev_close"] - 1.0
    e["vol_ratio"] = e["volume"] / e["avgvol20"]

    n0 = len(e)
    pool = e[np.isfinite(e["react_ret"])
             & (e["adv20"] >= LOCKED_ADV_FLOOR_INR)
             & np.isfinite(e["vol_ratio"])
             & (e["vol_ratio"] >= LOCKED_VOLCONF_MULT)].copy()
    log(f"volume-confirmed ADV-eligible event pool: n={len(pool)} (from {n0} mapped events)")
    pool = pool.sort_values("reaction_date", kind="mergesort").reset_index(drop=True)
    return pool


def causal_thresholds(pool: pd.DataFrame) -> pd.DataFrame:
    """Per unique reaction date d: percentile thresholds from the trailing
    volume-confirmed event cross-section with reaction_date in [d-60cal, d).
    STRICTLY-BEFORE inclusion — no event sees its own date or any later event.
    A pooled-window percentile here would be look-ahead; Phase 2 used one as a
    documented measurement simplification, and this script replaces it."""
    dates = pool["reaction_date"].to_numpy(dtype="datetime64[ns]")
    rets = pool["react_ret"].to_numpy(dtype=float)
    uniq = np.unique(dates)
    rows = []
    win = np.timedelta64(LOCKED_TRAILING_CAL_DAYS, "D")
    for d in uniq:
        lo_i = np.searchsorted(dates, d - win, side="left")
        hi_i = np.searchsorted(dates, d, side="left")   # strictly before d
        n_trail = hi_i - lo_i
        if n_trail >= LOCKED_MIN_TRAILING_EVENTS:
            seg = rets[lo_i:hi_i]
            thr_long = float(np.quantile(seg, LOCKED_PCT_LONG))
            thr_short = float(np.quantile(seg, LOCKED_PCT_SHORT))
        else:
            thr_long, thr_short = np.nan, np.nan
        rows.append({"reaction_date": pd.Timestamp(d), "n_trailing_events": int(n_trail),
                     "thr_long": thr_long, "thr_short": thr_short})
    thr = pd.DataFrame(rows)
    n_defined = int(thr["thr_long"].notna().sum())
    log(f"causal thresholds: defined on {n_defined}/{len(thr)} reaction dates "
        f"(min trailing events = {LOCKED_MIN_TRAILING_EVENTS})")
    return thr


# ---------------------------------------------------------------------------
# Candidate emission (both legs) + universe gate + ledger build
# ---------------------------------------------------------------------------

def emit_candidates(pool: pd.DataFrame, thr: pd.DataFrame) -> pd.DataFrame:
    e = pool.merge(thr, on="reaction_date", how="left")

    n_no_thr = int(e["thr_long"].isna().sum())
    e = e[e["thr_long"].notna()]
    log(f"events skipped (insufficient trailing cross-section): {n_no_thr}")

    long_mask = e["react_ret"] >= e["thr_long"]
    short_mask = e["react_ret"] <= e["thr_short"]
    cands = pd.concat([
        e[long_mask].assign(leg="long"),
        e[short_mask].assign(leg="short"),
    ], ignore_index=True)
    log(f"causal-percentile candidates: long={int(long_mask.sum())}  short={int(short_mask.sum())}")

    # Entry availability (Mode B next-session open) + full h10 forward path.
    n0 = len(cands)
    cands = cands[cands["entry_open"].notna() & (cands["entry_open"] > 0)
                  & cands["entry_date"].notna()]
    n1 = len(cands)
    cands = cands[np.isfinite(cands[f"exit_close_h{H_MAX}"])]
    log(f"dropped for missing entry open: {n0 - n1}; for incomplete {H_MAX}-session "
        f"forward path (delist/suspension survivorship caveat): {n1 - len(cands)}")

    # ProductionUniverseGate on the ENTRY date (LOCKED config, see header).
    from tools.sub9_research.production_universe import ProductionUniverseGate
    gate = ProductionUniverseGate(
        accepted_caps=set(LOCKED_UNIVERSE_CAPS),
        require_mis=LOCKED_REQUIRE_MIS,
        min_trading_days_required=0,
        min_daily_avg_volume=0,
    )
    ok = cands.apply(lambda r: gate.is_eligible(r["symbol"], r["entry_date"].date()), axis=1)
    log(f"ProductionUniverseGate (entry date, caps={sorted(LOCKED_UNIVERSE_CAPS)}, "
        f"require_mis={LOCKED_REQUIRE_MIS}): pass {int(ok.sum())} / {len(cands)}")
    cands = cands[ok].reset_index(drop=True)
    return cands


def build_ledger(cands: pd.DataFrame) -> pd.DataFrame:
    from services.symbol_metadata import get_cap_segment, in_fno_liquid_200

    # Tradability flags (current snapshots — NOT point-in-time; see header).
    import json
    with open(MTF_SNAPSHOT, encoding="utf-8") as f:
        mtf_entries = json.load(f)
    mtf_all = {str(x.get("tradingsymbol", "")).upper(): str(x.get("category", ""))
               for x in mtf_entries if isinstance(x, dict)}
    mtf_set = {s for s, c in mtf_all.items() if s and c != "etf"}
    fo_cat_set = {s for s, c in mtf_all.items() if s and c == "fo"}
    log(f"tradability sources: MTF snapshot n={len(mtf_set)} (non-ETF), "
        f"MTF category=='fo' n={len(fo_cat_set)}, fno_liquid_200 via symbol_metadata")

    # Descriptive ADV tiers over emitted candidates (tier1 = most illiquid; B1
    # convention). Reporting bucket ONLY — never a signal-time filter (Guard 7 note).
    cands = cands.copy()
    cands["adv_tier"] = pd.qcut(cands["adv20"], N_ADV_TIERS, labels=False,
                                duplicates="drop").astype(int) + 1

    high_cols = [f"high_p{k}" for k in range(LOCKED_ENTRY_LAG, LOCKED_ENTRY_LAG + H_MAX)]
    low_cols = [f"low_p{k}" for k in range(LOCKED_ENTRY_LAG, LOCKED_ENTRY_LAG + H_MAX)]

    rows = []
    for r in cands.itertuples(index=False):
        d = r._asdict() if hasattr(r, "_asdict") else dict(zip(cands.columns, r))
        entry = float(d["entry_open"])
        is_long = d["leg"] == "long"
        exit_h10 = float(d[f"exit_close_h{H_MAX}"])
        qty = max(1, int(POSITION_NOTIONAL_INR / entry))
        pnl_pct = ((exit_h10 - entry) if is_long else (entry - exit_h10)) / entry * 100.0

        max_high = max(float(d[c]) for c in high_cols)
        min_low = min(float(d[c]) for c in low_cols)
        if is_long:
            mfe_pct = (max_high - entry) / entry * 100.0
            mae_pct = (min_low - entry) / entry * 100.0
        else:
            mfe_pct = (entry - min_low) / entry * 100.0
            mae_pct = (entry - max_high) / entry * 100.0

        bare = str(d["symbol"])
        entry_date = pd.Timestamp(d["entry_date"])
        cap_seg = get_cap_segment(f"NSE:{bare}", session_date=entry_date.date())

        row = {
            # ---- canonical sanity_csv_schema columns (H10 canonical exit) ----
            "signal_date": pd.Timestamp(d["reaction_date"]).date(),
            "symbol": f"NSE:{bare}",
            "side": "LONG" if is_long else "SHORT",
            "entry_price": entry,
            "exit_price": exit_h10,
            "qty": qty,
            "pnl_pct": pnl_pct,
            "exit_reason": "time_stop",
            "same_bar": False,
            "cap_segment": cap_seg,
            # ---- per-event research columns (multiday, both legs) ----
            "leg": d["leg"],
            "reaction_date": pd.Timestamp(d["reaction_date"]).date(),
            "entry_date": entry_date.date(),
            "entry_open": entry,
            "reaction_ret_pct": float(d["react_ret"]) * 100.0,
            "vol_ratio": float(d["vol_ratio"]),
            "adv20_inr": float(d["adv20"]),
            "adv_tier": int(d["adv_tier"]),
            "causal_thr_pct": float(d["thr_long" if is_long else "thr_short"]) * 100.0,
            "n_trailing_events": int(d["n_trailing_events"]),
            "tradable_fo": bool(in_fno_liquid_200(f"NSE:{bare}") or bare in fo_cat_set),
            "tradable_mtf": bool(bare in mtf_set),
            "mfe_pct": mfe_pct,
            "mae_pct": mae_pct,
        }
        for h in LOCKED_HOLDS:
            row[f"fwd_h{h}_pct"] = (float(d[f"exit_close_h{h}"]) / entry - 1.0) * 100.0
        rows.append(row)

    ledger = pd.DataFrame(rows).sort_values(
        ["reaction_date", "symbol", "leg"], kind="mergesort").reset_index(drop=True)

    # DISCOVERY-WINDOW ASSERTION (hard): raise if any signal date leaked outside.
    sd = pd.to_datetime(ledger["signal_date"])
    if ((sd < DISCOVERY_START) | (sd > DISCOVERY_END)).any():
        bad = ledger.loc[(sd < DISCOVERY_START) | (sd > DISCOVERY_END),
                         ["symbol", "signal_date"]].head()
        raise RuntimeError(f"DISCOVERY-WINDOW VIOLATION: signal dates outside "
                           f"[{DISCOVERY_START.date()} .. {DISCOVERY_END.date()}]:\n{bad}")
    return ledger


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(ledger: pd.DataFrame) -> None:
    fwd_cols = [f"fwd_h{h}_pct" for h in LOCKED_HOLDS]
    for leg in ("long", "short"):
        sub = ledger[ledger["leg"] == leg]
        log(f"--- {leg.upper()} leg: n={len(sub)} ---")
        if not len(sub):
            continue
        for h in LOCKED_HOLDS:
            f = sub[f"fwd_h{h}_pct"]
            hit = (f > 0).mean() if leg == "long" else (f < 0).mean()
            log(f"  H={h:2d}: mean fwd {f.mean():+.3f}%  median {f.median():+.3f}%  hit {hit:.3f}")
        log(f"  MFE/MAE (h10 window, leg-aware): mean mfe {sub['mfe_pct'].mean():+.2f}%  "
            f"mean mae {sub['mae_pct'].mean():+.2f}%")
        log(f"  per ADV tier (tier1=most illiquid): n / mean h7 / mean h10")
        for t, g in sub.groupby("adv_tier"):
            log(f"    tier{t}: n={len(g):4d}  h7 {g['fwd_h7_pct'].mean():+.3f}%  "
                f"h10 {g['fwd_h10_pct'].mean():+.3f}%")
        log(f"  per cap segment: n / mean h7 / mean h10")
        for c, g in sub.groupby("cap_segment"):
            log(f"    {c:<10s}: n={len(g):4d}  h7 {g['fwd_h7_pct'].mean():+.3f}%  "
                f"h10 {g['fwd_h10_pct'].mean():+.3f}%")
        if leg == "short":
            trad = sub[sub["tradable_fo"] | sub["tradable_mtf"]]
            log(f"  SHORT tradable subset (F&O or MTF): n={len(trad)} of {len(sub)}")
            if len(trad):
                for h in LOCKED_HOLDS:
                    log(f"    H={h:2d}: tradable mean {trad[f'fwd_h{h}_pct'].mean():+.3f}%  "
                        f"vs full {sub[f'fwd_h{h}_pct'].mean():+.3f}%")


def main() -> None:
    log("=" * 78)
    log(f"STAGE 4 SANITY: {SETUP_NAME}  (Discovery {DISCOVERY_START.date()} -> {DISCOVERY_END.date()})")
    log("=" * 78)
    dd = load_daily()
    ev = load_events()
    evr = map_reaction_rows(ev, dd)
    pool = build_pool(evr)
    thr = causal_thresholds(pool)
    cands = emit_candidates(pool, thr)
    ledger = build_ledger(cands)

    # GUARD 6: canonical schema validation (intraday-single-exit layer; see SCHEMA NOTE).
    from tools.methodology.sanity_csv_schema import validate
    res = validate(ledger, setup_name=SETUP_NAME, layer="filtered_trades")
    log(f"sanity_csv_schema: {res.summary()}")
    if not res.is_valid:
        raise RuntimeError("schema validation FAILED — fix before using this ledger")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(OUT_CSV, index=False)
    log(f"wrote {len(ledger)} rows -> {OUT_CSV}")

    report(ledger)


if __name__ == "__main__":
    main()
