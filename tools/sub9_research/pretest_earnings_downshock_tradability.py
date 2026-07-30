"""
PRE-BRIEF TRADABILITY GAUNTLET -- earnings_shock_down_continuation_short
=======================================================================

Candidate (sole survivor of the 2026-07-29 event-class cost-clearing screen):

    signal : earnings reaction-day close move <= -8.0% (FIXED threshold)
    entry  : T+1 session OPEN
    exit   : same-session CLOSE (h1)  -> intraday MIS SHORT
    stat   : matched-abnormal return (event ret minus same-date x same-ADV-tercile
             panel mean), net of ROUND_TRIP_COST_PCT = 0.31%

Why this runs BEFORE the Stage-0 brief
--------------------------------------
`asm_gsm_stage_transition` died at Phase 2 partly because ~26% of its events were
Stage III / trade-for-trade: the short leg could not be placed AT ALL.  That check
should have come first.  Here it does.  A drift that only exists on events we
cannot short is not an edge, it is a chart.

Checks
------
0. Event-set integrity   : dedupe.  The screen's `screen_earnings_shock` never
                           de-duplicated (symbol, reaction_date); multiple filings
                           for the same company/session are counted as separate
                           "events".  Reported explicitly because it moves era_A.
1. Circuit-band blocking : from 5m bars, did T+1 OPEN at/near a lower price band
                           (zero-range first bar with volume / price pinned across
                           the first N bars / gap sitting on a standard 2-5-10-20%
                           band level)?  Blocked => the SELL cannot fill.
2. ASM/GSM overlap       : NSE surveillance membership on T+1 (any ASM stage =>
                           100% margin => no MIS short).  Stage III/IV (T2T)
                           flagged separately -- no intraday square-off exists.
                           BSE GSM joined via ISIN as a secondary, thin flag.
3. MIS-shortability      : nse_all.json (CURRENT Zerodha snapshot).  ANACHRONISTIC
                           + survivorship-biased when applied to history -- this is
                           reported as an UPPER BOUND / sensitivity, never as the
                           primary filter (lesson #27).
4. COMBINED TRADEABLE    : (1) AND (2) applied; (3) reported as a sensitivity.
                           THE decisive table.
5. panic_crash conflict  : panic_crash_revert_long BUYS deep intraday crashes.
                           Does it fire on the same (symbol, T+1 date) that this
                           candidate is short?  Replays its config trigger.
6. Repair sensitivity    : recompute on earnings_events.parquet (repaired
                           2026-07-28) AND earnings_events_pre_repair_2026-07-28.
7. Execution realism     : confirm entry session is strictly after the reaction
                           session (no lookahead), and measure the overnight gap
                           reaction_close -> T+1 open (is the drift already spent?).
7b. OPEN-PRINT ARTIFACT  : `screen_event_classes_cost_clearing.load_prices` warns in
                           its OWN header that the daily panel carries "an open-print
                           measurement artifact in illiquid names (the 09:15 auction
                           print)".  This candidate enters AT that print, on low-ADV
                           names, and a date x ADV-tercile baseline can only remove
                           the tercile-AVERAGE artifact -- a post-earnings-shock name
                           is far more dislocated than its tercile peers.  So we
                           decompose the open->close short return into (a) open ->
                           09:20 (first 5m bar) and (b) 09:20 -> close, and re-run the
                           whole expectancy with a REALISTIC entry at the 09:20 print,
                           against a properly re-matched 09:20->close date x ADV
                           baseline rebuilt from the FULL 5m archive.  If the edge
                           lives inside bar 1, it is not tradeable.

Ledger note
-----------
Development-window feasibility check.  No cell is locked, no candidate is
validated, nothing is shipped, no ledger line.  Signals >= 2026-05-01 are DROPPED
(fresh pool stays clean).

Output: reports/sub9_sanity/_earnings_downshock_tradability.csv
"""

from __future__ import annotations

import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "sub9_research"))

import screen_event_classes_cost_clearing as S  # noqa: E402  (shared price panel + abnormal-return machinery)

# ---------------------------------------------------------------------------
# Constants -- explicit, exploratory pre-test (mirrors the screen's constants)
# ---------------------------------------------------------------------------
COST = S.ROUND_TRIP_COST_PCT           # 0.31
SHOCK_THRESHOLD_PCT = -8.0             # fixed, as pitched
HORIZON = 1                            # h1 = entry-day close

FIVEM_DIR = ROOT / "backtest-cache-download" / "monthly"
ASM_PATH = ROOT / "data" / "asm_gsm_history" / "asm_gsm_events.parquet"
EARN_CUR = ROOT / "data" / "earnings_calendar" / "earnings_events.parquet"
EARN_PRE = ROOT / "data" / "earnings_calendar" / "earnings_events_pre_repair_2026-07-28.parquet"
MIS_PATH = ROOT / "nse_all.json"
OUT_PATH = ROOT / "reports" / "sub9_sanity" / "_earnings_downshock_tradability.csv"

# --- circuit-band proxy parameters (documented, not tuned) ---
STD_BANDS_PCT = (2.0, 5.0, 10.0, 20.0)   # NSE non-F&O standard price bands
BAND_TOL_PP = 0.30                        # open within 0.30pp of a band level
PIN_BARS = 3                              # bars (15 min) with no trade away from one price
MIN_GAP_FOR_BLOCK_PCT = -1.5              # a "band open" must actually be a gap DOWN

# --- panic_crash_revert_long trigger (read from config/configuration.json) ---
PANIC_SETUP = "panic_crash_revert_long"

ROWS: list[dict] = []


def add(check: str, scope: str, era: str, **kw) -> None:
    ROWS.append(dict(check=check, scope=scope, era=era, **kw))


# ---------------------------------------------------------------------------
# 1. Event construction (mirrors screen_earnings_shock exactly, then extends)
# ---------------------------------------------------------------------------
def build_events(px, meta, parquet: Path, dedup: bool,
                 threshold: float = SHOCK_THRESHOLD_PCT) -> pd.DataFrame:
    df = pd.read_parquet(parquet)
    df["announce_date"] = pd.to_datetime(df["announce_date"], errors="coerce")
    df["announce_time"] = pd.to_datetime(df["announce_time"], errors="coerce")
    df["symbol"] = S.normalise_symbol(df["symbol"])

    hour = df["announce_time"].dt.hour.fillna(18)
    same_day = df["announce_time_class"].isin(["intraday", "BMO"]) & (hour < 15)
    df["react_anchor"] = np.where(
        same_day, df["announce_date"], df["announce_date"] + pd.Timedelta(days=1)
    )
    df["react_anchor"] = pd.to_datetime(df["react_anchor"])

    react = px[["symbol", "date", "close", "prev_close"]].sort_values("date")
    cols = ["symbol", "react_anchor", "isin", "announce_time", "announce_time_class"]
    d = df[cols].dropna(subset=["symbol", "react_anchor"]).sort_values("react_anchor")

    m = pd.merge_asof(
        d,
        react.rename(columns={"date": "react_date", "close": "react_close"}),
        left_on="react_anchor", right_on="react_date", by="symbol",
        direction="forward", allow_exact_matches=True,
        tolerance=pd.Timedelta(days=S.ENTRY_STALENESS_DAYS),
    )
    m = m[m["react_date"].notna() & m["prev_close"].notna() & (m["prev_close"] > 0)]
    m["react_move"] = (m["react_close"] / m["prev_close"] - 1.0) * 100.0
    m["signal_date"] = m["react_date"]
    m = m[["symbol", "signal_date", "react_move", "react_close", "isin",
           "announce_time", "announce_time_class"]]
    if dedup:
        m = m.sort_values(["symbol", "signal_date"]).drop_duplicates(
            ["symbol", "signal_date"], keep="first"
        )
    ev = S.attach_returns(m, px, meta)
    if ev.empty:
        return ev
    ev = ev[ev["react_move"] <= threshold].copy()
    # SHORT convention throughout
    ev["short_net"] = -ev[f"ret_h{HORIZON}"] - COST
    ev["short_gross"] = -ev[f"ret_h{HORIZON}"]
    ev["year"] = ev["signal_date"].dt.year
    return ev


def add_adv_tiers(ev: pd.DataFrame) -> pd.DataFrame:
    """Terciles of prior-only ADV WITHIN the event set -- same construction the
    screen's `emit(adv_split=True)` uses, so 'low+mid' means the same thing."""
    ev = ev.copy()
    d = ev["adv20"]
    q = d.quantile([0.33, 0.66]).to_list()
    ev["adv_tier"] = np.where(d.isna(), "adv_na",
                       np.where(d <= q[0], "adv_low",
                       np.where(d <= q[1], "adv_mid", "adv_high")))
    ev["lowmid"] = ev["adv_tier"].isin(["adv_low", "adv_mid"])
    return ev


# ---------------------------------------------------------------------------
# Reporting helper -- the one expectancy block used everywhere
# ---------------------------------------------------------------------------
def stats(sub: pd.DataFrame) -> dict:
    r = sub["short_gross"].dropna()
    n = len(r)
    if n == 0:
        return dict(n=0, mean_gross=np.nan, net_exp=np.nan, median_gross=np.nan,
                    hit_gross=np.nan, hit_net=np.nan, tstat=np.nan)
    sd = r.std(ddof=1) if n > 1 else np.nan
    return dict(
        n=n,
        mean_gross=float(r.mean()),
        net_exp=float(r.mean() - COST),
        median_gross=float(r.median()),
        hit_gross=float((r > 0).mean()),
        hit_net=float((r > COST).mean()),
        tstat=float(r.mean() / (sd / np.sqrt(n))) if (sd and sd > 0) else np.nan,
    )


def report(check: str, label: str, ev: pd.DataFrame, per_year: bool = False,
           echo: bool = True) -> None:
    lines = []
    for era in ("ALL", "era_A", "era_B"):
        sub = ev if era == "ALL" else ev[ev["era"] == era]
        for scope_lbl, s2 in (("all_adv", sub), ("low+mid_adv", sub[sub["lowmid"]])):
            st = stats(s2)
            add(check, f"{label}|{scope_lbl}", era, **st)
            if echo:
                lines.append(f"  {era:6s} {scope_lbl:12s} n={st['n']:5d} "
                             f"gross={st['mean_gross']:+.3f} NET={st['net_exp']:+.3f} "
                             f"hit={st['hit_gross']:.3f} t={st['tstat']:+.2f}")
    if per_year:
        for yr, sub in ev.groupby("year"):
            for scope_lbl, s2 in (("all_adv", sub), ("low+mid_adv", sub[sub["lowmid"]])):
                st = stats(s2)
                add(check, f"{label}|{scope_lbl}", f"year_{yr}", **st)
                if echo and scope_lbl == "low+mid_adv":
                    lines.append(f"  {yr}   {scope_lbl:12s} n={st['n']:5d} "
                                 f"gross={st['mean_gross']:+.3f} NET={st['net_exp']:+.3f} "
                                 f"hit={st['hit_gross']:.3f}")
    if echo:
        print(f"\n--- {check} :: {label} ---")
        print("\n".join(lines), flush=True)


# ---------------------------------------------------------------------------
# CHECK 1 + 5 + 7 need intraday bars on the ENTRY session
# ---------------------------------------------------------------------------
def load_entry_bars(ev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Single pass over the 5m archive returning BOTH:

    * `bars`  : 5m bars for exactly the (symbol, entry_date) pairs we need, plus
                the reaction session (used for the panic_crash same-day check).
    * `panel` : a PANEL-WIDE per (symbol, session) reduction for EVERY symbol
                trading on any of our entry dates -- first bar open, first bar
                close (09:20), second bar close (09:25), session close.  This is
                what the check-7b re-matched intraday baseline is built from.
    """
    want = set(zip(ev["symbol"], ev["entry_date"]))
    want |= set(zip(ev["symbol"], ev["signal_date"]))
    entry_sessions = set(ev["entry_date"].unique())
    months = sorted({f"{d.year}_{d.month:02d}" for _, d in want})
    syms = {s for s, _ in want}
    out, pan = [], []
    for mo in months:
        p = FIVEM_DIR / f"{mo}_5m_enriched.feather"
        if not p.exists():
            continue
        d = pd.read_feather(p, columns=["date", "symbol", "open", "close", "high",
                                        "low", "volume"])
        d["session"] = pd.to_datetime(d["date"]).dt.normalize()
        d = d.sort_values(["symbol", "date"])

        # ---- panel reduction on the entry sessions (ALL symbols) ----
        dp = d[d["session"].isin(entry_sessions)]
        if not dp.empty:
            red = dp.groupby(["symbol", "session"], sort=False).agg(
                p_open=("open", "first"), p_close=("close", "last"),
                p_b1c=("close", "first"), p_nbars=("close", "size")).reset_index()
            k = dp.groupby(["symbol", "session"], sort=False).cumcount()
            b2 = (dp.loc[k == 1, ["symbol", "session", "close"]]
                    .rename(columns={"close": "p_b2c"}))
            pan.append(red.merge(b2, on=["symbol", "session"], how="left"))

        # ---- the event's own bars ----
        de = d[d["symbol"].isin(syms)]
        if not de.empty:
            de = de[[t in want for t in zip(de["symbol"], de["session"])]]
            if not de.empty:
                out.append(de)

    bars = (pd.concat(out, ignore_index=True).sort_values(["symbol", "date"])
            .reset_index(drop=True)) if out else pd.DataFrame()
    panel = pd.concat(pan, ignore_index=True) if pan else pd.DataFrame()
    return bars, panel


def intraday_baseline(panel: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """date x ADV-tercile mean of the three intraday legs -- the SAME matched
    control construction the screen uses, but for 09:20/09:25 entries."""
    if panel.empty:
        return pd.DataFrame()
    tiers = px[["symbol", "date", "adv_tier_x"]].rename(columns={"date": "session"})
    p = panel.merge(tiers, on=["symbol", "session"], how="left")
    p = p[p["adv_tier_x"].notna() & (p["p_open"] > 0) & (p["p_b1c"] > 0)]
    p["leg_oc"] = (p["p_close"] / p["p_open"] - 1.0) * 100.0        # open  -> close
    p["leg_ob1"] = (p["p_b1c"] / p["p_open"] - 1.0) * 100.0         # open  -> 09:20
    p["leg_b1c"] = (p["p_close"] / p["p_b1c"] - 1.0) * 100.0        # 09:20 -> close
    p["leg_b2c"] = np.where(p["p_b2c"] > 0,
                            (p["p_close"] / p["p_b2c"] - 1.0) * 100.0, np.nan)
    return (p.groupby(["session", "adv_tier_x"])[["leg_oc", "leg_ob1", "leg_b1c", "leg_b2c"]]
             .mean().reset_index())


def circuit_flags(ev: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Per-event T+1-open circuit-band proxies."""
    recs = []
    if bars.empty:
        return pd.DataFrame(recs)
    g = {k: v for k, v in bars.groupby(["symbol", "session"], sort=False)}
    for row in ev.itertuples(index=False):
        key = (row.symbol, row.entry_date)
        b = g.get(key)
        rec = dict(symbol=row.symbol, entry_date=row.entry_date,
                   has_bars=b is not None)
        if b is None or len(b) == 0:
            recs.append(rec)
            continue
        b = b.sort_values("date")
        o = float(b["open"].iloc[0])
        hi1, lo1, v1 = float(b["high"].iloc[0]), float(b["low"].iloc[0]), float(b["volume"].iloc[0])
        rc = float(row.react_close)
        gap = (o / rc - 1.0) * 100.0
        day_hi, day_lo = float(b["high"].max()), float(b["low"].min())

        # bar 1 traded at a single price with real volume
        b1_zero_range = bool(hi1 == lo1 and v1 > 0)
        # leading bars pinned to one price (no trade away)
        pinned = 0
        px_ref = hi1
        for _, r2 in b.iterrows():
            if r2["high"] == r2["low"] == px_ref:
                pinned += 1
            else:
                break
        # open sits on a standard band level below the reaction close
        band_hit = any(abs(gap + bp) <= BAND_TOL_PP for bp in STD_BANDS_PCT)
        band_lvl = min(STD_BANDS_PCT, key=lambda bp: abs(gap + bp))
        locked_all_day = bool(day_hi == day_lo)
        n_bars = len(b)
        n_distinct_first6 = int(pd.unique(b["close"].head(6)).size)

        # ---- classification (strict primary; loose reported as sensitivity) ----
        blocked_strict = bool(
            locked_all_day
            or (b1_zero_range and pinned >= PIN_BARS and gap <= MIN_GAP_FOR_BLOCK_PCT)
        )
        blocked_loose = bool(
            blocked_strict
            or (b1_zero_range and gap <= MIN_GAP_FOR_BLOCK_PCT)
            or (band_hit and gap <= MIN_GAP_FOR_BLOCK_PCT and n_distinct_first6 <= 2)
            or n_bars < 10          # session barely traded -> no reliable fill
            or v1 <= 0              # no volume in the entry bar at all
        )
        rec.update(open_gap_pct=gap, b1_zero_range=b1_zero_range, pinned_bars=pinned,
                   band_hit=band_hit, nearest_band=band_lvl, locked_all_day=locked_all_day,
                   n_bars=n_bars, n_distinct_first6=n_distinct_first6,
                   entry_bar_volume=v1, blocked_strict=blocked_strict,
                   blocked_loose=blocked_loose)
        recs.append(rec)
    return pd.DataFrame(recs)


def panic_crash_flags(ev: pd.DataFrame, bars: pd.DataFrame, cfg: dict,
                      caps: dict, session_col: str) -> pd.DataFrame:
    """Replay panic_crash_revert_long's config trigger on a given session."""
    p = cfg["setups"][PANIC_SETUP]
    r3_max = float(p["crash_r3_pct_max"])
    vol_burst_min = float(p["vol_burst_min"])
    min_prior = int(p["min_bars_before_signal"])
    turn_floor = float(p["per_day_turnover_floor_rs"])
    allowed_caps = set(p["allowed_cap_segments"])
    w0 = pd.Timestamp(f"1900-01-01 {p['active_window_start']}").time()
    w1 = pd.Timestamp(f"1900-01-01 {p['active_window_end']}").time()

    recs = []
    if bars.empty:
        return pd.DataFrame(recs)
    g = {k: v for k, v in bars.groupby(["symbol", "session"], sort=False)}
    for row in ev.itertuples(index=False):
        sess = getattr(row, session_col)
        b = g.get((row.symbol, sess))
        rec = dict(symbol=row.symbol, entry_date=row.entry_date, panic_fires=False,
                   panic_first_time=pd.NaT, panic_cap_ok=(caps.get(row.symbol) in allowed_caps))
        if b is None or len(b) < 5:
            recs.append(rec)
            continue
        b = b.sort_values("date").reset_index(drop=True)
        turn = float((b["close"] * b["volume"]).sum())
        rec["day_turnover"] = turn
        if turn < turn_floor or not rec["panic_cap_ok"]:
            recs.append(rec)
            continue
        c = b["close"].to_numpy(float)
        v = b["volume"].to_numpy(float)
        t = pd.to_datetime(b["date"]).dt.time.to_numpy()
        for i in range(len(b)):
            if i < max(3, min_prior):
                continue
            if not (w0 <= t[i] <= w1):
                continue
            r3 = c[i] / c[i - 3] - 1.0
            if r3 > r3_max:
                continue
            prior_mean = v[:i].mean()
            if prior_mean <= 0 or (v[i] / prior_mean) < vol_burst_min:
                continue
            rec["panic_fires"] = True
            rec["panic_first_time"] = b["date"].iloc[i]
            rec["panic_r3"] = r3
            break
        recs.append(rec)
    return pd.DataFrame(recs)


# ---------------------------------------------------------------------------
def main() -> None:
    print("loading price panel ...", flush=True)
    px, meta = S.load_prices()
    print(f"prices {px.shape}  {px['date'].min().date()} -> {px['date'].max().date()}",
          flush=True)

    # =====================================================================
    # CHECK 0 / 6 : event-set integrity (dedupe) and repair sensitivity
    # =====================================================================
    print("\n================ CHECK 0/6 : EVENT-SET INTEGRITY ================")
    variants = {}
    for tag, pq in (("repaired", EARN_CUR), ("pre_repair", EARN_PRE)):
        for dd in (False, True):
            key = f"{tag}|{'dedup' if dd else 'raw_asfired'}"
            e = build_events(px, meta, pq, dedup=dd)
            e = add_adv_tiers(e)
            variants[key] = e
            for era in ("era_A", "era_B"):
                st = stats(e[e["era"] == era])
                add("0_event_integrity", key, era, **st)
                print(f"  {key:26s} {era}  n={st['n']:4d}  gross={st['mean_gross']:+.3f}  "
                      f"NET={st['net_exp']:+.3f}  hit={st['hit_gross']:.3f}", flush=True)

    # PRIMARY = repaired parquet, DE-DUPLICATED (one event per symbol-session)
    ev = variants["repaired|dedup"].copy()
    ev_asfired = variants["repaired|raw_asfired"].copy()
    print(f"\n  as-fired rows {len(ev_asfired)} -> unique (symbol, reaction_date) {len(ev)}"
          f"  ({len(ev_asfired)-len(ev)} duplicate filings dropped)")

    report("0_event_integrity", "PRIMARY_dedup_baseline", ev, per_year=True)
    report("0_event_integrity", "as_fired_with_duplicates", ev_asfired, per_year=False)

    # =====================================================================
    # CHECK 7 : execution realism
    # =====================================================================
    print("\n================ CHECK 7 : EXECUTION REALISM ================")
    bad = int((ev["entry_date"] <= ev["signal_date"]).sum())
    gap_days = (ev["entry_date"] - ev["signal_date"]).dt.days
    print(f"  entry_date <= signal_date (LOOKAHEAD) : {bad}  <- must be 0")
    print(f"  calendar gap signal->entry: median {gap_days.median():.0f}d  "
          f"p95 {gap_days.quantile(0.95):.0f}d  max {gap_days.max():.0f}d")
    late = ev["announce_time"].dt.hour
    print(f"  announce_time_class mix: "
          f"{ev['announce_time_class'].value_counts().to_dict()}")
    ev["overnight_gap_pct"] = (ev["entry_open"] / ev["react_close"] - 1.0) * 100.0
    for era in ("era_A", "era_B"):
        s = ev[ev["era"] == era]
        add("7_execution_realism", "overnight_gap", era,
            n=len(s), mean_gross=float(s["overnight_gap_pct"].mean()),
            median_gross=float(s["overnight_gap_pct"].median()),
            net_exp=np.nan, hit_gross=float((s["overnight_gap_pct"] < 0).mean()),
            hit_net=np.nan, tstat=np.nan)
        print(f"  {era} overnight gap react_close->T+1 open: mean {s['overnight_gap_pct'].mean():+.3f}%  "
              f"median {s['overnight_gap_pct'].median():+.3f}%  frac_down {(s['overnight_gap_pct']<0).mean():.3f}")
    print(f"  lookahead check: entry uses the NEXT session open, reaction close is "
          f"stamped at 15:30 of the prior session -> causal by construction.")

    # =====================================================================
    # CHECK 1 : circuit-band blocking (5m bars)
    # =====================================================================
    print("\n================ CHECK 1 : CIRCUIT-BAND BLOCKING ================")
    print("  loading 5m bars for entry + reaction sessions (+ panel reduction) ...", flush=True)
    bars, panel = load_entry_bars(ev)
    print(f"  bars {bars.shape}   panel {panel.shape}", flush=True)
    cf = circuit_flags(ev, bars)
    ev = ev.merge(cf, on=["symbol", "entry_date"], how="left")
    ev["has_bars"] = ev["has_bars"].fillna(False)
    for c in ("blocked_strict", "blocked_loose"):
        ev[c] = ev[c].fillna(False)
    # events with no 5m coverage cannot be assessed -> reported separately
    cov = float(ev["has_bars"].mean())
    print(f"  5m coverage of entry sessions: {cov:.3f}  "
          f"({int(ev['has_bars'].sum())}/{len(ev)})")
    for era in ("era_A", "era_B"):
        s = ev[(ev["era"] == era)]
        sa = s[s["has_bars"]]
        for lbl, mask in (("blocked_strict", sa["blocked_strict"]),
                          ("blocked_loose", sa["blocked_loose"]),
                          ("b1_zero_range", sa["b1_zero_range"] == True),
                          ("open_on_band_level", sa["band_hit"] == True),
                          ("locked_all_day", sa["locked_all_day"] == True)):
            frac = float(mask.mean()) if len(sa) else np.nan
            add("1_circuit", f"frac_{lbl}", era, n=len(sa), mean_gross=frac,
                net_exp=np.nan, median_gross=np.nan, hit_gross=np.nan,
                hit_net=np.nan, tstat=np.nan)
            print(f"  {era} {lbl:20s} frac={frac:.4f}  (n_assessable={len(sa)})")
    report("1_circuit", "EXCL_blocked_strict", ev[ev["has_bars"] & ~ev["blocked_strict"]], per_year=True)
    report("1_circuit", "EXCL_blocked_loose", ev[ev["has_bars"] & ~ev["blocked_loose"]])
    report("1_circuit", "ONLY_blocked_loose", ev[ev["has_bars"] & ev["blocked_loose"]])

    # =====================================================================
    # CHECK 2 : ASM / GSM overlap
    # =====================================================================
    print("\n================ CHECK 2 : ASM / GSM SURVEILLANCE OVERLAP ================")
    asm = pd.read_parquet(ASM_PATH)
    asm["date"] = pd.to_datetime(asm["date"])
    nse = asm[asm["exchange"] == "NSE"].copy()
    nse["symbol"] = S.normalise_symbol(nse["symbol"])
    nse_mem = nse.groupby(["symbol", "date"]).agg(
        asm_program=("surveillance_program", "first"),
        asm_stage=("stage", lambda s: max(s, key=lambda x: ["0", "I", "II", "III", "IV", "VI"].index(str(x))
                                          if str(x) in ["0", "I", "II", "III", "IV", "VI"] else 0)),
    ).reset_index().rename(columns={"date": "entry_date"})
    ev = ev.merge(nse_mem, on=["symbol", "entry_date"], how="left")
    ev["in_asm_nse"] = ev["asm_program"].notna()
    ev["asm_t2t"] = ev["asm_stage"].isin(["III", "IV"])

    # BSE GSM via ISIN (thin / secondary)
    bse = asm[(asm["exchange"] == "BSE") & (asm["surveillance_program"] == "GSM")]
    bse_key = set(zip(bse["isin"].astype(str), bse["date"]))
    ev["in_gsm_bse"] = [ (str(i), d) in bse_key for i, d in zip(ev["isin"].astype(str), ev["entry_date"]) ]
    ev["surv_any"] = ev["in_asm_nse"] | ev["in_gsm_bse"]

    for era in ("era_A", "era_B"):
        s = ev[ev["era"] == era]
        for lbl, m in (("in_asm_nse", s["in_asm_nse"]), ("asm_stage_III_IV_T2T", s["asm_t2t"]),
                       ("in_gsm_bse", s["in_gsm_bse"]), ("surveillance_any", s["surv_any"])):
            frac = float(m.mean())
            add("2_surveillance", f"frac_{lbl}", era, n=len(s), mean_gross=frac,
                net_exp=np.nan, median_gross=np.nan, hit_gross=np.nan, hit_net=np.nan, tstat=np.nan)
            print(f"  {era} {lbl:22s} frac={frac:.4f}  n_events={int(m.sum())}/{len(s)}")
    report("2_surveillance", "EXCL_surveillance", ev[~ev["surv_any"]], per_year=True)
    report("2_surveillance", "ONLY_surveillance", ev[ev["surv_any"]])

    # =====================================================================
    # CHECK 3 : MIS shortability (CURRENT snapshot -- anachronistic upper bound)
    # =====================================================================
    print("\n================ CHECK 3 : MIS SHORTABILITY (upper bound) ================")
    mis = json.load(io.open(MIS_PATH, encoding="utf-8"))
    mis_ok, caps = {}, {}
    for r in mis:
        bare = str(r["symbol"]).replace(".NS", "").upper()
        mis_ok[bare] = bool(r.get("mis_enabled")) and float(r.get("mis_leverage") or 0) > 1.0
        caps[bare] = r.get("cap_segment")
    ev["mis_shortable_now"] = ev["symbol"].map(mis_ok).fillna(False)
    ev["in_mis_snapshot"] = ev["symbol"].isin(mis_ok)
    print("  CAVEAT (lesson #27): nse_all.json is a CURRENT Zerodha snapshot applied to")
    print("  2023-2026 signals -- anachronistic AND survivorship-biased.  UPPER BOUND only.")
    for era in ("era_A", "era_B"):
        s = ev[ev["era"] == era]
        for lbl, m in (("in_broker_snapshot", s["in_mis_snapshot"]),
                       ("mis_shortable_now", s["mis_shortable_now"])):
            frac = float(m.mean())
            add("3_mis_upper_bound", f"frac_{lbl}", era, n=len(s), mean_gross=frac,
                net_exp=np.nan, median_gross=np.nan, hit_gross=np.nan, hit_net=np.nan, tstat=np.nan)
            print(f"  {era} {lbl:22s} frac={frac:.4f}  n={int(m.sum())}/{len(s)}")
    report("3_mis_upper_bound", "ONLY_mis_shortable_now", ev[ev["mis_shortable_now"]], per_year=True)

    # =====================================================================
    # CHECK 4 : THE COMBINED TRADEABLE SUBSET
    # =====================================================================
    print("\n================ CHECK 4 : COMBINED TRADEABLE SUBSET ================")
    ev["tradeable_12"] = ev["has_bars"] & ~ev["blocked_strict"] & ~ev["surv_any"]
    ev["tradeable_12_loose"] = ev["has_bars"] & ~ev["blocked_loose"] & ~ev["surv_any"]
    ev["tradeable_123"] = ev["tradeable_12"] & ev["mis_shortable_now"]
    ev["tradeable_123_loose"] = ev["tradeable_12_loose"] & ev["mis_shortable_now"]
    for era in ("era_A", "era_B"):
        s = ev[ev["era"] == era]
        print(f"  {era}: raw n={len(s)}  ->  1+2 strict n={int(s['tradeable_12'].sum())}  "
              f"1+2 loose n={int(s['tradeable_12_loose'].sum())}  "
              f"1+2+3 n={int(s['tradeable_123'].sum())}")
    report("4_combined", "TRADEABLE_1and2_strict", ev[ev["tradeable_12"]], per_year=True)
    report("4_combined", "TRADEABLE_1and2_loose", ev[ev["tradeable_12_loose"]], per_year=True)
    report("4_combined", "TRADEABLE_1and2and3_MISsens", ev[ev["tradeable_123"]], per_year=True)
    report("4_combined", "TRADEABLE_1and2and3_loose", ev[ev["tradeable_123_loose"]], per_year=True)
    report("4_combined", "UNTRADEABLE_residual", ev[~ev["tradeable_12"]])

    # =====================================================================
    # CHECK 5 : panic_crash_revert_long conflict
    # =====================================================================
    print("\n================ CHECK 5 : panic_crash_revert_long CONFLICT ================")
    cfg = json.load(io.open(ROOT / "config" / "configuration.json", encoding="utf-8"))
    pc_entry = panic_crash_flags(ev, bars, cfg, caps, "entry_date")
    pc_entry = pc_entry.rename(columns={"panic_fires": "panic_on_entry_day",
                                        "panic_first_time": "panic_entry_time"})
    ev = ev.merge(pc_entry[["symbol", "entry_date", "panic_on_entry_day", "panic_entry_time"]],
                  on=["symbol", "entry_date"], how="left")
    pc_react = panic_crash_flags(ev, bars, cfg, caps, "signal_date")
    pc_react = pc_react.rename(columns={"panic_fires": "panic_on_reaction_day"})
    ev = ev.merge(pc_react[["symbol", "entry_date", "panic_on_reaction_day"]],
                  on=["symbol", "entry_date"], how="left")
    for c in ("panic_on_entry_day", "panic_on_reaction_day"):
        ev[c] = ev[c].fillna(False)
    for era in ("era_A", "era_B"):
        s = ev[(ev["era"] == era) & ev["has_bars"]]
        st = s[s["tradeable_12"]]
        for lbl, m, base in (("panic_same_day_T1_all", s["panic_on_entry_day"], s),
                             ("panic_same_day_T1_tradeable", st["panic_on_entry_day"], st),
                             ("panic_on_reaction_day", s["panic_on_reaction_day"], s)):
            frac = float(m.mean()) if len(base) else np.nan
            add("5_panic_conflict", f"frac_{lbl}", era, n=len(base), mean_gross=frac,
                net_exp=np.nan, median_gross=np.nan, hit_gross=np.nan, hit_net=np.nan, tstat=np.nan)
            print(f"  {era} {lbl:30s} frac={frac:.4f}  n={int(m.sum())}/{len(base)}")
    report("5_panic_conflict", "TRADEABLE_minus_panic_overlap",
           ev[ev["tradeable_12"] & ~ev["panic_on_entry_day"]], per_year=True)
    report("5_panic_conflict", "ONLY_panic_overlap", ev[ev["tradeable_12"] & ev["panic_on_entry_day"]])

    # =====================================================================
    # CHECK 7b : OPEN-PRINT ARTIFACT / realistic 09:20 entry
    # =====================================================================
    print("\n================ CHECK 7b : OPEN-PRINT ARTIFACT ================")
    base = intraday_baseline(panel, px)
    print(f"  intraday baseline cells (date x adv_tier): {len(base)}  "
          f"built from {panel['symbol'].nunique()} symbols", flush=True)

    eb = bars[bars.set_index(["symbol", "session"]).index.isin(
        pd.MultiIndex.from_arrays([ev["symbol"], ev["entry_date"]]))]
    er = eb.groupby(["symbol", "session"], sort=False).agg(
        e_open=("open", "first"), e_close=("close", "last"),
        e_b1c=("close", "first")).reset_index()
    k = eb.groupby(["symbol", "session"], sort=False).cumcount()
    e2 = eb.loc[k == 1, ["symbol", "session", "close"]].rename(columns={"close": "e_b2c"})
    er = er.merge(e2, on=["symbol", "session"], how="left").rename(
        columns={"session": "entry_date"})
    ev = ev.merge(er, on=["symbol", "entry_date"], how="left")
    ev = ev.merge(base.rename(columns={"session": "entry_date"}),
                  on=["entry_date", "adv_tier_x"], how="left")

    ev["ev_leg_oc"] = (ev["e_close"] / ev["e_open"] - 1.0) * 100.0
    ev["ev_leg_ob1"] = (ev["e_b1c"] / ev["e_open"] - 1.0) * 100.0
    ev["ev_leg_b1c"] = (ev["e_close"] / ev["e_b1c"] - 1.0) * 100.0
    ev["ev_leg_b2c"] = (ev["e_close"] / ev["e_b2c"] - 1.0) * 100.0
    # SHORT gross, abnormal (matched), per entry timing
    ev["short_open"] = -(ev["ev_leg_oc"] - ev["leg_oc"])
    ev["short_0920"] = -(ev["ev_leg_b1c"] - ev["leg_b1c"])
    ev["short_0925"] = -(ev["ev_leg_b2c"] - ev["leg_b2c"])
    ev["bar1_leg_short"] = -(ev["ev_leg_ob1"] - ev["leg_ob1"])

    for era in ("era_A", "era_B"):
        s = ev[(ev["era"] == era) & ev["tradeable_12"]]
        for lbl in ("all_adv", "low+mid_adv"):
            s2 = s if lbl == "all_adv" else s[s["lowmid"]]
            tot = s2["short_open"].mean()
            b1 = s2["bar1_leg_short"].mean()
            rest = s2["short_0920"].mean()
            add("7b_open_print", f"decomposition_{lbl}", era, n=len(s2),
                mean_gross=tot, net_exp=tot - COST, median_gross=b1,
                hit_gross=rest, hit_net=np.nan, tstat=np.nan)
            print(f"  {era} {lbl:12s} n={len(s2):4d}  open->close SHORT {tot:+.3f}"
                  f"  =  bar1(09:15-09:20) {b1:+.3f}  +  09:20->close {rest:+.3f}"
                  f"   [bar1 share {100*b1/tot if tot else float('nan'):.0f}%]")

    for lbl, col in (("REALISTIC_entry_0920", "short_0920"),
                     ("REALISTIC_entry_0925", "short_0925"),
                     ("SCREEN_entry_open_print", "short_open")):
        tmp = ev[ev["tradeable_12"]].copy()
        tmp["short_gross"] = tmp[col]
        report("7b_open_print", lbl, tmp, per_year=True)

    # =====================================================================
    # CHECK 8 : shock-threshold robustness + cost break-even
    # =====================================================================
    print("\n================ CHECK 8 : THRESHOLD ROBUSTNESS + COST BREAK-EVEN ================")
    print("  Is -8% a real cliff or a fitted round number?  Rebuilt from -4% down,")
    print("  surveillance-clean, daily open->close abnormal basis (no 5m needed).")
    wide = build_events(px, meta, EARN_CUR, dedup=True, threshold=-4.0)
    wide = wide.merge(nse_mem, on=["symbol", "entry_date"], how="left")
    wide["surv"] = wide["asm_program"].notna()
    wide = wide[~wide["surv"]]
    for thr in (-4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -12.0):
        w = add_adv_tiers(wide[wide["react_move"] <= thr].copy())
        line = f"  shock<={thr:6.1f}%"
        for era in ("era_A", "era_B"):
            s = w[(w["era"] == era) & w["lowmid"]]
            st = stats(s)
            add("8_threshold", f"shock_lte_{abs(thr):.0f}|low+mid_adv", era, **st)
            line += (f"   {era} n={st['n']:4d} NET={st['net_exp']:+.3f} "
                     f"t={st['tstat']:+.2f}")
        print(line)

    print("\n  COST BREAK-EVEN (gross abnormal drift = max round-trip cost the edge survives)")
    for entry_lbl, col in (("open_print", "short_open"), ("realistic_0920", "short_0920")):
        for era in ("era_A", "era_B"):
            s = ev[(ev["era"] == era) & ev["tradeable_12"] & ev["lowmid"]]
            be = float(s[col].mean())
            add("8_threshold", f"breakeven_cost|{entry_lbl}|low+mid_adv", era,
                n=len(s), mean_gross=be, net_exp=be - COST, median_gross=np.nan,
                hit_gross=np.nan, hit_net=np.nan, tstat=np.nan)
            print(f"    {entry_lbl:15s} {era} break-even round-trip cost = {be:.3f}%  "
                  f"(assumed {COST:.2f}%; headroom {be - COST:+.3f}pp)")

    # =====================================================================
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(ROWS)
    out.to_csv(OUT_PATH, index=False, float_format="%.4f")
    print(f"\nwrote {OUT_PATH}  rows={len(out)}")

    trades_path = OUT_PATH.with_name("_earnings_downshock_tradability_events.csv")
    keep = ["symbol", "signal_date", "entry_date", "era", "year", "react_move",
            "react_close", "entry_open", "adv20", "adv_tier", "lowmid",
            "raw_h1", "base_h1", "ret_h1", "short_gross", "short_net",
            "overnight_gap_pct", "has_bars", "open_gap_pct", "b1_zero_range",
            "pinned_bars", "band_hit", "locked_all_day", "blocked_strict",
            "blocked_loose", "in_asm_nse", "asm_stage", "asm_t2t", "in_gsm_bse",
            "surv_any", "mis_shortable_now", "tradeable_12", "tradeable_12_loose",
            "tradeable_123", "panic_on_entry_day", "panic_on_reaction_day",
            "adv_tier_x", "e_open", "e_b1c", "e_b2c", "e_close",
            "ev_leg_oc", "ev_leg_ob1", "ev_leg_b1c", "leg_oc", "leg_ob1", "leg_b1c",
            "short_open", "short_0920", "short_0925", "bar1_leg_short"]
    ev[[c for c in keep if c in ev.columns]].to_csv(trades_path, index=False,
                                                    float_format="%.4f")
    print(f"wrote {trades_path}  rows={len(ev)}")


if __name__ == "__main__":
    main()
