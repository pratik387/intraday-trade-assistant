"""sanity_earnings_downshock_continuation_short.py -- Stage 4 (Phase 4) sanity.

REAL P&L SIMULATION for `earnings_downshock_continuation_short`.
Brief: specs/2026-07-29-brief-earnings_downshock_continuation_short.md (SS4 = LOCKED).
Lifecycle: docs/setup_lifecycle.md Stage 4 (6-failure-mode checklist, binding).

WHAT IS DIFFERENT FROM EVERY PRIOR RUN ON THIS CANDIDATE
-------------------------------------------------------
All evidence to date (`screen_event_classes_cost_clearing.py`,
`pretest_earnings_downshock_tradability.py`) is an **abnormal return**: the event's
return minus a same-date x same-ADV-tercile panel mean, charged a flat 0.31% notional
round trip.  That is an UPPER BOUND on a tradeable signal, for three reasons:
  (a) it subtracts a panel baseline that a real short book does NOT collect,
  (b) it charges a single flat cost instead of the real Zerodha MIS ladder, whose
      per-order Rs-20 brokerage cap and 0.025% sell-side STT do not scale linearly, and
  (c) it carries zero slippage.
This script replaces all three with an actual trade ledger: real 5m entry print, real
5m exit print, integer quantity from a fixed rupee notional, the project's Zerodha MIS
fee helper on the SHORT leg convention, and explicit per-side slippage at 20bp and 50bp.
NO abnormal-return figure is carried into the Stage-4 conclusion.

LOCKED CONSTRUCTION (brief SS4 -- constants below; NOT tuned after the first run)
--------------------------------------------------------------------------------
  signal   : earnings reaction-day close-to-prev-close return <= -8.0% (FIXED)
             reaction day from `announce_time_class` (AMC / scheduled-after-15:00
             -> next trading session; BMO / intraday-before-15:00 -> announce day),
             identical to `screen_event_classes_cost_clearing` / the gauntlet.
  dedupe   : (symbol, reaction_date) -- MANDATORY.  Standalone + consolidated filings
             for the same company-session are ONE event.  (SS7.1: this alone moved
             era_A from +0.18% to +0.06% on the abnormal basis.)
  entry    : T+1, the CLOSE of the 09:15-09:20 5m bar (i.e. 09:20), SHORT.
             NOT the 09:15 auction/open print (SS7.2).
  exit     : same-day session close (last 5m bar close).  Single exit.
             NO stop, NO target -- geometry is a Phase-5 question and is not added here.
  universe : low+mid ADV tercile; ProductionUniverseGate on the ENTRY date;
             exclude circuit-blocked entry sessions; exclude any NSE ASM / BSE GSM
             listing on the entry date.
  window   : DISCOVERY ONLY.  Reaction dates 2023-01-01 .. 2024-12-31 AND entry dates
             <= 2024-12-31.  Asserted; the script RAISES on violation.
             2025 (OOS), 2026 (Holdout) and the 2026-05-01+ fresh pool are NOT touched.

STAGE-4 ANTI-BIAS GUARDS (docs/setup_lifecycle.md Stage 4, 6-failure-mode checklist)
------------------------------------------------------------------------------------
 1. NO full-day aggregates at signal time.  The only session-level quantities used to
    ADMIT a trade are: bar-0 open/high/low/volume, the bar-0 timestamp, and the bar
    count -- plus the circuit test, which is deliberately evaluated on the full entry
    session (see the note in `circuit_blocked_strict`).  `day_high` / `day_low` /
    `day_close` appear in the ledger as METADATA ONLY and gate nothing.  Nothing that
    resolves after 09:20 is used as a filter.
 2. Volume / ADV baselines EXCLUDE the current day.  `adv20` comes from the shared
    daily panel as `rolling(20).mean().shift(1)` on turnover (see
    `screen_event_classes_cost_clearing.load_prices`); it is measured on the REACTION
    day's history, i.e. strictly before the entry session.  ADV terciles are formed on
    that shifted series.
 3. Entry-bar convention (Mode-B off-by-one).  Entry is the CLOSE of bar 0, so bar 0's
    intra-bar range is already spent at entry: the path walk (MFE/MAE) starts at bar 1,
    never bar 0.  Exit is the close of the last bar.
    *** EXPLICIT OPTIMISM DISCLOSURE ***: the 09:20 close is BOTH the decision point and
    the fill price.  This is the standard next-bar-decision convention applied to a 5m
    bar (the decision variable -- the reaction-day close -- was known the previous
    evening; the 09:20 print is the first executable price we are willing to use after
    rejecting the 09:15 auction print).  A live 09:20:00 market order would fill at the
    first trade of the 09:20-09:25 bar, not at the 09:20 mark.  On illiquid names that
    is worth roughly one spread -- which is exactly what the 20bp/50bp slippage columns
    are sized to absorb.  Read the 50bp column as the honest execution case.
 4. Same-bar pessimism: no SL and no target exist in this construction, so there is no
    same-bar SL/target ambiguity to resolve.  `same_bar` is emitted as False for every
    row.  Where pessimism CAN bite -- entry fill and exit fill -- both legs are charged
    slippage in the adverse direction (sell lower on entry, buy higher on exit).
 5. Filters are LOCKED constants at the top of this file (`LOCKED_FILTERS`).  Nothing
    was tuned after the first Discovery run.
 6. Output is validated with `tools/methodology/sanity_csv_schema.validate()`.
    SCHEMA NOTE: the canonical schema is intraday-single-exit compatible and is used
    UNMODIFIED.  `pnl_pct` is the RAW per-share SHORT return computed from
    (entry_price - exit_price)/entry_price*100 on CLEAN prices -- no fees, no slippage,
    no leverage -- per the schema contract.  All cost-loaded numbers live in extra
    columns.  `exit_reason` = "eod" (forced session-close exit).
 7. Lesson #27: `nse_all.json` (used by ProductionUniverseGate for cap_segment +
    mis_enabled) is a CURRENT Zerodha snapshot applied to 2023-2024 sessions.  It is
    anachronistic and survivorship-biased.  Applying it here makes the universe gate
    an UPPER BOUND on point-in-time MIS shortability, not a conservative filter.

COSTS / SIZING CONVENTIONS (stated once, held constant)
------------------------------------------------------
  sizing   : FIXED Rs 100,000 notional per trade -> qty = int(NOTIONAL / entry_price).
             This is the ACTUAL share count the broker holds.  MIS 5x leverage reduces
             the MARGIN required (~Rs 20k blocked), it does NOT multiply the position;
             fees are charged on the full Rs 1L notional on BOTH legs.  Hence
             `calc_fee(..., mis_leverage=1.0)` -- see its docstring / the 2026-05-14
             audit against production analytics.jsonl.  (Lesson #9 concerns sanity
             scripts that size to base qty and then forget the broker holds qty*lev;
             that failure mode does not apply to a fixed-notional convention.)
  fees     : `tools.sub7_validation.build_per_setup_pnl.calc_fee(side="SELL")` --
             Zerodha intraday equity: per-order brokerage min(0.03%, Rs20) x2,
             STT 0.025% on the SELL leg (= the ENTRY leg for a short), NSE txn
             0.00307%, SEBI 1e-6, IPFT 1e-6, stamp 0.003% on the BUY leg (= the EXIT
             leg for a short), GST 18% on (brokerage+txn+SEBI+IPFT).
  slippage : reported at BOTH 20bp and 50bp PER SIDE (brief SS9 risk 1).  Short:
             entry fills LOWER, exit fills HIGHER.  Fees are recomputed on the slipped
             prices.  A 0bp reference column is also emitted.

LEDGER NOTE
-----------
Development-window (Discovery) sanity.  No cell locked, nothing shipped, no OOS/Holdout
or fresh-pool statistic computed -> no `docs/experiment_ledger.jsonl` line required.

FEASIBILITY CORRECTION (2026-07-29) -- `--exit-bar-start`
---------------------------------------------------------
The LOCKED construction above exits at the session close (the 15:25-15:30 5m bar, i.e.
the 15:29 1m print).  `measure_slippage_earnings_downshock.py` flagged that this is
**INFEASIBLE for an MIS short**: Zerodha auto-squares MIS equity positions from ~15:20
(project Stage-1 note in `docs/setup_lifecycle.md`, lesson #4: Upstox/Angel 15:15,
ICICI 15:15-15:20, Zerodha 15:20-15:24).  The position cannot be held to 15:29.

`--exit-bar-start HH:MM` names the START stamp of the 5m bar whose CLOSE is the exit
print (5m bars in the archive are start-stamped; the last bar of a session is 15:25 and
its close IS the 15:30 session close).  So:
    --exit-bar-start 15:10   -> exit at the 15:15 print   <-- the FEASIBLE exit
    --exit-bar-start 15:15   -> exit at the 15:20 print   (sensitivity only)
    (omitted)                -> exit at the session close (the original, INFEASIBLE)

*** THIS IS A FEASIBILITY CORRECTION, NOT AN OPTIMISATION (lesson #31). ***
The corrected exit time is determined by BROKER MECHANICS ALONE: 15:15 is the latest
exit safely inside every relevant broker's pre-square-off window.  It was NOT selected
by sweeping exit times and taking the best.  15:20 and the original session close are
reported as SENSITIVITIES ONLY and are never permitted to become the verdict basis.

MEASURED SLIPPAGE COLUMNS (additive; the locked 0/20/50bp columns are untouched)
--------------------------------------------------------------------------------
`measure_slippage_earnings_downshock.py` measured this exact cohort:
    CENTRAL      18.7 bp/side  (adv_low 23.2 / adv_mid 14.3)
    CONSERVATIVE 27.5 bp/side
Emitted as `net_pct_central` / `net_pct_centralTier` / `net_pct_conservative`.
These are MEASURED rates substituted for the earlier flat guesses -- not new tuning.

Output:
  reports/sub9_sanity/_earnings_downshock_continuation_short_trades_discovery.csv
  ... _trades_discovery_exit<HHMM>.csv   when --exit-bar-start is given

Usage:
  .venv/Scripts/python tools/sub9_research/sanity_earnings_downshock_continuation_short.py
  .venv/Scripts/python tools/sub9_research/sanity_earnings_downshock_continuation_short.py --exit-bar-start 15:10
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "tools" / "sub9_research"))

import screen_event_classes_cost_clearing as S            # noqa: E402
from tools.sub9_research.production_universe import ProductionUniverseGate   # noqa: E402
from tools.sub7_validation.build_per_setup_pnl import calc_fee               # noqa: E402
from tools.methodology import sanity_csv_schema                              # noqa: E402


# =============================================================================
# LOCKED FILTERS -- brief SS4.  DO NOT EDIT AFTER THE FIRST DISCOVERY RUN.
# =============================================================================
LOCKED_FILTERS = {
    "shock_threshold_pct":      -8.0,          # reaction-day close move <= this
    "dedupe_key":               ("symbol", "reaction_date"),
    "entry_bar_index":          0,             # bar 0 = 09:15-09:20
    "entry_price_field":        "close",       # -> the 09:20 print
    "entry_bar_start_time":     "09:15",       # entry bar MUST start at 09:15
    "exit":                     "session_close",   # last 5m bar close, same day
    "stop_loss":                None,          # none -- geometry is Phase 5
    "target":                   None,          # none -- geometry is Phase 5
    "adv_tiers":                ("adv_low", "adv_mid"),
    "exclude_circuit_blocked":  True,          # strict rule (see circuit_blocked_strict)
    "exclude_asm_gsm_on_entry": True,          # any NSE ASM stage or BSE GSM
    "production_universe_gate": True,          # ProductionUniverseGate on ENTRY date
    "min_bars_in_session":      2,             # need an entry bar + >=1 later bar
    "notional_inr":             100_000.0,
    "slippage_bps_per_side":    (0.0, 20.0, 50.0),
}

# =============================================================================
# MEASURED slippage (reports/sub9_sanity/_earnings_downshock_slippage.csv).
# NOT tuned here -- these are the outputs of the measurement run, substituted for the
# 20/50bp guesses above.  Reporting-only additions; nothing selects on them.
# =============================================================================
MEASURED_SLIP = {
    "central_flat_bps":      18.7,
    "conservative_flat_bps": 27.5,
    "central_by_tier_bps":   {"adv_low": 23.2, "adv_mid": 14.3},
}

# ---- Discovery window.  Stage 4 must NOT touch 2025+ at all. ----
DISCOVERY_START = pd.Timestamp("2023-01-01")
DISCOVERY_END   = pd.Timestamp("2024-12-31")

# ---- DEMOTED development window (lifecycle amendment A1).
# 2025-01-01 .. 2026-04-30 is the old OOS + Holdout, demoted to *development* data for
# any NEW candidate.  It is usable ONCE, at Stage 5, on an ALREADY-LOCKED cell -- it is
# NOT sweepable and it is NOT the decisive gate (that is the 2026-05-01+ fresh pool).
DEMOTED_START = pd.Timestamp("2025-01-01")
DEMOTED_END   = pd.Timestamp("2026-04-30")

# HARD FLOOR: the fresh pool. NOTHING in this module may read a signal on/after this.
FRESH_POOL_START = pd.Timestamp("2026-05-01")

WINDOWS = {
    "discovery": (DISCOVERY_START, DISCOVERY_END),
    "demoted":   (DEMOTED_START,   DEMOTED_END),
}

# ---- circuit-band proxy (identical to the gauntlet's STRICT rule; not tuned) ----
PIN_BARS = 3                 # leading bars pinned to one price
MIN_GAP_FOR_BLOCK_PCT = -1.5 # a "band open" must actually be a gap DOWN

# ---- paths ----
FIVEM_DIR = _REPO / "backtest-cache-download" / "monthly"
ASM_PATH  = _REPO / "data" / "asm_gsm_history" / "asm_gsm_events.parquet"
EARN_PATH = _REPO / "data" / "earnings_calendar" / "earnings_events.parquet"
OUT_PATH  = (_REPO / "reports" / "sub9_sanity"
             / "_earnings_downshock_continuation_short_trades_discovery.csv")

SETUP_NAME = "earnings_downshock_continuation_short"

# Cap segments accepted by the gate.  The brief puts NO cap restriction on the
# universe (it is an ADV-tier universe), so every segment production knows about is
# accepted; the binding part of the gate is `require_mis=True`.
ACCEPTED_CAPS = {"large_cap", "mid_cap", "small_cap", "micro_cap", "unknown"}

FUNNEL: list[tuple[str, int, str]] = []


def funnel(step: str, n: int, note: str = "") -> None:
    FUNNEL.append((step, n, note))
    print(f"  [funnel] {step:<46s} n={n:6d}  {note}", flush=True)


# =============================================================================
# 1. Event construction  (mirrors the gauntlet's build_events EXACTLY, + dedupe)
# =============================================================================
def build_events(px: pd.DataFrame, meta: dict) -> pd.DataFrame:
    df = pd.read_parquet(EARN_PATH)
    funnel("earnings filings on disk (raw rows)", len(df))

    df["announce_date"] = pd.to_datetime(df["announce_date"], errors="coerce")
    df["announce_time"] = pd.to_datetime(df["announce_time"], errors="coerce")
    df["symbol"] = S.normalise_symbol(df["symbol"])

    # ---- reaction-day assignment (announce_time_class) ----
    # GUARD 1/3: this is a PRIOR-session decision variable.  The reaction-day close is
    # stamped 15:30 of the session BEFORE the entry session -> causal by construction.
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
    funnel("filings matched to a reaction session", len(m))

    shock = m[m["react_move"] <= LOCKED_FILTERS["shock_threshold_pct"]].copy()
    funnel(f"reaction move <= {LOCKED_FILTERS['shock_threshold_pct']}% (as-fired)",
           len(shock), "duplicate filings NOT yet removed")

    # ---- MANDATORY DE-DUPLICATION (brief SS7.1) ----
    dedup = (shock.sort_values(["symbol", "signal_date"])
                  .drop_duplicates(["symbol", "signal_date"], keep="first"))
    funnel("de-duplicated (symbol, reaction_date)", len(dedup),
           f"{len(shock) - len(dedup)} duplicate filings dropped")

    # ---- attach entry session (strictly AFTER the reaction date) + adv20 ----
    ev = S.attach_returns(dedup, px, meta)
    ev = ev.drop(columns=[c for c in ev.columns
                          if c.startswith(("ret_h", "raw_h", "base_h"))], errors="ignore")
    funnel("with a tradeable T+1 entry session", len(ev))
    return ev


# =============================================================================
# 2. ADV terciles  (same construction as the gauntlet / the screen's adv_split)
# =============================================================================
def add_adv_tiers(ev: pd.DataFrame, cuts: tuple | None = None) -> tuple[pd.DataFrame, tuple]:
    """Terciles of PRIOR-ONLY adv20 within the event set.

    GUARD 2: `adv20` is rolling(20).mean().shift(1) of turnover -- the reaction day
    itself is excluded from its own baseline.
    CAVEAT (documented, not hidden): the two tercile CUT POINTS are computed across the
    whole Discovery event set, so they use in-window cross-sectional information a
    live system would not have on day 1.  A point-in-time alternative (`adv_tier_x`,
    the panel's per-date cross-sectional tercile) is reported as a robustness line in
    `main`.  The brief's "low+mid ADV" refers to the event-set tercile, so that is what
    is LOCKED here.

    `cuts` -- when given, the (q33, q66) rupee-turnover cut points are IMPOSED rather
    than recomputed.  Stage-5 Step 3 passes the FROZEN Discovery cuts so that "run the
    locked cell UNCHANGED on the demoted window" means literally unchanged: the tier
    definition itself is part of the lock and must not be re-fitted on the new window.
    A second column `adv_tier_win` always carries the within-window tercile so the
    re-fitted variant can be reported as a robustness line without a second run.
    """
    ev = ev.copy()
    d = ev["adv20"]
    q_win = tuple(d.quantile([0.33, 0.66]).to_list())
    q = tuple(cuts) if cuts is not None else q_win

    def _tier(cut):
        return np.where(d.isna(), "adv_na",
                 np.where(d <= cut[0], "adv_low",
                 np.where(d <= cut[1], "adv_mid", "adv_high")))

    ev["adv_tier"] = _tier(q)
    ev["adv_tier_win"] = _tier(q_win)
    ev["lowmid"] = ev["adv_tier"].isin(LOCKED_FILTERS["adv_tiers"])
    ev["lowmid_win"] = ev["adv_tier_win"].isin(LOCKED_FILTERS["adv_tiers"])
    ev["adv_tier_x_lowmid"] = ev["adv_tier_x"].isin([0.0, 1.0])
    return ev, q


# =============================================================================
# 3. 5m bars for the entry sessions only
# =============================================================================
def load_entry_bars(ev: pd.DataFrame) -> pd.DataFrame:
    want = set(zip(ev["symbol"], ev["entry_date"]))
    months = sorted({f"{d.year}_{d.month:02d}" for _, d in want})
    syms = {s for s, _ in want}
    out = []
    for mo in months:
        p = FIVEM_DIR / f"{mo}_5m_enriched.feather"
        if not p.exists():
            print(f"    WARN missing 5m feather: {p.name}")
            continue
        d = pd.read_feather(p, columns=["date", "symbol", "open", "close",
                                        "high", "low", "volume"])
        d = d[d["symbol"].isin(syms)]
        if d.empty:
            continue
        d["session"] = pd.to_datetime(d["date"]).dt.normalize()
        d = d[[t in want for t in zip(d["symbol"], d["session"])]]
        if not d.empty:
            out.append(d)
    if not out:
        return pd.DataFrame()
    return (pd.concat(out, ignore_index=True)
              .sort_values(["symbol", "date"]).reset_index(drop=True))


def circuit_blocked_strict(b: pd.DataFrame, react_close: float) -> dict:
    """Gauntlet STRICT circuit-block proxy on the ENTRY session.

    NOTE ON GUARD 1 (full-day aggregates): `locked_all_day` reads the whole session.
    That is intentional and is NOT look-ahead in the harmful sense -- it is an
    EXCLUSION (it can only remove trades, never admit one), and a session in which the
    stock never traded away from one price is a session in which our SHORT could not
    have been filled or covered at all.  Modelling it as a trade would be the bias.
    Every ADMITTING condition in this script is 09:20-or-earlier information.
    """
    o = float(b["open"].iloc[0])
    hi1, lo1 = float(b["high"].iloc[0]), float(b["low"].iloc[0])
    v1 = float(b["volume"].iloc[0])
    gap = (o / react_close - 1.0) * 100.0
    day_hi, day_lo = float(b["high"].max()), float(b["low"].min())

    b1_zero_range = bool(hi1 == lo1 and v1 > 0)
    pinned, px_ref = 0, hi1
    for _, r2 in b.iterrows():
        if r2["high"] == r2["low"] == px_ref:
            pinned += 1
        else:
            break
    locked_all_day = bool(day_hi == day_lo)
    blocked = bool(locked_all_day
                   or (b1_zero_range and pinned >= PIN_BARS
                       and gap <= MIN_GAP_FOR_BLOCK_PCT))
    return dict(open_gap_pct=gap, b1_zero_range=b1_zero_range, pinned_bars=pinned,
                locked_all_day=locked_all_day, entry_bar_volume=v1,
                circuit_blocked=blocked)


# =============================================================================
# 4. Surveillance (NSE ASM any stage / BSE GSM) membership on the ENTRY date
# =============================================================================
def surveillance_flags(ev: pd.DataFrame,
                       win_start: pd.Timestamp = DISCOVERY_START,
                       win_end: pd.Timestamp = DISCOVERY_END) -> pd.DataFrame:
    asm = pd.read_parquet(ASM_PATH)
    asm["date"] = pd.to_datetime(asm["date"])
    asm = asm[(asm["date"] >= win_start - pd.Timedelta(days=5))
              & (asm["date"] <= win_end + pd.Timedelta(days=5))]

    nse = asm[asm["exchange"] == "NSE"].copy()
    nse["symbol"] = S.normalise_symbol(nse["symbol"])
    order = ["0", "I", "II", "III", "IV", "VI"]
    nse_mem = nse.groupby(["symbol", "date"]).agg(
        asm_program=("surveillance_program", "first"),
        asm_stage=("stage", lambda s: max(
            s, key=lambda x: order.index(str(x)) if str(x) in order else 0)),
    ).reset_index().rename(columns={"date": "entry_date"})

    ev = ev.merge(nse_mem, on=["symbol", "entry_date"], how="left")
    ev["in_asm_nse"] = ev["asm_program"].notna()
    ev["asm_t2t"] = ev["asm_stage"].isin(["III", "IV"])

    bse = asm[(asm["exchange"] == "BSE") & (asm["surveillance_program"] == "GSM")]
    bse_key = set(zip(bse["isin"].astype(str), bse["date"]))
    ev["in_gsm_bse"] = [(str(i), d) in bse_key
                        for i, d in zip(ev["isin"].astype(str), ev["entry_date"])]
    ev["surv_any"] = ev["in_asm_nse"] | ev["in_gsm_bse"]
    return ev


# =============================================================================
# 5. Metrics
# =============================================================================
def matched_panel_baseline(entry_sessions: set) -> pd.DataFrame:
    """Panel-wide mean 09:20->close return per (session, ADV tercile).

    DIAGNOSTIC ONLY -- gates nothing.  This is the SAME control the abnormal-return
    screen subtracts.  Recomputing it on exactly our simulated trades lets us
    decompose the Stage-4 raw return into (event-specific abnormal drift) +
    (unconditional panel drift the screen removed but a real short book collects),
    so the delta vs the screen's +0.48% is measured rather than asserted.
    """
    months = sorted({f"{d.year}_{d.month:02d}" for d in entry_sessions})
    out = []
    for mo in months:
        p = FIVEM_DIR / f"{mo}_5m_enriched.feather"
        if not p.exists():
            continue
        d = pd.read_feather(p, columns=["date", "symbol", "close"])
        d["session"] = pd.to_datetime(d["date"]).dt.normalize()
        d = d[d["session"].isin(entry_sessions)]
        if d.empty:
            continue
        d = d.sort_values(["symbol", "date"])
        red = d.groupby(["symbol", "session"], sort=False).agg(
            b1c=("close", "first"), c=("close", "last"), nb=("close", "size")
        ).reset_index()
        out.append(red[red["nb"] >= 2])
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


def pf_of(s: pd.Series) -> float:
    g = float(s[s > 0].sum())
    l = float(-s[s < 0].sum())
    if l <= 0:
        return float("inf") if g > 0 else float("nan")
    return g / l


def block(tag: str, df: pd.DataFrame, pnl_col: str, pct_col: str) -> dict:
    p = df[pnl_col]
    q = df[pct_col]
    n = len(df)
    sd = float(q.std(ddof=1)) if n > 1 else np.nan
    return dict(
        tag=tag, n=n,
        total_inr=float(p.sum()),
        exp_inr=float(p.mean()) if n else np.nan,
        exp_pct=float(q.mean()) if n else np.nan,
        median_pct=float(q.median()) if n else np.nan,
        pf=pf_of(p),
        win=float((p > 0).mean()) if n else np.nan,
        tstat=(float(q.mean()) / (sd / np.sqrt(n))) if (n > 1 and sd and sd > 0) else np.nan,
    )


def show(b: dict) -> None:
    print(f"  {b['tag']:<34s} n={b['n']:4d}  total=Rs{b['total_inr']:>11,.0f}  "
          f"exp={b['exp_pct']:+.3f}% / Rs{b['exp_inr']:+8.1f}  "
          f"PF={b['pf']:.3f}  win={b['win']:.3f}  t={b['tstat']:+.2f}")


# =============================================================================
# COHORT BUILDER -- shared by the Stage-4 sanity (`main`) and the Stage-5 sweep driver
# (`phase5_sweep_earnings_downshock_continuation_short.py`).  Extracted verbatim from
# `main` so there is exactly ONE implementation of the LOCKED construction.
# =============================================================================
def build_cohort(window: str = "discovery", adv_cuts: tuple | None = None):
    """Return (px, ev, bars_by_key, adv_cuts_used) for the requested window.

    `ev` is the fully-filtered event set: reaction move <= -8%, de-duplicated,
    tradeable T+1 entry, ADV tier in (low, mid), ProductionUniverseGate PASS on the
    entry date, no NSE ASM / BSE GSM on the entry date.  Circuit-blocking and the
    bar-level checks happen at simulation time (they need the bars).
    `bars_by_key` maps (symbol, entry_date) -> the session's 5m bars.
    """
    if window not in WINDOWS:
        raise ValueError(f"unknown window {window!r}; expected one of {list(WINDOWS)}")
    win_start, win_end = WINDOWS[window]
    if win_end >= FRESH_POOL_START:
        raise RuntimeError("window would touch the 2026-05-01+ fresh pool -- refused")

    print(f"\n--- loading CA-adjusted / bad-print-cleaned daily panel ---", flush=True)
    px, meta = S.load_prices()
    print(f"  panel {px.shape}  {px['date'].min().date()} -> {px['date'].max().date()}")

    print(f"\n--- FUNNEL  [window={window}: "
          f"{win_start.date()} .. {win_end.date()}] ---")
    ev = build_events(px, meta)

    # ---------- window enforcement ----------
    ev = ev[(ev["signal_date"] >= win_start) & (ev["signal_date"] <= win_end)]
    funnel(f"reaction_date in {window}", len(ev))
    n_spill = int((ev["entry_date"] > win_end).sum())
    ev = ev[ev["entry_date"] <= win_end].copy()
    funnel("entry_date also inside the window", len(ev),
           f"{n_spill} dropped: entry spilled past {win_end.date()}")

    if len(ev) == 0:
        raise RuntimeError(f"no {window} events -- abort")
    assert ev["signal_date"].min() >= win_start, "window violation (signal min)"
    assert ev["signal_date"].max() <= win_end, "window violation (signal max)"
    assert ev["entry_date"].max() <= win_end, "window violation (entry max)"
    assert ev["entry_date"].max() < FRESH_POOL_START, "FRESH POOL TOUCHED -- abort"
    assert (ev["entry_date"] > ev["signal_date"]).all(), "LOOKAHEAD: entry <= signal"
    print(f"  [assert] {window}-only window + causal entry + fresh pool untouched: OK")

    # ---------- ADV tiers ----------
    ev, cuts_used = add_adv_tiers(ev, cuts=adv_cuts)
    print(f"  [adv] tercile cut points used (Rs turnover): "
          f"q33={cuts_used[0]:,.0f}  q66={cuts_used[1]:,.0f}"
          f"{'  (FROZEN from Discovery)' if adv_cuts is not None else '  (within-window)'}")
    n_pre_adv = len(ev)
    ev = ev[ev["lowmid"]].copy()
    funnel("ADV tier in (low, mid)", len(ev), f"from {n_pre_adv} all-ADV events")

    # ---------- ProductionUniverseGate on the ENTRY date ----------
    gate = ProductionUniverseGate(
        accepted_caps=ACCEPTED_CAPS,
        require_mis=True,               # intraday MIS short
        min_trading_days_required=0,    # Lesson #17
        min_daily_avg_volume=0,         # Lesson #17
    )
    keep, caps = [], []
    for sym, ed in zip(ev["symbol"], ev["entry_date"]):
        keep.append(gate.is_eligible(sym, ed.date()))
        caps.append(gate._cap_segment(sym))
    ev["cap_segment"] = caps
    ev["universe_ok"] = keep
    n_pre = len(ev)
    ev = ev[ev["universe_ok"]].copy()
    funnel("ProductionUniverseGate(entry_date) PASS", len(ev),
           f"{n_pre - len(ev)} rejected (cap/MIS; lesson #27 = upper bound)")

    # ---------- surveillance ----------
    ev = surveillance_flags(ev, win_start, win_end)
    n_pre = len(ev)
    n_t2t = int(ev["asm_t2t"].sum())
    ev = ev[~ev["surv_any"]].copy()
    funnel("no NSE ASM / BSE GSM on entry date", len(ev),
           f"{n_pre - len(ev)} excluded ({n_t2t} were Stage III/IV T2T)")

    # ---------- 5m bars ----------
    print("\n--- loading 5m bars for entry sessions ---", flush=True)
    bars = load_entry_bars(ev)
    print(f"  bars {bars.shape}  symbols={bars['symbol'].nunique()}")
    g = {k: v for k, v in bars.groupby(["symbol", "session"], sort=False)}
    return px, ev, g, cuts_used


# =============================================================================
def main(exit_bar_start: str | None = None, window: str = "discovery") -> None:
    print("=" * 96)
    print(f"STAGE 4 SANITY -- {SETUP_NAME}  (REAL P&L SIMULATION, window={window})")
    print("=" * 96)
    if exit_bar_start is None:
        print("  EXIT = session close (last 5m bar close = the 15:30 print).")
        print("  *** INFEASIBLE for an MIS short: Zerodha auto-squares from ~15:20. ***")
        out_path = OUT_PATH
    else:
        et = pd.Timestamp(f"1900-01-01 {exit_bar_start}") + pd.Timedelta(minutes=5)
        print(f"  EXIT = close of the {exit_bar_start} 5m bar (the {et.strftime('%H:%M')} print).")
        print("  FEASIBILITY CORRECTION (lesson #31): fixed by broker MIS square-off")
        print("  mechanics, NOT selected by sweeping exit times.")
        out_path = OUT_PATH.with_name(
            OUT_PATH.stem + "_exit" + exit_bar_start.replace(":", "") + OUT_PATH.suffix)
    if window != "discovery":
        out_path = out_path.with_name(out_path.stem.replace("_discovery", f"_{window}")
                                      + out_path.suffix)

    px, ev, g, _cuts = build_cohort(window)

    # =====================================================================
    # SIMULATION
    # =====================================================================
    NOTIONAL = LOCKED_FILTERS["notional_inr"]
    ENTRY_T = pd.Timestamp(f"1900-01-01 {LOCKED_FILTERS['entry_bar_start_time']}").time()
    EXIT_T = (pd.Timestamp(f"1900-01-01 {exit_bar_start}").time()
              if exit_bar_start is not None else None)
    rows, n_nobars, n_short_sess, n_badstart, n_badqty, n_circuit = [], 0, 0, 0, 0, 0
    n_noexitbar = 0

    for r in ev.itertuples(index=False):
        b = g.get((r.symbol, r.entry_date))
        if b is None or len(b) == 0:
            n_nobars += 1
            continue
        b = b.sort_values("date").reset_index(drop=True)
        if len(b) < LOCKED_FILTERS["min_bars_in_session"]:
            n_short_sess += 1
            continue
        if b["date"].iloc[0].time() != ENTRY_T:
            # truncated session feed -- "the 09:20 print" would not be the 09:20 print
            n_badstart += 1
            continue

        cf = circuit_blocked_strict(b, float(r.react_close))
        if cf["circuit_blocked"]:
            n_circuit += 1
            continue

        # ---- ENTRY: close of bar 0 (09:20).  GUARD 3: walk starts at bar 1. ----
        entry_price = float(b["close"].iloc[0])
        entry_ts = b["date"].iloc[0]

        # ---- EXIT ----
        if EXIT_T is None:
            exit_idx = len(b) - 1
        else:
            # the bar START-stamped EXIT_T; its close IS the exit print.
            hit = np.flatnonzero(b["date"].dt.time.values == EXIT_T)
            if hit.size == 0:
                n_noexitbar += 1
                continue
            exit_idx = int(hit[0])
        if exit_idx <= 0:                      # exit bar must be strictly after entry
            n_noexitbar += 1
            continue
        exit_price = float(b["close"].iloc[exit_idx])
        exit_ts = b["date"].iloc[exit_idx]
        if not (entry_price > 0 and exit_price > 0):
            n_badqty += 1
            continue
        qty = int(NOTIONAL // entry_price)
        if qty < 1:
            n_badqty += 1
            continue

        path = b.iloc[1:exit_idx + 1]
        path_hi = float(path["high"].max())
        path_lo = float(path["low"].min())
        # SHORT: favorable = price falls
        mfe_pct = (entry_price - path_lo) / entry_price * 100.0
        mae_pct = (path_hi - entry_price) / entry_price * 100.0

        raw_pct = (entry_price - exit_price) / entry_price * 100.0   # schema contract
        rec = dict(
            # --- schema-required ---
            signal_date=r.signal_date.date(),
            symbol=f"NSE:{r.symbol}",
            side="SHORT",
            entry_price=round(entry_price, 4),
            exit_price=round(exit_price, 4),
            qty=qty,
            pnl_pct=raw_pct,
            exit_reason="eod",
            same_bar=False,
            # --- deliverable / diagnostics ---
            reaction_date=r.signal_date.date(),
            entry_date=r.entry_date.date(),
            entry_ts=entry_ts,
            exit_ts=exit_ts,
            react_move_pct=round(float(r.react_move), 4),
            react_close=round(float(r.react_close), 4),
            open_gap_pct=round(cf["open_gap_pct"], 4),
            entry_bar_volume=cf["entry_bar_volume"],
            adv20=float(r.adv20) if pd.notna(r.adv20) else np.nan,
            adv_tier=r.adv_tier,
            adv_tier_x=float(r.adv_tier_x) if pd.notna(r.adv_tier_x) else np.nan,
            adv_tier_x_lowmid=bool(r.adv_tier_x_lowmid),
            cap_segment=r.cap_segment,
            announce_time_class=r.announce_time_class,
            year=int(r.signal_date.year),
            mfe_pct=round(mfe_pct, 4),
            mae_pct=round(mae_pct, 4),
            n_bars=len(b),
            exit_bar_start=exit_ts.strftime("%H:%M"),
            n_bars_held=exit_idx,
            # --- exclusion flags (all False by construction; kept for audit) ---
            flag_circuit_blocked=False,
            flag_locked_all_day=bool(cf["locked_all_day"]),
            flag_b1_zero_range=bool(cf["b1_zero_range"]),
            flag_zero_vol_entry_bar=bool(cf["entry_bar_volume"] <= 0),
            flag_asm_nse=False,
            flag_gsm_bse=False,
            flag_universe_gate_pass=True,
            # metadata only -- MUST NOT be used as a filter dimension
            day_high=round(float(b["high"].max()), 4),
            day_low=round(float(b["low"].min()), 4),
            day_close=round(float(b["close"].iloc[-1]), 4),
            session_close=round(float(b["close"].iloc[-1]), 4),
        )

        # ---- cost stack at each slippage assumption ----
        for bps in LOCKED_FILTERS["slippage_bps_per_side"]:
            s = bps / 10_000.0
            e_fill = entry_price * (1.0 - s)     # short sells LOWER
            x_fill = exit_price * (1.0 + s)      # covers HIGHER
            gross = (e_fill - x_fill) * qty
            fee = calc_fee(e_fill, x_fill, qty, "SELL", 1.0)
            net = gross - fee
            sfx = f"{int(bps)}bp"
            rec[f"gross_pnl_{sfx}"] = round(gross, 2)
            rec[f"fees_{sfx}"] = round(fee, 2)
            rec[f"net_pnl_{sfx}"] = round(net, 2)
            rec[f"net_pct_{sfx}"] = round(net / (entry_price * qty) * 100.0, 5)

        # ---- MEASURED slippage variants (additive; nothing above is changed) ----
        for tag, bps in (
            ("central",      MEASURED_SLIP["central_flat_bps"]),
            ("conservative", MEASURED_SLIP["conservative_flat_bps"]),
            ("centralTier",  MEASURED_SLIP["central_by_tier_bps"].get(r.adv_tier,
                             MEASURED_SLIP["central_flat_bps"])),
        ):
            s = bps / 10_000.0
            e_fill = entry_price * (1.0 - s)
            x_fill = exit_price * (1.0 + s)
            gross = (e_fill - x_fill) * qty
            fee = calc_fee(e_fill, x_fill, qty, "SELL", 1.0)
            net = gross - fee
            rec[f"slip_bps_{tag}"] = bps
            rec[f"gross_pnl_{tag}"] = round(gross, 2)
            rec[f"fees_{tag}"] = round(fee, 2)
            rec[f"net_pnl_{tag}"] = round(net, 2)
            rec[f"net_pct_{tag}"] = round(net / (entry_price * qty) * 100.0, 5)

        # canonical aliases (0bp = no-slippage reference)
        rec["gross_pnl"] = rec["gross_pnl_0bp"]
        rec["fees"] = rec["fees_20bp"]
        rec["net_pnl"] = rec["net_pnl_20bp"]
        rec["net_pct"] = rec["net_pct_20bp"]
        # schema rupee cross-check triple (0bp basis): net == realized - fee
        rec["realized_pnl_inr"] = rec["gross_pnl_0bp"]
        rec["fee_inr"] = rec["fees_0bp"]
        rec["net_pnl_inr"] = rec["net_pnl_0bp"]
        rows.append(rec)

    funnel("5m entry session present", len(ev) - n_nobars, f"{n_nobars} missing")
    funnel("session >=2 bars and starts 09:15", len(ev) - n_nobars - n_short_sess - n_badstart,
           f"{n_short_sess} too short, {n_badstart} bad start bar")
    funnel("not circuit-blocked (strict)", len(ev) - n_nobars - n_short_sess
           - n_badstart - n_circuit, f"{n_circuit} blocked")
    funnel("exit bar present", len(ev) - n_nobars - n_short_sess - n_badstart
           - n_circuit - n_noexitbar,
           f"{n_noexitbar} missing the "
           f"{exit_bar_start if exit_bar_start else 'session-close'} exit bar")
    funnel("SIMULATED TRADES", len(rows), f"{n_badqty} dropped on price/qty")

    tr = pd.DataFrame(rows)
    if tr.empty:
        raise RuntimeError("zero simulated trades")

    # =====================================================================
    # SCHEMA VALIDATION
    # =====================================================================
    print("\n--- schema validation (tools/methodology/sanity_csv_schema.py) ---")
    res = sanity_csv_schema.validate(tr, setup_name=SETUP_NAME, layer="filtered_trades")
    print("  " + res.summary().replace("\n", "\n  "))
    if not res.is_valid:
        raise RuntimeError("SCHEMA VALIDATION FAILED")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tr.to_csv(out_path, index=False)
    print(f"\n  wrote {out_path}  rows={len(tr)}")

    # =====================================================================
    # RESULTS
    # =====================================================================
    print("\n" + "=" * 96)
    print("RESULTS -- REAL P&L SIMULATION (Rs 100,000 notional/trade, Zerodha MIS short)")
    print("=" * 96)

    print(f"\n  n={len(tr)}   symbols={tr['symbol'].nunique()}   "
          f"sessions={tr['entry_date'].nunique()}   "
          f"top-5 symbol share={tr['symbol'].value_counts().head(5).sum()/len(tr):.3f}")
    print(f"  turnover deployed = Rs {(tr['entry_price']*tr['qty']).sum():,.0f}")

    print("\n  GROSS (per-share, before any cost):")
    gb = block("gross_raw", tr.assign(_r=tr["pnl_pct"] * tr["entry_price"] * tr["qty"] / 100.0),
               "_r", "pnl_pct")
    show(gb)

    print("\n  NET by slippage assumption (locked 0/20/50bp reference grid):")
    nets = {}
    for bps in (0, 20, 50):
        nets[bps] = block(f"NET @ {bps}bp/side", tr, f"net_pnl_{bps}bp", f"net_pct_{bps}bp")
        show(nets[bps])

    print("\n  NET at MEASURED slippage (_earnings_downshock_slippage.csv):")
    meas = {}
    for tag, lbl in (("central",      f"CENTRAL {MEASURED_SLIP['central_flat_bps']}bp/side"),
                     ("centralTier",  "CENTRAL per-ADV-tier (23.2/14.3)"),
                     ("conservative", f"CONSERV {MEASURED_SLIP['conservative_flat_bps']}bp/side")):
        meas[tag] = block(lbl, tr, f"net_pnl_{tag}", f"net_pct_{tag}")
        show(meas[tag])
    print("\n  falsifier #1 (brief SS6.1): kill if net < +0.15%/trade")
    for tag in ("central", "centralTier", "conservative"):
        v = meas[tag]["exp_pct"]
        print(f"    {tag:<14s} net {v:+.4f}%/trade -> "
              f"{'PASS' if v >= 0.15 else 'FAIL'}")

    print("\n  PER YEAR at MEASURED slippage:")
    for yr, sub in tr.groupby("year"):
        for tag in ("central", "conservative"):
            show(block(f"{yr} NET {tag}", sub, f"net_pnl_{tag}", f"net_pct_{tag}"))

    print("\n  Cost decomposition (mean per trade, Rs):")
    for bps in (0, 20, 50):
        slip = tr[f"gross_pnl_0bp"] - tr[f"gross_pnl_{bps}bp"]
        print(f"    {bps:2d}bp/side: fees Rs{tr[f'fees_{bps}bp'].mean():6.1f}  "
              f"slippage Rs{slip.mean():6.1f}  "
              f"total cost Rs{(tr[f'fees_{bps}bp'] + slip).mean():6.1f}  "
              f"= {(tr[f'fees_{bps}bp'] + slip).mean() / (tr['entry_price']*tr['qty']).mean() * 100:.3f}% of notional")

    print("\n  PER YEAR (2023 vs 2024):")
    for yr, sub in tr.groupby("year"):
        for bps in (20, 50):
            b = block(f"{yr} NET @{bps}bp", sub, f"net_pnl_{bps}bp", f"net_pct_{bps}bp")
            show(b)

    print("\n  MFE / MAE distribution (%, SHORT convention):")
    for c in ("mfe_pct", "mae_pct"):
        q = tr[c].quantile([0.10, 0.25, 0.50, 0.75, 0.90])
        print(f"    {c}: mean {tr[c].mean():.3f}  p10 {q[0.10]:.3f}  p25 {q[0.25]:.3f}  "
              f"p50 {q[0.50]:.3f}  p75 {q[0.75]:.3f}  p90 {q[0.90]:.3f}  max {tr[c].max():.3f}")
    print(f"    mfe>=1%: {(tr['mfe_pct']>=1).mean():.3f}   mfe>=2%: {(tr['mfe_pct']>=2).mean():.3f}   "
          f"mae>=2%: {(tr['mae_pct']>=2).mean():.3f}   mae>=4%: {(tr['mae_pct']>=4).mean():.3f}")

    print("\n  ROBUSTNESS (reported, NOT used to select):")
    ptp = tr[tr["adv_tier_x_lowmid"]]
    b = block("point-in-time ADV tier low+mid", ptp, "net_pnl_20bp", "net_pct_20bp")
    show(b)
    nz = tr[~tr["flag_zero_vol_entry_bar"]]
    show(block("excl. zero-volume entry bar", nz, "net_pnl_20bp", "net_pct_20bp"))
    for cs, sub in tr.groupby("cap_segment"):
        if len(sub) >= 10:
            show(block(f"cap={cs}", sub, "net_pnl_20bp", "net_pct_20bp"))
    for at, sub in tr.groupby("adv_tier"):
        show(block(f"adv_tier={at}", sub, "net_pnl_20bp", "net_pct_20bp"))

    # =====================================================================
    # DELTA vs THE ABNORMAL-RETURN SCREEN (diagnostic; gates nothing)
    # =====================================================================
    print("\n" + "-" * 96)
    print("DELTA vs the abnormal-return screen (+0.481% era_A, low+mid, tradeable)")
    print("-" * 96)
    sessions = set(pd.to_datetime(tr["entry_date"]).unique())
    pan = matched_panel_baseline(sessions)
    if not pan.empty:
        tiers = px[["symbol", "date", "adv_tier_x"]].rename(columns={"date": "session"})
        pan = pan.merge(tiers, on=["symbol", "session"], how="left")
        pan = pan[pan["adv_tier_x"].notna() & (pan["b1c"] > 0)]
        pan["leg"] = (pan["c"] / pan["b1c"] - 1.0) * 100.0
        base = (pan.groupby(["session", "adv_tier_x"])["leg"].mean()
                   .rename("base_leg").reset_index())
        t2 = tr.copy()
        t2["session"] = pd.to_datetime(t2["entry_date"])
        t2 = t2.merge(base, on=["session", "adv_tier_x"], how="left")
        cov = float(t2["base_leg"].notna().mean())
        t2 = t2[t2["base_leg"].notna()]
        raw_short = float(t2["pnl_pct"].mean())
        base_short = float(-t2["base_leg"].mean())
        abn = raw_short - base_short
        print(f"  baseline coverage: {cov:.3f}  ({len(t2)}/{len(tr)} trades matched; "
              f"{pan['symbol'].nunique()} panel symbols over {len(sessions)} sessions)")
        print(f"  A. Stage-4 RAW gross short return (09:20 -> close)      {raw_short:+.3f}%")
        print(f"  B. matched panel baseline, short sign (what the screen")
        print(f"     subtracts and a real short book actually collects)   {base_short:+.3f}%")
        print(f"  C. ABNORMAL gross = A - B  (screen's basis)             {abn:+.3f}%")
        print(f"     screen reported ~+0.791% era_A gross-abnormal 09:20->close")
        print(f"  D. screen NET  = C - flat 0.31% notional cost           {abn - 0.31:+.3f}%")
        print(f"     screen reported +0.481%")
        print(f"  E. REAL fees @0bp slip                                  "
              f"{-(tr['fees_0bp']/(tr['entry_price']*tr['qty'])*100).mean():+.3f}%")
        print(f"  F. REAL slippage 20bp/side                              "
              f"{-((tr['gross_pnl_0bp']-tr['gross_pnl_20bp'])/(tr['entry_price']*tr['qty'])*100).mean():+.3f}%")
        print(f"  G. REAL slippage 50bp/side                              "
              f"{-((tr['gross_pnl_0bp']-tr['gross_pnl_50bp'])/(tr['entry_price']*tr['qty'])*100).mean():+.3f}%")
        print(f"  => Stage-4 NET @20bp = A + E + F = {nets[20]['exp_pct']:+.3f}%   "
              f"(screen said {abn - 0.31:+.3f}%)")
        print(f"  => Stage-4 NET @50bp = A + E + G = {nets[50]['exp_pct']:+.3f}%")

    print("\n  FUNNEL SUMMARY")
    for step, n, note in FUNNEL:
        print(f"    {step:<48s} {n:6d}   {note}")

    print("\n" + "=" * 96)
    print("PRODUCTION FIDELITY STILL MISSING AT STAGE 4 (do NOT quote as forward P&L):")
    print("  * no slot caps / concurrent-position limit  * no capital budget or margin check")
    print("  * no cross-setup ranking or priority        * no per-day trade cap")
    print("  * no point-in-time MIS list (lesson #27)    * no borrow/SLB or broker short-ban check")
    print("  * no OCI pipeline parity run                * no regime/index-direction gate")
    print("  Lesson #30: this is an unconstrained per-signal ledger, not a book.")
    print("=" * 96)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--exit-bar-start", default=None, metavar="HH:MM",
        help="START stamp of the 5m bar whose CLOSE is the exit print. "
             "15:10 -> exit at the 15:15 print (the FEASIBLE MIS exit). "
             "Omit to reproduce the original session-close exit.")
    ap.add_argument(
        "--window", default="discovery", choices=sorted(WINDOWS),
        help="'discovery' = 2023-01..2024-12 (Stage 4). "
             "'demoted' = 2025-01..2026-04, the A1-demoted development window; "
             "Stage-5 Step-3 ONE-SHOT on an already-locked cell only.")
    a = ap.parse_args()
    main(exit_bar_start=a.exit_bar_start, window=a.window)
