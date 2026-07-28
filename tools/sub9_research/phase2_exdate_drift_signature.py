"""phase2_exdate_drift_signature.py — Stage 2 (Phase-2 empirical signature) for
`exdate_drift_short` (specs/2026-07-28-brief-exdate_drift_short.md §5, BINDING grid).

Construction (fixed to the validated post_split_bonus_short core):
  Signal day D = corporate-action ex-date. Entry D+1 open, SHORT.
  Exit close of D+1 / D+3 / D+5 (hold 1/3/5 sessions; session 1 = D+1).
  Statistic: mean signed SHORT drift = (entry_open - exit_close)/entry_open * 100
  (positive = profitable short), vs same-universe unconditional baseline over the
  matched (era, hold, ADV-tier). RAW percent, no fees, no leverage (Phase-2).

CAUSALITY GUARDS (documented per lifecycle Stage 2 + Lesson #5):
  1. ADV tier: turnover ADV20 = rolling(20).mean().shift(1) — uses data through
     D-1 only; the tier is known before the D+1 entry. Cross-sectional quintile
     is computed within-date (no future dates involved).
  2. Salience: split/bonus ratio parsed from the corporate announcement subject
     (public well before ex-date); dividend yield = announced dividend amount /
     UNADJUSTED close on the session before D (consolidated_daily; known pre-entry).
  3. No post-signal data in any filter. Exits are pure fixed-horizon closes.
  4. Synthetic-bar filter (reused from sanity_post_split_bonus_short): a daily bar
     with O==H==L==C is a synthetic / halted / circuit artifact. The primary
     ("real_bar") variant requires the D+1 ENTRY bar to be real; the "incl_synth"
     variant keeps all — falsifier #3 compares the two.
  5. Signals (ex-dates) capped at 2026-04-30. Fresh pool 2026-05-01+ NEVER touched
     as signals; forward exit legs of late-Apr-2026 events may extend into May 2026
     (brief-sanctioned, exits only).
  6. Era split per amendment A5 is mandatory: every cell reported per-era, never
     pooled-only.

Data:
  Events (split/bonus): data/corporate_actions/split_bonus_events.parquet if
    present, else _tmp_split_bonus_events.parquet (fallback, reported).
  Events (dividends):   data/dividend_ex_date/dividend_events.parquet
    (FILTERED to ex_date <= 2026-04-30 regardless of file contents — a top-up
    scrape is appending May-Jul rows).
  Prices (drift):       cache/preaggregate/clean_daily_from5m.feather (CA-adjusted, A3).
  Prices (div yield):   cache/preaggregate/consolidated_daily.feather (UNADJUSTED —
    correct economic denominator for yield-at-the-time).

MANDATORY PRE-STEP (brief §5): adjustment spot-check — compare clean_daily closes
across split/bonus ex-dates against the announced ratio. Back-adjusted (continuous)
=> D->D+5 drift on clean_daily is legitimate. Unadjusted (mechanical gap present)
=> the "drift" would be the adjustment itself => study invalid on this feather.
The script runs this FIRST and prints an explicit verdict.

Output: reports/sub9_sanity/_exdate_drift_phase2_discovery.csv (ALL grid cells,
per-era, dead cells included, both bar-variants).

Usage: .venv/Scripts/python tools/sub9_research/phase2_exdate_drift_signature.py
"""
from __future__ import annotations

import datetime as dt
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# ---------------- LOCKED_GRID (pre-registered, brief §5 — DO NOT EDIT) ----------------
LOCKED_GRID = {
    "members": [
        "split",                 # face-value split ex-dates
        "bonus",                 # bonus-issue ex-dates
        "div_interim",           # interim dividends (all yields; salience = yield terciles)
        "div_final_y1p5_3",      # final dividends, yield 1.5-3.0% band (terciles within band)
    ],
    "holds": [1, 3, 5],          # sessions from D+1 open, exit at close
    "adv_tiers": [1, 2, 3, 4, 5],  # within-date quintile of shifted turnover ADV20 (1=lowest)
    "eras": ["A", "B"],          # A: 2023-01..2024-12 | B: 2025-01..2026-04 (amendment A5)
    "salience_buckets": ["T1", "T2", "T3"],  # per-member terciles (T1 smallest adjustment)
    "variants": ["real_bar", "incl_synth"],  # falsifier #3
}
SIGNAL_START = dt.date(2023, 1, 1)
SIGNAL_END = dt.date(2026, 4, 30)   # HARD cap — fresh pool untouched
ERA_A_END = dt.date(2024, 12, 31)
FINAL_DIV_YIELD_LO, FINAL_DIV_YIELD_HI = 1.5, 3.0
PHASE2_DELTA_FLOOR = 0.1            # lifecycle Stage-2 cheap-kill threshold (%)

SB_PARQUET_PROD = _REPO / "data" / "corporate_actions" / "split_bonus_events.parquet"
SB_PARQUET_TMP = _REPO / "_tmp_split_bonus_events.parquet"
DIV_PARQUET = _REPO / "data" / "dividend_ex_date" / "dividend_events.parquet"
CLEAN_DAILY = _REPO / "cache" / "preaggregate" / "clean_daily_from5m.feather"
CONSOL_DAILY = _REPO / "cache" / "preaggregate" / "consolidated_daily.feather"
OUT_CSV = _REPO / "reports" / "sub9_sanity" / "_exdate_drift_phase2_discovery.csv"


def era_of(d: dt.date) -> str:
    return "A" if d <= ERA_A_END else "B"


# ---------------- ratio / salience parsing ----------------
_SPLIT_RE = re.compile(r"[Ff]rom\s+R[se]\.?\s*(\d+(?:\.\d+)?).*?[Tt]o\s+R[se]\.?\s*(\d+(?:\.\d+)?)")
_BONUS_RE = re.compile(r"(\d+)\s*:\s*(\d+)")


def split_salience(subject: str) -> float:
    """Price-drop fraction of a face-value split: FV X -> Y => 1 - Y/X."""
    m = _SPLIT_RE.search(str(subject))
    if not m:
        return np.nan
    x, y = float(m.group(1)), float(m.group(2))
    if x <= 0 or y <= 0 or y >= x:
        return np.nan
    return 1.0 - y / x


def bonus_salience(subject: str) -> float:
    """Price-drop fraction of a bonus A:B (A new per B held) => A/(A+B)."""
    m = _BONUS_RE.search(str(subject))
    if not m:
        return np.nan
    a, b = float(m.group(1)), float(m.group(2))
    if a <= 0 or b <= 0:
        return np.nan
    return a / (a + b)


# ---------------- load prices ----------------
def load_clean_daily() -> pd.DataFrame:
    df = pd.read_feather(CLEAN_DAILY)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    print(f"[clean_daily] shape={df.shape}  dates {df['date'].min().date()} -> "
          f"{df['date'].max().date()}  symbols={df['symbol'].nunique()}")

    g = df.groupby("symbol", sort=False)
    # forward legs (entry D+1 open, exits close D+1/D+3/D+5) — POSITIONAL sessions
    df["open_d1"] = g["open"].shift(-1)
    df["synth_d1"] = ((g["open"].shift(-1) == g["high"].shift(-1))
                      & (g["open"].shift(-1) == g["low"].shift(-1))
                      & (g["open"].shift(-1) == g["close"].shift(-1)))
    df["synth_d0"] = (df["open"] == df["high"]) & (df["open"] == df["low"]) & (df["open"] == df["close"])
    for h in LOCKED_GRID["holds"]:
        df[f"close_h{h}"] = g["close"].shift(-h)
        df[f"drift_h{h}"] = (df["open_d1"] - df[f"close_h{h}"]) / df["open_d1"] * 100.0
    # shifted ADV20 (turnover) — causality guard #1
    df["_turnover"] = df["close"] * df["volume"]
    df["adv20"] = g["_turnover"].transform(lambda s: s.rolling(20, min_periods=20).mean().shift(1))
    df["adv_tier"] = (df.groupby("date")["adv20"].rank(pct=True)
                      .mul(5).apply(np.ceil).clip(1, 5))
    df.drop(columns=["_turnover"], inplace=True)
    return df


# ---------------- adjustment spot-check (MANDATORY pre-step) ----------------
def adjustment_spot_check(daily: pd.DataFrame, sb: pd.DataFrame) -> str:
    print("\n" + "=" * 78)
    print("MANDATORY PRE-STEP: clean_daily adjustment-status spot-check (brief §5)")
    print("=" * 78)
    idx = daily.set_index(["symbol", "date"]).sort_index()
    cand = sb.dropna(subset=["salience"]).copy()
    cand["exp_factor"] = 1.0 / (1.0 - cand["salience"])   # close(D-1)/close(D) if UNadjusted
    # take the largest-ratio events of EACH kind so both splits and bonuses are checked
    cand = (cand.sort_values("exp_factor", ascending=False)
            .groupby("kind", group_keys=False).head(8))
    cand = cand.sort_values("exp_factor", ascending=False)
    rows, verdicts = [], []
    kind_counts = {"split": 0, "bonus": 0}
    for _, ev in cand.iterrows():
        if kind_counts.get(ev["kind"], 0) >= 4:
            continue
        sym = ev["symbol"]
        exd = pd.Timestamp(ev["ex_date"])
        try:
            g = idx.loc[sym]
        except KeyError:
            continue
        dts = g.index
        pos = dts.searchsorted(exd)
        if pos == 0 or pos >= len(dts) or (dts[pos] - exd).days > 7:
            continue
        c_dm1 = g.iloc[pos - 1]["close"]
        c_d = g.iloc[pos]["close"]
        o_d = g.iloc[pos]["open"]
        if not (c_dm1 > 0 and c_d > 0):
            continue
        obs = c_dm1 / c_d
        exp = ev["exp_factor"]
        # obs≈1 => back-adjusted/continuous. obs≈exp => raw (mechanical gap present).
        verdict = "ADJUSTED(continuous)" if abs(obs - 1.0) < abs(obs - exp) else "UNADJUSTED(gap)"
        verdicts.append(verdict)
        kind_counts[ev["kind"]] = kind_counts.get(ev["kind"], 0) + 1
        rows.append((sym, ev["kind"], str(ev["ex_date"]), f"{exp:.2f}",
                     f"{c_dm1:.2f}", f"{o_d:.2f}", f"{c_d:.2f}", f"{obs:.3f}", verdict))
    print(f"{'symbol':<12} {'kind':<6} {'ex_date':<11} {'exp_x':>6} {'cls_Dm1':>9} "
          f"{'open_D':>9} {'cls_D':>9} {'obs_x':>7}  verdict")
    for r in rows:
        print(f"{r[0]:<12} {r[1]:<6} {r[2]:<11} {r[3]:>6} {r[4]:>9} {r[5]:>9} {r[6]:>9} {r[7]:>7}  {r[8]}")
    n_adj = sum(v.startswith("ADJUSTED") for v in verdicts)
    if not verdicts:
        overall = "INCONCLUSIVE — no checkable events; STUDY INVALID until resolved"
    elif n_adj == len(verdicts):
        overall = ("BACK-ADJUSTED (continuous across ex-date) — D->D+5 drift on "
                   "clean_daily is LEGITIMATE; study valid")
    elif n_adj == 0:
        overall = ("UNADJUSTED — mechanical gap present; drift would be the adjustment "
                   "itself; STUDY INVALID on this feather")
    else:
        overall = (f"MIXED ({n_adj}/{len(verdicts)} adjusted) — partial adjustment; "
                   "treat with suspicion, per-event audit required")
    print(f"\nSPOT-CHECK VERDICT: {overall}")
    return overall


# ---------------- events ----------------
def load_split_bonus() -> tuple[pd.DataFrame, str]:
    if SB_PARQUET_PROD.exists():
        src, path = "PRODUCTION", SB_PARQUET_PROD
    else:
        src, path = "TMP-FALLBACK", SB_PARQUET_TMP
    df = pd.read_parquet(path)
    df["ex_date"] = pd.to_datetime(df["ex_date"]).dt.date
    df = df[df["kind"].isin(["split", "bonus"])].copy()
    df = df[(df["ex_date"] >= SIGNAL_START) & (df["ex_date"] <= SIGNAL_END)]
    # dedupe concurrent bonus+split on same (symbol, ex_date): trade once (prior convention)
    pre = len(df)
    df = (df.sort_values(["symbol", "ex_date", "kind"])
          .drop_duplicates(subset=["symbol", "ex_date"], keep="first").reset_index(drop=True))
    df["salience"] = np.where(
        df["kind"] == "split", df["subject"].map(split_salience), df["subject"].map(bonus_salience))
    df["member"] = df["kind"]
    print(f"[split_bonus] source={src} ({path.name})  shape={df.shape} "
          f"(deduped {pre - len(df)})  ex_dates {df['ex_date'].min()} -> {df['ex_date'].max()}")
    print(f"  kinds: {df['kind'].value_counts().to_dict()}  "
          f"salience parsed: {df['salience'].notna().sum()}/{len(df)}")
    return df[["symbol", "ex_date", "member", "salience"]], src


def load_dividends(daily: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_parquet(DIV_PARQUET)
    df["ex_date"] = pd.to_datetime(df["ex_date"]).dt.date
    print(f"[dividends] raw shape={df.shape}  ex_dates {df['ex_date'].min()} -> {df['ex_date'].max()}")
    df = df[(df["ex_date"] >= SIGNAL_START) & (df["ex_date"] <= SIGNAL_END)].copy()
    print(f"  after HARD filter <= {SIGNAL_END}: {len(df)} rows "
          f"(May-Jul-2026 top-up rows excluded by construction)")
    df["symbol"] = df["symbol"].str.replace("^NSE:", "", regex=True)
    df = df[df["dividend_type"].isin(["interim", "final"])]
    df = df[df["dividend_amount_inr"].notna() & (df["dividend_amount_inr"] > 0)]
    # one row per (symbol, ex_date, type): multiple tranches same day are summed
    df = (df.groupby(["symbol", "ex_date", "dividend_type"], as_index=False)
          ["dividend_amount_inr"].sum())
    print(f"  grouped events: {len(df)}  by type: {df['dividend_type'].value_counts().to_dict()}")

    # UNADJUSTED prev close for yield (causality guard #2)
    syms = df["symbol"].unique().tolist()
    cons = pd.read_feather(CONSOL_DAILY, columns=["ts", "symbol", "close"])
    cons = cons[cons["symbol"].isin(syms)].copy()
    cons["ts"] = pd.to_datetime(cons["ts"]).dt.normalize()
    cons = cons.sort_values(["symbol", "ts"])
    print(f"[consolidated_daily] yield-lookup slice shape={cons.shape}  "
          f"dates {cons['ts'].min().date()} -> {cons['ts'].max().date()}")
    grp = {s: (g["ts"].values.astype("datetime64[D]"), g["close"].values)
           for s, g in cons.groupby("symbol")}

    prev_closes = []
    for sym, exd in zip(df["symbol"], df["ex_date"]):
        pc = np.nan
        if sym in grp:
            dts, cls = grp[sym]
            pos = np.searchsorted(dts, np.datetime64(exd, "D"))  # first >= ex_date
            if pos > 0:
                prev_close = cls[pos - 1]
                prev_date = dts[pos - 1].astype("O")
                if prev_close > 0 and (exd - prev_date).days <= 7:
                    pc = float(prev_close)
        prev_closes.append(pc)
    df["prev_close_unadj"] = prev_closes
    df["yield_pct"] = df["dividend_amount_inr"] / df["prev_close_unadj"] * 100.0
    # bad-print guard: implausible yields (>50%) from corrupt closes -> invalid
    df.loc[df["yield_pct"] > 50.0, "yield_pct"] = np.nan
    n_y = df["yield_pct"].notna().sum()
    print(f"  yield computed: {n_y}/{len(df)} (unadjusted prev close within 7cd)")

    interim = df[df["dividend_type"] == "interim"].copy()
    interim["member"] = "div_interim"
    interim["salience"] = interim["yield_pct"]
    final = df[(df["dividend_type"] == "final")
               & df["yield_pct"].between(FINAL_DIV_YIELD_LO, FINAL_DIV_YIELD_HI)].copy()
    final["member"] = "div_final_y1p5_3"
    final["salience"] = final["yield_pct"]
    print(f"  members: div_interim n={len(interim)}  div_final_y1p5_3 n={len(final)} "
          f"(band {FINAL_DIV_YIELD_LO}-{FINAL_DIV_YIELD_HI}%)")
    out = pd.concat([interim, final], ignore_index=True)
    return out[["symbol", "ex_date", "member", "salience"]]


# ---------------- attach price legs to events ----------------
def attach_legs(events: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    cols = (["symbol", "date", "adv_tier", "synth_d0", "synth_d1", "open_d1"]
            + [f"drift_h{h}" for h in LOCKED_GRID["holds"]])
    grp = {s: g.reset_index(drop=True) for s, g in daily[cols].groupby("symbol")}
    rows = []
    funnel = {"raw": 0, "no_symbol": 0, "no_exdate": 0, "no_entry": 0, "ok": 0}
    for _, ev in events.iterrows():
        funnel["raw"] += 1
        sym = ev["symbol"]
        if sym not in grp:
            funnel["no_symbol"] += 1
            continue
        g = grp[sym]
        dts = g["date"].values.astype("datetime64[D]")
        pos = int(np.searchsorted(dts, np.datetime64(ev["ex_date"], "D")))  # first >= ex_date
        if pos >= len(dts) or (dts[pos].astype("O") - ev["ex_date"]).days > 7:
            funnel["no_exdate"] += 1
            continue
        r = g.iloc[pos]
        if pd.isna(r["open_d1"]) or r["open_d1"] <= 0:
            funnel["no_entry"] += 1
            continue
        funnel["ok"] += 1
        rows.append({
            "symbol": sym, "ex_date": ev["ex_date"], "member": ev["member"],
            "salience": ev["salience"], "D_date": r["date"].date(),
            "era": era_of(ev["ex_date"]),
            "adv_tier": r["adv_tier"], "synth_d0": bool(r["synth_d0"]),
            "synth_d1": bool(r["synth_d1"]),
            **{f"drift_h{h}": r[f"drift_h{h}"] for h in LOCKED_GRID["holds"]},
        })
    print(f"  attach funnel: {funnel}")
    return pd.DataFrame(rows)


def main():
    print("=" * 78)
    print("exdate_drift_short — PHASE 2 EMPIRICAL SIGNATURE (pre-registered grid)")
    print("=" * 78)
    daily = load_clean_daily()
    sb_all = pd.read_parquet(SB_PARQUET_PROD if SB_PARQUET_PROD.exists() else SB_PARQUET_TMP)
    sb_all["ex_date"] = pd.to_datetime(sb_all["ex_date"]).dt.date
    sb_all = sb_all[sb_all["kind"].isin(["split", "bonus"])]
    sb_all["salience"] = np.where(sb_all["kind"] == "split",
                                  sb_all["subject"].map(split_salience),
                                  sb_all["subject"].map(bonus_salience))

    # -------- MANDATORY pre-step --------
    spot_verdict = adjustment_spot_check(daily, sb_all)
    if "INVALID" in spot_verdict:
        print("\nABORT: adjustment spot-check failed — no numbers below are trustworthy.")
        # still continue to print nothing further
        return

    # -------- events --------
    print("\n[events]")
    sb, sb_src = load_split_bonus()
    dv = load_dividends(daily)
    events = pd.concat([sb, dv], ignore_index=True)
    print(f"  pooled events: {len(events)}  by member: {events['member'].value_counts().to_dict()}")

    print("\n[attach forward legs]")
    ev = attach_legs(events, daily)
    print(f"  events with valid entry: {len(ev)}  by member x era:")
    print(ev.groupby(["member", "era"]).size().unstack(fill_value=0).to_string())

    # salience terciles per member (pooled edges — deterministic, pre-registered dims)
    ev["sal_bucket"] = "na"
    for m in LOCKED_GRID["members"]:
        mask = (ev["member"] == m) & ev["salience"].notna()
        if mask.sum() >= 9:
            # rank-based terciles: deterministic tie-break (many bonuses are 1:1 =>
            # salience exactly 0.5; qcut on raw values fails on duplicate edges)
            rk = ev.loc[mask, "salience"].rank(method="first")
            ev.loc[mask, "sal_bucket"] = pd.qcut(
                rk, 3, labels=["T1", "T2", "T3"]).astype(str)

    # -------- baselines: same-universe unconditional, matched (era, hold, tier) --------
    print("\n[baselines] same-universe unconditional short drift (matched era/hold/tier)")
    base = daily[daily["date"].dt.date <= SIGNAL_END].copy()
    base["era"] = np.where(base["date"].dt.date <= ERA_A_END, "A", "B")
    baselines = {}   # (variant, era, hold, tier|'all') -> mean short drift
    for variant in LOCKED_GRID["variants"]:
        b = base if variant == "incl_synth" else base[~base["synth_d1"].fillna(True)]
        for h in LOCKED_GRID["holds"]:
            col = f"drift_h{h}"
            v = b[b[col].notna() & (b["open_d1"] > 0)]
            for e in LOCKED_GRID["eras"]:
                ve = v[v["era"] == e]
                baselines[(variant, e, h, "all")] = ve[col].mean()
                for t, g in ve.groupby("adv_tier"):
                    baselines[(variant, e, h, int(t))] = g[col].mean()
    for e in LOCKED_GRID["eras"]:
        row = "  era_%s h5:  all=%+.3f%%  " % (e, baselines[("real_bar", e, 5, "all")])
        row += "  ".join(f"t{t}={baselines[('real_bar', e, 5, t)]:+.3f}%"
                         for t in LOCKED_GRID["adv_tiers"] if ("real_bar", e, 5, t) in baselines)
        print(row + "   (real_bar variant, short sign)")

    # per-event delta vs matched baseline
    def event_frame(variant: str) -> pd.DataFrame:
        d = ev.copy() if variant == "incl_synth" else ev[~ev["synth_d1"]].copy()
        for h in LOCKED_GRID["holds"]:
            d[f"delta_h{h}"] = d.apply(
                lambda r: r[f"drift_h{h}"] - baselines.get(
                    (variant, r["era"], h,
                     int(r["adv_tier"]) if pd.notna(r["adv_tier"]) else "all"),
                    baselines[(variant, r["era"], h, "all")])
                if pd.notna(r[f"drift_h{h}"]) else np.nan, axis=1)
        return d

    frames = {v: event_frame(v) for v in LOCKED_GRID["variants"]}
    print(f"  real_bar events: {len(frames['real_bar'])}  "
          f"incl_synth events: {len(frames['incl_synth'])} "
          f"(synthetic D+1 entry bars: {int(ev['synth_d1'].sum())})")

    # -------- full grid CSV --------
    def cell_row(d: pd.DataFrame, variant, m, e, h, tier, sal):
        g = d[(d["member"] == m) & (d["era"] == e)]
        if tier != "all":
            g = g[g["adv_tier"] == tier]
        if sal != "all":
            g = g[g["sal_bucket"] == sal]
        dr = g[f"drift_h{h}"].dropna()
        dl = g[f"delta_h{h}"].dropna()
        return {
            "variant": variant, "member": m, "era": e, "hold": h,
            "adv_tier": tier, "sal_bucket": sal, "n": len(dr),
            "mean_drift_pct": round(dr.mean(), 4) if len(dr) else np.nan,
            "hit_rate_pct": round(100.0 * (dr > 0).mean(), 2) if len(dr) else np.nan,
            "baseline_pct": round(baselines[(variant, e, h, "all")], 4) if tier == "all"
                            else round(baselines.get((variant, e, h, tier), np.nan), 4),
            "mean_delta_pct": round(dl.mean(), 4) if len(dl) else np.nan,
        }

    out_rows = []
    for variant, d in frames.items():
        for m in LOCKED_GRID["members"]:
            for e in LOCKED_GRID["eras"]:
                for h in LOCKED_GRID["holds"]:
                    # full cross product (dead cells included)
                    for tier in LOCKED_GRID["adv_tiers"]:
                        for sal in LOCKED_GRID["salience_buckets"]:
                            out_rows.append(cell_row(d, variant, m, e, h, tier, sal))
                    # marginals
                    for tier in LOCKED_GRID["adv_tiers"]:
                        out_rows.append(cell_row(d, variant, m, e, h, tier, "all"))
                    for sal in LOCKED_GRID["salience_buckets"]:
                        out_rows.append(cell_row(d, variant, m, e, h, "all", sal))
                    out_rows.append(cell_row(d, variant, m, e, h, "all", "all"))
    out = pd.DataFrame(out_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\n[csv] {len(out)} cell rows -> {OUT_CSV}")

    rb = frames["real_bar"]

    # -------- CORE RESULT: per-member per-era at hold=5 (real_bar) --------
    print("\n" + "=" * 78)
    print("CORE RESULT — per-member per-era short drift at hold=5 (real_bar, raw %)")
    print("=" * 78)
    print(f"{'member':<18} {'era':<4} {'n':>5} {'mean_drift':>11} {'baseline':>9} "
          f"{'delta':>8} {'hit%':>6}")
    for m in LOCKED_GRID["members"]:
        for e in LOCKED_GRID["eras"]:
            r = cell_row(rb, "real_bar", m, e, 5, "all", "all")
            print(f"{m:<18} {e:<4} {r['n']:>5} {r['mean_drift_pct']:>11} "
                  f"{r['baseline_pct']:>9} {r['mean_delta_pct']:>8} {r['hit_rate_pct']:>6}")

    # -------- Falsifier 1: pooling --------
    print("\nFALSIFIER 1 — pooling (dividends same-sign as split/bonus in BOTH eras?)")
    signs = {}
    for m in LOCKED_GRID["members"]:
        for e in LOCKED_GRID["eras"]:
            r = cell_row(rb, "real_bar", m, e, 5, "all", "all")
            signs[(m, e)] = r["mean_delta_pct"]
            for h in (1, 3):
                r2 = cell_row(rb, "real_bar", m, e, h, "all", "all")
                print(f"  {m:<18} era_{e} h{h}: delta={r2['mean_delta_pct']} (n={r2['n']})",
                      end="")
            print(f"  h5: delta={r['mean_delta_pct']} (n={r['n']})")

    # -------- Falsifier 2: salience monotonicity (hold=5) --------
    print("\nFALSIFIER 2 — salience monotonicity at hold=5 (real_bar, delta by tercile)")
    for m in LOCKED_GRID["members"]:
        for e in LOCKED_GRID["eras"]:
            vals = []
            for sal in LOCKED_GRID["salience_buckets"]:
                r = cell_row(rb, "real_bar", m, e, 5, "all", sal)
                vals.append((sal, r["n"], r["mean_delta_pct"]))
            print(f"  {m:<18} era_{e}: " + "  ".join(
                f"{s}: {d} (n={n})" for s, n, d in vals))

    # -------- Falsifier 3: synthetic-bar integrity (hold=5) --------
    print("\nFALSIFIER 3 — real_bar vs incl_synth delta at hold=5")
    for m in LOCKED_GRID["members"]:
        for e in LOCKED_GRID["eras"]:
            a = cell_row(frames["real_bar"], "real_bar", m, e, 5, "all", "all")
            b = cell_row(frames["incl_synth"], "incl_synth", m, e, 5, "all", "all")
            print(f"  {m:<18} era_{e}: real_bar={a['mean_delta_pct']} (n={a['n']})  "
                  f"incl_synth={b['mean_delta_pct']} (n={b['n']})")

    # -------- floor count --------
    print(f"\nFLOOR COUNT — full-grid cells (real_bar) with delta >= {PHASE2_DELTA_FLOOR}%")
    grid_cells = out[(out["variant"] == "real_bar")
                     & (out["adv_tier"] != "all") & (out["sal_bucket"] != "all")]
    for e in LOCKED_GRID["eras"]:
        ge = grid_cells[grid_cells["era"] == e]
        n_all = (ge["mean_delta_pct"] >= PHASE2_DELTA_FLOOR).sum()
        n_n20 = ((ge["mean_delta_pct"] >= PHASE2_DELTA_FLOOR) & (ge["n"] >= 20)).sum()
        n_dead = (ge["n"] == 0).sum()
        print(f"  era_{e}: {n_all}/{len(ge)} cells clear floor "
              f"({n_n20} with n>=20; {n_dead} dead cells)")

    print(f"\n[note] split/bonus events source: "
          f"{'PRODUCTION parquet' if SB_PARQUET_PROD.exists() else 'TMP fallback (_tmp_split_bonus_events.parquet)'}")
    print("[done]")


if __name__ == "__main__":
    main()
