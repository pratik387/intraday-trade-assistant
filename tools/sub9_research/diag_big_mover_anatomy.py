"""DIAGNOSTIC: the anatomy of big movers -- are the day's large movers
identifiable EX ANTE and tradeable, or not?

NOT a candidate. NOT a validation. This is a feasibility / market-physics study
run over DEVELOPMENT windows only. No ledger line, no cell selection, no
freeze. Nothing here may be cited as evidence for a setup; anything it suggests
must go through its own freeze + fresh-pool one-shot (docs/setup_lifecycle.md).

WHY
  Every setup this system owns is a mean-reversion FADE (gap_fade,
  up_spike_fade, or_window_fade, capitulation/panic revert, close_dn,
  below_vwap revert). By construction we are on the WRONG SIDE of, or absent
  from, the day's big movers. Before anyone writes a momentum/continuation
  brief we need to know whether big movers are catchable at all, when, and at
  what false-positive cost.

  Prior momentum deaths do NOT answer this: ORB / gap_and_go / ema5 are
  universal retail patterns; A4 / xsec are cross-sectional multi-week;
  first_hour_momentum failed holdout. None of them measured the POPULATION.

HARD CONSTRAINTS (enforced in code, printed to stdout)
  * No signal dated >= 2026-05-01 is touched.  Fresh pool stays clean.
  * Prices are CA-adjusted / bad-print-cleaned: intraday comes from the 5m
    enriched archive, daily from clean_daily_from5m.feather (lessons #24).
  * Every screen input is CAUSAL: volume baselines are rolling-20 SHIFT(1),
    ADV20 is shift(1), the checkpoint decision price is the close of the last
    bar STRICTLY BEFORE the checkpoint, and the entry price is the OPEN of the
    checkpoint bar (next-bar-open fill).
  * Gross returns only. This study asks whether an edge exists BEFORE costs; a
    screen whose gross mean is ~0 is dead on arrival and needs no fee model.

DEFINITIONS (explicit choices)
  universe   : MIS-enabled (nse_all.json) AND causal ADV20 turnover >= Rs 20L
  BIG MOVER  : day open->close >= +8% (UP) or <= -8% (DOWN); +/-5% sensitivity
  checkpoints: 09:30 / 10:00 / 11:00 / 13:00 IST
  era_A      : 2023-01-01 .. 2024-12-31
  era_B      : 2025-01-01 .. 2026-04-30

OUTPUTS
  reports/sub9_sanity/_big_mover_anatomy.csv   (tidy long: every table stacked)
  console: Q1..Q6 tables + verdict
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from services.symbol_metadata import get_cap_segment, get_mis_info  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 80)
pd.set_option("display.max_rows", 500)

MONTHLY = ROOT / "backtest-cache-download" / "monthly"
CLEAN_DAILY = ROOT / "cache" / "preaggregate" / "clean_daily_from5m.feather"
ASM_PARQUET = ROOT / "data" / "asm_gsm_history" / "asm_gsm_events.parquet"
SANITY = ROOT / "reports" / "sub9_sanity"
OUT_CSV = SANITY / "_big_mover_anatomy.csv"

CACHE_DIR = Path(
    r"C:\Users\PRATIK\AppData\Local\Temp\claude"
    r"\E--Codebase-intraday-trade-assistant"
    r"\d9c67968-368c-45aa-a25e-bd4d1cfb4906\scratchpad"
)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PANEL_CACHE = CACHE_DIR / "bigmover_checkpoint_panel.parquet"
EXIT_CACHE = CACHE_DIR / "bigmover_exit_sim.parquet"

FRESH_POOL_START = pd.Timestamp("2026-05-01")   # hard wall, never crossed
STUDY_MONTHS = [
    f"{y}_{m:02d}" for y in (2023, 2024, 2025, 2026) for m in range(1, 13)
]
STUDY_MONTHS = [m for m in STUDY_MONTHS if "2023_01" <= m <= "2026_04"]

ERA_A = (pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31"))
ERA_B = (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-04-30"))

CHECKPOINTS = ["09:30", "10:00", "11:00", "13:00"]
CP_MIN = {c: int(c[:2]) * 60 + int(c[3:]) for c in CHECKPOINTS}

ADV_MIN_TURNOVER = 2_000_000.0        # Rs 20 lakh
BIG_UP, BIG_DN = 0.08, -0.08
MID_UP, MID_DN = 0.05, -0.05

ADV_TURNOVER_BINS = [0, 1e7, 5e7, 2.5e8, 1e9, np.inf]
ADV_TURNOVER_LABELS = ["V1_<1cr", "V2_1-5cr", "V3_5-25cr", "V4_25-100cr", "V5_>=100cr"]

# screen sweep
MOVE_THRESHOLDS = [0.02, 0.03, 0.05]
VOL_MULTIPLES = [2.0, 3.0, 5.0]

_ROWS: list[dict] = []          # tidy CSV accumulator


def emit(table: str, **kw) -> None:
    _ROWS.append({"table": table, **kw})


def _norm(s: str) -> str:
    s = str(s).strip().upper()
    return s.split(":", 1)[1] if ":" in s else s


# ===================================================================== PASS 1
def reduce_month(path: Path) -> pd.DataFrame:
    """Collapse one month of 5m bars into a per (symbol, session) checkpoint row.

    Everything a screen may look at is computed from bars STRICTLY BEFORE the
    checkpoint; everything an entry may realise is computed from the checkpoint
    bar onwards. The two never overlap -- that separation is the whole point.
    """
    df = pd.read_feather(path)
    df["day"] = df["date"].values.astype("datetime64[D]")
    df["tmin"] = df["date"].dt.hour * 60 + df["date"].dt.minute
    df = df.sort_values(["symbol", "date"], kind="stable")

    key = ["symbol", "day"]
    g = df.groupby(key, sort=False)
    out = g.agg(
        n_bars=("close", "size"),
        day_open=("open", "first"),
        day_close=("close", "last"),
        day_high=("high", "max"),
        day_low=("low", "min"),
        day_vol=("volume", "sum"),
    )

    # ---- circuit-lock proxies (no band data on disk; structural proxies) ----
    dhigh = g["high"].transform("max")
    dlow = g["low"].transform("min")
    zr = (df["high"].values == df["low"].values) & (df["volume"].values > 0)
    df["_zr"] = zr
    df["_zr_hi"] = zr & (df["high"].values == dhigh.values)
    df["_zr_lo"] = zr & (df["low"].values == dlow.values)
    z = df.groupby(key, sort=False)[["_zr", "_zr_hi", "_zr_lo"]].sum()
    out["zr_bars"] = z["_zr"]
    out["zr_at_high"] = z["_zr_hi"]
    out["zr_at_low"] = z["_zr_lo"]

    # ---- per-checkpoint pre/post split ------------------------------------
    for c in CHECKPOINTS:
        t = CP_MIN[c]
        pre = df[df["tmin"] < t]
        post = df[df["tmin"] >= t]
        tag = c.replace(":", "")

        if len(pre):
            gp = pre.groupby(key, sort=False)
            out[f"px_dec_{tag}"] = gp["close"].last()
            out[f"vwap_{tag}"] = gp["vwap"].last()
            out[f"cumvol_{tag}"] = gp["volume"].sum()
            out[f"hi_pre_{tag}"] = gp["high"].max()
            out[f"lo_pre_{tag}"] = gp["low"].min()
        else:
            for col in ("px_dec", "vwap", "cumvol", "hi_pre", "lo_pre"):
                out[f"{col}_{tag}"] = np.nan

        at = df[df["tmin"] == t]
        out[f"px_ent_{tag}"] = at.groupby(key, sort=False)["open"].first() if len(at) else np.nan

        if len(post):
            gq = post.groupby(key, sort=False)
            out[f"mfe_px_{tag}"] = gq["high"].max()
            out[f"mae_px_{tag}"] = gq["low"].min()
            out[f"post_bars_{tag}"] = gq["close"].size()
        else:
            for col in ("mfe_px", "mae_px", "post_bars"):
                out[f"{col}_{tag}"] = np.nan

    return out.reset_index()


def build_panel(force: bool = False) -> pd.DataFrame:
    if PANEL_CACHE.exists() and not force:
        return pd.read_parquet(PANEL_CACHE)
    parts = []
    for m in STUDY_MONTHS:
        p = MONTHLY / f"{m}_5m_enriched.feather"
        if not p.exists():
            print(f"  !! missing {p.name}")
            continue
        t0 = time.time()
        r = reduce_month(p)
        parts.append(r)
        print(f"  {m}: {len(r):>7,} symbol-days  ({time.time()-t0:.1f}s)")
    panel = pd.concat(parts, ignore_index=True)
    panel["symbol"] = panel["symbol"].map(_norm)
    panel["day"] = pd.to_datetime(panel["day"])
    panel.to_parquet(PANEL_CACHE, index=False)
    return panel


# ================================================================ ENRICHMENT
def build_adv(panel_syms: set[str]) -> pd.DataFrame:
    """Causal ADV20 turnover from the CA-adjusted daily panel.

    adv20 at D uses the 20 sessions ENDING AT D-1 (rolling(20).shift(1)) --
    nothing from the signal day. Also carries next-day open for the overnight
    leg of the exit-geometry study.
    """
    d = pd.read_feather(CLEAN_DAILY)
    d["symbol"] = d["symbol"].map(_norm)
    d = d[d["symbol"].isin(panel_syms)].sort_values(["symbol", "date"]).reset_index(drop=True)
    d["turnover"] = d["volume"] * d["close"]
    g = d.groupby("symbol", sort=False)
    d["adv20_turnover"] = g["turnover"].transform(
        lambda s: s.rolling(20, min_periods=15).mean().shift(1))
    d["adv20_shares"] = g["volume"].transform(
        lambda s: s.rolling(20, min_periods=15).mean().shift(1))
    d["prev_close"] = g["close"].shift(1)
    d["next_open"] = g["open"].shift(-1)
    d["next_close"] = g["close"].shift(-1)
    return d[["symbol", "date", "adv20_turnover", "adv20_shares",
              "prev_close", "next_open", "next_close"]].rename(columns={"date": "day"})


def build_asm() -> set[tuple]:
    a = pd.read_parquet(ASM_PARQUET)
    a = a[a["exchange"] == "NSE"].copy()
    a["date"] = pd.to_datetime(a["date"])
    a["symbol"] = a["symbol"].map(_norm)
    # snapshot semantics: entry/no_change/stage_up/stage_down => member on that
    # date; exit => last day, treated as still restricted intraday.
    return set(zip(a["symbol"], a["date"]))


def enrich(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel[panel["day"] < FRESH_POOL_START].copy()
    adv = build_adv(set(p["symbol"].unique()))
    p = p.merge(adv, on=["symbol", "day"], how="left")

    mis_map = {s: get_mis_info(s)["mis_enabled"] for s in p["symbol"].unique()}
    cap_map = {s: get_cap_segment(s) for s in p["symbol"].unique()}
    p["mis_enabled"] = p["symbol"].map(mis_map).fillna(False)
    p["cap_segment"] = p["symbol"].map(cap_map).fillna("unknown")

    asm = build_asm()
    p["in_asm"] = [(s, d) in asm for s, d in zip(p["symbol"], p["day"])]

    p["ret_oc"] = p["day_close"] / p["day_open"] - 1.0
    p["adv_tier"] = pd.cut(p["adv20_turnover"], ADV_TURNOVER_BINS, labels=ADV_TURNOVER_LABELS)
    p["era"] = np.where(p["day"] <= ERA_A[1], "era_A",
                        np.where(p["day"] <= ERA_B[1], "era_B", "OUT"))
    p["qtr"] = p["day"].dt.to_period("Q").astype(str)

    # causal same-time-of-day volume baseline: rolling-20 mean, shift(1)
    p = p.sort_values(["symbol", "day"], kind="stable").reset_index(drop=True)
    gs = p.groupby("symbol", sort=False)
    for c in CHECKPOINTS:
        tag = c.replace(":", "")
        p[f"basevol_{tag}"] = gs[f"cumvol_{tag}"].transform(
            lambda s: s.rolling(20, min_periods=10).mean().shift(1))
        p[f"volx_{tag}"] = p[f"cumvol_{tag}"] / p[f"basevol_{tag}"]
        # entry price falls back to the causal decision price when the
        # checkpoint bar itself did not print (illiquid names)
        p[f"px_ent_{tag}"] = p[f"px_ent_{tag}"].fillna(p[f"px_dec_{tag}"])
        p[f"move_{tag}"] = p[f"px_dec_{tag}"] / p["day_open"] - 1.0
        p[f"fwd_{tag}"] = p["day_close"] / p[f"px_ent_{tag}"] - 1.0
    return p


def apply_universe(p: pd.DataFrame) -> pd.DataFrame:
    return p[
        p["mis_enabled"]
        & (p["adv20_turnover"] >= ADV_MIN_TURNOVER)
        & p["day_open"].notna() & p["day_close"].notna()
        & (p["day_open"] > 0)
        & (p["n_bars"] >= 20)
    ].copy()


# ===================================================================== PASS 3
def simulate_exits(targets: pd.DataFrame) -> pd.DataFrame:
    """Exact bar-by-bar exit simulation for a selected (symbol, day, cp) set.

    Conservative ordering: if a bar's range spans BOTH the stop and the target,
    the stop is assumed to fill first. Long and short both simulated on their
    own rows via `side`.
    """
    if EXIT_CACHE.exists():
        cached = pd.read_parquet(EXIT_CACHE)
        want = set(zip(targets["symbol"], targets["day"], targets["cp"], targets["side"]))
        have = set(zip(cached["symbol"], cached["day"], cached["cp"], cached["side"]))
        if want <= have:
            return cached

    targets = targets.copy()
    targets["month"] = targets["day"].dt.strftime("%Y_%m")
    out = []
    for m, grp in targets.groupby("month", sort=True):
        p = MONTHLY / f"{m}_5m_enriched.feather"
        if not p.exists():
            continue
        df = pd.read_feather(p)
        df["symbol"] = df["symbol"].map(_norm)
        df = df[df["symbol"].isin(set(grp["symbol"]))].copy()
        df["day"] = pd.to_datetime(df["date"].values.astype("datetime64[D]"))
        df["tmin"] = df["date"].dt.hour * 60 + df["date"].dt.minute
        df = df.sort_values(["symbol", "date"], kind="stable")
        bars = {k: v for k, v in df.groupby(["symbol", "day"], sort=False)}

        t0 = time.time()
        for r in grp.itertuples(index=False):
            b = bars.get((r.symbol, r.day))
            if b is None:
                continue
            b = b[b["tmin"] >= CP_MIN[r.cp]]
            if len(b) < 2:
                continue
            out.append(_sim_one(r, b))
        print(f"  exit-sim {m}: {len(grp):>6,} rows ({time.time()-t0:.1f}s)")

    res = pd.DataFrame([o for o in out if o is not None])
    if EXIT_CACHE.exists():
        res = pd.concat([pd.read_parquet(EXIT_CACHE), res], ignore_index=True)
        res = res.drop_duplicates(subset=["symbol", "day", "cp", "side"], keep="last")
    res.to_parquet(EXIT_CACHE, index=False)
    return res


def _sim_one(r, b: pd.DataFrame) -> dict | None:
    ent = float(r.px_ent)
    if not np.isfinite(ent) or ent <= 0:
        return None
    sgn = 1.0 if r.side == "long" else -1.0
    hi = b["high"].to_numpy(float)
    lo = b["low"].to_numpy(float)
    cl = b["close"].to_numpy(float)

    rec = {"symbol": r.symbol, "day": r.day, "cp": r.cp, "side": r.side,
           "px_ent": ent, "close": cl[-1]}

    # signed favourable / adverse excursions in return space
    if sgn > 0:
        mfe = hi.max() / ent - 1.0
        mae = lo.min() / ent - 1.0
    else:
        mfe = 1.0 - lo.min() / ent
        mae = 1.0 - hi.max() / ent
    rec["mfe"] = mfe
    rec["mae"] = mae
    rec["hold_close"] = sgn * (cl[-1] / ent - 1.0)
    rec["next_open_r"] = sgn * (float(r.next_open) / ent - 1.0) if np.isfinite(r.next_open) else np.nan

    for stop_pct in (0.02, 0.03):
        s_lvl = ent * (1 - sgn * stop_pct)
        tag = f"s{int(stop_pct*100)}"
        # --- stop-only (hold to close with a hard stop)
        hit_stop = (lo <= s_lvl) if sgn > 0 else (hi >= s_lvl)
        i_stop = int(np.argmax(hit_stop)) if hit_stop.any() else -1
        rec[f"{tag}_holdclose"] = -stop_pct if i_stop >= 0 else rec["hold_close"]
        # --- R targets
        for rmult in (1, 2, 3):
            t_lvl = ent * (1 + sgn * stop_pct * rmult)
            hit_t = (hi >= t_lvl) if sgn > 0 else (lo <= t_lvl)
            i_t = int(np.argmax(hit_t)) if hit_t.any() else 10**9
            i_s = i_stop if i_stop >= 0 else 10**9
            if i_s <= i_t and i_s < 10**9:
                v = -stop_pct
            elif i_t < 10**9:
                v = stop_pct * rmult
            else:
                v = rec["hold_close"]
            rec[f"{tag}_t{rmult}R"] = v
        # --- trailing stop off running favourable close
        for trail in (stop_pct, stop_pct * 1.5):
            run = np.maximum.accumulate(cl) if sgn > 0 else np.minimum.accumulate(cl)
            trail_lvl = run * (1 - sgn * trail)
            # shift so the trail level is set by BARS ALREADY CLOSED
            trail_lvl = np.concatenate([[ent * (1 - sgn * trail)], trail_lvl[:-1]])
            hitt = (lo <= trail_lvl) if sgn > 0 else (hi >= trail_lvl)
            it = int(np.argmax(hitt)) if hitt.any() else -1
            key = f"{tag}_trail{int(round(trail*1000))}"
            rec[key] = sgn * (trail_lvl[it] / ent - 1.0) if it >= 0 else rec["hold_close"]
    return rec


# ==================================================================== TABLES
def q1_population(u: pd.DataFrame) -> None:
    print("\n" + "=" * 120)
    print("Q1  POPULATION -- how many big movers are there?")
    print("=" * 120)
    for era, lab in (("era_A", "era_A 2023-01..2024-12"), ("era_B", "era_B 2025-01..2026-04")):
        e = u[u["era"] == era]
        nd = e["day"].nunique()
        rows = []
        for name, mask in (
            ("UP >= +8%", e["ret_oc"] >= BIG_UP),
            ("UP >= +5%", e["ret_oc"] >= MID_UP),
            ("DN <= -8%", e["ret_oc"] <= BIG_DN),
            ("DN <= -5%", e["ret_oc"] <= MID_DN),
        ):
            n = int(mask.sum())
            rows.append({"tier": name, "n": n, "per_day": round(n / nd, 2),
                         "pct_of_universe_days": round(100 * n / len(e), 3)})
        t = pd.DataFrame(rows)
        print(f"\n  {lab}   sessions={nd}  universe symbol-days={len(e):,} "
              f"(avg {len(e)/nd:.0f} names/day)")
        print(t.to_string(index=False))
        for r in rows:
            emit("q1_population", era=era, sessions=nd, universe_symbol_days=len(e), **r)

    print("\n  -- by ADV turnover tier (per-day count of |open->close| >= 8%) --")
    piv = (u[u["ret_oc"].abs() >= BIG_UP]
           .groupby(["adv_tier", "era"], observed=True).size().unstack(fill_value=0))
    days = u.groupby("era")["day"].nunique()
    for era in piv.columns:
        piv[era] = (piv[era] / days[era]).round(3)
    print(piv.to_string())
    for tier, row in piv.iterrows():
        emit("q1_by_adv_tier", adv_tier=str(tier),
             **{f"per_day_{k}": float(v) for k, v in row.items()})

    print("\n  -- quarterly trend (big movers >= |8%| per session) --")
    q = (u.assign(big=(u["ret_oc"].abs() >= BIG_UP))
           .groupby("qtr").agg(big=("big", "sum"), sessions=("day", "nunique"),
                               univ=("symbol", "size")))
    q["big_per_day"] = (q["big"] / q["sessions"]).round(2)
    q["univ_per_day"] = (q["univ"] / q["sessions"]).round(0)
    q["big_pct_of_univ"] = (100 * q["big"] / q["univ"]).round(3)
    print(q.to_string())
    for qq, row in q.iterrows():
        emit("q1_quarterly", qtr=qq, **{k: float(v) for k, v in row.items()})


def q2_catchability(u: pd.DataFrame) -> None:
    print("\n" + "=" * 120)
    print("Q2  CATCHABILITY -- how much of the move is already gone at each checkpoint?")
    print("=" * 120)
    for side, mask, lab in (("UP", u["ret_oc"] >= BIG_UP, "UP big movers (o->c >= +8%)"),
                            ("DN", u["ret_oc"] <= BIG_DN, "DOWN big movers (o->c <= -8%)")):
        b = u[mask]
        for era in ("era_A", "era_B"):
            e = b[b["era"] == era]
            if not len(e):
                continue
            rows = []
            for c in CHECKPOINTS:
                tag = c.replace(":", "")
                done = (e[f"px_dec_{tag}"] / e["day_open"] - 1.0) / e["ret_oc"]
                fwd = e[f"fwd_{tag}"] * (1 if side == "UP" else -1)
                rows.append({
                    "checkpoint": c, "n": int(done.notna().sum()),
                    "pct_done_med": round(100 * done.median(), 1),
                    "pct_done_mean": round(100 * done.mean(), 1),
                    "pct_done_p25": round(100 * done.quantile(.25), 1),
                    "pct_done_p75": round(100 * done.quantile(.75), 1),
                    "frac_ge_90pct_done": round(100 * (done >= 0.9).mean(), 1),
                    "remain_med_pct": round(100 * fwd.median(), 2),
                    "remain_mean_pct": round(100 * fwd.mean(), 2),
                    "frac_remain_pos": round(100 * (fwd > 0).mean(), 1),
                })
            print(f"\n  {lab}  [{era}]  n={len(e):,}")
            print(pd.DataFrame(rows).to_string(index=False))
            for r in rows:
                emit("q2_catchability", side=side, era=era, **r)


def q3_false_positives(u: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 120)
    print("Q3  THE CRUX -- FALSE POSITIVES.  Forward distribution of EVERY screened name.")
    print("=" * 120)
    print("  screen(UP)   move_from_open >= M   AND   cumvol >= K x same-time-of-day 20d baseline")
    print("               AND   decision price > session VWAP")
    print("  forward ret  = day_close / px_ent - 1   (entry = open of the checkpoint bar)")

    allrows = []
    for side in ("UP", "DN"):
        sgn = 1.0 if side == "UP" else -1.0
        for era in ("era_A", "era_B"):
            e = u[u["era"] == era]
            for c in CHECKPOINTS:
                tag = c.replace(":", "")
                mv = e[f"move_{tag}"] * sgn
                vx = e[f"volx_{tag}"]
                vw = (e[f"px_dec_{tag}"] > e[f"vwap_{tag}"]) if side == "UP" \
                    else (e[f"px_dec_{tag}"] < e[f"vwap_{tag}"])
                fwd = e[f"fwd_{tag}"] * sgn
                dayret = e["ret_oc"] * sgn
                ok = mv.notna() & vx.notna() & fwd.notna()
                for M in MOVE_THRESHOLDS:
                    for K in VOL_MULTIPLES:
                        sel = ok & (mv >= M) & (vx >= K) & vw
                        n = int(sel.sum())
                        if n == 0:
                            continue
                        f = fwd[sel]
                        dr = dayret[sel]
                        allrows.append({
                            "side": side, "era": era, "checkpoint": c,
                            "move_thr": M, "vol_x": K, "n_screened": n,
                            "n_per_day": round(n / e["day"].nunique(), 2),
                            "pct_became_big8": round(100 * (dr >= 0.08).mean(), 1),
                            "pct_became_big5": round(100 * (dr >= 0.05).mean(), 1),
                            "pct_faded_le2": round(100 * (dr <= 0.02).mean(), 1),
                            "pct_closed_neg": round(100 * (dr < 0).mean(), 1),
                            "fwd_mean_pct": round(100 * f.mean(), 3),
                            "fwd_med_pct": round(100 * f.median(), 3),
                            "fwd_win_pct": round(100 * (f > 0).mean(), 1),
                            "fwd_p10_pct": round(100 * f.quantile(.10), 2),
                            "fwd_p90_pct": round(100 * f.quantile(.90), 2),
                            "fwd_std_pct": round(100 * f.std(), 2),
                        })
    sw = pd.DataFrame(allrows)
    for r in allrows:
        emit("q3_screen_sweep", **r)

    for side in ("UP", "DN"):
        for era in ("era_A", "era_B"):
            s = sw[(sw["side"] == side) & (sw["era"] == era)]
            if not len(s):
                continue
            print(f"\n  ---- {side} screen sweep [{era}] ----")
            print(s.drop(columns=["side", "era"]).to_string(index=False))
    return sw


def q4_tradeability(u: pd.DataFrame, sw: pd.DataFrame) -> None:
    print("\n" + "=" * 120)
    print("Q4  TRADEABILITY -- circuit locks, surveillance, MIS")
    print("=" * 120)
    print("  circuit-lock proxy: >= 3 zero-range bars printed AT the day's extreme")
    print("  (no band table on disk; this is a structural proxy, not the NSE band)")

    full = u.copy()
    full["locked_up"] = full["zr_at_high"] >= 3
    full["locked_dn"] = full["zr_at_low"] >= 3
    rows = []
    for era in ("era_A", "era_B"):
        e = full[full["era"] == era]
        for name, mask in (("UP >= +8%", e["ret_oc"] >= BIG_UP),
                           ("DN <= -8%", e["ret_oc"] <= BIG_DN),
                           ("universe (all)", pd.Series(True, index=e.index))):
            g = e[mask]
            if not len(g):
                continue
            lock = g["locked_up"] if "UP" in name else g["locked_dn"]
            if name.startswith("universe"):
                lock = g["locked_up"] | g["locked_dn"]
            rows.append({
                "era": era, "group": name, "n": len(g),
                "pct_circuit_locked": round(100 * lock.mean(), 1),
                "pct_zero_range_gt10": round(100 * (g["zr_bars"] >= 10).mean(), 1),
                "pct_in_asm_gsm": round(100 * g["in_asm"].mean(), 1),
                "med_adv_turnover_cr": round(g["adv20_turnover"].median() / 1e7, 2),
                "pct_advlt1cr": round(100 * (g["adv20_turnover"] < 1e7).mean(), 1),
            })
    t = pd.DataFrame(rows)
    print("\n  " + t.to_string(index=False).replace("\n", "\n  "))
    for r in rows:
        emit("q4_tradeability", **r)

    # MIS-eligibility funnel on the RAW panel (pre-universe)
    print("\n  -- universe funnel (raw symbol-days -> tradeable universe) --")
    emit("q4_note", note="MIS funnel printed in main()")


def q5_exit_geometry(u: pd.DataFrame, best_cp: str) -> pd.DataFrame:
    print("\n" + "=" * 120)
    print(f"Q5  EXIT GEOMETRY -- entries at {best_cp}, exact bar-by-bar simulation")
    print("=" * 120)

    tag = best_cp.replace(":", "")
    tgts = []
    for side, sgn in (("long", 1.0), ("short", -1.0)):
        mv = u[f"move_{tag}"] * sgn
        vx = u[f"volx_{tag}"]
        vw = (u[f"px_dec_{tag}"] > u[f"vwap_{tag}"]) if sgn > 0 else (u[f"px_dec_{tag}"] < u[f"vwap_{tag}"])
        sel = (mv >= 0.03) & (vx >= 3.0) & vw & u[f"px_ent_{tag}"].notna()
        g = u[sel][["symbol", "day", f"px_ent_{tag}", "next_open", "ret_oc", "era",
                    "in_asm", "adv20_turnover"]].copy()
        g = g.rename(columns={f"px_ent_{tag}": "px_ent"})
        g["cp"] = best_cp
        g["side"] = side
        tgts.append(g)
    tg = pd.concat(tgts, ignore_index=True)
    print(f"  simulating {len(tg):,} screened entries (3% move x 3x vol x VWAP side)")

    sim = simulate_exits(tg)
    sim = sim.merge(tg[["symbol", "day", "cp", "side", "era", "ret_oc", "in_asm"]],
                    on=["symbol", "day", "cp", "side"], how="left")

    variants = [c for c in sim.columns if c.startswith(("s2_", "s3_"))] + ["hold_close"]
    for side in ("long", "short"):
        s = sim[sim["side"] == side]
        if not len(s):
            continue
        rows = []
        for v in ["hold_close"] + sorted(c for c in variants if c != "hold_close"):
            x = s[v].dropna()
            if not len(x):
                continue
            w, l = x[x > 0].sum(), -x[x < 0].sum()
            rows.append({"variant": v, "n": len(x),
                         "mean_pct": round(100 * x.mean(), 3),
                         "med_pct": round(100 * x.median(), 3),
                         "win_pct": round(100 * (x > 0).mean(), 1),
                         "gross_pf": round(float(w / l), 3) if l > 0 else np.nan,
                         "p90_pct": round(100 * x.quantile(.9), 2),
                         "p10_pct": round(100 * x.quantile(.1), 2)})
        print(f"\n  ---- {side.upper()} exit variants (gross, no costs) n={len(s):,} ----")
        print(pd.DataFrame(rows).to_string(index=False))
        for r in rows:
            emit("q5_exit_variants", side=side, checkpoint=best_cp, **r)

        ex = s[["mfe", "mae", "hold_close", "next_open_r"]].dropna(how="all")
        d = {"side": side, "n": len(ex)}
        for col in ("mfe", "mae", "hold_close", "next_open_r"):
            x = ex[col].dropna()
            if len(x):
                d[f"{col}_med"] = round(100 * x.median(), 2)
                d[f"{col}_mean"] = round(100 * x.mean(), 2)
                d[f"{col}_p90"] = round(100 * x.quantile(.9), 2)
        print(f"\n  MFE/MAE to close + overnight [{side}]: {d}")
        emit("q5_mfe_mae", **d)

        # tail concentration -- does clipping kill it?
        x = s["hold_close"].dropna().sort_values()
        if len(x) > 20:
            top5 = x.tail(max(1, len(x) // 20)).sum()
            print(f"  tail concentration [{side}]: top-5% of hold-to-close trades "
                  f"contribute {100*top5/max(abs(x.sum()),1e-9):.0f}% of total gross "
                  f"(total={100*x.sum():.1f}pct-pts, n={len(x)})")
            emit("q5_tail", side=side, top5pct_share=round(float(100*top5/max(abs(x.sum()),1e-9)), 1),
                 total_pct_pts=round(float(100*x.sum()), 1), n=len(x))
    return sim


def q5b_significance(sim: pd.DataFrame, u: pd.DataFrame, best_cp: str) -> None:
    """Era split + t-stats + an overnight BASELINE control.

    A gross mean of a few bps is meaningless without (a) its standard error and
    (b) what the same universe did unconditionally. Both are computed here so
    nobody reads +0.66% overnight as an edge when the whole universe drifted.
    """
    print("\n" + "=" * 120)
    print("Q5b SIGNIFICANCE -- era split, t-stats, and unconditional baselines")
    print("=" * 120)
    rows = []
    for side in ("long", "short"):
        for era in ("era_A", "era_B", "ALL"):
            s = sim[(sim["side"] == side)]
            if era != "ALL":
                s = s[s["era"] == era]
            for col in ("hold_close", "s3_t1R", "next_open_r"):
                x = s[col].dropna()
                if len(x) < 30:
                    continue
                t = float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))
                rows.append({"side": side, "era": era, "variant": col, "n": len(x),
                             "mean_pct": round(100 * x.mean(), 3),
                             "med_pct": round(100 * x.median(), 3),
                             "t_stat": round(t, 2)})
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    for r in rows:
        emit("q5b_significance", checkpoint=best_cp, **r)

    # ---- unconditional baseline: every universe symbol-day, same clock legs
    tag = best_cp.replace(":", "")
    base = u[u[f"px_ent_{tag}"].notna()].copy()
    base["b_hold"] = base["day_close"] / base[f"px_ent_{tag}"] - 1.0
    base["b_overnight"] = base["next_open"] / base[f"px_ent_{tag}"] - 1.0
    brows = []
    for era in ("era_A", "era_B", "ALL"):
        b = base if era == "ALL" else base[base["era"] == era]
        brows.append({
            "era": era, "n": len(b),
            f"uncond_{best_cp}_to_close_mean_pct": round(100 * b["b_hold"].mean(), 3),
            f"uncond_{best_cp}_to_nextopen_mean_pct": round(100 * b["b_overnight"].mean(), 3),
            f"uncond_{best_cp}_to_nextopen_med_pct": round(100 * b["b_overnight"].median(), 3),
        })
    bt = pd.DataFrame(brows)
    print("\n  UNCONDITIONAL universe baseline (no screen at all):")
    print("  " + bt.to_string(index=False).replace("\n", "\n  "))
    for r in brows:
        emit("q5b_baseline", checkpoint=best_cp, **r)
    print("\n  -> subtract the unconditional column from the screened means above.")
    print("     Anything that survives THAT subtraction is the only real signal here.")


def q7_subpopulations(u: pd.DataFrame) -> None:
    """Is there ANY slice of the UP screen with positive gross expectancy?

    Not a cell selection (no freeze, no ledger, development window only) -- a
    falsification pass: if every slice is <= 0, the mechanism is dead and no
    brief is warranted.
    """
    print("\n" + "=" * 120)
    print("Q7  SUB-POPULATION FALSIFICATION -- does ANY slice of the UP screen pay?")
    print("=" * 120)
    tag = "1100"
    mv = u["move_1100"]
    vx = u["volx_1100"]
    vw = u["px_dec_1100"] > u["vwap_1100"]
    sel = mv.notna() & vx.notna() & u["fwd_1100"].notna() & (mv >= 0.03) & (vx >= 3.0) & vw
    s = u[sel].copy()
    s["fwd"] = s["fwd_1100"]
    s["locked"] = s["zr_at_high"] >= 3
    print(f"  base: 11:00 screen 3% x 3x x >VWAP, n={len(s):,}")
    for dim in ("adv_tier", "cap_segment", "era", "in_asm", "locked"):
        g = s.groupby([dim, "era"], observed=True)["fwd"].agg(
            n="size", mean_pct=lambda x: round(100 * x.mean(), 3),
            med_pct=lambda x: round(100 * x.median(), 3),
            win_pct=lambda x: round(100 * (x > 0).mean(), 1))
        g = g[g["n"] >= 100]
        if not len(g):
            continue
        print(f"\n  -- by {dim} --")
        print("  " + g.to_string().replace("\n", "\n  "))
        for idx, row in g.iterrows():
            emit("q7_subpop", dim=dim, level=str(idx[0]), era=str(idx[1]),
                 **{k: float(v) for k, v in row.items()})
    print("\n  cost reference: Zerodha MIS equity round-trip ~ 8-12 bps on notional")
    print("  BEFORE slippage; illiquid names add 20-50 bps of spread/impact.")
    print("  Any gross mean below ~0.15% is not tradeable at any size.")


def q8_ex_lock(u: pd.DataFrame) -> None:
    """THE CORRECTION.  Re-run the screen excluding circuit-locked names.

    Q7 showed the `locked=True` slice carrying a +1.55% / +1.77% mean. That is
    NOT an edge -- a stock frozen at the upper band cannot be bought, so its
    printed forward return is unrealisable. Any headline number that includes
    those rows is an artefact. This block reports the screen with them removed,
    which is the only version a trader could have executed.

    Removal rule (conservative, causal at the checkpoint):
      * >= 3 zero-range bars printed AT the day's extreme  (structural lock), OR
      * the checkpoint decision bar itself is a zero-range bar at the extreme.
    """
    print("\n" + "=" * 120)
    print("Q8  EX-CIRCUIT-LOCK CORRECTION -- the only executable version of the screen")
    print("=" * 120)
    rows = []
    for side, sgn in (("UP", 1.0), ("DN", -1.0)):
        for era in ("era_A", "era_B"):
            e = u[u["era"] == era]
            lock = (e["zr_at_high"] >= 3) if side == "UP" else (e["zr_at_low"] >= 3)
            for c in CHECKPOINTS:
                tag = c.replace(":", "")
                mv = e[f"move_{tag}"] * sgn
                vx = e[f"volx_{tag}"]
                vw = (e[f"px_dec_{tag}"] > e[f"vwap_{tag}"]) if sgn > 0 \
                    else (e[f"px_dec_{tag}"] < e[f"vwap_{tag}"])
                fwd = e[f"fwd_{tag}"] * sgn
                on = (e["next_open"] / e[f"px_ent_{tag}"] - 1.0) * sgn
                base = mv.notna() & vx.notna() & fwd.notna() & (mv >= 0.03) & (vx >= 3.0) & vw
                for lab, sel in (("incl_locked", base), ("EX_locked", base & ~lock)):
                    n = int(sel.sum())
                    if n < 50:
                        continue
                    f, o = fwd[sel], on[sel].dropna()
                    se = f.std(ddof=1) / np.sqrt(n)
                    rows.append({
                        "side": side, "era": era, "checkpoint": c, "variant": lab,
                        "n": n,
                        "fwd_mean_pct": round(100 * f.mean(), 3),
                        "fwd_med_pct": round(100 * f.median(), 3),
                        "fwd_win_pct": round(100 * (f > 0).mean(), 1),
                        "t_stat": round(float(f.mean() / se), 2),
                        "overnight_mean_pct": round(100 * o.mean(), 3) if len(o) else np.nan,
                    })
    t = pd.DataFrame(rows)
    for side in ("UP", "DN"):
        s = t[t["side"] == side]
        print(f"\n  ---- {side} screen, 3% x 3x x VWAP-side, with/without circuit locks ----")
        print("  " + s.drop(columns=["side"]).to_string(index=False).replace("\n", "\n  "))
    for r in rows:
        emit("q8_ex_lock", **r)

    print("\n  reminder: unconditional universe overnight drift is +0.29% (era_A) /")
    print("  +0.14% (era_B) from 13:00 -- subtract it from any overnight_mean_pct above.")


def main() -> None:
    print("=" * 120)
    print("DIAGNOSTIC -- ANATOMY OF BIG MOVERS  (feasibility study; NOT a candidate)")
    print(f"  months {STUDY_MONTHS[0]} .. {STUDY_MONTHS[-1]}   fresh-pool wall: "
          f"no signal >= {FRESH_POOL_START.date()}")
    print(f"  universe: MIS-enabled AND causal ADV20 turnover >= Rs {ADV_MIN_TURNOVER/1e5:.0f}L")
    print(f"  BIG MOVER: open->close >= +8% / <= -8%   (+/-5% sensitivity tier)")
    print("=" * 120)

    print("\n### PASS 1 -- reduce 5m archive to per (symbol, session) checkpoint rows")
    panel = build_panel()
    print(f"  raw panel: {panel.shape}  symbols={panel['symbol'].nunique()}  "
          f"days={panel['day'].nunique()}  {panel['day'].min().date()}..{panel['day'].max().date()}")

    print("\n### PASS 2 -- enrich (causal ADV20, MIS, cap, ASM/GSM, volume baselines)")
    p = enrich(panel)
    print(f"  after fresh-pool wall: {p.shape}  max day={p['day'].max().date()}")
    n_raw = len(p)
    n_mis = int(p["mis_enabled"].sum())
    n_adv = int((p["adv20_turnover"] >= ADV_MIN_TURNOVER).sum())
    u = apply_universe(p)
    print(f"  FUNNEL  raw symbol-days {n_raw:,}"
          f" -> MIS-enabled {n_mis:,} ({100*n_mis/n_raw:.1f}%)"
          f" -> ADV20>=20L {n_adv:,} ({100*n_adv/n_raw:.1f}%)"
          f" -> UNIVERSE {len(u):,} ({100*len(u)/n_raw:.1f}%)")
    print(f"  universe symbols={u['symbol'].nunique()}  sessions={u['day'].nunique()}")
    emit("funnel", raw=n_raw, mis=n_mis, adv=n_adv, universe=len(u),
         symbols=int(u["symbol"].nunique()), sessions=int(u["day"].nunique()))

    q1_population(u)
    q2_catchability(u)
    sw = q3_false_positives(u)
    q4_tradeability(u, sw)

    # pick the checkpoint with the best UP screened mean forward return at the
    # canonical 3% x 3x screen -- purely descriptive, not a selection
    cand = sw[(sw["side"] == "UP") & (sw["move_thr"] == 0.03) & (sw["vol_x"] == 3.0)]
    best_cp = (cand.groupby("checkpoint")["fwd_mean_pct"].mean().idxmax()
               if len(cand) else "10:00")
    print(f"\n  [checkpoint chosen for Q5 exit geometry: {best_cp} "
          f"(highest mean fwd return at the 3% x 3x screen)]")

    sim = q5_exit_geometry(u, best_cp)
    q5b_significance(sim, u, best_cp)
    q7_subpopulations(u)
    q8_ex_lock(u)

    print("\n" + "=" * 120)
    print("Q6  DOWN movers -- shortability")
    print("=" * 120)
    dn = u[u["ret_oc"] <= BIG_DN]
    rows = []
    for era in ("era_A", "era_B"):
        e = dn[dn["era"] == era]
        if not len(e):
            continue
        rows.append({"era": era, "n_down_big": len(e),
                     "pct_mis_shortable": 100.0,   # universe is MIS-filtered by construction
                     "pct_in_asm_gsm": round(100 * e["in_asm"].mean(), 1),
                     "pct_lower_locked": round(100 * (e["zr_at_low"] >= 3).mean(), 1),
                     "pct_clean_shortable": round(
                         100 * ((~e["in_asm"]) & (e["zr_at_low"] < 3)).mean(), 1)})
    print(pd.DataFrame(rows).to_string(index=False))
    for r in rows:
        emit("q6_shortability", **r)

    SANITY.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_ROWS).to_csv(OUT_CSV, index=False)
    print(f"\n  wrote {OUT_CSV}  ({len(_ROWS)} rows)")


if __name__ == "__main__":
    main()
