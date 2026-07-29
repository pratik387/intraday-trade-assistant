"""
diag_bigmove_multiday_path.py — STUDY (Phase-1/2 exploration, no ledger line)

QUESTION: After a stock moves +/-X% in a single day (open->close), what is its path
over the NEXT 10 sessions from the EVENT CLOSE? How often does it give back a large
share of the move, how often does it continue, and is either tradeable?

Everything measured before this was SAME-SESSION (diag_big_mover_anatomy = checkpoint
-> close; earnings down-shock = T+1 open -> T+1 close). The multi-day path from the
event close has never been systematically measured.

DATA   : cache/preaggregate/clean_daily_from5m.feather (CA-adjusted)
UNIVERSE: causal ADV20 turnover >= Rs 20L (rolling 20d mean of close*volume, SHIFTED
          by 1 so the event day itself never enters its own liquidity filter)
FRESH POOL: signal (event) dates hard-filtered to <= 2026-04-30. Forward path is
          allowed to run into May-Jul 2026 (that is realised future, not signal peeking).

EVENT  : daily open->close move.
         UP   tiers: >= +8%, >= +10%, >= +15%
         DOWN tiers: <= -8%, <= -10%, <= -15%
FORWARD: close-to-close cumulative return at T+1, T+2, T+3, T+5, T+10 from event close.

GIVE-BACK: give_back_ratio = -(forward_return) / (event_move)
           UP event   (move > 0): fwd < 0 -> gbr > 0  (gave back)
           DOWN event (move < 0): fwd > 0 -> gbr > 0  (bounced back)
           gbr = 1.0 means the entire move has been retraced.
           gbr <= -0.25 means the move EXTENDED by >= 25% of its size.

TRADE  : the "retracement trade" is entered at the EVENT CLOSE and held H sessions.
         UP event   -> SHORT (pnl = -fwd_ret)   <-- NOT executable in this universe
         DOWN event -> LONG  (pnl = +fwd_ret)
         Costs: 0.31% round trip (intraday MIS reference, optimistic for a multi-day
         hold) and 0.60% round trip (CNC/MTF multi-day realism: STT on both legs for
         delivery, wider spread + impact in the illiquid tail). For H >= 1 the 0.60%
         model is the decision-relevant one; 0.31% is shown only as an upper bound.

Run: .venv/Scripts/python tools/sub9_research/diag_bigmove_multiday_path.py
Out: reports/sub9_sanity/_bigmove_multiday_path.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DAILY = ROOT / "cache" / "preaggregate" / "clean_daily_from5m.feather"
OUT_CSV = ROOT / "reports" / "sub9_sanity" / "_bigmove_multiday_path.csv"

EARN = ROOT / "data" / "earnings_calendar" / "earnings_events.parquet"
CA = ROOT / "data" / "corporate_actions" / "split_bonus_events.parquet"
BLOCK = ROOT / "data" / "block_deals" / "block_deals_events.parquet"
FNO = ROOT / "assets" / "fno_liquid_200.csv"

MONSTER_LEDGERS = [
    "_monster_cond_mtf_capitulation_revert_long_baseline.csv",
    "_monster_cond_low52_capitulation_revert_long_baseline.csv",
    "_monster_cond_zscore_oversold_revert_long_baseline.csv",
    "_monster_cond_crash2d_revert_long_baseline.csv",
]

FRESH_POOL_CUTOFF = pd.Timestamp("2026-04-30")
ERA_A = (pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31"))
ERA_B = (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-04-30"))

ADV_MIN = 20e5  # Rs 20 lakh
HORIZONS = [1, 2, 3, 5, 10]
COST_INTRADAY = 0.0031
COST_MULTIDAY = 0.0060

TIERS = {
    "up": [("ge_8", 0.08), ("ge_10", 0.10), ("ge_15", 0.15)],
    "down": [("ge_8", -0.08), ("ge_10", -0.10), ("ge_15", -0.15)],
}
BANDS = {
    "up": [("band_8_10", 0.08, 0.10), ("band_10_15", 0.10, 0.15), ("band_15p", 0.15, 9.99)],
    "down": [("band_8_10", -0.10, -0.08), ("band_10_15", -0.15, -0.10), ("band_15p", -9.99, -0.15)],
}


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("^(NSE|BSE):", "", regex=True).str.strip().str.upper()


# --------------------------------------------------------------------------- load
def load_daily() -> pd.DataFrame:
    df = pd.read_feather(DAILY)
    print(f"[load] clean_daily_from5m: shape={df.shape} "
          f"symbols={df['symbol'].nunique()} "
          f"dates={df['date'].min().date()}..{df['date'].max().date()}")
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df["symbol"] = _norm(df["symbol"])

    g = df.groupby("symbol", sort=False)
    df["turnover"] = df["close"] * df["volume"]
    # causal ADV20: mean of PRIOR 20 sessions (shift(1) so event day excluded)
    df["adv20"] = g["turnover"].transform(lambda s: s.shift(1).rolling(20, min_periods=15).mean())

    df["move"] = df["close"] / df["open"] - 1.0
    # close position within the day's range (1.0 = closed at high, 0.0 = at low)
    rng = (df["high"] - df["low"]).replace(0.0, np.nan)
    df["close_pos"] = (df["close"] - df["low"]) / rng
    # prior extension context: 5-session close-to-close return ending the day BEFORE the event
    df["prior5"] = g["close"].transform(lambda s: s.shift(1) / s.shift(6) - 1.0)

    for h in HORIZONS:
        df[f"fwd{h}"] = g["close"].transform(lambda s, h=h: s.shift(-h) / s - 1.0)

    print(f"[load] adv20 non-null: {df['adv20'].notna().mean():.1%}; "
          f"rows with adv20>=20L: {(df['adv20'] >= ADV_MIN).sum():,}")
    return df


def load_event_maps():
    """Return (earn_keys, ca_keys, block_keys) as sets of (symbol, date) with a window."""
    earn = set()
    try:
        e = pd.read_parquet(EARN)
        e["symbol"] = _norm(e["symbol"])
        e["d"] = pd.to_datetime(e["announce_date"], errors="coerce").dt.normalize()
        e = e.dropna(subset=["d"])
        # an AMC/after-hours announcement on D moves the stock on D+1; allow [-1, +1]
        for off in (-1, 0, 1):
            earn |= set(zip(e["symbol"], e["d"] + pd.Timedelta(days=off)))
        print(f"[load] earnings: rows={len(e):,} keys(+/-1d)={len(earn):,}")
    except Exception as ex:  # noqa: BLE001
        print(f"[load] earnings FAILED: {ex}")

    ca = set()
    try:
        c = pd.read_parquet(CA)
        c["symbol"] = _norm(c["symbol"])
        c["d"] = pd.to_datetime(c["ex_date"], errors="coerce").dt.normalize()
        c = c.dropna(subset=["d"])
        for off in (-2, -1, 0, 1, 2):
            ca |= set(zip(c["symbol"], c["d"] + pd.Timedelta(days=off)))
        print(f"[load] corp actions: rows={len(c):,} keys(+/-2d)={len(ca):,}")
    except Exception as ex:  # noqa: BLE001
        print(f"[load] corp actions FAILED: {ex}")

    blk = set()
    try:
        b = pd.read_parquet(BLOCK)
        b["symbol"] = _norm(b["symbol"])
        b["d"] = pd.to_datetime(b["trade_date"], errors="coerce").dt.normalize()
        b = b.dropna(subset=["d"])
        for off in (0, 1):
            blk |= set(zip(b["symbol"], b["d"] + pd.Timedelta(days=off)))
        print(f"[load] block deals: rows={len(b):,} keys(0/+1d)={len(blk):,} "
              f"span={b['d'].min().date()}..{b['d'].max().date()}")
    except Exception as ex:  # noqa: BLE001
        print(f"[load] block deals FAILED: {ex}")

    return earn, ca, blk


def load_fno() -> set:
    try:
        f = pd.read_csv(FNO)
        s = set(_norm(f["symbol"]))
        print(f"[load] F&O liquid list: {len(s)} symbols")
        return s
    except Exception as ex:  # noqa: BLE001
        print(f"[load] fno FAILED: {ex}")
        return set()


def load_capitulation_keys() -> set:
    keys = set()
    for fn in MONSTER_LEDGERS:
        p = ROOT / "reports" / "sub9_sanity" / fn
        try:
            d = pd.read_csv(p)
            d["symbol"] = _norm(d["symbol"])
            d["signal_date"] = pd.to_datetime(d["signal_date"]).dt.normalize()
            k = set(zip(d["symbol"], d["signal_date"]))
            keys |= k
            print(f"[load] {fn}: rows={len(d):,} uniq_keys={len(k):,}")
        except Exception as ex:  # noqa: BLE001
            print(f"[load] {fn} FAILED: {ex}")
    print(f"[load] capitulation union keys: {len(keys):,}")
    return keys


# ------------------------------------------------------------------------ metrics
def bucket_stats(sub: pd.DataFrame, direction: str, meta: dict) -> list[dict]:
    """One row per horizon for a bucket."""
    rows = []
    sign = 1.0 if direction == "down" else -1.0  # retracement trade direction on fwd ret
    for h in HORIZONS:
        col = f"fwd{h}"
        d = sub[sub[col].notna()]
        n = len(d)
        if n == 0:
            rows.append({**meta, "horizon": f"T+{h}", "n": 0, "status": "EMPTY"})
            continue
        fwd = d[col].to_numpy(float)
        move = d["move"].to_numpy(float)
        gbr = -fwd / move  # give-back ratio (works for both signs)
        pnl = sign * fwd  # gross retracement-trade return

        se = fwd.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        t = (pnl.mean() / (pnl.std(ddof=1) / np.sqrt(n))) if n > 1 and pnl.std(ddof=1) > 0 else np.nan

        # --- cluster-robust: big-move days cluster on market-wide crash/rally dates, so the
        #     naive t-stat is badly overstated. Collapse to one observation per EVENT DATE.
        bydate = d.groupby("date")[col].mean() * sign
        nd = len(bydate)
        t_date = (bydate.mean() / (bydate.std(ddof=1) / np.sqrt(nd))) if nd > 1 and bydate.std(ddof=1) > 0 else np.nan
        top10_share = float(d["date"].value_counts().head(10).sum() / n)

        # --- tail robustness: trimmed mean (drop best+worst 2.5% of forward returns)
        lo_q, hi_q = np.percentile(pnl, [2.5, 97.5])
        trimmed = pnl[(pnl >= lo_q) & (pnl <= hi_q)]
        trim_mean = float(trimmed.mean()) if len(trimmed) else np.nan

        # --- monthly consistency of the retracement trade (net of 0.60%)
        mo = pd.Series(pnl - COST_MULTIDAY, index=d["date"].values).groupby(
            pd.PeriodIndex(d["date"], freq="M")).mean()
        mo_pos = float((mo > 0).mean()) if len(mo) else np.nan

        rows.append({
            **meta,
            "horizon": f"T+{h}",
            "n": n,
            "n_symbols": int(d["symbol"].nunique()),
            "status": "OK",
            "event_move_med": float(np.median(move)),
            "fwd_mean": float(fwd.mean()),
            "fwd_med": float(np.median(fwd)),
            "fwd_p25": float(np.percentile(fwd, 25)),
            "fwd_p75": float(np.percentile(fwd, 75)),
            "fwd_se": float(se) if se == se else np.nan,
            "gbr_med": float(np.median(gbr)),
            "gbr_mean": float(np.mean(np.clip(gbr, -5, 5))),  # clipped: raw mean explodes on tiny moves
            "pct_gbr_ge25": float((gbr >= 0.25).mean()),
            "pct_gbr_ge50": float((gbr >= 0.50).mean()),
            "pct_gbr_ge75": float((gbr >= 0.75).mean()),
            "pct_gbr_ge100": float((gbr >= 1.00).mean()),
            "pct_continue_ge25": float((gbr <= -0.25).mean()),
            "wr_gross": float((pnl > 0).mean()),
            "wr_net_60": float((pnl - COST_MULTIDAY > 0).mean()),
            "exp_gross": float(pnl.mean()),
            "exp_net_31bp": float(pnl.mean() - COST_INTRADAY),
            "exp_net_60bp": float(pnl.mean() - COST_MULTIDAY),
            "t_stat": float(t) if t == t else np.nan,
            "fno_frac": float(d["is_fno"].mean()),
            "cap_overlap_frac": float(d["in_cap"].mean()),
            "cost_model": "0.60% CNC/MTF (decision) | 0.31% MIS (upper bound)",
        })
    return rows


def main() -> int:
    df = load_daily()
    earn, ca, blk = load_event_maps()
    fno = load_fno()
    cap_keys = load_capitulation_keys()

    # ---- universe + fresh-pool filter
    ev = df[(df["adv20"] >= ADV_MIN) & (df["date"] <= FRESH_POOL_CUTOFF)].copy()
    ev = ev[ev["move"].notna() & ev["open"].gt(0)]
    print(f"\n[universe] rows after ADV>=20L + date<=2026-04-30: {len(ev):,} "
          f"symbols={ev['symbol'].nunique()}")

    ev["era"] = np.where(ev["date"] <= ERA_A[1], "era_A", "era_B")
    ev.loc[ev["date"] < ERA_A[0], "era"] = "pre"
    ev = ev[ev["era"].isin(["era_A", "era_B"])]

    q = ev["adv20"]
    ev["adv_tier"] = pd.cut(q, [0, 1e7, 1e8, np.inf], labels=["micro_20L_1cr", "small_1_10cr", "liq_10cr+"])

    key = list(zip(ev["symbol"], ev["date"]))
    ev["is_earn"] = [k in earn for k in key]
    ev["is_ca"] = [k in ca for k in key]
    ev["is_block"] = [k in blk for k in key]
    ev["is_eventdriven"] = ev["is_earn"] | ev["is_ca"] | ev["is_block"]
    ev["is_fno"] = ev["symbol"].isin(fno)
    ev["in_cap"] = [k in cap_keys for k in key]

    print(f"[universe] era counts: {ev['era'].value_counts().to_dict()}")
    print(f"[universe] adv_tier: {ev['adv_tier'].value_counts().to_dict()}")

    rows: list[dict] = []

    for direction in ("up", "down"):
        for tier_name, thr in TIERS[direction]:
            base = ev[ev["move"] >= thr] if direction == "up" else ev[ev["move"] <= thr]
            if base.empty:
                continue
            # close-off-extreme flag is direction dependent
            b = base.copy()
            if direction == "up":
                b["near_extreme"] = b["close_pos"] >= 0.80
                b["pre_extended"] = b["prior5"] >= 0.10
            else:
                b["near_extreme"] = b["close_pos"] <= 0.20
                b["pre_extended"] = b["prior5"] <= -0.10

            def add(sub, cond_dim, cond_val, era):
                if len(sub) < 15:
                    rows.append({"direction": direction, "tier": tier_name, "era": era,
                                 "cond_dim": cond_dim, "cond_val": cond_val,
                                 "horizon": "-", "n": len(sub), "status": "N_BELOW_15"})
                    return
                rows.extend(bucket_stats(sub, direction, {
                    "direction": direction, "tier": tier_name, "era": era,
                    "cond_dim": cond_dim, "cond_val": str(cond_val)}))

            for era in ("ALL", "era_A", "era_B"):
                e = b if era == "ALL" else b[b["era"] == era]
                if e.empty:
                    continue
                add(e, "none", "all", era)
                for t in e["adv_tier"].dropna().unique():
                    add(e[e["adv_tier"] == t], "adv_tier", t, era)
                for flag, dim in (("near_extreme", "close_off_extreme"),
                                  ("is_eventdriven", "event_driven"),
                                  ("is_earn", "earnings"),
                                  ("is_block", "block_deal"),
                                  ("pre_extended", "pre_extended")):
                    add(e[e[flag]], dim, "yes", era)
                    add(e[~e[flag]], dim, "no", era)

        # magnitude bands (non-overlapping) — tests magnitude separation cleanly
        for band_name, lo, hi in BANDS[direction]:
            bb = ev[(ev["move"] >= lo) & (ev["move"] < hi)].copy()
            if bb.empty:
                continue
            bb["near_extreme"] = False
            for era in ("ALL", "era_A", "era_B"):
                e = bb if era == "ALL" else bb[bb["era"] == era]
                if len(e) < 15:
                    continue
                rows.extend(bucket_stats(e, direction, {
                    "direction": direction, "tier": band_name, "era": era,
                    "cond_dim": "magnitude_band", "cond_val": band_name}))

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\n[out] {OUT_CSV} rows={len(out)}")

    # -------- console summary
    pd.set_option("display.width", 250, "display.max_columns", 60)

    def show(title, sel, cols):
        print(f"\n=== {title} ===")
        s = out[sel]
        if s.empty:
            print("(empty)")
            return
        print(s[cols].to_string(index=False))

    ok = out.get("status", pd.Series(dtype=str)).eq("OK")

    show("1. PATH (median-led), cond=none, era ALL",
         ok & out["cond_dim"].eq("none") & out["era"].eq("ALL"),
         ["direction", "tier", "horizon", "n", "event_move_med", "fwd_med", "fwd_mean", "fwd_p25", "fwd_p75"])

    show("1b. PATH by era, ge_10 only",
         ok & out["cond_dim"].eq("none") & out["tier"].eq("ge_10"),
         ["direction", "era", "horizon", "n", "fwd_med", "fwd_mean", "wr_gross"])

    show("2. GIVE-BACK DISTRIBUTION, cond=none, era ALL",
         ok & out["cond_dim"].eq("none") & out["era"].eq("ALL"),
         ["direction", "tier", "horizon", "n", "gbr_med", "pct_gbr_ge25", "pct_gbr_ge50",
          "pct_gbr_ge75", "pct_gbr_ge100", "pct_continue_ge25"])

    show("3. SYSTEMATIC? retracement trade, cond=none",
         ok & out["cond_dim"].eq("none"),
         ["direction", "tier", "era", "horizon", "n", "wr_gross", "exp_gross",
          "exp_net_31bp", "exp_net_60bp", "t_stat", "fno_frac", "cap_overlap_frac"])

    show("5. CONDITIONING (era ALL, ge_10)",
         ok & out["tier"].eq("ge_10") & out["era"].eq("ALL") & out["cond_dim"].ne("none"),
         ["direction", "cond_dim", "cond_val", "horizon", "n", "fwd_med", "exp_net_60bp", "t_stat", "fno_frac"])

    show("5b. MAGNITUDE BANDS",
         ok & out["cond_dim"].eq("magnitude_band") & out["era"].eq("ALL"),
         ["direction", "tier", "horizon", "n", "fwd_med", "gbr_med", "exp_net_60bp", "t_stat"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
