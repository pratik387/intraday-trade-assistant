"""
Broad Phase-1/2 screen: which event classes produce moves large enough to clear
transaction costs?

Screening principle
-------------------
An event class is only worth a Stage-0 brief if its SIGNED drift clears the
round-trip cost with real margin in BOTH eras.  Classes are reported even when
dead -- the dead list is as valuable as the live one.

Method (uniform across every class so results are comparable)
------------------------------------------------------------
* Prices  : cache/preaggregate/clean_daily_from5m.feather (CA-adjusted daily).
* Signal  : the date on which the information is PUBLICLY actionable.  Per-class
            assumption is stated in the CLASS_NOTES dict below.
* Entry   : open of the first trading session STRICTLY AFTER the signal date
            (max 7 calendar days of staleness tolerance).
* Exits   : close of entry day (h1), entry+2 (h3), entry+4 (h5).
* Both directions measured -- no direction is assumed.  A long-side mean of
            -0.8% is a short-side edge of +0.8% (before cost).
* Cost    : ROUND_TRIP_COST_PCT = 0.31% charged once per trade, both directions.
* Eras    : era_A 2023-01-01..2024-12-31, era_B 2025-01-01..2026-04-30.
            Signals on/after FRESH_POOL_CUTOFF (2026-05-01) are DROPPED.
* Causality: every conditioning statistic (ADV, delivery baseline, basis
            percentile, prior close) uses prior data only (shift(1) / expanding
            windows that exclude the event bar).

Ledger note
-----------
This is a Phase-1/2 exploratory screen run on development windows only.  It is
exempt from the research ledger under the same clause as Discovery exploration
(no candidate is being validated, no cell is being locked, nothing is shipped).

Output: reports/sub9_sanity/_event_class_screen.csv
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]

# ----------------------------------------------------------------------------
# Screen constants (explicit; this is an exploratory screen, not production code)
# ----------------------------------------------------------------------------
ROUND_TRIP_COST_PCT = 0.31
BIG_MOVE_MULT = 3.0
BIG_MOVE_PCT = ROUND_TRIP_COST_PCT * BIG_MOVE_MULT  # 0.93%

ERA_A = ("2023-01-01", "2024-12-31")
ERA_B = ("2025-01-01", "2026-04-30")
FRESH_POOL_CUTOFF = pd.Timestamp("2026-05-01")

HORIZONS = (1, 3, 5)
ENTRY_STALENESS_DAYS = 7
MIN_N_REPORT = 15  # rows below this are still emitted but flagged thin

PRICE_PATH = ROOT / "cache" / "preaggregate" / "clean_daily_from5m.feather"
OUT_PATH = ROOT / "reports" / "sub9_sanity" / "_event_class_screen.csv"

CLASS_NOTES = {
    "pledging": "SEBI SAST Reg-31 disclosure; NSE filings post-close -> entry T+1 open",
    "index_recon": "NSE press release post-close (announcement) / index funds trade at "
    "close of day before effective -> both dates screened, entry T+1 open",
    "earnings_shock": "reaction day = announce day if pre-15:30 else next session; "
    "signal = reaction day close -> entry T+1 open",
    "split_bonus": "ex_date known well in advance; signal = ex_date -> entry T+1 open",
    "delivery": "NSE bhavcopy delivery% published post-close -> entry T+1 open",
    "basis": "futures/spot close -> entry T+1 open",
    "asm_gsm": "surveillance list published post-close, effective next session -> entry T+1 open",
    "block_deals": "block window deals reported post-close -> entry T+1 open",
    "bulk_deals": "bulk deals reported post-close -> entry T+1 open",
    "dividend": "ex_date known in advance; signal = ex_date -> entry T+1 open",
    "fno_ban": "TOO THIN (n=21) -- not screened",
}


# ----------------------------------------------------------------------------
# Price panel
# ----------------------------------------------------------------------------
def load_prices() -> tuple[pd.DataFrame, dict]:
    px = pd.read_feather(PRICE_PATH)
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["symbol", "date"]).reset_index(drop=True)
    px["gidx"] = np.arange(len(px), dtype=np.int64)
    px["pos"] = px.groupby("symbol").cumcount()
    n_by_sym = px.groupby("symbol")["pos"].transform("size")
    px["n_sym"] = n_by_sym
    # prior-only ADV: 20d mean turnover, shifted by 1 so the event bar is excluded
    px["turnover"] = px["close"] * px["volume"]
    px["adv20"] = (
        px.groupby("symbol")["turnover"]
        .transform(lambda s: s.rolling(20, min_periods=10).mean().shift(1))
    )
    px["prev_close"] = px.groupby("symbol")["close"].shift(1)
    # ------------------------------------------------------------------
    # MATCHED BASELINE.  The panel carries a structural open->close drift of
    # about -0.18%/day (median -0.28%), mirrored by a +0.27% overnight drift.
    # Part of that is the real intraday/overnight decomposition and part is an
    # open-print measurement artifact in illiquid names (the 09:15 auction
    # print).  Either way EVERY event class inherits it, so the raw signed
    # return is not a usable screening statistic -- an unconditional short
    # would "clear cost" on it.  We therefore also report ABNORMAL returns:
    # the event return minus the same-entry-date, same-liquidity-tercile panel
    # mean.  ADV terciles are cross-sectional within each date and use
    # prior-only ADV, so the control is causal.
    # ------------------------------------------------------------------
    px["adv_tier_x"] = (
        px.groupby("date")["adv20"]
        .transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False)
                   if s.notna().sum() >= 30 else np.nan)
    )
    close = px["close"].to_numpy(np.float64)
    open_ = px["open"].to_numpy(np.float64)
    gidx = px["gidx"].to_numpy(np.int64)
    pos = px["pos"].to_numpy(np.int64)
    nsym = px["n_sym"].to_numpy(np.int64)
    base_maps = {}
    for h in HORIZONS:
        ok = (pos + h - 1) < nsym
        r = np.where(ok, (close[np.where(ok, gidx + h - 1, 0)] / open_ - 1.0) * 100.0, np.nan)
        px[f"_pr{h}"] = r
        base_maps[h] = (
            px.groupby(["date", "adv_tier_x"])[f"_pr{h}"].mean().rename(f"base_h{h}")
        )
    meta = {
        "open": open_,
        "close": close,
        "n_rows": len(px),
        "base_maps": base_maps,
    }
    return px, meta


def normalise_symbol(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"^(NSE|BSE)\s*:\s*", "", regex=True)
    )


def attach_returns(events: pd.DataFrame, px: pd.DataFrame, meta: dict,
                   allow_fresh_pool: bool = False) -> pd.DataFrame:
    """events needs columns: symbol, signal_date (+ arbitrary passthrough cols).

    `allow_fresh_pool` -- by DEFAULT this function refuses to attach an entry session to
    any signal dated on/after `FRESH_POOL_CUTOFF` (2026-05-01): every screen in this
    module is development-window work and must not see the fresh pool.  A FROZEN
    candidate running its pre-registered A1 one-shot passes True, which prints a logged
    banner.  The filter stays in place; bypassing it is a deliberate, auditable act.
    """
    ev = events.copy()
    ev["symbol"] = normalise_symbol(ev["symbol"])
    ev["signal_date"] = pd.to_datetime(ev["signal_date"]).astype("datetime64[ns]")
    ev = ev[ev["signal_date"].notna() & ev["symbol"].notna()]
    if allow_fresh_pool:
        n_fresh = int((ev["signal_date"] >= FRESH_POOL_CUTOFF).sum())
        print(f"  !! attach_returns: FRESH-POOL CUTOFF BYPASSED (allow_fresh_pool=True) "
              f"-- {n_fresh} signals on/after {FRESH_POOL_CUTOFF.date()} RETAINED")
    else:
        ev = ev[ev["signal_date"] < FRESH_POOL_CUTOFF]
    if ev.empty:
        return ev

    ev = ev.sort_values("signal_date").reset_index(drop=True)
    right = px[["symbol", "date", "gidx", "pos", "n_sym", "open", "adv20",
                "prev_close", "adv_tier_x"]]
    right = right.sort_values("date")

    merged = pd.merge_asof(
        ev,
        right.rename(columns={"date": "entry_date", "open": "entry_open"}),
        left_on="signal_date",
        right_on="entry_date",
        by="symbol",
        direction="forward",
        allow_exact_matches=False,  # strictly AFTER the signal date
        tolerance=pd.Timedelta(days=ENTRY_STALENESS_DAYS),
    )
    merged = merged[merged["entry_date"].notna() & (merged["entry_open"] > 0)]
    if merged.empty:
        return merged

    gidx = merged["gidx"].to_numpy(np.int64)
    pos = merged["pos"].to_numpy(np.int64)
    n_sym = merged["n_sym"].to_numpy(np.int64)
    entry_open = merged["entry_open"].to_numpy(np.float64)
    close = meta["close"]

    for h in HORIZONS:
        exit_pos = pos + h - 1
        ok = exit_pos < n_sym
        exit_g = np.where(ok, gidx + h - 1, 0)
        exit_px = close[exit_g]
        ret = np.where(ok, (exit_px / entry_open - 1.0) * 100.0, np.nan)
        merged[f"raw_h{h}"] = ret

    # matched abnormal return: minus same-entry-date, same-ADV-tercile panel mean
    idx = pd.MultiIndex.from_arrays(
        [merged["entry_date"].to_numpy(), merged["adv_tier_x"].to_numpy()]
    )
    for h in HORIZONS:
        base = meta["base_maps"][h].reindex(idx).to_numpy(np.float64)
        merged[f"base_h{h}"] = base
        merged[f"ret_h{h}"] = merged[f"raw_h{h}"].to_numpy(np.float64) - base

    merged["era"] = np.where(
        merged["signal_date"] <= pd.Timestamp(ERA_A[1]),
        "era_A",
        np.where(merged["signal_date"] >= pd.Timestamp(ERA_B[0]), "era_B", "gap"),
    )
    merged = merged[merged["signal_date"] >= pd.Timestamp(ERA_A[0])]
    return merged


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------
ROWS: list[dict] = []


def emit(cls: str, subtype: str, df: pd.DataFrame, adv_split: bool = False) -> None:
    if df is None or len(df) == 0:
        ROWS.append(
            dict(
                event_class=cls, subtype=subtype, era="ALL", horizon=0, n=0,
                mean_abs_move=np.nan, mean_signed_long=np.nan, median_signed_long=np.nan,
                hit_rate_long=np.nan, net_exp_long=np.nan, net_exp_short=np.nan,
                frac_abs_gt_3x_cost=np.nan, tstat_signed=np.nan, verdict="NO_DATA",
            )
        )
        return

    groups = [("ALL", df)]
    for era in ("era_A", "era_B"):
        groups.append((era, df[df["era"] == era]))

    for era_lbl, sub in groups:
        for h in HORIZONS:
            d = sub[[f"ret_h{h}", f"raw_h{h}"]].dropna()
            r = d[f"ret_h{h}"]        # abnormal (primary screening statistic)
            rawr = d[f"raw_h{h}"]     # raw, for reference
            n = len(r)
            if n == 0:
                continue
            mean_signed = float(r.mean())
            std = float(r.std(ddof=1)) if n > 1 else np.nan
            tstat = mean_signed / (std / np.sqrt(n)) if (std and std > 0) else np.nan
            ROWS.append(
                dict(
                    event_class=cls,
                    subtype=subtype,
                    era=era_lbl,
                    horizon=h,
                    n=n,
                    mean_abs_move=float(r.abs().mean()),
                    mean_signed_long=mean_signed,
                    median_signed_long=float(r.median()),
                    hit_rate_long=float((r > 0).mean()),
                    net_exp_long=mean_signed - ROUND_TRIP_COST_PCT,
                    net_exp_short=-mean_signed - ROUND_TRIP_COST_PCT,
                    frac_abs_gt_3x_cost=float((r.abs() > BIG_MOVE_PCT).mean()),
                    tstat_signed=tstat,
                    raw_mean_signed_long=float(rawr.mean()),
                    raw_mean_abs_move=float(rawr.abs().mean()),
                    raw_net_exp_long=float(rawr.mean()) - ROUND_TRIP_COST_PCT,
                    raw_net_exp_short=-float(rawr.mean()) - ROUND_TRIP_COST_PCT,
                    verdict="THIN" if n < MIN_N_REPORT else "",
                )
            )

    if adv_split and "adv20" in df.columns:
        d = df[df["adv20"].notna()].copy()
        if len(d) >= 60:
            q = d["adv20"].quantile([0.33, 0.66]).to_list()
            d["adv_tier"] = np.where(
                d["adv20"] <= q[0], "adv_low",
                np.where(d["adv20"] <= q[1], "adv_mid", "adv_high"),
            )
            for tier, dd in d.groupby("adv_tier"):
                emit(cls, f"{subtype}|{tier}", dd.drop(columns=["adv_tier"]), adv_split=False)


# ----------------------------------------------------------------------------
# Class builders
# ----------------------------------------------------------------------------
def screen_pledging(px, meta):
    p = ROOT / "data" / "pledging_disclosures" / "pledging_events.parquet"
    df = pd.read_parquet(p)
    df["disclosure_time"] = pd.to_datetime(df["disclosure_time"], errors="coerce")
    df["disclosure_date"] = pd.to_datetime(df["disclosure_date"])
    # NSE corporate-announcement filings: treat as actionable next session open.
    df["signal_date"] = df["disclosure_date"]
    ev = attach_returns(
        df[["symbol", "signal_date", "event_type", "pledge_pct_before",
            "pledge_pct_after", "pledge_pct_change"]],
        px, meta,
    )
    if ev.empty:
        emit("pledging", "ALL", None)
        return

    emit("pledging", "all_disclosures", ev, adv_split=True)
    for et, sub in ev.groupby("event_type"):
        emit("pledging", f"type_{et}", sub)
    # forced-sale side: creation + invocation pooled (the brief's mechanism)
    forced = ev[ev["event_type"].isin(["pledge_creation", "pledge_invocation"])]
    emit("pledging", "creation_or_invocation", forced)
    rel = ev[ev["event_type"].isin(["pledge_revocation"])]
    emit("pledging", "revocation_only", rel)

    # pledged-% LEVEL conditioning (parsed subset)
    lvl = ev[ev["pledge_pct_after"].notna()]
    emit("pledging", "pct_after_parsed_all", lvl)
    emit("pledging", "pct_after_zero", lvl[lvl["pledge_pct_after"] <= 0.0])
    emit("pledging", "pct_after_gt0_lt20", lvl[(lvl["pledge_pct_after"] > 0) & (lvl["pledge_pct_after"] < 20)])
    emit("pledging", "pct_after_gte20", lvl[lvl["pledge_pct_after"] >= 20], adv_split=True)
    emit("pledging", "pct_after_gte50", lvl[lvl["pledge_pct_after"] >= 50])

    # pledged-% CHANGE conditioning
    chg = ev[ev["pledge_pct_change"].notna()]
    emit("pledging", "pct_change_parsed_all", chg)
    emit("pledging", "pct_change_up_gte1pp", chg[chg["pledge_pct_change"] >= 1.0], adv_split=True)
    emit("pledging", "pct_change_up_gte5pp", chg[chg["pledge_pct_change"] >= 5.0])
    emit("pledging", "pct_change_dn_lte_-1pp", chg[chg["pledge_pct_change"] <= -1.0], adv_split=True)
    emit("pledging", "pct_change_dn_lte_-5pp", chg[chg["pledge_pct_change"] <= -5.0])
    return ev


def screen_index_recon(px, meta):
    p = ROOT / "data" / "index_reconstitution" / "events.parquet"
    df = pd.read_parquet(p)
    for c in ("announcement_date", "effective_date"):
        df[c] = pd.to_datetime(df[c])

    for anchor in ("announcement_date", "effective_date"):
        d = df.rename(columns={anchor: "signal_date"})[
            ["symbol", "signal_date", "action", "index_name"]
        ]
        ev = attach_returns(d, px, meta)
        if ev.empty:
            emit("index_recon", f"{anchor}|ALL", None)
            continue
        tag = "ann" if anchor == "announcement_date" else "eff"
        emit("index_recon", f"{tag}|all", ev)
        for act, sub in ev.groupby("action"):
            emit("index_recon", f"{tag}|{act}", sub, adv_split=True)
            for idx, ss in sub.groupby("index_name"):
                emit("index_recon", f"{tag}|{act}|{idx.replace(' ', '')}", ss)
    return None


def screen_earnings_shock(px, meta):
    p = ROOT / "data" / "earnings_calendar" / "earnings_events.parquet"
    df = pd.read_parquet(p)
    df["announce_date"] = pd.to_datetime(df["announce_date"])
    df["announce_time"] = pd.to_datetime(df["announce_time"], errors="coerce")
    df["symbol"] = normalise_symbol(df["symbol"])

    # reaction day: intraday/BMO announcements react same day, AMC next session
    hour = df["announce_time"].dt.hour.fillna(18)
    same_day = df["announce_time_class"].isin(["intraday", "BMO"]) & (hour < 15)
    df["react_anchor"] = np.where(same_day, df["announce_date"], df["announce_date"] + pd.Timedelta(days=1))
    df["react_anchor"] = pd.to_datetime(df["react_anchor"])

    # map anchor -> the actual reaction session, and measure that session's move
    react = px[["symbol", "date", "close", "prev_close", "gidx", "pos", "n_sym", "adv20"]].sort_values("date")
    d = df[["symbol", "react_anchor"]].dropna().sort_values("react_anchor")
    m = pd.merge_asof(
        d, react.rename(columns={"date": "react_date"}),
        left_on="react_anchor", right_on="react_date", by="symbol",
        direction="forward", allow_exact_matches=True,
        tolerance=pd.Timedelta(days=ENTRY_STALENESS_DAYS),
    )
    m = m[m["react_date"].notna() & m["prev_close"].notna() & (m["prev_close"] > 0)]
    m["react_move"] = (m["close"] / m["prev_close"] - 1.0) * 100.0
    m["signal_date"] = m["react_date"]
    m = m[["symbol", "signal_date", "react_move"]]

    ev = attach_returns(m, px, meta)
    if ev.empty:
        emit("earnings_shock", "ALL", None)
        return
    # shock deciles computed per era so era_B is not ranked against era_A vol
    ev["abs_shock"] = ev["react_move"].abs()
    ev["shock_decile"] = ev.groupby("era")["abs_shock"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1
    )
    emit("earnings_shock", "all_events", ev)
    emit("earnings_shock", "shock_decile_1_5_small", ev[ev["shock_decile"] <= 5])
    emit("earnings_shock", "shock_decile_10_top", ev[ev["shock_decile"] == 10], adv_split=True)
    emit("earnings_shock", "shock_decile_9_10_top20pct", ev[ev["shock_decile"] >= 9])
    top = ev[ev["shock_decile"] == 10]
    emit("earnings_shock", "shock_top_decile_UP", top[top["react_move"] > 0], adv_split=True)
    emit("earnings_shock", "shock_top_decile_DOWN", top[top["react_move"] < 0], adv_split=True)
    t9 = ev[ev["shock_decile"] >= 9]
    emit("earnings_shock", "shock_top20_UP", t9[t9["react_move"] > 0])
    emit("earnings_shock", "shock_top20_DOWN", t9[t9["react_move"] < 0])
    return ev


def screen_split_bonus(px, meta):
    p = ROOT / "data" / "corporate_actions" / "split_bonus_events.parquet"
    df = pd.read_parquet(p)
    df["ex_date"] = pd.to_datetime(df["ex_date"])
    d = df.rename(columns={"ex_date": "signal_date"})[["symbol", "signal_date", "ca_type"]]
    ev = attach_returns(d, px, meta)
    if ev.empty:
        emit("split_bonus", "ALL", None)
        return
    emit("split_bonus", "ex_date_all", ev, adv_split=True)
    for ca, sub in ev.groupby("ca_type"):
        emit("split_bonus", f"ex_date_{ca}", sub)

    # pre-ex-date run-up window (signal 6 sessions before ex -> the documented leg)
    d2 = df.copy()
    d2["signal_date"] = d2["ex_date"] - pd.Timedelta(days=9)
    d2 = d2.rename(columns={"symbol": "symbol"})[["symbol", "signal_date", "ca_type"]]
    ev2 = attach_returns(d2, px, meta)
    emit("split_bonus", "pre_ex_9cd_window", ev2)
    return ev


def screen_delivery(px, meta):
    p = ROOT / "data" / "delivery_pct" / "delivery_history.parquet"
    df = pd.read_parquet(p, columns=["symbol", "date", "series", "delivery_pct",
                                     "total_traded_value"])
    df = df[df["series"] == "EQ"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"])
    # prior-only baseline
    g = df.groupby("symbol")["delivery_pct"]
    df["base"] = g.transform(lambda s: s.rolling(20, min_periods=10).median().shift(1))
    df = df[df["base"].notna() & (df["base"] > 0)]
    df["ratio"] = df["delivery_pct"] / df["base"]
    # liquidity floor so we are not screening untradeable names
    df = df[df["total_traded_value"] >= 2e7]
    df["signal_date"] = df["date"]

    hi = df[df["ratio"] >= 2.0][["symbol", "signal_date", "ratio", "delivery_pct"]]
    lo = df[df["ratio"] <= 0.5][["symbol", "signal_date", "ratio", "delivery_pct"]]
    xhi = df[df["ratio"] >= 3.0][["symbol", "signal_date", "ratio", "delivery_pct"]]
    xlo = df[df["ratio"] <= 0.35][["symbol", "signal_date", "ratio", "delivery_pct"]]

    for lbl, d in (("deliv_ratio_gte2x", hi), ("deliv_ratio_lte0p5x", lo),
                   ("deliv_ratio_gte3x", xhi), ("deliv_ratio_lte0p35x", xlo)):
        ev = attach_returns(d, px, meta)
        emit("delivery", lbl, ev, adv_split=(lbl in ("deliv_ratio_gte2x", "deliv_ratio_lte0p5x")))
    return None


def screen_basis(px, meta):
    p = ROOT / "data" / "futures_basis" / "2023_2026_basis.parquet"
    df = pd.read_parquet(p)
    df["session_date"] = pd.to_datetime(df["session_date"])
    df = df[df["days_to_expiry"].between(2, 45)]
    # basis mechanically shrinks into expiry -> percentile WITHIN dte bucket,
    # using only data available up to that session (expanding rank)
    df["dte_bucket"] = pd.cut(df["days_to_expiry"], [1, 5, 12, 22, 45],
                              labels=["dte_2_5", "dte_6_12", "dte_13_22", "dte_23_45"])
    df = df.sort_values("session_date")
    # Percentile rank WITHIN each dte bucket.  NOTE: this is a full-sample rank,
    # i.e. a mild look-ahead on the THRESHOLD only (never on returns).  Acceptable
    # for a Phase-1 magnitude screen; a Stage-0 brief must re-run with an
    # expanding-window rank.  Flagged in the report.
    df["pct_rank"] = df.groupby("dte_bucket", observed=True)["basis_pct"].rank(pct=True)
    df["signal_date"] = df["session_date"]

    cuts = {
        "basis_top1pct": df[df["pct_rank"] >= 0.99],
        "basis_top5pct": df[df["pct_rank"] >= 0.95],
        "basis_bot1pct": df[df["pct_rank"] <= 0.01],
        "basis_bot5pct": df[df["pct_rank"] <= 0.05],
    }
    for lbl, d in cuts.items():
        ev = attach_returns(d[["symbol", "signal_date", "basis_pct", "days_to_expiry"]], px, meta)
        emit("basis", lbl, ev, adv_split=(lbl in ("basis_top1pct", "basis_bot1pct")))
    return None


def screen_asm_gsm(px, meta):
    p = ROOT / "data" / "asm_gsm_history" / "asm_gsm_events.parquet"
    df = pd.read_parquet(p)
    df = df[df["exchange"] == "NSE"]
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["transition_type"].isin(["entry", "exit", "stage_up", "stage_down"])]
    df["signal_date"] = df["date"]
    for tt, sub in df.groupby("transition_type"):
        ev = attach_returns(sub[["symbol", "signal_date", "surveillance_program", "stage"]], px, meta)
        emit("asm_gsm", f"NSE_{tt}", ev)
    return None


def screen_block_deals(px, meta):
    p = ROOT / "data" / "block_deals" / "block_deals_events.parquet"
    df = pd.read_parquet(p)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["symbol"] = normalise_symbol(df["symbol"])
    df["signed_cr"] = np.where(df["buy_or_sell"].str.upper() == "BUY",
                               df["trade_value_cr"], -df["trade_value_cr"])
    agg = df.groupby(["symbol", "trade_date"]).agg(
        net_cr=("signed_cr", "sum"), gross_cr=("trade_value_cr", "sum")
    ).reset_index().rename(columns={"trade_date": "signal_date"})
    ev = attach_returns(agg, px, meta)
    if ev.empty:
        emit("block_deals", "ALL", None)
        return
    ev["deal_over_adv"] = (ev["gross_cr"] * 1e7) / ev["adv20"]
    emit("block_deals", "all_deal_days", ev)
    for lbl, q in (("size_top_decile_gross", 0.90), ("size_top_pct_gross", 0.99)):
        thr = ev["gross_cr"].quantile(q)
        emit("block_deals", lbl, ev[ev["gross_cr"] >= thr])
    big = ev[ev["gross_cr"] >= ev["gross_cr"].quantile(0.90)]
    emit("block_deals", "top_decile_NET_BUY", big[big["net_cr"] > 0])
    emit("block_deals", "top_decile_NET_SELL", big[big["net_cr"] < 0])
    d = ev[ev["deal_over_adv"].notna()]
    if len(d) > 100:
        thr = d["deal_over_adv"].quantile(0.90)
        emit("block_deals", "top_decile_deal_over_adv", d[d["deal_over_adv"] >= thr], adv_split=True)
    return None


def screen_bulk_deals(px, meta):
    p = ROOT / "data" / "bulk_deals_cache" / "nse_bulk_deals_2023_2026.parquet"
    df = pd.read_parquet(p)
    df["signal_date"] = pd.to_datetime(df["Date"], format="%d-%b-%Y", errors="coerce")
    df["symbol"] = normalise_symbol(df["Symbol"])
    qty = pd.to_numeric(df["QuantityTraded"].astype(str).str.replace(",", ""), errors="coerce")
    prc = pd.to_numeric(df["TradePrice/Wght.Avg.Price"].astype(str).str.replace(",", ""), errors="coerce")
    df["value_cr"] = qty * prc / 1e7
    df["signed_cr"] = np.where(df["Buy/Sell"].str.upper().str.startswith("B"),
                               df["value_cr"], -df["value_cr"])
    df = df[df["value_cr"].notna()]
    agg = df.groupby(["symbol", "signal_date"]).agg(
        net_cr=("signed_cr", "sum"), gross_cr=("value_cr", "sum")
    ).reset_index()
    ev = attach_returns(agg, px, meta)
    if ev.empty:
        emit("bulk_deals", "ALL", None)
        return
    ev["deal_over_adv"] = (ev["gross_cr"] * 1e7) / ev["adv20"]
    emit("bulk_deals", "all_deal_days", ev)
    thr = ev["gross_cr"].quantile(0.90)
    big = ev[ev["gross_cr"] >= thr]
    emit("bulk_deals", "size_top_decile_gross", big)
    emit("bulk_deals", "top_decile_NET_BUY", big[big["net_cr"] > 0])
    emit("bulk_deals", "top_decile_NET_SELL", big[big["net_cr"] < 0])
    d = ev[ev["deal_over_adv"].notna()]
    if len(d) > 100:
        t2 = d["deal_over_adv"].quantile(0.90)
        emit("bulk_deals", "top_decile_deal_over_adv", d[d["deal_over_adv"] >= t2], adv_split=True)
    return None


def screen_dividend(px, meta):
    p = ROOT / "data" / "dividend_ex_date" / "dividend_events.parquet"
    df = pd.read_parquet(p)
    df["ex_date"] = pd.to_datetime(df["ex_date"])
    df["symbol"] = normalise_symbol(df["symbol"])
    # prior-close for yield: use the price panel's prev_close on the ex-date row
    ref = px[["symbol", "date", "prev_close"]].rename(columns={"date": "ex_date"})
    df = df.merge(ref, on=["symbol", "ex_date"], how="left")
    df["div_yield_pct"] = df["dividend_amount_inr"] / df["prev_close"] * 100.0
    df["signal_date"] = df["ex_date"]
    ev = attach_returns(
        df[["symbol", "signal_date", "div_yield_pct", "dividend_type"]], px, meta
    )
    if ev.empty:
        emit("dividend", "ALL", None)
        return
    emit("dividend", "ex_date_all", ev)
    d = ev[ev["div_yield_pct"].notna()]
    if len(d) > 50:
        thr = d["div_yield_pct"].quantile(0.90)
        emit("dividend", "high_yield_top_decile", d[d["div_yield_pct"] >= thr], adv_split=True)
        emit("dividend", "yield_gte_3pct", d[d["div_yield_pct"] >= 3.0])
    return None


def screen_placebo(px, meta):
    """Two controls.

    1. Random (symbol, date) draws.  The matched-abnormal mean must be ~0 here,
       otherwise the date x ADV-tercile baseline is broken.
    2. ACTIVITY-CONDITIONED placebo.  Almost every event class in this screen
       selects stocks that just had an unusually active session (block deal,
       bulk deal, delivery spike, ASM entry, big earnings move, basis
       dislocation).  If active days are systematically followed by negative
       abnormal open->close returns, every class inherits a FAKE short edge.
       These rows quantify that discount, and it is direction-asymmetric, so a
       class's signed drift must be read against the matching control row --
       not against zero.
    """
    samp = px.sample(n=40000, random_state=7)[["symbol", "date"]].copy()
    ev = attach_returns(samp.rename(columns={"date": "signal_date"}), px, meta)
    emit("PLACEBO_control", "random_symbol_dates", ev)

    d = px.copy()
    d["day_ret"] = (d["close"] / d["prev_close"] - 1.0) * 100.0
    d["vol20"] = d.groupby("symbol")["volume"].transform(
        lambda s: s.rolling(20, min_periods=10).mean().shift(1)
    )
    d["vol_ratio"] = d["volume"] / d["vol20"]
    d = d.dropna(subset=["day_ret", "vol_ratio"])
    qr = d["day_ret"].abs().quantile([0.90, 0.99])
    qv = d["vol_ratio"].quantile([0.90, 0.99])
    sel = {
        "active_absret_top_decile": d[d["day_ret"].abs() >= qr[0.90]],
        "active_absret_top_1pct": d[d["day_ret"].abs() >= qr[0.99]],
        "active_volratio_top_decile": d[d["vol_ratio"] >= qv[0.90]],
        "active_volratio_top_1pct": d[d["vol_ratio"] >= qv[0.99]],
    }
    both = d[(d["day_ret"].abs() >= qr[0.90]) & (d["vol_ratio"] >= qv[0.90])]
    sel["active_absret_AND_vol_top_decile"] = both
    sel["active_both_top_decile_dayret_UP"] = both[both["day_ret"] > 0]
    sel["active_both_top_decile_dayret_DOWN"] = both[both["day_ret"] < 0]
    for lbl, s in sel.items():
        s = s.rename(columns={"date": "signal_date"})[["symbol", "signal_date"]]
        emit("PLACEBO_control", lbl, attach_returns(s, px, meta))


def screen_fno_ban(px, meta):
    p = ROOT / "data" / "fno_ban_history" / "fno_ban_events.parquet"
    df = pd.read_parquet(p)
    ROWS.append(
        dict(event_class="fno_ban", subtype=f"TOO_THIN_n{len(df)}", era="ALL", horizon=0,
             n=len(df), mean_abs_move=np.nan, mean_signed_long=np.nan,
             median_signed_long=np.nan, hit_rate_long=np.nan, net_exp_long=np.nan,
             net_exp_short=np.nan, frac_abs_gt_3x_cost=np.nan, tstat_signed=np.nan,
             verdict="NOT_SCREENED_TOO_THIN")
    )


# ----------------------------------------------------------------------------
def main():
    px, meta = load_prices()
    print(f"prices: {px.shape}  {px['date'].min().date()} -> {px['date'].max().date()}  "
          f"symbols={px['symbol'].nunique()}", flush=True)

    steps = [
        ("pledging", screen_pledging),
        ("index_recon", screen_index_recon),
        ("earnings_shock", screen_earnings_shock),
        ("split_bonus", screen_split_bonus),
        ("delivery", screen_delivery),
        ("basis", screen_basis),
        ("asm_gsm", screen_asm_gsm),
        ("block_deals", screen_block_deals),
        ("bulk_deals", screen_bulk_deals),
        ("dividend", screen_dividend),
        ("PLACEBO_control", screen_placebo),
        ("fno_ban", screen_fno_ban),
    ]
    for name, fn in steps:
        print(f"-- screening {name} ...", flush=True)
        try:
            fn(px, meta)
        except Exception as e:  # keep the screen going; a dead class is still a result
            import traceback
            traceback.print_exc()
            ROWS.append(dict(event_class=name, subtype="ERROR", era="ALL", horizon=0, n=0,
                             mean_abs_move=np.nan, mean_signed_long=np.nan,
                             median_signed_long=np.nan, hit_rate_long=np.nan,
                             net_exp_long=np.nan, net_exp_short=np.nan,
                             frac_abs_gt_3x_cost=np.nan, tstat_signed=np.nan,
                             verdict=f"ERROR:{e}"))

    out = pd.DataFrame(ROWS)
    out["signal_note"] = out["event_class"].map(CLASS_NOTES)
    out["cost_pct"] = ROUND_TRIP_COST_PCT
    # best tradeable side and whether it clears cost
    out["best_side"] = np.where(out["mean_signed_long"] >= 0, "long", "short")
    out["net_exp_best"] = out[["net_exp_long", "net_exp_short"]].max(axis=1)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False, float_format="%.4f")
    print(f"\nwrote {OUT_PATH}  rows={len(out)}")

    # -------- console summary --------
    pl = out[(out["event_class"] == "PLACEBO_control") & (out["era"] != "gap")]
    print("\n===== PLACEBO CONTROL (matched-abnormal mean must be ~0) =====")
    print(pl[["era", "horizon", "n", "mean_signed_long", "raw_mean_signed_long"]].to_string(index=False))

    piv = out[(out["era"].isin(["era_A", "era_B"])) & (out["n"] >= MIN_N_REPORT)
              & (out["event_class"] != "PLACEBO_control")]
    key = ["event_class", "subtype", "horizon"]
    cols = ["n", "mean_abs_move", "mean_signed_long", "net_exp_long",
            "net_exp_short", "frac_abs_gt_3x_cost", "tstat_signed"]
    w = piv.pivot_table(index=key, columns="era", values=cols)
    w = w.dropna(subset=[("net_exp_long", "era_A"), ("net_exp_long", "era_B")])

    show = ["n", "mean_abs_move", "mean_signed_long", "net_exp_long",
            "net_exp_short", "frac_abs_gt_3x_cost", "tstat_signed"]
    w = w[show]

    long_ok = w[(w[("net_exp_long", "era_A")] > 0) & (w[("net_exp_long", "era_B")] > 0)]
    short_ok = w[(w[("net_exp_short", "era_A")] > 0) & (w[("net_exp_short", "era_B")] > 0)]
    print("\n===== QUADRANT: BIG *AND* DIRECTIONAL -- LONG side net-positive in BOTH eras =====")
    print(long_ok.sort_values(("net_exp_long", "era_B"), ascending=False).to_string()
          if len(long_ok) else "(none)")
    print("\n===== QUADRANT: BIG *AND* DIRECTIONAL -- SHORT side net-positive in BOTH eras =====")
    print(short_ok.sort_values(("net_exp_short", "era_B"), ascending=False).to_string()
          if len(short_ok) else "(none)")

    # big moves but no direction: magnitude high, signed ~ 0 in both eras
    nodir = w[
        (w[("mean_abs_move", "era_A")] > BIG_MOVE_PCT)
        & (w[("mean_abs_move", "era_B")] > BIG_MOVE_PCT)
        & (w[("net_exp_long", "era_A")] < 0) & (w[("net_exp_long", "era_B")] < 0)
        & (w[("net_exp_short", "era_A")] < 0) & (w[("net_exp_short", "era_B")] < 0)
    ]
    print(f"\n===== QUADRANT: BIG MOVES BUT NO DIRECTION (n rows={len(nodir)}) — not tradeable as drift =====")
    print(nodir.head(40).to_string() if len(nodir) else "(none)")


if __name__ == "__main__":
    main()
