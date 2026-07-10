"""Target-exit study for the 4 multiday CNC/MTF capitulation setups (2026-07-10).

Question (user): exit at a TARGET if touched within the hold window, else the
usual K-day close — does it beat hold-to-close?

Three target families, all computed AT ENTRY from signal-date data (no lookahead):
  F: fixed pct of entry                        x in {2,3,4,5,6,8}%
  V: vol-scaled  entry*(1 + k*sigma20/close)   k in {0.5,0.75,1.0,1.5,2.0}
  M: mechanism-anchored — zscore: 20d mean; loser/low52: retrace r of the
     measured drop (r in {0.382, 0.5})

Touch rule: entry at T+1 open; a day whose HIGH >= target exits AT target
(gap-open above target exits at open). Else exit at the K-day close (baseline).
Fees: corrected MTF model (incl. Rs20/order brokerage + pledge/unpledge) at the
Rs1L-margin live-plan sizing; early exits pay FEWER MTF interest days.
Judged on per-period PF stability (2023/2024/2025/2026), not top-cell PF.

Caveats printed with results: daily-high touch flatters fills on illiquid names;
capping winners' tails is measured, not assumed free.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.sub7_validation.build_per_setup_pnl import calc_fee_mtf  # noqa: E402

CLEAN = ROOT / "cache" / "preaggregate" / "clean_daily_from5m.feather"
MTF = ROOT / "data" / "mtf_universe" / "approved_mtf_securities_2026-05-21.json"
MARGIN = 100000.0
LEV = 2.79

CELLS = {
    "mtf_capitulation_revert_long": ("trailing_loser_decile", {"lb": 5, "pct": 0.05, "K": 2, "shock": 2.0}),
    "low52_capitulation_revert_long": ("near_period_low", {"dmax": 0.05, "K": 2, "shock": 2.0}),
    "zscore_oversold_revert_long": ("zscore_oversold", {"zmax": -1.5, "K": 2, "shock": 1.5}),
    "crash2d_revert_long": ("trailing_loser_decile", {"lb": 2, "pct": 0.10, "K": 3, "shock": 2.0}),
}


def load_eligible():
    import json
    data = json.loads(MTF.read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else (data.get("securities") or data.get("data") or data)
    out = set()
    for r in rows if isinstance(rows, list) else rows.values():
        if isinstance(r, dict):
            s = r.get("tradingsymbol") or r.get("symbol")
            if s:
                out.add(str(s).replace("NSE:", "").upper())
    return out


def build_trades(dd, elig, mode, params):
    """Selection identical to backtest_mtf_replay.research_ledger, but keeps
    per-trade path arrays (highs/opens/closes over the hold) + at-entry features."""
    df = dd[dd["bare"].isin(elig)].sort_values(["symbol", "date"]).copy()
    g = df.groupby("symbol", sort=False)
    K = params["K"]
    df["open_next"] = g["open"].shift(-1)
    df["date_next"] = g["date"].shift(-1)
    for j in range(1, K + 1):
        df[f"h{j}"] = g["high"].shift(-j)   # day T+j high
        df[f"o{j}"] = g["open"].shift(-j)
        df[f"d{j}"] = g["date"].shift(-j)
    df["close_exit"] = g["close"].shift(-(1 + K))
    df["date_exit"] = g["date"].shift(-(1 + K))
    df[f"h{K+1}"] = g["high"].shift(-(K + 1))
    df[f"o{K+1}"] = g["open"].shift(-(K + 1))
    df["turnover"] = df["close"] * df["volume"]
    df["adv20"] = g["turnover"].transform(lambda s: s.rolling(20).mean())
    df["adv20_prior"] = g["turnover"].transform(lambda s: s.shift(1).rolling(20).mean())
    df["tshock"] = df["turnover"] / df["adv20_prior"]
    df["mean20"] = g["close"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    df["std20"] = g["close"].transform(lambda s: s.rolling(20, min_periods=20).std())
    base = df[(df.adv20 >= 2e6) & (df.close >= 5) & df.tshock.notna()
              & df.open_next.notna() & df.close_exit.notna()].copy()
    base["tier"] = base.groupby("date")["adv20"].transform(
        lambda s: pd.qcut(s, 5, labels=[1, 2, 3, 4, 5], duplicates="drop"))
    if mode == "trailing_loser_decile":
        base["sig"] = base.groupby("symbol")["close"].transform(lambda s: s / s.shift(params["lb"]) - 1)
        base["ref_prior"] = base.groupby("symbol")["close"].transform(lambda s: s.shift(params["lb"]))
        base = base[base["sig"].notna()]
        base["rk"] = base.groupby("date")["sig"].rank(pct=True)
        sel = base[(base.rk <= params["pct"]) & (base.tier == 1) & (base.tshock >= params["shock"])]
    elif mode == "near_period_low":
        low = base.groupby("symbol")["low"].transform(lambda s: s.rolling(252, min_periods=20).min())
        base["dist"] = base["close"] / low - 1
        base["ref_prior"] = base.groupby("symbol")["close"].transform(lambda s: s.shift(20))
        sel = base[(base.dist <= params["dmax"]) & (base.tier == 1) & (base.tshock >= params["shock"])]
    else:
        base["z"] = (base["close"] - base["mean20"]) / base["std20"]
        base = base[np.isfinite(base["z"])]
        base["ref_prior"] = base["mean20"]
        sel = base[(base.z <= params["zmax"]) & (base.tier == 1) & (base.tshock >= params["shock"])]
    return sel.copy(), K


def simulate(sel, K, target_px_series):
    """Vectorized-ish touch sim. Returns (net array, hit%, avg_hold_days)."""
    entry = sel["open_next"].values
    tgt = target_px_series
    exit_px = sel["close_exit"].values.copy()
    hold = np.full(len(sel), float(K + 1))  # calendar-ish trading days entry->close exit
    done = np.zeros(len(sel), bool)
    for j in range(1, K + 2):
        h = sel[f"h{j}"].values if f"h{j}" in sel else None
        o = sel[f"o{j}"].values
        if h is None:
            break
        # j==1 is the ENTRY day itself (entry at open); a touch after entry counts.
        touch = ~done & np.isfinite(tgt) & (tgt > entry) & (h >= tgt)
        if j == K + 1:
            # past the normal exit day — baseline already exited at K-day close
            break
        fill = np.where(o >= tgt, o, tgt)
        exit_px[touch] = fill[touch]
        hold[touch] = float(j)
        done |= touch
    return exit_px, hold, done


def net_pnl(entry, exit_px, hold_days):
    notional = MARGIN * LEV
    qty = notional / entry
    gross = (exit_px - entry) * qty
    fees = np.array([calc_fee_mtf(notional, float(e * q), MARGIN, int(h))
                     for e, q, h in zip(exit_px, qty, hold_days)])
    return gross - fees


def pf(x):
    w = x[x > 0].sum(); l = -x[x < 0].sum()
    return w / l if l > 0 else np.inf


def main():
    dd = pd.read_feather(CLEAN)
    dd["date"] = pd.to_datetime(dd["date"])
    dd["bare"] = dd["symbol"].astype(str).str.replace("NSE:", "", regex=False).str.upper()
    elig = load_eligible()
    for name, (mode, params) in CELLS.items():
        sel, K = build_trades(dd, elig, mode, params)
        sel["year"] = pd.to_datetime(sel["date_next"]).dt.year
        entry = sel["open_next"].values
        variants = [("BASELINE hold-to-close", None)]
        for x in (2, 3, 4, 5, 6, 8):
            variants.append((f"F fixed +{x}%", entry * (1 + x / 100.0)))
        s20 = (sel["std20"] / sel["close"]).values
        for k in (0.5, 0.75, 1.0, 1.5, 2.0):
            variants.append((f"V vol k={k}", entry * (1 + k * s20)))
        if mode == "zscore_oversold":
            variants.append(("M mean20", sel["mean20"].values))
        else:
            drop = (sel["ref_prior"] - sel["close"]).values  # measured decline at T
            for r in (0.382, 0.5):
                variants.append((f"M retr {r}", sel["close"].values + r * drop))
        print(f"\n=== {name} (n={len(sel)}, K={K}) ===")
        print(f"{'variant':<24}{'poolPF':>7}{'PF23':>6}{'PF24':>6}{'PF25':>6}{'PF26':>6}{'net':>10}{'hit%':>6}{'hold':>5}")
        for label, tgt in variants:
            if tgt is None:
                exit_px, hold, done = sel["close_exit"].values, np.full(len(sel), K + 1.0), np.zeros(len(sel), bool)
            else:
                exit_px, hold, done = simulate(sel, K, np.asarray(tgt, float))
            net = net_pnl(entry, exit_px, hold)
            yr = sel["year"].values
            pfs = [pf(net[yr == y]) if (yr == y).sum() > 5 else float("nan") for y in (2023, 2024, 2025, 2026)]
            print(f"{label:<24}{pf(net):>7.2f}{pfs[0]:>6.2f}{pfs[1]:>6.2f}{pfs[2]:>6.2f}{pfs[3]:>6.2f}{net.sum():>10.0f}{100*done.mean():>5.0f}%{hold.mean():>5.1f}")
    print("\nCaveats: daily-high touch flatters illiquid fills; targets capped winners' tails are IN these numbers.")


if __name__ == "__main__":
    main()
