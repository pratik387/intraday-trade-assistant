"""Monthly health check for close_dn_overnight_long: research-methodology
backtest (Cell #5, byte-identical sanity_5bar functions) over a recent
window, compared against the VM's paper mirror + live ledgers.

Usage:
    python tools/sub9_research/close_dn_monthly_check.py \
        --from 2026-08-03 --to 2026-08-28 \
        [--ledger-dir <dir with decay_tripwire_close_dn_overnight_long*.json>]

Requires the window's monthly 5m feathers (download_monthly_5m.py) plus the
prior ~2 months for baseline warmup. Ledgers default to state/ (run after
scp-ing them from the VM, or pass --ledger-dir).

First used 2026-07-26 (July check): backtest PF 0.91 / mirror 1.04 / live
0.47 — the month itself was edge-less; live structure doubled the damage.
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.sub9_research.sanity_close_dn_overnight_5bar import (  # noqa: E402
    _load_window, _compute_5bar_signals, _compute_trades)
from tools.sub9_research.sanity_close_dn_overnight_long import (  # noqa: E402
    MIN_TRADING_DAYS_COVERAGE, MIN_DAILY_AVG_VOLUME)


def pf(x: np.ndarray) -> float:
    w = x[x > 0].sum()
    l = -x[x < 0].sum()
    return w / l if l > 0 else float("inf")


def led(path: Path, src=None):
    d = json.load(open(path))
    out = []
    for t in d["trades"]:
        if src and t.get("source") != src:
            continue
        ts = (t.get("ts") or t.get("ts_iso") or "")[:10]
        out.append((ts, (t.get("symbol") or "").replace("NSE:", ""),
                    t.get("net_pnl_inr") or 0))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="d0", required=True)
    ap.add_argument("--to", dest="d1", required=True)
    ap.add_argument("--ledger-dir", default=str(ROOT / "state"))
    args = ap.parse_args()
    d0, d1 = date.fromisoformat(args.d0), date.fromisoformat(args.d1)

    df = _load_window(d0, d1)
    print("bars loaded", len(df), flush=True)
    days_per_sym = df[df["hhmm"] == "15:25"].groupby("symbol", observed=True)["d"].nunique()
    total_days = df[df["hhmm"] == "15:25"]["d"].nunique()
    keep_d = days_per_sym[days_per_sym >= int(total_days * MIN_TRADING_DAYS_COVERAGE)].index
    daily_vol = df.groupby(["symbol", "d"], observed=True)["volume"].sum()
    avg_vol = daily_vol.groupby(level=0, observed=True).mean()
    keep_v = avg_vol[avg_vol >= MIN_DAILY_AVG_VOLUME / 10].index
    df = df[df["symbol"].isin(set(keep_d) & set(keep_v))]
    print("universe", df["symbol"].nunique(), flush=True)

    signals = _compute_5bar_signals(df, d0, d1)
    trades = _compute_trades(signals)

    mtf = json.load(open(ROOT / "data" / "mtf_universe" / "approved_mtf_securities_latest.json"))
    rows = mtf if isinstance(mtf, list) else (mtf.get("securities") or mtf.get("data") or [])
    etfs = {str(r.get("tradingsymbol") or r.get("symbol")).upper()
            for r in rows if isinstance(r, dict)
            and str(r.get("category") or "").lower() == "etf"}
    trades["bare"] = trades["symbol"].str.replace("NSE:", "").str.upper()
    trades = trades[~trades["bare"].isin(etfs)]

    cell = trades[(trades["closing_30m_volume_z_bin"] == "extreme")
                  & (trades["prior_day_return_bin"] == "up_gt_3pct")].copy()
    cell["month"] = pd.to_datetime(cell["signal_date"].astype(str)).dt.strftime("%Y-%m")

    print()
    print("=== BACKTEST Cell #5 (research 5-bar methodology, Rs1L, CNC fees) ===")
    for mth, g in cell.groupby("month"):
        x = g["net_pnl_inr"].values
        print("%s n=%3d net=%9.0f PF=%5.2f WR=%4.1f%% mean=%6.0f"
              % (mth, len(g), x.sum(), pf(x), 100 * (x > 0).mean(), x.mean()))

    ldir = Path(args.ledger_dir)
    mir_p = ldir / "decay_tripwire_close_dn_overnight_long.json"
    liv_p = ldir / "decay_tripwire_close_dn_overnight_long_live.json"
    lo, hi = args.d0, (pd.Timestamp(args.d1) + pd.Timedelta(days=4)).strftime("%Y-%m-%d")
    if mir_p.exists() and liv_p.exists():
        mir = [r for r in led(mir_p, src="reconstructed") if lo <= r[0] <= hi]
        liv = [r for r in led(liv_p) if lo <= r[0] <= hi]
        print()
        for name, rows_ in [("MIRROR (Rs1L, production fires)", mir),
                            ("LIVE (real fills)", liv)]:
            by_m = {}
            for r in rows_:
                by_m.setdefault(r[0][:7], []).append(r[2])
            for mth, xs in sorted(by_m.items()):
                x = np.array(xs)
                print("%-32s %s n=%3d net=%9.0f PF=%5.2f"
                      % (name, mth, len(x), x.sum(), pf(x)))
        bt = {(str(r.next_trading_day), r.bare) for r in cell.itertuples()}
        mi = {(r[0], r[1]) for r in mir}
        print()
        print("parity: backtest_signals=%d mirror_fires=%d common=%d"
              % (len(bt), len(mi), len(bt & mi)))
    else:
        print("\n(ledgers not found in %s — scp them from the VM for the "
              "mirror/live comparison)" % ldir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
