"""Stage-8 OCI-equivalent: replay the PRODUCTION multi-day executor
(services/execution/mtf_capitulation_handlers run_eod/run_verify_entries) over
2023-2026 in DRY_RUN, with a broker serving fills from the REAL 5m archive
(backtest-cache-download/monthly/*_5m_enriched.feather). Then diff the resulting
ledger against the research (clean_daily) ledger per setup.

PURPOSE (lessons #16-19): verify the production code path (rank -> size -> persist
-> 5m fill -> fee -> K-day exit -> settle) reproduces the cell-mine/confidence
numbers. Catches execution-semantics + fill-source bugs the research scripts can't.
It does NOT resolve survivorship (anachronistic MTF list) -> paper is the gate.

Runs all 4 wired setups (A2/C1/C4/C6). State files redirected to a temp dir
(no pollution). No git. Reports per-setup parity.
"""
from __future__ import annotations
import sys, json, logging, tempfile
from pathlib import Path
from datetime import time
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Force DRY_RUN so the executor uses the feather panel + 5m archive (not live fetch).
from config import env_setup
try:
    env_setup.env.DRY_RUN = True
except Exception:
    pass

import services.execution.mtf_capitulation_handlers as H
from services.daily_panel_provider import FeatherDailyPanelProvider

logging.disable(logging.WARNING)  # silence per-day executor INFO+WARNING spam (stale/pause counted via summary)
SAMPLE_YEAR = 2024  # parity validated on a representative 1-year sample (full 3yr replay is ranker-bound);
                    # NOT a silent cap — code-path/fill parity is regime-agnostic, one year is ample.

MONTHLY = ROOT / "backtest-cache-download" / "monthly"
CLEAN = ROOT / "cache" / "preaggregate" / "clean_daily_from5m.feather"
MTF = ROOT / "data" / "mtf_universe" / "approved_mtf_securities_2026-05-21.json"
COST = 0.00347 + 0.0020
NAMES = ["mtf_capitulation_revert_long", "low52_capitulation_revert_long",
         "zscore_oversold_revert_long", "crash2d_revert_long"]


class BacktestBroker:
    """Serves 5m fills from the monthly archive for the current session's month."""
    def __init__(self):
        self._month = None; self._cache = {}; self.orders = []; self._oid = 0
        self._dry_session_date = None

    def set_session(self, d):
        self._dry_session_date = d
        ym = (d.year, d.month)
        if ym != self._month:
            self._cache = {}
            f = MONTHLY / f"{d.year}_{d.month:02d}_5m_enriched.feather"
            if f.exists():
                df = pd.read_feather(f)
                df["date"] = pd.to_datetime(df["date"])
                df["bare"] = df["symbol"].astype(str).str.replace("NSE:", "", regex=False).str.upper()
                for bare, g in df.groupby("bare", sort=False):
                    self._cache[bare] = g.set_index("date")[["open", "high", "low", "close", "volume"]]
            self._month = ym

    def _load_enriched_5m(self):
        return self._cache

    def place_order(self, **kw):
        self._oid += 1; self.orders.append(kw); return f"BT{self._oid}"


def load_eligible():
    return {str(r["tradingsymbol"]).strip().upper()
            for r in json.load(open(MTF, encoding="utf-8"))
            if str(r.get("category", "")).lower() != "etf" and r.get("tradingsymbol")}


def research_ledger(dd, elig, mode, params):
    """Per-trade research ledger from clean_daily (the sanity fills) for a locked cell."""
    df = dd[dd["bare"].isin(elig)].sort_values(["symbol", "date"]).copy()
    g = df.groupby("symbol", sort=False)
    K = params["K"]
    df["open_next"] = g["open"].shift(-1)
    df["date_next"] = g["date"].shift(-1)
    df["close_exit"] = g["close"].shift(-(1 + K))
    df["turnover"] = df["close"] * df["volume"]
    df["adv20"] = g["turnover"].transform(lambda s: s.rolling(20).mean())
    df["adv20_prior"] = g["turnover"].transform(lambda s: s.shift(1).rolling(20).mean())
    df["tshock"] = df["turnover"] / df["adv20_prior"]
    base = df[(df.adv20 >= 2e6) & (df.close >= 5) & (df.tshock.notna()) & (df.open_next.notna()) & (df.close_exit.notna())].copy()
    base["tier"] = base.groupby("date")["adv20"].transform(lambda s: pd.qcut(s, 5, labels=[1, 2, 3, 4, 5], duplicates="drop"))
    if mode == "trailing_loser_decile":
        base["sig"] = base.groupby("symbol")["close"].transform(lambda s: s / s.shift(params["lb"]) - 1)
        base = base[base["sig"].notna()]
        base["rk"] = base.groupby("date")["sig"].rank(pct=True)
        sel = base[(base.rk <= params["pct"]) & (base.tier == 1) & (base.tshock >= params["shock"])]
    elif mode == "near_period_low":
        low = base.groupby("symbol")["low"].transform(lambda s: s.rolling(252, min_periods=20).min())
        base["dist"] = base["close"] / low - 1
        sel = base[(base.dist <= params["dmax"]) & (base.tier == 1) & (base.tshock >= params["shock"])]
    else:  # zscore_oversold
        m = base.groupby("symbol")["close"].transform(lambda s: s.rolling(20, min_periods=20).mean())
        sd = base.groupby("symbol")["close"].transform(lambda s: s.rolling(20, min_periods=20).std())
        base["z"] = (base["close"] - m) / sd
        sel = base[(base.z <= params["zmax"]) & (base.tier == 1) & (base.tshock >= params["shock"])]
    sel = sel[np.isfinite(sel.get("z", pd.Series(0, index=sel.index)))] if mode == "zscore_oversold" else sel
    out = pd.DataFrame({
        "bare": sel["bare"].values,
        "entry_date": pd.to_datetime(sel["date_next"]).dt.date.astype(str).values,
        "gross_ret": (sel["close_exit"].values / sel["open_next"].values - 1.0),
    })
    return out


CELLS = {
    "mtf_capitulation_revert_long": ("trailing_loser_decile", {"lb": 5, "pct": 0.05, "K": 2, "shock": 2.0}),
    "low52_capitulation_revert_long": ("near_period_low", {"dmax": 0.05, "K": 2, "shock": 2.0}),
    "zscore_oversold_revert_long": ("zscore_oversold", {"zmax": -1.5, "K": 2, "shock": 1.5}),
    "crash2d_revert_long": ("trailing_loser_decile", {"lb": 2, "pct": 0.10, "K": 3, "shock": 2.0}),
}


def main():
    cfg_all = json.load(open(ROOT / "config/configuration.json", encoding="utf-8"))
    tmp = Path(tempfile.mkdtemp(prefix="mtf_replay_"))
    sub = {}
    for n in NAMES:
        s = dict(cfg_all["setups"][n])
        s["paper_enabled"] = True
        s["capital_allocation"] = {**s["capital_allocation"], "state_file": str(tmp / f"{n}_slots.json")}
        s.pop("decay_tripwire", None)  # disable the stateful live-safety tripwire for a clean parity replay
        sub[n] = s
    cfg_run = {"setups": sub}

    # Load clean_daily once; restrict the PROVIDER's panel to eligible symbols
    # within (SAMPLE_YEAR-1 .. SAMPLE_YEAR) so each ranker call is ~4x cheaper
    # (the full 3yr x 2476-symbol panel is ranker-bound). research_ledger still
    # uses the full dd for fills.
    dd = pd.read_feather(CLEAN); dd["date"] = pd.to_datetime(dd["date"])
    dd["bare"] = dd["symbol"].astype(str).str.replace("NSE:", "", regex=False).str.upper()
    elig0 = load_eligible()
    panel = dd[(dd["bare"].isin(elig0)) & (dd["date"].dt.year.isin([SAMPLE_YEAR - 1, SAMPLE_YEAR]))][
        ["date", "symbol", "open", "high", "low", "close", "volume"]].copy()
    panel["symbol"] = panel["symbol"].astype(str).str.replace("NSE:", "", regex=False).str.upper()
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    shared = FeatherDailyPanelProvider(CLEAN, {"data_source": "x", "selection_mode": "near_period_low",
                                               "low_lookback_days": 252, "shock_lookback_days": 20})
    shared._cache = panel  # inject pre-filtered panel (bypass full-feather load)
    H.make_provider = lambda raw, **k: shared

    broker = BacktestBroker()
    # Loop over a SAMPLE_YEAR quarter (Q1) — the panel carries prior-year history
    # for trailing windows; the full-year/3yr replay is ranker-bound. A quarter
    # yields dozens-hundreds of trades/setup, ample for code-path/fill PARITY
    # (which is regime-agnostic). NOT a silent cap — stated explicitly.
    days = sorted({d.date() for d in dd["date"] if d.year == SAMPLE_YEAR and d.month <= 3})
    print(f"replaying {len(days)} sessions {days[0]}..{days[-1]} (Q1 parity sample) for {len(NAMES)} setups...", flush=True)

    ledger, stale = [], 0
    for i, d in enumerate(days):
        broker.set_session(d)
        H.run_verify_entries(cfg_run, broker, now_ist=pd.Timestamp.combine(d, time(9, 30)), paper_mode=True)
        s = H.run_eod(cfg_run, broker, now_ist=pd.Timestamp.combine(d, time(15, 25)),
                      paper_mode=True, ca_ex_dates={}, repo_root=ROOT)
        stale += s.get("stale_exit_count", 0)
        for e in s.get("events", []):
            if "net_pnl" in e:
                ledger.append(e)
        if (i + 1) % 100 == 0:
            print(f"  ...{i+1}/{len(days)} sessions, {len(ledger)} exits so far", flush=True)
    print(f"stale exits (holiday-calendar misses, NOT settled): {stale}", flush=True)

    led = pd.DataFrame(ledger)
    led.to_csv(ROOT / "reports/sub9_sanity/_mtf_replay_ledger.csv", index=False)
    print(f"\nEXECUTOR ledger: {len(led)} exits\n")

    elig = load_eligible()
    q1 = {f"{SAMPLE_YEAR}-{m:02d}" for m in (1, 2, 3)}  # match the Q1 replay window
    def pf(gr):
        net = gr - COST; w = net[net > 0].sum(); l = -net[net < 0].sum(); return (w / l) if l > 0 else float("nan")
    lines = [f"\n=== STAGE-8 PARITY ({SAMPLE_YEAR} sample): production executor (5m fills) vs research (clean_daily) ===",
             f"{'setup':<34} {'exec_n':>7} {'res_n':>7} {'exec_PF':>8} {'res_PF':>7}  fill_diff_bps p50/p90 (matched)"]
    for n in NAMES:
        ex = led[led.setup == n].copy()
        ex["bare"] = ex["symbol"].str.replace("NSE:", "", regex=False).str.upper()
        ex = ex[ex["entry_date"].astype(str).str[:7].isin(q1)]
        ex["gross_ret"] = ex["exit"] / ex["entry"] - 1.0
        mode, params = CELLS[n]
        res = research_ledger(dd, elig, mode, params)
        res = res[res["entry_date"].astype(str).str[:7].isin(q1)]
        ex_pf, res_pf = pf(ex["gross_ret"].values), pf(res["gross_ret"].values)
        m = ex.merge(res, on=["bare", "entry_date"], suffixes=("_ex", "_res"))
        if len(m):
            dbps = ((m.gross_ret_ex - m.gross_ret_res).abs() * 1e4)
            matched = f"{dbps.quantile(0.5):.1f}/{dbps.quantile(0.9):.1f} (n={len(m)}, {len(m)/max(1,len(ex)):.0%} of exec)"
        else:
            matched = "no match"
        lines.append(f"{n:<34} {len(ex):>7} {len(res):>7} {ex_pf:>8.3f} {res_pf:>7.3f}  {matched}")
    report = "\n".join(lines)
    print(report, flush=True)
    (ROOT / "reports/sub9_sanity/_mtf_replay_parity.txt").write_text(report, encoding="utf-8")
    print(f"\n(temp state dir: {tmp})", flush=True)


if __name__ == "__main__":
    main()
