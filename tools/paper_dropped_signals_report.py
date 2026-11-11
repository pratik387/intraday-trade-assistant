"""Nightly report: what the paper book's admission limits cost.

Rules other than capital that drop paper signals (capital is a separate
switch, capital_management.paper_capital_limits_enabled; a capital refusal
shows as SHADOW_TRADE and is not covered here):
  - BAR_SCHED_BLOCK  risk:concurrent_cap_N/N, per-symbol cooloff, rate limit
  - SETUP_GROUP      setup_group_claimed (the symbol-day already traded by a
                     group member) / setup_group_lower_priority (a higher
                     ranked member won the same scan)
This tool reads a paper session's agent.log, prices every dropped signal as
if it had been taken, and reports the would-be P&L by reason.

How a dropped signal is priced (the three intraday setups share one exit
model; see structures/up_spike_fade_short_structure.py):
  - entry  : SHORT at the midpoint (open+close)/2 of the 1m bar holding the
             scan timestamp. immediate_execution setups fill on the first
             tick after the scan whatever the entry zone says (paper 11-06:
             SOLEX zone 655.9-659.8, filled 680.0), and the scan lands ~15-55s
             into a minute; on 16 real paper fills (11-06, 11-11) the midpoint
             was off by 0.38% mean / 0.26% median, vs 0.63% / 0.27% for the
             bar close and 0.51% / 0.34% for the open
  - sizing : services.risk.intraday_sizing.size_intraday_position with the
             orchestrator's config keys at capital_management.initial_capital,
             circuit-capped from the NSE price band -- the production sizer,
             not a copy of it
  - exits  : first of circuit cover (trigger circuit_cover.trigger_offset_pct
             below the upper circuit, filled AT the circuit), hard stop at
             entry x (1 + catastrophe_stop_pct), far target at entry x
             (1 - catastrophe_stop_pct), time exit at time_stop_at (open of
             that 1m bar). The walk starts at the bar AFTER the entry bar (the
             rest of the entry minute is not tested). A bar touching both a
             stop and the target counts the stop. Bars ending before
             time_stop_at = skipped, never priced at a truncated exit.
  - fees   : tools/report_utils.calculate_single_trade_charges (MIS, qty is
             the real share count in notional sizing)

The SAME model also re-prices every trade paper actually took that day and
prints model-vs-actual, so each report carries its own accuracy check.

The model ignores the daily stop and the margin pool, which is the point:
it measures the signals the rules removed. A dropped (symbol, setup) that
paper later traded anyway is listed but not counted, and a (symbol, setup)
dropped by several rules is priced once, at its first drop (all reasons
listed in `reasons`).

A night whose bars could not be fetched exits 3 and does not replace a
better earlier ledger entry for that date.

Usage (VM, after the close):
  .venv/bin/python tools/paper_dropped_signals_report.py --date 2026-11-06
  .venv/bin/python tools/paper_dropped_signals_report.py --date 2026-11-06 --logs-root logs
Writes reports/dropped_signals/<date>.json and upserts the date's rows into
reports/dropped_signals/ledger.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, asdict, field
from datetime import date, datetime, time as dtime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

try:
    from report_utils import calculate_single_trade_charges
except ImportError:
    from tools.report_utils import calculate_single_trade_charges

_TS = r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+"
_RE_BLOCK = re.compile(_TS + r".*BAR_SCHED_BLOCK \| (?P<sym>\S+) \| (?P<setup>\S+) \| (?P<reason>\S+)")
_RE_GROUP = re.compile(_TS + r".*SETUP_GROUP \| (?P<group>\S+) \| (?P<sym>\S+) \| (?P<setup>\S+) dropped: "
                       r"(?P<reason>\S+)(?: \(kept (?P<kept>\S+)\))?")


class ReportError(RuntimeError):
    pass


@dataclass
class Drop:
    symbol: str
    setup: str
    reason: str            # e.g. risk:concurrent_cap_5/5, setup_group_claimed
    rule: str              # bar_scheduler | setup_group
    scan_ts: datetime
    kept: Optional[str] = None
    repeats: int = 1
    reasons: List[str] = field(default_factory=list)


@dataclass
class Priced:
    symbol: str
    setup: str
    reason: str
    rule: str
    scan_ts: str
    repeats: int = 1
    kept: Optional[str] = None
    reasons: str = ""
    status: str = "priced"          # priced | traded_later | skipped
    skip_reason: Optional[str] = None
    entry: Optional[float] = None
    exit: Optional[float] = None
    exit_reason: Optional[str] = None
    exit_ts: Optional[str] = None
    qty: int = 0
    notional: float = 0.0
    sizing_clamp: Optional[str] = None
    ret_pct: Optional[float] = None
    gross: float = 0.0
    fees: float = 0.0
    net: float = 0.0
    actual_entry: Optional[float] = None    # calibration rows only
    actual_ret_pct: Optional[float] = None
    actual_exit_reason: Optional[str] = None
    extra: Dict = field(default_factory=dict)


def _cfg() -> dict:
    with open(_REPO / "config" / "configuration.json", encoding="utf-8") as f:
        return json.load(f)


def paper_run_dirs(logs_root: Path, d: date) -> List[Path]:
    runs = sorted(p for p in logs_root.glob(f"paper_{d:%Y%m%d}_*") if p.is_dir())
    if not runs:
        raise ReportError(f"no paper run folder for {d} under {logs_root}")
    return runs


def parse_drops(run_dirs: List[Path], setups: set) -> List[Drop]:
    """First drop per (symbol, setup) in the session; later repeats are counted."""
    seen: Dict[tuple, Drop] = {}
    for rd in run_dirs:
        log = rd / "agent.log"
        if not log.exists():
            continue
        with open(log, encoding="utf-8", errors="replace") as f:
            for line in f:
                if "BAR_SCHED_BLOCK" not in line and "SETUP_GROUP" not in line:
                    continue
                m = _RE_BLOCK.search(line)
                rule, kept = "bar_scheduler", None
                if not m:
                    m = _RE_GROUP.search(line)
                    rule = "setup_group"
                    if not m:
                        continue
                    kept = m.group("kept")
                setup = m.group("setup")
                if setup not in setups:
                    continue
                # One would-be trade per (symbol, setup): later blocks of the same
                # signal (next scan, another rule) are repeats, not new trades.
                key = (m.group("sym"), setup)
                if key in seen:
                    seen[key].repeats += 1
                    if m.group("reason") not in seen[key].reasons:
                        seen[key].reasons.append(m.group("reason"))
                    continue
                seen[key] = Drop(m.group("sym"), setup, m.group("reason"), rule,
                                 datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S"), kept,
                                 reasons=[m.group("reason")])
    return sorted(seen.values(), key=lambda x: x.scan_ts)


def executed_trades(run_dirs: List[Path], setups: set) -> pd.DataFrame:
    frames = []
    for rd in run_dirs:
        p = rd / "trade_report.csv"
        if p.exists() and p.stat().st_size:
            frames.append(pd.read_csv(p))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df[df["setup_type"].isin(setups)]
    if "executed" in df.columns:
        df = df[df["executed"].astype(str).str.lower().isin(("true", "1"))]
    return df


class Pricer:
    def __init__(self, cfg: dict, session: date, data_client):
        from services.price_bands import get_price_band_store
        self.cfg, self.session, self.dc = cfg, session, data_client
        self.capital = float(cfg["capital_management"]["initial_capital"])
        self.sz = cfg["intraday_sizing"]
        self.cc_trigger = float(cfg["circuit_cover"]["trigger_offset_pct"]) / 100.0
        self.bands = get_price_band_store(cfg)
        self._bars: Dict[str, Optional[pd.DataFrame]] = {}
        self._pdc: Dict[str, Optional[float]] = {}

    def bars(self, sym: str) -> Optional[pd.DataFrame]:
        if sym not in self._bars:
            d0 = datetime.combine(self.session, dtime(9, 15))
            d1 = datetime.combine(self.session, dtime(15, 30))
            self._bars[sym] = self.dc.get_historical_1m(sym, d0, d1)
        return self._bars[sym]

    def pdc(self, sym: str) -> Optional[float]:
        if sym not in self._pdc:
            # get_daily serves the last N bars ending today: reach back past the session.
            days = max(10, int((date.today() - self.session).days) + 10)
            df = self.dc.get_daily(sym, days)
            val = None
            if df is not None and len(df):
                idx = pd.to_datetime(df.index if not isinstance(df.index, pd.RangeIndex) else df["date"])
                keep = [t.date() < self.session for t in idx]
                prev = df[keep]
                pidx = [t for t, k in zip(idx, keep) if k]
                # The previous TRADING day's close only: a gap would feed a stale
                # close into the band / circuit math.
                if len(prev) and pidx[-1].date() == self.bands.previous_trading_day(self.session):
                    val = float(prev["close"].iloc[-1])
            self._pdc[sym] = val
        return self._pdc[sym]

    def size(self, setup: str, scfg: dict, sym: str, entry: float, pdc: float):
        from services.risk.intraday_sizing import resolve_sizing_mode, size_intraday_position
        cap, mult = self.capital, float(self.sz["book_size_multiplier"])
        dist = self.bands.circuit_distance_frac(sym, self.session, pdc, entry, "short")
        if dist is not None and dist <= self.cc_trigger:
            return None   # the orchestrator's circuit_at_entry: not a trade
        return size_intraday_position(
            setup_name=setup, sizing_mode=resolve_sizing_mode(scfg, setup),
            entry_price=entry, risk_per_share=entry * float(scfg["catastrophe_stop_pct"]) / 100.0,
            atr=None,
            vol_risk_budget_inr=float(self.sz["vol_risk_budget_pct_of_capital"]) * cap,
            stop_risk_budget_inr=float(self.sz["stop_risk_budget_pct_of_capital"]) * cap,
            target_notional_pct=scfg.get("target_notional_pct"), total_capital_inr=cap,
            min_notional_inr=float(self.sz["min_notional_pct_of_capital"]) * cap,
            max_notional_inr=float(self.sz["max_notional_pct_of_capital"]) * cap,
            book_size_multiplier=mult,
            circuit_risk_budget_inr=float(scfg["circuit_risk_budget_pct_of_capital"]) * cap * mult,
            circuit_distance_frac=dist,
            circuit_premium_frac=float(self.sz["circuit_risk_auction_premium_pct"]) / 100.0,
        )

    def price(self, row: Priced, scfg: dict, qty_override: Optional[int] = None) -> Priced:
        sym = row.symbol
        bars = self.bars(sym)
        if bars is None or bars.empty:
            row.status, row.skip_reason = "skipped", "no_1m_bars"
            return row
        scan = pd.Timestamp(row.scan_ts)
        bar_ts = scan.floor("min")
        if bar_ts not in bars.index:
            # No print in the scan minute: the first bar after it.
            later = bars.index[bars.index > bar_ts]
            if not len(later):
                row.status, row.skip_reason = "skipped", "no_bar_after_scan"
                return row
            bar_ts = later[0]
        entry = (float(bars.loc[bar_ts, "open"]) + float(bars.loc[bar_ts, "close"])) / 2.0
        pdc = self.pdc(sym)
        if pdc is None:
            row.status, row.skip_reason = "skipped", "no_previous_close"
            return row
        uc = self.bands.circuit_price(sym, self.session, pdc, "short")
        if qty_override is None:
            sz = self.size(row.setup, scfg, sym, entry, pdc)
            if sz is None:
                row.status, row.skip_reason = "skipped", "circuit_at_entry"
                return row
            if sz.qty <= 0:
                row.status, row.skip_reason = "skipped", f"sized_0:{sz.reason}"
                return row
            qty, row.sizing_clamp = int(sz.qty), sz.clamped
        else:
            qty = int(qty_override)
        stop_frac = float(scfg["catastrophe_stop_pct"]) / 100.0
        hard_sl, target = entry * (1 + stop_frac), entry * (1 - stop_frac)
        t_exit = pd.Timestamp(datetime.combine(self.session, datetime.strptime(
            scfg["time_stop_at"], "%H:%M").time()))
        exit_px = exit_reason = exit_ts = None
        for ts, b in bars[bars.index > bar_ts].iterrows():
            if ts >= t_exit:
                exit_px, exit_reason, exit_ts = float(b["open"]), "time_exit", ts
                break
            hi, lo = float(b["high"]), float(b["low"])
            trigger = uc * (1 - self.cc_trigger) if uc is not None else None
            # exit_executor covers whenever the hard stop is not below the cover trigger.
            if trigger is not None and hi >= trigger and hard_sl >= trigger:
                exit_px, exit_reason, exit_ts = float(uc), "circuit_cover", ts
                break
            if hi >= hard_sl:
                exit_px, exit_reason, exit_ts = hard_sl, "hard_sl", ts
                break
            if lo <= target:
                exit_px, exit_reason, exit_ts = target, "target", ts
                break
        if exit_px is None:
            last = bars[bars.index > bar_ts]
            if last.empty:
                row.status, row.skip_reason = "skipped", "no_bars_after_entry"
                return row
            row.status, row.skip_reason = "skipped", f"bars_end_{last.index[-1]:%H:%M}_before_time_stop"
            return row
        gross = (entry - exit_px) * qty
        fees = calculate_single_trade_charges(exit_px, entry, qty, 1.0)["total_charges"]
        row.entry, row.exit, row.exit_reason = round(entry, 2), round(exit_px, 2), exit_reason
        row.exit_ts, row.qty, row.notional = str(exit_ts), qty, round(entry * qty, 0)
        row.ret_pct = round(100.0 * (entry - exit_px) / entry, 3)
        row.gross, row.fees, row.net = round(gross, 0), round(fees, 0), round(gross - fees, 0)
        return row


def _summary(rows: List[Priced]) -> Dict:
    out: Dict[str, Dict] = {}
    for r in rows:
        if r.status != "priced":
            continue
        k = out.setdefault(r.reason, {"n": 0, "net": 0.0, "wins": 0})
        k["n"] += 1
        k["net"] += r.net
        k["wins"] += int(r.net > 0)
    return out


def upsert_ledger(path: Path, d: date, rows: List[Priced]) -> None:
    cols = ["date", "symbol", "setup", "reason", "reasons", "rule", "scan_ts", "repeats", "kept", "status",
            "skip_reason", "entry", "exit", "exit_reason", "qty", "notional", "ret_pct", "gross",
            "fees", "net"]
    keep = []
    if path.exists():
        with open(path, encoding="utf-8", newline="") as f:
            keep = [r for r in csv.DictReader(f) if r["date"] != d.isoformat()]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(keep)
        for r in rows:
            a = asdict(r)
            w.writerow({"date": d.isoformat(), **{c: a.get(c) for c in cols if c != "date"}})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--date", required=True, help="session YYYY-MM-DD")
    ap.add_argument("--logs-root", default=str(_REPO / "logs"))
    ap.add_argument("--out-dir", default=str(_REPO / "reports" / "dropped_signals"))
    args = ap.parse_args()
    d = date.fromisoformat(args.date)
    cfg = _cfg()
    setups_cfg = {n: s for n, s in cfg["setups"].items()
                  if isinstance(s, dict) and s.get("enabled") and s.get("paper_enabled")
                  and "catastrophe_stop_pct" in s and "time_stop_at" in s}
    if not setups_cfg:
        raise ReportError("no paper-enabled intraday setup with the catastrophe-stop exit model")
    tilted = [n for n, sc in setups_cfg.items()
              if isinstance(sc.get("results_size_tilt"), dict) and sc["results_size_tilt"].get("enabled")]
    if tilted:
        raise ReportError(f"results_size_tilt is on for {tilted}; this tool does not size the tilt yet")
    runs = paper_run_dirs(Path(args.logs_root), d)
    drops = parse_drops(runs, set(setups_cfg))
    trades = executed_trades(runs, set(setups_cfg))
    traded = set(zip(trades.get("symbol", []), trades.get("setup_type", []))) if len(trades) else set()

    from broker.upstox.upstox_data_client import UpstoxDataClient
    pricer = Pricer(cfg, d, UpstoxDataClient())

    rows: List[Priced] = []
    for x in drops:
        r = Priced(x.symbol, x.setup, x.reason, x.rule, x.scan_ts.isoformat(sep=" "), x.repeats, x.kept,
                   reasons=";".join(x.reasons))
        if (x.symbol, x.setup) in traded:
            r.status = "traded_later"
        else:
            try:
                pricer.price(r, setups_cfg[x.setup])
            except Exception as e:  # one bad symbol must not lose the night's report
                r.status, r.skip_reason = "skipped", f"{type(e).__name__}: {e}"
        rows.append(r)

    calib: List[Priced] = []
    for _, t in trades.iterrows():
        try:
            r = Priced(t["symbol"], t["setup_type"], "executed", "calibration", str(t["entry_ts"])[:19])
            pricer.price(r, setups_cfg[t["setup_type"]], qty_override=int(t["qty"]))
            r.actual_entry = float(t["entry_price"])
            if float(t["entry_price"]) and int(t["qty"]):
                # realized_pnl in trade_report.csv is GROSS (before charges), like ret_pct.
                r.actual_ret_pct = round(100.0 * float(t["realized_pnl"])
                                         / (float(t["entry_price"]) * int(t["qty"])), 3)
            r.actual_exit_reason = str(t.get("last_exit_reason"))
        except Exception as e:
            r = Priced(str(t.get("symbol")), str(t.get("setup_type")), "executed", "calibration", "",
                       status="skipped", skip_reason=f"{type(e).__name__}: {e}")
        calib.append(r)

    summ = _summary(rows)
    ok = [c for c in calib if c.status == "priced" and c.actual_ret_pct is not None]
    mae = (sum(abs(c.ret_pct - c.actual_ret_pct) for c in ok) / len(ok)) if ok else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "date": d.isoformat(), "capital": pricer.capital, "runs": [p.name for p in runs],
        "by_reason": summ, "dropped": [asdict(r) for r in rows],
        "calibration": {"n": len(ok), "mean_abs_ret_diff_pct": mae, "trades": [asdict(c) for c in calib]},
    }
    (out_dir / f"{d.isoformat()}.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    n_priced = sum(r.status == "priced" for r in rows)
    data_skips = [r for r in rows if r.status == "skipped" and not str(r.skip_reason).startswith(
        ("circuit_at_entry", "sized_0"))]
    ledger = out_dir / "ledger.csv"
    prev_priced = 0
    if ledger.exists():
        with open(ledger, encoding="utf-8", newline="") as f:
            prev_priced = sum(1 for x in csv.DictReader(f)
                              if x["date"] == d.isoformat() and x["status"] == "priced")
    if n_priced < prev_priced:
        print(f"LEDGER NOT UPDATED: this run priced {n_priced} < {prev_priced} already in the "
              f"ledger for {d}", file=sys.stderr)
    else:
        upsert_ledger(ledger, d, rows)

    print(f"DROPPED SIGNALS {d} | paper runs {', '.join(p.name for p in runs)} | "
          f"priced at Rs{pricer.capital:,.0f} capital")
    print(f"{'time':8} {'symbol':16} {'setup':32} {'reason':28} {'status':12} "
          f"{'entry':>9} {'exit':>9} {'why':13} {'ret%':>7} {'net Rs':>8}")
    for r in rows:
        print(f"{r.scan_ts[11:19]:8} {r.symbol:16} {r.setup:32} {r.reason:28} {r.status:12} "
              f"{r.entry or '':>9} {r.exit or '':>9} {(r.exit_reason or r.skip_reason or ''):13.13} "
              f"{r.ret_pct if r.ret_pct is not None else '':>7} {r.net if r.status == 'priced' else '':>8}"
              + (f"  [also {r.reasons}]" if ";" in r.reasons else ""))
    if not rows:
        print("(no dropped signals)")
    print("\nBY REASON (priced only)")
    for k, v in summ.items():
        print(f"  {k:30} n={v['n']:3d}  wins={v['wins']:3d}  net Rs{v['net']:>9,.0f}")
    print(f"\nMODEL CHECK on {len(ok)} executed paper trades: mean |model ret - actual ret| = "
          f"{mae:.3f} pts" if mae is not None else "\nMODEL CHECK: no executed trades to compare")
    for c in ok:
        print(f"  {c.symbol:16} {c.setup:32} model {c.ret_pct:>7.3f}% ({c.exit_reason}) | "
              f"actual {c.actual_ret_pct:>7.3f}% ({c.actual_exit_reason}) | entry {c.entry} vs {c.actual_entry}")
    if data_skips:
        print(f"\nDATA GAPS: {len(data_skips)} dropped signal(s) not priced "
              f"({', '.join(sorted({str(r.skip_reason) for r in data_skips}))})", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ReportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
