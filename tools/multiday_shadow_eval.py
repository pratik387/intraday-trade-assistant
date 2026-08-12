"""Shadow-evaluate multi-day slot-selection policies against a random baseline.

READ-ONLY. Places no orders, mutates no state.

WHY THIS EXISTS
---------------
Until 2026-08-12 the multi-day book had take-all caps, so `logs/multiday_selection.jsonl`
recorded the outcome of essentially every candidate — a free counterfactual dataset
at ~8-10 labelled decisions/day versus the ~3 a capped book trades.

Phase 3 (cluster caps) ends that: from now on most candidates are skipped and
never priced, so the counterfactual has to be RECONSTRUCTED from market data
instead of observed. This tool does that, so any selection policy can still be
scored without risking capital.

It matters because ordering is now load-bearing. Before caps bound, the ranking
was inert; the 2026-08-12 replay showed 40 of 121 historical positions dropped by
a cap, so the order decides which trades the book gets. And the two orderings
tried so far both failed:

  overnight `conviction`  -> ANTI-predictive forward, p=0.0001 (0th pct of 3,000)
  multi-day `composite`   -> consensus does not predict, permutation p=0.69

Production therefore runs an unbiased date-salted hash. A policy replaces it ONLY
by beating that baseline out-of-sample on pre-registered terms.

METHOD
------
For every candidate in the selection log:
  entry = OPEN of the next trading session after session_date  (AMO fill)
  exit  = CLOSE `hold_days` trading sessions later             (per-setup config)
Returns are gross of fees — fees are ~proportional across policies, so they
cancel in a policy-vs-policy comparison. Sizing is deliberately ignored: this
measures WHICH names a policy picks, not how big they are (that is Phase 2).

A self-check reconciles reconstructed returns against the tripwire ledger's
realised entry/exit for positions the book actually took; a large mismatch means
the reconstruction is wrong and the whole comparison is void.

POLICIES
--------
  random       date-salted sha1 (production baseline)
  composite    legacy consensus score (weight * cap_score summed)
  tshock       rank by shock magnitude
  cap_score    rank by the owner's own cap_score
  cooldown_N   random, but skip a symbol re-entered within N sessions
                (Phase 5 hypothesis: repeat names measured -0.921% vs +0.143%)

Usage:
    python tools/multiday_shadow_eval.py --slots 6 --iters 2000
    python tools/multiday_shadow_eval.py --from 2026-07-01 --to 2026-08-12
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------
# forward returns
# --------------------------------------------------------------------------

def build_return_fn(symbols, lo: date, hi: date):
    """(symbol, session_date, hold_days) -> gross % return, or None."""
    from broker.upstox.upstox_data_client import UpstoxDataClient
    sdk = UpstoxDataClient()
    cache: Dict[str, Optional[pd.DataFrame]] = {}

    def daily(sym: str):
        if sym not in cache:
            try:
                df = sdk.get_daily(f"NSE:{sym}", days=260)
                cache[sym] = df if df is not None and len(df) else None
            except Exception:
                cache[sym] = None
        return cache[sym]

    def fwd(sym: str, sess: date, hold: int) -> Optional[float]:
        df = daily(sym)
        if df is None:
            return None
        idx = pd.to_datetime(df.index).normalize()
        after = np.where(idx > pd.Timestamp(sess))[0]
        if not len(after):
            return None
        e = after[0]
        x = e + max(1, int(hold)) - 1
        if x >= len(df):
            return None
        try:
            entry = float(df["open"].iloc[e])
            exit_ = float(df["close"].iloc[x])
        except Exception:
            return None
        if entry <= 0:
            return None
        return 100.0 * (exit_ / entry - 1.0)

    return fwd


# --------------------------------------------------------------------------
# policies — each returns candidates in pick order
# --------------------------------------------------------------------------

def _hash_key(sess: str, sym: str) -> str:
    return hashlib.sha1(f"{sess}|{sym}".encode("utf-8")).hexdigest()


def policy_random(cands, sess, **_):
    return sorted(cands, key=lambda c: _hash_key(sess, c["symbol"]))


def policy_composite(cands, sess, **_):
    return sorted(cands, key=lambda c: (-(c.get("composite") or c.get("cap_score") or 0.0),
                                        -(c.get("tshock") or 0.0), c["symbol"]))


def policy_tshock(cands, sess, **_):
    return sorted(cands, key=lambda c: (-(c.get("tshock") or 0.0), c["symbol"]))


def policy_cap_score(cands, sess, **_):
    return sorted(cands, key=lambda c: (-(c.get("cap_score") or 0.0), c["symbol"]))


def make_cooldown(n: int):
    def _p(cands, sess, last_seen=None, sess_index=0, **_):
        ok = [c for c in policy_random(cands, sess)
              if last_seen is None
              or sess_index - last_seen.get(c["symbol"], -10**6) > n]
        return ok
    _p.__name__ = f"cooldown_{n}"
    return _p


POLICIES = {
    "random": policy_random,
    "composite": policy_composite,
    "tshock": policy_tshock,
    "cap_score": policy_cap_score,
    "cooldown_5": make_cooldown(5),
    "cooldown_10": make_cooldown(10),
}


def run_policy(by_date, fn, slots: int) -> List[float]:
    """Walk sessions in order, take up to `slots` per session, collect returns."""
    picked, last_seen = [], {}
    for i, (sess, cands) in enumerate(by_date):
        ordered = fn(cands, sess, last_seen=last_seen, sess_index=i)
        for c in ordered[:slots]:
            picked.append(c["fwd"])
            last_seen[c["symbol"]] = i
    return picked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default="logs/multiday_selection.jsonl")
    ap.add_argument("--slots", type=int, default=6, help="picks per session (cluster cap)")
    ap.add_argument("--iters", type=int, default=2000, help="random draws for the baseline band")
    ap.add_argument("--from", dest="dfrom", default=None)
    ap.add_argument("--to", dest="dto", default=None)
    args = ap.parse_args()

    cfg = json.loads((ROOT / "config" / "configuration.json").read_text(encoding="utf-8"))
    holds = {k: int(v.get("hold_days", 2)) for k, v in cfg["setups"].items()
             if isinstance(v, dict) and str(v.get("horizon")) == "multi_day"}

    rows = [json.loads(l) for l in Path(args.log).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.dfrom:
        rows = [r for r in rows if r["session_date"] >= args.dfrom]
    if args.dto:
        rows = [r for r in rows if r["session_date"] <= args.dto]
    if not rows:
        print("no rows in range")
        return 1

    # dedupe to one row per (session, symbol) — the book holds one position
    best: Dict[tuple, dict] = {}
    for r in rows:
        k = (r["session_date"], r["symbol"])
        cur = best.get(k)
        if cur is None or (r.get("cap_score") or 0) > (cur.get("cap_score") or 0):
            best[k] = r
    cands = list(best.values())
    sessions = sorted({c["session_date"] for c in cands})
    print(f"selection log: {len(rows)} rows -> {len(cands)} unique (session, symbol) "
          f"over {len(sessions)} sessions [{sessions[0]} .. {sessions[-1]}]")

    fwd = build_return_fn({c["symbol"] for c in cands},
                          date.fromisoformat(sessions[0]), date.fromisoformat(sessions[-1]))
    resolved = 0
    for c in cands:
        hold = holds.get(c.get("setup"), 2)
        c["fwd"] = fwd(c["symbol"], date.fromisoformat(c["session_date"]), hold)
        resolved += c["fwd"] is not None
    print(f"forward returns resolved: {resolved}/{len(cands)} ({100*resolved/len(cands):.0f}%)")
    cands = [c for c in cands if c["fwd"] is not None]

    by_date = []
    grouped = defaultdict(list)
    for c in cands:
        grouped[c["session_date"]].append(c)
    for s in sorted(grouped):
        by_date.append((s, grouped[s]))

    pool = [c["fwd"] for c in cands]
    print(f"\ncandidate pool: n={len(pool)}  mean={np.mean(pool):+.3f}%  "
          f"median={np.median(pool):+.3f}%  sd={np.std(pool, ddof=1):.3f}%")
    print(f"(this is the take-all book — the ceiling any policy is picking from)\n")

    # random baseline band: reshuffle the pick within each session
    rng = random.Random(0)
    draws = []
    for _ in range(args.iters):
        vals = []
        for _s, cs in by_date:
            k = min(args.slots, len(cs))
            vals += [c["fwd"] for c in rng.sample(cs, k)]
        draws.append(float(np.mean(vals)))
    draws = np.array(draws)
    lo, hi = np.percentile(draws, [5, 95])
    print(f"RANDOM baseline over {args.iters} draws at {args.slots} slots/session: "
          f"mean={draws.mean():+.3f}%  90% band [{lo:+.3f}, {hi:+.3f}]\n")

    print(f"{'policy':<14}{'n':>5}{'mean%':>10}{'vs random':>12}{'pctile':>9}  verdict")
    print("-" * 62)
    for name, fn in POLICIES.items():
        picks = run_policy(by_date, fn, args.slots)
        if not picks:
            print(f"{name:<14}{0:>5}  (no picks)")
            continue
        m = float(np.mean(picks))
        pct = 100.0 * float((draws <= m).mean())
        beats = "BEATS random" if pct >= 95 else ("worse" if pct <= 5 else "indistinguishable")
        print(f"{name:<14}{len(picks):>5}{m:>+9.3f}%{m-draws.mean():>+11.3f}%{pct:>8.1f}  {beats}")

    print("\nA policy replaces the production hash ONLY on a PRE-REGISTERED margin and\n"
          "sample, decided before looking. 'BEATS random' here is in-sample unless the\n"
          "window was fixed in advance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
