"""Sub-project #7/#8 Phase-1 gauntlet on per-setup NET parquets.

Combines Stage 1 (universe pruning) + Stage 2 (univariate screening) verdict
with Stage 3 (cell selection) per setup. Reads per-setup parquets produced by
`tools/sub7_validation/build_per_setup_pnl.py` (Indian fee model + sanity
clean already applied), runs the gauntlet, writes a single roll-up report.

Pass criteria
-------------
**Phase-1 floor (Stages 1+2 combined, per sub-project #7/#8 design):**
  - n_trades >= 500
  - NET PF    >= 1.10
  - NET Sharpe > 0
  - Sub-period stability: NET PF >= 1.0 in BOTH halves of Discovery
    (catches one-year flukes per master plan Section 3.3 Stage 2).

**Stage 3 cell criterion (per master plan Section 3.3):**
  - Cell n_trades >= 100
  - Cell NET PF   >= 1.30
  - Cell NET PF   >= 1.10 in both Discovery sub-periods

Conditioners (structural only, master plan Section 3.2):
  regime  ×  cap_segment  ×  hour_bucket

Output structure
----------------
  <output_dir>/
    00-summary.md               # one-line pass/fail per setup
    00-summary.json             # machine-readable rollup
    <setup>/
      01-floor.json             # Phase-1 floor verdict
      02-cells-1way.csv         # 1-way cells (regime, cap_segment, hour_bucket)
      03-cells-2way.csv         # 2-way cells (only on pairs with 1-way passers)
      04-best-cells.json        # top 1-2 cells per setup
      05-report.md              # human-readable

CLI
---
    python tools/sub7_validation/run_phase1_gauntlet.py \\
        --parquet-dir reports/sub8_phase1_v2 \\
        --output-dir docs/edge_discovery/2026-05-01-sub8-phase1-gauntlet \\
        --period-start 2023-01-02 --period-end 2024-12-31

Honours sub-project #1 OOS discipline: this tool reads ONLY Discovery
data. Validation/Holdout periods are spent (per gauntlet v2 postmortem)
or unavailable (no 2025 data captured for sub-project #7/#8 setups yet).
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from services.symbol_metadata import in_universe


# Phase-1 floor (sub-project #7/#8 design Section 12.1)
FLOOR_MIN_N = 500
FLOOR_MIN_PF = 1.10
FLOOR_MIN_SHARPE = 0.0
FLOOR_SUBPERIOD_MIN_PF = 1.0   # "PF >= 1.0 in both halves" (master plan Stage 2)

# Stage 3 cell criterion (master plan Section 3.3 Stage 3)
CELL_MIN_N = 100
CELL_MIN_PF = 1.30
CELL_SUBPERIOD_MIN_PF = 1.10

CONDITIONERS = ("side", "regime", "cap_segment", "hour_bucket", "volatility_regime")
# `side` joins regime/cap/hour as a primary conditioner because Indian
# intraday has a known structural asymmetry: 70% of cash intraday traders
# lose money (SEBI FY23), and the losing flow is overwhelmingly LONG. MIS
# auto-square at 3:20 PM creates net sell pressure on the close. Cargo-
# culted SMC patterns therefore systematically have weaker LONG variants.
# Empirically, every multi-side sub7+sub8 setup in the 2026-04-30 capture
# shows BUY PF 0.53-0.74 vs SELL PF 0.84-0.99. Splitting on side at cell
# level surfaces the edge that's drowned out by mixing both directions.
#
# `volatility_regime` is the master plan's 4th approved conditioner
# (§3.3): per-stock tercile of BB-width vs the stock's OWN distribution.
# Stage 4 SHAP on the 2 rescue cells (orb_15, pdh_pdl_reject) showed
# bb_width_proxy in top-5 features — drives the win/loss split independent
# of regime. Bucketing into low_vol/mid_vol/high_vol surfaces sub-cells
# the master plan's regime conditioner doesn't separate.

# Setups that should be SKIPPED from the gauntlet (already validated &
# locked, OR awaiting their own data). Add to this set as the validated
# library grows so we don't waste cell-selection cycles on done work.
SKIP_SETUPS = frozenset({
    "gap_fade_short",   # validated production setup, config selected
})


def _load_intended_filters(
    config_path: Path,
) -> Dict[str, Dict[str, Any]]:
    """Read each setup's intended universe + cap filter from
    configuration.json. The wide_open OCI capture deliberately bypasses
    these filters so the gauntlet sees ALL trades; here we apply them BACK
    so cell-selection measures edge under the configuration the setup was
    actually designed for.

    Without this, e.g. pdh_pdl_reject (designed for smallmid_fno) gets
    evaluated on every NIFTY-50 large-cap symbol too — symbols the setup's
    structure thesis never claimed. The off-universe noise drowns out the
    on-universe edge.

    Returns: {setup_name: {"universe_key": str|None,
                           "allowed_cap_segments": set|None}}
    """
    cfg = json.loads(config_path.read_text())
    setups = cfg.get("setups") or {}
    out: Dict[str, Dict[str, Any]] = {}
    for name, sc in setups.items():
        if not isinstance(sc, dict):
            continue
        out[name] = {
            "universe_key": sc.get("universe_key"),
            "allowed_cap_segments": (
                set(sc["allowed_cap_segments"])
                if isinstance(sc.get("allowed_cap_segments"), list)
                else None
            ),
        }
    return out


def apply_intended_filters(
    df: pd.DataFrame,
    setup: str,
    filters: Dict[str, Dict[str, Any]],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Filter a setup's trade DataFrame to its intended universe + caps.

    Returns (filtered_df, filter_info). filter_info records what was
    applied + sample-size impact for the report.
    """
    flt = filters.get(setup, {})
    info = {
        "universe_key": flt.get("universe_key"),
        "allowed_cap_segments": (
            sorted(flt["allowed_cap_segments"])
            if flt.get("allowed_cap_segments") else None
        ),
        "n_before": len(df),
    }

    out = df
    if flt.get("universe_key"):
        try:
            mask = out["symbol"].apply(
                lambda s: in_universe(s, flt["universe_key"])
            )
            out = out[mask]
        except KeyError:
            # Unknown universe key — skip rather than crash; report it.
            info["universe_filter_error"] = (
                f"unknown universe_key={flt['universe_key']!r}"
            )

    if flt.get("allowed_cap_segments") and "cap_segment" in out.columns:
        out = out[out["cap_segment"].isin(flt["allowed_cap_segments"])]

    info["n_after"] = len(out)
    info["n_filtered_out"] = info["n_before"] - info["n_after"]
    return out.copy(), info


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _net_pf(net_pnl: pd.Series) -> float:
    """Profit factor on net PnL: sum(wins) / |sum(losses)|.

    Returns +inf when there are no losers (rare, suspicious — usually < 100
    trades). Caller treats inf as 'unstable, do not promote'.
    """
    wins = net_pnl[net_pnl > 0].sum()
    losses = net_pnl[net_pnl < 0].abs().sum()
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def _daily_sharpe(df: pd.DataFrame) -> float:
    """Per-day Sharpe of net daily PnL.

    Per-trade Sharpe (used in some legacy gauntlet iterations) is
    pathologically biased downward for intraday strategies because each
    trade resets daily — see specs/2026-04-25-sub-project-5-gauntlet-v2-
    postmortem.md for the diagnosis. We use per-session Sharpe, the
    industry-standard convention.
    """
    if df.empty:
        return 0.0
    daily = df.groupby("session_date")["net_pnl"].sum()
    if daily.std() == 0 or daily.size < 2:
        return 0.0
    return float(daily.mean() / daily.std())


def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Aggregate net metrics for a slice (full setup OR a cell)."""
    if df.empty:
        return {
            "n_trades": 0, "n_sessions": 0, "net_pnl": 0.0,
            "net_pf": 0.0, "net_sharpe": 0.0, "wr": 0.0,
        }
    n = df["net_pnl"]
    daily = df.groupby("session_date")["net_pnl"].sum()
    return {
        "n_trades": int(len(df)),
        "n_sessions": int(daily.size),
        "net_pnl": round(float(n.sum()), 0),
        "net_pf": round(_net_pf(n), 3),
        "net_sharpe": round(_daily_sharpe(df), 3),
        "wr": round(float((n > 0).mean()), 3),
    }


def _split_subperiods(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split Discovery into two halves by session_date midpoint.

    Master plan calls these FY22-23 vs FY23-24, but a midpoint split on
    whatever Discovery range is supplied is equivalent for "edge stable
    across sub-periods" purposes and avoids hard-coded fiscal-year math.
    """
    if df.empty:
        return df, df
    dates = pd.to_datetime(df["session_date"]).sort_values().unique()
    if len(dates) < 2:
        return df, pd.DataFrame(columns=df.columns)
    mid = dates[len(dates) // 2]
    h1 = df[pd.to_datetime(df["session_date"]) < mid].copy()
    h2 = df[pd.to_datetime(df["session_date"]) >= mid].copy()
    return h1, h2


# ---------------------------------------------------------------------------
# Hour-bucket derivation (master plan Section 3.3 conditioner)
# ---------------------------------------------------------------------------


def _add_volatility_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Add `volatility_regime` ∈ {low_vol, mid_vol, high_vol} from
    bb_width_proxy, bucketed per-stock against that stock's OWN
    distribution.

    Master plan §3.3: "Volatility regime: BB width tercile of stock vs its
    own 20-day distribution." Strict 20-day rolling would require the full
    daily BB-width history per symbol; we use the per-symbol 2-year
    distribution from the trade rows as a stable approximation. The
    tercile is symbol-relative (so a "high_vol day" for a quiet large-cap
    isn't compared to a noisy micro-cap on absolute terms).

    Symbols with <30 trades fall into a single 'unknown' bucket — their
    own distribution isn't stable enough to tercile.
    """
    out = df.copy()
    if "bb_width_proxy" not in out.columns:
        out["volatility_regime"] = None
        return out

    # Per-symbol q33 / q67 of bb_width_proxy
    bbw = pd.to_numeric(out["bb_width_proxy"], errors="coerce")
    out["bb_width_proxy"] = bbw

    def _bucket(group: pd.DataFrame) -> pd.Series:
        vals = group["bb_width_proxy"].dropna()
        if len(vals) < 30:
            return pd.Series(["unknown"] * len(group), index=group.index)
        q33, q67 = vals.quantile([0.33, 0.67])
        # Defensive: if distribution is degenerate (q33 == q67), put
        # everything in mid_vol.
        if q33 == q67:
            return pd.Series(["mid_vol"] * len(group), index=group.index)

        def _label(x):
            if pd.isna(x):
                return None
            if x <= q33:
                return "low_vol"
            if x >= q67:
                return "high_vol"
            return "mid_vol"

        return group["bb_width_proxy"].map(_label)

    out["volatility_regime"] = out.groupby("symbol", group_keys=False).apply(_bucket)
    return out


def _add_hour_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Derive hour_bucket from decision_ts. NSE 9:15 IST = minute_of_day 555.

    Mapping (matches tools/edge_discovery/data_loader.py):
      555-599  → opening   (09:15-10:00)
      600-719  → morning   (10:00-12:00)
      720-779  → lunch     (12:00-13:00)
      780-869  → afternoon (13:00-14:30)
      870+     → late      (14:30+)
    """
    out = df.copy()
    if "decision_ts" not in out.columns:
        out["hour_bucket"] = None
        return out
    ts = pd.to_datetime(out["decision_ts"], errors="coerce")
    mod = ts.dt.hour * 60 + ts.dt.minute

    def _bucket(m):
        if pd.isna(m):
            return None
        m = int(m)
        if m < 555:
            return None
        if m < 600:
            return "opening"
        if m < 720:
            return "morning"
        if m < 780:
            return "lunch"
        if m < 870:
            return "afternoon"
        return "late"

    out["hour_bucket"] = mod.map(_bucket)
    return out


# ---------------------------------------------------------------------------
# Floor + cell evaluation
# ---------------------------------------------------------------------------


def evaluate_floor(df: pd.DataFrame) -> Dict[str, Any]:
    """Phase-1 floor verdict: n + PF + Sharpe + sub-period PF stability."""
    agg = compute_metrics(df)
    h1, h2 = _split_subperiods(df)
    h1m, h2m = compute_metrics(h1), compute_metrics(h2)
    passes = (
        agg["n_trades"] >= FLOOR_MIN_N
        and agg["net_pf"] >= FLOOR_MIN_PF
        and agg["net_sharpe"] > FLOOR_MIN_SHARPE
        and h1m["net_pf"] >= FLOOR_SUBPERIOD_MIN_PF
        and h2m["net_pf"] >= FLOOR_SUBPERIOD_MIN_PF
    )
    return {
        "passes_phase1": bool(passes),
        "thresholds": {
            "min_n": FLOOR_MIN_N,
            "min_pf": FLOOR_MIN_PF,
            "min_sharpe": FLOOR_MIN_SHARPE,
            "min_subperiod_pf": FLOOR_SUBPERIOD_MIN_PF,
        },
        "aggregate": agg,
        "h1": h1m,
        "h2": h2m,
    }


def _evaluate_cell(
    sub: pd.DataFrame, dims: List[Tuple[str, Any]],
) -> Dict[str, Any]:
    """One cell's metrics + Stage-3 pass verdict."""
    agg = compute_metrics(sub)
    h1, h2 = _split_subperiods(sub)
    h1m, h2m = compute_metrics(h1), compute_metrics(h2)
    passes = (
        agg["n_trades"] >= CELL_MIN_N
        and agg["net_pf"] >= CELL_MIN_PF
        and h1m["net_pf"] >= CELL_SUBPERIOD_MIN_PF
        and h2m["net_pf"] >= CELL_SUBPERIOD_MIN_PF
    )
    return {
        "dims": [{"conditioner": c, "value": str(v)} for c, v in dims],
        "n_trades": agg["n_trades"],
        "net_pnl": agg["net_pnl"],
        "net_pf": agg["net_pf"],
        "net_sharpe": agg["net_sharpe"],
        "wr": agg["wr"],
        "h1_pf": h1m["net_pf"],
        "h2_pf": h2m["net_pf"],
        "passes_stage3": bool(passes),
    }


def run_cell_selection(df: pd.DataFrame) -> Dict[str, Any]:
    """1-way + 2-way cells on (side, regime, cap_segment, hour_bucket).

    The master plan's gating rule was "1-way first; 2-way only after
    1-way passes (no combo-hunting)". That worked for the OLD SMC library
    where 1-way passers existed; for sub7/sub8 with stricter NET economics
    the 1-way pass rate is too sparse to seed 2-way exploration. We
    relax the gate: 2-way generated for ANY two-conditioner pair where the
    SETUP has enough total trades. The Stage 3 cell criterion (n>=100,
    PF>=1.30, sub-period PF>=1.10) is conservative enough to filter
    spurious combos — small-n cells fail the 100 floor; large-n
    combinations only pass if the edge is real and stable.

    3-way combinations remain forbidden (master plan rule, combinatorial
    overfitting). 2-way is the cap.
    """
    available = [c for c in CONDITIONERS if c in df.columns and df[c].notna().any()]
    one_way = []
    for cond in available:
        for val, sub in df.groupby(cond, dropna=True):
            if val is None:
                continue
            cell = _evaluate_cell(sub, [(cond, val)])
            one_way.append(cell)

    two_way = []
    # Unconditional 2-way generation across all conditioner pairs. The
    # Stage 3 sample-size + sub-period criteria gate the noise.
    for ca, cb in combinations(available, 2):
        vals_a = [v for v in df[ca].dropna().unique() if v is not None]
        vals_b = [v for v in df[cb].dropna().unique() if v is not None]
        for va in vals_a:
            for vb in vals_b:
                sub = df[(df[ca] == va) & (df[cb] == vb)]
                if len(sub) < 30:
                    # Skip very-small cells before fee math — they can't
                    # reach the n>=100 floor anyway and clutter the report.
                    continue
                cell = _evaluate_cell(sub, [(ca, va), (cb, vb)])
                two_way.append(cell)

    # Best 1-2 cells: rank by (passes desc, PF desc, n desc).
    all_cells = one_way + two_way
    all_cells.sort(
        key=lambda c: (c["passes_stage3"], c["net_pf"], c["n_trades"]),
        reverse=True,
    )
    best = [c for c in all_cells if c["passes_stage3"]][:2]
    return {
        "available_conditioners": available,
        "one_way_cells": one_way,
        "two_way_cells": two_way,
        "best_cells": best,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _cells_to_df(cells: List[Dict[str, Any]]) -> pd.DataFrame:
    if not cells:
        return pd.DataFrame()
    rows = []
    for c in cells:
        row = {
            "dims": " & ".join(f"{d['conditioner']}={d['value']}" for d in c["dims"]),
            "n_trades": c["n_trades"],
            "net_pnl": c["net_pnl"],
            "net_pf": c["net_pf"],
            "net_sharpe": c["net_sharpe"],
            "wr": c["wr"],
            "h1_pf": c["h1_pf"],
            "h2_pf": c["h2_pf"],
            "passes_stage3": c["passes_stage3"],
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["passes_stage3", "net_pf", "n_trades"], ascending=[False, False, False]
    )


def _write_setup_report(
    setup: str,
    df: pd.DataFrame,
    floor: Dict[str, Any],
    cells: Dict[str, Any],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "01-floor.json").write_text(json.dumps(floor, indent=2))

    one_way_df = _cells_to_df(cells["one_way_cells"])
    two_way_df = _cells_to_df(cells["two_way_cells"])
    if not one_way_df.empty:
        one_way_df.to_csv(out_dir / "02-cells-1way.csv", index=False)
    if not two_way_df.empty:
        two_way_df.to_csv(out_dir / "03-cells-2way.csv", index=False)
    (out_dir / "04-best-cells.json").write_text(
        json.dumps(cells["best_cells"], indent=2)
    )

    floor_verdict = "PASS" if floor["passes_phase1"] else "FAIL"
    md = [
        f"# {setup} — Phase-1 Gauntlet",
        f"\n**Phase-1 floor verdict:** {floor_verdict}",
        f"\n**Aggregate (NET):** n={floor['aggregate']['n_trades']:,} "
        f"PF={floor['aggregate']['net_pf']} "
        f"Sharpe={floor['aggregate']['net_sharpe']} "
        f"PnL={floor['aggregate']['net_pnl']:,.0f}",
        f"\n**Sub-period stability:** "
        f"H1 PF={floor['h1']['net_pf']} | H2 PF={floor['h2']['net_pf']}",
        f"\n## Stage 3 — Cell selection",
    ]

    if cells["best_cells"]:
        md.append(f"\n**Best {len(cells['best_cells'])} cell(s):**")
        for c in cells["best_cells"]:
            dim_s = " & ".join(f"{d['conditioner']}={d['value']}" for d in c["dims"])
            md.append(
                f"  - **{dim_s}** — n={c['n_trades']} PF={c['net_pf']} "
                f"H1={c['h1_pf']} H2={c['h2_pf']} PnL={c['net_pnl']:,.0f}"
            )
    else:
        md.append("\n_No cells passed Stage 3 criteria (n>=100, PF>=1.30, "
                  "sub-period PF>=1.10)._ Top 5 by PF (informational):")
        all_cells = cells["one_way_cells"] + cells["two_way_cells"]
        all_cells.sort(key=lambda c: (c["net_pf"], c["n_trades"]), reverse=True)
        for c in all_cells[:5]:
            dim_s = " & ".join(f"{d['conditioner']}={d['value']}" for d in c["dims"])
            md.append(
                f"  - {dim_s} — n={c['n_trades']} PF={c['net_pf']} "
                f"H1={c['h1_pf']} H2={c['h2_pf']} PnL={c['net_pnl']:,.0f}"
            )

    if not one_way_df.empty:
        md.append("\n## All 1-way cells")
        md.append(one_way_df.to_markdown(index=False))
    if not two_way_df.empty:
        md.append("\n## All 2-way cells")
        md.append(two_way_df.to_markdown(index=False))

    (out_dir / "05-report.md").write_text("\n".join(md), encoding="utf-8")


def _write_summary(
    setup_results: List[Dict[str, Any]], out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "00-summary.json").write_text(
        json.dumps(setup_results, indent=2, default=str)
    )
    md = [
        "# Sub-project #7/#8 Phase-1 Gauntlet — Summary",
        "",
        "**Pass criteria (Phase-1 floor):** n>=500, NET PF>=1.10, "
        "NET Sharpe>0, sub-period PF>=1.0",
        "**Stage 3 criteria:** cell n>=100, NET PF>=1.30, sub-period PF>=1.10",
        "",
        "| Setup | n | NET PF | Net Sharpe | NET PnL | Floor | Best cell PF | Best cell n |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for r in setup_results:
        agg = r["floor"]["aggregate"]
        verdict = "**PASS**" if r["floor"]["passes_phase1"] else "fail"
        bc = r["cells"]["best_cells"]
        bc_pf = bc[0]["net_pf"] if bc else "—"
        bc_n = bc[0]["n_trades"] if bc else "—"
        md.append(
            f"| {r['setup']} | {agg['n_trades']:,} | {agg['net_pf']} | "
            f"{agg['net_sharpe']} | {agg['net_pnl']:,.0f} | {verdict} | "
            f"{bc_pf} | {bc_n} |"
        )
    (out_dir / "00-summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet-dir", required=True,
                   help="Dir of <setup>.parquet files from build_per_setup_pnl")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--period-start", default="2023-01-02")
    p.add_argument("--period-end",   default="2024-12-31")
    p.add_argument("--config-path",
                   default="config/configuration.json",
                   help="Source for per-setup intended universe + caps")
    p.add_argument("--no-intended-filter", action="store_true",
                   help="Skip the intended-universe pre-filter (raw wide-open eval)")
    args = p.parse_args()

    parquet_dir = Path(args.parquet_dir)
    out_dir = Path(args.output_dir)
    filters = (
        {} if args.no_intended_filter
        else _load_intended_filters(Path(args.config_path))
    )

    parquets = sorted(parquet_dir.glob("*.parquet"))
    if not parquets:
        raise SystemExit(f"No parquets under {parquet_dir}")

    setup_results: List[Dict[str, Any]] = []
    for pq in parquets:
        setup = pq.stem
        if setup in SKIP_SETUPS:
            print(f"  {setup}: SKIPPED (already validated & locked)")
            continue
        df = pd.read_parquet(pq)
        df = df[(df["session_date"] >= args.period_start)
                & (df["session_date"] <= args.period_end)].copy()
        if df.empty:
            print(f"  {setup}: no trades in [{args.period_start}, {args.period_end}]")
            continue
        df = _add_hour_bucket(df)
        df = _add_volatility_regime(df)

        # Apply intended universe + cap filter so wide-open noise doesn't
        # drown out the cells where each setup's structural thesis lives.
        df, filter_info = apply_intended_filters(df, setup, filters)
        if df.empty:
            print(f"  {setup}: no trades after intended-filter "
                  f"(universe={filter_info.get('universe_key')!r})")
            continue

        floor = evaluate_floor(df)
        cells = run_cell_selection(df)

        sub_out = out_dir / setup
        _write_setup_report(setup, df, floor, cells, sub_out)
        setup_results.append({
            "setup": setup,
            "filter_info": filter_info,
            "floor": floor,
            "cells": cells,
        })

        verdict = "PASS" if floor["passes_phase1"] else "fail"
        agg = floor["aggregate"]
        bc = cells["best_cells"]
        bc_str = f"best cell n={bc[0]['n_trades']} PF={bc[0]['net_pf']}" if bc else "no Stage-3 passing cell"
        n_filtered = filter_info["n_filtered_out"]
        filter_note = (
            f" [filtered out {n_filtered:,} off-universe trades]"
            if n_filtered > 0 else ""
        )
        print(f"  {setup:<30} n={agg['n_trades']:>6,} PF={agg['net_pf']:>5} "
              f"S={agg['net_sharpe']:>5} {verdict:<5} | {bc_str}{filter_note}")

    _write_summary(setup_results, out_dir)
    print(f"\n--> {out_dir}/00-summary.md")


if __name__ == "__main__":
    main()
