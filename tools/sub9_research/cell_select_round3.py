"""Cell selection on round-3 sanity trades.

Aggregate PF for all 3 round-3 candidates was < 0.65 on 2yr data.
But the gauntlet's whole point is that individual cells (regime ×
cap × hour_bucket × side) might pass even when aggregate fails.

Reads the 3 sanity trades CSVs at reports/sub9_sanity/, joins NIFTY
5m for regime tagging, computes per-cell PF/WR/n across the same
dimensions used by circuit_t1's Stage-3 cell selection.

Pass criteria (locked, same as circuit_t1 gauntlet):
  - n >= 30 per cell
  - NET PF >= 1.10
  - WR delta from aggregate < 10pp (stability — flagged as warning,
    not a hard fail since gauntlet's Stage-3 doesn't enforce it)

Output: reports/sub9_sanity/round3_cell_selection.md
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import List

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
SANITY_DIR = _REPO_ROOT / "reports" / "sub9_sanity"
NIFTY_5M_PATH = (
    _REPO_ROOT / "backtest-cache-download" / "index_ohlcv"
    / "NSE_NIFTY_50" / "NSE_NIFTY_50_1minutes.feather"
)

PASS_PF = 1.10
PASS_N = 30
WARN_WR_DELTA_PP = 10.0


def load_nifty_5m() -> pd.DataFrame:
    """Load NIFTY 50 1m, resample to 5m, compute EMA20 for regime tag."""
    print("loading NIFTY 50 1m + resampling 5m + EMA20 ...")
    df = pd.read_feather(NIFTY_5M_PATH)
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df = df[(df["date"] >= pd.Timestamp("2023-01-01"))
            & (df["date"] < pd.Timestamp("2025-01-01"))].set_index("date")
    agg = df.resample("5min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    agg["ema20"] = agg["close"].ewm(span=20, adjust=False).mean()
    agg = agg.reset_index().rename(columns={"date": "ts"})
    return agg[["ts", "close", "ema20"]]


def hour_bucket(ts: pd.Timestamp) -> str:
    """5 hour buckets matching the existing gauntlet's data_loader."""
    h = ts.hour
    m = ts.minute
    mod = h * 60 + m
    if mod < 600:    return "opening"     # < 10:00
    if mod < 720:    return "morning"     # 10:00-12:00
    if mod < 780:    return "lunch"       # 12:00-13:00
    if mod < 870:    return "afternoon"   # 13:00-14:30
    return "late"                         # 14:30+


def tag_regime_per_trade(trades: pd.DataFrame, nifty: pd.DataFrame) -> pd.DataFrame:
    """Tag each trade with NIFTY regime at entry time.

    Regime = "trend_up" if NIFTY close > EMA20 at entry bar; else "trend_down".
    Aligns with circuit_t1's gauntlet definition.
    """
    trades = trades.copy()
    trades["entry_ts"] = pd.to_datetime(trades["entry_ts"])
    nifty = nifty.copy()
    nifty["ts"] = pd.to_datetime(nifty["ts"])
    # 5m floor join
    trades["entry_ts_5m"] = trades["entry_ts"].dt.floor("5min")
    nifty_idx = nifty.set_index("ts")[["close", "ema20"]]
    j = trades.set_index("entry_ts_5m").join(nifty_idx, how="left").reset_index()
    j["regime"] = j.apply(
        lambda r: "trend_up" if r["close"] > r["ema20"] else "trend_down"
        if pd.notna(r["close"]) and pd.notna(r["ema20"]) else "unknown",
        axis=1,
    )
    return j


def cell_summary(df: pd.DataFrame, *cols, agg_pf=None, agg_wr=None) -> pd.DataFrame:
    """Per-cell n, gross PnL, fees, NET PnL, NET PF, WR. cols = grouping keys."""
    rows = []
    grouped = df.groupby(list(cols)) if cols else [(("__all__",), df)]
    for key, g in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        pnl = g["net_pnl"].astype(float)
        if len(pnl) == 0:
            continue
        wins = (pnl > 0).sum()
        gw = pnl[pnl > 0].sum()
        gl = abs(pnl[pnl < 0].sum())
        pf = gw / max(gl, 1e-9)
        wr = 100.0 * wins / len(pnl)
        wr_delta = (wr - agg_wr) if agg_wr is not None else None
        rows.append({
            "cell": " × ".join(f"{c}={v}" for c, v in zip(cols, key)),
            "n": len(g),
            "wr_pct": wr,
            "wr_delta_pp": wr_delta,
            "gross_pnl": float(pnl.sum()),
            "net_pf": pf,
            "passed": bool(len(g) >= PASS_N and pf >= PASS_PF),
        })
    return pd.DataFrame(rows).sort_values("net_pf", ascending=False).reset_index(drop=True)


def report_setup(name: str, trades: pd.DataFrame, nifty: pd.DataFrame) -> List[str]:
    """Return markdown lines for one setup."""
    out = [f"\n## {name}\n"]
    if trades.empty:
        out.append("_no trades_")
        return out

    trades = tag_regime_per_trade(trades, nifty)
    trades["hour_bucket"] = trades["entry_ts"].apply(hour_bucket)

    # Aggregate
    pnl = trades["net_pnl"].astype(float)
    wins = (pnl > 0).sum()
    gw = pnl[pnl > 0].sum()
    gl = abs(pnl[pnl < 0].sum())
    agg_pf = gw / max(gl, 1e-9)
    agg_wr = 100.0 * wins / len(trades)
    out.append(f"**Aggregate: n={len(trades)}, NET PF={agg_pf:.3f}, WR={agg_wr:.1f}%**\n")
    out.append("\n### Univariate cells\n")
    for dim in ("side", "cap_segment", "regime", "hour_bucket"):
        out.append(f"\n**{dim}:**\n")
        out.append("| cell | n | NET PF | WR | WR Δ pp | pass |")
        out.append("|---|---|---|---|---|---|")
        for _, r in cell_summary(trades, dim, agg_wr=agg_wr).iterrows():
            mark = "✓" if r["passed"] else "✗"
            wd = f"{r['wr_delta_pp']:+.1f}" if r["wr_delta_pp"] is not None else "—"
            out.append(
                f"| {r['cell']} | {r['n']} | {r['net_pf']:.3f} | {r['wr_pct']:.1f}% | {wd} | {mark} |"
            )

    out.append("\n### Bivariate cells (top 15 by PF, n>=30)\n")
    out.append("| cell | n | NET PF | WR | WR Δ pp | pass |")
    out.append("|---|---|---|---|---|---|")
    bivariate = []
    for d1, d2 in [("side", "cap_segment"), ("side", "regime"),
                   ("side", "hour_bucket"), ("regime", "cap_segment"),
                   ("regime", "hour_bucket"), ("cap_segment", "hour_bucket")]:
        s = cell_summary(trades, d1, d2, agg_wr=agg_wr)
        s = s[s["n"] >= PASS_N]
        bivariate.append(s)
    if bivariate:
        all_b = pd.concat(bivariate, ignore_index=True).sort_values("net_pf", ascending=False).head(15)
        for _, r in all_b.iterrows():
            mark = "✓" if r["passed"] else "✗"
            wd = f"{r['wr_delta_pp']:+.1f}" if r["wr_delta_pp"] is not None else "—"
            out.append(
                f"| {r['cell']} | {r['n']} | {r['net_pf']:.3f} | {r['wr_pct']:.1f}% | {wd} | {mark} |"
            )

    # 3-way: regime × side × cap
    out.append("\n### Trivariate (regime × side × cap, n>=30)\n")
    out.append("| cell | n | NET PF | WR | WR Δ pp | pass |")
    out.append("|---|---|---|---|---|---|")
    s3 = cell_summary(trades, "regime", "side", "cap_segment", agg_wr=agg_wr)
    s3 = s3[s3["n"] >= PASS_N]
    for _, r in s3.head(15).iterrows():
        mark = "✓" if r["passed"] else "✗"
        wd = f"{r['wr_delta_pp']:+.1f}" if r["wr_delta_pp"] is not None else "—"
        out.append(
            f"| {r['cell']} | {r['n']} | {r['net_pf']:.3f} | {r['wr_pct']:.1f}% | {wd} | {mark} |"
        )
    return out


def main():
    nifty = load_nifty_5m()

    sections = [
        "# Round-3 Cell Selection on 2yr Sanity Trades",
        "",
        "**Date:** 2026-05-06",
        f"**Pass criteria:** n>=30 AND NET PF>=1.10 (matches circuit_t1 cell selection)",
        "",
        "All 3 candidates failed aggregate sanity (PF 0.47-0.63). This report",
        "checks whether any individual cell (regime × cap × hour × side) has",
        "PF>=1.10 + n>=30, in which case the candidate could ship at narrow scope.",
    ]
    for setup_name, csv_name in [
        ("vwap_deviation_meanrevert", "vwap_deviation_meanrevert_trades.csv"),
        ("index_stock_divergence_revert", "index_stock_divergence_revert_trades.csv"),
        ("volume_spike_exhaustion_reversal", "volume_spike_exhaustion_reversal_trades.csv"),
    ]:
        csv_path = SANITY_DIR / csv_name
        if not csv_path.exists():
            sections.append(f"\n## {setup_name}\n_no trades csv at {csv_path}_")
            continue
        trades = pd.read_csv(csv_path)
        sections.extend(report_setup(setup_name, trades, nifty))

    out_path = SANITY_DIR / "round3_cell_selection.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"Wrote: {out_path}")
    print()
    # Print quick console summary of any passing cells
    for setup_name, csv_name in [
        ("vwap_deviation_meanrevert", "vwap_deviation_meanrevert_trades.csv"),
        ("index_stock_divergence_revert", "index_stock_divergence_revert_trades.csv"),
        ("volume_spike_exhaustion_reversal", "volume_spike_exhaustion_reversal_trades.csv"),
    ]:
        csv_path = SANITY_DIR / csv_name
        if not csv_path.exists():
            continue
        trades = tag_regime_per_trade(pd.read_csv(csv_path), nifty)
        trades["hour_bucket"] = pd.to_datetime(trades["entry_ts"]).apply(hour_bucket)
        pass_cells = []
        for dims in [("side",), ("cap_segment",), ("regime",), ("hour_bucket",),
                     ("side", "cap_segment"), ("side", "regime"),
                     ("side", "hour_bucket"), ("regime", "cap_segment"),
                     ("regime", "hour_bucket"), ("cap_segment", "hour_bucket"),
                     ("regime", "side", "cap_segment")]:
            s = cell_summary(trades, *dims)
            s = s[(s["n"] >= PASS_N) & (s["net_pf"] >= PASS_PF)]
            for _, r in s.iterrows():
                pass_cells.append(f"  PASS  {r['cell']:60} n={r['n']:>5} PF={r['net_pf']:.3f} WR={r['wr_pct']:.1f}%")
        print(f"=== {setup_name}: {len(pass_cells)} passing cells ===")
        for line in pass_cells[:20]:
            print(line)
        if len(pass_cells) > 20:
            print(f"  ... and {len(pass_cells)-20} more")
        print()


if __name__ == "__main__":
    main()
