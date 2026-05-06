"""Cell selection on round-4 sanity trades (IV-rank revert + earnings).

Round-4 aggregate sanity verdict:
  - options_vol_iv_rank_revert: PF 0.843 (LONG drag); SHORT-only PF 1.104, n=503
  - sector_rotation_relative_strength: PF 0.72 (skip — n=59 is too thin for cells)
  - earnings_day_intraday_fade: PF 1.245, n=170 (skip — n too thin for cells)

Per the gauntlet pattern (Stage-3 cell selection), individual cells may
pass even when aggregate fails. This script applies the same per-cell
PF/WR check to IV-rank trades, stratified by:
  - side (LONG / SHORT)
  - cap_segment (large_cap is dominant per F&O 200 reality)
  - iv_rank_bucket (the key new dim for this candidate):
      SHORT: 0.80-0.85, 0.85-0.90, 0.90-0.95, 0.95-1.00
      LONG:  0.00-0.05, 0.05-0.10, 0.10-0.15, 0.15-0.20
  - regime (NIFTY trend at entry — re-tagged from NIFTY 50 EMA20 5m)

Pass criteria (locked, same as circuit_t1 / round-3 gauntlet):
  - n >= 30 per cell (n_floor for trustworthy point estimate)
  - NET PF >= 1.10
  - WR delta from aggregate < 10pp (warning, not hard fail)

Output: reports/sub9_sanity/round4_iv_rank_cell_selection.md
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
SANITY_DIR = _REPO_ROOT / "reports" / "sub9_sanity"
NIFTY_1M_PATH = (
    _REPO_ROOT / "backtest-cache-download" / "index_ohlcv"
    / "NSE_NIFTY_50" / "NSE_NIFTY_50_1minutes.feather"
)

PASS_PF = 1.10
PASS_N = 30
WARN_WR_DELTA_PP = 10.0


def load_nifty_5m() -> pd.DataFrame:
    df = pd.read_feather(NIFTY_1M_PATH)
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df = df[(df["date"] >= pd.Timestamp("2023-01-01"))
            & (df["date"] < pd.Timestamp("2025-01-01"))].set_index("date")
    agg = df.resample("5min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    agg["ema20"] = agg["close"].ewm(span=20, adjust=False).mean()
    return agg.reset_index().rename(columns={"date": "ts"})[["ts", "close", "ema20"]]


def tag_regime(trades: pd.DataFrame, nifty: pd.DataFrame) -> pd.DataFrame:
    trades = trades.copy()
    trades["entry_ts"] = pd.to_datetime(trades["entry_ts"])
    trades["entry_ts_5m"] = trades["entry_ts"].dt.floor("5min")
    nifty_idx = nifty.set_index("ts")[["close", "ema20"]]
    j = trades.set_index("entry_ts_5m").join(nifty_idx, how="left").reset_index()
    j["regime"] = j.apply(
        lambda r: "trend_up" if (pd.notna(r["close"]) and pd.notna(r["ema20"])
                                  and r["close"] > r["ema20"])
        else ("trend_down" if (pd.notna(r["close"]) and pd.notna(r["ema20"]))
              else "unknown"),
        axis=1,
    )
    return j


def iv_bucket(side: str, ivr: float) -> str:
    if side == "SHORT":
        if ivr >= 0.95: return "0.95-1.00"
        if ivr >= 0.90: return "0.90-0.95"
        if ivr >= 0.85: return "0.85-0.90"
        return "0.80-0.85"
    else:
        if ivr <= 0.05: return "0.00-0.05"
        if ivr <= 0.10: return "0.05-0.10"
        if ivr <= 0.15: return "0.10-0.15"
        return "0.15-0.20"


def cell_summary(df: pd.DataFrame, *cols, agg_wr=None) -> pd.DataFrame:
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
        rows.append({
            "cell": " × ".join(f"{c}={v}" for c, v in zip(cols, key)),
            "n": len(g),
            "wr_pct": wr,
            "wr_delta_pp": (wr - agg_wr) if agg_wr is not None else None,
            "gross_pnl": float(pnl.sum()),
            "net_pf": pf,
            "passed": bool(len(g) >= PASS_N and pf >= PASS_PF),
        })
    return pd.DataFrame(rows).sort_values("net_pf", ascending=False).reset_index(drop=True)


def report_setup(name: str, trades: pd.DataFrame, nifty: pd.DataFrame) -> List[str]:
    out = [f"\n## {name}\n"]
    if trades.empty:
        out.append("_no trades_")
        return out

    trades = tag_regime(trades, nifty)
    trades["iv_rank_bucket"] = trades.apply(
        lambda r: iv_bucket(r["side"], float(r["iv_rank"])), axis=1)

    pnl = trades["net_pnl"].astype(float)
    wins = (pnl > 0).sum()
    gw = pnl[pnl > 0].sum()
    gl = abs(pnl[pnl < 0].sum())
    agg_pf = gw / max(gl, 1e-9)
    agg_wr = 100.0 * wins / len(trades)
    out.append(f"**Aggregate: n={len(trades)}, NET PF={agg_pf:.3f}, WR={agg_wr:.1f}%**\n")

    out.append("\n### Univariate cells\n")
    for dim in ("side", "cap_segment", "regime", "iv_rank_bucket"):
        out.append(f"\n**{dim}:**\n")
        out.append("| cell | n | NET PF | WR | WR Δ pp | pass |")
        out.append("|---|---|---|---|---|---|")
        for _, r in cell_summary(trades, dim, agg_wr=agg_wr).iterrows():
            mark = "✓" if r["passed"] else "✗"
            wd = f"{r['wr_delta_pp']:+.1f}" if r["wr_delta_pp"] is not None else "—"
            out.append(
                f"| {r['cell']} | {r['n']} | {r['net_pf']:.3f} | "
                f"{r['wr_pct']:.1f}% | {wd} | {mark} |"
            )

    out.append("\n### Bivariate cells (top 20 by PF, n>=30)\n")
    out.append("| cell | n | NET PF | WR | WR Δ pp | pass |")
    out.append("|---|---|---|---|---|---|")
    bivariate = []
    for d1, d2 in [("side", "cap_segment"), ("side", "regime"),
                   ("side", "iv_rank_bucket"), ("regime", "iv_rank_bucket"),
                   ("cap_segment", "iv_rank_bucket"), ("regime", "cap_segment")]:
        s = cell_summary(trades, d1, d2, agg_wr=agg_wr)
        s = s[s["n"] >= PASS_N]
        bivariate.append(s)
    if bivariate:
        all_b = pd.concat(bivariate, ignore_index=True).sort_values(
            "net_pf", ascending=False).head(20)
        for _, r in all_b.iterrows():
            mark = "✓" if r["passed"] else "✗"
            wd = f"{r['wr_delta_pp']:+.1f}" if r["wr_delta_pp"] is not None else "—"
            out.append(
                f"| {r['cell']} | {r['n']} | {r['net_pf']:.3f} | "
                f"{r['wr_pct']:.1f}% | {wd} | {mark} |"
            )

    out.append("\n### Trivariate (side × iv_rank_bucket × regime, n>=30)\n")
    out.append("| cell | n | NET PF | WR | WR Δ pp | pass |")
    out.append("|---|---|---|---|---|---|")
    s3 = cell_summary(trades, "side", "iv_rank_bucket", "regime", agg_wr=agg_wr)
    s3 = s3[s3["n"] >= PASS_N]
    for _, r in s3.head(20).iterrows():
        mark = "✓" if r["passed"] else "✗"
        wd = f"{r['wr_delta_pp']:+.1f}" if r["wr_delta_pp"] is not None else "—"
        out.append(
            f"| {r['cell']} | {r['n']} | {r['net_pf']:.3f} | "
            f"{r['wr_pct']:.1f}% | {wd} | {mark} |"
        )
    return out


def main():
    nifty = load_nifty_5m()

    sections = [
        "# Round-4 Cell Selection: options_vol_iv_rank_revert",
        "",
        "**Date:** 2026-05-06",
        "**Pass criteria:** n>=30 AND NET PF>=1.10 (matches circuit_t1 / round-3 cell selection)",
        "",
        "Round-4 IV-rank-revert aggregate failed (PF 0.843, n=5224, LONG dragged "
        "everything down). This report checks whether any individual cell — "
        "particularly tighter iv_rank buckets like 0.95-1.00 (extremely high IV) "
        "or 0.00-0.05 (extremely low IV) — has PF>=1.10 + n>=30, in which case "
        "the candidate could ship at narrow scope.",
    ]

    csv_path = SANITY_DIR / "options_vol_iv_rank_revert_trades.csv"
    if not csv_path.exists():
        sections.append(f"\n_no trades csv at {csv_path}_")
    else:
        trades = pd.read_csv(csv_path)
        sections.extend(report_setup("options_vol_iv_rank_revert", trades, nifty))

    out_path = SANITY_DIR / "round4_iv_rank_cell_selection.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"Wrote: {out_path}")
    print()

    # console quick summary
    if csv_path.exists():
        trades = tag_regime(pd.read_csv(csv_path), nifty)
        trades["iv_rank_bucket"] = trades.apply(
            lambda r: iv_bucket(r["side"], float(r["iv_rank"])), axis=1)
        pass_cells = []
        for dims in [("side",), ("cap_segment",), ("regime",), ("iv_rank_bucket",),
                     ("side", "cap_segment"), ("side", "regime"),
                     ("side", "iv_rank_bucket"), ("regime", "iv_rank_bucket"),
                     ("cap_segment", "iv_rank_bucket"),
                     ("side", "iv_rank_bucket", "regime")]:
            s = cell_summary(trades, *dims)
            s = s[(s["n"] >= PASS_N) & (s["net_pf"] >= PASS_PF)]
            for _, r in s.iterrows():
                pass_cells.append(
                    f"  PASS  {r['cell']:60} n={r['n']:>5} "
                    f"PF={r['net_pf']:.3f} WR={r['wr_pct']:.1f}%"
                )
        print(f"=== options_vol_iv_rank_revert: {len(pass_cells)} passing cells ===")
        for line in pass_cells[:30]:
            print(line)
        if len(pass_cells) > 30:
            print(f"  ... and {len(pass_cells)-30} more")


if __name__ == "__main__":
    main()
