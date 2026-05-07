"""Verify block_deals_events.parquet — counts, F&O alignment, NSE/BSE overlap, spot-validation.

CLI:
    python tools/block_deal_calendar/verify_block_deals.py \\
        --parquet data/block_deals/block_deals_events.parquet \\
        --out specs/2026-05-07-sub-project-9-block-deal-backfill-validation.md

Outputs both stdout summary and a markdown report. Spot-validation embeds
NSE / BSE archive URLs the human reviewer can open to confirm each sample
event.

Sanity targets (from the brief):
- Per-quarter NSE counts roughly stable.
- Top-20 most-frequent block-traded names should be familiar liquid names
  (RELIANCE, HDFCBANK, ICICIBANK, INFY, etc.).
- ≥₹25 cr filter retains a meaningful number of events (brief locked
  threshold; raw parquet keeps everything ≥₹10 cr).
- NSE-vs-BSE overlap should be 60-90% (cross-validation working).
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PARQUET = _REPO_ROOT / "data" / "block_deals" / "block_deals_events.parquet"
_DEFAULT_OUT = (
    _REPO_ROOT / "specs" / "2026-05-07-sub-project-9-block-deal-backfill-validation.md"
)
_FNO_UNIVERSE_PATH = _REPO_ROOT / "assets" / "fno_liquid_200.csv"

_BRIEF_MIN_VALUE_CR = 25.0


def _build_nse_url() -> str:
    """NSE archive URL — reviewer narrows by date manually."""
    return "https://www.nseindia.com/report-detail/display-bulk-and-block-deals"


def _build_bse_url() -> str:
    return (
        "https://www.bseindia.com/markets/equity/EQReports/BulknBlockDeals?flag=2"
    )


def _load_fno(path: Path) -> set[str]:
    """Return set of NSE-prefixed F&O symbols (e.g. {'NSE:RELIANCE', ...})."""
    df = pd.read_csv(path)
    col = df.columns[0]
    syms = df[col].dropna().astype(str).str.strip().tolist()
    out: set[str] = set()
    for s in syms:
        s = s.upper()
        out.add(s if s.startswith("NSE:") else f"NSE:{s}")
    return out


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet", type=Path, default=_DEFAULT_PARQUET,
    )
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument(
        "--fno-universe", type=Path, default=_FNO_UNIVERSE_PATH,
    )
    parser.add_argument(
        "--seed", type=int, default=20260507,
        help="random seed for reproducible sampling",
    )
    parser.add_argument("--n-spot", type=int, default=5)
    parser.add_argument(
        "--min-value-cr", type=float, default=_BRIEF_MIN_VALUE_CR,
        help=(
            "brief filter threshold for ≥₹25cr report stage "
            "(parquet keeps ≥₹10cr raw)"
        ),
    )
    args = parser.parse_args(argv)

    if not args.parquet.exists():
        print(f"ERROR: {args.parquet} does not exist", file=sys.stderr)
        return 2
    df = pd.read_parquet(args.parquet)
    if df.empty:
        print(f"ERROR: {args.parquet} is empty", file=sys.stderr)
        return 3

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    fno = _load_fno(args.fno_universe)

    n_total = len(df)
    by_year_total = (
        df.groupby(df["trade_date"].dt.year).size().to_dict()
    )
    by_year_exchange = (
        df.groupby([df["trade_date"].dt.year, "exchange"]).size().to_dict()
    )
    by_quarter = (
        df.groupby([
            df["trade_date"].dt.year,
            df["trade_date"].dt.quarter,
            "exchange",
        ]).size().to_dict()
    )

    # ≥₹25 cr filter
    df_25 = df[df["trade_value_cr"] >= args.min_value_cr]
    n_25 = len(df_25)

    # F&O 200 universe filter (NSE-prefix matching)
    df_fno = df[df["symbol"].isin(fno)]
    n_fno = len(df_fno)

    # Combined ≥₹25cr + F&O 200
    df_strict = df_25[df_25["symbol"].isin(fno)]
    n_strict = len(df_strict)

    # Top-20 most-frequent NSE-prefix symbols (sanity)
    df_nse_only = df[df["exchange"] == "NSE"]
    top20 = df_nse_only["symbol"].value_counts().head(20)
    top20_strict = df_strict["symbol"].value_counts().head(20)

    # NSE / BSE overlap — at the (trade_date, raw_symbol) granularity.
    # Many BSE rows reference NSE-listed tickers via the Company column;
    # if BSE.raw_symbol matches NSE-side raw_symbol on the same date, that's
    # an overlap.
    nse_keys = set(
        zip(
            df.loc[df["exchange"] == "NSE", "trade_date"].dt.strftime("%Y-%m-%d"),
            df.loc[df["exchange"] == "NSE", "raw_symbol"].astype(str).str.upper(),
        )
    )
    bse_keys = set(
        zip(
            df.loc[df["exchange"] == "BSE", "trade_date"].dt.strftime("%Y-%m-%d"),
            df.loc[df["exchange"] == "BSE", "raw_symbol"].astype(str).str.upper(),
        )
    )
    overlap = nse_keys & bse_keys
    n_nse_keys = len(nse_keys)
    n_bse_keys = len(bse_keys)
    overlap_pct_nse = (
        100.0 * len(overlap) / n_nse_keys if n_nse_keys else 0.0
    )
    overlap_pct_bse = (
        100.0 * len(overlap) / n_bse_keys if n_bse_keys else 0.0
    )

    # Spot-validation: 5 random events across a mix of NSE + BSE
    rng_state = args.seed
    spot_pool = df[df["trade_value_cr"] >= 10.0]
    if len(spot_pool) >= args.n_spot:
        spot = spot_pool.sample(
            n=args.n_spot, random_state=rng_state,
        ).sort_values("trade_date").reset_index(drop=True)
    else:
        spot = spot_pool.copy().reset_index(drop=True)

    # Markdown report
    lines: list[str] = []
    lines.append("# Sub-project 9 round-5 — block-deal backfill validation\n\n")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
    lines.append(f"Source parquet: `{args.parquet.relative_to(_REPO_ROOT)}`\n\n")

    lines.append("## Coverage stats\n\n")
    lines.append(f"- Total raw events ingested: **{n_total}**\n")
    lines.append(
        f"- Date range: {df['trade_date'].min().date()} -> "
        f"{df['trade_date'].max().date()}\n"
    )
    lines.append(
        f"- NSE rows: {int((df['exchange']=='NSE').sum())} ; "
        f"BSE rows: {int((df['exchange']=='BSE').sum())}\n"
    )
    lines.append(f"- Unique symbols (any exchange): {df['symbol'].nunique()}\n")

    lines.append("\n### Events by calendar year (combined)\n\n")
    for y, c in sorted(by_year_total.items()):
        lines.append(f"- {y}: {c}\n")

    lines.append("\n### Events by calendar year + exchange\n\n")
    for (y, exch), c in sorted(by_year_exchange.items()):
        lines.append(f"- {y} {exch}: {c}\n")

    lines.append("\n### Events by quarter + exchange\n\n")
    lines.append("| year | quarter | exchange | count |\n|---|---|---|---|\n")
    for (y, q, exch), c in sorted(by_quarter.items()):
        lines.append(f"| {y} | Q{q} | {exch} | {c} |\n")

    lines.append("\n## Filter funnels\n\n")
    lines.append(f"- After ≥₹{args.min_value_cr:.1f} cr filter: **{n_25}** rows\n")
    lines.append(f"- After F&O 200 universe (NSE-prefix only): **{n_fno}** rows\n")
    lines.append(
        f"- After ≥₹{args.min_value_cr:.1f} cr + F&O 200 (strict): "
        f"**{n_strict}** rows\n"
    )

    # Per-side split for strict subset
    if n_strict > 0:
        side_counts = df_strict["buy_or_sell"].value_counts().to_dict()
        lines.append(f"- Strict subset side split: {side_counts}\n")

    lines.append("\n## Top-20 most-frequent block-traded names (NSE rows)\n\n")
    lines.append("| symbol | events |\n|---|---|\n")
    for sym, c in top20.items():
        lines.append(f"| {sym} | {c} |\n")

    lines.append(
        "\n## Top-20 strict subset (F&O 200 ∩ ≥₹25cr)\n\n"
    )
    if len(top20_strict):
        lines.append("| symbol | events |\n|---|---|\n")
        for sym, c in top20_strict.items():
            lines.append(f"| {sym} | {c} |\n")
    else:
        lines.append("(none)\n")

    lines.append("\n## NSE vs BSE reconciliation\n\n")
    lines.append(f"- NSE (date,symbol) keys: **{n_nse_keys}**\n")
    lines.append(f"- BSE (date,symbol) keys: **{n_bse_keys}**\n")
    lines.append(f"- Overlap (intersection): **{len(overlap)}**\n")
    lines.append(
        f"- Overlap as % of NSE keys: **{overlap_pct_nse:.1f}%**\n"
    )
    lines.append(
        f"- Overlap as % of BSE keys: **{overlap_pct_bse:.1f}%**\n\n"
    )
    lines.append(
        "Brief target: 60-90% overlap = healthy cross-validation. "
        "<40% overlap = one source is missing data.\n"
    )

    # Spot-validation
    lines.append(f"\n## Spot-validation ({len(spot)} events)\n\n")
    lines.append(
        "Open the archive URL, narrow to the trade_date, and confirm the "
        "symbol + qty + price match the row below.\n\n"
    )
    for i, row in spot.iterrows():
        url = _build_nse_url() if row["exchange"] == "NSE" else _build_bse_url()
        td = row["trade_date"]
        td_str = td.date() if hasattr(td, "date") else td
        lines.append(
            f"### {i + 1}. {row['symbol']} ({row['exchange']}) — {td_str}\n\n"
            f"- Client: `{row['client_name']}`\n"
            f"- Side: `{row['buy_or_sell']}`\n"
            f"- Qty: `{row['qty']:,}`\n"
            f"- Price: `₹{row['trade_price']:.2f}`\n"
            f"- Value: `₹{row['trade_value_cr']:.2f} cr`\n"
            f"- Archive URL: {url}\n\n"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(lines), encoding="utf-8")
    print(f"OK: validation report written to {args.out}")
    print(
        f"total={n_total} ge25cr={n_25} fno={n_fno} strict={n_strict} "
        f"overlap={len(overlap)} ({overlap_pct_nse:.1f}% nse / "
        f"{overlap_pct_bse:.1f}% bse)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
