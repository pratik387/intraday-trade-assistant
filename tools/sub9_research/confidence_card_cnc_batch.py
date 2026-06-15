"""Stage-6 CNC confidence cards for the 4-setup MTF paper batch, using the
CNC-VALID framework (specs/2026-06-15-cnc-confidence-card-methodology-research.md):
stationary block bootstrap of the daily MTM portfolio-return series + Lo(2002)
autocorrelation-corrected Sharpe + Deflated Sharpe Ratio haircut.

This SUPERSEDES the earlier per-setup cards (`confidence_card_low52/zscore/...`,
`_tmp_cnc_a2_confidence_card`) which used the intraday iid per-trade BCa bootstrap +
Harvey-Liu — INVALID for overlapping multi-day cross-sectional holds (CIs too tight,
Sharpe overstated).

Setups (locked cells, Discovery 2023-24; OOS 2025 + HO 2026 one-shot):
  A2 mtf_capitulation_revert_long  : trailing-5d bottom-5%  x tier1 x shock>=2 x K=2  (M=540)
  C1 low52_capitulation_revert_long: close<=5% of 252d-low  x tier1 x shock>=2 x K=2  (M=58)
  C4 zscore_oversold_revert_long   : close<=-1.5sd(20d)      x tier1 x shock>=1.5 x K=2 (M=66)
  C6 crash2d_revert_long           : trailing-2d bottom-10%  x tier1 x shock>=2 x K=3  (M=72)

Basis note: the DSR selection test (SR_hat vs SR0, V[trial SR]) is computed on the
signal-day-aggregated daily return series, CONSISTENTLY across the locked cell and
every swept trial. The headline PF/expectancy CIs and Sharpe use the full MTM daily
portfolio-return series (block bootstrap + Lo HAC). Both are reported.
"""
from __future__ import annotations
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.methodology.confidence.confidence_card_cnc import compute_cnc_card
from tools.methodology.confidence.cnc_daily_returns import simulate_slot_admission

_DAILY = ROOT / "cache" / "preaggregate" / "clean_daily_from5m.feather"
_MTF = ROOT / "data" / "mtf_universe" / "approved_mtf_securities_2026-05-21.json"
_OUT = ROOT / "reports" / "confidence_cards_cnc"

BASE_COST = 0.00547           # STT(0.20) + brokerage(0.06) + charges(0.047) + slippage(0.20), round-trip %
MTF_INT_PER_DAY = 0.0004      # MTF interest on funded days
ADV_FLOOR = 2e6
SHIP_MIN_N = 200

N_GRID = [1, 2, 3, 5, 10]
K_GRID = [1, 2, 3]
TIER_GRID = [1, 2, 3]
SHOCK_GRID = [1.0, 1.5, 2.0, 3.0]
DECILE_GRID = [0.05, 0.10, 0.20]
LOW52_GRID = [0.01, 0.03, 0.05, 0.10]
ZSCORE_GRID = [-1.5, -2.0, -2.5]


def load_panel():
    df = pd.read_feather(_DAILY).rename(columns={"ts": "date"})
    df["date"] = pd.to_datetime(df["date"])
    if getattr(df["date"].dt, "tz", None) is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df["symbol"] = df["symbol"].str.replace("NSE:", "", regex=False).str.upper()
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def load_mtf_eligible():
    import json
    return {str(r["tradingsymbol"]).strip().upper()
            for r in json.load(open(_MTF, encoding="utf-8"))
            if str(r.get("category", "")).lower() != "etf" and r.get("tradingsymbol")}


def prep():
    """Full contiguous MTF-eligible panel + per-row signal features.

    Returns (panel, base):
      panel = full per-symbol contiguous OHLC (for MTM return computation, NO gaps)
      base  = ADV-floor-filtered rows with tier + features (for signal SELECTION)
    """
    df = load_panel()
    elig = load_mtf_eligible()
    df = df[df["symbol"].isin(elig)].sort_values(["symbol", "date"]).reset_index(drop=True)
    g = df.groupby("symbol", sort=False)
    for n in set(N_GRID):
        df[f"ret{n}"] = g["close"].transform(lambda s, n=n: s / s.shift(n) - 1)
    df["turnover"] = df["close"] * df["volume"]
    df["adv20"] = g["turnover"].transform(lambda s: s.rolling(20).mean())
    df["adv20_prior"] = g["turnover"].transform(lambda s: s.shift(1).rolling(20).mean())
    df["tshock"] = df["turnover"] / df["adv20_prior"]
    df["low252"] = g["low"].transform(lambda s: s.rolling(252, min_periods=60).min())
    df["dist_low"] = df["close"] / df["low252"] - 1.0
    df["mean20"] = g["close"].transform(lambda s: s.rolling(20).mean())
    df["std20"] = g["close"].transform(lambda s: s.rolling(20).std())
    df["zscore"] = (df["close"] - df["mean20"]) / df["std20"]
    df["open_next"] = g["open"].shift(-1)
    for k in set(K_GRID):
        df[f"fwd{k}"] = g["close"].shift(-(1 + k)) / df["open_next"] - 1
    df["yr"] = df["date"].dt.year

    panel = df[["symbol", "date", "open", "close"]].copy()
    base = df[(df.adv20 >= ADV_FLOOR) & (df.close >= 5) & (df.tshock.notna())].copy()
    base["tier"] = base.groupby("date")["adv20"].transform(
        lambda s: pd.qcut(s, 5, labels=[1, 2, 3, 4, 5], duplicates="drop"))
    return panel, base


def _signal_day_sharpe(sel: pd.DataFrame, fwd_col: str, cost: float):
    """Per-period Sharpe of the signal-day-aggregated net-return series (the
    equal-weight basket return per signal-day). Returns (sharpe, n_signal_days)."""
    if len(sel) == 0:
        return np.nan, 0
    net = sel[fwd_col] - cost
    daily = net.groupby(sel["date"]).mean()
    if len(daily) < 2 or daily.std(ddof=1) == 0:
        return np.nan, len(daily)
    return float(daily.mean() / daily.std(ddof=1)), len(daily)


# Per-setup selection: returns the rows of `m` matching (cell params).
def sel_decile(m, lookback, dec, tier, shock):
    rk = m.groupby("date")[f"ret{lookback}"].rank(pct=True)
    return m[(rk <= dec) & (m["tier"] == tier) & (m["tshock"] >= shock)]


def sel_low52(m, v, tier, shock):
    return m[(m["dist_low"] <= v) & (m["tier"] == tier) & (m["tshock"] >= shock)]


def sel_zscore(m, v, tier, shock):
    return m[(m["zscore"] <= v) & (m["tier"] == tier) & (m["tshock"] >= shock)]


# k_grid per setup MUST match each setup's ORIGINAL cell-mine sweep (else M is
# mis-counted): A2 swept K in {1,2,3}; the 2-3day batch (C1/C4/C6) swept K in {2,3}.
LIVE_SLOTS = 8  # _live_max_concurrent_slots per setup (config)

SETUPS = [
    dict(name="mtf_capitulation_revert_long", kind="decile", k_grid=[1, 2, 3],
         grid=[("N", N_GRID), ("dec", DECILE_GRID)],
         locked=dict(N=5, dec=0.05, K=2, tier=1, shock=2.0), M_doc=540),
    dict(name="low52_capitulation_revert_long", kind="low52", k_grid=[2, 3],
         grid=[("v", LOW52_GRID)],
         locked=dict(v=0.05, K=2, tier=1, shock=2.0), M_doc=58),
    dict(name="zscore_oversold_revert_long", kind="zscore", k_grid=[2, 3],
         grid=[("v", ZSCORE_GRID)],
         locked=dict(v=-1.5, K=2, tier=1, shock=1.5), M_doc=66),
    dict(name="crash2d_revert_long", kind="decile", k_grid=[2, 3],
         grid=[("dec", DECILE_GRID)],
         locked=dict(N=2, dec=0.10, K=3, tier=1, shock=2.0), M_doc=72),
]


def _score_series(setup, sel):
    """Priority score for slot admission (LOWER = stronger signal). Decile: the
    cross-sectional return rank; low52: distance to the 252d low; zscore: the
    z-score itself."""
    if setup["kind"] == "decile":
        N = setup["locked"]["N"]
        return sel.groupby("date")[f"ret{N}"].rank(pct=True)
    if setup["kind"] == "low52":
        return sel["dist_low"]
    return sel["zscore"]


def _iter_cells(setup):
    """Yield dicts of cell params for the full grid (setup-specific dims x K x tier x shock)."""
    dim_names = [d[0] for d in setup["grid"]]
    dim_vals = [d[1] for d in setup["grid"]]
    for combo in itertools.product(*dim_vals, setup["k_grid"], TIER_GRID, SHOCK_GRID):
        cell = dict(zip(dim_names, combo[:len(dim_names)]))
        cell["K"], cell["tier"], cell["shock"] = combo[-3], combo[-2], combo[-1]
        if setup["kind"] == "decile" and "N" not in cell:
            cell["N"] = setup["locked"]["N"]   # C6: lookback fixed at 2
        yield cell


def _select(setup, base, cell):
    K = cell["K"]
    m = base[base[f"fwd{K}"].notna()].copy()
    if setup["kind"] == "decile":
        return sel_decile(m, cell["N"], cell["dec"], cell["tier"], cell["shock"]), f"fwd{K}"
    if setup["kind"] == "low52":
        return sel_low52(m, cell["v"], cell["tier"], cell["shock"]), f"fwd{K}"
    return sel_zscore(m, cell["v"], cell["tier"], cell["shock"]), f"fwd{K}"


def run_setup(setup, panel, base):
    locked = setup["locked"]
    K = locked["K"]
    cost = BASE_COST + MTF_INT_PER_DAY * K

    # --- sweep: trial Sharpes (signal-day basis) + ship-eligible daily series ---
    trial_sharpes, ship_series = [], {}
    m_total = 0
    for cell in _iter_cells(setup):
        sel, fwd_col = _select(setup, base, cell)
        disc = sel[sel.yr.isin([2023, 2024])]
        if len(disc) < SHIP_MIN_N:
            continue
        m_total += 1
        sr, _ = _signal_day_sharpe(disc, fwd_col, cost)
        if not np.isnan(sr):
            trial_sharpes.append(sr)
        # ship-eligible (net daily-mean PF on Discovery >= 1.20) -> keep series for rho_bar
        net = disc[fwd_col] - cost
        daily = net.groupby(disc["date"]).mean()
        w = daily[daily > 0].sum(); l = -daily[daily < 0].sum()
        pf = (w / l) if l > 0 else np.nan
        if not np.isnan(pf) and pf >= 1.20:
            ship_series[tuple(sorted(cell.items()))] = daily

    # --- rho_bar: mean pairwise corr of ship-eligible cells' signal-day series ---
    rho_bar = 0.0
    if len(ship_series) >= 2:
        aligned = pd.DataFrame(ship_series)
        corr = aligned.corr().values
        iu = np.triu_indices_from(corr, k=1)
        vals = corr[iu]
        vals = vals[np.isfinite(vals)]
        rho_bar = float(np.clip(np.mean(vals), 0.0, 1.0)) if len(vals) else 0.0

    # --- locked cell: ledgers (+ priority score) per period + DSR-basis Sharpe ---
    sel, fwd_col = _select(setup, base, dict(locked))
    sel = sel.copy()
    sel["score"] = _score_series(setup, sel)
    ledgers = {}
    for label, yrs in [("disc", [2023, 2024]), ("oos", [2025]), ("ho", [2026])]:
        sub = sel[sel.yr.isin(yrs)]
        ledgers[label] = (sub[["date", "symbol", "score"]]
                          .rename(columns={"date": "signal_date"}).reset_index(drop=True))
    sr_hat_dsr, n_obs_dsr = _signal_day_sharpe(sel[sel.yr.isin([2023, 2024])], fwd_col, cost)

    def _card(trades, aggregate):
        return compute_cnc_card(
            setup_name=setup["name"],
            trades_disc=trades["disc"], trades_oos=trades["oos"], trades_ho=trades["ho"],
            panel=panel, k_hold=K, cost=cost, aggregate=aggregate,
            trial_sharpes=trial_sharpes, m_trials=m_total, rho_bar=rho_bar,
            sr_hat_dsr=sr_hat_dsr, n_obs_dsr=n_obs_dsr,
        )

    # capped variant: ration to LIVE_SLOTS concurrent (top-ranked), fixed-Rs/slot sum
    capped = {lbl: simulate_slot_admission(ledgers[lbl], panel, k_hold=K,
                                           max_slots=LIVE_SLOTS, score_col="score")
              for lbl in ("disc", "oos", "ho")}

    cards = {
        "mean (equal-weight book)": _card(ledgers, "mean"),
        "sum (uncapped fixed-Rs/slot)": _card(ledgers, "sum"),
        f"capped-{LIVE_SLOTS} (live slot book)": _card(capped, "sum"),
    }
    meta = dict(m_total=m_total, n_ship=len(ship_series), cost=cost, K=K,
                n_disc=len(ledgers["disc"]), n_disc_capped=len(capped["disc"]))
    return cards, meta


def _row(label, card):
    pf = card.pf_ci
    t = (card.sharpe_per_period / card.sharpe_se) if card.sharpe_se else float("nan")
    return (f"| {label:<30} | {pf.point_estimate:5.3f} [{pf.ci_lower:5.3f}, {pf.ci_upper:5.3f}] "
            f"| {card.sharpe_per_period:+.4f} (t {t:+.2f}) | {card.oos_pf:5.3f} | {card.ho_pf:5.3f} |")


def render_sensitivity(setup_name, cards, meta):
    any_card = next(iter(cards.values()))
    lines = [
        f"# CNC CONFIDENCE CARD (capital-model sensitivity): {setup_name}",
        "",
        f"**Discovery:** {meta['n_disc']:,} trades (uncapped) / {meta['n_disc_capped']:,} admitted at "
        f"{LIVE_SLOTS} slots; K={meta['K']} hold; round-trip cost {meta['cost']*100:.3f}%.",
        "",
        "## Capital-model sensitivity",
        "Deep-research recipe (stationary block bootstrap of the daily portfolio-return series",
        "+ Lo(2002) HAC Sharpe) applied to three capital-deployment models. PF CI = 95% block",
        "bootstrap, B=5000.",
        "",
        "| Model | PF [95% block-boot CI] | daily Sharpe (HAC; t) | OOS PF | HO PF |",
        "|---|---|---|---|---|",
    ]
    for label, card in cards.items():
        lines.append(_row(label, card))
    c = any_card
    lines += [
        "",
        "## Selection-bias haircut — Deflated Sharpe Ratio (shared; signal-day basis)",
        "",
        f"- **Trials:** M={c.m_trials} swept (doc {meta.get('m_doc','?')}), rho_bar={c.rho_bar:.2f} "
        f"-> effective independent N={c.effective_n:.1f}",
        f"- **SR0 (expected max of N unskilled):** {c.sr0:.4f}  (V[trial SR]={c.var_trial_sharpe:.5f})",
        f"- **Locked-cell SR_hat (signal-day basis):** {c.sr_hat_dsr:+.4f}",
        f"- **DSR = P(true SR > SR0):** {c.dsr:.3f}",
        "",
        "## Interpretation",
        "- The three rows bracket the deployment-model uncertainty (equal-weight = pessimistic;",
        "  uncapped sum = raw edge; capped = production capacity). A future ML allocation model",
        "  would replace the fixed top-rank rationing with a learned take/size policy.",
        "- Per-day Sharpe t-stats are sub-2 across models on ~200-600 signal-days; paper trading",
        "  remains the production-faithful go/no-go gate (resolves survivorship + tier-1 slippage).",
        "",
        "---",
        "_Recipe: specs/2026-06-15-cnc-confidence-card-methodology-research.md._",
        "_Supersedes the intraday iid per-trade BCa + Harvey-Liu cards (invalid for overlapping holds)._",
    ]
    return "\n".join(lines)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # cards contain em-dash / approx glyphs
    except Exception:
        pass
    print("Loading clean daily panel + computing features...")
    panel, base = prep()
    print(f"  panel rows={len(panel):,} | base (ADV-floor) rows={len(base):,} | "
          f"symbols={base['symbol'].nunique()} | years={sorted(base['yr'].unique())}\n")
    _OUT.mkdir(parents=True, exist_ok=True)

    for setup in SETUPS:
        print(f"===== {setup['name']} (K={setup['locked']['K']}) =====")
        cards, meta = run_setup(setup, panel, base)
        meta["m_doc"] = setup["M_doc"]
        any_card = next(iter(cards.values()))
        print(f"  swept M={meta['m_total']} (doc {setup['M_doc']}) | ship-eligible pool={meta['n_ship']} | "
              f"rho_bar={any_card.rho_bar:.2f} -> eff N={any_card.effective_n:.1f} | "
              f"disc n {meta['n_disc']} -> {meta['n_disc_capped']} capped@{LIVE_SLOTS}")
        md = render_sensitivity(setup["name"], cards, meta)
        out = _OUT / f"{setup['name']}_cnc_card.md"
        out.write_text(md, encoding="utf-8")   # write first (utf-8 always succeeds)
        print(md)
        print(f"  [written] {out}\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
