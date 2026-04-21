# Sub-Project #3: Cross-Sectional Features — Design Spec

**Date:** 2026-04-21
**Sub-project:** #3 of 6 in the Trading System Rebuild
**Status:** Design approved, awaiting writing-plans
**Predecessor:** Sub-project #1 (Edge Discovery Gauntlet) — produces 90 approved Stage-5 rules + Stage 5b ruleset simulation showing aggregate PF 1.36 on 368 trades/day
**Successor:** Sub-project #2 (Conviction Architecture) — final 15-20 trades/day ranking

---

## 1. Context: why this exists

Stage 5b of the gauntlet showed the 90 approved rules produce an aggregate PF of 1.36 with session Sharpe 0.74 across 368 trades/day. Production target is 15-20 trades/day.

Sub-project #3 sits between #1 (edge discovery) and #2 (conviction scoring) as a pre-ranking filter layer. Its job: cut 368/day → ≤~250/day via cross-sectional signals that capture **correlated-move dilution** (the biggest aggregate-drag risk per Stage 5b data) and **volume-flow concentration** (the biggest-edge predictor within cap-classified universe per empirical probe).

Sub-project #2 then does the final 250 → 15-20 cut via conviction-ranking.

### What this sub-project is NOT

- A conviction scorer (sub-project #2)
- A deployment-quality filter for circuit-hit / non-MIS stocks (SEBI tradability; MIS filter already in place via `backtest_mis_filtering`)
- A slippage-realism model (sub-project #4 Shadow/Parity Loop)
- Per-trade features (those already exist in detector context — `pdz_confluence_count` etc.)

---

## 2. Empirical basis

Four candidate cross-sectional features were probed against 178,151 Discovery trades (matching the 90 approved Stage-5 rules):

| Feature | Universe result | Unknown cap_segment result |
|---|---|---|
| **RVOL** (volume / 20d-same-mod avg) | **DOWN** for small/mid/micro | **FLAT** |
| **Crowdedness** (same-setup trades in ±5min) | **DOWN** universally | **DOWN** (modest) |
| Intraday return rank | FLAT | FLAT |
| Volatility ratio rank | FLAT | FLAT |
| NIFTY context (macro) | FLAT | FLAT |

**Two features survived:** RVOL (conditional on cap) + Crowdedness (universal).

### Mechanism explanation for unknown-cap ceiling

Per Pacific-Basin Finance Journal 2023 ("Momentum, reversals and liquidity: Indian evidence") and ScienceDirect 2023 ("Cross-sectional reversal of intraday returns and investor heterogeneity in an emerging market"):

> Cross-sectional signals in emerging-market intraday require **institutional-vs-retail flow heterogeneity** within the universe. The signal discriminates because some stocks have more informed flow than others at a given moment.

Our "unknown" cap_segment (T-group / penny-tier / IBULLSLTD / RPOWER / FOODSIN class) has **near-zero institutional presence**. All stocks there are pure retail flow — no cross-sectional heterogeneity to exploit. Only AGGREGATE BEHAVIORAL BUNCHING signals (crowdedness — how many retail desks pile into the same pattern simultaneously) survive this regime.

### Mechanistic intuition for the two features

**RVOL DOWN (low RVOL = better PF in cap-classified names):** Our 90 rules are predominantly reversal setups (`premium_zone_short`, `range_bounce_short`, `order_block_short`, `resistance_bounce_short`). For reversal, high RVOL means the retail crowd has already piled in aggressively → fading into aggressive flow means getting run over before reversion kicks in. Low RVOL = the pattern formed on a quiet tape = smaller counterparty force pushing against the fade = reversion works cleanly.

**Crowdedness DOWN:** When many correlated stocks fire the same setup simultaneously, it's a regime-wide move, not 40 independent trades. Edge gets diluted across correlated positions. Entry at lower crowd = capturing pattern on a single-stock idiosyncratic move, not a universe-wide drift.

Both signals point to the same underlying story: **our edge lives in quiet, uncorrelated tape, not loud regime-aligned moves.**

---

## 3. Feature specifications

### F1: RVOL low-filter (conditional on cap_segment)

**Signal:** For each 5m bar, compute `rvol = current_bar_volume / mean(prior 20 sessions' volume at same minute-of-day)`.

**Cross-sectional step:** Rank `rvol` across all trades firing in the same 5m window, within the same cap_segment tier (small / mid / micro), → percentile 0-100.

**Filter:** Skip trades where `rvol_pct_in_tier >= 70` (top-30% highest RVOL in peer tier).

**Conditional application:**
- **Apply** for cap_segment ∈ {small_cap, mid_cap, micro_cap} (~77% of trades)
- **Skip** for cap_segment ∈ {unknown, large_cap} — no empirical signal
- **Skip** for hour_bucket = late — effect direction reverses there (MIS-unwind momentum)

**Data requirements:**
- Per-trade: stock's current-bar volume (from `bar5` in events.jsonl, already logged)
- Historical: 20-session same-mod average volume per symbol
- Cross-sectional: per-bar grouping of candidates firing at same (session_date, minute_of_day, cap_segment)

### F2: Crowdedness low-filter (universal)

**Signal:** For each candidate trade at (session, setup_type, decision_ts), count the number of OTHER candidates with the same setup_type firing within ±5min across the entire MIS-eligible universe.

**Filter:** Skip trades where `crowdedness >= 40` (top-30% of observed crowdedness distribution on Discovery — threshold comes from literature convention of ~10 per minute-equivalent × 5min window × same-setup factor; final threshold is empirically locked once, then frozen).

**Application:** Universal across all cap_segments, all setups, all hours. Validated across every slice.

**Data requirements:**
- All candidate events (trades + their setup_type + decision_ts)
- Rolling window aggregation per (setup_type, ±5min)

### F3 (deferred): Float-adjusted move

**Signal:** For each stock, compute `move_per_free_float_rupee = (current_bar_return × mcap / free_float_mcap)`.

**Why deferred:** Requires NSE bhavcopy shareholding-pattern data we don't currently load. Marginal expected lift based on penny-stock practitioner literature. Revisit if F1 + F2 don't deliver enough.

---

## 4. Architecture

### 4.1 Module structure

```
services/cross_sectional/           ← NEW package
├── __init__.py
├── universe_rvol.py                ← F1: per-symbol rolling RVOL + cross-sectional rank
├── crowdedness_counter.py          ← F2: rolling same-setup window counter
└── gate.py                         ← CrossSectionalGate: applies F1 + F2 filters

services/gates/                     ← existing
├── ...
└── cross_sectional_gate.py         ← thin wrapper calling services/cross_sectional/gate.py
```

### 4.2 Data flow (live + backtest)

```
Pattern detector fires setup signal
         ↓
Decision pipeline proposes trade at (symbol, setup_type, decision_ts)
         ↓
CrossSectionalGate.evaluate(candidate):
    ├── F1: fetch rvol_pct for candidate (cap-tier cross-sectional rank)
    │      skip if rvol_pct >= 70 AND cap in {small,mid,micro} AND hour != late
    ├── F2: fetch crowdedness for candidate (same-setup ±5min count)
    │      skip if crowdedness >= 40
    └── Return: ALLOW or REJECT (reason logged)
         ↓
If ALLOW: trade proceeds to sizing + execution
```

### 4.3 RVOL state maintenance

**Backtest:** Pre-compute per (symbol, mod) rolling 20-session mean from monthly feather cache (already proven workable in `probe_3_candidates.py`).

**Live:** `UniverseRVOLState` singleton maintains per-symbol deque of (date, mod, volume). On each 5m bar close, updates deque and recomputes the rolling-20 mean for next bar. Cross-sectional ranking happens at candidate-evaluation time by fetching current-bar rvol for ALL candidates firing in the same 5m window.

**Cross-sectional ranking per cap_segment:** maintained as separate ranked series for {small_cap, mid_cap, micro_cap}. Updated on each 5m bar close.

### 4.4 Crowdedness state maintenance

`CrowdednessCounter` singleton keeps a per-setup_type sliding 10-min window of signal timestamps. On each candidate evaluation, returns the count in ±5min of candidate's `decision_ts`. Pure in-memory; ~5min of events per setup = trivial storage.

### 4.5 Integration with existing pipelines

**Entry point:** `services/gates/trade_decision_gate.py` already orchestrates pre-trade gates. Add `CrossSectionalGate` as a new gate, positioned:
- AFTER detector + setup-level filters (we only apply cross-sectional to approved candidates)
- BEFORE sizing + execution (rejecting cheaply)

**Config:** Add `cross_sectional_gate` section to `config/configuration.json`:
```json
"cross_sectional_gate": {
  "enabled": true,
  "f1_rvol_enabled": true,
  "f1_rvol_threshold_pct": 70,
  "f1_applicable_caps": ["small_cap", "mid_cap", "micro_cap"],
  "f1_skip_hour_buckets": ["late"],
  "f2_crowdedness_enabled": true,
  "f2_crowdedness_threshold": 40,
  "f2_crowdedness_window_min": 5
}
```

All parameters config-driven per project standard (no hardcoded thresholds in code).

### 4.6 Backtest equivalence

`tools/edge_discovery/stages/stage5b_ruleset_simulation.py` already simulates ruleset behavior. Extend it (or create `stage5c_cross_sectional_simulation.py`) to also apply F1 + F2 filters and report:
- Trade count reduction vs baseline
- PF delta
- Session Sharpe delta
- Per-cap, per-hour breakdown of filter impact

This becomes the formal sub-project #3 deliverable metric.

---

## 5. Testing

### 5.1 Unit tests

- `test_universe_rvol.py`: synthetic per-symbol volume series, verify 20-session same-mod rolling mean matches hand-computed expectation
- `test_crowdedness_counter.py`: synthetic signal stream, verify ±5min window count is correct (inclusive/exclusive boundaries, tie-breaks)
- `test_cross_sectional_gate.py`: mock candidates, verify filter logic (skip/allow decisions) match config

### 5.2 Integration test

- End-to-end: load synthetic mini-run fixture (reusing `tests/edge_discovery/fixtures/make_fixtures.py` patterns), run gauntlet + cross-sectional gate, verify output aggregate metrics improve per hypothesis

### 5.3 Empirical backtest validation

- Run full Discovery gauntlet with F1 + F2 active; verify:
  - Trade count drops from 368/day to target range (~230-280/day)
  - Aggregate PF maintained or improved (expected lift ~5-10%)
  - Per-cap breakdown shows F1 effect on small/mid/micro and F2 effect on unknown
  - NO PF degradation in any specific cell (specifically: unknown+afternoon must retain its PF)

### 5.4 Live/paper parity

- When live/paper integration is built: verify that CrossSectionalGate's filter decisions match backtest (no divergence in crowdedness counts or RVOL percentiles for identical trade streams)

---

## 6. Success criteria

This sub-project ships when:

1. **F1 + F2 implemented + unit tested** (pytest green)
2. **Integrated into `trade_decision_gate`** with config-driven toggles
3. **Discovery re-run shows:**
   - Trade count: 368/day → 230-280/day (acceptable range)
   - Aggregate PF: 1.36 → ≥ 1.45 (+7% lift)
   - Session Sharpe: ≥ 0.74 (no degradation)
   - Per-cap-segment sanity: no single cap_segment's PF drops vs unfiltered baseline
4. **Documented handoff to sub-project #2:** pre-filtered candidate stream, ready for conviction scoring to pick top 15-20/day

---

## 7. Out of scope (explicit)

- **Float-adjusted move (F3):** needs NSE bhavcopy shareholding data; marginal expected value; deferred
- **Sector rotation / sector RS:** requires sector-mapping infrastructure; can add later if edge warrants
- **Conviction scoring / top-N selection:** sub-project #2
- **Slippage realism / shadow-loop fills:** sub-project #4
- **Unknown-cap refinement beyond F2:** research shows ceiling (no institutional heterogeneity to exploit); future work should look at per-trade features (confluence counts) via sub-project #2 or shadow-loop fills via sub-project #4
- **Live universe-wide polling infrastructure:** assumes `early_mis_universe_filter` already subscribes all MIS-eligible symbols for live; if not, that's infrastructure work outside this sub-project

---

## 8. Known risks & mitigations

| Risk | Mitigation |
|---|---|
| F1 thresholds (70th pct, cap-tier list) are data-fitted on Discovery | Thresholds come from penny-stock practitioner literature (70th pct), not tuned on data. Cap-tier list comes from empirical direction test, not threshold tuning. Re-validate on Validation period before Holdout. |
| Crowdedness threshold (40) appears derived from data | Empirical direction test showed monotonic DOWN across quantile buckets; 40 is the top-30% boundary per literature convention (practitioner "drop top-30%" rule). Treated as a locked parameter pending Validation. |
| F1 might leave edge on table for unknown-cap (no filter) | Confirmed by research — no cross-sectional signal exists there structurally. Unknown-cap edge refinement belongs to sub-project #2 (per-trade conviction) or sub-project #4 (shadow-loop fill-quality). |
| Live data for universe-wide RVOL computation may have gaps | Bar density gate already filters sub-70% 1m bars. Cross-sectional ranking skips symbols with insufficient history rather than assuming a default percentile. |
| Cross-sectional state singleton may lag on late bars | Design uses end-of-bar update; decision-time queries always see last-closed-bar state. Matches existing detector patterns. |

---

## 9. Sources

**Academic:**
- [Momentum, reversals and liquidity: Indian evidence](https://www.sciencedirect.com/science/article/abs/pii/S0927538X23002640) — Pacific-Basin Finance Journal 2023
- [Cross-sectional reversal of intraday returns and investor heterogeneity in an emerging market](https://www.sciencedirect.com/science/article/pii/S2214845023000029) — ScienceDirect 2023
- [Investor clientele and intraday patterns in the cross section of stock returns](https://link.springer.com/article/10.1007/s11156-024-01319-8) — Review of QFA 2024
- [Jegadeesh & Titman (1993)](https://www.bauer.uh.edu/rsusmel/phd/jegadeesh-titman93.pdf) — seminal momentum/reversal paper

**Indian-market practitioner:**
- [Nifty Microcap 250 Factsheet (NSE)](https://nsearchives.nseindia.com/content/indices/Factsheet_Nifty_Microcap_250_Index.pdf)
- [Relative Volume (RVOL) for NSE — Goodwill](https://www.gwcindia.in/blog/how-to-use-relative-volume-rvol-for-better-entry-timing-in-indian-stocks/)
- [Nithin Kamath on non-F&O liquidity (Business Standard)](https://www.business-standard.com/markets/news/markets-equities-zerodhas-kamath-flags-risks-amid-soaring-mtf-book-126012101037_1.html)
- [BE-series intraday prohibition (NSE)](https://tradesmartonline.in/help/trading/what-does-eq-and-be-series-stand-for-in-nse/)

**Empirical probes (this project):**
- `tools/edge_discovery/volume_probe_v2.py` — RVOL direction test (98.8% coverage)
- `tools/edge_discovery/probe_3_candidates.py` — Return rank / NIFTY / volatility rank tests
- `tools/edge_discovery/probe_crowdedness_per_cap.py` — Crowdedness per cap_segment validation

---

**End of design spec.**
