# Multi-Day Composite Selection & Ranking — Design

**Date:** 2026-06-29
**Status:** Design approved; plan pending
**Scope:** Cross-setup selection/ranking and dedupe for the multi-day CNC/MTF
capitulation-reversion family. **Technical selection only** — capital/margin-pool
arbitration is explicitly out of scope (setups are in paper testing; caps lifted).
`close_dn_overnight_long` (overnight/BTST) is a separate family and is NOT in scope.

---

## 1. Problem

Four `horizon == "multi_day"` setups now share one execution path
(`services/execution/mtf_capitulation_handlers.py`) and one tiny tier-1-ADV
universe:

| Setup | `selection_mode` | Hold | Confidence-card adj-Sharpe |
|---|---|---|---|
| `mtf_capitulation_revert_long` (A2) | `trailing_loser_decile` (5d) | 2d | 0.80 |
| `low52_capitulation_revert_long` (C1) | `near_period_low` (252d) | 2d | 0.93 |
| `zscore_oversold_revert_long` (C4) | `zscore_oversold` (20d, −1.5σ) | 2d | 0.73 |
| `crash2d_revert_long` (C6) | `trailing_loser_decile` (2d) | 3d | 0.93 |

`run_eod(phase="entries")` loops these setups and **places AMO BUYs inside the
loop**, each setup independent: its own `PositionPersistence` store, its own
caps, and a dedupe check (`persistence.get_position`) that sees only its own
store. Because the four are **same-side (all long) and same-mechanism
(capitulation reversion) — i.e. highly correlated** — this produces:

1. **Hidden concentration via cross-setup duplicates.** One name can be a
   5d-loser *and* near its 252d low *and* z-score-oversold the same day, so it is
   bought by 2–4 setups into 2–4 separate stores → a multiplied position on one
   symbol, invisible to each setup's own dedupe.
2. **Cross-day re-entry.** A name still inside its 2–3-day hold that re-signals is
   re-entered, because the held-check is per-store.
3. **No selection priority when oversubscribed.** When combined candidates exceed
   the (eventual) cap, Python dict iteration order over setups decides who enters.

Grinold-Kahn (`IR = IC × √Breadth`) names the core issue: **four correlated
signals are far fewer than four independent bets**; treating them as independent
sleeves overstates diversification and concentrates risk on exactly the consensus
names.

## 2. Approach — Integrated Composite (chosen)

Professional practice combines correlated same-side signals via the **integrated
(composite) approach**: normalize each signal to a common scale, blend into one
composite score per name, then build a single basket. For same-side signals there
is no long/short "netting" — overlap is **pure agreement**, so dedupe and a
consensus boost both fall out of the composite for free (no separate dedupe pass,
no hand-tuned consensus knob).

Rejected alternatives:
- **Mixed sleeves + dedupe-by-precedence** — overlap merely removed; no consensus
  signal; cross-mode ranking leans only on `rank_pct` comparability.
- **Dual-view (sleeves + composite)** — most faithful but most moving parts;
  deferred. The chosen design already preserves per-setup measurement (§5), so the
  separate sleeve book is unnecessary now.

References: GSAM "Combining Investment Signals in Long/Short Strategies"; FactSet
"A Practical Approach to Weighting Signals"; Grinold-Kahn Fundamental Law of
Active Management.

## 3. Architecture

Invert the entries phase from **place-in-loop** to **collect → compose →
place-once**:

```
run_eod(phase="entries"):
  1. for each eligible multi-day setup s:
         basket_s = CrossSectionalRanker(s).rank(panel, today, eligible)   # unchanged + cap_score
  2. chosen = MultiDayCompositeSelector.select(baskets, held_union, config) # NEW module
  3. place AMO BUYs for `chosen`, once; persist (owner store) + tag contributors
```

- `services/cross_sectional_ranker.py` — **unchanged except one additive output
  field** (`cap_score`, §4 Step A).
- `services/multiday_composite_selector.py` — **new**, the only new module.
- `services/execution/mtf_capitulation_handlers.py` — `_run_entries` restructured
  from per-setup place-loop into collect-baskets → call selector → place-once.
  Exits, persistence, fee math, AMO placement: unchanged.

## 4. The composite score

### Step A — Normalize (in the ranker, additive)
The ranker emits one new field per basket name, `cap_score` = the
**cross-sectional standardized capitulation magnitude**:
- Orient the mode's signal so "more capitulated = larger":
  - `trailing_loser_decile`: magnitude = `−signal` (signal = trailing return; more
    negative = more capitulated).
  - `near_period_low`: magnitude = `−signal` (signal = `close/low − 1`; smaller =
    more capitulated).
  - `zscore_oversold`: magnitude = `−signal` (signal = z; more negative = more
    oversold).
- Z-score the magnitude over **that day's full qualifying cross-section** (the
  `today` frame already computed in `rank()`, not just the selected basket — keeps
  the scale stable), then **clip negatives to 0** (only the capitulated tail
  contributes).
- ~5 lines where the full cross-section already lives; no extra pass.

### Step B — Blend across setups (the selector)
For each symbol `i`:

```
composite_i = Σ over setups s that selected i of  w_s × cap_score_{s,i}
```

- **Sum, not mean** — the breadth/consensus mechanism. A name flagged by 3 setups
  accumulates 3 contributions and floats up; a single-setup *deeply* capitulated
  name can still out-rank a 3-setup *mild* consensus.
- **Equal weights `w_s = 1.0` for v1** — robust ("1/N") default; 4 borderline
  setups is where optimized weights overfit. Per-setup `composite_weight` config
  key makes this explicit and tunable. IC/confidence-card weighting is the
  documented next step, justified by §6 diagnostics.
- **Tiebreaker:** `max` `tshock` across selecting setups (universally comparable
  turnover-shock conviction).

### Step C — Select
- Drop any symbol already held by **any** setup (union of stores, §5).
- Rank by `composite_i` desc (tiebreak `tshock`); take **top-N**, where N =
  config-driven combined `max_new_per_day` (generous in paper → effectively "take
  all", but the ordering is real and the cap exists for when it matters).

Result: one deduped, consensus-ranked basket for the whole family.

## 5. Dedupe, attribution & held-filter

- **Book deduped:** one position per symbol, stored once in the **owner** setup's
  existing `PositionPersistence` (owner = highest `w_s × cap_score` contributor).
  Exits / PnL / persistence / fee math unchanged.
- **Measurement per-setup (testing tension resolved):** the position record tags
  **all** contributing setups (`contributors: [...]`); on exit the realized net
  PnL is fed to **every** contributor's `DecayTripwire`, not just the owner's. Each
  setup's standalone edge reflects every name it flagged — shared or not — while
  the book holds the name once.
- **Cross-day held-filter:** the selector's "already held" set is the **union** of
  all setups' position stores; those names are excluded before ranking. Kills the
  re-entry-during-hold double-up. No store unification — exits still run per-setup
  against their own store.

## 6. Config (CLAUDE.md rule 1 — no hardcoded defaults)

- **Per-setup:** new `composite_weight` key (all `1.0` to start — written, not
  implied).
- **New family block** `multi_day_portfolio`: combined `max_new_per_day`,
  `max_concurrent` (selection caps; generous in paper), `cap_score_clip`,
  tiebreaker source. Missing key = startup error (fail fast).

## 7. Diagnostics (the evidence that later tunes weights)

Per-day selection record (jsonl): overlap rate (names flagged by ≥2 setups),
composite ranking, owner + contributors per name, consensus distribution. This
**empirically measures the family's effective breadth** and whether consensus
names actually outperform — the evidence that justifies or rejects IC-weighting.
Without it, weight choices are guesses.

## 8. Testing / validation

- **TDD `MultiDayCompositeSelector`:** per-mode normalization orientation;
  sum-blend; consensus boost (3-setup vs 1-setup deep); dedupe to one row;
  held-filter (union of stores); top-N cap; multi-contributor attribution (PnL fed
  to all contributors).
- **Ranker `cap_score`:** orientation + clip per mode (unit tests).
- **Restructure `run_eod(entries)`:** integration test — collect→compose→place
  places once, respects combined cap, feeds all contributors' tripwires; exits
  unaffected.
- **Backtest parity:** dry-run replay over `cache/preaggregate/clean_daily_from5m.feather`;
  measure the real overlap rate; compare composite-book vs naive-sleeves on the
  same dates.

## 9. Live/backtest symmetry

The selector is pure function of (baskets, held-union, config) — identical in
dry-run, paper, and live. IST-naive throughout. No wall-clock dependence.

## 10. Out of scope (explicit)

- Capital/margin shared-pool arbitration (revisit at live activation).
- IC/confidence-card composite weighting (v2, gated on §7 diagnostics).
- Score-every-name integrated variant (selecting setups only contribute in v1).
- `close_dn_overnight_long` (separate family).
