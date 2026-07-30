# Brief: `pead_reaction_drift` — post-earnings reaction-day-surprise drift (both legs)

**Date:** 2026-07-27
**Stage:** 0 (idea) + Stage 1 (Indian-market research) combined
**Family:** EVENT-DRIVEN continuation, 1-10 day CNC horizon
**Status:** **KILLED at Phase 5 (2026-07-28)** — Discovery cell PF_net 2.45 (n=258, broadest
cell, year-stable) collapsed to PF_net 0.67 / WR 40% (n=157) on the one-shot demoted-window
check (2025-01 → 2026-04), gap 1.78 vs the 0.30 overfit ceiling. 2026 (PF 0.36) worse than
2025 (0.87). Ledger lines logged; fresh pool (2026-05+) never touched. **Revival path (the
only one):** pre-registered falsifier #3 fired on schedule — the `announcements_fr` source
died Mar-2025, degrading reaction-day assignment for exactly the failing window; a revival
requires REPAIRED announcement-timing data first, and constitutes a NEW ledger-burning
evaluation, not a salvage. Evidence: `tools/sub9_research/pead_reaction_drift_cell_lock.json`.

**[2026-07-28 FALSIFIER-3 RE-TEST ON REPAIRED DATA: KILL CONFIRMED, FILE CLOSED PERMANENTLY.]**
The timestamp repair (commits 68ee6f3/8692ae8; 1,308 classes corrected, 2025Q2 scheduled
942→96) was run and the locked cell re-tested once on the demoted window under pre-registered
bands (>=1.20 revive / 1.00-1.20 ambiguous / <1.00 confirmed): repaired PF_net 0.851 → KILL
CONFIRMED. Decisive decomposition: ZERO traded events had changed reaction dates; the
unchanged-cohort trades lose identically under both parquets (PF 0.669 pre vs 0.644 repaired);
the apparent improvement came entirely from 33 newly-recovered events (composition, not
correction); 2026JanApr sits at PF ~0.33 under BOTH parquets. The era decay is real. Integrity
check: Discovery reproduced exactly (n=258, PF 2.4505, zero deviation). Fresh pool never
touched. Do not spend a fresh-pool shot on this candidate. Evidence:
`reports/sub9_sanity/_pead_repaired_retest_trades.csv` + ledger line 2026-07-28
(falsifier3_retest_repaired_data).
**Lifecycle:** governed by `docs/setup_lifecycle.md` incl. 2026-07-27 amendments A1-A4.

---

## 1. Why this candidate exists (provenance — read first)

This is the **disciplined redo of the B1 weak kill** flagged by `tasks/lessons.md` #28
(2026-06-15). The original B1 "KILL" tested exactly ONE construction and is invalid as a
verdict. What was actually tested (driver `_tmp_b1_pead_killtest.py`, preserved at repo root):

- LONG leg only (top decile of reaction-day return)
- ONE surprise proxy: raw announcement-reaction-day return
- ONE entry timing: reaction-day + 2 trading days, at open
- Holds {3, 5, 7, 10} trading days, exit close
- ADV floor Rs.20L, 5 ADV tiers, CNC cost model (delivery STT 0.20% + brokerage 0.06% +
  charges 0.047% round-trip) + slippage sweep {10, 20, 30}bp
- Pooled 2023 → 2026-04 (no window discipline)

Untested: the SHORT leg on negative surprises, any market-adjusted/abnormal surprise proxy,
volume confirmation, decile variations, entry T+1/T+3, BMO/AMC class splits, and the
revenue-confirmation conditioning the literature says strengthens the drift.
Per lesson #28 rule 5, B1's honest status is "not found under the variants tested so far" —
this brief pre-registers the full grid.

**Fresh-pool status:** the June 2026 kill-test ran on `clean_daily_from5m` ending
2026-04-30 — the May-2026+ fresh holdout pool (amendment A1) is UNTOUCHED by B1.

## 2. Mechanism statement (one sentence)

Indian mid/small-cap stocks underreact to quarterly earnings surprises because information
diffuses slowly through a retail-dominated holder base and arbitrage is capacity-constrained
(no easy cash-market shorting, thin SLB, MTF margin limits), so the announcement-reaction-day
move — our surprise proxy, since no SUE feed exists — continues to drift in the same
direction for days-to-weeks after the announcement.

## 3. Indian prior evidence (Stage-1 requirement: ≥2 sources)

1. **Sharma & Anand-type PEAD study, Theoretical Economics Letters / SCIRP (2018),
   "Post-Earnings-Announcement Drift Anomaly in India: A Test of Market Efficiency"**
   (https://www.scirp.org/journal/paperinformation?paperid=88060): 2002-2017 sample,
   statistically significant PEAD on Indian stocks — positive-surprise names drift up and
   negative-surprise names drift down over ~64 post-announcement days; anomaly persists
   after controlling for beta, market cap, price-to-book, illiquidity, and idiosyncratic
   volatility. Also reports drift is stronger when revenue surprise confirms the earnings
   surprise (a conditioning dimension in our grid).
2. **"Post-Earning Announcement Drift and Value-Glamour Anomalies in NSE Listed Firms",
   Trends Economics and Management (2021)** (https://journals.vut.cz/index.php/trends/article/view/541,
   also EconStor https://www.econstor.eu/handle/10419/308924): NSE-listed firms 2014-2018,
   1,130 observations, expectations-formation approach around earnings; documents abnormal
   post-announcement return association with earnings surprise on NSE specifically.

Both are academic (satisfies the "not retail-educator" source rule). Caveat honestly noted:
the literature is mixed — at least one Indian study finds efficiency w.r.t. announcements;
this is a real edge question, not settled science. That is what Phase 2 is for.

## 4. Participants (who is on each side)

- **Slow side (we join):** retail holders who anchor on pre-announcement price and
  under-adjust; domestic MF/institutional flow that accumulates gradually over days due to
  execution constraints in mid/small-cap liquidity.
- **Constrained side (why it persists):** arbitrageurs cannot cheaply short the
  negative-surprise leg in the cash market (T+1 delivery, thin SLB), and long-side
  capacity in illiquid names is limited by impact — classic limits-to-arbitrage.
- **Asymmetric direction rule (§3.2 #5):** BOTH legs will be tested, but each leg must
  pass independently; the short leg is additionally gated on a tradability quantification
  (which subset is actually shortable: F&O names intraday-to-multiday via futures, or
  MIS-day-by-day) per lesson #28 rule 3 — no hand-waved "tradability-masked" claims.

## 5. Pre-registered variant grid (the FULL battery, lesson #28 rule 1)

| Dimension | Values |
|---|---|
| Surprise proxy | raw reaction return; market-adjusted (minus Nifty same-day); volume-confirmed (reaction volume ≥ 2x ADV20) |
| Reaction window | 1 day; 2 days (announce + next) |
| Legs | LONG on positive surprise; SHORT on negative surprise (with tradable-subset quantification) |
| Threshold | top/bottom decile; top/bottom 5%; fixed |reaction| ≥ 3% |
| Entry | reaction+1 open; reaction+2 open |
| Hold | 3, 5, 7, 10 trading days (exit close) |
| ADV tier | 5 tiers as in B1; report per-tier, don't pool |
| Conditioning | none; revenue-confirmation (if extractable); BMO/AMC class |

Kill rule per cell family: same as lifecycle Phase 2 (drift delta < 0.1% net directional
footprint = dead dimension). Only after the grid is dry across every cell is "no edge" a
defensible verdict.

## 6. Falsifiers (3, pre-registered)

1. **Mechanism falsifier:** drift must be STRONGER in lower-ADV tiers and in
   volume-confirmed reactions (slow diffusion + limits-to-arbitrage). If the drift
   concentrates in the top-ADV/large-cap tier, the mechanism story is wrong regardless of
   PF — kill.
2. **Regime falsifier:** if the drift sign flips between FII-inflow and FII-outflow regimes
   (7-regime schema), the "edge" is a regime artifact (the pre_results_t1_fade failure
   mode) — kill or regime-gate only via Stage-5 Discovery cell-lock.
3. **Data-integrity falsifier:** announcement-time classification (BMO/AMC/intraday) drives
   reaction-day assignment; the `announcements_fr` source died Mar-2025 so post-2025 AMC
   classification is degraded. If drift results differ materially between
   well-classified (pre-2025) and degraded (post-2025) cohorts, the signal is a
   timestamping artifact — kill.

## 7. Data feasibility (Stage-1 Gate B) + prerequisites

| Input | On disk | Coverage | Action needed |
|---|---|---|---|
| `data/earnings_calendar/earnings_events.parquet` | yes | 36,530 events, 2022-01-13 → 2026-05-04 | **refresh to current** before fresh-pool eval |
| `cache/preaggregate/clean_daily_from5m.feather` (CA-adjusted, amendment A3) | yes | 2023-01-02 → 2026-04-30 | **rebuild through 2026-07** from monthly feathers (which exist to 2026_07) |
| MTF/MIS tradability lists | yes | data/mtf_universe/ + live Zerodha MIS fetcher | intersect short-leg universe (A3) |
| ProductionUniverseGate | yes | tools/sub9_research/production_universe.py | mandatory from Phase 2 (A3) |

## 8. Regulatory sensitivity

- SEBI LODR Reg 30/33: earnings disclosure timing rules — stable; no known 2024-26 cutover
  affecting announcement mechanics.
- Delivery STT 0.20% round-trip dominates cost at CNC horizon → holds < 3 days are
  structurally fee-fragile; the grid respects this.
- Oct-2025 F&O MWPL changes: irrelevant to the cash-CNC long leg; RELEVANT to any
  futures-implemented short leg — short-leg validation must use post-Oct-2025 data for its
  decisive check.

## 9. Adjacent setups (Stage-0 requirement, feeds effective-M and A4 factor budget)

- `xsec_momentum_demeaned` (sibling brief, same date): formation returns include
  earnings-reaction moves, so the two candidates will be partially correlated — if both
  ship, measure pairwise daily-PnL correlation before assigning separate factor budgets.
- Capitulation-reversion cluster (A2/C1/C4/C6 + panic_crash): expected LOW correlation
  (entries not conditioned on crashes; opposite-direction short leg possible) — but the
  NEGATIVE-surprise LONG-side reversion trade is explicitly OUT of scope here (that's the
  cluster's territory).
- Retired relatives and their lessons baked in: `earnings_day_intraday_fade` (look-ahead
  SL bias — we hold multi-day, no intraday SL from day aggregates),
  `pre_results_t1_fade` (FII-regime dependence — falsifier #2), `earnings_t1_runup_fade_short`
  (drafted-never-built; different window, fade not drift).

## 9b. Phase-5 sweep pre-registration (added 2026-07-28, BEFORE any Phase-5 run)

Stage-4 result: LONG n=258 on Discovery (long-leg-primary; SHORT parked pending a tradable
implementation case — F&O-shortable subset was n=38/thin). Phase-5 dimensions, locked now:

- **Cell dimensions (filter sweep):** threshold {causal pct5, causal pct3}; hold {5, 7, 10
  sessions}; cap_segment {all, large+mid, mid+small+unknown}; ADV tier {all, tiers 1-4 (drop
  most-liquid)}. Floors: n >= 100 per cell (Discovery), PF floor per lifecycle. NO other
  dimensions may be added after seeing results (lesson #2).
- **Exit sweep:** hold-to-H close (baseline); vol-scaled target (2-sigma, multiday
  target-exit precedent) with H time-stop; NO tight stop variants — Stage-4 MAE profile
  (mean −6.3%) says stops would clip the drift; a stopped variant may be measured but only
  as a diagnostic, never selected without both windows agreeing.
- **Costs:** CNC delivery round-trip (STT 0.20% + brokerage + charges) + 20bp slippage,
  the mtf_capitulation cost model.
- **Window discipline (amendment A1):** sweep + cell-lock on Discovery 2023-24; the locked
  cell then runs ONCE on the demoted development window (2025-01 → 2026-04-30) as a
  stability check — logged to the experiment ledger; freeze commit; then ONE shot on the
  fresh pool (2026-05-01 → present) which is decisive together with paper.

## 10. A1/A2 compliance plan

- **Development data:** Discovery 2023-2024 + demoted windows (2025-01 → 2026-04-30),
  treated as one development pool for the grid.
- **Freeze:** after Phase 5 cell-lock, spec + locked cell + geometry committed (freeze
  commit) BEFORE any evaluation touching 2026-05+.
- **Decisive gate:** one-shot on fresh pool (2026-05-01+, refreshed data) + paper forward
  run. Every evaluation on demoted or fresh windows appends a `docs/experiment_ledger.jsonl`
  line in the same session (this grid will add many lines — that is the point).
