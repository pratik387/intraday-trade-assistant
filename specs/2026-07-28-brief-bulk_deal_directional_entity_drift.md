# Brief: `bulk_deal_directional_entity_drift` — entity-class-conditioned bulk-deal drift

**Date:** 2026-07-28
**Stage:** 0 (idea) + 1 (Indian-market research) combined
**Family:** EVENT-DRIVEN, entity-conditioned, T+1 → T+5 CNC horizon (long-primary on directional
buys; sell-side measured with tradability quantification)
**Status:** **KILLED at Phase 2 (2026-07-28, uncommitted per user instruction).** Class
falsifier FAILED: directional-entity deals drift NEGATIVE at every era×hold (best marginal
delta +0.10%, at the kill line); the only directional-vs-churn separation is churn reversing
*harder*. Salience INVERTED (larger deal-size/ADV → stronger reversal; institutions ≤
individuals) — the footprint is generic post-disclosure fade, not slow-diffusing information.
The tempting inverse (fade directional buys: era_A −1.14%) is era-inconsistent under A5
(era_B −0.06%) and churn-inseparable — no inverse-edge rescue. The three anonymous-family
kills stand. **Reusable asset:** 8,142 churn entity-date classifications (Graviton-class
mechanical flow) as an "ignore" tag for any future deal-based candidate. Evidence:
`reports/sub9_sanity/_bulk_entity_drift_phase2.csv` + ledger line 2026-07-28.
**Lifecycle:** `docs/setup_lifecycle.md` incl. amendments A1-A5.

---

## 1. Provenance + the structural discovery (2026-07-28 recon)

Three adjacent corpses, all of which treated bulk/block deals as ANONYMOUS events:
- `block_deal_accumulation` (C-01, retired at sanity): same-session level-defense; both
  directions failed. Its revival condition #1 explicitly demanded client_name enrichment —
  this brief is that enrichment.
- `bulk_block_buy_continuation` (retired at sanity, PF 0.636): entity-blind VALUE-aggregate,
  T+1 intraday-after-gap — the construction note said the literature edge is T+1→T+5
  cumulative and the gap eats the intraday capture.
- `nse_block_deal_counter_flow` (drafted, never built): anonymous FOMO fade.

**The recon's structural finding:** of 71,971 bulk rows (2023-01 → 2026-04, ClientName 100%
populated), the top-50 entities = 54.3% of ALL rows and are almost entirely HFT/prop shops
(Graviton, HRTI, QE, NK, Citadel, Jump, iRage, XTX...) that are **100% both-sided per
symbol-day with median |net qty| ≈ 0** — disclosure-threshold churn carrying zero positioning
information. 108 of 235 ≥30-row entities are churn-like, covering **51.5% of the dataset**.
Every prior anonymous construction averaged real signal with this majority-noise. The
signal-bearing population: **directional entities** (<30% both-sided sym-days) — MFs, FPIs,
emerging-market funds, named HNIs — 5,375 directional event-units (era_A 3,371 / era_B 2,004)
across 1,276 symbols.

## 2. Mechanism statement (one sentence)

Bulk/block deals by DIRECTIONAL entities (funds/institutions/HNIs that take positions home,
identified by their non-churn footprint) carry slow-diffusing information that resolves as a
T+1 → T+5 drift in the deal's direction, invisible in prior studies because 51.5% of the
disclosure tape is HFT threshold-churn with zero net position.

## 3. Indian anchors (Stage-1)

- SEBI/NSE bulk-deal disclosure: ≥0.5% of equity in a day must be disclosed with client name
  — the ONLY Indian public feed naming counterparties daily (published ~18:00 IST T+0 →
  earliest actionable = T+1 09:15 open; both parquets are date-only, consistent).
- The churn-vs-directional split is itself an Indian-microstructure artifact: prop shops
  crossing the 0.5% threshold intraday in illiquid names is an NSE-specific disclosure
  quirk — the "noise floor" other markets' block feeds don't have.
- Institutional-follow literature (the prior brief's 50-150bps T+1→T+5 citation) applies to
  the directional subset only.

## 4. Entity classification (pre-registered, computed on rolling PRIOR history only)

Per entity, trailing both-sided share of symbol-days (causal — computed from deals strictly
before the event date, min 10 prior appearances):
- **churn** (>80% both-sided): EXCLUDED — this is the noise filter, the core of the thesis
- **directional** (<30%): the signal population
- **ambiguous** (30-80%) or <10 prior appearances: EXCLUDED (no salvage-mining the middle)
Cheap normalization (case/punctuation/suffix) + a small manual alias table (~10-20 entries:
reordered individual names, LLP/PVT variants, the MANSI sibling handoff) — recon verified
no heavy fuzzy-matching burden. Entity-CLASS conditioning only; per-entity track-record
conditioning is pre-declared OUT of scope (only 1 directional entity has ≥100 events —
un-lockable under n-floors).

## 5. Pre-registered Phase-2 grid

Event = (directional entity, symbol, date, side); entry T+1 open; RAW drift, no fees.
Dimensions: side {BUY→long drift, SELL→short drift} × hold {1, 3, 5 sessions} × entity class
{directional-institution, directional-individual} × deal-size-vs-ADV bucket (terciles) ×
era {A, B} (A5 split mandatory) × source {bulk, block}. CONTROL COHORT (mandatory, same
grid): churn-entity deals — falsifier #1 needs the side-by-side.

## 6. Falsifiers (3, pre-registered)

1. **Class falsifier (load-bearing):** if directional-entity drift ≈ churn-entity drift
   (no material separation per era), entity conditioning adds nothing — the anonymous kills
   stand and this candidate dies with them. This is the single number the thesis lives on.
2. **Salience falsifier:** drift must scale with deal size relative to ADV and be stronger
   for institution-class than individual-class. Flat = generic post-disclosure beta — kill.
3. **Coverage falsifier (era_A honesty):** era_A clean_daily coverage of bulk symbols is
   <45% (the 2023 archive cohort = survivorship, lesson #18). Era_A results are computed on
   the covered cohort with that caveat and demoted to sign-only evidence; **era_B is the
   primary development read**; if era_B and era_A(covered) disagree in sign, treat as
   era-inconsistent per A5 unless the coverage bias itself explains it (must be argued
   explicitly, not assumed).

## 7. Data prerequisites

1. Parsing shim for the bulk parquet (string qty/price, Indian commas, %d-%b-%Y) — recon
   verified zero parse failures.
2. Normalization + alias table (small, manual, committed as an asset).
3. Top-up scrape: both parquets end 2026-04-30 — bulk/block May→July needed BEFORE any
   fresh-pool one-shot (~3 months accrual; fetchers exist per the parquets' provenance).
4. NO new infra otherwise; prices from clean_daily_from5m (A3).

## 8. Adjacent setups

- Event-driven long-ish family: distinct mechanism from capitulation cluster (not conditioned
  on crashes) and from the killed anonymous constructions (different conditioning variable).
- SELL-side short leg subject to the same multiday-shorting tradability quantification as
  PEAD's short leg was (no hand-waving, lesson #28 rule 3).
- If the class falsifier passes, the churn-entity list itself becomes a reusable asset
  (a "mechanical flow, ignore" tag for any future deal-based candidate).

## 9. A1/A2 compliance

Development = 2023-01 → 2026-04-30 with A5 era-split (era_A carrying the §6.3 coverage
caveat); freeze before any 2026-05+ statistic; fresh-pool one-shot + paper decisive; every
demoted/fresh evaluation logged to `docs/experiment_ledger.jsonl`. Prior anonymous-family
evaluations are covered by the pre-ledger M≥200 baseline.
