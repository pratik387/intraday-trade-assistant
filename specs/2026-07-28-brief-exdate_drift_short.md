# Brief: `exdate_drift_short` — corporate-action ex-date anchoring drift (pooled sleeve)

**Date:** 2026-07-28
**Stage:** 0 (idea) + 1 (Indian-market research) combined
**Family:** EVENT-DRIVEN short, 1-5 day horizon (variant A intraday-D+1 MIS; variant B 5d CNC-short measured-only pending SLB)
**Status:** DRAFT — awaiting user review (lesson #20 gate). Not committed until approved.
**Lifecycle:** `docs/setup_lifecycle.md` incl. amendments A1-A5.

---

## 1. Provenance (read first — this is a frequency-fix pooling, not a new idea)

Consolidation of three prior threads (full record: 2026-07-28 consolidation, agent report):

1. **`post_split_bonus_short` — the validated core.** Phase-4 (locked construction: ANY NSE
   split/bonus ex-date D, unconditional SHORT at D+1 09:15 open, cover D+1 close [A] or D+5
   close [B]): Discovery PF 5.31/4.90, OOS-2025 2.31/2.88, Holdout 1.70/8.58, WR 71-77%,
   n=210. **Edge positive in BOTH eras (A5-compliant before A5 existed).** KILLED 2026-06-01
   on frequency alone (~55-60 trades/yr; `specs/2026-06-01-post_split_bonus_short-PHASE5-KILL.md`
   explicitly preserves the signature as real and mandates an n/yr screen — which this sleeve
   is designed to pass).
2. **`dividend_ex_date_t1_intraday` — failed its gate, never formally retired.** The 2026-05-08
   Discovery run of the intraday open-fade construction: net PF 0.93 (gross 1.32 — fees ate it,
   exactly as its brief predicted). THIS SLEEVE DOES NOT REUSE THAT CONSTRUCTION. Informational
   sub-slices from that run (interim dividends PF 4.30 n=14; yield 1.5-3% positive; yield 5%+
   toxic) are treated as *hypotheses carrying selection risk*, admissible only under the new
   drift construction. This brief also serves as the formal record that the old dividend
   intraday-fade construction FAILED (gap flagged in consolidation §6.3).
3. **`buyback_tender_intraday` — excluded.** Different mechanism (arb unwind, not anchoring),
   no data on disk. Not a member; revisit separately if ever.
4. The drafted split/bonus brief's 09:20 gap-revert-to-theoretical mechanic was NEVER tested —
   it remains untested and is NOT claimed here.

## 2. Mechanism statement (one sentence)

After a mechanical ex-date price adjustment (split, bonus, or large dividend), retail
participants anchoring on the pre-adjustment price — "the stock is cheap now", "free shares",
mis-set GTTs and stale stop levels — generate temporary net demand that resolves as a
multi-day DOWNWARD drift from the D+1 open, harvestable as a short.

## 3. Indian anchors (Stage-1)

- SEBI LODR Reg 42 / NSE circulars force the mechanical adjustment timing (observable,
  rule-anchored precondition — the edge-integrity rule-watch applies).
- SEBI FY23: retail dominance in small/mid caps where these events concentrate; the
  split/bonus "affordability illusion" is a documented retail behavior in Indian market
  commentary (Zerodha Varsity corporate-actions module) and the prior Phase-2 measured its
  footprint directly: +4.30% mean 5-day short drift, 73.7% hit, positive all 4 years.
- T+1 settlement compresses the record-date/ex-date confusion window (dividend brief's
  research, retained).

## 4. Members + pooling logic ("one mechanism" is a falsifiable claim)

| Member | Events/yr (raw) | Prior evidence | Role |
|---|---|---|---|
| Splits + bonuses | ~100-130 (fires ~58% → ~55-60 trades/yr) | Phase-4 validated both eras | Core |
| Dividends, anchoring subset (interim OR yield 1.5-3%) | ~150-300 after filters | Sub-slice hints only (selection risk) | Frequency carrier — must EARN membership at Phase 2 |

Pooled target: ~200-350 trades/yr → monthly monitoring viable (the kill doc's binding
constraint, cleared). Members are validated per-member AND pooled; the sleeve ships only if
the pooling falsifier (below) survives.

## 5. Pre-registered Phase-2 grid

Construction fixed to the validated core: signal day D = ex-date, entry D+1 open, SHORT.
Dimensions: member {split, bonus, interim-div, final-div-yield-1.5-3} × hold {1, 3, 5
sessions} × ADV tier (5) × era (A5 split mandatory) × adjustment-size bucket (split ratio /
bonus ratio / dividend yield). Statistic: mean signed drift vs same-universe baseline, both
eras separately. Data per A3: `clean_daily_from5m.feather` (CA-adjusted) with the
synthetic-OHLC ex-day bar filter from the prior sanity; the consolidated_daily
adjustment-status tension (consolidation §6.5) must be spot-verified BEFORE any number is
trusted.

## 6. Falsifiers (3, pre-registered)

1. **Pooling falsifier (the load-bearing one):** if the dividend members' post-ex drift does
   not match the split/bonus drift in sign and rough shape (per-era), the "one anchoring
   mechanism" claim is FALSE — the sleeve degenerates to splits-only and dies on the same
   frequency kill. No salvage by re-filtering dividends post-hoc.
2. **Mechanism falsifier:** drift must scale with adjustment salience (bonus/split ratio,
   dividend yield within band) and concentrate in low/mid-ADV retail-heavy tiers. Flat
   drift across salience = generic post-event beta, not anchoring — kill.
3. **Data-integrity falsifier:** ex-day bars are exactly where adjusted feathers emit
   synthetic OHLC. If the drift shrinks materially on the verified-clean subset (real-bar
   entries only), the edge is an adjustment artifact — kill. (The prior Phase-4 already
   filtered synthetic bars; this must be preserved and audited.)

## 7. Data prerequisites (chores, start on approval)

1. Materialize `data/corporate_actions/split_bonus_events.parquet` via the EXISTING
   production fetcher `tools/corporate_actions/fetch_split_bonus.py` (currently only a
   `_tmp_` parquet ending 2026-04-30 exists) + top-up through July 2026.
2. Dividend re-scrape through July 2026 (`tools/dividend_ex_date/fetch_dividends.py`;
   parquet ends 2026-04-30; record_date 23% null in 2023-24 — document, don't repair).
3. Fresh-pool feasibility per A5: dividends fine (~350-450 events accrued May-Jul);
   splits/bonuses thin (~25-35) — the sleeve's fresh-pool one-shot pools members, and
   splits-only conclusions there will be flagged low-n.

## 8. Adjacent setups + side-findings

- SHORT event-driven: orthogonal to the capitulation cluster (A4) and to xsec_momentum;
  complements the book's missing short/event regime exposure.
- Variant B (5d CNC short) requires SLB infra that does not exist — variant A (MIS
  intraday-D+1) is the shippable leg; B is measured for evidence only.
- **Side-finding for the parity file (independent of this brief):** `close_dn_overnight_long`
  computes prior_day_return from live unadjusted prices but backtests on back-adjusted
  feathers → ex-date behavior differs live-vs-backtest (spurious −50% prior_ret on live
  bonus ex-dates blocks fires; backtest sees adjusted continuity). Low severity (blocks,
  never false-fires) but belongs in `analysis/backtest_findings.md`; this sleeve's event
  calendar can double as close_dn's ex-date annotation input.

## 9. A1/A2 compliance

Development = Discovery 2023-24 + demoted 2025-Apr-2026 with mandatory A5 era-split;
freeze commit before ANY statistic on 2026-05+; fresh-pool one-shot + paper decisive;
every demoted/fresh evaluation → `docs/experiment_ledger.jsonl` line. The prior
evaluations (split/bonus Phase-4 2026-05-24, dividend fade 2026-05-08) touched
Discovery/OOS/HO pre-ledger and are covered by the M≥200 baseline.
