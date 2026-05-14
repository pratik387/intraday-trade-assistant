# Retired Trading Setups — Evidence & Reasoning

This document is the **single source of truth** for setups that were prototyped, (believed-)validated, then retired. It exists so we don't burn weeks re-implementing dead theses.

**Hard rule: do NOT re-implement any setup listed here without:**

1. **Reading the retirement evidence below in full** — including the specific failure mode that killed it.
2. **Writing a corrected sanity script** that explicitly addresses that failure mode (e.g., no look-ahead bias on intraday aggregates, real-time regime classification, exit modeling that doesn't double-count favorable fills).
3. **Running new Discovery + OOS + Holdout passes** that meet ALL gates simultaneously (PF ≥ 1.10 net of full Indian fee stack at MIS-leveraged qty, n ≥ 30 per period, WR delta vs OOS ≤ 10pp, Sharpe > 0).
4. **Producing a falsification analysis** showing what would have to be true for the corrected setup to ship — and confirming that condition is testable on independent data.

If any of those four are skipped, the setup stays retired regardless of new PF.

**Active setups at the time of writing (2026-05-14):** `gap_fade_short`, `circuit_t1_fade_short`, `delivery_pct_anomaly_short`. All three carry the load. The 3-setup portfolio produces NET +Rs.1.13M FINAL after tax on 2yr Discovery (OCI run `20260514-002008_full`).

---

## Common failure modes (read first)

These patterns killed multiple setups. Watch for them when designing or validating a new candidate:

1. **Look-ahead bias on intraday aggregates.** Computing `day_high` / `day_low` / `day_vwap` as `max()`/`min()`/`mean()` over the *entire* trading day at signal time. A SHORT trade entering at 13:00 cannot legitimately use the 14:30 high to place its SL. Sanity scripts that aggregate this way produce SL distances 3-5× wider than production — and the SL stops never fire → PF gets inflated by avoiding all the losses that would have happened in real trading.

2. **wide_open OCI capture passed off as validation.** OCI captures with `wide_open: true` bypass cell filters (active windows, ADV bands, gap bands, cap segments). Numbers from those captures are NOT the production setup. They quantify the raw signal universe; the validated setup is a subset cell within that universe. Reporting wide-open PF as if it were validated PF is the most common silent failure.

3. **Cell-mining without OOS lock.** Finding the best `cap_segment × ADV-band × time-window × gap-bucket` cell on Discovery data and shipping it without OOS confirmation. The "best cell" is almost always overfit. Required: pre-register the cell criteria, then test on untouched OOS.

4. **Regime classifier non-reproducibility.** Cell-locked to `regime=trend_down` but the regime label was computed with a one-off helper that doesn't match production's per-symbol classifier. Cells that look profitable under one regime definition disappear under another. Always validate with the production classifier (`services/regime_classifier.py:classify_per_symbol` or equivalent), not an ad-hoc Discovery-only label.

5. **Holdout n below floor.** A setup that "passes" Holdout PF but has n < 30 (Phase B3 floor) is not validated; it has 1-2 lucky trades. The PF on n=11 is statistical noise.

6. **WR delta vs OOS.** Required check: |WR_Holdout - WR_OOS| ≤ 10pp. Violations mean the trade-quality distribution shifted — likely overfit cell, not stationary edge.

7. **MIS-leveraged fee math.** Fees scale with `qty × mis_leverage` (5× for MIS-eligible names), not base qty. Discovery PFs that were computed on base-qty fees are over-stated by Rs.650K+ on 2yr aggregates. See commit 271a149.

8. **Target-anchor stripping in plan dict.** Detector emits `target_anchor_type="r_multiple"` on the TradePlan, but the dict re-builder in `services/screener_live.py` strips the field — executor sees a missing key and defaults to `"structural"`, which keeps detect-time T1/T2 instead of recomputing from actual entry. Always verify `R_MULT_RECALCULATED` log lines fire for r-multiple detectors. See commits ee7e0a3 + 2c32a03.

---

## `earnings_day_intraday_fade` — RETIRED 2026-05-14

**Retired:** 2026-05-14
**Predecessor spec:** `specs/2026-05-06-sub-project-9-brief-earnings_day_intraday_fade.md` (since archived)
**Last commit with detector code:** (this commit's parent)

### Original thesis

Indian post-earnings-announcement intraday retail-FOMO fade. SEBI LODR Reg. 30 forces a 30-minute disclosure window → BMO (07:30-09:00) or AMC (16:00-19:00) announcement times → 09:15 gap → retail FOMO extends in first hour → institutional desks fade the 11:00-14:30 mid-session leg. Bidirectional fade of BMO/AMC earnings-gap.

### Universe + filters

- MIS-eligible + ADV ≥ Rs.10 cr/day, large_cap + mid_cap
- T-1 earnings announcement of class BMO or AMC (from `data/earnings_calendar/earnings_events.parquet`)
- 09:15 gap from PDC in [0.5%, 6%] (either direction)
- Entry window 10:30-15:00 IST
- Time stop 15:10 IST

### Claimed validation (pre-retire)

Per config `_status_2026_05_12`:

- **Discovery 2yr:** PF 1.64, n=1,569
- **OOS Jan-Sep 2025:** PF 1.53, n=510
- **Holdout Oct'25-Apr'26:** PF 1.25, n=298
- Locked R-multiples T1=1.0R / T2=1.0R / SL=2.5x from 3D sweep

### Why retired (the actual failure)

**The sanity script has look-ahead bias on SL placement.** `tools/sub9_research/sanity_earnings_day_intraday_fade.py:250-253` (since deleted) aggregated `day_high` / `day_low` over the **entire trading day**:

```python
day_meta = df.groupby(["symbol", "d"]).agg(
    day_high=("high", "max"),  # MAX over ALL DAY'S BARS (09:15 to 15:30)
    day_low =("low",  "min"),
)
```

A SHORT trade entering at 13:00 had its SL computed as `day_high * (1 + buffer)` — where `day_high` was the maximum high across the entire session including bars **after** the entry. Result: SL placed above the day's actual maximum → price by definition could not reach it → **0% stop hit rate in sanity** vs **4.7% in production (28 of 592 trades, avg -Rs.4,868)**.

**Same trade, sanity vs production fix evidence:**

| | sanity (T1=T2=1.0R/SL=2.5x cell) | production OCI Discovery 2yr |
|---|---|---|
| n | 413 | 592 |
| time_stop | 98.0% | 92.2% |
| T2 hit | 1.9% | 3.0% |
| **hard_sl** | **0.0%** | **4.7%** |
| total stop drag | 0 | **−Rs.136K** |
| **PF** | **1.306** (inflated) | **0.782** (real) |
| **NET** | +Rs.9,454 (inflated) | **−Rs.97K** (real) |

The "validated" PF chain (1.64 → 1.53 → 1.25) is invalid. Same look-ahead bug propagated through Discovery, OOS, and Holdout.

### Symptom-level evidence in OCI

Production trades that DID fire across 2yr Discovery (`20260514-002008_full`):

- Side × cap_segment breakdown shows all 4 quadrants negative or near-zero
- 92% time-stop rate at near-zero PnL
- 4.7% stops at avg −Rs.4,868 each = −Rs.136K total drag
- 3.0% T2 hits at avg +Rs.4,920 each = +Rs.88K total
- Net: −Rs.97K NET, PF 0.782

### Conditions for revival (if anyone attempts it)

1. Re-write the sanity to use only bars *up to and including* the entry bar when computing `day_high` / `day_low`. The detector already does this correctly (`structures/earnings_day_intraday_fade_structure.py:300` before deletion); the sanity must match.
2. Re-run the 3D sweep on the corrected sanity. If best PF at any (T1, T2, SL) combo is below 1.10 on Discovery, retire permanently.
3. If a sweep cell passes 1.10 on Discovery, propagate the same corrected sanity to OOS + Holdout. Both must pass independently.
4. Confirm the cell does not rely on `target_anchor_type` semantics that the production plan dict can strip — exercise the full screener_live → executor path in a smoke before claiming production parity.

### Code removed in this cleanup

- `structures/earnings_day_intraday_fade_structure.py`
- `tests/structures/test_earnings_day_intraday_fade.py`
- `tools/sub9_research/sanity_earnings_day_intraday_fade.py`
- All 30 `reports/sub9_sanity/earnings_day_intraday_fade_*` sweep result dirs
- `setups.earnings_day_intraday_fade` block in `config/configuration.json`
- Registry entries in `structures/main_detector.py` + `services/plan_orchestrator.py`
- Universe entries in `services/setup_universe.py`
- Earnings-calendar download in `oci/docker/entrypoint.py` (was the only consumer)

---

## `capitulation_long_morning` — RETIRED 2026-05-12

**Retired:** 2026-05-12
**Predecessor spec:** `specs/2026-05-07-sub-project-9-brief-capitulation_long_morning.md` (archived)

### Original thesis

LONG-side mirror of `gap_fade_short`. 09:15 gap-down 1.5-8% on no-news, confirmed by 09:25-10:00 5m bar with `lower_wick/body ≥ 0.5` + `body ≤ 30%` + green + no fresh low, faded back toward PDC. Retail panic-capitulation under stop/margin pressure exhausts in 5-10 minutes; institutional flow absorbs and reverts.

### Universe + filters

- MIS-eligible, mid_cap, ADV 10-30 Cr (cell-locked from round-6 cell-mining)
- 09:15 gap-down 1.5-8%
- 5-day no-news filter (no earnings within ±2 days; no corporate announcements in prior 5 trading days)
- Exhaustion candle 09:25-10:00
- NIFTY regime = `trend_down` (cell-locked)

### Claimed validation (pre-retire)

- **Discovery aggregate:** PF 0.813, n=7,471 — FAILED on aggregate
- **Discovery cell** (trend_down × mid_cap × liq 10-30cr): PF 1.238, n=443 — passed gates
- Shipped to OCI for Phase-1 capture on the cell

### Why retired (the actual failure)

Per config `_status_2026_05_12_RETIRED`:

> "Setup retired permanently. No fix path identified; the gap-down + reversal mechanic is not asymmetric enough to overcome friction in 2025-2026 conditions."

**Two independent failures:**

1. **Holdout with cell + reproducible regime classifier:** n=174, PF **0.631** — catastrophic. The cell-locked Discovery PF 1.238 was a regime-classifier artifact. When Holdout used the production per-symbol `close<SMA20 T-1` classifier (not the ad-hoc Discovery label), the trend_down filter selected a different set of days and the cell collapsed.
2. **Own sanity script aggregate:** PF 0.813 Discovery (n=7,471), PF 0.899 Holdout (n=3,035) — both fail the Phase B3 floor of 1.10 on aggregate. The cell was a single thin slice that didn't generalize.

This was the canonical example of **Common Failure Mode #3 (cell-mining without OOS lock)** + **#4 (regime classifier non-reproducibility)**. The trend_down regime was computed on Discovery using one method, then reproduced differently in Holdout — different days qualified, different trades fired, the apparent edge evaporated.

### Conditions for revival

Effectively none under the gap-down mechanic itself. The thesis (gap-down panic exhaustion in mid_cap, faded LONG toward PDC) is symmetric in form to gap_fade_short but **asymmetric in participant identity**: gap-down flows are more news-driven than gap-up flows, and the 5-day no-news filter wasn't load-bearing enough to compensate. Anyone reviving this owes:

1. A working real-time news filter (the 5-day prior-announcement check + earnings-calendar exclusion was implemented but didn't separate news-driven gap-downs from sentiment-driven ones).
2. A regime classifier locked to production semantics (per-symbol, not aggregate index regime).
3. Falsification: pre-register a cell, test on OOS, retire if Holdout WR delta > 10pp from OOS.

### Code removed in this cleanup

- `structures/capitulation_long_morning_structure.py`
- `tests/structures/test_capitulation_long_morning_structure.py`
- `tools/sub9_research/sanity_capitulation_long_morning.py`
- `setups.capitulation_long_morning` block in `config/configuration.json`
- Registry entries in `structures/main_detector.py` + `services/plan_orchestrator.py`

---

## `expiry_pin_strike_reversal` — RETIRED 2026-05-12

**Retired:** 2026-05-12 (with full code purge 2026-05-14)
**Predecessor spec:** `specs/2026-04-29-expiry_pin_strike_reversal-plan.md` (archived)

### Original thesis

NSE F&O weekly+monthly expiry sessions exhibit a "pin magnet" — market-makers minimize gamma exposure by pulling spot toward the highest-OI strike post 13:30 IST. Detector fires only on expiry days, only on NIFTY top-10 heavyweights, only when spot ≥ 0.3% from pin (room for the magnet pull), and only after RSI extreme reading earlier in session.

Sources cited: Wright Research, ICFM, OptionX, PL Capital.

### Universe + filters

- NIFTY heavyweights from `assets/nifty_heavyweights.csv` (top ~20)
- large_cap only
- Active window 13:30-15:15
- min 30 bars of session data
- NIFTY spot within 0.3% of a strike
- RSI extreme (>70 or <30) for reversal signal
- Weekly Thursday or monthly expiry day

### Claimed validation (pre-retire)

**None.** Per config `_status_2026_05_12_RETIRED`:

> "Shipped without any validation of the actual signal logic."

The OI data pipeline (`tools/option_chain/`) was built first. The detector was wired second. Sanity was deferred pending OI-data backfill. OCI captures were run in `wide_open: true` mode — which bypassed the RSI-decay confirmation, heavyweight universe filter, AND cap segment guard.

### Why retired (the actual failure)

Per config `_status_2026_05_12_RETIRED`:

1. **wide_open OCI captures are noise, not validation** — Common Failure Mode #2. 47k fires across 1,218 symbols in Discovery wide-open, when the real detector should fire on top-10 heavyweights only.
2. **Post-hoc heavyweight filtering on wide-open captures:** Discovery PF **0.526**, n=377. Fails both `PF ≥ 1.10` AND `n ≥ 500` gates.
3. **Cell-mining rescue attempt overfit:** found a 3D cell with Discovery PF 1.43, but it collapsed to PF 0.80 OOS / **0 SELL trades** with rank ≥ 2.0 in Holdout. Classic overfit.
4. **Production OCI Discovery (`20260514-002008_full` and prior):** 2 fires over 2yr (after honoring the production filter stack). Both losers, NET −Rs.6,130 MIS-leveraged. Sample size of 2 is statistical noise; the filter stack is genuinely too narrow to ever produce meaningful n.

### Conditions for revival

1. Write `tools/sub9_research/sanity_expiry_pin_strike_reversal.py` that exercises the FULL detector logic (RSI decay, spot-vs-pin distance, heavyweight universe, expiry day check, OI lookup) on local 2023-2024 OI parquets.
2. Confirm that with all filters honored, n ≥ 500 over 2yr Discovery. If not, the setup's filter stack is too narrow for statistical validation — redesign the universe (e.g., broaden to top-30 heavyweights, or test on bank-NIFTY constituents) before any PF claim.
3. Independent OOS + Holdout passes, each n ≥ 100.

The OI data infrastructure was preserved in commit history if revival is attempted. NOT preserved live: the detector code itself, the wide-open capture results (they were noise), or the sub8 override entry.

### Code removed in this cleanup

- `structures/expiry_pin_strike_reversal_structure.py`
- `tests/structures/test_expiry_pin_strike_reversal_structure.py`
- `setups.expiry_pin_strike_reversal` block in `config/configuration.json`
- `setups.expiry_pin_strike_reversal` entry in `config/sub8_oci_overrides.json`
- Registry entries in `structures/main_detector.py` + `services/plan_orchestrator.py`
- OI pipeline: `tools/option_chain/`, `oci/tools/upload_option_chain.py`
- `services/option_chain_loader.py`, `services/index_spot_loader.py` (only consumer was this setup)
- Option-chain download in `oci/docker/entrypoint.py`

---

## `options_vol_iv_rank_revert` — RETIRED 2026-05-07

**Retired:** 2026-05-07 (with full code purge 2026-05-14)
**Predecessor spec:** `specs/2026-05-06-sub-project-9-brief-options_vol_iv_rank_revert.md` (archived)

### Original thesis

Indian single-stock-options ATM IV-rank extreme = retail F&O option-buyer concentration. Per SEBI FY23 study, 91-93% of retail F&O traders lose money. At `iv_rank ≥ 0.85` (top 15% of 252-day window) the underlying mean-reverts down on T+0 mid-session as institutional vol-arb desks unwind. EOD IV-rank → T+0 11:00-15:00 intraday revert (same daily-stale-signal pattern as circuit_t1_fade_short).

### Universe + filters

- F&O 200 + ATM-IV-available stocks (empirically large_cap-dominant: 143/153 large, 4 mid, 0 small)
- iv_rank ≥ 0.85 (later tightened to ≥ 0.95 post-OOS)
- SHORT-only (LONG variant at iv_rank ≤ 0.20 dragged aggregate to PF 0.819)
- 11:00 entry, T+0 mid-session window
- T1=1.0R, T2=1.0R-2.0R, SL=2.5x (1% fixed structural stop)

### Validation chain (pre-retire)

Per config `_status_2026_05_07_RETIRED`:

| | n | PF | Result |
|---|---|---|---|
| **Sanity Discovery 2yr aggregate** | (mixed) | 0.843 | RETIRE — but cell-rescue triggered |
| **Sanity Discovery cell** (SHORT × iv_rank ≥ 0.85) | 448 | 1.131 | passed |
| **Discovery OCI** (`20260506-220013_full`) | 445 | 1.153 | held |
| **OOS** (`20260507-082527_full`, iv ≥ 0.85) | 83 | 0.843 | **FAILED** |
| **OOS** (iv ≥ 0.95 sub-cell, tightened) | 56 | 1.197 | passed |
| **Holdout** (`20260507-*`, iv ≥ 0.95) | **11** | 1.436 | n FAIL |

### Why retired (the actual failure)

Three of three falsification criteria triggered at Holdout:

1. **n=11 < 30-floor.** Phase B3 minimum is n ≥ 30. Eleven trades is not validation; it's a handful of lucky picks.
2. **WR delta +13.6pp > 10pp limit.** Holdout WR 63.6% vs OOS WR 50.0% — trade quality distribution shifted, classic overfit signal.
3. **Q1 2026 monthly decay observed.** 2026-02 had a single losing trade; 2026-03 PF 0.77 over 4 trades. The cell's edge was already dying in the most recent data.

This was a **methodologically clean retire** — the falsification thresholds were locked pre-experiment (see `reports/sub9_sanity/round4_iv_rank_cell_selection.md`), and all three triggered at Holdout. Retired permanently.

### Conditions for revival

The IV-rank → underlying-revert thesis itself may have merit on a different data layer (longer holding period, different universe slice, options-side trade instead of underlying-side). But re-implementation requires:

1. A working IV-rank data pipeline (the previous one in `services/iv_rank_service.py` is being deleted; it was thin wrapper).
2. n ≥ 500 over 2yr on Discovery within the candidate cell — the 0.95 cell only produced 56 OOS / 11 Holdout, which is below any reasonable statistical floor regardless of PF.
3. WR delta and PF both stable across OOS → Holdout windows.

### Code removed in this cleanup

- `structures/options_vol_iv_rank_revert_structure.py`
- `tests/structures/test_options_vol_iv_rank_revert_structure.py`
- `tools/sub9_research/sanity_options_vol_iv_rank_revert.py`
- `tools/sub9_research/cell_select_round4.py` (round-4 was this setup)
- All `reports/sub9_sanity/options_vol_iv_rank_revert_*` sweep dirs
- `setups.options_vol_iv_rank_revert` block in `config/configuration.json`
- `setups.options_vol_iv_rank_revert` entry in `config/sub8_oci_overrides.json`
- Registry entries in `structures/main_detector.py` + `services/plan_orchestrator.py`
- `services/iv_rank_service.py`, `oci/tools/upload_iv_rank.py`
- IV-rank download in `oci/docker/entrypoint.py`

---

## Earlier retirements (pre-2026-05)

The following setups were retired earlier and their code already deleted. Listed here for completeness — do not re-implement without reading the original brief and the lessons in `tasks/lessons.md`.

### From the 2026-04-15 to 2026-05-01 cleanup cycle

`mis_unwind_short`, `closing_hour_reversal`, `orb_15`, `pdh_pdl_reject`, `pdh_pdl_sweep_reclaim`, `gap_and_go_continuation`, `ema5_alert_pullback`, `camarilla_l3_reversal`, `cpr_mean_revert`, `narrow_cpr_breakout`, `vwap_first_pullback`

All source files DELETED in the commit cycle ending 2026-05-01. Common reasons: failed Phase B3 PF gates after the sub-7 / sub-8 cleanup audit, or invalidated by deeper review of the participant model.

### From the 2026-05-12 first-wave retire (per task #123)

`capitulation_long_morning` ✓ (this doc), `expiry_pin_strike_reversal` ✓ (this doc).

`circuit_t1_fade_short` was on the retire list but was subsequently re-validated via a corrected SL/target sweep (`tools/sub9_research/_circuit_t1_sl_target_sweep.py`) and is currently active at PF 1.50 in production OCI. Lesson: a "failed audit" tag is not final — a sweep that fixes the broken assumption can revive the setup.

---

## Maintenance protocol

When a new setup is retired:

1. Add a section here with the four standard fields: thesis, universe+filters, claimed validation, actual failure mode + evidence.
2. List the specific code files removed.
3. Add the "conditions for revival" — what would have to be true for someone to legitimately try this again.
4. Move the original brief from `specs/` to `specs/archived/` (or delete if low-value).
5. Update `tasks/lessons.md` with any pattern that other setups should watch for.

The point of this document is **negative knowledge** — what does NOT work and why. It is at least as valuable as the active-setup documentation.
