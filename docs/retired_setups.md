# Retired Trading Setups — Evidence & Reasoning

This document is the **single source of truth** for setups that were prototyped, (believed-)validated, then retired. It exists so we don't burn weeks re-implementing dead theses.

**Inventory:** 18 retired setups across 4 retire batches:

- **Sub-5 ICT/SMC batch** (2026-04-25) — 3 setups: cargo-culted US/forex literature
- **Sub-7 generic-pattern batch** (2026-04-25) — 5 setups: Indian-published but universal mechanics
- **Sub-8 generic-pattern batch** (2026-04-26 to 2026-05-01) — 6 setups: same root cause as sub-7
- **Sub-9 narrow-cell batch** (2026-05-07 to 2026-05-14) — 4 setups: each killed by a specific bug or threshold failure (look-ahead bias, regime non-reproducibility, never sanity-validated, Holdout n below floor)

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

## Sub-projects #5, #7, #8 — the 14-setup cargo-cult batch (2026-04 to 2026-05-01)

Eleven setups failed across sub-projects #5/#7/#8. The post-mortem in `specs/2026-05-01-sub-project-9-microstructure-first-redesign.md` is the canonical reference; the summary below is for quick lookup.

### Shared root cause

Every failure was a **cargo-cult of a universal or Indian-published pattern** rather than an Indian-microstructure asymmetry. From the sub-9 redesign spec, §2:

> "The research process was 'find an Indian-published setup → implement it'. Indian-published patterns are mostly translations of US/forex patterns, used by retail traders. **Retail usage ≠ institutional edge.** SEBI's FY23 study: 70% of cash intraday traders lose, 93% of F&O traders lose. The patterns retail uses are the patterns retail loses on."

Every survivor (`gap_fade_short`, `circuit_t1_fade_short`, `delivery_pct_anomaly_short`) exploits a **specific Indian-microstructure asymmetry** — small-cap retail FOMO at the open, single-day circuit-band post-FOMO mean reversion, NSE bhavcopy low-delivery-pump signature. Every failure is a generic pattern dressed up as Indian-specific because someone Indian published it.

The §3.2 binding rule, post-failure: a candidate must satisfy ALL of:
1. **Indian-specific** asymmetry (not a universal pattern)
2. **Identifiable participants** on each side ("retail vs institutional", not "buyers vs sellers")
3. **Persistence rationale** (regulatory / behavioral / structural)
4. **Prior evidence** from academic / regulatory / institutional sources (SEBI, NSE, Zerodha Varsity, EPAT projects — NOT Subasish Pani / TraderCarnival / MyAlgomate alone)
5. **Asymmetric direction** (long-only or short-only — two-sided setups inherit long-side losses)

A new setup that doesn't pass §3.3 brief gate is not allowed code review.

### Sub-5 ICT/SMC group — 3 setups

Source: ICT/Smart-Money-Concepts literature (24/5 forex). Failed because session-open, T+1 settlement, gap risk, circuit halts, and asymmetric short rules in Indian equity break the structural assumptions ICT was built on.

| Setup | Failure evidence |
|---|---|
| `order_block` (long + short) | Discovery NET PF 1.77 on n=71 — too thin to validate. Holdout: gauntlet drag. Retired with the broader Phase-1 cleanup. |
| `premium_zone` | Discovery NET PF 1.27, n=6,741, NET +Rs.915K — biggest single contributor in sub-5. Failed because the "discount/premium" zone math is a Fibonacci-retracement re-skin with no Indian-specific edge; gauntlet v2 post-mortem concluded the apparent edge was joint-optimization variance against the unvalidated library. |
| `vwap_lose_short` | Discovery NET PF 0.75, NET −Rs.16K — already losing at Discovery, dropped pre-OOS. |

Predecessor doc: `specs/2026-04-25-sub-project-5-gauntlet-v2-postmortem.md` (named the cargo-cult problem; specs/2026-04-25/26/05-01 sub-projects #7/#8/#9 are direct responses).

### Sub-7 generic-pattern group — 5 setups

Sub-7 was meant to be the "Indian-native redesign" of the sub-5 library. It added 1 genuine Indian-microstructure setup (`gap_fade_short`, survived) and 4 generic patterns (all failed).

| Setup | Original thesis | Failure evidence |
|---|---|---|
| `mis_unwind_short` | SEBI requires MIS positions square off by 3:20 PM → forced unwind in last 60-90 min creates asymmetric net-sell pressure (since retail is structurally net-long) | n=326 over 2yr Discovery — sample size below floor. Stop multiplier 0.8×ATR (vs source-cited 1.2-1.5×ATR for end-of-day high-vol window) — most signals stopped before the unwind began. Cap-segment filter (small/mid only) starved the sample. Thesis itself is plausible; the mechanic was wrong. The sub-9 candidate `closing_hour_reversal_short_redo` retries this thesis with different mechanic. |
| `closing_hour_reversal` | Generic closing-bar reversal pattern (no Indian-specific anchor) | Universal pattern; failed Phase-1 floor (PF ≥ 1.10). The sub-9 redo brief at `specs/2026-05-07-sub-project-9-brief-closing_hour_reversal_short_redo.md` re-ran the R-sweep on a different mechanic; the redo is in `reports/sub9_sanity/closing_hour_reversal_short_redo_*` but did not ship to production. |
| `narrow_cpr_breakout` | "Narrow Central Pivot Range expands → directional breakout follows" (CPR pivot trading is universal — Frank Ochoa's pattern, taught in Zerodha Varsity for Indian audiences) | Pivot indicator is universal; the Indian-published version is a translation. Failed Phase-1 with `cpr_mean_revert`. |
| `vwap_first_pullback` | "Pull back to VWAP after a trend leg → continuation" — generic VWAP pullback pattern | Universal pattern; failed Phase-1 PF floor. |
| `cpr_mean_revert` | Lunch lull (11:30-13:30) has low volume → range trading dominates → stocks mean-revert to CPR. Zerodha Varsity-documented | PF 0.64, n=247/day Discovery. Three root causes per sub-8 retrospective (`specs/2026-04-26-sub-project-8-indian-native-setups-extended-design.md:18-19`): (a) Stop multiplier 0.5×ATR vs source-cited 0.75-1.0×ATR (wicked out before mean reversion). (b) Single-shot exit at CPR midpoint — no T1 partial. (c) Wrong universe: ran on 1000+ symbols incl. micro-caps; CPR is a Bank Nifty / Nifty 50 heavyweight tool per Frank Ochoa. The fix path (right universe + tiered exits + correct stop) was deferred — the setup never shipped a v2. |

Delete commits: `da7b978` (mis_unwind_short + closing_hour_reversal), `1579297` (cpr_mean_revert + mis_unwind_short structures), `2251396` (vestigial sub-7/8 cleanup).

### Sub-8 generic-pattern group — 6 setups

Sub-8 was meant to be the "extended" Indian-native library after sub-7's mixed results. Despite explicit binding rules ("every parameter Indian-source-cited", "no universal patterns as PRIMARY thesis"), all six new setups failed.

| Setup | Origin / thesis | Failure |
|---|---|---|
| `orb_15` | Opening Range Breakout — first 15-minute range sets the day's institutional intent; full-bar close outside the range = breakout. Source: Toby Crabel (US), Algotest, In-the-Money by Zerodha | Universal pattern. Failed Phase-1 PF floor. Crabel's ORB has US-specific microstructure assumptions; the Indian translation didn't carry. |
| `pdh_pdl_reject` | Price rejects at Prior Day High / Prior Day Low → continuation in the opposite direction | Universal level-rejection pattern. Failed Phase-1. |
| `pdh_pdl_sweep_reclaim` | ICT-style sweep+reclaim: price takes out PDH/PDL liquidity then reclaims the level → reversal | Universal ICT pattern (same family as sub-5's `order_block`). Failed Phase-1. |
| `gap_and_go_continuation` | LONG bias gap continuation: gap-up + first 15-min momentum → ride to PDC or VWAP-based target | This is **the losing flow per SEBI FY23**. Long-bias retail intraday flow systematically loses. The asymmetric-direction rule (binding rule §3.2 #5) was violated. Predictably failed. |
| `ema5_alert_pullback` | Subasish Pani's 5-EMA pullback pattern — generic trend-pullback indicator | Indian-published, not Indian-microstructure-edged. Subasish Pani is a retail educator; this is a pattern retail uses (and loses on). Failed Phase-1. |
| `camarilla_l3_reversal` | Camarilla pivots level 3 reversal — universal pivot indicator | Universal pattern; not Indian-specific. Failed Phase-1. |

Delete commit: `9726fc7` ("chore(sub9): delete 6 failed sub-7/sub-8 detectors and all related code"). Predecessor task #57.

Specs preserved (do not re-implement without reading these first):
- `specs/2026-04-25-sub-project-7-indian-native-setups-design.md` — original sub-7 design
- `specs/2026-04-25-sub-project-7-indian-native-setups-plan.md` — sub-7 plan
- `specs/2026-04-26-sub-project-8-indian-native-setups-extended-design.md` — sub-8 design with binding rules
- `specs/2026-04-26-sub-project-8-indian-native-setups-extended-plan.md` — sub-8 plan
- `specs/2026-05-01-sub-project-9-microstructure-first-redesign.md` — the canonical post-mortem
- `specs/2026-04-25-sub-project-5-gauntlet-v2-postmortem.md` — sub-5 ICT/SMC post-mortem

### Conditions for revival (sub-5/7/8 group)

These eleven setups have a **group-level retire reason** that's harder to overturn than the per-setup ones in the 2026-05-14 batch. The sub-9 spec's §3.2 binding rules apply: any revival must satisfy all five asymmetry attributes. In practice that means:

1. **Don't revive the pattern; revive the asymmetry.** `mis_unwind_short` is the canonical example — the SEBI MIS-square-off thesis is real, but the mechanic (short above-VWAP names with negative 3-bar momentum at 15:00) was wrong. A revival has to come from the asymmetry side: what's the right entry signal for forced unwind flow? (E.g., institutional algos initiate VWAP liquidations starting 14:45 — if you can detect those algos' price impact, you have an edge. The generic "above VWAP with weakening momentum" doesn't detect them.)
2. **Don't revive universal patterns at all.** ORB-15, PDH/PDL reject, PDH/PDL sweep, ema5 pullback, camarilla L3, vwap_first_pullback, narrow_cpr_breakout, cpr_mean_revert — these are universal patterns. They are not allowed to be the PRIMARY thesis of a new setup. They can be MECHANICS used to harvest an Indian-specific asymmetry — but the asymmetry has to come first.
3. **The `circuit_t1_fade_short` precedent.** Circuit_t1 was on the retire list per task #123 but was subsequently revived via a corrected SL/target sweep and now ships at PF 1.50. That revival succeeded because the underlying asymmetry (post-FOMO mean-reversion on Indian-specific circuit-band stocks) was sound; only the SL/target geometry was broken. None of the 11 setups above have that profile — they fail on the asymmetry, not on geometry.

---

## `fno_ban_t1_fade_short` — RETIRED 2026-05-16 (pre-implementation)

**Retired:** 2026-05-16
**Predecessor spec:** Candidate #2 of `specs/2026-05-14-research-post-sebi-edges.md`
**Status:** Never implemented as a detector. Killed at falsifier stage before any sanity-script or detector code was written.

### Original thesis

Under SEBI's Nov 3, 2025 intraday FutEq OI monitoring framework, stocks can enter the F&O ban list at any of 4 random intraday MWPL snapshots. Once a stock enters ban, fresh positions are blocked — only existing-position closures allowed. This creates a one-way exit flow with no fresh buying, leading to predictable downside drift on the ban-entry day and/or T+1 session.

**Direction:** SHORT on/after ban-entry.
**Falsifier (pre-registered):** WR ≥ 55% AND median T+1 return ≤ −0.5%.

### Universe + filters

- All F&O underlyings that appear in the daily NSE F&O ban list
- Window for falsifier: 2025-10-01 → 2026-04-30 (full post-SEBI rule-change window)
- Data: `data/fno_ban_history/fno_ban_events.parquet` (EOD-resolution only; intraday endpoint not found)

### Falsifier evidence (n=19, after 2 missing-bar drops)

Path-A (intraday on ban_date, open → close):
- WR 52.6%, median **−0.46%**, mean −0.33%, PF (short view) 1.29
- Misses both thresholds (WR < 55%, median > −0.5%).

Path-B (close ban_date → close ban_date+1, literal "T+1" reading):
- WR **47.4%**, median **+0.57%**, mean +1.11%, PF (short view) **0.375**
- INVERSE direction — the literal hypothesis goes the wrong way. T+1 mean-reverts UP on news already priced in by EOD ban_date.

Path-C (full ban-window, open ban_date → close exit−1):
- WR 52.6%, median −0.18%, mean +1.35%, PF (short view) 0.586

Per-event CSV: `reports/sub9_sanity/_fno_ban_falsifier_per_event.csv`
Falsifier script: `tools/sub9_research/_fno_ban_falsifier.py`

### Why retired (the actual failure)

1. **Sample is fatally thin.** Only 21 ban-entry events across 5 symbols in the full 7-month post-SEBI window. The original brief assumed n=200+ achievable. The reality (21 EOD-resolution events) is an order of magnitude smaller.
2. **Symbol concentration is fatal.** SAMMAANCAP (10) + SAIL (6) = 14/19 events. Two-symbol risk masquerades as a "pattern."
3. **Outlier-driven mean.** KAYNES (−13.66%) drives most of Path A's negative mean. Without that one event, mean ≈ +0.30% (LONG bias, not SHORT).
4. **Path B is inverse.** The literal "T+1 fade" interpretation has mean +1.11% — i.e., the stock RECOVERS after the ban day. Whatever mechanism the brief proposed for forced selling is dominated by the next-day rebound from over-extended selling on ban_date itself.
5. **Even the closest-to-passing path (A) is below transaction-cost economics.** Median −0.46% gross vs ~0.15-0.20% round-trip MIS fees = net ~−0.25%/trade. With 19 trades over 7 months ≈ 3 trades/month, this is operationally pointless even if the gross PF held.

### Conditions for revival

1. **Intraday ban-list endpoint must be found.** The EOD-only data inherently misses the actual "first moment" the ban activates (one of 4 random intraday snapshots per SEBI Nov 3 2025 framework). The mechanism the brief describes is INTRADAY (no fresh buying within the session that the snapshot triggers), but EOD data tests next-day behavior — that's the wrong window.
2. **Sample must be ≥ 100 events.** The current 21 across 5 symbols cannot support even a basic falsifier. Need at least intraday-resolution data + a wider F&O universe + a longer window (12+ months post-SEBI).
3. **Mechanism reformulation.** The data suggests the OPPOSITE of the original thesis on Path B (T+1 is a LONG). If revived, the candidate should be tested as a SHORT-ON-BAN-DAY-INTRADAY only, with the falsifier raised to PF ≥ 1.40 to compensate for transaction costs at this small return size.
4. **Symbol diversification.** ≤ 30% of events from any single symbol. The current 53% SAMMAANCAP+SAIL share would have to drop.

If any of those four are missing, the setup stays retired.

### Files of record

- `tools/sub9_research/_fno_ban_falsifier.py` — falsifier script (retained for future revival reference)
- `reports/sub9_sanity/_fno_ban_falsifier_per_event.csv` — per-event returns table
- `data/fno_ban_history/fno_ban_events.parquet` — source data (retained — still relevant to other candidates that consume F&O ban events as a covariate, not as a primary signal)
- No detector code, no sanity script, no config keys — nothing to remove.

---

## `pre_open_auction_direction_follow` — RETIRED 2026-05-16 (pre-implementation)

**Retired:** 2026-05-16
**Predecessor spec:** Candidate #4 of `specs/2026-05-14-research-post-sebi-edges.md`
**Status:** Never implemented. Killed at data-recon — the required upstream data feed is unavailable.

### Original thesis

Feb 1, 2025 SEBI rule eliminated leverage on long options. Retail option-buying flow that used to position pre-open is now gone or much smaller. Pre-open auction depth and pattern should have shifted predictably — opening gap behavior in retail-favorite names should be more reliable post-rule.

**Direction:** Both LONG and SHORT, based on first-15-minute follow-through after pre-open auction direction in NIFTY-50 names.
**Falsifier (pre-registered):** WR ≥ 60% on next-60-min prediction from (pre-open direction + first 5m bar direction).

### Data state at recon (2026-05-16)

1. `data/pre_open_auction/pre_open_events.parquet` — **does not exist**. Backfill (`tools/pre_open_auction/fetch_pre_open.py`) attempted 25/818 sessions, all returned 404 across legacy / 2024 / nsearchives URL forms.
2. Scraper docstring explicitly states: *"NSE has retired the public pre-open archive endpoint. None of the documented URLs resolve for historical dates."* This was already documented in May 2026 when the scraper was written; the candidate brief in `specs/2026-05-14-research-post-sebi-edges.md` overlooked that note.
3. `cache/ohlcv_archive/RELIANCE.NS/*_1minutes.feather` — **0 bars before 09:15** across 809 sessions. The pre-open call-auction window (09:00-09:15) is not in our intraday archive either. Same is true for all symbols (the broker pipeline starts ingesting at 09:15 market open).
4. Live API mode (`--use-live-api`) exists in the scraper but only snapshots T-0 (current day), which cannot build a historical dataset.

### Why retired (the actual failure)

The brief's hypothesis falsifier ("pre-open direction + first 5m bar direction predicts next-60min with WR ≥ 60%") cannot be tested — we have **zero historical observations of pre-open IEP, imbalance quantity, or pre-09:15 bars**. The brief's "~7000 events" sample-availability estimate assumed the scrape would succeed; it did not, and the upstream endpoint has been retired.

Unlike #2 (where 21 thin events still let us run a falsifier), this candidate has no path to even a single data point. There is nothing to score.

### Conditions for revival

1. **Wire up the live-API snapshotter as a daily cron** at 09:14:30 IST. After 60-90 trading days (~3-4 months), a forward-only sample exists. The current `fetch_pre_open.py --use-live-api` flag is the entry point.
2. **Once n ≥ 200 sessions × 50 NIFTY-50 stocks ≈ 10K events**, re-run the falsifier on forward-only data. If WR ≥ 60% on the (pre-open direction + first 5m bar) → next-60-min signal, proceed to sanity-script and Discovery/OOS phases.
3. **Recognize this is now a 2026-Q4+ project**, not a 2026-Q2 project. Earliest realistic ship date is end-2026 if the cron starts today.
4. **Even with data, the post-Feb-2025 option-rule-change thesis is the WEAKEST mechanism in the original candidate list** — it's a "should have shifted" hypothesis, not a "must have shifted" one. Before investing the cron-collection time, write a stronger mechanism statement OR explicitly accept that this is an exploratory data-mining project.

If any of those four are missing, the setup stays retired.

### Files of record

- `tools/pre_open_auction/fetch_pre_open.py` — scraper (kept, also useful for future forward-only collection)
- `tools/pre_open_auction/verify_pre_open.py` — verification stub
- `data/pre_open_auction/_backfill.log` — evidence of failed backfill attempt
- No detector code, no sanity script, no config keys — nothing to remove.

---

---

## `block_deal_accumulation` (C-01) — RETIRED 2026-05-16 (at sanity)

**Retired:** 2026-05-16
**Predecessor spec:** CANDIDATE-01 of `specs/2026-05-16-new-setup-candidates.md`
**Status:** Sanity script run on all 3 windows (Discovery / OOS / Holdout) in both directions (LONG and SHORT). Mechanism doesn't generalize. No detector code written. No production wiring.

### Original thesis

When a confirmed NSE institutional BUY block deal (>= Rs. 5 cr) prints at price X on day T, that price becomes a "defended" level on subsequent retests within the same session. The institution that took size at X is unlikely to let price materially cross X (they'll defend or add at that level). Pro-Indian-trader interpretation: morning blocks (08:45-09:00 IST) are reported by ~10:00, actionable from 10:30 IST onwards with 30 min buffer for confirmation. The retest from above (price went up after block, came back to X) is the entry trigger.

### Universe + filters

- NSE-listed, MIS-enabled
- Block trade_value_cr >= 5.0 (SEBI institutional-block minimum)
- BUY rows (one row per block)
- 5m bars cached for the symbol
- Discovery: 2023-01-01 to 2024-12-31
- OOS: 2025-01-01 to 2025-09-30
- Holdout: 2025-10-01 to 2026-04-30 (block deal data backfilled 2026-05-16 specifically for this evaluation)

### Trade simulation

- Disclosure cutoff: 10:30 IST
- AWAY-move: bar high after 10:30 must reach >= block_price * 1.005
- Entry: first bar after AWAY-move where bar.low <= block_price (limit fill at level)
- SL: 0.5% from entry (above for short, below for long)
- T1: 1R, T2: 2R
- Time stop: 15:10
- Risk per trade: Rs.1000

### Cross-window results (both directions tested)

| Window | LONG direction (candidate spec) | SHORT direction (inverse-edge test) |
|---|---|---|
| Discovery (n=236) | PF **0.56**, WR 29.2%, NET -Rs.73K | PF **1.35**, WR 51.7%, NET +Rs.42K |
| OOS (n=217) | PF 0.80, WR 38.7%, NET -Rs.22K | PF **0.29**, WR 19.8%, NET -Rs.128K |
| Holdout (n=80) | PF 1.12, WR 43.8%, NET +Rs.5K | PF 0.31, WR 26.2%, NET -Rs.38K |

Trade CSVs preserved in `reports/sub9_sanity/_block_deal_accumulation_{long,short}_trades_{window}.csv`.

### Why retired (the actual failure)

**The mechanism doesn't generalize across regimes.** Both directions fail at least one window:

1. **LONG direction (candidate spec):** Fails Discovery (PF 0.56, WR 29.2%) — classic inverse-edge signature where MAE median (1.77R) significantly exceeded MFE median (0.86R). Holdout PF 1.12 looks marginally OK but n=80 is below the 200-floor for any statistical confidence, AND the cell breakdown shows large_cap losing while small/mid win — opposite to where institutional block deals concentrate (large_cap dominates the sample by 60%).

2. **SHORT direction (inverse test):** Looked promising on Discovery (PF 1.35) but **catastrophically collapsed in both OOS (PF 0.29) and Holdout (PF 0.31)**. 30+ percentage point WR drop (51.7% -> 19.8%). This is a clear sign that the Discovery edge was a regime artifact of 2023-24 (FII-outflow-dominated period), not a stationary signal.

3. **Direction-flipping across regimes is the signature of an unstable mechanism**, not a real edge. A stationary edge should hold the same direction (positive or negative) across windows even if magnitude varies.

### Mechanism-level diagnosis (what's actually happening)

The candidate spec's claim that "institutions defend block_price on retest" is incorrect for intraday cash equity. The block-deal disclosure event is NOT sufficient signal to predict intraday price behavior. Likely reasons:

1. **Block-deal participants are often longer-term holders** — they don't actively defend intraday positions. They took size for portfolio rebalancing or strategic accumulation; they don't sit at the level adding more.
2. **Many block deals are portfolio-rebalancing** between AMC schemes (e.g., HDFC Liquid Fund -> HDFC Mid Cap Fund) — no directional positioning at all; level has no defender.
3. **By the time disclosure is public (10:00+), the level has already been digested** by HFT and market makers within minutes of execution. Pro-trader 10:30 entry is too late for any same-session edge.
4. **The ICT-style "institutional level defense" thesis is mostly retail education**, not market mechanics. Institutions don't behave like the folk theory suggests — they're not constantly active at every price they touched.

### Conditions for revival

If revived in the future, would need to address ALL of the following:

1. **Filter to TRUE STRATEGIC accumulation blocks** (not portfolio rebalancing). This requires client_name enrichment — classify each block as "long-only mutual fund acquisition" vs "intra-fund transfer" vs "PMS rebalance" vs "promoter pledge release." Without this enrichment, all blocks look the same in the data, but only a small fraction are mechanistically defended.
2. **Test T+1 retest, not same-session.** Same-session edge appears arbed-away within minutes of disclosure. Strategic positions may show T+1 defense (next session after institution has settled in).
3. **Find a stable direction.** Both LONG and SHORT failed in different windows. Without finding ONE direction that works across all 3 windows, this is not a tradeable signal.
4. **Cap-segment specialization may help** — small/mid_cap blocks in Holdout LONG showed PF 1.57 (mid) and 3.82 (small), but with n=18 and 9 respectively, this is below any statistical floor. Could test with a longer accumulation window.

### Files of record

- `tools/sub9_research/sanity_block_deal_accumulation_long.py` (handles both LONG and SHORT via --direction flag, despite the legacy filename)
- `reports/sub9_sanity/_block_deal_accumulation_long_trades_{discovery,oos,holdout}.csv`
- `reports/sub9_sanity/_block_deal_accumulation_short_trades_{discovery,oos,holdout}.csv`
- `data/block_deals/block_deals_events.parquet` (kept — useful for other research that may consume block deal data as covariate)
- No detector code, no test code, no config keys — nothing to remove.

### Lesson

Confirms that the inverse-edge memory rule (`feedback_inverse_edge_signature.md`) saved time here. The Discovery LONG result alone (PF 0.56, WR 29%) would have been a retire — but flipping to SHORT immediately revealed Discovery PF 1.35. Without the OOS check, this would have shipped as a SHORT setup. OOS revealed the Discovery SHORT was a regime artifact. Net: gauntlet did its job, the methodology held up, and we learn the C-01 thesis is broken in 3 windows.

---

## `volume_spike_reversal_midsession` (C-09) — RETIRED 2026-05-16 (at sanity)

**Retired:** 2026-05-16
**Predecessor spec:** CANDIDATE-09 of `specs/2026-05-16-new-setup-candidates.md`
**Status:** Sanity script tested across 2 universes (large_cap only with ADV>=500cr, and large+mid_cap with ADV>=100cr). Both fail decisively. No detector code written. No production wiring.

### Original thesis

During mid-session (10:30-13:30 IST) when morning retail FOMO has settled but MIS-unwind has not yet started, large-cap institutional VWAP-tracking algorithm flows can exhaust at intraday extremes - manifesting as a volume-spike bar with rejection wick near session VWAP. Different participant than gap_fade_short's retail-FOMO exhaustion: this should capture institutional algo flow exhaustion. Mechanism backed by HFT/microstructure literature on order flow toxicity (Easley/Lopez de Prado/O'Hara).

### Filters tested

Universe v1: large_cap, MIS-eligible, 20-day ADV >= Rs.500 cr (122 symbols)
Universe v2: large_cap + mid_cap, MIS-eligible, 20-day ADV >= Rs.100 cr (461 symbols)

Signal bar (both universes): vol_z >= 2.0 AND vol_ratio (vs 20-bar median) >= 3.0 AND body >= 0.3% of price AND wick rejection >= 50% on the move's extreme AND price within 2x ATR of VWAP. Active window 10:30-13:30 IST. R-multiple trade geometry: SL = 0.3% beyond wick, T1=1R, T2=2R, time-stop 15:10.

### Discovery results (both universes fail)

| Universe | n | PF | WR | LONG | SHORT |
|---|---|---|---|---|---|
| large_cap (122 sym) | 86 | 0.604 | 37.2% | WR 24%, PF 0.35 | WR 49%, PF 0.92 |
| large+mid (461 sym) | 737 | 0.790 | 42.3% | WR 39%, PF 0.70 | WR 44%, PF 0.84 |

Both directions lose in both universes. Aggregate PF 0.60-0.79.

### Why retired (mechanism-level diagnosis)

The candidate spec's claim - "institutional algo flow exhaustion creates mean-reversion at VWAP" - does NOT manifest in Indian intraday cash equity. The empirical pattern is the OPPOSITE: a volume-spike bar with a rejection wick in mid-session is more often a **momentum-continuation signal** (the move that created the wick continues) than a reversal signal.

This makes mechanical sense in retrospect:
1. Institutional VWAP-anchored algos execute regardless of price - they don't "exhaust" at extensions; they just keep buying/selling on schedule.
2. Mid-session volume spikes in large-caps are typically driven by news, block-deal anticipation, or sectoral rotation - all of which create CONTINUATION pressure, not exhaustion.
3. The "wick rejection" interpretation assumes a single large order met an opposing wall and bounced. In reality, intraday volume spikes are aggregated activity (many small orders), not single-block exhaustion.
4. The HFT/microstructure literature backing (Easley/Lopez de Prado on order flow toxicity) operates at tick-scale, not 5-minute-bar scale. At 5m bars, the aggregation washes out the toxicity signature.

### Comparison to gap_fade_short (which DOES work)

gap_fade_short captures retail-FOMO exhaustion in SMALL-CAPS at the OPEN (09:25-10:00 specifically). Different mechanism entirely:
- Small-cap retail FOMO is mechanically distinct from large-cap institutional algo flow
- Morning open is the highest-emotion window of the day; mid-session is the lowest
- The "exhaustion candle" with wick rejection IS the retail-FOMO signature, but only in the small-cap + open-window context

C-09 tried to generalize this pattern to a different universe + timing, but the participant mix is fundamentally different, and the mechanism doesn't carry.

### Conditions for revival

Genuinely none under the current filter geometry. A revived version would need to:
1. **Find a different anchor than wick-rejection-at-VWAP.** Wick rejection in mid-session large-caps does NOT predict reversal. A revived version would need a fundamentally different mechanic.
2. **Identify a specific participant group whose behavior creates exhaustion.** Institutional VWAP-tracking algos don't exhaust. Block-deal-anticipation flows? Sectoral-rotation flows? Need a clear participant hypothesis with regulatory or behavioral backing.
3. **Test mid_cap-only or specific high-RVOL sub-cells.** The current data may have edge in some narrow cell we didn't slice. But after Discovery showing both directions lose in both universes, the burden of proof for a revival is high.

### Files of record

- `tools/sub9_research/sanity_volume_spike_reversal.py` - sanity script (retained for future revival reference)
- `reports/sub9_sanity/_volume_spike_reversal_trades_discovery.csv` - per-trade outputs
- No detector code, no test code, no config keys - nothing to remove.

### Lesson

Reinforces the binding rule from earlier sub-9 work: **mechanism must come BEFORE filters**. The candidate spec specified the trade geometry (vol_z, wick ratio, VWAP proximity, ATR) thoroughly but couldn't articulate a SPECIFIC participant whose behavior creates intraday large-cap exhaustion. Without that participant story, the geometry was data-mining a pattern that doesn't have an underlying mechanism in this universe.

---

## `compression_fii_anchor_breakout` (C-06) — RETIRED 2026-05-16 (at data recon)

**Retired:** 2026-05-16
**Predecessor spec:** CANDIDATE-06 of `specs/2026-05-16-new-setup-candidates.md`
**Status:** Killed at data-recon stage before any sanity script was written. No data exists to evaluate the candidate.

### Original thesis

When a NIFTY-50 stock enters a tight intraday range (height < 0.7% over 20+ bars) during the 11:30-13:30 lunchtime regime, the eventual breakout direction is predictable from concurrent FII net cash flow direction. Institutional flow concentrating on one side during low-vol periods should tip the squeeze in their favor when liquidity returns post-13:30.

### Data state at recon (2026-05-16)

The candidate hinges on **intraday FII/DII net cash flow data**. We have:
- T-1 EOD FII/DII data via NSE bhavcopy (next-day-available, not real-time intraday)
- NO intraday FII/DII feed in `data/` or any current scraper

NSE publishes intraday FII/DII updates ONLY on the live website (paywalled / scraping-fragile). Moneycontrol displays "live FII activity" but it's also delayed and not exposed via clean API. We do not have this scraper built.

### Why retired (the actual failure)

The mechanism requires the directional anchor (FII flow sign) to be observable AT ENTRY TIME (post-13:30 IST). Without intraday FII data, we cannot make that observation. The candidate's entire premise collapses without the anchor.

Like `pre_open_auction_direction_follow` (retired 2026-05-16) and `fno_ban_t1_fade_short` (retired 2026-05-16), this candidate dies at the data-availability stage. The pre-registered candidate spec assumed the data was attainable; data-recon revealed otherwise.

### Conditions for revival

1. **Wire up an intraday FII scraper** — possible sources: Moneycontrol's live-activity page (HTML scraping with rate limits), Bloomberg/Reuters terminal subscriptions, NSE intraday bhavcopy if it exists. Note: latency is the killer here — even if a scraper works, FII updates published 30+ minutes after execution may be too late to anchor a 13:30 entry decision.
2. **Test on forward-collected data once available** — minimum 60-90 trading days of paired (compression-event, concurrent-FII-direction) observations.
3. **Independent validation** that NIFTY-50 lunchtime compression events have meaningful predictive value at all. The compression-then-expansion mechanism is academically real (volatility clustering); the FII-direction anchor is theoretical — would need to test compression events even without the FII anchor to establish a baseline.

### Files of record

- No sanity script, no detector code, no test code, no config keys.
- The candidate spec in `specs/2026-05-16-new-setup-candidates.md` is preserved for revival reference.

### Why compression-alone is not a valid fallback

A natural question: "Without the FII anchor, the candidate is still a compression-then-expansion trade. Why not test that on its own?"

Answer: That mechanism was already tested as `narrow_cpr_breakout` (sub-7, retired) and failed Phase-1. Compression-alone without an external directional anchor is direction-blind: the volatility-clustering mechanism (low-vol periods followed by high-vol periods) is academically real, but predicting WHICH direction the expansion takes is essentially 50/50 without an external signal. Re-running compression-alone in this candidate would be a re-test of a known retire, not new research.

The FII-anchor was THE differentiating feature of C-06. Without it, there's no novel research path to pursue. Hence retire-at-recon is correct.

### Lesson

Three of the candidates that surfaced from the old_main review (#6 F&O ban T+1, #4 pre-open auction, C-06 here) all died at data recon because their proposed external-data anchors were not actually available in the system. The lesson encoded in `memory/feedback_data_availability_pre_check.md` (always print df.shape + symbol concentration before scoring a brief) needs to extend to: **always verify the required external data feed exists in `data/` or has a working scraper, before assigning any confidence rating to a candidate.**

---

## `sectoral_lag_catchup_long` (C-05) — RETIRED 2026-05-16 (at sanity)

**Retired:** 2026-05-16
**Predecessor spec:** CANDIDATE-05 of `specs/2026-05-16-new-setup-candidates.md`
**Status:** Full 6-step LPGD cycle completed (sanity Discovery / OOS / Holdout + inverse-edge + per-sector cell analysis). No salvageable cell holds across all 3 windows. No detector code written, no production wiring.

### Original thesis

When a NIFTY sectoral index (NIFTY Bank, NIFTY IT, NIFTY Auto, etc.) breaks its prior-day-high with volume conviction, the heaviest-weighted constituents that have NOT yet broken their own PDH have a documented tendency to catch up within 30-60 min. Index-level conviction drags lagging constituents on positioning rebalance flow.

### Implementation details

- Sectoral universe: 11 NIFTY sectoral lists (Bank, IT, Auto, Pharma, FMCG, Metal, Energy, Finance, Realty, Oilgas, PSUBank). 163 unique constituents total.
- Sectoral proxy: equal-weighted aggregate of constituent closes (normalized to first-bar-of-day = 1.0). NOT actual NSE market-cap-weighted index.
- Active window: 10:30 - 14:30 IST
- Signal: sectoral aggregate breaks morning_high (max sectoral level by 10:00) by >= 0.05% AND constituent is BELOW own PDH at that moment.
- LONG entry on lagging constituent.
- SL: 0.3% below signal-bar low; T1=constituent PDH (structural); T2=PDH + 0.5*ATR.

### 3-window aggregate results

| Window | n | PF | WR | NET |
|---|---|---|---|---|
| Discovery (2023-24) | 33,174 | 0.96 | 40.1% | -Rs. 864K |
| OOS (2025 Q1-Q3) | 12,053 | **0.83** | 38.7% | -Rs.1.27M |
| Holdout (2025-10 to 26-04) | 8,427 | **1.10** | 43.4% | +Rs. 477K |

### Inverse-edge test

- Original LONG: PF 0.96, WR 40.1% (Discovery)
- INVERSE-SHORT: PF 0.50, WR 18.0% — CATASTROPHIC, original direction is correct

### Per-sector cross-window cell stability (Discovery-passing sectors)

| Sector | Disc | OOS | Hold | Cross-window |
|---|---|---|---|---|
| NIFTY_PSUBANK | 1.10 | **0.67** | 1.18 | OOS crashes |
| NIFTY_AUTO | 1.03 | 0.94 | 0.95 | drifts down |
| NIFTY_METAL | 1.03 | **0.72** | 1.21 | OOS crashes then recovers |
| NIFTY_REALTY | 1.04 | 0.91 | 1.57 | OOS dips then recovers |
| NIFTY_IT | 0.95 | 1.16 | 0.92 | Only OOS-positive (anti-pattern) |

**No sector holds PF >= 1.00 in all 3 windows.** Holdout shows broad recovery (8/11 sectors >= 1.00) but Discovery + OOS aggregate both fail.

### Why retired

1. **Aggregate fails Discovery (0.96) AND OOS (0.83).** Even with Holdout pass (1.10), 2/3 windows below floor is a clean fail.
2. **Per-sector cells don't generalize.** PSUBANK and METAL CRASHED in OOS (PF 0.67, 0.72) then recovered in Holdout (1.18, 1.21). This is regime-dependent noise, not stationary edge.
3. **Holdout-only positive pattern is suspicious.** Holdout window includes post-SEBI Oct 2025 + war months (Feb-Apr 2026) - regime with unusual sectoral correlations from sectoral leadership rotation, FII flow shifts, and rate-cut anticipation. The recovery is likely regime-specific, not mechanism-driven.
4. **Inverse-edge test failed.** Original direction is correct; flipping makes it worse. Edge is real-ish but unstable.

### Mechanism-level diagnosis

The intermarket-arbitrage thesis (sectoral index breaks PDH → lagging constituents catch up) IS academically real on **daily** timeframes (multiple SSRN papers document this for US markets). The failure here is the **intraday 5m-bar timeframe** AND the **equal-weighted constituent proxy** for the sectoral index.

Two issues:
1. **Equal-weighted aggregate doesn't match NSE's market-cap-weighted index calculation.** The actual sectoral index has very different intraday dynamics than my proxy. A NIFTY_BANK heavyweight (HDFC, ICICI) moves the actual index 10-30x more than a smaller constituent. The equal-weighted proxy treats all constituents equally, blurring the true index move.
2. **Intraday catch-up effects are weak.** On daily timeframes, "lagging constituents catch up" plays out over 1-3 days, not 30-60 min. The candidate spec was overly optimistic about intraday speed.

### Conditions for revival

Genuinely thin. If revived in the future, would need:
1. **Direct sectoral index intraday data** (not constituent aggregation proxy). NSE publishes intraday index values; would need a separate scraper.
2. **Market-cap-weighted aggregate at minimum** (not equal-weighted) if direct index data unavailable.
3. **Longer holding window** (intraday 60-120 min vs current 30-60 min) - daily lag-catchup takes longer than intraday.
4. **Re-evaluate after the war-regime is over.** The Holdout pattern (2025-10 to 2026-04) overlaps with regime change events; can't separate edge from regime.

If those four don't change, the setup stays retired.

### Files of record

- `tools/sub9_research/sanity_sectoral_lag_catchup.py` (retained for future revival reference)
- `reports/sub9_sanity/_sectoral_lag_catchup_trades_{discovery,oos,holdout}.csv`
- No detector code, no production wiring, no config keys.

### Lesson

Reinforces that **intermarket-arbitrage signals operate at slower timescales than intraday 5m**. Daily lag-catchup is real (academic literature confirms); intraday lag-catchup in 30-60 min is wishful thinking. Future intermarket candidates should be validated on multi-bar (e.g., 30min/60min) bars before assuming 5m granularity.

---

## Maintenance protocol

When a new setup is retired:

1. Add a section here with the four standard fields: thesis, universe+filters, claimed validation, actual failure mode + evidence.
2. List the specific code files removed.
3. Add the "conditions for revival" — what would have to be true for someone to legitimately try this again.
4. Move the original brief from `specs/` to `specs/archived/` (or delete if low-value).
5. Update `tasks/lessons.md` with any pattern that other setups should watch for.

The point of this document is **negative knowledge** — what does NOT work and why. It is at least as valuable as the active-setup documentation.
