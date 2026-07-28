# Retired Trading Setups — Evidence & Reasoning

This document is the **single source of truth** for setups that were prototyped, (believed-)validated, then retired. It exists so we don't burn weeks re-implementing dead theses.

**Inventory:** 23 retired setups across 6 retire batches:

- **Sub-5 ICT/SMC batch** (2026-04-25) — 3 setups: cargo-culted US/forex literature
- **Sub-7 generic-pattern batch** (2026-04-25) — 5 setups: Indian-published but universal mechanics
- **Sub-8 generic-pattern batch** (2026-04-26 to 2026-05-01) — 6 setups: same root cause as sub-7
- **Sub-9 narrow-cell batch** (2026-05-07 to 2026-05-14) — 4 setups: each killed by a specific bug or threshold failure (look-ahead bias, regime non-reproducibility, never sanity-validated, Holdout n below floor)
- **Sub-9 regulatory-regime batch** (2026-05-19) — 3 setups + 2 revival fails: `mis_unwind_vwap_revert_short`, `round_number_sweep_short`, `circuit_release_fade_short` (each shipped to production then collapsed in Holdout post-SEBI-Oct-2025); `capitulation_long_v2` (Phase 1-5 revival of earlier `capitulation_long_morning` — failed at Holdout ship gate); `pre_results_t1_fade` (Phase 1-5 candidate — failed v2 conservative re-run after data-classification correction; regime-conditioned edge with FII-positioning dependency)
- **Active-trio retirement batch** (2026-05-22 → 2026-07-27) — the 3 production intraday setups themselves: `delivery_pct_anomaly_short` (Stage 13 soft-disable 2026-05-22), `circuit_t1_fade_short` (Stage 14 2026-05-23), `gap_fade_short` (disabled 2026-06-25/29, retirement confirmed after paper validation 2026-07-27). All three cleared the historical gauntlet, shipped, and died on forward/OCI-canonical data — the empirical basis for the 2026-07-27 lifecycle amendments (forward-only validation + experiment ledger, see `docs/setup_lifecycle.md`)

**Hard rule: do NOT re-implement any setup listed here without:**

1. **Reading the retirement evidence below in full** — including the specific failure mode that killed it.
2. **Writing a corrected sanity script** that explicitly addresses that failure mode (e.g., no look-ahead bias on intraday aggregates, real-time regime classification, exit modeling that doesn't double-count favorable fills).
3. **Running new Discovery + OOS + Holdout passes** that meet ALL gates simultaneously (PF ≥ 1.10 net of full Indian fee stack at MIS-leveraged qty, n ≥ 30 per period, WR delta vs OOS ≤ 10pp, Sharpe > 0).
4. **Producing a falsification analysis** showing what would have to be true for the corrected setup to ship — and confirming that condition is testable on independent data.

If any of those four are skipped, the setup stays retired regardless of new PF.

**Active setups at the time of writing (2026-05-14):** `gap_fade_short`, `circuit_t1_fade_short`, `delivery_pct_anomaly_short`. All three carry the load. The 3-setup portfolio produces NET +Rs.1.13M FINAL after tax on 2yr Discovery (OCI run `20260514-002008_full`).

**[UPDATE 2026-07-27]** All three of the above are now retired: `circuit_t1_fade_short` (Stage 14, 2026-05-23, commit `0ae102b`), `delivery_pct_anomaly_short` (Stage 13 soft-disable, 2026-05-22, commit `845dc6a`), and `gap_fade_short` (disabled 2026-06-25/29, retirement confirmed after paper validation 2026-07-27). The +Rs.1.13M 2yr-Discovery portfolio figure no longer stands — it was dominated by gap_fade's Discovery-only PF, which did not reproduce out-of-sample (faithful re-sim: OOS 0.60 / Holdout 0.74). Re-baseline the portfolio before any intraday live activation. See the three entries below.

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

## `gap_fade_short` — RETIRED 2026-07 (after paper validation; disabled 2026-06-25/29)

**Retired:** 2026-07-27 (user-confirmed after paper trades; production disable 2026-06-29, commit `59c4b2c`; original disable 2026-06-25 was parked on branch `research/2026-06-25-gap-fade-directional-veto` and never reached main — the VM kept trading it in paper until 2026-06-29) [config `_disabled_2026_06_29`, configuration.json:111; commit `59c4b2c` message]
**Predecessor specs:** `specs/2026-05-16-gap-fade-short-paper-trade-validation.md` (pre-registered kill rule), `specs/2026-06-03-gap_fade_short-illiquidity-filter-prereg.md` (never activated), `specs/2026-06-16-gap_fade_short-squeeze-filter-investigation.md` (filter rejected)
**Config status:** `enabled: false` with `_disabled_2026_06_29` evidence string (configuration.json:110-111). Detector `structures/gap_fade_short_structure.py` + universe builder retained.

### Original thesis

SHORT fade of small-cap retail-FOMO gap-ups at the open. 09:15 gap-up 1.5-8% above PDC, exhaustion candle (upper wick ≥ 0.5× body, body ≤ 30%), volume decline after the gap, faded back toward PDC. Indian-microstructure anchor: small-cap retail-driven gap-up exhaustion + structural opening-window mechanics + SHORT-only (aligned with the SEBI FY23 long-bias-loses finding). [config `mechanism_notes` line 115, `_audit_decision_2026_05_01` line 161]

### Universe + filters (shipped/locked geometry)

- small_cap only (the Phase-7-validated cell; mid/micro not validated) [config lines 139-141, 153]
- Gap vs PDC in [1.5%, 8%], entry window 09:15-09:30, min_upper_wick_ratio 0.5, max_body_size_pct 30, require_volume_decline_after_gap [config lines 129-138]
- Locked exits from 2026-05-12 sweep: stop_buffer 0.10% above gap high, ATR mult 2.0, time_stop 13:00, partial_50_be_trail [config lines 126, 142-149]
- Priority 70, capital_budget_pct 30 — was the PRIMARY intraday short [config lines 156-157]

### Claimed validation (pre-retire)

- `_audit_decision_2026_05_01`: NET PF 1.49 on 5,848 trades, 2yr Discovery [configuration.json:161]
- Legacy sub7/sub8 chain: Discovery PF 1.153, Phase-1 v1/clean/v2/v3 1.118-1.227, OOS PF 0.993 (marginal), Holdout PF 1.128 [tasks/lessons.md:738]
- 2026-05-12 SL/target sweep claim: Disc 2.36 / OOS 1.71 / Hold 1.69, "Net Discovery jumps to Rs.~3M" [config `_status_2026_05_12_SWEEP_FIX`, line 126]
- `_live_status`: "PF=1.36, WR=70%, N=797 in 117-session Holdout" — already flagged non-reproducible by `_live_status_drift_note` 2026-05-15 (stored parquet `reports/sub8_oos_holdout_clean/gap_fade_short.parquet` shows PF 1.13, WR 64%) [config lines 153-154]
- Stage-13 confidence card (OCI, 2023-01→2026-04, n=3,452): PF 1.588 CI [1.448, 1.749], net +Rs.594,111, adj Sharpe 3.733 after 25.8% Harvey-Liu haircut; but war_vol_2026 regime already PF 0.789 (n=153, −Rs.13,419) [reports/confidence_cards/gap_fade_short_confidence_card.md]

### Why retired (the actual failure)

**None of the claimed PF chains reproduce on a faithful re-sim, and the setup failed its own pre-registered kill rule.**

1. **40-month faithful re-sim of the SHIPPED/LOCKED geometry** (sb=0.10 / atr=2.0 / ts=13:00 / partial_50_be_trail, via the canonical simulator `tools/sub9_research/_gap_fade_short_sl_target_sweep.simulate_trade`, driver `_tmp_gap_fade_rr_bucket.py`):
   - Pooled n=6,836: **PF 0.953, WR 61%, net −Rs.70k** — a net loser.
   - **Discovery 1.26 / OOS 0.60 / Holdout 0.74** (OOS+Holdout n=3,185, both net-negative).
   - **20/40 months positive (coin-flip).** Catastrophe months: 2025-04 (PF 0.46, −66k), 2025-05 (0.37, −57k), **2026-04 (PF 0.29, −Rs.153k — worst month and the most recent full month pre-live)**.
   - The sweep claim (2.36/1.71/1.69) and `_live_status` 1.36 are NOT reproducible — the drift was already flagged in `_live_status_drift_note`.
   [memory `project_gap_fade_short_paper_trade_validation.md`; summarized in config `_disabled_2026_06_29` (configuration.json:111): "pooled PF 0.95, Disc 1.26 / OOS 0.60 / Holdout 0.74"]

2. **Pre-registered kill rule executed.** `specs/2026-05-16-gap-fade-short-paper-trade-validation.md:118-120`: "If live PROD Holdout PF stays below 1.00 ... AND NET is negative, disable `gap_fade_short` from production." The same spec (line 15) already showed Holdout PF 0.93 for the locked combo.

3. **The full June 2026 paper book confirmed forward failure** (computed 2026-07-27 from VM `~/intraday_fixed/intraday-trade-assistant/logs/paper_*/analytics.jsonl`, net of fees): **2026-06-01 → 2026-06-29, 18 sessions, n=87 trades, net −Rs.7,821, PF 0.71, WR 64%** — the win-small/lose-big signature at the predicted WR. Consistent earlier partial reads: PF 0.81 (memory, partial window), PF 0.61 / −Rs.7,804 on the 11-day audit 2026-06-01→16 during which the other 4 setups netted +Rs.5,421 [memory `project_intraday_paper_audit_2026_06.md`].

4. **The index-direction-veto rescue was validated on backtest but NEVER paper-tested, and the avenue is closed.** The raw loss is entirely against-the-index shorts: **63% of fires short into a GREEN Nifty, PF 0.76 = 100% of the loss (−Rs.246k)**. An index-direction veto ("no short when Nifty green at entry", ±0.2% band; driver `_tmp_gap_fade_nifty_dir.py`) restored pooled PF 1.39 / Disc 1.44 / OOS 1.11 / Holdout 1.57 on backtest — this is why retirement was PAUSED Jun-Jul 2026 pending a per-setup hard directional block in `services/gates/directional_bias.py` (which is enabled:false + boost-only). The block was never wired: VM paper logs show **zero gap_fade fires after the 2026-06-29 disable** — no veto-conditioned variant ever traded paper. **User confirmed retirement 2026-07-27 on the raw June paper book, closing the veto avenue rather than funding its wiring + a fresh paper cycle.** [config `_disabled_2026_06_29` (configuration.json:111); memory `project_gap_fade_short_paper_trade_validation.md`; VM paper logs 2026-06/07]

5. **Falsification done (do not re-litigate):** `rr_t2` reward-headroom gate does not generalize (keep≥0.8 → Disc 1.43 but OOS/Hold 0.58/0.71); NOT an inverse edge (PF≈0.95 at ~50% winning months — long-flip loses to double fees); daily-squeeze regime filter REJECTED as non-stationary (squeeze-cohort PF: Discovery 1.24, full-cut 1.93, Holdout 0.76 — filtering = overfit to 2025-26; `specs/2026-06-16-gap_fade_short-squeeze-filter-investigation.md`).

### Root-cause pattern

Regime non-stationarity, not geometry: PF-2 months and PF-0.3 months alternate with no trade-level separable signal; the payoff shape is win-small/lose-big (needs ~73% WR, gets ~61-63%). The Discovery window (2023-24) was simply the good regime.

### Conditions for revival

1. A dedicated regime-separation study isolating the PF-2 vs PF-0.3 months with a PRODUCTION-reproducible classifier, passing Disc+OOS+Holdout PF ≥ 1.10 simultaneously on the filtered subset. Geometry/exit tweaks alone won't fix regime non-stationarity.
2. The index-direction veto is closed as gap_fade's revival avenue (paper-failed), though the veto principle itself remains validated for other directional setups.
3. Config's own condition "Re-enable ONLY after wiring + validating that veto" is superseded by the paper failure — a revival needs a NEW pre-registered thesis, not the veto. [configuration.json:111]

### Files/code status

- `config/configuration.json` `setups.gap_fade_short`: enabled=false, block retained with full evidence strings (lines 109-163)
- `structures/gap_fade_short_structure.py`, universe builder `services.setup_universe.gap_fade_universe`: retained
- Confidence card: `reports/confidence_cards/gap_fade_short_confidence_card.md` (pre-retire numbers — treat as the "claimed validation" artifact, not current truth)
- Holdout parquet: `reports/sub8_oos_holdout_clean/gap_fade_short.parquet`
- Disable commits: research-branch disable 2026-06-25 (branch `research/2026-06-25-gap-fade-directional-veto`), main disable `59c4b2c` 2026-06-29
- Still listed in `screening.allowed_setups` (configuration.json:1067) — harmless (detector gates on enabled) but should be cleaned up

---

## `circuit_t1_fade_short` — RETIRED 2026-05-23 (Stage 14; config flag flipped 2026-06-01)

**Retired:** 2026-05-23, commit `0ae102b` ("setup-lifecycle: retire circuit_t1_fade_short (Stage 14)"). The commit set the lifecycle status but left `enabled: true` in config, so the detector kept loading a 6-symbol universe and consuming slots until `enabled: false` was set on 2026-06-01. [config `_status_2026_06_01_RETIRED`, configuration.json:439; commit `0ae102b`]
**Predecessor spec:** `specs/2026-05-01-sub-project-9-brief-circuit_t1_fade_short.md`
**Config status:** `enabled: false`, block retained (configuration.json:437-490). Detector + universe builder + sanity script preserved as negative-knowledge artifacts per the retire commit.

### Original thesis

SHORT fade T+1 after an upper-circuit-pin day in small/mid-cap: T+0 up ≥4.5% closing at/near high (high_to_close ≥ 0.995) with volume ≥1.5× 20d, T+1 gap-up 1-5%, entered at the 10:30 bar, ride to T2 = T+0 close, time-stop 15:10. Mechanism: post-circuit retail FOMO + operator pump exhaustion — Indian-microstructure-specific (NSE DPR flexing, no US/forex analog), 5 peer-reviewed sources. Sub-project #9's first APPROVED setup. [config `mechanism_notes` line 443, `_research_notes` line 488, thresholds lines 457-471]

### Universe + filters

mid_cap + small_cap (cell-level mid_cap-only restriction FAILED holdout and was reverted 2026-05-06); no regime restriction (trend_up-only cell also invalidated by holdout); locked exits per 2026-05-12 sweep: t1_qty_pct=0.0 (no partial — ride to T2=t0_close), stop = T1-high + 0.25%, min_stop 0.5%. [config lines 465-478 incl. `_comment_cap_post_holdout`, `_comment_regime_post_holdout`, `_comment_t1_qty_pct`]

### Claimed validation (pre-retire)

- Sub9 sanity: NET PF 1.473 on 654 trades over 2024 [config `_research_notes` line 488]; Discovery PF 1.404 (only 11mo of 2024, not full 2yr), OOS PF **0.982** (below the 1.10 ship gate), Holdout PF 1.890 — shipped on Discovery+Holdout despite the OOS failure [tasks/lessons.md:736]
- 2026-05-12 sweep: production partial_50_be_trail was actively losing (PF 0.49 Disc / 0.91 Hold); locked all-in-to-T2 mechanic gave PF 1.74 Discovery / 1.88 Holdout [config `_status_2026_05_12_SWEEP_FIX` line 454]
- `_status_2026_05_08_oci`: "Holdout PF 1.89 (Oct'25-Apr'26) confirmed robust through war + regulatory regime" — later found to be **sanity-sourced, not OCI** (Lesson #13 violation) [config line 455; commit `0ae102b`]
- backtest_findings.md 8e table (run `20260519-123643_full`): OOS 1.29 / HO pre-war **1.88** / HO war **0.53** — initially classified "war-only collapse (kept active per portfolio decision)" [analysis/backtest_findings.md:254, 282]

### Why retired (the actual failure)

**Stage 14 triggered by the OCI-v2 confidence card, and the "war-only collapse, keep active" story collapsed on decomposition:**

- Stage 14 confidence card on OCI v2 canonicals (2023-02→2026-04, n=632, net +Rs.33,299): aggregate **PF 1.174 CI [0.937, 1.459]** — CI lower bound 0.937 < 0.95 ship-floor. Expectancy +Rs.52.69/trade CI [−20.40, +127.45] crosses zero. [reports/confidence_cards/circuit_t1_fade_short_confidence_card.md]
- **Pre-war Holdout (2025-10-01 → 2026-02-27, war excluded): n=51, net PF 0.74 — STILL fails the ship gate with the war carved out. Structurally broken, not regime-paused.** [commit `0ae102b`]
- Two of seven regimes drag the CI below floor: post_tariff_consolidation PF 0.695 (n=84, −Rs.8,621) and war_vol_2026 PF 0.219 (n=30, −Rs.12,260). [confidence card per-regime table]
- The prior HO PF 1.89 "robustness" claim was sanity-sourced (`_concern_2026_05_22_oci_v2_ho_discrepancy` resolved with full evidence chain in the retire commit) — the real OCI holdout never showed it. [commit `0ae102b`]
- Post-retire hygiene bug: the Stage 14 commit didn't flip `enabled`, so the detector kept consuming universe slots for ~9 days until the 2026-06-01 config fix. [config `_status_2026_06_01_RETIRED` line 439]

### Conditions for revival

1. The 2026-05-19 decision principle ("war-only collapse = regime-temporary, keep active") no longer applies — the pre-war-HO decomposition (PF 0.74, n=51) shows structural decay. Any revival must first reproduce a NET-positive pre-war holdout on OCI-pipeline (not sanity) numbers. [commit `0ae102b`; analysis/backtest_findings.md:285]
2. Never accept sanity-sourced holdout numbers as OCI validation (Lesson #13). The 1.89-vs-0.74 gap IS the retirement.
3. Standard doc gates (corrected sanity, Disc+OOS+Holdout all ≥1.10 net, n≥30/period) — note this setup NEVER passed OOS (0.982) even in its best claimed chain.

### Files/code status

- `config/configuration.json` `setups.circuit_t1_fade_short`: enabled=false, block retained (lines 437-490)
- `structures/circuit_t1_fade_short_structure.py`, `services.setup_universe.circuit_t1_universe`, `tools/sub9_research/sanity_circuit_t1_fade_short.py`: preserved as negative-knowledge artifacts [commit `0ae102b`]
- Confidence card: `reports/confidence_cards/circuit_t1_fade_short_confidence_card.md`

---

## `delivery_pct_anomaly_short` — DISABLED 2026-05-22 (Stage 13 soft-disable; Stage 14 decision never recorded)

**Retired:** 2026-05-22 soft-disable, commit `845dc6a` ("config(delivery_pct_anomaly_short): disable per Stage 13 confidence card"). Status is formally "soft-disabled pending researcher decision on Stage 14 full retirement vs chop-regime gate redesign" — no Stage 14 record exists as of 2026-07-27, and the setup has stayed disabled, so this entry documents it as de-facto retired. [config `_status_2026_05_22_disabled`, configuration.json:493]
**Predecessor spec:** `specs/2026-05-08-sub-project-9-brief-nse_delivery_pct_anomaly.md`
**Config status:** `enabled: false`, block retained (configuration.json:491-545).

### Original thesis

SHORT fade of next-day price action after a delivery-percentage anomaly: NSE bhavcopy delivery_pct < 20% + same-day pump > 3% = retail/operator pump signature (churned intraday, not held → speculator inventory must unwind). Cross-day RVOL confirmation rejects intra-session false signals. Mechanism CRITICALLY depends on F&O single-stock position math (MWPL) — the speculator-unwind flow this fades. [config `mechanism_notes` line 497, `_comment_depends_on` line 510, `_research_notes` line 543]

### Universe + filters (shipped Cell B)

mid_cap only (large_cap 0.83 / small_cap 0.76 diluted the edge), ADV Rs.100-1000 cr, prior-day return ≥ 3%, delivery_pct ≤ 20%, gap in [−2%, +3%], entry window 09:30-10:30, T1=0.25R (50% partial) / T2=0.75R, time-stop 13:00. [config lines 514-534 incl. `_comment_allowed_cap_segments`]

### Claimed validation (pre-retire)

- Sub9 sanity chain: Discovery PF 1.44 / OOS 1.90 / Holdout 1.13 / War-period 5.03 at tight targets [config `_status_2026_05_08_enabled` line 512; tasks/lessons.md:739]
- Cell B production-replay (analytics.jsonl fee model), Jan-Sep 2025 OCI: PF 1.245, n=295 [config `_status_2026_05_14_paper_cell_B` line 511]
- backtest_findings.md 8e table: OOS 1.43 / HO pre-war 1.29 / HO war 1.09 — "graceful war drag" [analysis/backtest_findings.md:280]

### Why retired (the actual failure)

**Two stacked failures: a SEBI-Oct-2025 mechanism break on the shipped cell, then a Stage 13 confidence card showing the decay is structural (pre-dates the war AND partly pre-dates SEBI) and fee-dominated.**

1. **SEBI Oct 1, 2025 F&O rule change broke the load-bearing dependency.** Tightened MWPL + single-stock position limits altered the speculator-inventory-unwind flow the setup fades: Cell B PF 1.245 (OOS, pre-SEBI) → **0.879** (Holdout Oct'25-Apr'26, post-SEBI). This was known at ship time — the 2026-05-14 decision was to run Cell B at PAPER only to capture forward data. [config `_status_2026_05_14_paper_cell_B` line 511, `_comment_depends_on` line 510; same Oct-1-2025 cutover documented as breaking `mis_unwind_vwap_revert_short` and `circuit_release_fade_short` — tasks/lessons.md:627, analysis/backtest_findings.md:234]
2. **Stage 13 OCI confidence card (2023-02→2026-04, 3.3y, n=542, net −Rs.7,530):** aggregate **PF 0.954 CI [0.777, 1.151]** — CI crosses 1.0, lower bound < 0.95 floor. Harvey-Liu adjusted Sharpe **−0.274 (NEGATIVE)** on the M=6 corpus — the "survivor" is consistent with the best of 6 coin flips. [reports/confidence_cards/delivery_pct_anomaly_short_confidence_card.md; config `_status_2026_05_22_disabled` line 493]
3. **Decay is STRUCTURAL, not war-only** — multiple PRE-war regimes ≤ 1.0: pre_election_calm 0.94, post_election_consolidation 0.84, tariff_recovery_rally 0.58, post_tariff_consolidation 0.81; war_vol_2026 itself only 0.93. So the SEBI break is real but not the whole story — the aggregate edge was never robust across regimes even before Oct 2025, and the "war-only kept active for revival" principle (Lesson #4) explicitly does NOT apply. [confidence card per-regime table; commit `845dc6a`]
4. **Fee-drag dynamic (Lesson #7):** gross PF 1.16 (a real, small signal) wiped by ~Rs.32K fees on 542 trades — only the "chop" regime would be net-positive after fees, and gating to it post-hoc would violate the Stage-5 Discovery cell-lock rule (Lesson #2). [config `_status_2026_05_22_disabled` line 493; commit `845dc6a`]

Note the contradiction worth preserving: the 2026-05-19 per-period table (backtest_findings.md:280) showed 1.43/1.29/1.09 "graceful war drag" — three days later the OCI-canonical confidence card showed 0.954 aggregate. The lifecycle answer: pooled-period PF on one run ≠ regime-decomposed canonical PF with CIs; the card is the source of truth.

### Conditions for revival

1. Stage 14 decision is formally still open per the config string (full 7-step retirement vs chop-regime gate redesign) — if redesigned, the chop gate must be a Stage-5 Discovery cell-lock, NOT a post-hoc filter. [config line 493]
2. Re-validate the MWPL dependency on every SEBI position-limit change; the mechanism is `regulatory_sensitivity: rule_dependent` — any revival must show post-Oct-2025 data alone passing net gates. [config lines 501-510]
3. Fee floor: with gross PF 1.16 and this trade size, revival needs either bigger per-trade R or a materially higher gross edge; Apr-1-2026 STT hike adds further pressure. [config `_comment_depends_on` line 510]

### Files/code status

- `config/configuration.json` `setups.delivery_pct_anomaly_short`: enabled=false, block retained (lines 491-545)
- `structures/delivery_pct_anomaly_short_structure.py`, `services.setup_universe.delivery_pct_universe`, `tools/sub9_research/sanity_nse_delivery_pct_anomaly.py`: retained per soft-disable ("detector/sanity/universe code retained pending researcher decision") [commit `845dc6a`]
- Confidence card: `reports/confidence_cards/delivery_pct_anomaly_short_confidence_card.md`
- Related deferred work: `specs/2026-05-09-delivery-pct-rvol-live-wiring-plan.md` (live wiring — moot unless revived)

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
3. **The `circuit_t1_fade_short` precedent.** Circuit_t1 was on the retire list per task #123 but was subsequently revived via a corrected SL/target sweep and now ships at PF 1.50. That revival succeeded because the underlying asymmetry (post-FOMO mean-reversion on Indian-specific circuit-band stocks) was sound; only the SL/target geometry was broken. None of the 11 setups above have that profile — they fail on the asymmetry, not on geometry. *[2026-07-27: circuit_t1_fade_short has since been retired itself (Stage 14, 2026-05-23 — see its entry above); the precedent stands only as an example of the geometry-fix revival path, not as a live endorsement.]*

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

**[Editorial note 2026-07-27: gap_fade_short was itself retired in July 2026 — see its entry in this document. The comparison below is kept as written (2026-05-16) because the mechanism contrast it draws is still the reason C-09 died at sanity; "DOES work" reflects the state of belief at the time.]**

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

## `mis_unwind_vwap_revert_short` (C-08) — RETIRED 2026-05-19 (Holdout collapse)

**Thesis (original):** SEBI requires MIS positions to be auto-squared by 15:20 IST. Retail LONGS in overbought small/mid-cap stocks at 15:00+ face forced-sell flow during 15:15-15:25, creating a 10-minute window of concentrated sell-side pressure that SHORT entries before 15:15 can capture.

**Mechanism research (2026-05-19, Phase 1):** Heterogeneous broker auto-square is REAL:
- Upstox/Angel One: 15:15 IST auto-square
- Zerodha: 15:20 IST (drifts to 15:24)
- ICICI Direct: 15:15-15:20 IST
- All charge ~₹50+GST = ₹59 penalty per auto-squared position

Volume profile (Discovery 2-yr, 374K symbol-days): confirmed bulge at 15:15-15:25 is real — small/mid-cap mean volume at 2.5-3.0x the 11:00-13:00 baseline. Forced-liquidation signature is unambiguous.

**Universe + filters tested:**
- Cap: small_cap + mid_cap, MIS-eligible
- VWAP extension >= 0.5%
- RSI >= 85 (TIGHTENED from original 75 after cell-sweep)
- vol_ratio >= 15x cum-vol (TIGHTENED from original 7x)
- SL: 0.5% above entry
- T2: 3.0R (full exit)
- Active window: 15:00-15:10 entry, 15:15 5m-bar close exit (= 15:20:00 IST, just before Zerodha auto-square — no penalty)

**Validation chain (locked cell, all anti-bias guards verified):**
- Discovery (2023-01..2024-12): n=622, PF_realized=1.50, PF_net=**1.213**, WR=39%, 19/24 winning months, +Rs 72,728
- OOS (2025-01..09):              n=261, PF_realized=1.49, PF_net=**1.216**, WR=35%, 7/9 winning months,  +Rs 32,195
- Holdout (2025-10..2026-04):     n=198, PF_realized=0.91, PF_net=**0.751**, WR=32%, ? mwin,           **-Rs 34,562** ❌

**Failure mode: regulatory regime break (same pattern as `delivery_pct_anomaly_short`).**

Discovery + OOS both PREDATE the SEBI Oct 2025 rule changes (MWPL tightening, F&O position limits cut, single-stock SLB changes). PF parity Disc 1.213 ↔ OOS 1.216 was misleading because both periods reflect the SAME pre-SEBI retail MIS-positioning regime.

Holdout (Oct 2025 onwards) covers post-SEBI: retail MIS concentration in small/mid-cap dropped, the forced-sell pressure mechanism weakened, multi-bar subset PF also collapsed (3.43 Disc → 4.14 OOS → 1.47 HO). Setup is **regulatory-decay-dead**.

**Diagnostic depth notes:**
- 71-73% of trades same-bar exit (entry bar 15:05 hit either SL or T2 within 5 min). High vol_ratio≥15 cells are inherently volatile.
- Same-bar subset PF_net 0.84 (Disc) / 0.86 (OOS) / 0.60 (HO) — sanity's pessimistic same-bar-SL-priority rule made these net-negative. Production tick-level execution could differ.
- Multi-bar subset PF_net 3.43 (Disc) / 4.14 (OOS) / 1.47 (HO) — even the "clean" carve-out collapsed in HO.
- Signal-bar range as a pre-entry filter for same-bar prediction: tested 9 thresholds, monotonically HURTS PF (wider cap = higher PF). No clean predictor found.

**Code files removed:**
- `structures/mis_unwind_vwap_revert_short_structure.py`
- `tools/sub9_research/sanity_mis_unwind_vwap_revert.py`
- Function `mis_unwind_vwap_revert_short_universe` in `services/setup_universe.py`
- Config block `setups.mis_unwind_vwap_revert_short` in `config/configuration.json`
- Entry in `services/dispatch/worker.py:_STRUCTURE_TO_SETUP_TYPE`
- Comment refs in `services/setup_universe.py:compute_static_universes` and `services/screener_live.py:_run_dispatch_path`

**Preserved (negative-knowledge artifacts):**
- `tools/sub9_research/sanity_mis_unwind_REAL_window.py` — the proper Phase-4 sanity with anti-bias guards
- `tools/sub9_research/_phase2_empirical_signature.py` — Phase-2 Indian-markets empirical signature check
- `reports/sub9_sanity/_mis_unwind_locked_trades_{discovery,oos,holdout}.csv` — 3-period evidence trail
- `reports/sub9_sanity/_phase2_mis_unwind_drift_records.csv` — 374K symbol-day population data

**Conditions for revival:**
This setup should NOT be re-implemented unless:
1. **A new post-SEBI regulatory event creates a different MIS-unwind mechanism** (e.g., new auto-square timing, new forced-deliverability rule). Then run Phase 1 research again with the new mechanism.
2. **Live tick-level reproduction shows same-bar trades are NOT 50/50** (i.e., genuine tick-order info makes them profitable in production where 5m sanity rates them noise). Then re-test with tick-level sanity.
3. **A new cell beyond RSI/vol_ratio is discovered** (e.g., gap_pct conditioning, NIFTY-direction conditioning, sector flow) that survives Discovery + OOS + Holdout. The Phase-3-style brief in `tasks/lessons.md` 2026-05-12 ship gate applies.

## `round_number_sweep_short` (C-02) — RETIRED 2026-05-19 (all 3 periods fail)

**Thesis (original):** Indian retail traders cluster stop-losses at round-number prices (Rs.100, 150, 200, 250) far more than at PDH/PDL because retail-education courses teach this (Subasish Pani, Powerof Stocks, Zerodha Varsity). Rs.100-250 stocks are prime retail territory (cheap, accessible, hyped on YouTube). When intraday price pokes above a round number briefly and closes back below, the upside stop-cluster has failed — retail breakout buyers are trapped. Their panic-sell cascades down for a SHORT fade.

**Universe + filters (production cell-lock):**
- Cap: small_cap MIS-eligible
- Price band: Rs.100-250 (cheap-stock retail territory)
- Round-number increment: Rs.50 (50, 100, 150, 200, 250)
- Poke pct: 0.15% (bar pierces RN by ≥ this)
- Volume ratio: ≥ 2.0× session-cumulative-mean
- Active window: 11:00-12:30 IST
- SL: 0.5% above sweep_high
- T1=1R FULL EXIT (100% qty at T1, T2=2R retained as backup but inert with qty_pct=0)
- Time stop: 15:00 IST

**Claimed validation (`_status_2026_05_16`):** 3-window PFs Disc 1.24 (n=300) / OOS 1.21 (n=176) / Holdout 1.17 (n=126). Cell-locked from aggregate sanity which failed all 3 windows (PF 0.80/0.84/0.81).

**Actual production result (OCI 17-month run, 2025-01 → 2026-04):**
- ALL: n=217, PF_real **1.075**, WR 50%, net +Rs 5,135 over 16 months, mwin **8/16 (50%)**
- IS (9mo): n=118, PF 1.06, mwin 5/9 (56%)
- HO (7mo): n=99, PF 1.09, mwin **3/7 (43%)**

PF marginally above 1.0 realized, but after Indian fee stack (~0.5% round-trip), PF_net falls below 1.0. Monthly winning months in HO collapse to 43%, below the 67% gate.

**Audit (2026-05-19): cell-locked re-evaluation on aggregate sanity CSVs**

Applied production cell-lock filter (11:00-12:30, small_cap, Rs.100-250) + production T1-full-exit geometry to the existing aggregate sanity CSVs:

| Period | n | PF_real | PF_net | WR | mwin | Net PnL |
|---|---|---|---|---|---|---|
| Discovery (2yr) | 683 | 0.86 | **0.69** | 45% | **4/24 (17%)** | -Rs 91,144 |
| OOS (9mo) | 314 | 0.85 | **0.69** | 46% | **0/9 (0%)** | -Rs 44,439 |
| HO (7mo) | 270 | 0.94 | **0.73** | 48% | **1/7 (14%)** | -Rs 29,711 |

**ALL THREE PERIODS NET-LOSING.** No period produces a winning result on PF_net. Monthly winning months 0-17% across the chain. The `_status_2026_05_16` claim (3-window PFs 1.24/1.21/1.17) does NOT reproduce under disciplined cell-lock + T1-full-exit geometry.

**Failure mode:** Cell-mining illusion. The original "cell-locked" claim was found by sweeping 37K Discovery aggregate trades for cell+R-geometry combos. The cell that appeared to pass was data-mined — when the same filter + R-geometry is applied to a clean evaluation of the same data, PF_net is 0.69 not 1.24. The `_status_2026_05_16` numbers were not reproducible from the underlying CSVs.

**Sanity script bug (minor):** `tools/sub9_research/sanity_round_number_sweep.py` walked from `entry_idx = i+1` while entry_price was `i+1.close` — bar i+1's high/low (which happened BEFORE entry-at-close) were used for SL/T2 checks. Same-bar rate was only 0.3%, so net impact small. Bug noted but not the dominant cause of retirement.

**Code files removed:**
- `structures/round_number_sweep_short_structure.py`
- `tools/sub9_research/sanity_round_number_sweep.py`
- Function `round_number_sweep_short_universe` in `services/setup_universe.py`
- Config block `setups.round_number_sweep_short` in `config/configuration.json`
- Entry in `services/dispatch/worker.py:_STRUCTURE_TO_SETUP_TYPE`
- Comment refs in `services/setup_universe.py:compute_static_universes` and `services/screener_live.py:_run_dispatch_path`

**Preserved (negative-knowledge artifacts):**
- `reports/sub9_sanity/_round_number_sweep_trades_{discovery,oos,holdout}.csv` — original aggregate sanity (37K Disc / 17K OOS / 11K HO)
- `reports/sub9_sanity/_round_number_sweep_cells_*.csv` — cell-mining outputs that produced the (non-reproducible) shipped claim

**Conditions for revival:**
This setup should NOT be re-implemented unless:
1. **The reproducibility gap is explained** — why does the existing aggregate sanity recompute to PF_net 0.69 across all periods while the `_status_2026_05_16` claim was PF 1.24? Need to identify the methodological gap and confirm it's not an OOS-overfit artifact.
2. **A genuinely new mechanism is found** for round-number price clustering. The Subasish Pani / Powerof Stocks precedent is YouTube-anecdotal, not data-confirmed. Phase-2 empirical signature check on Discovery would need to demonstrate retail stop-clustering signature is detectable in NSE intraday data.
3. **A different price band or window** shows stable 3-window edge. Sweeps to date have only found data-mined cells.

## `circuit_release_fade_short` (C-03) — RETIRED 2026-05-19 (regulatory decay, pre-war)

**Thesis (original):** SHORT fade of failed re-test of morning circuit-pin in small/mid-cap NSE names. Indian retail FOMO drives a stock to upper circuit-band (5/10/20%) early. Sellers appear mid-day, price drops 1-2% from pin. Retail buyers re-engage, price re-tests day high from below. When the re-test FAILS, trapped FOMO buyers panic-sell, cascade-down.

**Validation chain (claimed Disc 2.12 / OOS 3.13 / Holdout 4.53 — but those were aggregate WIDE_OPEN PFs, not production-config performance).**

**Actual production results (OCI 17-month run, full 6-setup portfolio, 2025-01..2026-04):**

Per-period breakdown (raw PnL):

| Period | n | PF_real | Avg/trade | Sum |
|---|---|---|---|---|
| Discovery (2023-24) | 1,414 | 1.19 | +Rs 83 | +Rs 117,924 |
| OOS (Jan-Sep 2025) | 472 | 1.26 | +Rs 112 | +Rs 52,716 |
| **HO_pre_war (Oct-Dec 2025)** | 110 | **0.84** ❌ | -Rs 84 | **-Rs 9,207** |
| **HO_war (Jan-Apr 2026)** | 159 | **0.60** ❌ | -Rs 222 | **-Rs 35,271** |

After MIS leverage + Indian fees + 31.2% tax (OOS+HO combined): **-Rs 63,429 NET (LOSING)**. Per `analysis/reports/3year_backtest/run_20260519_130847/detailed_report.txt`.

**Failure mode: regulatory decay starting Oct 2025 (pre-war).** The HO_pre_war period (Oct-Dec 2025) is post-SEBI-Oct-2025 changes but BEFORE the Jan 2026 war volatility. PF dropped 1.26 → 0.84 in this 3-month window. The war period (Jan-Apr 2026) deepened decay to 0.60 but did NOT cause it.

This pattern is distinct from `circuit_t1_fade_short` (which was PROFITABLE in HO_pre_war at PF 1.88, only war hurt it) and matches `mis_unwind_vwap_revert_short` (decay started post-SEBI cutover, deepened in war).

**Cell salvage attempts (2026-05-19 exhaustive sweep):**

Per the methodology used for `mis_unwind` and `round_number`, ran 1D, 2D, and 3D cell sweeps on day_gain_pct, rejection_pct, entry_hour, cap_segment.

| Best cell tried | Disc | OOS | HO_pre | HO_war |
|---|---|---|---|---|
| hour=12 + rej_pct 0.4-1.0 (post-hoc winner) | 1.25 | 1.44 | 1.44 (n=31) | **0.53** ❌ |
| hour=13 (Disc/OOS-locked winner) | 1.34 | 1.32 | **0.53** ❌ | 0.55 ❌ |
| All day_gain bands | 1.02-1.31 | 1.04-1.51 | 0.69-1.03 | 0.49-0.94 |

**NO 3D cell with n>=20 + PF>1.10 ships in ALL 4 periods.** Cell-mining illusion confirmed: the "hour=12 cell that survives HO_pre" was a post-hoc artifact — if locked on Disc+OOS independently, hour 13 would've won and that cell broke harder in HO_pre.

**Code files removed:**
- `structures/circuit_release_fade_short_structure.py`
- Function `circuit_release_fade_short_universe()` in `services/setup_universe.py`
- Config block `setups.circuit_release_fade_short` in `config/configuration.json`
- Entry in `services/dispatch/worker.py:_STRUCTURE_TO_SETUP_TYPE`
- Comment refs in `services/setup_universe.py` and `services/screener_live.py`

**Preserved (negative-knowledge artifacts):**
- `tools/sub9_research/sanity_circuit_release_fade.py` — anti-bias-fixed sanity script (Phase A look-ahead removed May 18-19, useful for any future revival attempt)
- `tools/sub9_research/_circuit_release_fade_sweep_cellmine.py` — cell-mining tool
- `tools/sub9_research/_tmp_retrace_sweep*.py` — retrace-filter remediation attempts (didn't help; documented for future reference)
- `reports/sub9_sanity/_circuit_release_fade_short_trades_oos_{CLEAN,MODE_A,PHASE_A,PHASE_B2}.csv` — sanity reproducibility evidence trail
- `reports/sub9_sanity/_circuit_release_fade_short_trades_holdout_PHASE_A.csv` — holdout under fixed-anti-bias sanity

**Conditions for revival:**
This setup should NOT be re-implemented unless:
1. **A new post-Oct-2025 mechanism emerges** that restores retail circuit-pin behavior (e.g., SEBI loosening F&O / leverage rules)
2. **Holdout regime stabilizes** (war volatility recedes) AND **a tightened cell** shows PF >= 1.10 in 6-month rolling window with n >= 100
3. **Mode-A execution (tick zone re-touch) testing reveals** the production was systematically failing on entry mechanics distinct from regime (we tested Mode A in May 2026 — PF only dropped 14%, so not the cause)

## `capitulation_long_v2` — RETIRED 2026-05-19 (revival attempt failed at Holdout)

**Retired:** 2026-05-19 (never reached production — failed at sanity Phase 5 Holdout)
**Predecessor:** `capitulation_long_morning` (retired earlier in sub-9 batch). v2 was a disciplined Phase 1-5 revival under the chain documented in `tasks/lessons.md` 2026-05-19 #3.
**Sanity scripts (preserved):** `tools/sub9_research/sanity_capitulation_long_v2.py`, `tools/sub9_research/_phase2_capitulation_signature.py`

### Original thesis (v2)

Indian small/mid-cap MIS-eligible stocks gapping down -3% to -5% (per intradaylab.com Phase-1 research: large gap-downs recover more reliably than -1.5/-2% danger zone). Phase 2 empirical signature on 21,548 Discovery 2023-24 gap events confirmed mid_cap × gap [-5%, -3%] is the sweet spot (MFE/MAE 1.57, mean +0.47% return at 14:30). LONG bias on exhaustion candle (09:25-10:00 window, lower_wick/body ≥ 0.5, body ≤ 30%, green, no new low) with Mode B next-bar-open entry.

### Universe + filters (locked from Phase 2)

- MIS-eligible + mid_cap (cap_segment lookup from `services/symbol_metadata`)
- 09:15 gap from PDC in [-5.0%, -3.0%] (gap-size-only filter, no news filter per user accept)
- Exhaustion candle 09:25-10:00 IST (lower_wick/body ≥ 0.5, body ≤ 30%, green, no new low)
- Mode B: entry at next-bar OPEN from i+1, path walk from i+1
- Locked cell (from Discovery sweep): SL=0.7%, T2=3.0R, TS=13:00, ride-to-T2 geometry

### Validation chain (3-period locked cell)

| Period | n | PF_real | **PF_net** | WR | mwin | Net |
|---|---|---|---|---|---|---|
| Discovery (2yr) | 527 | 1.30 | **1.127** | 43.3% | 12/24 (50%) | +Rs 35,283 |
| OOS (Jan-Sep 2025) | 350 | 1.87 | **1.617** ⭐ | 51.1% | 5/9 (56%) | +Rs 96,996 |
| **HO (Oct'25-Apr'26)** | 216 | 1.19 | **1.031** ❌ | 40.3% | **3/7 (43%)** | +Rs 3,943 |

**HO war vs pre-war breakdown:**
- HO pre-war (Oct-Dec 2025): n=20, PF_net 1.030, WR 35.0%, +Rs 372
- HO war (Jan-Apr 2026): n=196, PF_net **1.116**, WR 38.3%, +Rs 13,816

War period generated 91% of HO trades; war-period PF (1.12) actually exceeded HO pre-war (1.03). LONG bias benefits from gap-down volatility — the war did NOT hurt this setup.

### Ship gate evaluation

| Gate | Required | Actual | Pass? |
|---|---|---|---|
| HO PF_net | ≥ 1.10 | 1.031 | ❌ FAIL (below floor) |
| HO mwin | ≥ 4/7 (57%) | 3/7 (43%) | ❌ FAIL |
| WR delta OOS→HO | ≤ 10pp | -10.8pp | ❌ MARGINAL FAIL |
| HO n | ≥ 100 | 216 | ✅ |
| Same-bar % | < 30% | 1.4% | ✅ |

### Failure mode — Disc/OOS-favorable-regime illusion (same pattern as `mis_unwind`)

Discovery PF_net 1.13 → OOS PF_net 1.62 → HO PF_net 1.03 trajectory is **structurally identical to `mis_unwind_vwap_revert_short`** (Disc 1.21 → OOS 1.22 → HO 0.75). The OOS strength (PF 1.62) was a favorable-regime artifact, not validated edge. HO returned to Discovery-like marginal performance.

This is the **Disc+OOS parity ≠ HO survival** failure that `tasks/lessons.md` #1 (2026-05-19) was written to prevent — and this revival fell into it anyway. Both periods (Disc and HO) cluster at PF_net ~1.03-1.13 (effectively break-even after fees + tax); OOS was the outlier on the positive side, not Discovery on the negative side.

### Notable positive signal (preserved for future revival)

War-period PF_net 1.12 outperforms most existing production SHORT setups in the same window:
- `gap_fade_short` HO war: 1.08
- `delivery_pct_anomaly_short` HO war: 1.09
- `or_window_failure_fade_short` HO war: 0.92
- `circuit_t1_fade_short` HO war: 0.53

The setup is **complementary** (LONG-side, volatility-favoring) — just not above the ship-gate floor on its own.

### Code state

**No detector code was ever written** — v2 stopped at sanity-validation phase. Nothing to remove from production. The following research artifacts are preserved as negative-knowledge:

**Preserved:**
- `tools/sub9_research/_phase2_capitulation_signature.py` — empirical signature analysis on 21,548 Discovery 2023-24 gap-down events (confirms mid_cap × gap [-5,-3] sweet spot)
- `tools/sub9_research/sanity_capitulation_long_v2.py` — disciplined anti-bias sanity (full Phase 1-5 chain, Mode B from i+1, locked cell sweeps)
- `reports/sub9_sanity/_phase2_capitulation_signature.csv` — Phase 2 signature output
- `reports/sub9_sanity/_capitulation_long_v2_grid_{discovery,oos,holdout}.csv` — full grid sweep outputs
- `reports/sub9_sanity/_capitulation_long_v2_trades_{discovery,oos,holdout}.csv` — locked-cell trade ledgers

### Conditions for revival

This setup should NOT be re-implemented unless:

1. **Post-war regime cleanup** — re-test after war volatility recedes (Q3-Q4 2026 fresh Holdout). The HO war PF 1.12 hints volatility-favoring behavior; calmer regime may still produce PF_net < 1.10 (the pre-war HO sub-cell already showed 1.03).
2. **A genuinely additive filter** lifts HO PF_net above 1.10 floor — e.g., news-event filter restricting to non-news gaps (was deferred for v2 per user accept on gap-size-only). Adding a filter post-hoc to chase HO is data-mining; the filter must be pre-registered AND tested on fresh untouched HO.
3. **Portfolio diversification thesis** — if existing 5-setup portfolio's HO drawdown is dominated by SHORT correlation, adding even a marginal LONG (PF_net 1.03) at small size might still improve portfolio-level Sharpe. Requires correlation analysis, not standalone ship-gate.
4. **The Disc/HO-cluster vs OOS-outlier asymmetry is explained** — why was OOS PF 1.62 a 50% premium over both Disc and HO? Until the cause is identified, the 1.10 ship gate is the only defense against OOS-outlier shipping.

## `pre_results_t1_fade` — RETIRED 2026-05-19 (regime-conditioned edge; OOS confirmed fail with corrected filter)

**Status: RETIRED. Retirement was briefly reversed (data classification concern), then re-confirmed via conservative re-run with corrected filter.**

### Investigation chain (4 layers)

1. **Layer 1 — Period aggregates:** OOS PF_net 0.82 vs Disc 1.10 / HO 1.15. Initial retirement (premature).
2. **Layer 2 — Monthly breakdown (user's challenge):** OOS is the OUTLIER, Disc + HO are similar. Suggested structural shift, not non-stationary edge.
3. **Layer 3 — Data audit:** NSE `announcements_fr` source died after Mar 2025. AMC events for 2025+ got demoted to "scheduled" class via lower-priority `board_meetings` source. Hypothesis: setup's `AMC-only` filter missed the demoted-AMC events.
4. **Layer 4 — Conservative re-run (Path 2 — Phase 4 grid + Phase 5 with corrected `{AMC, scheduled}` filter):** Recovered ~50% more OOS events. OOS PF_net got WORSE (0.816 → **0.784**). Demoted events were ALSO losers. Hypothesis falsified.

### Final v2 evidence (corrected filter)

| Period | n_v2 | PF_real | PF_net | WR | mwin | Net (after tax) | Median monthly PF |
|---|---|---|---|---|---|---|---|
| Discovery (24mo) | 1,940 | 1.388 | 1.103 | 52.7% | 20/23 | Rs +365K | 1.438 |
| **OOS (9mo)** | 766 | 0.990 | **0.784** ❌ | 45.0% | 4/9 | **-Rs 509K** | 0.976 |
| HO (7mo) | 755 | 1.449 | 1.159 | 52.6% | 4/6 | +Rs 219K | 1.482 |

**OOS+HO combined NET: -Rs 290K LOSS** (16 post-Discovery months).

### Ship gate evaluation

| Gate | Required | Actual | Pass? |
|---|---|---|---|
| HO PF_net | ≥ 1.10 | 1.159 | ✅ |
| HO mwin | ≥ 4/7 (57%) | 4/6 | ✅ |
| WR delta OOS→HO | ≤ 10pp | 7.5pp | ✅ |
| HO n | ≥ 100 | 755 | ✅ |
| Same-bar % | < 30% | 29.9% | ✅ (barely) |
| **OOS+HO combined NET** | > 0 | -Rs 290K | ❌ |
| **Stationarity max-min PF_net** | ≤ 0.30 | 0.375 | ❌ |

### Failure mode — regime-conditioned edge with FII-positioning dependency

The setup mechanism (institutional T-1 de-risking → SHORT signal) **requires institutions to be net-LONG** to have de-risking flow to fade. When FIIs are net-sellers (e.g., Jan-Mar 2025: ~₹1L cr outflow), there's no LONG positioning to unwind → setup has no signal → losses.

Monthly evidence supports this: 2025-07 (FII return month) was the strongest non-Disc month at PF 2.20. HO Nov 2025+ recovery aligns with sustained FII inflows.

**This is structurally different from prior retirements:**

| Pattern | Disc | OOS | HO | Examples |
|---|---|---|---|---|
| Disc/OOS-favorable-regime illusion (lesson #1, #10) | high | very high ⭐ | low ❌ | capitulation_long_v2, mis_unwind |
| Disc-only overfit (cell-mining illusion #2) | high | low | low | round_number_sweep_short |
| Regulatory decay | high | high | low ❌ | circuit_release_fade_short |
| **Regime-conditioned (FII-dependent)** | high | **low (FII-out regime)** | high (FII-in regime) | **pre_results_t1_fade** |

### Original (v1) retirement evidence (preserved for traceability — version superseded by v2)

| Period | v1 n | PF_real | PF_net |
|---|---|---|---|
| Discovery | 1879 | 1.381 | 1.097 |
| OOS | 509 | 1.030 | 0.816 |
| HO | 679 | 1.440 | 1.153 |

### Code state

**No detector code was ever written.** All Phase 1-5 artifacts preserved as negative-knowledge:

- `tools/sub9_research/phase2_pre_results_t1_signature.py` — Phase 2 empirical signature
- `tools/sub9_research/sanity_pre_results_t1_fade.py` — v1 sanity (AMC-only filter)
- `tools/sub9_research/sanity_pre_results_t1_fade_v2.py` — v2 sanity (AMC + scheduled filter)
- `tools/sub9_research/phase5_pre_results_t1_validation.py` — v1 Phase 5 driver
- `tools/sub9_research/phase5_pre_results_t1_validation_v2.py` — v2 Phase 5 driver
- `reports/sub9_sanity/_phase2_pre_results_t1_signature.csv` — Phase 2 signature
- `reports/sub9_sanity/_pre_results_t1_grid_discovery.csv` — v1 Phase 4 grid
- `reports/sub9_sanity/_pre_results_t1_v2_grid_discovery.csv` — v2 Phase 4 grid
- `reports/sub9_sanity/_pre_results_t1_trades_{discovery,oos,holdout}.csv` — v1 trade ledgers
- `reports/sub9_sanity/_pre_results_t1_v2_trades_{discovery,oos,holdout}.csv` — v2 trade ledgers
- `reports/sub9_sanity/_pre_results_t1_monthly_breakdown.csv` — monthly stability evidence

### Conditions for revival

This setup should NOT be re-implemented unless:

1. **FII-flow gate is added to the universe filter** — only fire when FII net flow has been positive for the prior 30 days. Requires FII data infrastructure (NSDL FPI flow file or equivalent). Even with gating, the post-2025 data shows the setup's "active" regime is narrower than originally assumed.
2. **Mechanism revalidation in post-war regime** — re-test on fresh Holdout (Q3-Q4 2026) AFTER war volatility recedes AND FII positioning stabilizes. The HO war period showed PF 1.504 but war is a non-replicable forward state.
3. **Stationarity gate must clear** — corrected filter showed max-min PF_net of 0.375 across 3 periods. Any revival must demonstrate ≤ 0.30 across Disc/OOS/HO with the FII-gate applied.
4. **OOS+HO combined NET must be positive** — non-negotiable. The v2 result shows -Rs 290K combined, which would be production-fatal.

## Maintenance protocol
**Predecessor:** First-pass thesis (no prior version). Phase 1-5 disciplined revival chain documented in `tasks/lessons.md` 2026-05-19 #3.
**Sanity scripts (preserved):** `tools/sub9_research/phase2_pre_results_t1_signature.py`, `tools/sub9_research/sanity_pre_results_t1_fade.py`, `tools/sub9_research/phase5_pre_results_t1_validation.py`

### Original thesis

Pre-results-day T-1 institutional-de-risking SHORT. SEBI LODR Reg. 30 forces results disclosure 30+ minutes before/after trading hours → institutional desks de-risk on the trading day BEFORE results (T-1) → produces directional intraday selling pressure on names with results pending at T+0. Trade fires during T-1 trading hours, exits by EOD (MIS-compatible, not overnight).

### Universe + filters (locked from Phase 2 BEFORE Phase 4 sweep)

- MIS-eligible + cap ∈ {large_cap, mid_cap, unknown} (loose filter to include F&O liquid names like M&M/BAJAJ-AUTO that tag unknown due to missing market_cap_cr)
- `announce_class == "AMC"` (Phase 2: n=3,957 / 57.3% negative / -0.348% mean — cleanest signal with largest n)
- `prior_5d_ret_pct > 0.0` (Phase 2 Table C: momentum filter — institutions de-risk what they're sitting on profits in; prior_5d > +5% → 61% negative)
- T-1 trading day has ≥ 30 bars (data quality)

### Phase 2 empirical signature (Discovery 2023-01..2024-12, n=4,179 measured)

| Metric | Value |
|---|---|
| T-1 mean intraday return | -0.351% |
| Random-day baseline mean | +0.238% |
| Delta (signal − baseline) | **-0.589%** |
| AMC class % negative days | 57.3% (n=3,957) |
| Strongest momentum bucket | prior_5d > +5% → 61.3% negative (n=569) |

Phase 2 verdict: STRONG SIGNAL. Proceeded to Phase 4.

### Phase 4 locked cell (Discovery sweep, 192 combos)

```
window=(10:00, 12:00), SL=0.3%, T2=2.5R, T1_partial=0.5, TS=15:10
Discovery: n=1879, PF_real=1.381, PF_net=1.097, WR=52.7%, mwin=18/20 (90%), +Rs 332K
Exit mix: SL=73.7% T2=25.8% TS=0.5% SameBar=35.5%
```

Discovery passed both gates (PF ≥ 1.10, mwin ≥ 12/24). Proceeded to Phase 5.

### Phase 5 validation chain (the trap)

| Period | n | PF_real | **PF_net** | WR | mwin | Net (after tax) |
|---|---|---|---|---|---|---|
| Discovery (24mo) | 1879 | 1.381 | **1.097** | 52.7% | 18/20 | +Rs 332K |
| **OOS (9mo)** | 509 | 1.030 | **0.816** ❌ | 46.0% | 4/9 | **-Rs 283K** |
| HO (7mo) | 679 | 1.440 | **1.153** | 52.3% | 4/6 | +Rs 191K |

**OOS+HO combined (16 post-Discovery months): NET -Rs 92K LOSS.**

**HO sub-periods:**
- HO pre-war (Oct'25-Dec'25): n=186, PF_real 1.269, +Rs 23,559
- HO war (Jan'26-Apr'26): n=493, PF_real **1.504**, +Rs 117,870

War-period strength drove HO recovery. OOS (no war) was the catastrophic window.

### Ship gate evaluation

| Gate | Required | Actual | Pass? |
|---|---|---|---|
| HO PF_net | ≥ 1.10 | 1.153 | ✅ |
| HO mwin | ≥ 4/7 (57%) | 4/6 (66.7%) | ✅ |
| WR delta OOS→HO | ≤ 10pp | 6.3pp | ✅ |
| HO n | ≥ 100 | 679 | ✅ |
| **Same-bar %** | **< 30%** | **30.3%** | ❌ (by 0.3pp) |

**Technically fails 1 of 5 gates. But the REAL retirement signal is the trajectory pattern.**

### Failure mode — NEW pattern: OOS-catastrophic / HO-recovery (non-stationary edge)

Distinct from prior retirements:

| Pattern | Disc PF_net | OOS PF_net | HO PF_net | Examples |
|---|---|---|---|---|
| Disc/OOS-favorable-regime illusion | high | very high ⭐ | low ❌ | capitulation_long_v2, mis_unwind_vwap_revert_short |
| Disc-only overfit | high | low | low | round_number_sweep_short |
| Regulatory decay | high | high | low ❌ | circuit_release_fade_short |
| **Non-stationary edge (this)** | **marginal** | **low ❌** | **high** | **pre_results_t1_fade** |

**Key insight:** The Disc → OOS → HO trajectory is non-monotonic AND high-magnitude (PF_net swings 1.10 → 0.82 → 1.15). Edge appears in Discovery, vanishes in OOS, reappears in HO (war-period). Ship a setup like this and you don't know which regime you're in until 9 months of drawdown educates you. **A real edge should be approximately stationary across periods** — non-monotonic ±0.3-PF swings indicate the apparent edge is regime-conditioned, not a stable mechanism.

**Architectural fragility (Phase 4 red flag confirmed):** Tight SL 0.3% won the Phase 4 grid (classic overfit pattern). Same-bar exit rate climbed regime-sensitively: Disc 35.5% → OOS 44.0% → HO 30.3% — the entry-bar volatility-eats-stop problem is non-stationary too.

**Borrowed-mechanism risk (Phase 1 yellow flag confirmed):** No Indian academic literature documents T-1 pre-earnings institutional de-risking. Closest analog is US Ben-Rephael (2024). The Indian-market mechanism remains unproven; what we measured in Phase 2 may be a regime artifact, not a stationary participant behavior.

### Code state

**No detector code was ever written** — stopped at sanity-validation phase. Nothing to remove from production. Research artifacts preserved as negative-knowledge:

**Preserved:**
- `tools/sub9_research/phase2_pre_results_t1_signature.py` — empirical signature (4,179 Discovery events, T-1 mean -0.35% vs baseline +0.24%)
- `tools/sub9_research/sanity_pre_results_t1_fade.py` — disciplined anti-bias sanity (Mode B, path walk from i+2, same-bar SL priority, full locked filters)
- `tools/sub9_research/phase5_pre_results_t1_validation.py` — Phase 5 driver
- `reports/sub9_sanity/_phase2_pre_results_t1_signature.csv` — Phase 2 per-event signature
- `reports/sub9_sanity/_pre_results_t1_grid_discovery.csv` — 192-combo Phase 4 grid
- `reports/sub9_sanity/_pre_results_t1_trades_{discovery,oos,holdout}.csv` — locked-cell trade ledgers

### Conditions for revival

This setup should NOT be re-implemented unless:

1. **Stationary-edge demonstration** — find a configuration where Disc / OOS / HO PF_net are all within ±0.15 of each other. The 0.82-1.15 PF_net swing in this attempt is the disqualifier. Variance across periods this large = no stable mechanism.
2. **Indian-market literature backing** — locate a published Indian-market study (NSE research, academic paper, broker quant report) documenting the T-1 institutional de-risking pattern. Without it, this is borrowed-mechanism speculation.
3. **Architecturally robust SL** — current sanity locked SL=0.3% (tightest grid value), which is fee-stack-fragile and same-bar-prone. Revival needs a wider SL that survives at PF_net ≥ 1.10 across all 3 periods, not a tight scalp.
4. **Regime explanation for OOS catastrophe** — explain WHY OOS (Jan-Sep 2025) lost -Rs 283K while HO (Oct'25-Apr'26) made +Rs 191K. What mechanism produces opposite signals in adjacent periods? Until this is answered, the setup is regime-roulette.

## Maintenance protocol

**Updated 2026-05-20.** Walk-forward tier classification (RED / AMBER / GREEN) is DEPRECATED. The thresholds (≤ 5/13 windows pass = RED, etc.) were folklore, not literature-backed. See `tasks/lessons.md` #15 and `docs/setup_lifecycle.md` Stage 14.

When a new setup is retired:

1. **Read the confidence card.** Per `docs/setup_lifecycle.md` Stage 14, retirement requires evidence from `reports/confidence_cards/<setup>_confidence_card.md` generated on OCI canonical data via `tools/methodology/confidence/confidence_card.py`. Any one of these intervals is sufficient evidence — researcher confirms:
   - PF CI lower bound below 0.95 on OCI canonical with n > 500 in the most recent 12-month window
   - Adjusted Sharpe (Harvey-Liu) flips negative on OCI canonical
   - Per-regime breakdown shows the edge has collapsed to a single regime AND that regime has ended

2. Add a section here with the four standard fields: thesis, universe+filters, claimed validation, actual failure mode + evidence.

3. **Include the OCI confidence card snapshot** (or sanity card with the Lesson #13 caveat if no OCI data exists): aggregate PF CI, per-regime table, raw Sharpe + adjusted Sharpe. This is the primary retirement evidence.

4. **Include mechanism_tags + mechanism_notes** that were pre-registered (verify via `git log` on the config file). If the actual decay cause differs from the pre-registered mechanism, document the divergence — that's a Lesson candidate.

5. List the specific code files removed.

6. Add the "conditions for revival" — what would have to be true for someone to legitimately try this again. Be specific: which regime would have to return, which mechanism would have to be re-validated by fresh data.

7. Move the original brief from `specs/` to `specs/archived/` (or delete if low-value).

8. Update `tasks/lessons.md` ONLY if a new failure pattern was discovered. Repeat patterns (e.g., "regime-conditioned edge") do not warrant a new lesson.

**Historical retirement entries** were written under earlier methodologies (3-period chronological then walk-forward tier classification). They are preserved as historical record. New entries must use the confidence framework.

The point of this document is **negative knowledge** — what does NOT work and why. It is at least as valuable as the active-setup documentation.
