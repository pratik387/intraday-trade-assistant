# `lunch_lull_breakout_fail_short` — Stage 0 brief

> **STATUS: KILLED at Phase 2 (2026-05-22)** — preserved as negative knowledge.
>
> **Phase 2 verdict — KILL.** Falsifier #1 PASS (vol_ratio median 0.71, thin-volume signature real). Drift delta **+0.0624% — WRONG SIGN** (SHORT required ≤ -0.15%).
>
> **Phase 2 evidence (script: `tools/sub9_research/phase2_lunch_lull_breakout_fail_signature.py`, output: `reports/sub9_sanity/_phase2_lunch_lull_breakout_fail_signature.csv`, 103,484 rows):**
>
> | Metric | Value | Pass? |
> |---|---|---|
> | n_signal | 23,233 | ✅ (200 floor) |
> | n_baseline | 80,251 | — |
> | vol_ratio median | 0.71 | ✅ Falsifier #1 (< 1.0) |
> | Fraction vol_ratio >= 1.0 | 20.04% | ✅ (< 40%) |
> | Signal mean ret_to_1430 | -0.002% | — |
> | Baseline mean ret_to_1430 | -0.064% | — |
> | **DRIFT DELTA** | **+0.0624%** (WRONG SIGN) | ❌ |
>
> **All cohort splits POSITIVE delta** (no SHORT footprint anywhere):
>
> | Cohort | n | delta |
> |---|---|---|
> | pre_2024 | 4,600 | +0.045% |
> | post_2024 | 18,633 | +0.060% |
> | pre_sebi_oct2025 | 17,485 | +0.040% |
> | post_sebi_oct2025 | 5,748 | +0.100% |
> | cap=small_cap | 11,880 | +0.055% |
> | cap=mid_cap | 11,353 | +0.064% |
>
> vol_ratio buckets NON-MONOTONIC — the thin-volume signature is real but doesn't drive directional drift.
>
> **Indian retail consensus was literally right:** "lunch trades fail" — they fail by going sideways or slightly UP, not DOWN. The Bayesian inferential step from cited sources to "post-lunch fade direction" (flagged in Phase 1 as raised-bar) was empirically wrong. Lesson reinforces: don't extrapolate direction beyond what sources explicitly state.
>
> **Inverse-edge check (Lesson #1):** aggregate +0.0624% is too small for LONG either (need +0.15%). Edge is too thin to survive Indian fee stack in either direction.
>
> **Conditions for revival:** would need a fundamentally different signal definition that produces directional drift >= |0.15%| at 5m resolution. The current "fresh intraday-high in 11:30-13:00 on thin volume" trigger does NOT predict either direction. Mid-day SHORT setups using a different trigger (e.g., explicit news/event-driven gap-up at midday rather than thin-volume drift) are not pre-empted by this kill.
>
> ---

> **STATUS: Phase 1 PASS (2026-05-22)** — advance to Phase 2.
>
> **Gate A — 3 Indian sources operationalize the mechanism:**
> 1. **Monash NSE liquidity paper** (cited in `tasks/lessons.md` #4) — J/U-shape volume profile confirms 11:00-13:00 = ~1.0× baseline LOW
> 2. **TradeSmart Online (Indian broker blog)** — explicit on NSE midday: "Volumes drop sharply... false signals; trend trades often fail" + names 13:00-14:30 as "afternoon trend session, second best window for intraday trading"
> 3. **Zerodha Varsity (Technical Analysis — Volume chapter)** — operationalizes the bull-trap signature: "price increases but volume decreases → Caution – weak hands buying... small retail participation... possible bull trap"
>
> **Mechanism components covered:**
> - (a) Lunch-lull = low-volume window: YES (Monash + TradeSmart + Lesson #4)
> - (b) Intraday-high on below-baseline vol = thin retail FOMO / bull trap: YES (Zerodha Varsity explicit)
> - (c) Post-lunch high-vol window reverses unconfirmed lunch breakouts: **PARTIAL** — Indian sources say "avoid midday, re-engage post-13:30 for trend continuation"; the SHORT-fade direction is one Bayesian inference beyond cited material (failed lunch breakouts MEAN-REVERT because the trend was never real). Phase 2 must validate the fade direction empirically.
>
> **Gate B — Data feasibility: PASS** (5m feathers, consolidated_daily, get_cap_segment, ProductionUniverseGate all confirmed)
>
> **Capacity outlook: POSITIVE.** Mainstream Indian retail advice is "avoid midday trades" → the contrarian fade side is likely under-arbed precisely because retail sits out. Low arbitrage risk.
>
> **Universe sizing:** 892 small+mid caps in consolidated_daily.feather → 150-300/day effective universe after ProductionUniverseGate → comfortable for `n_signal ≥ 200` over 2023-2026.
>
> **Phase 2 evidentiary bar (raised due to partial-(c)):**
> - Falsifier #1 (vol_ratio median < 1.0) MUST be computed and reported BEFORE any drift-delta peeking
> - Drift-delta threshold ≤ -0.15% holds as written
>
> ---

**Date:** 2026-05-22
**Stage:** 0 → 1 (Phase 1 complete, Phase 2 dispatched)
**Branch:** `research/2026-05-22-new-setups-batch`
**Predecessor:** Brainstorming continuation of 2026-05-22 session, post 3-KILL batch
**Direction:** SHORT
**Window:** Intraday MIS (square 15:25). Signal 11:30-13:00, exit 13:30-14:30.
**Portfolio rationale:** Mid-day window (11:00-14:00) has zero active or paper-pending setups. All active SHORTs operate in open window (09:15-10:30). Adds time-diversification without overlapping any current trigger.

## 1. Mechanism statement (ONE sentence)

Small/mid-cap NSE MIS-eligible stocks (cap_segment ∈ {small_cap, mid_cap}, **explicitly excluding large_cap** per 2026-05-22 C-H kill which proved large_caps drift UP in 11-13 window) that print a fresh intraday high in the 11:30-13:00 IST lunch-lull window (Monash NSE liquidity paper: 11:00-13:00 is the J/U-shape volume LOW at 1.0× baseline vs 5× at open) with `vol_ratio < 1.2×` cumulative-prior-mean (volume BELOW baseline confirms the breakout is thin retail FOMO into low-liquidity, not real institutional demand) get SHORT-faded in 13:30-14:30 as the post-lunch high-volume window restarts and the unconfirmed break reverts.

## 2. Falsifiers (3 conditions that would invalidate)

1. **Mechanism falsifier (volume signature):** Thesis requires breakouts to be on BELOW-baseline volume. Test: across 200+ fires, signal-bar `vol_ratio` distribution should have median < 1.0×. If median >= 1.0× OR if >40% of fires have `vol_ratio >= 1.0`, the breakout is real demand and the SHORT fade fails → KILL.

2. **Regime falsifier (retail concentration):** Mechanism depends on retail FOMO concentration in small/mid-cap. During FII-exit regimes (R4) or war-vol regimes (R7), retail risk-on flow drops. Per-regime PF CI lower bound > 1.0 must hold in at least 4 of 7 regimes including R1, R2, R5. If <4 → KILL.

3. **Infra falsifier (NSE J/U volume profile):** Mechanism depends on stable J/U-shape intraday volume profile. If NSE changes session structure (e.g., pre-lunch break, continuous trading, mid-day auction) that flattens the lunch-lull volume LOW, the "thin-volume FOMO" signature disappears. Monitor NSE circulars during 2024-2026 validation.

## 3. Adjacent setups + correlation/effective-M

| Setup | Status | Direction | Window | Mechanism overlap | M penalty |
|---|---|---|---|---|---|
| `or_window_failure_fade_short` | active | SHORT | 09:30-10:30 | **Same family** (intraday-high failure fade) | **1.5-2.0** (highest in portfolio) |
| `gap_fade_short` | active | SHORT | 09:25-10:00 | Different trigger (gap, not intraday-high) | 0.5 |
| `circuit_t1_fade_short` | active | SHORT | 10:30 entry | Different trigger (circuit) | 0.3 |
| `long_panic_gap_down` | active | LONG | 09:15-09:20 | Opposite direction | 0 |

**Effective M estimate (Harvey-Liu input):** 1.5-2.0 against `or_window_failure_fade_short`. This is the highest correlation in today's brainstorm — both setups exploit "intraday high prints with weak follow-through." Phase 5 confidence card must apply Bonferroni haircut at M=2+ to discount selection bias.

**Mitigation:** explicitly different time window (no entry-time overlap), explicitly different universe band (or_window fires on full universe; this candidate restricts to small/mid only). Trade-overlap on same symbols is possible if a stock prints intraday-high failure both in OR window AND lunch-lull — should be rare (~5% expected) but worth tracking in Phase 2 telemetry.

## 4. Phase 1 outline (Gate A + Gate B)

### Gate A — Indian sources to find (≥2 required)

1. **Monash NSE liquidity paper** — already cited in Lesson #4 for J/U-shape volume profile (11:00-13:00 = 1.0× baseline)
2. **intradaylab.com** — "lunch-lull false breakout" or "noon breakout failure" pattern catalog
3. **Zerodha Varsity (Module 5)** — volume-confirmation requirement for intraday breakouts
4. **SEBI 2024 retail study** — retail FOMO concentration in low-volume windows

Acceptance: Monash + 1 of [intradaylab / Varsity / SEBI] explicitly operationalizes "intraday-high breakouts on below-baseline volume fail in next high-vol window."

### Gate B — Data feasibility (preliminary check)

| Required data | On disk? |
|---|---|
| 5m bars per symbol | ✅ |
| `cap_segment` metadata (small/mid only) | ✅ |
| Cumulative-prior-volume baseline (Lesson #5 #2 — exclude current bar) | ✅ |
| `ProductionUniverseGate` (Lesson #19) | ✅ |
| `consolidated_daily.feather` for ADV gate | ✅ |

**Verdict (predicted):** Gate B clearly passes. No new data needed.

## 5. Phase 2 plan (preview)

- **Universe:** `cap_segment ∈ {small_cap, mid_cap}` AND MIS-eligible AND `ProductionUniverseGate` passes per-date
- **Signal:** for each (sym, date), find FIRST 5m bar in 11:30-13:00 IST where `bar.high > max(bar.high for prior intraday bars TODAY)` (fresh intraday high using bars[:i+1] only — Lesson #5 #1) AND `bar.volume / mean(volume of prior intraday bars EXCLUDING current bar) < 1.2`
- **Baseline:** same universe + same intraday-high condition + `vol_ratio >= 1.2` (volume-confirmed control)
- **Target return:** `ret_to_1430 = (close_at_1425 - signal_close) / signal_close * 100` (SHORT direction → negative is good)
- **Acceptance:** drift delta `(signal_mean - baseline_mean) <= -0.15%` AND `n_signal >= 200`
- **Required splits:** pre/post-2024, pre/post-SEBI-Oct-2025, cap=small vs cap=mid, vol_ratio buckets (<0.8 / 0.8-1.0 / 1.0-1.2)
- **C-H carry-over check:** explicitly verify large_cap is EXCLUDED from universe (re-running the bug would re-create the C-H finding)

## 6. Status checklist

- [x] Gate A — sources cited (Monash + TradeSmart + Zerodha Varsity)
- [x] Gate B — data feasibility confirmed
- [x] Universe excludes large_cap (carry-over from C-H 2026-05-22 finding)
- [x] Phase 2 source-window-cohort splits planned
- [x] M penalty noted vs `or_window_failure_fade_short` (1.5-2.0)
- [x] **Phase 2 pre-registration: vol_ratio median computed and reported BEFORE drift-delta peek** — Falsifier #1 PASS (median 0.71); discipline held
- [x] **Phase 2 verdict: KILL** — drift delta +0.0624% (wrong sign for SHORT)

## 7. Next action

KILLED. Brief preserved as negative knowledge per maintenance protocol. No detector code written, no production wiring.
