# `lunch_lull_breakout_fail_short` — Stage 0 brief

**Date:** 2026-05-22
**Stage:** 0 — Idea (awaiting Phase 1)
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

- [ ] Gate A — sources cited (Monash + 1+ Indian retail/pro operationalization)
- [ ] Gate B — data feasibility confirmed
- [ ] Universe excludes large_cap (carry-over from C-H 2026-05-22 finding)
- [ ] Phase 2 source-window-cohort splits planned
- [ ] M penalty noted vs `or_window_failure_fade_short` (1.5-2.0)

## 7. Next action

Phase 1 research (Gate A + Gate B) — dispatched as parallel research agent per the proven Phase 1 template from today's session.
