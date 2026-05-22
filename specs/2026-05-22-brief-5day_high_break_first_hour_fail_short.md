# `5day_high_break_first_hour_fail_short` — Stage 0 brief

**Date:** 2026-05-22
**Stage:** 0 — Idea (awaiting Phase 1)
**Branch:** `research/2026-05-22-new-setups-batch`
**Predecessor:** 4-KILL brainstorm continuation (5th candidate today); first multi-day-signal candidate in this batch
**Direction:** SHORT
**Window:** Intraday MIS (square 15:25). Signal scan 09:30-11:30; entry at failure bar (typically 10:00-12:00); exit 13:30.
**Portfolio rationale:** First multi-day-signal → intraday-trigger candidate this batch. Active `circuit_t1_fade_short` proves multi-day-event + intraday-trigger works in this system; this candidate extends to multi-day-TECHNICAL-LEVEL (5-day high) signal, untouched territory.

## 1. Mechanism statement (ONE sentence)

Small/mid-cap NSE MIS-eligible stocks (cap_segment ∈ {small_cap, mid_cap}, large_cap and unknown excluded per 2026-05-22 C-H / lunch_lull learnings) where `5day_high / 5day_T-1_close >= 1.02` (multi-day uptrend with real resistance level visible to retail momentum chasers) that BREAK above the prior-5-day-high during 09:30-11:30 intraday but FAIL to hold above that level within the next 30 minutes (close falls back below `5day_high * 0.999`) signal that the multi-day breakout was thin retail FOMO into stale resistance distribution; SHORT-fade at the fail bar's close, exit at 13:25-13:30 5m bar close as the failure plays out into the post-lunch high-volume window.

## 2. Falsifiers (3)

1. **Mechanism falsifier (volume signature on break vs fail):** Thesis = break is retail FOMO (elevated volume), fail is supply distribution (rejection on equal-or-heavier volume). Test: across 200+ fires, break-bar `vol_ratio` median > 1.5× cumulative-prior-mean AND fail-bar `vol_ratio` median >= break-bar `vol_ratio`. If break-bar vol_ratio is unremarkable (median < 1.2×), the break wasn't FOMO and the mechanism is wrong → KILL.

2. **Regime falsifier (retail momentum concentration):** Depends on retail momentum-chasing concentration in small/mid-cap multi-day breakouts. During FII-exit regimes (R4) retail risk-on flow drops; war-vol regimes (R7) change retail behavior. Per-regime PF CI lower bound > 1.0 must hold in at least 4 of 7 regimes including R1, R2, R5. If <4 → KILL.

3. **Infra falsifier (technical-level visibility):** Mechanism depends on the 5-day high being a widely-watched retail level. If a major broker/scanner platform removes the 5-day-high indicator (or institutional algorithmic flow systematically arbitrages multi-day breakouts at sub-second resolution), the retail-trap dynamic shrinks. Monitor for major changes to Indian retail trading-platform tooling (Streak, Tradetron, Zerodha) during 2024-2026 validation.

## 3. Adjacent setups + correlation/effective-M

| Setup | Status | Direction | Window | Mechanism overlap | M penalty |
|---|---|---|---|---|---|
| `or_window_failure_fade_short` | active | SHORT | 09:30-10:30 entry | Similar trigger (level break + fail) but uses OPENING RANGE high, not 5-day high. Levels differ for most stocks; overlap expected ~15-25% on same-symbol fires | **1.0-1.5** |
| `gap_fade_short` | active | SHORT | 09:25-10:00 | Gap-based trigger, different mechanic | 0.3 |
| `circuit_t1_fade_short` | active | SHORT | 10:30 entry | Multi-day signal (T-1 circuit), different price-level mechanic | 0.5 |
| `lunch_lull_breakout_fail_short` | KILLED today | n/a | 11:30-13:00 | Same fail-fade family but lunch-lull-specific; KILLED → not portfolio overlap | 0 |

**Effective M estimate: 1.0-1.5** primarily against `or_window_failure_fade_short`. Mitigation: explicitly check break-level (5-day high) vs OR-high overlap in Phase 2 telemetry. If >50% of signals coincide with same-symbol OR breaks within the same day, mechanism is partially derivative.

## 4. Phase 1 outline (Gate A + Gate B)

### Gate A — Indian sources to find (≥2 required)

1. **intradaylab.com** — multi-day high break failure / "fake breakout at multi-day resistance" patterns specific to NSE small/mid-cap
2. **Zerodha Varsity (Technical Analysis)** — multi-day resistance break-and-fail framework + retail trap psychology
3. **TradingView India** — Indian retail-facing breakout-failure scanner mentions
4. **SEBI 2024 retail study** — confirm retail concentration in multi-day-breakout small/mid-cap names
5. **Indian broker quant blogs** — Motilal Oswal / Edelweiss intraday-momentum-failure patterns

Acceptance: ≥1 explicit Indian-source operationalization of "multi-day high break + intraday fail = SHORT fade in small/mid-cap retail-driven names" + the supporting technical/retail context.

### Gate B — Data feasibility (predicted PASS)

| Required data | On disk? |
|---|---|
| 5m bars per symbol | ✅ |
| `consolidated_daily.feather` (for 5-day high computation T-5..T-1) | ✅ |
| `cap_segment` metadata (small/mid only) | ✅ |
| `ProductionUniverseGate` (Lesson #19) | ✅ |

## 5. Phase 2 plan (preview)

- **Universe:** `cap_segment ∈ {small_cap, mid_cap}` AND MIS-eligible AND `ProductionUniverseGate` per-date AND 5-day daily-data available
- **Multi-day filter:** `5day_high = max(daily.high for T-5..T-1)` from consolidated_daily; `5day_T-1_close = daily.close[T-1]`. Require `5day_high / 5day_T-1_close >= 1.02` (real momentum).
- **Intraday signal (SHORT entry):** on T+0 5m bars 09:30-11:30, find FIRST bar where `bar.high > 5day_high`. Track next 6 bars (30 min). If any of those bars has `close < 5day_high * 0.999`, that's the FAIL bar — record signal at FAIL bar's close.
- **Intraday baseline (control):** same universe + same break condition but NO fail in next 30 min (held above resistance). Record at `min(break_bar_idx + 6, last_bar_in_1130)` close.
- **Target return:** `ret_to_1330 = (close_at_1325 - signal_close) / signal_close * 100` (SHORT direction → negative is good)
- **Acceptance:** drift delta ≤ -0.15% AND `n_signal >= 200`
- **Required splits:** pre/post-2024, pre/post-SEBI-Oct-2025, cap=small vs cap=mid, break-bar-vol-ratio buckets (<1.0, 1.0-1.5, 1.5-2.0, ≥2.0), OR-high-overlap (signal coincides with OR high or not)

## 6. Status checklist

- [ ] Gate A — ≥2 Indian sources cited
- [ ] Gate B — data feasibility confirmed (predicted PASS)
- [ ] Universe excludes large_cap AND unknown (carry-over from C-H + lunch_lull learnings)
- [ ] Phase 2 OR-high-overlap telemetry planned (M-penalty validation)
- [ ] Break-vs-fail volume Falsifier #1 pre-registered

## 7. Next action

Phase 1 research — dispatch agent per proven template.
