# `5day_decile_leader_morning_reject_short` — Stage 0 brief

**Date:** 2026-05-22
**Stage:** 0 — Idea (awaiting Phase 1)
**Branch:** `research/2026-05-22-new-setups-batch`
**Predecessor:** Multi-day-signal batch #3 of 5
**Direction:** SHORT
**Window:** Intraday MIS (square 15:25). Signal 10:25-10:30 (first-hour close); entry 10:30 close; exit 13:30.
**Portfolio rationale:** Cross-sectional rank-based multi-day signal (untouched signal class). Captures crowded-momentum positioning saturation that intraday OR-failure setups don't gate on.

## 1. Mechanism statement (ONE sentence)

Small/mid-cap NSE MIS-eligible stocks (cap ∈ {small_cap, mid_cap}) that as of T-1 close rank in the TOP DECILE of 5-day cumulative return WITHIN their cap_segment (cross-sectional retail momentum-chase leaders, positioning saturated) AND on T+0 the 10:25-10:30 5m bar close < the 09:15 5m bar open (first-hour failure to extend the multi-day momentum) signal that retail FOMO-chasing is exhausted at the open and that profit-taking + stop-loss cascades take over; SHORT-fade at the 10:30 close, exit 13:30.

## 2. Falsifiers (3)

1. **Mechanism falsifier (decile-rank stability):** Thesis requires "decile leaders" to be a STABLE crowd, not a noisy day-to-day shuffle. Test: across 100+ days, fraction of T-1 top-decile stocks that were ALSO in T-2 top decile should be ≥ 60%. If stability < 50%, the signal is rank-noise, not positioning-saturation → KILL.

2. **Regime falsifier (retail momentum flow):** Depends on retail momentum-chasing concentration. FII-exit regimes (R4) suppress retail flow. Per-regime PF CI lower bound > 1.0 must hold in ≥ 4 of 7 regimes including R1, R2, R5. If <4 → KILL.

3. **Infra falsifier (screener-tool changes):** Mechanism depends on retail screening for momentum leaders via tools like Streak / Tradetron / Chartink. Material changes to these tools' default screeners (or institutional algos arbitraging the screener output at sub-second resolution) shrink the trap.

## 3. Adjacent setups + correlation/effective-M

| Setup | Status | Direction | Window | Mechanism overlap | M penalty |
|---|---|---|---|---|---|
| `or_window_failure_fade_short` | active | SHORT | 09:30-10:30 | **Same window + same direction**, different trigger (OR break vs cross-sectional rank). Same-symbol overlap possible if decile leader also breaks OR | **1.0** |
| `gap_fade_short` | active | SHORT | 09:25-10:00 | Different trigger (gap-only); some overlap on decile leaders that gap up | 0.5 |
| `5day_high_break_first_hour_fail_short` | this batch | SHORT | 09:30-11:30 | Both fire on first-hour failure but different gating (multi-day high vs cross-sectional rank). Independent gates. | 0.5 |

**Effective M estimate: 1.0** vs `or_window_failure_fade_short`. Phase 2 telemetry must report overlap fraction.

## 4. Phase 1 outline (Gate A + Gate B)

### Gate A — Indian sources to find (≥2 required)

1. **intradaylab.com** — Indian momentum-chasing failure patterns / crowded-long unwind
2. **Streak / Tradetron docs** — momentum-leader screening behavior + retail usage statistics
3. **SEBI 2024 retail study** — retail concentration in small/mid-cap momentum leaders + 89% F&O loss rate context
4. **Indian broker quant blogs** — multi-day momentum leader breakdown documentation

Acceptance: ≥1 explicit Indian operationalization of "cross-sectional momentum leader + first-hour rejection = SHORT fade" + retail-screener behavior context.

### Gate B — Data feasibility (predicted PASS)

| Required data | On disk? |
|---|---|
| 5m bars per symbol | ✅ |
| `consolidated_daily.feather` (cross-sectional 5-day return rank) | ✅ |
| `cap_segment` metadata (per-segment ranking) | ✅ |
| `ProductionUniverseGate` (Lesson #19) | ✅ |

## 5. Phase 2 plan (preview)

- **Universe:** `cap_segment ∈ {small_cap, mid_cap}` AND MIS-eligible AND `ProductionUniverseGate` per-date AND 5-day daily-data available.
- **Multi-day rank computation:** for each T-1, compute `5day_cumret[sym] = (daily.close[sym,T-1] / daily.close[sym,T-6]) - 1`. Within each cap_segment cohort, compute the 90th-percentile threshold of `5day_cumret`. A stock is a "decile leader" if `5day_cumret >= 90th_percentile_of_its_cap_segment`. NO look-ahead: rank uses T-6..T-1 daily data only.
- **Intraday signal (SHORT entry):** at T+0 10:30 (the 10:25-10:30 5m bar), check `bars_at_1025_1030.close < bars_at_0915_0920.open`. Signal at the 10:25-10:30 bar's close.
- **Baseline (control):** same decile-leader universe but `close_at_1030 >= open_at_0915` (first-hour continued, no failure). Use the 10:25-10:30 bar's close as no-signal anchor.
- **Target return:** `ret_to_1330 = (close_at_1325 - signal_close) / signal_close * 100`. SHORT → negative is good.
- **Acceptance:** drift `<= -0.15%` AND `n_signal >= 200`.
- **Required splits:** pre/post-2024, pre/post-SEBI-Oct-2025, cap=small vs cap=mid, 5-day-return magnitude buckets (5-10%, 10-15%, >15%), OR-overlap (also fired `or_window_failure_fade_short` same day?).

## 6. Status checklist

- [ ] Gate A — ≥2 Indian sources cited
- [ ] Gate B — data feasibility confirmed (predicted PASS)
- [ ] Universe excludes large_cap AND unknown
- [ ] Phase 2 OR-overlap telemetry planned (M penalty validation)
- [ ] Decile-rank stability Falsifier #1 pre-registered

## 7. Next action

Phase 1 research — parallel dispatch with batch.
