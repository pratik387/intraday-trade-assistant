# `5day_volume_buildup_morning_distribution_short` — Stage 0 brief

**Date:** 2026-05-22
**Stage:** 0 — Idea (awaiting Phase 1)
**Branch:** `research/2026-05-22-new-setups-batch`
**Predecessor:** Multi-day-signal batch #4 of 5
**Direction:** SHORT
**Window:** Intraday MIS (square 15:25). Signal 10:25-10:30 first-hour close; entry 10:30 close; exit 13:30.
**Portfolio rationale:** Volume-magnitude multi-day signal — under-tested in the system. The retired `delivery_pct_anomaly_short` used EOD-volume signal, never an intraday-fail companion. This candidate fills the gap.

## 1. Mechanism statement (ONE sentence)

Small/mid-cap NSE MIS-eligible stocks (cap ∈ {small_cap, mid_cap}) where 5-day cumulative volume z-score is ≥ +2σ vs trailing-60-day rolling-5-day-sum baseline (multi-day "accumulation" signature retail interprets as bullish per Zerodha Varsity's "volume confirms price" framework) AND T+0 first-hour close (10:30) falls in the LOWER HALF of the first-hour range (close ≤ (open_at_0915 + low_in_first_hour) / 2) signal that the multi-day volume buildup was actually institutional distribution into retail bid; SHORT-fade at the 10:30 bar's close, exit 13:30.

## 2. Falsifiers (3)

1. **Mechanism falsifier (delivery% corroboration):** Distribution thesis should show LOWER delivery% on signal days than the multi-day-volume universe average (institutional distribution leaves higher non-delivery / intraday-trade share). Test: signal cohort median delivery% < baseline cohort median delivery% (NSE EOD delivery% from `consolidated_daily.feather`). If signal delivery% ≥ baseline, the volume buildup was real accumulation → KILL.

2. **Regime falsifier (retail volume interpretation):** Depends on retail reading multi-day volume buildup as buy-signal. FII-exit regimes (R4) suppress retail interpretation flow. Per-regime PF CI lower bound > 1.0 must hold in ≥ 4 of 7 regimes including R1, R5.

3. **Infra falsifier (NSE volume reporting):** Mechanism depends on NSE EOD volume disclosure pattern that retail watches. Material changes to delivery%/volume reporting timing (e.g., real-time delivery% during session) collapse the asymmetry.

## 3. Adjacent setups + correlation/effective-M

| Setup | Status | Direction | Mechanism overlap | M penalty |
|---|---|---|---|---|
| `or_window_failure_fade_short` | active | SHORT | Same window + direction, different trigger (no volume gating) | 0.7 |
| `delivery_pct_anomaly_short` | DISABLED | SHORT | Same volume-family signal source (delivery% + volume z); different timing (EOD-fade) | 0.3 |
| `5day_high_break_first_hour_fail_short` | this batch | SHORT | Both first-hour-fail family, different multi-day gate | 0.5 |
| `5day_decile_leader_morning_reject_short` | this batch | SHORT | Both first-hour-fail family, different multi-day gate | 0.5 |

**Effective M estimate: 0.7** vs or_window_failure. Within-batch M is moderate (3 SHORT setups using first-hour failure with different multi-day gates — Phase 5 confidence card must apply Bonferroni haircut at M=3+).

## 4. Phase 1 outline (Gate A + Gate B)

### Gate A — Indian sources to find (≥2 required)

1. **Zerodha Varsity (Volume chapter)** — "volume confirms price" framework that retail uses to interpret volume buildup
2. **intradaylab.com** — distribution-into-retail-bid patterns / "fake accumulation" descriptions
3. **NSE delivery% / cash-segment volume documentation** — institutional vs retail volume split mechanics
4. **SEBI 2024 retail study** — retail interpretation of volume signals + 76% loss rate context
5. **Indian academic studies on NSE delivery% as institutional-flow proxy** — academic backing for delivery% as a distribution indicator

Acceptance: Zerodha Varsity (already cited in `lunch_lull` Phase 1 PASS) + ≥1 Indian operationalization of "multi-day volume buildup + intraday distribution = SHORT fade."

### Gate B — Data feasibility (predicted PASS)

| Required data | On disk? |
|---|---|
| 5m bars per symbol | ✅ |
| `consolidated_daily.feather` (5-day volume sum + delivery%) | ✅ |
| `cap_segment` metadata | ✅ |
| `ProductionUniverseGate` (Lesson #19) | ✅ |
| 60-day rolling baseline computation feasibility | ✅ (standard pandas rolling) |

## 5. Phase 2 plan (preview)

- **Universe:** `cap_segment ∈ {small_cap, mid_cap}` AND MIS-eligible AND `ProductionUniverseGate` per-date AND 65+ daily bars available (for 60-day baseline + 5-day sum).
- **Multi-day filter:** for each T-1, compute `5day_vol_sum[T-5..T-1] = sum(daily.volume[T-5..T-1])`. Compute rolling-5-day-sum series for days T-65..T-1. `60day_baseline_mean = mean(rolling_5day_sums[T-65..T-6])` (60-day baseline EXCLUDING current 5-day window, no overlap). `60day_baseline_std = std(...)`. Require `(5day_vol_sum - 60day_baseline_mean) / 60day_baseline_std >= 2.0`.
- **Intraday signal (SHORT entry):** at T+0 10:30 (10:25-10:30 5m bar), compute `first_hour_low = min(bars.low for 09:15-10:30)` and `first_hour_open = bars[0915-0920].open`. Trigger when `bars[1025-1030].close <= (first_hour_open + first_hour_low) / 2`. Signal at 10:30 close.
- **Baseline:** same multi-day-vol-buildup universe but `close_at_1030 > (open + first_hour_low) / 2` (upper-half close, no distribution signature). Use 10:30 close as anchor.
- **Target:** `ret_to_1330`. SHORT.
- **Acceptance:** drift `<= -0.15%` AND `n_signal >= 200`.
- **Required splits:** pre/post-2024, pre/post-SEBI-Oct-2025, cap=small vs cap=mid, vol-z buckets (2.0-2.5, 2.5-3.0, ≥3.0), signal vs baseline delivery% (Falsifier #1 corroboration).

## 6. Status checklist

- [ ] Gate A — ≥2 Indian sources cited
- [ ] Gate B — data feasibility confirmed
- [ ] Universe excludes large_cap AND unknown
- [ ] Phase 2 delivery%-corroboration telemetry planned (Falsifier #1)
- [ ] Within-batch M=3+ tracked for Phase 5 confidence card

## 7. Next action

Phase 1 research — parallel dispatch with batch.
