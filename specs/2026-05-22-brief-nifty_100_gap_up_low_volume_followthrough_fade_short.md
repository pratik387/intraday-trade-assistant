# `nifty_100_gap_up_low_volume_followthrough_fade_short` — Stage 0 brief

**Date:** 2026-05-22
**Stage:** 0 — Idea (awaiting Phase 1)
**Branch:** `research/2026-05-22-new-setups-batch`
**Predecessor:** Post 11-KILL meta-reframe (2026-05-22). Refines the gap-fade family with a NUANCED institutional-flow gate (volume-followthrough failure) AND a large-cap universe shift.
**Direction:** SHORT
**Window:** Intraday MIS (square 15:25). Signal at 10:15 (= post-first-hour); entry at 10:15 close; exit 13:30.
**Portfolio rationale:** First large-cap SHORT setup (active `gap_fade_short` targets small-cap). Volume-followthrough gate is novel; not retail-screenable.

## 1. Mechanism statement (ONE sentence)

NIFTY 100 stocks (top-100 by market cap, F&O-eligible) that GAP UP >= +0.8% at 09:15 open (from prior-day close) AND have a first-hour (09:15-10:15) cumulative volume LESS THAN 0.7× same-stock prior-5-day same-window average (low followthrough — the gap was retail/headline-trader / pre-market FOMO driven, NOT real institutional flow) SHORT-fade at the 10:15 5m bar's close as institutional supply concentrates in 10:30-13:30 to fade the un-absorbed retail gap, exit at 13:30 (close of 13:25-13:30 bar = wall-clock 13:30).

## 2. Falsifiers (3)

1. **Volume-followthrough signature (institutional-flow tell):** Mechanism requires the first-hour volume to be GENUINELY LOW (institutional skip). Test: across 200+ fires, signal-cohort `first_hour_vol_ratio_vs_5day` median <= 0.7× (by construction). Additionally: SUBSEQUENT 11:00-13:30 cumulative volume should be ELEVATED (>= 1.0× of 5d-same-window-avg) on FADE days — that's the institutional sell. If 11:00-13:30 volume STAYS low, there's no institutional fade either → KILL.

2. **Direction (post-signal drift must be negative):** signed-mean ret_to_1330 (signal_close → 13:25_close) must be <= -0.20% for the SHORT thesis. Indian fee-stack at 5x MIS leverage needs ~0.4% gross to survive. If aggregate drift > -0.15%, mechanism wrong → KILL.

3. **Regime (large-cap headline-flow concentration):** Depends on pre-market headline-trading patterns. During war-vol regimes (R7) or FII-exit regimes (R4), large-cap gap-ups may be different. Per-regime PF CI lower bound > 1.0 in >= 4 regimes.

## 3. Adjacent setups + correlation/effective-M

| Setup | Status | Direction | Mechanism overlap | M penalty |
|---|---|---|---|---|
| `gap_fade_short` | active | SHORT | Same family (gap-fade) but DIFFERENT universe (small-cap vs large-cap NIFTY 100) and ADDITIONAL gate (low first-hour volume). Triggers are mutually exclusive by cap_segment. | **0.3-0.5** |
| `or_window_failure_fade_short` | active | SHORT | Same direction, similar window (09:30-10:30 vs 10:15 entry), different trigger (OR-break-fail vs gap+vol-gate) | 0.3 |
| `nifty_heavy_vwap_reclaim_long` | KILLED today | LONG | Opposite direction; KILLED | 0 |

**Effective M estimate:** 0.3-0.5 vs `gap_fade_short` (different cap segment is the main differentiator; volume gate adds independence). Phase 5 confidence card with Bonferroni at M=1.

## 4. Phase 1 outline (Gate A + Gate B)

### Gate A — Indian sources to find (≥2 required)

1. **intradaylab.com** — "low-volume gap fade" / "fake gap-up" in large-cap NSE
2. **Zerodha Varsity Volume chapter** — "low-volume rally" / "fakeout" framework (volume confirms price)
3. **SEBI 2024 retail study** — large-cap pre-market headline-trading concentration
4. **Indian broker quant reports (Motilal Oswal, Edelweiss)** — gap-fade pattern operationalization on Nifty stocks
5. **Chartink / TradingView India scanners** — verify the EXACT low-volume-gap-up filter is NOT in standard scanners (preserve edge from retail)

Acceptance: ≥1 Indian source operationalizes "gap-up + low first-hour volume = institutional skip → fade" on large-cap. The volume-followthrough gate must NOT be a standard retail scanner (verify by searching Chartink top-50 screeners).

### Gate B — Data feasibility (predicted PASS with one-time infra)

| Required data | On disk? |
|---|---|
| 5m bars per symbol | ✅ |
| `consolidated_daily.feather` (for PDC + 5-day baseline) | ✅ |
| NIFTY 100 list | ⚠️ Need build (~30 min) |
| `ProductionUniverseGate` (Lesson #19) | ✅ |

## 5. Phase 2 plan (preview)

- **Universe:** NIFTY 100 (top-100 by market cap as of T-1), F&O-eligible, `ProductionUniverseGate` per-date, ≥6 daily bars
- **Multi-day filter:** none (pure intraday event-conditional)
- **Gap signal:** for each (sym, T+0), gap_pct = `(open_at_0915 / daily.close[T-1]) - 1`. Require `gap_pct >= 0.008`.
- **Volume gate:** `first_hour_cum_vol = sum(volume for bars 09:15-10:15)`. `5day_first_hour_avg = mean(first_hour_cum_vol for prior 5 trading days, same window, same symbol)`. Require `first_hour_cum_vol / 5day_first_hour_avg <= 0.7`.
- **Signal:** if both conditions hold, signal at the 10:10-10:15 bar's close (Mode A — entry at signal bar close). Walk from bars[i+1] for exit.
- **Baseline:** same gap-up >= 0.8% universe but `first_hour_vol_ratio > 0.7` (volume-confirmed control). Anchor 10:15 close.
- **Target:** `ret_to_1330 = (close_at_1325 - signal_close) / signal_close * 100`. SHORT direction → negative is good.
- **Acceptance:** drift <= -0.15% AND n_signal >= 200
- **Required splits:** pre/post-2024, pre/post-SEBI-Oct-2025, gap-magnitude buckets (0.8-1.5%, 1.5-2.5%, ≥2.5%), 11:00-13:30 follow-on volume sub-cohorts (Falsifier #1 #2 corroboration)

## 6. Status checklist

- [ ] Gate A — ≥2 Indian sources cited
- [ ] Gate B — NIFTY 100 list built
- [ ] Falsifier #1 (post-signal 11:00-13:30 volume signature) pre-registered
- [ ] Chartink scanner check completed (verify gate is novel)

## 7. Next action

Phase 1 research + NIFTY 100 list build, then Phase 2 dispatch.
