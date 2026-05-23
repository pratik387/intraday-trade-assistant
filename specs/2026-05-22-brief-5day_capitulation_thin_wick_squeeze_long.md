# `5day_capitulation_thin_wick_squeeze_long` — Stage 0 brief

**Date:** 2026-05-22
**Stage:** 0 — Idea (awaiting Phase 1 supplement + Phase 2 dispatch)
**Branch:** `research/2026-05-22-new-setups-batch`
**Predecessor:** Mechanism-rephrase of `5day_oversold_first_hour_bid_long` (Phase 2 KILL 2026-05-22 — Falsifier #1 failed, but signal pattern produced +0.93% raw drift on aggregate). Pattern works empirically; original "institutional bid" mechanism falsified; this brief reframes as thin-tape short-cover squeeze.
**Direction:** LONG
**Window:** Intraday MIS (square 15:25). Signal scan 09:30-10:30; entry at wick-bar close; exit 13:30.
**Portfolio rationale:** Lesson #1 inverse-edge restatement. Empirical pattern: +0.93% signal vs -0.32% baseline (n=7394 / 5864) on 5-day-capitulation + first-hour wick LONG cohort. Mechanism story rewritten to match the data.

## 1. Mechanism statement (ONE sentence)

Small/mid-cap NSE MIS-eligible stocks where `5day_cumret <= -8%` AND `daily.close[T-1] <= 5day_low * 1.01` (sustained capitulation; multi-day decline likely produces elevated retail intraday-short positioning per Indian retail leverage cascade behavior) that print a 5m bar in 09:30-10:30 with `lower_wick / body >= 0.5` AND `close > open` AND no fresh low in the next 30 minutes (a thin-tape liquidity spike — the wick is NOT institutional bid per Phase 2 Falsifier-#1 failure, but rather a microstructure trigger where any opportunistic buy initiates a short-cover cascade in a low-volume stalled tape) drift UPWARD into 13:30 because the multi-day capitulation primes short interest and the thin-tape wick triggers the squeeze; LONG entry at the wick bar's close, exit 13:30.

## 2. Falsifiers (3)

1. **Squeeze proxy (delivery% asymmetry):** Short-cover squeeze requires elevated prior-day intraday short positioning. NSE EOD delivery% is the proxy — LOW delivery% (e.g., < 40%) indicates high intraday turnover/positioning. Signal cohort (wick + capitulation, LONG-rebound) should have LOWER 5-day mean delivery% than baseline cohort (no-wick capitulation, no-rebound). Test: `signal_5day_delivery_median < baseline_5day_delivery_median - 2pp`. If signal delivery% >= baseline (positioning wasn't elevated), squeeze mechanism is wrong → KILL.

2. **Next-day gap-up corroboration:** Squeeze patterns produce elevated next-day open-to-prior-close gap-up rates (squeeze continues into next session). Signal cohort should have higher fraction of T+1 gap-ups (≥0.5%) than baseline cohort. Test: `signal_T+1_gap_up_rate > baseline_T+1_gap_up_rate + 5pp`. If similar or lower, squeeze story is wrong → KILL.

3. **Universe filter (cap_segment):** Mechanism depends on retail concentration in small/mid-cap. Phase 2 prior shows both small_cap and mid_cap deliver positive drift (small: +0.90%, mid: +0.99%). Phase 2 must reconfirm both cap segments work. If only one passes, regime story narrows.

## 3. Adjacent setups + correlation/effective-M

| Setup | Status | Direction | Mechanism overlap | M penalty |
|---|---|---|---|---|
| `long_panic_gap_down` | active | LONG | Same family (capitulation LONG) but trigger differs (gap-day vs multi-day cumulative oversold + wick). Phase 2 prior showed only 10.49% overlap on same-day gap-down events. | **1.0** (down from 1.5 — Phase 2 measured overlap is lower than prior estimate) |
| `below_vwap_volume_revert_long` | paper-pending | LONG | Different mechanism (VWAP revert), different window. | 0.3 |
| `close_dn_overnight_long` | paper-pending | LONG | Overnight CNC. Different timeframe. | 0 |
| `5day_oversold_first_hour_bid_long` | KILLED today | LONG (this candidate's predecessor) | Same signal, different mechanism story. Not both can ship. | 0 |

**Effective M estimate:** 1.0 vs long_panic_gap_down (Phase 2 prior overlap of 10.5% is acceptable; Phase 5 confidence card applies Bonferroni haircut at M=2).

## 4. Phase 1 outline (Gate A + Gate B) — INHERIT from predecessor

### Gate A — Inherited PASS (with mechanism reframe)

Phase 1 sources from `5day_oversold_first_hour_bid_long` PASS (Upstox Intraday Hammer Strategy + Zerodha Varsity Hammer + IntradayLab + SEBI). These sources operationalize the PATTERN — multi-day decline + intraday wick rejection LONG entry. They DON'T specifically operationalize "short-cover squeeze on thin tape."

**Additional source needed (≥1):** Indian source for "short-cover squeeze" / "intraday short squeeze" / "thin-tape rebound" on retail-MIS Indian small/mid-cap. Likely candidates: SEBI intraday short positioning data, NSE delivery%-as-positioning-proxy academic studies, intradaylab short-squeeze patterns.

### Gate B — Inherited PASS + delivery% data

All data confirmed in #2 LONG Phase 1 + `data/delivery_pct/delivery_history.parquet` (confirmed working from #4 Phase 2 via explicit (sym, date) left-join at 96.8% coverage).

## 5. Phase 2 plan (preview)

- **Universe:** cap ∈ {small_cap, mid_cap}, MIS-eligible, ProductionUniverseGate, ≥6 daily bars (5-day cumret + low + delivery_pct lookback)
- **Multi-day filter:** `5day_cumret = (close[T-1] / close[T-6]) - 1`; `5day_low = min(daily.low[T-5..T-1])`; require `5day_cumret <= -0.08` AND `close[T-1] <= 5day_low * 1.01`. Plus compute `5day_delivery_median = median(delivery_pct[T-5..T-1])` via left-join on delivery parquet.
- **Intraday signal (LONG entry):** same as predecessor #2 — first 5m bar in 09:30-10:30 with `lower_wick/body >= 0.5`, `close > open`, no fresh low in next 6 bars. Mark at wick-bar close.
- **Baseline:** same universe + same multi-day filter but NO wick-bar bid in 09:30-10:30. Anchor: 09:55 close.
- **Target return:** `ret_to_1330`. LONG → positive.
- **NEW Falsifier #1 metrics:** record `5day_delivery_median` for signal and baseline rows; compute median delta. Also compute `T+1_open_to_T_close_gap` for signal/baseline rows; report gap-up rate (gap ≥ +0.5%).
- **Acceptance:** drift ≥ +0.15% AND n_signal ≥ 200 AND (Falsifier #1: `signal_delivery_median < baseline_delivery_median - 2pp`) AND (Falsifier #2: `signal_gap_up_rate > baseline_gap_up_rate + 5pp`). All four required.
- **Required splits:** pre/post-2024, pre/post-SEBI-Oct-2025, cap=small vs mid, 5-day delivery% buckets (<30%, 30-40%, 40-50%, ≥50%), overlap with `long_panic_gap_down` trigger.

## 6. Status checklist

- [x] Phase 1 Gate A inherited (pattern operationalization) — Upstox Hammer + Varsity Hammer + IntradayLab
- [ ] Phase 1 Gate A reframe: ≥1 Indian source for "short-cover squeeze" / "thin-tape rebound"
- [x] Phase 1 Gate B inherited + delivery_pct parquet wiring confirmed (#4 Phase 2 join worked at 96.8%+)
- [ ] Phase 2 squeeze-proxy Falsifier #1 (delivery% asymmetry) pre-registered
- [ ] Phase 2 next-day gap-up Falsifier #2 pre-registered

## 7. Next action

Phase 1 supplementary research (mechanism-reframe source) + Phase 2 dispatch with delivery%-asymmetry + next-day-gap-up Falsifiers pre-registered.
