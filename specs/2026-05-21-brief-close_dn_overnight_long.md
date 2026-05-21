# `close_dn_overnight_long` — Phase 3 brief

**Date:** 2026-05-21
**Branch:** `research/europe-open-13ist`
**Predecessor:** `specs/2026-05-21-phase2-volume-4angles-RESULTS.md`
(Phase 2 v2 batched signature test; Angle 4 PROCEED with pooled |spread| = 0.3124% across 2023+2024. Mean next-day open return on STRONG_DN_closing = +0.5034% with WR=76.3% at thresh=0.5. Monotonic in thresh, strong on STRONG_DN side, asymmetric — STRONG_UP_closing also positive but much weaker [+0.19% mean].)

## 1. Mechanism statement (ONE sentence)

When a stock's closing 30-minute window (15:00–15:25 IST) is dominated by sell-volume (`signed_vol_ratio ≤ -0.5`, where `signed_vol_ratio = Σ(volume × sign(close-open)) / Σ(volume)`), the closing flush represents forced supply (MIS-unwind, retail panic, EOD margin calls); the next-day opening auction at 09:15 partially reverses the flush as overnight short-cover + value-buyer bid restores price, producing a measurable mean-revert in next-day open vs prior-day close.

## 2. Indian-microstructure anchor

Phase 2 v2 pooled 2023+2024 results show:

| thresh | direction (LONG side) | n_ev (pooled) | mean_next_open_ret | win_rate |
|-------:|:----------------------|--------------:|-------------------:|---------:|
| 0.1 | STRONG_DN_closing | 230,400 | +0.4150% | 73.0% |
| 0.2 | STRONG_DN_closing | 185,997 | +0.4316% | 73.8% |
| 0.3 | STRONG_DN_closing | 144,618 | +0.4507% | 74.6% |
| 0.5 | STRONG_DN_closing | **75,542** | **+0.5034%** | **76.3%** |

Monotonic increase in both threshold and win rate. Mirror side `STRONG_UP_closing` shows much weaker mean (+0.19% at thresh=0.5, WR=57%) — asymmetry confirms LONG-only signature.

The mechanism is consistent with documented Indian intraday + overnight microstructure:

1. **NSE MIS auto-square-off at 15:15** (Zerodha; ICICIDirect; standard across brokers): brokers force-exit MIS positions starting ~15:00, peaking 15:10-15:25. For LONG MIS holders sitting in a falling stock, the forced exit IS selling pressure — the closing flush we detect.
2. **EOD margin-call cascades** (SEBI peak-margin rules 2021-onwards): brokers raise margin requirements at session end; under-margined positions get sold by broker risk desks at the close, concentrated in the same 15:00-15:25 window.
3. **NSE pre-open auction at 09:00-09:08** (NSE circular 2010, refined 2017): overnight buy orders accumulate before 09:08 cutoff and clear at the 09:15 opening auction. Bottom-fishing institutional bids on stocks that flushed at prior close → opening above the prior-day close → the +0.5% mean reversion we measure.
4. **Zerodha Varsity Module 5** + intradaylab.com discuss "EOD reversal" and "overnight gap-up after panic close" as separate-but-related patterns; the volume-signed mechanism here operationalises the trigger.

## 3. Falsifiers (3 conditions that would invalidate)

1. **Mechanism falsifier (signal sub-bar artifact):** if the +0.5% mean is mostly the GAP from prior-close to next-day open (no further intraday move), and that gap is driven by overnight ADR/news rather than mean-revert, then the signal would not generalize to days WITHOUT news catalysts. Phase 5 must test: per-regime sub-bucket on a `news_proximity` feature (days within ±1 of an earnings announcement or major news flag) — if signal disappears outside news days, mechanism is wrong → KILL.

2. **Fee falsifier (delivery STT eats the edge):** CNC round-trip ≈ 0.17% (driven by 0.10% STT on delivery sell). Phase 2 mean signal = 0.503% gross → +0.329% net per trade. Acceptance gate requires NET PF (after CNC fees) >= 1.20 on Discovery, >= 1.10 on OOS+HO. If fee-adjusted PF falls below floor → KILL.

3. **Overnight gap-risk falsifier (tail loses):** the 24% of trades that don't win can include catastrophic gap-downs (post-close earnings miss, FII flow reversal, ADR crash). If the worst 1% of trades produce losses >= 5× the mean win, the right-skew of the PnL distribution dies and aggregate PF collapses even with 76% WR. Phase 5 must compute P99 loss vs mean win — if ratio >= 5×, the setup is gambling, not edge.

## 4. Pro/retail precedent (≥2 Indian sources)

1. **Zerodha Varsity Module 5** ("Trading Strategies") discusses "gap-down recovery" and "overnight buy-the-dip" patterns; without the volume-signed-direction trigger, this is partial precedent.
2. **intradaylab.com** documents "EOD panic close → next-day reversal" as a named pattern in their pattern catalog — direct mechanism precedent.
3. **SEBI 2024 retail study**: confirms retail traders are forced sellers at session close (MIS auto-square + margin-shortfall liquidation); the supply side of our LONG thesis.
4. **NSE-IFMR working paper on pre-open auction** (2018, 2021 revisions): documents systematic information-revelation in the 09:00-09:08 pre-open window. Institutional + algorithmic bid replenishment at the open is the demand side of our thesis.

**STRONG precedent** — both the supply mechanism (MIS unwind + margin calls) and demand mechanism (pre-open auction discovery) are operationally documented. Our contribution: (a) the `signed_vol_ratio` operationalization of the closing flush; (b) the +0.5% threshold cell-locking; (c) Indian-universe sample-size validation across 2023-2026.

## 5. Pre-registered cell-sweep dimensions

Per Lesson #2 (anti-salvage), dimensions are locked BEFORE running cell sweep.

**Filter dim_pool (max 2D combinations):**

| Dimension | Bins | Source |
|---|---|---|
| `cap_segment` | large_cap, mid_cap, small_cap, micro_cap, unknown | per_row (universe builder, `nse_all.json`) |
| `signed_vol_ratio_bin` | [-0.5, -0.6), [-0.6, -0.75), [-0.75, -0.9), [-0.9, -1.0] | computed at 15:25 from closing-30min bars |
| `prior_day_return_pct_bin` | down_gt_3pct, down_1to3pct, flat, up_1to3pct, up_gt_3pct | (prior-day close - prior-prior close) / prior-prior close |
| `closing_30m_volume_z` | normal (z<1), high (1<=z<2), extreme (z>=2) | closing 30m total volume z-score vs prior-20d closing-30m mean |
| `news_proximity` | within_1day_earnings, otherwise | from earnings calendar (`data/earnings_calendar/`) |

**Forbidden dimensions:** anything using next-day data at signal time (Lesson #5 failure mode #1). Next-day open IS the target, not a filter.

**No R-grid / no SL/T1/T2 to sweep** — this is an all-in directional position with fixed entry (15:25 close) and fixed exit (next-day 09:15 open). The "geometry" sweep is trivial. The cell sweep is purely about filter dims.

**Time stop:** N/A — exit is locked at 09:15 next trading day (or first bar after if 09:15 has no fill). No intraday management.

## 6. Mechanic (single sentence)

LONG CNC entry at the 15:25 bar's close (Mode B for "next bar" doesn't apply — entry is the LAST bar of the session) when:
- `signed_vol_ratio <= -0.5` over the 15:00-15:25 window (computed as `Σ(volume × sign(close-open)) / Σ(volume)` across the six 5m bars)
- `closing_30m_volume_z >= 1.0` (today's closing-30m volume is at least 1σ above the prior-20d closing-30m mean — confirms flush is real, not just thin trading)
- stock is in declared liquidity universe (>= 200d trading-day coverage, >= 50K daily avg vol)
- stock has NO earnings announcement scheduled for next session (skip news days)
- no other position open in same symbol

Exit: SELL at next-day 09:15 bar OPEN (or first available fill after open).

## 7. Active window

**Signal window:** 15:25 IST (the last 5m bar of regular session). Signal is computed at bar close.
**Entry window:** 15:25 (signal bar) to 15:29 (immediately after signal). Order placed as CNC BUY.
**Exit window:** 09:15 next trading day (pre-open auction match or first opening tick). Order placed pre-market as CNC SELL.
**Position holding:** ~18 hours overnight (Mon-Thu) / ~66 hours (Fri close → Mon open) / longer over public holidays.

## 8. Independence-from-existing-edges story

| Existing setup | Direction | Window | Conflict? |
|---|---|---|---|
| `gap_fade_short` (active) | SHORT | 09:15-09:30 | No — SHORT direction; this exits at 09:15 BEFORE gap_fade fires |
| `long_panic_gap_down` (active) | LONG | 09:15-09:20 | **Possible same-symbol overlap**: this setup may hold a symbol overnight that gaps down → long_panic_gap_down could fire on the same name. Manage via dispatcher's per-symbol cooloff or universe exclusion. |
| `or_window_failure_fade_short` (active) | SHORT | 09:30-10:30 | No — this setup is already exited by 09:15 |
| `circuit_t1_fade_short`, `delivery_pct_anomaly_short` (active) | SHORT | morning | No — opposite direction, different window |
| `below_vwap_volume_revert_long` (active, SHIPPED 2026-05-21) | LONG (intraday CNC^) | 13:00-14:55 | No — both LONG, different windows (this setup fires at 15:25, below_vwap exits by 14:30). No same-bar overlap. But same-symbol exposure can compound: if below_vwap holds a LONG into 14:30 time-stop, then this setup fires on the same name at 15:25, we'd have two consecutive LONG positions. Acceptable but worth telemetry. |
| `mis_unwind_short` (retired 2026-05-19) | SHORT | 14:30-15:00 | RETIRED — no conflict |

**No active LONG overnight setup exists.** Closest mechanic-overlap is `long_panic_gap_down` which fires at 09:15-09:20 the next day — exactly when this setup is EXITING. Could compound for the same symbol but they target different conditions (this setup conditions on close-direction; long_panic on gap-down magnitude). Telemetry should track joint fires.

^ below_vwap_volume_revert_long is intraday MIS (not CNC); naming is unrelated.

## 9. Sample-size feasibility

Phase 2 v2 pooled 2023+2024 yielded at thresh=0.5:
- STRONG_DN_closing (LONG cohort): **75,542** events across 250+ unique trading days

Per-day average: ~150 events. With cell-locking (cap_segment × signed_vol_ratio_bin × closing_30m_volume_z), expected per-day count drops to 5-20 — manageable.

After splitting by Discovery / OOS / Holdout per `assets/regime_schema.yaml`:
- Discovery (60%): ~45K events
- OOS (20%): ~15K events
- Holdout (20%): ~15K events

Cell-locked expected n per Disc cell: 1,000-5,000. PF ≥ 1.20 with n ≥ 500 floor easily achievable.

## 10. Acceptance criteria (Phase 5 ship gate)

**Critical: all PF computations include CNC fees.** Fee model: brokerage Rs 20 flat per side + STT 0.10% on sell-side (delivery) + SEBI/txn/GST/stamp per `tools/sub7_validation/build_per_setup_pnl.calc_fee_cnc` (to be added — current `calc_fee` is intraday-MIS only).

1. **Discovery:** Cell with n >= 500, **fee-adjusted** PF_net >= 1.20 on Disc+OOS combined
2. **OOS one-shot:** fee-adjusted PF_net >= 1.10 with WR within 10pp of Discovery
3. **Holdout one-shot:** fee-adjusted PF_net >= 1.10
4. **Stationarity:** max-min PF_net across (Disc, OOS, HO) <= 0.30
5. **Mechanism falsifier (news days excluded):** fee-adjusted PF on non-news subset >= 1.15 (slightly relaxed from main gate since news exclusion shrinks n)
6. **Gap-risk falsifier:** P99 loss / mean win <= 5×
7. **Confidence framework verdict** (per Lesson #15): pooled D+OOS+HO PF CI lower bound > 1.0; adjusted Sharpe > 0 after Harvey-Liu haircut for M effective setups

If any acceptance gate fails -> KILL.

## 11. Files of record

- Phase 2 v2 batched run: `tools/sub9_research/phase2_volume_4angles_v2.py`
- Phase 2 results: `specs/2026-05-21-phase2-volume-4angles-RESULTS.md`
- Phase 2 sensitivity CSVs: `reports/sub9_sanity/_phase2_volume_v2_angle4_{per_year,pooled}.csv`
- This brief: `specs/2026-05-21-brief-close_dn_overnight_long.md`
- Sanity script (to write): `tools/sub9_research/sanity_close_dn_overnight_long.py`
- Earnings calendar dependency: `data/earnings_calendar/` (existing, used for news_proximity dim)
- CNC fee helper (to write or extend): `tools/sub7_validation/build_per_setup_pnl.py` — add `calc_fee_cnc(buy_value, sell_value, qty)`

## 12. Open infrastructure questions (resolve before Phase 4 sanity)

1. **CNC fee helper:** existing `calc_fee` is intraday-MIS only. Need a parallel `calc_fee_cnc` that uses delivery STT (0.10% sell), delivery stamp (0.015% buy), no MIS leverage. Implement in `tools/sub7_validation/build_per_setup_pnl.py`.

2. **Earnings calendar pull:** the `data/earnings_calendar/` exists but coverage across the full 2023-2026 backtest period needs verification. If gaps exist, news_proximity dim is unreliable — sanity should report the coverage percentage upfront and abort if < 95% of cell-locked days have valid earnings data.

3. **Pre-open exit modelling:** the sanity script's "next-day OPEN" exit assumes the 09:15 open price is fillable. In reality, pre-open auction matching may not fill 100% of the order (especially for thin small-caps). For Phase 4 sanity: just use the 09:15 OPEN price as a clean theoretical fill (matches the Phase 2 sensitivity output). For paper trading: add slippage model based on pre-open order-book depth.

4. **Symbol-level position compounding:** if this setup holds a LONG overnight and `long_panic_gap_down` fires on the same name at 09:15, the dispatcher would attempt to open a second LONG. Per the spec section 8, this should be guarded by per-symbol cooloff. Existing dispatcher infrastructure supports this — need to verify in Phase 4 implementation.
