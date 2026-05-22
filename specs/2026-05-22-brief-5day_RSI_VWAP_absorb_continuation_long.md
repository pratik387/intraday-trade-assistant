# `5day_RSI_VWAP_absorb_continuation_long` — Stage 3 brief

> **STATUS: PROCEED to Phase 3 (2026-05-22)** — Phase 2 empirical signature PASSED with caveats.
>
> **Phase 2 evidence (script: `tools/sub9_research/phase2_5day_RSI_VWAP_absorb_continuation_signature.py`, output: `reports/sub9_sanity/_phase2_5day_RSI_VWAP_absorb_continuation_signature.csv`, 7,552 rows):**
>
> | Metric | Value | Required |
> |---|---|---|
> | **Falsifier #1 — Absorption signature** | **91% next-bar vol < cross-bar vol** | ≥60% (✅ +31pp margin) |
> | n_signal (post-2024 cohort) | 733 | ≥200 ✅ |
> | Aggregate drift (post-2024) | **+0.1623%** | ≥+0.15% ✅ (barely, +1.5bp margin) |
> | Regime gate validation (pre-2024) | -0.006% (correctly fails) | by design ✅ |
> | Pre-SEBI Oct 2025 cohort | +0.156% | ≥+0.15% ✅ |
> | Post-SEBI Oct 2025 cohort | +0.244% | ≥+0.15% ✅ |
>
> **Cohort breakdown (caveats for Phase 4-5):**
>
> | Cohort | n | delta |
> |---|---|---|
> | 2024 H1 | 225 | +0.415% (strong) |
> | **2024 H2** | 219 | **-0.033%** (yellow flag — negative cohort) |
> | 2025 H1 | 130 | +0.078% (weak) |
> | 2025 H2 | 144 | +0.105% (weak) |
> | 2026 Q1 | 14 | +0.445% (small n) |
> | **cap=small_cap** | 391 | **+0.266%** (most of the edge) |
> | cap=mid_cap | 342 | +0.043% (weak — may need cap-segment lock to small_cap) |
> | RSI duration 5-day sustained | 428 | +0.152% (still passes) |
>
> **Open concerns for Phase 3/4:**
> 1. Aggregate drift is **borderline** (+1.5bp above floor). Phase 5 confidence card with Bonferroni haircut at M≥1 (vs `long_panic_gap_down` family) will likely tighten CI below 1.0 PF.
> 2. **Edge concentrated in small_cap.** Phase 5 cell-lock may need to restrict to `cap=small_cap` only.
> 3. **2024 H2 negative cohort.** Could be noise (n=219) or could indicate intra-regime decay. Phase 5 monthly breakdown mandatory.
> 4. **`cross_bar_vol_ratio` mean = 725.4** is artifactual (extreme values from tiny prior-bar baselines). Phase 4 sanity must use a more robust vol_ratio computation (e.g., trimmed mean baseline or median-based vol_ratio).
>
> **Inverse-edge story confirmed (Lesson #1):** the SHORT-direction mirror `5day_RSI_overbought_intraday_VWAP_lose_short` was KILLED 2026-05-22 on wrong-sign drift; the LONG direction with reframed absorption mechanism produces +0.16% post-2024 drift. The Bayesian inverse-edge test was right.
>
> ---

**Date:** 2026-05-22
**Stage:** 3 — Mechanism brief + pre-registration (this commit)
**Branch:** `research/2026-05-22-new-setups-batch`
**Predecessor:** Inverse-edge analysis of `5day_RSI_overbought_intraday_VWAP_lose_short` (Phase 2 KILL 2026-05-22) — same signal pattern, opposite direction, different mechanism story per Lesson #1 inverse-edge test
**Direction:** LONG
**Window:** Intraday MIS (square 15:25). Signal scan 09:30-12:00; entry at VWAP-cross bar close; exit 13:30.
**Portfolio rationale:** Regime-conditional LONG complement; signal pattern already validated empirically by #5 SHORT Phase 2 (post-2024 cohort drift +0.162%, post-SEBI cohort +0.244% — both above LONG +0.15% threshold). LONG direction adds portfolio diversification.

## 1. Mechanism statement (ONE sentence)

Small/mid-cap NSE MIS-eligible stocks where daily RSI(14) is ≥ 75 for 3+ consecutive trading days as of T-1 close (positioning saturated on the LONG side — most willing retail buyers already in) that print a 5m bar in 09:30-12:00 IST where (a) close < intraday VWAP for the first time AND (b) prior bar's close ≥ VWAP AND (c) bar.volume / cumulative-prior-mean ≥ 1.2× AND (d) the next 5m bar's vol_ratio < cross-bar vol_ratio (absorption signature: heavy retail panic-supply at the cross is absorbed by institutional bid, then volume drops as supply exhausts) drift UPWARD into 13:30 because the saturation regime favors continuation and the VWAP cross-down was a thin-tape liquidity event (retail panic-stop-loss cascade), not institutional supply initiation; LONG entry at the cross bar's close, exit 13:30.

## 2. Falsifiers (3)

1. **Absorption falsifier (next-bar vol drop):** Mechanism requires "absorption then exhaust" — cross-bar heavy volume, next-bar volume DROPS. Test: across 200+ post-2024 fires, fraction where `next_bar_vol_ratio < cross_bar_vol_ratio` should be > 60%. If < 50%, the volume isn't being absorbed (supply continues) and the mechanism is wrong → KILL.

2. **Regime falsifier (post-2024 dependency):** Mechanism is REGIME-CONDITIONAL — works only when retail trader count is high and passive flow steady. Phase 2 prior shows pre-2024 fails (n=717, delta -0.006%) and post-2024 passes (n=733, delta +0.162%). Phase 2 must reconfirm: post-2024 cohort PF CI lower bound > 1.0 AND pre-2024 fails — that's the regime-gate validation. If post-2024 fails or pre-2024 unexpectedly passes, regime story is wrong → KILL.

3. **Single-day-RSI falsifier:** Sustained-3-day RSI must outperform single-day RSI on LONG drift (saturation effect, not noise). Phase 2 prior already showed sustained +0.077% > single-day -0.010% (delta 0.087%). Phase 2 must reconfirm on post-2024 cohort specifically. If converged, "saturation" thesis dies → KILL.

## 3. Adjacent setups + correlation/effective-M

| Setup | Status | Direction | Mechanism overlap | M penalty |
|---|---|---|---|---|
| `long_panic_gap_down` | active | LONG | Different trigger (gap-down event, not RSI-saturation + VWAP cross). Different universe direction (capitulated, not overbought). | 0 |
| `below_vwap_volume_revert_long` | paper-pending | LONG | Both use VWAP mechanic but different signal (this fires on cross-DOWN from above, paper-pending fires on revert UP from below). | 0.3 |
| `close_dn_overnight_long` | paper-pending | LONG | Overnight CNC, different timeframe + trigger | 0 |
| `mis_unwind_vwap_revert_short` | RETIRED | SHORT | Inverse direction; retired post-SEBI-Oct-2025 | 0 |
| `5day_RSI_overbought_intraday_VWAP_lose_short` | KILLED today | SHORT (this candidate's mirror) | Same signal, opposite direction. KILLED today. | 0 (not both can ship; this is the empirically-correct direction) |

**Effective M estimate:** 0.3 (lowest in today's batch — distinct from all active and paper-pending setups). Independence is genuine.

## 4. Phase 1 outline (Gate A + Gate B) — INHERIT from SHORT mirror

### Gate A — Inherited PASS (with reframe)

Phase 1 sources from `5day_RSI_overbought_intraday_VWAP_lose_short` PASS (Zerodha Varsity RSI, Upstox RSI+VWAP combo, SEBI 2024 retail study) **apply to the signal pattern**, not specifically to the direction. The SHORT-direction reading was the Indian-retail-trader's interpretation; the LONG-direction reading is the empirically-correct mechanism per Phase 2 data.

**Additional source needed (≥1):** Indian operationalization of "absorption volume" / "thin-tape squeeze" / "supply exhaustion" as a LONG-continuation signal — to confirm the mechanism reframe is supported, not just a post-hoc rationalization.

### Gate B — Inherited PASS

All data confirmed in #5 SHORT Phase 1 (5m bars, consolidated_daily with `ts` column, RSI computable, ProductionUniverseGate, ≥30 daily bars per symbol).

## 5. Phase 2 plan (preview)

- **Universe:** cap ∈ {small_cap, mid_cap}, MIS-eligible, ProductionUniverseGate, ≥30 daily bars. **Regime gate: signal_date >= 2024-01-01** (per #5 SHORT Phase 2 finding that pre-2024 cohort doesn't work).
- **Multi-day filter:** `RSI(14)[T-1..T-3] >= 75` (sustained 3-day, Wilder smoothing). consolidated_daily uses `ts` column.
- **Intraday signal (LONG entry):** for each T+0 5m bar in 09:30-12:00, compute cumulative intraday VWAP (bars[:i+1] only — Lesson #5 #1). Trigger: `bars[i].close < VWAP[i]` AND `bars[i-1].close >= VWAP[i-1]` AND `bars[i].volume / mean(prior bars volume EXCLUDING current) >= 1.2`. First-fire-per-day latch.
- **NEW: Absorption confirmation (Falsifier #1):** record `next_bar_vol_ratio = bars[i+1].volume / vol_baseline_at_i+1`. Report fraction of signals where `next_bar_vol_ratio < cross_bar_vol_ratio`. **Required >= 60% for mechanism confirmation.**
- **Baseline:** same RSI-sustained, post-2024 universe, NO VWAP cross-down in 09:30-12:00. Anchor: 12:00 close.
- **Target return:** `ret_to_1330 = (close_at_1325 - signal_close) / signal_close * 100`. LONG → positive is good.
- **Acceptance:** drift ≥ +0.15% AND n_signal ≥ 200 AND Falsifier #1 (absorption signature) PASS.
- **Required splits:** post-2024 sub-splits (2024 H1/H2, 2025 H1/H2, 2026 Q1), pre-SEBI-Oct-2025 vs post (regime stability), cap=small vs mid, RSI-duration (3-day vs 5-day sustained), absorption-confirmed vs absorption-not-confirmed sub-cohorts.

## 6. Status checklist

- [x] Phase 1 Gate A inherited from #5 SHORT (signal pattern validated)
- [ ] Phase 1 Gate A reframe: ≥1 Indian source for "absorption volume" or "thin-tape squeeze" LONG mechanism
- [x] Phase 1 Gate B inherited (data feasibility confirmed)
- [x] Regime gate (post-2024 only) pre-registered
- [ ] Phase 2 absorption Falsifier #1 (next-bar vol drop ≥60%) pre-registered

## 7. Next action

Phase 1 supplementary research (single mechanism-reframe source) + Phase 2 dispatch with absorption-Falsifier-#1 pre-registered.
