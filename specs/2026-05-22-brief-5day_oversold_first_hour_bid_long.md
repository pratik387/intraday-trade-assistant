# `5day_oversold_first_hour_bid_long` — Stage 0 brief

**Date:** 2026-05-22
**Stage:** 0 — Idea (awaiting Phase 1)
**Branch:** `research/2026-05-22-new-setups-batch`
**Predecessor:** Multi-day-signal batch #2 of 5
**Direction:** LONG
**Window:** Intraday MIS (square 15:25). Signal scan 09:30-10:30; entry at wick-bar close; exit 13:30.
**Portfolio rationale:** LONG-side complement using multi-day capitulation signal. Distinct mechanic from `long_panic_gap_down` (which is single-day gap event); this uses 5-day cumulative oversold + near-low close as the multi-day gating.

## 1. Mechanism statement (ONE sentence)

Small/mid-cap NSE MIS-eligible stocks (cap ∈ {small_cap, mid_cap}) where `5day_cumret <= -8%` AND `daily.close[T-1] <= 5day_low * 1.01` (sustained capitulation, T-1 close near multi-day low — not a single-day spike) that print a 5m bar in 09:30-10:30 IST with `lower_wick / body >= 0.5` AND `close > open` (green bar with rejection wick) AND no fresh low in the next 30 minutes (institutional discretionary bid stepping in to resolve the cascade) get LONG-faded at the wick bar's close, exit at 13:30 as the rebound plays out into the post-lunch high-volume window.

## 2. Falsifiers (3)

1. **Mechanism falsifier (bid volume signature):** Thesis = institutional discretionary bid steps in on declining retail supply. Wick-bar `vol_ratio` (vs cumulative-prior-mean) must be elevated (>=1.2× median across 200+ fires). If median vol_ratio < 1.0×, the wick was thin noise, not real bid → KILL.

2. **Regime falsifier (FII flow dependency):** Mechanism depends on institutional discretionary bid being available. During FII-exit regimes (R4) institutional bid weakens. Per-regime PF CI lower bound > 1.0 must hold in ≥ 4 of 7 regimes including R1, R5. If <4 → KILL.

3. **Infra falsifier (MIS leverage policy):** Mechanism depends on retail MIS leverage cascade exhausting within 60-90 min. SEBI MIS minimum-margin changes that reduce leverage below 5× shrink cascade magnitude.

## 3. Adjacent setups + correlation/effective-M

| Setup | Status | Direction | Mechanism overlap | M penalty |
|---|---|---|---|---|
| `long_panic_gap_down` | active | LONG | **Same family** (capitulation LONG). Different signal (T+0 gap-down event vs multi-day cumulative oversold). Triggers may co-fire if 5-day decline ended with a T+0 gap-down. | **1.5** (highest in candidate set) |
| `capitulation_long_v2` | RETIRED | n/a | Was mid-cap gap-down. Retired. | 0 |
| `below_vwap_volume_revert_long` | paper-pending | LONG | Afternoon VWAP mean-revert, different window | 0.3 |
| `close_dn_overnight_long` | paper-pending | LONG | Overnight CNC, different timeframe | 0.2 |

**Effective M estimate: 1.5** vs `long_panic_gap_down` (highest in candidate set). Mitigation: Phase 2 telemetry must report fraction of fires that overlap with same-day `long_panic_gap_down` trigger criteria; if >50%, mechanism is partially derivative.

## 4. Phase 1 outline (Gate A + Gate B)

### Gate A — Indian sources to find (≥2 required)

1. **intradaylab.com** — small-cap multi-day oversold + intraday reversal patterns
2. **Zerodha Varsity (Module 5)** — capitulation low + bid-bar pattern documentation
3. **SEBI 2024 retail study** — small/mid-cap retail MIS leverage cascade exhaustion timing
4. **Indian broker quant blogs (Motilal Oswal / Edelweiss / ICICI)** — multi-day decline reversal patterns

Acceptance: ≥1 Indian source operationalizes "multi-day oversold + intraday bid-rejection wick = LONG fade in small/mid-cap retail-driven names" + supporting retail-cascade timing context.

### Gate B — Data feasibility (predicted PASS)

| Required data | On disk? |
|---|---|
| 5m bars per symbol | ✅ |
| `consolidated_daily.feather` (for 5-day cumret + low) | ✅ |
| `cap_segment` metadata | ✅ |
| `ProductionUniverseGate` (Lesson #19) | ✅ |

## 5. Phase 2 plan (preview)

- **Universe:** `cap_segment ∈ {small_cap, mid_cap}` AND MIS-eligible AND `ProductionUniverseGate` per-date AND 5-day daily-data available
- **Multi-day filter:** `5day_cumret = (daily.close[T-1] / daily.close[T-6]) - 1`; `5day_low = min(daily.low[T-5..T-1])`. Require `5day_cumret <= -0.08` AND `daily.close[T-1] <= 5day_low * 1.01`.
- **Intraday signal (LONG entry):** for each T+0 5m bar in 09:30-10:30: compute `lower_wick = min(open, close) - low` and `body = abs(close - open)`. Trigger fires when `lower_wick / max(body, 0.0001) >= 0.5` AND `close > open` AND `min(low for next 6 bars) >= bars[i].low`. Mark signal at the wick-bar's close. Take FIRST qualifying bar per (sym, date).
- **Baseline (control):** same universe + same multi-day filter but NO wick-bar bid signal in 09:30-10:30. Use 09:55 5m bar close as no-signal anchor.
- **Target return:** `ret_to_1330 = (close_at_1325 - signal_close) / signal_close * 100` (LONG → positive is good)
- **Acceptance:** drift delta `>= +0.15%` AND `n_signal >= 200`
- **Required splits:** pre/post-2024, pre/post-SEBI-Oct-2025, cap=small vs cap=mid, wick-bar vol_ratio buckets (<1.0, 1.0-1.5, ≥1.5), overlap with same-day `long_panic_gap_down` trigger

## 6. Status checklist

- [ ] Gate A — ≥2 Indian sources cited
- [ ] Gate B — data feasibility confirmed (predicted PASS)
- [ ] Universe excludes large_cap AND unknown
- [ ] Phase 2 overlap-with-long_panic_gap_down telemetry planned (high M risk)
- [ ] Bid-bar vol_ratio Falsifier #1 pre-registered

## 7. Next action

Phase 1 research — parallel dispatch with batch.
