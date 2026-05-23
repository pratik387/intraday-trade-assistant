# `5day_RSI_overbought_intraday_VWAP_lose_short` — Stage 0 brief

**Date:** 2026-05-22
**Stage:** 0 — Idea (awaiting Phase 1)
**Branch:** `research/2026-05-22-new-setups-batch`
**Predecessor:** Multi-day-signal batch #5 of 5
**Direction:** SHORT
**Window:** Intraday MIS (square 15:25). Signal scan 09:30-12:00; entry at VWAP-lose bar; exit 13:30.
**Portfolio rationale:** Technical-indicator multi-day signal (sustained daily RSI overbought) — fundamentally different from price-level (#1), return-magnitude (#2), rank (#3), volume (#4) classes. Maximizes signal-class diversity in the batch.

## 1. Mechanism statement (ONE sentence)

Small/mid-cap NSE MIS-eligible stocks (cap ∈ {small_cap, mid_cap}) where daily RSI(14) is ≥ 75 for 3 or more CONSECUTIVE trading days as of T-1 close (sustained overbought = positioning saturated, NOT a transient single-day RSI spike) that on T+0 print a 5m bar in 09:30-12:00 IST where (a) `close < intraday_VWAP_at_bar_close` AND (b) the prior 5m bar's close was `>= intraday_VWAP_at_prior_close` (first VWAP cross-DOWN from above) AND (c) `bar.volume / cumulative-prior-mean-excluding-current >= 1.2×` (institutional supply confirmation) signal the saturated retail-RSI positioning is unwinding via institutional VWAP-anchored selling; SHORT-fade at the VWAP-cross bar's close, exit 13:30.

## 2. Falsifiers (3)

1. **Mechanism falsifier (single-day vs sustained):** Thesis requires SUSTAINED overbought (3+ days). Test: compare drift delta on single-day-RSI≥75 cohort vs 3-day-sustained cohort. If single-day and sustained produce equal drift deltas, "sustained = positioning saturation" thesis is wrong (RSI is just noise indicator either way) → mechanism wrong, KILL.

2. **Regime falsifier (retail RSI chasing):** Depends on retail RSI-chasing concentration. FII-exit regimes (R4) suppress retail flow. Per-regime PF CI lower bound > 1.0 must hold in ≥ 4 of 7 regimes including R1, R5.

3. **Infra falsifier (RSI tool changes):** Mechanism depends on retail using RSI via Indian broker platforms (Zerodha, Upstox, Angel, Tradingview India). Wholesale change to default indicators or replacement by other oscillators (Stochastic, MFI) would shift the positioning.

## 3. Adjacent setups + correlation/effective-M

| Setup | Status | Direction | Mechanism overlap | M penalty |
|---|---|---|---|---|
| `or_window_failure_fade_short` | active | SHORT | Same direction, different window + trigger. Some overlap on RSI-high names that also break OR. | 0.5 |
| `gap_fade_short` | active | SHORT | Different window (open vs scan-to-noon), different trigger | 0.3 |
| `mis_unwind_vwap_revert_short` | RETIRED | n/a | Used RSI≥85 as FILTER + late-day window. Different mechanism class (entry filter, not multi-day signal). Retired post-SEBI-Oct-2025. | 0 |
| Other batch candidates | this batch | SHORT | Different multi-day signal classes — independent gates | 0.3 each |

**Effective M estimate: 0.5** vs or_window_failure (lowest in candidate set — distinct signal class).

## 4. Phase 1 outline (Gate A + Gate B)

### Gate A — Indian sources to find (≥2 required)

1. **Zerodha Varsity (Technical Analysis — RSI chapter)** — RSI overbought/oversold framework + retail usage statistics
2. **intradaylab.com** — Indian RSI-extreme + intraday reversal patterns
3. **SEBI 2024 retail study** — retail TA-indicator concentration + behavior
4. **Indian broker tutorials (Upstox, Motilal Oswal)** — RSI as primary retail intraday-positioning signal
5. **Indian academic literature on RSI predictive power in NSE intraday** — backing for sustained vs single-day distinction

Acceptance: Zerodha Varsity (RSI module) + ≥1 Indian operationalization of "sustained RSI overbought + intraday supply confirmation = SHORT fade in small/mid-cap."

### Gate B — Data feasibility (predicted PASS)

| Required data | On disk? |
|---|---|
| 5m bars per symbol (for intraday VWAP) | ✅ |
| `consolidated_daily.feather` (daily OHLCV for RSI(14) computation) | ✅ |
| `cap_segment` metadata | ✅ |
| `ProductionUniverseGate` (Lesson #19) | ✅ |
| RSI computation feasibility | ✅ (standard 14-day Wilder smoothing, computable from daily data) |

## 5. Phase 2 plan (preview)

- **Universe:** `cap_segment ∈ {small_cap, mid_cap}` AND MIS-eligible AND `ProductionUniverseGate` per-date AND 30+ daily bars available (for stable RSI(14)).
- **Multi-day filter:** compute daily RSI(14) using Wilder smoothing from `consolidated_daily.feather` per symbol. Require `RSI[T-1] >= 75` AND `RSI[T-2] >= 75` AND `RSI[T-3] >= 75` (sustained 3-day overbought, NOT single-day spike).
- **Intraday signal (SHORT entry):** for each T+0 5m bar in 09:30-12:00:
  - Compute cumulative intraday VWAP through this bar (typical_price × volume, cumulative; Lesson #5 #1 — no look-ahead beyond current bar)
  - Compute `vol_baseline = mean(volume of prior intraday bars EXCLUDING current)` (Lesson #5 #2)
  - Trigger fires when `bars[i].close < VWAP[i]` AND `bars[i-1].close >= VWAP[i-1]` (cross-down from above) AND `bars[i].volume / vol_baseline >= 1.2`. Signal at bar i's close.
  - Take FIRST qualifying bar per (sym, date).
- **Baseline (control):** same RSI-sustained universe + NO VWAP cross-DOWN in 09:30-12:00 (price stayed above intraday VWAP throughout). Use 12:00 close as no-signal anchor.
- **Target return:** `ret_to_1330 = (close_at_1325 - signal_close) / signal_close * 100`. SHORT.
- **Acceptance:** drift `<= -0.15%` AND `n_signal >= 200`.
- **Required splits:** pre/post-2024, pre/post-SEBI-Oct-2025, cap=small vs cap=mid, RSI-duration (3-day vs 5-day vs 7-day sustained), VWAP-cross-bar vol_ratio buckets.

## 6. Status checklist

- [ ] Gate A — ≥2 Indian sources cited
- [ ] Gate B — data feasibility confirmed
- [ ] Universe excludes large_cap AND unknown
- [ ] Falsifier #1 single-day-vs-sustained cohort comparison pre-registered
- [ ] RSI(14) Wilder-smoothing implementation locked

## 7. Next action

Phase 1 research — parallel dispatch with batch.
