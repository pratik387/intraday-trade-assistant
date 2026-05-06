# §3.3 Brief: `volume_spike_exhaustion_reversal`

**Sub-project:** #9 (microstructure-first redesign)
**Status:** **DRAFT — awaiting user APPROVE/REJECT/RETIRE before sanity-check.**
**Date:** 2026-05-06

**Predecessors:**
- specs/2026-05-01-sub-project-9-microstructure-first-redesign.md (defines §3.3 gate process)
- specs/2026-05-01-sub-project-9-asymmetry-research-findings.md (Round-1)
- specs/2026-05-05-sub-project-9-asymmetry-feasibility-round-2.md (Round-2)
- specs/2026-05-06-sub-project-9-asymmetry-feasibility-round-3.md (Round-3, this candidate is §3)
- specs/2026-05-01-sub-project-9-brief-circuit_t1_fade_short.md (APPROVED template)

This is one of three round-3 §3.3 briefs (the others are `vwap_deviation_meanrevert` and `index_stock_divergence_revert`). All three target the 11:00-15:15 IST middle-of-day window left empty by the existing morning-only setups.

---

## Asymmetry

**Name:** Indian-equity middle-of-day volume-spike exhaustion reversal (5m).

**Indian-specific source:**
- **3x+ normal-volume single-bar spikes in the 11:00-15:10 window are almost never fundamental news events** in NSE intraday. Fundamental events (earnings, M&A, policy announcements) cluster in pre-market, opening 09:15-10:00, or post-close 15:30+. Mid-session 5m bars with `volume ≥ 3 × rolling_20_bar_mean` AND `wick ≥ 60% of bar range` are signature of **retail capitulation** (stop-cascade) or **retail FOMO peak** (last-buyer-in, exhausting the move).
- **Indian intraday volume profile is U-shaped:** high at 09:15-10:00, low 11:30-13:30, mild rebound 14:00-15:00, peak final 30 min. A `vol_z ≥ 3.0` AGAINST this U-shape baseline is a genuine anomaly, not just opening flow.
- **Wick signature confirms exhaustion vs continuation:** a high-volume bar that closes near its high with a tiny wick is institutional accumulation (continuation expected); a high-volume bar with a 60%+ rejection wick is the OPPOSITE — buyers/sellers ABSORBED THE FLOW, signal exhaustion. The 60%+ wick filter is the key distinguishing feature between continuation-spike and reversal-spike.
- **SEBI FY23 losing-flow asymmetry:** retail flow that capitulates (panic-sell) or peaks (FOMO-buy) on a 3x volume bar is the LOSING side per SEBI 70%/93% statistic. Fading the spike is structurally aligned with the asymmetry.

The exploitable asymmetry is the **5m bar with `vol_z ≥ 3.0` AND `(top_wick or bottom_wick) ≥ 60% of range` followed by a confirming reversal candle, faded in the OPPOSITE direction of the wick-rejected side**.

## Participants

- **Volume-spike-with-upper-wick (SHORT side, primary edge):** retail FOMO peak in mid/small-cap F&O — last-buyer-in on a momentum chase, then immediate rejection by sellers. Common single-stock impulse sources: retail screener "top gainer" attention spike, sector-leader halo (e.g., HDFCBANK news → all banking pops then fades), TradingView alert cascade. The fade is filled by mean-reversion algos + institutional VWAP-target sellers + profit-taking from earlier longs.
- **Volume-spike-with-lower-wick (LONG side, secondary edge):** retail capitulation / stop-cascade — last-seller-out on a stop-loss flush, then immediate absorption by buyers. Long side carries the long-bias 11-failure risk.

We are on the disciplined side of an attention-driven (or panic-driven) mispricing — the wick is the literal evidence that the OTHER side has already pushed back.

## Persistence

Three structural reasons:

1. **Retail FOMO/panic behaviour is structural** — Barber-Odean attention-driven trading literature documents persistent retail bias toward attention-grabbing stocks. SEBI FY23 confirms the Indian-specific magnitude. Volume-spikes with wick-rejection are the kinetic signature of this structural retail behaviour. Persists as long as Indian retail keeps using single-stock screeners — i.e., decades.
2. **U-shaped intraday volume profile is regulatory/microstructural** — NSE session structure (pre-open auction, opening flow, lunch lull, closing rebalance) creates the U-shape. Volume spikes in the lunch-trough window are statistically distinctive precisely because the baseline is low. The U-shape is not market-cycle-dependent.
3. **Internal precedent on this team** — `MEMORY.md project_detector_bugs_and_improvements.md` notes "volume spike reversal P1-P6" as an existing internal priority. Independent confirmation that this mechanism was already on the team's radar from prior backtest analysis. Re-implementing it under the §3.3 sanity-check gate gives a fresh, OOS-disciplined evaluation of an idea the team had organic conviction on.

Greenwood/Sammon decay caveat: volume-spike fading is a known pattern. Decay risk is real but is mitigated by the wick-confirmation filter (most retail volume-spike algos don't include wick-direction logic; the wick adds an idiosyncratic edge). Annual re-validation built into falsification criteria.

## Project-wide caveats addressed

- **Long-bias caveat:** the brief proposes a **bidirectional** detector with the SHORT side as primary edge. Per sub7/sub8 11-failure long-bias pattern, LONG-side ship requires sanity-check long-side PF ≥ short-side PF × 0.85. Volume-spike LONG (capitulation reclaim) is structurally similar to the long setups that failed in sub7/8 — the gate must be enforced strictly.
- **Decay risk (Greenwood/Sammon):** Acknowledged. The wick-direction filter is the differentiator that delays decay relative to plain volume-spike algos.
- **Time-of-day filter is critical** (round-3 §3 caution): must distinguish 11:00-15:10 retail-spike from 09:15-10:30 opening-flow institutional spike. The time gate is locked at 11:00-15:10 and is the single most important parameter — opening-flow spikes do NOT mean-revert in the same way and including them contaminates the dataset.
- **Continuation-vs-reversal disambiguation:** the 60% wick threshold is the structural separator. Without the wick filter, the setup would catch institutional accumulation (which continues) — losing the edge. The wick is non-negotiable.

## Evidence

Round-3 spec §3 lists 5 sources (Wright Research, TradingView India scripts, Mastertrust, Streak, internal MEMORY.md). All are operational publications or internal precedent on Indian intraday volume-spike at retail-MIS scale. Per round-3 mandate (Gate A passed), Indian retail-algo precedent is primary; in-house event study during sanity-check is the formal evidence substitute.

Peer-reviewed (secondary):
1. **Springer "Mean-Reverting Tendency in Stock Returns" chapter** (Indian-data study). https://link.springer.com/chapter/10.1007/978-81-322-1590-5_4
2. **Barber & Odean "All That Glitters" (Review of Financial Studies)** — attention-driven retail trading, foundational reference for FOMO peaks. (Cited as foundational; not Indian-specific but the mechanism transports.)

Indian retail-algo / internal sources (Gate A primary):
1. Wright Research top-5 reversal strategies — https://www.wrightresearch.in/blog/top-5-reversal-trading-strategies-for-traders/
2. TradingView India spike scripts — https://in.tradingview.com/scripts/spike/
3. TradingView India volume-breakout scripts — https://in.tradingview.com/scripts/volumebreakout/
4. Mastertrust beginner intraday strategies — https://www.mastertrust.co.in/blog/beginners-guide-to-intraday-trading-strategies
5. Streak strategies (built-in volume-anomaly + reversal-candle scanners) — https://www.streak.tech/strategies
6. TrendSpider Volume Spikes mechanism reference — https://help.trendspider.com/kb/indicators/volume-spikes
7. Internal: MEMORY.md `project_detector_bugs_and_improvements.md` — volume spike reversal P1-P6 (team-organic precedent)

## Direction

**Bidirectional**, with **SHORT side as the primary edge** and additional gate on long-side ship decision (see project-wide caveat above).

- **SHORT** when 5m bar has `vol_z ≥ 3.0` AND `(high − close) / (high − low) ≥ 0.6` (upper-wick rejection ≥ 60% of bar range) AND next bar is bearish reversal.
- **LONG** when 5m bar has `vol_z ≥ 3.0` AND `(close − low) / (high − low) ≥ 0.6` (lower-wick rejection ≥ 60% of bar range) AND next bar is bullish reversal.

Wick-direction determines side; wick-magnitude filter is the same threshold. Symmetric mechanic by design — the asymmetry between SHORT and LONG performance will come from the SEBI losing-flow tilt, NOT from the mechanic itself.

## Mechanic

**Setup name:** `volume_spike_exhaustion_reversal`
**Side:** Bidirectional (SHORT primary, LONG conditional ship).

**Sequence:**

1. **Continuous detection (every 5m bar close, 11:00-15:05 IST):**
   - For each F&O-200 mid/small-cap symbol, on the closing 5m bar:
     - `vol_mean_20 = rolling 20-bar mean of volume (excluding current bar)`
     - `vol_std_20 = rolling 20-bar stdev of volume`
     - `vol_z = (volume − vol_mean_20) / vol_std_20`
     - `bar_range = high − low`
     - `upper_wick_frac = (high − close) / bar_range` (if `bar_range > 0`)
     - `lower_wick_frac = (close − low) / bar_range`
   - **Trigger:**
     - SHORT: `vol_z ≥ 3.0` AND `upper_wick_frac ≥ 0.6` AND bar is red (close < open)
     - LONG: `vol_z ≥ 3.0` AND `lower_wick_frac ≥ 0.6` AND bar is green (close > open)
   - Threshold rationale (research-locked; post-sanity may reveal need to revise but DO NOT iterate on validation data per lessons.md 2026-05-01):
     - `vol_z ≥ 3.0`: 3-sigma volume anomaly. Round-3 §3 cites z=3.0 as the standard volume-spike threshold across Indian retail-algo platforms (Streak, TradingView). Below 2.5σ the noise floor (intraday volume bursts that don't carry structural meaning) dominates; above 3.5σ the sample size collapses too sharply.
     - `wick_frac ≥ 0.6`: 60% of bar range as rejection wick. This is the threshold Wright Research and TradingView volume-spike-reversal scripts converge on (range 50-70%); 60% is mid-range and balances signal-purity vs sample-size.

2. **Reversal-candle confirmation (next 5m bar):**
   - For SHORT: next bar's close < open AND close < prior bar's close.
   - For LONG: next bar's close > open AND close > prior bar's close.
   - Confirmation is single-bar — no extended formation requirement (preserves the timing edge; volume-spike-fade unwinds within 30-60 min).

3. **Entry:**
   - **Entry price:** confirmation bar's CLOSE.
   - **Direction:** as triggered.
   - **Active window gate:** 11:00-15:05 IST inclusive on the trigger bar (so confirmation bar latest is 15:05-15:10 = entry at 15:10). Excludes 09:15-10:55 (opening institutional flow, spikes do NOT mean-revert here per round-3 §3 caution) and 15:10+ (MIS auto-square contamination).

4. **Stop-loss:**
   - **Hard SL (SHORT):** trigger bar's high + 0.3% buffer (the wick high — clears the rejected level).
   - **Hard SL (LONG):** trigger bar's low − 0.3% buffer.
   - **Min stop distance:** 0.6% of entry.
   - The wick-extreme is a structurally meaningful stop level — if the wick is broken, the rejection thesis is invalidated.

5. **Targets:**
   - **T1** (50% qty): 1.0R move (R = entry − stop distance). First take-profit at risk-equivalent reward.
   - **T2** (50% qty): 2.0R move OR mean-revert to bar midpoint (`(high + low) / 2` of the trigger bar — symmetric pendulum exit), whichever first.
   - **Time stop:** 6 bars from entry (≈ 30 min) OR 15:15 IST hard stop, whichever first.

6. **Latch:** one fire per (symbol, day, direction) — no re-entry same direction same session.

**target_anchor_type:** `r_multiple` — UNLIKE the other two round-3 candidates (which are structural-anchored on VWAP / spread z-score), this setup uses arithmetic R-multiples for T1/T2 because the volume-spike bar itself is the only structural reference (the wick high/low) and beyond that there's no natural intraday level. R-multiples are the cleaner exit framework for spike-fade mechanics. T2 secondary anchor (bar midpoint) is structural and acts as a fallback if the R-multiple target is far away.

This explicit divergence from the other two round-3 candidates is a feature, not a bug — different asymmetry sources warrant different anchor types.

## Universe

**Universe:** F&O 200 with cap-segment filter:
- **Allowed cap segments:** `mid_cap`, `small_cap`
- **Excluded:** `large_cap` (volume-spike-with-wick is rarer — large-cap volume is dominated by institutional + index-fund flow; spikes that occur are usually fundamental-news-driven and DON'T mean-revert) and `micro_cap` (not in F&O 200).
- **Universe filter file:** `assets/fno_liquid_200.csv` (verified on disk per round-3 §3 Gate B).
- **Liquidity gate:** 20-day average `volume × close` ≥ ₹3 Cr (defends against thin-tape spikes that are computational artifacts of low-volume baselines — a single 100-share bar can trigger 5σ spikes spuriously in low-liquidity names).

**Why this universe:**
- Mid/small-cap F&O is where retail-FOMO + retail-panic concentrate (per SEBI FY23 + round-3 §3).
- F&O 200 inclusion ensures borrowable for MIS short.
- The liquidity gate is more important here than in candidate 1 because the spike-detection math itself depends on a meaningful volume baseline.

**Approximate symbol count after cap filter:** ~120-140 stocks (F&O 200 minus ~60-80 large-caps). Sample-size feasibility per round-3 §3: 8-20/day at vol_z ≥ 3.0; conservative annual count 2,000-5,000 events. Highest fire rate of the three round-3 candidates.

## Active window

**Setup formation + entry:** 11:00-15:05 IST (trigger bar). Entry on confirmation bar close (latest entry = 15:10).
**Hold horizon:** 6 bars (≈ 30 min) OR 15:15 IST hard stop.
**Latest possible exit:** 15:15 IST (5 min before MIS auto-square at 15:20).

**Why 11:00-15:05 entry window:**
- 09:15-10:55 = opening institutional flow. Volume spikes here are accumulation/distribution, NOT exhaustion. Fading them is the structural failure mode of generic volume-spike algos that don't have the time-of-day gate. Round-3 §3 explicitly flags this.
- 11:00-13:30 = lunch trough. The U-shaped baseline makes spikes here MOST distinctive (signal-to-noise highest).
- 13:30-15:05 = afternoon trend window. Volume spikes are increasingly retail-FOMO or panic, not institutional positioning.
- 15:05+ = closing-30-min institutional rebalance flow contaminates the spike interpretation; spikes here are mostly real positioning that continues.
- The 30-min hold is shorter than candidates 1 and 2 because spike-fade unwinds quickly — by 30 min either the rejection has played out or the original move resumes.

## Risks / falsification criteria

The setup is **wrong** (and should be retired) if:

1. **Phase-1 floor fails on validation/holdout:**
   - n < 500 trades over 12 months
   - NET PF < 1.10
   - NET Sharpe (daily) ≤ 0
2. **Direction asymmetry collapses:** if SHORT-side PF < 1.10 while LONG-side PF passes, the post-FOMO short thesis is weak; reconsider, but trust SHORT bias per sub7/sub8 long-bias failure pattern.
3. **Long-side ships and loses live:** if LONG passes sanity but underperforms in paper/live by ≥ 30% PF, retire LONG immediately, keep SHORT-only.
4. **Time-of-day filter fails:** if 09:15-10:55 spikes (separately tested in sanity report) show comparable PF to 11:00-15:05 spikes, the time-gate edge is illusory; the entire premise (retail-vs-institutional spike differentiation) is wrong, retire.
5. **Wick-filter fails:** if no-wick-filter spikes show comparable PF to wick-filtered spikes, the wick-direction edge is illusory, retire (or revisit detector design without wick filter — but that's a different setup).
6. **Decay signal:** rolling-60-trade NET PF drops below 1.05 sustained for 60 calendar days post-launch.

**Pre-coding sanity check** (mandatory per §3.3, BEFORE writing detector):
- Use existing 12-month 2024 5m enriched feathers on disk
- Compute `vol_z` (rolling 20-bar) and `wick_frac` per (symbol, bar)
- Identify trigger bars where `vol_z ≥ 3.0` AND `wick_frac ≥ 0.6` (per direction) AND time ∈ [11:00, 15:05]
- Apply reversal-candle confirmation
- Simulate entry → 6-bar / 15:15 exit with 1R/2R or bar-midpoint T1/T2 + 0.6% min-stop hard SL
- Compute NET PF using existing Indian fee model
- **Sensitivity analysis** (locked, report-only): also report PF at vol_z ∈ {2.5, 3.5} and wick_frac ∈ {0.5, 0.7}; do NOT re-tune on validation data per lessons.md 2026-05-01
- **Diagnostic comparisons**:
   - 09:15-10:55 spike PF (must be < 11:00-15:05 PF for time-gate edge to be real)
   - No-wick-filter spike PF (must be < wick-filtered PF for wick edge to be real)
   - vol_z ∈ {2.0, 2.5} PF (sub-threshold robustness check)
- Report PF / WR / Sharpe per direction (SHORT vs LONG separately) and per cap segment
- **Decision per §3.3:** PF ≥ 1.10 (SHORT side) → strong proceed; 1.0-1.10 → marginal; PF < 1.0 → retire. LONG-side PF ≥ SHORT-side PF × 0.85 → ship bidirectional; else SHORT-only.

## Data engineering plan

Required new components (only if sanity check passes):

1. **`tools/sub9_research/sanity_volume_spike_exhaustion_reversal.py`** — pre-coding sanity check. Reads 5m enriched feathers; computes vol_z + wick logic; no detector code yet. Will be retired after used.

2. **(post-sanity-check, only if APPROVED for full implementation):**
   - `services/volume_zscore_calculator.py` — rolling 20-bar vol_z, incremental for live mode (CLAUDE.md rule 3).
   - `structures/volume_spike_exhaustion_reversal_structure.py` — the detector.
   - Config keys (added to `config/configuration.json`, NO hardcoded defaults per CLAUDE.md rule 1):
     - `volume_spike_exhaustion_reversal.vol_z_threshold` = 3.0
     - `volume_spike_exhaustion_reversal.vol_window_bars` = 20
     - `volume_spike_exhaustion_reversal.wick_frac_threshold` = 0.6
     - `volume_spike_exhaustion_reversal.entry_window_start` = "11:00"
     - `volume_spike_exhaustion_reversal.entry_window_end` = "15:05"
     - `volume_spike_exhaustion_reversal.time_stop_bars` = 6
     - `volume_spike_exhaustion_reversal.time_stop_hard_ist` = "15:15"
     - `volume_spike_exhaustion_reversal.sl_buffer_pct` = 0.003
     - `volume_spike_exhaustion_reversal.min_stop_pct` = 0.006
     - `volume_spike_exhaustion_reversal.t1_r_multiple` = 1.0
     - `volume_spike_exhaustion_reversal.t2_r_multiple` = 2.0
     - `volume_spike_exhaustion_reversal.t2_use_bar_midpoint_fallback` = true
     - `volume_spike_exhaustion_reversal.cap_segments_allowed` = ["mid_cap", "small_cap"]
     - `volume_spike_exhaustion_reversal.min_adv_inr_cr` = 3.0
     - `volume_spike_exhaustion_reversal.long_side_enabled` = false   # toggled true only if sanity passes long-side gate

   No new ingestion needed. Volume column already in 5m enriched feather (round-3 §3 Gate B verified). Lowest infrastructural lift of the three round-3 candidates per round-3 spec.

## Sample-size feasibility

Per round-3 §3: 8-20 events/day across F&O 200 at vol_z ≥ 3.0 with wick_frac ≥ 0.6; conservative annual count 2,000-5,000 events. After reversal-candle confirmation (~50% pass rate) and active-window gate (11:00-15:05 = 65% of session): expect ~1,500-3,500 entries/year. Highest sample-size of the three round-3 candidates by 2-3x. n ≥ 500 met with comfortable margin.

Honest acknowledgement: this means slippage / fee drag is the binding constraint, not signal frequency. The sanity-check fee model must be precisely calibrated; if vol_z=3 spikes are dispersed across 120 mid/small-cap symbols, average position size will be thin and fixed-cost components (STT, exchange fees, GST) will materially compress the gross-to-net ratio. NET PF threshold 1.10 is a hard test for this candidate.

## Honest comparison to surviving setups

| Aspect | gap_fade_short (TRUSTED) | circuit_t1_fade_short (APPROVED) | volume_spike_exhaustion_reversal (proposed) |
|---|---|---|---|
| Indian-specific source | retail open momentum exhaustion | T+0 retail FOMO + operator pump | retail FOMO/panic peak + wick-rejection |
| Direction | short-only | short-only | bidirectional, SHORT primary |
| Active window | T+0 09:15-09:30 | T+1 10:30 single bar | T+0 11:00-15:05 continuous |
| Universe | small_cap | mid_cap, small_cap (no F&O restriction) | F&O-200 mid_cap, small_cap |
| Hold | 15-30 min MIS | 4h 45m MIS | 30 min MIS |
| Evidence base | empirical sub-7 validation | 5 peer-reviewed papers | 5+ Indian retail-algo platforms + internal precedent |
| Anchor | structural (PDC) | structural (gap edges) | **r_multiple** (with structural fallback) |
| Trigger type | gap level | circuit-band level | volume + wick (single-bar pattern) |
| Sample size (annual) | ~3-5K | ~750-1750 | ~1500-3500 (highest of round-3) |
| Correlation w/ existing | n/a | low | low (different time + different trigger class) |

The setup complements: highest-fire-rate of the three round-3 candidates, only one using r_multiple anchor, only one with single-bar pattern trigger (other two use multi-bar / cross-asset triggers). Lowest infrastructural lift. Internal team precedent (volume spike reversal P1-P6) gives organic conviction.

---

## Decision required

User to indicate:
- [ ] APPROVED — proceed to pre-coding sanity-check script
- [ ] REJECTED — reason
- [ ] RETIRE — kill candidate

Per sub-9 §3.3, no detector code is written until APPROVED and sanity-check passes (NET PF ≥ 1.10).
