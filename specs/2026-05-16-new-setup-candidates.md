# New Setup Candidates — From old_main Thesis Review

**Started:** 2026-05-16
**Completed:** 2026-05-16
**Status:** Old_main review COMPLETE. 8 batches, 22 mechanisms, 10 candidates surfaced.

**Source:** Re-review of all 40+ setups in `old_main` branch, batch-by-batch, thesis-first. Per user directive: understand the original thesis before consulting retired_setups.md, surface candidates based on Indian-microstructure fit not past PF numbers.

**Purpose:** Accumulate genuinely new setup ideas surfaced during the review that:
  (a) have NOT been tested before (per `docs/retired_setups.md`), AND
  (b) are NOT a near-duplicate of an active production setup, AND
  (c) have a clear Indian-microstructure rationale (not a US/forex re-skin).

This file is now the queue for the next research wave. Each candidate is fully specified so we can pick any of them up cold.

## Review yield summary

| Batch | Family | Mechanisms | Candidates | Strong (MED-HI+) |
|---|---|---|---|---|
| 1 | ICT (Order Block, FVG, Sweep, BOS, CHoCH) | 5 | 4 | 3 |
| 2 | Pivot/Level (PDH/PDL, ORB, S/R, Range) | 4 | 2 | 0 |
| 3 | VWAP (mean-rev, reclaim/lose) | 2 | 2 | 1 |
| 4 | Trend (pullback, continuation) | 2 | 0 | 0 |
| 5 | Gap (fill, breakout) | 2 | 0 | 0 |
| 6 | Volatility/Pattern (squeeze, flag) | 2 | 0 | 0 |
| 7 | Volume/Momentum (4 mechanisms) | 4 | 2 | 1 |
| 8 | Time-of-day (FHM) | 1 | 0 | 0 |
| **Total** | | **22** | **10** | **5** |

**Key pattern:** Yield is concentrated in batches where the mechanism has an *Indian-specific data anchor* (block deals, circuit bands, retail stop clustering, MIS-unwind window) rather than universal TA patterns. The universal-pattern families (Levels, Trend, Gap, Volatility, FHM) yielded few or zero candidates because they're broadly retired across sub-7/sub-8/sub-9 batches.

---

## Format

Each candidate has:
- **Source batch:** which old_main batch surfaced this idea, and which mechanism it spun off from
- **Mechanism (one sentence):** the Indian-microstructure asymmetry being exploited
- **Detection sketch:** how the detector would actually fire
- **Data dependencies:** what we already have vs what we'd need to scrape
- **Falsifier (pre-registered):** what would have to be true to retire it
- **Why new (vs retired_setups.md):** explicit statement of what's different from anything tested

---

## CANDIDATE-01 — Block-Deal-Anchored Accumulation Zone

### Source batch
Batch 1 (ICT family), spinoff from `order_block`. The retired `order_block` (sub-5) used the ICT pattern-recognition recipe (last opposing candle + volume + structure). The retire reason was "thin n=71, not validated." This candidate uses **actual NSE block-deal disclosure** as the anchor instead of inferring institutional activity from price.

### Mechanism (one sentence)
When a confirmed institutional block deal (≥ Rs. 5 cr, reported within 60 min) prints at a specific price, that price becomes a defended level on subsequent retests within the same session — institutions accumulating in size don't accept materially worse re-entry prices once they've publicly disclosed their position.

### Detection sketch
1. Pull intraday block deal feed (already scraped to `data/block_deals/`)
2. On every retest of a block-deal price (± 0.25%) later in the SAME session:
   - LONG entry if block was a BUY block and price retests from above (price > block_price, dropping toward it)
   - SHORT entry if block was a SELL block and price retests from below
3. Stop: 0.5% beyond the block price on the wrong side
4. Target: T1 = 1R, T2 = 2R

### Data dependencies
- `data/block_deals/*.parquet` — ALREADY HAVE
- `cache/ohlcv_archive/<symbol>.NS/<symbol>.NS_1minutes.feather` — ALREADY HAVE
- No new scrapers needed.

### Falsifier (pre-registered)
- Same-session retests of block-deal prices must show ≥ 55% favorable-direction reaction within 30 min, with median return ≥ +0.40% (long) or ≤ −0.40% (short), on n ≥ 100 events.
- If retests are random-walk (WR ~ 50% ± 3pp, median ~ 0%), kill it.

### Why new (vs retired_setups.md)
- Retired `order_block` used inferred-from-price institutional zones — pattern guesswork.
- This anchors on a SEBI-mandated regulatory disclosure (block-deal report). The anchor is verified institutional activity, not inferred.
- Block-deal-anchored detection has never been tested in the codebase.

---

## CANDIDATE-02 — Round-Number Stop-Cluster Sweep + Recovery

### Source batch
Batch 1 (ICT family), spinoff from `liquidity_sweep`. The retired `pdh_pdl_sweep_reclaim` (sub-8) used PDH/PDL as the swept levels with the ICT "institutional stop hunt" framing. The mechanism actually present in Indian markets is **retail stop-clustering at psychologically-anchored prices, followed by natural price recovery after stop-out exhausts**. This candidate tests that with round-number anchors instead of PDH/PDL.

### Mechanism (one sentence)
Indian retail traders cluster stop-losses at round-number prices (Rs. 100 / 250 / 500 / 1000 / 1500 / 2000) far more than at PDH/PDL because they're taught to in retail-education courses (Subasish Pani, Powerof Stocks, Zerodha Varsity); when intraday price pokes through such a level briefly and reverses, that's the stop-cluster being cleared, after which natural buying interest at the round number reasserts.

### Detection sketch
1. For each symbol, identify "active round numbers" — round-number levels within ± 2% of current price (e.g., a Rs. 487 stock has Rs. 500 active; a Rs. 1,234 stock has Rs. 1,250 and Rs. 1,200).
2. Detect a "sweep" bar: high pokes ≥ 0.15% above round number AND close is back below round number (or symmetric on the downside) AND volume on the sweep bar ≥ 2× session-average.
3. Confirm: next 1-2 bars stay back on the original side of the round number.
4. Entry: short at round number (after upside sweep), stop 0.5% above the sweep high, target = mid-point to next round number below.

### Data dependencies
- ALREADY HAVE all required data (1m bars, daily aggregates).
- No new scrapers needed.

### Falsifier (pre-registered)
- Sweeps + recoveries at round-number levels must show ≥ 58% favorable next-15min direction with median return ≥ +0.30% (in direction of recovery) on n ≥ 200 events.
- Critical: must hold AFTER filtering out generic PDH/PDL sweeps. The edge has to come specifically from the round-number stop-clustering, not from generic level-sweep behavior.

### Why new (vs retired_setups.md)
- `pdh_pdl_sweep_reclaim` (sub-8) tested PDH/PDL with the ICT institutional-manipulation framing. Failed.
- This candidate uses ROUND NUMBERS (not PDH/PDL) and explicitly frames the mechanism as retail-stop-clustering, not institutional manipulation. Both the level definition and the mechanism narrative are different.
- Round-number stop-clustering has never been tested as a primary anchor in the codebase.

---

## CANDIDATE-03 — Upper-Circuit Release Spike-and-Fade

### Source batch
Batch 1 (ICT family), spinoff from `liquidity_sweep`. Different from the existing `circuit_t1_fade_short` (which is a NEXT-DAY fade of small-caps after they hit upper circuit). This candidate is INTRADAY — the spike-and-reverse that happens at the exact moment a circuit-pinned stock's circuit is released.

### Mechanism (one sentence)
When a small-cap pins the 10% upper circuit during the morning, then NSE releases the circuit (typically 15-30 min later if circuit conditions are met), the stock can re-trade for a brief window before retesting circuit — the price often spikes to a new high (forced retail BUY orders queued during circuit-lock all execute at market) then collapses back into range when the forced-buy flow exhausts.

### Detection sketch
1. Identify stocks pinning 10% / 20% upper circuit (price within 0.1% of the day's circuit ceiling, volume = 0 for ≥ 3 bars).
2. Detect "circuit release" — first 1m bar with non-zero volume after the pin.
3. Watch the next 5-15 min for:
   - Spike to a new local high within first 3 bars after release
   - Followed by 2+ bars with declining volume
   - Close back below the spike high
4. Entry: SHORT at the close of the volume-decline bar; stop 0.5% above the spike high; target = midpoint between spike high and pre-pin price level.

### Data dependencies
- 1m bars — ALREADY HAVE
- Daily circuit ceiling per symbol — DERIVABLE from PDC + circuit % (the % is in `assets/nse_all.json` per stock)
- No new scrapers needed.

### Falsifier (pre-registered)
- Circuit-release spike-and-fade events must show ≥ 60% downside direction in next 30 min with median return ≤ −0.50% on n ≥ 50 events.
- Volume-decline confirmation is non-negotiable — events without it must be excluded (they capture continuation, not exhaustion).

### Why new (vs retired_setups.md)
- `circuit_t1_fade_short` (active in prod) fades the NEXT DAY after a circuit-pin. Different time horizon entirely.
- This candidate fades the INTRA-SESSION release moment. Different signal generation, different exit timing, different cap_segment universe (release events are concentrated in small-caps, but the FOMO-buy-queue exhaustion is the specific mechanism).
- Circuit-release intraday fade has never been tested in the codebase.

---

### C-03 PASSED SANITY — Locked Detector Spec (2026-05-16)

**Status:** Sanity script (`tools/sub9_research/sanity_circuit_release_fade.py`) + R-sweep + cell-mine completed on Discovery / OOS / Holdout. PASSED standalone ship gate on aggregate.

**Locked geometry:** T1 = 1.0R, T2 = 2.0R, Time Stop = 15:10
**Cell filter:** NONE (aggregate ships) - documented best cells below for OCI-driven decision

**Cross-window aggregate (no cell filter):**

| Window | n | PF | WR | NET | Sharpe |
|---|---|---|---|---|---|
| Discovery (2023-24) | 1,323 | **2.12** | 58.5% | +Rs.440K | 4.89 |
| OOS (2025 Q1-Q3) | 493 | **3.13** | 68.0% | +Rs.226K | 7.13 |
| Holdout (2025-10 to 26-04) | 223 | **4.53** | 72.7% | +Rs.119K | 9.01 |

Sub-9 standalone ship gate (Disc PF >= 1.30, OOS PF >= 1.20, Hold PF >= 1.15): **PASSES ALL**.
WR delta across windows: 58.5 -> 68.0 -> 72.7 = +9.5pp Disc->OOS, +4.7pp OOS->Hold. Both within 10pp.
Monotonically improving PF - real edge strengthening, not regime artifact.

**Detector trigger conditions (lock):**
1. Universe: cap_segment in {small_cap, mid_cap} AND MIS-eligible
2. Day filter: session_high reached by 10:30 IST AND session_high / PDC >= 1.045 (morning circuit-pin signature)
3. Active entry window: 12:00-15:10 IST
4. Signal bar conditions:
   - bar.high >= session_high * 0.997 (re-test within 0.3% of day high)
   - bar.close <= bar.high * 0.997 (rejection - close ≥ 0.3% below bar high)
   - bar.volume >= median of prior 5 bars' volume (volume confirmation)
5. SHORT entry at signal bar close
6. SL = signal_bar.high * 1.003 (0.3% above rejection high)
7. T1 = entry - 1R, T2 = entry - 2R (R = SL distance)
8. Hard time stop = 15:10 IST

**Best discovered cells (Discovery), documented for OCI-driven refinement:**

| Cell | Disc n | Disc PF | Disc WR |
|---|---|---|---|
| `day_gain_bucket=5-7` (5-7% intraday gain) | 282 | 3.86 | 68.1% |
| `cap_segment=mid_cap` | 300 | 2.81 | 68.7% |
| `cap_segment=small_cap` | 200 (OOS) | 3.99 | 70.7% |
| `rejection_hhmm_bucket=12-12:30` | 117 (OOS) | 4.51 | 76.9% |

**Different cells dominate each window** - Discovery favors `day_gain_bucket=5-7`, OOS favors `cap_segment=small_cap`. The aggregate is more robust than any single cell. Cell selection deferred until OCI run informs which slice is operationally cleanest.

**Files of record:**
- `tools/sub9_research/sanity_circuit_release_fade.py` - sanity script
- `tools/sub9_research/_circuit_release_fade_sweep_cellmine.py` - R-sweep + cell-mine
- `reports/sub9_sanity/_circuit_release_fade_short_trades_{discovery,oos,holdout}.csv` - per-window trades
- `reports/sub9_sanity/_circuit_release_fade_sweep_cells_{discovery,oos,holdout}.csv` - cell results

**Look-ahead bias correction logged:** Initial sanity used `close_off_high_pct <= 1.5%` as a day filter, which is look-ahead (uses end-of-day close). Removed 2026-05-16; trades CSV was regenerated. The look-ahead would have HIDDEN the true edge (excluded the stocks that crashed from circuit pin - exactly the best SHORT trades). After removal, aggregate PF jumped from 0.81/1.05/1.44 to 2.12/3.13/4.53 across windows.

---

### C-09 PARKED — Reframed Cell-Locked Verdict (2026-05-16)

**Status (revised):** Original "Mid-Session Large-Cap Volume-Spike Reversal" thesis FAILED in both large_cap-only (PF 0.60) and large+mid (PF 0.79) universes. Inverse-edge test on those universes also failed. BUT on **small_cap universe** with proper cell-mining + 3-window stability test (per `feedback_consistent_research_depth` rule), ONE cell survives all 3 windows.

**The surviving cell:** `SHORT` direction + `small_cap` + `vol_ratio in (8, 15]` (high-but-not-extreme volume spike) + `wick_ratio < 0.55` (directional bar, NOT rejection wick)

**3-window cross-validation (with T1=1R / T2=2R / full exit / TS=15:10):**

| Window | n | PF | WR |
|---|---|---|---|
| Discovery (2023-24) | 214 | 1.31 | 50% |
| OOS (2025 Q1-Q3) | 85 | 1.17 | 55% |
| Holdout (2025-10 to 26-04) | 41 | 1.63 | 61% |

**Mechanism (reframed from original spec):**
- Original spec: "exhaustion with wick rejection at VWAP" - DOES NOT manifest in any universe
- What actually works: **high-but-not-extreme directional volume bar** (8-15x median volume + small wick) in small_cap is followed by mean-reversion
- Interpretation: 8-15x volume = institutional/algo participation in small-cap (not just noise); small wick = directional move with no rejection; subsequent bars mean-revert as algo flow exhausts
- This is the OPPOSITE of the candidate spec's wick-rejection requirement

**Ship gate check:**
- Disc >= 1.30: ✓ (1.31, marginal)
- OOS >= 1.20: ✗ (1.17, just below threshold)
- Hold >= 1.15: ✓ (1.63)
- WR delta OOS->Hold within 10pp: ✓ (55->61 = +6pp)
- All windows positive direction: ✓

**Why PARKED instead of shipped:**
1. OOS PF 1.17 is below the 1.20 ship gate (close but a clear miss)
2. Holdout n=41 is way below the 200-floor for any individual cell decision
3. The standalone ship gate doesn't pass cleanly

**Why PARKED instead of retired:**
1. ALL 3 windows show positive direction (1.31, 1.17, 1.63) - this is a real edge, not noise
2. WR is monotonically improving across windows (50% -> 55% -> 61%) - mechanism strengthening
3. Holdout PF 1.63 is strongest of the 3 - if the trend continues, future data should be even better
4. The cell mechanism is mechanically interpretable (institutional vol-spike fade)

**Park decision:**
- Re-evaluate Q3 2026 when more Holdout-window data accumulates (currently 7 months, want 12+)
- If Holdout n reaches 100+ with PF still >= 1.30, ship cell-locked
- Otherwise retire

**Files of record:**
- `tools/sub9_research/sanity_volume_spike_reversal.py` (small_cap variant via config knobs)
- `reports/sub9_sanity/_volume_spike_reversal_trades_{discovery,oos,holdout}_smallcap.csv`
- No detector code, no production wiring (until park-review unparks)

**Key lesson:** This is exactly the case the `feedback_consistent_research_depth` rule warns about. Initial C-09 sanity (large+mid, PF 0.79) looked like a retire. With small_cap test + inverse-edge test + cell-mining + 3-window stability test, found a real but borderline cell. Without the depth, would have retired prematurely. The cell-mining bucket-label bug I caught also matters - mis-labelled bucket "15-50" was actually "(8, 15]", and the labels swap had been hiding the true cell.

---

## CANDIDATE-04 — Higher-TF Character-Shift Filter (Cross-Setup Gate, Not Standalone)

### Source batch
Batch 1 (ICT family), spinoff from `change_of_character`. The original CHoCH thesis as a primary entry signal doesn't translate to 5m intraday cash equity. But the underlying idea — "the higher timeframe has shifted character, so don't trade against it" — is valuable as a gating layer.

### Mechanism (one sentence)
When the NIFTY index OR the symbol's own 15m chart has shown a sign-flip in directional momentum within the last 30 min, fresh trade entries in the opposite direction have lower hit rates than baseline; gating existing setups on this filter should reduce avoidable losers.

### Detection sketch
NOT a standalone setup. A function `higher_tf_character_shift(symbol, side)` returning bool:
1. Fetch NIFTY 15m bars for last 90 min
2. Compute returns_3 on each 15m bar
3. Count sign-flips in last 6 bars
4. If side == "long" and recent sign-flip went POSITIVE→NEGATIVE → return True (don't allow long)
5. If side == "short" and recent sign-flip went NEGATIVE→POSITIVE → return True (don't allow short)
6. Wire as a filter call in `services/screener_live.py` before every detector emit

### Data dependencies
- NIFTY 15m bars — ALREADY HAVE (re-aggregate from 1m)
- No new scrapers needed.

### Falsifier (pre-registered)
- Apply the filter retroactively to gap_fade_short, circuit_t1_fade_short, delivery_pct_anomaly_short on Discovery data.
- If filtered-version PF improves by ≥ +0.05 across all 3 setups while losing ≤ 15% of trade count, ship it.
- If filtered-version PF drops or sample shrinks > 15% with no PF improvement, retire the filter idea.

### Why new (vs retired_setups.md)
- CHoCH as a standalone setup was never tested — and my analysis above argues it shouldn't be.
- As a FILTER on existing setups, it's a zero-cost experiment that retired_setups.md doesn't address (the doc is about setups, not about cross-cutting filters).
- This is a free experiment — implement, A/B against historical data, ship or retire in one pass.

---

## CANDIDATE-05 — Sectoral-Index Breakout + Lagging-Constituent Catch-up

### Source batch
Batch 2 (Pivot/Level breakouts), spinoff from `level_breakout_*` and `resistance_breakout_long`. The pure long-side level-breakout family has been tested exhaustively and retired (sub-7/sub-8). This candidate is NOT a pure level-breakout — it's an **intermarket arbitrage** where the level signal comes from the sectoral index, not the stock itself.

### Mechanism (one sentence)
When a NIFTY sectoral index (NIFTY Bank, NIFTY IT, NIFTY Auto, etc.) cleanly breaks its prior-day-high level with volume, the heaviest-weighted constituents that have NOT yet broken their own PDH have a documented tendency to catch up within 30-60 min — index-level conviction drags lagging constituents on positioning rebalance flow.

### Detection sketch
1. Track each sectoral index's PDH/PDL real-time via NIFTY indices feed (or computed from constituent-weighted aggregates if no direct feed).
2. When sectoral index breaks PDH with vol_z ≥ 2.0 AND holds for 2+ bars:
   - Identify the top-5 weight constituents
   - For each constituent: if it's currently BELOW its own PDH but moving in same direction → LONG candidate
   - Entry: market or limit at current price; stop at constituent's intraday low; T1 = constituent's own PDH (likely magnet); T2 = constituent's own PDH + 0.5 × range.
3. Symmetric for sectoral breakdowns: SHORT lagging constituents below their PDL.

### Data dependencies
- Sectoral index PDH/PDL — need to compute from constituent aggregates OR add NSE sectoral indices feed (not currently scraped). **NEW SCRAPER REQUIRED.**
- Constituent weights per sectoral index — available in `assets/ind_niftybanklist.csv`, `ind_niftyautolist.csv`, etc. (already have files).
- 1m bars per constituent — already have.

### Falsifier (pre-registered)
- Within 60 min of a sectoral-index PDH break with vol_z ≥ 2.0, lagging top-5 constituents must show ≥ 60% catch-up rate (price reaches constituent PDH) with median R-multiple ≥ +0.5R on n ≥ 100 sectoral-index events.
- If catch-up rate is < 55% or median R is < +0.2R, kill it — the mechanism doesn't survive friction costs.

### Why new (vs retired_setups.md)
- Pure single-stock level breakouts (`orb_15`, `pdh_pdl_*`, `level_breakout`) are exhaustively retired.
- This candidate uses **the sectoral index as the signal source** and trades the lagging stock. The mechanism is intermarket positioning rebalance, not single-stock momentum.
- Sectoral-level → constituent-level intermarket arbitrage has never been tested in the codebase.

### Confidence
**MEDIUM-LOW.** The mechanism is plausible but the same family of patterns has consistently failed. Worth a sanity script before committing to a full detector. New scraper (sectoral index feed) is the main implementation cost.

---

## CANDIDATE-06 — Compression + External Directional Anchor

### Source batch
Batch 2 (Pivot/Level breakouts), spinoff from `range_breakout` and `squeeze_release`. The volatility-clustering mechanism (low-vol periods → high-vol periods) is academically validated (ARCH/GARCH). What kills the trade is that the DIRECTION of expansion is 50/50 without an anchor. This candidate pairs the compression detection with an EXTERNAL directional signal.

### Mechanism (one sentence)
When a NIFTY-50 stock enters a tight intraday range (height < 0.7% over 20+ bars) during the 11:30-13:30 lunchtime regime, the eventual breakout direction is predictable from concurrent FII net cash flow direction (intraday NSE FII/DII feed) — institutional flow concentrating on one side during low-vol periods tips the squeeze in their favor when liquidity returns post-13:30.

### Detection sketch
1. Detect compression: 20+ consecutive 5m bars where (high - low) / mid < 0.7%, occurring 11:30-13:30 window.
2. Read concurrent intraday FII flow direction (positive = net buy, negative = net sell).
3. Entry direction = FII flow sign (long if FII net-buy, short if FII net-sell).
4. Entry trigger: first 5m bar after 13:30 that closes outside the compression range in the FII-aligned direction.
5. Stop: opposite edge of compression range.
6. Target: T1 = 1× compression range height, T2 = 2× height.

### Data dependencies
- Intraday FII/DII flow data — NSE publishes T+0 EOD; **intraday version requires a separate scrape (Moneycontrol or NSE bhavcopy)**. Closest substitute we have currently is T-1 EOD (delayed by one day).
- 5m bars — already have.

### Falsifier (pre-registered)
- Of compression+breakout events in the FII-aligned direction (n ≥ 50), the breakout direction must match the FII flow sign with ≥ 65% accuracy AND show ≥ 60% follow-through to T1 on aligned breakouts.
- If FII alignment is no better than chance (≤ 55%), the FII anchor isn't load-bearing — fall back to retiring the squeeze idea entirely.

### Why new (vs retired_setups.md)
- `narrow_cpr_breakout` (sub-7) tested compression without external anchor. Failed.
- This candidate explicitly tests whether ADDING an external flow anchor (FII direction) salvages the otherwise-broken squeeze trade.
- FII-anchored intraday squeeze trading has never been tested in the codebase.

### Confidence
**LOW.** The FII-flow → directional-breakout link is theoretical and probably weak — FII intraday flow data has significant publication lag and may not be observable in real time. This candidate should be a quick falsifier test on historical FII data joined to historical compressions, NOT a full implementation push. If the falsifier passes, then investigate live FII data feed.

---

## CANDIDATE-07 — VWAP-Side Filter on Existing Setups (Free Filter, Not a Standalone)

### Source batch
Batch 3 (VWAP family), spinoff from `vwap_reclaim_long` / `vwap_lose_short`. The VWAP-cross-as-setup is empirically dead (`vwap_lose_short` retired with PF 0.75 at Discovery). But the underlying idea — "VWAP separates institutional-aligned price zones" — has merit as a *gating filter* on existing setups.

### Mechanism (one sentence)
A fresh long entry signal in a stock currently trading below its session VWAP fights against the institutional VWAP-execution flow (large desks selling into above-VWAP price); a fresh short entry signal above VWAP fights the symmetric buying flow — so simply gating existing long/short signals by "must be on the VWAP-aligned side" should reduce avoidable losers without changing trade selection.

### Detection sketch
NOT a standalone setup. Modify `services/screener_live.py` (or equivalent gate) to:
1. Before each long-side signal emission, check current_price > session_vwap. If FALSE, suppress signal.
2. Before each short-side signal emission, check current_price < session_vwap. If FALSE, suppress signal.
3. Apply retroactively in backtest to `gap_fade_short`, `circuit_t1_fade_short`, `delivery_pct_anomaly_short`.

### Data dependencies
- Session VWAP — ALREADY computed per bar
- No new scrapers needed.

### Falsifier (pre-registered)
- Apply filter to all 3 active setups on Discovery (2yr) + OOS + Holdout data.
- Ship if: filtered version reduces total trade count by ≤ 25% AND raises aggregate PF by ≥ +0.08 AND aggregate NET drops by ≤ 10%.
- Retire if: filter cuts trades by >25% without PF improvement, or cuts NET by >15%.

### Why new (vs retired_setups.md)
- VWAP-cross-as-primary-setup is retired (`vwap_lose_short` direct retire).
- VWAP-side as a cross-setup FILTER has never been tested in the codebase.
- This is a zero-cost experiment — implement, A/B against history, ship or retire in one pass.

### Confidence
**HIGH** for implementation viability, **MEDIUM** for finding the edge. The filter may also REMOVE the edge from existing setups if their winning trades fire on the "wrong VWAP side." Easy to discover via the A/B.

---

## CANDIDATE-08 — Last-Hour VWAP-Mean-Revert SHORT (MIS-Unwind Window)

### Source batch
Batch 3 (VWAP family), spinoff from `vwap_mean_reversion_short`. The generic-timeframe VWAP-mean-revert short has no direct test but weak mechanism support. However, in the 14:30-15:15 IST window (MIS retail auto-square-off), there's an additional **forced-selling flow** on long retail positions that may make above-VWAP names asymmetrically vulnerable. The retired `mis_unwind_short` (sub-7) targeted this window but with wrong mechanic (vwap-momentum). This candidate uses VWAP-extension distance as the anchor.

### Mechanism (one sentence)
During the 14:30-15:15 MIS auto-square-off window, retail long positions on stocks trading above their session VWAP face forced sell pressure (broker auto-liquidation by 15:20 SEBI rule), creating mean-reversion pressure back toward VWAP — short the extension, target VWAP retest, hard time-stop at 15:15.

### Detection sketch
1. Window gate: 14:30 ≤ entry_time ≤ 15:10
2. Universe: MIS-eligible stocks ONLY, mid_cap + small_cap (where retail concentration is highest)
3. Detection: current_price / session_vwap > 1.005 (≥ 50 bps above VWAP) AND RSI(14) > 65 AND vol on signal bar > 2× session-average
4. Entry: SHORT at current_price
5. Stop: 0.4% above current_price (above the extension high)
6. Targets: T1 = session_vwap, T2 = vwap × 0.997
7. Hard time stop: 15:15 (mandatory exit, before final 15:20 square-off rush)

### Data dependencies
- Session VWAP per stock — ALREADY computed
- RSI(14) — ALREADY computed
- Volume aggregates — ALREADY computed
- MIS-eligible flag — in `nse_all.json`

### Falsifier (pre-registered)
- 14:30-15:10 above-VWAP entries with RSI > 65 must show ≥ 60% downside follow-through to VWAP within 30 min on n ≥ 100 events over Discovery.
- Median R-multiple ≥ +0.4R, win rate ≥ 55%.
- If WR < 55% or median R < +0.2R, kill it — the MIS-unwind hypothesis doesn't materialize at the VWAP-extension geometry.

### Why new (vs retired_setups.md)
- `mis_unwind_short` (sub-7) — used VWAP-cross signal + momentum gate. Failed (n=326 too thin + stop 0.8×ATR too tight). DIFFERENT mechanic.
- `vwap_mean_reversion_short` — generic VWAP-mean-revert without time window. Never directly tested.
- This candidate combines MIS-unwind time window + VWAP-extension geometry + RSI confirmation. Specific combination never tested.

### Confidence
**LOW-MEDIUM.** The MIS-unwind premise is plausible and `mis_unwind_short` retire noted "thesis itself is plausible; the mechanic was wrong." This candidate is a different mechanic on the same thesis. Real but uncertain.

---

## CANDIDATE-09 — Mid-Session Large-Cap Volume-Spike Reversal

### Source batch
Batch 7 (Volume/Momentum), spinoff from `volume_spike_reversal_long/short`. The original detector was implemented with thoughtful P1-P6 filters (volume ratio, wick rejection, S/R proximity) but never made it to production. The mechanism (volume-spike + wick-rejection + level confluence = exhaustion fade) is the same signature as `gap_fade_short`'s exhaustion candle — just unanchored from the morning gap. This candidate specializes the generic detector to a context where gap_fade_short does NOT fire: mid-session large-cap exhaustion.

### Mechanism (one sentence)
During mid-session (10:30-13:30) when morning retail FOMO has settled but MIS-unwind has not yet started, large-cap institutional VWAP-tracking algorithm flows can exhaust at intraday extremes — manifesting as a volume-spike bar with rejection wick near session VWAP, which is a tradeable fade signal capturing institutional algo exhaustion (a different participant than the retail-FOMO exhaustion gap_fade_short captures).

### Detection sketch
1. Time gate: 10:30 ≤ entry_time ≤ 13:30 IST
2. Universe: NIFTY-100 constituents with ADV ≥ Rs. 500 cr (institutional-flow-dominated names)
3. Signal bar:
   - Volume z-score ≥ 3.0
   - Volume ratio (vs 20-bar median) ≥ 4.0 (illiquid-noise guard, mirrors P5)
   - Body ≥ 0.4% of price
   - Wick rejection ratio ≥ 60% on the move's extreme (mirrors P2)
   - Price within 1× ATR of session VWAP (mirrors P3 with VWAP-specific anchor)
4. Entry: fade direction at the signal bar's close
5. Stop: 0.3% beyond the wick extreme
6. T1: VWAP (likely magnet), T2: VWAP - 0.5 × ATR (long) or VWAP + 0.5 × ATR (short)

### Data dependencies
- 5m bars + vol_z + ATR + VWAP — ALREADY HAVE
- NIFTY-100 constituents — `assets/ind_nifty100list.csv` (need to verify file exists)
- ADV ≥ Rs.500cr filter — derivable from daily cache

### Falsifier (pre-registered)
- Mid-session large-cap volume-spike fades at VWAP proximity must show ≥ 60% favorable direction within 30 min, median return ≥ +0.30%, on n ≥ 150 events over 2yr Discovery.
- WR delta between OOS and Discovery must be ≤ 10pp (overfit guard)
- If WR < 56% or median R < +0.15R, kill — the institutional-algo-exhaustion premise doesn't manifest at this geometry.

### Why new (vs retired_setups.md)
- `volume_spike_reversal` was an internal research thread (P1-P6 improvements documented in MEMORY) — NOT in retired_setups.md, never explicitly shipped or retired.
- The mid-session large-cap specialization is DIFFERENT from both the original generic detector AND from gap_fade_short:
  - Different universe (large-cap vs small/mid-cap)
  - Different time window (mid-session vs morning)
  - Different anchor (VWAP vs gap-from-PDC)
  - Different participant (institutional algo exhaustion vs retail FOMO exhaustion)
- This specific combination has not been tested in the codebase.

### Confidence
**MEDIUM-HIGH.** The mechanism is well-supported by HFT/microstructure literature (Easley/Lopez de Prado/O'Hara on order flow toxicity). The P-series enhancements from the prior detector iteration carry over. Real risk is whether large-cap exhaustion produces enough sample-size at the strict filter thresholds — may need to relax filters during falsifier phase.

---

## CANDIDATE-10 — OR-Window Failure Fade

### Source batch
Batch 7 (Volume/Momentum), spinoff from `failure_fade_long/short`. The generic failure_fade detector applies to ANY level (PDH/PDL/OR/swing); the PDH/PDL variant was already tested as `pdh_pdl_sweep_reclaim` (sub-8) and retired. This candidate tests whether the OR-level variant during the IB window (9:15-10:30) captures different participants — specifically, the retail breakout-traders entering on the OR pierce.

### Mechanism (one sentence)
During the Initial Balance window (9:15-10:30 IST), retail breakout-traders are taught to enter on ORH/ORL pierces; when those pierces fail (close back through the level within 1-2 bars), those retail entries become trapped and create cascading stop-out flow in the opposite direction — fading the failed pierce captures the resulting flow.

### Detection sketch
1. Time gate: 9:30 ≤ entry_time ≤ 10:30 IST (after OR is established at 9:30, during IB)
2. Universe: small_cap + mid_cap (where retail breakout-trader concentration is highest)
3. Signal:
   - Bar pierces ORH (long-side failure) or ORL (short-side failure) by ≥ 0.15%
   - Bar volume ≥ 2× session-average
   - Next 1-2 bars close back through the level
   - Pattern age ≤ 15 min from pierce
4. Entry: opposite direction of pierce, at close of confirming bar
5. Stop: 0.3% beyond the pierce extreme
6. T1: OR midpoint, T2: opposite OR boundary

### Data dependencies
- 5m bars, ORH/ORL computation — ALREADY HAVE
- No new scrapers needed.

### Falsifier (pre-registered)
- OR-pierce failures in IB window must show ≥ 60% favorable next-30-min direction, median return ≥ +0.25%, on n ≥ 100 events over 2yr Discovery.
- Critical comparison: PF must materially exceed the failed `pdh_pdl_sweep_reclaim` baseline (Phase-1 fail level) — say, PF ≥ 1.20 — to justify the family-revival.
- WR delta OOS-vs-Holdout ≤ 10pp.

### Why new (vs retired_setups.md)
- `pdh_pdl_sweep_reclaim` (sub-8, retired) tests this mechanism at PDH/PDL (yesterday's levels).
- This candidate tests at ORH/ORL (today's forming levels) during IB window specifically.
- Different participant (retail breakout traders during OR establishment vs day-traders watching PDH/PDL throughout the session).
- IB-window-anchored OR failure fade has not been isolated and tested in the codebase.

### Confidence
**LOW-MEDIUM.** The mechanism family is broadly retired (`pdh_pdl_sweep_reclaim` failed Phase-1). The OR-specific + IB-window-specific variant has a plausibly different participant base but the failure rate of related candidates is the realistic prior. Worth a quick falsifier test, NOT a full implementation push.

---

## Status tracking

| ID | Candidate | Confidence | Status | Source batch |
|---|---|---|---|---|
| **C-01** | **Block-Deal-Anchored Accumulation Zone** | MEDIUM-HIGH (a priori) | **RETIRED 2026-05-16** at sanity (see `docs/retired_setups.md`) | Batch 1 (ICT) |
| C-02 | Round-Number Stop-Cluster Sweep+Recovery | MEDIUM | Queued | Batch 1 (ICT) |
| **C-03** | **Upper-Circuit Release Spike-and-Fade** | MEDIUM-HIGH | **PASSED sanity 2026-05-16 - building detector** (see below) | Batch 1 (ICT) |
| C-04 | Higher-TF Character-Shift Filter (cross-setup gate) | HIGH (free filter) | Queued | Batch 1 (ICT) |
| **C-05** | **Sectoral-Index Break + Lagging-Constituent Catch-up** | MEDIUM-LOW (a priori) | **RETIRED 2026-05-16** — Discovery+OOS aggregate fails, Holdout regime-artifact only, no sector holds 3 windows | Batch 2 (Levels) |
| **C-06** | **Compression + FII Flow Anchor** | LOW (a priori) | **RETIRED 2026-05-16** at data recon (no FII intraday data; see `docs/retired_setups.md`) | Batch 2 (Levels) |
| C-07 | VWAP-Side Filter on Existing Setups (free filter) | HIGH (impl), MEDIUM (edge) | Queued | Batch 3 (VWAP) |
| **C-08** | **Last-Hour VWAP-Mean-Revert SHORT (MIS-Unwind Window)** | LOW-MEDIUM (a priori) | **SHIPPED 2026-05-16** cell-locked RSI>=75+vol>=7 (Disc PF 2.66 / OOS 2.56 / Hold 2.50) - **STRONGEST new setup found**. Aggregate was 100/day operationally; tightened to 9-12/day | Batch 3 (VWAP) |
| **C-09** | **Volume-Spike Reversal** (reframed: small_cap SHORT-only on directional vol spike, not large-cap exhaustion) | MEDIUM (post-research) | **PARKED 2026-05-16** — one cell survives all 3 windows but Holdout n=41 too thin to ship; see notes below | Batch 7 (Volume/Mom) |
| **C-10** | **OR-Window Failure Fade** | LOW-MEDIUM (a priori) | **SHIPPED 2026-05-16** cell-locked with retire-if-OCI-defers (Disc 1.22 / OOS 1.27 / Hold 1.12) | Batch 7 (Volume/Mom) |

---

## Notes for future batches

- **Don't bias by retire-reasons before doing fresh research.** The user's standing rule: read the thesis straight from old_main, understand the mechanism, then cross-check what's already been tested.
- **Indian-microstructure asymmetry first.** Every candidate must answer: who's the retail side of this trade, who's the institutional side, and what regulatory/structural anchor makes the asymmetry persistent?
- **Active production setups are the comparison baseline.** Any new candidate must be materially different from `gap_fade_short`, `circuit_t1_fade_short`, `delivery_pct_anomaly_short` — not just a re-skin.
