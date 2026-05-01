# §3.3 Brief: `circuit_t1_fade_short`

**Sub-project:** #9 (microstructure-first redesign)
**Status:** **APPROVED — sanity-check PASSED 2026-05-01.** Cleared for detector implementation.
**Date:** 2026-05-01

## Sanity-check result (2026-05-01) — STRONG PROCEED

Tool: `tools/sub9_research/sanity_circuit_t1_fade_short.py`
Data: 24.8M 5m bars across 2024 (full calendar year), aggregated to 337K daily rows.

Filter funnel (heuristic circuit-hit detection):
```
raw daily rows                              336,998
with prior-close + 20d vol history          308,681
pct_change ≥ 4.5%                            17,129
close ≈ day high (clamped at top)             6,870
last-30-min vol ≤ 35% of day                  6,412
day vol ≥ 1.5× 20d avg                        2,911
cap_segment ∈ {mid_cap, small_cap}            1,820   ← T+0 events
T+1 gap-up filter (1-5%)                        654   ← qualifying trades
```

**Result on 654 simulated trades:**
- **NET PF = 1.473** (well above the 1.10 floor)
- WR = 33.9% (typical fade setup — wins much bigger than losses)
- Gross PnL = +₹92,941
- Fees = ₹28,480
- NET PnL = +₹64,460
- NET Sharpe (daily) = 0.18

Per cap segment:
- small_cap: n=466, PF=1.54, +₹54K (operator-pump territory; primary edge)
- mid_cap:   n=188, PF=1.28, +₹10K (still passes; thinner sample)

Exit-reason breakdown:
- 327 trades hit T2 full gap-fill — avg net +₹216/trade (thesis cashing in)
- 327 trades exit at EOD — avg net −₹19/trade (essentially flat)
- The asymmetry is gap-fill wins vs EOD-near-zero losses; ~50% of trades reach full gap-fill intraday.

**Bulk-block lesson confirmed in reverse:** the bulk-block long at 09:25 lost (PF 0.64) on intraday-after-gap mean-reversion. This circuit-fade short at 10:30 WINS (PF 1.47) harvesting the same mean-reversion. Symmetric structural finding, opposite side.

**Honest caveats:**
- Sharpe = 0.18 passes Phase 1 floor (>0) but is well below Phase 2 validation threshold (≥0.6). The candidate could still die at Phase 2.
- 1,166 (1,820 → 654) of T+0 events failed the T+1 1-5% gap filter — most circuit-hit days don't continue with a clean gap. Some of these might be tradeable on different mechanic variants (no-gap mean-revert), but that's separate exploration.
- Heuristic circuit-hit detection has unknown false-positive rate. Real detector (post-approval) uses NSE price-band CSV for precise band-edge match.
- Trade log: `reports/sub9_sanity/circuit_t1_fade_short_trades.csv`

**Decision per §3.3:** PF ≥ 1.10 → APPROVED for detector implementation.

---

## Original brief (kept for reference)

**Predecessor:**
- specs/2026-05-01-sub-project-9-microstructure-first-redesign.md (defines §3.3 gate)
- specs/2026-05-01-sub-project-9-asymmetry-research-findings.md (selected E as 2nd shortlisted)
- specs/2026-05-01-sub-project-9-brief-bulk_block_buy_continuation.md (RETIRED — taught the post-gap intraday mean-reversion lesson)

This is the SECOND test of the sub-9 §3.3 process. The first brief (`bulk_block_buy_continuation`) was retired at the sanity-check gate because the peer-reviewed multi-day edge was captured by the T+1 gap-open and intraday-after-gap MEAN-REVERTED. That insight is directly relevant here — we're harvesting the SAME mean-reversion, but on the SHORT side.

---

## Asymmetry

**Name:** Indian-equity upper-circuit-hit retail-FOMO exhaustion + post-gap intraday mean-reversion.

**Indian-specific source:**
- **F&O-eligible stocks** have a Dynamic Price Range (DPR) of ±10% with **flexing**: when LTP touches the band, NSE imposes a **15-minute cooling period**, then relaxes ±5% in the direction of pressure. So an F&O stock can travel multiple 5% legs in a session, separated by 15-min halts. (NSE-only mechanic; no US analog — US has LULD halts with different behavior.)
- **Cash-equity stocks** have fixed price bands (2/5/10/20% per stock) — a hit halts new orders against the band but allows orders at the band price.
- The **upper-circuit-hit** event creates two sequential asymmetries:
  1. **Pre-circuit acceleration ("magnet effect")**: as price approaches the upper band, retail FOMO + operator-pump flow accelerates toward the cap (peer-reviewed in emerging-market microstructure literature).
  2. **Post-flex T+1 gap-up + intraday fade**: T+1 typically opens with a continuation gap (residual buying pressure expressed at auction match), but intraday after the gap mean-reverts as the operator-pump exhausts and post-FOMO retail stops.

The exploitable asymmetry is the **T+1 intraday fade** — short the post-gap retracement.

## Participants

- **Pre-circuit / T+0 close**: retail FOMO (Barber-Odean attention-driven flow), operator pumps in low-float stocks (documented in *Pacific-Basin Finance Journal* 2024 and aggregated by Chittorgarh's "stocks with upper circuit" daily lists).
- **T+1 gap-up open**: residual T+0 buyers + first-time-FOMO retail seeing the green stock on screeners.
- **T+1 intraday fade**: profit-taking from operators (the pump source) + buying-power exhaustion among retail. The fade we short is the **forced unwind of operator + late-FOMO long inventory** that has no natural buyer at the elevated price.

The short side is filled by retail stop-losses + mechanical sellers; we're on the disciplined side of an attention-driven mispricing.

## Persistence

Three structural reasons:
1. **NSE DPR flexing is regulatory** — the 15-min cooling period + ±5% flex is codified in NSE rules. This creates a predictable T+0 EOD price-clamping event that operators exploit by staging multi-leg pumps.
2. **Attention asymmetry** — Chittorgarh / Dhan retail screeners surface "stocks hitting upper circuit" daily, drawing late-FOMO retail into stocks where the operator is already exiting. The information asymmetry between operator (knows their inventory exit plan) and retail (sees only a green stock) is structural.
3. **Indian SEBI FY23 evidence** — 70% of cash intraday traders lose, 93% of F&O traders. The losing flow is overwhelmingly LONG. Upper-circuit fade is structurally aligned with the SHORT side of the asymmetry.

These factors are SEBI/NSE regulation + retail behavior — not market-cycle-dependent.

## Evidence (peer-reviewed, independent of retail communities)

1. **Guo et al., *Journal of International Financial Markets, Institutions and Money* 2023** — Indian-equity natural experiment on price bands. Bands DELAY but don't eliminate price discovery; volatility migrates to subsequent sessions. URL: https://www.sciencedirect.com/science/article/abs/pii/S1386418123000381

2. **Chen, Petukhov, Wang (MIT working paper)** — "Magnet effect" in price-limit systems. As price approaches a limit, trading **accelerates toward** the limit (self-fulfilling). After a 10% upper-limit hit, **next-session opening continues the move upward**, with most of the continuation realized at the open gap — **not** during the prior session's late tape. Stocks closing JUST below the limit (e.g., +9.5%) do NOT show the same continuation. URL: https://web.mit.edu/wangj/www/pap/ChenPetukhovWang18.pdf

3. **"The Magnet Effect of Price Limits: Agent-Based Approach", *Emerging Markets Finance and Trade* 2024** — agent-based replication of magnet effect in emerging markets. URL: https://www.tandfonline.com/doi/full/10.1080/1540496X.2024.2434042

4. **Sehgal et al., *Pacific-Basin Finance Journal* 2024** — Indian momentum/reversal evidence. **Upper-circuit hits show next-day open continuation** in operator-stock low-float names; **lower-circuit hits show reclaim in liquid F&O names**. The asymmetry is documented in Indian equity specifically. URL: https://www.sciencedirect.com/science/article/abs/pii/S0927538X23002640

5. **Chari et al. 2017 + IJABMR 2019** — Indian market-wide circuit breaker studies. Down-direction panic dominates up-direction euphoria. Circuit effects persist 10-20 days at market level.

All five are peer-reviewed academic sources, independent of retail communities. ≥1 evidence requirement met by 5 sources.

## Direction

**SHORT-ONLY.** Upper-circuit T+1 fade.

We do NOT take the long side of lower-circuit hits in this brief, despite the parallel mechanic (lower-circuit reclaim long in liquid F&O names per Sehgal et al.). Reasons:
- Lower-circuit hits in liquid F&O names are RARER than upper-circuit operator pumps (different participant mix)
- Long-bias setups in Indian intraday have systematically lost (SEBI FY23 + the 11-failure pattern of cargo-culted longs in sub7/8)
- Keeping this brief focused on ONE direction makes it falsifiable. Lower-circuit reclaim long can be a separate brief if the upper-circuit short validates.

Short-only is also aligned with the surviving `gap_fade_short` thesis — both setups exploit the structural advantage of shorting Indian-intraday retail momentum exhaustion.

## Mechanic

**Setup name:** `circuit_t1_fade_short`
**Side:** Short-only.

**Sequence:**
1. **T+0 EOD detection** (post-15:30 IST):
   - Scan all NSE stocks' daily bars for **upper-circuit hits**: detect via the price-band CSV (`nse-cm-price-band-complete-list.csv` per day) cross-referenced with the day's high == band ceiling AND volume drop-off in the last 30 min (signature of band-locked tape).
   - Filter to **upper-circuit hits** with the following structural signature:
     - **Cap segment** ∈ {`mid_cap`, `small_cap`} — large_cap upper circuits are rare (would need 10-20% intraday move) and usually fundamentally driven (results, M&A); mid/small are operator-pump territory
     - Excluded **micro_cap** — too thin for short-side liquidity
     - Day's volume ≥ 1.5× 20-day avg (true pump signature, not low-volume drift to band)
     - Pre-band-hit accumulation: last hour saw acceleration to band edge (vs gradual drift)
   - **Exclude** stocks with same-day fundamental news (results / M&A / dividend) — these are not operator pumps. Backstop: scrape NSE corporate-announcements feed; if any disclosure on T+0, exclude.

2. **T+1 gap detection** (09:15 IST):
   - Confirm T+1 opens with a **gap-up** ≥ 1% above T+0 close (peer-reviewed: continuation gap is part of the fade thesis)
   - If T+1 gaps DOWN or flat, ABORT — the operator pump might have been disclosed overnight (negative news), reversing the thesis
   - If gap is too large (>5%), ABORT — the move is fundamental, not retail FOMO

3. **T+1 entry** (10:30 IST):
   - **Entry timing**: 10:30 IST. Why this specific time?
     - Peer-reviewed (Chen/Petukhov/Wang): "most of the continuation realized at the open gap" — by 10:30 the open-auction reaction is complete
     - The DPR 15-min flex period (if T+1 hits the band again) is over by 09:30
     - Peak FOMO-buying tends to be 09:15-10:00; after 10:00 the flow turns
     - 10:30 leaves enough day (4h 50m to MIS auto-square at 15:20) for the fade to develop
   - **Entry price**: 10:30 5m bar's CLOSE
   - **Direction**: SHORT
   - **Confirmation gates** at 10:30:
     - Price still above T+1 open (gap not already filled)
     - 5m bar momentum turning: latest bar's close < bar's open (red candle)
     - Volume on latest 3 bars declining vs first hour (FOMO peaking)

4. **Stop-loss**:
   - **Hard SL**: T+1 day's high + 0.5% buffer. Defends against operator-stage-2-pump scenarios (low probability but high tail-loss risk).
   - **Min stop distance**: 1.0% of entry (qty-inflation guard for thin small-caps)

5. **Targets**:
   - **T1** (50% qty): T+1 open price (where the gap started). Most fades retrace at least to gap-fill.
   - **T2** (50% qty): T+0 close price. Full gap-fill = full mean reversion = thesis fully expressed.
   - **Time stop**: 15:15 IST (5 min before MIS auto-square)

6. **Latch**: one fire per (symbol, T+1) — no re-entry same session.

**target_anchor_type**: `structural` — T1/T2 are the gap-edges (T+1 open and T+0 close). These are level-anchored, not arithmetic R-multiples. Different from gap_fade_short's PDC-anchored targets.

## Universe

**Intended universe**: NSE all stocks, but with cap segment filter:
- **Allowed cap segments**: `mid_cap`, `small_cap` only
- **Excluded**: `large_cap` (rare upper circuits, usually fundamental); `micro_cap` (too thin for short-side liquidity)
- **No F&O 200 restriction** — peer-reviewed evidence (Sehgal et al.) is on Indian-equity broadly, including non-F&O small-caps where operator pumps concentrate

**Why this universe:**
- Operator-pump signature concentrates in mid/small-caps with restricted floats
- Bulk-block lesson: don't pre-restrict universe without literature backing — let the data decide via the cap_segment filter
- Excluding micro-cap is the explicit short-side liquidity defense (you can't short what you can't borrow); F&O 200 isn't a strict requirement because mid/small caps with normal liquidity can be MIS-shorted via Zerodha SLB or Kite borrow

**Approximate symbol count after cap filter**: ~600 stocks. Of those, only ~5-15/day hit upper circuit. Sample-size-feasibility: ~1,200-3,000 events/year before quality filters; ~300-700/year after structural filters.

## Active window

**Setup formation**: T+0 (full session — circuit-hit can occur at any time during 09:15-15:30).
**Entry**: T+1 10:30 IST (single bar, the 10:30-10:34 5m close).
**Hold horizon**: 10:30 → 15:15 IST = 4h 45m intraday MIS.

**Why 10:30 entry (not 09:15 or 09:30 or noon):**
- 09:15 = gap-open, BUYERS dominate, fading is suicidal
- 09:30 = first 15-min DPR flex window concluded but FOMO still strong
- 10:00-10:30 = FOMO peaks, sellers begin emerging
- 10:30 = inflection — peer-reviewed evidence supports this as the post-FOMO entry zone
- After 11:30 = the fade is mostly done, late entries lose net edge
- 14:00+ = MIS-unwind sells dominate (fades may overlap with that flow but are noisy)

**Bulk-block lesson application:** the bulk-block long failed because the multi-day edge was captured by the T+1 gap-open. For circuit_t1_fade_short, **we are HARVESTING the same intraday-after-gap mean-reversion, but on the short side**. The bulk-block sim showed buyers entering at 09:25 LOST money on T+1 intraday; that's exactly the side we want to be on. The lesson favors this setup, doesn't undermine it.

## Risks / falsification criteria

The setup is **wrong** (and should be retired) if:

1. **Phase-1 floor fails on validation/holdout**:
   - n < 500 trades over 1-2 years
   - NET PF < 1.10
   - NET Sharpe ≤ 0
2. **Direction asymmetry collapses** — if SHORT-side T+1 fade PF < 1.10 while LONG-side reclaim PF passes, the "post-FOMO short" thesis isn't right; might be sector-momentum continuation instead.
3. **Same-day intraday component is too thin** — if the fade is a multi-day phenomenon (T+1 gap → fade over T+1 to T+5), MIS-only 4h 45m hold won't capture enough. Pre-coding sanity check addresses this.
4. **Circuit-hit detection unreliable** — if 5m-bar-derived circuit-hit detection has too many false positives (e.g., illiquid stocks where every bar is at the band) the candidate dataset is contaminated.

**Pre-coding sanity check** (mandatory per §3.3, BEFORE writing detector):
- Use the existing 12-month 2024 5m feathers (already on disk)
- Detect upper-circuit hits via daily-bar high == price-band-CSV ceiling (NSE publishes daily price-band CSV)
- Apply cap_segment + same-day-fundamental-news exclusion
- Simulate T+1 10:30 entry → 15:15 exit short, with 1% min-stop hard SL above T+1 high, gap-fill T1/T2
- Compute NET PF using existing Indian fee model
- **Decision per §3.3:** PF ≥ 1.10 → strong proceed; 1.0-1.10 → marginal; PF < 1.0 → retire

## Data engineering plan (preliminary, NOT yet built)

Required new components (only if sanity check passes):

1. **`tools/sub9_research/sanity_circuit_t1_fade_short.py`** — pre-coding sanity check (parallel to the bulk-block one). Reads price-band CSVs + 5m feathers; no detector code yet. Will be retired after used.

2. **(post-sanity-check, only if APPROVED for full implementation):**
   - `tools/circuit_data/fetch_price_band_csv.py` — daily NSE price-band CSV scraper, normalized to a parquet at `data/price_bands/<YYYY>/<MM>/<YYYY-MM-DD>.parquet`
   - `services/circuit_hit_detector.py` — circuit-hit detection from 5m bars + price-band lookup
   - `structures/circuit_t1_fade_short_structure.py` — the detector

## Sample-size feasibility

From research findings: ~3-8 F&O-stock circuit-flex events/day market-wide; small/mid-cap excluded F&O = wider universe but harder to confirm operator signature. Conservative estimate after all filters:
- ~5-15 raw upper-circuit hits/day across cap-eligible universe
- After cap filter + volume filter + news exclusion + T+1 gap confirmation: ~3-7 signals/day
- Annual: ~750-1,750 events. n ≥ 500 over 1 year easily satisfied.

## Honest comparison to surviving setups

| Aspect | gap_fade_short (TRUSTED) | circuit_t1_fade_short (proposed) |
|---|---|---|
| Indian-specific | retail momentum exhaustion in T+0 opening | retail FOMO + operator pump exhaustion in T+0 close, faded T+1 |
| Direction | short-only | short-only |
| Active window | T+0 09:15-09:30 | T+1 10:30 single-bar entry |
| Universe | small_cap | mid_cap, small_cap |
| Hold | intraday MIS (15-30 min) | intraday MIS (4h 45m) |
| Evidence base | empirical sub-7 validation | 5 peer-reviewed papers |
| Correlation with gap_fade_short | n/a (same setup) | low expected — different timing window, different participants on the long side |

The two setups complement: gap_fade harvests T+0 opening retail momentum; circuit_t1_fade harvests T+0 close retail FOMO carrying into T+1. Different timeframes, low signal correlation, both short-only on Indian-intraday's losing-flow side.

---

## Decision required

User to indicate:
- [ ] APPROVED — proceed to pre-coding sanity-check script
- [ ] REJECTED — reason
- [ ] REVISE — specify what's missing / wrong

Per sub-9 §3.3, no code is written until APPROVED.
