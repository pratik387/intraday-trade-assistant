# §3.3 Brief: `vwap_deviation_meanrevert`

**Sub-project:** #9 (microstructure-first redesign)
**Status:** **DRAFT — awaiting user APPROVE/REJECT/RETIRE before sanity-check.**
**Date:** 2026-05-06

**Predecessors:**
- specs/2026-05-01-sub-project-9-microstructure-first-redesign.md (defines §3.3 gate process)
- specs/2026-05-01-sub-project-9-asymmetry-research-findings.md (Round-1)
- specs/2026-05-05-sub-project-9-asymmetry-feasibility-round-2.md (Round-2)
- specs/2026-05-06-sub-project-9-asymmetry-feasibility-round-3.md (Round-3, this candidate is §1)
- specs/2026-05-01-sub-project-9-brief-circuit_t1_fade_short.md (APPROVED template)

This is one of three round-3 §3.3 briefs (the others are `index_stock_divergence_revert` and `volume_spike_exhaustion_reversal`). All three target the 11:00-15:15 IST middle-of-day window left empty by the existing morning-only setups (`gap_fade_short` and `circuit_t1_fade_short`).

---

## Asymmetry

**Name:** Indian-equity intraday VWAP-deviation mean-reversion (5m).

**Indian-specific source:**
- **NSE intraday is heavily VWAP-anchored** because (a) institutional execution algos at large desks are VWAP-targeted by mandate (the desk's evaluation benchmark is "did the trade execute close to session VWAP?"); (b) MIS auto-square-off mechanics cause large-account intraday flow to cluster around VWAP late-day (operators target VWAP-or-better exits before 15:20); (c) Zerodha Kite, Streak, and most major Indian retail-algo platforms publish VWAP as the default intraday anchor — making it both a real institutional target AND a retail-algo magnet (self-reinforcing).
- **Retail-FOMO chase pattern** in India systematically over-extends prices above VWAP on intraday momentum bursts (small/mid-cap especially), creating mean-reversion edge on the SHORT side of stretched-above-VWAP bars. SEBI FY23 study: 70% cash-intraday + 93% F&O retail traders lose; the losing flow is overwhelmingly LONG. Stretched-above-VWAP bars are the kinetic signature of this losing-side momentum.
- **India VIX < 14 modal regime** (the dominant 2024-26 regime) makes VWAP behave as a strong magnet (Goodwill Securities Indian-broker guide — VWAP-revert specifically conditioned on VIX < 14).

The exploitable asymmetry is the **5m bar that breaches a fixed deviation band above VWAP, prints a reversal candle, and reverts toward VWAP within the next 4-8 bars** — particularly on the SHORT side in mid/small-cap F&O names where retail-FOMO concentrates.

## Participants

- **Stretched-above-VWAP bars (SHORT side, primary edge):** retail FOMO + small operator-led pumps in mid/small-cap F&O names; momentum-chasing retail algos firing on simple breakout signals. The fade is filled by (a) institutional VWAP-target sellers who treat the stretch as a free-execution gift, (b) mechanical mean-reversion algos (the "smart-money" side of Indian retail asymmetry), (c) profit-taking from earlier-positioned longs.
- **Stretched-below-VWAP bars (LONG side, secondary edge):** intraday capitulation + stop-cascades; the long side here is "buying institutional VWAP-targeted accumulation." In Indian intraday, this side is LESS reliable than the short side because the losing-flow asymmetry (SEBI FY23) means downside capitulations frequently keep going — the "buy-the-dip" trade has historically failed more often than not in this universe. **Long side carried for diagnostic comparison only; not the primary edge.**

We are on the disciplined side of an attention-driven mispricing.

## Persistence

Three structural reasons:

1. **Institutional VWAP-execution mandate is regulatory/professional standard** — large desks (FII/DII, mutual fund execution) are evaluated against session-VWAP benchmark (industry-standard, codified in execution-quality reports). This creates a *guaranteed* counter-flow whenever price stretches from VWAP. Not market-cycle-dependent.
2. **Retail-platform VWAP saturation** — Zerodha Kite, Upstox Pro, Streak, Dhan, all publish VWAP as default intraday overlay. Making VWAP a Schelling point for intraday decisions across millions of retail accounts. The visibility itself is the persistence mechanism.
3. **SEBI FY23 losing-flow asymmetry is structural** — 70% cash-intraday + 93% F&O retail traders lose, and the losing flow is LONG-biased. Short-side fade of stretched-above-VWAP is structurally aligned with the asymmetry; the persistence will outlive any single market regime as long as Indian retail keeps losing money on the long side.

Greenwood/Sammon decay caveat: documented attention-driven microstructure asymmetries decay 30-50% per decade as participants adapt. This setup is expected to retain edge over a 3-5 year horizon but should be re-evaluated annually for half-life signals (PF compression, signal-frequency drop).

## Project-wide caveats addressed

- **Long-bias caveat:** the brief proposes a **bidirectional** detector but prefers SHORT side. Per sub7/sub8 11-failure long-bias pattern, any LONG side allocation must clear an additional gate during sanity-check: long-side PF must be ≥ short-side PF × 0.85 to ship long. If long-side PF is materially weaker, ship SHORT-only first and revisit long after one quarter of live data.
- **Decay risk (Greenwood/Sammon):** Acknowledged. VWAP-revert is a documented retail-algo strategy with 5+ Indian platforms publishing templates — adaptation is already partial. The half-life of this edge is plausibly shorter than peer-reviewed once-discovered asymmetries. Annual re-validation built into the falsification criteria.
- **VIX regime caveat:** Goodwill Securities source explicitly conditions VWAP-revert effectiveness on India VIX < 14. India VIX intraday history is NOT on disk per round-3 §1. The mechanic uses the *daily* India VIX close as a regime gate (proxy via NIFTY ATM IV from the EOD option chain if needed). High-VIX days (≥ 18) are excluded from entry — VWAP loses its magnet property in panic regimes and the setup is not meant to fight panic.

## Evidence

Round-3 spec §1 lists 5 Indian retail-algo platform sources (Zerodha Streak, Zerodha Varsity, Goodwill Securities, Share India, Stoxra). All 5 are operational publications on Indian intraday VWAP-revert at retail-MIS scale. Per round-3 mandate (Gate A passed), Indian retail-algo precedent is primary; in-house event study during sanity-check is the formal evidence substitute.

Peer-reviewed (secondary):
1. **Springer "Mean-Reverting Tendency in Stock Returns" chapter** (Indian-data study) — documents general mean-reversion in Indian equity markets. https://link.springer.com/chapter/10.1007/978-81-322-1590-5_4
2. **Zarattini & Aziz, SSRN 4631351** — VWAP general framework, intraday-execution academic context. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4631351

Indian retail-algo sources (Gate A primary):
1. Zerodha Streak template — https://www.youtube.com/watch?v=9O6ZHoVSLcQ
2. Zerodha Varsity supplementary indicators chapter — https://zerodha.com/varsity/chapter/supplementary-notes-1/
3. Goodwill Securities (VIX-regime-aware) — https://www.gwcindia.in/blog/vwap-volume-weighted-average-price-a-pro-tool-for-intraday-traders/
4. Share India — https://www.shareindia.com/knowledge-center/intraday-trading/how-to-use-vwap-indicator-for-intraday-trading
5. Stoxra Blog top-5 algo strategies — https://stoxra.com/blog/top-5-algorithmic-trading-strategies-professional-traders

## Direction

**Bidirectional**, with **SHORT side as the primary edge** and additional gate on long-side ship decision (see project-wide caveat above).

- **SHORT** when 5m bar's `(close − vwap) / vwap_intraday_stdev` ≥ +2.0 AND next bar prints a bearish reversal candle.
- **LONG** when 5m bar's `(close − vwap) / vwap_intraday_stdev` ≤ −2.0 AND next bar prints a bullish reversal candle.

The bidirectional design lets the sanity-check produce *evidence* that long-bias is structurally weaker (the 11-failure pattern). If long-side PF passes the differential gate, ship both. If not, ship SHORT-only.

## Mechanic

**Setup name:** `vwap_deviation_meanrevert`
**Side:** Bidirectional (SHORT primary edge, LONG conditional ship).

**Sequence:**

1. **Continuous detection (every 5m bar close, 11:00-15:00 IST):**
   - For each F&O-200 mid/small-cap symbol, compute on the closing 5m bar:
     - `vwap` (already in 5m enriched feather)
     - `vwap_intraday_stdev` = rolling stdev of `(close − vwap)` over the 9 most recent intraday bars (≈ 45 min, captures intraday volatility regime without leaking session-end noise)
     - `z = (close − vwap) / vwap_intraday_stdev`
   - **Trigger:** `|z| ≥ 2.0` (research-locked threshold — see Pre-coding sanity check; post-sanity may reveal need to revise but DO NOT iterate on validation data).
     - The +2.0 threshold rationale: at modal Indian-intraday VIX < 14 regime, a 2-sigma stretch from VWAP corresponds to roughly +0.7-1.2% deviation — empirically the level above which the FOMO-driven extension becomes statistically distinct from random walk drift (per Goodwill blog and Stoxra retail-algo template ranges of 1.5-2.5 sigma).

2. **Reversal-candle confirmation (next 5m bar):**
   - For SHORT entry: next bar's close < open AND close < prior bar's close AND (high − close) / (high − low) ≥ 0.5 (rejection wick).
   - For LONG entry: next bar's close > open AND close > prior bar's close AND (close − low) / (high − low) ≥ 0.5.
   - Single-bar reversal — no multi-bar formation requirement (keeps signal frequency in the 5-15/day target zone).

3. **Entry:**
   - **Entry price:** confirmation bar's CLOSE (5m bar close after the reversal candle).
   - **Direction:** as triggered.
   - **Active window gate:** 11:00-15:00 IST inclusive on the trigger bar. Excludes 09:15-10:55 (overlaps `gap_fade_short` and `circuit_t1_fade_short`; opening-flow VWAP stretches are institutional, not retail FOMO) and 15:00+ (MIS auto-square contamination).

4. **Stop-loss:**
   - **Hard SL (SHORT):** trigger bar's high + 0.3% buffer.
   - **Hard SL (LONG):** trigger bar's low − 0.3% buffer.
   - **Min stop distance:** 0.6% of entry (qty-inflation guard for thin small-caps).
   - Buffer of 0.3% empirically clears intraday spread + slippage in F&O-200 mid/small-cap names per the existing sub7 stop-distance work.

5. **Targets:**
   - **T1** (50% qty): VWAP touch (price = `vwap` on any subsequent bar). This is the literal mean-reversion thesis cashing in.
   - **T2** (50% qty): VWAP minus 0.5 × `vwap_intraday_stdev` for SHORT (or VWAP plus 0.5 × stdev for LONG) — overshoot of mean to capture the typical pendulum swing.
   - **Time stop:** 8 bars from entry (≈ 40 min) OR 15:10 IST hard stop, whichever first.

6. **Latch:** one fire per (symbol, day, direction) — no re-entry same direction same session. Opposite direction allowed if independent z-score breach later in the session.

**target_anchor_type:** `structural` — T1/T2 are anchored to the live VWAP and stdev band (not arithmetic R-multiples). VWAP is a structural intraday level (institutional benchmark + retail-platform Schelling point), so structural anchoring is the correct semantic and matches the asymmetry source.

## Universe

**Universe:** F&O 200 with cap-segment filter:
- **Allowed cap segments:** `mid_cap`, `small_cap`
- **Excluded:** `large_cap` (institutional VWAP-target flow dominates, less retail-FOMO stretch — edge is muted) and `micro_cap` (not in F&O 200 anyway; thin liquidity + short-borrow risk).
- **Universe filter file:** `assets/fno_liquid_200.csv` (verified on disk per round-3 §2 Gate B).
- **Liquidity gate:** 20-day average `volume × close` ≥ ₹3 Cr on the daily bar (defends against thin-tape failures-to-fill on MIS short side).

**Why this universe:**
- F&O 200 is the borrowable + liquid Indian intraday-MIS short universe (Zerodha SLB / Kite borrow viable).
- Mid/small-cap is where retail FOMO concentrates per SEBI FY23 + the operator-pump literature.
- Excluding micro-cap is the explicit short-side liquidity defense.

**Approximate symbol count after cap filter:** ~120-140 stocks (F&O 200 minus ~60-80 large-caps). Sample-size feasibility per round-3 §1: 5-15/day at ±2.0 sigma; conservative annual count 1,200-3,500 events. n ≥ 500 over 1 year easily satisfied.

## Active window

**Setup formation + entry:** 11:00-15:00 IST. Entry on the post-trigger reversal-confirmation 5m bar (so latest possible entry is the 14:55-15:00 bar close = entry at 15:00).
**Hold horizon:** 8 bars (≈ 40 min) OR 15:10 IST hard stop, whichever first.
**Latest possible exit:** 15:10 IST (10 min before MIS auto-square at 15:20).

**Why 11:00-15:00 entry window (not 09:15 or 15:10+):**
- 09:15-10:55 = opening-flow + circuit-fade window. VWAP is unstable in the first hour (intraday VWAP definition itself accumulates few bars). Stretches are dominated by institutional opening flow, not retail FOMO. Overlap with `gap_fade_short` (09:15-09:30) and `circuit_t1_fade_short` (10:30) is a separate concern; this setup must not double-allocate on the same flow.
- 11:00 onwards = VWAP stabilises (≥ 21 bars accumulated), retail-FOMO flow becomes the dominant source of stretch.
- 14:00-15:00 = retail-FOMO chase in trending stocks intensifies (afternoon-trend phenomenon), VWAP magnet still strong.
- 15:00+ = MIS auto-square unwinding contaminates the tape; exits cluster, bid/ask widens, the 40-min hold horizon doesn't fit.

## Risks / falsification criteria

The setup is **wrong** (and should be retired) if:

1. **Phase-1 floor fails on validation/holdout:**
   - n < 500 trades over 12 months
   - NET PF < 1.10
   - NET Sharpe (daily) ≤ 0
2. **Direction asymmetry collapses:** if SHORT-side PF < 1.10 while LONG-side PF passes, the "post-FOMO short" thesis isn't right; might be sector-momentum continuation, retire SHORT and re-evaluate LONG separately.
3. **Long-side ships but loses live:** if LONG-side passes sanity but underperforms in paper/live by ≥ 30% PF (the sub7/sub8 11-failure pattern), retire LONG immediately, keep SHORT-only.
4. **VIX regime dependency too strong:** if NET PF in VIX ≥ 18 days is < 0.85, gate the detector to VIX < 18 entry only and re-validate.
5. **Decay signal:** if rolling-60-trade NET PF drops below 1.05 sustained for 60 calendar days post-launch, retire (Greenwood/Sammon adaptation kicked in).
6. **VWAP value mismatch with feather:** if the 5m enriched feather's `vwap` column is computed differently than session-cumulative `Σ(price × volume) / Σ(volume)` (e.g., rolling VWAP), the sanity script must abort and re-derive on raw 5m + 1m source.

**Pre-coding sanity check** (mandatory per §3.3, BEFORE writing detector):
- Use existing 12-month 2024 5m enriched feathers on disk
- Pre-aggregate to compute `vwap_intraday_stdev` per (symbol, day) on the 9-bar rolling window
- Identify trigger bars where `|z| ≥ 2.0` AND time ∈ [11:00, 15:00]
- Apply reversal-candle confirmation on next 5m bar
- Simulate entry → 8-bar / 15:10 exit with structural T1/T2 + 0.6% min-stop hard SL
- Compute NET PF using existing Indian fee model (Zerodha MIS + STT + GST + stamp)
- Report PF / WR / Sharpe per direction (SHORT vs LONG separately) and per cap segment (mid_cap vs small_cap)
- **Decision per §3.3:** PF ≥ 1.10 (SHORT side) → strong proceed; 1.0-1.10 → marginal; PF < 1.0 → retire. LONG-side PF ≥ SHORT-side PF × 0.85 → ship bidirectional; else SHORT-only.

## Data engineering plan

Required new components (only if sanity check passes):

1. **`tools/sub9_research/sanity_vwap_deviation_meanrevert.py`** — pre-coding sanity check (parallel to the circuit_t1 one). Reads 5m enriched feathers + computes z-score / reversal logic; no detector code yet. Will be retired after used.

2. **(post-sanity-check, only if APPROVED for full implementation):**
   - `services/vwap_band_calculator.py` — computes `vwap_intraday_stdev` on the rolling 9-bar window incrementally (live-replay-compatible per CLAUDE.md rule 3).
   - `structures/vwap_deviation_meanrevert_structure.py` — the detector.
   - Config keys (added to `config/configuration.json`, NO hardcoded defaults per CLAUDE.md rule 1):
     - `vwap_deviation_meanrevert.z_threshold` = 2.0
     - `vwap_deviation_meanrevert.stdev_window_bars` = 9
     - `vwap_deviation_meanrevert.reversal_wick_min_pct` = 0.5
     - `vwap_deviation_meanrevert.entry_window_start` = "11:00"
     - `vwap_deviation_meanrevert.entry_window_end` = "15:00"
     - `vwap_deviation_meanrevert.time_stop_bars` = 8
     - `vwap_deviation_meanrevert.time_stop_hard_ist` = "15:10"
     - `vwap_deviation_meanrevert.sl_buffer_pct` = 0.003
     - `vwap_deviation_meanrevert.min_stop_pct` = 0.006
     - `vwap_deviation_meanrevert.t1_anchor` = "vwap"
     - `vwap_deviation_meanrevert.t2_anchor_overshoot_stdev` = 0.5
     - `vwap_deviation_meanrevert.cap_segments_allowed` = ["mid_cap", "small_cap"]
     - `vwap_deviation_meanrevert.min_adv_inr_cr` = 3.0
     - `vwap_deviation_meanrevert.vix_max_entry` = 18.0
     - `vwap_deviation_meanrevert.long_side_enabled` = false   # toggled true only if sanity passes long-side gate

   No new ingestion needed. VWAP column already in `cache/ohlcv_archive/{symbol}.NS/{symbol}.NS_5minutes_enriched.feather` (round-3 §1 Gate B verified). India VIX daily close: needs ingestion from NSE indices CSV — minor (1 daily file/yr, ~250 rows/yr) — defer to post-approval sub-task.

## Sample-size feasibility

Per round-3 §1: 5-15 events/day across F&O 200 at ±2.0 sigma; ~2-5/day at ±2.5 sigma. Conservative annual count: 1,200-3,500 events. After reversal-candle confirmation (~50% pass rate empirically) and active-window gate (11:00-15:00 = 60% of session): expect ~700-2,100 entries/year. n ≥ 500 over 1 year met.

Honest acknowledgement: signal frequency is sensitive to the +/- 2.0 sigma threshold. ±2.5 sigma drops sample by 60-70% — sanity-check report must include sensitivity to threshold within ±0.5 sigma but the LOCKED parameter is 2.0 and won't be re-tuned post-result on validation data (lessons.md 2026-05-01).

## Honest comparison to surviving setups

| Aspect | gap_fade_short (TRUSTED) | circuit_t1_fade_short (APPROVED) | vwap_deviation_meanrevert (proposed) |
|---|---|---|---|
| Indian-specific source | retail momentum exhaustion in T+0 opening | T+0 retail FOMO + operator pump | institutional VWAP-target + retail-FOMO stretch |
| Direction | short-only | short-only | bidirectional, SHORT primary |
| Active window | T+0 09:15-09:30 | T+1 10:30 single bar | T+0 11:00-15:00 continuous |
| Universe | small_cap | mid_cap, small_cap (no F&O restriction) | F&O-200 mid_cap, small_cap |
| Hold | 15-30 min MIS | 4h 45m MIS | 40 min MIS |
| Evidence base | empirical sub-7 validation | 5 peer-reviewed papers | 5 Indian retail-algo platforms + 2 academic |
| Anchor | structural (PDC) | structural (gap edges) | structural (VWAP + stdev) |
| Correlation w/ existing | n/a | low (different time, different participants) | low (different time + cross-bar trigger) |

The setup complements: harvests middle-of-day retail-FOMO stretches that the morning setups don't see. Different timeframe, different trigger (z-score vs gap), low signal correlation expected. Bidirectional design is the differentiator within sub-9 (others are short-only) but the long side is gated behind a passes-comparable-to-short sanity check.

---

## Decision required

User to indicate:
- [ ] APPROVED — proceed to pre-coding sanity-check script
- [ ] REJECTED — reason
- [ ] RETIRE — kill candidate

Per sub-9 §3.3, no detector code is written until APPROVED and sanity-check passes (NET PF ≥ 1.10).
