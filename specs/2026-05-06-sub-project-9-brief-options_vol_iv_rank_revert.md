# §3.3 Brief: `options_vol_iv_rank_revert`

**Sub-project:** #9 (microstructure-first redesign)
**Status:** **DRAFT — awaiting user APPROVE/REJECT/REVISE before sanity-check.**
**Date:** 2026-05-06

**Predecessors:**
- specs/2026-05-01-sub-project-9-microstructure-first-redesign.md (defines §3.3 gate process)
- specs/2026-05-01-sub-project-9-asymmetry-research-findings.md (Round-1)
- specs/2026-05-05-sub-project-9-asymmetry-feasibility-round-2.md (Round-2)
- specs/2026-05-06-sub-project-9-asymmetry-feasibility-round-3.md (Round-3, Round-4 inherits methodology)
- specs/2026-05-01-sub-project-9-brief-circuit_t1_fade_short.md (APPROVED template — daily-stale-signal pattern)

This is one of three round-4 §3.3 briefs (the others target sector-rotation and earnings-day asymmetries). All three reuse already-running daily-data-ingestion infrastructure to generate T+0 intraday signals — the same daily-stale-signal pattern that worked for `circuit_t1_fade_short`.

---

## Asymmetry

**Name:** Indian single-stock-options IV-rank extreme → T+0 intraday underlying mean-reversion.

**Indian-specific source:**
- **SEBI FY23 retail-F&O loss study:** 91-93% of individual F&O traders lose money; aggregate retail F&O loss ~₹52,000 crore in FY24 (SEBI consultation paper Sep 2024 update). The retail flow is overwhelmingly LONG-options (calls + puts), and the systematic lossage is the option-buyer side. https://www.sebi.gov.in/reports-and-statistics/research/sep-2024/study-on-the-losses-of-individual-traders-in-fno-segment_86852.html
- **IV-rank extreme is the kinetic signature of this lossage in real-time:**
  - **High IV-rank (≥80)** — ATM IV is in the top quintile of its 252-day distribution. In single-stock options, this state typically arises from concentrated retail call-buying (FOMO-greed peak) or concentrated retail put-buying (panic peak). The underlying has been pushed by positioning more than fundamentals; option-writers (institutional desks, market-makers) are over-short vol relative to fair value. The mean-reversion is two-fold: (a) IV crushes back toward its 252-day median over the next 1-3 sessions, (b) underlying drifts opposite to the retail directional bias as the position-driven push exhausts. The exploitable T+0 intraday component is the **underlying revert**, not the IV crush itself.
  - **Low IV-rank (≤20)** — ATM IV at the bottom quintile. Retail short-vol exhausted (e.g., post-expiry calm, post-event quiescence). The market is structurally underpricing realized variance for that name; expansion follows. Direction of expansion in single-stock space is empirically biased toward the **prevailing NIFTY trend** (low-IV stocks tend to catch up to index drift when their idiosyncratic vol re-engages). Hence: long underlying when IV-rank ≤ 20 AND NIFTY uptrend gate.
- **Single-stock options vs index options:** the asymmetry is sharper in single-stock F&O because (a) retail concentration is higher per name, (b) market-maker offsetting flow is thinner than in NIFTY/BANKNIFTY, (c) IV-rank percentiles are computed against the stock's own 252-day history (relative measure, not absolute volatility level). NIFTY/BANKNIFTY IV-rank does not produce the same edge — those are dominated by institutional vol traders, not retail.
- **Why "EOD-IV-rank applied to T+0 intraday" (daily-stale signal pattern):**
  - IV-rank is computed from yesterday's NSE F&O bhavcopy (already ingested via `tools/option_chain/fetch_oi_snapshot.py`). It is a daily-frequency signal.
  - Underlying intraday revert plays out in the 11:00-15:00 window — after the morning open auction has digested overnight positioning and before late-day MIS auto-square noise.
  - This is the **same data-shape pattern** as `circuit_t1_fade_short`: daily-frequency feature (T-1 EOD) → T+0 intraday entry. Already proven viable for `circuit_t1_fade_short` (PF 1.47 sanity).

The exploitable asymmetry is the **T+0 11:00-15:00 underlying revert** when IV-rank is at an extreme percentile of its 252-day single-stock-options history.

## Participants

- **High IV-rank ≥ 80 → SHORT underlying (primary edge):**
  - **Long side faded:** retail option-buyers (calls in greed regimes, puts in panic regimes) who have driven IV up via concentrated demand. Their hedging counterparties (market-makers) are short gamma / short vol and are pinned at delta-hedge driven price levels.
  - **Short underlying flow we join:** institutional vol-arb desks (selling rich vol, hedging via short-stock); profit-taking from operators who pumped the name into the IV peak; mean-reversion algos.
  - **Why this is the disciplined side:** retail option flow is the documented losing side per SEBI FY23 (91-93% lose). When IV-rank pins ≥80, the retail position is at its most concentrated AND at its most overpriced — the structural worst entry point for the retail buyer.

- **Low IV-rank ≤ 20 + NIFTY uptrend → LONG underlying (secondary edge, gated):**
  - **Position faded:** retail short-vol writers who have collected premium and are now structurally short gamma at low IV (the post-event quiescence trade). Their offsetting flow when realized vol re-engages is to delta-cover by buying underlying.
  - **Long underlying flow we join:** vol-expansion algos; index-tracking flow if the name is index-eligible; the NIFTY drift itself.
  - **Why the NIFTY uptrend gate:** without it, low-IV-rank long is a long-bias setup similar to those that failed 11x in sub7/sub8. The NIFTY uptrend filter ensures we are only long when the broader regime is trending up — this is a regime gate, not a directional override.

## Persistence

Three structural reasons:

1. **SEBI FY23 retail-F&O lossage is structural.** Retail option-buying is the documented losing flow; this is regulatory-level evidence (SEBI consultation paper, then-current research wing publication). Persists as long as retail demographics + smartphone-broker access continue to expand — i.e., decades.
2. **NSE F&O bhavcopy is a regulated public daily artifact.** EOD ATM IV per (symbol, expiry) is reproducible from the bhavcopy (NSE publishes settlement IV for every contract). The signal is not platform-dependent; it does not decay if a single broker changes their UI.
3. **Single-stock IV percentile-rank dispersion is structural.** Indian single-stock options have lower volume than NIFTY/BANKNIFTY and higher percentile-rank dispersion across the 252-day window — i.e., extremes occur with sufficient frequency to be tradeable. This is mechanically different from US single-stock IV which is dominated by index-flow spillover; Indian single-stock IV is more stock-specific.

Greenwood/Sammon decay caveat: IV-rank reversion is a known pattern in US literature (Hill et al., RMS publications). Indian single-stock-options IV-rank-revert is NOT widely operationalised at retail-algo scale (Stratzy publishes it as a concept; Choice India mentions it in retail education; no retail-algo platform ships it as a turnkey strategy). This relative-obscurity-in-India delays decay relative to volume-spike or VWAP setups.

## Project-wide caveats addressed

- **Long-bias caveat (sub7/sub8 11-failure pattern):** The brief proposes a **bidirectional** detector but with **explicit asymmetric gating**. SHORT side (high IV-rank) is the primary edge — directly aligned with the SEBI FY23 retail-loss asymmetry. LONG side (low IV-rank) is gated by an additional NIFTY uptrend filter that the SHORT side does not require. Per the established pattern, LONG-side ship requires sanity-check long-side PF ≥ short-side PF × 0.85; otherwise SHORT-only ships first and LONG re-evaluates.
- **Daily-stale signal coupling risk:** Using yesterday's IV-rank for today's entry means signal staleness is up to 1 trading day. Mitigation: re-validate the IV-rank against intraday-IV-proxy (ATM-strike option price changes during the 09:15-11:00 window) at entry time. If intraday IV has already collapsed by ≥30% relative to yesterday's IV by 11:00 (i.e., the revert has already happened), ABORT. This guards against the "signal already exhausted by entry" failure mode.
- **Sample-size tightness:** F&O 200 universe × ~250 trading days/year × ~10% IV-extreme rate ≈ **5,000 candidate-days/year before structural filters**. After NIFTY-trend gate (LONG side) and intraday-confirmation gates: estimated **800-2,000 entries/year**. n ≥ 500 is comfortably feasible over 1 year; over 2 years (sanity has 2024 + 2025) the count is robust.
- **Liquidity caveat (single-stock options thin names):** Some F&O 200 names have illiquid options where ATM IV is unreliable (wide bid-ask, low volume). Mitigation: filter to contracts where bhavcopy `vol ≥ 100` AND `oi ≥ 1000` per (symbol, expiry, day) — this excludes the long tail of dead F&O names.

## Evidence

**Indian retail-algo precedent (Gate A — required ≥2 distinct sources):**

1. **Stratzy** — IV-rank / IV-percentile is enumerated in their common-algo-strategies post as a mainstream Indian-retail-algo concept; explicit publication of the high-IV-fade and low-IV-buy mechanic. https://stratzy.in/blog/common-algo-trading-strategies-and-examples/
2. **Choice India broker education** — IV-rank/IV-percentile threshold conventions (≥80 high, ≤20 low) are published as standard for Indian retail F&O traders. https://choiceindia.com/blog/iv-rank-iv-percentile-options-trading/
3. **Sensibull** — Indian options-platform that publishes IV-rank per single-stock contract and IV-rank-based strategy templates (long-vol when low, short-vol when high). https://web.sensibull.com/
4. **Quantsapp** — Indian options-analytics platform publishing IV percentile and IV rank by stock. https://quantsapp.com/
5. **Internal precedent** — `tools/option_chain/fetch_oi_snapshot.py` (already running, ingests EOD bhavcopy to `data/option_chain/<YYYY>/<MM>/*.parquet`) is the exact ingestion infra needed. Built originally for `expiry_pin_strike_reversal` (Phase A2-A3, specs/2026-04-29). Independent confirmation that the team has already built the data layer for option-chain features.

**Peer-reviewed / academic (secondary):**

1. **Hill, Balasubramanian, Gregory-Allen — "Volatility Risk Premium in the Indian Equity Market"** (Indian-specific). Documents structural overpricing of implied vol vs realized vol in NIFTY options; logical extension to single-stock options where retail concentration is higher. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2495568
2. **Bollen & Whaley, *Journal of Finance* 2004 — "Does Net Buying Pressure Affect the Shape of Implied Volatility Functions?"** — foundational paper showing retail-driven option demand distorts IV away from fundamental fair value, with subsequent reversion. The exact mechanism we are exploiting, established globally with the Indian-equity instance documented separately.
3. **Indian-equity mean-reversion (Springer chapter, already cited in round-3 spec)** — general framework. https://link.springer.com/chapter/10.1007/978-81-322-1590-5_4

≥2 Indian retail-algo precedent sources met (Stratzy, Choice India, Sensibull, Quantsapp = 4). ≥2 peer-reviewed sources met (Hill et al., Bollen-Whaley). Internal precedent (already-built ingestion) is the data-feasibility validator.

## Direction

**Bidirectional with asymmetric gates.**

- **SHORT underlying** when ATM IV-rank ≥ 80 (high-IV crush thesis → underlying mean-reverts down).
  - **Primary edge.** Aligned with SEBI FY23 retail-loss asymmetry. No additional regime gate required beyond the IV-rank threshold and intraday-confirmation gates.
- **LONG underlying** when ATM IV-rank ≤ 20 AND NIFTY 50 5m close > NIFTY 50 5m EMA(20) at 11:00 IST (low-IV expansion → underlying continues with index drift).
  - **Secondary, gated edge.** The NIFTY-uptrend gate is the safety against the sub7/sub8 long-bias failure pattern. Drops the LONG side in choppy or down-trending NIFTY regimes, where low-IV-rank long is statistically the same kind of trade that failed 11 times.

If sanity-check long-side PF < short-side PF × 0.85, ship SHORT-only and re-evaluate LONG separately.

## Mechanic

**Setup name:** `options_vol_iv_rank_revert`
**Side:** Bidirectional with asymmetric gating (SHORT primary, LONG gated by NIFTY trend).

**Sequence:**

1. **T-1 EOD ingestion** (post-18:00 IST after bhavcopy publish):
   - Use already-running `tools/option_chain/fetch_oi_snapshot.py` to fetch yesterday's bhavcopy. Output: `data/option_chain/<YYYY>/<MM>/<YYYY-MM-DD>.parquet`.
   - For each F&O 200 stock, identify the **ATM strike of the nearest-expiry contract** (closest call/put to underlying close; ignore weekly index options).
   - Compute **ATM IV** per contract from bhavcopy (settlement IV column; if absent, derive from close-price + Black-Scholes inversion using risk-free rate from `assets/`).
   - Reject contracts with `vol < 100` OR `oi < 1000` (illiquidity exclusion).

2. **IV-rank computation** (T-1 EOD, per symbol):
   - Pull last **252 trading days** of ATM IV for the symbol from the parquet store.
   - `IV_rank = (today's_ATM_IV − min(252d)) / (max(252d) − min(252d)) × 100`
   - This is the **standard published convention** (Stratzy, Choice India, Sensibull all use 252-day window with min-max scaling — NOT percentile-rank, NOT 1-year-on-rolling-basis).
   - Persist `(symbol, date, atm_iv, iv_rank)` to a daily cache.

3. **T+0 09:00 IST signal preparation:**
   - Symbol qualifies for SHORT side if `iv_rank ≥ 80`.
   - Symbol qualifies for LONG side if `iv_rank ≤ 20` AND (will be re-checked at 11:00 against NIFTY trend).

4. **T+0 11:00 IST entry candidate check:**
   - Confirm symbol is in F&O 200 universe (`assets/fno_liquid_200.csv`).
   - **NIFTY trend gate (LONG side only):** NIFTY 50 5m close at 11:00 > NIFTY 50 5m EMA(20). If false, abort LONG side for the day.
   - **Intraday IV-staleness check:** ATM-call price at 11:00 vs ATM-call close yesterday. If intraday IV proxy (ATM-call premium relative to underlying spot) has moved ≥30% in the same direction the daily signal anticipated, the revert has already partially happened — ABORT.
   - **Underlying confirmation candle (5m bar 10:55-10:59 close at 11:00):**
     - SHORT: 5m bar close < bar open AND close < VWAP(intraday-since-09:15) (red candle with downside conviction)
     - LONG: 5m bar close > bar open AND close > VWAP(intraday-since-09:15) (green candle with upside conviction, plus NIFTY trend already gated)

5. **T+0 entry** at 11:00 IST 5m bar close:
   - **Direction**: per side qualification above.
   - **Entry price**: 5m bar's CLOSE.
   - **Stop**: entry × (1 ± 1.0%) — 1% hard stop. Locked param.
   - **T1**: 1R (50% qty closed, breakeven trail activated for remaining).
   - **T2**: 2R (remaining 50% closed).
   - **Time stop**: 15:10 IST (5m before MIS auto-square cutoff). Locked param.
   - **Latch**: one fire per (symbol, T+0).

6. **target_anchor_type**: `arithmetic` — T1/T2 are R-multiples of the 1% stop. Different from `circuit_t1_fade_short` which uses gap-edges (structural anchors). Choice rationale: IV-rank-revert mechanics do not have a natural price-level anchor (no gap, no circuit band) — R-multiples are the correct choice for this structural type.

## Universe

**Allowed universe**: F&O 200 stocks (`assets/fno_liquid_200.csv`). This is a **strict requirement** — the IV-rank signal only exists where stock-options exist, which is exactly the F&O universe.

**Why F&O 200:**
- Single-stock options exist only for F&O-listed names. The signal is mechanically inaccessible outside this universe.
- F&O 200 (vs full F&O list of ~180-220 names) is the operating universe used elsewhere in this codebase. Consistency with `gap_fade_short` and round-3 candidates.

**Symbol count after universe + per-day liquidity filters:**
- 200 candidate symbols × ~10% have IV-rank in extreme zone on any given day = ~20 candidates/day pre-filtering.
- After intraday-confirmation gates: ~3-8 entries/day.
- Annual: ~750-2,000 entries/year. n ≥ 500 over 1 year easily satisfied; sanity over 2024+2025 = 2 years gives n ~1,500-4,000.

## Active window

**Setup formation**: T-1 EOD (bhavcopy ingestion + IV-rank computation; ~5 min compute per session).
**Entry**: T+0 11:00 IST single-bar entry at the 10:55-10:59 5m close.
**Hold horizon**: 11:00 → 15:10 IST = 4h 10m intraday MIS.

**Why 11:00 entry (not 09:15, 09:30, or 14:00):**
- 09:15-10:00 = morning gap-noise + opening auction overhang. Daily-stale IV-rank signal contaminated by overnight news flows.
- 10:00-10:30 = first wave of FOMO-buying / panic-selling resolves. IV-rank-driven mean-revert is not yet in play.
- **10:30-11:00** = inflection. Morning momentum exhausted; institutional vol-arb desks active in single-stock options. IV-rank revert begins.
- **11:00 = entry**. Aligns with locked param spec; matches the 11:00-15:00 active window mandate from round-3.
- 14:00+ = late-day MIS auto-square unwind contaminates the signal. Time-stop at 15:10 leaves enough cushion before 15:20 auto-square.

**Active window**: 11:00-15:10 IST (entry only at 11:00; trade is held through window with stops/targets active throughout).

**Time-of-day diversification fit:** Existing portfolio has `gap_fade_short` (09:15-09:30) and `circuit_t1_fade_short` (10:30 single-bar). This brief's 11:00 entry adds the third leg in the morning-to-midday transition; round-3 candidates (`vwap_deviation_meanrevert`, `index_stock_divergence_revert`, `volume_spike_exhaustion_reversal`) cover the broader 11:00-15:10 window throughout the day. Low correlation expected (different signal source — option-chain IV vs equity OHLCV).

## Risks / falsification criteria

The setup is **wrong** (and should be retired) if:

1. **Phase-1 floor fails on validation/holdout:**
   - n < 500 trades over 2 years (2024+2025 sanity)
   - NET PF < 1.10
   - NET Sharpe ≤ 0
2. **Direction asymmetry collapses:** if SHORT-side PF < LONG-side PF (after NIFTY-gate), the SEBI-aligned thesis is wrong; the signal might be picking up a different mechanism (e.g., low-IV expansion with no retail-loss component).
3. **LONG side fails the asymmetric gate test:** if LONG-side PF < SHORT-side PF × 0.85, ship SHORT-only — same gating logic as `volume_spike_exhaustion_reversal` round-3 brief.
4. **Daily-stale signal too stale:** if PF improves substantially when the IV-rank cutoff is recomputed at 09:15 IST T+0 (using opening option prices) vs T-1 EOD, the daily-stale pattern is not viable here. (Addresses the staleness concern empirically.)
5. **Liquidity excludes too much:** if the `vol ≥ 100, oi ≥ 1000` filter drops > 70% of candidate-days, the F&O 200 universe is mostly illiquid in options and the n drops below 500.
6. **IV-computation noise dominates:** if bhavcopy settlement IV vs Black-Scholes-inverted IV produce materially different signals, the IV measurement itself is unreliable.

**Falsification thresholds (locked):** PF < 1.10 OR n < 500 OR NET Sharpe ≤ 0 → retire.

## Pre-coding sanity check (mandatory per §3.3, BEFORE detector implementation)

Tool to build: `tools/sub9_research/sanity_options_vol_iv_rank_revert.py`

Steps:
1. Load EOD bhavcopy parquets from `data/option_chain/<YYYY>/<MM>/*.parquet` for 2024-01-01 → 2025-12-31 (2 years).
2. Per symbol, per session: identify ATM strike of nearest expiry; extract ATM IV; apply `vol ≥ 100, oi ≥ 1000` liquidity filter.
3. Compute 252-day rolling IV-rank per symbol. Drop symbols with < 100 days of history at any point.
4. Tag candidate days: `iv_rank ≥ 80` (SHORT) or `iv_rank ≤ 20` AND NIFTY-trend-up (LONG).
5. Apply 11:00 intraday confirmation gates from per-symbol 5m enriched feathers.
6. Simulate trades: 1% hard stop, T1 = 1R, T2 = 2R with breakeven trail, time stop 15:10.
7. Compute NET PF using existing Indian fee model (per-trade MIS leverage, brokerage, STT, GST per `tools/report_utils.py`).
8. Decision per §3.3: PF ≥ 1.10 → APPROVE for detector implementation; 1.0-1.10 → marginal (revisit param sensitivity); PF < 1.0 → retire.

Output: `reports/sub9_sanity/options_vol_iv_rank_revert_trades.csv`.

## Data engineering plan (preliminary, NOT yet built)

Required new components (only if sanity check passes):

1. **`tools/option_chain/compute_iv_rank.py`** — daily batch tool that reads `data/option_chain/<YYYY>/<MM>/*.parquet`, computes per-symbol ATM IV + 252-day IV-rank, writes to `data/iv_rank/<YYYY>/<MM>/<YYYY-MM-DD>.parquet`. Backfill from 2023-01 for 2-year history (2 years × 252 days = 504 sessions; computation budget < 30 min total).
2. **`structures/options_vol_iv_rank_revert_structure.py`** — the detector. Reads daily IV-rank cache + intraday 5m bars + NIFTY 50 5m feather. Fires at 11:00 IST.
3. **Live integration:** existing `tools/option_chain/fetch_oi_snapshot.py` is already in production schedule. Add a downstream cron (or inline call) for `compute_iv_rank.py` to run T-1 EOD post-bhavcopy. Detector reads the resulting daily parquet at session start T+0 09:00 IST.
4. **Config keys to add to `config/configuration.json`** (per CLAUDE.md mandatory rule: NO hardcoded defaults):
   - `options_vol_iv_rank_revert.iv_rank_lookback_days = 252`
   - `options_vol_iv_rank_revert.iv_rank_high_threshold = 80`
   - `options_vol_iv_rank_revert.iv_rank_low_threshold = 20`
   - `options_vol_iv_rank_revert.entry_time_ist = "11:00"`
   - `options_vol_iv_rank_revert.time_stop_ist = "15:10"`
   - `options_vol_iv_rank_revert.stop_pct = 0.01`
   - `options_vol_iv_rank_revert.t1_r_multiple = 1.0`
   - `options_vol_iv_rank_revert.t2_r_multiple = 2.0`
   - `options_vol_iv_rank_revert.intraday_iv_staleness_pct = 0.30`
   - `options_vol_iv_rank_revert.option_min_volume = 100`
   - `options_vol_iv_rank_revert.option_min_oi = 1000`
   - `options_vol_iv_rank_revert.nifty_trend_ema_period = 20`

## Sample-size feasibility

- F&O 200 stocks × ~250 trading days = 50,000 symbol-days/year baseline.
- After liquidity filter (`vol ≥ 100, oi ≥ 1000`): conservatively retain ~70% of symbol-days = 35,000.
- IV-extreme rate (≥80 or ≤20) by definition of percentile = 40% of distribution but with serial correlation (an IV-rank-≥80 day is often followed by another) → effective independent days ≈ 10-15% of universe-days = 3,500-5,250 candidate-days/year.
- After NIFTY trend gate (LONG-side only halves the LONG candidates) and intraday confirmation gates: estimated **800-2,000 entries/year**.
- 2-year sanity (2024+2025): **n = 1,600-4,000 trades**. n ≥ 500 floor easily satisfied.

**Sample-size honesty:** the brief assumes IV-extreme dispersion across F&O 200 is similar to US single-stock options (~10-15% effective independent days at the extreme percentiles). This is an unverified-in-Indian-data assumption; sanity-check will validate. If Indian single-stock options have lower IV-rank dispersion (e.g., most names are clustered in middle quintiles), n could collapse to 200-500/year and the candidate dies at Phase-1 floor.

## Honest comparison to surviving / shortlisted setups

| Aspect | gap_fade_short (TRUSTED) | circuit_t1_fade_short (APPROVED) | options_vol_iv_rank_revert (proposed) |
|---|---|---|---|
| Indian-specific source | retail momentum exhaustion at 09:15 open | upper-circuit operator-pump exhaustion T+1 | retail F&O option-buyer lossage (SEBI FY23) |
| Direction | short-only | short-only | bidirectional, SHORT primary + LONG NIFTY-gated |
| Active window | T+0 09:15-09:30 | T+1 10:30 single-bar | T+0 11:00 single-bar |
| Universe | small_cap (cash) | mid_cap, small_cap (cash) | F&O 200 |
| Hold | intraday MIS (15-30 min) | intraday MIS (4h 45m) | intraday MIS (4h 10m) |
| Signal source | intraday OHLCV | T-1 daily + price-band CSV | **T-1 EOD option-chain bhavcopy** |
| Daily-stale-signal pattern | no | yes | **yes** (same pattern as circuit_t1) |
| Evidence base | sub-7 validation | 5 peer-reviewed papers | SEBI FY23 + 4 Indian retail-algo + 2 peer-reviewed |
| Correlation with morning setups | n/a | low (different timing) | low (different signal source — option-chain not equity OHLCV) |

The proposed setup is the **first option-chain-derived signal** in the portfolio. All existing setups (and round-3 shortlist) use equity OHLCV directly. Adding an option-chain-derived signal:
- Uses already-running ingestion (no new data infrastructure)
- Targets the documented retail-F&O lossage (SEBI-validated asymmetry)
- Fires at 11:00 (fills the gap between 10:30 circuit_t1 and the round-3 mid-day setups)
- Has materially uncorrelated signal source vs the rest of the portfolio (data orthogonality reduces sequence-of-correlated-losses risk)

The two parallel round-4 briefs (sector-rotation, earnings) target different daily-stale signals (sector-momentum and corporate-events). All three round-4 candidates share the daily-stale-signal architecture pioneered by `circuit_t1_fade_short`.

---

## Decision required

User to indicate:
- [ ] APPROVED — proceed to pre-coding sanity-check script
- [ ] REJECTED — reason
- [ ] REVISE — specify what's missing / wrong

Per sub-9 §3.3, no code is written until APPROVED.
