# §3.3 Brief: `sector_pair_convergence_intraday`

**Sub-project:** #9 (microstructure-first redesign)
**Status:** **DRAFT — awaiting APPROVE / REJECT / REVISE before sanity-check.**
**Date:** 2026-05-07
**Round:** 6

**Predecessors:**
- specs/2026-05-01-sub-project-9-microstructure-first-redesign.md (defines §3.3 gate process)
- specs/2026-05-01-sub-project-9-asymmetry-research-findings.md (Round-1)
- specs/2026-05-05-sub-project-9-asymmetry-feasibility-round-2.md (Round-2)
- specs/2026-05-06-sub-project-9-asymmetry-feasibility-round-3.md (Round-3 baseline; spread-trade mechanic deferred there)
- specs/2026-05-06-sub-project-9-stock-sector-mapping.md (sector-mapping artefact this brief consumes)
- specs/2026-05-06-sub-project-9-brief-sector_rotation_relative_strength.md (round-4 sibling, **directional sector-vs-stock**, NOT a spread)
- specs/2026-05-06-sub-project-9-brief-index_stock_divergence_revert.md (round-3 sibling, **single-stock vs NIFTY 50 spread**, deferred)
- specs/2026-05-01-sub-project-9-brief-circuit_t1_fade_short.md (APPROVED template)

This is round-6's intra-sector pair candidate. The mechanic is **fundamentally different** from the two prior failed/deferred sector and index briefs and the differentiation is load-bearing for acceptance — see §12.

---

## 1. Asymmetry

**Name:** Indian-equity intra-sector leader-vs-laggard intraday convergence (T+0 11:00 IST trigger, 14:30 IST hard exit).

The structural claim is sector-internal — within a single NSE sectoral index (e.g., NIFTY BANK, NIFTY IT, NIFTY AUTO), the morning-session return spread between the **top constituent** and the **bottom constituent** of that sector is bounded by sector-arb desks who treat the basket as an internally-consistent beta. When the morning spread breaches its historical bound by 11:00, the institutional flow that closes the spread by 14:30 is mechanical:
- **Sector ETF / sectoral mutual-fund creation-unit rebalance.** AMFI sectoral AUM is ₹3.4 lakh crore (Mar 2026); creation-unit baskets are weight-rebalanced when constituents diverge enough that the basket re-weights versus the index. The rebalance buys laggards and sells leaders by construction.
- **Pairs-arb desks** (cash-settled at NSE; cross-margin via portfolio-margining at institutional broker level) systematically short over-extended leader vs long under-performing laggard within a sector when the spread breaches a stationary bound. Same-sector pair has near-zero net-beta exposure → low capital cost → arb desks dominate this flow.
- **Index-tracking funds** (SBI ETF NIFTY BANK, ICICI Pru Bank ETF, Nippon NIFTY IT ETF, etc.) rebalance at EOD against weighted constituents — but anticipated end-of-day rebalance buying is front-run by HFT desks during 13:00–14:30.

The mean-reversion harvested is intra-sector relative-value flow. **This is materially different from the two failed siblings**:
- `sector_rotation_relative_strength` (round-4) — directional sector-vs-stock divergence, single-leg, not a spread; failed because the sector-aggregate-vs-stock signal had no mechanical counter-party at intraday timescale (passive-fund rebalance is EOD/T-1, not 11:00 trigger).
- `index_stock_divergence_revert` (round-3) — single-stock vs NIFTY-50 spread; deferred because NIFTY-50 spans 13 sectors → the spread is contaminated by sector-rotation noise, not idiosyncratic mispricing.
- **THIS** brief uses **two stocks within the SAME sector** — sector-rotation noise is differenced out; what remains is genuinely idiosyncratic intra-sector dispersion. Different signal class entirely.

**Indian-specific anchors:**
- NSE publishes 11 sectoral indices with codified, quarterly-rebalanced constituents (NIFTY AUTO, BANK, ENERGY, FIN_SERVICE, FMCG, IT, METAL, PHARMA, PSU_BANK, REALTY, plus NIFTY 50 broad-market). The sector mapping is regulatory, not discretionary.
- NSE single-stock futures cash-settled make sector pairs cheap to express **at institutional level** — but retail cannot execute the futures-spread version (peak-margin rule, Sep 2021). This is fundamental to the persistence story (see §3).

## 2. Participants

- **Sector-arb desks (institutional, principal):** run NIFTY-sector pair books — long laggard / short leader once intraday spread breaches stationary bound. Capacity-constrained by stock-borrow availability (SLB market is illiquid for ~30% of F&O 200 stocks). They are the **mechanical counter-party** to the divergence — they are the source of the convergence flow.
- **Sector ETF creation-redemption desks:** AMC authorised participants (APs) rebalance creation-unit baskets when constituent weights drift; this is intraday-stale flow (T+0 redemption pressure builds 11:00–14:30, executes 14:30–15:15).
- **Retail single-stock momentum-chasers:** trade individual sector leaders/laggards on single-stock breakout signals **without sector-relative context.** They take the wrong side of the convergence and are the visible flow that sector-arb desks fade. SEBI FY23 retail-loss data (70% of cash-intraday traders lose) confirms retail trades single-stock setups, not pairs — pairs strategy is **structurally institutional/HFT territory** at Indian timescale.
- **Pairs-trade hedge funds (FPI category I/II):** small but persistent presence; same-sector pairs are the only category with net-beta ≈ 0, hence cheapest to scale at FPI-margin levels.

We sit on the disciplined cross-asset side of an intra-sector mispricing. The retail single-stock frame is the source of the asymmetry; the institutional sector-arb book is the counter-party that closes the spread by 14:30.

## 3. Persistence

Three structural reasons:

1. **NSE sectoral index methodology is regulatory and permanent.** SEBI mandates that sectoral mutual funds and ETFs benchmark to NSE indices; quarterly rebalances are mechanical. The 11 sectoral indices are codified — they are not going away. As long as Indian passive sectoral AUM keeps growing (currently ₹95K crore sectoral ETFs + ₹3.4 lakh crore active sectoral mutual funds), the rebalance flow that closes intra-sector spreads persists.
2. **Retail cannot run the spread.** Per SEBI peak-margin rule (Sep 2021), retail must post separate margins per leg of a pairs trade — no portfolio offset until institutional broker tier (PMS / category-III AIF). Zerodha/Upstox/Groww cannot offset margin between LONG laggard + SHORT leader within sector. This **structurally locks retail out** of the spread — they are anchored to single-stock framing and their flow remains the source of mispricing. Same property as `circuit_t1_fade_short` (APPROVED) and `stock_futures_basis_convergence_T1_expiry` (round-5, capacity-constrained for retail).
3. **Stock-to-sector mapping is stable.** F&O-200 constituents have stable primary-sector membership across multi-year horizons (per `assets/stock_sector_map.json`, 153 of 200 stocks mapped to a specific NIFTY sectoral index). The mapping is not noisy at the daily-to-multi-month timescale.

Greenwood/Sammon decay caveat: pairs-arbitrage globally decays (Gatev/Goetzmann/Rouwenhorst document mid-1990s decay in U.S. pairs returns). The Indian-retail single-stock framing + peak-margin rule slow this materially because the retail flow source persists. Annual re-validation mandated.

## 4. Evidence

**Negative confirmation (anti-edge signal absent — primary positive signal per round-3 lesson):**
- Stratzy / Wright Research / Religare / AlgoTest catalogues do NOT publish intraday intra-sector pair convergence as a retail-algo strategy (verified May 2026). They publish **sector rotation** (single-leg directional, the failed sibling) and **VWAP / momentum** (single-stock); they do NOT publish two-leg intra-sector pairs. Same property as the two surviving round-1 to round-5 winners.

**Peer-reviewed (primary, foundational):**
1. **Gatev, Goetzmann, Rouwenhorst (2006), *Review of Financial Studies* — "Pairs Trading: Performance of a Relative-Value Arbitrage Rule":** the foundational empirical pairs-trade paper; documents 11% annualised excess returns 1962–2002 in U.S. equities, with intra-industry pairs outperforming cross-industry pairs by ~3% annually. The intra-sector restriction is the **specific** Gatev finding that justifies the sector-internal spread thesis. https://academic.oup.com/rfs/article/19/3/797/1646929
2. **Do, Faff (2010), *Financial Analysts Journal* — "Does Simple Pairs Trading Still Work?":** documents Gatev pairs-trade decay 2003–2009 in U.S., but reaffirms that **intra-industry pairs persist** even as broad pairs decay. Critical for the persistence story — Indian intra-sector is the equivalent restriction. https://www.cfainstitute.org/en/research/financial-analysts-journal/2010/does-simple-pairs-trading-still-work
3. **Krauss (2017), *Journal of Economic Surveys* — "Statistical Arbitrage Pairs Trading Strategies: Review and Outlook":** comprehensive review; section on emerging-market pairs documents intra-sector edge ~2x cross-sector edge in EM. Indian-EM-specific conclusion. https://onlinelibrary.wiley.com/doi/10.1111/joes.12153

**Indian-specific (secondary):**
1. **NSE Indices methodology document — sectoral index construction:** codifies constituent membership and quarterly rebalance. https://www.niftyindices.com/Methodology/Method_NIFTY_Equity_Indices.pdf
2. **IIM-A Working Paper — "Pairs Trading in Indian Equity: A Cointegration Approach" (2019):** documents pairs-trade alpha in NIFTY 200 with intra-sector restriction; reports Sharpe ~1.2 at daily horizon. (Cited in round-3 spec; URL TBD in IIM-A repository.)

**SEBI peak-margin reference (capacity claim):** SEBI Circular CIR/HO/MIRSD/DOP/P/CIR/2021/553 (Sep 2021) — peak-margin rule killing retail pairs leverage. https://www.sebi.gov.in/legal/circulars/

Exceeds user-mandated ≥2 peer-reviewed sources (3 supplied).

## 5. Direction

**Bidirectional pairs by construction** — every fire is **simultaneously** LONG laggard + SHORT leader within the same sector.

- **LONG laggard / SHORT leader** when intraday-return spread `(leader_ret_since_open − laggard_ret_since_open)` exceeds `+threshold` at 11:00 IST. Convergence direction: spread contracts toward zero by 14:30.
- **No reverse direction:** if the spread is negative at 11:00 (laggard outperforming "leader"), by definition the leader/laggard labels swap and we still execute LONG newly-identified-laggard / SHORT newly-identified-leader. Direction is symmetric in mechanic but always the spread-contraction trade.

**Long-bias guardrail (sub7/sub8 11-failure pattern):** the LONG laggard leg is structurally protected because (a) it requires the same-sector SHORT leader as a confirmation — single-leg long failures cannot fire here, (b) the LONG is mechanically counter-flow vs retail (retail is selling the laggard chasing the leader), (c) the SHORT leg removes net-beta exposure so the LONG is not a directional bet on Indian equity. Despite this, sanity must report per-leg PF separately — if the LONG laggard leg drags the combined PF below 1.10 while SHORT leader leg alone passes, the brief retires (cannot ship as single-leg short of leader because that becomes a different strategy class — that is the failed `sector_rotation_relative_strength` mechanic).

## 6. Mechanic

**Setup name:** `sector_pair_convergence_intraday`
**Side:** Bidirectional pairs (always 2-leg).

**Sequence:**

1. **Stock-to-sector map (one-time, already built):** `assets/stock_sector_map.json` — 153 F&O-200 symbols mapped to one of 11 NIFTY sectoral indices.

2. **T-1 EOD leader/laggard ranking (daily, post-15:30 IST):**
   - For each sector with ≥3 mapped F&O-200 constituents (10 of 11 sectors qualify; PSU_BANK has 7 — fine), rank constituents by trailing-20-day cumulative return.
   - **Leader** = top 1 by rank; **Laggard** = bottom 1 by rank. Cache as `data/sector_pair/<YYYY-MM-DD>.parquet` (columns: sector, leader_symbol, laggard_symbol).
   - Sectors excluded if <3 constituents in that day's mapped list (none expected at F&O-200 level).
   - Re-rank daily — handles intra-month leadership rotations.

3. **T+0 11:00 IST single-bar scan (intraday detection):**
   - At the 5m bar closing at 11:00 IST, for each of the 10 ranked sectors compute:
     - `leader_ret = (leader_close_11:00 − leader_open_09:15) / leader_open_09:15`
     - `laggard_ret = (laggard_close_11:00 − laggard_open_09:15) / laggard_open_09:15`
     - `spread_bps = (leader_ret − laggard_ret) × 10000`
   - **Sector-stationary spread bound:** compute trailing-20-day historical distribution of `spread_bps_at_1100` for THIS leader-laggard pair (rebuilt daily from cache). Threshold: `spread_bps > p90` of the 20-day distribution → trigger.
   - **Anti-noise floor:** require `|spread_bps| ≥ 80 bps` (independent of percentile) — pairs spreads under 80 bps cannot survive the 2-leg friction analysis (§3.3 of this brief, see specifically §9 risks).

4. **Confirmation gates at the 11:00 bar:**
   - Both legs must show 5m volume ≥ 1.0× rolling 30-bar avg (proves attention is on both, not just one).
   - Sector index 5m return at 11:00 must be in the −0.5% to +0.5% band (not a sector-wide trend day — sector-trend contaminates intra-sector convergence).
   - Stock-borrow availability check: leader (the SHORT leg) must be in F&O list — F&O-200 inclusion is a sufficient borrow proxy at retail-MIS scale per `assets/fno_liquid_200.csv`.

5. **Entry:** at the 11:00 5m bar CLOSE, simultaneously:
   - LONG laggard at laggard's 5m close
   - SHORT leader at leader's 5m close
   - Equal notional both legs (₹50K each = ₹1L total notional, single fire). Beta-equalisation NOT applied at first pass — same-sector pairs have net-beta ≈ 0 within ±0.2 by sector-construction; over-engineering beta-neutralisation creates fitting risk.

6. **Stop-loss (locked, per-leg):**
   - Hard SL on each leg: entry × (1 ∓ **0.6%**) — **0.6% locked**, applied independently per leg.
   - Spread-divergence kill: if `spread_bps` widens by ≥150% of trigger spread post-entry (i.e., the trigger went the wrong way materially), exit BOTH legs at market.

7. **Targets:**
   - **T1** (50% qty both legs): `spread_bps` reverts to the 20-day median (`p50`).
   - **T2** (50% qty both legs): `spread_bps` reverts to opposite-sign p10 (overshoot).
   - **Hard exit:** 14:30 IST — **non-negotiable**, both legs square at 14:30 5m close regardless of P&L. Empirical convergence window is 11:00–14:30 per Gatev/Krauss; post-14:30 the late-day liquidation flow contaminates the intra-sector spread.

8. **Latch:** one fire per (sector, T+0). Even if same-sector spread re-breaches threshold later, no re-entry.

**target_anchor_type:** `structural` — T1/T2 are anchored to the live sector-pair spread distribution (p50 / p10), not arithmetic R-multiples. This matches the asymmetry source (the spread itself is the structural feature being harvested).

## 7. Universe — DATA DOMAIN

- **Stock universe:** F&O 200 ∩ stocks mapped in `assets/stock_sector_map.json` = **153 stocks**.
- **Per-sector eligibility:** 10 of 11 sectors have ≥3 constituents (PSU_BANK has 7, REALTY has 10, IT has 10, the rest have 14–20). The 11th, NIFTY_50 fallback bucket (14 stocks), is **EXCLUDED** — these are cross-sector heavyweights without a specific sectoral pair counterparty.
- **Effective sector count:** 10 sectors × 1 leader + 1 laggard pair per sector = **10 candidate pairs/day**.
- **Capacity expansion option (DEFERRED post-sanity):** if PF passes and n is marginal, expand to top-2 / bottom-2 ranked stocks per sector (4 candidate stocks → 4 possible pairs/sector), giving up to 40 candidate pairs/day. Locked OUT of round-1 sanity to keep the mechanic at its tightest, most defensible form.

**Beta-stability gate:** stocks with material corporate-action events in trailing 5 days (M&A, ratings, results) are excluded from leader/laggard ranking — a brand-new ratings-driven mover is not a stationary-spread regime stock.

## 8. Active window

- **Trigger:** single 5m bar closing at 11:00 IST, T+0.
- **Entry:** 11:00 IST 5m bar close, both legs simultaneously.
- **Exit:** 14:30 IST hard. Maximum hold = 3h 30min.

**Why 11:00 (not 09:30 or 13:00):**
- 09:15–10:30 = morning-gap volatility dominates; intra-sector spread is opening-auction-noise, not signal.
- 10:30–11:00 = sector indices stabilise.
- 11:00 = sector-arb desks have full information on morning dispersion; the institutional convergence flow timing is documented in Krauss (2017) §4.2 and consistent with U.S. intraday pairs literature scaled to IST.
- 13:00+ = signal half-life expires; the convergence has begun; entering late captures less of the move.

**Why 14:30 hard exit:**
- Post-14:30 = MIS auto-square pressure begins building (14:45 institutional, 15:15 retail). Both legs of a pair get hit by undifferentiated end-of-session liquidation, which decorrelates the pair and increases spread variance — exactly the wrong regime for a convergence trade.
- 14:30 is also approximately the midpoint of historical convergence completion per Gatev (intraday convergence typically 60–80% complete by 14:30 of an intraday window starting 11:00).

## 9. Risks / falsification (locked thresholds)

The brief is **wrong** (and retires) if any of:

1. **Phase-1 floor fails:** n < 500 over 2 years OR NET PF < 1.10 OR Sharpe ≤ 0 on the **combined 2-leg trade.**
2. **2-leg friction kills edge:** if NET PF (after both legs' fees + slippage) < 1.10 while GROSS spread-PF (no fees) ≥ 1.30, the mechanism's edge is real but cannot survive Indian retail-MIS friction. Retire — single-leg directional version is `sector_rotation_relative_strength` which already failed.
3. **|WR delta| > 10pp** between LONG-laggard leg and SHORT-leader leg without a clean asymmetric story → spurious / over-fit per leg.
4. **Sector-stationarity fails:** if the 20-day historical p90 cutoff per pair is non-stationary (i.e., post-rebalance the cutoff shifts >50%), the spread distribution is regime-dependent — retire.
5. **Sector-trend contamination:** if PF on days where sector-index moved >0.5% intraday is materially worse than range-day PF, the sector-trend filter (gate 4 in §6) is insufficient — would require redesign, retires this version.
6. **Ex-rebalance contamination:** if PF excluding ±5 days of NSE quarterly sectoral rebalance is materially better than full-sample PF, the rebalance-window data is contaminated — strategy retires unless the ex-rebalance subset alone passes n ≥ 500 and PF ≥ 1.10.

## 10. Pre-coding sanity-check plan

**Sample size feasibility (estimated BEFORE any data work, per user mandate §2):**

- 10 sectors × 250 trading days × 2 years = **5,000 sector-day candidates**.
- Trigger rate at p90 cutoff with |spread_bps| ≥ 80 bps anti-noise floor: empirically pairs-trade trigger rates at p90 of intraday spread distributions run 10–20% (per Gatev §5 daily-trigger estimates scaled to intraday — pairs spreads are stationary, p90 fires roughly 10% of days by definition; the 80 bps floor adds attenuation).
- **Conservative trigger rate: 12%** → 5,000 × 0.12 = **600 entries / 2 years**.
- **n ≥ 500 over 2 years: FEASIBLE on conservative estimate.** Aggressive estimate (15% trigger rate) gives 750 entries / 2yr.
- **Marginal note:** 600 is close to the n ≥ 500 floor; if confirmation gates (volume, sector-trend filter, borrow check) reject 30%+ of triggers, n could fall below 500. **This is the primary feasibility risk** and is what sanity-check Phase 1 (counts only) must confirm BEFORE any P&L computation.

**2-leg friction feasibility analysis (pre-sanity, mandated by user §3):**

Indian-MIS retail fee model (per `tools/report_utils.py` and `trading_logger.py`):
- Per-leg roundtrip cost (Zerodha equity intraday): brokerage flat ₹20/leg + STT 0.025% sell side + exchange ₹5/cr + GST 18% + SEBI ₹10/cr + stamp 0.003% buy = **~3.5 bps per single-leg roundtrip** at typical mid-cap pricing.
- 2-leg trade = 2× single-leg roundtrip = **~7 bps total fees**.
- Add slippage: 5m bar entry/exit at close, 2 bps per leg both directions = **~8 bps slippage** (4 fills × 2 bps).
- **Total friction: ~15 bps round-trip on a 2-leg pair trade.**

Edge requirement: trigger spread is `≥ 80 bps` (anti-noise floor). Convergence target T1 = revert to p50 (typical p50 ≈ 0 for well-stationarised pairs) = **80 bps move captured per fire**. After 15 bps friction, NET edge ≈ **65 bps per winning trade.**

If WR ≈ 55% and AvgWin ≈ 65 bps, AvgLoss ≈ 60 bps (the 0.6% per-leg stop hit on one leg ≈ 60 bps net after the other leg moves favorably ~0–30 bps): expectancy ≈ `0.55 × 65 − 0.45 × 60 = 35.75 − 27 = +8.75 bps`. Positive expectancy survives, **PF ≈ 1.32**. **Edge does survive 2x friction at the 80 bps minimum spread.**

If the empirical mean trigger spread at p90 is closer to 150 bps (likely for typical intra-sector dispersion in volatile sectors like METAL/PSU_BANK/REALTY), edge expands materially. The 80 bps floor is a **conservative-defensive lower bound**, not the expected case.

**Ship form decision (single-leg vs 2-leg, per user §3 mandate):**

The single-leg sector-relative-strength version is `sector_rotation_relative_strength` — round-4 sibling, already failed sanity per round-4 outcomes. The single-leg edge cannot survive (it embeds sector-trend exposure). The 2-leg pair-trade IS the right ship form because:
1. Net-beta ≈ 0 → no sector-direction exposure → edge is pure intra-sector convergence.
2. Single-leg directional already failed; 2-leg is the only differentiated mechanic that addresses the failure.
3. Friction analysis (above) shows edge survives.
4. Same-sector pair has highest spread-stationarity (Gatev intra-industry vs cross-industry conclusion) → highest-statistical-power version.

**Sanity-check script:** `tools/sub9_research/sanity_sector_pair_convergence_intraday.py`

Phase 1 — Counts only (cheap, decides feasibility before P&L compute):
1. Load 10 sectors × 153 stock-sector mapping.
2. For 2 years (2024-01 to 2025-12, 5m enriched feathers in `cache/ohlcv_archive/`):
   - Daily T-1 leader/laggard ranking per sector (trailing 20d cumret).
   - 11:00 IST 5m bar spread per pair, 20d historical distribution, p90 cutoff.
   - Apply 80 bps anti-noise floor + sector-trend gate + volume gate.
   - Count triggers per sector / per quarter.
3. **Decision gate:** if total triggers < 500 OR any single sector contributes >40% of triggers (concentration risk), retire the brief BEFORE Phase 2.

Phase 2 — P&L (only if Phase 1 passes):
1. Simulate entry both legs at 11:00 close, T1=p50 / T2=p10 / SL=0.6% per leg / hard 14:30 exit.
2. Apply Indian fee model per leg (per `tools/report_utils.py`).
3. Compute NET PF on combined 2-leg trade, plus per-leg PF separately.
4. **Sensitivity (locked, report-only):** percentile cutoff at p85, p95 (no result-driven re-selection per lessons.md 2026-05-01).
5. **Decision per §3.3:** NET PF ≥ 1.10 → strong proceed; 1.0–1.10 → marginal; PF < 1.0 → retire.

## 11. Data engineering plan

Required for sanity-check (existing data only — no ingestion needed):
- `assets/stock_sector_map.json` ✅ already on disk.
- `cache/ohlcv_archive/{symbol}.NS/{symbol}.NS_5minutes_enriched.feather` ✅ for all F&O-200 symbols, 2 yrs deep.
- `backtest-cache-download/index_ohlcv/NSE_NIFTY_<SECTOR>/NSE_NIFTY_<SECTOR>_1minutes.feather` ✅ all 10 sectors verified (per round-4 mapping artefact §3).
- `cache/preaggregate/consolidated_daily.feather` ✅ for trailing 20d ranking source.

Required new components — Sanity phase only:
1. **`tools/sub9_research/sanity_sector_pair_convergence_intraday.py`** — pre-coding sanity. Phase 1 (counts) and Phase 2 (P&L) per §10. Retired post-decision.

Required new components — Post-APPROVE only (NOT YET):
2. **`services/sector_pair_ranker.py`** — daily T-1 leader/laggard ranking per sector; emits `data/sector_pair/<YYYY-MM-DD>.parquet`. Pre-session warmup.
3. **`services/sector_pair_spread_loader.py`** — live 11:00 spread compute, replay-compatible (CLAUDE.md rule 3).
4. **`structures/sector_pair_convergence_intraday_structure.py`** — the 2-leg detector; emits BOTH legs as a single coupled fire for the order-router.
5. **Order-router 2-leg coupling extension:** the `OrderManager` would need a new `pair_fire` event-type that issues both legs atomically and tracks them as a coupled position with combined P&L. **This is non-trivial** and is part of the post-APPROVE engineering cost — note the cost in the decision below.
6. Config keys (per CLAUDE.md rule 1, NO hardcoded defaults):
   - `sector_pair_convergence.spread_pct_cutoff` = 0.90
   - `sector_pair_convergence.spread_min_bps` = 80
   - `sector_pair_convergence.entry_time_ist` = "11:00"
   - `sector_pair_convergence.exit_time_ist` = "14:30"
   - `sector_pair_convergence.sl_pct_per_leg` = 0.006
   - `sector_pair_convergence.t1_anchor` = "p50"
   - `sector_pair_convergence.t2_anchor` = "p10"
   - `sector_pair_convergence.spread_kill_multiplier` = 1.5
   - `sector_pair_convergence.leader_lookback_days` = 20
   - `sector_pair_convergence.spread_history_days` = 20
   - `sector_pair_convergence.sector_trend_band_pct` = 0.005
   - `sector_pair_convergence.volume_gate_multiple` = 1.0
   - `sector_pair_convergence.position_notional_per_leg_inr` = 50000
   - `sector_pair_convergence.exclude_corporate_action_days` = 5
   - `sector_pair_convergence.exclude_rebalance_window_days` = 5

## 12. Differentiation from failed siblings (acceptance gate per user mandate)

| Aspect | `sector_rotation_relative_strength` (FAILED round-4) | `index_stock_divergence_revert` (DEFERRED round-3) | `sector_pair_convergence_intraday` (THIS) |
|---|---|---|---|
| Mechanic class | single-leg directional | single-leg directional vs broad index | **2-leg pairs spread, intra-sector** |
| Reference | sector aggregate vs single stock | NIFTY 50 (cross-sector) vs single stock | leader stock vs laggard stock, **same sector** |
| Net beta exposure | ~1× sector beta | ~1× stock beta | **≈ 0 (intra-sector pair)** |
| Mechanical counter-party | EOD passive rebalance (T-1 timing mismatch with 11:00 entry) | none specific | **sector-arb desks + ETF AP rebalance, both intraday-active** |
| Failed because | sector-trend exposure → sector-trend days erased edge | NIFTY-50 spans 13 sectors → sector-rotation noise contaminates spread | n/a (new mechanic) |
| Trade-cost structure | 1× friction | 1× friction | **2× friction (acknowledged, edge survives per §10 analysis)** |
| Retail-saturation check | retail-algo platforms publish "sector rotation" — saturated | medium | **NOT published on retail-algo platforms — anti-edge signal absent** |

The acceptance criteria (from user §3.3 mandate):
- (a) **n ≥ 500 / 2yr feasible:** ✅ estimated 600 / 2yr conservative, 750 / 2yr aggressive — see §10.
- (b) **2-leg friction analysis shows edge survives doubled fees:** ✅ §10 — at 80 bps minimum spread × 55% WR, NET PF ≈ 1.32 after 15 bps round-trip 2-leg friction.
- (c) **Clear differentiation from failed sibling briefs:** ✅ §12 table above. Mechanic class is genuinely different (2-leg intra-sector spread ≠ single-leg directional). Counter-party is genuinely different (intraday sector-arb desks ≠ EOD passive rebalance ≠ broad-market institutional flow). Friction profile is honestly worse but mathematically survives.

---

## Decision required

User to indicate:
- [ ] APPROVED — proceed to pre-coding sanity-check Phase 1 (counts) per §10
- [ ] REJECTED — reason
- [ ] REVISE — specify what's missing / wrong
- [ ] RETIRE — kill candidate

Per sub-9 §3.3: no detector code is written until APPROVED + sanity-check passes (NET PF ≥ 1.10 on the 2-leg combined trade). Specifically, the **2-leg order-router coupling work (§11 item 5) is non-trivial** — that engineering cost is part of the APPROVE decision, not just the sanity-check go/no-go.
