# §3.3 Brief: `bulk_block_buy_continuation`

**Sub-project:** #9 (microstructure-first redesign)
**Status:** DRAFT — pending user review per sub-9 spec §3.3 gate
**Date:** 2026-05-01
**Predecessor:**
- specs/2026-05-01-sub-project-9-microstructure-first-redesign.md (defines §3.3 gate)
- specs/2026-05-01-sub-project-9-asymmetry-research-findings.md (selected H as top candidate)

This is the FIRST test of the sub-9 §3.3 process. Per the gate, this brief must pass user review BEFORE any code is written.

---

## Asymmetry

**Name:** Indian-equity bulk-deal / block-deal information asymmetry on the BUY side.

**Indian-specific source:**
- **Bulk deal**: SEBI/NSE-defined transaction where a single client's cumulative trade > **0.5% of listed equity** in a stock during a session (executed in normal trading window). Reported same-day by 17:00 IST or within 1 hour of close.
- **Block deal**: SEBI/NSE-defined transaction ≥ **5 lakh shares OR ≥ ₹10 cr** in either of two special windows (08:45-09:00 morning, 14:05-14:20 afternoon), within ±1% of reference price. Must result in delivery; intraday squaring forbidden by rule. Disclosed same day after market close with client name + buy/sell flag.
- These categories and disclosure rules are SEBI/NSE-only — no US/forex analog. The US equivalent (Section 13D/13G filings) operate on different thresholds and timing windows.

The asymmetry is **end-of-day public disclosure with imperfect overnight repricing**. The deal happens during T+0; the public learns identity + side after market close; the first liquid repricing opportunity is T+1 open.

## Participants

- **Buy-side (informed)**: institutional accumulators — large mutual funds, FPIs, HNIs, strategic-PE buyers — entering positions on the basis of due-diligence work that retail/HFT have not yet seen. Disclosed buyer identity (visible in the bulk-deal CSV) often signals quality (e.g., named foreign portfolio investor with public mandate vs unknown HNI).
- **Sell-side counterparty in a buy-block** is heterogeneous: could be liquidity providers, exiting mutual funds, secondary-market sellers — generally NOT informed (they were filling the buy demand).
- **The slow-reacting public** (retail, smaller PMS, slower HFT) prices the disclosure imperfectly between T+0 close and T+1 open + first hour. This is where the edge lives.

The peer-reviewed paper (Agarwalla & Pandey, IIM-A NSE Research Initiative) explicitly attributes the **purchases > sales** asymmetry to information content: purchases are informed accumulation; sells are largely liquidity-driven (fund redemptions, portfolio rebalancing exits).

## Persistence

Three reasons the edge persists:

1. **Disclosure timing** — bulk/block deal data is published end-of-day, NOT intraday. The T+0 close → T+1 open gap is the structural delay; this is a SEBI rule on disclosure cadence, fixed by regulation.
2. **Information processing latency** — retail and slower professional flows take days to react fully (Managerial Finance 2022 documents 5-7 day post-disclosure drift). The price fully reflects the buy-side information only over T+1 to T+5.
3. **Asymmetric attention** — bulk/block deal feeds are public but not heavily aggregated by retail platforms; aggregators (Trendlyne, BSE/NSE official) exist but the signal is buried in daily noise of 30-80 bulk + 5-25 block events. Retail flow doesn't systematically arbitrage it.

These factors are SEBI/NSE-structural, not market-cycle-dependent. The edge has been documented across 2004-2019 in peer review (Chaturvedula EMR 2015 + Managerial Finance 2022); the regulation hasn't changed since.

## Evidence (peer-reviewed, independent of retail communities)

1. **Agarwalla & Pandey** — NSE Research Initiative paper, IIM-Ahmedabad faculty. Sample: NSE block trades (institutional). Findings:
   - ~1.32% average same-day return on block trades
   - 77% of block trades show positive abnormal returns
   - **Permanent price impact > for purchases than sales**
   - Sales mostly liquidity-driven; purchases mostly informed
   - URL: https://nsearchives.nseindia.com/content/research/NSE_Proposal_216_Final_Paper.pdf
2. **Chaturvedula, Bang, Rastogi & Kumar (Emerging Markets Review, 2015)** — bulk trades 2004-2012, NSE+BSE. CARs up to 7.49% around event; **front-running before disclosure** documented; post-disclosure drift continues.
   - URL: https://www.sciencedirect.com/science/article/abs/pii/S1566014115000138
3. **"What's hidden behind bulk deals" (Managerial Finance, 2022)** — NSE 2010-2019. Front-runners earn 5-7% within a week around event day; **pre-deal CARs higher for buy than sell** (the latter is liquidity-driven and weaker).
   - URL: https://www.emerald.com/insight/content/doi/10.1108/MF-08-2021-0374/full/html

All three are peer-reviewed academic / institutional sources, independent of retail trader communities. ≥1 evidence requirement met by 3 sources.

## Direction

**LONG-ONLY.** Buy-block T+1 continuation only.

Sell-side blocks have NO documented edge after fees:
- Sell blocks are largely liquidity-driven (fund redemptions, position-rebalances)
- Permanent price impact is small for sells (Agarwalla & Pandey)
- Net of round-trip costs, sell-side T+1 fade ≈ noise

This is a notable departure from naive symmetric setups. We are NOT taking the short side. SEBI FY23 long-bias-loses concern doesn't apply here because the LONG entry is on an explicitly informed-flow signal — not on retail momentum (which is what loses).

## Mechanic

**Setup name:** `bulk_block_buy_continuation`
**Side:** Long-only.
**Sequence:**

1. **T+0 EOD ingestion** (post 18:00 IST):
   - Pull NSE daily bulk-deals CSV (`https://nseindia.com/report-detail/display-bulk-and-block-deals`) and BSE equivalent
   - Filter to **buy-side rows only** (deal_type field)
   - Per stock per day, aggregate: total buy quantity, total buy value, distinct buy clients, weighted avg buy price
   - Apply **filters**:
     - Aggregate buy value ≥ ₹50 cr (avoid promoter-offload nuisance deals; literature shows higher-value deals carry more signal)
     - Stock must be in **F&O 200 universe** (per services/symbol_metadata.fno_liquid_200) — liquidity ensures price discovery, blocks manipulation
     - Stock cap_segment ∈ {`large_cap`, `mid_cap`} — exclude small/micro_cap (promoter-offload + cross-deal signal corruption higher there)
     - **Exclude promoter buys** (from SEBI PIT disclosures, looked up T+1 morning) — promoter accumulation is structurally different signal
     - **Exclude same-day cross-deals** where buy_client and sell_client share a parent group (heuristic on client name string match)
2. **T+1 entry** (09:15 IST, opening bar):
   - **Trigger**: surviving filter list from step 1
   - **Entry price**: opening 5m bar's close (09:20 close), NOT the 09:15:00 open (avoids opening auction noise)
   - **Direction**: LONG
3. **Stop**:
   - **Hard SL**: 1.5 × ATR(14d, daily) from entry. Indian retail standard.
   - Or alternatively: previous-day low − 0.3% (whichever is closer to entry, capping risk)
   - **Min stop distance**: 0.5% of entry (qty-inflation guard)
4. **Targets**:
   - **T1** (50% qty): +1R (one ATR above entry)
   - **T2** (50% qty): +2R or end-of-day trail-stop
   - **Time stop**: 15:15 IST EOD MIS auto-square (this is the MIS tradeoff — see Risks)
5. **Latch**: one fire per (symbol, T+1) — no re-entry same session.

**target_anchor_type**: `r_multiple` — this is an R-driven setup, not level-anchored. (Different from gap_fade_short which is `structural`.)

## Universe

**Intended universe**: `fno_liquid_200` (F&O top-200 by liquidity).
**Allowed cap segments**: `large_cap`, `mid_cap` only. Excludes small_cap, micro_cap.

**Why these:**
- F&O 200 ensures sufficient liquidity that bulk-deal disclosure causes price discovery rather than noise. Smaller stocks see manipulation deals (promoter friendly counterparty, etc.) that contaminate the buy-block signal.
- Large + mid cap brackets are where institutional accumulation actually concentrates. Peer-reviewed evidence (Agarwalla & Pandey) is on **block** trades (≥₹10 cr), which are by definition large enough to require liquid stocks.
- Excluding small/micro cap is the explicit defense against the cargo-cult failure mode: small-cap bulk deals are dominated by retail-driven promoter offloads and cross-deals — opposite signal.

**Approximate symbol count:** F&O 200 with large+mid filter ≈ 150 stocks at any given time.

## Active window

**Setup formation**: T+0 (any time during trading session, captured at EOD post-disclosure).
**Entry window**: T+1 09:15-09:25 IST (first two 5m bars).

**Why this window:**
- The structural information event is the EOD disclosure on T+0. The first liquid repricing is the T+1 open.
- 09:15 itself is the call-auction matched price; 09:20 close is the first regular-session bar. We use 09:20 close to avoid the call-auction artefact.
- After 09:25 (10+ minutes into the session), the disclosure information is increasingly priced in. Late entries lose net edge.
- This is the equivalent of gap_fade_short's tight 09:15-09:30 window — early-session entries on a structural-information event.

**Hold horizon**: same-day MIS, 09:25 to ≤15:15 (forced auto-square).

**MIS LIMITATION (honest disclosure):**

The peer-reviewed evidence documents 50-150 bps net edge cumulative T+1 to T+5 (5 trading days). Our system runs MIS-only, capturing only the T+1 portion of that edge. Same-day T+1 component is empirically smaller; the brief is testable but the multi-day part of the documented edge is unreachable in current framework.

**This is a real limitation and the test is the right way to learn:** if T+1-only is enough to clear fees + slippage with PF≥1.10, the setup is viable in our framework. If not, two paths:
- (a) Retire the candidate as not-system-fit
- (b) Add CNC mode (deliveries, overnight risk) to capture the multi-day edge — separate sub-project

We commit to (a) if T+1-only fails Phase-1.

## Risks / falsification criteria

The setup is **wrong** (and should be retired) if:

1. **Phase-1 floor fails on validation period (FY24-25 or fresh OCI capture)**:
   - n < 500 trades over the validation period (insufficient sample)
   - NET PF < 1.10 (no edge after fees)
   - NET Sharpe ≤ 0 (negative risk-adjusted return)
2. **Direction asymmetry collapses**: if BUY-side T+1 PF < 1.10 AND there's no spread vs sell-side performance, the buy-asymmetry thesis isn't surviving in our intraday framework.
3. **MIS-vs-multi-day collapse**: if T+1-same-day-MIS-only PF is materially worse than T+1-to-T+5 PF (computed via separate analysis), the documented edge is multi-day; system framework can't harvest it.

Pre-coding sanity check (BEFORE writing detector): pull 6 months of NSE bulk-deals data, simulate the filter + naive T+1-9:25-to-15:15 hold on a small Python script, see if NET PF on filtered events is ≥ 1.10. If under 1.0 even on the rough simulation, **retire the candidate before writing detector code.**

## Data engineering plan (preliminary)

Required new components (NOT yet built):

1. **`tools/bulk_deals/fetch_bulk_deals.py`** — daily CSV scraper from NSE + BSE, normalises into a single per-day parquet under `data/bulk_deals/<YYYY>/<MM>/<YYYY-MM-DD>.parquet` with columns:
   `session_date, symbol, deal_type (BULK/BLOCK), client_name, buy_sell, qty, weighted_price, source_exchange`
2. **`tools/bulk_deals/aggregate_buy_signals.py`** — per-day filter + aggregation pipeline producing T+1 long-candidate list with all the filters from Mechanic step 1.
3. **`structures/bulk_block_buy_continuation_structure.py`** — the detector. Reads T+0 aggregated buy-signals (computed during overnight backtest pre-step), checks at T+1 09:20 for symbol presence + emits StructureEvent.

Data backfill: NSE has bulk-deals archives back to ~2005. Need to backfill 2023 onward to align with the existing OCI capture window.

## Sample-size feasibility

From research findings: ~30-80 bulk + 5-25 block deals/day. After §filter (BUY only + ≥₹50 cr + F&O 200 + large/mid cap + non-promoter + no-cross-deal):

- Rough estimate: 5-10 surviving signals/day
- 250 trading sessions × 5-10 = **1,250-2,500 events/year**
- **n ≥ 500 over 2 years**: comfortably satisfied

## Honest comparison to surviving setups

| Aspect | gap_fade_short (TRUSTED) | bulk_block_buy_continuation (proposed) |
|---|---|---|
| Indian-specific | retail momentum exhaustion in opening | institutional accumulation disclosure lag |
| Direction | short-only | long-only |
| Active window | T+0 09:15-09:30 | T+1 09:15-09:25 |
| Universe | small_cap | F&O 200 large+mid_cap |
| Hold | intraday MIS | intraday MIS (T+1 same-day) |
| Evidence base | empirical sub-7 validation | 3 peer-reviewed papers |
| Correlation with gap_fade_short | none expected — different participants, opposite direction | uncorrelated diversification |

The two setups are structurally complementary: gap_fade harvests retail-momentum-exhaustion (short small-caps); bulk_block_buy harvests institutional-accumulation-information (long large/mid-caps). Different participants, different directions, different universes, different hold mechanics. Library diversification rather than overlap.

---

## Decision required

User to indicate:
- [ ] APPROVED — proceed to data-engineering plan + detector implementation
- [ ] REJECTED — reason
- [ ] REVISE — specify what's missing / wrong

Per sub-9 §3.3, no code is written until APPROVED.
