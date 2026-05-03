# §3.3 Brief: `nifty500_deletion_short`

**Sub-project:** #9 (microstructure-first redesign)
**Status:** **DRAFT — pending user approval for §3.3 sanity check**
**Date:** 2026-05-03
**Predecessor:**
- specs/2026-05-01-sub-project-9-microstructure-first-redesign.md (defines §3.3 gate)
- specs/2026-05-01-sub-project-9-asymmetry-research-findings.md (G — CONDITIONAL, peer-reviewed)
- specs/2026-05-01-sub-project-9-brief-bulk_block_buy_continuation.md (RETIRED at sanity)
- specs/2026-05-01-sub-project-9-brief-circuit_t1_fade_short.md (APPROVED + IMPLEMENTED)

This is the **THIRD §3.3 brief** in sub-9. The asymmetry research findings doc (lines 86-99) listed **G — Index rebalancing** as CONDITIONAL with verdict "Worth building only the deletion-side / effective-day-reversion variants. Smaller effect than H or E." The shortlist line 125-130 promoted only H and E to brief stage. **This brief promotes G** off the conditional list because (a) the original two shortlisted candidates have been worked (one retired, one implemented), and (b) G has the next-strongest peer-reviewed evidence among remaining candidates.

---

## Asymmetry

**Name:** Indian-equity passive-AUM forced selling on NIFTY 500 deletions, with effective-day mean-reversion.

**Indian-specific source:**
- NIFTY 50 / Bank / Next 50 / 500 are reconstituted **semi-annually** (March / September) by NSE Indices.
- Announcement-to-effective gap is **~28 days**.
- Indian passive AUM tracking Nifty indices ≈ **₹9.7 lakh cr** (Oct 2025), per AMFI ETF data — which forces mechanical selling of deletions and buying of additions to align tracking error.
- The flow is **predictable in direction** (sell deletions, buy additions) and **time-bounded** (must complete by effective close).

**The exploitable asymmetry:**
- Additions: announcement-day CAAR +1.10% (Marisetty 2025) is **not statistically significant** in Indian data — front-running the addition has weak edge.
- Deletions: show **stronger initial negative reaction** at announcement, continued drift down through pre-effective period as passive funds rebalance, **partial recovery** beginning effective day.
- Therefore: **shorting the deletion** between announcement and effective day captures the forced-selling drift; **before the effective-day mean-reversion** kicks in.

## Participants

- **Pre-effective sellers**: passive ETFs and index mutual funds (~₹9.7 lakh cr AUM) executing mechanical rebalancing trades. Forced flow, no price-discovery role; sells regardless of fundamentals.
- **Pre-effective buyers**: arbitrageurs and active funds front-running the effective-day reversal, **but** Marisetty 2025 shows their participation is insufficient to neutralize the passive flow during the pre-effective window in Indian markets.
- **Post-effective buyers (effective day onward)**: arb-flow + active funds + the same passive funds who finished selling now becoming neutral to long-bias on price recovery.

We're on the disciplined side: **shorting the forced sellers' wave** before the active buyers arrive in size.

## Persistence

Three structural reasons:
1. **Regulatory schedule** — NSE Indices publishes the rebalance list ~28 days ahead. Both retail and active institutions know which stocks are being deleted; the front-running is publicly visible. Yet the effect persists because **passive funds cannot front-run their own rebalance** (tracking error rules force them to trade on/near effective day).
2. **Asymmetric AUM** — Indian active-fund AUM that could absorb forced flow has shifted toward passive (passive index funds + ETFs grew from ~₹0.5 lakh cr in 2018 to ~₹9.7 lakh cr in 2025 per AMFI). The forced-selling pool has grown faster than the active counterparty pool.
3. **Index inclusion criteria** — deletions in Indian markets often correlate with declining business fundamentals (low free-float, declining mcap, F&O exit). The forced flow pressure compounds genuine selling pressure, accelerating the drop.

These are SEBI/AMFI structural factors — not market-cycle-dependent. The "disappearing index effect" globally (Greenwood/Sammon NBER w30748) shows the US S&P 500 effect declined from 7.4% (1990s) to ~1% (2010s) as passive AUM saturated. **Indian markets are at the rapid-passive-growth phase the US was in 2000s** — the effect is currently strong but expected to decay over the coming decade. Backtest will show whether 2023-2025 still has tradable edge.

## Evidence (peer-reviewed, independent of retail communities)

1. **Marisetty 2025** (SSRN 5642110) — 81 NIFTY events 2010-2024. Additions: announcement-day CAAR +1.10%, p=0.54 (**not significant**), reversing to **−1.17% by effective day**. Deletions: stronger initial negative reaction with partial recovery. URL: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5642110

2. **Chhatwani (Publishing India)** — additions get permanent positive abnormal returns; reversal within 60 days. URL: https://www.indianjournals.com/

3. **Greenwood & Sammon NBER w30748** — global "disappearing index effect" — even in US S&P 500, effect declined from 7.4% (1990s) to ~1% (2010s). Indian markets at earlier stage (Indian passive AUM grew 2018-2025 vs US plateau post-2010). URL: https://www.nber.org/system/files/working_papers/w30748/w30748.pdf

3 peer-reviewed academic sources, independent of retail communities. ≥1 evidence requirement met.

**Counter-evidence:** Greenwood/Sammon's "disappearing effect" is the strongest argument FOR caution. The Indian effect is real today but on a decay curve. Sanity check on 2024 data establishes whether the 2024 effect is still tradable AFTER NET costs.

## Direction

**SHORT-ONLY.** Short pre-effective NIFTY 500 deletions.

We do **NOT** take the long side of additions per Marisetty — Indian addition CAAR pre-effective is +1.10% (p=0.54, statistically insignificant). The asymmetry between deletions (significant negative drift) and additions (insignificant positive drift) is itself the Indian-specific finding.

We do **NOT** take the long side of post-effective deletion recovery (the +1.17% reversal). Reasons:
- Effect is delayed (60-day window per Chhatwani) — incompatible with intraday MIS infrastructure
- Mean-reversion buy-side has multiple sources (active funds, arbs, retail) — less concentrated than the forced-selling short-side edge
- Project-wide pattern: long-bias setups in Indian intraday have systematically lost (SEBI FY23 + sub7/8's 11-failure pattern of cargo-culted longs)

Short-only on the forced-flow side aligns with the surviving setup library (gap_fade_short, circuit_t1_fade_short), all on the structural "retail/forced-flow loses" side of Indian intraday.

## Mechanic

**Setup name:** `nifty500_deletion_short`
**Side:** Short-only.

**Sequence:**

1. **Event detection (T-N, where N = 28 calendar days before effective)**:
   - On NSE Indices announcement day, parse the rebalance list:
     - Source: NSE Indices press release at `https://www.niftyindices.com/equity-indices` (or PDF circulars dated Feb/Aug).
     - Specifically the deletion list for NIFTY 500 (broader = more deletions = larger sample). Optional addition: include NIFTY 50, BANK, MIDCAP 100, NEXT 50 for additional events.
   - For each deletion announced, compute the **effective_date** (typically last Friday of March/September).

2. **Position sizing horizon (T-N+1 to T-1, the pre-effective short window)**:
   - **Two-stage entry strategy** (each is a separate trade):
     - **Stage A — Announcement entry** (T-N+1 09:15 IST): short at open with 1/3 of position size budget. Captures earliest forced-flow signal.
     - **Stage B — Concentration entry** (T-5 09:15 IST): if Stage A trade still open, scale-add 2/3 of position size. Captures the final-week forced-flow concentration.
   - **OR** — simpler single-entry variant for sanity check:
     - **Single entry** (T-1 09:15 IST): short at open, hold one full session. Captures the **highest-density forced-selling day** (passive funds typically rebalance at/near effective day).

   For sanity check, propose **the single-entry variant** (simpler, fewer parameters, smaller failure surface).

3. **T-1 entry** (single-entry variant):
   - Entry price: T-1 09:15 5m bar's CLOSE
   - Direction: SHORT
   - Confirmation gates at 09:15:
     - Stock is in NSE Indices' deletion list for the upcoming effective_date
     - Stock has NSE 5m data (i.e. is currently still trading on NSE — sanity check that delisting hasn't already happened due to corporate action like merger)
     - Average daily volume over T-30 to T-2 ≥ ₹5 cr (liquidity gate for short-side via SLB)

4. **Hold horizon**: T-1 09:15 → T-1 15:15 (intraday MIS), one trading day.
   - Why a single intraday day? Compatibility with existing MIS infrastructure. Multi-day hold would require CNC/SLB, which is a separate infra task.
   - The peer-reviewed effect IS multi-day (Marisetty's CAAR is computed over a window). Sanity check tests whether ONE DAY (T-1) captures enough of the cumulative effect to be tradable NET of fees.
   - If single-day MIS doesn't capture enough (PF < 1.10), the answer is "this setup is real but doesn't fit MIS infra" — retire and revisit when CNC/SLB infra is built. Don't loosen the brief to chase variants.

5. **Stop-loss**:
   - **Hard SL**: T-1 day's previous high (T-2 close + 1.0% buffer, or PDH × 1.005, whichever is higher). Defends against announcement-arbitrage rallies (unusual but possible if late deletion-list correction).
   - **Min stop distance**: 1.0% of entry (qty-inflation guard for thin small-caps that often get deleted).

6. **Targets**:
   - **T1** (50% qty): entry × (1 - 1.0%) — first 1% intraday move down.
   - **T2** (50% qty): entry × (1 - 2.0%) — second 1% (target full intraday day's typical move on a forced-flow day).
   - **Time stop**: 15:10 IST (5 min before MIS auto-square).

7. **Latch**: one fire per (symbol, T-1) — no re-entry same session.

**target_anchor_type**: `r_multiple` — T1/T2 are arithmetic R-multiples, not structural levels. (Different from circuit_t1's structural gap-edges; deletion forced-flow doesn't have a clean structural anchor.)

## Universe

**Intended universe**: NSE all stocks that appear on a NIFTY 500 (or related family index) **deletion list** for the upcoming reconstitution.
- **Cap segment** filter: any. Deletions can occur in any cap segment — large_cap deletion (e.g., a stock falling out of NIFTY 50 due to mcap rank) is rare but sample. Small_cap and mid_cap deletions are more common (NIFTY 500 churn).
- **No F&O 200 restriction** — peer-reviewed evidence is on broader index family.
- **Liquidity gate**: T-30 avg daily volume ≥ ₹5 cr (₹50,000,000 turnover) to ensure short-side fillability via SLB.

**Sample size feasibility:**
- Per Marisetty 2025: 81 NIFTY events 2010-2024, but this is for NIFTY 50 only. NIFTY 500 churn is much higher.
- Estimated NIFTY 500 deletions: ~50/yr (from research findings doc).
- Adding NIFTY Bank, Midcap 100, Next 50, Next 100 family: ~80-120 deletions/yr total.
- Over 2024 (sanity period): ~80-120 single-event trades.
- **Honest gap vs n ≥ 500 floor**: this is a TIGHT setup. A single year's sanity might have only ~80-120 trades. To reach n=500 over 2-3 years backtest needed. State this upfront in sanity result.

## Active window

**Setup formation**: announcement (T-N) — handled offline / in event-detection module, not at scan time.
**Entry**: T-1 09:15 IST (single bar — first 5m of the day before effective day).
**Hold horizon**: 09:15 → 15:15 IST = 6 hours intraday MIS, single day.

**Why T-1 (not T-N to T-1 multi-day):**
- MIS infrastructure compatibility (existing fee model, existing exit logic, existing risk gates)
- T-1 is the **highest-density forced-selling day** per Marisetty's CAAR concentration around effective
- Single-day hold avoids the overnight gap risk that compounds in multi-day setups
- If T-1 alone isn't tradable NET, the setup retires; we don't chase multi-day variants without a separate CNC/SLB infra task

**Why 09:15 entry (not 10:30 like circuit_t1):**
- Forced flow is concentrated at OPEN (passive funds use VWAP / TWAP execution starting at open)
- The full day's forced-selling is what we want to capture; intraday late entry misses the bulk
- Different mechanic from circuit_t1 (FOMO exhaustion, peaks 10:00-10:30) — index deletion forced-flow peaks 09:15 onwards

## Risks / falsification criteria

The setup is **wrong** (and should be retired) if:

1. **Phase-1 floor fails on validation/holdout**:
   - n < 500 trades over 1-2 years (high risk given sample size constraints — see "Sample size feasibility" above)
   - NET PF < 1.10
   - NET Sharpe ≤ 0
2. **"Disappearing index effect" applies to Indian markets** — if 2024 backtest PF is materially below 2010-2020 expectations from Marisetty's data, the effect has decayed faster than projected.
3. **Single-day MIS doesn't capture enough** — if T-1 alone gives PF < 1.10 but multi-day CNC would give PF > 1.50, the answer is "real edge, wrong infra" — retire from sub-9 and revisit when CNC/SLB infra is built.
4. **Liquidity gate fails too many candidates** — if the ₹5 cr ADV gate excludes >70% of deletions (because deleted stocks are often illiquid by definition), the tradable universe is too thin.

**Pre-coding sanity check** (mandatory per §3.3, BEFORE writing detector):
- Manually enumerate NIFTY 500 deletions for March 2024 + September 2024 reconstitutions (2 events, ~50-70 deletions each). Source: NSE Indices press releases (PDFs at https://www.niftyindices.com/equity-indices).
- For each deleted stock, simulate T-1 09:15 entry → 15:15 exit short, with R-multiple T1/T2 and PDH-based hard SL.
- Compute NET PF using existing Indian fee model (`tools/sub7_validation/build_per_setup_pnl.py:calc_fee`).
- **Decision per §3.3:** PF ≥ 1.10 → strong proceed; 1.0-1.10 → marginal; PF < 1.0 → retire.

## Data engineering plan (preliminary, NOT yet built)

Required new components (only if sanity check passes):

1. **`tools/sub9_research/sanity_nifty500_deletion_short.py`** — pre-coding sanity check (parallel to circuit_t1's). Reads:
   - Manually-curated CSV of (symbol, announcement_date, effective_date) for 2024 deletions
   - Existing 2024 5m feathers
   - Existing consolidated_daily.feather for PDH
   - No detector code yet. Will be retired after used.

2. **(post-sanity-check, only if APPROVED for full implementation):**
   - `tools/index_rebalance/fetch_nifty_reconstitutions.py` — automated NSE press-release scraper, normalized to a parquet at `data/index_rebalance/nifty500_deletions.parquet`
   - `services/index_rebalance_loader.py` — load deletion events keyed by effective_date, lookup by date
   - `structures/nifty500_deletion_short_structure.py` — the detector (cross-day state: detect() at T-1 09:15 reads upcoming-deletion list)

## Honest comparison to surviving setups

| Aspect | gap_fade_short (TRUSTED) | circuit_t1_fade_short (APPROVED) | nifty500_deletion_short (proposed) |
|---|---|---|---|
| Indian-specific | retail momentum exhaustion in T+0 opening | retail FOMO + operator pump exhaustion in T+0 close | passive-AUM forced selling on index deletion |
| Direction | short-only | short-only | short-only |
| Active window | T+0 09:15-09:30 | T+1 10:30 single-bar | T-1 09:15 (event-driven, semi-annual) |
| Universe | small_cap | mid_cap, small_cap | any cap, NIFTY 500 deletions |
| Hold | intraday MIS (15-30 min) | intraday MIS (4h 45m) | intraday MIS (6h) |
| Evidence base | empirical sub-7 validation | 5 peer-reviewed papers | 3 peer-reviewed papers |
| Expected n/yr | several thousand | ~500-700 | ~80-120 (TIGHT) |
| Correlation w/ existing | n/a | low (different timing) | low (event-driven, semi-annual) |
| Decay risk | low (regulatory + behavioral) | low (regulatory + behavioral) | **moderate** (Greenwood/Sammon "disappearing effect") |

**Honest summary of fit:**
- **Good fit:** Short-only, Indian-microstructure-specific, peer-reviewed evidence, intraday MIS-compatible, low correlation with existing setups.
- **Concerns:** Tight sample size (likely ~100/yr vs. ≥500 floor), event-driven (semi-annual) so backtest data is concentrated, decay risk per Greenwood/Sammon. PF could pass on 2024 data but be weakening; need to flag this in sanity result.

The brief explicitly accepts these concerns and asks the §3.3 gate to determine whether 2024 NIFTY 500 deletions still produce a tradable NET edge. If yes, proceed; if no, retire and the brief itself counts as evidence that "Indian index-effect has decayed past tradable threshold" — useful negative finding.

---

## Decision required

**User action:**
1. **APPROVE** for sanity-check coding → I write `tools/sub9_research/sanity_nifty500_deletion_short.py`, manually curate 2024 deletion list, run sanity, report PF.
2. **REJECT** with revisions → I revise specific points and re-submit.
3. **RETIRE before sanity** → if you judge the sample-size or decay-risk concerns alone are disqualifying, we skip G entirely and do a fresh research round for new candidates.

**My read:** APPROVE for sanity check. The peer-reviewed evidence + Indian-specific asymmetry justify spending the ~2 hour sanity-check budget. Sample-size is a known risk that the sanity will quantify. If PF < 1.10, we retire decisively (lesson 2026-04-22: don't iterate on a failed sanity).
