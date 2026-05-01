# Sub-project #9 — Asymmetry Research Findings

**Date:** 2026-05-01
**Predecessor:** specs/2026-05-01-sub-project-9-microstructure-first-redesign.md (§4 candidate inventory)
**Sub-project DoD item:** #3 ("Candidate asymmetry inventory reviewed by user with ≥1 asymmetry shortlisted for §3.3 brief")

---

## Methodology

For each candidate asymmetry in the §4 inventory of the sub-9 spec, a deep-research subagent investigated:
1. The exact mechanic (regulatory / structural)
2. Empirical price-impact evidence — restricted to **peer-reviewed academic / institutional / SEBI / NSE** sources, not retail trading communities
3. Direction asymmetry
4. Persistence rationale
5. Frequency / sample size
6. Data availability for backfill 2023-26
7. Verdict against the 5 sub-9 quality criteria

Sources cited per asymmetry. Findings synthesized below.

---

## Verdict matrix

| ID | Asymmetry | Indian-specific | Identifiable participants | Persistence rationale | Independent evidence | Asymmetric direction | **Verdict** |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| C | MIS auto-square retry | ✓ | ✓ | ✓ | ✓ | ✓ | **REJECT — exploitable residual too thin; CAS Mar-2026 invalidates pre-2026 backtest** |
| D | F&O ban-list crossing | ✓ | ✓ | ✓ | ✗ | unproven | **CONDITIONAL — promote only after in-house preliminary event study** |
| E | Circuit-band hit + recovery | ✓ | ✓ | ✓ | ✓ | ✓ | **PASS — strong** |
| F | FII / DII flow asymmetry | ✓ | ✓ | ✓ | ✓ | ✓ | **PASS — overlay only (aggregate-only data)** |
| G | Index rebalancing flows | ✓ | ✓ | partial | ✓ | ✓ (deletions > additions) | **CONDITIONAL — small Indian effect (~1%), favours deletions** |
| H | Bulk / block deals | ✓ | ✓ | ✓ | ✓ | ✓ (buy ≫ sell) | **PASS — strongest** |

---

## Per-asymmetry summary (evidence anchors)

### H. Bulk / block deals — STRONGEST CANDIDATE

**Mechanic:** Bulk deal = single client > 0.5% of equity in a stock; reported within 1 hour of close. Block deal = ≥ 5 lakh shares OR ≥ ₹10 cr in special windows (08:45-09:00, 14:05-14:20). Disclosed end-of-day with client name + buy/sell flag.

**Evidence:**
- Agarwalla & Pandey (IIM-A NSE Research Initiative): ~1.32% same-day return; 77% of block trades show positive abnormal returns; **purchases more informative than sales**
- Chaturvedula et al. (Emerging Markets Review 2015): CARs up to 7.49%; front-running before disclosure
- "What's hidden behind bulk deals" (Managerial Finance 2022): post-disclosure 5-7% drift within a week

**Direction:** Buy blocks > Sell blocks in informational content. Sell blocks often liquidity-driven (fund redemptions) → frequent reversal; buy blocks signal informed accumulation → continuation.

**Net of fees:** ~50-150 bps net edge T+1 to T+5 on buy side (peer-reviewed). Sell side ≈ noise after costs.

**Frequency:** Combined NSE+BSE ~30-80 bulk + 5-25 block per day. Annual ~10-20K events; after dedup + liquidity filter ≥ 2K-5K high-quality buy events. n ≫ 500.

**Data:** Free, daily CSV at `nseindia.com/report-detail/display-bulk-and-block-deals` and BSE equivalent. Backfillable for 2023-26.

### E. Circuit-band hit + recovery — STRONG

**Mechanic:** F&O stocks have ±10% Dynamic Price Range with **flexing** — when LTP touches band, NSE imposes 15-min cooling, then relaxes ±5% in pressure direction. Cash equity has fixed 2/5/10/20% bands per stock.

**Evidence:**
- Guo et al. (J. Int'l Fin. Markets 2023): Indian-equity natural experiment; bands delay but don't eliminate price discovery; volatility migrates to subsequent sessions
- Chen, Petukhov, Wang (MIT WP): magnet effect — price accelerates toward limit; post-limit-hit T+1 open continues the move (gap), then fades intraday
- Sehgal et al. (Pacific-Basin Finance Journal 2024): Indian momentum/reversal evidence — upper-circuit hits show next-day continuation in operator stocks; lower-circuit hits show reclaim in liquid F&O names

**Direction:** Upper-circuit hits → T+1 open continuation, then mid-session fade (operator-pump signature in low-float). Lower-circuit hits → reclaim long in liquid F&O names; continued sell in micro-caps.

**Frequency:** Within F&O 200 universe, ~3-8 circuit-flex events/day on average; ~1,200-2,000 annually. Concentrated mid-cap.

**Data:** Daily NSE `nse-cm-price-band-complete-list.csv` for per-stock bands. Circuit-hit detection rule from 5m bars: `(high == close == band_upper) AND volume drops to <30% of prior bar`. Backfillable.

### F. FII / DII flow asymmetry — PASS but overlay-only

**Mechanic:** Aggregate daily flow (FPI + DII cash segment net buy/sell) published 6-7 PM IST by NSE / NSDL. **Stock-level FII flow is NOT public daily** (only monthly via NSDL AUC).

**Evidence:**
- Acharya, Anshuman, Kumar (NYU Stern / IGC 2019): high-FII-flow vs low-FII-flow large-caps show **+2.14% Day-0 differential**, partially reverses within 1 week
- Springer J. Asset Mgmt 2024: FII sells more informative than FII buys
- ICRIER WP-109: positive autocorrelation in daily FII flow (multi-day liquidations)

**Direction:** FII-sell > FII-buy in information content. DII flows largely steady-state SIP-driven (mechanical mutual-fund inflows ~₹25,000 cr/month). Asymmetric: FII-sell + DII-buy (offsetting) is the high-information regime.

**Stock-level:** Effect concentrates on NIFTY-50 large-caps where FII free-float is 20-45%. Mid/small-caps see negligible direct effect.

**Limitation:** Aggregate-only data. **Cannot stock-pick within NIFTY-50 from FII flow alone.** Best used as a **regime overlay/filter** for other setups, not as a standalone entry trigger.

### G. Index rebalancing — CONDITIONAL

**Mechanic:** NIFTY 50 / Bank / Next 50 / 500 reconstituted semi-annually (March/September). Announcement-to-effective gap ~28 days. Indian passive AUM tracking Nifty indices ≈ ₹9.7 lakh cr (Oct 2025) → forced flow on inclusions/deletions.

**Evidence:**
- Marisetty 2025 (SSRN): 81 NIFTY events 2010-2024 — additions show announcement-day CAAR +1.10% (p=0.54, **not statistically significant**), reversing to **−1.17% by effective day**. Deletions show stronger initial negative reaction with partial recovery.
- Chhatwani (Publishing India): additions get permanent positive abnormal returns; reversal within 60 days
- Greenwood & Sammon NBER w30748: "disappearing index effect" globally — even in US S&P 500, effect declined from 7.4% (1990s) to ~1% (2010s)

**Direction:** Indian asymmetry — **deletions stronger than additions**, opposite of textbook front-running. Effective-day reversal favours short-deletions pre-effective rather than long-additions.

**Frequency:** ~50-80 events/yr across NIFTY 50 + Bank + Next 50 + 500. NIFTY 50 alone has only ~12-24 events / 3 yr — too thin. Use NIFTY 500 universe.

**Verdict:** Worth building only the deletion-side / effective-day-reversion variants. Smaller effect than H or E.

### D. F&O ban-list crossing — CONDITIONAL

**Mechanic:** SEBI MWPL rule. When stock-level FutEq OI > 95% of MWPL → enters ban list (only delta-reducing trades allowed; new positions blocked; margin escalation). Exits when OI < 80% of MWPL (asymmetric hysteresis).

**Evidence gap:** Despite extensive search across SEBI / SSRN / IUP / IIM working papers, **no peer-reviewed event study on Indian MWPL ban-entry/exit returns was located**. Closest analog (Switzer & Tu, SPY US) suggests removing limits improved efficiency, indirectly implying imposing limits degrades efficiency. Retail/broker commentary makes claims but is excluded by sub-9 spec criteria.

**Verdict:** Mechanic is real and Indian-specific, but **fails the "≥1 evidence source independent of retail communities" gate**. Could promote after in-house preliminary event study (build MWPL utilization series from F&O bhavcopy + free-float, run own T-5..T+5 returns analysis around ban events). If statistically significant, that becomes the independent evidence; otherwise reject decisively.

### C. MIS auto-square retry — REJECT as standalone

**Why mis_unwind_short failed (sub-7 evidence):** n=304, WR=9.2%, PF=0.355, Sharpe=-0.20, 89.4% losing days. Worst regime: trend_up — detector was systematically shorting strong-trend afternoons.

**Root causes:**
- Auto-cut is diffuse (15:10-15:25 broker-staggered) and pre-arbitraged
- Retail bypasses via CNC conversion before 3:20 PM
- Index-level signal collapses at stock level (Zerodha: "post-2011 negative-intraday pattern is clean at Nifty index level but much noisier and far less consistent at the stock level")
- Trend_up afternoons have institutional VWAP-target buying + Da/Baltussen "intraday momentum" hedging-demand drift to close

**Regulatory regime change:** NSE Circular CMTR63915 announces VWAP-based Closing Auction Session (CAS) using 15:00-15:15 reference window for F&O-eligible stocks (Mar 2026). Pre-2026 backtest baselines are regime-stale.

**Verdict:** Treat as a contextual filter / vetoer for existing afternoon longs (suppress gap_and_go after 14:30 in trending small-caps) rather than a generative short setup. Skip standalone version until ≥6 months of post-CAS data accumulates.

---

## Shortlist for §3.3 brief stage

In priority order, only setups with ALL 5 criteria passing on peer-reviewed evidence:

1. **H. Bulk-block buy-deal T+1 open continuation** — strongest evidence, largest sample, lowest data-engineering cost. **Recommended first brief.**
2. **E. Upper-circuit T+1 open fade** (F&O-eligible) — second-strongest evidence, distinct mechanic, uncorrelated signal source.

The two together form a complementary library: H is a **next-day continuation play** on informed buy-side flow; E is a **next-day fade play** on retail-FOMO-driven upper-circuit hits. Different participants on each side; uncorrelated.

**F. FII/DII flow** is admitted as a regime overlay (not a standalone setup). It can serve as a same-day or next-day filter for both H and E (e.g., suppress E's bullish-side fade on FII-buy days).

**D, G** are deferred. **C** is rejected.

## Sources (consolidated)

### Bulk / block deals
- Agarwalla & Pandey, NSE Research Initiative — https://nsearchives.nseindia.com/content/research/NSE_Proposal_216_Final_Paper.pdf
- Chaturvedula et al., Emerging Markets Review 2015 — https://www.sciencedirect.com/science/article/abs/pii/S1566014115000138
- "What's hidden behind bulk deals", Managerial Finance 2022 — https://www.emerald.com/insight/content/doi/10.1108/MF-08-2021-0374/full/html
- SEBI Master Circular Ch.1 Trading (Dec 2024) — https://www.sebi.gov.in/sebi_data/commondocs/dec-2024/RE_Chapter%201%20-%20Trading%20-%20NEW_p.pdf

### Circuit-band hits
- Guo et al., Journal of International Financial Markets 2023 — https://www.sciencedirect.com/science/article/abs/pii/S1386418123000381
- Chen, Petukhov, Wang, MIT WP — https://web.mit.edu/wangj/www/pap/ChenPetukhovWang18.pdf
- Sehgal et al., Pacific-Basin Finance Journal 2024 — https://www.sciencedirect.com/science/article/abs/pii/S0927538X23002640

### FII / DII flow
- Acharya, Anshuman, Kumar (NYU Stern 2019) — https://w4.stern.nyu.edu/sternfin/vacharya/public_html/pdfs/Foreign-Fund-Flows_May_2019.pdf
- Springer J. Asset Management 2024 — https://link.springer.com/article/10.1057/s41260-024-00387-8
- ICRIER WP-109 (Batra) — https://www.icrier.org/pdf/wp109.pdf

### Index rebalancing
- Marisetty 2025, SSRN — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5642110
- Greenwood & Sammon NBER w30748 — https://www.nber.org/system/files/working_papers/w30748/w30748.pdf

### MIS auto-square
- NSE CAS circular CMTR63915 (Mar 2026) — https://nsearchives.nseindia.com/content/circulars/CMTR63915.pdf
- Zerodha Auto Square-off Timings — https://support.zerodha.com/category/trading-and-markets/trading-faqs/market-sessions/articles/intraday-auto-square-off-timings
