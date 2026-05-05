# Sub-project #9 — Asymmetry Feasibility Round 2 (Indian Microstructure)

**Date:** 2026-05-05
**Predecessor:** specs/2026-05-01-sub-project-9-asymmetry-research-findings.md
**Lesson driving this:** tasks/lessons.md 2026-05-05 — "Run feasibility / pro-precedent check BEFORE drafting a §3.3 brief"
**Methodology:** Two sequential gates per candidate. Gate 1 (feasibility precedent at retail-intraday-MIS scale, ≥2 distinct Indian sources) → only YES advances → Gate 2 (≥1 peer-reviewed Indian-market study, 2018+).

---

## Executive summary

Six named candidates were evaluated against the new two-gate protocol.

| # | Candidate | Gate 1 (precedent) | Gate 2 (peer-reviewed ≥2018) | Status |
|---|---|:---:|:---:|---|
| 1 | NSE Closing Auction Session (CAS) post-launch | **PARTIAL** | n/a | DEFER (mechanic-only references; CAS not yet live in cash equity, August 2026 phased rollout) |
| 2 | Pre-open auction → 09:15 open mismatch | **YES** | **FAIL (recency)** | DEFERRED (algo precedent strong; only pre-2018 academic evidence located in 30-min budget) |
| 3 | F&O OI-velocity intraday afternoon directional buildup | **YES** | **PASS** | **SHORTLIST — recommended §3.3 brief target** |
| 4 | T+1 settlement next-day liquidation pressure | **NO** | n/a | RETIRE (no retail-intraday algo precedent; the only peer-reviewed Indian T+1 paper documents *improved* liquidity, opposite of the hypothesised stress) |
| 5 | Bulk / block deal intraday post-disclosure fade | **NO (structural)** | n/a | RETIRE (mechanic infeasible — bulk/block disclosure is post-close within 1hr; no intraday window exists post-disclosure) |
| 6 | Anchor-investor lock-in expiry day intraday | **PARTIAL** | n/a | DEFER (mechanic + retail-blog precedent; no algo-platform / pro retail-intraday precedent located) |

**Recommended §3.3 brief target: Candidate 3 — F&O OI-velocity intraday directional buildup (afternoon variant)** (single passing candidate; do not pad shortlist).

The asymmetry is well-precedented across uTrade Algos, StockEdge, Combiz, Zerodha Varsity (independent Indian retail-algo sources). Recent (2020+) Indian-market peer-reviewed literature confirms OI's informational role for return prediction in Indian equity / index futures. Sample size, data availability (NSE OI-spurts feed + bhavcopy + per-strike chain), and direction asymmetry (long-buildup vs short-buildup is a published, asymmetric signal — not a generic price pattern) all align with sub-9 §3.2 quality criteria.

---

## Per-candidate findings

### 1. NSE Closing Auction Session (CAS) — post-launch behaviour

**Mechanic.** NSE Circular CMTR63915 (March 18, 2026) introduces a 20-min closing auction session for F&O-eligible cash-equity stocks (Phase 1 from August 3, 2026). Reference price = VWAP of trades 15:00-15:15. Auction phases: 15:15-15:20 reference window display, 15:20-15:25 limit + market orders, 15:25-15:30 limit only, 15:30-15:35 matching. ±3% price band. Algo market orders explicitly permitted (relaxation of earlier rules).

The candidate hypothesis is that the 15:00-15:15 VWAP-reference window changes pro behaviour pre-CAS, and post-CAS launch will create new asymmetries (front-running of institutional CAS-VWAP-target trades, fade of stretched VWAP-pumping in the reference window, etc.).

**Gate 1 — PARTIAL.** No Indian retail-algo source operationalises this at our scale; retail-broker explainers (Angel One, Ventura, Bigul, ProStocks, niftytrader.in) describe the mechanic but no published algo strategy exists yet — CAS launches August 2026. Pre-CAS behavioural research in Indian equity by professional retail algos is essentially non-existent in 30-min search budget.

- mechanic-only references:
  - https://www.angelone.in/news/market-updates/nse-to-bring-closing-auction-session-what-does-it-mean-for-stock-prices-and-investors
  - https://nsearchives.nseindia.com/content/circulars/CMTR63915.pdf
  - https://www.venturasecurities.com/news/daily-market-updates/sebi-changes-closing-price-calculation-auction-based-closing-to-begin-from-august-2026-know-how-it-will-work/

**Verdict: DEFER.** Revisit ≥6 months after CAS goes live (Q1 2027). Pre-launch mechanic-only literature does not satisfy the §3.2 "evidence" criterion, and any pre-Aug-2026 backtest baseline is regime-stale by construction.

---

### 2. Pre-open auction (09:00-09:08) → 09:15 open mismatch

**Mechanic.** Pre-open call-auction collects orders 09:00-09:07/08 (random cutoff to prevent end-of-window manipulation), 09:08-09:12 matching/confirmation. Indicative equilibrium price (IEP) is disseminated continuously during the order-collection phase. The IEP frequently differs from the eventual 09:15 continuous-session open price; the asymmetry hypothesis is that this mismatch (or a stretched IEP move at high volume) is exploitable in the first 15 minutes.

**Gate 1 — YES.** Multiple distinct Indian retail-algo sources operationalise pre-open / first-15-min gap strategies:

1. **Elearnmarkets (StockEdge family)** — *Pre Open Market Strategy* — explicit retail-intraday playbook for pre-open IEP analysis as the first step in an intraday plan. https://www.elearnmarkets.com/school/units/intraday-trading/pre-open-market-strategy
2. **Elearnmarkets** — *Gap Up and Gap Down in Share Market Trading* — gap-reversal intraday strategy with stop = mid-opening-range, target = gap size. https://blog.elearnmarkets.com/gap-up-and-gap-down-in-share-market-trading/
3. **Zerodha Streak v4** — pre-built intraday strategies including VWAP-confirmation, opening-range and gap-based variants deployable on Streak. https://streak.zerodha.com/ , https://zerodha.com/z-connect/streak/introducing-streak-v4
4. **NSEcharts / preopenmarket.in** — published "100% working" pre-open gap-trade strategy targeting first-15-min behaviour. https://nsecharts.com/gap-up-and-gap-down-trading-how-to-trade-nse-openings/ , https://preopenmarket.in/nse-pre-open-market-strategy.html

Real-world precedent: **YES**, retail intraday MIS, broker = Zerodha + others, hold ≤ 1 day, retail capital scale. Multiple distinct sources, not retail-community claims (per sub-9 §2.5).

**Gate 2 — FAIL on recency criterion.** Peer-reviewed Indian-market evidence located in 30-min budget is all pre-2018:

- Sharma & Gupta, *Indian Journal of Finance* 2015 — pre-open auction improved price discovery efficiency (mixed sign for some specifications). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2690803
- Agarwalla, Jacob, Pandey, IIM-A 2015 — high-frequency-data study of call-auction impact on price discovery. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2188542
- Mallikarjunappa & Harish 2015 — BSE Sensex stocks, pre-open call-auction efficiency study, including documented serial correlation in returns indicating *price reversal* from call-auction price to continuous-session open. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2783244
- Cogent Economics & Finance 2014 — Pre-open call auction and price discovery: evidence from India. https://www.tandfonline.com/doi/full/10.1080/23322039.2014.944668

Mallikarjunappa & Harish 2015 explicitly documents reversal between IEP and continuous-session open — the exact asymmetry the candidate proposes — but it predates the 2018 cutoff specified in this round's Gate 2. No 2018+ Indian-equity peer-reviewed paper on the same effect was located in budget.

**Verdict: DEFERRED.** Real precedent at retail-intraday-MIS scale exists (Gate 1 strong); academic foundation locatable in 30-min search is the right effect but predates the recency cutoff. Path forward (per sub-9 §2.5 protocol): once an in-house event study on captured pre-open IEP data confirms the post-2018 effect persists, the candidate is promotable. Until then, we have setup-design risk that the pre-2018 effect has decayed (as Greenwood & Sammon NBER w30748 documents for index-effect generally — emerging-market microstructure asymmetries decay as algo participation grows).

---

### 3. F&O OI-velocity intraday afternoon directional buildup

**Mechanic.** Stock-specific futures + per-strike option OI changes intraday reveal where new contracts are being written. The asymmetry hypothesis: large directional OI builds in specific intraday windows (post-lunch 13:00-13:30 stockwise) signal informed positioning that becomes the next-day or end-of-day move. The mechanic is INDIAN-SPECIFIC because (a) NSE publishes OI-spurts and per-strike OI live during continuous trading — most exchanges don't, (b) stock-future OI is heavily retail-driven in India (per SEBI FY23 study, F&O retail = 88-93% of participants, 91-93% lose money), and (c) institutional positioning shows up identifiable against the retail-noise floor on intraday-OI-velocity readings rather than on closing-OI levels. Different from sub7's `mis_unwind_short` (which used MIS auto-square as the participant model — wrong); here participants are identified as informed-buyer vs retail-trapped-short via the OI direction × price direction quadrant (long buildup, short buildup, long unwind, short covering — a four-quadrant classification standard in Indian retail algo literature).

**Distinct from sub-7's REJECTED `mis_unwind_short`.** That setup hypothesised retail MIS auto-square ≈ structural net-sell pressure 15:10-15:25, traded short against trend. This candidate operates in an EARLIER window (~13:00-13:30), uses OI-velocity as the participant-identification signal (not the time-of-day proxy), and has a different direction logic (4-quadrant OI×price classification, not unconditional short).

**Gate 1 — YES.** Multiple distinct Indian retail/pro algo sources operationalise OI-velocity intraday strategies:

1. **uTrade Algos** — *Understanding OI-Based Strategies in Algo Trading* — full guide on OI-spurts, change-in-OI heatmaps, four-quadrant intraday classification. uTrade is a SEBI-registered algo platform serving retail intraday MIS users. https://www.utradealgos.com/blog/the-ultimate-guide-understanding-oi-based-strategies-in-algo-trading
2. **StockEdge** — *Future Long Position Scans / Future Short Position Scans* — daily-refreshed scans of historical OI-velocity-classified stocks with explicit intraday-tradable filtering. https://stockedge.com/ , https://blog.stockedge.com/how-to-find-stocks-with-stockedge-scans/
3. **Combiz Services** — *NIFTY Algo Trading Strategy Best Intraday Strategy for Nifty & Bank Nifty* — published 2-minute-chart algo strategy explicitly using OI direction + supertrend + RSI + volume filters for intraday Nifty/Bank Nifty trades. https://www.combiz.org/blogs/NIFTY-Algo-Trading-Strategy-Best-Intraday-Strategy-for-Nifty-Bank-Nifty
4. **Zerodha Varsity** — *Open Interest in Futures* — canonical Indian retail education on long-buildup / short-buildup / long-unwind / short-covering quadrant classification, used as the educational foundation for downstream algo platforms. https://zerodha.com/varsity/chapter/open-interest/ , https://zerodha.com/varsity/chapter/open-interest-2/
5. **Tradejini** — *How to Interpret Open Interest and Price Data: A Trader's Guide* — Indian broker-published guide to intraday OI velocity. https://www.tradejini.com/blogs/how-to-interpret-open-interest-and-price-data-a-traders-guide
6. **Jainam** — *What OI Spurts Indicate in the Stock Market* — operationalised intraday "where new money is entering" analysis. https://www.jainam.in/blog/oi-spurts-stock-market/
7. **NSE OI-Spurts feed** — official live intraday OI-change feed, published during trading hours (13:00 onwards is the institutional window per multiple sources). https://www.nseindia.com/market-data/oi-spurts

Real-world precedent at retail-intraday-MIS scale: **YES, multiple platform-published algos and broker-published methodologies**.

**Gate 2 — PASS.** Recent Indian-market peer-reviewed literature on OI's predictive power for returns:

- Avinash & Mallikarjunappa, *Asia-Pacific Journal of Management Research and Innovation* (Sage), 2020 — *Informational Role of Open Interest and Transaction Volume of Options: A Meta-Analytic Review* — meta-analysis confirming OI + transaction volume as predictors of future stock prices, future volatility, and announcement-day behaviour; explicitly states "open interest and transaction volume are more informative in an emerging market compared to a developed market" (Indian-market focus). Direction = informed-positioning detection (long-buildup → continuation; short-buildup → fade in liquid F&O names). https://journals.sagepub.com/doi/abs/10.1177/2319714520980662
- *Option Volume and Open Interest for Predicting Underlying Return — A Study of Index Option in Indian Stock Market*, Springer 2024 — peer-reviewed Indian-market evidence that index-option OI + volume predicts underlying returns. https://link.springer.com/chapter/10.1007/978-981-97-6242-2_6

Both are 2020 and 2024 respectively — pass the 2018+ recency cutoff.

**Direction.** Long-buildup (OI↑ + price↑) → next-day continuation (informed accumulation signal); short-buildup (OI↑ + price↓) → fade in liquid F&O names (retail-trapped-short signature, vulnerable to short squeeze). Asymmetric — the four quadrants do NOT have symmetric edge; literature plus retail-algo precedent both treat long-buildup-in-mid-cap-F&O as the most informative case. Unambiguous direction logic per quadrant satisfies §3.2 criterion 5.

**Sample size.** F&O eligible universe ≈ 200 stocks. NSE OI-spurts feed publishes ~30-60 names/day with material intraday OI changes. Conservative estimate: 5-15 high-conviction long-buildup events / day across the F&O 200, ~1,200-3,500/year. Independent of bulk/block deal events (different mechanism) and circuit hits (different mechanism) — uncorrelated to existing setups H and E from round 1.

**Infra fit.** Intraday MIS native (entry on intraday OI-velocity reading 13:00-13:30, exit ≤ 15:15). Data needs: NSE OI-spurts feed + per-symbol futures OI minute-snapshot. We already capture per-strike option OI for `expiry_pin_strike_reversal` infrastructure (sub-9 §1) — extending to per-symbol-future OI snapshots is incremental. Backfillable for 2023-26 from F&O bhavcopy + intraday OI tick captures (existing OCI capture). Cost low.

**Precedent caution noted in literature** (uTrade Algos, Tradejini): "Institutions tend to falsely entice retailers with early OI buildup and later shift direction" — i.e., raw-OI-buildup is well-known to retail and is partly faded by institutional reversal. The setup design will need a confirming filter (price + volume + 2nd-leg OI confirmation) rather than naive OI-spurts → entry. This is a setup-design constraint, not a gate failure — the asymmetry exists, the mechanic just needs care.

**Verdict: SHORTLIST. Recommended §3.3 brief target.**

---

### 4. T+1 settlement next-day liquidation pressure

**Mechanic.** India transitioned to T+1 settlement Jan 2023. Hypothesis: stocks bought on broker-margin face next-day liquidation pressure (margin calls or auto-cuts on T+1 if collateral / payment fails), creating predictable opening-window selling on T+1.

**Gate 1 — NO.** No Indian retail-algo source located that operationalises a "T+1 next-day-liquidation-pressure short" setup. Retail/broker content about T+1 (Bajaj Broking, JM Financial, Razorpay, Bajaj Finserv, Cleartax) describes the mechanic from the *capital-rotation-benefit* angle (faster fund release, lower margin requirements) — not from the *liquidation-pressure* angle the candidate hypothesis assumes. None describe an algo strategy at retail-intraday-MIS scale targeting this asymmetry.

Searched: zerodha.com, plindia.com, utradealgos.com, stockedge.com, tickertape.in, business-standard markets section, ion-group, deutsche-bank flow.

The peer-reviewed Indian-market study on T+1 — Bhanu, Nath, Patnaik, *Economics Letters* / SSRN 2023 "Unlocking liquidity through shortened settlement cycle: Empirical evidence from India" — documents the *opposite* of the hypothesised stress: T+1 *improved* market liquidity, narrowed quoted spreads, and the improvement was *greater for illiquid stocks*. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4745303 , https://www.sciencedirect.com/science/article/abs/pii/S0165176524002192

**Verdict: RETIRE.** The hypothesised asymmetry does not survive its own background literature. Gate 2 is moot because Gate 1 fails and the only peer-reviewed Indian study points opposite.

---

### 5. Bulk / block deal intraday post-disclosure fade

**Mechanic (proposed).** Retire from the round-1 `bulk_block_buy_continuation` candidate (which entered at T+1 09:15 and mean-reverted intraday). The new angle proposed: same disclosure event, INTRADAY post-disclosure fade.

**Gate 1 — NO (structural infeasibility).** SEBI Master Circular Ch.1 Trading + NSE bulk/block disclosure rules: bulk-deal disclosure is *end-of-day* (within 1 hour of close), block-deal disclosure is *immediately after the special-window match* (block windows are 08:45-09:00 and 14:05-14:20). The bulk-deal event therefore has **no intraday post-disclosure window** on the trade day — disclosure occurs after 15:30 close.

Block-deal post-disclosure intraday (post-14:20 special window → trades through 15:30 close) is theoretically a 70-min intraday window, but:
- Block-deal volumes are tiny vs daily ADV (the 14:05-14:20 window matches once per stock per day at a single negotiated price)
- The block-deal disclosure is already in the historical Agarwalla & Pandey (IIM-A 2010) and Chaturvedula et al. (Emerging Markets Review 2015) literature — and round-1's candidate H already covers buy-block continuation as the strongest post-disclosure signal

No Indian retail/pro algo source located that operationalises a separate "intraday post-disclosure fade" strategy distinct from H.

**Verdict: RETIRE.** Mechanic infeasible for bulk deals (no intraday window); for block deals, the buy-side post-disclosure signal is already covered by candidate H from round 1. No additional asymmetry to exploit.

---

### 6. Anchor-investor lock-in expiry day intraday

**Mechanic.** SEBI rule: anchor-investor IPO allocations are locked for 30 days (50%) and 90 days (50%) from listing. On lock-in expiry day, anchor selling can pressure the stock — observed e.g. Waaree Energies 9% intraday drop on April 2025 lock-in expiry. Hypothesis: short the stock intraday on anchor-lock-in expiry day (or fade post-drop reclaim).

**Gate 1 — PARTIAL.** Several Indian retail-finance blogs and IPO-tracking sites document the mechanic and price impact:

- Chittorgarh — *Lock-in Period Details for Anchor Investors in IPOs* — published lock-in-end-date calendar for all listed IPOs. https://www.chittorgarh.com/report/anchor-investor-lock-in-end-dates/156/all/
- StockGro — *What is the lock-in period for IPOs* — retail-investor-facing explainer. https://www.stockgro.club/blogs/ipo/ipo-lock-in-period-2/
- Eqwires — anchor-lock-in expiry to unlock ₹2.36 trn worth of shares (April-May 2025). https://www.eqwires.com/tutorials/anchor-lock-in-expiry-to-unlock-%E2%82%B92-36-trn-worth-shares-in-april-may-2025/
- IPO Central — *20 IPOs Anchor Lock-in Expiry In March, Shares Worth INR 6,766 Cr Unlock*. https://ipocentral.in/ipos-with-anchor-lock-in-expiry-in-march/
- Tejimandi — *Anchor Lock-ins Ending: Should You Worry or Rejoice?* https://tejimandi.com/blog/feature-articles/anchor-lock-ins-ending-should-you-worry-or-rejoice
- Bajaj Broking, mStock, Choice — retail explainers on lock-in expiry mechanics.

However: searches across uTrade Algos, Streak, Tradetron, AlgoBulls, AlgoTest, Combiz, PL Capital, StockEdge, Zerodha Varsity for an *operationalised algo strategy* targeting anchor-lock-in expiry day at retail-intraday-MIS scale returned **zero matches**. The mechanic is well-known and retail-blog-discussed; no broker-published or algo-platform-published intraday strategy was located.

This is the same failure mode as round-1 candidate G (NIFTY 500 deletion short) per the 2026-05-05 lesson: known mechanic, retail-blog-level commentary, no pro-retail-algo precedent at our scale.

**Verdict: DEFER.** Mechanic is real and Indian-specific. Promote only after either (a) ≥2 retail-algo-platform-published strategies appear (revisit annually), or (b) in-house event study on captured 2023-26 lock-in-expiry calendar produces a statistically significant intraday-short edge — which would make the in-house study itself the §3.2 evidence, similar to candidate D (F&O ban-list) treatment in round 1.

---

## Shortlist for §3.3 brief stage

Single passing candidate. Per the 2026-05-01 lesson "Stop offering 4-option lists at the end of every response," this is one item, not three.

1. **Candidate 3 — F&O OI-velocity intraday afternoon directional buildup**
   - Gate 1 = YES (7 distinct Indian sources: uTrade Algos, StockEdge, Combiz, Zerodha Varsity, Tradejini, Jainam, NSE OI-spurts feed)
   - Gate 2 = PASS (Avinash & Mallikarjunappa 2020 Sage; Springer 2024 Indian index-option OI study)
   - Direction asymmetric: long-buildup → continuation, short-buildup → fade-in-F&O-200, four-quadrant classification published
   - Sample n ≈ 1,200-3,500 events/year across F&O 200
   - Infra fit: intraday MIS native; data needs incremental on existing per-strike OI capture
   - Distinct mechanic from `mis_unwind_short` (rejected) and `expiry_pin_strike_reversal` (deferred): same data plumbing, different participant model and time window
   - Setup-design caution: literature explicitly notes institutional-fakeout risk — the §3.3 brief must specify the confirming-filter design (price + volume + 2nd-leg OI confirmation) to avoid trading on raw spurts

**Round-1 shortlist** (from specs/2026-05-01-sub-project-9-asymmetry-research-findings.md) of bulk-block-buy continuation (H) and circuit T+1 continuation (E) remains the dominant body of work; this round-2 shortlist adds Candidate 3 as an uncorrelated third leg using a different data source (F&O OI vs cash-equity disclosure / cash-equity circuit-band).

---

## Sources (consolidated)

### NSE CAS (mechanic-only references)
- https://www.angelone.in/news/market-updates/nse-to-bring-closing-auction-session-what-does-it-mean-for-stock-prices-and-investors
- https://nsearchives.nseindia.com/content/circulars/CMTR63915.pdf
- https://www.venturasecurities.com/news/daily-market-updates/sebi-changes-closing-price-calculation-auction-based-closing-to-begin-from-august-2026-know-how-it-will-work/
- https://www.niftytrader.in/markets/a-new-era-for-market-closing-prices/
- https://www.prostocks.com/blog/337-closing-auction-session-in-the-equity-segment-what-clients-should-know.html
- https://bigul.co/blog/nse-implements-comprehensive-framework-to-govern-algo-trading-practices-across-all-participants

### Pre-open auction (Gate 1 YES, Gate 2 FAIL on recency)
Retail-algo precedent:
- https://www.elearnmarkets.com/school/units/intraday-trading/pre-open-market-strategy
- https://blog.elearnmarkets.com/gap-up-and-gap-down-in-share-market-trading/
- https://streak.zerodha.com/
- https://zerodha.com/z-connect/streak/introducing-streak-v4
- https://nsecharts.com/gap-up-and-gap-down-trading-how-to-trade-nse-openings/
- https://preopenmarket.in/nse-pre-open-market-strategy.html
- https://zerodha.com/z-connect/general/pre-marketpost-marketafter-market-orders

Peer-reviewed (pre-2018, fail recency):
- Sharma & Gupta, Indian Journal of Finance 2015 — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2690803
- Agarwalla, Jacob, Pandey, IIM-A 2015 — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2188542
- Mallikarjunappa & Harish 2015 — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2783244
- Cogent Economics & Finance 2014 — https://www.tandfonline.com/doi/full/10.1080/23322039.2014.944668

### F&O OI velocity intraday (Gate 1 YES, Gate 2 PASS) — SHORTLISTED
Retail-algo / pro precedent:
- uTrade Algos — https://www.utradealgos.com/blog/the-ultimate-guide-understanding-oi-based-strategies-in-algo-trading
- StockEdge — https://stockedge.com/ , https://blog.stockedge.com/how-to-find-stocks-with-stockedge-scans/
- Combiz Services — https://www.combiz.org/blogs/NIFTY-Algo-Trading-Strategy-Best-Intraday-Strategy-for-Nifty-Bank-Nifty
- Zerodha Varsity (foundational) — https://zerodha.com/varsity/chapter/open-interest/ , https://zerodha.com/varsity/chapter/open-interest-2/
- Tradejini — https://www.tradejini.com/blogs/how-to-interpret-open-interest-and-price-data-a-traders-guide
- Jainam — https://www.jainam.in/blog/oi-spurts-stock-market/
- NSE OI-Spurts feed — https://www.nseindia.com/market-data/oi-spurts

Peer-reviewed (2018+):
- Avinash & Mallikarjunappa, Sage 2020 (meta-analytic, India-focused) — https://journals.sagepub.com/doi/abs/10.1177/2319714520980662
- Springer 2024, Indian index-option OI/volume → underlying return — https://link.springer.com/chapter/10.1007/978-981-97-6242-2_6

### T+1 settlement (Gate 1 NO)
Background (mechanic + opposite-direction evidence):
- Bhanu, Nath, Patnaik, SSRN 2023 / Economics Letters 2024 — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4745303 , https://www.sciencedirect.com/science/article/abs/pii/S0165176524002192
- Zerodha Varsity — https://zerodha.com/varsity/chapter/clearing-and-settlement-process/
- Business Standard — https://www.business-standard.com/markets/news/stock-market-settlement-cycles-india-s-shift-from-t-2-to-t-1-explained-123080100560_1.html

### Bulk / block deal intraday (Gate 1 NO — structural)
- SEBI Master Circular Ch.1 Trading — https://www.sebi.gov.in/sebi_data/commondocs/dec-2024/RE_Chapter%201%20-%20Trading%20-%20NEW_p.pdf
- Round-1 candidate H already covers buy-side continuation: see specs/2026-05-01-sub-project-9-asymmetry-research-findings.md

### Anchor lock-in expiry intraday (Gate 1 PARTIAL)
- Chittorgarh anchor-lock-in calendar — https://www.chittorgarh.com/report/anchor-investor-lock-in-end-dates/156/all/
- StockGro — https://www.stockgro.club/blogs/ipo/ipo-lock-in-period-2/
- Eqwires — https://www.eqwires.com/tutorials/anchor-lock-in-expiry-to-unlock-%E2%82%B92-36-trn-worth-shares-in-april-may-2025/
- IPO Central — https://ipocentral.in/ipos-with-anchor-lock-in-expiry-in-march/
- Tejimandi — https://tejimandi.com/blog/feature-articles/anchor-lock-ins-ending-should-you-worry-or-rejoice
- Academic context (peer-reviewed but pre-2018): Bubna & Prabhala (ISB) — https://w4.stern.nyu.edu/finance/docs/WP/2014/AnchorIPOs_BubnaPrabhala.pdf ; Mahajan & Singh 2011 Sage — https://journals.sagepub.com/doi/abs/10.1177/0258042X1103600203 ; ScienceDirect 2017 — https://www.sciencedirect.com/science/article/pii/S0970389617305608

---

## Methodology notes (round 2 vs round 1)

Round 1 (2026-05-01) ran a single-gate process: peer-reviewed evidence ≥1 source. The 2026-05-05 lesson added a MANDATORY upfront feasibility-precedent gate — peer-reviewed evidence is necessary but not sufficient. Round 2 applies both gates in sequence to the candidates listed in the user prompt.

The round-2 gate ordering (precedent first, peer-reviewed second) is by design: a feasibility-precedent failure terminates the search before academic-evidence search is even started, saving budget. This round had two such early terminations (T+1 settlement; bulk/block intraday).

Round 1's three passing candidates (H, E, F-as-overlay) are NOT re-evaluated under the new two-gate rule in this spec. They were validated under the original single-gate process and should be re-checked separately if the user wants Round 1 backfilled for the new gate. Spot-check of the Round 1 list against precedent literature: H (bulk-block buy-deal continuation) → uTrade / Tickertape / StockEdge all publish bulk-deal screeners; precedent OK. E (circuit T+1 continuation) → Indian retail blogs document circuit-hit follow-through; precedent OK. F (FII/DII overlay) → admitted as overlay, not a standalone setup; not subject to the standalone-precedent gate.
