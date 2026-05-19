# Research: FII/DII Flow Asymmetry (sub-9 §4 candidate F)

**Branch:** `research/post-sebi-edge-setups`
**Date:** 2026-05-14
**Status:** RETIRE — mechanism pre-falsified by academic literature; aggregate-signal → per-stock-trade structure fails sub-9 round-3 Gate A precedent test.
**Predecessor:** `specs/2026-05-01-sub-project-9-microstructure-first-redesign.md` §4 candidate F (never previously attempted)

---

## Mechanism (one sentence)

When FIIs print a heavy net-sell day in the NSE cash segment (EOD aggregate published ~17:30 IST), the implied institutional positioning de-risking should bleed into the next session via gap-down opens and continued intraday weakness — the T+1 fade trades SHORT on NIFTY-50 (FII-sensitive) constituents on heavy-FII-sell days.

## Direction

**SHORT-only** on T+1, restricted to NIFTY-50 cash-segment constituents (FIIs barely hold mid/small caps). Long side rejected per sub-9 prior (70-93% loss rate on long-bias intraday setups, SEBI FY23 study).

## Regulatory dependencies (per `data/sebi_calendar/` taxonomy)

- **regulatory_sensitivity:** `rule_orthogonal` — no SEBI rule creates this asymmetry; it depends on participant behaviour, so the setup is robust against the regulatory regime breaks that killed `delivery_pct_anomaly_short`.
- **depends_on:** `["MIS_leverage", "STT_drag"]` — both load-bearing for break-even calculation. Post-2026-04-01 STT hike (cash equity intraday MIS 0.025% sell-leg unchanged) is the relevant fee floor.

## Data accessibility audit

### Aggregate FII/DII daily flow — AVAILABLE

| Source | URL | Format | Cost |
|---|---|---|---|
| **NSE official** | `https://www.nseindia.com/reports/fii-dii` and historical archive `https://www.nseindia.com/static/all-reports/historical-equities-fii-fpi-dii-trading-activity` | CSV/XLSX downloads, daily aggregate (buy_value / sell_value / net_value, segregated by FII vs DII, by cash + F&O) | Free, public |
| **NSDL FPI Reports** (gold standard for FPI breakdown) | `https://www.fpi.nsdl.co.in/Reports/Latest.aspx` | XLSX, FPI category-wise daily | Free |
| Third-party aggregators (cross-reference) | NiftyTrader / 5paisa / Sensibull / EquityPandit | HTML scrape | Free |
| Community wrappers | `dhruvitdiyora/nse-tools`, `hi-imcodeman/stock-nse-india` (GitHub) | Python | Free |

**No on-disk data today**, but a ~1-day scraper to backfill 2022-2026 from NSE archives is straightforward (same pattern as `tools/delivery_pct/fetch_delivery.py`). Data is NOT a blocker.

### Per-stock FII attribution — NOT AVAILABLE

- NSE does NOT publish per-stock FII flow daily. Only the daily AGGREGATE buy/sell value across the universe is published.
- SEBI publishes bulk-deal data per-stock (already accessible) but bulk deals capture only the ~10-30 largest trades/day, not aggregate institutional positioning per name.
- Quarterly shareholding patterns publish FII holding per stock, but only at quarter-end (90-day lag, useless for T+1 trades).

**Implication:** any setup built on this data is necessarily AGGREGATE_signal → PER-STOCK trade. We pick which stocks to short on T+1 by a static proxy of "FII-sensitivity" (NIFTY-50 membership, free-float FII holding from quarterly disclosure), NOT by which stocks the FIIs actually sold yesterday.

## Why this candidate fails sub-9 round-3 gates

The round-3 feasibility framework (specs/2026-05-06-sub-project-9-asymmetry-feasibility-round-3.md) requires Gate A (Indian retail-algo precedent) + Gate B (data on disk) + Gate C (time-of-day fits 11:00-15:15 gap). Apply to FII/DII flow:

### Gate A — FAIL

1. **Academic literature already tested the T+1 mechanism and rejected it.**
   - "Short-term (1-day) impact is variable and not reliably predictive" — multiple Indian-data peer-reviewed studies (ICRIER WP109; ScienceDirect S1925209924003978; PMC9145119).
   - Granger causality on FII-net-flow → NIFTY return tests as monthly/lag-N: bi-directional or non-significant at lag-1 in four sub-periods studied; correlation strongest at 10-20 day windows, NOT at T+1.
   - This is the same failure mode that retired round-3 candidate 5 (EOD-PCR → T+1 intraday) — aggregate EOD signal, weak T+1 link.
2. **No Indian retail-algo platform publishes an operationalised T+1 FII-fade strategy.** Numerous explainer blogs ("FII selling = bearish") but zero published Streak / Stratzy / Algotest / Sensibull strategy script that trades a daily FII-net-sell threshold → next-session SHORT. By the round-3 evidence bar (≥2 distinct algo-platform-published strategies), this fails.
3. **Sentiment & commentary precedent only** — Samco / Outlook Business pieces discuss FII flow regime in qualitative terms ("DIIs catching the bid", "FII exhaustion"); none operationalise a tradeable trigger.

### Gate B — PASS (data scrape-able)

See accessibility audit above. ~1 day to ingest.

### Gate C — PARTIAL

T+1 fade fires at 09:15 (gap-down open) and continues throughout the session. Directly overlaps with `gap_fade_short` (09:15-09:30) — and the FII-sell-day correlates with gap-down probability, which is precisely what `gap_fade_short` already captures more directly via its own price-action signal. **The FII-flow signal is mechanically downstream of the gap signal.** Adding an FII-flow gate to `gap_fade_short` is a refinement (not a new setup), and gap-fade already uses price gap as the trigger, which absorbs the FII-flow information without needing to scrape it.

## Cells (would have been pre-registered if proceeding)

Not registered. The candidate is retired before cell-lock.

If revisited in future (only after a published Indian-algo-platform precedent emerges):
- `cap_segment`: NIFTY-50 only (FIIs largest holders)
- `signal`: FII net cash flow on day T <= -3000 cr (95th-percentile sell day)
- `entry`: 09:20 (after gap settles, before 09:30 gap-fade window)
- `direction`: SHORT
- `target/stop`: same as `gap_fade_short` cells; difference is universe filter (NIFTY-50 vs full F&O 200)

## Falsifiers (would have been pre-registered)

Same template as `mwpl_recalc_forced_rebalance_fade`:
- Discovery (2024-09 .. 2025-09): n ≥ 100, NET PF ≥ 1.30, WR ≥ 38%, per-month stability (no month carries > 40% of PF).
- Auto-retire if: per-stock attribution unavailable (✓ confirmed), Discovery PF < 1.20, signal subsumed by `gap_fade_short` (long-short overlap > 50% of fires).

## Verdict

**RETIRE — DATA AVAILABLE, MECHANISM PRE-FALSIFIED.**

Specifically:
1. The data exists publicly (NSE archive + NSDL) and is scrape-able in ~1 engineering day. **Data is NOT the blocker.**
2. Peer-reviewed Indian-market studies have already tested the T+1 lag-1 hypothesis and found it non-significant; the predictive horizon for FII flow is 10-20 days, which is a position-trading horizon, not an intraday MIS setup. This is a strong prior against running our own sanity (we would be re-doing a published null result).
3. Aggregate-signal → per-stock-trade has no Indian retail-algo platform precedent that survives the round-3 Gate A bar (≥2 published platforms operationalising the mechanism). Same failure as round-3 candidate 5 (EOD-PCR).
4. Gate C overlap: the FII-flow signal is mechanically downstream of the gap-down signal that `gap_fade_short` already trades. Any incremental edge from FII flow is captured by the existing gap-based setup without ingesting a new data source.

**Sanity script NOT built. Discovery run NOT attempted.** Per the prompt's "data prerequisite required → STOP" rule and the round-3 lessons (2026-05-05 "feasibility/pro-precedent check BEFORE any §3.3 brief"; 2026-05-01 "stop padding shortlists"), this candidate is retired at the brief stage.

## Possible future re-investigation paths (not active)

These are flagged for an entirely separate research line, NOT this branch:

1. **Multi-day FII regime swing trade (NOT intraday MIS).** Academic literature supports the 10-20 day horizon. If we ever build a swing-trade leg (CNC, multi-day hold), the FII-3-day-or-5-day net-sell signal could anchor a NIFTY-50 short. Out of scope for an intraday-MIS-only system.
2. **FII flow as a regime gate, not a signal.** Add daily FII net flow as a feature to `services.regime_break_detector` (or a successor regime classifier) to scale-down existing short setups when FIIs are buying heavily and scale-up when FIIs are selling. This is a portfolio-level overlay, not a standalone setup; would also require a fresh sub-project to design properly.
3. **NSDL FPI sectoral breakdown (monthly).** NSDL publishes FPI sectoral allocation monthly. Sector-rotation candidate (round-3 candidate 4, currently DEFER) could be enhanced by this for a longer-horizon signal — but again, monthly granularity, not intraday.

## Files

- `specs/2026-05-14-research-fii_dii_flow.md` — this research brief (only output)

NOT created (deliberately):
- `tools/sub9_research/sanity_fii_dii_followthrough.py` — would re-test a published null result against an aggregate-signal-per-stock-trade structure that has no platform precedent. Skipped per round-3 lessons.

## Sources

### Data accessibility
- NSE FII/DII reports: https://www.nseindia.com/reports/fii-dii
- NSE historical archive: https://www.nseindia.com/static/all-reports/historical-equities-fii-fpi-dii-trading-activity
- NSDL FPI reports: https://www.fpi.nsdl.co.in/Reports/Latest.aspx
- GitHub wrappers: https://github.com/dhruvitdiyora/nse-tools , https://github.com/hi-imcodeman/stock-nse-india

### Academic (mechanism pre-falsification)
- ICRIER WP109 — FII Trading Strategies: https://www.icrier.org/pdf/wp109.pdf
- ScienceDirect — FII Investments / Volatility (Indian Experience): https://www.sciencedirect.com/science/article/pii/S1925209924003978
- PMC — Trading Behaviour of FIIs (Indian Stock Markets): https://pmc.ncbi.nlm.nih.gov/articles/PMC9145119/
- Upstox learning — FII/DII Impact on NIFTY-50: https://upstox.com/learning-center/share-market/impact-of-fii-and-dii-flows-on-nifty-50/article-1602/

### Cross-references (within this repo)
- `specs/2026-05-01-sub-project-9-microstructure-first-redesign.md` §4 candidate F
- `specs/2026-05-06-sub-project-9-asymmetry-feasibility-round-3.md` (round-3 gate framework)
- `specs/2026-05-14-research-post-sebi-edges.md` (parent research roadmap)
- Round-3 candidate 5 (EOD-PCR → T+1 intraday) — same retire-mode pattern
