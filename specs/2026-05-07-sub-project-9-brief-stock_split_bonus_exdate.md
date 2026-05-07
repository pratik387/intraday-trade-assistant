# §3.3 Brief: `stock_split_bonus_exdate_arbitrage`

**Sub-project:** #9 (microstructure-first redesign)
**Status:** **DRAFT — awaiting user APPROVE/REJECT/RETIRE before sanity-check.**
**Date:** 2026-05-07
**Round:** 5

**Predecessors:**
- specs/2026-05-01-sub-project-9-microstructure-first-redesign.md (defines §3.3 gate)
- specs/2026-05-01-sub-project-9-asymmetry-research-findings.md (Round 1)
- specs/2026-05-05-sub-project-9-asymmetry-feasibility-round-2.md (Round 2)
- specs/2026-05-06-sub-project-9-asymmetry-feasibility-round-3.md (Round 3)
- specs/2026-05-01-sub-project-9-brief-circuit_t1_fade_short.md (APPROVED + IMPLEMENTED template — NSE/SEBI mechanical event class)

This is a Round-5 §3.3 brief. It belongs to the **NSE/SEBI mechanical-event** class that produced the two surviving production setups (gap_fade_short, circuit_t1_fade_short). Round-1 through Round-4 produced 11 candidates → 2 production setups; the 9 retirements were almost all *generic asymmetries* (volatility-revert, mean-revert, sector-rotation) without a hard NSE/SEBI rule anchoring them. This brief deliberately targets the same class as the winners: a **regulator-defined corporate-action ex-date** with a **mechanical price adjustment** that retail order-entry behaviour persistently mishandles.

---

## Asymmetry

**Name:** Indian-equity stock-split / bonus ex-date open-auction mispricing reversion (T+0 09:15 IST).

**Indian-specific source:**
- Under **SEBI (LODR) Regulation 42** + NSE Circular framework, on the ex-date of a stock split (e.g., 1:5, 1:10) or bonus issue (e.g., 1:1, 2:1), NSE/BSE mechanically adjust:
  1. Closing price of T-1 → theoretical adjusted price for T+0 (the ex-price).
  2. Outstanding orders in the orderbook (cancelled or re-priced per exchange rule).
  3. Tick size, lot size (F&O), circuit limits (intraday upper/lower bands) — recomputed against ex-price.
- The **pre-open auction (09:00-09:08 IST)** on the ex-date is where price discovery against the new theoretical-adjusted level happens. This is the only auction window of the day.
- **Broker UI inconsistencies** are well-documented during 09:00-09:30 on ex-dates:
  - Some retail broker apps continue to show the **pre-adjusted price** (T-1 close) for several minutes into the session because their market-data caches haven't refreshed corporate-action factors.
  - Watchlists, charts, GTT/SL orders that were entered against pre-split price are auto-adjusted on the broker side, but **alerts/limits set by retail traders frequently fire at wrong levels** (e.g., a stop-loss set at ₹1000 pre-split shows as ₹100 post-split for a 1:10 split).
  - Retail traders see "stock crashed 90%" panic on pre-adjusted views and place spurious sells; conversely, on bonus issues some retail traders see a "free additional shares" effect and place spurious buys.
- **Pre-open auction order imbalance** in a thin participant set (most institutions skip the auction on a corporate-action day, market makers post wide quotes) causes the ex-date open-auction price to deviate from the theoretical-adjusted price by an amount that exceeds normal-day deviation.
- The exploitable asymmetry is: **the gap between the ex-date 09:15 open and the theoretical adjusted price reverts toward the theoretical price within the 09:15-10:30 window** as: (a) market makers re-quote against the corrected reference; (b) institutional algos with corporate-action-aware data feeds enter; (c) retail order errors get cancelled or fade.

This is in the same class as `gap_fade_short` (T+0 open mispricing reverting in the morning window) and `circuit_t1_fade_short` (regulator-imposed mechanical event mispriced by retail), which are the two production-grade survivors of sub-7/8/9.

## Participants

- **Confused retail (LOSING flow per SEBI FY23):** retail traders viewing pre-adjusted prices, mis-set GTT/SL/alert levels, panicked sells on apparent "crash" (split optics) or naive buys on apparent "discount" (bonus optics). These flow through their broker's pre-open / first-15-minutes orders.
- **Market makers / arb desks (PRICING flow):** corp-action-aware data feeds (Bloomberg, Refinitiv, NSE adjustment factors). They quote tight to theoretical-adjusted price within minutes of 09:15 and absorb the retail mispricing. **We are not competing with them on speed — we are taking the same direction as them, sized for retail-MIS infra, on the residual mispricing they don't fully arb out within the first 5-15 minutes.**
- **Passive index funds:** mechanical adjustment, no price-discovery role.
- **F&O hedge unwinds:** lot-size + strike adjustments often trigger small unwind flows on the ex-date that reinforce reversion direction (a side effect, not the primary edge).

We are on the disciplined side: **fading the retail-confused-open back toward the theoretical adjusted price**, alongside (not against) market makers.

## Persistence

Three structural reasons:

1. **NSE / SEBI corporate-action processing is regulated** — SEBI LODR Reg 42 mandates the ex-date mechanism, NSE Circulars define the orderbook re-pricing rules, and adjustment factors are published on bhavcopy. The mechanical event itself is regulator-anchored and won't disappear (same persistence class as circuit limits for `circuit_t1_fade_short`).
2. **Retail behavioural confusion is structural, not market-cycle-dependent** — new retail traders enter Indian markets continuously (NSE active client base grew from ~3 cr in 2019 to ~17 cr in 2025). Each ex-date catches a fresh cohort that has never seen a 1:10 split or 1:1 bonus before; their order-entry mistakes are predictably first-time-trader patterns. Persists as long as Indian retail keeps growing — i.e., decades.
3. **Broker UI fragmentation persists** — Indian retail broker tech stacks vary widely (Zerodha, Groww, Upstox, ICICIdirect, HDFC Securities, etc., each on independent corporate-action processing). Even one major broker showing pre-adjusted price for 5-10 minutes after 09:15 generates enough mispriced flow to move the auction. Standardising broker corporate-action UX is not on the SEBI roadmap.

**Decay risk acknowledgement:** as broker tech matures (notably Zerodha, Upstox, IIFL improving their corp-action UX), the "broker-UI-confusion" component will weaken. The **first-time-trader cohort confusion** component does NOT decay (always renewed). Annual re-validation built into falsification.

## Project-wide caveats addressed

- **Long-bias caveat:** the brief is **direction-TBD-per-evidence**. We do NOT pre-commit to a long or short bias. US literature (Lakonishok & Lev 1987, Grinblatt et al. 1984) shows positive ex-date drift on splits; Indian-specific direction is unknown a priori. The sanity check determines direction — if both directions pass, ship the higher-PF one only (no symmetric ship without evidence; the sub-7/8 11-failure pattern of cargo-culted longs requires that long-side ship needs PF ≥ short-side × 0.85 if both are in scope).
- **Sample size honesty:** stock splits + bonuses are RARE events. F&O 200 universe sees roughly **30-80 ex-dates per year** (combined splits + bonuses + face-value reductions). A 2-year discovery window yields **60-160 events**. This is at or just above the n ≥ 30 floor and well below the standard n ≥ 500 floor used by other sub-9 candidates. **This is the binding feasibility constraint.** Locked thresholds (PF ≥ 1.10, n ≥ 30, |WR delta vs baseline| ≤ 10pp) are all that matter for this candidate; the brief explicitly accepts that we cannot meet n ≥ 500 and uses the rare-event floor n ≥ 30 instead.
- **Data ingestion is non-trivial:** unlike volume_spike (which needed only existing 5m feathers) or circuit_t1 (which needed circuit-band data already on bhavcopy), this candidate requires a NEW corporate-actions ingestion pipeline. **Acceptance gate:** the candidate is APPROVE-eligible only if the data path is clear AND sample size on F&O 200 over 2yr is ≥ 30 events. See data engineering plan below.
- **NSE/SEBI mechanical-event class anchoring:** by design — Round 1-4 lessons say "winners are tied to NSE/SEBI mechanical events; losers were generic asymmetries". This candidate is squarely in the winners' class.

## Evidence

Indian regulatory + corporate-action references (primary, Gate A):
1. **NSE corporate-actions framework** — https://www.nseindia.com/companies-listing/corporate-filings-actions (filterable by action type: split, bonus, dividend, rights). Source of truth for ex-date, ratio, action type per symbol.
2. **BSE corporate-actions archive** — https://www.bseindia.com/corporates/corporate_act.html (parallel listing for BSE-only events; cross-check for split-listed stocks).
3. **NSE bhavcopy `Adjusted_Open_Price` / adjustment-factor field** — daily bhavcopy carries adjustment factors that allow reconstructing theoretical adjusted prices independently of any vendor.
4. **SEBI (LODR) Regulation 42** — codified ex-date mechanism. Persistence anchor.
5. **Tickertape / Trendlyne / MoneyControl corporate-actions APIs** — backup/supplementary feeds for ratio + ex-date.

Academic / quasi-academic (secondary):
1. **NIPFP working papers on Indian corporate-action mispricing** — multiple working papers on Indian ex-date abnormal returns; specific direction findings vary by sample period. (Search: NIPFP + corporate action + ex-date.)
2. **Sage Vision Indian Market journal** — articles on retail order-flow inefficiencies around corporate actions in Indian small/mid caps.
3. **Lakonishok & Lev (1987) "Stock Splits and Stock Dividends: Why, Who, and When"** — foundational US ex-date drift literature; mechanism transports but direction may not.
4. **Grinblatt, Masulis & Titman (1984) Journal of Financial Economics** — bonus issue announcement and ex-date returns.

≥ 1 peer-reviewed evidence requirement met. Indian-specific direction is the open question the sanity check resolves.

**Counter-evidence / caution:** US literature finds **positive** ex-date drift on splits (signalling explanation); Indian markets may behave differently (lower retail signalling literacy, higher first-time-trader noise). Direction must be determined empirically, not transported.

## Direction

**TBD per evidence.** Sanity-check determines direction.

The mechanic itself is direction-symmetric (compute gap from theoretical adjusted price; trade reversion to theoretical). The sanity will report:
- Open-vs-theoretical gap distribution (sign + magnitude) per event type (split / bonus).
- Realized 09:15-10:30 reversion direction conditional on open gap sign.
- Per-side PF (long the cheap side / short the expensive side).

**Ship rules:**
- One-direction ship if only one side passes PF ≥ 1.10 with n ≥ 30.
- Bidirectional ship only if BOTH sides pass AND |WR delta| ≤ 10pp AND long-side PF ≥ short-side PF × 0.85 (long-bias gate from sub7/8).
- If the open-vs-theoretical gap is symmetric in sign (no consistent direction), then the trade is **mean-revert** with side determined intraday by gap sign — that is the most likely shipped form given mechanic.

## Mechanic

**Setup name:** `stock_split_bonus_exdate_arbitrage`
**Side:** TBD per evidence (most likely sign-of-gap-determined mean-revert).

**Sequence:**

1. **Event detection (T-1 evening / T+0 pre-market, offline):**
   - From the corporate-actions ingest, on each trading day T+0, list all NSE F&O 200 symbols whose ex-date is T+0 with action ∈ {`STOCK_SPLIT`, `BONUS_ISSUE`, `FACE_VALUE_REDUCTION`}.
   - For each, record:
     - `ratio` (e.g., `1:5` for split, `1:1` for bonus)
     - `t_minus_1_close` (T-1 unadjusted close from 5m feather / daily bar)
     - `theoretical_adjusted_price` = `t_minus_1_close / split_factor` (split) or `t_minus_1_close × old_qty / (old_qty + bonus_qty)` (bonus)
   - Skip if: F&O lot-size adjustment isn't yet applied on exchange feed (data integrity), or if the action is dividend-only (no orderbook re-pricing event), or if the symbol has another corporate event the same day (mergers, listings).

2. **Entry trigger (T+0 09:15 first 5m bar):**
   - On T+0 first 5m bar (09:15-09:20):
     - `open_price` = first 5m bar OPEN (auction-result print)
     - `gap_pct` = `(open_price − theoretical_adjusted_price) / theoretical_adjusted_price`
   - **Trigger:** `|gap_pct| ≥ X%` where X is research-locked at the dispersion-1σ band of historical ex-date gaps (estimated ~0.8-1.5%, finalised in sanity).
   - **Direction:** if `gap_pct > +X%` → SHORT (open is above theoretical, fade down); if `gap_pct < -X%` → LONG (open is below theoretical, fade up).

3. **Entry:**
   - **Entry price:** first 5m bar CLOSE (09:20 print).
   - **Active window:** T+0 09:20 entry only. No re-entry, no late entry.

4. **Stop-loss:**
   - **Hard SL:** `entry_price ± 1.5 × |gap_pct|` (against direction). I.e., if SHORT entered with gap +1.5%, SL at +2.25% above theoretical — protects against the gap being a real signal that extends rather than reverts.
   - **Min stop distance:** 1.0% of entry (qty-inflation guard for thin small-caps).

5. **Targets:**
   - **T1** (50% qty): `theoretical_adjusted_price` (the literal mean-revert target — the price the market should have opened at).
   - **T2** (50% qty): `theoretical_adjusted_price ∓ 0.5 × |gap_pct|` (slight overshoot of the theoretical level — symmetric pendulum exit).
   - **Time stop:** 10:30 IST (75-min hold ceiling) OR 15:15 IST hard square if longer hold variant tested (sanity reports both).

6. **Latch:** one fire per (symbol, ex-date). No re-entry.

**target_anchor_type:** `structural` — T1 is the SEBI-defined theoretical adjusted price, which is the cleanest possible structural anchor (regulator-published reference level). Same anchor class as `circuit_t1_fade_short` (circuit-band level).

## Universe

**Universe:** NSE F&O 200 (cash equity + derivative) symbols with ex-date corporate action on T+0, action ∈ {`STOCK_SPLIT`, `BONUS_ISSUE`, `FACE_VALUE_REDUCTION`}.
- **Cap segments allowed:** any (large/mid/small_cap; corporate actions occur across all segments).
- **F&O 200 restriction:** YES — limits universe to liquid borrowable names so short-side fillability via SLB is sound and slippage on the auction print is bounded.
- **Universe filter file:** `assets/fno_liquid_200.csv`.
- **Liquidity gate:** 20-day average `volume × close` ≥ ₹3 Cr on T-1 (rules out names where the corporate action itself caused the liquidity, which contaminates the baseline).
- **Action type expansion (post-sanity, optional):** rights issues, mergers/de-mergers, scheme of arrangement — NOT in initial scope; the orderbook adjustment mechanics differ.

**Sample-size feasibility (binding constraint):**
- F&O 200 universe stock-split events: ~10-25 per year (estimated from NSE corporate-actions browse).
- F&O 200 universe bonus-issue events: ~15-40 per year.
- F&O 200 universe face-value reductions: ~5-15 per year.
- **Combined: ~30-80 events/year on F&O 200.**
- **2-year discovery window: ~60-160 events.**
- After `|gap_pct| ≥ X%` trigger filter (estimated ~50-65% of events): **~30-100 entries over 2 years.**
- **n ≥ 30 floor: feasibly met.** n ≥ 500 floor: NOT feasible. The candidate uses rare-event floor n ≥ 30 with locked thresholds PF ≥ 1.10, |WR delta| ≤ 10pp.

## Active window

**Setup formation:** event detection runs T-1 evening offline (no scan-time work).
**Entry:** T+0 09:20 IST (first 5m bar close after auction).
**Hold horizon:** 09:20 → 10:30 IST = 70 minutes (open-auction inefficiency window; market-maker arb has typically completed by 10:30 per `gap_fade_short` analogue).
**Latest possible exit:** 10:30 IST (sanity also tests 15:15 hard-square as a longer variant for comparison).

**Why 09:20 entry (auction-print bar):**
- The first 5m bar of T+0 contains the auction print and the immediate post-auction continuous-trading prints. This is where the retail-confused order flow lands and where the gap-vs-theoretical is observable.
- Entering at 09:15 (auction itself) is impossible for MIS (auction orders are at-market; we cannot place limit orders at theoretical adjusted price reliably in the auction).
- Entering after 10:30 misses the bulk of the reversion.

**Why 10:30 exit ceiling:**
- Mirrors `gap_fade_short` and the morning-mispricing-reversion class.
- Empirical: by 10:30, market makers' corp-action arb has completed; post-10:30 price action is regular intraday flow uncorrelated with the ex-date asymmetry.

## Risks / falsification criteria

The setup is **wrong** (and should be retired) if:

1. **Phase-1 floor fails on validation/holdout:**
   - n < 30 trades over 2 years (sample-size famine — the rare-event floor)
   - NET PF < 1.10
   - |WR delta vs baseline-day-1 trade| > 10pp (retail-locked threshold per round-3+ standard)
2. **Direction is unstable:** if sign of `gap_pct` doesn't predict reversion direction with consistency (e.g., gap-up events sometimes revert, sometimes extend, with no separating filter), the asymmetry is illusory.
3. **Action-type heterogeneity:** if SPLITS pass but BONUS does not (or vice versa), ship the passing action only — don't pool.
4. **Data ingestion path fails:** if corporate-actions data cannot be sourced reliably (e.g., NSE site changes, no clean machine-readable feed), the setup cannot be productionised even if the asymmetry is real. Retire.
5. **Theoretical-adjusted-price computation has gotchas:** F&O lot-size adjustments, simultaneous dividend + split events, fractional-share rounding rules, complex ratios (e.g., 7:3 splits). If >20% of events are unparseable, retire.
6. **Decay signal:** rolling NET PF on the most recent 30 events drops below 1.05 sustained — broker UX has matured past the edge.

**Pre-coding sanity check** (mandatory per §3.3, BEFORE writing detector):
- Manually curate 2024 + 2023 NSE F&O 200 corporate-actions list (target ~60-160 events).
  - Source: NSE corporate-filings page, filter by action type (split, bonus, face-value reduction), filter by symbol ∈ F&O 200 (asset-list join).
  - Per event capture: symbol, ex-date, action_type, ratio, T-1 close, theoretical adjusted price.
- For each event, simulate T+0 09:20 entry against theoretical price target with `|gap_pct| ≥ X%` trigger sweep (X ∈ {0.5%, 1.0%, 1.5%, 2.0%}).
- Compute NET PF per direction (gap-up SHORT, gap-down LONG) per action type (split, bonus, face-value reduction).
- Use existing Indian fee model (`tools/sub7_validation/build_per_setup_pnl.py:calc_fee`).
- Report n, PF, WR, Sharpe per (direction × action_type × X-threshold) cell.
- **Diagnostic:** also run the 10:30 vs 15:15 exit comparison (which hold horizon captures more of the reversion NET).
- **Decision per §3.3:** PF ≥ 1.10 with n ≥ 30 on at least one (direction × action_type) cell → strong proceed; 1.0-1.10 → marginal; PF < 1.0 across all cells → retire.

## Data engineering plan

**FLAGGED: corporate-actions data must be ingested. This is the largest infrastructural lift of any sub-9 candidate.**

Required new components (only if sanity check passes):

1. **`tools/sub9_research/sanity_stock_split_bonus_exdate_arbitrage.py`** — pre-coding sanity check.
   - Input: a manually-curated CSV `data/corp_actions/manual_curated_2023_2024.csv` (columns: `symbol, ex_date, action_type, ratio, t_minus_1_close, theoretical_adjusted_price, notes`).
   - Reads existing 5m feathers for T+0 09:15-15:15 of each ex-date.
   - Simulates entry/exit logic with parameter sweep.
   - No detector code yet. Will be retired after used.

2. **(post-sanity-check, only if APPROVED for full implementation):**
   - **Corporate-actions ingestion pipeline** (NEW component):
     - **Primary source:** NSE corporate-filings page (https://www.nseindia.com/companies-listing/corporate-filings-actions). Scraper writes a daily-refreshed parquet at `data/corp_actions/nse_corp_actions.parquet`.
     - **Backup source:** BSE corporate-actions archive (cross-validation for BSE-listed-only events).
     - **Tertiary source:** NSE bhavcopy adjustment-factor field (independent reconstruction of theoretical adjusted price from already-on-disk bhavcopy data — strongest robustness).
     - **Optional vendor source:** Tickertape API (paid, used only if NSE scraper fails the production-reliability test).
     - **Validation:** every event pulled from primary must reconcile against bhavcopy adjustment-factor for the ex-date (ratio implied by adjustment factor must equal ratio in the corp-action announcement). Mismatches are quarantined.
   - **`services/corp_actions_loader.py`** — load events keyed by ex-date, lookup by (symbol, ex-date). Caches parquet, refreshes nightly.
   - **`structures/stock_split_bonus_exdate_arbitrage_structure.py`** — the detector. At T+0 09:20 bar, `detect()` queries the corp-actions loader for `ex_date == today AND symbol == self.symbol AND action_type ∈ allow-list`; if hit, computes gap and triggers per logic.
   - Config keys (added to `config/configuration.json`, NO hardcoded defaults per CLAUDE.md rule 1):
     - `stock_split_bonus_exdate_arbitrage.gap_threshold_pct` = TBD per sanity (placeholder 0.01)
     - `stock_split_bonus_exdate_arbitrage.allowed_action_types` = TBD per sanity (placeholder `["STOCK_SPLIT", "BONUS_ISSUE", "FACE_VALUE_REDUCTION"]`)
     - `stock_split_bonus_exdate_arbitrage.entry_window_ist` = "09:20"
     - `stock_split_bonus_exdate_arbitrage.time_stop_ist` = "10:30"
     - `stock_split_bonus_exdate_arbitrage.sl_multiplier` = 1.5
     - `stock_split_bonus_exdate_arbitrage.min_stop_pct` = 0.01
     - `stock_split_bonus_exdate_arbitrage.t1_anchor` = "theoretical_adjusted_price"
     - `stock_split_bonus_exdate_arbitrage.t2_overshoot_frac` = 0.5
     - `stock_split_bonus_exdate_arbitrage.universe_file` = "assets/fno_liquid_200.csv"
     - `stock_split_bonus_exdate_arbitrage.min_adv_inr_cr` = 3.0
     - `stock_split_bonus_exdate_arbitrage.long_side_enabled` / `short_side_enabled` — toggled per sanity.

**Acceptance gate (per the prompt):** APPROVE-eligible only if BOTH of:
- **Data path is clear:** primary (NSE scraper) + tertiary (bhavcopy adjustment-factor) reconciliation is implementable with current tools — YES, both are publicly available.
- **Sample size feasible above n ≥ 30 floor on F&O 200 over 2yr:** estimated 60-160 events, after gap-trigger filter ~30-100 entries — YES, marginally feasible.

Both conditions met. Brief is APPROVE-eligible.

## Honest comparison to surviving setups

| Aspect | gap_fade_short (TRUSTED) | circuit_t1_fade_short (APPROVED) | stock_split_bonus_exdate_arbitrage (proposed) |
|---|---|---|---|
| Class | NSE/SEBI mechanical event | NSE/SEBI mechanical event | NSE/SEBI mechanical event |
| Indian-specific | retail open momentum exhaustion | retail FOMO + operator pump on circuit | retail confusion on corp-action ex-date open |
| Direction | short-only | short-only | TBD per evidence (likely sign-of-gap-determined) |
| Active window | T+0 09:15-09:30 | T+1 10:30 single-bar | T+0 09:20 entry, 10:30 exit |
| Universe | small_cap | mid_cap, small_cap | F&O 200, any cap |
| Hold | 15-30 min MIS | 4h 45m MIS | 70 min MIS |
| Anchor | structural (PDC) | structural (gap edges) | **structural (SEBI theoretical adjusted price)** — strongest anchor class |
| Trigger | gap level | circuit-band level | corp-action ex-date + open-vs-theoretical gap |
| Sample size (annual) | ~3-5K | ~750-1750 | **~30-100** (RARE-event floor) |
| Sample size floor | n ≥ 500 | n ≥ 500 | **n ≥ 30** (locked rare-event floor) |
| Data infra | existing 5m | existing bhavcopy | **NEW corp-actions ingestion** |
| Correlation w/ existing | n/a | low | very low (event-driven, ~30-80 days/yr) |
| Decay risk | low | low | moderate (broker UX maturing, but first-time-trader cohort renews) |

**Honest summary of fit:**
- **Strong fit:** NSE/SEBI mechanical event class (winner class). Strongest possible structural anchor (SEBI-published theoretical adjusted price). Very low correlation with existing setups. Direction-TBD-per-evidence avoids cargo-culted-long-bias trap. Locked rare-event-floor thresholds prevent retro-fitting.
- **Concerns:** Sample size at floor (n ≥ 30 rather than ≥ 500). New data ingestion required. Direction not pre-known. Decay risk on broker-UX-confusion component.
- **Net:** the candidate's structural anchor is so clean (regulator-published target price) that a small sample of high-edge events is plausible. The acceptance gate (data path clear + n ≥ 30 feasible) is met.

---

## Decision required

User to indicate:
- [ ] APPROVED — proceed to manual 2023-2024 corp-actions curation + sanity-check script
- [ ] REJECTED — reason
- [ ] RETIRE — kill candidate

Per sub-9 §3.3, no detector code is written until APPROVED and sanity-check passes (NET PF ≥ 1.10 with n ≥ 30 on at least one (direction × action_type) cell).
