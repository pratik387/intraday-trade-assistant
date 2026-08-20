# Execution-Fidelity Systematic Trading — Diligence Pack

NSE intraday / overnight / multi-day systematic equity
Prepared 2026-08-19 · all figures traceable to run ledgers in this repository

---

## 1. Thesis

> **Most systematic managers cannot tell you why their live P&L differs from
> their backtest. We can — decomposed per basis point, per leg — and we retire
> our own validated signals when forward data says to.**

The claim is not that our research is more clever. It is that the number our
research produces and the number our account produces are the *same number*,
and where they are not, we can name the difference and its cause.

In NSE small- and mid-cap equity, that is the binding constraint. Signals are
abundant; surviving implementation is not. Our comparative advantage is a
measured, instrumented path from backtest to live — not a signal library.

---

## 2. What this is

| | |
|---|---|
| Market | NSE cash equity — intraday (MIS), overnight (MTF/CNC), 2–3 day multi-day |
| Style | Short-horizon mean-reversion and event-reaction; no leverage beyond broker MIS/MTF |
| Instruments | ~1,200–2,400 symbol universe, cap- and liquidity-filtered per setup |
| Active setups | **9** of **148** researched |
| Broker | Zerodha (orders) + Upstox (market data) |
| Capital deployed | Paper ₹5L; live overnight book ₹50k/slot |

Three independent books share one engine, one config, one validation pipeline.
Each has its own capital pool, decay tripwire, and kill switch.

---

## 3. Why the backtest can be trusted

The evidence an allocator should weigh is not what we shipped. It is **what we
refused to ship, and what we killed after shipping.**

### 3.1 Rejection rate

**148 researched candidates → 9 active.** Every rejected candidate retains its
brief, its data, and its kill reason in `specs/` and `docs/retired_setups.md`
(140 documented entries). Nothing is quietly abandoned.

### 3.2 Validation is one-shot and pre-registered

- A candidate's cell (parameter region) is **locked before** out-of-sample data
  is touched. OOS is spent once; iterating on it is treated as spending
  multiple shots and is recorded as such.
- Variants are pre-registered before testing, not selected after.
- A **17-stage documented lifecycle** (`docs/setup_lifecycle.md`, Stages 0–14
  plus amendments A1–A5) governs promotion. No setup reaches live without
  passing every stage in order.

### 3.3 Multiplicity is priced, not ignored

Every experiment is written to an **experiment ledger**, and confidence
statements carry a **Harvey–Liu haircut** at M = the number of ship-eligible
variants tested. This directly answers the question that ends most
emerging-manager meetings: *"how many strategies did you try before this one?"*
We can produce the count.

### 3.4 We kill our own validated alpha

The strongest single piece of evidence. Our overnight slot-ranking rule
(deepest-|signed-volume-ratio| first) was validated across Discovery, OOS and
Holdout. Forward, it **reversed**:

| | |
|---|---|
| Trades the ranker took | **−0.306%** |
| Trades it skipped | **+0.624%** |
| Permutation test | **0.0th percentile of 20,000 draws, p = 0.0001** |

We did not re-fit it. We **replaced it with a deliberately random ranker**
(`sha1(date\|symbol)`), accepting the random-pick expected value because the
evidence said our ordering was worse than chance. Attributable cost of having
run it: ~₹11.5k of the −₹17.9k July live result.

### 3.5 Parameters do not move in response to losses

Stop-loss and setup-geometry changes are made **only through the backtest
pipeline**, never as a live reaction to a drawdown. A recent −₹55.7k paper loss
on a single name did not trigger a stop-loss change; it triggered a *sizing*
review, because the stop had behaved exactly as designed (§6.2).

---

## 4. The backtest-to-live gap, measured

This is the section most managers cannot write.

### 4.1 The gap is decomposed, not assumed

For the overnight book, measured against its own idealised backtest fills:

| component | measured | status |
|---|---|---|
| **Selection** — which signals we take when more fire than slots | **−0.474%/trade** | **fixed 2026-08-04** |
| **Execution** — fills vs idealised | **−0.101%/trade** | partly structural |

Execution breaks down further: entry slippage median **+3.8 bp** / mean
**+7.5 bp**; exit median **0.0 bp** / mean **−2.8 bp**. 39 of 79 trades land
within ±0.25% of idealised. **The damage is a tail of six entries at 92–147 bp**
— which tells us where to spend effort, and that the median trade is fine.

**Independently re-measured 2026-08-20 on a larger sample (n = 99), the ranking
matters ~10× more than the fills.** Matching every live trade to its idealised
twin and comparing at *reference* prices — where execution cannot interfere:

| | n | mean / trade |
|---|---|---|
| Trades the book **took** | 99 | **−₹63** |
| Trades the book **skipped** | 221 | **+₹1,084** |
| Execution cost on the trades taken | 99 | **−₹115** |

Selection was worth **₹1,147/trade** more than execution cost. This is the same
defect as §3.4, measured a second way on a different sample, and it is why the
ranker was replaced rather than tuned. Post-fix (from 2026-08-04) the live book
is **+₹623 over 27 trades**, against −₹18,182 over the preceding 72 — directionally
right, but 27 trades is far too few to claim.

### 4.2 Slippage is measured per-trade, per-book — not parameterised

Backtests run at `fees_slippage_bps = 5.0`, and **we state plainly that this is
optimistic.** Slippage is measured separately for each book, because the two
differ by an order of magnitude and averaging them would mislead.

**Overnight book — measured, not modelled.** Every live trade is matched 1:1
against an independently reconstructed idealised twin (same symbol, same date,
reference prices), giving a direct per-trade measurement. n = 99, 2026-07-01 →
2026-08-20:

| leg | mean | median | p90 |
|---|---|---|---|
| Entry | **+3.0 bp** | +1.8 bp | +79.7 |
| Exit | +10.3 bp | **0.0 bp** | 0.0 |
| **Round-trip** | **+13.2 bp** | **+3.8 bp** | +81.3 |

Against a **46.2 bp/side break-even**, the median trade consumes 8% of the
available headroom. 32% of trades land within ±10 bp, and the 10th percentile is
**negative** — we sometimes fill better than the reference.

The mean is a tail, and the tail is identifiable: the two worst exits are the
`CREATIVEYE` cancel/fill race (+800 bp) and the `REGENCERAM` stuck partial AMO
(+218 bp) — both in the incident register, both fixed. **Excluding those two,
round-trip slippage is 6.0 bp mean / 3.0 bp median (₹94/trade).** 97 of 99 exits
fill at exactly the reference price, because the exit AMO executes at the
opening print the reference is drawn from.

Post-fix trend is visible in the data: round-trip mean **14.8 bp before
2026-08-04 → 9.1 bp after**.

**Intraday illiquid shorts** are a different regime and are reported separately:
central **18.7 bp/side**, conservative **27.5**, recent worst **~30**. We
therefore report candidate economics at *both* figures. Our newest candidate
shows true-OOS **+0.525% / PF 1.564 / t = +2.41** at 18.7 bp — and **its
confidence interval crosses zero at 27.5 bp.** We publish the second number
alongside the first.

### 4.3 Entry basis is validated against what was actually reachable

We re-anchored the overnight book's entry from the theoretical close to the
price actually available at decision time:

| basis | mean/trade | PF |
|---|---|---|
| 15:25 open (reachable) | **+0.349%** | 1.812 |
| 15:30 close (assumed) | +0.240% | 1.527 |

Live fills come in ~**18.7 bp** worse than the price available at their own
entry time — so of a +0.349% achievable edge, roughly 0.19% is given back in
fill quality. Both numbers are in the record.

### 4.4 Data-path fidelity is measured

The live WebSocket path was found to drop **10–25% of bars for illiquid
symbols** — the root cause of a 20–30% paper/backtest divergence. Replaced with
an API-first ingestion at 40 RPS. Divergence is now monitored per session, not
assumed away.

### 4.5 Fees are real, not modelled

Zerodha charges are computed per-trade from the actual schedule (measured
MIS-short cost **0.083% of notional**), applied per-trade before aggregation,
with MIS leverage applied per-trade rather than to the aggregate — and tax
computed on **net annual FY income**, not per-trade.

---

## 5. Mitigation framework

Each control below exists because a specific failure occurred and was
root-caused. The full incident register is
`docs/LIVE_TRADING_INCIDENTS.md` — 24 incidents, every commit hash and code
anchor verified.

### 5.1 The failure taxonomy we design against

| class | example | control |
|---|---|---|
| **Backtest-invisible surfaces** | three detectors passed full validation and could not fire live (bar labelling, bar availability, today-only intraday data) | live/backtest parity harness; sentinel-bar pattern; per-session fire-count monitoring |
| **Execution erosion** | partial fills orphaned, cancel/fill races, margin pinned account-wide, circuit-band rejections | orderbook reconciliation sweep; partial-fill attachment on all rollback paths; shortfall-based retry sizing |
| **Silent degradation** | a failed data-prep job left a stale snapshot; the cost landed the *next* day in a *different* log | failures that degrade a downstream consumer now log CRITICAL and surface in the run summary |
| **Selection drift** | a validated ranker reversed forward | forward-validation of selection rules with permutation testing; default to unbiased ordering |
| **Accounting error** | P&L aggregation over final legs only misread a profitable book as losing | reporting audited against an independent source before any decision |

### 5.2 Standing controls

- **Decay tripwire** per setup — rolling-window PF floor, auto-pause on
  sustained breach, evaluated on one regime only (ledgers segment at any
  parameter boundary rather than pooling across it).
- **Circuit-breaker state** as an allowlist — unknown states block rather than
  permit.
- **Regime segmentation** — when sizing or selection changes, the ledger splits;
  statistics are never pooled across a boundary, and per-trade sizing provenance
  is recorded so any trade can be restated at any book size exactly.
- **Cluster-level correlation caps** on the multi-day book (measured pairwise
  ρ = 0.227; volatility-targeted sizing on a stated risk budget).
- **Kill switches** at setup, cluster and book level.

---

## 6. Track record, presented honestly

### 6.1 Two sizing regimes — reported separately, not blended

On 2026-08-14 we deliberately raised the intraday book's size multiplier. Rupee
results are therefore **not one sample**, and we do not present them as one.

**Primary track — 1× regime, homogeneous sizing (43 sessions, 123 trades):**

| | |
|---|---|
| Net | **+₹12,515** |
| Profit factor | **1.22** |
| Win rate | 55% |
| Winning days | 25 / 43 |
| Mean / trade | **+0.285%** (SD 3.47%) |
| **t-statistic** | **0.91** |
| 95% CI | **[−0.332%, +0.895%]** |
| Median notional | ₹29,945 |
| Daily P&L SD | ₹1,946 → **6.2% annualised** |
| Max drawdown | ₹5,699 (**1.1%** of capital) |

**We state plainly: t = 0.91 is not statistically significant.** The point
estimate is positive, the confidence interval includes zero, and 123 trades is
not a track record. We present it as *evidence the process runs correctly at
size*, not as proof of edge.

**Secondary — 10× regime (2 sessions, 7 trades):** net **−₹70,035**, PF 0.10,
daily SD ₹33,855 (**107.5% annualised**), max drawdown 14.0% of capital.

This is disclosed rather than excluded. Two sessions is no verdict on edge, but
it is a clear verdict on sizing, and §6.2 states what we concluded.

### 6.2 What the 10× regime taught us

One position accounts for 97% of that loss: an event-reaction short that moved
**−11.17%** against a ₹498k notional. Its catastrophe stop (15%) did not fire —
**correctly**. That stop is deliberately set above the worst adverse excursion
ever observed (max 12.97%, p99 11.39%) so that it never becomes a strategy exit;
the real exit is a 15:10 time stop. The design behaved exactly as specified.

**What failed was sizing, not the setup.** A no-stop strategy with an 11–13%
tail, sized at ₹500k against ₹5L, places ~11% of capital on one name's tail. At
its validated ₹1L footprint the same adverse move costs ₹11,132.

Two process points follow, and both are the point of this section:

1. **It happened in paper, not live.** The size increase was tested on simulated
   capital first. That is what the paper book is for.
2. **We did not change the stop.** The stop was correct. We changed the sizing
   control — and any stop or setup-geometry change would go back through the
   backtest pipeline, never a live reaction.

### 6.3 Other books

- **Overnight (live, real money, ₹50k/slot):** the only book with a live-money
  record. July net −₹17.9k, of which ~₹11.5k is attributed to the selection rule
  retired on 2026-08-04. Post-fix performance is being accumulated.
- **Multi-day:** relaunched 2026-08-14 under volatility-targeted sizing with
  cluster correlation caps. Replay of the prior book under the new rules reduced
  daily P&L SD from ₹33,035 to ₹8,316 and worst day from −₹60,452 to −₹11,271.
  Forward record begins 2026-08-14.

---

## 7. Limitations — stated before you find them

1. **No statistically significant live edge yet.** t = 0.91 on the primary
   track. We are not claiming proven alpha.
2. **Short record, small capital.** Months, not years; ₹5L paper and ₹50k/slot
   live.
3. **Capacity is genuinely constrained — and we have measured it.** Median
   daily turnover of the names we actually trade is **₹9.9 crore**; 62% of trades
   are in names under 500K median share volume. Current per-trade participation
   is **0.052% of daily turnover** (p95 0.272%), so today's size is not the
   constraint — but scaling is.

   Holding per-trade notional at its current share of capital, the AUM at which
   trades begin hitting a participation cap:

   | cap on daily turnover | binding on least-liquid decile | on p25 | on median trade |
   |---|---|---|---|
   | 1% of ADV | **₹21 lakh** | ₹33 lakh | ₹96 lakh |
   | 3% of ADV | **₹64 lakh** | ₹98 lakh | ₹2.9 crore |
   | 5% of ADV | **₹1.07 crore** | ₹1.63 crore | ₹4.8 crore |

   The least-liquid decile binds, because that is where the edge lives — 88% of
   P&L comes from the illiquid tail. **Realistic capacity is ₹0.6–1.6 crore**
   at 3–5% participation. It reaches ~₹5 crore only by dropping the illiquid
   names, which forfeits most of the measured edge.

   **Strategic implication, stated plainly:** at this capacity the strategy is
   not a standalone institutional allocation. It is viable as a **proprietary
   book, a family-office mandate, or a sleeve inside a multi-strategy platform**
   — and we would rather say so than have it discovered in diligence.

   The capacity constraint is a property of the edge, not of the
   implementation: our measured slippage (18.7 bp central) against a 46.2 bp
   break-even leaves real headroom per trade, but the illiquid tail caps
   aggregate size regardless.
4. **A measured regime break.** The illiquidity premium we trade **turned
   negative from 2025Q4**. We measured it, and it invalidated several
   2023–24-derived candidates, which were retired. We regard having detected it
   as more informative than any single backtest.
5. **Infrastructure is early.** Single VM, some manually-started processes,
   alerting added only recently. This is a funding-stage gap, not a design one.
6. **Known parity asymmetry, accepted:** live computes one signal from
   unadjusted prices while backtests use adjusted series, so live *blocks* some
   fires the backtest contains. One-way (never false fires), quantified,
   documented rather than hidden.

---

## 8. What we are asking for, and what we would do with it

The honest ask is not "allocate to a proven strategy." It is:

- **Capital sufficient to make the live sample statistically decisive** at the
  measured capacity ceiling, and
- **Infrastructure funding** to close the operational gaps in §7.5.

What a partner gets that is unusual: a manager who can hand over the
implementation-shortfall decomposition, the rejection ledger, the incident
register, and the list of its own validated ideas it has killed — **before**
being asked for any of them.

---

## Appendix — verification

Every figure is reproducible from this repository:

| claim | source |
|---|---|
| Track record, both regimes | `logs/paper_*/analytics.jsonl`, aggregated per-lifecycle over **all** exit legs |
| Selection / execution decomposition | `docs/SYSTEM_HANDOVER.md` §11 |
| Slippage and break-even | `docs/SYSTEM_HANDOVER.md` §§222–223, 348–349 |
| Overnight slippage (n=99) | `state/decay_tripwire_close_dn_overnight_long_live.json` matched 1:1 against the reconstructed idealised ledger `state/decay_tripwire_close_dn_overnight_long.json` on (symbol, settle-date) |
| Ranker reversal and permutation test | commit `4adcd63` |
| Incident register | `docs/LIVE_TRADING_INCIDENTS.md` (24 entries, anchors verified) |
| Rejection rate | `specs/` (148 briefs), `docs/retired_setups.md` (140 entries) |
| Capacity analysis | traded-symbol turnover from `cache/preaggregate/consolidated_daily.feather`, 60-session median, joined to realised per-trade notionals |
| Lifecycle and amendments | `docs/setup_lifecycle.md` |
| Experiment ledger | `docs/experiment_ledger.jsonl` |

**Reporting integrity note.** P&L is aggregated over every exit leg per trade,
filtered to currently-active setups, and never pooled across a sizing or
selection boundary. Each of those three rules exists because violating it once
produced a materially wrong number — all three failures are documented in the
incident register.
