# Plan: Multi-day capital management + slot selection

**Date:** 2026-08-12
**Scope:** `horizon: multi_day` book — `crash2d_revert_long`, `zscore_oversold_revert_long`,
`low52_capitulation_revert_long`, `mtf_capitulation_revert_long`. Currently PAPER only.
**Status:** Phases 0-4 + 6 IMPLEMENTED and deployed 2026-08-12. Phase 5 TESTED AND REJECTED.
See S11 for outcomes. Original plan text below is left intact as the pre-registration. Every number below is labelled DERIVED (measured),
CHOSEN (a judgement call, not fitted), or UNVALIDATED (needs its own test).

---

## 0. Why this exists

The multi-day book has **no capital management at all**:

```python
limit = min(max_new_per_day, max_concurrent - held)   # 100, 200 -> never binds
qty   = int((margin_per_slot * leverage) // close)    # flat Rs1L margin, every position
```

Take-all selection, flat notional, no risk scaling, no correlation awareness, no capacity
check. Margin-pool arbitration is explicitly "out of scope (paper testing)".

**Measured consequence (DERIVED, 23 entry days, 121 deduped book positions):**

| | |
|---|---|
| per-position return SD | 3.87% (worst −10.20%) |
| daily book P&L SD | **Rs 33,035** |
| worst day / worst week | −Rs 60,452 / −Rs 56,145 |
| median notional | Rs 281,566 per position |
| entries/day × mean hold | 4.9 × 2.72d → **~13 concurrent** |
| mean pairwise correlation | +0.227 |

At 6–8 slots that is **65–87% annualised volatility**. Institutional systematic books target
10–15%. The book is running 5–8× a professional risk budget.

---

## 1. Phase 0 — GATE: do not allocate capital to a negative-expectancy book

**This precedes every sizing question.** DERIVED:

| setup | n | mean/position | SD |
|---|---|---|---|
| crash2d_revert_long | 49 | **−0.984%** | 4.13% |
| zscore_oversold_revert_long | 79 | +0.366% | 3.77% |
| low52_capitulation_revert_long | 17 | +0.896% | 2.79% |
| mtf_capitulation_revert_long | 9 | +0.117% | 3.94% |
| **BOOK (deduped)** | **121** | **−0.014%** | **3.87%** |

The book mean is negative; `crash2d` accounts for it. Fractional Kelly on a negative edge
sizes to zero. **No allocation scheme fixes this** — sizing a losing book correctly only
loses more slowly.

**Gate:** a setup receives capital only with a defensible positive expectancy. On current
evidence that admits `zscore_oversold` and `low52`; `crash2d` allocates **zero** until its
decay tripwire resolves (below floor since 2026-07-23, auto-pauses ~2026-09-03).
`mtf_capitulation` is n=9 — too thin to judge either way; hold at zero pending sample.

Do NOT hand-disable `crash2d` early — it is paper, there is no bleed, and killing it
destroys the tripwire's pre-registered evidence.

---

## 2. Phase 1 — Add the missing primitive: an explicit risk budget

There is currently no number anywhere in config expressing "how much risk is this book
allowed to take." Everything else derives from it.

**New config block (CHOSEN structure, values TBD by the operator):**

```
multi_day_portfolio.risk_budget:
  daily_vol_target_pct        # book daily P&L SD as % of capital
  max_daily_loss_pct          # hard stop, halts new entries for the session
  max_book_drawdown_pct       # peak-to-trough halt
  capital_inr                 # the denominator — must be explicit, not implied
```

**Capital implied by the target, at CURRENT sizing (DERIVED):**

| daily vol target | annualised | capital needed | worst day | worst week |
|---|---|---|---|---|
| 0.63% | 10% | Rs 52.4 lakh | −1.2% | −1.1% |
| 0.95% | 15% | **Rs 34.8 lakh** | −1.7% | −1.6% |
| 1.26% | 20% | Rs 26.2 lakh | −2.3% | −2.1% |
| 1.90% | 30% | Rs 17.4 lakh | −3.5% | −3.2% |

**If capital is fixed, positions scale (DERIVED, at 15% annual):**

| capital | position size vs today | notional/position |
|---|---|---|
| Rs 5 lakh | 14% | Rs 40,486 |
| Rs 10 lakh | 29% | Rs 80,971 |
| Rs 25 lakh | 72% | Rs 202,428 |
| Rs 35 lakh | 101% | Rs 283,399 |

**Risk binds, not margin** — utilisation lands at ~37% of capital at every level, so MTF
leverage is not the constraint. Volatility is.

⚠ **Small-capital warning:** below ~Rs 10 lakh, position notional falls under Rs 80k. The
Rs 20/order brokerage cap stops helping below ~Rs 67k notional, while delivery STT (0.2%
round-trip) stays proportional. Small positions make an already-thin edge thinner. If the
answer is "we have Rs 5–10 lakh", the implication is **fewer, better setups** — not scaling
to 14% and proceeding.

**Set the target from drawdown tolerance, NOT from backtested returns.** Choosing the vol
target to maximise historical P&L is the same fitting error as tuning a threshold.

---

## 3. Phase 2 — Size by risk, not by rupees

Replace `qty = (margin_per_slot * leverage) // close` with vol-targeted sizing:

```
position_notional = per_position_risk_budget / sigma_symbol
```

`sigma20_pct` is **already carried** in the selector's candidate dicts, so this needs no new
data. Today a Rs 919cr illiquid name and a large cap receive identical Rs 1L margin despite
very different volatility — that is the core defect of flat sizing.

Per-position risk budget derives from the book target and the correlation-adjusted position
count (Phase 3), not from a hand-picked per-trade rupee figure.

---

## 4. Phase 3 — Cap by correlation cluster, not per setup

**DERIVED correlations:** crash2d × zscore **+0.68**, crash2d × mtf +0.50, mtf × zscore
+0.38; `low52` × everything −0.19 to +0.08.

Three of four setups are **one bet**. Per-setup slot pools would let the book hold the same
capitulation position four times over — precisely when it hurts, because they co-fire.

**Structure (CHOSEN):**
- **Sleeve A** — crash2d + zscore + mtf_capitulation: one shared budget and one slot cap
- **Sleeve B** — low52: its own allocation, the genuine diversifier
- Book-level concurrency cap sized on DERIVED demand: ~13 concurrent typical, p90 8
  entries/day

**Correlation haircut (DERIVED from the standard equal-weight constant-correlation identity
`sigma_p = sigma * sqrt((1+(n-1)*rho)/n)`):** at the measured rho = 0.227, 8 positions carry
**1.61×** the risk of 8 independent ones; 12 carry 1.87×. Sleeve A must size down by that
factor for equal risk. This formula is textbook, not invented.

---

## 5. Phase 4 — Selection: random null + cost filter, NOT a quality score

**The current composite ranking is unsupported by the data.** The selector does:

```python
composite += weight * min(cap_score, cap_score_clip)   # consensus raises rank
```

Its entire purpose is to promote names multiple setups agree on. **DERIVED test on 121
deduped positions:**

| #contributors | n | mean | PF | win |
|---|---|---|---|---|
| 1 | 86 | −0.049% | 1.01 | 44% |
| 2 | 26 | **−1.033%** | 0.55 | 46% |
| 3 | 8 | +1.204% | 2.42 | 62% |
| 4 | 1 | +1.541% | — | 100% |

Consensus (>1) −0.448% vs solo −0.049%, t=−0.52, **permutation p=0.69**. Non-monotonic,
statistically nothing, and the positive cells are n=8 and n=1.

This is the `conviction` ranker's shape again — a plausible ordering, validated in backtest,
that does not reproduce forward. `conviction` was measured **anti-predictive at p=0.0001**
and had to be replaced with `unbiased_hash`.

**Plan (CHOSEN, mirrors the overnight remedy):**
1. **Keep the dedupe** — it is sound and must stay. One book position per symbol.
2. **Rank the deduped set randomly** (date-salted hash, same as overnight), not by composite.
3. **Keep `composite` computed and logged but unused**, so it can be scored against the
   random baseline continuously.
4. **Owner assignment stays** — it drives `close`/`sigma20_pct` for sizing, which is a sizing
   input, not a selection claim. Make it explicit rather than a by-product of highest
   cap_score.
5. **Add a cost/liquidity filter** — skip names whose expected execution cost eats a
   meaningful share of the edge. Cost is predictable; alpha demonstrably is not.

---

## 6. Phase 5 — Re-entry cooldown (UNVALIDATED, test before shipping)

**DERIVED:** 16 of 102 symbols were traded more than once (max 3×). Those 35 positions
average **−0.921%** against **+0.143%** for one-off names. ASALCBR, SHANTIGEAR and TCI were
each traded three times.

The `held` guard correctly blocks pyramiding while a position is open, but nothing prevents
re-entry days later. Plausible mechanism: every setup here is capitulation-reversion, so a
name that keeps re-qualifying is one that keeps falling — the thesis is failing on it.

**Do NOT ship this as a config change now.** n=35 across 16 symbols, and it was not a
pre-registered hypothesis. Pre-register a cooldown of N days, test on the accruing mirror,
then decide.

---

## 7. Phase 6 — Validation harness before any of this touches capital

The unconstrained paper book **is** the counterfactual dataset: it takes every fire, so the
outcome of every skipped candidate is observable. That is ~8–10 labelled decisions/day versus
the ~3 a capped book would trade.

**Harness:** nightly, score every candidate policy against the mirror — random baseline,
composite, cost-filtered, cooldown variants. Pre-register the margin and the sample size
BEFORE looking. A policy touches capital only after beating random out-of-sample on
pre-registered terms.

**Build this before imposing caps.** Once caps bind you stop observing the counterfactuals,
and you would be choosing a policy blind at the moment it starts costing money.

---

## 8. Go-live gates (all must hold)

1. Book expectancy positive with excluded setups removed (Phase 0)
2. Explicit risk budget set from drawdown tolerance, and capital sufficient for it (Phase 1)
3. Vol-targeted sizing live in paper and reproducing expected book vol (Phase 2)
4. Cluster caps in place; no sleeve able to hold the same bet 4× (Phase 3)
5. Selection policy has beaten the random baseline out-of-sample on pre-registered terms
   (Phases 4–5 via Phase 6)

---

## 9. Explicitly NOT in this plan

- **No ML selection layer.** ~860 labelled trades across three books, one regime. The one
  ranker deployed (`conviction`) was negatively skilled. Sizing/capital has denser labels
  and bounded downside — that is where a learned layer belongs first, not selection.
- **No day-of-week filter.** Investigated and rejected: "Friday worst" was an exit-day
  artifact of Wednesday entries, and Wednesday is 76% one day of three trades across only
  4 distinct Wednesdays.
- **No hand-disabling `crash2d`.** The tripwire resolves it ~2026-09-03 on pre-registered
  terms; it is paper, so there is no bleed to stop.
- **No tuning of the vol target or Kelly fraction to backtested returns.**

---

## 10. External anchors

- Volatility targeting — size inversely to volatility to hold portfolio risk constant; the
  institutional default underlying risk-parity and managed-futures programmes.
- Fractional Kelly — practitioners use 25–50% of full Kelly because the overbetting penalty
  far exceeds the underbetting cost, and a 10% error in expected-return estimation can cause
  ~50% overbetting. Half-Kelly retains ~75% of the growth rate at materially lower variance.

Both point the same way here: we currently size in rupees with no vol adjustment, on an edge
estimate whose confidence interval crosses zero.


---

## 11. OUTCOMES (2026-08-12, after implementation)

| phase | status | commit |
|---|---|---|
| 0 crash2d disabled | DONE | `aaa7a86` |
| 1 risk budget (Rs 10L, 0.95%/day) | DONE | `aaa7a86` / `2c0e76a` |
| 2 vol-targeted sizing | DONE | `aaa7a86` |
| 3 cluster caps | DONE | `aaa7a86` |
| 4 unbiased-hash ranking | DONE | `8680a18` |
| 5 re-entry cooldown | **REJECTED** — see below | — |
| 6 shadow harness | DONE | `8a4e657` |

### 11a. Replay of the paper history under the new rules

| | n | net | mean/position | PF |
|---|---|---|---|---|
| actual (flat Rs 1L, take-all) | 121 | −Rs 8,923 | −0.164% | 0.98 |
| new rules | 59 | +Rs 62,589 | +0.783% | 1.92 |

Risk moved as designed: daily P&L SD **Rs 33,035 → Rs 8,316** (0.83% of Rs 10L
against a 0.95% target); worst day **−Rs 60,452 → −Rs 11,271** (−1.13% of capital);
median notional Rs 282,297 → Rs 148,867.

Skips: cluster_concurrent 25, crash2d_disabled 22, cluster_new_per_day 15. Zero
`below_min_notional` — the Rs 25k floor did not cut the illiquid tail on this
sample, contrary to the concern raised in S3.

⚠ **The P&L improvement is in-sample and proves nothing** — every parameter
(cluster membership, rho, caps, crash2d's removal) was chosen from this data. The
RISK numbers are real, because vol targeting is an identity rather than a
forecast.

### 11b. Ordering: nothing beats random (Phase 6 first run)

166 of 221 candidates resolved, 32 sessions, 6 slots, 2,000 draws:

```
candidate pool    mean +0.128%          <- take-all ceiling
RANDOM baseline   mean +0.231%   90% band [+0.083, +0.383]

random      +0.360%  pctile 92.0  indistinguishable
composite   +0.235%  pctile 52.5  indistinguishable
tshock      +0.177%  pctile 28.2  indistinguishable
cap_score   +0.184%  pctile 30.1  indistinguishable
```

`composite` lands at the 52nd percentile — dead centre — independently confirming
the permutation p=0.69 consensus result by a different method. **Three orderings
have now been tested (conviction, composite, tshock/cap_score) and none beats
chance.** The unbiased hash stands.

The `random` row at the 92nd percentile is NOT evidence the hash is good: it is
one seed drawn from the distribution the baseline describes, and sits inside the
band by construction.

### 11c. Phase 5 REJECTED

```
cooldown_5   +0.229%  pctile 50.0  indistinguishable
cooldown_10  +0.053%  pctile  2.3  WORSE
```

The re-entry cooldown does not survive. The −0.921% vs +0.143% repeat-name
finding was tail noise across 16 symbols — which is exactly why it was
pre-registered rather than shipped. **Do not revive it without a new mechanism
and a fresh sample.**

### 11d. What remains true

The gate in S1 is UNMET. Removing crash2d moves the book from negative to
**neutral**, not to positive: the surviving three all have confidence intervals
crossing zero, and monthly stability is poor (July +0.587%, August −0.566%). The
risk framework now sizes the book sanely; it does not create an edge.
