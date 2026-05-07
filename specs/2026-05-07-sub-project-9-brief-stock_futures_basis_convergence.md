# §3.3 Brief: `stock_futures_basis_convergence_T1_expiry`

**Sub-project:** #9 (microstructure-first redesign)
**Status:** **DRAFT — DATA-BLOCKED. Awaiting APPROVE/REJECT/REVISE before any sanity work.**
**Date:** 2026-05-07
**Round:** 5

**Predecessors:** sub-9 microstructure-first-redesign (§3.3 gate); circuit_t1_fade_short (APPROVED template); round-3 feasibility (3-gate filter).

---

## 1. Asymmetry

**Name:** Stock-futures cash-settlement convergence on T-1 of monthly expiry, used as a **single-leg directional spot signal** (NOT a futures+spot spread — retail margin cannot run that).

NSE single-stock futures are cash-settled at expiry against a 15:00–15:30 IST spot VWAP (NSE F&O contract specs). On T-1 (day before last-Thursday monthly expiry), basis `(F − S)/S` must collapse to ≤5 bps by 15:30. If at 11:00 IST basis is materially wider than the symbol's 20-day median (**> +25 bps premium** or **< −15 bps discount**), arb desks force-close the leg over the next 4h. We trade the **residual spot drift** — premium → spot drifts up; discount → spot drifts down.

Source: https://www.nseindia.com/products-services/equity-derivatives-contract-specifications

## 2. Participants

- **Institutional cash-futures arb desks** capture the spread; by 11:00 T-1 they force-buy (premium) or force-sell (discount) spot to close the leg before settlement.
- **Retail cannot run the spread.** Zerodha/Upstox/Groww require separate margins per leg (no portfolio offset); SEBI peak-margin rules (Sep 2021) killed residual leverage.
- **Single-leg directional spot is unsaturated capacity:** ~zero retail watches futures basis; institutions are committed to the spread. Residual spot drift is unfilled capacity for a spot directional trader.

## 3. Persistence

1. **NSE-codified cash-settlement is regulatory.** Basis MUST converge to zero at expiry; only the time path varies.
2. **Capacity-constrained spread.** Stock-borrow (SLB) is illiquid for most F&O 200 names; peak-margin caps leverage. Residual dislocation persists into T-1.
3. **Retail-algo absent.** Stratzy / Wright / Religare / AlgoTest catalogues do NOT publish futures-basis convergence (verified May 2026). **Anti-edge signal absent — same property as the 2 round-1-to-4 winners.**

## 4. Evidence

- **NSE F&O contract specs:** https://www.nseindia.com/products-services/equity-derivatives-contract-specifications
- **Roll, Schwartz, Subrahmanyam 2007, *J. Financial Economics* — "Liquidity and the Law of One Price: The Futures-Cash Basis":** foundational global paper on basis convergence + arb capacity limits. https://www.sciencedirect.com/science/article/pii/S0304405X07000487
- **NSE Working Paper Series — cash-futures basis dynamics in Indian single-stock futures:** empirical T-2/T-1 convergence path. https://www.nseindia.com/research/research-publications
- **Negative confirmation:** Stratzy/Wright/Religare/AlgoTest do NOT list futures-basis-convergence. Per round-3 lessons, retail-algo absence is a positive signal.

## 5. Direction

**Bidirectional, sign-locked to dislocation:**
- Basis at 11:00 > median + **25 bps** → **LONG spot** (spot catches up upward).
- Basis at 11:00 < median − **15 bps** → **SHORT spot** (spot drops to meet futures).

Asymmetric thresholds: premium dislocations are more frequent (carry + retail-call-buying via futures); discounts are rarer and signal genuine selling, so tighter cutoff.

**Long-bias guardrail:** if sanity LONG-PF < SHORT-PF × 0.85, ship SHORT-only first.

## 6. Mechanic

`stock_futures_basis_convergence_T1_expiry`

1. **T-2 EOD prep:** 20-day rolling median basis per F&O 200 symbol from EOD bhavcopy futures-close vs spot-close.
2. **T-1 11:00 IST scan:** `basis_bps = (F − S)/S × 10000` from 5m bars; `dislocation = basis_bps − median_20d_bps`.
3. **Entry:** LONG if dislocation > +25 bps; SHORT if < −15 bps. Confirm: spot 5m bar 10:55–10:59 closes in trade direction; spot vol ≥ 1.0× 20-day intraday-vol-at-11:00 median.
4. **Stop:** entry × (1 ± **0.6%**) — locked. Tight because expected move (~20 bps) is small; 60 bps absorbs noise.
5. **Targets:** T1 = 1R (50% qty, breakeven trail); T2 = exit at 15:20 MIS auto-square.
6. **Latch:** one fire per (symbol, T-1). **target_anchor_type:** `arithmetic`.

## 7. Universe

F&O 200 (`assets/fno_liquid_200.csv`) ∩ near-month FUTSTK with `vol ≥ 500, oi ≥ 5000` on T-2. Excluded: index futures (different mechanic, no spot-leg analog at retail).

## 8. Active window

Triggers only on **T-1 of monthly expiry** (last-Wednesday-of-month under NSE last-Thursday calendar; ~12 sessions/year). Entry 11:00 IST single-bar, exit by 15:20 IST. Estimated n: 200 × 12 × ~12% post-gate ≈ **290 entries/year** → 2-year sanity yields ~580. **n ≥ 500 marginal but feasible.**

## 9. Risks / falsification (locked thresholds)

Retire if any of:
1. n < 500 over 2 years OR NET PF < 1.10 OR Sharpe ≤ 0.
2. |WR delta| > 10pp LONG vs SHORT without clean asymmetric story → spurious.
3. PF collapses outside ±5 bps band of (25/15) cutoffs → cell-overfit.
4. **Expiry-mechanic falsification:** if PF on non-T-1 days at same cutoffs ≥ 0.9× T-1 PF, the expiry-convergence mechanic is NOT the source of edge — retire.

## 10. Pre-coding sanity-check plan

**BLOCKED on data (§11).** Once data lands, `tools/sub9_research/sanity_stock_futures_basis_convergence.py`:
1. Last-Wed-of-month T-1 calendar 2023–2025.
2. **EOD-only first pass:** 20-day median basis from daily futures-close; use **T-1 close basis** as proxy for 11:00 basis (cheap test).
3. If EOD PF ≥ 1.10, escalate to intraday: futures 5m bars + basis at 11:00, simulate 11:00→15:20 spot trade with locked stop/targets.
4. NET PF using Indian fee model. Decision per §3.3.

## 11. Data engineering plan — **BLOCKED, NOT YET BUILT**

**Critical block:** repo has spot 5m feathers + daily spot consolidated, AND EOD F&O **options** bhavcopy under `data/option_chain/`. **No stock-futures pricing.** Inspection of `data/option_chain/2023/01/2023-01-02.parquet` confirms only `option_type ∈ {CE, PE}`.

**Buildable path:**

1. **EOD futures — LOW effort, ~1 day.** `tools/option_chain/_nse_bhavcopy_client.py` already downloads the full NSE F&O bhavcopy ZIP but filters out futures (line 192 keeps `OPTIDX/OPTSTK` legacy; line 195 keeps `STO/IDO` new). Adding `FUTSTK` (legacy) and `STF` (new) to the keep-list, writing to a parallel `data/futures_eod/<YYYY>/<MM>/*.parquet` store, captures stock-futures EOD prices for 2023–2025 immediately. Unblocks 20-day median basis + EOD-only sanity.

2. **Intraday futures 5m bars — MEDIUM effort, ~3–5 days.**
   - **Upstox F&O API historical (recommended):** free with our broker session; 40 req/s validated. Need FUTSTK→Upstox-instrument-key mapping per expiry month. ~200 sym × 36 months × ~24 sess × 75 bars ≈ 13M bars; ~250 MB feather. ~2-day ingestion + 1-day validation vs EOD close.
   - **Kite Connect historical:** already have Kite session in `broker/kite/`. Same magnitude.
   - **NSE Data Vendor (paid, ~₹5K/year):** only if free APIs insufficient.

**Recommended sequencing:**
1. Relax bhavcopy filter (~1 day) → 20-day median basis + **EOD-only sanity** using T-1 close basis as proxy.
2. **If EOD sanity PF < 1.10, candidate dies cheaply — do NOT invest in intraday ingestion.**
3. If EOD sanity PF ≥ 1.10, invest in intraday ingestion (3–5 days), re-run at 11:00 IST.

**Acceptance:** candidate is APPROVE-eligible because (a) the NSE-codified cash-settlement mechanic is named explicitly, and (b) a clear, low-cost first-step data path exists (~1 day) that gates whether deeper investment is justified.

---

## Decision required

- [ ] APPROVED — proceed: relax bhavcopy filter to ingest FUTSTK EOD, then run EOD-only sanity. Escalate to intraday only if EOD PF ≥ 1.10.
- [ ] REJECTED — reason.
- [ ] REVISE — specify what's missing / wrong.

Per sub-9 §3.3, no detector code is written until APPROVED **and** sanity-check PF ≥ 1.10.
