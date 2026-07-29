# Brief: `earnings_downshock_continuation_short`

**Date:** 2026-07-29
**Stage:** 0 (idea) + 1 (Indian-market research)
**Family:** EVENT-DRIVEN SHORT continuation, intraday (T+1 09:20 → same-day close)
**Status:** DRAFT — Stage-0. Pre-Stage-4 gauntlet already PASSED (see §7).
**Lifecycle:** `docs/setup_lifecycle.md` incl. A1 (fresh-pool decisive), A5 + **A5-b** (single-leg
cash ⇒ era consistency on the ABSOLUTE statistic), A2 (ledger), A3 (hardened data).

---

## 1. Why this exists

Sole survivor of the 2026-07-29 ten-class event screen
(`tools/sub9_research/screen_event_classes_cost_clearing.py`), which was run because the
previous six candidates all died the same death: a real edge smaller than the ~0.31%
round trip. The screen ranked every event class on disk by *signed drift net of cost, in both
eras*. Nine classes failed. This one clears by ~2.5×.

Two contaminants were neutralised before any class was read, and both matter here: the daily
panel carries a structural **−0.18%/day open→close drift** (an unconditional short looks
profitable on raw returns), and event days carry an attention bias. All numbers below are
**abnormal returns** vs a same-date, same-ADV-tercile panel baseline; placebo on 35,337 random
symbol-dates = ±0.003%; the direction-matched activity control (prior day down + high volume,
no earnings) = **+0.04%** against this candidate's +0.48/+0.52.

## 2. Mechanism (one sentence)

When an Indian small/mid-cap reports earnings that the market punishes with a ≥8% down day,
the selling is not complete at that close — a retail-dominated, analyst-uncovered holder base
digests the news over the following session, so the stock continues to bleed intraday on T+1,
and that continuation is short-able for one session.

## 3. Indian anchors

- SEBI LODR Reg 33 fixes the disclosure event; announce-mix here is AMC 422 / intraday 305 /
  scheduled 8 / BMO 5, so the reaction day is well-defined and the T+1 session is genuinely
  post-information.
- Retail concentration + zero analyst coverage in the sub-₹5cr-turnover tier is why price
  discovery takes more than one session (the same holder-base argument that underpins the
  capitulation cluster, applied to a *dated* catalyst instead of an intraday crash).
- **The edge grows in era_B**, consistent with the 2026-07-28 era finding: the illiquidity
  premium flipped negative from 2025Q4, so weak illiquid names now underperform structurally.
  This is a candidate the new market structure *favours* — see the A5 structure-born note in §8.

## 4. Construction (LOCKED for Stage 4)

- **Signal:** earnings reaction-day return ≤ **−8%** (fixed threshold, NOT a full-sample
  decile). Reaction day derived from `announce_time_class` per the repaired
  `data/earnings_calendar/earnings_events.parquet`.
- **De-duplicate `(symbol, reaction_date)`** — MANDATORY, see §7.1.
- **Entry: T+1 09:20, the close of the first 5m bar** — NOT the 09:15 open print (§7.2).
- **Exit:** same-day close (single session, intraday MIS short).
- **Universe:** low+mid ADV tiers; exclude circuit-blocked opens; exclude any NSE ASM / BSE GSM
  listing on T+1; `ProductionUniverseGate` at Stage 4 (A3).
- Costs: full MIS round trip + per-trade slippage at Stage 4 (see §9 risk 1).

## 5. Evidence (abnormal-return basis, net of 0.31%; de-duplicated, tradeable subset, low+mid ADV)

| | n | NET | hit | t |
|---|---|---|---|---|
| era_A (2023-01→2024-12) | 232 | **+0.481%** | 63.8% | 3.92 |
| era_B (2025-01→2026-04) | 225 | **+0.516%** | 63.7% | 3.67 |

Per year: 2023 **+0.427** · 2024 +0.541 · 2025 +0.410 · 2026(→Apr) +0.806. **Positive every
year, near-identical across eras** (A5 satisfied on the absolute statistic per A5-b).
Break-even round-trip cost **0.791% (era_A) / 0.826% (era_B)** vs 0.31% assumed — ~2.5× headroom,
the margin that nine other classes lacked. Drift is **monotone in shock depth**
(−4% +0.00/+0.16 → −6% +0.16/+0.35 → −8% +0.28/+0.72 → −10% +0.47/+0.50), so −8% is not a
fitted cliff. 528 symbols, top-5 share 3.1% — no concentration. ~150-210 events/yr.

## 6. Falsifiers (pre-registered)

1. **Cost falsifier (the one that killed the last six):** if realistic Stage-4 fills — real MIS
   fees + measured slippage on low-ADV names — push net expectancy below +0.15%/trade in
   EITHER era, kill. The abnormal-return screen is an upper bound; only a real fill simulation
   settles this.
2. **Mechanism falsifier:** the drift must remain monotone in shock depth and must be
   *stronger* in lower-ADV tiers. If it concentrates in liquid names, the
   slow-diffusion/uncovered-holder story is wrong regardless of PF — kill.
3. **Shortability falsifier:** if the genuinely short-able subset (point-in-time MIS + no
   broker intraday-short ban + no borrow constraint) is materially smaller than the ~98%
   current-snapshot estimate, and the edge does not survive on it, kill. (This is the
   `asm_gsm_stage_transition` failure mode; here T2T is 0% and ASM is 7-9%, but the MIS list
   used was a current snapshot — lesson #27.)

## 7. Pre-Stage-4 gauntlet — ALREADY RUN (2026-07-29), PASSED

`tools/sub9_research/pretest_earnings_downshock_tradability.py` +
`reports/sub9_sanity/_earnings_downshock_tradability{,_events}.csv`.

**7.1 De-duplication (mandatory correction).** The screen counted standalone + consolidated
filings for the same company-session as separate events: 835 rows → **740 unique**. era_A's
pitched +0.183% (n=446) is **+0.057% (n=380)** on a clean set — i.e. zero. era_B unaffected.
Do not carry 446/389 anywhere.

**7.2 Entry-timing (mandatory correction).** Decomposing the open→close short shows the
09:15→09:20 bar is noise that **flips sign between eras** (era_A −0.191, era_B +0.196) — it was
the source of the apparent era instability, not the edge. The 09:20→close leg is stable
(+0.791 / +0.826). Entering at 09:20 raises era_A from +0.296 to **+0.481**, cures the flat
2023 (+0.041 → +0.427), and makes the eras near-equal. Entering at the 09:15 auction print in
low-ADV names was never executable anyway.

**7.3 Circuit blocking — NOT a killer:** 0.26%/0.28% strict, 1.05%/0.56% loose. T+1 opens flat
vs the reaction close (median +0.32%/+0.26%), so these names bleed over a full session and
open unlocked. 5m coverage of entry sessions 740/740.

**7.4 Surveillance — 9.2%/6.7% ASM, and 0.00% Stage III/IV (T2T) in BOTH eras.** The ASM/GSM
failure mode does not repeat (26% T2T there). Excluding surveillance names *improves* era_A —
the ASM subset was its loss centre (n=35, −1.09).

**7.5 MIS coverage 97.1%/98.1%** — reported as an upper bound only (current snapshot applied
to history; anachronistic + survivorship-biased, lesson #27). Applying it costs ~1pp of n and
~0.03pp of expectancy.

**7.6 `panic_crash_revert_long` conflict: ZERO.** Config-faithful replay fires on 0/380 era_A
and 0/360 era_B T+1 sessions (3.2%/1.7% on the reaction day, which we do not trade). Near-
disjoint by construction: panic needs a −7%-in-15-min collapse; this is a ~1%/session drift.

**7.7 Data-repair independence.** era_A identical to 4 decimals on pre- and post-repair
parquets; era_B moved 0.001pp. **No finding depends on the 2026-07-28 earnings repair.**

**7.8 Execution realism.** `entry_date > signal_date` for 740/740, median gap 1 session. The
overnight gap is flat, not spent — only 34.5%/38.1% gap down.

## 8. Adjacent setups + factor position

Orthogonal to everything live: the book is one capitulation-reversion LONG factor plus
close_dn (2026-07-27 factor study). This is a **short**, event-dated, single-session — the
exposure the study said was missing after the intraday shorts were retired. Zero overlap with
`panic_crash_revert_long` (§7.6). Distinct from the dead PEAD drift (killed twice, incl. on
repaired data): that traded *small* surprises over 3-10 sessions on the long side; this trades
*large* down-shocks for one session on the short side. **A5 structure-born note:** the era_B
strength is consistent with the post-2025Q4 illiquidity-premium flip, but the candidate does
NOT claim the exemption — it is era-consistent on its own (2023 +0.427), so it validates
conventionally.

## 9. Open risks the brief owns

1. **This is an abnormal-return screen, not a P&L simulation.** Stage 4 must re-run through the
   real machinery — MIS fees, `ProductionUniverseGate`, per-trade slippage — per the lifecycle,
   NOT another pooled `_tmp_` screen (lessons #3 / #28b). At a pessimistic 50bp slippage the
   headline roughly halves; the 2.5× break-even headroom is what makes that survivable, but it
   must be measured.
2. Short availability beyond MIS (borrow/SLB, per-broker intraday-short bans) is **untested**.
3. The universe is confined to the 5m archive (~1,451-2,574 symbols); the event population was
   born inside it.
4. era_A is the weaker half on every cut before the 09:20 fix and remains the thinner one after.

## 10. A1/A2 compliance

Development = 2023-01 → 2026-04-30 with the A5 era split. Freeze commit after Stage-5
cell-lock, BEFORE any statistic on 2026-05-01+. Decisive gate = fresh-pool one-shot + paper.
Every demoted/fresh evaluation logs a `docs/experiment_ledger.jsonl` line. Screens to date were
development-window Phase-1/2 and are exempt; the gauntlet likewise.
