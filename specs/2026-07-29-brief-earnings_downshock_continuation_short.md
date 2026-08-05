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
- **Exit: the 15:15 print** (close of the 15:10-15:15 5m bar). **CORRECTED 2026-07-29 —
  the original "same-day close" was INFEASIBLE**: Zerodha auto-squares MIS equity from ~15:20
  (lifecycle Stage-1 note / lesson #4: Upstox-Angel 15:15, ICICI 15:15-20, Zerodha 15:20-24),
  and this is an MIS short, so the 15:29 leg cannot be held. 15:15 is fixed by BROKER
  MECHANICS, not chosen by performance — 15:20 scores higher (+0.650% vs +0.552%) and was
  deliberately NOT taken because it sits inside the square-off window (lesson #31).
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

## 9b. STAGE 4 RESULT (2026-07-29) — PASSED on real P&L with a feasible exit

`tools/sub9_research/sanity_earnings_downshock_continuation_short.py` (Discovery only,
n=220 after dedup → ADV tier → ProductionUniverseGate → ASM/GSM-clean → circuit-clean;
177 symbols, 97 sessions, canonical schema, no deviations, first run, zero tuning).

| basis (15:15 feasible exit) | net %/trade | PF | win | t |
|---|---|---|---|---|
| RAW gross | +1.007% | 2.19 | 65.9% | +4.45 |
| real Zerodha MIS fees only (**0.083%**) | +0.924% | 2.06 | 65.0% | +4.08 |
| **CENTRAL slippage 18.7bp/side (measured)** | **+0.552%** | **1.55** | 61.4% | +2.43 |
| CONSERVATIVE 27.5bp/side | +0.377% | 1.35 | 58.6% | +1.66 |

Per year: 2023 +0.617% PF 1.82 / 2024 +0.481% PF 1.37 (CENTRAL) — both positive, both tiers
positive (adv_low +0.520%, adv_mid +0.584%). **Break-even 46.5bp/side vs 18.1 measured.**
Falsifier #1 (+0.15% floor): CENTRAL is **3.7×** it, CONSERVATIVE **2.5×**. PASS.

**Cost structure (the key discovery):** real MIS-short fees are only **0.083%** of notional
(brokerage is ₹20-capped both legs) — an order of magnitude below the 0.31% flat assumption.
**Slippage is therefore the entire verdict**, which is why it was measured rather than assumed
(`tools/sub9_research/measure_slippage_earnings_downshock.py`, 5 methods triangulated on 1m
data, 220/220 matched): CENTRAL 18.7bp/side, CONSERVATIVE 27.5, STRESS 46.6. Spread is
tier-invariant (~11bp); **impact** separates the tiers (13.2 adv_low vs 5.2 adv_mid).
Model-free check: filling at real 5m-block VWAP is −5.4bp/side vs the assumed mark, so the
mark carries **no** optimistic bias.

**The exit correction cost nothing.** 15:15 vs the infeasible 15:29: gross +1.007% vs +0.993%,
net +0.552% vs +0.538% — the last 15 minutes contribute ≈ −1.4bp to this short, i.e. the
continuation drift is complete by 15:15 and the closing print adds nothing. Exit-leg slippage
is also slightly *cheaper* at 15:15 (16.1 vs 17.4bp): the close has more turnover but is 2.3×
more volatile, and the two effects net in our favour.

**What keeps this from being a clean pass:** the STRESS slippage case (39.8bp/side, spread =
full intra-bar half-range) gives +0.132%/trade, PF 1.11 — a **FAIL**. STRESS is a definitional
ceiling, not the pre-registered basis, but it is the honest upper bound on execution risk for a
post-earnings gap-down book. Weakest grid cell: 2024 CONSERVATIVE (PF 1.222, t 0.79).

**Execution requirement (not optional):** at ₹1L the order is ~25% of the entry *minute's*
turnover in adv_low (49% at the exit minute) — it must be **worked across the 5m block**, never
fired as a single-minute market order. A naive market order costs materially more than any
number above.

## 9c. PHASE-5 PRE-REGISTRATION (written BEFORE any Phase-5 run)

- **Fixed, NOT swept:** signal ≤−8%, de-dup, 09:20 entry, **15:15 exit** (broker mechanics),
  ProductionUniverseGate, ASM/GSM + circuit exclusions, measured per-tier slippage.
- **Cell dimensions:** ADV tier {low, mid, both} × shock depth {−8%, −10%, −12%} ×
  announce class {AMC, intraday, all}. Nothing else; no dimension added after seeing results.
- **Geometry sweep (the MFE p50 2.65% vs MAE p50 1.48% profile says there is room):**
  stop ∈ {none, 2%, 3%, 4%} × target ∈ {none, 1.5%, 2.5%}. **`none/none` (the Stage-4
  construction) is the incumbent and wins ties** — a geometry only displaces it on a strictly
  better result in BOTH eras, not on pooled numbers.
- **Lockable-cell rule (A5 + A5-b):** net expectancy > 0 on the ABSOLUTE statistic in BOTH eras
  at CONSERVATIVE slippage, n ≥ 100/era, pooled PF ≥ 1.20. Stability-first selection
  (smallest era gap at comparable PF), never top-PF-only.
- **Then:** one-shot demoted-window check (2025-01→2026-04), ledger-logged; freeze commit;
  fresh-pool one-shot + paper as the decisive gates (A1).

## 9d. PHASE-5 RESULT + **FREEZE** (2026-07-29)

**Sweep:** 324 combinations (27 cells × 12 geometries), all reported in
`reports/sub9_sanity/_earnings_downshock_phase5_cells.csv`.

**Eligible: 2 of 324, both inside ONE cell.** The binding gate was n ≥ 100/era — Discovery is
220 trades (115/105 by year), so only the un-subsetted cell can clear it (21/27 cells pass the
era-expectancy gate and 23/27 pass pooled PF ≥ 1.20). **The mechanism is broad; the sample is
not.** Stability-first selection was trivially satisfied (one eligible cell, year gap 0.081pp).

**LOCKED CELL = `both ADV tiers / −8% / all announce classes / geometry none-none` — i.e. the
Stage-4 construction, unchanged.** Discovery: CENTRAL +0.565%/PF 1.55/t 2.48; CONSERVATIVE
+0.396%/PF 1.36 (2023 +0.434%, 2024 +0.354%).

**No geometry displaced the incumbent.** Inside the locked cell, 0 of 11 non-incumbent
geometries are strictly better in BOTH eras; every stop degrades it (sl2 PF 1.36→1.07, sl3
1.11, sl4 1.17), every target degrades it (tp1.5 0.99, tp2.5 1.21), and 8 of 11 are pooled
net-NEGATIVE at CONSERVATIVE. The only 6 pairs that beat their incumbent sit in dead cells
(n=9-48) where the incumbent is negative. Disclosed bias: measured slippage is anchored on the
09:20/15:14 minutes so an intraday stop-out is charged the 15:15 rate — this **flatters**
geometries and never the incumbent, i.e. the bias points away from displacement.

**Demoted-window check (ONE SHOT, 2025-01-01 → 2026-04-30, cell unchanged):** n=219, slippage
re-measured on this cohort (wider: CENTRAL 19.3 / CONSERVATIVE 29.5 bp/side).
CENTRAL **+0.432%/PF 1.441**; CONSERVATIVE **+0.229%/PF 1.212**.
**PF gap 0.114 (CENTRAL) / 0.152 (CONSERVATIVE) — well inside the 0.30 overfit threshold**, and
the opposite of `pead_reaction_drift`, which died at this exact step with a gap of 1.78.
Robustness: re-fitting ADV terciles in-window instead of using frozen Discovery cuts gives
n=216, +0.234%, PF 1.216 — no dependence on the frozen cut.

**A5/A5-b now hold on the TRUE era split, on the absolute statistic, at CONSERVATIVE slippage:**
era_A +0.396%/PF 1.364 · era_B +0.229%/PF 1.212 — both net-positive.

**⚠ NAMED WEAKNESS (must not be buried):** the edge **decays within era_B**. 2025 (n=160) is
+0.327%/PF 1.310 conservative, but **2026 Jan-Apr (n=59) is net-NEGATIVE at conservative
(−0.037%, PF 0.969)** and the win rate falls 62.3% → 55.3%. The fresh-pool one-shot is
therefore **genuinely decisive, not confirmatory**.

### 🔒 FREEZE — declared 2026-07-29, this commit

Frozen: signal ≤−8% reaction move · de-dup (symbol, reaction_date) · entry T+1 09:20 ·
**exit 15:15** · universe low+mid ADV + ProductionUniverseGate + ASM/GSM-clean + circuit-clean ·
geometry none/none · real Zerodha MIS-short fees · measured per-tier slippage · order worked
across the 5m block. Cell lock:
`tools/sub9_research/earnings_downshock_continuation_short_cell_lock.json`.

**PRE-REGISTERED FRESH-POOL DECISION RULE (fixed here, before the run):** window 2026-05-01 →
present, ONE shot, no iteration, ledger-logged.
- n < 40 → **POWER-BLOCKED**: report counts only, compute no verdict, schedule the re-shoot for
  the month projected to reach 40 (the C-09 pattern). Do not burn the verdict.
- net ≥ **+0.15%/trade at CONSERVATIVE measured slippage** → **PASS** → detector build + paper.
- 0 to +0.15% → **MARGINAL** → hold, re-shoot at larger n; no detector work.
- < 0 → **KILL**.
Judged on CONSERVATIVE, not CENTRAL — the 2026 decay above is the reason to use the pessimistic
basis rather than the central one.

## 9e. FRESH-POOL ONE-SHOT (2026-07-29) — **PASS by 1.4bp. Fresh pool now BURNED.**

Window 2026-05-01 → 2026-07-24. Power gate: **n=77 ≥ 40 → proceed** (counted before any
outcome; frozen ADV cuts imposed, never re-fitted). 77 trades / 77 distinct symbols / 21
sessions. Guard bypass required an explicit `--i-am-burning-the-fresh-pool` flag naming freeze
commit `33d2046`; a second undocumented fresh-pool guard was found in the shared screen module
(it silently returned n=0 on the first run) and was made an explicit opt-in rather than deleted.

| basis | n | net %/trade | PF | win | t |
|---|---|---|---|---|---|
| GROSS | 77 | +0.847% | 2.13 | 61.0% | — |
| fees only (0.083%) | 77 | +0.764% | 1.97 | 61.0% | — |
| CENTRAL 18.7bp | 77 | +0.391% | 1.41 | 57.1% | +1.21 |
| **CONSERVATIVE 30.1bp — the verdict basis** | 77 | **+0.164%** | **1.15** | 53.2% | **+0.49** |

**VERDICT: PASS** (+0.164% ≥ the +0.15% pre-registered floor), applied mechanically.

**Slippage method deviation, disclosed:** 1m feathers end 2026-04, so the frozen 1m method is
not runnable on this cohort. Slippage was measured on 5m bars with identical impact formulas
and **calibrated against the 1m truth on the demoted cohort** (n=180 overlap, per-trade
correlation 0.66/0.74). Fresh-vs-demoted proxy ratio 0.995 central / 1.047 conservative — the
fresh tape is **not** tighter. Chosen before any outcome was computed.

### The disclosures that belong next to the PASS

1. **The margin is 1.4bp of expectancy / 0.7bp of slippage.** Net crosses the +0.15% floor at
   30.8bp/side against 30.07 measured. It crosses zero at 38.4bp. **t = +0.49 — not significant.**
2. **One month carries the entire result.** Per signal month (CONSERVATIVE):
   **2026-05 n=53 +0.983% PF 2.49 · 2026-06 n=17 −1.487% PF 0.19 · 2026-07 n=7 −2.034% PF 0.13.**
   **Ex-May = −1.647% on n=24.**
3. **Tail-fragile:** median +0.277% and a 3+3 trimmed mean of +0.243% are healthy, but dropping
   the single best trade gives +0.077% (MARGINAL band) and dropping the top 3 gives −0.073%
   (KILL band).
4. **Not a book:** one session (2026-06-02) holds 17 of 77 trades and −₹25,263, larger than the
   total net of +₹12,578. Production slot caps would never take 17 concurrent shorts. This is a
   per-signal ledger (lesson #30).
5. Not driven by a bad print: the one suspect CA row (SILVERTUC −88.5%) is a LOSER; excluding it
   *improves* the result to +0.203%.

### Standing interpretation (NOT a construction change)

The mechanism is earnings-driven and Indian results seasons are inherently seasonal (Q4/annual
≈ Apr-May, Q1 ≈ Jul-Aug, Q2 ≈ Oct-Nov, Q3 ≈ Jan-Feb). May is peak season (n=53); June is a dead
month (n=17); July is a season just starting and truncated at the 24th (n=7). **Whether the
negative months are decay or off-season noise is UNRESOLVED at n=24 combined.**
**Per lesson #31 this must NOT be turned into a seasonal filter** — proposing "trade only in
results season" after seeing this month split would be fitting to the window that revealed it.
If it is ever tested it requires its own pre-registration on data that did not generate the
observation.

### Status → PAPER, sized as a genuine test

The rule permits the detector build. The honest forward expectation this evidence supports is
**near zero, not the +0.39% central figure**. Paper is the decisive instrument (lesson #30:
paper > OCI > research) and must carry pre-registered kill criteria before it starts, sized
small. A conservative override (declining the PASS) is permitted by A5-b and was **considered
and not taken**, because unlike `xsec_momentum_demeaned` — declined for being net-NEGATIVE in
era_B — this candidate is net-POSITIVE in every window tested (era_A +0.396%, era_B +0.229%,
fresh +0.164%), merely declining. That distinction is the whole basis for the different call.

## 10. A1/A2 compliance

Development = 2023-01 → 2026-04-30 with the A5 era split. Freeze commit after Stage-5
cell-lock, BEFORE any statistic on 2026-05-01+. Decisive gate = fresh-pool one-shot + paper.
Every demoted/fresh evaluation logs a `docs/experiment_ledger.jsonl` line. Screens to date were
development-window Phase-1/2 and are exempt; the gauntlet likewise.

---

## §10 — V2 REFINED CONSTRUCTION (pre-registered 2026-07-31, NOT validated)

### 10a. What Stage-8 actually established

Production ran the frozen V1 construction through the real pipeline (OCI, real MIS-short
fees, ProductionUniverseGate, measured slippage) across three windows:

| window | n | mean/trade @27.5bp | PF | t |
|---|---|---|---|---|
| Discovery Apr-23 → Dec-24 | 150 | +0.373% | 1.350 | +1.34 |
| OOS 2025 | 129 | +0.305% | 1.306 | +1.04 |
| Holdout 2026 Jan → Jul-24 | 106 | +0.404% | 1.397 | +1.23 |
| **true out-of-sample (OOS+HO)** | **235** | **+0.349%** | **1.348** | **+1.60** |

Research replication is near-exact (Discovery +0.377→+0.373, OOS +0.327→+0.305; PF 1.35→1.350,
1.31→1.306). **Stage-8 parity is closed.**

Two structural findings travel with it:

1. **The V1 trigger was measured on the wrong price series.** Research computed the reaction
   move from LAST-5m-PRINT closes (`clean_daily_from5m`); production and live use OFFICIAL NSE
   closes (a last-30-min VWAP) via `consolidated_daily` / the Upstox daily API. The two disagree
   on ~21% of the population at the −8% boundary — yet expectancy is identical, so the edge is
   NOT knife-edge on the trigger definition. Official close is also the correct variable on
   mechanism grounds: it is the number every other participant reacts to.
2. **V1 is unholdable as a book.** 16/30 months positive, rolling-12m PF below 1.0 in 6 of 28
   windows, max drawdown −Rs28,567 and a **15-month underwater stretch** (Oct-2023 → Feb-2025)
   sitting in the middle of the sample, not the tail.

### 10b. The two V2 filters

| filter | config key | rule |
|---|---|---|
| reaction BAND | `shock_floor_pct: -12.0` | fire only for `reaction_move_pct ∈ [−12%, −8%]` |
| cap restriction | `allowed_cap_segments: ["small_cap"]` | small_cap only (was all five) |

Era-stability is the selection criterion, not pooled PF:

| bucket | DISC | OOS | HOLD |
|---|---|---|---|
| reaction −9..−8 | +0.87 | +0.38 | +0.51 |
| reaction −12..−10 | +0.52 | +0.14 | +0.82 |
| reaction −15..−12 | +1.33 | −0.55 | **−2.02** |
| reaction ≤−15 | −0.76 | +1.04 | +0.01 |
| small_cap | +0.45 | +0.54 | +0.86 |
| mid_cap | +0.30 | −0.91 | −0.20 |

**Mechanism.** −8% is a disappointment the retail holder base digests over T+1; ≤−12% is
capitulation that reverts — and it is precisely the cohort `panic_crash_revert_long` goes LONG
on (≤−7% deep illiquid crash → EOD snapback). Shorting it fights our own book and doubles the
capitulation-factor exposure the 2026-07 factor study flagged. The cap filter follows the stated
thesis: analyst-uncovered retail-held names digest slowly; mid_caps carry coverage and price the
news same-day, leaving no T+1 drift.

In-sample effect: n=210, +0.629%/trade, PF 1.77, t +3.10, 21/27 months positive,
max DD −Rs14,705 (vs −Rs28,567).

### 10c. What these numbers are NOT

**The filters were mined from the same data they improve.** ~17 bucket evaluations across
regime × reaction-depth × cap-segment. A crude Bonferroni at M=17 puts the bar near t≈3.0;
V2 sits at **3.10 — on the line, not clear of it** — and discards 45% of trades, so part of the
drawdown improvement is simply less exposure. **The V2 in-sample figures are not a forward
expectation.**

**Regime conditioning was tested and rejected as noise**: chop +1.17/−0.38/+1.04,
trend_up +0.21/−0.19/+1.11, trend_down +0.11/+1.09/−0.50. There is no "right regime" to trade
this in. Any future seasonal or regime filter needs its own pre-registration (lesson #31).

### 10d. Concurrency cap — derived, not inherited

The 35%/10 used for the Stage-8 parity run is retired. Across 3.3 years the V2 construction
**never exceeds 5 concurrent positions** (per-day max 5, p95 4, median 1; only 8 sessions exceed
3). At Rs 500k paper capital and Rs 1L notional with 5× MIS margin (Rs 20k blocked/position),
5 slots = Rs 100k = 20%. So `max_concurrent_positions: 5` / `capital_budget_pct: 20` truncates
**nothing** — no capped-subset expectancy correction is required for V2 — while halving V1's
capital footprint (V1 peaked at 8 concurrent / 32%).

### 10e. Pre-registered forward gate

Every archive window is spent: Discovery, OOS, Holdout and the original fresh pool have all been
evaluated. **No clean historical data remains.** Forward data from **2026-08-01** is the only
admissible test, and this section is committed before any August bar exists.

- **ONE SHOT** at **n ≥ 40** filtered trades, counted BEFORE any outcome is computed
- Expected ~4–5 trades/month post-filter → shot due **~Dec-2026**, after Q1 (Jul-Aug) and
  Q2 (Oct-Nov) results seasons
- Statistic: mean net %/trade at **CONSERVATIVE 27.5 bp/side**, real Zerodha MIS-short fees
- **PASS floor +0.15%/trade**, applied mechanically (same floor as the V1 fresh-pool one-shot)
- Scored on the **official-close** trigger — what production actually reads
- No re-filtering, no seasonal filter, no regime filter
- **If V2 fails, V1 is not resurrected as a fallback — the candidate retires.**

### 10f. Seasonality and reporting-wave position — TESTED, DELIBERATELY NOT FILTERED

Both were tested on 2026-07-31 (the config note required any seasonal filter to carry its own
pre-registration). Both are REAL in V1 and both are ABSORBED by the V2 filters. Neither is
added to the construction.

**Season phase** (peak = Feb/May/Aug/Nov, ramp = Jan/Apr/Jul/Oct, off = Mar/Jun/Sep/Dec):

| phase | V1 mean / PF | V1 eras D/O/H | V2 mean / PF | V2 eras D/O/H |
|---|---|---|---|---|
| peak | +0.557% / 1.58 | +0.63/+0.66/+0.34 | +0.638% / 1.82 | +0.41/+0.69/+0.99 |
| ramp | −0.013% / 0.99 | −0.58/−0.72/+1.12 | +0.817% / 1.84 | +0.44/−0.33/+2.45 |
| off  | +0.041% / 1.04 | −0.06/−0.16/+0.25 | +0.302% (n=12) | — |

**Reporting-wave position** (days since quarter-end anchor; SEBI LODR Reg 33 deadline = 45d):

| bucket | V1 mean / PF | V1 eras | V2 mean / PF | V2 eras |
|---|---|---|---|---|
| 21–35 mid | +0.132% / 1.12 | −0.54/−0.20/+0.91 | +0.894% / 1.93 | +0.50/+0.53/+1.84 |
| 36–45 deadline | +0.841% / 1.96 | +0.88/+0.93/+0.60 | +0.883% / 2.31 | +0.43/+1.23/+1.41 |
| 46–60 late | +0.113% / 1.10 | +0.22/+0.06/+0.04 | +0.159% / 1.16 | +0.34/−0.48/+1.04 |

**Why neither becomes a filter:**

1. **They are proxies, not causes.** V1's dead ramp months (−0.013%) and dead mid-wave bucket
   (+0.132%) both turn strongly positive under V2 (+0.817% / +0.894%). Off-peak periods simply
   held a worse MIX — more deep capitulation shocks and more covered mid_caps. Filtering the
   trade characteristic directly is strictly better than filtering the calendar slot it
   clustered in.
2. **The wave mechanism runs backwards.** If late filing were the tell, weaker/more retail-held
   late reporters should drift MORE. Measured drift is LESS (+0.159% vs +0.88%). A filter whose
   mechanism contradicts its measured sign is far more likely to be noise.
3. **Multiplicity.** Adding these would take the search to ~27 bucket evaluations, pushing the
   Bonferroni bar above the t=3.10 that V2 already only just clears, while cutting n by a
   further 22–29% and delaying the forward one-shot past Dec-2026.

**Monitored, not filtered.** Both dimensions are recorded here so the forward one-shot can
report them as a PRE-SPECIFIED SECONDARY analysis. They are descriptive only and CANNOT be
promoted to filters on the strength of the forward window that also scores the primary gate —
that would be the same re-mining this section exists to prevent. Promotion requires its own
fresh pre-registration and its own subsequent data.

### 10g. TAXONOMY INPUT CHANGE 2026-08-05 — market-cap backfill (declared, not silent)

**This changes the INPUT to the pre-registered V2 cap filter. It is recorded here
because the forward one-shot must be scored on a known taxonomy, not one that
shifted underneath it.**

**What was wrong.** `nse_all.json` derives `cap_segment` from `market_cap_cr`.
1,037 of 2,333 symbols (44%) had `market_cap_cr == 0.0` and fell through to
`cap_segment == "unknown"`. Verified: EVERY `unknown` symbol had exactly 0.0, and
no symbol with a real cap was `unknown` — so `unknown` was a **missing-value
sentinel, never a size class**. V2 admits `small_cap` only, so every in-band
candidate that happened to be unclassified was silently discarded.

The gap was **growing**, which is why it did not show up in validation:

| year | `unknown` share of this setup's trades |
|---|---|
| 2023 | 0.0% (0 of 60) |
| 2024 | 0.0% (0 of 90) |
| 2025 | 2.3% (3 of 129) |
| 2026 | **38.7% (41 of 106)** |

The two eras that established small_cap's edge contained essentially no `unknown`,
so the filter's measured benefit is unaffected — but forward it was discarding an
accelerating share of candidates. On 2026-08-05 the setup's ONLY in-band candidate
(BUTTERFLY, −9.78%) was dropped solely for a missing cap.

**Fix.** `tools/backfill_market_caps.py`. Chain, all public / no auth:
`NSE symbol → ISIN (EQUITY_L.csv) → BSE scripcode (local scrip master) → MktCapFull`.
`MktCapFull` is natively in ₹ crore — unit verified against Reliance
(17,38,931 = ₹17.4 lakh crore). Bands reverse-engineered from the already-classified
symbols, every cut point falling inside an observed gap with no overlap:
micro < 500 < small < 5,000 < mid < 20,000 < large.

**Applied 2026-08-05**, 648 of 1,037 resolved:

| | before | after |
|---|---|---|
| small_cap | 519 | **736 (+42%)** |
| micro_cap | 67 | 432 |
| mid_cap | 378 | 428 |
| large_cap | 332 | 348 |
| unknown | 1037 | 389 |

The residual 389 are NSE-only listings (SME/recent) with no BSE counterpart; they
remain `unknown` and stay excluded.

**Safety properties.** Only symbols *currently* `unknown` are touched — an existing
classification is never overwritten, so the validated population cannot move.
`nse_all.json.bak-2026-08-05` holds the pre-change file and
`data/cap_segments/market_cap_backfill_2026-08-05.json` holds every applied delta,
so the exact taxonomy in force on any date is reconstructible.

**How this must be treated at the one-shot.** The V2 in-sample figures (n=210,
+0.629%, PF 1.77) were computed with these 217 symbols classified `unknown`, i.e.
excluded. Forward they are *included*. This is a correction of a data defect, not a
loosening of the filter — the filter is unchanged and still `small_cap` only — but
the eligible population is ~42% larger than the one that produced the in-sample
number. Two consequences to carry into December:

1. **Do not read a higher forward trade count as edge decay or improvement.** The
   candidate pool genuinely widened on 2026-08-05.
2. **The recovered names skew smaller: median ₹919 cr vs ₹2,262 cr for the
   pre-existing small_caps.** They will land disproportionately in `adv_low`, where
   measured slippage is worst (18.7–30 bp/side against a 46.2 bp break-even). If the
   forward result underperforms the in-sample figure, the size-mix shift is the first
   hypothesis to test — before decay.
