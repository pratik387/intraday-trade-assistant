# Sub-project #9 — Microstructure-First Setup Redesign

**Status:** Draft. Captures the systemic diagnosis from 11 setup failures across sub-projects #5/#7/#8 and locks in the corrected research process for any future setup additions.

**Date:** 2026-05-01

**Branch:** `feat/premium-zone-ict-fix`

**Predecessor diagnoses:**
- specs/2026-04-25-sub-project-5-gauntlet-v2-postmortem.md (named the cargo-cult problem)
- tasks/lessons.md 2026-05-01 entries (period scheme + variant shopping)

---

## 1. The 11-failure pattern

Across sub-projects #5/#7/#8, eleven detectors have failed gauntlet/validation:

| # | Detector | Sub-project | Origin |
|---|---|---|---|
| 1 | order_block (long+short) | #5 | ICT/SMC (US/forex literature) |
| 2 | premium_zone | #5 | ICT/SMC |
| 3 | vwap_lose_short | #5 | Generic VWAP play |
| 4 | mis_unwind_short | #7 | Generic 3:20 PM mean-revert (thesis right, mechanic wrong) |
| 5 | closing_hour_reversal | #7 | Generic closing-bar reversal |
| 6 | narrow_cpr_breakout | #7 | CPR pivot — universal |
| 7 | vwap_first_pullback | #7 | VWAP pullback — universal |
| 8 | orb_15 | #8 | Opening Range Breakout (Toby Crabel, US) |
| 9 | pdh_pdl_reject | #8 | Level rejection — universal |
| 10 | pdh_pdl_sweep_reclaim | #8 | ICT-style sweep+reclaim |
| 11 | gap_and_go_continuation | #8 | Long-bias gap continuation (the LOSING flow per SEBI FY23) |
| 12 | ema5_alert_pullback | #8 | Subasish Pani 5-EMA — generic trend-pullback |
| 13 | camarilla_l3_reversal | #8 | Camarilla pivots — universal |

**Survivors:**

| Detector | Status | Why it works |
|---|---|---|
| `gap_fade_short` | ✅ validated | Indian-specific asymmetry: small-cap retail-driven gap-up + structural opening-window exhaustion + SHORT-only (alignment with 70%-of-traders-lose-on-longs) |
| `expiry_pin_strike_reversal` | not yet tested | Indian-specific asymmetry: F&O gamma-pin on NIFTY expiry days. OI data infrastructure broken on last OCI run; pending re-capture |

**The pattern is unambiguous:** every survivor (or potential survivor) exploits an Indian-specific microstructure asymmetry. Every failure is a universal/cargo-culted pattern.

## 2. Diagnosis: research-phase failure

Sub-project #5's postmortem already named the root cause:

> *"Setup library is cargo-culted from US/forex SMC literature. ICT/SMC patterns originate in 24/5 forex with no MIS, circuits, or asymmetric short rules. Indian equity has session opens, T+1 settlement, gap risk, circuit halts, and asymmetric short rules — the structural assumptions break."*

> *"Recommended next action: Open a new sub-project scoped to redesign the setup library for Indian intraday market structure rather than continue gauntlet-style optimization over the existing library."*

Sub-project #7/#8 was meant to be that redesign. What actually happened:
- The research process was **"find an Indian-published setup → implement it"** (Subasish Pani 5-EMA, Sahi.com sweep+reclaim, Camarilla pivots, NSE-trader-published ORB variants).
- Indian-published patterns are mostly translations of US/forex patterns, used by retail traders.
- **Retail usage ≠ institutional edge.** SEBI's FY23 study: 70% of cash intraday traders lose, 93% of F&O traders lose. The patterns retail uses are the patterns retail loses on.
- Sub7/8 added 1 genuinely Indian-microstructure-driven setup (`expiry_pin`) and 7 cargo-culted ones. The cargo-culted ones died on validation, exactly as the postmortem predicted.

**The gap:** "Indian-published" was conflated with "Indian-edge". Adding more Indian-published patterns produces more cargo-cult failures. The fix isn't more research depth on existing pattern libraries — it's a **different kind of research**.

## 3. The corrected research process: microstructure-first

For any new setup addition under this sub-project:

### 3.1 Start from asymmetry, not pattern

**Wrong:** "There's a 5-EMA pullback pattern Subasish Pani teaches; let's implement it and see if it works."

**Right:** "MIS auto-square at 3:20 PM creates asymmetric net-sell pressure because retail intraday flow is overwhelmingly net-long. What setup harvests this specific asymmetry?"

The unit of work is **the asymmetry**, not the pattern. A pattern is one of many possible mechanics for harvesting an asymmetry; designing the pattern without the asymmetry is cargo-cult.

### 3.2 Required asymmetry attributes

Before designing a setup, the asymmetry must satisfy ALL of:

1. **Indian-specific** — derives from Indian market structure, regulation, or participant mix. Universal patterns (ORB, level rejection, pivots, gap-continuation) are forbidden as the PRIMARY thesis. They can be MECHANICS used to harvest an Indian-specific asymmetry.
2. **Identifiable participants** — name who is on each side of the trade. "Retail vs institutional" is acceptable; "buyers vs sellers" is not.
3. **Persistence rationale** — name the regulatory, behavioral, or structural reason the edge persists. "It's been there historically" is not enough; "MIS auto-square is a SEBI rule that won't change" is.
4. **Prior evidence (academic / regulatory / institutional)** — at least one source independent of retail trading communities. Zerodha Varsity, SEBI publications, QuantInsti EPAT projects, or peer-reviewed Indian-equity microstructure papers count. Subasish Pani / TraderCarnival / MyAlgomate alone don't count.
5. **Asymmetric direction** — the edge must be unambiguous on direction (long-only or short-only). Per SEBI FY23 + the gauntlet-v2 postmortem, long-bias setups in Indian intraday systematically lose. Two-sided setups inherit the long-side losses unless filtered.

### 3.3 Setup-design quality gate

A new setup is admitted into the library only after a one-page brief that answers:

```
Asymmetry: [name + Indian-specific source]
Participants: [who's on each side]
Persistence: [why edge persists]
Evidence: [≥1 source from §3.2 list]
Direction: [long-only / short-only / why both]
Mechanic: [how the setup harvests this asymmetry]
Universe: [intended cap segments + symbols + WHY these]
Active window: [time-of-day + WHY this window]
```

The brief is reviewed by the user before any code is written.

### 3.4 What's allowed without re-vetting

The 2 surviving setups stay validated:
- `gap_fade_short` — already on production
- `expiry_pin_strike_reversal` — pending re-capture with OI fix; no design changes needed

No other setup is allowed in the library until the §3.3 brief passes user review.

## 4. Candidate asymmetries inventory (initial)

These are CANDIDATES to research further. Each requires the §3.3 brief before becoming a setup. Listed in roughly decreasing prior-evidence strength.

| # | Asymmetry | Source | Direction | Already attempted? |
|---|---|---|---|---|
| A | F&O expiry gamma-pin (NIFTY weeklies, monthlies) | OI dynamics, SEBI publications | both (mean-revert) | ✓ expiry_pin (pending OI fix) |
| B | Small-cap opening-window retail-momentum exhaustion | empirical (gap_fade_short) | short | ✓ gap_fade_short (validated) |
| C | MIS auto-square 3:00-3:20 PM forced selling | SEBI rule + retail-flow asymmetry | short | ✗ mis_unwind_short failed; thesis right, mechanic wrong; worth retry with different mechanic |
| D | F&O ban-list crossing (95% MWPL → margin shock) | SEBI rule | likely short on entry, long on exit | ✗ not attempted |
| E | Circuit-band hit + recovery dynamics | SEBI rule | depends on direction | ✗ not attempted |
| F | FII/DII flow asymmetry (foreign vs domestic positioning) | NSE published flow data | depends on flow regime | ✗ not attempted |
| G | Index rebalancing (NIFTY/BankNifty quarterly) | NSE rebalance schedule | depends on inclusion/exclusion | ✗ not attempted |
| H | Block deals / bulk deals (large reported transactions) | NSE published deals data | depends on deal direction + size | ✗ not attempted |
| I | T+1 settlement gap dynamics (overnight risk asymmetry) | SEBI rule | short on next-day gap-up | already partly inside gap_fade_short |
| J | Pre-market call-auction → opening tick mismatch | NSE pre-market mechanics | likely small-cap-specific | already partly inside gap_fade_short |

This inventory is a starting list. Items D-H are most-promising for substantive future work because they exploit specifically-Indian regulatory/structural mechanics with no obvious US/forex analog.

## 5. Disposition of failed setups

Per master plan §2.5, the 6 sub7/8 candidates that failed validation are set to **DISABLED**:
- orb_15
- pdh_pdl_reject
- pdh_pdl_sweep_reclaim
- gap_and_go_continuation
- ema5_alert_pullback
- camarilla_l3_reversal

`enabled: false` in `config/configuration.json`, audit decision documented in each detector's plan doc.

The 4 sub7-era failed setups (mis_unwind_short, closing_hour_reversal, narrow_cpr_breakout, vwap_first_pullback) are already removed from the codebase; no further action needed.

## 6. Period accounting (2023-24 = validation, spent)

Per the user's 2026-05-01 correction: 2023-24 OCI capture is **validation territory** for sub7/8 (since sub-project #5 already used it as Discovery for the SMC library and exhausted it).

The 5 gauntlet iterations on this validation slice spent the validation budget multiple times. Per OOS discipline, **the 2023-24 validation slice is now spent for any sub-project #7/#8 successor library**.

Any future setup library requires fresh validation data. The next OCI capture should target 2025-04 onward (FY25-26) — that becomes the new validation period.

## 7. Definition of Done for this sub-project

Sub-project #9 is complete when ALL of:

1. ✓ This spec committed (the diagnosis + corrected process is now project-state)
2. The 6 failed sub7/8 detectors set to `enabled: false` with audit decisions
3. Candidate asymmetry inventory (§4) reviewed by user with ≥1 asymmetry shortlisted for §3.3 brief
4. ≥1 §3.3 brief passes user review (this is the gate to start sub-project #10 or whatever the next library iteration is named)
5. tasks/lessons.md captures the 11-failure pattern as a permanent lesson

## 8. What this sub-project explicitly does NOT do

- Does NOT add any new detector to the library
- Does NOT fix the failed detectors
- Does NOT redesign the gauntlet
- Does NOT modify the OOS protocol

This sub-project's deliverable is **research-process discipline**, not code.
