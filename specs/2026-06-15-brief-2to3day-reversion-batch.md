# Brief: 2-3 Day CNC/MTF Reversion Batch (Stage 0-1)

**Status:** Stage 0-1 (idea + Phase-1). Three sibling candidates to validate together, then paper as a batch alongside A2 (`mtf_capitulation_revert_long`).
**Date:** 2026-06-15
**Lifecycle:** `docs/setup_lifecycle.md` Stages 0→6, daily-cross-sectional CNC/MTF variant (same machine as A2: C1 ranker → C2 panel → C3 store → C4 executor).
**Horizon constraint (user):** hold 2-3 days only (MTF utilisation).

---

## Why these three (and not momentum/PEAD/lead-lag)

Established this session (evidence, then properly: see lessons #28/#28b): drift/continuation edges (momentum, PEAD, breakout) need ~weeks to pay and are market-beta-exposed; at the **2-3 day horizon the only mechanism that resolves fast enough is over-extension reversion** (the system's proven family: `gap_fade_short`, `panic_crash_revert_long`, `up_spike_fade_short`, and A2 at the CNC/MTF horizon). These three are **distinct reversion TRIGGERS** (not A2 re-skinned — different signal definitions), validated together to diversify *how* the bounce is detected.

## The three candidates

### C1 — `low52_capitulation_revert_long`
- **Mechanism (one sentence):** An illiquid MTF-eligible name trading **at/near its trailing-252-day low on a turnover shock** is in terminal capitulation into a liquidity vacuum; absent institutional bid support, the over-shoot snaps back over 2-3 sessions.
- **Distinct from A2:** A2 triggers on *recent 5-day % drop* (rate of decline); C1 triggers on *price level* (proximity to the 1-year low) — a stock can be near its low without a sharp recent drop, and vice-versa.
- **Falsifiers:** (1) mechanism — if it's terminal decline not capitulation, near-low names keep making new lows (no bounce); (2) regime — needs volatility; dead in low-vol consolidation; (3) infra — if MTF leverage on tier-1 names is withdrawn, capital ROI collapses.

### C2 — `downstreak_revert_long`
- **Mechanism (one sentence):** A name printing **≥3 consecutive down days on a turnover shock** has exhausted the impatient-seller cohort; mean-reversion of the streak resolves over 2-3 sessions.
- **Distinct from A2:** streak *count* (path) vs A2's cumulative *magnitude*; a 3-day grind-down can be a small cumulative move (would miss A2's bottom-decile cut) yet still exhaust sellers.
- **Falsifiers:** (1) streaks reflect informed selling that continues (no bounce); (2) regime — trending-down regime extends streaks; (3) shock filter removes the edge (quiet grind-downs continue).

### C3 — `crash1d_revert_long`
- **Mechanism (one sentence):** A **single-session deep drop (bottom-decile 1-day return) on a turnover shock** is a panic flush; the panic-crash bounce (system's live `panic_crash_revert_long` thesis) expressed at the daily/CNC horizon resolves in 2-3 days.
- **Distinct from A2:** single-day shock vs A2's 5-day cumulative; isolates the acute flush from the slow bleed.
- **Falsifiers:** (1) the 1-day drop is news-driven (continues, like the killed earnings-down-bounce); (2) regime; (3) shock filter.

## Phase-1: Indian-market basis (≥2 sources each)

1. **System's own LIVE validated reversion edges** — `gap_fade_short`, `panic_crash_revert_long`, `up_spike_fade_short` (OCI-validated, in production). Same universe (illiquid NSE), same over-extension→reversion family. Strongest possible precedent: production P&L.
2. **A2 `mtf_capitulation_revert_long`** — confidence-card PASS at this exact CNC/MTF 2-3d horizon (PF 1.356 [1.17,1.60], 6/7 regimes, Harvey-Liu adj-Sharpe 0.80 at M=540). Direct horizon precedent.
3. **Literature/practitioner:** Jegadeesh (1990) short-term reversal — strongest in small/illiquid; intradaylab.com gap-down-recovery write-ups; the capitulation/liquidity-vacuum mechanism (thin book → over-shoot) is the documented Indian-illiquid microstructure feature.

## Phase-1: Data feasibility (Gate B)
- **On disk:** `cache/preaggregate/clean_daily_from5m.feather` (CA-adjusted, bad-print-clean, 2023-01-02..2026-04-30 — the production-faithful daily panel, NOT `consolidated_daily`); MTF approved-list snapshot (`data/mtf_universe/approved_mtf_securities_2026-05-21.json`). No earnings/CA data needed. **Feasible.**
- **Universe (production-faithful):** MTF-eligible (exclude ETF) ∩ ADV ≥ ₹20L ∩ ADV-tier-1 (most-illiquid quintile) — identical to A2's `CrossSectionalRanker` universe.

## Phase-1: Regulatory sensitivity
- CNC/MTF **delivery** (not MIS, not F&O) → SEBI F&O Oct-2025 changes do NOT apply. STT delivery 0.10%/side (round-trip 0.20%); MTF interest ~0.04%/day on borrowed; leverage ~3x on tier-1. Apr-2026 STT hike is F&O-only → no impact.
- **Survivorship caveat (Lesson #27):** the 2026 MTF list applied to 2023-25 is anachronistic; no point-in-time MTF list exists. **Paper is the production-faithful gate** for all three (same as A2).

## Falsifier shared across batch
If, after proper Stage-2→6, the demeaned (market-relative) bounce is NOT positive across Discovery/OOS/HO regimes, OR net-of-real-MTF-fees fails the confidence-card CI lower-bound > 1.0, the candidate is killed (no salvage mining — Lesson #2).

---

## Decision (Stage 0 gate)
- [x] Proceed all three to Stage 2 (Phase-2 signature, Discovery-only).

---

## RESULTS (Stages 2-6, 2026-06-15) — real lifecycle, production universe, real delivery fees

**Stage 2 (Phase-2 signature, Discovery-only, market-relative footprint):** all three PASS the 0.1% gate.
| | hold-3 delta vs tier-1 universe | win | n |
|---|---|---|---|
| C1 low52 | +2.21% | 63% | 255 |
| C2 downstreak | +1.05% | 57% | 981 |
| C3 crash1d | +0.42% | 50% | 1428 |

**Stage 4-5 (Discovery cell-lock → one-shot OOS 2025 → one-shot HO 2026, net of real fees):**

- **C1 `low52_capitulation_revert_long` — PASS (borderline).** 58 cells swept, 10 net-positive in ALL 3 periods. Most-stable (min ΔPF): **low52 ≤ 5% × K=2 × tier-1 × shock ≥ 2×**, PF Disc/OOS/26 = **1.86 / 1.08 / 1.39**.
- **C2 `downstreak` — KILL.** 42 ship-eligible on Discovery but **0 net-positive in all 3 periods**; every cell's 2026 HO PF is 0.44–0.71 (Disc+OOS-strong → HO-collapse, lesson #10). Regime-dependent; died in 2026.
- **C3 `crash1d` — KILL.** Only 7 ship-eligible; **fails OOS one-shot** (2025 PF < 1.0). Also a duplicate of A2's existing N=1 mine cell (not a distinct mechanism). Fold into A2, don't ship.

**Stage 6 (C1 confidence card, Discovery-only CI + one-shot validators, true M=58):**
- BCa PF **1.86 [1.41, 2.44]**, expectancy ₹570/trade [308, 857], win 47.7%.
- Harvey-Liu adj-Sharpe **0.93 at M=58** (42% haircut) — positive.
- Per-regime: strong `pre_election_calm` (2.53) / `election_vol_spike` (3.01); **weak/negative `fed_pivot_china_rotation_FII_exit` (0.32)** and `post_election_consolidation` (0.91). **Regime-conditioned — needs volatility, dies in FII-exit.**
- One-shot: OOS 2025 **1.08** (marginal, just under the informal 1.10 gate), HO 2026 **1.39** (positive in the down market).

**Verdict:** C1 is a *real, borderline* candidate — strong Discovery + positive 2026 HO, but OOS marginal and regime-conditioned. Worth paper-validating as a distinct-trigger complement to A2 (paper is the production-faithful gate per risk #2). C2/C3 killed at the proper gates — and notably my earlier toy-screen had falsely flagged downstreak as OK; the real cell-mine with delivery fees + the 2026 HO exposed the collapse.

**Batch for paper:** A2 (`mtf_capitulation_revert_long`) + C1 (`low52_capitulation_revert_long`). Two distinct reversion triggers; C2/C3 dropped.
