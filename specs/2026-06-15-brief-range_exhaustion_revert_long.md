# Brief: `range_exhaustion_revert_long` (Stage 0-1)

**Status:** Stage 0 (idea). Candidate #5 for the 2-3 day CNC/MTF reversion batch (after A2 trailing-5d-loser, C1 low52, C4 z-score; C2 downstreak / C3 1d-decile killed).
**Date:** 2026-06-15
**Lifecycle:** `docs/setup_lifecycle.md` Stages 0→6, daily-cross-sectional CNC/MTF variant. Reuses the C1-C4 machine via a new `selection_mode`.
**Horizon:** 2-3 day hold (user constraint, MTF utilisation).

## Mechanism (one sentence)
An MTF-eligible tier-1 illiquid name that prints a **climactic wide-range down day** — daily range `(high − low) / prev_close` in the cross-sectional **top decile** AND **close < open** (closed weak, sellers in control) **on a turnover shock** — has experienced a single-session selling-climax flush in a thin book; the over-shoot from forced/panic liquidation snaps back over 2-3 sessions once the climactic seller is exhausted.

## Why distinct (not A2/C1/C4 re-skinned)
The four tested triggers key on the **close-to-close move** in different forms: raw 5d %-return (A2), price *level* vs 252d low (C1), close vs 20d band in σ units (C4) — and the two killed ones on 1-day-decile (C3) / consecutive-down-count (C2). **None of them look at the intraday RANGE (the bar's high-low spread).** Range exhaustion selects on **how violent the session was**, not how far the close fell:

- A wide-range day can have a **modest net close-to-close move** (gap down, intraday plunge, partial recovery → close still red but only −1 to −2%) — it would **miss A2's bottom-decile/large-%-cut and C4's σ-cut**, yet the wide bar marks a genuine intraday selling climax.
- Conversely a name can post a large clean close-to-close drop on a **narrow orderly bar** (a controlled markdown, no climax) — captured by A2/C3/C4 but **not** by range-exhaustion.
- The `close < open` requirement is a separate intraday-direction filter none of the others use (they are all close-to-prev-close).

So C5 selects a different cohort — the *climax/capitulation-bar* names rather than the *cumulative-decliner* names → genuine diversification of "how the bounce is detected." This is the same over-extension→reversion family as the system's LIVE `panic_crash_revert_long`, but expressed via the **daily bar's range** at the CNC/MTF horizon, which `panic_crash` (intraday/MIS) does not encode at the daily level.

## Fallback candidate (if C5 weak): C6 `crash2d_revert_long`
2-day cumulative return in the cross-sectional **bottom decile** + turnover shock → 2-3d revert. Distinct by **horizon-of-formation**: sits *between* C3's 1-day decile (killed: too acute, news-driven, duplicate of A2's N=1 cell) and A2's 5-day cumulative (slow bleed). A 2-day flush captures a short, sharp multi-session panic that a 1-day window cuts off mid-move and a 5-day window dilutes. Only tested if C5 fails Stage 2 (<0.1% delta) or dies in the cell-mine.

## Phase-1: Indian-market basis (≥2 sources)
1. **System's own LIVE validated reversion edges** — `panic_crash_revert_long`, `gap_fade_short`, `up_spike_fade_short` (OCI-validated, in production) + A2/C1 (confidence-card PASS at this exact CNC/MTF 2-3d horizon). Same universe (illiquid NSE), same over-extension→reversion family. The wide-range selling-climax is the canonical "capitulation bar" these fade/bounce on.
2. **Literature/practitioner:** the **selling-climax / wide-range reversal bar** is a documented technical pattern (Wyckoff selling-climax; classic "climactic volume + wide spread" reversal in Indian retail TA — Zerodha Varsity volatility/candlestick chapters; intradaylab.com gap-down-recovery write-ups). Jegadeesh (1990) short-term reversal is strongest in small/illiquid names where the thin book lets the intraday over-shoot exceed fair value. The wide-range bar + close-weak is the price footprint of forced/panic liquidation into a vacuum.

## Phase-1: Data feasibility (Gate B)
On disk: `cache/preaggregate/clean_daily_from5m.feather` (CA-adjusted, bad-print-clean, 2023-01-02..2026-04-30) has daily OHLC → `range = (high − low) / prev_close`, `close < open`, cross-sectional decile rank per day all computable. MTF snapshot (`data/mtf_universe/approved_mtf_securities_2026-05-21.json`) for eligibility. **Feasible**, no new data.

**Bad-print caveat (MEMORY):** wide-range is the dimension MOST exposed to a single bad print (a spurious high/low spike inflates range and would fake-trigger). The data file is the CA-adjusted, bad-print-cleaned `clean_daily_from5m.feather` (built precisely to remove this); the cross-sectional top-decile cut on a per-day basis further bounds the tail. Stage-2 will report `df.shape` + the trigger's symbol concentration to confirm it is not a handful of bad-print names (Lesson #26 / data-cleaning lesson).

## Phase-1: Regulatory sensitivity
CNC/MTF **delivery** (not MIS, not F&O) → SEBI F&O Oct-2025 changes do NOT apply. STT delivery 0.10%/side (round-trip 0.20%); real cost modelled = `0.00347 + 0.0020` (Zerodha CNC brokerage Rs0, %-symmetric charges + 20bp slip) — identical to the rest of the batch. Apr-2026 STT hike is F&O-only → no impact.
**Survivorship caveat (Lesson #27):** the 2026 MTF list applied to 2023-25 is anachronistic; no point-in-time MTF list exists. **Paper is the production-faithful gate** (same as A2/C1).

## Falsifiers (pre-registered)
1. **Mechanism:** if the wide-range down bar reflects *informed* selling on real bad news (not a noise/liquidity over-shoot), the climax names keep falling (no bounce) — like the killed earnings-down-bounce / C3 news-driven flushes.
2. **Regime:** needs volatility/dispersion to generate climax bars; expected dead/weak in low-vol consolidation and the FII-exit regime (the same weakness C1 showed).
3. **Infra/data:** if the edge is concentrated in a few names with residual bad prints, or vanishes when the shock filter is removed (quiet wide bars), the footprint is an artifact, not a tradable climax.

## Shared batch falsifier
If, after proper Stage-2→6, the market-relative bounce is NOT positive across Discovery/OOS/HO, OR net-of-real-delivery-fees fails the confidence-card CI lower-bound > 1.0 / collapses in the 2026 HO, the candidate is killed (no salvage mining — Lesson #2).

## Decision (Stage 0 gate)
- [x] Proceed C5 to Stage 2 (Phase-2 signature, Discovery-only). C6 fallback on file if C5 fails.

---

## RESULTS (Stages 2-6, 2026-06-15) — real lifecycle, production universe, real delivery fees

Scripts: `tools/sub9_research/phase2_range_signature.py`, `cellmine_range.py`, `confidence_card_range.py` (separate candidate-specific copies; shared batch scripts untouched).

**Stage 2 (Phase-2 signature, Discovery-only, market-relative footprint vs tier-1 universe):**
| | hold-2 delta | hold-3 delta | win h3 | n | unique syms |
|---|---|---|---|---|---|
| C5 range_exhaustion | **−0.147%** | **−0.101%** | 45% | 1326 | 337 |
| C6 crash2d | +0.243% | +0.579% | 51% | 1055 | 343 |

C5's footprint is **NEGATIVE** (signal underperforms the universe) — wide-range/close-weak names *continue down*, not revert. It clears the |delta|≥0.1% magnitude gate but in the **wrong direction for a LONG** (a continuation/inverse footprint). Concentration is healthy (337 syms, top name PVP=17 of 1326) so it is NOT a bad-print artifact — it is a real, wrong-signed footprint. C6 is positive in both Discovery sub-years (2023 +0.31%, 2024 +0.82%).

**Stage 4-5 (Discovery cell-lock → one-shot OOS 2025 → one-shot HO 2026, net of real fees):**
- **C5 range_exhaustion — KILL.** M=72 cells swept, **0 ship-eligible** on Discovery (no cell reaches netPF≥1.20). Consistent with the negative Stage-2 delta — not a long-revert edge.
- **C6 crash2d — PASS.** M=72 cells swept, 10 ship-eligible, **5 net-positive in ALL 3 periods.** Most-stable (min ΔPF=0.198): **2-day-return cross-sectional rank ≤ 0.10 × K=3 × tier-1 × turnover-shock ≥ 2.0×**, PF Disc/OOS/26 = **1.36 / 1.17 / 1.37**.

**Stage 6 (C6 confidence card, Discovery-only CI + one-shot validators, true M=72):**
- BCa PF **1.355 [1.122, 1.606]** (n=1105) — CI lower bound > 1.0. Expectancy **₹372/trade [145, 585]**, win 47.6%.
- Harvey-Liu adj-Sharpe **0.93 at M=72** (42.5% haircut) — positive (raw Sharpe 1.62).
- Per-regime: strong `pre_election_calm` (1.34) / `election_vol_spike` (4.01) / `post_election_consolidation` (1.75); **weak/negative `fed_pivot_china_rotation_FII_exit` (0.85, net −₹31K)**. Regime-conditioned — same FII-exit weakness as C1 (pre-registered falsifier #2 confirmed). No data in tariff/war regimes (Discovery window ends 2024-12).
- One-shot: OOS 2025 **1.17** (win 46%, n=766), HO 2026 **1.37** (win 47%, n=254) — both positive, HO holds up in the down market.

**Verdict (this candidate):**
- **C5 `range_exhaustion_revert_long` = KILL** (Stage-2 negative delta + 0 ship-eligible cells; inverse/continuation footprint).
- **C6 `crash2d_revert_long` = PASS (borderline, real).** Numerically the *cleanest* of the batch's tested triggers: positive in ALL 3 periods, OOS 1.17 ≥ the informal 1.10 gate (C1's OOS was 1.08, just under), HO 1.37, BCa CI lower bound 1.12 > 1.0, positive Harvey-Liu at the true M=72. Genuinely distinct horizon-of-formation (2-day window) vs A2 (5d) / C1 (level) / C4 (σ-band) / C3-killed (1d). Worth adding to the paper batch as the 4th trigger; paper is the production-faithful gate (Lesson #27 anachronistic-MTF caveat).
