# Brief: monster-conditioning for multiday capitulation composite — PARKED (forward-mirror fail)

**Date:** 2026-07-21
**Status:** PARKED — historical card passed, forward mirror REVERSED. Do not ship.
**Scope:** conditioning filter on the 4 multiday CNC/MTF capitulation setups
(A2 mtf_capitulation / C1 low52 / C4 zscore_oversold / C6 crash2d), at the
composite-selector layer. Not a new setup.

## Hypothesis (pre-stated before P&L testing)

User observation: the multiday book produces 1-2 "monster" winners (>= +5%)
per week that carry the whole P&L (measured: 6 of 39 composite-era paper
trades = 138% of total net). Screen question: do monsters share an entry-time
signature that the current selection features (cap_score, tshock, consensus)
miss?

17-feature screen (RSI2/14, ATR14%, BB %B, SMA50/200 dist, ret5/20/60,
52w position, down-streak, vol-z, gap, range, close-pos, log-price, tshock)
on 10,414 historical fires, requiring |AUC-0.5| >= 0.07 with consistent sign
across Disc 2023-24 / OOS 2025 / Rec Jan-Apr 2026:

- **ATR14% high** — AUC 0.62-0.77, passes all 4 setups x 3 periods
- **Trailing weakness deep** (dist_sma50 / ret20 low) — AUC 0.22-0.37 (inverted)
- cap_score-family (tshock) AUC ~ 0.5 everywhere → current ranking features
  cannot see monsters (consistent with the n=33 paper-trade join).

## Validation (tools/sub9_research/monster_conditioning_validation.py)

Parameter-free rules (nothing fitted on Disc): RULE-X = day-relative median
split; RULE-T = trailing-120d fire-median thresholds. Real MTF fees at Rs1L
live-plan sizing, K-day-close baseline geometry.

RULE-T pooled: PF 1.96 / 1.58 / 1.81 (Disc/OOS/Rec) vs baseline 1.47 / 1.25 /
1.33; keeps 39% of fires, 88% of net; improves all 4 setups in all 3 periods
(12/12 cells). Confidence card (M=64 Harvey-Liu): pooled PF CI [1.63, 1.91],
all 7 regimes net-positive, adj Sharpe 1.04 (per-setup up to 1.87).
Cards: reports/confidence_cards/monster_cond_rule_t_*.md

**Weak cell noted in-sample:** post_tariff_consolidation PF 1.06 [0.88, 1.29]
— the only regime where the rule adds ~nothing. This was the tell.

## Forward mirror (Jun-30..Jul-21 composite paper book, n=47) — REVERSED

Adaptive median split on the forward window (proxy for a live trailing rule
after the May-Jul vol compression; frozen April thresholds keep only 3/47 —
distributions shifted, ATR median 4.4 -> 3.5):

- high-ATR + deep:   n=16 net **-24,377** PF **0.76** (4 monsters)
- low-ATR + shallow: n=14 net **+34,475** PF **3.33** (0 monsters)

Exact inversion of the historical pattern. Forward losers ARE the deep/volatile
names (SALASAR, EPIGRAL re-entry, BIRLANU, KALPATARU, NITTAGELA, BIRLAMONEY);
forward winners are shallow/low-ATR (MAXESTATES, WESTLIFE, TEAMLEASE, CLSEL).

## Interpretation

The rule's edge is plausibly regime-conditional: deep+volatile capitulations
snap back hard in vol regimes (war_vol PF 2.12, election_vol 4.50 in-sample)
and keep bleeding in grinding consolidation (post_tariff 1.06 in-sample;
Jun-Jul forward tape is consolidation-like and also the window where overnight
close_dn went cold). n=16/14 forward is small — this is a veto under the
post-dow shipping bar (historical AND forward must both pass), not proof of
inverse edge.

## Revisit triggers

1. Forward paper reaches ~100 composite-era owner trades → re-run the
   forward split (tools: /tmp-style fwd check documented in session; features
   via UpstoxDataClient get_daily, Wilder ATR14, SMA50).
2. Regime shifts back to a vol regime → re-test as a REGIME-GATED filter
   (only active when regime in {war_vol, election_vol, ...}) — that variant
   was NOT searched here and would need its own Harvey-Liu accounting.
3. If revived: implement at MultiDayCompositeSelector as ranker-emitted
   features (like sigma20_pct), paper A/B before any cap binding.

## Artifacts

- tools/sub9_research/monster_conditioning_validation.py (rules + fees study)
- tools/sub9_research/monster_conditioning_cards.py (confidence cards)
- reports/sub9_sanity/_monster_cond_*.csv (trade baskets)
- reports/confidence_cards/monster_cond_rule_t_*.md
