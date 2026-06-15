# Algo Research Backlog — 2026-06-02

**Origin:** Session that ran (1) a failure-mode triage of the 0/22 recent batch, (2) a data-asset inventory + explored-dimensions census, (3) a verified deep-research landscape of what algo traders trade on (107-agent harness, 24 sources, report archived at `reports/research/2026-06-02-algo-landscape/`).

**Central reframe:** every prior candidate was a *time-series* setup ("detect pattern X on symbol Y at time T"). The literature says the signals alive in this system's universe are *cross-sectional* ("rank the universe daily, trade the extremes"). The unexplored paradigm is cross-sectional factor ranking on the illiquid universe — Feasible-now from data on disk.

**The binding constraint, named:** microcap edges are "more apparent than real" *because of transaction cost* (Hou-Xue-Zhang RFS 2020). The same illiquid universe where signals are statistically strongest is where cost/impact is most punishing. Every item below carries a **net-of-cost kill-test as the FIRST gate** — never build before proving tradeable (the delivery%-accumulation lesson: an edge can live entirely in a non-capturable component).

Priority: **P0** = foundational/blocking · **P1** = high-conviction, do next · **P2** = promising, speculative · **P3** = housekeeping.

---

## Track 0 — Foundational (gates everything)

- [ ] **T0.1 (P0) — Realistic transaction-cost + market-impact model for illiquid names.** Extend the fee stack (STT/brokerage/exchange/SEBI/stamp/GST) with a **market-impact term for <500K-vol stocks** (impact ∝ order-size/ADV, or half-spread proxy from daily range). CNC round-trip STT alone is ~0.20% (0.1% buy + 0.1% sell) vs MIS 0.025% sell-only — this dominates short-horizon CNC economics. **Deliverable:** a `net_return(symbol, date, gross_ret, side, product, notional)` function usable by every study below.
- [ ] **T0.2 (P0) — Cross-sectional ranking harness.** Daily-rank the universe by any characteristic, form long-only decile baskets, measure net-of-cost spread. Reusable infra for Track A. (First version embedded in the A1 study script.)

## Track A — Cross-sectional factor setups (NEW paradigm: long-only, daily rank, illiquid universe)

- [ ] **A1 🆕 (P1, TOP PICK) — Cross-sectional short-term reversal.** Rank small/mid/SME by trailing 1-5d return; long the loser decile, exit after K days (CNC, T+1-open entry). Literature's strongest microcap-revived signal (Jegadeesh 1990; HXZ). **Kill-test:** loser-decile net-of-T0.1-cost spread > 0 across 2023-25 AND survives 2026; locate which ADV tier it survives in. *(Study started 2026-06-02.)*
- [ ] **A2 🆕 (P1) — Liquidity-shock reversal.** Amihud/dollar-volume spike + sharp drop → revert. **Kill-test:** does conditioning A1 on a turnover/Amihud shock beat plain A1 net-of-cost? If not, fold into A1.
- [ ] **A3 🆕 (P2) — MAX / lottery-demand fade.** Extreme trailing max-daily-return decile underperforms (Bali-Cakici-Whitelaw); retail lottery demand = this universe. Long-only = avoid-filter; short = MIS intraday. **Kill-test:** high-MAX forward underperformance net-of-cost; short leg must clear MIS regulatory + intraday tradeability.
- [ ] **A4 🆕 (P2) — Cross-sectional momentum at swing horizon.** CS momentum (1-4 wk) in under-covered illiquid names (slow diffusion); one of few robust survivors. **Kill-test:** winner−loser decile net-of-cost; confirm it's not just the size factor.

## Track B — Event / slow-diffusion (CNC overnight-swing)

- [ ] **B1 🆕 (P1) — PEAD-CNC drift.** SUE-rank post-earnings, hold day +2 to +N, CNC. India-validated (~4.8-6% decile spread, 2002-17). **Kill-test (mandatory):** tradeable **T+2-open→exit** drift net-of-cost — NOT close-to-close — confirm not trapped in announcement gap or eaten 70-100% by cost in illiquid names.
- [ ] **B2 🆕 (P2) — Liquid→illiquid daily lead-lag (within-sector).** Prior lead-lag deaths were intraday/index-level (arbed); untested at daily horizon via slow diffusion. **Kill-test:** liquid-peer move predicts illiquid-name next-day move net-of-cost, beyond own-reversal (A1).

---

## 2026-06-15 — CNC/MTF candidate exhaustion (evidence; full battery per lesson #28)

Tested on `cache/preaggregate/clean_daily_from5m.feather` (CA-adjusted) with market-relative (demeaned) controls and net-of-cost gates. Scripts: `_tmp_b1_pead_exhaustive.py`, `_tmp_a4_momentum_exhaustive.py`, `_tmp_regime_drift_vs_reversion.py`, `_tmp_b1_hedged_netpnl.py`.

- **A2 (liquidity-shock reversal) — SHIPPED candidate, code-complete, awaiting paper.** Demeaned reversion alpha is **STABLE across regimes** (2023 +0.36% / 2024 +0.32% / 2025 +0.23% / 2026 +0.38%, 2d hold). The durable CNC/MTF edge. Built as `mtf_capitulation_revert_long` (branch `research/2026-06-14-mtf-capitulation-revert`).
- **A4 (CS momentum) — PARK (real but DECAYING alpha).** Demeaned winner alpha is REAL and strongest in tier-1 illiquid (slow-diffusion, literature-consistent: L60/H20 tier1 +3.1% / net +2.6%), so NOT survivorship. BUT the alpha itself decays: 2023 +2.0% → 2024 +1.4% → 2025 +0.3% → **2026 −1.0%** (demeaned). Don't build into a fading edge; revisit only if the regime turns trending.
- **B1 (price-surprise PEAD; SUE impossible — calendar has no fundamentals) — PARK (real alpha, NOT harvestable).** Abnormal (vs equal-weight universe) drift is positive ALL years (2023 +2.97 / 2024 +3.15 / 2025 +1.36 / 2026 +1.38%, top-10% reaction, +2 entry, 10d). BUT: long-only carries full market beta (≈flat-to-neg net); the SHORT leg's down-drift lives in non-shortable illiquid names (F&O-shortable subset LOSES −0.90% net) = tradability-masked; and hedging with the only liquid instrument (MID150 futures) turns net **NEGATIVE** (−0.57% 10d; 2025 −1.68 / 2026 −1.80) because the illiquid basket lagged the midcap index. Edge trapped in a non-capturable component (the T0-gate failure mode). Revisit only if a tradable small/micro short appears.
- **B2 (liquid→illiquid daily lead-lag) — BLOCKED + already fee-dead.** (a) Sector map `assets/stock_sector_map.json` covers only 153 LIQUID leaders — no classification for the illiquid followers → "liquid→illiquid" not buildable without new data. (b) The intraday version was already studied: `reports/sub9_sanity/sector_pair_convergence_intraday_trades.csv` = gross +33K / **net −265K** (mean −93/trade). Briefs (`sector_pair_convergence_intraday`, `intraday_leadlag_continuation_5m`) are stalled DRAFTs; the family is fee-dead at 1× friction.
- **A3 (MAX/lottery fade) — not retested here** (short-leg is MIS-intraday, not CNC/MTF; out of this track's scope).

**Conclusion (evidence-grounded):** the only net-harvestable CNC/MTF edge in this universe is **REVERSION (A2)**. The drift/diffusion candidates each show a real statistical signature but fail the *harvestable-net-of-cost* gate (B1 basis-risk-trapped, A4 decaying, B2 cost-dead + data-blocked). 2023-24 was a trending regime; 2025-26 turned reversion-dominated. **Next action: get A2 to paper (start the clock), not more factor mining.**

---

## Track C — Methodology force-multipliers (low risk, broad upside)

- [ ] **C1 🆕 (P1) — Volume/dollar information-driven bars** from 1m data, vs 5m time bars, for microcap detectors (López de Prado). **Kill-test:** any existing detector's OOS improves on volume bars.
- [ ] **C2 (P2) — Meta-labeling layer** over `close_dn` + `below_vwap`: primary picks side, ML filter decides take/skip + size.
- [ ] **C3 (P2) — Deflated Sharpe / CPCV / PBO** into the confidence framework, extending the Harvey-Liu haircut.

## Track D — Portfolio construction

- [ ] **D1 (P2) — Sparse-event basket (Lever 2).** Bundle frequency-killed *real* edges (`post_split_bonus` +4.30%/73.7%, `fno_ban`, `anchor_lockin` if survives) into one ~450/yr strategy. **Prerequisite:** each component re-passes the tradeable test (T0.1).

## Track E — Housekeeping

- [ ] **E1 (P3)** — Resolve killed `delivery_accumulation` brief (keep as Phase-2-kill record or delete); branch-commit/clean the ~35 `_tmp_*` probes per Lesson #20.

---

**Recommended sequence:** T0.1 + T0.2 → **A1** (flagship + the cost-survival measurement that answers "is the illiquid edge real or 'more apparent than real'?") → **B1** in parallel. C1 anytime.
