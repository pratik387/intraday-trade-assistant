# Research brief: `block_deal_followthrough_short`

**Branch:** `research/post-sebi-edge-setups`
**Sub-project:** sub-9 §4 candidate "H" (Block deals / bulk deals — never attempted as a standalone setup)
**Status:** **DRAFT — pre-registered, pre-sanity. One-pager research brief.**
**Date:** 2026-05-14

---

## Mechanism (one sentence)

Large reported NSE block-deal **SELL** disclosures (institutional / promoter / FII) are *informed-selling* events on illiquid mid/small-cap names — when the disclosed sold-quantity is meaningful relative to a session's traded volume the price shock keeps unwinding into the next session as the institutional seller's remaining inventory continues to be worked, producing a predictable T+1 negative drift.

This is the **sell-block continuation** hypothesis (option #2 in the brief's three candidates), preferred over counter-flow because:
- The prior `nse_block_deal_counter_flow` sanity (F&O 200 universe, both sides) returned PF 0.79 LONG / 0.74 SHORT on n=585 — i.e. fading either side is *unprofitable on liquid large caps* where the block size is fully absorbed in one session.
- Project rule §3.2 #5 (sub-9): asymmetric SHORT-only direction; do not test two-sided setups.

## Direction

**SHORT-only on T+1** (continuation of seller-driven drift).

## regulatory_sensitivity

`rule_orthogonal` — block-deal disclosure rules (SEBI block window 08:45-09:00 / 14:05-15:30, >₹10 Cr threshold, NSE/BSE archive publication T+0) have been stable since 2017 and are not part of the Oct 2025 SEBI overhaul. The mechanism does not depend on MWPL, FutEq OI, or F&O ban rules.

## depends_on

`["MIS_leverage", "STT_drag"]`

- `MIS_leverage` — sizing & break-even depend on 5x MIS leverage being available on the target universe (mid/small-cap).
- `STT_drag` — post Apr 1, 2026 STT hike (0.025% delivery, 0.025% equity intraday sell) compresses the T+1 SHORT edge by ~2-3 bps per round trip; net-PF must clear after STT-adjusted fees via `tools/sub7_validation/build_per_setup_pnl::calc_fee`.

NOT regime-sensitive to: MWPL, single_stock_FO, F&O_speculation, intraday_ban, options_premium_upfront, expiry_lot_size.

## Universe / cell pre-registration

Pre-registered cells (locked before sanity execution):

| Cell | Filter |
|---|---|
| Direction | `buy_or_sell == "SELL"` (no buy-side, no aggregated net) |
| Size impact | `block_qty / avg_T0_session_volume >= 5%` (true volume-shock threshold, NOT a notional ₹-cut) |
| Notional floor | `side_total_cr >= 5.0` (filters out micro disclosures; SEBI block-window min is ₹10 Cr but exchange archive also lists qualifying bulk deals below that) |
| Cap segment | `cap_segment in ["mid_cap", "small_cap"]` (large-caps absorb in T+0; the structural edge is in thinner books — counter-flow run confirmed large-cap is dead at PF 0.74-0.86) |
| Latch | one fire per (symbol, T+1, side) — collapse multi-line same-day disclosures |
| Universe | NSE-listed only; F&O 200 NOT required (mid/small caps are mostly out of F&O 200) |

Entry mechanic (mirrors `sanity_nse_block_deal_counter_flow.py`, single side):
- Block disclosed on date `T0`.
- T+1 first 5m bar @ 09:15; entry at second 5m bar OPEN (09:25 IST).
- Hard SL 1.5% (min 1.0%), T1 at 1.0R (50% qty), T2 at 2.0R (50% qty).
- BE-trail after T1 (`active_sl = entry_price if t1_hit else hard_sl`).
- Time stop 14:30 IST.
- Risk Rs.1,000/trade.
- Fees via `tools/sub7_validation/build_per_setup_pnl::calc_fee` (STT-aware, Zerodha-spec).

## Ship gates (gauntlet-v2)

| Gate | Threshold |
|---|---|
| n (Discovery, post-latch) | ≥ 100 |
| Net PF (post-fee) | ≥ 1.20 |
| Sharpe (daily) | > 0 |
| Per-month stability | majority of trading months net-PF > 1.0 (no month-cluster bias) |
| Cap-segment cross-check | mid and small individually pass PF ≥ 1.15 |

## Falsifiers (pre-registered)

1. **Mechanism falsifier — leakage check**: if T+0 same-day EOD SHORT (control simulation, entered at 15:10 close) PF >= T+1 PF, the asymmetry is pre-disclosure leakage already priced in, NOT a next-session continuation edge → RETIRE.
2. **Sample falsifier**: if n < 100 post-latch on 2023-2024 NSE sell-blocks, mid/small only — DATA-UNAVAILABLE, defer until 2025+ block-deal backfill lands.
3. **Stability falsifier**: if Discovery 2023 PF >> Discovery 2024 PF (i.e. >0.4 absolute drift), regime-instability — defer to OOS-only validation.
4. **Cap-segment falsifier**: if large_cap rerun (relaxing universe) shows PF >= small/mid PF, the mechanism is NOT illiquidity-driven, and the cell-selection is just data-mined → RETIRE.

## Sanity script + run plan

- Script: `tools/sub9_research/sanity_block_deal_continuation_short.py`
- Window: **Discovery 2023-01-05 → 2024-12-31** (full coverage of the scraped parquet — the spec's nominal "2024-09 to 2025-09" Discovery window would only return ~4 months of overlap with available data).
- Out: `reports/sub9_sanity/block_deal_followthrough_short_trades.csv` + `_t0_control.csv`.

## Decision-tree post-sanity

| Outcome | Action |
|---|---|
| All gauntlet-v2 gates pass + T+0 leakage clean | STRONG PROCEED → write full brief, OOS validation on 2025+ data once scraped to date |
| Gates pass but T+0 leakage detected | MARGINAL — log to `tasks/lessons.md`, mechanism is mis-identified; revisit as a different setup (pre-disclosure short on the institutional client name) |
| Gates fail (PF < 1.20 or n < 100) but per-cap mid-only passes | MARGINAL — narrow universe and revisit |
| Gates fail across cells | RETIRE → document in `docs/retired_setups.md` |
