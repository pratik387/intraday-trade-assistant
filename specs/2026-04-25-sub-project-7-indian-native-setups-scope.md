# Sub-project #7 — Indian-Native Setup Library (SCOPE PROPOSAL)

**Status:** Scope proposal, NOT a design. Requires brainstorming/design pass before implementation.
**Date:** 2026-04-25
**Trigger:** Sub-project #5 (Gauntlet v2) post-mortem identified Failure G as root cause: existing setup library is cargo-culted from US/forex SMC literature and structurally mismatched to Indian intraday equity.

## Premise (drawn from sub-project #5 post-mortem + research)

The existing setup library (`structures/`) implements ICT/SMC patterns originating in 24/5 forex: order blocks, fair value gaps, premium/discount zones, change-of-character, break-of-structure. These patterns assume:
- Continuous liquidity (no auction open/close)
- No forced position closure (no MIS-style square-off)
- No circuit halt mechanics
- No T+1 settlement gap risk
- Symmetric long/short rules

Indian intraday equity violates ALL of these. The empirical result:
- Long variants of SMC patterns systematically lose money (PF 0.6-0.9 net).
- Short variants work only when they happen to align with structural retail-long unwind in 3:00-3:20 PM window.
- Net Sharpe 0.34 — well below professional benchmarks (1.0-2.0).

A redesigned setup library, built around Indian intraday market structure rather than borrowed forex concepts, is the path to capturing the available edge.

## Goal

Build a setup library specifically designed around Indian intraday equity microstructure, with fee-aware net edge as the primary success metric and Sharpe ≥ 0.7 net as the realistic target (research-validated as deployable floor for a real intraday Indian system).

## Indian-specific edge sources to exploit (research-sourced)

The following structural edges are documented in Indian quant trading literature and should be the candidate setup space:

1. **3:00-3:20 PM MIS unwind shorts** — predictable structural net-sell as leveraged retail longs are force-closed. Last-hour drift down is asymmetric.
2. **Opening 9:15-9:30 gap fade in mid/small caps** — FII gap-up + retail chase → exhaustion, a mean-reversion edge.
3. **CPR / Pivot mean reversion in 11:30-13:30 lunch lull** — low-volume window where range-trading dominates (Zerodha Varsity documented).
4. **Pump-and-dump morning fade in micro/small caps** — manipulator accumulation reversal patterns (academic literature documented).
5. **FII/DII flow signal as next-day direction prior** — previous day's FII net is a known weak signal for opening direction.
6. **India VIX-conditioned directional bias** — high VIX days favor mean reversion, low VIX days favor breakouts.
7. **F&O expiry week dynamics** — last week of month has rolling/hedging flow patterns.
8. **Sector rotation signals** — cross-sectional intraday momentum within sectors.
9. **Circuit-limit exhaustion fades** — when stocks approach upper circuit, forced cap creates fade opportunity.

## Scope (in / out)

**In scope:**
- Net-of-fees PF/Sharpe as the optimization target from day one.
- Setup detectors aligned with the 9 edge sources above (or a justified subset).
- Time-of-day-aware setup activation (some setups only fire in specific windows).
- Re-derivation of rule survivors from the new setup library on the existing wide_open OCI captures (for Discovery 2023-2024 only; OOS 2025+ data is spent).
- Reuse of existing infrastructure: parity simulator, wide_open OCI capture format, LiveGateChain composed gate framework.
- Reuse of existing OCI Discovery data (no new wide_open run required for setup-library iteration; setups only need 5m bar data, which we already have).

**Out of scope:**
- Re-running gauntlet v2 over the OLD setup library (don't polish a flawed catalog).
- Re-using the 74 rule survivors from sub-project #1 (they're derived from the wrong setup library).
- Re-using best_config.json from sub-project #5 Optuna (it's tuned to wrong setups).
- Validation/holdout testing of the new system on EXISTING 2025-2026 OOS data (those datasets were spent in sub-project #5; need fresh data for clean OOS).
- F&O / options strategies (this sub-project stays in cash equity intraday).
- Multi-asset / cross-instrument signals beyond FII/DII flow.

## Open questions (require brainstorming)

The following decisions are not pre-made and must be resolved in the brainstorming phase:

1. **Build new setups from scratch vs. refactor existing detectors?** Some existing detectors (range_bounce_short, premium_zone_short) DO work — possibly because they accidentally align with Indian structure. Worth distinguishing "Indian-aligned by accident" from "redesigned for Indian market" detectors.

2. **Time-of-day weighting: hard time gates vs. soft scoring features?** A hard rule "premium_zone_short only fires 14:30-15:20" vs a feature "minutes_to_mis_close" that conviction model uses.

3. **How to obtain FII/DII data?** Daily NSE bhavcopy provides previous-day net flow. Real-time intraday flow requires NSE/BSE proprietary feeds or proxy via large-cap futures order flow.

4. **India VIX integration:** intraday VIX prints exist; need ingestion from NSE indices feed.

5. **OOS reservation:** if we burn validation/holdout from sub-project #5, what's the fresh OOS plan? Options: (a) wait 3-4 months for live data (April-July 2026); (b) run a second independent OCI capture with a TIME-disjoint period reserved.

6. **Fee model precision:** simple per-trade fee or order-book-aware (slippage modeling)? For Indian retail intraday, simple per-trade fee is probably sufficient; institutional sizing would need slippage.

7. **Fail criteria:** what constitutes failure of THIS sub-project, and what's the next escalation if it fails?

## Suggested phasing

**Phase 0: Brainstorming + design** — resolve the 7 open questions above. Output: design document.

**Phase 1: Net-aware fee model** — extend trade_report.csv pipeline to compute net PnL at trade time. Refactor build_pnl_index to use net.

**Phase 2: Indian-native setup detectors** — implement the 9 edge sources (or the subset chosen in Phase 0). Each setup is a standalone module with clear documentation of which Indian-specific structural fact it exploits.

**Phase 3: Local backtest per setup** — measure each setup's net PF/Sharpe in isolation on Discovery 2023-2024. Drop setups with net PF < 1.1.

**Phase 4: Composition** — combine surviving setups via existing LiveGateChain framework. Re-derive rule survivors. Tune via Optuna with NET as objective. (This is gauntlet v2 redux, but with a better setup library.)

**Phase 5: Fresh OOS test** — once new live data is available (≥3 months post-launch of new pipeline), formal Phase 3/4 OOS validation per master-plan discipline.

## What this sub-project does NOT promise

- It does not promise net Sharpe ≥ 0.7. Research suggests 0.7 IS achievable for a well-designed Indian intraday system, but achieving it requires that the chosen edge sources actually deliver net of fees. This is an empirical question.
- It does not promise that all 9 edge sources will yield positive net edge. Some may not. Phase 3 (per-setup backtest) will surface this.
- It does not promise to be a multi-month project. The actual cost depends on Phase 0 design choices (especially: build new vs refactor existing).

## Estimated cost (rough)

- Phase 0 (brainstorming + design): ~1 week
- Phase 1 (fee model integration): ~3-5 days
- Phase 2 (setup detectors): ~2-3 weeks
- Phase 3 (per-setup backtest): ~1 week
- Phase 4 (composition + Optuna): ~1 week
- Phase 5 (OOS test): waits for fresh data (3-6 months elapsed time)

Total active engineering: ~6-8 weeks. Total elapsed (including OOS reservation): 4-7 months.

## Decision required from human

This is a SCOPE PROPOSAL. Before any code:

1. Confirm the diagnosis from sub-project #5 post-mortem (Failure G is root cause).
2. Confirm appetite for a multi-week redesign vs alternative (e.g., halt all sub-project work and accept current production behavior).
3. Resolve the 7 open questions in a brainstorming pass.
4. Greenlight Phase 0 design work.

If this scope is approved, the next step is invoking the brainstorming skill on this document to produce a proper design.
