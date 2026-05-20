# Setup Lifecycle — Idea → Live → Retirement

**Status:** Active as of 2026-05-20. Single source of truth for moving a setup from initial hypothesis to live deployment and through retirement.

**Replaces:** `docs/methodology_walk_forward.md` (deprecated — per-window tier methodology was abandoned, see `tasks/lessons.md` #15). The 3-period chronological validation (Discovery / OOS / Holdout) survives as a research tool; the **decision** at each gate is now made via the confidence framework (`tools/methodology/confidence/`), not via tier classification.

## Core principles

1. **Each stage has a gate, not a threshold.** The confidence framework outputs INTERVALS. The researcher reads them and judges. No code in this project hardcodes a ship/no-ship threshold; doing so would be folklore (Lesson #15).
2. **Sanity inflates, OCI is gold standard.** A setup that looks dead on sanity is dead. A setup that looks alive on sanity might still be dead in production. The Lesson #13 caveat applies at every sanity-stage gate.
3. **Pre-register mechanism before testing windows.** `mechanism_tags` + `mechanism_notes` go in config ≥ 1 commit before any window-based evidence. Falsifiable, not post-hoc (Lesson #12).
4. **Three numeric inputs only — researcher judges.** At every gate the input is: aggregate PF CI, per-regime breakdown, adjusted Sharpe (Harvey-Liu sign-preserving haircut for effective M setups tested historically).

## Stage map

| Stage | Owner artifact | Gate criterion (read, don't compute) | Decision |
|---|---|---|---|
| 0. Idea | `specs/YYYY-MM-DD-brief-<name>.md` | Hypothesis stated; falsifier identified; mechanism rooted in NSE/SEBI/Indian-market behavior | proceed / kill |
| 1. Phase 1 empirical signature | First-pass aggregate stats notebook/script | Signal vs random-day baseline differs meaningfully; n large enough to not be folklore | proceed / kill |
| 2. Phase 2 sanity script | `tools/sub9_research/sanity_<name>.py` | Discovery aggregate stats consistent with Phase 1 prediction | proceed / kill |
| 3. Phase 3-4 cell-lock | Locked cell JSON in `tools/sub9_research/` | Cell narrows aggregate without overfitting (researcher judges) | proceed / kill |
| 4. Phase 5 OOS + Holdout | D + OOS + Holdout sanity CSVs in `reports/sub9_sanity/` | All three periods point in same direction; consistency of cell behavior | proceed / kill |
| 5. **Confidence card (sanity)** | `reports/confidence_cards/<setup>_confidence_card.md` | PF CI on combined D+OOS+Holdout, per-regime breakdown, sanity-adjusted Sharpe | proceed to structure / kill |
| 6. Structure code | `structures/<setup>_structure.py` + `services/setup_universe.py` entry + tests | code-reviewer agent approves; mechanism_tags pre-registered in config ≥ 1 commit prior | proceed / kill |
| 7. OCI backtest | OCI run directory with `analytics.jsonl` per session | Backtest completes successfully on 2023-2026 window | proceed / fix structure |
| 8. **Confidence card (OCI)** | OCI-source `confidence_card.md` | **Gold-standard verdict** on production-equivalent data | proceed / fix-or-retire |
| 9. Paper trade | 30-90 days `--paper-trading` mode | Paper PnL distribution matches OCI backtest expectation (CI overlap) | proceed / fix |
| 10. Live small | `position_size_multiplier: 0.25`, `cb_state: enabled` | 60-90 days; live distribution within OCI CI | scale up / hold / retire |
| 11. Scale up | `position_size_multiplier: 1.0` | Live verdict matches OCI expectation across multiple regimes | full deployment |
| 12. Ongoing | Quarterly OCI re-run + confidence card refresh | Per-regime breakdown surfaces decay before circuit breaker | continue / re-cell-lock / retire |
| 13. Retire | `docs/retired_setups.md` entry | PF CI crosses 1.0 OR adj Sharpe goes negative on accumulated OCI data | document + remove |

## Stage-by-stage runbook

### Stage 0 — Idea

Write `specs/YYYY-MM-DD-brief-<setup_name>.md` with:

- Mechanism: WHY this should work in NSE intraday (cite the institutional or retail behavior driving the edge)
- Falsifier: a specific outcome that would prove the thesis wrong
- Adjacent setups: which existing setups would correlate with this one (matters for effective M at Stage 8)

A brief that doesn't pass code review here probably won't survive Phase 5 either.

### Stage 1-2 — Phase 1 / Phase 2 sanity

Pattern established in `tools/sub9_research/sanity_*.py`. Output is a per-trade CSV with the schema validated by `tools/methodology/sanity_csv_schema.py`.

The point of Phase 1-2 is to confirm the signal is detectable at all. Pure noise candidates should die here, cheap.

### Stage 3-4 — Cell-lock

Cell-mining happens on Discovery only. Lock the cell (filters: vol-ratio, dist-from-VWAP, time-of-day, cap-segment, etc.) BEFORE running OOS or Holdout.

Cells that materially outperform the aggregate on Discovery and then fail to reproduce on OOS are the typical death pattern. See `docs/retired_setups.md` for examples.

### Stage 5 — Confidence card on sanity

Aggregate the D+OOS+Holdout trade CSVs into the canonical schema, then run the confidence framework:

```bash
# 1. Aggregate to canonical (one-time per setup)
#    Edit SETUP_CSV_MAP in aggregate_sanity_to_canonical.py to add the setup,
#    then run:
.venv/Scripts/python tools/methodology/aggregate_sanity_to_canonical.py

# 2. Generate confidence card
.venv/Scripts/python tools/methodology/confidence/confidence_card.py \
    --include-sanity --setups <setup_name>
```

Read `reports/confidence_cards/<setup>_confidence_card.md`. The questions to ask:

- Is the aggregate PF CI lower bound > 1.0? (necessary, not sufficient)
- In which regimes does the edge live? Is it concentrated in 1-2 regimes or spread across 5-7? Concentrated = regime-conditioned (high attrition risk)
- Is the adjusted Sharpe positive after Bonferroni haircut? Sanity-source haircut is informational; the real test is at Stage 8

**Lesson #13 caveat at this stage:** sanity-GREEN does NOT permit live deployment. It only justifies the investment of Stage 6 structure-code work. Sanity-RED kills the setup here without spending the structure-code week.

### Stage 6 — Structure code

Write the detector in `structures/<setup>_structure.py`. Wire universe in `services/setup_universe.py`. Add tests in `tests/` (gitignored individually but tracked test files exist).

Pre-register mechanism in `config/configuration.json` under `setups.<name>`:
```json
"mechanism_tags": ["tag_from_assets/mechanism_tags_registry.yaml"],
"mechanism_notes": "Why this should work and which regimes might break it"
```
Commit pre-registration as its own commit BEFORE running OCI. This is enforced by `tools/methodology/pre_registration.py`.

Run the code-reviewer agent on the structure code. CLAUDE.md mandatory rules apply (no hardcoded defaults, IST-naive timestamps, live/backtest compatibility).

### Stage 7 — OCI backtest

Deploy structure code to OCI. Run backtest over 2023-01..present. Sessions land as `<YYYY-MM-DD>/analytics.jsonl` records.

### Stage 8 — Confidence card on OCI (gold-standard verdict)

Aggregate the OCI run into canonical and regenerate the confidence card:

```bash
.venv/Scripts/python tools/methodology/aggregate_oci_to_canonical.py \
    --run-dir <oci_run_path> --setup <setup_name>
.venv/Scripts/python tools/methodology/confidence/confidence_card.py
```

This card is the **gold-standard verdict**. Same questions as Stage 5, but now the answers are production-equivalent (real entry filter, real fills, real fees). If the OCI confidence card disagrees with the sanity card, trust the OCI card.

The regime breakdown matters most here. A setup with PF CI [1.10, 1.40] aggregate that has 4/7 weak regimes is a candidate for regime-gating (Lopez de Prado tactical paradigm), not a clean ship.

### Stage 9 — Paper trade

Run `python main.py --paper-trading` with the setup enabled. 30-90 days. Compare actual paper P&L distribution to the OCI confidence card's expectancy CI. They should overlap.

If paper diverges materially from OCI: stop. Investigate (often a live-vs-backtest infrastructure issue — see `feedback_production_mindset.md` and `project_paper_backtest_parity.md`).

### Stage 10 — Live small

Config:
```json
"enabled": true,
"position_size_multiplier": 0.25,
"cb_state": "enabled",
"cb_drawdown_threshold": -<2σ of OCI per-day distribution>,
"cb_lookback_days": 60,
"cb_min_trades_for_signal": 30
```

60-90 days. Compare live distribution to OCI expectation. Daily circuit breaker (`jobs/check_circuit_breakers.py`) monitors trailing 60-day net PnL.

### Stage 11 — Scale up

Only after Stage 10 closes positive AND matches OCI expectation across at least 2 regimes (e.g., the live window spanned post_tariff_consolidation + war_vol_2026). Move `position_size_multiplier` to 0.5 → 1.0 over 30-60 days.

### Stage 12 — Ongoing monitoring

Quarterly: re-run `aggregate_oci_to_canonical` on accumulated production data, regenerate confidence card. Compare new card to the Stage 8 baseline:

- PF CI tightening → edge is real and stable
- PF CI widening or drifting toward 1.0 → degradation; investigate regime cause
- Per-regime breakdown shows new weak regime → consider regime gate

Daily: circuit breaker. If it trips, the setup goes `cb_state=disabled` automatically; researcher inspects and decides retire vs un-disable.

### Stage 13 — Retirement

Triggers (any one is sufficient evidence — researcher confirms):

- PF CI lower bound below 0.95 on OCI canonical with n > 500 in the most recent 12-month window
- Adjusted Sharpe (Harvey-Liu) flips negative on OCI canonical
- Per-regime breakdown shows the edge has collapsed to a single regime AND that regime has ended

Documentation requirements (replacing the prior walk-forward-table requirement):

1. Section in `docs/retired_setups.md` with: thesis, universe+filters, claimed validation, **OCI confidence card snapshot** (aggregate PF CI + per-regime table + adjusted Sharpe), failure mode, conditions for revival
2. Mechanism tags in config block must reflect the actual decay cause (e.g., add `regime_X_dependent` if regime breakdown revealed it)
3. Code removal list
4. New `tasks/lessons.md` entry IF a new failure pattern was discovered (not for repeat patterns)

## Anti-patterns (don't)

- **Don't compute new thresholds without literature backing.** PF > 1.10, win-rate > 55%, 9/13 windows — all folklore. The confidence framework outputs intervals; trust them.
- **Don't un-retire a setup based on a sanity-GREEN result.** Sanity inflates (Lesson #13). Revival requires fresh OCI evidence with structure code.
- **Don't ship at full size from Stage 8.** Even with a clean OCI card, Stage 9-10 are non-negotiable. Production has live-vs-backtest infrastructure risk.
- **Don't skip pre-registration.** A post-hoc "the mechanism is X because window 7 failed" claim is unfalsifiable.

## References

- Confidence framework code: `tools/methodology/confidence/`
- Regime schema: `assets/regime_schema.yaml`
- Aggregators: `tools/methodology/aggregate_{oci,sanity}_to_canonical.py`
- Lesson #12: pre-registration requirement
- Lesson #13: sanity-vs-production inflation
- Lesson #14: per-trade actual fees (not flat constant)
- Lesson #15: framework redesign + sign-preservation bug
- Research backing for the framework: `reports/sub9_sanity/_per_trade_validation_research.md` (gitignored but exists locally), `reports/sub9_sanity/_indian_regime_schema_research.md`, `reports/sub9_sanity/_indian_regime_global_events_research.md`
- Stale doc: `docs/methodology_walk_forward.md` (per-window tier methodology, abandoned)
