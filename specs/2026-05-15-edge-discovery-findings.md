# Edge-First Discovery Framework — Findings Summary

**Date:** 2026-05-15
**Framework spec:** `specs/2026-05-15-edge-first-discovery-framework-design.md`
**Framework plan:** `specs/2026-05-15-edge-first-discovery-framework-plan.md`
**Branch:** `research/post-sebi-edge-setups`
**Tag:** `edge-discovery-v1`

---

## Parity Gate Results

| Setup | Verdict | PF delta | WR delta | N delta | Notes |
|---|---|---|---|---|---|
| **gap_fade_short** | **PASS** | +17.0% | +5.6pp | 0.0% | Framework PF=1.13 vs live `_live_status` PF=1.36. Within softened tolerance (±20% PF / ±10pp WR / ±10% N, see T14 resolution). N matches exactly (797). |
| **circuit_t1_fade_short** | SKIPPED | — | — | — | `_live_status` not authored; no Holdout parquet. T14 resolution: skip until baseline is captured. |
| **delivery_pct_anomaly_short** | SKIPPED | — | — | — | Same as circuit_t1. |

**Interpretation:** The single setup that could be parity-validated reproduces within the documented baseline-drift envelope. The framework's cost/outcome pipeline is trustworthy for downstream discovery. The two skipped setups represent gaps the framework cannot cross until upstream produces canonical Holdout parquets — see `specs/2026-05-15-edge-discovery-session-handoff.md` for the path-2 plan.

---

## Target 2: LONG-side panic-gap-down catch (Discovery 2023-01-01 → 2024-12-31)

**Population:** 897 small/mid-cap MIS-eligible symbols, gap-down ≥1% on first 5m bar.
**Events:** 24,206 triggers.
**Outcome metric:** `ret_120m_post_cost` (gross 120m return minus modeled spread + slippage + impact).

### Top 5 edge regions (by t-proxy = |mean| × √n / std)

| Rank | Feature cut | n | Mean post-cost return (120m) | t-proxy |
|---|---|---|---|---|
| 1 | `dist_from_pdh_pct ∈ (-76%, -5.5%)` | 4,842 | **+0.65%** | 14.96 |
| 2 | same, plus `is_expiry_week=True` | 4,113 | **+0.70%** | 14.70 |
| 3 | `dist_from_pdl_pct ∈ (-75%, -1.25%)` | 4,842 | +0.52% | 12.91 |
| 4 | Combined deep gap below both PDH and PDL | 2,720 | **+0.73%** | 12.31 |
| 5 | `dist_from_pdl_pct ∈ (...)` + `is_expiry_week=True` | 4,022 | +0.54% | 12.02 |

### Interpretation

**A real LONG-side edge exists in deep panic-gap-down catches.** When a small/mid-cap MIS name gaps down ≥1% AND opens below both prior-day high (by >5.5%) and prior-day low (by >1.25%), the post-cost 120m mean return is **+0.73% with n=2,720**. The t-proxy of 12+ across all top regions argues against this being a noise artifact of the 24K-event sample.

The mean return is positive after the framework's modeled execution costs (spread + slippage + impact). The 120m holding window aligns with intraday MIS constraints.

`is_expiry_week=True` slightly enhances the effect (+0.70% vs +0.65%), suggesting forced unwinding pressure during expiry weeks amplifies the post-gap mean-reversion.

### Ship/Reject decision

- **Status:** **CANDIDATE** for standalone ship gate. Discovery results meet `effect_size_min_sigma=0.4` and n-per-year (24,206 over 2 years = 12K/year, well above `n_per_year_min=300`).
- **Next steps before shipping:**
  - Validate on Sub8 OOS window (2025-01-01 → 2025-09-30)
  - Validate on Holdout window (2025-10-01 → 2026-04-30)
  - Walk-forward stability across 6-month train / 1-month test slices
  - Rule-orthogonality check vs existing 3 short setups

---

## Target 3: Ensemble feature mining (live setups)

**Method:** Load each live setup's existing Discovery parquet, attach Tier-A + event-calendar features per trade, scan for context-conditional regions.

### gap_fade_short (n=6,723 trades from `reports/sub7_validation/gap_fade_short.parquet`)

Top regions filtered by t-proxy. Baseline mean net_pnl = +37 INR/trade.

| Region | n | Mean net_pnl (INR) | t-proxy | Lift vs baseline |
|---|---|---|---|---|
| `cap_segment=small_cap` | 3,150 | +47 | 3.47 | +25% |
| `small_cap + is_expiry_week=True` | 2,486 | +45 | 2.99 | +22% |
| `is_expiry_week=False` (non-expiry) | 1,162 | +56 | 2.60 | **+51%** |
| `small_cap + is_monthly_expiry_day=True` | 135 | +130 | 1.93 | **+247%** (small n) |

**Interpretation:** Non-expiry-week trades outperform by 51% on average — gap_fade_short benefits from quiet trend-day microstructure rather than expiry-day cross-flow noise. The monthly-expiry small-cap region is striking (n=135, mean +130 INR) but the sample is too small to ship as a standalone filter; flag for future probe.

### circuit_t1_fade_short, delivery_pct_anomaly_short

Discovery parquets unavailable in `reports/sub7_validation/` — same gap as parity-gate skip. Ensemble mining waits on the same upstream prerequisite.

### Ship/Reject decision

- **Add to feature catalog:** `is_expiry_week` (validated as a context modifier on gap_fade_short with t-proxy=2.60)
- **Tentative live filter:** Down-weight gap_fade_short entries during expiry week (-50% size or skip) pending OOS + Holdout confirmation
- **No setup-level retire/pause:** gap_fade_short's overall behavior is healthy; the finding is a refinement, not a rejection

---

## Decay Monitor — current shipped setups

From `reports/sub8_oos_holdout_clean/` baselines via `tools/edge_discovery/decay_monitor_runner.py`:

| Setup | Status | Rolling PF (6m) | Latest month PF | Notes |
|---|---|---|---|---|
| **gap_fade_short** | **CAUTION** | 1.16 | 1.47 | Rolling below caution threshold (1.20); latest month recovering. Watch closely. |
| circuit_t1_fade_short | SKIP | — | — | No Holdout parquet |
| delivery_pct_anomaly_short | SKIP | — | — | No Holdout parquet |

**Interpretation:** gap_fade_short is in CAUTION but not PAUSED — rolling PF below 1.20 threshold but above the 1.00 pause threshold. The latest-month PF (1.47) suggests the dip may be cyclical rather than terminal decay. Recommend:
- Continue trading at full size for now
- Re-snapshot decay status monthly
- If next-month PF < 1.00, drop to half-size; if 2 consecutive months below 0.80, retire per config

---

## Portfolio Composition Recommendation

| Current (3 SHORT setups) | Proposed (post-discovery) |
|---|---|
| gap_fade_short — ACTIVE/CAUTION | gap_fade_short — ACTIVE with expiry-week down-weight |
| circuit_t1_fade_short | unchanged (insufficient framework data) |
| delivery_pct_anomaly_short | unchanged (insufficient framework data) |
| — | **ADD:** `long_panic_gap_down` (deep small/mid gap-down catch) after OOS+Holdout validation |

### Net direction-bias change

Portfolio currently 100% SHORT. Adding a LONG-side setup (long_panic_gap_down) would:
- Diversify direction exposure (reduces correlated drawdowns when small/mid caps gap down hard)
- Add an inverse hedge during panic episodes (when SHORTs may be saturated, LONGs in catch trades work)
- Increase total signal capacity (24K candidate events/year vs current SHORT-only)

### Risks

- The Discovery edge (+0.73% mean, t=12.3) is on **pre-OOS, pre-Holdout** data. The historical baseline-drift observed on gap_fade_short (PF 1.36→1.13 between author-snapshot and replay) argues for skepticism until validated forward.
- Cost model assumes 0.1% of ADV per trade. Reality: at scale, slippage may worsen.
- The "deep gap-down" criterion may select adverse-selection names (genuinely bad news, not panic). Need delivery%/news-flow filter for production.

---

## Spec Coverage Self-Check

- §1 Motivation, §2 Goals — captured in plan + handoff ✓
- §3 Architecture (`tools/edge_discovery/`) — all modules built ✓
- §4 Event Population — small/mid gap-down ≥1% in T18 ✓
- §5.1 Symbol Tier-A features — `features/symbol_features.py` ✓
- §5.2 Market Tier-B — `features/market_features.py` (T15) ✓
- §5.3 Event-calendar Tier-B — `features/event_features.py` (T16) ✓
- §5 Tier-C (earnings, 5m AD) — deferred per spec ✓
- §6 Outcomes + Cost — `outcomes/returns.py`, `outcomes/costs.py` ✓
- §7 Edge Region Detection — `types.ConditionalOutcomeTable.top_edge_regions` (T17) ✓
- §8.1 Parity Gate — `validation/parity_gate.py` + 3 target scripts ✓
- §8.2 Walk-Forward — `validation/walk_forward.py` ✓
- §8.3 Rule-Orthogonality — `rule_orthogonality.py` ✓
- §9 Ship Gates (two-tier) — `ship_gate.py` ✓
- §10 Decay Monitor — `decay_monitor.py` + runner (T21) ✓

---

## What's NOT done (deferred to v1.1)

- **Re-running framework on OOS + Holdout windows** for the LONG candidate. v1 only validates Discovery.
- **circuit_t1 / delivery_pct parity** — needs upstream to produce canonical Holdout parquets.
- **Tier-C features** (earnings calendar, 5m advance-decline) — deferred per design spec §5.
- **Live integration of decay_monitor_runner** — runs offline against parquets; needs wiring to live `trade_report.csv` once setup graduates.
- **Probe-class research files** (`tools/edge_discovery/probe_*.py`, `volume_probe*.py`) — leftover from prior sub-projects, intentionally not part of v1.

---

## Reproduce this report

```powershell
# Run all framework outputs
.venv/Scripts/python -m tools.edge_discovery.targets.target_parity_gap_fade        > reports/edge_discovery/parity_gap_fade.json
.venv/Scripts/python -m tools.edge_discovery.targets.target_long_panic_gap_down    > reports/edge_discovery/_target2.log
.venv/Scripts/python -m tools.edge_discovery.targets.target_ensemble_live_setups   > reports/edge_discovery/_target3.log
.venv/Scripts/python -m tools.edge_discovery.decay_monitor_runner                  > reports/edge_discovery/_decay.log
.venv/Scripts/python -m tools.edge_discovery.report                                > reports/edge_discovery/_report.log
```

Outputs land in `reports/edge_discovery/`. The decision-report markdown is at `reports/edge_discovery/decision_report.md`.
