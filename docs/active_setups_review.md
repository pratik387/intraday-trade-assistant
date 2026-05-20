# Active Setups — Walk-Forward Consistency Review

**Generated:** 2026-05-20 (after Schema Stage 3 of walk-forward methodology rollout)

Per **Option C** (user decision 2026-05-19): active setups that did not clear the new walk-forward GREEN threshold are kept running at full size, but are now monitored by the daily circuit breaker. **No proactive retires.**

## Important context

Per **lesson #13** (sanity Mode B over-optimism): canonical walk-forward on sanity data is **NOT a reliable production verdict**. Sanity Mode B (next-bar-OPEN entry) can over-estimate OR under-estimate production PF depending on the execution-semantics gap. For active setups, the production OCI runs use cell-locked filters that sanity aggregates miss.

**Therefore the verdicts in this document are INFORMATIONAL, not actionable.** Setups continue to run based on documented production performance unless live drawdown trips the circuit breaker.

## Walk-forward tier per active setup

| Setup | Sanity walk-forward tier | Trades (sanity) | Production status (per config) | Disposition |
|---|---|---|---|---|
| gap_fade_short | N/A — no per-trade CSV available | — | Disc 2.36 / OOS 1.71 / HO 1.69 (post-sweep, small_cap only) | Keep running. Watch list. |
| circuit_t1_fade_short | N/A — only 12 months of data (2024 + HO) | 646 (insufficient) | Disc 1.74 / OOS 0.98 / HO 1.89 (per agent analysis, 2026-05-19) | Keep running. Watch list. |
| delivery_pct_anomaly_short | RED 0/13 (sanity SHORT-only aggregate) | 2856 SHORT | Disc 1.49 / OOS 1.40 / HO 1.46 (aggregate, no cell) | Keep running. Aggregate-vs-cell gap suggests cell-locked production beats sanity. Watch list. |
| long_panic_gap_down | RED 3/13 (sanity aggregate) | 17324 | Disc 1.45 / OOS 1.40 / HO 1.72 (Cell B narrow: dist_from_pdl [-5,-3]) | Keep running. Cell B is much narrower than sanity aggregate. Watch list. |
| or_window_failure_fade_short | RED 3/13 (sanity SHORT-only aggregate) | 19925 | Disc 1.22 / OOS 1.27 / HO 1.12 (Cell B narrow: vol-ratio (8,15]) | Keep running. Watch list. |

## Why all 5 are on the watch list

Either:
1. **Sanity data is insufficient or unavailable** (gap_fade_short, circuit_t1_fade_short) — can't walk-forward.
2. **Sanity walk-forward shows RED but production uses cell-locked filters that aren't captured by sanity** (delivery_pct, long_panic, or_window_failure). The aggregate-vs-cell mismatch is real and consistent.

Per **lesson #13**, this is the predicted state when running walk-forward on sanity Mode B data instead of production trade_report.csv. **The right next step is Stage 6: walk-forward on production OCI trade_report.csv** — that would give the actionable verdict.

## Circuit breaker thresholds (auto-disable if breached)

For each active setup, the circuit breaker job (`jobs/check_circuit_breakers.py`) monitors trailing-60d NET PnL. Thresholds (auto-computed from sanity walk-forward, mean − 2σ):

| Setup | cb_drawdown_threshold (Rs) | cb_state |
|---|---|---|
| delivery_pct_anomaly_short | -301.59 | (not auto-set — Option C keeps enabled) |
| long_panic_gap_down | -5162.49 | (not auto-set — Option C keeps enabled) |
| or_window_failure_fade_short | -692.29 | (not auto-set — Option C keeps enabled) |
| gap_fade_short | (no walk-forward run — sanity data missing) | enabled |
| circuit_t1_fade_short | (no walk-forward run — insufficient data) | enabled |

**Manual action required:** when the circuit breaker job is first deployed, set explicit `cb_drawdown_threshold` values for gap_fade_short and circuit_t1_fade_short based on production trade_report.csv distribution (not sanity).

## Recommended Stage 6 (next session)

Build a "production-data walk-forward" pathway:
1. Read latest OCI `analysis/reports/3year_backtest/run_<latest>/trade_report.csv`
2. Filter by `setup_type` for each active setup
3. Run walk-forward on production trades (real execution outcomes, not sanity Mode B)
4. Compare verdicts to current sanity-based verdicts in this document
5. Cell-locked production verdicts should agree directionally with the config `_status_*` claims — that's the validation

This would be the **actionable** walk-forward for active setup decisions.

## Monthly review checklist

Researchers should review this file monthly:
- Check trailing-60d PnL for each watch-list setup vs cb_drawdown_threshold
- If any setup approaches threshold, investigate
- If any setup trips the circuit breaker, treat as auto-disable signal
- If 3 consecutive months below mean, schedule a re-run of Stage 6 walk-forward
