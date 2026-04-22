# Stage 5c — Cross-Sectional Filter Simulation

**Purpose:** Replay F1 (RVOL cap-conditional) + F2 (crowdedness universal) filters on the Stage-5b trade stream. Report before/after aggregate metrics.

## Scenarios

| scenario | n_trades | n_sessions | trades_per_day | total_pnl | pf | wr_pct | session_sharpe | losing_days_pct |
|---|---|---|---|---|---|---|---|---|
| Before CrossSectionalGate | 178151 | 484 | 368.1 | 32087340.0 | 1.363 | 55.5 | 0.745 | 16.7 |
| After CrossSectionalGate (F1+F2) | 99891 | 484 | 206.4 | 19381751.0 | 1.397 | 56.5 | 0.699 | 19.0 |

## Top rejection reasons

| reason | count |
|---|---|
| f1_rvol_pct=100.0>=70.0 | 543 |
| f2_crowded_count=40>=40 | 435 |
| f2_crowded_count=43>=40 | 412 |
| f2_crowded_count=41>=40 | 412 |
| f2_crowded_count=44>=40 | 409 |
| f2_crowded_count=42>=40 | 399 |
| f2_crowded_count=45>=40 | 383 |
| f2_crowded_count=47>=40 | 380 |
| f2_crowded_count=46>=40 | 373 |
| f2_crowded_count=48>=40 | 369 |
