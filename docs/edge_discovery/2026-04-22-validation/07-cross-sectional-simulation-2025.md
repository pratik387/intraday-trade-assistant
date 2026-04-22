# Stage 5c — Cross-Sectional Filter Simulation

**Purpose:** Replay F1 (RVOL cap-conditional) + F2 (crowdedness universal) filters on the Stage-5b trade stream. Report before/after aggregate metrics.

## Scenarios

| scenario | n_trades | n_sessions | trades_per_day | total_pnl | pf | wr_pct | session_sharpe | losing_days_pct |
|---|---|---|---|---|---|---|---|---|
| Before CrossSectionalGate | 73690 | 181 | 407.1 | 9247036.0 | 1.236 | 53.1 | 0.389 | 29.3 |
| After CrossSectionalGate (F1+F2) | 41573 | 181 | 229.7 | 6424324.0 | 1.298 | 54.5 | 0.432 | 29.8 |

## Top rejection reasons

| reason | count |
|---|---|
| f2_crowded_count=40>=40 | 271 |
| f2_crowded_count=41>=40 | 257 |
| f2_crowded_count=43>=40 | 247 |
| f2_crowded_count=42>=40 | 247 |
| f1_rvol_pct=100.0>=70.0 | 230 |
| f2_crowded_count=44>=40 | 214 |
| f2_crowded_count=46>=40 | 213 |
| f2_crowded_count=45>=40 | 209 |
| f2_crowded_count=47>=40 | 197 |
| f2_crowded_count=48>=40 | 177 |
