# Stage 5c — Cross-Sectional Filter Simulation

**Purpose:** Replay F1 (RVOL cap-conditional) + F2 (crowdedness universal) filters on the Stage-5b trade stream. Report before/after aggregate metrics.

## Scenarios

| scenario | n_trades | n_sessions | trades_per_day | total_pnl | pf | wr_pct | session_sharpe | losing_days_pct |
|---|---|---|---|---|---|---|---|---|
| Before CrossSectionalGate | 97404 | 242 | 402.5 | 12267136.0 | 1.24 | 53.7 | 0.398 | 28.9 |
| After CrossSectionalGate (F1+F2) | 54921 | 242 | 226.9 | 7889100.0 | 1.278 | 54.7 | 0.411 | 28.9 |

## Top rejection reasons

| reason | count |
|---|---|
| f2_crowded_count=40>=40 | 356 |
| f2_crowded_count=42>=40 | 317 |
| f2_crowded_count=41>=40 | 308 |
| f1_rvol_pct=100.0>=70.0 | 299 |
| f2_crowded_count=43>=40 | 298 |
| f2_crowded_count=44>=40 | 295 |
| f2_crowded_count=46>=40 | 274 |
| f2_crowded_count=45>=40 | 272 |
| f2_crowded_count=47>=40 | 260 |
| f2_crowded_count=48>=40 | 234 |
