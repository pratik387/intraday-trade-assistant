# Stage 5e - Budgeted Selector Simulation (Illiquid-aware)

Six-constraint selection replacing FIFO cap: opening embargo, time-bucket
quotas, ADV cap, bar participation cap, per-symbol rate limit, cap-segment
concurrency. Daily trade count is emergent, not an input.

## Scenarios

| scenario | n_trades | n_sessions | trades_per_day | total_pnl | pf | wr_pct | session_sharpe | losing_days_pct |
|---|---|---|---|---|---|---|---|---|
| Before BudgetedSelector | 99891 | 484 | 206.4 | 19381751.0 | 1.397 | 56.5 | 0.699 | 19.0 |
| After BudgetedSelector | 21145 | 484 | 43.7 | 4123729.0 | 1.407 | 55.6 | 0.63 | 25.4 |

## Admitted by time bucket

| bucket | admitted_count |
|---|---|
| morning | 12094 |
| afternoon | 5665 |
| lunch | 3386 |

## Admitted by cap segment

| cap_segment | admitted_count |
|---|---|
| small_cap | 8869 |
| mid_cap | 5369 |
| large_cap | 3471 |
| unknown | 3001 |
| micro_cap | 435 |

## Top rejection reasons

| reason | count |
|---|---|
| bucket_quota:morning_25_of_25 | 52394 |
| bucket_quota:lunch_7_of_7 | 11856 |
| bucket_quota:opening_0_of_0 | 11138 |
| bucket_quota:afternoon_15_of_15 | 1797 |
| embargo:opening_micro_unknown | 1561 |
