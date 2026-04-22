# Stage 5e - Budgeted Selector Simulation (Illiquid-aware)

Six-constraint selection replacing FIFO cap: opening embargo, time-bucket
quotas, ADV cap, bar participation cap, per-symbol rate limit, cap-segment
concurrency. Daily trade count is emergent, not an input.

## Scenarios

| scenario | n_trades | n_sessions | trades_per_day | total_pnl | pf | wr_pct | session_sharpe | losing_days_pct |
|---|---|---|---|---|---|---|---|---|
| Before BudgetedSelector | 41573 | 181 | 229.7 | 6424324.0 | 1.298 | 54.5 | 0.432 | 29.8 |
| After BudgetedSelector | 7424 | 181 | 41.0 | 1223111.0 | 1.325 | 53.7 | 0.454 | 33.1 |

## Admitted by time bucket

| bucket | admitted_count |
|---|---|
| morning | 4525 |
| afternoon | 1635 |
| lunch | 1264 |

## Admitted by cap segment

| cap_segment | admitted_count |
|---|---|
| small_cap | 2907 |
| mid_cap | 1645 |
| large_cap | 1425 |
| unknown | 1250 |
| micro_cap | 197 |

## Top rejection reasons

| reason | count |
|---|---|
| bucket_quota:morning_25_of_25 | 24886 |
| bucket_quota:lunch_7_of_7 | 4436 |
| bucket_quota:opening_0_of_0 | 3605 |
| embargo:opening_micro_unknown | 743 |
| bucket_quota:afternoon_15_of_15 | 479 |
