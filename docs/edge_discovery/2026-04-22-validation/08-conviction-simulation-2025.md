# Stage 5d — Conviction Gate Simulation

## Scenarios

| scenario | n_trades | n_sessions | trades_per_day | total_pnl | pf | wr_pct | session_sharpe | losing_days_pct |
|---|---|---|---|---|---|---|---|---|
| Before ConvictionGate | 41573 | 181 | 229.7 | 6424324.0 | 1.298 | 54.5 | 0.432 | 29.8 |
| After ConvictionGate | 8970 | 181 | 49.6 | 1398121.0 | 1.293 | 54.6 | 0.394 | 33.1 |

## Top rejection reasons

| reason | count |
|---|---|
| daily_cap_reached count=50>=50 | 14833 |
| below_threshold predicted_r=0.299<0.3 | 137 |
| below_threshold predicted_r=0.297<0.3 | 130 |
| below_threshold predicted_r=0.295<0.3 | 123 |
| below_threshold predicted_r=0.283<0.3 | 122 |
| below_threshold predicted_r=0.287<0.3 | 121 |
| below_threshold predicted_r=0.276<0.3 | 119 |
| below_threshold predicted_r=0.286<0.3 | 119 |
| below_threshold predicted_r=0.278<0.3 | 113 |
| below_threshold predicted_r=0.224<0.3 | 111 |
