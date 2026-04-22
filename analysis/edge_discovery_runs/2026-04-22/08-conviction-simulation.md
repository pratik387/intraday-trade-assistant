# Stage 5d — Conviction Gate Simulation

## Scenarios

| scenario | n_trades | n_sessions | trades_per_day | total_pnl | pf | wr_pct | session_sharpe | losing_days_pct |
|---|---|---|---|---|---|---|---|---|
| Before ConvictionGate | 178151 | 484 | 368.1 | 32087340.0 | 1.363 | 55.5 | 0.745 | 16.7 |
| After ConvictionGate | 23502 | 484 | 48.6 | 3455226.0 | 1.286 | 53.9 | 0.424 | 35.1 |

## Top rejection reasons

| reason | count |
|---|---|
| daily_cap_reached count=50>=50 | 34304 |
| below_threshold predicted_r=0.296<0.3 | 5450 |
| below_threshold predicted_r=0.299<0.3 | 4247 |
| below_threshold predicted_r=0.167<0.3 | 4179 |
| below_threshold predicted_r=0.236<0.3 | 3291 |
| below_threshold predicted_r=0.274<0.3 | 2601 |
| below_threshold predicted_r=0.286<0.3 | 2335 |
| below_threshold predicted_r=0.263<0.3 | 2283 |
| below_threshold predicted_r=0.277<0.3 | 1971 |
| below_threshold predicted_r=0.222<0.3 | 1772 |
