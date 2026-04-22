# Stage 5d — Conviction Gate Simulation

## Scenarios

| scenario | n_trades | n_sessions | trades_per_day | total_pnl | pf | wr_pct | session_sharpe | losing_days_pct |
|---|---|---|---|---|---|---|---|---|
| Before ConvictionGate | 99891 | 484 | 206.4 | 19381751.0 | 1.397 | 56.5 | 0.699 | 19.0 |
| After ConvictionGate | 24156 | 484 | 49.9 | 6323156.0 | 1.551 | 59.4 | 0.798 | 21.5 |

## Top rejection reasons

| reason | count |
|---|---|
| daily_cap_reached count=50>=50 | 36307 |
| below_threshold predicted_r=0.286<0.3 | 340 |
| below_threshold predicted_r=0.287<0.3 | 310 |
| below_threshold predicted_r=0.288<0.3 | 279 |
| below_threshold predicted_r=0.296<0.3 | 268 |
| below_threshold predicted_r=0.284<0.3 | 267 |
| below_threshold predicted_r=0.285<0.3 | 265 |
| below_threshold predicted_r=0.280<0.3 | 261 |
| below_threshold predicted_r=0.293<0.3 | 260 |
| below_threshold predicted_r=0.274<0.3 | 258 |
