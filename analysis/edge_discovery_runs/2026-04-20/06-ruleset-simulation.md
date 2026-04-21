# Stage 5b — Ruleset Simulation

**Purpose:** Apply approved Stage-5 rules as union filter; report aggregate PF / WR / session-Sharpe / daily-count to verify ruleset coherence before Validation.

**Approved rules simulated:** 90

## Scenarios

| scenario | n_trades | n_sessions | trades_per_day | total_pnl | pf | wr_pct | session_sharpe | losing_days_pct | median_daily_pnl |
|---|---|---|---|---|---|---|---|---|---|
| Baseline (raw wide-open, no filter) | 389535 | 484 | 804.8 | 18750051.0 | 1.092 | 52.5 | 0.536 | 25.0 | 41876.0 |
| All approved rules (union filter) | 178151 | 484 | 368.1 | 32087340.0 | 1.363 | 55.5 | 0.745 | 16.7 | 67001.0 |
| Exclude opening+morning entries | 16578 | 481 | 34.5 | 2861214.0 | 1.439 | 56.4 | 0.373 | 28.1 | 4546.0 |
| Late-hour entries only | 946 | 354 | 2.7 | 142265.0 | 1.566 | 54.9 | 0.216 | 37.9 | 270.0 |
| Afternoon+late entries only | 14807 | 480 | 30.8 | 2548163.0 | 1.451 | 56.3 | 0.346 | 27.9 | 3913.0 |

## Per-hour breakdown (filtered set, entry-time hour_bucket)

| hour_bucket | n | total_pnl | avg_pnl | pf | wr_pct |
|---|---|---|---|---|---|
| afternoon | 13861 | 2405898.0 | 173.6 | 1.446 | 56.4 |
| late | 946 | 142265.0 | 150.4 | 1.566 | 54.9 |
| lunch | 1771 | 313051.0 | 176.8 | 1.361 | 57.0 |
| morning | 104177 | 18897984.0 | 181.4 | 1.353 | 55.6 |
| opening | 57396 | 10328142.0 | 179.9 | 1.364 | 54.9 |

## Per-setup breakdown (filtered set)

| setup | n | total_pnl | avg_pnl | pf | wr_pct |
|---|---|---|---|---|---|
| order_block_short | 6825 | 1331449.0 | 195.1 | 1.398 | 58.2 |
| premium_zone_short | 104583 | 17321099.0 | 165.6 | 1.336 | 54.8 |
| range_bounce_short | 57812 | 11836129.0 | 204.7 | 1.398 | 55.9 |
| resistance_bounce_short | 7998 | 1445512.0 | 180.7 | 1.423 | 58.4 |
| vwap_lose_short | 933 | 153151.0 | 164.1 | 1.363 | 58.0 |
