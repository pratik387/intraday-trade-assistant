# gap_fade_short — Gauntlet-Lite Filter Sweep

**Baseline**: n=6,723 | PF=1.153 | Sharpe=0.092 | WR=47.0% | Net PnL=Rs210,280

## Top 10 Single-Dim Slices (by PF)

| Filter | n | PF | Sharpe | WR | Net PnL |
|--------|---|----|--------|----|---------|
| minute=15 | 6 | 999.000 | 0.645 | 33.3% | Rs756 |
| minute=35 | 1 | 999.000 | 0.000 | 100.0% | Rs646 |
| cap_segment=micro_cap | 148 | 1.591 | 0.135 | 38.5% | Rs14,983 |
| minute=25 | 2,038 | 1.334 | 0.153 | 46.1% | Rs115,819 |
| dow=1 | 1,546 | 1.293 | 0.150 | 48.6% | Rs79,919 |
| regime=squeeze | 876 | 1.254 | 0.096 | 48.3% | Rs38,016 |
| dow=4 | 1,395 | 1.234 | 0.181 | 48.1% | Rs65,419 |
| regime=trend_up | 3,244 | 1.232 | 0.140 | 48.3% | Rs163,455 |
| dow=3 | 1,208 | 1.206 | 0.120 | 49.4% | Rs50,547 |
| cap_segment=small_cap | 3,762 | 1.187 | 0.104 | 47.7% | Rs147,150 |

## Top 10 2-Dim Crosses (n >= 200, by PF)

| Filter | n | PF | Sharpe | WR | Net PnL |
|--------|---|----|--------|----|---------|
| dow=1 & minute=25 | 479 | 1.780 | 0.260 | 49.3% | Rs51,244 |
| dow=4 & minute=20 | 317 | 1.576 | 0.196 | 47.0% | Rs30,418 |
| regime=squeeze & minute=25 | 258 | 1.531 | 0.155 | 47.7% | Rs20,445 |
| regime=squeeze & dow=1 | 218 | 1.531 | 0.138 | 51.4% | Rs17,620 |
| dow=2 & minute=25 | 366 | 1.480 | 0.212 | 47.3% | Rs25,203 |
| cap_segment=small_cap & minute=25 | 1,147 | 1.472 | 0.169 | 47.3% | Rs90,639 |
| regime=trend_up & dow=1 | 716 | 1.333 | 0.209 | 47.6% | Rs45,021 |
| regime=trend_up & dow=4 | 705 | 1.332 | 0.228 | 50.1% | Rs49,763 |
| cap_segment=small_cap & dow=1 | 835 | 1.326 | 0.159 | 49.6% | Rs49,198 |
| regime=trend_down & dow=1 | 462 | 1.324 | 0.105 | 49.6% | Rs22,305 |

## Best Combined Filter (n >= 1000)

**Filter**: `{'regime': 'trend_up'}`
**Metrics**: n=3,244 | PF=1.232 | Sharpe=0.140 | WR=48.3% | Net PnL=Rs163,455

## Phase-2 Verdict (PF >= 1.25 AND Sharpe >= 0.60 AND n >= 1000)

**FAIL** — no filter meets Phase-2 thresholds