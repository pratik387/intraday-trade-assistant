# Gauntlet v2 — Search Report (sub5-T7)

- **Study:** gauntlet_v2
- **n_trials_requested:** 5
- **n_trials_completed:** 5
- **best_sharpe:** -inf (no qualifying trial)

## Best config overrides

```json
{
  "conviction_gate": {
    "daily_cap": 69,
    "min_predicted_r": 1.4014286128198323
  },
  "dedup_gate": {
    "cooloff_bars": 9,
    "require_setup_change": true
  },
  "cross_sectional_gate": {
    "f1_rvol_threshold_pct": 57.01975341512912,
    "f2_crowdedness_threshold": 14,
    "f2_crowdedness_window_min": 14
  },
  "rank_pctl_min": 0.7005575058716045
}
```

## Top 10 by sharpe

|   daily_cap |   min_predicted_r |   cooloff_bars | require_setup_change   |   f1_rvol_threshold_pct |   f2_crowdedness_threshold |   f2_crowdedness_window_min |   rank_pctl_min |   value |   number |   m_n_admits |   m_n_filled |   m_n_sessions |   m_fill_rate |   m_total_pnl |   m_pf |   m_sharpe |   m_wr_filled |   m_admits_per_day |   m_filled_per_day |   m_losing_days_pct |
|------------:|------------------:|---------------:|:-----------------------|------------------------:|---------------------------:|----------------------------:|----------------:|--------:|---------:|-------------:|-------------:|---------------:|--------------:|--------------:|-------:|-----------:|--------------:|-------------------:|-------------------:|--------------------:|
|          69 |         1.40143   |              9 | True                   |                 57.0198 |                         14 |                          14 |        0.700558 |    -inf |        0 |            0 |            0 |              0 |             0 |          0    |  0     |      0     |         0     |               0    |               0    |                 0   |
|         112 |        -0.458831  |             12 | True                   |                 58.1821 |                         23 |                           6 |        0.662378 |    -inf |        1 |          276 |          276 |            205 |             1 |       5167.32 |  1.078 |      0.033 |         0.504 |               1.35 |               1.35 |                51.2 |
|          76 |         0.0824583 |              7 | False                  |                 66.4863 |                         42 |                          13 |        0.499837 |    -inf |        2 |          354 |          354 |            205 |             1 |       7650.25 |  1.093 |      0.039 |         0.506 |               1.73 |               1.73 |                50.7 |
|          87 |         0.684829  |              0 | True                   |                 52.9273 |                         77 |                          15 |        0.804199 |    -inf |        3 |            0 |            0 |              0 |             0 |          0    |  0     |      0     |         0     |               0    |               0    |                 0   |
|          59 |        -0.304656  |              8 | True                   |                 72.283  |                         12 |                          14 |        0.52939  |    -inf |        4 |          276 |          276 |            205 |             1 |       5167.32 |  1.078 |      0.033 |         0.504 |               1.35 |               1.35 |                51.2 |