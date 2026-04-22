# Conviction architecture decision

Comparison: per-setup XGBoost (trained only on the setup's trades) vs
the universal model (trained on all 74-survivor trades with setup_type
one-hot).  Both use pseudo-Huber loss + R<=5 Winsorization.  OOS eval
applies the 74-survivor rule filter AND restricts to the target setup.

| Setup | N (OOS) | Universal RMSE | Per-setup RMSE | Universal PF@50 | Per-setup PF@50 | Winner |
|---|---|---|---|---|---|---|
| premium_zone_short | 44728 | 6.300 | 6.303 | 6.431 | 6.250 | universal |
| range_bounce_short | 22893 | 5.642 | 5.645 | 6.816 | 6.764 | universal |

## Decision: Ship universal only (per-setup does not beat universal by >5% PF)
