# Stage 4: SHAP feature importance (interpretation aid)
Per design §3.3: identifies missed structural drivers; does NOT kill rules.
Discovery period: 2023-01-01 → 2024-12-31

## closing_hour_reversal
N=9,197 | win_rate=32.8% | n_features=9

Top features by mean |SHAP|:

| Rank | Feature | Mean abs SHAP |
|---:|:---|---:|
| 1 | `minute_of_day` | 0.7553 |
| 2 | `regime_trend_down` | 0.0482 |
| 3 | `cap_segment_large_cap` | 0.0446 |
| 4 | `regime_squeeze` | 0.0416 |
| 5 | `regime_trend_up` | 0.0348 |
| 6 | `regime_chop` | 0.0177 |
| 7 | `cap_segment_unknown` | 0.0132 |
| 8 | `cap_segment_mid_cap` | 0.0099 |
| 9 | `hour_bucket_late` | 0.0000 |

**Note**: top-5 includes non-conditioner feature(s) ['minute_of_day']. Consider whether any are STRUCTURAL (stable across regimes, interpretable) and could be added to Stage 3 conditioners.

## gap_fade_short
N=6,796 | win_rate=63.0% | n_features=9

Top features by mean |SHAP|:

| Rank | Feature | Mean abs SHAP |
|---:|:---|---:|
| 1 | `minute_of_day` | 0.0950 |
| 2 | `cap_segment_small_cap` | 0.0629 |
| 3 | `regime_chop` | 0.0517 |
| 4 | `regime_trend_down` | 0.0339 |
| 5 | `cap_segment_mid_cap` | 0.0278 |
| 6 | `regime_trend_up` | 0.0258 |
| 7 | `cap_segment_micro_cap` | 0.0257 |
| 8 | `regime_squeeze` | 0.0167 |
| 9 | `hour_bucket_opening` | 0.0000 |

**Note**: top-5 includes non-conditioner feature(s) ['minute_of_day']. Consider whether any are STRUCTURAL (stable across regimes, interpretable) and could be added to Stage 3 conditioners.

## pdh_pdl_reject
N=7,241 | win_rate=41.0% | n_features=11

Top features by mean |SHAP|:

| Rank | Feature | Mean abs SHAP |
|---:|:---|---:|
| 1 | `minute_of_day` | 0.1878 |
| 2 | `regime_chop` | 0.0961 |
| 3 | `regime_squeeze` | 0.0608 |
| 4 | `regime_trend_down` | 0.0549 |
| 5 | `regime_trend_up` | 0.0368 |
| 6 | `hour_bucket_lunch` | 0.0222 |
| 7 | `cap_segment_large_cap` | 0.0204 |
| 8 | `hour_bucket_afternoon` | 0.0198 |
| 9 | `hour_bucket_morning` | 0.0185 |
| 10 | `cap_segment_mid_cap` | 0.0119 |
| 11 | `hour_bucket_late` | 0.0027 |

**Note**: top-5 includes non-conditioner feature(s) ['minute_of_day']. Consider whether any are STRUCTURAL (stable across regimes, interpretable) and could be added to Stage 3 conditioners.
