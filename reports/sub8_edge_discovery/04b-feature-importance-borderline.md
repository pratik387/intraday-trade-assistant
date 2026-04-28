# Stage 4: SHAP feature importance (interpretation aid)
Per design §3.3: identifies missed structural drivers; does NOT kill rules.
Discovery period: 2023-01-01 → 2024-12-31

## orb_15
N=47,754 | win_rate=46.8% | n_features=10

Top features by mean |SHAP|:

| Rank | Feature | Mean abs SHAP |
|---:|:---|---:|
| 1 | `minute_of_day` | 0.0620 |
| 2 | `regime_trend_up` | 0.0287 |
| 3 | `regime_squeeze` | 0.0266 |
| 4 | `regime_chop` | 0.0212 |
| 5 | `regime_trend_down` | 0.0144 |
| 6 | `cap_segment_mid_cap` | 0.0075 |
| 7 | `hour_bucket_morning` | 0.0064 |
| 8 | `cap_segment_large_cap` | 0.0053 |
| 9 | `cap_segment_unknown` | 0.0033 |
| 10 | `hour_bucket_opening` | 0.0011 |

**Note**: top-5 includes non-conditioner feature(s) ['minute_of_day']. Consider whether any are STRUCTURAL (stable across regimes, interpretable) and could be added to Stage 3 conditioners.

## narrow_cpr_breakout
N=36,534 | win_rate=45.1% | n_features=11

Top features by mean |SHAP|:

| Rank | Feature | Mean abs SHAP |
|---:|:---|---:|
| 1 | `minute_of_day` | 0.0814 |
| 2 | `regime_squeeze` | 0.0354 |
| 3 | `regime_trend_down` | 0.0254 |
| 4 | `regime_trend_up` | 0.0207 |
| 5 | `regime_chop` | 0.0163 |
| 6 | `hour_bucket_morning` | 0.0132 |
| 7 | `cap_segment_large_cap` | 0.0111 |
| 8 | `hour_bucket_afternoon` | 0.0098 |
| 9 | `hour_bucket_lunch` | 0.0088 |
| 10 | `cap_segment_unknown` | 0.0039 |
| 11 | `hour_bucket_opening` | 0.0037 |

**Note**: top-5 includes non-conditioner feature(s) ['minute_of_day']. Consider whether any are STRUCTURAL (stable across regimes, interpretable) and could be added to Stage 3 conditioners.

## cpr_mean_revert
N=72,196 | win_rate=27.7% | n_features=11

Top features by mean |SHAP|:

| Rank | Feature | Mean abs SHAP |
|---:|:---|---:|
| 1 | `regime_trend_down` | 0.1044 |
| 2 | `minute_of_day` | 0.0731 |
| 3 | `cap_segment_small_cap` | 0.0300 |
| 4 | `cap_segment_large_cap` | 0.0299 |
| 5 | `cap_segment_mid_cap` | 0.0189 |
| 6 | `regime_squeeze` | 0.0170 |
| 7 | `regime_trend_up` | 0.0139 |
| 8 | `regime_chop` | 0.0130 |
| 9 | `hour_bucket_morning` | 0.0105 |
| 10 | `hour_bucket_lunch` | 0.0096 |
| 11 | `hour_bucket_afternoon` | 0.0078 |

**Note**: top-5 includes non-conditioner feature(s) ['minute_of_day']. Consider whether any are STRUCTURAL (stable across regimes, interpretable) and could be added to Stage 3 conditioners.

## vwap_first_pullback
N=101,773 | win_rate=37.9% | n_features=12

Top features by mean |SHAP|:

| Rank | Feature | Mean abs SHAP |
|---:|:---|---:|
| 1 | `minute_of_day` | 0.0760 |
| 2 | `regime_trend_up` | 0.0148 |
| 3 | `regime_squeeze` | 0.0100 |
| 4 | `hour_bucket_morning` | 0.0091 |
| 5 | `regime_chop` | 0.0084 |
| 6 | `regime_trend_down` | 0.0076 |
| 7 | `hour_bucket_lunch` | 0.0055 |
| 8 | `hour_bucket_afternoon` | 0.0048 |
| 9 | `cap_segment_mid_cap` | 0.0041 |
| 10 | `cap_segment_large_cap` | 0.0025 |
| 11 | `cap_segment_unknown` | 0.0017 |
| 12 | `hour_bucket_late` | 0.0009 |

**Note**: top-5 includes non-conditioner feature(s) ['minute_of_day']. Consider whether any are STRUCTURAL (stable across regimes, interpretable) and could be added to Stage 3 conditioners.
