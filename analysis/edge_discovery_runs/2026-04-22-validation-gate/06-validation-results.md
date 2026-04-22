# Validation Gate Results

**Criteria (per spec §3.4, ALL required):** pf_val >= 1.0 AND n_val >= 50 AND |wr_delta| <= 10.0pp

**Rules evaluated:** 90
**Passed:** 74
**Failed:** 16

## Per-rule results

| rule_id | status | n_val | pf_val | wr_val | wr_disc | wr_delta | fail_reasons |
|---|---|---|---|---|---|---|---|
| resistance_bounce_short__cap_segment+hour_bucket=unknown+afternoon | PASS | 55 | 2.642 | 63.64% | 63.69% | 0.05pp | — |
| range_bounce_short__cap_segment+hour_bucket=unknown+afternoon | PASS | 212 | 2.078 | 59.43% | 59.16% | 0.27pp | — |
| range_bounce_short__regime+cap_segment=trend_down+unknown | PASS | 646 | 1.898 | 60.84% | 58.34% | 2.5pp | — |
| range_bounce_short__regime+cap_segment=trend_down+micro_cap | PASS | 224 | 1.786 | 64.29% | 58.8% | 5.49pp | — |
| order_block_short__cap_segment+hour_bucket=unknown+opening | PASS | 147 | 1.777 | 59.86% | 55.74% | 4.12pp | — |
| order_block_short__cap_segment=unknown | PASS | 193 | 1.72 | 59.59% | 57.55% | 2.04pp | — |
| order_block_short__regime+cap_segment=trend_down+unknown | PASS | 89 | 1.701 | 58.43% | 59.8% | 1.37pp | — |
| range_bounce_short__cap_segment=unknown | PASS | 2185 | 1.657 | 57.3% | 56.44% | 0.86pp | — |
| range_bounce_short__cap_segment+hour_bucket=unknown+morning | PASS | 1906 | 1.626 | 57.19% | 56.18% | 1.01pp | — |
| range_bounce_short__regime+cap_segment=chop+unknown | PASS | 671 | 1.604 | 55.74% | 54.98% | 0.76pp | — |
| range_bounce_short__cap_segment=micro_cap | PASS | 709 | 1.574 | 60.93% | 55.51% | 5.42pp | — |
| range_bounce_short__regime+cap_segment=chop+small_cap | PASS | 1624 | 1.574 | 57.82% | 57.01% | 0.81pp | — |
| premium_zone_short__regime+cap_segment=trend_down+unknown | PASS | 1139 | 1.573 | 57.51% | 57.8% | 0.29pp | — |
| range_bounce_short__cap_segment+hour_bucket=micro_cap+morning | PASS | 667 | 1.566 | 61.02% | 55.56% | 5.46pp | — |
| resistance_bounce_short__cap_segment=unknown | PASS | 466 | 1.566 | 59.23% | 57.2% | 2.03pp | — |
| vwap_lose_short__regime=trend_up | PASS | 97 | 1.55 | 61.86% | 59.66% | 2.2pp | — |
| range_bounce_short__cap_segment+hour_bucket=small_cap+lunch | PASS | 162 | 1.548 | 60.49% | 59.38% | 1.11pp | — |
| range_bounce_short__regime+cap_segment=trend_up+unknown | PASS | 845 | 1.532 | 55.86% | 56.4% | 0.54pp | — |
| vwap_lose_short__cap_segment=large_cap | PASS | 236 | 1.482 | 62.71% | 57.66% | 5.05pp | — |
| premium_zone_short__cap_segment=unknown | PASS | 4223 | 1.468 | 54.49% | 55.85% | 1.36pp | — |
| premium_zone_short__cap_segment+hour_bucket=unknown+opening | PASS | 1554 | 1.427 | 51.16% | 56.11% | 4.95pp | — |
| range_bounce_short__regime+cap_segment=chop+micro_cap | PASS | 95 | 1.42 | 54.74% | 55.78% | 1.04pp | — |
| range_bounce_short__regime+cap_segment=trend_up+small_cap | PASS | 2682 | 1.41 | 56.0% | 56.58% | 0.58pp | — |
| range_bounce_short__regime+hour_bucket=chop+morning | PASS | 3518 | 1.404 | 55.26% | 55.7% | 0.44pp | — |
| range_bounce_short__regime=chop | PASS | 3915 | 1.403 | 55.22% | 56.03% | 0.81pp | — |
| vwap_lose_short__regime=squeeze | PASS | 102 | 1.398 | 56.86% | 60.47% | 3.61pp | — |
| range_bounce_short__regime+cap_segment=squeeze+micro_cap | PASS | 157 | 1.384 | 57.32% | 54.41% | 2.91pp | — |
| range_bounce_short__cap_segment+hour_bucket=small_cap+morning | PASS | 9402 | 1.372 | 55.71% | 57.09% | 1.38pp | — |
| premium_zone_short__regime+cap_segment=squeeze+small_cap | PASS | 5215 | 1.372 | 55.38% | 57.57% | 2.19pp | — |
| range_bounce_short__cap_segment=small_cap | PASS | 10467 | 1.368 | 55.68% | 57.35% | 1.67pp | — |
| range_bounce_short__regime+cap_segment=squeeze+small_cap | PASS | 2971 | 1.361 | 57.05% | 59.61% | 2.56pp | — |
| range_bounce_short__regime+hour_bucket=chop+afternoon | PASS | 306 | 1.352 | 52.61% | 59.62% | 7.01pp | — |
| order_block_short__cap_segment+hour_bucket=small_cap+morning | PASS | 282 | 1.35 | 56.74% | 62.63% | 5.89pp | — |
| resistance_bounce_short__cap_segment=small_cap | PASS | 2189 | 1.331 | 58.15% | 60.47% | 2.32pp | — |
| range_bounce_short__cap_segment+hour_bucket=small_cap+afternoon | PASS | 809 | 1.322 | 55.5% | 59.22% | 3.72pp | — |
| order_block_short__regime+hour_bucket=chop+opening | PASS | 155 | 1.312 | 54.19% | 56.48% | 2.29pp | — |
| order_block_short__regime+hour_bucket=squeeze+morning | PASS | 173 | 1.304 | 60.69% | 67.11% | 6.42pp | — |
| range_bounce_short__hour_bucket=morning | PASS | 20736 | 1.297 | 54.75% | 55.66% | 0.91pp | — |
| range_bounce_short__regime+hour_bucket=trend_up+morning | PASS | 6020 | 1.295 | 54.52% | 55.04% | 0.52pp | — |
| range_bounce_short__regime=trend_up | PASS | 6659 | 1.292 | 54.36% | 55.12% | 0.76pp | — |
| order_block_short__regime+cap_segment=trend_down+small_cap | PASS | 387 | 1.289 | 52.71% | 60.18% | 7.47pp | — |
| range_bounce_short__regime+hour_bucket=trend_up+afternoon | PASS | 479 | 1.288 | 55.11% | 55.41% | 0.3pp | — |
| range_bounce_short__hour_bucket=lunch | PASS | 346 | 1.27 | 55.2% | 56.78% | 1.58pp | — |
| range_bounce_short__regime+hour_bucket=trend_down+morning | PASS | 5985 | 1.27 | 53.88% | 55.9% | 2.02pp | — |
| range_bounce_short__regime=trend_down | PASS | 6650 | 1.259 | 53.67% | 56.2% | 2.53pp | — |
| premium_zone_short__regime=squeeze | PASS | 13705 | 1.258 | 54.26% | 55.69% | 1.43pp | — |
| range_bounce_short__regime=squeeze | PASS | 5669 | 1.257 | 55.58% | 57.4% | 1.82pp | — |
| resistance_bounce_short__cap_segment+hour_bucket=small_cap+afternoon | PASS | 233 | 1.25 | 54.94% | 61.18% | 6.24pp | — |
| range_bounce_short__regime+cap_segment=trend_down+small_cap | PASS | 3190 | 1.247 | 53.04% | 57.28% | 4.24pp | — |
| premium_zone_short__cap_segment=small_cap | PASS | 19369 | 1.247 | 52.43% | 55.73% | 3.3pp | — |
| range_bounce_short__hour_bucket=afternoon | PASS | 1626 | 1.235 | 53.44% | 57.89% | 4.45pp | — |
| order_block_short__cap_segment=small_cap | PASS | 1058 | 1.226 | 52.65% | 58.99% | 6.34pp | — |
| premium_zone_short__regime+hour_bucket=squeeze+opening | PASS | 4505 | 1.21 | 53.36% | 57.14% | 3.78pp | — |
| premium_zone_short__cap_segment+hour_bucket=small_cap+afternoon | PASS | 466 | 1.206 | 52.15% | 58.41% | 6.26pp | — |
| range_bounce_short__regime+hour_bucket=trend_down+afternoon | PASS | 512 | 1.198 | 52.34% | 59.27% | 6.93pp | — |
| order_block_short__hour_bucket=morning | PASS | 741 | 1.194 | 56.01% | 60.89% | 4.88pp | — |
| order_block_short__cap_segment+hour_bucket=small_cap+opening | PASS | 773 | 1.184 | 51.23% | 57.16% | 5.93pp | — |
| order_block_short__regime+cap_segment=squeeze+small_cap | PASS | 257 | 1.182 | 54.47% | 64.24% | 9.77pp | — |
| premium_zone_short__regime+cap_segment=trend_down+small_cap | PASS | 5354 | 1.179 | 51.01% | 56.87% | 5.86pp | — |
| resistance_bounce_short__hour_bucket=afternoon | PASS | 750 | 1.17 | 53.73% | 56.12% | 2.39pp | — |
| order_block_short__regime=trend_down | PASS | 809 | 1.164 | 51.92% | 58.78% | 6.86pp | — |
| premium_zone_short__regime=trend_down | PASS | 13846 | 1.157 | 51.08% | 55.09% | 4.01pp | — |
| premium_zone_short__cap_segment+hour_bucket=small_cap+opening | PASS | 7127 | 1.157 | 50.29% | 55.48% | 5.19pp | — |
| premium_zone_short__hour_bucket=opening | PASS | 17799 | 1.135 | 50.08% | 54.79% | 4.71pp | — |
| range_bounce_short__regime+hour_bucket=squeeze+afternoon | PASS | 329 | 1.101 | 53.5% | 61.17% | 7.67pp | — |
| premium_zone_short__regime+hour_bucket=trend_down+opening | PASS | 4065 | 1.1 | 48.73% | 55.85% | 7.12pp | — |
| order_block_short__regime+hour_bucket=trend_down+opening | PASS | 586 | 1.09 | 48.98% | 57.74% | 8.76pp | — |
| order_block_short__regime=squeeze | PASS | 523 | 1.086 | 52.77% | 62.3% | 9.53pp | — |
| order_block_short__regime=chop | PASS | 321 | 1.071 | 50.16% | 59.33% | 9.17pp | — |
| vwap_lose_short__regime+cap_segment=trend_down+large_cap | PASS | 70 | 1.068 | 57.14% | 63.64% | 6.5pp | — |
| premium_zone_short__cap_segment+hour_bucket=unknown+afternoon | PASS | 148 | 1.056 | 52.7% | 57.77% | 5.07pp | — |
| premium_zone_short__hour_bucket=afternoon | PASS | 1715 | 1.031 | 50.44% | 54.94% | 4.5pp | — |
| vwap_lose_short__regime=trend_down | PASS | 128 | 1.017 | 52.34% | 59.09% | 6.75pp | — |
| order_block_short__cap_segment+hour_bucket=mid_cap+morning | PASS | 245 | 1.0 | 53.06% | 61.14% | 8.08pp | — |
| vwap_lose_short__regime+cap_segment=trend_up+large_cap | FAIL | 56 | 2.522 | 71.43% | 58.6% | 12.83pp | wr_delta=12.8pp>10.0 |
| resistance_bounce_short__hour_bucket=opening | FAIL | 49 | 1.93 | 63.27% | 60.0% | 3.27pp | n_val=49<50 |
| order_block_short__regime+cap_segment=chop+unknown | FAIL | 49 | 1.85 | 63.27% | 58.16% | 5.11pp | n_val=49<50 |
| order_block_short__cap_segment+hour_bucket=unknown+morning | FAIL | 40 | 1.396 | 57.5% | 63.78% | 6.28pp | n_val=40<50 |
| vwap_lose_short__cap_segment=small_cap | FAIL | 63 | 1.154 | 50.79% | 65.91% | 15.12pp | wr_delta=15.1pp>10.0 |
| order_block_short__cap_segment=mid_cap | FAIL | 689 | 0.998 | 49.64% | 57.59% | 7.95pp | pf_val=0.998<1.0 |
| order_block_short__cap_segment+hour_bucket=mid_cap+opening | FAIL | 440 | 0.996 | 47.73% | 55.3% | 7.57pp | pf_val=0.996<1.0 |
| order_block_short__regime+hour_bucket=squeeze+opening | FAIL | 347 | 0.982 | 48.41% | 58.81% | 10.4pp | pf_val=0.982<1.0; wr_delta=10.4pp>10.0 |
| resistance_bounce_short__cap_segment=micro_cap | FAIL | 252 | 0.98 | 50.0% | 56.72% | 6.72pp | pf_val=0.980<1.0 |
| premium_zone_short__regime+hour_bucket=trend_down+afternoon | FAIL | 462 | 0.957 | 49.78% | 56.5% | 6.72pp | pf_val=0.957<1.0 |
| range_bounce_short__hour_bucket=late | FAIL | 156 | 0.951 | 47.44% | 56.0% | 8.56pp | pf_val=0.951<1.0 |
| order_block_short__regime+hour_bucket=chop+morning | FAIL | 159 | 0.897 | 45.91% | 61.87% | 15.96pp | pf_val=0.897<1.0; wr_delta=16.0pp>10.0 |
| premium_zone_short__hour_bucket=late | FAIL | 83 | 0.863 | 53.01% | 53.37% | 0.36pp | pf_val=0.863<1.0 |
| range_bounce_short__cap_segment+hour_bucket=small_cap+late | FAIL | 81 | 0.851 | 45.68% | 57.08% | 11.4pp | pf_val=0.851<1.0; wr_delta=11.4pp>10.0 |
| order_block_short__regime+cap_segment=squeeze+mid_cap | FAIL | 172 | 0.839 | 46.51% | 62.81% | 16.3pp | pf_val=0.839<1.0; wr_delta=16.3pp>10.0 |
| range_bounce_short__regime+hour_bucket=trend_up+late | FAIL | 52 | 0.582 | 30.77% | 55.56% | 24.79pp | pf_val=0.582<1.0; wr_delta=24.8pp>10.0 |
