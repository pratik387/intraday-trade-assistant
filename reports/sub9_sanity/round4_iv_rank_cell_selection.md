# Round-4 Cell Selection: options_vol_iv_rank_revert

**Date:** 2026-05-06
**Pass criteria:** n>=30 AND NET PF>=1.10 (matches circuit_t1 / round-3 cell selection)

Round-4 IV-rank-revert aggregate failed (PF 0.843, n=5224, LONG dragged everything down). This report checks whether any individual cell — particularly tighter iv_rank buckets like 0.95-1.00 (extremely high IV) or 0.00-0.05 (extremely low IV) — has PF>=1.10 + n>=30, in which case the candidate could ship at narrow scope.

## options_vol_iv_rank_revert

**Aggregate: n=5224, NET PF=0.843, WR=46.4%**


### Univariate cells


**side:**

| cell | n | NET PF | WR | WR Δ pp | pass |
|---|---|---|---|---|---|
| side=SHORT | 503 | 1.104 | 50.5% | +4.1 | ✓ |
| side=LONG | 4721 | 0.819 | 46.0% | -0.4 | ✗ |

**cap_segment:**

| cell | n | NET PF | WR | WR Δ pp | pass |
|---|---|---|---|---|---|
| cap_segment=large_cap | 5168 | 0.848 | 46.5% | +0.1 | ✗ |
| cap_segment=mid_cap | 56 | 0.508 | 39.3% | -7.1 | ✗ |

**regime:**

| cell | n | NET PF | WR | WR Δ pp | pass |
|---|---|---|---|---|---|
| regime=trend_down | 634 | 1.006 | 50.2% | +3.8 | ✗ |
| regime=trend_up | 4590 | 0.821 | 45.9% | -0.5 | ✗ |

**iv_rank_bucket:**

| cell | n | NET PF | WR | WR Δ pp | pass |
|---|---|---|---|---|---|
| iv_rank_bucket=0.85-0.90 | 59 | 1.241 | 50.8% | +4.4 | ✓ |
| iv_rank_bucket=0.95-1.00 | 262 | 1.192 | 53.1% | +6.7 | ✓ |
| iv_rank_bucket=0.90-0.95 | 127 | 0.977 | 44.9% | -1.5 | ✗ |
| iv_rank_bucket=0.80-0.85 | 55 | 0.872 | 50.9% | +4.5 | ✗ |
| iv_rank_bucket=0.10-0.15 | 1307 | 0.862 | 48.1% | +1.7 | ✗ |
| iv_rank_bucket=0.05-0.10 | 1569 | 0.843 | 46.3% | -0.1 | ✗ |
| iv_rank_bucket=0.00-0.05 | 907 | 0.794 | 44.2% | -2.2 | ✗ |
| iv_rank_bucket=0.15-0.20 | 938 | 0.743 | 44.1% | -2.3 | ✗ |

### Bivariate cells (top 20 by PF, n>=30)

| cell | n | NET PF | WR | WR Δ pp | pass |
|---|---|---|---|---|---|
| regime=trend_down × iv_rank_bucket=0.85-0.90 | 35 | 1.944 | 60.0% | +13.6 | ✓ |
| regime=trend_down × iv_rank_bucket=0.95-1.00 | 143 | 1.262 | 53.1% | +6.7 | ✓ |
| side=SHORT × iv_rank_bucket=0.85-0.90 | 59 | 1.241 | 50.8% | +4.4 | ✓ |
| cap_segment=large_cap × iv_rank_bucket=0.85-0.90 | 59 | 1.241 | 50.8% | +4.4 | ✓ |
| side=SHORT × regime=trend_down | 275 | 1.238 | 53.8% | +7.4 | ✓ |
| cap_segment=large_cap × iv_rank_bucket=0.95-1.00 | 258 | 1.219 | 53.5% | +7.1 | ✓ |
| regime=trend_down × iv_rank_bucket=0.05-0.10 | 118 | 1.218 | 54.2% | +7.8 | ✓ |
| regime=trend_down × iv_rank_bucket=0.80-0.85 | 31 | 1.210 | 61.3% | +14.9 | ✓ |
| side=SHORT × iv_rank_bucket=0.95-1.00 | 262 | 1.192 | 53.1% | +6.7 | ✓ |
| side=SHORT × cap_segment=large_cap | 498 | 1.119 | 50.8% | +4.4 | ✓ |
| regime=trend_up × iv_rank_bucket=0.95-1.00 | 119 | 1.105 | 52.9% | +6.5 | ✓ |
| regime=trend_down × cap_segment=large_cap | 625 | 1.017 | 50.2% | +3.8 | ✗ |
| regime=trend_up × iv_rank_bucket=0.90-0.95 | 61 | 0.996 | 41.0% | -5.4 | ✗ |
| cap_segment=large_cap × iv_rank_bucket=0.90-0.95 | 127 | 0.977 | 44.9% | -1.5 | ✗ |
| side=SHORT × iv_rank_bucket=0.90-0.95 | 127 | 0.977 | 44.9% | -1.5 | ✗ |
| regime=trend_down × iv_rank_bucket=0.90-0.95 | 66 | 0.964 | 48.5% | +2.1 | ✗ |
| side=SHORT × regime=trend_up | 228 | 0.937 | 46.5% | +0.1 | ✗ |
| cap_segment=large_cap × iv_rank_bucket=0.80-0.85 | 54 | 0.895 | 51.9% | +5.5 | ✗ |
| side=SHORT × iv_rank_bucket=0.80-0.85 | 55 | 0.872 | 50.9% | +4.5 | ✗ |
| regime=trend_down × iv_rank_bucket=0.10-0.15 | 109 | 0.863 | 47.7% | +1.3 | ✗ |

### Trivariate (side × iv_rank_bucket × regime, n>=30)

| cell | n | NET PF | WR | WR Δ pp | pass |
|---|---|---|---|---|---|
| side=SHORT × iv_rank_bucket=0.85-0.90 × regime=trend_down | 35 | 1.944 | 60.0% | +13.6 | ✓ |
| side=SHORT × iv_rank_bucket=0.95-1.00 × regime=trend_down | 143 | 1.262 | 53.1% | +6.7 | ✓ |
| side=LONG × iv_rank_bucket=0.05-0.10 × regime=trend_down | 118 | 1.218 | 54.2% | +7.8 | ✓ |
| side=SHORT × iv_rank_bucket=0.80-0.85 × regime=trend_down | 31 | 1.210 | 61.3% | +14.9 | ✓ |
| side=SHORT × iv_rank_bucket=0.95-1.00 × regime=trend_up | 119 | 1.105 | 52.9% | +6.5 | ✓ |
| side=SHORT × iv_rank_bucket=0.90-0.95 × regime=trend_up | 61 | 0.996 | 41.0% | -5.4 | ✗ |
| side=SHORT × iv_rank_bucket=0.90-0.95 × regime=trend_down | 66 | 0.964 | 48.5% | +2.1 | ✗ |
| side=LONG × iv_rank_bucket=0.10-0.15 × regime=trend_down | 109 | 0.863 | 47.7% | +1.3 | ✗ |
| side=LONG × iv_rank_bucket=0.10-0.15 × regime=trend_up | 1198 | 0.862 | 48.2% | +1.8 | ✗ |
| side=LONG × iv_rank_bucket=0.00-0.05 × regime=trend_up | 841 | 0.823 | 44.8% | -1.6 | ✗ |
| side=LONG × iv_rank_bucket=0.05-0.10 × regime=trend_up | 1451 | 0.815 | 45.6% | -0.8 | ✗ |
| side=LONG × iv_rank_bucket=0.15-0.20 × regime=trend_up | 872 | 0.749 | 44.0% | -2.4 | ✗ |
| side=LONG × iv_rank_bucket=0.15-0.20 × regime=trend_down | 66 | 0.659 | 45.5% | -0.9 | ✗ |
| side=LONG × iv_rank_bucket=0.00-0.05 × regime=trend_down | 66 | 0.542 | 36.4% | -10.0 | ✗ |