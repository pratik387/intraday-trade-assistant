# Round-3 Cell Selection on 2yr Sanity Trades

**Date:** 2026-05-06
**Pass criteria:** n>=30 AND NET PF>=1.10 (matches circuit_t1 cell selection)

All 3 candidates failed aggregate sanity (PF 0.47-0.63). This report
checks whether any individual cell (regime × cap × hour × side) has
PF>=1.10 + n>=30, in which case the candidate could ship at narrow scope.

## vwap_deviation_meanrevert

**Aggregate: n=736, NET PF=0.472, WR=39.9%**


### Univariate cells


**side:**

| cell | n | NET PF | WR | WR Δ pp | pass |
|---|---|---|---|---|---|
| side=SHORT | 330 | 0.515 | 40.9% | +1.0 | ✗ |
| side=LONG | 406 | 0.440 | 39.2% | -0.8 | ✗ |

**cap_segment:**

| cell | n | NET PF | WR | WR Δ pp | pass |
|---|---|---|---|---|---|
| cap_segment=mid_cap | 736 | 0.472 | 39.9% | +0.0 | ✗ |

**regime:**

| cell | n | NET PF | WR | WR Δ pp | pass |
|---|---|---|---|---|---|
| regime=trend_down | 324 | 0.474 | 41.4% | +1.4 | ✗ |
| regime=trend_up | 412 | 0.470 | 38.8% | -1.1 | ✗ |

**hour_bucket:**

| cell | n | NET PF | WR | WR Δ pp | pass |
|---|---|---|---|---|---|
| hour_bucket=late | 48 | 0.781 | 43.8% | +3.8 | ✗ |
| hour_bucket=morning | 414 | 0.509 | 41.1% | +1.1 | ✗ |
| hour_bucket=afternoon | 111 | 0.433 | 38.7% | -1.2 | ✗ |
| hour_bucket=lunch | 163 | 0.351 | 36.8% | -3.1 | ✗ |

### Bivariate cells (top 15 by PF, n>=30)

| cell | n | NET PF | WR | WR Δ pp | pass |
|---|---|---|---|---|---|
| cap_segment=mid_cap × hour_bucket=late | 48 | 0.781 | 43.8% | +3.8 | ✗ |
| side=SHORT × hour_bucket=morning | 175 | 0.587 | 42.9% | +2.9 | ✗ |
| side=SHORT × regime=trend_down | 116 | 0.564 | 44.8% | +4.9 | ✗ |
| regime=trend_up × hour_bucket=morning | 243 | 0.524 | 41.2% | +1.2 | ✗ |
| side=SHORT × cap_segment=mid_cap | 330 | 0.515 | 40.9% | +1.0 | ✗ |
| regime=trend_up × hour_bucket=afternoon | 55 | 0.515 | 41.8% | +1.9 | ✗ |
| cap_segment=mid_cap × hour_bucket=morning | 414 | 0.509 | 41.1% | +1.1 | ✗ |
| side=LONG × hour_bucket=afternoon | 62 | 0.497 | 41.9% | +2.0 | ✗ |
| side=SHORT × regime=trend_up | 214 | 0.493 | 38.8% | -1.2 | ✗ |
| regime=trend_down × hour_bucket=morning | 171 | 0.489 | 40.9% | +1.0 | ✗ |
| regime=trend_down × cap_segment=mid_cap | 324 | 0.474 | 41.4% | +1.4 | ✗ |
| side=SHORT × hour_bucket=lunch | 80 | 0.473 | 40.0% | +0.1 | ✗ |
| regime=trend_up × cap_segment=mid_cap | 412 | 0.470 | 38.8% | -1.1 | ✗ |
| side=LONG × hour_bucket=morning | 239 | 0.459 | 39.7% | -0.2 | ✗ |
| side=LONG × regime=trend_up | 198 | 0.443 | 38.9% | -1.1 | ✗ |

### Trivariate (regime × side × cap, n>=30)

| cell | n | NET PF | WR | WR Δ pp | pass |
|---|---|---|---|---|---|
| regime=trend_down × side=SHORT × cap_segment=mid_cap | 116 | 0.564 | 44.8% | +4.9 | ✗ |
| regime=trend_up × side=SHORT × cap_segment=mid_cap | 214 | 0.493 | 38.8% | -1.2 | ✗ |
| regime=trend_up × side=LONG × cap_segment=mid_cap | 198 | 0.443 | 38.9% | -1.1 | ✗ |
| regime=trend_down × side=LONG × cap_segment=mid_cap | 208 | 0.438 | 39.4% | -0.5 | ✗ |

## index_stock_divergence_revert

**Aggregate: n=4578, NET PF=0.604, WR=39.5%**


### Univariate cells


**side:**

| cell | n | NET PF | WR | WR Δ pp | pass |
|---|---|---|---|---|---|
| side=LONG | 1547 | 0.697 | 41.3% | +1.8 | ✗ |
| side=SHORT | 3031 | 0.562 | 38.6% | -0.9 | ✗ |

**cap_segment:**

| cell | n | NET PF | WR | WR Δ pp | pass |
|---|---|---|---|---|---|
| cap_segment=unknown | 71 | 0.748 | 46.5% | +7.0 | ✗ |
| cap_segment=mid_cap | 135 | 0.664 | 38.5% | -1.0 | ✗ |
| cap_segment=large_cap | 4372 | 0.601 | 39.4% | -0.1 | ✗ |

**regime:**

| cell | n | NET PF | WR | WR Δ pp | pass |
|---|---|---|---|---|---|
| regime=trend_down | 1864 | 0.705 | 42.4% | +2.9 | ✗ |
| regime=trend_up | 2714 | 0.539 | 37.5% | -2.0 | ✗ |

**hour_bucket:**

| cell | n | NET PF | WR | WR Δ pp | pass |
|---|---|---|---|---|---|
| hour_bucket=afternoon | 2420 | 0.642 | 40.5% | +1.0 | ✗ |
| hour_bucket=late | 2158 | 0.558 | 38.4% | -1.1 | ✗ |

### Bivariate cells (top 15 by PF, n>=30)

| cell | n | NET PF | WR | WR Δ pp | pass |
|---|---|---|---|---|---|
| regime=trend_down × cap_segment=mid_cap | 60 | 0.939 | 50.0% | +10.5 | ✗ |
| regime=trend_up × cap_segment=unknown | 43 | 0.879 | 48.8% | +9.3 | ✗ |
| cap_segment=mid_cap × hour_bucket=afternoon | 71 | 0.863 | 46.5% | +7.0 | ✗ |
| side=LONG × cap_segment=mid_cap | 51 | 0.776 | 35.3% | -4.2 | ✗ |
| side=LONG × hour_bucket=afternoon | 800 | 0.762 | 42.8% | +3.2 | ✗ |
| regime=trend_down × hour_bucket=afternoon | 1000 | 0.762 | 43.2% | +3.7 | ✗ |
| cap_segment=unknown × hour_bucket=afternoon | 41 | 0.758 | 48.8% | +9.3 | ✗ |
| cap_segment=unknown × hour_bucket=late | 30 | 0.729 | 43.3% | +3.8 | ✗ |
| side=SHORT × regime=trend_down | 1644 | 0.713 | 42.8% | +3.3 | ✗ |
| side=LONG × regime=trend_up | 1327 | 0.707 | 41.6% | +2.1 | ✗ |
| regime=trend_down × cap_segment=large_cap | 1776 | 0.700 | 42.2% | +2.7 | ✗ |
| side=LONG × cap_segment=large_cap | 1471 | 0.692 | 41.3% | +1.8 | ✗ |
| side=SHORT × cap_segment=unknown | 46 | 0.689 | 43.5% | +4.0 | ✗ |
| side=LONG × regime=trend_down | 220 | 0.641 | 39.5% | +0.0 | ✗ |
| cap_segment=large_cap × hour_bucket=afternoon | 2308 | 0.633 | 40.2% | +0.7 | ✗ |

### Trivariate (regime × side × cap, n>=30)

| cell | n | NET PF | WR | WR Δ pp | pass |
|---|---|---|---|---|---|
| regime=trend_down × side=SHORT × cap_segment=mid_cap | 52 | 1.015 | 50.0% | +10.5 | ✗ |
| regime=trend_up × side=LONG × cap_segment=mid_cap | 43 | 0.811 | 32.6% | -7.0 | ✗ |
| regime=trend_down × side=SHORT × cap_segment=large_cap | 1567 | 0.708 | 42.6% | +3.1 | ✗ |
| regime=trend_up × side=LONG × cap_segment=large_cap | 1262 | 0.701 | 41.8% | +2.2 | ✗ |
| regime=trend_down × side=LONG × cap_segment=large_cap | 209 | 0.639 | 38.8% | -0.8 | ✗ |
| regime=trend_up × side=SHORT × cap_segment=large_cap | 1334 | 0.409 | 33.6% | -5.9 | ✗ |
| regime=trend_up × side=SHORT × cap_segment=mid_cap | 32 | 0.125 | 25.0% | -14.5 | ✗ |

## volume_spike_exhaustion_reversal

**Aggregate: n=739, NET PF=0.631, WR=36.8%**


### Univariate cells


**side:**

| cell | n | NET PF | WR | WR Δ pp | pass |
|---|---|---|---|---|---|
| side=SHORT | 350 | 0.700 | 38.6% | +1.8 | ✗ |
| side=LONG | 389 | 0.581 | 35.2% | -1.6 | ✗ |

**cap_segment:**

| cell | n | NET PF | WR | WR Δ pp | pass |
|---|---|---|---|---|---|
| cap_segment=mid_cap | 739 | 0.631 | 36.8% | +0.0 | ✗ |

**regime:**

| cell | n | NET PF | WR | WR Δ pp | pass |
|---|---|---|---|---|---|
| regime=trend_up | 353 | 0.653 | 35.7% | -1.1 | ✗ |
| regime=trend_down | 386 | 0.611 | 37.8% | +1.0 | ✗ |

**hour_bucket:**

| cell | n | NET PF | WR | WR Δ pp | pass |
|---|---|---|---|---|---|
| hour_bucket=late | 190 | 0.677 | 37.4% | +0.6 | ✗ |
| hour_bucket=afternoon | 244 | 0.674 | 38.1% | +1.3 | ✗ |
| hour_bucket=morning | 107 | 0.663 | 35.5% | -1.3 | ✗ |
| hour_bucket=lunch | 198 | 0.528 | 35.4% | -1.5 | ✗ |

### Bivariate cells (top 15 by PF, n>=30)

| cell | n | NET PF | WR | WR Δ pp | pass |
|---|---|---|---|---|---|
| regime=trend_up × hour_bucket=morning | 60 | 1.309 | 48.3% | +11.5 | ✓ |
| regime=trend_down × hour_bucket=late | 106 | 0.884 | 39.6% | +2.8 | ✗ |
| side=SHORT × hour_bucket=late | 92 | 0.872 | 40.2% | +3.4 | ✗ |
| side=SHORT × regime=trend_down | 211 | 0.789 | 40.8% | +4.0 | ✗ |
| regime=trend_down × hour_bucket=afternoon | 125 | 0.766 | 41.6% | +4.8 | ✗ |
| side=SHORT × hour_bucket=afternoon | 112 | 0.755 | 41.1% | +4.3 | ✗ |
| side=LONG × hour_bucket=morning | 65 | 0.742 | 33.8% | -3.0 | ✗ |
| side=SHORT × cap_segment=mid_cap | 350 | 0.700 | 38.6% | +1.8 | ✗ |
| side=LONG × regime=trend_up | 214 | 0.700 | 36.0% | -0.8 | ✗ |
| cap_segment=mid_cap × hour_bucket=late | 190 | 0.677 | 37.4% | +0.6 | ✗ |
| cap_segment=mid_cap × hour_bucket=afternoon | 244 | 0.674 | 38.1% | +1.3 | ✗ |
| cap_segment=mid_cap × hour_bucket=morning | 107 | 0.663 | 35.5% | -1.3 | ✗ |
| regime=trend_up × cap_segment=mid_cap | 353 | 0.653 | 35.7% | -1.1 | ✗ |
| side=LONG × hour_bucket=afternoon | 132 | 0.620 | 35.6% | -1.2 | ✗ |
| regime=trend_down × cap_segment=mid_cap | 386 | 0.611 | 37.8% | +1.0 | ✗ |

### Trivariate (regime × side × cap, n>=30)

| cell | n | NET PF | WR | WR Δ pp | pass |
|---|---|---|---|---|---|
| regime=trend_down × side=SHORT × cap_segment=mid_cap | 211 | 0.789 | 40.8% | +4.0 | ✗ |
| regime=trend_up × side=LONG × cap_segment=mid_cap | 214 | 0.700 | 36.0% | -0.8 | ✗ |
| regime=trend_up × side=SHORT × cap_segment=mid_cap | 139 | 0.568 | 35.3% | -1.6 | ✗ |
| regime=trend_down × side=LONG × cap_segment=mid_cap | 175 | 0.448 | 34.3% | -2.5 | ✗ |