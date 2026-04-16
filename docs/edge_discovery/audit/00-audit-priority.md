# Audit Priority Ranking

## Methodology
1. Highest trade-count detectors first (more downstream impact if buggy)
2. ICT/SMC detectors next (recently refactored, more risk)
3. Detectors with multiple recent commits (layer-on-layer bug risk)
4. Everything else alphabetically

## Inputs used
- Trade counts: `/tmp/setup_trade_counts.txt` (snapshot from 20260413-102859_full)
- Recent commits: `/tmp/recent_commit_counts.txt` (3-month window as of 2026-04-15)

## Detector trade-count aggregation

| Detector | Aggregated trade count | Top setup |
|----------|------------------------|-----------|
| ICTStructure | ~928,383 | discount_zone_long (542,744) |
| RangeStructure | 190,453 | range_bounce_short (190,453) |
| SupportResistanceStructure | 75,926 | resistance_bounce_short (75,926) |
| MomentumStructure | 7,596 | momentum_breakout_long (7,585) |
| LevelBreakoutStructure | 3,294 | breakout_long (3,294) |
| VolumeStructure | 553 | volume_spike_reversal_long (457) |
| FailureFadeStructure | 257 | failure_fade_short (257) |
| VWAPStructure | 219 | vwap_lose_short (219) |
| ORBStructure | 68 | orb_breakdown_short (40) |
| SqueezeReleaseStructure | 46 | squeeze_release_short (46) |
| FlagContinuationStructure | 1 | flag_continuation_short (1) |
| FHMStructure | 0 | — |
| GapStructure | 0 | — |
| TrendStructure | 0 | — |
| VolumeBreakoutStructure | 0 | — |

## Ranking

| Rank | Detector | File | Top setup (trade count) | Recent commits | Status |
|------|----------|------|-------------------------|----------------|--------|
| 01 | ICTStructure | structures/ict_structure.py | discount_zone_long (542,744) | 2 | DONE-FIXED-AND-TRUSTED |
| 02 | RangeStructure | structures/range_structure.py | range_bounce_short (190,453) | 6 | DONE-FIXED-AND-TRUSTED |
| 03 | SupportResistanceStructure | structures/support_resistance_structure.py | resistance_bounce_short (75,926) | 4 | DONE-FIXED-AND-TRUSTED |
| 04 | MomentumStructure | structures/momentum_structure.py | momentum_breakout_long (7,585) | 0 | DONE-FIXED-AND-TRUSTED (split: momentum_breakout_* FIXED, trend_continuation_* DISABLED) |
| 05 | LevelBreakoutStructure | structures/level_breakout_structure.py | breakout_long (3,294) | 0 | DONE-FIXED-AND-TRUSTED |
| 06 | VolumeStructure | structures/volume_structure.py | volume_spike_reversal_long (457) | 1 | DONE-FIXED-AND-TRUSTED |
| 07 | VolumeBreakoutStructure | structures/volume_breakout_structure.py | — (0) | 3 | DONE-FIXED-AND-TRUSTED (re-enabled per fresh-OOS methodology; gauntlet decides) |
| 08 | TrendStructure | structures/trend_structure.py | — (0) | 2 | DONE-FIXED-AND-TRUSTED (split: trend_pullback_* FIXED, trend_continuation_* DISABLED via audit/04) |
| 09 | FailureFadeStructure | structures/failure_fade_structure.py | failure_fade_short (257) | 1 | DONE-FIXED-AND-TRUSTED + RE-ENABLED long (per fresh-OOS) |
| 10 | VWAPStructure | structures/vwap_structure.py | vwap_lose_short (219) | 1 | DONE-FIXED-AND-TRUSTED + RE-ENABLED 3 setups (per fresh-OOS) |
| 11 | ORBStructure | structures/orb_structure.py | orb_breakdown_short (40) | 1 | DONE-FIXED-AND-TRUSTED + RE-ENABLED orb_breakout_long (per fresh-OOS) |
| 12 | SqueezeReleaseStructure | structures/squeeze_release_structure.py | squeeze_release_short (46) | 1 | DONE-FIXED-AND-TRUSTED + RE-ENABLED long |
| 13 | FHMStructure | structures/fhm_structure.py | — (0) | 1 | DONE-FIXED-AND-TRUSTED + RE-ENABLED both (per fresh-OOS) |
| 14 | FlagContinuationStructure | structures/flag_continuation_structure.py | flag_continuation_short (1) | 0 | DONE-FIXED-AND-TRUSTED + RE-ENABLE long + cap-block parity added |
| 15 | GapStructure | structures/gap_structure.py | — (0) | 0 | DONE-FIXED-AND-TRUSTED + RE-ENABLE all 4 + cap-block parity (substantive canonical rewrite deferred to post-OOS — see audit/15) |

## Rationale notes
- **Rank 01 (ICTStructure):** Top by trade count (~928K trades aggregated across 12 setup types) AND highest recent commit activity (2 in 3 months, including the in-flight `feat/premium-zone-ict-fix` branch). Both rule #1 and rule #2 point here.
- **Ranks 02-05:** Trade-count order. RangeStructure's single `range_bounce_short` setup alone produces more trades than 11 of the 15 detectors combined.
- **Ranks 06-08:** Detectors with 1 recent commit each (rule #3) — promoted above zero-activity detectors even when trade counts are low or zero.
- **Ranks 09-15:** Ordered by residual trade count then alphabetically (rule #4). Zero-trade detectors at the bottom — they are either dormant, disabled, or emit nothing in the current backtest window, so audit urgency is lowest.

## Status legend
- NOT_STARTED — not yet audited
- IN_PROGRESS — audit doc started
- AWAITING_USER — assistant has filled in items 1-7, awaiting user disposition decision
- DONE-TRUSTED
- DONE-FIXED-AND-TRUSTED
- DONE-DISABLED
