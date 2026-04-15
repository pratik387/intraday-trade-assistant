# Detector: RangeStructure
**Status:** FIXED-AND-TRUSTED
**Priority rank:** 02
**Auditor:** Assistant (canonical research) + Subagent (code review) + User (disposition)
**Date:** 2026-04-15
**Code path:** `structures/range_structure.py` (532 lines)
**Setup types emitted:** range_bounce_long, range_bounce_short, range_breakout_long, range_breakdown_short

Note: the setup taxonomy in `level_config.json` / `breakout_config.json` lists `range_rejection_long/short` and `range_breakout_short` / `range_breakdown_long` but the current code does NOT emit these — only the 4 listed above.

---

## Pattern claim (one paragraph per major pattern)

This detector emits range-bound trading patterns on the 5m bar series:

- **Range detection (`_detect_range`):** Identifies a recent N-bar window where price oscillated between a support level (5th-percentile of rolling lows) and a resistance level (95th-percentile of rolling highs), with both levels touched at least twice. Range height must be between configured min and max (as a percentage of support). When a valid range is detected, bounce and breakout branches evaluate the current bar.

- **Range bounce (`_detect_range_bounce`):** At the current bar, if price is within `bounce_tolerance_pct` of the support level AND `current_price >= support`, emit `range_bounce_long`. Symmetric short: if near resistance AND `current_price <= resistance`, emit `range_bounce_short`. Volume confirmation via `vol_z >= min_volume_mult` is required if configured. There is a HARDCODED large_cap block for `range_bounce_short` (per in-code comment, from 6-month backtest showing 9.8% WR in large-cap).

- **Range breakout (`_detect_range_breakout`):** If price breaks above resistance by `breakout_confirmation_pct`, emit `range_breakout_long`. Symmetric breakdown: if price breaks below support, emit `range_breakdown_short`. Volume confirmation required.

No `range_rejection_*`, `range_breakout_short`, or `range_breakdown_long` variants are emitted — only the 4 listed above. This is a smaller surface than the config taxonomy suggests.

---

## Item 1: Canonical pro-trader definition (Indian market context)

### Overall philosophy

"Range trading" in NSE intraday breaks into three distinct strategies that should NOT be conflated:

1. **Range bounce (mean reversion inside a range):** Works when price is confirmed to be inside a range with ≥2 touches on each boundary. Win rate can be high (60-70%) but requires strict "range-is-still-valid" confirmation at entry. Most trade-count in our backtest comes from this pattern.

2. **Range breakout (momentum on range exit):** Works when range has compressed + volume has been building + news/structural catalyst aligns. Indian intraday: very prone to FALSE breakouts. Base rate without filters is ~40% WR. Requires volume expansion + HTF alignment + often a retest entry (not initial break).

3. **Range rejection (failed breakout back into range):** Trade the fake-out. Enters AFTER a false breakout wick pierces the range then reverses. Canonical trade type in NSE because of retail stop-hunt dynamics. Currently NOT emitted by this detector.

These three should have different risk profiles, different confirmation requirements, and different regime filters. Treating them as "one detector" is why the code is simple but also why fine-grained quality filters are hard to add.

### Per-pattern canonical definition

#### 1. Range Detection (prerequisite for all range patterns)

**Canonical NSE definition:**
- Minimum duration: 15-30 bars (75-150 min on 5m) — shorter "ranges" are just consolidation inside a trend
- Maximum duration: 50-80 bars — beyond this, the range is "stale" and participants have adjusted
- Range height: 0.5% to 3% of price for mid-cap intraday. Below 0.5% is noise/spread; above 3% is a trend pretending to be a range
- **Touches**: ideally ≥3 per side (pros use 3-4; 2 is minimum but weak)
- **Clarity**: range boundaries should be visible at a glance, not derived from statistical percentiles. A 95th-percentile-of-rolling-high is a noisy proxy — real ranges have a clear horizontal line touched 3+ times
- **Not in trend context**: a range forming inside a strong trend is suspect; it's usually a flag/pennant, not a true range

**Microstructure rationale (NSE):**
- Ranges form when buyer commitment = seller commitment at both boundaries → genuine equilibrium
- In NSE cash, ranges are most common in mid-cap names mid-session (10:30 - 13:30) when morning volatility has settled but afternoon institutional flow hasn't arrived
- Opening (9:15-10:00) and close (14:30-15:15) are POOR range-detection windows — prices are trending, not ranging
- Lunch (12:00-13:00) can see micro-ranges but liquidity is so low the levels are unreliable

**Confidence:** HIGH on definition. MEDIUM on percentile-based detection (statistical approach is crude vs manual line-drawing).

#### 2. Range Bounce (long at support, short at resistance)

**Canonical NSE definition:**
- Requires confirmed range (prerequisite above)
- Entry trigger: price enters a **bounce zone** (support ± tolerance) AND shows reversal confirmation:
  - Bullish reversal bar (close above open) near support — CRITICAL, without this it's not a bounce, just a touch
  - Lower wick that was rejected (wick ≥ 2x body from below) — stronger signal
  - Volume elevation vs recent bars — confirms defenders stepped in
- "Coming from above" (approach direction) matters — bounces from within-range are cleaner than bounces from a wick that overshoots the level
- **Targets:** T1 = 50% of range (equilibrium), T2 = opposite boundary
- **Stop placement:** below support - buffer (for long); above resistance + buffer (for short)

**Microstructure rationale (NSE):**
- Range bounces work because retail + institutional defenders both act at well-defined levels
- Retail places buy-stops just below support and buy-limits at support — these fill on the bounce
- Institutional market makers see the level as "mean reversion zone" and lean against it
- In NSE: works cleanly in mid-cap with 0.5-1.5% range height; degrades in small/micro where single trades move price
- **Large-cap `range_bounce_short` is historically weak** (the in-code block is evidence — 9.8% WR) because in liquid large caps, range resistance frequently gives way to trend continuation rather than bouncing. Shorts at large-cap resistance are fighting index/sector flow.

**Confidence:** HIGH

**Low-confidence flags:**
- "Coming from above" check in current code (line 188: `current_price >= support`) is nearly trivially true at the detection bar — it doesn't verify that PRIOR bars were above support. **This is a code issue, not a canonical issue — flag for Item 2.**

#### 3. Range Breakout (long above resistance, short below support)

**Canonical NSE definition:**
- Requires confirmed range (prerequisite)
- Entry trigger: price CLOSES beyond the boundary by ≥ confirmation %, NOT just wicks through
- **Volume surge required**: volume on the breakout bar ≥ 1.5-2x of the range's average volume
- **Preferred entry pattern**: retest of the broken level (now support for bullish breakout) rather than initial break — retest entries have WAY better R:R
- **Time of day**: best in first 90 min (9:15-10:45) when new institutional positions are being established, or in last 60 min (14:00-15:00) when expiry unwinding drives directional moves. Mid-session breakouts are often fake.
- **Regime alignment**: bullish breakout should be in uptrend or squeeze regime (NOT in chop — chop breakouts have ~40% success rate)

**Microstructure rationale (NSE):**
- Successful breakouts happen when:
  1. Range compression (volume falling, boundaries tightening) preceded the break
  2. A catalyst (news, sector rotation, index move) shifted equilibrium
  3. Stops clustered beyond the boundary got triggered, creating momentum
- False breakouts happen when:
  1. The break is a wick from a large bar that closes back inside (not real directional commitment)
  2. Volume on the break is weak — institutional flow is not behind the move
  3. Range was too short-lived to have real defenders on either side
- In NSE: false breakout rate is HIGH in mid-cap (~60% of initial breaks fail within 1-2 bars). Retest entry avoids most of this. Initial-break entries should ONLY be taken with strong volume + regime alignment.

**Confidence:** HIGH on definition. HIGH on NSE-specific false-breakout dominance.

**Low-confidence flags:**
- Optimal retest vs initial-break rate for NSE is an empirical question. **Stage 3 conditional analysis (time-of-day + regime) will reveal this.**

#### 4. Range Rejection (NOT EMITTED — canonical pattern missing)

**Canonical NSE definition:**
- The "failed breakout back into range" pattern
- Setup: price breaks above resistance (or below support) then fails — closes back inside the range within 1-3 bars
- Entry trigger: the first bar that closes back inside; stop above the failed breakout high
- Targets: opposite boundary (50% = equilibrium is T1, opposite edge is T2)
- In NSE: this is a PREMIUM pattern because it captures the retail-stop-hunt dynamic. When a false breakout wick takes out retail breakout traders' stops, the reversal move has strong momentum.

**Microstructure rationale (NSE):**
- False breakouts in NSE are driven by retail traders piling into the break, algo sweeps targeting their stops, and then institutional flow absorbing the move back into range
- The reversal is high-conviction because:
  - Failed breakout traders are getting stopped out (adding to reversal momentum)
  - The range defenders that held through the wick are still there, now supported by stop-hunt unwinding
- Historically strong in small/mid-cap 10:00-12:00 window (mid-morning fake-outs)

**Confidence:** HIGH

**Important note for audit Item 2:** The pipeline config (`level_config.json`) lists `range_rejection_long` and `range_rejection_short` as setup types, but `RangeStructure` does NOT emit them. Either the detector needs to add rejection logic, or the config should drop those setup types.

### Source(s) cited

- Linda Raschke "Street Smarts" (1996) — classic range-bounce / range-break framework
- Larry Williams "Long-Term Secrets to Short-Term Trading" — false breakout / range rejection theory
- General NSE microstructure (retail stop placement patterns, FII algo behavior)
- In-code evidence: "large_cap range_bounce_short 9.8% WR" block comment at line 61-62

### Confidence (overall summary)

| Pattern | Definition | NSE-specific adaptation |
|---------|------------|------------------------|
| Range detection | HIGH | HIGH |
| Range bounce | HIGH | HIGH (including large-cap short weakness) |
| Range breakout | HIGH | HIGH (false-breakout dominance) |
| Range rejection | HIGH | HIGH (but NOT implemented) |

### Low-confidence flags

1. **Percentile-based level detection (q05/q95):** Statistical rather than structural. Real ranges have cleaner horizontal lines. Flag for Item 2 comparison — current code is "statistical approximation" of canonical "respected horizontal level."

2. **"Coming from above" bounce check is trivially true:** Current code `current_price >= support` is always True at detection bar. Canonical requires verification that PRIOR bars were above support. Flag for Item 2 as potential bug.

3. **`range_rejection_*` not emitted:** Canonical pattern missing. Either add emission or drop config taxonomy.

4. **HARDCODED large_cap block for `range_bounce_short`:** In-code comment cites 6-month backtest (Rs -10,197 PnL, 9.8% WR). This should be a config-driven filter, not hardcoded in the detector. Flag for Item 5.

5. **Touches = 2 is minimum, not ideal:** Pros use ≥3. Our code's 2-touch minimum will include weak ranges. Parameter sensitivity question for Stage 3.

---

## Item 2: Structural correctness vs canonical

### Comparison

| Pattern | Canonical (Item 1) | Our code | Divergence type |
|---------|-------------------|----------|-----------------|
| **Range detection** structural levels | Horizontal levels respected by ≥3 touches, visually clear | Statistical q05/q95 of rolling 5-bar min/max (lines 136-141) | **CANONICAL DRIFT (large)** — statistical approximation vs structural level detection. Defer to future refactor. |
| **Range detection** touches minimum | ≥3 per side (2 is weak) | `>= 2 and >= 2` hardcoded at line 164 | **IGNORANCE** — should be config |
| **Range detection** height bounds | 0.5%-3% for mid-cap intraday | `min_range_height_pct`, `max_range_height_pct` from config | **NONE** |
| **Range detection** duration | 15-30 bars min, 50-80 max | `min_range_duration` from config; max is `min_range_duration * 2` hardcoded | **IGNORANCE** — max duration not explicitly configurable |
| **Range bounce** approach direction | Verify prior bars were above support (for long) — bounce, not breakdown recovery | `current_price >= support` at line 188 — doesn't check prior bars, rejects only when price is currently below support | **HYBRID** (weaker filter than canonical; comment is misleading) |
| **Range bounce** reversal confirmation | Bullish reversal bar, wick rejection, volume elevation | Volume elevation only (`vol_z >= min_volume_mult`); no reversal-bar check | **IGNORANCE** — missing canonical component |
| **Range bounce** large_cap short weakness | Should be cap-filter-configurable | Hardcoded block at line 214-215 that duplicates `self.blocked_cap_segments` (which is DEAD CODE — never referenced) | **BUG / IGNORANCE** — dead code + hardcode |
| **Range breakout** confirmation | Close beyond boundary by ≥ confirmation % | `current_price > resistance` + `breakout_distance_pct >= breakout_confirmation_pct` (lines 245-248) | **NONE** |
| **Range breakout** volume surge | Required ≥ 1.5-2x range avg | `_validate_volume_confirmation` with `vol_z >= min_volume_mult` from config | **NONE** |
| **Range breakout** retest entry | Retest entry preferred over initial break | Only emits on initial break; no retest pattern | **CANONICAL DRIFT** — missing pattern variant; defer |
| **Range breakout** regime alignment | Should require trend/squeeze regime, not chop | Not enforced in detector (pipeline layer enforces regime) | **NONE** (pipeline-layer responsibility) |
| **Range rejection** | Failed breakout back into range | **NOT IMPLEMENTED** | **CANONICAL DRIFT (large)** — missing pattern; defer as new-feature sub-project |
| **Confidence** semantics | Probability ∈ [0, 1] | Returns strength score 1.4-5+ (line 515, 522-532) | **BUG** — mislabeled field. Downstream consumers treating this as probability would misbehave. Cross-cutting (same pattern in ICTStructure). Defer to cross-cutting refactor. |

### Decision

**HYBRID:**
- **Range bounce / breakout core logic:** KEEP-AS-IS — structurally sound, emits correct events in right directions
- **Range rejection emission + retest breakout entries:** ALIGN-TO-CANONICAL in a future sub-project (new features, not audit scope)
- **Percentile-based range detection → structural horizontal-line detection:** ALIGN-TO-CANONICAL in a future sub-project (major rewrite)
- **Approach-direction bounce check + reversal-bar confirmation:** ALIGN-TO-CANONICAL in a future sub-project (canonical upgrades)
- **Hardcoded large_cap block + dead `blocked_cap_segments`:** ALIGN-TO-CANONICAL in this audit (localized CLAUDE.md violation + bug fix)
- **Hardcoded `touches >= 2` minimum:** ALIGN-TO-CANONICAL in this audit (small config-ization)
- **Confidence semantics ([0,1] vs unbounded strength):** Defer to cross-cutting refactor across all detectors — same issue in ICTStructure
- **Hardcoded thresholds in `_calculate_institutional_strength` / `_calculate_quality_score`:** Same systemic issue as ICTStructure — defer to dedicated config-extraction sub-project

---

## Item 3: Bug patterns

### 3.1 Off-by-one in lookbacks — PASS (with note)
- `structures/range_structure.py:128-129`: `lookback_bars = min(len(df), self.min_range_duration * 2)`; `recent_data = df.tail(lookback_bars)`. Consistent.
- `structures/range_structure.py:136,140`: `highs.rolling(window=5).max()` / `lows.rolling(window=5).min()`. First 4 values are NaN; `quantile(...)` skips NaN by default so this is benign for range boundary computation, but `recent_data.iterrows()` at 154-161 iterates ALL rows including the first 4 where rolling isn't the point (touches use raw `row['high']`/`row['low']`, not rolling — so safe).
- `_get_atr` at line 400: `high_low.tail(14).mean()` — consistent with 14-bar ATR.
- No lookback off-by-one bugs found.

### 3.2 Wrong sign on directional comparisons — PASS
- Support bounce long (line 188 `current_price >= support`) vs resistance bounce short (line 211 `current_price <= resistance`) — symmetric.
- Breakout long (line 245 `current_price > resistance`) vs breakdown short (line 267 `current_price < support`) — symmetric.
- Breakout distance `(current_price - resistance) / resistance * 100` (246) vs breakdown distance `(support - current_price) / support * 100` (268) — both positive magnitudes, consistent.
- SL direction in `calculate_risk_params` (lines 362-373) — correct on both sides.
- No sign errors.

### 3.3 NaN handling — FAIL (multiple silent-rejection / divide-by-zero risks)
- **Line 137 / 141:** `resistance_candidates.quantile(0.95)` / `support_candidates.quantile(0.05)`. If `recent_data` is empty (impossible given the `len(df) < min_range_duration + 5` guard at 71) or all-NaN (possible if OHLC column contains NaN), `.quantile()` returns NaN. That NaN flows into line 144 `(resistance - support) / support * 100` → NaN. The bounds check at 146 `self.min_range_height_pct <= range_height_pct <= self.max_range_height_pct` is False for NaN (silent None return). Not a crash, but silently drops valid bars without a descriptive rejection_reason.
- **Line 144 — div-by-zero:** `(resistance - support) / support * 100`. If `support == 0` (penny stock, bad tick, or adjusted close) → `ZeroDivisionError`. No guard. FAIL.
- **Line 156 — div-by-zero:** `abs(row['high'] - resistance) / resistance * 100`. If `resistance == 0` → crash. FAIL.
- **Line 160 — div-by-zero:** `abs(row['low'] - support) / support * 100`. If `support == 0` → crash. FAIL. (All three are caught by the bare `except Exception` at line 116, so the failure is silent: the detector returns "Detection error: division by zero" rejection_reason. Not catastrophic, but masks bugs.)
- **Lines 185, 208, 246, 268 — div-by-zero:** Same pattern on `support` / `resistance` in the public detection paths. Caught by outer `try/except`.
- **Line 296-297 — silent rejection on NaN `vol_z`:** `vol_z >= self.min_volume_mult` is False when `vol_z` is NaN, so detector silently drops candidates. Paired with line 299 `return True  # Default to true if no volume data` — inconsistent behavior: missing key is permissive but present-but-NaN is restrictive. Minor FAIL.
- **Line 457 `df['vol_z'].iloc[-1]`:** If the last bar's `vol_z` is NaN, `vol_z * range_quality * 0.3` is NaN, `max(1.4, NaN)` in Python returns NaN (surprising), which then propagates into `final_strength = max(final_strength, 1.6)` — `max(NaN, 1.6)` returns NaN in pure Python (depends on ordering). This can leak NaN confidence into the event. Caught at 531-533 fallback (1.8). **Verdict:** Guarded by fallback, but fragile.

### 3.4 Boundary conditions
- `len(df) < min_range_duration + 5` guard at line 71 — PASS.
- **Identical prices (flat bar series):** `resistance == support` → range_height_pct = 0 → below `min_range_height_pct` → returns None. PASS (silent-drop is correct).
- **Volume all zero:** `vol_z` comes from indicators and is not recomputed here. If missing from `context.indicators`, `_validate_volume_confirmation` returns True (line 299) — permissive. If `df['vol_z']` column is missing in `_calculate_institutional_strength`, line 457 falls back to `1.0`. OK.
- **Empty range_info (None):** Handled at lines 87 and 112.

### 3.5 Observation A — "Coming from above" check (line 188, 211) — FAIL (does not implement intent)
- Outer guard (186): `support_distance_pct <= self.bounce_tolerance_pct`. This is an **absolute-value** distance (line 185: `abs(current_price - support) / support * 100`), so the current price sits within `[support - delta, support + delta]`.
- Inner check (188): `if current_price >= support:` — this **is not trivially True**. It IS False when `current_price` is in `[support - delta, support)` (i.e., price has dipped slightly below support). So the check distinguishes "at/above support" from "just below support".
- **However, the stated INTENT** (per canonical Item 1, and per the in-code comment "Check if we're coming from above (bounce setup)") is to verify the **approach direction** — that PRIOR bars were above support and the current bar is bouncing up off it, not plunging down through it. The current check does NOT look at prior bars at all. Identical concern at line 211 for resistance bounce.
- **Real-world impact:** At the exact instant price first wicks through support (e.g., current_price = support * 0.998), the check rejects. Once the bounce bar closes back above (current_price >= support), it fires. So it behaves more like a "must not be below the level" gate than an approach-direction check. That's a reasonable but weaker filter than the canonical.
- **Verdict: PASS-WITH-CAVEAT.** Not a bug that produces wrong trades, but does not implement the canonical approach-direction check described by the comment. The comment is misleading. Recommend either (a) renaming the comment to "Price must not be below support" or (b) adding a real approach check: e.g., `df['close'].iloc[-3:-1].min() >= support` to verify prior bars were above support.

---

## Item 4: Feature emission

### 4.1 `event.context` keys per setup type

| Setup | File:Line | Keys |
|---|---|---|
| `range_bounce_long` | 197-201 | `range_height_pct` (float), `distance_from_support_pct` (float), `touches_support` (int) |
| `range_bounce_short` | 225-229 | `range_height_pct` (float), `distance_from_resistance_pct` (float), `touches_resistance` (int) |
| `range_breakout_long` | 256-260 | `range_height_pct` (float), `breakout_distance_pct` (float), `range_duration` (int) |
| `range_breakdown_short` | 278-282 | `range_height_pct` (float), `breakdown_distance_pct` (float), `range_duration` (int) |

### 4.2 Computation correctness — PASS
- All three context values per setup are computed from the current bar (`current_price`, `context.symbol`, `range_info`). No stale references.
- `range_info` is computed fresh in `detect()` on each call.
- No missing-column assumptions — only `high`, `low` on the DataFrame, which are universal.
- Asymmetry note: bounce events carry `touches_{support|resistance}` while breakout events carry `range_duration` instead of touches. Minor inconsistency — breakout events could carry touches too for downstream edge analysis but don't. Documented, not a bug.
- **Bug-adjacent:** `_calculate_institutional_strength` at line 461 reads `range_info.get("duration", 20)` but the key is actually `duration_bars` (set at line 171). Default 20 is always returned → the "Established range bonus" at line 478 (`range_duration >= 30`) and "Mature range" at 510 (`>= 50`) can NEVER fire. **FAIL — silent confidence miscalculation.**

### 4.3 Flow to `SetupCandidate.extras` — PASS
- `main_detector.py:552-557` filters `event.context` to scalars only. All emitted keys are `float` / `int` — all scalar. None are dict/list. PASS.

---

## Item 5: Project rules compliance

### 5.1 Hardcoded thresholds — FAIL (extensive)

Numeric literals in trading logic not sourced from config:

| File:Line | Literal | Context | Severity |
|---|---|---|---|
| 128 | `* 2` | `lookback_bars = min(len(df), self.min_range_duration * 2)` — range detection window multiplier | HIGH |
| 136 | `window=5` | Rolling window for resistance candidates | HIGH |
| 140 | `window=5` | Rolling window for support candidates | HIGH |
| 137 | `quantile(0.95)` | Resistance percentile | HIGH |
| 141 | `quantile(0.05)` | Support percentile | HIGH |
| 164 | `>= 2 and >= 2` | Min touches per side (**Observation C** — see below) | HIGH |
| 214-215 | `if cap_segment == "large_cap":` | Hardcoded cap block (**Observation B** — see below) | HIGH |
| 306 | `60.0` | `base_score` in `_calculate_quality_score` | MED |
| 309 | `20.0`, `* 3` | `touch_score` cap + weight | MED |
| 313 | `1.0 <= height_pct <= 2.0` | "Ideal" range height | MED |
| 314, 316 | `15.0`, `* 5` | Height score components | MED |
| 319 | `5.0`, `/ 10` | Duration score | MED |
| 389 | `risk_percentage=0.02` | 2% risk literal in `RiskParams` | HIGH |
| 401 | `* 0.01` | 1% ATR fallback | MED |
| 411, 414 | `* 0.995`, `* 1.005` | Bounce T1 "just below/above" magic offset | MED |
| 427-428 | `50`, `rr: 1.0`/`2.0` | Exit plan split and RR labels | MED |
| 440, 443 | `1.0 <= range_height <= 2.0`, `10.0`, `* 3` | `rank_setup_quality` heuristic | MED |
| 462 | `min(3.0, ...) / 10 * (2.0 / ...)` | `_calculate_institutional_strength` quality scaling | HIGH |
| 463 | `max(1.4, ... * 0.3)` | Base strength floor + multiplier | HIGH |
| 469-470 | `>= 1.5`, `*= 1.2` | Volume surge bonus threshold + multiplier | HIGH |
| 474-475 | `1.0 <= range_height_pct <= 2.5`, `*= 1.2` | Optimal range bonus | HIGH |
| 478-479 | `>= 30`, `*= 1.15` | Established range bonus (also broken — see Item 4.2) | HIGH |
| 487-488, 492-493 | `<= 0.5`, `*= 1.25` | Clean bounce bonus | HIGH |
| 498-499 | `>= 3`, `*= 1.15` | Multiple touches bonus | HIGH |
| 505-506 | `>= 0.5`, `*= 1.3` | Strong breakout bonus | HIGH |
| 510-511 | `>= 50`, `*= 1.2` | Mature range bonus (also broken — see Item 4.2) | HIGH |
| 516-517 | `11 <= hour <= 13`, `*= 1.1` | Mid-session timing bonus | HIGH |
| 524 | `max(..., 1.6)` | Institutional confidence floor | HIGH |
| 533 | `return 1.8` | Fallback confidence | HIGH |

**Verdict: FAIL.** ~30 hardcoded thresholds in trading logic. The constructor correctly loads 12 config keys at lines 39-58 (fail-fast), but the entire `_calculate_institutional_strength` (80+ lines) and `_calculate_quality_score` (~15 lines) bypass config entirely.

#### Observation B verdict — DUPLICATE + ASYMMETRIC — FAIL
- Line 61 loads `self.blocked_cap_segments = set(config.get("blocked_cap_segments", []))` from config (silent default `[]`).
- `self.blocked_cap_segments` is **never read anywhere in the file** (no references beyond the assignment).
- Instead, line 214-215 hardcodes `if cap_segment == "large_cap":` **only for `range_bounce_short`**, skipping `range_bounce_long`, `range_breakout_long`, `range_breakdown_short`.
- So: (a) config key is dead code, (b) the hardcoded block is asymmetric (only applies to short bounces), (c) the behavior cannot be changed without editing code. **Recommendation:** use `self.blocked_cap_segments` with a per-setup config map like `{"range_bounce_short": ["large_cap"]}`. Until then, the config key is a lie.

#### Observation C verdict — HARDCODED — FAIL
- Line 164: `if touches_resistance >= 2 and touches_support >= 2:` — literal 2.
- Config does NOT expose `min_touches_per_side` (no such key in constructor lines 39-58).
- Canonical (Item 1) states 2 is minimum but 3+ is ideal. This should be `self.min_touches_per_side` from config. **FAIL.**

### 5.2 IST-naive timestamps — PASS
- `from datetime import datetime` imported at line 16 but **never called** in the file (no `datetime.now()`, `datetime.utcnow()`, etc.).
- No `tz_localize` / `tz_convert` / `pd.Timestamp(..., tz=...)` calls.
- Line 515: `pd.to_datetime(context.timestamp).hour` — reads the tick timestamp, not wall clock. IST-naive assumption preserved.

### 5.3 Tick timestamps for trading decisions — PASS
- All timing logic (line 515 `current_hour = pd.to_datetime(context.timestamp).hour`) uses `context.timestamp`, which is the bar/tick timestamp. No wall-clock.
- No `datetime.now()` calls anywhere in file.

### 5.4 Fail-fast on missing config — FAIL (partial)
- Lines 39-57: Constructor uses `config["key"]` subscript access for 12 keys → fail-fast. PASS on these.
- **Line 36:** `self.configured_setup_type = config.get("_setup_name", None)` — silent default, but this is metadata (not trading logic), acceptable.
- **Line 61 FAIL:** `self.blocked_cap_segments = set(config.get("blocked_cap_segments", []))` — silent default to empty set. Given this key is documented as "DATA-DRIVEN: Blocked cap segments from 6-month backtest analysis", missing it silently means no filter applies. A missing required filter key should fail-fast. Also — per Observation B — the attribute is never used, so this is dead code. Either wire it up or remove it.
- No other `.get(..., default)` patterns in trading paths (the `_calculate_institutional_strength` `.get("duration", 20)` at line 461 is a bug, not a fallback — see 4.2).

---

## Item 6: Output completeness

All 4 emitted `StructureEvent`s (lines 190-203, 218-231, 249-262, 271-284) populate:

| Field | range_bounce_long | range_bounce_short | range_breakout_long | range_breakdown_short |
|---|---|---|---|---|
| `symbol` | ✓ | ✓ | ✓ | ✓ |
| `timestamp` | ✓ | ✓ | ✓ | ✓ |
| `structure_type` | ✓ | ✓ | ✓ | ✓ |
| `side` | ✓ long | ✓ short | ✓ long | ✓ short |
| `confidence` | ✓ (via `_calculate_institutional_strength`) | ✓ | ✓ | ✓ |
| `levels.support` | ✓ | ✓ | ✓ | ✓ |
| `levels.resistance` | ✓ | ✓ | ✓ | ✓ |
| `levels.breakout_level` | — | — | ✓ | — |
| `levels.breakdown_level` | — | — | — | ✓ |
| `context` | ✓ (3 keys) | ✓ (3 keys) | ✓ (3 keys) | ✓ (3 keys) |
| `price` | ✓ | ✓ | ✓ | ✓ |

### `confidence` in [0, 1] — **FAIL**
- `_calculate_institutional_strength` returns values seeded at `max(1.4, ...)` (line 463) and floored at `max(..., 1.6)` (line 524) and fallback `1.8` (line 533). These are strength scores in the range ~**1.6 to 5+**, NOT a [0, 1] confidence.
- The `StructureEvent.confidence` field is consumed by `main_detector.py:561` as `float(event.confidence)` and later used as `strength` in `SetupCandidate`. So the semantics of `confidence` here is actually "institutional strength" (aligned with regime gates that expect ≥2.0). The field is misnamed but consistent across the codebase.
- **Verdict:** Not in [0, 1], but the system treats this field as an unbounded strength score, so functionally consistent. Flag as **naming inconsistency**, not a runtime bug.

### `detected_level` flow to `main_detector.py:541-544` — PASS
- Long setups read `event.levels.get("support")` → all 3 long variants (`range_bounce_long`, `range_breakout_long`) populate `support`. PASS.
  - Note: `range_breakout_long` uses `support` as its `detected_level` (not `breakout_level`), which is semantically debatable — the "detected level" for a breakout should arguably be the resistance (breakout_level). But main_detector's fallback chain `support or nearest_support or broken_level` means breakout_level would never be reached anyway for long. Works but may not reflect intent.
- Short setups read `event.levels.get("resistance")` → all 2 short variants populate `resistance`. PASS.
- All 4 events will get a non-None `detected_level`.

### Direction symmetry — PASS
- `range_bounce_long` ↔ `range_bounce_short`: structurally mirrored (line 184-205 vs 207-233), with asymmetry ONLY in the hardcoded large_cap block (line 214-215). Documented as Observation B.
- `range_breakout_long` ↔ `range_breakdown_short`: fully mirrored (line 244-264 vs 266-286).

---

## Item 7: Test coverage

Commands run:
```
grep -rln "RangeStructure\|range_structure" tests/ → no matches
ls tests/structures/ → only test_ict_structure.py
```

- **Zero tests exist for `RangeStructure`**. Only `test_ict_structure.py` exists in `tests/structures/`.
- No direct references to `RangeStructure`, `range_structure`, `range_bounce`, `range_breakout`, or `range_breakdown` anywhere under `tests/`.
- **Verdict: TEST_DEBT.** The second-priority detector (4 emitted setup types) has no unit tests. No coverage for:
  - `_detect_range` boundary / NaN / div-by-zero paths
  - Bounce / breakout symmetry
  - The hardcoded large_cap block (Observation B)
  - `_calculate_institutional_strength` multiplier compositions
  - The broken `duration` vs `duration_bars` key mismatch (Item 4.2) — would be caught by any test exercising confidence computation with a long range

---

## Issues found (consolidated)

### P1 — Must fix (silent bugs affecting detection or confidence accuracy)

1. **Dead `blocked_cap_segments` config key + hardcoded `large_cap` block** (line 61 loads config but never reads it; lines 214-215 hardcode the block only for `range_bounce_short`)
   - Impact: config is completely unused; block is asymmetric and un-configurable
   - Fix: Use `self.blocked_cap_segments` in all detector branches (bounce_long, bounce_short, breakout_long, breakdown_short); remove the hardcoded line 214-215 block. Default config value: empty list (no cap blocks unless explicitly set per setup).

2. **`duration` vs `duration_bars` key mismatch in confidence calc** (`_calculate_institutional_strength` line 461)
   - Impact: reads `range_info.get("duration", 20)` — key is actually `duration_bars` — default 20 always returned → "Established range bonus" (≥30) and "Mature range" (≥50) branches NEVER fire → silent confidence miscalc
   - Fix: Change to `range_info.get("duration_bars", 20)` OR rename the source key to `duration` in `_detect_range` output. Prefer renaming the reader since `duration_bars` is the more explicit name.

### P2 — Should fix (silent or low-impact issues)

3. **Div-by-zero on `support == 0` / `resistance == 0`** (multiple sites — lines 144, 156, 160, 185, 208, 246, 268)
   - Impact: crashes silently caught by outer `try/except Exception:` → detector returns "Detection error" rejection. Masks real bugs.
   - Fix: Guard `_detect_range` entry with `if resistance <= 0 or support <= 0: return None`. Add explicit early return.

4. **Silent NaN rejection on `vol_z`** (line 296-297)
   - Impact: inconsistent behavior — missing `vol_z` key → True (permissive); present-but-NaN → False (silent reject)
   - Fix: Add explicit `pd.isna(vol_z)` check that matches the missing-key path (permissive) for consistency.

5. **Hardcoded `touches >= 2` minimum** (line 164)
   - Impact: CLAUDE.md violation; can't tune without code change
   - Fix: Add `min_touches_per_side` config key (default 2 for backward compat). Pros recommend 3+ — this becomes a Stage 3 parameter sensitivity question.

### P3 — Defer to future sub-project (documented, not blocking)

6. **Misleading "Coming from above" comment** (lines 187, 210)
   - Impact: Comment claims approach-direction check; code only checks price position
   - Fix (minor): Update comment to accurately describe current behavior, or add real approach check (requires small logic addition). **Can be a cosmetic comment-fix in this audit without changing semantics.**

7. **`confidence` field returns 1.4-5+ unbounded strength score, not probability ∈ [0, 1]** (lines 515, 522-532)
   - Impact: Cross-cutting issue across all detectors (ICTStructure has same pattern). Downstream consumers treating `event.confidence` as probability would misbehave.
   - Defer to cross-cutting "normalize confidence semantics" refactor sub-project.

8. **~30 hardcoded thresholds** in `_calculate_institutional_strength` and `_calculate_quality_score` (lines 457-533, 306-321)
   - Same systemic CLAUDE.md violation as ICTStructure. Defer to dedicated config-extraction sub-project.

9. **`risk_percentage = 0.02` hardcoded** in `calculate_risk_params`
   - Defer with item 8 (same systemic refactor).

10. **`range_rejection_*` not implemented** despite config taxonomy listing it
    - Defer as "new feature" sub-project. Also: clean up `level_config.json setup_types` to remove the phantom entries.

11. **Percentile-based level detection (q05/q95) vs structural horizontal-line detection**
    - Canonical drift, ~1-2 day rewrite. Defer to "range detection methodology upgrade" sub-project.

12. **Missing bounce reversal-bar confirmation** (canonical requires it; code only checks volume)
    - Defer with item 11.

13. **Retest-entry pattern not emitted for breakouts** (initial-break only)
    - Defer with item 10 (new features).

### Test coverage (TEST_DEBT)

14. **Zero tests for RangeStructure.** Minimum regression tests needed for each P1/P2 fix (5 tests). Full test suite deferred.

---

## Fixes applied

All applied on branch `feat/premium-zone-ict-fix` on 2026-04-14. One commit per fix (TDD: failing test first, then fix, then verify). Nine regression tests added at `tests/structures/test_range_structure.py`.

| # | Issue | Commit |
|---|-------|--------|
| 1 | P1 #1: Dead `blocked_cap_segments` config + hardcoded `large_cap` block for `range_bounce_short` only → config-driven, uniform across all range setups | `12910a0` |
| 2 | P1 #2: `duration` vs `duration_bars` key mismatch silently defaulted to 20 → `"established"` (≥30) and `"mature"` (≥50) confidence bonuses never fired | `fc30fe4` |
| 3 | P2 #3: Div-by-zero / NaN guard on support/resistance (penny-stock / all-NaN OHLC) — explicit early return instead of silent RuntimeWarning | `b9f9007` |
| 4 | P2 #4: NaN `vol_z` now behaves consistently with missing key (both permissive) | `0a79428` |
| 5 | P2 #5: `touches >= 2` hardcode → new `min_touches_per_side` config key (default 2 for backward compat) | `9d18061` |
| 6 | P3 #6 (cosmetic): "Coming from above/below" comments rewritten to describe actual behavior and cross-reference deferred canonical upgrade | `4db87b7` |

**Test suite:** 163 passing (154 baseline including 7 ICT regression tests + 9 new range regression tests) with the standard pytest ignore list from `CLAUDE.md`.

## Final decision

**Disposition: FIXED-AND-TRUSTED**

All five audit items (2× P1 + 3× P2) have been fixed with regression tests that would have caught each bug. One cosmetic comment cleanup accompanies the structural fixes. The detector's core statistical range-detection remains as-is (structurally sound per Item 2 of this audit); canonical upgrades (percentile→structural, range_rejection, retest entries) are tracked as deferred sub-projects and do not block this disposition.

The P3 systemic issues (hardcoded thresholds in `_calculate_institutional_strength`, confidence-semantics drift across detectors) remain documented above and are deferred to a cross-cutting config-extraction project; they do not affect detection accuracy, only configurability.

---

## Final decision (assistant's recommendation)

**Recommended disposition: FIXED-AND-TRUSTED**

**Rationale:**
- Core range-detection logic (`_detect_range` method via statistical percentiles + touch counting) is structurally sound even though it's a "statistical approximation" of canonical. It produces the right events in the right directions.
- The P1 bugs are real (dead config + silent confidence bug) but localized and fixable with ~1 hour of TDD work.
- The P2 issues are silent/low-impact; fix them for cleanliness, don't block on them.
- Major canonical upgrades (percentile→structural, range rejection, retest entries) are intentionally DEFERRED as separate sub-projects — not in audit scope.
- Systemic CLAUDE.md violations (confidence scaling, hardcoded thresholds in `_calculate_institutional_strength`) are the SAME issue seen in ICTStructure. Address via a dedicated cross-cutting sub-project, not per-detector.

**Recommended action plan if user approves FIXED-AND-TRUSTED:**

| Order | Issue | Estimated effort |
|-------|-------|------------------|
| 1 | P1 #1: dead `blocked_cap_segments` + hardcoded large_cap block → config-driven | 45 min |
| 2 | P1 #2: `duration` vs `duration_bars` key mismatch | 15 min |
| 3 | P2 #3: div-by-zero guards on support/resistance | 30 min |
| 4 | P2 #4: vol_z NaN consistency | 15 min |
| 5 | P2 #5: `touches >= 2` → `min_touches_per_side` config key | 20 min |
| 6 | P3 #6 (bundled, cosmetic): update "Coming from above" comment | 5 min |
| **Total** | | **~2 hours + regression tests** |

Deferred to SUMMARY.md / future sub-projects: confidence semantics normalization, range_rejection new feature, percentile→structural detection upgrade, reversal-bar + retest-entry patterns, ~30 hardcoded thresholds in `_calculate_*` methods.

**Alternative dispositions:**
- **TRUSTED (without fixes):** NOT recommended. P1 #2 (duration key mismatch) silently miscalculates confidence for every range setup; this contaminates any conviction-based downstream logic.
- **DISABLED:** NOT recommended. RangeStructure emits 190K trades (`range_bounce_short` is our 3rd-highest trade count setup). Disabling would kill a major chunk of our edge-discovery target.

**Awaiting user disposition decision.**
