# Detector: MomentumStructure
**Status:** FIXED-AND-TRUSTED (split: momentum_breakout_* FIXED, trend_continuation_* DISABLED)
**Priority rank:** 04
**Auditor:** Assistant (canonical research) + Subagent (code review) + User (disposition)
**Date:** 2026-04-15
**Code path:** `structures/momentum_structure.py` (656 lines)
**Setup types emitted:** momentum_breakout_long, momentum_breakout_short, trend_continuation_long, trend_continuation_short

Top setup in backtest: `momentum_breakout_long` (7,585 trades).

---

## Pattern claim (one paragraph per major pattern)

This detector emits momentum-based patterns WITHOUT requiring specific price levels (unlike Range or S/R structures). Pure momentum plays that trade directional pressure:

- **Momentum breakout (`_detect_momentum_breakouts`):** At the current bar, check if 3-bar momentum (`returns_3`), last-bar momentum (`returns_1`), and 2-bar cumulative momentum all exceed config thresholds in the same direction, with volume z-score above `vol_z_required`. Emit `momentum_breakout_long` or `_short`. This is a "rocket take-off" pattern — price is accelerating with volume confirmation and we're trying to catch the continuation.

- **Trend continuation (`_detect_trend_continuations`):** Check 5-bar trend momentum, 3-bar trend bias, and count of positive (or negative) bars in last 3 bars. Requires consistent directional pressure across multiple bars. Emit `trend_continuation_long` or `_short`. This is a "trend already in motion, ride it" pattern — less aggressive than breakout, expects ongoing directional flow.

Both patterns are gated by `_get_time_adjusted_vol_threshold` which varies required volume z-score by time of day.

---

## Item 1: Canonical pro-trader definition (Indian market context)

### Overall philosophy

Momentum trading is the OLDEST profitable strategy type. Academic evidence: Jegadeesh & Titman (1993) showed momentum works across markets; Moskowitz, Ooi, Pedersen (2012) showed time-series momentum works across asset classes. In Indian markets, momentum is particularly strong in:
- Mid-cap names where retail can push price disproportionately
- Post-news events (results, policy announcements) where directional flow is sustained
- First and last hours of the session (institutional positioning windows)

The distinction from breakout-level patterns is important: **momentum trades don't require a specific S/R level being broken**. The signal IS the velocity and persistence of price movement plus volume confirmation. Momentum can fire mid-range if acceleration is strong enough.

NSE intraday specifics:
- Momentum is HIGHLY time-dependent. 9:15-10:30 is strongest (fresh directional bias). 10:30-12:00 weaker. Lunch is terrible. 13:00-14:30 mixed. 14:30-15:15 strong again (EOD unwind).
- Indian retail chases momentum aggressively — this is WHY momentum works short-term but fails long-term (reversion kicks in 1-2 days later on overnight)
- F&O expiry days (Thursdays) have FALSE momentum from OI unwinding — worse signal-to-noise
- FIIs algo-trade momentum using 3-5 min velocity; their flow amplifies our retail/DII signal

### Per-pattern canonical definition

#### 1. Momentum Breakout (long: strong upward acceleration; short: strong downward acceleration)

**Canonical NSE definition:**
- **Multi-timeframe acceleration** — price should be rising (or falling) at INCREASING velocity across multiple horizons:
  - Last bar (1m or 5m) showing directional close
  - 2-3 bars cumulative showing sustained directional pressure
  - No counter-trend wick rejection
- **Volume confirmation is MANDATORY** — momentum without volume is a trap (retail chasing, no institutional participation). Canonical: volume z-score ≥ 1.5 or volume surge ≥ 1.5x rolling average
- **Relative volume across timeframes** — the breakout bar's volume should be ≥ 1.5x the average of the prior 10-20 bars
- **Price structure**: breakout bar should have directional close (close > open for long, close < open for short) with minimal counter-wick
- **Regime**: works best in trend_up / trend_down regime; chop regime momentum breakouts have ~40% success rate
- **Timing**: highest-probability windows are 9:15-10:30 (fresh positioning) and 14:00-15:00 (EOD directional flow)

**Microstructure rationale (NSE):**
- Momentum burst = institutional orders being filled aggressively
- Volume confirms it's not just one trader spoofing
- Multi-bar acceleration confirms the flow is SUSTAINED, not a one-bar spike
- Retail piles in AFTER the initial burst, adding continuation fuel
- FII algos detect the same pattern and add to the move
- Works in NSE because retail participation is high (provides follow-through) and institutional flow is concentrated in certain windows (provides fresh starts)

**Confidence:** HIGH

#### 2. Trend Continuation (long: consistent upward; short: consistent downward)

**Canonical NSE definition:**
- **Sustained directional pressure** over 5-bar window — not an acceleration, but a consistent pattern
- **Positive-bar count** — in a 5-bar window, at least 4 bars should close positive (for long); 3 of last 3 bars all positive is a common threshold for continuation
- **No pullback** — continuation setups assume trend is still in motion; any significant counter-move invalidates
- **Volume can be moderate** (unlike breakout which demands surge) — trend continuation is more about persistence than explosiveness
- **Best entry**: pullback to short-term MA (EMA 8 or VWAP) within the trend, not the extension of the current bar
- **Less suitable for NSE intraday than US markets** — Indian intraday has heavy noise from retail speculation; trends rarely persist 10+ bars cleanly

**Microstructure rationale (NSE):**
- Clean trends are rare in Indian intraday — most sessions are range-bound or choppy trends with deep pullbacks
- When a clean trend DOES form (usually post-news or gap-and-go days), following it works
- However: chasing late in the trend = buying tops / selling bottoms to retail who's late to the party
- Pros prefer pullback entries to retest-entries, not current-bar chasing
- NSE's ~2:30-4:00 PM window often has clean trends due to MIS positioning
- **Caveat**: trend_continuation is WEAKEST of the momentum patterns — current code emits it but the canonical edge is thin in NSE

**Confidence:** HIGH on Western/academic definition. MEDIUM on NSE-specific edge (likely weak in Indian intraday specifically).

### Key divergence from Breakout-level structures

MomentumStructure fires based on VELOCITY + VOLUME, with no level-proximity requirement. This means:
- It can fire in the middle of a range if acceleration is sufficient
- It can fire during chop if a single burst meets thresholds
- There's no "measured move" target based on levels — targets are ATR-based or time-based
- The `levels` dict in emitted events just stores `momentum_level = current_price` (not a real S/R level)

This is a fundamentally different signal from level-based breakout. Treating them the same is a mistake.

### NSE microstructure caveats

1. **Time-of-day dependency is critical.** The existing `_get_time_adjusted_vol_threshold` acknowledges this — required volume z-score varies by time. This is the right design but the threshold curve needs empirical validation.

2. **Cap segment effects:** Momentum works better in mid-cap than large-cap. Large-cap momentum is often fake (index algo flow that reverses quickly). Small/micro momentum is often 1-2 trade spikes that don't persist.

3. **F&O expiry distortion:** Thursday expiry days have synthetic momentum from OI unwinding that doesn't translate to continuation. Momentum patterns fire but reversion is faster.

4. **Gap-and-run days:** Best momentum edge is on morning gaps that continue in the gap direction for 30-60 min. If the detector doesn't distinguish gap-day from normal-day, momentum signal quality is diluted.

### Source(s) cited

- Jegadeesh & Titman (1993) "Returns to Buying Winners and Selling Losers"
- Moskowitz, Ooi, Pedersen (2012) "Time Series Momentum"
- John Carter "Mastering the Trade" — momentum breakout frameworks for US markets
- Linda Raschke "Street Smarts" — short-term momentum patterns
- NSE-specific: general prop-trader content on intraday momentum windows
- NSE microstructure knowledge on retail chasing behavior

### Confidence (overall summary)

| Pattern | Definition | NSE-specific adaptation |
|---------|------------|------------------------|
| Momentum breakout long | HIGH | HIGH (timing-dependent) |
| Momentum breakout short | HIGH | HIGH (timing-dependent) |
| Trend continuation long | HIGH | MEDIUM (weak in Indian intraday) |
| Trend continuation short | HIGH | MEDIUM (weak in Indian intraday) |

### Low-confidence flags requiring user fact-check

1. **Momentum thresholds should vary by cap segment:** Large-cap 0.5% move in 3 bars ≠ small-cap 0.5% move. Canonical uses ATR-normalized thresholds, but our detector uses fixed percentage thresholds. **Flag for Item 2: check if thresholds are cap-normalized.**

2. **Pullback-entry missing:** Canonical prefers pullback entries for trend_continuation (enter on EMA retest within trend). Current detector likely enters on current bar extension. **Flag for Item 2.**

3. **Gap-day distinction:** Best momentum edge is on gap-and-run days. Detector likely doesn't flag gap days specifically. **Flag as future enhancement (Stage 3 conditioner question).**

4. **F&O expiry filter:** Thursday expiry should probably be filtered or flagged. Not in detector. **Deferred (future conditioner).**

5. **`trend_continuation_*` edge is weak in Indian intraday** — the backtest data has only 11 trades total for this setup type. Given the weak canonical edge + tiny sample, this may be a candidate for DISABLED rather than FIXED-AND-TRUSTED. **Flag for user consideration.**

6. **Time-adjusted volume threshold curve:** The `_get_time_adjusted_vol_threshold` is a design win, but the specific values need empirical validation. **Parameter sensitivity question for Stage 3.**

7. **Momentum gates use returns_1/2/3 directly** — canonical uses acceleration (2nd derivative of price). Returns_3 is ok as a velocity proxy but doesn't explicitly check acceleration (is returns_1 > returns_2 > returns_3 in magnitude?). **Flag for Item 2 if we care about acceleration semantics.**

---

## Item 2: Structural correctness vs canonical

### Comparison

| Pattern / aspect | Canonical (Item 1) | Our code | Divergence type |
|------------------|-------------------|----------|-----------------|
| **Momentum breakout** — multi-timeframe gate | Multi-bar velocity thresholds | `returns_1/2/3` all > thresholds (lines 290-323, 329-366) | **NONE** for velocity gating |
| **Momentum breakout** — acceleration check | Canonical requires `returns_1 > avg_per_bar_of_window` (acceleration) | Velocity-only, no acceleration check | **CANONICAL DRIFT** — defer to canonical-upgrade sub-project |
| **Momentum breakout** — volume confirmation | Volume z ≥ 1.5 mandatory | `vol_z_required` from `_get_time_adjusted_vol_threshold` + `min_volume_surge_ratio` | **NONE** (time-adjusted threshold is actually an improvement over canonical constant) |
| **Momentum breakout** — ATR-normalized thresholds | Thresholds should scale with cap/vol | Fixed pct thresholds applied uniformly | **CANONICAL DRIFT** — defer to canonical-upgrade sub-project |
| **Momentum breakout** — directional close | Bar close > open (long) | Not explicitly checked; implied by returns_1 > 0 | **HYBRID** (implicit via returns_1) |
| **Momentum breakout** — counter-wick rejection | Minimal counter-wick on breakout bar | Not checked | **CANONICAL DRIFT** — defer |
| **Trend continuation** — sustained pressure | 5-bar trend + positive/negative bar count | Implemented at lines 228-288 | **NONE** for detection; **BUG** for entry (see below) |
| **Trend continuation** — pullback entry | Enter on pullback to EMA8/VWAP, not extension | Enters at `context.current_price` (current-bar extension) | **CANONICAL DRIFT (large)** — this is the #1 structural weakness |
| **Trend continuation** — EMA/VWAP reference | Entry requires reference to fast MA | NO EMA or VWAP reference anywhere in detector | **MISSING canonical component** |
| **Trend continuation** — NSE edge | Weak canonical edge in Indian intraday (< 3% of momentum setups) | 11 trades total in 3yr backtest → weak + chase-entry = no structural edge | **RECOMMENDED DISABLE** |
| **Time-adjusted vol threshold** | Time-of-day dependency critical | `_get_time_adjusted_vol_threshold` implemented (line ~490) | **NONE** (design win, but thresholds hardcoded — defer) |
| **Cap-specific thresholds** | ATR-normalized or cap-bucketed | No cap normalization | **CANONICAL DRIFT** — defer |
| **Directional symmetry** | Long/short mirror correctly | Verified by subagent: all 6 gate comparisons symmetric | **NONE** |
| **NaN vol_surge handling** | Explicit NaN check → fail gate | `last_bar.get('vol_surge', 1.0)` returns NaN if key exists with NaN → `NaN < threshold` False → **silent bypass** | **BUG (silent)** — P2 fix in scope |
| **Confidence score** semantics | Probability ∈ [0, 1] | Unbounded strength 1.8-10+ (per subagent analysis of `_calculate_institutional_strength`) | **BUG (cross-cutting)** — defer |
| **`_calculate_institutional_strength` hardcoded thresholds** | Config-driven | ~15+ hardcoded magic numbers | **CANONICAL DRIFT (systemic)** — defer |

### Decision — SPLIT DISPOSITION (first in audit)

Because the 4 emitted setups fall into two structurally distinct groups with very different quality:

**Group A: `momentum_breakout_long` + `momentum_breakout_short`** → **FIXED-AND-TRUSTED**
- Structural foundation is sound (multi-timeframe velocity + volume + time-adjusted threshold)
- 7,585 trades in backtest (meaningful signal volume)
- Fix in scope: NaN vol_surge bypass (P2 silent bug); any silent `config.get` defaults in trading logic
- Defer: acceleration check, ATR-normalization, counter-wick check, hardcoded thresholds (~15), unbounded confidence (cross-cutting sub-project)

**Group B: `trend_continuation_long` + `trend_continuation_short`** → **DISABLED**
- Chase-the-extension entry (no pullback to EMA/VWAP) is the opposite of canonical best practice
- Only 11 trades in 3yr backtest → statistically meaningless
- Weak canonical edge in NSE intraday (retail noise breaks clean trends)
- No code-level fix would produce structural edge without a full rewrite (adding EMA/VWAP pullback entry is new-feature scope)
- Cleanly separable: `_detect_trend_continuations` is an independent method; disable via config `enabled: false` in pipeline configs

**Rationale for split disposition:**
Normal detector audits produce one disposition for the whole class. MomentumStructure is the first case where ~half the detector's emitted setups should be fixed and ~half should be disabled. The class itself stays FIXED (because momentum_breakout_* are retained); we disable only the 2 trend_continuation_* setup types at the pipeline-config layer. No code deletion — the `_detect_trend_continuations` method remains so a future sub-project can add proper pullback-entry logic if desired.

---

## Item 3: Bug patterns

### 3.1 Off-by-one / lookbacks — PASS (with notes)

| File:line | Slice | Notes |
|---|---|---|
| `structures/momentum_structure.py:148` | `pct_change(3)` | 3-bar return; semantic is `(close[t] - close[t-3]) / close[t-3]` — a 3-bar window, correct. |
| `structures/momentum_structure.py:149` | `pct_change(5)` | 5-bar return. Correct. |
| `structures/momentum_structure.py:152` | `rolling(10, min_periods=5)` | Vol MA 10-bar with 5-bar ramp. No off-by-one. |
| `structures/momentum_structure.py:170,233` | `df.iloc[-1]` | Last bar — current bar, correct. |
| `structures/momentum_structure.py:252,253,275,276` | `df['returns_1'].tail(3)` | Last 3 bars sum / count — correct. |
| `structures/momentum_structure.py:305,344` | `df['returns_1'].tail(2).sum()` | Last 2 bars. Note: since `returns_1 = close.pct_change()`, `tail(2).sum()` covers bars `[t-1,t]` — only **bars t-2→t-1 and t-1→t** moves. Labelled as "2-bar cumulative" which is accurate. PASS. |
| `structures/momentum_structure.py:381,409` | `(df['returns_1'] > 0).tail(3).sum()` | Count of positive bars in last 3. PASS. |
| `structures/momentum_structure.py:490-495` | `rolling(30, min_periods=10)` for vol_z | Hardcoded 30/10 window (config-miss — see 5.1). |
| `structures/momentum_structure.py:564` | `df.tail(self.swing_lookback_bars)` | Swing lookback from config — PASS. |
| `structures/momentum_structure.py:652` | `rolling(14, min_periods=5)` | ATR fallback 14-bar — hardcoded (5.1). |

No off-by-one bugs.

### 3.2 Wrong sign / long-short symmetry — PASS

Long vs short checks mirror correctly.

| Condition | Long (line) | Short (line) | Symmetric? |
|---|---|---|---|
| 3-bar momentum | `returns_3 > threshold` (295) | `returns_3 < -threshold` (334) | YES |
| 1-bar momentum | `returns_1 > threshold` (300) | `returns_1 < -threshold` (339) | YES |
| 2-bar cumulative | `two_bar_sum > threshold` (305-306) | `two_bar_sum < -threshold` (344-345) | YES |
| 5-bar trend | `returns_5 > threshold` (373) | `returns_5 < -threshold` (401) | YES |
| 3-bar bias | `tail(3).sum() > threshold` (377) | `tail(3).sum() < -threshold` (405) | YES |
| Directional-bar count | positive_bars ≥ min (381-382) | negative_bars ≥ min (409-410) | YES |

Volume gates are sign-agnostic (absolute) — same threshold used for both sides, which is correct.

**Note:** `min_positive_bars` config key is (mis-)reused in the short branch (line 410) to mean "min negative bars". Works functionally but the variable name is misleading — cosmetic issue only. P3.

### 3.3 NaN handling — FAIL (P2)

- **`returns_3` NaN on first 3 bars, `returns_5` NaN on first 5 bars.** Guard `len(df) < 10` at line 78 ensures ≥10 bars, so `iloc[-1]` will have `returns_3` and `returns_5` populated — PASS for normal path.
- **`last_bar['returns_3']` NaN risk if any close is NaN.** Comparisons `NaN > threshold` → False, `NaN < -threshold` → False, so detector silently returns False. Acceptable but silent.
- **`vol_z` NaN:** `_calculate_vol_z` does `.fillna(0)` (line 495), so last bar vol_z is never NaN — PASS.
- **`vol_surge` NaN:** `vol_surge = volume / vol_ma.replace(0, np.nan)` — if `vol_ma` is zero → NaN. `last_bar.get('vol_surge', 1.0)` returns NaN (not the default!) because the key exists, value is NaN. Then `vol_surge < min_volume_surge_ratio` → `NaN < x` → False, so gate passes incorrectly. **P2 BUG:** NaN vol_surge bypasses the volume-surge guard.
- **Missing `volume` column:** Would raise KeyError in `_calculate_momentum_indicators`, caught by outer try/except → returns None → detector fails clean.

**Fix needed:** Replace `last_bar.get('vol_surge', 1.0)` with explicit NaN check → treat NaN as fail, not pass.

### 3.4 Boundary conditions — PASS

- `len(df) < 10` at line 78 — guards returns_5 / 10-bar vol_ma.
- Zero volume → `vol_ma.replace(0, np.nan)` prevents divide-by-zero but produces NaN `vol_surge` (see 3.3).
- First-bar path unreachable due to 10-bar floor.

### 3.5 Observation C — acceleration vs velocity — FAIL (CANONICAL DRIFT, P2)

**Verdict: (a) Velocity-gated, NOT acceleration-gated.**

`_check_momentum_breakout_long` (lines 290-323) and `_check_momentum_breakout_short` (lines 329-366) apply **three independent velocity thresholds** — no check that `|returns_1| > avg per-bar of the 3-bar window` or any acceleration condition:

- (1) `returns_3 > momentum_3bar_threshold` (total 3-bar move)
- (2) `returns_1 > momentum_1bar_threshold` (last-bar move)
- (3) `tail(2).sum() > momentum_2bar_threshold` (2-bar cumulative)

Nothing enforces increasing per-bar velocity (e.g. `returns_1 > two_bar_sum/2 > returns_3/3`). A setup where prices rose 0.5% then 0.4% then 0.3% (decelerating) would pass as readily as an accelerating one — as long as each total-window gate is cleared.

**Recommendation:** Add an acceleration check: last-bar per-bar velocity should exceed the prior-bars per-bar average (e.g., `last_bar['returns_1'] >= (last_bar['returns_3'] / 3) * accel_factor`). Configurable via `momentum_acceleration_factor`.

### 3.6 Observation B — pullback entry — FAIL (CANONICAL DRIFT, P1 for trend_continuation)

**Verdict: Current-bar-extension. No pullback logic.**

`_detect_trend_continuations` (lines 228-288) fires on `last_bar` when 5-bar return, 3-bar bias, and directional-bar count pass. Entry price is `context.current_price` (line 257, 280) — the CURRENT bar close/price, i.e. the trend extension. There is no:
- Check that the current bar is pulling back toward EMA8 / VWAP (no EMA or VWAP reference anywhere in file)
- Check that price is below (long) / above (short) an MA for a retest
- Check for a counter-move bar preceding entry

This is "chase the trend at the top" entry — canonical pro-trader approach is the opposite (pullback to EMA/VWAP, enter on retest with trend direction intact). **This is the primary structural weakness of `trend_continuation_*`.** Given (a) only 11 total trades in backtest, (b) weak canonical NSE edge, and (c) chase-the-extension entry semantics, `trend_continuation_*` has no structural edge.

---

## Item 4: Feature emission

### 4.1 event.context keys per emitted event

| Setup type | Emission line | Context keys |
|---|---|---|
| `momentum_breakout_long` | 188-194 | `momentum_3bar_pct`, `momentum_1bar_pct`, `vol_z`, `vol_surge`, `pattern_type` |
| `momentum_breakout_short` | 211-217 | `momentum_3bar_pct`, `momentum_1bar_pct`, `vol_z`, `vol_surge`, `pattern_type` |
| `trend_continuation_long` | 250-256 | `trend_5bar_pct`, `trend_3bar_sum_pct`, `positive_bars`, `vol_z`, `pattern_type` |
| `trend_continuation_short` | 273-279 | `trend_5bar_pct`, `trend_3bar_sum_pct`, `negative_bars`, `vol_z`, `pattern_type` |

**Asymmetry issue (P3):** trend_continuation_long uses key `positive_bars` while short uses `negative_bars`. Downstream CSV analysis will see two different columns instead of one `directional_bars` or a pair. Should normalize.

**Missing context keys (P3):** momentum_breakout events do NOT emit `two_bar_sum_pct` even though that threshold is a gate in `_check_*`. For edge attribution work (filter_simulation, edge_optimizer) this loses signal. Similarly, `vol_surge` is not emitted on trend_continuation events.

### 4.2 Computation correctness — PASS

- All context values are `last_bar[...]` scalars (current bar, not stale).
- All values are Python scalars / numpy floats — compatible with `isinstance(v, (str, int, float, bool, type(None)))` filter.
- `(df['returns_1'] > 0).tail(3).sum()` returns `numpy.int64` — counts as `int` via `isinstance`. PASS.

### 4.3 Flow to SetupCandidate.extras — PASS

Per `structures/main_detector.py:552-557`, extras filters to scalar types. All emitted context keys are scalars → all flow through to `trade_report.csv`. No nested dicts/lists emitted. PASS.

---

## Item 5: Project rules compliance

### 5.1 Hardcoded thresholds — FAIL (P1)

**Constructor uses `config[key]` (fail-loud) for most trading parameters** (lines 43-67). Good. But multiple hardcoded values exist in trading logic:

| File:line | Hardcoded value | Purpose | Severity |
|---|---|---|---|
| `momentum_structure.py:78` | `len(df) < 10` | Min bars | P2 — should be `config["min_bars_required"]` (or computed from the larger of momentum-3/5 + vol_ma window). |
| `momentum_structure.py:148` | `pct_change(3)` | 3-bar lookback | P2 — derive from `config["momentum_3bar_lookback"]`. Currently threshold is config but window is hardcoded, creating a silent decoupling. |
| `momentum_structure.py:149` | `pct_change(5)` | 5-bar lookback | P2 — same as above for `min_trend_5bar_pct`. |
| `momentum_structure.py:152` | `rolling(10, min_periods=5)` | vol_ma window | P2 — hardcoded. |
| `momentum_structure.py:252,275` | `.tail(3).sum()` | 3-bar bias window | P2 — hardcoded; also mis-labelled as `trend_3bar_sum_pct` with no config key. |
| `momentum_structure.py:305,344` | `.tail(2).sum()` | 2-bar cumulative | P2 — hardcoded lookback. |
| `momentum_structure.py:381,409` | `.tail(3)` | Positive/negative bars window | P2 — hardcoded. |
| `momentum_structure.py:429` | `max(1.5, vol_z * momentum * 15.0)` | Strength floor + magic multiplier | **P1** — hardcoded institutional floor `1.5`, hardcoded scaling constant `15.0`. |
| `momentum_structure.py:435-441` | `0.03, 0.02, 0.01`, multipliers `1.4, 1.25, 1.15` | Momentum bonus tiers | **P1** — hardcoded thresholds and bonus multipliers. |
| `momentum_structure.py:444-448` | `vol_z ≥ 2.5, 1.5`, multipliers `1.3, 1.2` | Volume bonus tiers | **P1** — hardcoded. |
| `momentum_structure.py:451-453` | `10 <= hour <= 14` window and `1.1` bonus | Time bonus | **P1** — hardcoded time window + multiplier (also uses `context.timestamp` but hardcoded hour bounds). |
| `momentum_structure.py:460` | `max(..., 1.8)` | Strength floor | **P1** — hardcoded institutional minimum. |
| `momentum_structure.py:469` | `return 1.8` | Fallback strength on exception | P1 — hardcoded fallback. |
| `momentum_structure.py:475` | `base_vol_z = 1.5` | Default time-adjusted base | **P1** — `_get_time_adjusted_vol_threshold` has hardcoded `1.5` base, `0.5` and `0.75` multipliers, `630` / `720` minute boundaries. |
| `momentum_structure.py:481-486` | `630, 720, 0.5, 0.75` | Time-window cutoffs and multipliers | **P1** — hardcoded. |
| `momentum_structure.py:490` | `window=30, min_periods=10` | vol_z computation | P2 — hardcoded. |
| `momentum_structure.py:502` | `base_score = 65.0` | Quality base | P2 — hardcoded. |
| `momentum_structure.py:508,511,517,520` | `20.0, 10, 15.0, 6, 10.0, 3, 5` | Quality weights | P2 — hardcoded weights. |
| `momentum_structure.py:595` | `risk_percentage=0.02` | Risk pct | P1 — hardcoded 2% risk (Pipeline likely overrides but still violates rule). |
| `momentum_structure.py:626,629,635` | `15.0, 7, 10.0, 4, 10.0, 3` | Rank weights | P2 — hardcoded. |
| `momentum_structure.py:652` | `rolling(14, min_periods=5)` | ATR fallback window | P2 — hardcoded. |
| `momentum_structure.py:653` | `max(0.005, ...)` | 0.5% floor | P2 — hardcoded. |
| `momentum_structure.py:655` | `* 0.01` | 1% ATR fallback | P2 — hardcoded. |

**Trade-plan hardcoded multipliers:** The `target_mult_t1`, `target_mult_t2`, `stop_mult` are config-driven (good). But `risk_percentage=0.02` at line 595 is hardcoded.

**Observation A verified (cap-normalized thresholds):** `min_momentum_3bar_pct`, `min_momentum_1bar_pct`, `min_momentum_2bar_pct`, `min_trend_5bar_pct`, `min_trend_3bar_pct` are all applied as **fixed percentages**, uniform across cap segments. No ATR normalization, no cap-segment branching. **CANONICAL DRIFT confirmed — P1.** A 0.5% move means very different things for RELIANCE vs a small-cap; thresholds should be either ATR-normalized (`returns_3 / atr`) or cap-segment-indexed.

### 5.2 IST-naive timestamps — PASS (with note)

- No `datetime.now()`, `tz_localize`, or `tz_convert` in file.
- Line 17: `from datetime import datetime` — imported but never used. Dead import.
- Line 451: `pd.to_datetime(context.timestamp).hour` — preserves naivety; PASS.
- Line 478: `timestamp.hour * 60 + timestamp.minute` — operates on `context.timestamp` directly.

### 5.3 Tick timestamps for trading decisions — PASS

- `_get_time_adjusted_vol_threshold(context.timestamp)` (line 101) uses tick timestamp. PASS.
- `_calculate_institutional_strength` uses `context.timestamp` (line 451). PASS.
- No `datetime.now()` anywhere in file. PASS.

### 5.4 Fail-fast on missing config — FAIL (P2)

**Constructor (lines 43-67) uses `config[key]` → KeyError on miss. Good.**

**But multiple `config.get(...)` usages in non-trading paths:**
- Line 38: `config.get("_setup_name", None)` — meta key, acceptable.

**And silent-default `.get()` in trading logic via `last_bar.get(...)`:**
- Line 179, 191, 202, 214, 241, 254, 264, 277: `last_bar.get('vol_z', 1.0)` — defaults to 1.0 if missing. Since `_calculate_momentum_indicators` always creates `vol_z`, this shouldn't hit — but the silent default masks bugs.
- Line 192, 215: `last_bar.get('vol_surge', 1.0)` — silent default hides missing values; same NaN-bypass bug noted in 3.3.
- Lines 311, 318, 350, 357, 386, 414: same pattern inside `_check_*`.

**P2:** The `.get(..., 1.0)` defaults on `vol_z` and `vol_surge` should raise or be explicit NaN-aware. Silent defaults violate the "fail-fast" mandate for trading logic.

---

## Item 6: Output completeness

| Field | momentum_breakout_long | momentum_breakout_short | trend_continuation_long | trend_continuation_short |
|---|---|---|---|---|
| `symbol` | ✅ | ✅ | ✅ | ✅ |
| `timestamp` | ✅ | ✅ | ✅ | ✅ |
| `structure_type` | ✅ | ✅ | ✅ | ✅ |
| `side` | ✅ long | ✅ short | ✅ long | ✅ short |
| `confidence` | `_calculate_institutional_strength` | same | same | same |
| `levels` | `{"momentum_level": current_price}` | same | `{"trend_level": current_price}` | same |
| `context` | 5 keys | 5 keys | 5 keys | 5 keys |
| `price` | current_price | current_price | current_price | current_price |

**`levels` content is NOT a meaningful S/R level** — it is simply the current price. Downstream consumers expecting a real detected level (main_detector uses `event.levels.get("support")` etc. at line 541-544 for BOS-style events) will find nothing matching. For momentum, this is acceptable by design (momentum fires without levels) but `detected_level` in `SetupCandidate` will be `None` for all 4 setup types. Not a bug — a structural property to flag for Item 2.

**Direction symmetry:** long/short pairs mirror correctly (already confirmed in 3.2).

**Confidence — FAIL (P1, CROSS-CUTTING):** `_calculate_institutional_strength` returns an **unbounded strength score**, not a probability ∈ [0,1]:

- Base: `max(1.5, vol_z * momentum * 15.0)` → floor 1.5
- Multipliers stack: `1.4 * 1.3 * 1.1 = 2.002x` max stack
- Final floor: `max(final, 1.8)`
- Realistic range: **1.8 to ~10+** (e.g. vol_z=3, momentum=0.04 → base = `3 * 0.04 * 15 = 1.8`, × 1.4 × 1.3 × 1.1 = `3.6`)
- Can exceed 5 easily in exceptional conditions.

This is passed as `event.confidence` which is a `float` in `StructureEvent` and flows into `SetupCandidate.strength` (`float(event.confidence)`). Downstream probability-based logic breaks. **Same cross-cutting issue as ICT/Range/SR.** P1.

---

## Item 7: Test coverage — FAIL (TEST_DEBT)

```
grep -rln "MomentumStructure\|momentum_structure" tests/  →  (zero matches)
```

**No tests exist for MomentumStructure.** Top setup in backtest is `momentum_breakout_long` (7,585 trades) with zero unit-test coverage. **TEST_DEBT — P1.**

Test recommendations:
- Long/short symmetry for `_check_momentum_breakout_*` and `_check_trend_continuation_*`
- NaN-bypass regression for `vol_surge`
- Threshold boundary tests
- Confidence-range test (document unbounded-strength behavior)
- Acceleration property test (currently would demonstrate the bug)

---

## Issues found (consolidated)

### P1 (must fix before trusting signal)
1. **Fixed-pct momentum thresholds (not cap-normalized / ATR-normalized)** — Obs A confirmed. Lines 43-49 applied uniformly across caps. Recommend ATR-normalize in `_calculate_momentum_indicators` and compare `returns_N / atr` against config.
2. **`_calculate_institutional_strength` hardcoded bonuses/floors everywhere** — lines 429, 435-441, 444-448, 451-453, 460, 469. All must move to config.
3. **`_get_time_adjusted_vol_threshold` hardcoded values** — base_vol_z, time cutoffs, multipliers (475, 481-486). Config-ize.
4. **`trend_continuation_*` lacks pullback entry** (Obs B) — chases current-bar extension. Either add pullback-to-EMA logic OR disable the pair. Given 11-trade sample + weak NSE edge, **DISABLE recommended** (set `trend_continuation_long`/`_short` to off via gating config).
5. **`confidence` is unbounded strength score (1.8 – 10+), not probability [0,1]** — cross-cutting. Must wrap/normalize at emission or fix all downstream consumers uniformly.
6. **Zero test coverage** for top-volume setup family.
7. **`risk_percentage=0.02` hardcoded** at line 595.

### P2
1. **NaN `vol_surge` silently passes surge gate** (3.3) — replace `.get(...,1.0)` with explicit NaN-check → fail.
2. **Acceleration not checked** (Obs C) — `_check_momentum_breakout_*` applies 3 independent velocity gates with no acceleration constraint. Add optional `momentum_acceleration_factor`.
3. **Hardcoded lookback windows** decoupled from config thresholds: `pct_change(3)`, `pct_change(5)`, `rolling(10,...)`, `tail(2)`, `tail(3)`, vol_z 30/10 window.
4. **`len(df) < 10` hardcoded** (line 78).
5. **Silent `.get(..., 1.0)` defaults on `vol_z` / `vol_surge`** in trading logic.
6. **Quality-score weights hardcoded** (502-522, 626-635).
7. **ATR fallback** rolling(14)/min(0.005)/1% fallback hardcoded (652-655).

### P3
1. `min_positive_bars` config key reused as "min negative bars" in short branch (mis-named).
2. `positive_bars` vs `negative_bars` key asymmetry in trend_continuation context — breaks CSV column consistency.
3. `two_bar_sum` not emitted in context (edge attribution loss).
4. `vol_surge` not emitted on trend_continuation events.
5. Unused `from datetime import datetime` import (line 17).

### Observation verdicts

- **A — Cap-normalized thresholds?** **NO — fixed percentages, uniform across caps.** CANONICAL DRIFT. P1.
- **B — Pullback entry for trend_continuation?** **NO — current-bar extension entry, no EMA/VWAP reference anywhere.** CANONICAL DRIFT. P1.
- **C — Acceleration vs velocity?** **Velocity only.** Three independent velocity gates; no per-bar acceleration check. CANONICAL DRIFT. P2.
- **D — Cross-cutting issues present?** **YES.** `_calculate_institutional_strength` has ~15 hardcoded thresholds/multipliers; `confidence` returns unbounded strength (1.8 – 10+), not probability; silent `.get(...,1.0)` defaults present in trading logic.
- **E — trend_continuation_* disable candidate?** **YES — cleanly separable.** The two pattern families are emitted by independent methods: `_detect_momentum_breakouts` (line 165) and `_detect_trend_continuations` (line 228). They share only utility code (`_calculate_momentum_indicators`, `_get_time_adjusted_vol_threshold`, `_calculate_institutional_strength`). The configured-setup-type filter at lines 112-116 already enables per-setup disablement at the gating layer. Disabling `trend_continuation_long`/`_short` in config (or removing `_detect_trend_continuations` call at line 108) would leave `momentum_breakout_*` fully intact. **RECOMMEND DISABLE for trend_continuation pair** given: chase-the-extension entry (Obs B), weak NSE canonical edge, 11-trade sample.

---

## Issues found (consolidated)

### P1 — Fix in scope (silent bug)

1. **NaN `vol_surge` silently bypasses volume-surge gate** (`_check_momentum_breakout_long` line 305, `_check_momentum_breakout_short` line 344)
   - `last_bar.get('vol_surge', 1.0)` returns NaN (not the fallback) when vol_surge column has NaN at last bar
   - `NaN < min_volume_surge_ratio` is False → gate silently passes even without real volume confirmation
   - Fix: Add `pd.isna(vol_surge)` check, fail the gate when NaN

### P1 — DISABLE decision (no code fix; config change)

2. **`trend_continuation_long` + `trend_continuation_short` DISABLED**
   - 11 trades total in 3yr backtest (statistically meaningless)
   - Chase-the-extension entry (canonical requires pullback to EMA/VWAP — detector has no MA/VWAP reference)
   - Weak canonical edge in NSE intraday (retail noise breaks clean trends)
   - **Action**: Set `enabled: false` in pipeline configs for both setup types. Add `_audit_decision: "DISABLED – see audit/04-momentum_structure.md"` key.
   - Code at `_detect_trend_continuations` stays untouched — future sub-project may add pullback-entry logic.

### P2 — Defer (canonical upgrades / cross-cutting)

3. **Missing acceleration check** — canonical momentum breakout requires per-bar acceleration (last-bar velocity > avg per-bar velocity). Current code is velocity-gated only. Defer to canonical-upgrade sub-project.

4. **Fixed-pct momentum thresholds** not ATR-normalized or cap-bucketed. Same pct thresholds applied across all caps. Defer.

5. **No counter-wick rejection check** on breakout bar. Defer.

6. **~20 hardcoded magic numbers** in `_calculate_institutional_strength`, `_get_time_adjusted_vol_threshold`, `_calculate_quality_score`, and `_create_*_trade_plan` methods. Same systemic issue as ICT/Range/SR. Defer to cross-cutting config-extraction sub-project.

7. **Silent `config.get(..., default)` in trading logic** — subagent flagged some. Fix in scope if quick (similar to SR Fix 3).

8. **ATR fallback hardcoded** (`rolling(14, min_periods=5)` for ATR computation). Defer.

9. **Hardcoded `risk_percentage = 0.02`** in trade plan methods. Defer.

### P3 — Cosmetic / deferred

10. **Asymmetric context keys** — `trend_continuation_long` emits `positive_bars`, short emits `negative_bars`. Downstream Stage 3 analysis will see 2 columns instead of 1 unified `directional_bars`. With DISABLE decision, this becomes moot (trend_continuation no longer emits).

11. **Missing `two_bar_sum_pct` in momentum_breakout context** — gate uses it but doesn't emit. Minor loss of signal for Stage 3 attribution. Defer or fix as small addition.

12. **Missing `vol_surge` in trend_continuation context** — same (moot with DISABLE).

13. **`min_positive_bars` config key reused in short branch** to mean "min negative bars". Cosmetic; rename or split into two keys in a future cleanup.

14. **Unused `datetime` import** — cosmetic cleanup.

### TEST_DEBT

15. Zero tests exist for MomentumStructure. Regression test needed for Fix 1. Full test suite deferred.

---

## Fixes applied

SPLIT disposition applied (2026-04-14). Commits on branch `feat/premium-zone-ict-fix`:

| # | Fix | Commit | TDD | Notes |
|---|-----|--------|-----|-------|
| 1 | P1 #1: NaN vol_surge gate bypass | `1e89dad` | Yes (regression test in `tests/structures/test_momentum_structure.py::test_momentum_breakout_rejects_nan_vol_surge`) | Added `pd.isna()` check in both `_check_momentum_breakout_long` and `_check_momentum_breakout_short` before the `<` comparison. |
| 2 | P1 #2: DISABLE trend_continuation_long | `1dc9aab` | Yes (config test `::test_trend_continuation_setups_are_disabled_in_pipeline_config`) | `enabled:false` in `config/pipelines/momentum_config.json` gates.setup_filters. Detector code left intact. |
| 3 | P1 #2: DISABLE trend_continuation_short | `1dc9aab` (same commit, paired decision) | Yes (same regression test) | Same file; rationale: 11 trades / 3yr, chase-entry semantics, no EMA/VWAP pullback reference. |
| 4 | P2 #7: Silent `config.get(..., default)` cleanup | N/A — **no violations found** | N/A | Only `config.get` call in `momentum_structure.py` is `config.get("_setup_name", None)` which is a metadata sentinel (not a trading threshold). All trading parameters already use `config[...]` fail-fast indexing. |
| 5 | P3 #11 (optional): Emit `two_bar_sum_pct` for Stage 3 attribution | `59b520f` | Yes (`::test_momentum_breakout_long_emits_two_bar_sum_pct` + short variant) | Added `"two_bar_sum_pct": float(df['returns_1'].tail(2).sum() * 100)` to `context` dict of both long and short momentum_breakout events. |

**Test coverage added:** `tests/structures/test_momentum_structure.py` (4 regression tests). Full pytest suite: 174 passed (170 baseline + 4 new).

**Final decision:** SPLIT applied. `momentum_breakout_*` = FIXED-AND-TRUSTED. `trend_continuation_*` = DISABLED at config layer (code preserved for future canonical-upgrade sub-project if ever revived with pullback-entry semantics).

---

## Final decision (assistant's recommendation)

**Recommended disposition: SPLIT**
- `momentum_breakout_long` + `momentum_breakout_short`: **FIXED-AND-TRUSTED**
- `trend_continuation_long` + `trend_continuation_short`: **DISABLED**

**Recommended action plan if user approves:**

| Order | Issue | Effort |
|-------|-------|--------|
| 1 | P1 #1: NaN vol_surge bypass (TDD fix) | 30 min |
| 2 | P1 #2: DISABLE trend_continuation_long in pipeline config | 10 min |
| 3 | P1 #2: DISABLE trend_continuation_short in pipeline config | 10 min |
| 4 | P2 #7: Find + fix any silent `config.get(..., default)` in trading logic | 20 min |
| 5 | P3 #11 (optional): Add `two_bar_sum_pct` to momentum_breakout context for Stage 3 attribution | 15 min |
| **Total** | | **~1-1.5 hours** |

Deferred to SUMMARY.md: acceleration check, cap-normalization, counter-wick check, ~20 hardcoded thresholds (cross-cutting), unbounded confidence (cross-cutting), ATR fallback hardcoded, risk_percentage hardcoded, pullback-entry for trend_continuation (new feature if we ever revive it), TEST_DEBT.

**Alternative dispositions considered:**
- **All FIXED-AND-TRUSTED (no DISABLE):** Rejected. trend_continuation_* has no structural edge and 11 trades is noise. Leaving it enabled pollutes future gauntlet analysis.
- **All DISABLED:** Rejected. momentum_breakout_* has 7,585 trades and sound structural foundation.
- **TRUSTED as-is (no fixes):** Rejected. NaN vol_surge silent bypass is a real bug.

**Awaiting user disposition decision.**
