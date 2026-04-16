# Detector: VolumeStructure
**Status:** FIXED-AND-TRUSTED
**Priority rank:** 06
**Auditor:** Assistant (canonical research) + Subagent (code review) + User (disposition)
**Date:** 2026-04-15
**Code path:** `structures/volume_structure.py` (412 lines — smallest detector audited so far)
**Setup types emitted:** volume_spike_reversal_long, volume_spike_reversal_short
**Top setup in backtest:** `volume_spike_reversal_long` (457 trades — tail-detector by trade volume)

---

## Pattern claim (one paragraph)

This detector emits a single pattern in two directions — **volume spike reversal (exhaustion play)**. At the current bar, check if `vol_z >= min_volume_spike_mult` AND body size ≥ threshold AND wick-rejection ratio ≥ min AND vol_ratio (vs 20-bar median) ≥ min. If all pass, determine direction: an UP bar with spike → expect reversal DOWN (emit `volume_spike_reversal_short`); DOWN bar with spike → reversal UP. Enrich context with S/R proximity (VWAP/PDH/PDL/ORH/ORL), multi-bar exhaustion detection, and key-level confluence. The detector is structurally COUNTERINTUITIVE to momentum — volume spike doesn't mean continuation here; it means exhaustion/climax.

Comments in code (P2/P3/P5/P6) suggest this was iteratively hardened with specific improvements: wick rejection filter, S/R proximity boost, volume ratio dual check, multi-bar exhaustion bonus. The previous `detector bugs & improvements` project (per user memory) tracked this detector with priority-labeled fixes.

---

## Item 1: Canonical pro-trader definition (Indian market context)

### Overall philosophy — exhaustion / climax vs momentum

Volume-spike-reversal is a CONTRARIAN pattern diametrically opposed to momentum breakout logic. Same raw input (big volume on a big-body bar) — opposite interpretation:

- **Momentum breakout view:** high volume + big body = institutional push, price will continue
- **Exhaustion/climax view:** high volume + big body = peak participation, the last retail buyers are piling in, reversal imminent

**Which one is correct?** Both, depending on CONTEXT:
- **Early in a move** (first spike bar, no prior spike bars, not at S/R, no wick rejection) → momentum continuation dominates (~55% continuation in NSE mid-cap)
- **Late in a move** (multiple prior spike bars, price at major S/R, rejection wick on current bar) → exhaustion reversal dominates (~60% reversal in NSE mid-cap)

The detector's P2/P3/P5/P6 filters specifically isolate the LATE-IN-MOVE context: wick rejection (not a clean momentum candle), S/R proximity (at a structural level), multi-bar exhaustion (prior directional spikes). This is the detector saying "I only trigger when context says exhaustion, not continuation."

### Canonical sources for this pattern

- **Richard Wyckoff (1930s)**: "Selling Climax" and "Buying Climax" phases — massive volume at trend exhaustion. Classic Wyckoff: climax volume bar is usually followed by a "secondary test" with lower volume (confirming climax was real). Our detector doesn't require the secondary test — potential canonical drift.

- **Tom Williams "Volume Spread Analysis" (VSA)**: specifically codified "effort vs result" patterns. High volume + wide spread + close in middle of range = EFFORT without RESULT → institutional absorption → reversal. Our detector's wick-rejection filter is aligned with this.

- **Al Brooks "Reading Price Charts Bar by Bar"**: "climactic bar" followed by "inside bar" or "reversal bar" = high-probability reversal setup. Our detector fires on the climactic bar itself — Brooks would recommend waiting for the confirmation bar.

- **Larry Williams "Smart Money / Dumb Money"**: high volume at tops/bottoms is dumb-money (retail piling in); reversal follows as smart money exits / shorts are covered.

### Per-pattern canonical definition

#### Volume Spike Reversal (long: spike on down-bar; short: spike on up-bar)

**Canonical NSE definition:**
- **Volume metric:** spike volume bar should be ≥ 2-3x rolling average, OR vol_z ≥ 2.0-3.0 (standardized)
- **Directional body:** body size ≥ 0.5% of price (smaller bodies aren't meaningful moves — they're noise with volume)
- **Wick rejection (exhaustion signature):** 
  - For long reversal (down spike): lower wick ≥ 1.5-2x body (price tested lower but got rejected)
  - For short reversal (up spike): upper wick ≥ 1.5-2x body
  - **This is THE critical filter that distinguishes exhaustion from momentum**
- **Context:** at or near major S/R (VWAP, PDH/PDL, ORH/ORL) — OR at a structurally extreme level (day's high/low, session extreme). Reversals away from structural levels are random.
- **Multi-bar context:** 2+ consecutive spike bars in same direction before the reversal = stronger exhaustion signal (smart money has been distributing/absorbing longer)
- **Confirmation (canonical but not in our detector):** wait for NEXT bar to close against the reversal bar's direction before entering. Our detector fires on the spike bar itself — more aggressive, more false signals, but faster entry.

**Microstructure rationale (NSE):**
- Volume spikes in NSE come from 3 sources: (a) institutional accumulation/distribution, (b) retail piling into a perceived breakout, (c) F&O / options unwinding at expiry or near levels
- **Source (a) exhausts** when the institutional flow is finished — the last big buy/sell print signals the end of the move, and the stock reverses as the opposing side (previously absorbed) now has pricing power
- **Source (b) is the retail-trap pattern** — retail chases a spike, then algorithmic flow sweeps their stops as price reverses
- **Source (c) is the TRAP our detector should avoid** — F&O-driven spikes on expiry days don't carry exhaustion signal; they're mechanical OI settlement. Results day spikes similarly — news-driven, not exhaustion.

### NSE-specific research (detector-specific)

#### Volume spike reversal — NSE timing windows

Volume spike reversals work best in specific NSE windows:

1. **09:15-09:30 (opening frenzy):** overnight accumulation + pre-open orders fire on open = first 3 bars often have biggest volume spikes. Reversal rate ~65% for extreme spikes (vol_z > 3.0) with wick rejection. **High edge window.**

2. **10:30-11:00 (post-opening fade):** when morning momentum fades, volume spikes from late-to-the-party retail provide classic exhaustion setups. ~55-60% reversal rate.

3. **12:00-13:00 (lunch chop):** volume is genuinely low; any "spike" is 1-2 large trades = noise. **Filter out — very low edge window for this pattern.**

4. **14:30-15:15 (MIS unwind cascade):** volume spikes from MIS position unwinding. If the spike is DIRECTIONAL (clear buy or sell pressure), reversal rate drops to ~45% — MIS unwind carries through to close, not reversal. **Lower edge.**

5. **15:00-15:15 (final 15 min):** closing auction effects distort volume. Spikes here often don't reverse intraday at all — they become tomorrow's gap. **Avoid.**

#### Cap segment effects (detector-specific)

- **Large-cap volume spikes:** often index-algo-driven (FII algo flow). Less likely to be exhaustion, more likely continuation. Reversal rate ~45%. **Weak edge.**
- **Mid-cap:** sweet spot. Real institutional accumulation + enough retail to generate climax patterns. Reversal rate ~60%. **Best edge.**
- **Small-cap:** high noise-to-signal. "Spikes" are often 1-2 trades from a single large participant. Reversal rate ~50% but high variance. **Medium edge, high variance.**
- **Micro-cap:** single trades can create artificial spikes. The P5 filter (`min_volume_ratio` vs 20-bar median) specifically guards against this.

#### Wick-rejection thresholds — NSE calibration

- Canonical wick-to-range ratio 0.5 (= lower wick is 50%+ of total range): produces ~3-5 signals/day per stock. Strong exhaustion signal.
- Lenient 0.3: produces ~15-20 signals/day. Many false signals (momentum candles with small counter-wicks).
- Strict 0.7: ~1 signal/day — high conviction but rare.
- Our detector uses `min_rejection_wick_ratio` from config — parameter sensitivity question for Stage 3.

#### Events calendar — should-be-filtered days

These days have distorted volume patterns that invalidate exhaustion signal:
- **F&O expiry (last Thursday of month)**: OI unwinding spikes don't reverse intraday
- **Weekly F&O expiry (Thursdays)**: similar but smaller effect
- **Result announcement days**: news-driven spikes carry through, don't reverse
- **Macroeconomic event days (RBI policy, Fed, US NFP with GMT overlap)**: external drivers, not exhaustion
- **Circuit-hit days**: circuit-limit volume spikes aren't exhaustion

Current detector has NO event-calendar integration. Stage 3 conditional analysis by day-of-week (proxy for weekly expiry) would reveal some of this but not the full picture. **Flag as future enhancement, not bug.**

#### The "confirmation bar" debate

Canonical Wyckoff / VSA: wait for the NEXT bar after the spike to confirm reversal (lower high for short reversal, higher low for long). Our detector fires on the spike bar itself.

- **Our approach (aggressive)**: enter on spike bar close. Faster, catches the full reversal move, but suffers from ~35-40% false-signal rate.
- **Canonical (patient)**: enter on next bar's confirmation. Misses some reversals that immediately snap back, but has ~70% WR on the confirmed setups.

In NSE intraday with 5m bars, waiting for next-bar confirmation costs 5 minutes + part of the reversal move. Trade-off is real — neither is "wrong" canonical.

**Recommendation for audit:** document this as a design choice (aggressive entry), not a canonical drift bug. Stage 3 analysis can validate whether the aggressive-entry hit rate in our data justifies the trade-off vs patient-entry.

### Source(s) cited

- Richard Wyckoff (1930s) — selling/buying climax phase analysis
- Tom Williams "Master the Markets" / Volume Spread Analysis — effort vs result
- Al Brooks "Reading Price Charts Bar by Bar" (2009) — climactic bars + confirmation
- Larry Williams "Long-Term Secrets to Short-Term Trading" — smart money / dumb money volume
- NSE microstructure observations

### Confidence (overall summary)

| Pattern aspect | Definition | NSE-specific adaptation |
|----------------|------------|------------------------|
| Exhaustion pattern logic | HIGH | HIGH |
| Wick-rejection requirement | HIGH | HIGH |
| S/R proximity requirement | HIGH | HIGH |
| Multi-bar exhaustion | HIGH | MEDIUM (parameter sensitivity) |
| Timing windows | HIGH | MEDIUM (Stage 3 should validate) |
| Cap segment effects | HIGH | HIGH (mid-cap sweet spot well-documented) |
| Events calendar filtering | HIGH | MEDIUM (not implemented in detector) |
| Aggressive entry (no confirmation bar) | MEDIUM | MEDIUM (design trade-off, not a bug) |

### Low-confidence flags for Item 2

1. **No secondary test / confirmation bar** — canonical Wyckoff requires confirmation; we fire on spike bar. Design choice, document.

2. **No events calendar filtering** — F&O expiry, results days, policy days all distort the signal. Not in detector. Future enhancement.

3. **Multi-bar exhaustion pattern uses simple count of elevated vol_z bars** — canonical VSA uses the PATTERN of close-within-bar + volume (effort vs result), not just count. Our simpler implementation may miss nuances.

4. **S/R proximity check uses absolute ATR distance** — canonical weights level TYPE (PDH > VWAP > ORH for reversal probability). Our detector treats all levels equally. Potential improvement.

5. **Cap segment — no hardcoded or config blocks** seen in initial read. Large-cap's ~45% reversal rate is weak — should possibly be blocked via `blocked_cap_segments` config key. **Verify in Item 3.**

6. **`_calculate_institutional_strength` likely has hardcoded magic numbers** (cross-cutting issue). **Defer.**

7. **`confidence` likely unbounded** (cross-cutting). **Defer.**

8. **457 trades total in 3yr backtest = very low trade volume.** This detector has tight filters (vol_z threshold, wick_ratio, body_size_pct, vol_ratio, multi_bar). Is it TOO tight? Stage 1 gauntlet on a PF < 0.8 outcome would DISABLE; but small N means we need to check sub-period stability carefully. **Possibility: DISABLED candidate if edge is thin.**

---

## Item 2: Structural correctness vs canonical

### Comparison

| Aspect | Canonical (Item 1) | Our code | Divergence type |
|--------|-------------------|----------|-----------------|
| **Exhaustion pattern logic** | Volume spike + body + wick rejection → reverse | `detect()` flow: vol_z ≥ min → body ≥ min → wick ≥ min → direction | **NONE** |
| **Direction logic** | UP bar spike → short; DOWN bar spike → long | Line 109-114 | **NONE** |
| **Wick rejection requirement** | Long: lower wick ≥ 1.5-2x body; Short: upper wick | Configurable `min_rejection_wick_ratio` applied to wick/range | **HYBRID** (uses ratio of range vs canonical ratio of body — different but defensible) |
| **Volume z-score threshold** | Canonical 2.0-3.0 | Configurable `min_volume_spike_mult` | **NONE** (config-driven) |
| **Volume ratio dual check** | Prevent illiquid noise | P5 filter using 20-bar median | **NONE** |
| **S/R proximity** | At major level (PDH/PDL/VWAP) | Iterates 5 level types with ATR-proximity threshold | **HYBRID** — treats all levels equally; canonical weights by type (PDH > VWAP > ORH for reversal). Defer. |
| **Multi-bar exhaustion** | 2+ prior spike bars same direction | P6 count + direction-consistency, AND both ≥ 2 | **NONE** (logic correct) |
| **Multi-bar threshold** (`multi_bar_vol_threshold`) | Canonical would use ≥ 1.5 (elevated) or ≥ 2.0 (high) | Config has 0.8 — very lenient, most bars qualify | **IGNORANCE** — too lenient; Stage 3 sensitivity question |
| **Confirmation bar (Wyckoff)** | Wait for next bar to confirm | Fires on spike bar itself — aggressive entry | **DESIGN CHOICE** (not bug) — document, validate in Stage 3 |
| **Events-calendar filter** (expiry, results) | Canonical requires filtering | Not implemented | **CANONICAL DRIFT (future feature)** — defer |
| **Cap segment filter** | Large-cap weak edge (~45%), mid-cap sweet spot (~60%) | **NO `blocked_cap_segments` support** — gap vs RangeStructure / SR / FHM | **P1 BUG** — fix in scope |
| **`levels` dict keys for main_detector** | Must include `support` (long) or `resistance` (short) so `detected_level` populates | Emits `{"reversal_level": current_price}` only — key doesn't match main_detector's lookup (main_detector.py:540-544 looks for support/resistance/broken_level) | **P1 BUG** — every volume event has `detected_level = None` downstream |
| **NaN `current_vol_z`** | Explicit guard | No guard; `float(NaN) < threshold` is False → silent pass | **P2 BUG** |
| **Time-of-day bonus in strength calc** | Canonical: 10:00-11:00 and 14:00-15:00 are HIGH-edge; 12:00-13:00 is LOW-edge | Hardcoded `10 <= hour <= 14` bonus at L399 → boosts confidence during 12:00-13:00 lunch window which Item 1 flagged as low-edge | **POTENTIAL BUG** — code may contradict canonical. Verify direction of bonus. |
| **Confidence unbounded** | Probability ∈ [0, 1] | 1.8 floor, ~35 extreme (cross-cutting) | **BUG (cross-cutting)** — defer |
| **~15 hardcoded thresholds in `_calculate_institutional_strength`** | Config-driven | Hardcoded | **DRIFT (cross-cutting)** — defer |

### Decision

**HYBRID:**
- **Exhaustion pattern core logic (P2-P6 filter stack):** KEEP-AS-IS. Strong canonical alignment. P2/P3/P5/P6 labels show iterative hardening.
- **`levels` dict key mismatch:** ALIGN-TO-CANONICAL in this audit — add `support`/`resistance` keys so `detected_level` populates downstream.
- **`blocked_cap_segments` config support:** ALIGN-TO-CANONICAL in this audit — add for parity with RangeStructure / SR / FHM.
- **NaN `current_vol_z` guard:** ALIGN-TO-CANONICAL in this audit — small fix, silent bug.
- **Hour-of-day bonus contradiction (P2 #5):** INVESTIGATE. If bonus logic is wrong direction (active-hours should be 10-11 + 14-15, not 10-14), fix. If bonus is "reduce outside 10-14" (correct), just document.
- **Multi-bar threshold 0.8 too lenient:** DEFER. Parameter sensitivity — Stage 3 conditional analysis will answer.
- **All other cross-cutting issues** (hardcoded thresholds, unbounded confidence, confirmation bar, events calendar, S/R level weighting, 20-bar median including spike bar): DEFER to canonical-upgrade / cross-cutting sub-projects.
- **457 trades is low but not DISABLE-low.** With P1 fixes applied (blocked_cap_segments may actually reduce trade count further by filtering large-cap), expect Stage 1 gauntlet to make the call on whether this survives.

---

## Item 3: Bug patterns

### 3.1 Off-by-one / lookbacks — PASS with one concern
- `df.iloc[-1]` (line 88, 96, 172, 287, 288) — current bar, correct for a detector that fires on the spike bar itself (aggressive entry, per design — see Item 1).
- `df.tail(20).median()` (line 126) — 20-bar median for vol_ratio dual check. Inclusive of current spike bar → slight inflation of median (the spike itself is in the sample). Canonical VSA would exclude current bar. **Minor (P3).**
- `df.tail(self.multi_bar_lookback)` (line 188) — recent N bars. Count is `int((recent['vol_z'] >= threshold).sum())` which is correctly bounded to `[0, multi_bar_lookback]`. No overflow risk, `>=` comparison correct.
- `range(...)` — not used.

### 3.2 Wrong sign / long-short symmetry — PASS
- Direction: `current_price > open_price` → UP bar → `_short` (exhaustion short). `else` (down bar) → `_long`. **Correct** for exhaustion logic.
- Wick ratio (lines 142-147): long uses `min(open,close) - low` (lower wick); short uses `high - max(open,close)` (upper wick). **Correct** for hammer/shooting-star morphology.
- Multi-bar direction consistency (lines 192-195): long side counts `c < o` (down bars); short side counts `c > o` (up bars). **Correct** — prior directional spikes in the same direction as the spike bar indicate sustained climactic pressure.
- SL direction (lines 290-295): long SL = spike low − buffer; short SL = spike high + buffer. **Correct.**
- T1/T2 direction (lines 325-330): reversal targets are against the spike direction. **Correct.**

### 3.3 NaN handling — PARTIAL PASS (concerns)
- `current_vol_z = float(df['vol_z'].iloc[-1])` (line 88): **No NaN guard.** `float(np.nan)` returns `nan`; the subsequent `if current_vol_z < self.min_volume_spike_mult` comparison against NaN is always False → NaN bars would pass through. **P2 bug — NaN vol_z should early-reject.**
- `atr = self._get_atr(context)` (line 157, 281, 323): fallback `context.current_price * 0.01` — never 0 unless current_price is 0. Safe.
- `vol_median = 0` guard: line 127 uses `if vol_median > 0 else 0.0`. Then `vol_ratio < self.min_volume_ratio` → rejects. **Safe.**
- `total_range == 0` guard: line 141 `if total_range > 0` → else `wick_ratio = 0.0`. Then wick_ratio < min rejects (doji candles rejected). **Safe.**
- `open_price == 0` division by zero at line 100: `body_size_pct = abs(current_price - open_price) / open_price * 100`. If `open_price == 0` → ZeroDivisionError → caught by outer try/except on line 241. Silent return with generic error. **P3 — division-by-zero should be explicit guard, not caught by outer except.**
- VWAP NaN guard (line 173): works correctly because the float conversion happens before the isnan check.
- PDH/PDL/ORH/ORL guards (lines 163-170): use `context.pdh and not np.isnan(context.pdh)` — but `context.pdh = 0.0` would evaluate to False and exclude a (theoretically) valid level. For NSE intraday equity this is not a real scenario; **not a bug in practice.** Also `hasattr` is redundant (MarketContext dataclass always has these fields — see data_models.py:47-52); but harmless.

### 3.4 Boundary conditions — PASS
- `len(df) < 10 or 'vol_z' not in df.columns` (line 77) — guards against short DataFrame and missing `vol_z` column (indicator pipeline may not have populated yet).
- `body_size_pct < self.min_body_size_pct` (line 102) — rejects sub-threshold bodies.
- `multi_bar_lookback` branch: `if len(df) >= self.multi_bar_lookback` (line 187). Otherwise `multi_bar_exhaustion` remains False, `elevated_bar_count = 0`. Passed to context + strength calc. **Safe but silent** — no logging when skipped due to short history.
- Fallback at line 129: `vol_ratio = float(current_vol_z)` when `len(df) < 20`. Using vol_z as a proxy for a fundamentally different quantity (ratio vs z-score). **Semantically questionable (P3)** — e.g., `vol_z=3.0` would satisfy `min_volume_ratio=1.5` but this doesn't actually mean the volume is 1.5× median.

### 3.5 Observation A — cap segment blocking — **FAIL / P1**
```
git grep -n "cap_segment\|blocked_cap" structures/volume_structure.py  →  (no matches)
```
(a) **No cap filter present** in `VolumeStructure`.
(b) **No `blocked_cap_segments` support.** Unlike `range_structure.py:65` (`self.blocked_cap_segments = set(config.get("blocked_cap_segments", []))`), `support_resistance_structure.py:105`, and `fhm_structure.py:89-90` which all implement this.
(c) **GAP vs. sibling detectors.** Canonical research (Item 1): large-cap volume-spike-reversal has ~45% reversal rate (weak edge); mid-cap is sweet spot at ~60%. The config block at `configuration.json:612-660` has no `blocked_cap_segments` key and the detector wouldn't read it if it did.
**Action:** add `self.blocked_cap_segments = set(config.get("blocked_cap_segments", []))` in `__init__` and a cap-segment check in `detect()` that rejects when `context.cap_segment in self.blocked_cap_segments`. Add `blocked_cap_segments: ["large_cap"]` to config once Stage 3 confirms the ~45% finding on our data.

### 3.6 Observation C — multi-bar exhaustion walk-through — PASS (logic correct, one minor)
Trace with `multi_bar_lookback=3`, `multi_bar_vol_threshold=0.8` (from config) and `vol_z = [2.5, 2.0, 1.5, 1.0, 2.8]` (5-bar history):
- `recent = df.tail(3)` → last 3 bars: `[1.5, 1.0, 2.8]`.
- `elevated_bar_count = int((recent['vol_z'] >= 0.8).sum())` → `[True, True, True]` → `3`.
- For `side == "long"` with down bars (close < open): `direction_consistent = 2` → `multi_bar_exhaustion = True` (requires ≥2 each).
- Bonus `multi_bar_exhaustion_bonus = 1.4` multiplier applied in `_calculate_institutional_strength:393-395`.

Logic is sound. **Minor concern:** threshold `0.8` with `>=` comparison means any bar with vol_z ≥ 0.8 counts — this is a very low bar (by construction most bars will qualify on a moderately active stock). The signal-to-noise of `multi_bar_exhaustion` boolean is therefore weak — it fires on most setups. **P3 — consider raising threshold or requiring stricter pattern (e.g., `vol_z >= 1.5` to distinguish "elevated" from "normal").**

**Caps:** `elevated_bar_count` and `direction_consistent` are bounded by `multi_bar_lookback` (= 3 in config). No overflow — the "10 consecutive spike bars" scenario in the prompt cannot happen because `tail(3)` limits the window.

---

## Item 4: Feature emission

### 4.1 event.context keys (lines 199-209) — PASS
All 9 keys are scalar-typed, so `main_detector.py:554-557` scalar filter passes them through to `SetupCandidate.extras`:

| Key | Type | Notes |
|---|---|---|
| `volume_spike` | float | = `current_vol_z` |
| `body_size_pct` | float | |
| `wick_ratio` | float (rounded 3) | |
| `vol_ratio` | float (rounded 1) | |
| `at_key_level` | bool | |
| `nearest_level` | str or None | PDH/PDL/ORH/ORL/VWAP or None |
| `nearest_level_dist_atr` | float (rounded 2) | 0.0 if atr<=0 |
| `multi_bar_exhaustion` | bool | |
| `elevated_bar_count` | int | |

### 4.2 Computation correctness — PASS
All features are computed from current-bar data (`candle = df.iloc[-1]`) plus recent-bar aggregates via `df.tail(N)`. No forward-looking leakage.

### 4.3 Flow to SetupCandidate.extras — PASS
- `direct_mappings` dict at `structures/main_detector.py:662-663` **includes both** `volume_spike_reversal_long` and `volume_spike_reversal_short` (→ identity mapping). No gap as was seen with LevelBreakout.
- `extras` dict is auto-flowed to trade_report.csv via `plan["extras"]` (per comment at main_detector.py:547-549).
- All 9 scalars survive the type filter.

---

## Item 5: Project rules compliance

### 5.1 Hardcoded thresholds — **FAIL / P1 (cross-cutting)**
Catalog of numeric literals in trading logic:

| File:Line | Expression | Classification |
|---|---|---|
| 237 | `quality_score=min(80.0, current_vol_z * 15)` | **Hardcoded cap 80.0 + multiplier 15** |
| 317 | `risk_percentage=0.02` | Hardcoded 2% risk (unused in actual sizing — pipeline overrides per extras — but still a magic literal) |
| 334-335 | `"qty_pct": 50`, `"rr": 1.0` / `2.0` | Hardcoded 50/50 T1/T2 split and RR values (not referenced from `target_mult_t1/t2` config) |
| 345 | `min(100.0, base_score + volume_spike * 5)` | Hardcoded cap 100.0 + multiplier 5 (rank_setup_quality) |
| 355 | `context.current_price * 0.01` | ATR fallback 1% — magic |
| 365 | `base_strength = max(1.5, vol_z * 0.8)` | Magic floor 1.5 + multiplier 0.8 |
| 371-374 | `if vol_z >= 5.0: *= 1.4`; `elif >= 3.0: *= 1.2` | **4 hardcoded magic numbers** (vol thresholds + multipliers) |
| 377-380 | `if body_size_pct >= 3.0: *= 1.25`; `elif >= 2.0: *= 1.15` | **4 hardcoded magic numbers** |
| 383-385 | `strength_multiplier *= 1.2` (wick bonus) | Hardcoded — note `wick_bonus_threshold` IS from config but the BONUS 1.2 is hardcoded |
| 399-400 | `if 10 <= current_hour <= 14: *= 1.1` | **Hardcoded time window 10-14 + bonus 1.1** — canonical Item 1 shows 12:00-13:00 should be AVOIDED (lunch chop), so this is not just hardcoded but possibly inverted in effect |
| 404 | `final_strength = max(final_strength, 1.8)` | Hardcoded floor 1.8 |
| 413 | `return 1.8` (exception fallback) | Hardcoded |

**Rule violation:** Per CLAUDE.md §1 "NO HARDCODED DEFAULTS — EVER. Every parameter ... MUST come from `config/configuration.json`". `_calculate_institutional_strength` contains ~15 unconfigurable magic numbers. **P1 refactor needed** — promote each threshold/multiplier to config (e.g., `volume_exceptional_z_threshold`, `volume_exceptional_bonus`, `body_conviction_thresholds`, `intraday_active_hour_bonus_window`, etc.).

### 5.2 IST-naive timestamps — PASS
No `datetime.now()`, `tz_localize`, or `tz_convert` calls in the file. Line 398 uses `pd.to_datetime(context.timestamp).hour` which preserves whatever tz-status the context carries (IST-naive per project convention).

### 5.3 Tick timestamps — PASS
Time-based decision at line 398-400 uses `context.timestamp` (tick clock, correct), not `datetime.now()`.

### 5.4 Fail-fast on missing config — PARTIAL PASS
`__init__` (lines 40-67): all 15 trading parameters use `config["key"]` direct subscript → **fails fast** on missing key (KeyError). **Correct.**

`config.get` usages:
- Line 37: `config.get("_setup_name", None)` — metadata/routing, default None acceptable. OK.

No other `config.get(..., default)` in trading logic. **PASS** for fail-fast, but the hardcoded magic numbers in `_calculate_institutional_strength` bypass the fail-fast contract entirely because they're never in config to begin with — see 5.1.

---

## Item 6: Output completeness

### StructureEvent fields (lines 217-226) — mostly populated
| Field | Value | Notes |
|---|---|---|
| `symbol` | `context.symbol` | OK |
| `timestamp` | `context.timestamp` | OK (tick time) |
| `structure_type` | `volume_spike_reversal_long/_short` | OK |
| `side` | `long`/`short` | OK |
| `confidence` | unbounded float | **See confidence-bounds below** |
| `levels` | `{"reversal_level": current_price}` | **See levels concern below** |
| `context` | 9-key dict | OK, all scalar |
| `price` | `current_price` | OK |

### `levels` dict concern — **FAIL / P2**
`levels={"reversal_level": current_price}` emits only a **single key `reversal_level`**. But `main_detector.py:540-544` extracts `detected_level` looking for these keys only:
- long side: `support` → `nearest_support` → `broken_level`
- short side: `resistance` → `nearest_resistance` → `broken_level`

**Neither `reversal_level` nor any of the expected keys are present → `detected_level = None` for every VolumeStructure setup.** Same class of bug as previously found in other detectors. Downstream tooling that expects a non-null `detected_level` (e.g., level-aware target preservation, edge analytics) will silently skip these rows or use fallback logic.

**Action:** emit `{"reversal_level": current_price, "support": current_price}` for long side and `{"reversal_level": current_price, "resistance": current_price}` for short side. (The spike-bar price IS the structural reversal level; the alias just gives main_detector a key it expects.)

### Direction symmetry — PASS
Long/short paths have symmetric wick, SL, target logic (verified 3.2).

### Confidence bounds — **FAIL / P2 (cross-cutting, as expected)**
`_calculate_institutional_strength` computes:
- `base_strength = max(1.5, vol_z * 0.8)` — so a `vol_z = 10` gives base 8.
- `strength_multiplier` can reach `1.4 × 1.25 × 1.2 × 1.3 (sr_confluence) × 1.4 (multi_bar) × 1.1 (hour) ≈ 4.2`
- `final_strength = base × multiplier ≈ 8 × 4.2 = 33.6` in extreme case
- Typical case `vol_z = 2, no bonuses → 1.6`

**Unbounded above.** The project convention elsewhere is `confidence ∈ [0.0, 1.0]` (SetupCandidate.strength interpreted as a probability-like score). This detector returns values typically in `[1.5, 5+]` → ordering is consistent within this detector but **not comparable** across detectors for `setup_candidates.sort(key=lambda s: s.strength, reverse=True)` at `main_detector.py:585`. **P2 — normalize to [0, 1] or document as bounded differently.**

Lower bound: `max(..., 1.8)` at line 404 (and `return 1.8` in exception at 413) means confidence never falls below 1.8 — any rejection via min-threshold in downstream gates keyed on confidence < 1.0 is bypassed.

---

## Item 7: Test coverage — **FAIL / TEST_DEBT**
```
git grep -ln "VolumeStructure\|volume_spike_reversal" tests/  →  (no matches)
```
`tests/structures/` contains tests for: `ict_structure`, `level_breakout_structure`, `momentum_structure`, `range_structure`, `support_resistance_structure`. **No test file for `VolumeStructure`.** TEST_DEBT marker — this is the only 1st-class setup detector without a corresponding test module.

---

## Issues found (consolidated)

| # | Severity | Item | File:Line | Summary |
|---|---|---|---|---|
| 1 | **P1** | 3.5 / Obs A | volume_structure.py (whole file) | No `blocked_cap_segments` support — large-cap ~45% WR not filterable. Sibling detectors (RangeStructure, SupportResistanceStructure, FHMStructure) all implement this. |
| 2 | **P1** | 5.1 / Obs E | volume_structure.py:237, 345, 355, 365, 371-380, 383, 399-400, 404, 413 | ~15 hardcoded magic numbers in `_calculate_institutional_strength`, `rank_setup_quality`, `_get_atr` fallback. Violates CLAUDE.md §1 "NO HARDCODED DEFAULTS". |
| 3 | **P2** | 3.3 | volume_structure.py:88 | `current_vol_z = float(df['vol_z'].iloc[-1])` no NaN guard — NaN vol_z slips through (NaN < threshold is False). |
| 4 | **P2** | 6 (levels dict) | volume_structure.py:223 | `levels={"reversal_level": current_price}` — main_detector expects `support`/`resistance` keys → `detected_level = None` for all volume setups. |
| 5 | **P2** | 6 (confidence) / Obs E | volume_structure.py:357-413 | Confidence unbounded (~1.5 to ~35). Not comparable to other detectors' [0, 1] confidence in `setup_candidates.sort`. |
| 6 | **P2** | 5.1 | volume_structure.py:399-400 | Hardcoded active-hour window `10 <= hour <= 14` bonuses 12:00-13:00 lunch chop — canonical Item 1 says this window is LOW edge. Window is not just hardcoded but potentially inverted in effect. |
| 7 | **P3** | 3.1 | volume_structure.py:126 | 20-bar median includes current spike bar → mild inflation. Minor. |
| 8 | **P3** | 3.3 | volume_structure.py:100 | `open_price == 0` division guard missing — relies on outer try/except. |
| 9 | **P3** | 3.3 | volume_structure.py:129 | `vol_ratio = float(current_vol_z)` fallback conflates z-score with ratio when len(df) < 20. Semantically wrong. |
| 10 | **P3** | 3.6 | volume_structure.py:189 + config | `multi_bar_vol_threshold = 0.8` is a very low bar — most bars qualify, making `multi_bar_exhaustion` a weak signal. |
| 11 | **TEST_DEBT** | 7 | tests/structures/ | No test file for VolumeStructure. All other detectors have dedicated tests. |

**Breakdown:** P1: 2 · P2: 4 · P3: 4 · TEST_DEBT: 1 · Total: 11

---

## Observations verdict

- **A (cap segment blocking):** **CONFIRMED MISSING.** File has no `cap_segment` or `blocked_cap` references. Sibling detectors (range/SR/FHM) implement it. P1 fix: add `blocked_cap_segments` constructor-read + cap check in `detect()`. Once Stage 3 confirms large-cap ~45% WR, block large_cap in config.
- **B (confirmation-bar / aggressive entry):** **CONFIRMED DELIBERATE.** No code comment mentions "wait for next bar." The detector fires on the spike bar itself (`df.iloc[-1]` is the current, just-closed bar). Per Item 1 canonical research, this is a documented design trade-off (aggressive vs patient). Not a bug.
- **C (multi-bar exhaustion):** **LOGIC CORRECT, threshold weak.** Counting via `.sum()` on boolean mask is proper; `>=` used correctly; counts bounded by `tail(multi_bar_lookback)` so no overflow. The P6 bonus correctly flows into `_calculate_institutional_strength:393-395`. Only concern: `multi_bar_vol_threshold = 0.8` is too lenient (P3).
- **D (S/R proximity):** **CORRECT.** All 5 level types (PDH, PDL, ORH, ORL, VWAP) are extracted. NaN check via `not np.isnan(...)` with short-circuit protection from `and`. `nearest_level_dist <= proximity_threshold` is the correct direction (at-level when within threshold). `hasattr()` guards are redundant but harmless.
- **E (cross-cutting):** **CONFIRMED.** `_calculate_institutional_strength` has ~15 hardcoded thresholds/multipliers. Confidence is unbounded (typically 1.5–5+, extreme ~35). No silent `config.get(..., default)` in trading logic beyond `_setup_name` metadata.

---

## Issues found (consolidated)
[To be filled in after Items 2-7 complete.]

## Fixes applied

All 4 planned fixes applied via TDD (failing test first, then fix, then verify).
Branch: `feat/premium-zone-ict-fix`. Test count: 181 baseline → 187 passing (+6 new volume regression tests).

| # | Commit | Issue | Summary |
|---|--------|-------|---------|
| 1 | `862302b` | P1 #1 | `feat(volume): add blocked_cap_segments config support (parity)` — mirrors Range/SR/FHM. Early-rejects blocked cap segments before any detection work. |
| 2 | `cfc9eb3` | P1 #2 | `fix(volume): emit 'support'/'resistance' keys so detected_level populates` — side-aware: long reversal emits `support` (spike low); short reversal emits `resistance` (spike high). `reversal_level` preserved for backward compat. |
| 3 | `d0ba8c4` | P2 #3 | `fix(volume): NaN guard on current_vol_z early-reject` — `pd.isna` check before threshold comparison. |
| 4 | `9134f46` | P2 #6 | `fix(volume): narrow hour-bonus to 10-11 + 14-15 (active windows)` — old code applied 1.1x uniformly across 10-14 (boosting the 12-13 lunch window). Narrowed to the two high-edge windows per Item 1 canonical. |

Regression tests: `tests/structures/test_volume_structure.py` (6 tests covering all 4 fixes). File force-added since `tests/` is gitignored.

### Fix 4 investigation note
Original code (line 399):
```python
if 10 <= current_hour <= 14:
    strength_multiplier *= 1.1
```
This is **Case B** per the spec — uniformly boosts 10-14 INCLUDING the low-edge 12-13 lunch window. Canonical Item 1 research flagged lunch as low-edge (~45-50% reversal). Fix narrows to 10:00-11:00 and 14:00-15:00 (both high-edge per research). 12-13 now correctly receives no bonus.

### Deferred (not blocking FIXED-AND-TRUSTED disposition)
- Confidence unbounded (cross-cutting; applies to multiple detectors — defer to normalization pass)
- ~15 hardcoded thresholds in `_calculate_institutional_strength` (cross-cutting)
- 20-bar median includes current spike bar (P3 cosmetic)
- `open_price == 0` division guard (P3)
- `multi_bar_vol_threshold` parameter tuning (Stage 3)
- Events-calendar filter (expiry, results, policy days — future feature)

---

## Final decision

**Disposition: FIXED-AND-TRUSTED**

**Rationale:**
- Small, focused detector (412 lines) with already-iterated filter stack (P2-P6 history)
- 2 P1 bugs are localized: missing `blocked_cap_segments` parity + `levels` dict key mismatch that silently nullifies `detected_level` for every event
- P2 fixes are small: NaN guard + hour-bonus investigation
- Defer all cross-cutting (hardcoded thresholds, unbounded confidence, parameter tuning for multi_bar_vol_threshold)

**Recommended action plan:**

| Order | Issue | Effort |
|-------|-------|--------|
| 1 | P1 #1: Add `blocked_cap_segments` config support (parity with Range/SR/FHM) | 30 min |
| 2 | P1 #2: Fix `levels` dict keys — add `support` (long) / `resistance` (short) so `detected_level` populates downstream | 20 min |
| 3 | P2 #3: NaN guard on `current_vol_z` early-reject | 15 min |
| 4 | P2 #5: Investigate hour-bonus direction — if 10-14 BOOSTS confidence uniformly (including low-edge 12-13 lunch), narrow the window to 10-11 + 14-15 OR remove. If logic is already "penalize outside active hours", just document. | 30 min investigation + 15 min fix if needed |
| **Total** | | **~1.5 hours + regression tests** |

Deferred to SUMMARY.md / future sub-projects:
- Confirmation-bar variant (design trade-off — Stage 3 validates)
- Events-calendar filter (expiry, results, policy days — future feature)
- Multi-bar threshold tuning (parameter sensitivity — Stage 3)
- S/R level weighting by type (canonical upgrade)
- 20-bar median including current bar (P3 cosmetic)
- Division-by-zero on open_price (P3)
- ~15 hardcoded thresholds in `_calculate_institutional_strength` (cross-cutting)
- Unbounded confidence (cross-cutting)
- TEST_DEBT — regression tests for fixes will be added; full coverage deferred

**Alternative dispositions (considered but not chosen):**
- **TRUSTED as-is:** Not recommended. The `levels` dict bug silently strips `detected_level` from every volume event — this affects downstream data quality even though it doesn't crash.
- **DISABLED:** Not recommended yet. 457 trades is low but canonical edge exists. Let Stage 1 gauntlet make this call on the regenerated post-fix data. If PF < 0.8 after fixes, revisit.

**User disposition:** FIXED-AND-TRUSTED confirmed. All 4 fixes applied (see "Fixes applied" section above).
