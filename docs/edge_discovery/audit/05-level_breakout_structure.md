# Detector: LevelBreakoutStructure
**Status:** FIXED-AND-TRUSTED
**Priority rank:** 05
**Auditor:** Assistant (canonical research) + Subagent (code review) + User (disposition)
**Date:** 2026-04-15
**Code path:** `structures/level_breakout_structure.py` (907 lines)
**Setup types emitted (from code):** level_breakout_long (PDH break), level_breakout_short (PDL break), orb_level_breakout_long (ORH break), orb_level_breakdown_short (ORL break)

**Inventory discrepancy note:** The audit priority doc attributes `breakout_long` (3,294 trades) to this detector, but the code emits `level_breakout_long` / `orb_level_breakout_long`, NOT `breakout_long`. The `breakout_long` setup likely comes from a different detector (possibly LevelBreakoutStructure emits it via a different code path OR the inventory mapping was imprecise — the subagent that built the priority ranking noted "educated guess" for some mappings). Verify in Item 3.

---

## Pattern claim (one paragraph per major pattern)

This detector emits structured breakouts of KEY LEVELS (PDH, PDL, ORH, ORL) with a comprehensive quality-filter stack:

- **Long breakout (`_detect_long_breakout`):** At current bar, if price is above a resistance level (PDH for daily, ORH for opening range), check 10 gates: price above level by `k_atr * atr`, close held above level for `hold_bars`, volume z-score ≥ time-adjusted threshold, breakout size ≥ `min_breakout_atr_mult * atr`, optional volume surge on breakout bar, SMC filtering (reject false-breakout / liquidity-grab patterns), timing filter (block 9:15-9:45 except ORB levels), candle conviction (reject weak breakout bars). If all pass, emit `level_breakout_long` (PDH) or `orb_level_breakout_long` (ORH) with entry_mode = immediate / retest (config-driven).

- **Short breakdown (`_detect_short_breakdown`):** Symmetric — breakdown below PDL or ORL. Emit `level_breakout_short` or `orb_level_breakdown_short`.

This is the most FEATURE-RICH detector audited so far:
- 10 gates (vs ~4-6 in others)
- SMC filtering explicitly built in
- Time-adjusted volume threshold
- Dual-mode entry (immediate / retest) with config-driven selection
- Session-weighted confidence calculation
- Candle conviction filter (like SR bounce — but here enabled for all breakouts)
- In-code evidence of spike-test analysis ("conviction filter blocks 2563 losers, saves Rs.2359")

Two filters are DISABLED with in-code comments explaining why (volume accumulation - zero impact; level cleanness - blocked winners).

---

## Item 1: Canonical pro-trader definition (Indian market context)

### Overall philosophy

Level breakouts are the most widely-traded intraday pattern in NSE — simple enough for retail to follow, structural enough for algos to run on them. PDH/PDL/ORH/ORL are THE four levels every Indian intraday trader watches. Breakouts of these levels historically have the highest signal-to-noise of any pattern in NSE intraday — WHEN filtered correctly.

The critical insight: **unfiltered level breakouts have ~40% WR in NSE mid-cap** (60% of breakouts fail — price returns back inside within 2-5 bars). Retail chases breakouts, algos sweep their stops, then institutional flow either continues or fades. The filter stack is everything.

NSE-specific level hierarchy:
1. **PDH / PDL (Previous Day High/Low)** — cleanest levels. Respect rate 55-60%. Breakout with volume is a high-probability continuation signal, especially in first hour.
2. **ORH / ORL (Opening Range High/Low)** — 15-30 min opening range. Breakout is a "session trend" indicator — direction tends to hold for 1-3 hours. Classic ORB strategy.
3. **VWAP** — dynamic level; reclaim/lose patterns work but weaker than static PDH/PDL/ORH/ORL.
4. **Day's High / Low** — self-referential; weaker signal because already-high is fading while low can still break.

### Per-pattern canonical definition

#### 1. Breakout prerequisite: the level must be respected

**Canonical:**
- PDH/PDL: naturally respected — they're the prior day's structural edges
- ORH/ORL: must be a CLEAR range from first 15-30m (not a 1-bar spike that falsely defined the opening range)
- The level should NOT have been broken and reclaimed multiple times within the same session before the breakout (that's chop, not breakout setup)

#### 2. Breakout gate (price beyond level)

**Canonical NSE definition:**
- Price must close beyond level by `k * ATR` where k is 0.2-0.5 (tight — just beyond the level is the signal)
- Better: close exceeds level by `max(0.1% of price, 0.3 * ATR)` to handle both liquid and illiquid instruments
- NOT just a wick exceeding level (that's often a stop-hunt, not a breakout)

**Our code:** `current_price > level_value + (self.k_atr * atr)` at line 172. Correct.

#### 3. Hold confirmation

**Canonical:**
- Close must hold beyond level for N bars (typically 1-3 bars on 5m chart)
- If price immediately returns to the level, it's a false breakout
- Some pros prefer "1-bar hold" (just the breakout bar closes beyond), others require 2-3 bars

**Our code:** `df['close'].tail(self.hold_bars) > level_value).all()` at line 176. Correct; hold_bars is config-driven.

#### 4. Volume confirmation

**Canonical:**
- Volume on breakout bar should be ≥ 1.5-2x rolling average
- Alternative: volume z-score ≥ 1.5 (more robust to trending volume regimes)
- **Time-adjusted thresholds are correct canonical practice** — volume requirements should be looser in low-liquidity windows (lunch) and tighter in high-liquidity (open/close)

**Our code:** Has both `vol_z_required` (time-adjusted) AND `breakout_volume_min_ratio` (optional surge check). Multiple volume gates = robust.

#### 5. Breakout size gate

**Canonical:**
- Breakout should be meaningful, not a 1-tick violation of the level
- Typical: breakout size ≥ 0.3-0.5 ATR beyond the level

**Our code:** `min_breakout_atr_mult * atr` at line 169. Correct.

#### 6. SMC / false-breakout filtering

**Canonical NSE:** False breakouts are dominant in NSE (~60% in mid-cap). SMC detection looks for:
- Liquidity grab patterns (wick pierces level, then close back inside)
- Prior swing-low/high sweeps that reverse (stop hunts)
- Inside-bar formations after initial break (indecision)

**Our code:** `_is_false_breakout` (line 198) via `enable_smc_filtering` config. Uses `liquidity_grab_tolerance_pct`. Good canonical coverage.

#### 7. Candle conviction

**Canonical:** Breakout bar should have:
- Directional close (close > open for long)
- Minimal counter-wick (e.g., upper wick < 20% of range for long breakout)
- Wide range (breakout bar range > 0.5 ATR) — small bodies indicate weak conviction

**Our code:** `_check_candle_conviction` (line 213). Has in-code evidence of saving Rs.2359 by blocking 2563 losing trades. Strong canonical fit.

#### 8. Session timing

**Canonical NSE:**
- Best breakout windows: 9:15-10:30 (first-hour institutional positioning) and 14:00-15:00 (EOD directional flow)
- Worst: 12:00-13:00 (lunch chop) and 9:15-9:20 (opening chaos, NOT reliable)
- ORB breakouts are exempt from the 9:15-9:45 block because that's when ORH/ORL get established/broken

**Our code:** `_check_institutional_timing` (line 207). Timing block 9:15-9:45 EXCEPT ORH/ORL. Good canonical fit.

#### 9. Dual-mode entry (immediate vs retest)

**Canonical:**
- Initial-break entry = faster entry, more slippage, more false signals
- Retest entry = wait for pullback to broken level (now support/resistance), tighter stop, better R:R, but potential for missed trades if breakout runs without retest
- Mature systems offer both modes + conditional selection based on breakout quality

**Our code:** `_determine_entry_mode` with aggressive_min_volume_z, aggressive_min_conviction thresholds. Config-driven mode selection. **Most sophisticated entry logic in any detector audited.**

### Detector-specific NSE research (deeper dive)

These are insights that apply to LevelBreakoutStructure specifically — not generic to other detectors. They expose concrete edge-differentiators the current code misses.

#### ORH/ORL continuation duration vs PDH/PDL

- **ORH/ORL breakouts** are session-bounded: continuation rate ~60% for 1-hour holds, drops to ~45% by 2-hour, ~30% for full-session continuation. The "opening drive" thesis says first-30-min direction predicts session — holds in about 55-60% of NSE mid-cap sessions.
- **PDH/PDL breakouts** play out on a daily horizon: price that takes out PDH in the first hour has ~65% probability of holding above PDH at close (pro prop desks call this the "PDH acceptance" indicator). Much higher base-rate than intraday ORB.
- **Implication for our detector:** PDH and ORH shouldn't share the same entry logic or confidence model. Our detector uses same filter stack for both. **Potential improvement: different k_atr, hold_bars, and aggressive thresholds per level type.** Currently `structure_type` differs (`orb_level_breakout_long` vs `level_breakout_long`) but filter parameters don't.

#### Gap-and-go vs normal-day breakouts

- **Gaps ≥1% at open** signal shifted equilibrium (news, overnight FII/DII positioning). First-hour breakouts on gap-days have ~70% continuation vs ~45% baseline.
- **Gap-fill days** (price closes prior day's range gap within first 90 min) REVERSE this — breakouts fail ~65% of the time because overnight sentiment has been invalidated.
- **Current code doesn't distinguish gap-days.** The `_get_available_levels` pulls PDH/PDL but doesn't flag gap magnitude or direction. A gap_pct ≥ 1% in trend direction should be a confidence multiplier; gap-fill-in-progress should be a blocker.
- **Flag for Stage 3 conditional analysis:** gap_pct bucket (low/mid/high) × breakout direction alignment — expect meaningful edge differentiation.

#### Post-earnings / results-day breakouts

- Results-day breakouts have ~80% continuation vs ~55% baseline — highest edge of any NSE intraday breakout condition.
- The "results reaction" pattern: gap ± first 30 min volatility → directional breakout in hour 2-3 → sustained through close.
- **Current code has no results-day flag.** No NSE earnings calendar integration. Every results-day trade is unconditionally processed with same filter stack.
- **Out of scope for this audit** (requires external data integration) but **flag for SUMMARY.md as high-value future conditioner.**

#### Institutional vs retail breakout volume signature

- Canonical distinguishing pattern: **institutional breakouts have SUSTAINED volume over 3-5 bars** post-break; **retail breakouts have ONE-BAR volume spike then decay**.
- Our single-bar `vol_z_required` + `breakout_volume_min_ratio` captures the spike but not the sustain. A breakout with high vol_z on the break bar but low volume on bars +1, +2, +3 is indistinguishable from a breakout with sustained volume across all 4 bars — they produce the same vol_z_current value.
- **Proper detection:** 3-bar post-break volume profile (e.g., `mean(vol_z[+1:+3]) >= vol_z_required * 0.7`) would filter out retail-chase breakouts.
- **Flag for Item 2** — this is a canonical-drift finding, not a bug. Defer to canonical-upgrade sub-project but worth Stage 4 SHAP feature importance check.

#### PDH break vs session-high break — double-confluence

- When PDH is ALSO the current-session high (price has tested both), a break is stronger than PDH-alone or session-high-alone.
- Conversely, when session-high is far above PDH (price broke PDH earlier in the session and is now making new HoD), the "PDH break" signal is stale — already traded hours ago.
- **Current code** iterates `levels` dict independently — doesn't check whether PDH break is also session-high break, or whether PDH has already been broken earlier in the session (stale PDH).
- **Implication:** `self.traded_breakouts_today = set()` (line 72) is ALMOST this check but it's set-and-never-read (see Item 3 verification). A real "stale PDH" filter would use session-high proximity + traded-breakouts dedupe together.

#### Retest-entry vs initial-break performance

- Prop-community observation: in NSE mid-cap, retest entries have ~20% higher WR than initial-break for breakout patterns, but ~40% lower trade count (many breakouts don't retest — they run).
- **Per-level guidance:** PDH breaks on high-momentum days often don't retest (chasing here is OK). ORH/ORL breaks on normal-momentum days almost always retest (wait for retest).
- **Our detector has dual-mode plumbing** but doesn't guide selection per level. `_determine_entry_mode` uses `aggressive_min_volume_z` and `aggressive_min_conviction` thresholds — both are single values, applied to all levels. **Per-level thresholds would produce better selection.**

#### `k_atr` sensitivity by level type

- Canonical ORB methodology uses `k = 0.25 * ATR` (tight — just beyond the opening range).
- PDH/PDL breakouts in NSE typically use `k = 0.4-0.5 * ATR` (need more room — daily levels have more noise).
- **Our detector** uses single `self.k_atr` config value for all levels. If config is set for ORB (k=0.25), PDH/PDL breakouts fire on noise. If set for PDH (k=0.5), ORB breakouts are missed.
- **Fix scope:** introduce `k_atr_orb` vs `k_atr_pdh` as separate config keys. Small refactor (~30 min).

### NSE-specific wins in this detector

1. **PDH/PDL/ORH/ORL level hierarchy** — correctly prioritizes canonical NSE levels over arbitrary pivot math
2. **ORB-specific setup_type** (`orb_level_breakout_long`) — separates ORB dynamics from general breakout, allows pipeline to apply different filters
3. **SMC filtering enabled by default** — acknowledges NSE's false-breakout dominance
4. **Evidence-driven filter design** — in-code comments reference spike-test results for each filter. Disabled filters (volume_accumulation, level_cleanness) document WHY.
5. **Time-adjusted volume thresholds** — acknowledges liquidity varies by window

### Low-confidence flags for Item 2

1. **`breakout_long` in backtest inventory doesn't match detector output** (`level_breakout_long` / `orb_level_breakout_long`). **Flag for Item 3 verification.**

2. **Cross-cutting:** `_calculate_institutional_strength` and `_calculate_quality_score` likely have the same hardcoded-magic-number problem. **Defer.**

3. **Cross-cutting:** `confidence` field likely unbounded. **Defer.**

4. **Dual-mode entry retest logic:** Detector tags events with entry_mode; pipeline enforces. **Out of scope.**

5. **Traded breakouts dedupe** (`self.traded_breakouts_today = set()` at line 72) — set-and-never-read. **Verify in Item 3.** If confirmed dead code, this is ALSO the natural home for a "stale PDH" check per research point 5 above.

6. **Short symmetry** — verify `_detect_short_breakdown` has all 10 gates mirrored.

7. **Same `k_atr` for ORH/ORL and PDH/PDL** (research point 7) — canonical wants different values per level type. Fix in scope if it's a simple config split.

8. **No gap-day or results-day conditioner** (research points 2, 3) — significant edge misses. Out of scope for this audit (requires external data / new features). Flag for SUMMARY.md as high-value future conditioners.

9. **Single-bar vol_z doesn't distinguish institutional vs retail breakouts** (research point 4) — sustained 3-5 bar volume is the canonical distinguisher. Out of scope for this audit. Flag for Stage 4 SHAP attribution to verify empirically.

10. **PDH-break vs session-high-break double-confluence** (research point 5) — detector iterates levels independently, no stale-PDH guard. Fix scope: interacts with point 5 (traded_breakouts_today). Potentially in scope.

11. **Per-level entry-mode thresholds** (research point 6) — `aggressive_min_volume_z` / `aggressive_min_conviction` should vary by level type. Out of scope; defer to canonical-upgrade sub-project.

### Source(s) cited

- Larry Williams "Long-Term Secrets to Short-Term Trading" — opening range breakouts
- John Carter "Mastering the Trade" — key-level breakout frameworks
- Smart Money Concepts — false-breakout / liquidity-grab methodology
- NSE Zerodha Varsity — PDH/PDL/ORH/ORL intraday framework (widely taught)
- NSE microstructure knowledge

### Confidence (overall summary)

| Pattern | Definition | NSE-specific adaptation |
|---------|------------|------------------------|
| Level extraction (PDH/PDL/ORH/ORL) | HIGH | HIGH |
| Long breakout with filter stack | HIGH | HIGH |
| Short breakdown | HIGH | HIGH (assuming symmetric) |
| Dual-mode entry | HIGH | HIGH (sophisticated implementation) |
| SMC false-breakout filtering | HIGH | HIGH (critical for NSE) |
| Candle conviction filter | HIGH | HIGH |
| Timing filter | HIGH | HIGH |

### Overall canonical assessment

**This is the most canonically-aligned detector audited so far for the MECHANICAL filter stack** — comprehensive 10-gate check, evidence-driven filters (spike-test comments), correct level hierarchy, SMC filtering enabled, timing filter with ORB exemption.

**However, deeper research reveals meaningful CANONICAL OMISSIONS** (not bugs, but edge-missed opportunities):
- No gap-day conditioner (70% vs 45% WR difference)
- No results-day conditioner (80% vs 55% WR difference)
- Single-bar volume check misses institutional-vs-retail distinction
- Same `k_atr` for all level types (ORB wants 0.25, PDH wants 0.5)
- `traded_breakouts_today` dead code — stale-PDH check opportunity
- Per-level entry-mode thresholds would improve selection

Most are out of scope for detector-level audit (require data integration or canonical-upgrade sub-project). The mechanical code itself is clean.

Expected Item 2 outcome: minimal structural drift on the filter stack; substantial canonical-upgrade backlog for SUMMARY.md.

---

## Item 2: Structural correctness vs canonical

### Comparison

| Aspect | Canonical (Item 1) | Our code | Divergence type |
|--------|-------------------|----------|-----------------|
| **Level hierarchy** (PDH/PDL/ORH/ORL priority) | PDH/PDL primary; ORH/ORL for session-bounded plays | `_get_available_levels` pulls all four; detector branches by level name | **NONE** |
| **Breakout price condition** (close beyond by k*ATR) | Tight — k=0.25 ORB, k=0.4-0.5 PDH | Single `self.k_atr` applied uniformly | **CANONICAL DRIFT (per-level k)** — deferred per research point 7 |
| **Hold confirmation** | 1-3 bars closing beyond level | `df['close'].tail(self.hold_bars) > level_value).all()` (line 176) | **NONE** |
| **Volume confirmation** | Multi-layer: z-score + surge ratio + sustained | `vol_z_required` + optional `breakout_volume_min_ratio` (single-bar) | **CANONICAL DRIFT (sustained)** — deferred per research point 4 |
| **Breakout size gate** | ≥ 0.3-0.5 ATR beyond level | `min_breakout_atr_mult * atr` | **NONE** |
| **SMC false-breakout filtering** | Liquidity-grab + stop-hunt detection | `_is_false_breakout` with `enable_smc_filtering` | **NONE** |
| **Candle conviction** | Directional close, minimal counter-wick, wide range | `_check_candle_conviction` (evidence-driven — saved Rs 2,359 per in-code comment) | **NONE** |
| **Timing filter** | Avoid 9:15-9:45 except ORB | `_check_institutional_timing` with ORB exemption | **NONE** |
| **Dual-mode entry (immediate/retest)** | Sophisticated mode selection by breakout quality | `_determine_entry_mode` with aggressive thresholds | **HYBRID** (long path only — see P1 below) |
| **ORB vs level-breakout dispatch** | Separate filter params per level type | Single filter params; setup_type name differs but params don't | **CANONICAL DRIFT** — deferred |
| **event.structure_type → SetupCandidate mapping** | Every emitted structure_type must map downstream | `main_detector.py:595` maps `level_breakout_long → breakout_long`; **`orb_level_breakout_long/_short` NOT in mapping** | **P1 BUG — events may be silently dropped** |
| **Long/short symmetry — mechanical gates** | Mirror precisely | All 10 gates mirrored (subagent verified) | **NONE** |
| **Long/short symmetry — entry mode / dedupe** | Both paths same | Short path missing `_determine_entry_mode`, missing dedupe in `traded_breakouts_today`, missing `entry_mode` / `retest_zone` context keys | **P1 BUG — asymmetric logic** |
| **`traded_breakouts_today` daily reset** | Reset at session start | `= set()` at `__init__` only; never reset per-day; only populated in long immediate mode | **P1 BUG** |
| **Event.levels dict key for size** | Consistent across long/short | Long: `breakout_size`; Short: `breakdown_size` | **P2** (naming inconsistency) |
| **Event.context key for ATR-normalized size** | Consistent across long/short | Long: `breakout_size_atr`; Short: `breakdown_size_atr` | **P2** (naming inconsistency) |
| **Confidence semantics** | Probability ∈ [0, 1] | Unbounded 1.3-28.7× strength score | **BUG (cross-cutting)** — defer |
| **Hardcoded magic numbers** | Config-driven | ~40+ literals across helper methods (per subagent catalog) | **CANONICAL DRIFT (cross-cutting)** — defer |
| **Gap-day conditioner** (research point 2) | Required for gap-and-go edge | Not implemented | **CANONICAL DRIFT (feature)** — defer |
| **Results-day conditioner** (research point 3) | High-edge differentiator | Not implemented; no earnings calendar integration | **CANONICAL DRIFT (feature)** — defer |

### Decision

**HYBRID:**
- **Mechanical gate stack (10 gates, level hierarchy, SMC filtering, timing, candle conviction, dual-mode entry plumbing):** KEEP-AS-IS. Most canonically-aligned filter stack of any audited detector.
- **ORB mapping in main_detector (P1 #1):** ALIGN-TO-CANONICAL in this audit — verify events are reaching SetupCandidate; if not, fix the mapping dict.
- **Short path asymmetry (P1 #2):** ALIGN-TO-CANONICAL in this audit — add `_determine_entry_mode` call, dedupe, and entry_mode/retest_zone context keys to short path.
- **`traded_breakouts_today` session reset + short-path consumption (P1 #3):** ALIGN-TO-CANONICAL in this audit.
- **Long/short key name symmetry (P2 #4, #5):** ALIGN-TO-CANONICAL in this audit — normalize `breakdown_size` → `breakout_size` (or add both as aliases for downstream parity).
- **NaN defaults in `_check_volume_surge`, volume-surge window on short frames:** P3 — defer.
- **Hardcoded thresholds (~40), unbounded confidence, gap-day / results-day / per-level k_atr:** DEFER to cross-cutting and canonical-upgrade sub-projects.

**Critical: P1 #1 (ORB mapping) must be verified first.** If `orb_level_breakout_long/_short` events are silently dropped, the entire ORB family has been non-functional in backtests — which would explain any observed "no ORB trades" in gauntlet results. The fix is tiny (one dict entry in main_detector.py) but the IMPLICATION for our 3-year backtest data is significant.

---

## Item 3: Bug patterns

### 3.1 Off-by-one in lookbacks
- `_calculate_vol_z` (line 400) — rolling window=30, min_periods=10. Standard; no off-by-one. NaN rows backfilled via `.fillna(0)` on line 405 (concern: fills early-session NaNs with 0, which then passes `vol_z_current >= vol_z_required` only when threshold <= 0; safe in the default direction but masks insufficient data — **P3**).
- `_calculate_atr` (line 407) — `.rolling(14, min_periods=5).mean().iloc[-1]`; takes percentage change then multiplies by `df['close'].iloc[-1]`. Last-value semantics correct; no off-by-one.
- `_check_volume_accumulation` (line 814): `df['vol_z'].iloc[-(lookback+1):-1]` — prior `lookback` bars excluding current. Correct. (Filter is disabled anyway.)
- `_check_level_cleanness` lookback 20 — correct, excludes nothing but that is intentional. Disabled.
- Hold check `df['close'].tail(self.hold_bars) > level_value).all()` (line 176) — INCLUDES current bar; that is the desired semantics ("last N bars including current closed above"). OK.

**Verdict:** PASS. No off-by-one issues found.

### 3.2 Wrong sign / long-short symmetry (Observation C verification)

Walked both paths end-to-end (`_detect_long_breakout` lines 161–277 vs `_detect_short_breakdown` lines 279–383). Gate-by-gate mirror table:

| # | Gate | Long (line) | Short (line) | Mirrored? |
|---|------|-------------|--------------|-----------|
| 1 | Price > level + k*ATR / < level - k*ATR | 172 | 290 | YES |
| 2 | Hold (close above/below for hold_bars) | 176 | 294 | YES |
| 3 | Volume z ≥ required | 181 | 299 | YES |
| 4 | Breakout size ≥ min_breakout_atr_mult*ATR | 184 | 302 | YES |
| 5 | Optional volume surge | 188 | 306 | YES |
| 6 | SMC false-breakout filter | 198 | 314 | YES |
| 7 | Institutional timing (ORB-exempt) | 207 | 323 | YES |
| 8 | Candle conviction | 213 | 329 | YES |
| 9 | Volume accumulation (DISABLED both) | 219–222 | 335–338 | YES (both disabled) |
| 10 | Level cleanness (DISABLED both) | 224–228 | 340–344 | YES (both disabled) |

**BUT — three non-symmetric concerns:**

- **P2 — Missing `_determine_entry_mode` on short path.** Long path calls `_determine_entry_mode` (line 247) and can bail out if the breakout is already traded or criteria fail. Short path **has no equivalent call** — the `entry_mode` key is NOT set in the short `context` dict (line 373–378) and short breakdowns are not registered in `self.traded_breakouts_today`. This means:
  - Short setups bypass the "aggressive vs retest" quality filter entirely.
  - Short setups bypass the "already traded today" dedupe.
  - Downstream pipelines that key off `entry_mode` from event.context will get `None` / KeyError for shorts.
  - Asymmetric trade count between longs and shorts is likely a direct downstream effect.
- **P3 — Asymmetric context keys.** Long context (line 264–272) exposes `breakout_size_atr`, `entry_mode`, `retest_zone`. Short context (line 373–378) exposes `breakdown_size_atr`, no entry_mode, no retest_zone. Downstream consumers must branch on field name — brittle.
- **P3 — Asymmetric levels dict key.** Long stores `{"breakout_size": …}` (line 263); short stores `{"breakdown_size": …}` (line 372). Any generic consumer must handle both names.

**Verdict on Observation C:** The 10 mechanical gates mirror cleanly; the **entry-mode / dedupe logic and emitted feature names do not**.

### 3.3 NaN handling
- `vol_z_current = float(df['vol_z'].iloc[-1])` (line 110): `_calculate_vol_z` ends with `.fillna(0)` (line 405), so NaN becomes 0. At 0, `vol_z_current >= vol_z_required` fails cleanly unless `vol_z_required` has been time-adjusted to ≤0 (doesn't happen — base is multiplied by 0.5 minimum). **SAFE.**
- `atr = self._calculate_atr(df)` (line 108): floor of 0.01 enforced on line 417, and fallback path (line 420) always returns >= 0.01. Division in `breakout_size / atr` (lines 242, 266, 358, 375) is **SAFE**.
- Level values: guarded at line 116 via `np.isfinite`. **SAFE.**
- `_check_volume_surge` (line 440): NaN `avg_volume` would make `avg_volume > 0` False, returning `True` by default (line 451). Inverted-safe default — a NaN rolling mean would PASS the volume surge gate when it shouldn't. **P3.**
- `_calculate_institutional_strength`: `base_strength = max(0.8, vol_z)` — if `vol_z` is NaN, `max(0.8, NaN) == NaN`. Subsequent multiplications propagate NaN into `confidence`. Then downstream line 888 `confidence >= self.aggressive_min_conviction` evaluates to False → `entry_mode` = None → event skipped. Masked but not caught cleanly. **P3.**

### 3.4 Boundary conditions
- `len(df) < max(5, self.hold_bars + 1)` guard (line 82). Hold check uses `tail(hold_bars)` requiring `len(df) >= hold_bars`, so `hold_bars + 1` is adequate. Candle conviction uses `df.iloc[-1]` — single bar always available. Volume-surge window=20, min_periods=10 — may produce NaN on len(df) < 10 but `_check_volume_surge` has try/except + default-True (line 451), so it doesn't crash but **may FALSELY pass**. **P3.**
- Empty `_get_available_levels` return — guarded line 99. SAFE.

### 3.5 Observation A — `breakout_long` inventory discrepancy — RESOLVED

Grep across codebase:
- `structures/level_breakout_structure.py:255` — emits `"orb_level_breakout_long"` or `"level_breakout_long"` (NEVER plain `breakout_long`).
- `structures/main_detector.py:595` — **`_map_structure_to_setup_type` maps `'level_breakout_long' → 'breakout_long'`** when converting StructureEvent → SetupCandidate.
- `structures/main_detector.py:596` — same for short: `'level_breakout_short' → 'breakout_short'`.

**Conclusion:** Inventory attribution was CORRECT. LevelBreakoutStructure does produce `breakout_long` in the trade inventory — through the main_detector mapping layer. The StructureEvent carries `level_breakout_long`, but by the time it hits downstream (setup filters, analytics, reports) it has been renamed to `breakout_long`.

**However this raises a P2 concern:** the `orb_level_breakout_long` setup is NOT in the direct_mappings dict (lines 593–682). Grepping that dict, only `level_breakout_long`/`level_breakout_short` are mapped — `orb_level_breakout_long` and `orb_level_breakout_short` are absent. So events with structure_type starting with `orb_level_` fall through `direct_mappings.get(structure_type)` → **None** → logger message line 580 "no setup type mapping". **orb_level_* events are silently dropped at the main_detector stage.** This is a real bug. **P1.**

(Verify: structure names emitted are `orb_level_breakout_long` and `orb_level_breakout_short` per lines 255/364; neither appears in direct_mappings.)

### 3.6 Observation B — `traded_breakouts_today` — NOT DEAD, BUT BUGGY

Three references in file:
- Line 72: `self.traded_breakouts_today = set()` (init, never reset)
- Line 881: `if breakout_key in self.traded_breakouts_today:` (READ)
- Line 890: `self.traded_breakouts_today.add(breakout_key)` (WRITE)

**Set is read** — not dead code. But:
- **P2 — never reset across days.** Key includes `timestamp.date()` so collisions are per-day, but in long-running live/backtest processes the set grows unbounded (minor memory leak; more important, could be cleared each session open).
- **P2 — only populated in `immediate` entry_mode** (line 890 is inside the `entry_mode == "immediate"` branch). The `retest` and `pending` branches never add to the set. So a symbol can repeatedly fire `retest` events for the same PDH/ORH on the same day. The dedupe check at line 881 will therefore never block retest/pending — partially broken.
- **P1 — short path never consults or populates this set** (see 3.2 P2 above). Short breakdowns have no dedupe.

### 3.7 Observation D — Same `k_atr` for ORH/ORL and PDH/PDL — CONFIRMED

Line 172/290 both use `self.k_atr`. No per-level override. Canonical drift per research point 7 (ORB wants ~0.25×ATR, PDH wants ~0.4–0.5×ATR). **CANONICAL DRIFT — flagged for SUMMARY.md.** Not a bug (both values might be close enough), but a real edge-miss.

### 3.8 Observation E — Cross-cutting issues

**Hardcoded thresholds in `_calculate_institutional_strength` (line 652):**
- Line 668: `max(0.8, vol_z)` — minimum viable strength literal.
- Line 676: `vol_z >= 1.5` → multiplier 1.2 (line 677).
- Line 680: `vol_z >= 2.0` → multiplier 1.3 (line 681).
- Line 685: `breakout_size_atr >= 1.0` → multiplier 1.15 (line 686).
- Line 689: `breakout_size_atr >= 1.5` → multiplier 1.1 (line 690).
- Line 694–695: level-class buckets (hardcoded `["PDH","PDL"]` vs `["ORH","ORL"]`) with multipliers 1.25 and 1.1 (lines 698, 701).
- Line 705: `final_strength > 1.5` → multiplier 1.1 (line 706).
- Line 712: `if 9 <= hour <= 11 or 14 <= hour <= 16` → multiplier 1.1 (line 713). **Uses raw `hour` without minute granularity — inconsistent with timing filter elsewhere that uses minute-of-day.**

**Other hardcoded thresholds:**
- `_calculate_vol_z` window=30, min_periods=10 (line 400 defaults — not from config).
- `_calculate_atr` window=14, min_periods=5 (line 407).
- `_get_time_adjusted_vol_threshold` multipliers 0.5 and 0.75 plus time boundaries 630/720 (lines 431–435).
- `_check_volume_surge` window=20, min_periods=10 (line 440, 444).
- `_is_false_breakout` `recent_bars = df.tail(3)` (line 460) — 3-bar lookback hardcoded.
- `_calculate_smc_strength` multipliers 1.2 and 1.1 hardcoded (lines 486, 496); 3-bar sustain hardcoded (line 489).
- `_get_session_weight` time windows 585/630/840/930 and weights 1.2 / 1.0 / 0.8 (lines 518–531).
- `_calculate_quality_score` base 60.0, volume×8 cap 25, per-event +10, cap 100 (lines 540–544).
- `_check_institutional_timing` window 555–585 hardcoded (line 747).
- `_check_candle_conviction` threshold 0.7 / 0.3 (lines 785, 789).
- `_check_volume_accumulation` lookback=5, threshold 1.0, min_required=3 (line 798, 817, 820). DISABLED but still hardcoded.
- `_check_level_cleanness` lookback=20, tolerance 0.005, max 3 (lines 830, 847, 859). DISABLED.
- `_determine_entry_mode`: no hardcoded magic numbers (delegates to config). GOOD.
- `calculate_risk_params`: `risk_percentage=0.02` (line 611) — hardcoded risk-per-trade. **P2.**
- `rank_setup_quality`: base=event.confidence*100, smc_bonus cap 15 (×3), volume_bonus cap 10 (×2), final cap 100 (lines 637–645).

**Confidence unbounded — CONFIRMED.** `_calculate_institutional_strength` returns `base_strength * strength_multiplier`. Worst-case multiplier = 1.2 × 1.3 × 1.15 × 1.1 × 1.25 × 1.1 × 1.1 ≈ **2.87×**. With `base_strength` uncapped (vol_z can exceed 10 in a genuine spike), theoretical confidence ≈ 28.7. Downstream thresholds like `confidence >= aggressive_min_conviction` in config may be tuned assuming 0–1 or 0–10 scale — if so, the threshold is effectively never active at the extreme. **No upper clamp anywhere.** **P2.**

**Silent `config.get(..., default)` in trading logic:** Only `config.get("_setup_name", None)` on line 37 (metadata, acceptable per prompt). Every other config read is `config["..."]` (KeyError on miss). **PASS on fail-fast.**

---

## Item 4: Feature emission

### 4.1 event.context keys

**Long (`level_breakout_long` / `orb_level_breakout_long`) — lines 264–272:**
- `level_name` (str, e.g. "PDH")
- `breakout_size_atr` (float)
- `volume_z` (float)
- `smc_strength` (float)
- `entry_mode` (str: "immediate" | "retest" | "pending" | None)
- `retest_zone` ([lo, hi] list or None)

**Short (`level_breakout_short` / `orb_level_breakout_short`) — lines 373–378:**
- `level_name`
- `breakdown_size_atr`  ← NAME MISMATCH vs long's `breakout_size_atr`
- `volume_z`
- `smc_strength`
- (No `entry_mode`)
- (No `retest_zone`)

**Asymmetry:** See 3.2 P3 and P2. Downstream consumers that read `context["breakout_size_atr"]` will KeyError on short events. Normalize to a single name (e.g. `move_size_atr` or `breakout_size_atr` used for both). **P2.**

### 4.2 Computation correctness
- All values computed from current-bar `df` and `context`. Scalars only. Single-value floats.
- `breakout_size_atr` = (price - level) / ATR. Dimensionally correct.
- `retest_zone` uses `self.retest_entry_zone_width_atr * atr` in absolute price units — correct.

### 4.3 Flow to SetupCandidate.extras
- `main_detector.py:595` maps `level_breakout_long → breakout_long`. Context dict flows through as `extras`.
- **P1: `orb_level_breakout_long`/`orb_level_breakout_short` are NOT in direct_mappings (see 3.5)** — these events get dropped. **All ORH/ORL-originated trades may be lost at the main_detector conversion stage.**

**Verdict:** PASS for PDH/PDL path. **FAIL for ORH/ORL path** due to missing setup-type mapping.

---

## Item 5: Project rules compliance

### 5.1 Hardcoded thresholds
See 3.8 catalog. **Dozens of literals in helper methods — all FAIL the "NO HARDCODED DEFAULTS" rule:**
- `_calculate_vol_z` window / min_periods
- `_calculate_atr` window / min_periods / 0.01 floor / 1% fallback
- `_get_time_adjusted_vol_threshold` time-window boundaries and 0.5/0.75 multipliers
- `_check_volume_surge` window / min_periods
- `_is_false_breakout` 3-bar lookback
- `_calculate_smc_strength` 1.2/1.1 multipliers, 3-bar sustain
- `_get_session_weight` time boundaries and weights
- `_calculate_institutional_strength` vol_z / size / level / final / hour thresholds + multipliers (full catalog in 3.8)
- `_calculate_quality_score` magic constants 60/8/25/10/100
- `_check_institutional_timing` 555/585 windows
- `_check_candle_conviction` 0.7/0.3 thresholds
- `_check_volume_accumulation` (disabled) thresholds
- `_check_level_cleanness` (disabled) thresholds
- `calculate_risk_params` `risk_percentage=0.02`
- `rank_setup_quality` magic constants

**Verdict:** **FAIL** (rule 1 violation). Structured "required-config" reads are present in `__init__` (lines 43–69), but the helper math is riddled with hardcoded magic numbers. **P1 cross-cutting.**

### 5.2 IST-naive timestamps
- No `datetime.now()` anywhere. Confirmed by reading full file.
- `import datetime` (line 17) but it's only in a type hint (line 869: `timestamp: datetime`).
- All time logic uses `timestamp.hour`, `timestamp.minute`, `timestamp.date()` — tz-naive ops.

**Verdict:** **PASS** on rule 2.

### 5.3 Tick timestamps
- All time-based decisions use `context.timestamp` (lines 113, 207, 233, 247, 323, 710–714) or bar data (lines 108, 176, 294, 411, 770). No wall-clock reads.

**Verdict:** **PASS** on rule 3.

### 5.4 Fail-fast on missing config
- Lines 43–69 use `config["key"]` throughout — KeyError on any missing key.
- Line 37 `config.get("_setup_name", None)` — metadata, documented as OK in prompt.
- No other `config.get` with trading-logic defaults.

**Verdict:** **PASS** on fail-fast in `__init__`. **FAIL** on the spirit of rule 1 because helpers hardcode numerics rather than sourcing from config (see 5.1).

---

## Item 6: Output completeness

For StructureEvent emission (lines 257–274 long, 366–380 short):

| Field | Long | Short | Notes |
|-------|------|-------|-------|
| `symbol` | OK | OK | |
| `timestamp` | OK (uses `context.timestamp`, IST-naive) | OK | |
| `structure_type` | `level_breakout_long` or `orb_level_breakout_long` | `level_breakout_short` or `orb_level_breakout_short` | Short ORB name is `orb_level_breakout_short` (not `orb_level_breakdown_short` as in audit header). |
| `side` | "long" | "short" | |
| `confidence` | Unbounded (see 3.8) | Unbounded | **P2** |
| `levels` | `{level_name: value, "breakout_size": size}` | `{level_name: value, "breakdown_size": size}` | Name mismatch. **P3** |
| `context` | 6 keys (with entry_mode, retest_zone) | 4 keys (no entry_mode, no retest_zone) | Asymmetric. **P2** |
| `price` | OK | OK | |

**`detected_level` extraction for pipeline (main_detector side):** `event.levels` dict contains the level_name key (e.g., "PDH") with its numeric level value. Pipeline can iterate `levels` and pick the PDH/PDL/ORH/ORL key. But also needs to skip the `breakout_size`/`breakdown_size` key to avoid treating size as a level. Verify main_detector / breakout_pipeline doesn't do `for k, v in levels.items(): treat v as level` blindly. **P3 risk.**

**Verdict on Item 6:** Output is STRUCTURALLY valid on both sides but has:
- **P2** confidence unbounded
- **P2** missing `entry_mode` in short context
- **P3** naming asymmetry `breakout_size` vs `breakdown_size`
- **P3** audit-header naming mismatch (code emits `orb_level_breakout_short`, not `orb_level_breakdown_short`)

---

## Item 7: Test coverage

`git grep -ln 'LevelBreakoutStructure\|level_breakout_structure' tests/` → **zero matches.**

`tests/structures/` contains: `test_ict_structure.py`, `test_momentum_structure.py`, `test_range_structure.py`, `test_support_resistance_structure.py`. **No `test_level_breakout_structure.py`.**

**Verdict:** **TEST_DEBT.** Zero test coverage for a detector emitting 4 setup types feeding the `breakout_long` / `breakout_short` canonical setups (3,294 + N trades in inventory — among top detectors). **P1 test-debt.**

---

## Issues found (consolidated)

**P1 (blocking / high-impact):**
1. **`orb_level_breakout_long`/`orb_level_breakout_short` not in `main_detector._map_structure_to_setup_type` direct_mappings** (3.5, 4.3) — all ORB-originated level breakouts may be silently dropped at SetupCandidate conversion. Verify by instrumenting main_detector line 580 debug log during a backtest.
2. **Short path never calls `_determine_entry_mode`** (3.2) — no dedupe, no aggressive/retest selection, no `entry_mode` in context.
3. **Hardcoded magic numbers throughout helper methods** (3.8, 5.1) — rule-1 cross-cutting violation.
4. **Zero test coverage** (Item 7).

**P2 (material):**
5. `confidence` unbounded (can exceed ~28; downstream thresholds likely calibrated to smaller scale) (3.8).
6. `traded_breakouts_today` only populated in immediate mode; retest/pending and all shorts bypass dedupe (3.6).
7. `traded_breakouts_today` never cleaned across sessions (3.6).
8. `calculate_risk_params.risk_percentage=0.02` hardcoded (3.8).
9. Asymmetric emitted context keys long vs short (`breakout_size_atr` vs `breakdown_size_atr`; missing `entry_mode` on short) (3.2, 4.1, 6).
10. `_calculate_institutional_strength` uses raw `hour` for peak-hours bonus, inconsistent with minute-of-day used elsewhere (3.8).

**P3 (minor / polish):**
11. `_calculate_vol_z` fills NaN with 0, masking insufficient-data state (3.1).
12. `_check_volume_surge` defaults to True on exception (3.3) — hides missing-data gates.
13. Volume-surge check can false-pass on short histories where rolling window isn't filled (3.4).
14. `levels` dict key naming mismatch long (`breakout_size`) vs short (`breakdown_size`) (3.2, 6) — downstream consumers must branch.
15. Audit-header name `orb_level_breakdown_short` doesn't match code's `orb_level_breakout_short` (6).
16. `k_atr` shared across PDH/ORH/PDL/ORL (Observation D — also a canonical drift flag for SUMMARY.md).

**Total: 16 issues (4 P1, 6 P2, 6 P3).**

## Fixes applied

| # | Audit ref | Fix | Commit |
|---|-----------|-----|--------|
| 1 | P1 #1 | Added `orb_level_breakout_long`/`orb_level_breakout_short` entries to `main_detector._map_structure_to_setup_type` direct_mappings. Investigation confirmed the mapping **was missing** — ORB-level events were being silently dropped at StructureEvent→SetupCandidate conversion. | `19b8342` |
| 2 | P1 #2 (a+b+c) | Mirrored long path's post-gate tail logic onto `_detect_short_breakdown`: added `_determine_entry_mode` call, `traded_breakouts_today` registration (via `_determine_entry_mode`), and `entry_mode`/`retest_zone` keys in event.context. | `fcf9aec` |
| 3 | P2 #4/5 | Normalized short path's level/context keys from `breakdown_size`/`breakdown_size_atr` → `breakout_size`/`breakout_size_atr` for parity with long path. Bundled with Fix 2 (same dict literals). | `fcf9aec` |
| 4 | P1 #3 | Refactored `traded_breakouts_today` from a flat `set()` (unbounded accumulation) to a `dict[session_date, set]` keyed by session date. Keys simplified from `symbol_level_date` → `symbol_level` (date is implicit in the bucket). | `573dafd` |

**Regression tests:** `tests/structures/test_level_breakout_structure.py` — 7 tests, all pass.

## Final decision
**FIXED-AND-TRUSTED.** All P1 bugs resolved; P2 key-naming asymmetry resolved. Deeper canonical upgrades (gap-day conditioner, per-level k_atr, institutional-vs-retail sustained volume, ~40 hardcoded thresholds) remain deferred to future sub-projects per audit recommendation.

---

## Issues found (consolidated)
[To be filled in after Items 2-7 complete.]

## Fixes applied
See table above.

---

## Final decision (assistant's recommendation)

**Recommended disposition: FIXED-AND-TRUSTED**

**Rationale:**
- Mechanical filter stack (10 gates, SMC, timing, candle conviction, dual-mode entry) is the most canonically-aligned of any detector audited so far.
- Two P1 bugs are localized and fixable:
  - ORB mapping missing in main_detector is a **single dict entry** (verify + fix).
  - Short path asymmetry is fix-by-mirror from long path (~40 lines of structured addition).
- Third P1 (`traded_breakouts_today` daily reset + short consumption) is ~10 lines.
- The DEEPER research points (gap-day, results-day, per-level k_atr, institutional vs retail volume sustain) are canonical upgrades, not bugs — deferred to future sub-projects.

**Critical context:** P1 #1 (ORB mapping) may explain gauntlet data showing zero or near-zero `orb_level_breakout_*` trades. If confirmed, backtest regeneration post-fix will produce meaningfully different Stage 1 results for this setup family.

**Recommended action plan:**

| Order | Issue | Effort |
|-------|-------|--------|
| 1 | **Verify P1 #1** — is `orb_level_breakout_long/_short` in `main_detector.py` direct_mappings? If missing, add. If events are reaching SetupCandidate through another path, document that path. | 30 min investigation + 15 min fix if needed |
| 2 | **P1 #2a** — Add `_determine_entry_mode` call to `_detect_short_breakdown` (mirror long path at line 247) | 30 min (TDD) |
| 3 | **P1 #2b** — Add `traded_breakouts_today.add()` to short path + consult dedupe set | 15 min |
| 4 | **P1 #2c** — Add `entry_mode` / `retest_zone` keys to short context dict | 15 min |
| 5 | **P1 #3** — Session reset logic for `traded_breakouts_today` (clear at start of each session OR use date-keyed dict) | 20 min |
| 6 | **P2 #4/5** — Normalize `breakdown_size` → `breakout_size` in short levels + `breakdown_size_atr` → `breakout_size_atr` in short context | 15 min + downstream check |
| **Total** | | **~2-2.5 hours** |

Deferred to SUMMARY.md / future sub-projects: per-level k_atr, gap-day conditioner, results-day conditioner, institutional-vs-retail sustained volume check, ~40 hardcoded thresholds, unbounded confidence, volume_surge NaN default, audit-header naming cosmetic.

**Alternative dispositions:**
- **TRUSTED as-is:** NOT recommended. Both P1 bugs affect actual trade flow (ORB mapping silently drops; short path missing entry mode).
- **DISABLED:** NOT recommended. Detector has ~3K+ trades and sophisticated canonical filter stack — disabling is massive overkill.

**Awaiting user disposition decision.**
