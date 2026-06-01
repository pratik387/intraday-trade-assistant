# Lessons Learned

Patterns and rules captured from corrections during development sessions.
Review at the start of each session to avoid repeating mistakes.

---

<!-- Format:
### [Date] - Short title
**What went wrong:** ...
**Rule:** ...
-->

### 2026-06-01 (#22) — Audit input data BEFORE concluding from rejection patterns

**What went wrong:** User reported "only gap_fade_short trading on paper, no other setups firing." After SSH-ing into the VM and reading `screening.jsonl`, I saw `or_window_failure_fade_short:vol_ratio=X outside cell band (8.0, 15.0]` repeated dozens of times with X in the 0.26-2.01 range, and concluded: "the cell-lock requires 8-15× baseline volume; today's max was 2.01×, the cell-lock is the bottleneck — matches the decay PAUSED status, retire the setup."

User push-back: "u should check the data first before coming to conclusion... like for ORH/ORL — are they proper?"

Right. I concluded from the *output* of a gate without auditing the *inputs* feeding that gate. The cell-lock check is `vol_ratio = bar_volume / baseline_vol`. If `baseline_vol` is wrong (stale parquet, wrong session_date lookup, wrong hhmm bucket), the ratio is wrong, and the whole "cell-lock too tight" theory is built on garbage. Same for the `sweep_high < poke_threshold` rejections — if ORH itself was computed from the wrong 09:15-09:30 window (e.g. tz-bug, missing bars, prior-day data), then `poke_threshold = ORH * (1 + buffer)` is wrong, and "sweep_high < poke_threshold" rejections are noise, not signal. I jumped to "cell-lock too tight → retire" without verifying any of the upstream computations.

**Why:** Rejection-reason strings look like diagnostic evidence but they're really *derived* outputs. A wrong input upstream silently produces "valid-looking" rejections downstream. The honest verification chain for any "setup X is silently filtering everything" hypothesis is: (1) inputs to the gate are computed correctly, (2) the gate threshold is reasonable for current market, (3) the gate rejects for the right reason. I skipped (1) and went straight to (3).

This is structurally adjacent to but distinct from Lesson 2026-05-05 (data feasibility — does the data exist at all). That lesson asks "do we have data?" This one asks "is the data we have correct?"

**Rule:**
1. **Before concluding from rejection_reason patterns** in a paper/live log, audit the *inputs* feeding each gate. For or_window: `ORH`, `ORL`, `poke_threshold`, `baseline_vol`. For below_vwap: `session_vwap`, `vwap_dev_pct`, `cap_segment`, `baseline_vol(symbol, session_date, hhmm)`. For long_panic: `gap_pct`, `dist_from_pdh`, `dist_from_pdl`, `regime_density_count`. Each must be cross-checked against an independent source (e.g. raw 5m parquet → manual VWAP calc; baseline parquet → grep for the symbol+date+hhmm tuple).
2. **Concretely**: grep `events.jsonl` or `screening.jsonl` for a SPECIFIC symbol-bar where the rejection occurred, dump the input values, and recompute them by hand from the raw 5m bar data. If they don't match → upstream bug, not gate-too-tight.
3. **When the conclusion would trigger a retire/disable action**, the input-audit step is mandatory, not optional. "Retire setup X because gate Y rejected everything" is a destructive decision built on the assumption that the gate's *inputs* were correct.
4. **In rejection-pattern responses to the user**, include the input-audit step in the analysis BEFORE the conclusion. Frame as: "Gate Y rejected with reason Z. Inputs to that gate look like [observed values]. Cross-check against [independent source] confirms / refutes the inputs. Therefore the gate is / isn't actually the bottleneck."
5. **Add to any future paper-trade diagnostic script** (e.g. `tools/diagnose_silent_setups.sh`) a section that dumps the gate's input values per rejected symbol, not just the rejection_reason string.

---

### 2026-05-29 (#21) — close_dn_overnight_long slot caps lifted for paper validation: MUST be restored before live cutover

**What went wrong (preventive — captured before it could):**
Paper-validation phase needs the full PF distribution observable, but the
production slot caps (max_concurrent_slots=4, max_new_positions_per_day=2)
were sized for live capital and would cap us to 2 fires out of 24+ valid
signals per day. Set in `configuration.json:capital_allocation` block.

**Rule (LIVE CUTOVER CHECKLIST):**
Before flipping `close_dn_overnight_long` to `enabled: true` in live mode,
restore `configuration.json` `setups.close_dn_overnight_long.capital_allocation`
to the live values stored as sibling `_live_*` keys:
  - active_margin_inr: 400000   (NOT 10000000)
  - cushion_inr: 100000          (NOT 1000000)
  - max_concurrent_slots: 4      (NOT 100)
  - max_new_positions_per_day: 2 (NOT 100)

The `_status_PAPER_ONLY_CAPS_LIFTED_2026_05_29` field in the config block
loudly flags the situation; the `_live_rationale` field documents WHY
those numbers are what they are (Phase 5 cell-mining confidence card was
computed under those caps; lifting them in live invalidates the PF=2.44
expectation and oversizes the real account).

This is a TEMPORARY paper-only relaxation. The PF / Sharpe numbers that
graduated this setup from research assumed 2-per-day; observing more than
that in paper is fine for evidence-gathering, but live risk caps must
match what was validated.

---

### 2026-05-22 (#20) — Set up the work environment BEFORE producing artifacts: research branch first, then draft briefs visible to user, then commit

**What went wrong:** During the new-setup brainstorm session (3 candidates: `first_hour_low_retest_fail_long`, `nifty_heavy_vwap_reclaim_long`, `pre_results_t0_morning_accumulation_fade_short`), I committed 3 batches of work (briefs + Phase 1/2 scripts) directly to `main` without ever creating a research branch. I also wrote 8-13KB brief files and committed them to `main` after only showing the user a summary table of mechanism statements — not the actual brief drafts. User pushback: "u should hv created a new branch first... discussed actual briefs".

**Why this is two missed habits, not one:**

1. **No research branch.** The project convention is to do research/exploratory work on `research/<topic>` branches (e.g., `research/europe-open-13ist` from the recent `close_dn_overnight_long` brief). I committed directly to main even though origin/main hadn't moved — my 3 commits forced a `git checkout -b research/2026-05-22-new-setups-batch && git reset --hard origin/main` to retrofit the branch.

2. **No actual-content review gate.** The `superpowers:brainstorming` skill has an explicit "User reviews spec? — Wait for user's response" gate before transitioning to implementation. I conflated "user approved the design summary" (mechanism statements + falsifiers as a compact table) with "user approved the full brief content" (8-13KB files with 7+ sections each). The user wanted to see the actual mechanism wording, the actual falsifier formulations, the actual adjacent-setup correlation tables — BEFORE those went into commits.

**Rules:**

1. **Every research session starts with `git checkout -b research/YYYY-MM-DD-<topic>`** BEFORE the first commit. Even if origin/main hasn't moved, even if I think the work is small, even if I'm planning to merge anyway. Branch first, work second, commit third. The cost of branching is zero; the cost of retrofitting is non-zero git surgery.

2. **The brainstorming skill's "User reviews spec" gate means the user reads the SPEC FILE CONTENTS, not just a summary.** Workflow:
   - Compact summary at design approval (mechanism statement + falsifier + adjacent setups in 3-5 lines per candidate) ✅ — what I did
   - Write the brief files to disk ✅ — what I did
   - **PAUSE. Tell user "briefs written, please read X/Y/Z and approve before commit"** ❌ — what I skipped
   - Commit only after user has confirmed they've read the actual file content

3. **Don't conflate "approve design" with "approve text."** "Yes write all 3 briefs" is permission to draft, not permission to commit. The commit gate is a separate user action.

4. **Default branch for ANY code/spec/research work that touches the repo: NOT main.** Main is for merged, reviewed, integrated work. Research and exploration goes on a branch. Even if the work doesn't ship, the branch keeps main clean and gives the user a clear merge/discard decision.

5. **If I miss the branch at the start, fix it BEFORE the second commit** — not after 3 commits. The retrofit cost grows linearly with commit count.

**Mechanical retrofit recipe (preserved here for next time):**
```bash
git checkout -b research/<topic>       # save work on a branch
git checkout main                      # back to main
git reset --hard origin/main           # ONLY if origin/main hasn't been pushed past
# Work continues on research/<topic>
```

---

### 2026-05-22 (#19) — Empirical confirmation: sanity's per-bar logic is identical to production. Sanity is tainted by 3 OTHER mechanics (NOT detector logic).

**What went wrong (continued from #18):** After Lesson #18 identified universe-data-source asymmetry as the dominant root cause, I empirically tested whether the per-bar detector logic itself diverges. Result: **for all 260 OCI HO fires, sanity's per-bar compute logic ALSO says "WOULD_FIRE"** — 260/260 match. So the detector is identical; the divergence is in how the per-bar fires are aggregated into trades.

Building on that empirical baseline, the apples-to-apples comparison (overlap of (sym, date) keys in both sanity and OCI HO) surfaced TWO MORE execution-semantics gaps:

1. **Sanity doesn't respect the per-(sym, date) latch.** Production's detector has `_fired_today: set` so each (sym, date) fires at most once. Sanity fires on EVERY qualifying bar within the cell window — for below_vwap (13:00-14:55), the same symbol can have 3-4 fires per day. Sanity-on-overlap had 61 trades on 17 unique (sym, date) keys (3.6× over-count). **Sanity over-counts fires per day.**

2. **Sanity's synthetic R-walk exit gives different PnL than OCI's real tick execution on the same fire.** On the 17 overlap (first-fire-per-day) keys, sanity reports PF 0.43 vs OCI's 0.87 on the same (sym, date) pairs. Both use the same entry bar; the realized PnL diverges because sanity's same-bar SL-wins-tie pessimism + Mode B next-bar-open entry differs from OCI's real tick execution.

**Combined projection (sanity-first-fire-per-day + OCI-only fires sanity missed):** 296 trades, PF 1.13, mean +Rs 37/trade. The gap to OCI's actual PF 0.965 is dominated by execution-semantics, not signal divergence. Either way the candidate is marginal.

**The three mechanics inflating sanity's PF claim vs production:**

| Mechanic | Direction | Magnitude (below_vwap HO) |
|---|---|---|
| #18-A: Universe-data-source asymmetry (5m vs daily archive coverage) | Sanity → narrow | 235/260 OCI fires on symbols sanity rejects |
| #18-B: nse_all-missing symbols default to `cap=unknown` | Sanity → wide | 151 spurious sanity inclusions, PF 0.83 (losers drag) |
| #19-A: No per-day latch | Sanity → high count | 3.6× more fires per (sym, date) than OCI |
| #19-B: Synthetic R-walk exit (pessimistic) | Sanity → low per-trade PnL | Same-day overlap PF 0.43 (sanity) vs 0.87 (OCI) |

Net: #18 inflates sanity's PF (by including only the "easier" old cohort); #19 distorts it in mixed directions. They partly cancel but the combined sanity PF doesn't represent realistic production.

**Rules:**

1. **Per-bar detector logic is sound.** Don't chase ghosts there. Any sanity-vs-OCI divergence on detector outputs has a different source (universe gate, latch, exit walk).

2. **Sanity scripts MUST implement the same per-(sym, date) latch as the production detector.** Use `df.drop_duplicates(subset=['bare','signal_date'], keep='first')` or equivalent. Without it, sanity reports inflated trade counts.

3. **For cell-locked setups: realistic forward expectation = OCI actual PF, not sanity PF.** Sanity's PF should only be used to identify candidates worth running OCI on, not to size live positions. Treat sanity as a candidate-filter, not a backtest.

4. **Tooling: `tools/sub9_research/production_universe.py:ProductionUniverseGate`** — reusable per-date universe filter mirroring production's setup_universe.py builders. Use this INSTEAD of window-level coverage filters in future sanity scripts.

5. **Methodology change: all future per-setup sanities must:**
   - (a) Use `ProductionUniverseGate` for the symbol filter (consolidated_daily.feather based, per-date check)
   - (b) Implement per-(sym, date) latch (one fire per day per symbol)
   - (c) Document the exit-walk semantics explicitly (synthetic R-walk vs real tick); when comparing to OCI, expect a few % PF gap from this alone

---

### 2026-05-22 (#18) — Sanity vs OCI parity has THREE root causes, layered, with universe-data-source asymmetry as DOMINANT

**What went wrong:** Continued the OCI vs sanity parity investigation for `below_vwap_volume_revert_long` after Lesson #17 fixed legacy filters. Found that even after universe-filter alignment, OCI PF (0.965, net loser) still diverged from corrected sanity PF (1.32). I suspected "per-bar signal divergence" (different VWAP, vol_ratio, latch). Spent time looking at specific bars (RPPINFRA 2026-03-30 14:55) and confirmed the DETECTOR criteria fire identically. The real divergence was elsewhere.

**Root cause hierarchy (in order of magnitude):**

| Cause | Direction | Impact on below_vwap HO |
|---|---|---|
| **A. Universe-data-source asymmetry** — sanity uses 5m feather coverage (`>=80% of HO days`); production uses `consolidated_daily.feather` (independent coverage) | Sanity → narrower (rejects newly-tracked SME) | **235 of 260 OCI fires (90%) on symbols sanity wholly rejected** |
| **B. cap_segment default for nse_all-missing symbols = "unknown"** | Sanity → wider (wrongly) | 151 spurious sanity inclusions; PF 0.83 drags overall PF DOWN |
| **C. True per-bar signal divergence on same-universe subset** | Sanity → optimistic | Possibly real (OCI PF 0.80 vs sanity 1.32 on the 23 trades both saw) but small n |

**The "5m archive vs daily archive coverage" asymmetry** (Lesson #16 reprise): `consolidated_daily.feather` has cap=unknown symbols going back to 2023. `monthly/*_5m_enriched.feather` only has 5m bars for SME names since ~Dec 2025. Production's universe builder uses consolidated_daily → symbol passes. Production's screener uses monthly feathers → has bars from Dec onward → fires. Sanity uses 5m feathers as its universe-gate (via the 80% window-coverage filter) → rejects symbols with only 5 months of bars → never even tries to detect.

**Concrete proof:** For 5 OCI-only fire cases (RPPINFRA, SIMPLEXINF, TRACXN, KAMOPAINTS, CIFL):
- All 5 in nse_all.json with cap=unknown + mis_enabled=True
- All 5 in consolidated_daily.feather since 2023 (152 HO days each)
- All 5 in monthly 5m feathers ONLY from 2025-12 onwards (5 of 10 HO months)
- Sanity's `days_per_sym >= 144` (80% of 180 HO days) rejects them at ~100 days
- Production's per-date filter accepts them

**Implication:** sanity PF claims for SME-heavy cell-locked setups (`cap=unknown`) are inflated by SURVIVORSHIP — they're computed only on the "older tracked" sub-cohort of cap=unknown. The newer SME names that production fires on have different edge dynamics. OCI PF is the realistic forward expectation; sanity PF is optimistic.

**Rules:**

1. **The DOMINANT root cause of sanity vs production parity gap is universe-data-source asymmetry.** Not per-bar detector logic. Verify by computing the OCI-only vs sanity-only fire counts and segment by 5m-feather coverage. If most OCI-only fires are on symbols with partial 5m coverage, the universe gate is the issue.

2. **Sanity's universe filter is a SURVIVORSHIP filter.** Window-level coverage thresholds reject newly-tracked symbols entirely. Production's per-date rolling filter accepts them once they cross the threshold. These give different universes. For honest sanity-to-OCI comparison: align sanity to use per-date rolling coverage (NOT window-level), OR use the SAME data source (consolidated_daily.feather) for the universe gate.

3. **Don't trust sanity PF for SME-heavy cell-locked setups.** When the cell targets `cap=unknown` (where 5m archive coverage is most sparse), sanity backtest is computed on a non-representative sub-cohort. The setup may have completely different forward edge once production fires on the newer SME names. Trust OCI PF as the realistic expectation.

4. **For `below_vwap_volume_revert_long`:** OCI HO PF 0.965 (NET LOSER on 141 trades). Sanity HO PF 1.32 is computed on a DIFFERENT universe than production trades. **Do NOT activate for live capital** based on sanity numbers alone — paper-trade validation phase is required, and the realistic expectation should be the OCI number, not the sanity number.

5. **For `close_dn_overnight_long`:** less affected because the cell accepts large+mid+small+unknown caps. Only unknown sub-cohort hit by the 5m coverage gap. 10.3% rejection rate (vs below_vwap's 26.2%). Still needs an OCI run on the latest code (with legacy filters zeroed via commit `35b0c96`) before paper-trade activation.

---

### 2026-05-21 (#17) — Legacy universe filters (min_trading_days, min_daily_avg_volume) are inappropriate for cell-locked setups — remove them

**What went wrong:** OCI vs sanity parity diagnostic for `below_vwap_volume_revert_long` showed:
- Sanity HO PF 1.17 (577 cell-locked trades)
- OCI HO PF 0.96 (141 completed trades, NET LOSER)
- (sym, date) overlap: only 6.9%

My initial diagnosis pointed at `consolidated_daily.feather` coverage gaps. Built a universe-parity simulator (`tools/sub9_research/universe_parity_simulator.py`) that "confirmed" production was a strict subset of sanity. Recommended backfilling consolidated_daily as P0.

User correction: "no these filters shouldn't be there... they are generalised filters which are from legacy code." The min_trading_days_required=30 + min_daily_avg_volume=50000 in `services/setup_universe.py:below_vwap_volume_revert_long_universe` are intraday-MIS-era defaults. For cell-locked setups, the cell itself does the selection (e.g., `vol_ratio_bin=gte_10` already requires 10× signal-bar volume; cap=unknown locks the universe). Legacy filters are redundant AND harmful — they clip out the very tail names where edge lives.

**Surprise inversion after removing legacy filters + correctly handling `get_cap_segment` defaulting:**
- Sanity ORIGINAL (577 trades): PF 1.17
- Production-aligned (426 trades after rejecting `mis_disabled` symbols): **PF 1.32**
- The 151 wrongly-included symbols (mis_disabled, sanity treats as cap=unknown by default): **PF 0.83** (NET LOSERS dragging down original sanity)

So the legacy filters were the wrong target. The real fix: sanity overstates universe because `services.symbol_metadata.get_cap_segment` returns `"unknown"` as default for nse_all-missing symbols, accidentally classifying 151 untradeable symbols into the cell. The corrected universe-aligned sanity claims PF 1.32 — HIGHER than originally reported.

OCI still shows PF 0.96 (0.36 below corrected sanity 1.32). The remaining gap is per-bar signal divergence (root cause #2), not investigated yet — likely VWAP cumulative compute, cross_day_rvol baseline source, or detector latch state differences between research and production code paths.

**Rules:**

1. **For cell-locked setups, the universe builder should NOT apply intraday-MIS-era legacy filters** (`min_trading_days_required`, `min_daily_avg_volume`). The cell already filters for the right cohort. Setting both to 0 in config disables them; the `len(ddf) < 0` and `avg_vol < 0` checks always evaluate False, while `ddf is None or ddf.empty` still rejects truly missing data. (Done 2026-05-21 commit `35b0c96` for `below_vwap_volume_revert_long` + `close_dn_overnight_long`.)

2. **`get_cap_segment` defaults to `"unknown"` for missing symbols.** This means setups with `required_cap="unknown"` accidentally pass nse_all-missing symbols. For sanity scripts, this overstates the universe vs production (production also requires `mis_enabled=True` which defaults to False for missing symbols, so production naturally rejects them — but the sanity output shows them with cap_segment="unknown" and they enter cell-locks). When designing a cell-lock check, either:
   - Make sanity require explicit nse_all presence (`bare in nse_all` AND `nse_all[bare].cap_segment == required_cap`)
   - OR accept that sanity will overstate and post-filter with `mis_enabled` to match production

3. **OCI-vs-sanity parity has TWO root causes, not one.** Don't conflate them:
   - **Cause A**: universe-builder filter mismatch (sanity is permissive about new listings + nse_all-missing symbols). Fixed by aligning filters and validating with `tools/sub9_research/sanity_to_production_universe_align.py`.
   - **Cause B**: per-bar signal divergence on same-universe symbols (different VWAP, baseline source, latch state). Requires per-bar input comparison to diagnose.

4. **Cell-locked setup migration checklist** (for any future shipped setup):
   - Set legacy universe filters to 0 in config (cell does the selection)
   - Verify sanity = production universe via the alignment tool
   - Investigate per-bar signal divergence if a 0.2+ PF gap remains

5. **Triaged research vs production gap quickly** (the right path took ~30 min once the user corrected the diagnosis): (a) run alignment tool with both legacy filters at original values → see gap; (b) zero out legacy filters → see gap closes ~partially; (c) inspect remaining rejected trades' nse_all presence → identify `mis_disabled` as the residual; (d) recompute realistic PF on aligned set.

---

### 2026-05-21 (#16) — Sanity script's universe ≠ production universe — verify monthly-feather coverage by cap segment BEFORE shipping

**What went wrong:** Shipped `below_vwap_volume_revert_long` to OCI based on Phase 5 research (PF=1.587 / OOS 1.782 / Hold 1.606 on 3712 trades, 2023-2026). First OCI backtest (20260521-155634_full, Jan-Apr 2023): **zero fires across 79 days**. Spent time chasing entrypoint / cross_day_rvol upload / configuration / `wide_open` mode. Real cause: data-coverage asymmetry between caches.

**The asymmetry:**
- `consolidated_daily.feather`: 725 cap=unknown symbols with 2023 daily data (universe builder reads this)
- `backtest-cache-download/monthly/2023_04_5m_enriched.feather`: 97 cap=unknown symbols (screener reads this for df_5m)
- Intersection that passes universe filter (>=30 days, >=50K vol): **0**
- Result: universe picks 203 symbols → screener has 0 of them in `_precomputed_5m` → `active_syms=612 with_data=408` → detector never called → silent zero

**Why Phase 5 missed it:** Sanity script reads the same monthly feathers and labels `cap_segment` from current `nse_all.json`. It found 3712 trades among the **~102 cap=unknown symbols that happened to have historical 5m data** (older symbols reclassified down to "unknown"). Cell-lock JSON says `"unique_symbols": 102` — the entire research was on ~100 symbols. Production diverges because its universe builder picks the 203 *currently*-cap=unknown symbols with daily history, which are mostly newly-listed SME names (median 1m-feather start: 2025-12) with no historical 5m data.

**Coverage by month (cap=unknown in monthly 5m feather):**
- 2023_01: 94 / 2023_04: 97 / 2024_01: 101 / 2025_01: 116 / **2026_03: 719 / 2026_04: 987**

Cap=unknown 5m bulk-download only happened March-April 2026. Validates on 2026-only data (20260521-163406_full: 205 trades / 42 days fired across Jan-Apr 2026).

**Rules:**

1. **Before shipping any setup whose cell-lock includes a cap-segment filter, verify the production universe overlaps the research universe.** Concretely: take the universe builder's filter, run it against `consolidated_daily.feather` for a sample backtest date, then intersect with the corresponding monthly `_5m_enriched.feather`. If the intersection is < ~50% of the universe builder's output, the setup will silently misfire in production.

2. **`unique_symbols` in the cell-lock JSON is a red flag, not a footnote.** When research is concentrated in ≤200 unique symbols, the cell may be picking up which symbols happened to be in the 5m archive — not a structural edge. Add to confidence-card review: cross-check that the universe builder's filter (run against historical daily data) produces an overlap with research symbols of ≥80%.

3. **Add a loud failure mode to the detector dispatch.** When `len(active_syms) - len(active_syms_with_data)` exceeds e.g. 50% of a single setup's universe (computable from `_tag_map` per-setup membership), log `DISPATCH_DATA_GAP | <setup> | universe=N with_data=M (M/N=X%)` as WARN, not silent INFO. Would have caught this on day 1.

4. **Asymmetric data coverage exists across caches.** `consolidated_daily.feather` (built from daily downloads, has wide coverage incl. SME names back to 2023) and `monthly/YYYY_MM_5m_enriched.feather` (built from per-symbol 5m archives, only newer SME names have history) are NOT in sync. The cross_day_rvol baseline parquet inherits the 5m gap because it's aggregated from monthly feathers. Don't assume "symbol has daily data" implies "symbol has 5m data."

5. **Quick triage flowchart for "setup didn't fire":** (a) Universe builder log line shows non-zero count? → universe OK; (b) `DISPATCH_OPEN_WINDOW` for the setup? → window OK; (c) `active_syms - with_data` for the setup's bars shows the universe entirely missing? → **data coverage gap** (not config, not detector logic). Tools to confirm: simulate the universe builder against `consolidated_daily.feather` and intersect with `monthly/YYYY_MM_5m_enriched.feather`.

---

### 2026-05-20 (#15) — Per-trade walk-forward methodology was wrong; rebuilt as research-backed confidence framework

**What went wrong:** Spent multiple rounds building a walk-forward validator that tier-classifies each 3-month window (Green / Yellow / Red) based on PF/DSR/PBO thresholds. User pushed back hard:
- "the solution seems premature. this not at all serves the purpose of getting confidence on a setup. the methodology seems fucked completely"
- "[the thresholds] seem very random and not research backed"

I caved and admitted: "I made up the numbers — 1.05, 0.95, 0.8×, the 'ship at 25%' — none of those have research backing." A deep-research dive (`reports/sub9_sanity/_per_trade_validation_research.md`) confirmed:
- DSR is defined on per-PERIOD returns, not per-trade
- PBO requires testing N candidate configurations and choosing the best — meaningless on a single setup's trade list
- "200-500 trades minimum" is mis-attributed to López de Prado
- All numerical thresholds in practitioner blogs are folklore, not literature-backed

**The redesign (research-backed):**
- Component 1: Bootstrap BCa CI (Efron-Tibshirani 1993) on aggregate PF / expectancy / win-rate
- Component 2: Per-regime decomposition (López de Prado tactical 2019) with 7-regime Bai-Perron-style schema for Jan 2023 - Apr 2026 in `assets/regime_schema.yaml`. Each regime has evidence_class: EVENT+DATA, DATA, or INTERPOLATED
- Component 3: Selection-bias haircut (Harvey-Liu 2015) via ONC clustering effective-M (López de Prado & Lewis 2019)

Output is a **confidence card** (intervals, not verdicts) per setup. Framework refuses to produce ship/no-ship binaries — the researcher reads the intervals.

**Subtle bug found AND fixed during redesign:** The first Harvey-Liu implementation took `abs(t_stat)` before computing p-values, then back-solved through `ppf(1 - p/2)` which is ALWAYS positive. Result: a setup with raw Sharpe -2.6 came out with adjusted Sharpe +2.4 (sign flip). Caught when reviewing the mis_unwind card showed adj Sharpe positive for an obvious losing setup. Fixed via the standard haircut-factor approach: `SR_adj = SR_raw × t_crit_1 / t_crit_M`. Added regression test `test_harvey_liu_haircut_preserves_sign_for_negative_sharpe`.

**Calibration sanity check (works):** Across 8 OCI production setups, the framework cleanly sorts 3 retired setups to the bottom (PF CI crosses 1.0, negative adj Sharpe, weak in all regimes) and 5 active setups to the top — without using ship/retired labels as input. Marginal middle: circuit_t1_fade_short, delivery_pct_anomaly_short, round_number_sweep_short — no clean threshold separates them, which is honest reporting.

**Rules:**

1. **Per-trade outcome list ≠ per-period returns.** DSR / Sharpe / PBO are defined on per-period equity curves. To apply them to per-trade data, FIRST aggregate to daily P&L, THEN compute the statistic on the daily series. Never compute "Sharpe of per-trade pnls" — it has no meaning under the literature definition.

2. **If you can't cite a paper for a threshold, don't propose it.** "PF > 1.05" / "DSR > 0.95" / "win 9/13 windows" are folklore. The framework outputs INTERVALS; the researcher judges.

3. **Selection bias is real and quantifiable.** Effective M from ONC clustering, then Bonferroni haircut factor (sign-preserving) gives the adjusted Sharpe. For our 8 setups, M=8 (all independent), haircut factor ≈ 0.72. Document the M source — if you discover 30 more variants tested historically, M jumps and the haircut tightens.

4. **Regime evidence has tiers.** R3 (post_election_consolidation) is INTERPOLATED — boundary inferred from absence of evidence, not from a detected break. Wide CIs in R3 are EXPECTED, not a setup weakness. Don't gate on R3 alone.

5. **Calibration test for ANY new component:** run it on the full 8-setup OCI portfolio and check that retired setups land near the bottom WITHOUT being labeled as such. If retired and active are intermingled, the component isn't measuring what we want.

6. **When the user says "very random and not research backed" — they're right.** Don't defend numbers I made up. Cite or remove.

---

### 2026-05-20 (#14) — Use ACTUAL per-trade fees, not a calibrated global constant

**What went wrong (multiple times):**

**Round 1:** Walk-forward defaulted `fee_pct_round_trip = 0.5` (% of capital basis), picked as a "conservative round number." User said "I think your system is broken somehow." Traced 100 trades from pre_results_t1 single-leg — measured fee_pct_capital = 0.2484 (std 0.0002). Changed default to 0.25, felt confident. Lesson #14 v1 was triumphalist about "calibration."

**Round 2:** User said "I still think there are bugs." Did a per-SETUP audit. Discovered fee_pct varies dramatically:
  - pre_results_t1 single-leg: 0.248 <- my 100-trace was biased to this
  - mis_unwind: 0.296
  - capitulation_long_v2: 0.437
  - circuit_release: 0.411
  - long_panic_gap_down: 0.461
  - or_window_failure: 0.444
  - delivery_pct: 0.488
  - capitulation_long_morning T1-partial: 0.531

  My "calibration" to 0.25 was wrong by 0.1-0.3pp for every setup except pre_results_t1 single-leg. The "fix" made some verdicts WORSE.

**Final fix:** walk_forward now uses per-trade ACTUAL fees (`fee_inr` column from canonical CSV) when available. Falls back to flat 0.25 only if fee_inr missing. **Verdicts under per-trade fees revert closer to the original 0.5 verdicts** because real fees average ~0.4 across setups, much closer to 0.5 than 0.25.

**Why fee % varies by setup:**
- Brokerage cap (Rs 20/order) bites differently at different trade sizes
- T1-partial trades have 3 orders, not 2 (more brokerage)
- STT is side-asymmetric (sell-only), affecting LONG vs SHORT
- Stamp duty is side-asymmetric (buy-only)

**Painful meta-lesson:**

1. **A "verified against real data" calibration can still be wrong if the sample is biased.** I sampled 100 trades but they were all the same setup, same side, same partial mode. The variation I missed was across DIFFERENT setups.

2. **Calibration sample must span the diversity it will be applied to.** A global constant calibrated on one slice is a constant, not a calibration.

3. **When the cost of being wrong is high, don't generalize a constant — use per-trade actuals.** Fee data is already in the CSV. Use it.

4. **Repeat user pushback is signal, not noise.** The user said "broken" twice. The first time I found one bug but my fix introduced another. The second time forced me to look at the structure (per-setup variation) instead of the constant.

5. **"100 tests passing" is not a verdict on correctness.** All 100+ tests passed both with fee=0.5 AND fee=0.25 AND now with per-trade actuals. They tested the formula, not the parameter calibration. Calibration tests need to test the EVIDENCE behind the parameter, not just that the formula computes.

6. **For Indian retail intraday SPECIFICALLY:** use per-trade `fee_inr` column. Don't use a flat fee_pct unless you absolutely must (e.g., synthetic fixture data without fee column). When forced to use a flat default, 0.40 is closer to the per-setup median than 0.25 or 0.5.

### 2026-05-20 (#13) — Sanity Mode B walk-forward systematically OVER-ESTIMATES production PF

**What went wrong:** Ran canonical walk-forward on `_circuit_release_fade_short_trades_*.csv` (post-Stage-2 adapter). Result: GREEN 9/13 with HO_war PF 4.35. Retired_setups.md cited production HO_pre PF 0.84 / HO_war PF 0.60 — net LOSING -Rs 63K. The 4x+ discrepancy in same time window suggested either an adapter bug, a cell mismatch, or methodology drift. **Root cause** (from production config comment, already documented at retirement): sanity Mode B (next-bar-OPEN entry) captures ALL signals as trades. Production uses `entry_zone retest` filter (price must return UP within 0.3% of signal close before entry triggers). For SHORT setups, this filter has selection bias — fast-mover winners (price drops immediately) never retest, so production MISSES them. Sanity sees the winners and reports inflated PF.

**Quote from production config:** "Sanity Mode B (PF 2.44) was wrong because it assumed all signals = good signals."

**Why this matters now:** The whole point of Schema Stage 2 was to build trustworthy walk-forward on sanity data. But sanity data carries a systematic optimism bias for setups where execution semantics matter. circuit_release sanity showed GREEN but production was RED. If we'd un-retired based on sanity walk-forward, we'd ship a losing setup with 5x MIS leverage.

**Rule:**

1. **Walk-forward on sanity Mode B output OVER-ESTIMATES production PF systematically.** Especially for SHORT setups where the entry_zone retest filter has selection bias against fast-mover winners. A GREEN tier on sanity data is necessary but NOT sufficient for production ship.

2. **Cross-check sanity walk-forward verdict against production OCI data BEFORE acting.** For un-retire decisions: require that the OCI production trade_report.csv ALSO shows GREEN tier on walk-forward. If sanity is GREEN but production is RED, the sanity is over-optimistic — investigate execution-semantics gap.

3. **Going forward, walk-forward should consume `trade_report.csv` from OCI runs**, not sanity outputs. The trade_report.csv has real execution outcomes including entry_zone filtering, slippage, latency. Stage 5 of the walk-forward methodology project should be reframed: validate setups via production-data walk-forward, not sanity-data walk-forward.

4. **Mode B sanity validations are still useful for the OTHER direction.** If sanity Mode B shows RED, production will be even more RED (since production has less-favorable execution). RED on sanity = honest retirement; GREEN on sanity = needs further validation.

5. **Document `_comment_no_immediate_execution`-style findings on the SETUP's config block.** This circuit_release comment captured the critical sanity-vs-production execution-semantics gap. Future setups should have similar `_comment_sanity_vs_production_drift` blocks if a delta is observed.

**Examples from current session (2026-05-20):**
- circuit_release_fade_short: sanity GREEN 9/13, production RED (HO PF 0.84/0.60). Discrepancy was the entry_zone retest filter. Retirement stands.
- pre_results_t1_fade: sanity RED 0/13, production RED. Mode B optimism didn't save it. Honest retirement.
- capitulation_long_v2: sanity RED 3/13, retirement matches.
- mis_unwind_vwap_revert_short: sanity RED 1/13, retirement matches.

### 2026-05-19 (#12) — Walk-forward methodology replaces chronological 3-period validation

**What changed:** Replaced 3-period chronological validation (Disc 24mo / OOS 9mo / HO 7mo) with walk-forward (13 × 3-month windows + per-window bootstrap CI + 3-tier classification GREEN/AMBER/RED). Triggered by recognizing that 5 of 5 recently-retired setups followed the same 2-of-3-pass pattern with the failing period aligned to known Indian regime breaks (SEBI Oct 2024, FII Jan-Mar 2025, war Jan-Apr 2026). The old methodology couldn't distinguish "edge gone" from "one bad regime window dominated PF."

**New discipline:**
1. **Walk-forward is mandatory for all new Phase 5 validation.** No more single-OOS/single-HO ship gate. Implementation in `tools/methodology/walk_forward.py`.
2. **Mechanism pre-registration via git timestamp is mandatory for AMBER tier.** `mechanism_tags` field in setup config must be committed BEFORE walk-forward runs; engine refuses otherwise (`tools/methodology/pre_registration.py`). Prevents post-hoc rationalization.
3. **Three-tier outcome:** GREEN ≥ 9 of 13 windows → ship full size. AMBER 6-8 of 13 + mechanism docs → 90-day forward-validation at 25% size. RED ≤ 5 of 13 → retired.
4. **Per-window bootstrap CI gate.** Window passes iff PF_net ≥ 1.10 AND 95% bootstrap CI lower bound > 1.0. Defends against small-n noise.
5. **Circuit breaker is the live safety net.** Daily-EOD `jobs/check_circuit_breakers.py` computes trailing-60d NET PnL per setup; auto-disables if below threshold (mean − 2σ of backtest window distribution). Manual re-enable required.
6. **Active setup consistency check (Option C):** active setups below GREEN tier are NOT proactively retired — they stay live at full size AND get circuit breaker monitoring AND go on a watch list (`docs/active_setups_review.md`). Drawdown is the signal, not retroactive retirement.
7. **OOS+HO combined NET PnL is still the most damning indicator** for retirement (lesson #11). Walk-forward + circuit breaker don't replace this gut check.

**Reference docs:**
- Spec: `docs/superpowers/specs/2026-05-19-walk-forward-methodology-design.md`
- Plan: `docs/superpowers/plans/2026-05-19-walk-forward-methodology.md`
- Methodology runbook: `docs/methodology_walk_forward.md`
- Active review: `docs/active_setups_review.md` (populated during Phase 3)

### 2026-05-19 (#11) — Monthly breakdown is mandatory pre-retirement, but data-classification hypotheses MUST be falsified before retiring or shipping

**What went wrong (4-layer investigation chain):**

I retired `pre_results_t1_fade` on Layer 1 (period aggregates: Disc 1.10 / OOS 0.82 / HO 1.15). User asked: "did u check month by month? i see a pattern... discovery is similar to ho and oos is different altogether... maybe some kind of indian market shift?"

- **Layer 2 (monthly breakdown):** OOS is the outlier, Disc + HO are similar (medians 1.36 / 0.79 / 1.61). Suggested structural cause.
- **Layer 3 (data audit):** NSE `announcements_fr` source died after Mar 2025. AMC events for 2025+ got demoted to "scheduled" class. **Hypothesis: setup missed events due to source-priority misclassification.**
- **Layer 4 (conservative re-run with corrected `{AMC, scheduled}` filter):** Recovered ~50% more OOS events. **OOS PF_net got WORSE (0.816 → 0.784).** Demoted events were ALSO losers. Hypothesis falsified.

**Final verdict: RETIRE (confirmed). Real failure mode: regime-conditioned edge with FII-positioning dependency** — setup mechanism (institutional T-1 de-risking) requires FIIs net-LONG to have something to de-risk. Jan-Mar 2025 FII outflow (~₹1L cr) destroyed the mechanism. Even when AMC universe restored, the events fired during FII-out regime were structurally losers.

**Why this matters as a lesson:** The user's monthly-breakdown instinct caught a LEGITIMATE data quality concern that I should have caught myself. But the data-correction hypothesis turned out to be wrong — the events WERE misclassified, but those events were ALSO losers. Two truths can be simultaneously valid: (a) data ingestion has issues; (b) the underlying mechanism is regime-broken. Conflating them is the trap.

**Rules:**

1. **Phase 5 MUST include monthly within-period breakdown.** Always compute median monthly PF + Q25-Q75 + losing-months count. If a period's median differs by ≥0.30 from other periods, investigate STRUCTURAL cause before final retire/ship.

2. **Outlier-month structural-cause hypothesis is required.** Cross-reference outlier months with:
   - Indian-market regime (FII flows, SEBI rule changes, election cycles, sector corrections, war/macro shocks)
   - Data-source coverage (calendar mutation history, scrape completeness, `.bak_*` files, dead endpoints)
   - Mechanism dependencies (e.g., setup depends on FIIs LONG → fails when FIIs net-sell)

3. **Hypotheses MUST be falsified, not just observed.** If you propose "OOS bleed = data classification artifact," CONFIRM by re-running with the correction. If the correction makes OOS PF *worse* or *unchanged*, the hypothesis is dead → retire. If it makes OOS PF *better*, validate via stationary-edge check before ship.

4. **Earnings calendar / event-data freshness is a load-bearing dependency.** Setups keyed on `data/earnings_calendar/`, `data/fno_ban_history/`, etc. must verify per-period coverage AND source-priority completeness in Phase 5. NSE endpoints DO die (e.g., `announcements:Financial Result Updates` returned 0 rows for all 2025-04+ chunks).

5. **Source-priority audits are part of Phase 5.** Print `df.groupby([period, source]).size()` for every period. Missing high-priority sources is a smoking gun for misclassification — but requires Layer 4 (re-run with corrected filter) to determine if it actually drove the result.

6. **OOS+HO combined NET PnL is the single most damning gate.** v1 result was -Rs 92K (already retired-worthy). v2 result with corrected filter: -Rs 290K (3x worse). A setup that loses money across all post-Discovery data combined is not shippable, regardless of HO standalone passing other gates.

7. **Conservative re-run (Path 2) beats trust-the-correction (Path 1).** When user is offered "trust the fix" vs "re-run + re-validate," default to re-validate. User chose Path 2 here and the conservative approach revealed v1 retirement was correct (just for partly wrong reasons). Path 1 would have shipped a corrected filter without re-validating the underlying edge.

8. **"Non-stationary edge" is the wrong default label.** The right vocabulary: "regime-conditioned edge with structural dependency on X" — where X is a specific identifiable macro/regulatory factor. Naming the mechanism dependency is part of the retire verdict.

### 2026-05-19 (#10) — Phase 1-5 discipline does NOT make a setup ship; the HO ship gate does

**What went wrong:** Ran a full disciplined Phase 1-5 revival of `capitulation_long_morning` (renamed v2). Phase 1 Indian-market research (intradaylab.com gap-down recovery). Phase 2 empirical signature on 21,548 Discovery 2023-24 gap events confirmed mid_cap × gap [-5,-3] sweet spot. Phase 4 sanity v2 with anti-bias guards (Mode B entry from i+1, anti-look-ahead, locked filters before sweep). Phase 5 3-period parity: Disc PF_net 1.13, OOS PF_net **1.62** ⭐, HO PF_net **1.03** ❌. Trajectory was structurally identical to `mis_unwind_vwap_revert_short` (Disc 1.21 → OOS 1.22 → HO 0.75) — the exact pattern lesson #1 was written to prevent. Revival fell into the same trap despite full process discipline.

**Why:** Process discipline (Phase 1-5 chain) makes the EVIDENCE reliable. It does NOT make the EDGE real. A setup can pass every Phase 1-5 gate honestly and still die at the HO ship gate because OOS happened to be a favorable-regime window. The Disc+OOS-favorable-regime illusion is a property of the data, not a property of the methodology. Phase 1-5 discipline is necessary but NOT sufficient.

**Rule:**
1. **The HO ship gate is the ONLY gate that matters for production decisions.** Disc and OOS are evidence-gathering; HO is the verdict. If HO PF_net < 1.10 OR mwin < 4/7 OR |WR delta OOS→HO| > 10pp, retire — regardless of how disciplined the chain was.
2. **Disc+OOS parity ≠ HO survival, FULL STOP.** Even Disc+OOS+favorable-direction-pattern (OOS > Disc, suggesting "improving edge") is NOT a green light. v2 OOS at 1.62 was a 50% premium over Disc 1.13 — that gap should have been a red flag, not encouragement.
3. **War-period resilience is interesting but insufficient.** v2 HO_war PF 1.12 beat 4 of 5 production SHORT setups in the same window. That hints volatility-favoring LONG complement, but standalone PF_net 1.03 still fails ship-gate. Portfolio-correlation thesis is the only path forward, and that requires correlation analysis, not standalone ship.
4. **When you see Disc 1.1 → OOS 1.6 → HO 1.0 pattern emerge, retire FAST.** Don't try variant-tweaks (tighter cell, news filter, different TS). That's the cell-mining illusion lesson #2 in a new costume — chasing HO after the fact is data-mining.



**What went wrong:** During paper-trade readiness audit I listed `cache/ohlcv_archive/` and `cache/preaggregate/` as VM deployment requirements. User pushed back: "I don't think we use cache folder any way in live trading. If we are then it's wrong." Audit confirmed: cache is correctly gated, but the assumption that it might be needed was sloppy — I hadn't traced the production data path before recommending VM setup.

**Why:** In live/paper, the data path is `WebSocket → 1m bars → 5m aggregate → DispatchPlanner → detectors`. Zero feather reads. Cache exists ONLY for backtest replay (`_precomputed_5m` initialized only inside `if env.DRY_RUN:` block at `services/screener_live.py:563-564`). All cache truthiness checks (`elif self._precomputed_5m and s in ...`) safely fail in live (empty dict).

**Rule:**
1. **Before recommending any deployment artifact, trace the live data path end-to-end** — don't pattern-match from backtest setup docs to live VM checklist.
2. **Every cache/feather/precomputed access must be gated by `env.DRY_RUN`** OR safely fall through (empty container truthiness check). Verify by grep at every regression-prone change to data loading.
3. **VM paper deployment surface**: `nse_all.json`, `broker/kite/token.txt`, `assets/nse_holidays.json`, `config/configuration.json`, requirements.txt, env vars. NOT cache/.
4. **Defensive guard candidate**: add startup assertion `assert not args.paper_trading or not self._precomputed_5m` if regression in cache gating ever fires.

### 2026-05-19 (#8) — Setup retirement protocol: deletion is part of the architectural discipline

**What went wrong:** Retired 3 setups in one session (mis_unwind_vwap_revert_short, round_number_sweep_short, circuit_release_fade_short). Each retirement required updating 7 places: detector file, sanity script, universe builder function, config block, dispatch worker map, retired_setups.md, analysis/backtest_findings.md. Easy to miss one and leave dead code or stale config.

**Why:** Setup config is multi-referenced by design (dispatch refactor reduced this from 7 to 4 but still spreads). Half-retired setups create silent footguns — a future audit may resurrect a setup whose universe builder still exists but detector doesn't, or vice versa.

**Rule — retirement checklist (mandatory all 7):**
1. Delete `structures/<setup>_structure.py`
2. Delete `tools/sub9_research/sanity_<setup>.py` (UNLESS the sanity has anti-bias-fix work worth preserving — then KEEP and note in retired_setups.md "preserved as artifact")
3. Delete function `<setup>_universe()` in `services/setup_universe.py`
4. Delete `setups.<setup>` block in `config/configuration.json`
5. Delete entry in `services/dispatch/worker.py:_STRUCTURE_TO_SETUP_TYPE`
6. Add entry to `docs/retired_setups.md` — must include thesis, claimed validation, actual production result, failure mode + evidence, code files removed, preserved artifacts, conditions for revival
7. Update `analysis/backtest_findings.md` with corrected verdict + reference back to retired_setups.md

**Verify cleanly**: `python -c "import json; assert '<setup>' not in json.load(open('config/configuration.json'))['setups']"` + `grep -rE "<setup>" services/ structures/ | wc -l` should be ≤1 (only the comment in `setup_universe.py:compute_static_universes`).

### 2026-05-19 (#7) — FinalNet (Gross × MIS − Charges − Tax) is the only deployment-relevant metric

**What went wrong:** Reported PF 1.07 and "+Rs 5K net" for `round_number_sweep_short` and was about to retire. User pointed at `comprehensive_run_analyzer` output showing the full Indian fee + tax stack on the same setup: **−Rs 63K net (LOSING after charges).** Raw realized PF doesn't survive Indian fees — and `realized_pnl` × 5x MIS leverage further skews vs unleveraged charges if you size off raw R-multiples.

**Why:** Indian retail trading economics:
- MIS leverage multiplies PnL (both wins and losses) but charges scale on UNLEVERAGED notional × qty.
- STT (0.025% on sell-side for short equity), brokerage (Rs 20 capped per order × 2 legs), exchange fees, SEBI, IPFT, stamp duty, plus 18% GST on (brokerage + exchange + SEBI + IPFT) — total ~0.05-0.10% per round-trip.
- Income tax at 31.2% on NET annual speculative income (FY April-March, Section 73 — losses offset profits within same FY, net losses carry forward 4 years).
- A setup with PF 1.07 realized → typically PF_net ~0.85 after fees → -tax → break-even or losing.

**Rule:**
1. **NEVER report PF/profit using raw `realized_pnl`** when assessing ship readiness. Always run `comprehensive_run_analyzer` or use `tools/report_utils.calculate_per_trade_final_pnl()` which applies the full stack.
2. **Ship-gate PF threshold is PF_NET ≥ 1.15** (not realized). Tighter for setups with high trade frequency (charges scale linearly with trade count).
3. **Annual ROI on Rs 5L paper capital is the headline metric** for whether to deploy at retail-individual scale. Below ~50%/yr the setup is not worth the operational overhead.
4. **Per-setup FinalNet from analyzer Section 4B** decides which setups stay enabled. Setup with negative Avg/trade (Status: LOSS) gets retired unless decay is regime-only (war period).

### 2026-05-19 (#6) — Systematic debugging Phase 1: no fixes without root cause evidence

**What went wrong:** When OCI showed 34% hard_sl in production for `circuit_release_fade_short` while sanity showed 0%, I initially hypothesized "tick-level intra-bar SL checks fire on transients the 5m bar high doesn't record." User shut this down: "dont speculate." Followed `superpowers:systematic-debugging` Phase 1 properly — picked ONE specific overlap trade, looked at actual bar data, and discovered the truth: the 197 OCI-only trades had `day_high > morning_high` (post-morning new high) — sanity's `morning_high < day_high * 0.999` was hindsight filtering. Speculation would have wasted hours on wrong fix.

**Why:** "Tick-level execution" or "intra-bar volatility" is a generic-sounding explanation that pattern-matches lots of real-world bugs. It's easy to land on as a hypothesis. But Phase 1 evidence (one trade, walk the bars, see what differs) cuts directly to the real issue — usually something specific and unromantic (filter uses hindsight aggregate, off-by-one in path walk, etc.).

**Rule:**
1. **`superpowers:systematic-debugging` Phase 1 is mandatory before proposing any fix** — and "I think it might be X" counts as a fix proposal.
2. **Pick ONE specific data point** (one trade, one event, one bar) and trace it END-TO-END before generalizing. Aggregate stats lie about which mechanism is dominant.
3. **When stuck, the user's "stop guessing" / "don't speculate" / "look at the data" feedback is THE signal** — return to Phase 1 evidence gathering immediately.
4. **Hypothesis quality test**: can you predict (before checking) what specific bars/values will show? If "well, it's probably tick noise" — that's not a hypothesis, it's a wave-of-the-hand. A real hypothesis predicts a SPECIFIC measurable.

### 2026-05-19 (#5) — Sanity script anti-bias checklist (the 6 recurring failure modes)

**What went wrong:** Re-discovered the SAME sanity-script biases that retired_setups.md documents in its "common failure modes" section. mis_unwind sanity had 88% same-bar look-ahead. round_number sanity walked from `i+1` when entry was at `i+1.close`. circuit_release sanity used EOD `day_high` for morning-pin check.

**Why:** Sanity scripts are written quickly during research; each one adopts a convention slightly differently. Without an explicit guard checklist, the same six bugs recur.

**Rule — every new/edited sanity script must explicitly verify against `retired_setups.md` failure modes:**
1. **Intraday aggregate look-ahead**: NEVER use `day_high`, `day_low`, `day_vwap`, `day_close` as filters. Use `session_high_so_far = bars[:i+1].high.max()` at signal time. Rule of thumb: "what value would I have known at this exact bar's close?" — anything later is leakage.
2. **Volume baseline**: `cum_vol_mean = bars.volume.expanding(min_periods=2).mean().shift(1)` (current bar EXCLUDED) OR `prior_bars.iloc[:-1].volume.mean()`. Including the signal bar in its own baseline inflates the vol_ratio at the most volatile bars.
3. **Mode B entry walk**: `entry_idx = i + 1`, `path_walk = bars.iloc[i + 1:]` only AFTER the entry bar. If entry is at `bars[i+1].open`, walking starts at `bars[i+1]` is OK (entry happened at bar's open, the bar's full intra-bar range happens AFTER entry). If entry is at `bars[i+1].close`, walking must start at `i+2` (entry happened at end of bar i+1).
4. **Same-bar exit ambiguity**: when both `hi >= hard_sl` AND `lo <= t2_target` on the same bar, sanity must pick stop (pessimistic). Production tick path could go either way; sanity must not assume the favorable outcome.
5. **Cell-locked filters at signal time**: filter must reproduce in production exactly. If sanity uses `cum_vol_mean` and production uses `prior_bars.mean()`, results diverge silently.
6. **Reproducibility floor**: same trade IDs in (sanity, production) must produce same realized_pnl ±Rs 10. Per-trade-match diff is the audit tool when aggregate PF diverges.

### 2026-05-19 (#4) — Indian-market microstructure facts that drive setup design

**What went wrong:** Designed `sanity_mis_unwind_REAL_window.py` assuming "Zerodha auto-square at 15:20 IST means forced-sell window is 15:20-15:25." User pointed out Upstox/Angel auto-square at **15:15** — entire 10-minute window of MIS unwinding pressure starts 5 minutes earlier than my assumption. Set EXIT_BAR_HHMM accordingly only after the correction.

**Why:** Indian retail brokers have non-uniform MIS auto-square timing, and the user's mental model is "Indian retail flow happens during the union of all broker windows." Single-broker assumptions miss the full forcing function.

**Rule — Indian-microstructure facts to encode into any setup research:**
1. **Broker MIS auto-square timing (heterogeneous):**
   - Upstox: 15:15 (₹50+GST penalty for auto-squared positions)
   - Angel One: 15:15 (~₹50)
   - Zerodha: 15:20-15:24 (₹50+GST)
   - ICICI Direct: 15:15-15:20 (~₹50-100)
   - For setups that fade retail MIS-long flow: pressure window is **15:15-15:25**, not 15:20-15:25.
2. **MIS leverage**: SEBI minimum 20% margin = max 5x leverage. Retail brokers typically offer 4-5x for stocks, less for high-volatility names.
3. **NSE intraday volume profile** (J/U-shape from Monash NSE liquidity paper): 09:15-09:30 is 5x baseline, 11:00-13:00 is 1.0 (mid-day quiet), 15:00-15:15 is 1.8-2.0x, **15:15-15:25 is 2.5-3.0x** baseline (last bin includes closing auction).
4. **Retail concentration** is highest in small/mid-cap (<Rs 250 stocks especially). SEBI 2024 study: 76% of intraday traders under 30 lose money.
5. **Regulatory cutover dates that broke setups** (track in `data/sebi_calendar/`):
   - **Oct 1, 2025**: SEBI F&O rule changes (MWPL tightened, single-stock position limits cut) — broke `delivery_pct_anomaly_short`, `mis_unwind_vwap_revert_short`, `circuit_release_fade_short`.
   - **Apr 1, 2026**: STT hike (futures 0.02%→0.05%, options premium 0.1%→0.15%) — further fee compression.
6. **War period (Jan-Apr 2026)**: high realized vol hurt short-bias setups. PF aggregate dropped 1.46 → 0.88. Distinguishes regulatory decay (pre-war drop, structural) from regime-temporary decay (war-only, kept active for revival post-war).

### 2026-05-19 (#3) — Phase 1-5 disciplined research chain for new/revived setups

**What went wrong:** First attempt at "saving" `mis_unwind_vwap_revert_short` was ad-hoc cell sweeping on OOS data. Found `RSI≥85 + vol≥15 + SL=0.5 + T2=3.0` cell with PF 1.31 — looked shippable. User pushed back: "go through lessons.md… how we work on new setups first." Restarted with disciplined chain.

**Why:** Ad-hoc cell sweeps on OOS are data-mining (lessons.md 2026-04-22, 2026-05-01). The disciplined chain locks the methodology BEFORE seeing the data — that's the only path to a real edge claim.

**Rule — for any new or revived setup, run all 5 phases IN ORDER, NEVER skipping:**

**Phase 1 — Indian-market research (~15 min web/precedent):**
- Operating mechanism: what real-world Indian flow does this fade/catch? Cite ≥2 retail/pro Indian sources.
- Data feasibility (Gate B from lessons.md 2026-05-05): list exact data inputs; verify each is on disk OR has clear acquisition path.
- Regulatory sensitivity: which SEBI/STT/MIS rules govern the mechanism? Flag if a 2024-26 cutover affects it.

**Phase 2 — Empirical signature check on Discovery (~10 min compute):**
- Quantify the mechanism in raw data BEFORE writing the sanity. Volume bulge? Directional drift? Effect size?
- If signature doesn't exist or is weak (<0.1% net drift), abandon — no methodology will rescue a non-existent edge.

**Phase 3 — Mechanism brief (one sentence + falsifiers):**
- Mechanism statement: ONE sentence describing the captured edge with Indian-microstructure anchor.
- Falsifiers: 3 conditions that would invalidate the thesis (mechanism, regime, infra).
- Pro/retail precedent: ≥2 Indian sources operationalize this on retail-MIS infra.

**Phase 4 — Sanity v2 with explicit anti-bias guards:**
- Apply the 6-failure-mode checklist (#5 lesson). Document each guard in the script header.
- Discovery-only run: 2-yr Discovery window with locked filter set.

**Phase 5 — Cell mine on Discovery, lock cell, then OOS one-shot, then HO:**
- Cell sweep ONLY on Discovery. Lock the winning (filter × R-multiple) cell.
- Run OOS ONCE on locked cell. If passes ship gate (PF_net ≥ 1.10), proceed.
- Run HO ONCE on locked cell. If passes ship gate, write the brief + add to production config.
- If HO fails: classify (war-only collapse → pause; pre-war decay → retire).

**Total time: ~2 hours of disciplined work** (mostly async compute). NEVER take a shortcut.

### 2026-05-19 (#2) — Cell-mining illusion: post-hoc selection ≠ validated edge

**What went wrong:** Searching for a salvage cell in `circuit_release_fade_short` after Holdout collapse. Found `hour=12 + rejection_pct 0.4-1.0` survives HO_pre_war at PF 1.44. Almost shipped it. But this cell was SELECTED AFTER seeing HO data — if I'd locked on Disc+OOS independently, the best cell would have been **hour=13** (Disc 1.34, OOS 1.32). That cell broke HARDER in HO_pre (PF 0.53).

**Why:** Cell-mining is a confirmation tool, not a discovery tool. The same mechanism that produces look-ahead bias in features produces overfitting in cells: ANY large enough cell grid will surface "winners" by chance, and choosing AFTER seeing the data is post-hoc.

**Rule:**
1. **Lock cell on Discovery + OOS independently** — pretend Holdout doesn't exist when selecting. The cell that wins on Disc+OOS combined is the cell you test on HO. Not the other way around.
2. **If no cell with n≥200 + PF≥1.20 wins on Disc+OOS combined**, the setup is dead. Don't dig further — that's how illusions are born.
3. **Post-hoc cell selection is structurally equivalent to p-hacking**. The "hour=12 survives HO_pre" finding has the same epistemic status as "let me try one more conditioner" (2026-05-01 lesson).
4. **3D cell sweep with PF gate**: across all (cap × time × feature) intersections with n≥20 each in ALL 4 periods (Disc, OOS, HO_pre, HO_war), require PF_net ≥ 1.10 in each. If zero cells pass, retire. circuit_release: zero. round_number: zero. mis_unwind: had one but HO killed it.

---

### 2026-05-12 (#3) — Plan-as-source-of-truth is binding architecture; the gate chain accumulated cruft that anti-selected our setups

**What went wrong:** The system had a 5-stage filter chain after detector accept:
1. Stage-0 top-K (drops to top-1000 by intraday momentum — anti-selecting for our setups)
2. RuleFilter (admit-all, no-op)
3. CrossSectionalGate (RVOL + crowdedness — partly useful, partly redundant)
4. ConvictionGate (XGBoost — rejected 90% of profitable trades; HIGH conv +50% per-trade but ~7x absolute profit lost)
5. DedupGate (per-symbol cooloff — useful)

The chain accumulated incrementally. Each gate solved a specific case at the time, but
together they nullified the per-setup edge our sweeps proved. Smoke showed 8 detector
accepts → 1 final admit on 2024-08-29 (87% rejected).

Plus the executor had its own override forest (eod_scale_out, score_drop, generic
time_stop, sl_time_widening) that overrode plan-owned exits — the trade that DID fire
exited at 15:00 via eod_scale_out, not at the plan's locked 13:00 time_stop.

**Rule:**
1. **Each setup config is the SOLE authority** for: entry conditions, SL, targets,
   qty_pct per target, time_stop_hhmm, trail, priority, capital_budget_pct,
   max_concurrent, cooloff, rate-limit.
2. **Executor honors ONLY plan fields** — no global overrides for time-stop or partial
   qty. Global EOD remains as a hard safety cap (MIS auto-square), but plan's earlier
   time_stop wins.
3. **Universe-union is the shortlist** — Stage-0's intraday momentum ranking can't see
   cross-day signals (circuit hits, earnings, delivery anomalies, gaps from PDC), so
   it anti-selects for these setups. Each setup declares its qualifying universe;
   union is the per-bar shortlist.
4. **Bar-level scheduling resolves capital contention** via explicit priority
   (setup_priority × detector.quality_score / 100). When capital is tight, high-PF
   setups get first dibs. No black-box ML deciding which trades survive.
5. **ConvictionGate's XGBoost may have value, but as a SCORER not a FILTER** — to
   re-introduce, multiply plan.priority by predicted_r. Never use as a hard reject.
   And retrain on POST-refactor data, not stale sub7+sub8 trades.

**Lines saved by the refactor:** ~2000 LOC deleted (gate_chain/, cross_sectional/,
conviction gate+scorer, executor exit cruft). System is leaner, plan is genuinely
authoritative, model retraining is now optional (not a precondition for shipping).

---

### 2026-05-12 (#2) — Detector implementation must EXACTLY mirror validated sanity logic — silent exit-logic divergence destroyed circuit_t1
**What went wrong:** `circuit_t1_fade_short` sanity script (`tools/sub9_research/sanity_circuit_t1_fade_short.py`) validated a SINGLE-EXIT mechanic: enter short at T+1 10:30, ride to T2 (= t0_close = full gap fill) OR stop OR time-stop 15:10. No T1 partial, no breakeven trail. Reproduced PF=1.404 (Discovery) / 1.890 (Holdout). But the **production detector** + executor added a T1 partial at t1_open (`t1_qty_pct=0.5`) plus `exit_t1_book_pct` execution logic plus breakeven-trail-after-T1. Sweep finding (240 combos × 585 Discovery trades): production's `partial_50_be_trail` mode drops PF from 1.34 → **0.49** on the same trades. The 41% of trades that hit T1 partial then bounced into the BE-trail SL would have ridden to T2 in the validated mechanic. Production was actively losing money on its own validated trade set.

**Root cause discovery path:** User pushed me to verify SL/target sweep on the 3 "prod-ready" setups. I wrote `_circuit_t1_sl_target_sweep.py` that replays the trigger CSV through alternative SL/target/partial-mode params. The sweep instantly showed: sanity baseline reproduces PF=1.34, production current = 0.49. This kind of defect cannot be caught by aggregate PF alone — the production parquet PF=1.40 numbers I saw earlier came from the SANITY script outputs, NOT the production OCI captures (which were wide_open and don't isolate the real production exit logic).

**Architectural finding:** The executor's `_partial_exit_t1` reads `self.t1_book_pct` from execution config (line 1804) which is a SINGLE GLOBAL value applied to every setup that emits a T1 target. Setting `t1_qty_pct=0.0` in a per-setup detector config does NOT prevent T1 partial because executor ignores the plan's per-target qty_pct. Fixed in this commit by modifying `circuit_t1_fade_short_structure.py` to emit ONLY T2 (no T1 entry in the targets list) when `t1_qty_pct <= 0.0`, which is what the executor checks.

**Rule:**
1. **Sanity script defines the validated mechanic** — production detector + executor TOGETHER must reproduce sanity PF on the same trigger set, EXACTLY. Acceptance: ±2% PF drift, no more.
2. **Pre-ship reproducibility test**: feed each shipped setup's exact trigger conditions into local sanity-mode + production-mode and compare net_pnl per trade. Differences > Rs.10/trade are a defect.
3. **Per-setup partial/exit logic must be opt-in, not opt-out** — when a sanity validates single-exit, production must NOT add T1 partial just because the executor supports it. The detector emits exactly the exit ladder the sanity validated.
4. **SL/target sweep is the audit tool** — any setup shipped before its SL/target sweep produced concrete numbers is a candidate for hidden defects. Run the sweep retroactively on every shipped setup.

### 2026-05-12 — Production ship requires reproducible full validation chain — Discovery cell-mining + OOS + Holdout, all preserved
**What went wrong:** Multiple sub9 setups were shipped with incomplete validation:
- `capitulation_long_morning` (shipped 2026-05-07): Cell-mining script was never preserved (claim `n=443 PF 1.238` exists only in commit message); OOS validation was explicitly deferred (`will re-validate at OOS`) and never done; Holdout was captured but never cell-level-analyzed; production detector used wrong regime classifier (ADX-based) vs cell-mining used per-symbol close<SMA20 T-1; mass-enabled commit `98f520d` flipped `enabled=true` without the deferred OOS step ever being completed. Reproducibility audit on 2026-05-12: cell at per-sym SMA classifier gives Discovery PF 1.290 (close to claim) but **Holdout PF 0.631 — FAILS decisively**.
- `circuit_t1_fade_short`: Discovery PF 1.404 (only 11mo 2024, not full 2yr), OOS **PF 0.982** (below 1.10 ship gate), Holdout PF 1.890 — regime-dependent, fails OOS but recovers in war-period Holdout. Shipped on Discovery+Holdout despite OOS failure.
- `expiry_pin_strike_reversal`: Discovery PF 0.568, OOS PF 0.736, Holdout PF 0.508 — **loses money in ALL periods**. Should never have shipped. Currently silent in production because OI data is not loaded.
- `gap_fade_short`: HAD full validation chain (sub7 Discovery PF 1.153, sub8 Phase-1 v1/clean/v2/v3 all in 1.118-1.227 range, OOS PF 0.993 [marginal], Holdout PF 1.128). Initial audit incorrectly reported no OOS/Holdout because validation artifacts live in `reports/sub8_oos_*/`, not the sub9 directory pattern. **Audit lesson**: always check ALL `reports/` subdirectories for a setup's history, not just the latest sub-pattern; legacy setups span multiple sub-projects.
- `delivery_pct_anomaly_short`: shipped with claimed full chain (Discovery 1.44, OOS 1.90, Holdout 1.13) but failed Phase-1 OCI capture → retired.

**Contrast — `earnings_day_intraday_fade` (shipped 2026-05-12)** went through proper rigor: Discovery PF 1.64 + OOS PF 1.53 + Holdout PF 1.25 all reproducible; T1×T2×SL 3D sweep done with documented script and 45 cell outputs; production detector code reproducible against sanity; cell-mining methodology preserved.

**Why:** "Shipped" was being used to mean "merged a detector and flipped enabled=true" rather than "passed Discovery + OOS + Holdout with reproducible evidence." Commit messages contained claimed numbers that couldn't be reproduced from any preserved artifact. Production detector code drifted from the validated cell-mining code in subtle ways (regime classifier, T1/T2 sweep choices) without anyone noticing because no end-to-end reproducibility test was required.

**Rule — production ship gate (binding):**
1. **Cell-mining script must be preserved** in `tools/sub9_research/_cell_mine_*.py` with output saved in `analysis/_cell_mine_*.log`. If you can't point at the script that produced the cell's claimed PF, the cell isn't shipped.
2. **T1/T2/SL R-multiplier sweep is mandatory** with output saved in `reports/sub9_sanity/<setup>_t1_*_t2_*_sl_*/summary.json`. The shipped R combo is the one that maximizes the cell's PF on Discovery, not arbitrarily chosen defaults.
3. **Discovery + OOS + Holdout must ALL pass** at the locked cell + R config:
   - Discovery: n ≥ 200, net PF ≥ 1.20, Sharpe > 0.5
   - OOS (Jan-Sep 2025): net PF ≥ 1.10 with no major-cell collapse vs Discovery
   - Holdout (Oct 2025-Mar 2026): net PF ≥ 1.10 with no major-cell collapse vs OOS
4. **Reproducibility check**: production detector's filter logic (regime classifier, cap_segment, ADV band, T1/T2/SL) must be byte-identical to what the cell-mining used. If you change the regime computation, you must re-mine the cell.
5. **TODO debt is not allowed at ship time**: commit messages saying "will re-validate at OOS" + `enabled=false` are NOT a valid path to production. Either complete OOS before merge, or don't merge.
6. **No mass-enable commits without per-setup audit**: a commit that flips `enabled=true` on multiple setups must explicitly document the OOS + Holdout passes for each.

**Process check before any future `enabled=true`:**
- [ ] Cell-mining script committed + output file in `analysis/`
- [ ] R-multiplier sweep script committed + 9+ cell outputs in `reports/sub9_sanity/`
- [ ] Discovery sanity reproducible (script can be re-run, exact PF reproduces)
- [ ] OOS validation done with `--allow-oos` flag, output in `reports/sub9_iv_rank_oos/`
- [ ] Holdout validation done, output in `reports/sub9_iv_rank_holdout/`
- [ ] Production detector regression test against sanity (same trade IDs, same PnL)
- [ ] No regression in `tests/structures/` suite

### 2026-04-22 — Never run architecture/variant comparisons on OOS data; reserve OOS for ONE confirmation check on the chosen winner
**What went wrong:** After the chained gauntlet showed Stage 5e (Budgeted Selector) beat Stage 5d (FIFO) on Discovery 2023-24, I ran a head-to-head on 2025 OOS. Then, when the user asked to also test a simplified variant (5e without ADV/bar/rate/concurrency caps), I started to run the variant comparison ALSO on 2025. User caught it: "why not on 23 24 data? isn't it the whole point? also the split checks? shouldn't we do all this on 23 24 data only?"
**Why:** This is the standard ML train/validation/holdout discipline. Every evaluation on OOS data consumes its informational value. If you keep running variants on OOS until one looks good, you've implicitly overfitted selection choices to the "holdout" set and lost the ability to claim OOS generalization. The correct sequence is: all architecture + hyperparameter + variant selection happens on IN-SAMPLE data, then you validate the ONE chosen winner on OOS (single check, no shopping), then final holdout gate before deploy.
**Rule:**
1. Architecture decisions / variant comparisons / hyperparameter tuning → IN-SAMPLE only (Discovery 2023-24 in this project)
2. Validation period (2025 Q1-Q3) → used ONCE per major design decision to confirm the chosen architecture generalizes. Not for A/B variant shopping.
3. Holdout period (Oct 2025 - Mar 2026) → untouched until final deploy gate
4. Before running ANY backtest variant on validation/holdout data, ask: "am I choosing between options here, or confirming a pre-chosen winner?" If choosing, you're on the wrong dataset.
5. If you have a bunch of variants to compare, build a variant-comparison script that runs them ALL on in-sample in one pass, not one-at-a-time on OOS.

### 2026-04-15 — Don't ask permission for fixes you've already identified
**What went wrong:** I identified a quality issue (canonical research drifting to recycled generalities across Tasks 8-10 audit Item 1), listed 7 specific missing research points, then asked the user "A/B/C — which do you want?". User called this out: "if you've identified it as a problem and know the fix, just fix it — don't ask permission."
**Rule:** When I identify a real problem AND know the right fix AND the scope is clearly within the current task, just DO the fix. Asking "A/B/C" when the honest answer is obviously A wastes the user's turn and signals ceremony-over-substance. Save questions for genuine decisions where multiple paths are defensible.

### 2026-04-15 — Don't shortcut audits for low-trade-count detectors
**What went wrong:** In Task 17 (SqueezeReleaseStructure, 46 trades) I wrote a significantly abbreviated audit doc — compressed Item 1 canonical research, condensed Items 3-7 into single-line summaries, reused "same pattern as prior detectors" framing. User pushed back: "dont assume anything... as u saw there are bugs due to which maybe trade count were low... do it as u did for high priority structures only... no shortcuts". The point: low trade counts might BE THE BUG — if detector has silent failures, it will APPEAR low-trade. Abbreviated audit = missed bugs.
**Rule:** Every detector audit (regardless of trade count) gets the SAME rigor as ICT/Range/LevelBreakout:
- Full detector-specific NSE canonical research (≥5 insights specific to THIS detector, not recycled)
- Full code read — not assume "same pattern" without verifying
- Item 2 comparison table with every relevant aspect
- Items 3-7 with full PASS/FAIL + file:line refs
- Proper disposition proposal with alternative considerations
Low trade count is a SIGNAL to look harder, not easier. "It's rare so skip detail" is exactly backwards.

### 2026-04-15 — Don't defer to pre-rebuild evidence; let the new gauntlet decide
**What went wrong:** For VolumeBreakoutStructure (rank 07, 0 trades in current backtest because config-disabled), I jumped to DISABLED based on OLD evidence: "491 historical trades = -0.88L, only profitable in trend_down regime." User correctly pushed back: that was BEFORE the rebuild, before our audit-driven detector fixes, before our OOS protocol. We're regenerating the wide-open backtest anyway — Stage 1 of the gauntlet (PF ≥ 0.8 on Discovery period + N ≥ 500) will make the call.
**Rule:** When auditing a detector for the first time under the new OOS protocol, base disposition on (a) canonical structural soundness (Item 1), (b) code quality (Items 3-6), (c) documented bugs that affect detection accuracy. Do NOT defer to pre-rebuild PnL evidence — that data was generated by older / overfit / unfixed code under no OOS discipline. If the structural foundation is sound and code is clean, ENABLE the detector for the wide-open backtest and let the fresh gauntlet decide. Pre-rebuild evidence is documentation of what was tried, not authoritative ground truth.

### 2026-04-19 — Always suggest the correct solution, not the easy solution
**What went wrong:** When the OCI Docker image needed a rebuild for entrypoint.py changes (gzip-on-upload), I proposed Option C: "move gzip into the Python trading code so it ships via the code tarball — no Docker rebuild needed." This is architecturally wrong. Compression of logs before upload is an infrastructure concern that belongs in entrypoint.py (the deployment layer). Moving it into the application layer would mix concerns — the trading system shouldn't know about OCI Object Storage upload optimization. The correct answer was always Option A (rebuild the Docker image), which I should have stated directly without offering the shortcut.
**Why:** This is the production-mindset lesson. Shortcuts that save 10 minutes now create second code paths, mixed concerns, and maintenance debt. The user's standard is: suggest the RIGHT solution, not the EASY solution. If the right solution requires more work (Docker rebuild), say so and do it.
**Rule:** When proposing solutions, evaluate them on architectural correctness FIRST, implementation convenience SECOND. If one option is clearly the right architecture (separation of concerns, single responsibility, proper layering) but requires more work, recommend THAT option directly. Don't offer shortcuts that pollute clean boundaries just because they avoid a step. "No shortcuts" applies to infrastructure decisions, not just code.

### 2026-04-21 — No payroll, no deadline. Correctness beats shipping speed.
**What went wrong:** While brainstorming sub-project #2 (Conviction Architecture), I proposed a Phase 1/Phase 2 sequencing: "ship a dummy-scorer scaffolding first so we have a baseline, then build ML v2 as a drop-in replacement." User pushed back: "we are not on any payroll here to ship it... its about getting it to work... so u need to put in ur lessons.md... only my money is on the line right now so going live fast is not as important as getting it correct."
**Why:** I was implicitly optimizing for a VC-backed-startup ship-fast-iterate pattern. This is the user's personal capital — the priority is CORRECTNESS, not velocity. Building placeholders / dummy scorers just to hit phase gates creates throwaway code and delays the real work of getting the actual solution right. Shipping v1 early has no value when there's no deadline; the only value is "it works and I can trust it with my money."
**Rule:** When I identify a design approach and I'm tempted to propose "let's ship a simpler version first and upgrade later," ASK first whether speed is a real constraint. For personal-capital projects with no deadline, the right sequence is:
1. Build the CORRECT design immediately (not a placeholder)
2. Take time for proper feature engineering / validation / SHAP interpretation / OOS discipline
3. Don't use "phase gates" as an excuse to ship something less-right
4. Shadow-trade / paper-trade before live is mandatory regardless of speed
5. Integration scaffolding gets built AROUND the correct design, not as a placeholder waiting for the real design

Shortcut patterns to avoid: "v1 baseline to measure against," "ship scaffolding first with dummy internals," "heuristic to bootstrap ML." All legitimate in deadline-driven work, ALL wrong when the constraint is correctness.

### 2026-04-21 — Don't offer shortcut options alongside the correct fix
**What went wrong:** When Stage 2 of the edge gauntlet killed all 5 PF-consistent setups via Sharpe ≥ 0.7 (because we compute Sharpe per-trade, not per-session, making the gate structurally unpassable for intraday), I presented three options to the user: (1) lower the Sharpe threshold to ~0.08, (2) change the metric to daily/session Sharpe, (3) accept 0-survivors. Option 1 is threshold-tuning to hide a design bug — a shortcut. Option 3 abandons real findings. Only option 2 is the correct fix. User responded: "its obviously 2... u dont read from lessons.md? no shortcuts we are going for the correct fix always fucker."
**Why:** Same production-mindset lesson as the Docker-rebuild incident (2026-04-19). Offering a shortcut alongside the correct fix dilutes the recommendation and wastes the user's turn. When one option is clearly the right architecture/methodology and the others are thresholds-moving or giving up, don't enumerate — just state the correct fix and do it.
**Rule:** When I identify a methodology or design bug, the ONLY option I present is the architecturally-correct fix. Don't list "lower the threshold" as an alternative to "fix the metric definition". Don't list "accept failure" as an alternative to fixing a bug. Options are for genuine tradeoffs with multiple defensible answers, not for hiding a shortcut next to a correct answer.

### 2026-04-15 — Canonical research must be detector-SPECIFIC, not recycled NSE generalities
**What went wrong:** In detector audits 8-10 (SupportResistance, Momentum, LevelBreakout), my Item 1 canonical research increasingly leaned on recycled observations from prior audits (time windows, cap-segment weakness, retest preference). User pointed out "I don't see any research-driven points since last 2-3 detector audits."
**Rule:** For each detector audit Item 1, generate AT LEAST 3-5 detector-SPECIFIC research insights that couldn't apply verbatim to prior detectors. Generic NSE microstructure knowledge is fine as context but must not be the primary content. Specifically research: (a) pattern-specific WR / continuation statistics, (b) pattern-specific NSE timing windows, (c) pattern-specific cap/regime interactions, (d) pattern-specific parameter sensitivity ranges from pro community knowledge. If the detector is simple (e.g., MomentumStructure), acknowledge the research budget is limited rather than padding with recycled content.

### 2026-04-29 — Don't dispatch one autonomous agent for many trading-logic tasks
**What went wrong:** User asked me to "code all 6 plans" (~120 tasks across 6 detector implementations). I dispatched a single general-purpose implementer agent to autonomously execute Phases 0+1+2 (32 tasks) of the orb_15 redesign in one shot, then planned to do the same for the other 5 plans sequentially. User stopped me: "no shortcuts... treat each and every code change u make as important..."
**Why:** Trading-logic code is production-critical. Bulk-delegating 32 tasks to a single agent — even with TDD discipline in the prompt — means none of those changes get the careful review they deserve. The agent might pass tests yet introduce subtle bugs (wrong threshold semantics, off-by-one in the latch, regime-allowlist logic inverted, etc.). The "fresh-agent-per-task + 2-stage review" pattern in `superpowers:subagent-driven-development` exists exactly for this: each task gets implemented by a focused agent, then reviewed for spec compliance, then reviewed for code quality, BEFORE moving to the next. Letting one agent burn through 32 tasks autonomously is the shortcut.
**Rule:**
1. For trading-logic implementations, NEVER dispatch one agent to execute multiple consecutive tasks autonomously.
2. The right pattern is one of:
   - **Per-task subagent-driven-development** (project's standard): fresh implementer per task → spec reviewer → code quality reviewer → user-visible commit → next task
   - **I implement myself**, task by task, reading every file before changing, running tests after every change, committing per logical unit
3. "Treat each code change as important" means: read every line, verify against CLAUDE.md rules (no hardcoded defaults, IST-naive, live/backtest compat), verify against the plan, run tests, only THEN commit.
4. If the agent has already done partial work, DON'T just accept its diff — read every line, verify, fix, then commit (or rewrite). Reviewing carefully is acceptable; rubber-stamping is not.
5. Estimate scope realistically — "6 plans, ~120 tasks" is days of careful work, not one autonomous afternoon. Tell the user so they can decide cadence.

### 2026-05-01 — Period scheme is project-state-dependent, not derived from master plan calendar
**What went wrong:** Asked to run the gauntlet on sub7/sub8 candidates over the 2-year OCI capture (2023-01-02 to 2024-12-31). I assumed master plan §3.2 calendar applied unchanged (Discovery=FY22-23+FY23-24, Validation=FY24-25). On that basis I attempted to "discipline" by carving out FY24-25 from analysis. User correction: "we r considering 23 24 data as validation now". The project state had moved on — for sub7/8, 2023-24 is **validation** because (1) sub-project #5's gauntlet-v2 already used 2023-24 as Discovery for the SMC library and exhausted it on Optuna search, and (2) sub7/8 detectors were research-driven (design IS in-sample work), so 2023-24 is OOS for them.
**Why:** The master plan calendar is the DEFAULT for sub-project #1; subsequent sub-projects can REASSIGN periods based on what's been spent + what the new sub-project's "in-sample" actually was. Treating the master plan calendar as immutable across sub-projects is wrong. The current period scheme lives in project state (which the user holds), not in the master plan doc.
**Rule:**
1. Before running ANY OOS-disciplined analysis on a new sub-project, ASK what the period scheme is for THIS sub-project (Discovery/Validation/Holdout boundaries). Don't infer from master plan calendar.
2. If a sub-project's "in-sample" is research/design (not data-mining), there is no Discovery DATA — the design doc IS the in-sample work, and any captured data starts at Validation.
3. Track period assignment per sub-project in a versioned doc; don't rely on the master plan + memory.
4. Once the period scheme for THIS sub-project is confirmed, execute the gauntlet ONCE on the right slice. Don't iterate.

### 2026-05-01 — Validation is one-shot; iterating the gauntlet on validation data spends multiple shots
**What went wrong:** With 2023-24 properly identified as validation for sub7/8, my 4 successive gauntlet iterations (v1→v5: added side, intended-universe filter, unconditional 2-way, volatility_regime) on the same 2023-24 data spent the validation budget 5 times. Master plan §5 disciplines: "ONE SHOT at Validation. Failed rule = dead OR redesign from Stage 1. No tuning to pass." Each iteration was effectively re-validating with a different config to find passing cells.
**Why:** Validation is supposed to be a single binary measurement on a frozen rule produced by Discovery (or by the design itself for research-driven sub-projects). Iterating the cell-selection criteria on validation data — even when each individual change is master-plan-approved — is the same as tuning a Discovery rule on validation data.
**Rule:**
1. For sub-projects where 2023-24 = validation, the gauntlet config MUST be locked before the first run on 2023-24. Locking happens via the design doc + research-justified additions, not via "let's see what works".
2. If the validation gauntlet produces 0 passing cells, that's the final answer for the candidate setup library on this period. The validation budget for 2023-24 is then SPENT.
3. Re-running with different conditioners or thresholds after seeing the result = budget already spent. Honest accounting: you don't have validation evidence anymore on the same period.
4. If a future iteration of the setup library is desired, capture FRESH validation data (e.g. 2025-26) and design that library's gauntlet config UPFRONT before touching the new data.

### 2026-05-01 — Don't iterate gauntlet config in response to results — that's variant-shopping
**What went wrong:** The first run of the Phase-1 gauntlet found 0 passing cells across 6 sub7/sub8 candidates. I proposed an "update": added `side` as conditioner, made 2-way unconditional, added intended-universe filter, then added `volatility_regime`. After 4 iterations (v1→v5), 2 passing cells emerged. Each individual change was research-defensible (Indian-asymmetry, master-plan conditioner, etc.), but the AGGREGATE pattern is variant-shopping: "found 0 → expand search space → found 2". Master plan §3.2: "No retroactive criteria changes. Criteria locked before run." Lessons.md 2026-04-22: "build a variant-comparison script that runs them ALL on in-sample in one pass, not one-at-a-time." I ran them one-at-a-time, each iteration informed by the previous result.
**Why:** Adding conditioners after a "no passing cells" result is the same as lowering a threshold after seeing the data. Even if every conditioner is master-plan-approved, choosing WHICH ones to include AFTER seeing the failure is implicit shopping. The disciplined path is: lock the conditioner set + criteria UPFRONT (based on master plan + research, before running), execute once, accept the result.
**Rule:**
1. Before the first gauntlet run on a setup library, write down (in commit message or design doc) the EXACT conditioner set + threshold values that will be used. Lock it.
2. Conditioner additions justified by research (e.g. "side per gauntlet-v2 postmortem", "volatility_regime per master plan §3.3") MUST be locked BEFORE the gauntlet runs, not after seeing the result.
3. If a gauntlet run produces 0 passing cells, accept that as the answer. Don't re-run with more conditioners hoping cells appear.
4. Stage 4 SHAP IS the master-plan-sanctioned path to add conditioners — if SHAP reveals a missed structural driver, add it AND re-run Stage 3 ONCE. Don't iterate beyond that single re-run.
5. Variant-shopping looks like productive iteration but is statistically equivalent to p-hacking. The "found 0 → expanded → found 2" pattern is the tell.

### 2026-05-01 — Stop offering 4-option lists at the end of every response
**What went wrong:** Across the gauntlet session, every response ended with "Want me to: 1. X, 2. Y, 3. Z, 4. W?" Lessons.md 2026-04-15 ("Don't ask permission for fixes you've already identified"), 2026-04-19 ("Always suggest the correct solution"), and 2026-04-21 ("Don't offer shortcut options alongside the correct fix") all say the same thing: when the right next step is identifiable, just propose it; don't enumerate alternatives. User pushback: "u r fucking this up... u hv not gone through lessons.md or the master plan."
**Why:** Enumeration is a habit signaling ceremony-over-substance. It creates the impression of optionality where there isn't real ambiguity. It also burns the user's turn on a meta-question (which option) instead of progressing the work. Three lessons say the same thing — repeated user pushback means I haven't internalised them.
**Rule:**
1. When the master plan or research clearly identifies the next step, propose ONE concrete action and either execute or ask for go-ahead.
2. Reserve option lists for genuine forks where multiple paths are defensible AND the user has unique context to decide.
3. Before posting an option list, ask: "Could I just identify the correct option and propose it?" If yes, do that.
4. After ANY response that ends with "1. X, 2. Y, 3. Z, 4. W", flag it internally as a lesson violation and edit before sending.

### 2026-05-05 — DATA feasibility check too: does on-disk data support the sanity check?
**What went wrong:** Drafted §3.3 brief #4 (oi_long_buildup_continuation) under the new precedent-first protocol. Brief specified "Leg 1: stock-future OI increased ≥ +1.5% from 13:00 to 13:30 (30-min window)." Only AFTER user approval, when starting the sanity-check tool, I checked what data was on disk: 5m enriched feathers have no OI column, option_chain parquets are EOD option OI (not stock-future, not intraday), F&O bhavcopy is EOD daily snapshot (cannot interpolate intraday velocity). The signal the brief specified cannot be tested with available data. The brief's casual "F&O bhavcopy + interpolates intraday OI" claim was wrong — bhavcopy is single-snapshot per day.
**Why:** The 2026-05-05 precedent-feasibility lesson covers WHETHER pros trade this. It doesn't cover WHETHER WE CAN TEST IT LOCALLY. Both gates are needed before a brief is committable: precedent (does it exist?) AND data feasibility (can we sanity-check it on what we have?). Drafting a brief whose sanity requires data we'd need to spend a week capturing is the same waste class as drafting a brief for a setup pros don't trade.
**Rule:**
1. Before drafting a §3.3 brief, run TWO feasibility gates:
   - **Gate A — precedent**: ≥2 distinct Indian retail/pro intraday-MIS algo sources operationalising it (per 2026-05-05 lesson)
   - **Gate B — data**: list the EXACT data inputs the sanity check will read; verify each is on disk OR has a clear acquisition path that fits within the sanity-check budget (1-2 hours, NOT 1-2 weeks).
2. If Gate B fails, EITHER (a) defer the brief until the data acquisition is a separate completed task, OR (b) restructure the brief to use alternative data we already have. Do NOT speculate ("we'll interpolate from bhavcopy") without verifying the speculation.
3. Concrete check before drafting: 5-min `Bash` command to verify the data file/columns exist. E.g. for circuit_t1 we already had consolidated_daily.feather; for nifty500_deletion we had nothing programmatic but accepted manual curation; for oi_long_buildup we have NEITHER intraday OI ticks NOR a curation path.
4. The brief's "Data engineering plan" section is for the production detector (post-sanity), NOT the sanity itself. The sanity tool MUST work with on-disk data without new captures. If it needs new captures, it's not a sanity tool — it's a data-engineering project pretending to be a research check.
5. Add to the round-2-style asymmetry research feasibility doc: a "Gate B — data on disk?" column alongside the precedent gate. A "no" downgrades the candidate to DEFERRED until data acquisition is scheduled.

### 2026-05-05 — Run feasibility / pro-precedent check BEFORE drafting a §3.3 brief
**What went wrong:** Promoted candidate G (NIFTY 500 deletion short) off the CONDITIONAL list, drafted a 214-line §3.3 brief (`specs/2026-05-03-sub-project-9-brief-nifty500_deletion_short.md`), got user approval, started designing the sanity-check tool — and only THEN searched for whether pro Indian intraday algos actually trade this asymmetry. They don't. Pros who trade index rebalancing operate in modes incompatible with our infra (index arb desks, multi-day positional CNC, Tier-1 quant prop with Bloomberg feeds). Zero hits in Indian retail/pro intraday algo literature. The brief was wasted; the answer was already determinable in 5 min of search. User: "then why the fucking are we wasting time on these? dont u do feasibility research before going through all this fucker"
**Why:** A peer-reviewed academic asymmetry can be REAL but **infrastructurally infeasible** for our scale (intraday MIS, retail-size, no paid data feeds, no SLB). Skipping the "do pros operationalize this on infra like ours?" gate means writing a brief for a setup that's structurally a non-starter. Even the brief's own falsification criterion #3 ("real edge wrong infra") was prescient — but it was placed AFTER sanity, not BEFORE brief.
**Rule:**
1. Before drafting any §3.3 brief, run a 5-minute feasibility-precedent check: search Indian quant blogs / Github / paid-platform docs (uTrade, PL Capital, Zerodha Varsity, StockEdge, TickerTape, public Indian quant Github) for retail/pro algos using THIS asymmetry on infra COMPATIBLE with ours (intraday MIS, retail-size, no paid feeds).
2. If no precedent at our scale: state that openly, propose either (a) RETIRE the candidate, or (b) defer until the infra it needs (CNC/SLB/paid feeds) is built. **Do NOT write the brief.**
3. The asymmetry research findings doc (sub-9 §2.5) needs an additional column: "operationalized at retail-intraday scale by pros (yes/no/unknown)". A "no" downgrades the candidate to DEFERRED before §3.3 stage.
4. Peer-reviewed evidence is necessary but not sufficient. **Real-world precedent at our infrastructure is the second gate.** A setup that academia validates but pros don't trade is either (a) recently-decayed, (b) requires infra we lack, or (c) execution-arb-only edge. None are workable for us.
5. Sanity-check budget is precious. Spending it on a candidate that fails the feasibility-precedent gate is the same waste class as variant-shopping on validation data (lessons 2026-04-22 + 2026-05-01).

### 2026-05-19 — Discovery+OOS parity is necessary but NOT sufficient when both predate a regulatory cutover

**What went wrong:** I re-validated retired `mis_unwind_vwap_revert_short` by sweeping cells on OOS data, finding a "shippable" cell (RSI≥85 + vol≥15 + SL=0.5 + T2=3.0). To test for OOS-overfit, I ran the same locked cell on Discovery (2023-24). Discovery PF_net 1.213 vs OOS PF_net 1.216 — near-perfect parity (-0.003 drift). This looked like genuine, stable edge across 2.75 years. I declared the mechanism "real" and proposed shipping.

The Holdout test (Oct 2025 → Apr 2026) collapsed PF_net to 0.751 — catastrophic. Same trajectory as previously-retired `delivery_pct_anomaly_short`.

**Why:** Discovery + OOS both PREDATE the SEBI Oct 1, 2025 rule changes (MWPL tightening, F&O position limits cut). Both reflected the SAME pre-SEBI retail MIS-positioning regime. Their parity was confirming "the regime is consistent across 2.75 years" — which it was — but NOT "the mechanism survives the regulatory break." Holdout was the only honest test of post-SEBI viability, and it failed.

This is also the SAME failure mode I'd documented in the post-SEBI research brief (`specs/2026-05-14-research-post-sebi-edges.md`): "Regulatory regimes break setups silently. Gauntlets must NEVER straddle a high/critical rule change for any of the strategy's depends_on tags." I had the warning but didn't apply it to my own validation pipeline.

**Rule:**
1. **When Discovery + OOS both predate a known regulatory cutover, Disc↔OOS parity proves regime-internal consistency, NOT regime-transition robustness.** The "validation" claim must include the cutover.
2. **For any setup with `regulatory_sensitivity != rule_orthogonal` OR `depends_on` including post-cutover-sensitive tags (MIS_leverage, F&O OI, STT_drag, mis_auto_square, dpr_circuit, etc.):** require Holdout PF gates as STRICTLY as Discovery — Holdout PF_net >= 1.10 is binding, not informational.
3. **Run `services.regime_break_detector.check_window` BEFORE any Disc+OOS celebration.** If the window straddles a cutover, the parity result is informational only — never sufficient for ship.
4. **Look at depends_on tags explicitly:** mis_unwind had `depends_on=["MIS_leverage", "STT_drag", "mis_auto_square"]` — ALL THREE were affected by 2025-26 SEBI changes. I should have predicted this from the depends_on tags alone.
5. **The "shippable Discovery+OOS cell with similar PF" finding is a known overfit pattern under regulatory regimes**, not validated edge. Don't celebrate parity until Holdout post-cutover confirms.

### 2026-04-29 — Commit per logical phase, not per task
**What went wrong:** While implementing pdh_pdl_sweep_reclaim with manual per-task discipline, I made 14 commits for one plan (one per task: config block, register category, sub8_oci, opening_bell, scaffold, fixture, state machine, mirror test, negative tests, gap-context, multi-day, plan_*_strategy, wide_open, register, wire). User pushed back: "this is not best way to commit". Compare to orb_15 which was 3 commits (Phase 0 + Phase 1 + lessons) and was cleaner.
**Why:** Per-task commits create noisy git history. A reviewer reading `git log --oneline` for one plan sees 14 lines of "Task 1.4 / Task 1.5 / Task 1.6" instead of 3 meaningful units. Bisect / revert / squash all become harder. The plan's `commit per task` instruction was meant as a TDD-discipline cue (don't batch all changes into one mega-commit), not a literal "every task is its own commit." The real rule: commit per LOGICAL feature unit. For a detector implementation, that's typically ~3 commits: Phase 0 (configs + registry), Phase 1 (detector + tests + conftest), Phase 2 (wiring).
**Rule:**
1. For new detector / setup implementations, target ~3 commits per plan: Phase 0 (config scaffolding bundled), Phase 1 (TDD detector + tests as ONE coherent unit), Phase 2 (wiring + smoke).
2. When the plan says "commit per task", read it as "commit per logical feature unit" — group related task commits if they share a coherent message.
3. Sub-tasks of TDD development (write failing test → implement → make it pass) are part of ONE logical unit and should be ONE commit at the GREEN state. Don't commit RED states unless the plan explicitly requires them.
4. If you've already over-committed, soft-reset and re-commit with the right cadence — the user's "no shortcuts" feedback applies to fixing past mistakes, not just future ones.
5. Atomic per-task commits are appropriate for: (a) tightly-coupled bug fixes where each commit must be revertable in isolation, (b) plans that explicitly call for it.
