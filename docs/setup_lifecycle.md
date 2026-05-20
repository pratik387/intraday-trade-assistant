# Setup Lifecycle — Idea → Live → Retirement

**Status:** Active as of 2026-05-20. Single source of truth for moving a setup from initial hypothesis to live deployment and through retirement.

**Replaces:** `docs/methodology_walk_forward.md` (deprecated — per-window tier methodology was abandoned, see `tasks/lessons.md` #15). The 3-period chronological validation (Discovery / OOS / Holdout) survives as a research tool; the **decision** at each gate is now made via the confidence framework (`tools/methodology/confidence/`), not via tier classification.

## Core principles

1. **Each stage has a gate, not a threshold.** The confidence framework outputs INTERVALS. The researcher reads them and judges. No code in this project hardcodes a ship/no-ship threshold; doing so would be folklore (Lesson #15).
2. **Sanity inflates, OCI is gold standard.** A setup that looks dead on sanity is dead. A setup that looks alive on sanity might still be dead in production. The Lesson #13 caveat applies at every sanity-stage gate.
3. **Pre-register mechanism before testing windows.** `mechanism_tags` + `mechanism_notes` go in config ≥ 1 commit before any window-based evidence. Falsifiable, not post-hoc (Lesson #12).
4. **Three numeric inputs only — researcher judges.** At every gate the input is: aggregate PF CI, per-regime breakdown, adjusted Sharpe (Harvey-Liu sign-preserving haircut for effective M setups tested historically).

## Gauntlet vs this lifecycle (NOT the same thing)

The **per-setup lifecycle** below is what this document covers — Phase 1-5 + confidence framework + ship + monitor + retire.

**Gauntlet v2** is a separate system-level tool that tunes the ~8 shared gate-config parameters (daily_cap, crowdedness threshold, cooloff_bars, min_predicted_r, etc.) once per major rebuild. Per setup you do NOT run Gauntlet. Spec at `specs/2026-04-23-sub-project-5-gauntlet-v2-design.md`, runbook at `docs/gauntlet_v2/README.md`.

The only intersection: a setup arriving at Stage 8 (OCI backtest) runs through whatever gate-config the latest Gauntlet pass produced. If gate-config has drifted materially since the last Gauntlet refresh, your per-setup numbers diverge from prior baselines — that's a Gauntlet refresh signal, not a setup failure.

## Stage map

| Stage | Owner artifact | Gate criterion (read, don't compute) | Decision |
|---|---|---|---|
| 0. Idea | `specs/YYYY-MM-DD-brief-<name>.md` | Hypothesis stated; falsifier identified; mechanism rooted in NSE/SEBI/Indian-market behavior | proceed / kill |
| 1. Phase 1 — Indian-market research | Brief section + citations | ≥2 Indian sources; data feasibility confirmed; SEBI/STT regulatory exposure documented | proceed / kill |
| 2. Phase 2 — empirical signature | `tools/sub9_research/phase2_<name>_signature.py` + CSV | Signal vs baseline delta ≥ 0.1% drift (cheap kill threshold) | proceed / kill |
| 3. Phase 3 — mechanism brief + pre-registration | Brief finalized + `mechanism_tags` committed to config | ONE-sentence mechanism + 3 falsifiers + 2 precedents | proceed / kill |
| 4. Phase 4 — sanity script (Discovery only) | `tools/sub9_research/sanity_<name>.py` + `_<setup>_trades_discovery.csv` | Anti-bias guards documented in header; Discovery aggregate PF > 1.0; schema valid | proceed / kill |
| 5. Phase 5 — cell-lock + OOS + Holdout | Locked cell JSON + OOS + Holdout sanity CSVs | Disc+OOS combined cell n≥200 + PF≥1.20; HO one-shot pass | proceed / kill |
| 6. **Confidence card (sanity)** | `reports/confidence_cards/<setup>_confidence_card.md` | PF CI on D+OOS+Holdout, per-regime breakdown, sanity adj Sharpe | proceed to structure / kill |
| 7. Structure code | `structures/<setup>_structure.py` + universe + tests | code-reviewer agent approves; mechanism pre-registered ≥ 1 commit prior | proceed / kill |
| 8. OCI backtest | OCI run directory with `analytics.jsonl` per session | Backtest completes; per-trade sanity-vs-OCI reproducibility floor ≤ Rs 10 | proceed / fix structure |
| 9. **Confidence card (OCI)** | OCI-source `confidence_card.md` | **Gold-standard verdict** on production-equivalent data | proceed / fix-or-retire |
| 10. Paper trade | 30-90 days `--paper-trading` mode | Paper PnL distribution matches OCI backtest expectation (CI overlap) | proceed / fix |
| 11. Live small | `position_size_multiplier: 0.25`, `cb_state: enabled` | 60-90 days; live distribution within OCI CI | scale up / hold / retire |
| 12. Scale up | `position_size_multiplier: 1.0` | Live verdict matches OCI expectation across multiple regimes | full deployment |
| 13. Ongoing | Quarterly OCI re-run + confidence card refresh | Per-regime breakdown surfaces decay before circuit breaker | continue / re-cell-lock / retire |
| 14. Retire | `docs/retired_setups.md` entry | PF CI crosses 1.0 OR adj Sharpe goes negative on accumulated OCI data | document + remove |

## Stage-by-stage runbook

### Stage 0 — Idea

Write `specs/YYYY-MM-DD-brief-<setup_name>.md` with:

- **Mechanism statement** — WHY this should work in NSE intraday. Cite the institutional or retail behavior driving the edge.
- **Falsifier** — a specific, observable outcome that would prove the thesis wrong (not "if it loses money" — specific to the mechanism).
- **Adjacent setups** — which existing setups would correlate with this one (matters for effective M at Stage 9).

A brief that doesn't pass code review here probably won't survive Phase 5 either.

### Stage 1 — Phase 1: Indian-market research (~15 min)

Goal: kill candidates with no real-world Indian-market basis BEFORE writing any code.

**Output:** a one-page section in the brief answering three questions:

1. **Operating mechanism.** What real-world Indian flow does this fade or catch? Cite ≥2 retail/pro Indian sources (intradaylab.com, Zerodha Varsity, broker quant reports, NSE research). No source = borrowed-mechanism speculation; kill.
2. **Data feasibility (Gate B, Lesson #1).** List exact data inputs (5m bars, OI, delivery%, F&O bhavcopy, earnings calendar). Verify each is on disk OR has a clear acquisition path. A setup needing intraday OI velocity when you only have EOD bhavcopy dies here.
3. **Regulatory sensitivity.** Which SEBI/STT/MIS rules govern the mechanism? Flag if a 2024-26 cutover affects it (especially **SEBI F&O Oct 2025** which broke `delivery_pct_anomaly_short`, `mis_unwind_vwap_revert_short`, `circuit_release_fade_short`).

**Indian-microstructure facts to encode (Lesson #4):**
- Broker MIS auto-square timing is heterogeneous: Upstox/Angel 15:15, ICICI 15:15-15:20, Zerodha 15:20-15:24 → retail MIS pressure window is **15:15-15:25**, not 15:20-15:25.
- MIS leverage SEBI minimum 20% margin → max 5x.
- NSE intraday volume profile (J/U-shape): 09:15-09:30 is 5x baseline, 11:00-13:00 is 1.0, 15:00-15:15 is 1.8-2.0x, **15:15-15:25 is 2.5-3.0x**.
- Retail concentration is highest in small/mid-cap (< Rs 250 stocks).

**Kill criteria:** can't cite 2 Indian sources; can't acquire required data; mechanism depends on a rule that already changed.

### Stage 2 — Phase 2: Empirical signature on Discovery (~10 min compute)

Goal: confirm the mechanism leaves a footprint in raw data **before writing the sanity script**.

**Output:** a `tools/sub9_research/phase2_<setup>_signature.py` script + a CSV.

Quantify the mechanism in raw data. Examples:
- Volume bulge on the trigger candle? (`vol_ratio` vs prior 5-day median)
- Directional drift? (signed mean intraday return on trigger days vs random days)
- Effect size? (signal_mean − baseline_mean, in %)

You're measuring the **mechanism's footprint**, not validating a strategy yet. No fees, no leverage, no exits — just raw drift.

**Example (`pre_results_t1_fade` Phase 2 on n=4,179 trigger days):**

| Metric | Value |
|---|---|
| T-1 mean intraday return | -0.351% |
| Random-day baseline mean | +0.238% |
| **Delta (signal − baseline)** | **-0.589%** |
| AMC class % negative days | 57.3% |

**Kill criterion:** net drift **< 0.1%** = signal doesn't exist. No methodology rescues a non-existent edge. Abandon — this is the cheapest kill in the pipeline.

### Stage 3 — Phase 3: Mechanism brief + pre-registration

Goal: lock the falsifiable hypothesis BEFORE looking at OOS data.

**Output:** finalized `specs/YYYY-MM-DD-brief-<setup>.md` + a **mechanism-pre-registration commit** in `config/configuration.json` (Lesson #12).

1. **Mechanism statement** — ONE sentence with Indian-microstructure anchor. Not "the market reverts" — instead "Mid-cap MIS-eligible stocks with -5% to -3% gap-downs revert by EOD because retail FOMO-shorts get squeezed when institutional bids come in 10:30-11:00."
2. **3 falsifiers** — specific conditions that would invalidate the thesis:
   - **Mechanism falsifier:** "If the retail-FOMO-short hypothesis is right, we should see option PCR spikes at the open on signal days."
   - **Regime falsifier:** "If this depends on positive FII flow, performance should drop during FII-exit regimes (R4 `fed_pivot_china_rotation_FII_exit`)."
   - **Infra falsifier:** "If MIS leverage limits change, the mechanism breaks."
3. **Pro/retail precedent** — ≥2 Indian sources that operationalize this on retail-MIS infra.

**Pre-registration commit:** add `mechanism_tags` (from `assets/mechanism_tags_registry.yaml`) and `mechanism_notes` (free text) to the setup block in `config/configuration.json`. Commit message: `chore(config): pre-register mechanism for <setup>`. This must be committed ≥ 1 commit BEFORE any walk-window evidence is collected. `tools/methodology/pre_registration.py` enforces this via `git log -S "mechanism_tags"` check.

**Kill criteria:** can't write the one-sentence mechanism; can't articulate 3 falsifiers; no precedent.

### Stage 4 — Phase 4: Sanity script with anti-bias guards (Discovery only)

Goal: produce a Discovery trade ledger that **the OCI run will reproduce within ±Rs 10 per trade**. Sanity bugs here cause every downstream verdict to lie.

**Output:** `tools/sub9_research/sanity_<setup>.py` + `reports/sub9_sanity/_<setup>_trades_discovery.csv` validated against `tools/methodology/sanity_csv_schema.py`.

**The 6-failure-mode checklist (Lesson #5). Every sanity script header MUST explicitly document each guard.** These six bugs killed multiple setups in 2026:

1. **Intraday aggregate look-ahead.** NEVER use `day_high`, `day_low`, `day_vwap`, `day_close` as filters at signal time. Use `session_high_so_far = bars[:i+1].high.max()` — only bars up to and including the current bar. Rule of thumb: "what value would I have known at this exact bar's close?" — anything later is leakage.

   *Case:* `earnings_day_intraday_fade` aggregated `day_high` over the full session for SL placement → 0% stop-hit in sanity vs 4.7% in production. Shipped → retired post-OCI.

2. **Volume baseline includes current bar.** Use `cum_vol_mean = bars.volume.expanding(min_periods=2).mean().shift(1)` (current bar EXCLUDED). Including the signal bar in its own baseline inflates `vol_ratio` exactly when the bar is most volatile — guaranteed positive bias.

3. **Mode B entry walk off-by-one.** Mode B = entry at NEXT bar's open (`bars[i+1].open`), never current bar. Walking semantics:
   - If entry is at `bars[i+1].open`: path walk starts at `bars[i+1]` (the bar's full intra-bar range happens AFTER entry — OK).
   - If entry is at `bars[i+1].close`: path walk must start at `bars[i+2]` (entry was at end of bar i+1; that bar's intra-bar range is already past).

   *Case:* `round_number` sanity walked from `i+1` when entry was at `i+1.close` — selected fictional fills.

4. **Same-bar exit ambiguity.** When both `hi >= hard_sl` AND `lo <= t2_target` on the same bar, sanity MUST pick stop (pessimistic). Production's tick-by-tick path could go either way; sanity must not assume the favorable outcome.

   *Case:* `mis_unwind` sanity had 88% same-bar look-ahead → systematically picked favorable outcomes. Disc 1.21 → OOS 1.22 → HO 0.75 → retired with -Rs 492K production.

5. **Cell-locked filters at signal time.** Every filter in the sanity must reproduce in production exactly. If sanity uses `cum_vol_mean` and production uses `prior_bars.mean()`, results diverge silently. The filter is part of the contract; document them in a top-of-file constants block.

6. **Reproducibility floor (Stage 4 → Stage 7 audit).** Once OCI structure code exists, per-trade-match diff between sanity and OCI for the SAME trade IDs must be ≤ Rs 10. If aggregate PF diverges, run a per-trade audit to find the bug. This is the audit tool when something feels off.

**Anti-bias guards from `docs/retired_setups.md` Common failure modes (read in addition):**

7. **wide_open OCI capture ≠ validated production.** `wide_open_mode=true` bypasses cell filters. Wide-open numbers quantify the raw signal universe, NOT the validated cell. Reporting wide-open PF as validated PF is the most common silent failure.
8. **Regime classifier non-reproducibility.** If you cell-lock to `regime=trend_down`, validate with the **production** classifier (`services/regime_classifier.py:classify_per_symbol`), not an ad-hoc Discovery-only label.
9. **MIS-leveraged fee math.** Fees scale with `qty × mis_leverage` (5× for MIS-eligible names), not base qty. Discovery PFs computed on base-qty fees are over-stated by ~Rs 650K on 2-year aggregates.

**Locked-filter discipline:**
- Write the filter list AT THE TOP of the script as constants BEFORE running. No "I'll just try one more conditioner" later.
- Discovery window only — typically 2023-01 to 2024-12. Do NOT touch 2025 OOS or 2026 Holdout in Stage 4.
- Output schema matches `tools/methodology/sanity_csv_schema.py` (signal_date, symbol, side, entry_price, exit_price, qty, pnl_pct RAW, exit_reason, same_bar).

**Sanity script skeleton (copy this pattern):**

```python
# tools/sub9_research/sanity_<setup>.py
#
# Anti-bias guards (Lesson #5):
#   1. No day_high/day_low/day_vwap at signal time — only bars[:i+1]
#   2. Volume baseline EXCLUDES current bar (.shift(1))
#   3. Mode B entry at bars[i+1].open; walk starts at bars[i+1]
#   4. Same-bar SL+T2 picks STOP (pessimistic)
#   5. Filters listed in LOCKED_FILTERS constant, no late tuning
#   6. Output validated via sanity_csv_schema.validate()
#
# Locked filters (DO NOT EDIT after first Discovery run):
LOCKED_FILTERS = {
    "cap_segment": ["mid_cap", "small_cap"],
    "vol_ratio_min": 1.5,
    "vol_ratio_max": 4.0,
    "time_window": ("10:00", "14:30"),
    # ...
}

DISCOVERY_START = date(2023, 1, 2)
DISCOVERY_END   = date(2024, 12, 31)

for each session in DISCOVERY:
    bars = load_5m_bars(symbol, session)  # IST-naive
    for i, bar in enumerate(bars):
        # Compute filters from bars[:i+1] only — no look-ahead
        session_high_so_far = bars.iloc[:i+1].high.max()
        prior_vol_mean = bars.iloc[:i].volume.tail(60).mean()  # shifted
        vol_ratio = bar.volume / prior_vol_mean

        if not passes(LOCKED_FILTERS, bar, session_high_so_far, vol_ratio):
            continue

        # Mode B entry
        if i + 1 >= len(bars):
            continue  # no next bar
        entry_price = bars.iloc[i+1].open
        entry_ts = bars.iloc[i+1].index

        # Walk forward from bars[i+1] (entry happened at its OPEN)
        for j in range(i+1, len(bars)):
            future = bars.iloc[j]
            # Same-bar ambiguity: SL wins (pessimistic)
            if future.high >= hard_sl and future.low <= t2_target:
                exit_price, exit_reason = hard_sl, "same_bar_sl"
                break
            if future.high >= hard_sl:
                exit_price, exit_reason = hard_sl, "sl"
                break
            if future.low <= t2_target:
                exit_price, exit_reason = t2_target, "t2"
                break
        else:
            exit_price, exit_reason = bars.iloc[-1].close, "eod"

        # pnl_pct = RAW per-share % return, NO fees, NO leverage
        pnl_pct = (entry_price - exit_price) / entry_price * 100.0  # SHORT
        rows.append({...})

df = pd.DataFrame(rows)
result = sanity_csv_schema.validate(df, setup_name="<setup>", layer="filtered_trades")
assert result.is_valid, result.summary()
df.to_csv(f"reports/sub9_sanity/_{setup}_trades_discovery.csv", index=False)
```

**Kill criterion:** Discovery PF on aggregate < 1.0 even before cell-lock. The mechanism is wrong, the filters are wrong, or there's a sanity bug — find which.

### Stage 5 — Phase 5: Cell-lock + R-multiple sweep + OOS + Holdout

Goal: select the narrowest stable (filter × R-multiple) cell on Discovery + OOS combined, then test once on Holdout.

This stage has TWO sweeps over Discovery data — a **filter cell sweep** and an **R-multiple grid sweep** — and both close before any OOS or Holdout data is observed.

**Step 1 — Filter cell sweep (Discovery only).** Vary (cap_segment × time-window × vol-ratio band × dist-from-VWAP band × gap-bucket × any setup-specific dimension). For each intersection cell, compute PF, n, win-rate, expectancy on Discovery. Pre-register the dimension list in the brief BEFORE the sweep — no "I'll add one more conditioner" later (Lesson #2).

**Step 2 — R-multiple grid sweep (Discovery only).** For the surviving filter cells, sweep `(T1, T2, hard_sl)` in R-units. Typical grids: T1 ∈ {0.5R, 1.0R, 1.5R}, T2 ∈ {1.5R, 2.0R, 2.5R, 3.0R}, SL ∈ {1.0R, 1.5R, 2.0R}. Lock the winning `(filter cell, R-multiple)` joint cell. Save as `tools/sub9_research/<setup>_cell_selection_locked.json` with the exact values.

**Step 3 — Run OOS ONCE on the locked joint cell.** Typically 2025. If aggregate PF_net diverges materially from Discovery (|PF_disc − PF_oos| > 0.30), the cell was overfit — kill.

**Step 4 — Run Holdout ONCE on the locked joint cell.** Typically 2026+. Same gate.

**Anti-patterns (Lesson #2):**
- **Post-hoc cell selection** — choosing a cell AFTER seeing OOS is p-hacking. The cell that wins on Disc+OOS combined (Step 1+2 + a confirmatory Step 3) is the cell tested on Holdout. Not the other way around.
- **Salvage mining** — if no `(filter × R)` cell with n≥200 + PF≥1.20 wins on Disc+OOS combined, the setup is dead. Stop digging.
- **Re-sweeping R after seeing OOS/HO** — same epistemic status as cell-mining. Lesson #2: "The same mechanism that produces look-ahead bias in features produces overfitting in cells."

**MIS-leveraged fee math (Failure mode #7 from `retired_setups.md`):** the R-multiple sweep MUST include MIS-leveraged fees, not base-qty fees. Discovery PFs that were computed on base-qty fees are over-stated by ~Rs 650K on 2-year aggregates.

**Use the shared helper (`tools/methodology/cell_sweep.py`).** Replaces the ad-hoc per-setup `_*_sweep_cellmine.py` scripts that each re-implemented the same pattern with slightly different bugs. Audit of 25 production scripts (2026-05-20) found 3 target paradigms, all supported:

| `target_unit` | Used by | Grid sweeps |
|---|---|---|
| `"R"` | 56% of setups (long_panic_gap_down, or_window_failure_fade_short, circuit_release_fade_short, capitulation_long_v2, etc.) | `(T1_R, T2_R, ts_hhmm, partial_mode)` |
| `"pct"` | 8% (block_deal_t0_short, block_deal_continuation_short) | `(T1_pct, T2_pct, sl_pct, ts_hhmm, partial_mode)` |
| `"structural"` | 24% (gap_fade_short, circuit_t1_fade_short, etc.) | `(ts_hhmm, partial_mode)` only — T1/T2/SL are per-row prices |

The helper:

- `validate_candidates_schema` rejects look-ahead dims by exact match (`day_high`, `day_low`, `day_vwap`, `day_close`, `day_volume`, `day_range`, `day_atr`) and prefix (`close_off_high*`, `EOD_*`, `eod_*`, `session_close_*`). Legitimate dims like `day_gain_bucket` (whose underlying value is computed from `session_high_so_far` at signal) are allowed.
- `simulate_exit` implements same-bar pessimism (Failure mode #4) and side-aware PnL; all three modes converge to R-unit exit logic internally.
- `partial_mode` is a swept grid parameter, not a config-wide constant. Three options:
  - `"all_in"` — full qty, T1 doesn't fire; only T2 / SL / TS resolve
  - `"partial_50_no_trail"` — 50% at T1, 50% to T2 or TS close
  - `"partial_50_be_trail"` — 50% at T1; remaining 50% exits at BE if mae stayed close to SL post-T1 (conservative approximation when only summary `mfe_r`/`mae_r` are present; exact when `mfe_r_pre_t1` + `mae_r_post_t1` are also emitted — these are OPTIONAL_COLUMNS in `tools/methodology/sanity_csv_schema.py`)
- `select_best_cell` returns `None` if no cell meets ship-eligibility — kill signal, not silent-pick-best (Lesson #2 anti-salvage).
- `lock_cell` writes JSON; refuses overwrite without `force=True` (lock-once discipline).

**Candidates DataFrame contract:**

| target_unit | Required columns (per row) |
|---|---|
| `R` | `entry_ts`, `entry_price`, `qty`, `mfe_r` (unsigned), `mae_r` (unsigned), `R_per_share`, `close_at_<HHMM>` |
| `pct` | `entry_ts`, `entry_price`, `qty`, `mfe_pct`, `mae_pct`, `close_at_<HHMM>` |
| `structural` | `entry_ts`, `entry_price`, `qty`, `mfe_r`, `mae_r`, `R_per_share`, `t1_price`, `t2_price`, `close_at_<HHMM>` |

Plus the filter dimensions as columns (declared in `dim_pool`).

**Per-setup dimension registry (`assets/setup_dimension_registry.yaml`).** Central record of what filter dimensions each setup uses, what their data source is (`signal_time` / `session_time` / `per_row_at_signal` / `precomputed`), and what dimensions have been explicitly forbidden (with the look-ahead / rejection reason).

Pass `setup_name=` to `run_cell_sweep(...)` to cross-check `dim_pool` against the registry:
- Errors on any dim in `dim_pool` that's in the setup's `forbidden_dims` (e.g., `day_gain_bucket` on `circuit_release_fade_short` — uses EOD value, replaced by `day_gain_at_signal_bucket`).
- Warns on any dim not in the registry's `allowed_dims` (likely a new dim — confirm it's not a look-ahead, then add it to the YAML).
- Warns on any registered dim missing from `dim_pool` (intentional skip or oversight?).

```python
results = run_cell_sweep(
    disc_candidates, cfg,
    setup_name="circuit_release_fade_short",   # triggers registry cross-check
)
```

**Usage example (R-mode setup, e.g. circuit_release_fade_short):**

```python
from tools.methodology.cell_sweep import (
    CellSweepConfig, GridEntry, run_cell_sweep, select_best_cell, lock_cell,
)

grid = []
for T1 in (0.5, 1.0):
    for T2 in (1.5, 2.0, 2.5):
        for ts in (1300, 1430, 1500):
            for pm in ("partial_50_no_trail", "partial_50_be_trail"):
                grid.append(GridEntry(
                    label=f"T1={T1}/T2={T2}/TS={ts}/{pm}",
                    ts_hhmm=ts, partial_mode=pm,
                    t1=T1, t2=T2, sl=1.0,
                ))

cfg = CellSweepConfig(
    side="SHORT", target_unit="R",
    grid=grid,
    dim_pool=["cap_segment", "dow", "day_gain_bucket", "rejection_hhmm_bucket"],
    n_min_floor=100, pf_min_floor=1.10,
    n_min_ship=200,  pf_min_ship=1.30,
)
results = run_cell_sweep(disc_candidates, cfg)
best = select_best_cell(results, cfg)
if best is None:
    raise SystemExit("KILL: no ship-eligible cell on Discovery")
lock_cell(best, setup_name="circuit_release_fade_short", window_label="Discovery",
          output_path=REPO / "tools/sub9_research/circuit_release_fade_short_cell_selection_locked.json")
```

**Usage example (structural setup, e.g. gap_fade_short with PDC targets):**

```python
# disc_candidates already has t1_price (e.g. midpoint), t2_price (e.g. PDC)
# computed per-row at signal time by the sanity script.
grid = [
    GridEntry(label=f"TS={ts}/{pm}", ts_hhmm=ts, partial_mode=pm)
    for ts in (1015, 1045, 1130, 1300, 1510)
    for pm in ("partial_50_no_trail", "partial_50_be_trail", "all_in")
]
cfg = CellSweepConfig(
    side="SHORT", target_unit="structural",
    grid=grid,
    dim_pool=["cap_segment", "gap_pct_bucket", "dim_wick"],
)
results = run_cell_sweep(disc_candidates, cfg)
```

After Stage 5 locks the cell, re-run the sanity at the locked `(filter_cell, grid_entry)` to emit the final canonical trade CSV for Stage 6 (sanity confidence card).

### Stage 6 — Confidence card on sanity


Aggregate the D+OOS+Holdout trade CSVs into the canonical schema, then run the confidence framework:

```bash
# 1. Aggregate to canonical (one-time per setup)
#    Edit SETUP_CSV_MAP in aggregate_sanity_to_canonical.py to add the setup,
#    then run:
.venv/Scripts/python tools/methodology/aggregate_sanity_to_canonical.py

# 2. Generate confidence card
.venv/Scripts/python tools/methodology/confidence/confidence_card.py \
    --include-sanity --setups <setup_name>
```

Read `reports/confidence_cards/<setup>_confidence_card.md`. The questions to ask:

- Is the aggregate PF CI lower bound > 1.0? (necessary, not sufficient)
- In which regimes does the edge live? Is it concentrated in 1-2 regimes or spread across 5-7? Concentrated = regime-conditioned (high attrition risk)
- Is the adjusted Sharpe positive after Bonferroni haircut? Sanity-source haircut is informational; the real test is at Stage 8

**Lesson #13 caveat at this stage:** sanity-GREEN does NOT permit live deployment. It only justifies the investment of Stage 7 structure-code work. Sanity-RED kills the setup here without spending the structure-code week.

### Stage 7 — Structure code

Write the detector in `structures/<setup>_structure.py`. Wire universe in `services/setup_universe.py`. Add tests in `tests/` (gitignored individually but tracked test files exist).

Pre-register mechanism in `config/configuration.json` under `setups.<name>`:
```json
"mechanism_tags": ["tag_from_assets/mechanism_tags_registry.yaml"],
"mechanism_notes": "Why this should work and which regimes might break it"
```
Commit pre-registration as its own commit BEFORE running OCI. This is enforced by `tools/methodology/pre_registration.py`.

Run the code-reviewer agent on the structure code. CLAUDE.md mandatory rules apply (no hardcoded defaults, IST-naive timestamps, live/backtest compatibility).

### Stage 8 — OCI backtest

Deploy structure code to OCI. Run backtest over 2023-01..present. Sessions land as `<YYYY-MM-DD>/analytics.jsonl` records.

**Reproducibility floor (Failure mode #6 from Lesson #5):** before generating the OCI confidence card, run a **per-trade sanity-vs-OCI diff** on the same date range. For each (date, symbol, side) tuple present in both:

- `|realized_pnl_inr_sanity − realized_pnl_inr_oci| ≤ Rs 10` on a like-for-like notional basis (sanity is naked qty, OCI is leveraged; normalize before diffing)
- Same `exit_reason` distribution within 5 percentage points

If the per-trade diff fails, STOP. Find the sanity bug (one of the 6 failure modes) or the structure-code bug. Do NOT generate the confidence card until the diff passes — the card will report inflated PF that you'll then ship from.

### Stage 9 — Confidence card on OCI (gold-standard verdict)

Aggregate the OCI run into canonical and regenerate the confidence card:

```bash
.venv/Scripts/python tools/methodology/aggregate_oci_to_canonical.py \
    --run-dir <oci_run_path> --setup <setup_name>
.venv/Scripts/python tools/methodology/confidence/confidence_card.py
```

This card is the **gold-standard verdict**. Same questions as Stage 6, but now the answers are production-equivalent (real entry filter, real fills, real fees). If the OCI confidence card disagrees with the sanity card, trust the OCI card.

The regime breakdown matters most here. A setup with PF CI [1.10, 1.40] aggregate that has 4/7 weak regimes is a candidate for regime-gating (Lopez de Prado tactical paradigm), not a clean ship.

### Stage 10 — Paper trade

Run `python main.py --paper-trading` with the setup enabled. 30-90 days. Compare actual paper P&L distribution to the OCI confidence card's expectancy CI. They should overlap.

If paper diverges materially from OCI: stop. Investigate (often a live-vs-backtest infrastructure issue — see `feedback_production_mindset.md` and `project_paper_backtest_parity.md`).

### Stage 11 — Live small

Config:
```json
"enabled": true,
"position_size_multiplier": 0.25,
"cb_state": "enabled",
"cb_drawdown_threshold": -<2σ of OCI per-day distribution>,
"cb_lookback_days": 60,
"cb_min_trades_for_signal": 30
```

60-90 days. Compare live distribution to OCI expectation. Daily circuit breaker (`jobs/check_circuit_breakers.py`) monitors trailing 60-day net PnL.

### Stage 12 — Scale up

Only after Stage 11 closes positive AND matches OCI expectation across at least 2 regimes (e.g., the live window spanned post_tariff_consolidation + war_vol_2026). Move `position_size_multiplier` to 0.5 → 1.0 over 30-60 days.

### Stage 13 — Ongoing monitoring

Quarterly: re-run `aggregate_oci_to_canonical` on accumulated production data, regenerate confidence card. Compare new card to the Stage 9 baseline:

- PF CI tightening → edge is real and stable
- PF CI widening or drifting toward 1.0 → degradation; investigate regime cause
- Per-regime breakdown shows new weak regime → consider regime gate

Daily: circuit breaker. If it trips, the setup goes `cb_state=disabled` automatically; researcher inspects and decides retire vs un-disable.

### Stage 14 — Retirement

Triggers (any one is sufficient evidence — researcher confirms):

- PF CI lower bound below 0.95 on OCI canonical with n > 500 in the most recent 12-month window
- Adjusted Sharpe (Harvey-Liu) flips negative on OCI canonical
- Per-regime breakdown shows the edge has collapsed to a single regime AND that regime has ended

Documentation requirements (replacing the prior walk-forward-table requirement):

1. Section in `docs/retired_setups.md` with: thesis, universe+filters, claimed validation, **OCI confidence card snapshot** (aggregate PF CI + per-regime table + adjusted Sharpe), failure mode, conditions for revival
2. Mechanism tags in config block must reflect the actual decay cause (e.g., add `regime_X_dependent` if regime breakdown revealed it)
3. Code removal list
4. New `tasks/lessons.md` entry IF a new failure pattern was discovered (not for repeat patterns)

## Anti-patterns (don't)

- **Don't compute new thresholds without literature backing.** PF > 1.10, win-rate > 55%, 9/13 windows — all folklore. The confidence framework outputs intervals; trust them.
- **Don't un-retire a setup based on a sanity-GREEN result.** Sanity inflates (Lesson #13). Revival requires fresh OCI evidence with structure code.
- **Don't ship at full size from Stage 9.** Even with a clean OCI card, Stage 10-11 are non-negotiable. Production has live-vs-backtest infrastructure risk.
- **Don't skip pre-registration.** A post-hoc "the mechanism is X because window 7 failed" claim is unfalsifiable.

## References

- Confidence framework code: `tools/methodology/confidence/`
- Regime schema: `assets/regime_schema.yaml`
- Aggregators: `tools/methodology/aggregate_{oci,sanity}_to_canonical.py`
- Lesson #12: pre-registration requirement
- Lesson #13: sanity-vs-production inflation
- Lesson #14: per-trade actual fees (not flat constant)
- Lesson #15: framework redesign + sign-preservation bug
- Research backing for the framework: `reports/sub9_sanity/_per_trade_validation_research.md` (gitignored but exists locally), `reports/sub9_sanity/_indian_regime_schema_research.md`, `reports/sub9_sanity/_indian_regime_global_events_research.md`
- Stale doc: `docs/methodology_walk_forward.md` (per-window tier methodology, abandoned)
