# Lessons Learned

Patterns and rules captured from corrections during development sessions.
Review at the start of each session to avoid repeating mistakes.

---

<!-- Format:
### [Date] - Short title
**What went wrong:** ...
**Rule:** ...
-->

### 2026-05-19 (#9) — Cache directory is BACKTEST-ONLY; live/paper must touch zero feathers

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
