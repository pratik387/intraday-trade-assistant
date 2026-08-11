# System Handover — Intraday Trade Assistant

**Written:** 2026-08-02
**Scope:** what exists, what runs, how the pieces connect, and what is knowingly broken or unfinished.

This is a working handover, not a sales sheet. Where something is fragile, unproven, or a
trap, it says so. Read §9 before touching production.

---

## 1. What this system is

An algorithmic trading system for **NSE (Indian) equities**, primarily intraday MIS, with a
smaller overnight/multi-day CNC+MTF wing. It scans the whole liquid NSE universe on 5-minute
bars, detects a fixed library of setups, plans trades, executes through a broker, and manages
exits.

The differentiating constraint is **research discipline**, not execution speed. The system has
an unusually strict setup-lifecycle process (§7) because its historical failure mode was
shipping setups that backtested well and died forward. Most of the machinery in `tools/` and
`docs/` exists to prevent that.

**Core domain facts baked into the code:**
- MIS intraday leverage ~5×; Zerodha auto-squares MIS positions from ~15:20 IST
- The trading session is 09:15–15:30 IST; 5m bars are **start-labelled** (the bar labelled
  15:10 covers 15:10–15:15 and its close is the 15:15 print — this has caused real bugs)
- All internal timestamps are **IST-naive** (no tzinfo). Helpers in `utils/time_util.py`
- ASM/GSM surveillance, circuit bands, and T2T settlement are real constraints that gate trades

---

## 2. Operating modes

| Mode | Command | Data | Orders |
|---|---|---|---|
| **Backtest** | `python main.py --dry-run --session-date YYYY-MM-DD` | archived feathers | simulated |
| **Paper** | `python main.py --paper-trading --data-source upstox` | live Upstox | simulated |
| **Live** | `python main.py` | live | **real** |

All three run the *same* scanner, detectors, planner and exit logic. Only the broker and data
source swap. That is deliberate — there is no second code path for backtesting.

**Mode is decided by CLI flags, not config.** This matters: see §9.1.

---

## 3. Architecture — the request path

```
                    ┌─────────────────────────────────────────┐
   data source ───► │ ScreenerLive  (services/screener_live.py)│
   (feather /       │  • builds universe once per session     │
    Upstox WS+REST) │  • per-5m-bar scan of all symbols       │
                    │  • daily_dict cache (ctx.df_daily)      │
                    └────────────────┬────────────────────────┘
                                     │ batches
                    ┌────────────────▼────────────────────────┐
                    │ Dispatch (services/dispatch/)           │
                    │  planner.py → worker.py                 │
                    │  builds MarketContext per symbol:       │
                    │   df5, df_daily, levels, regime         │
                    └────────────────┬────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────┐
                    │ Structures (structures/*_structure.py)  │
                    │  detect() → StructureResult             │
                    │  plan_*_strategy() → entry/stop/targets │
                    └────────────────┬────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────┐
                    │ PlanOrchestrator (plan_orchestrator.py) │
                    │  gate: enabled + cb_state allowlist     │
                    │  → sizing, capital, slot caps           │
                    └────────────────┬────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────┐
                    │ Execution (services/execution/)         │
                    │  TriggerAwareExecutor → broker          │
                    │  ExitExecutor: SL / targets / time-stop │
                    └─────────────────────────────────────────┘
```

**Key files:**
- `services/screener_live.py` — the scanner loop; owns the universe and the daily-bar cache
- `services/dispatch/{planner,worker,setup_registry}.py` — fan-out and context construction
- `structures/<setup>_structure.py` — one file per setup; detection + plan
- `services/plan_orchestrator.py` — the enable/circuit-breaker gate and sizing
- `services/execution/` — order placement, exits, overnight/multi-day handlers
- `config/configuration.json` — **single source of truth for every parameter**

### 3.1 The no-defaults rule

Every threshold comes from config. Structures read with `config["key"]`, not `.get(key, default)`,
so a missing key raises `KeyError` at startup rather than silently trading a default. When adding
a feature, **add the config key first**. Tests assert this (`test_missing_config_key_raises`).

---

## 4. Data pipeline — the part that bites

There are **four** distinct price stores and they are not interchangeable. Most of the hard bugs
in this system have come from mixing them up.

### 4.1 Per-symbol archives — `cache/ohlcv_archive/{SYM}.NS/`

| file | contents |
|---|---|
| `{SYM}.NS_1days.feather` | daily bars |
| `{SYM}.NS_1minutes.feather` | 1-minute bars |
| `{SYM}.NS_5minutes.feather` | 5-minute bars |
| `{SYM}.NS_5minutes_enriched.feather` | 5m + vwap/adx/rsi/bb_width |

Written by `upstox_cache_downloader.py`:
```bash
python upstox_cache_downloader.py --data-type daily|minute|5minute --from YYYY-MM-DD --to YYYY-MM-DD --workers 4
```
Unauthenticated public Upstox v3 endpoint (no token needed). **Rate limit: 8 rps.** Higher rates
429-storm on months-old data — a documented incident lost 14% of symbols at 40 rps. Minute data
chunks at 28 days per request; daily fetches any range in one call. Re-running is idempotent
(append + dedup).

**Symbol list gotcha:** the downloader caches `cache/upstox_instrument_map.json` and returns it
without rebuilding if present. A stale map silently narrows the universe — it was found holding
1,575 symbols against a real universe of ~2,220. Delete the file to force a rebuild
(`upstox_nse_instruments.json` ∩ `nse_all.json`).

### 4.2 Monthly pre-aggregates — `backtest-cache-download/monthly/`

Built by `tools/create_preaggregated_cache.py --from … --to … [--output-dir …] [--skip-daily]`.

| file | consumed by | purpose |
|---|---|---|
| `YYYY_MM_1m.feather` | `broker/mock/feather_tick_loader.py:134` | **tick replay** — backtest execution |
| `YYYY_MM_5m_enriched.feather` | screener precompute | structure detection |

**If `_1m.feather` is missing for a month, the loader silently drops to "SLOW MODE", reads
per-symbol archives, loads 0 symbols, and the whole month produces zero trades across every
setup.** This is not an error — it is a warning line in the log. It cost a full holdout run.
Always check session counts by month before trusting a backtest.

`tools/sub9_research/download_monthly_5m.py YYYY-MM` fetches 5m monthly files directly, but
writes **base OHLCV only, no enrichment columns** (deliberately — so consumers that need them
fail loudly). Empirically the intraday path tolerates their absence; this was verified for
`earnings_downshock` only, **not** for the other setups.

### 4.3 `cache/preaggregate/consolidated_daily.feather` — the production daily panel

Schema `ts, open, high, low, close, volume, symbol`. Built incrementally from the `_1days`
archives. **This is what `ctx.df_daily` is in backtest** (via
`broker/mock/mock_broker.py:_load_from_consolidated_daily`, with a strict look-ahead cut at the
session date).

- **UNADJUSTED for corporate actions.** Price *levels* for a symbol that later split differ ~10×
  from adjusted sources on every prior bar. One-day *returns* are unaffected except across the
  CA date itself (~0.07% of rows).
- Contains ~2,434 symbols, more than `nse_all.json` (2,333) — the extra are historical
  leftovers outside the tradeable universe.
- Its path is duplicated in `mock_broker.get_daily` and `oci/docker/entrypoint.py`; a divergence
  once silently broke a setup for 8 days. There is a comment in the builder warning about this.

### 4.4 `cache/preaggregate/clean_daily_from5m.feather` — research only

CA-adjusted, bad-print-cleaned, built by resampling 5m bars (`_tmp_build_clean_daily.py`). Used
by `tools/sub9_research/*` and by four paper-only multi-day setups in local `--mode multi_day`
dry-runs. **It is never `ctx.df_daily` on the intraday path.**

### 4.5 ⚠ The three-source trap

The same conceptual variable — "yesterday's close" — resolves differently per mode:

| mode | source | close definition | adjusted? |
|---|---|---|---|
| research scripts | `clean_daily_from5m` | last 5m print | yes |
| backtest | `consolidated_daily` | official NSE close | no |
| live/paper | Upstox daily API | official NSE close | yes |

NSE's official close is a **last-30-minute VWAP**, not the last trade. On days a stock is dumped
into the bell these diverge materially. For a threshold-based trigger this reshuffles ~21% of the
selected population at a −8% boundary.

**Rule:** any setup conditioned on a daily value must be researched on the series production
actually reads, or explicitly re-validated. Do not assume the research script and the engine
agree.

### 4.6 Event data

- `data/earnings_calendar/earnings_events.parquet` — announce dates/times, `announce_time_class`
- `data/asm_gsm_history/asm_gsm_events.parquet` — surveillance state (fails **open** by design)
- `data/delivery_pct/`, `data/cross_day_rvol/`, `data/corporate_actions/`, `data/mtf_universe/`

Enrichers in `services/*_enrichment.py` attach these to the daily frame on demand.

---

## 5. Backtesting at scale — OCI

Local single-day: `python main.py --dry-run --session-date 2026-05-05`.

For multi-month runs, work is fanned out to an Oracle Kubernetes cluster, one pod per session:

```bash
python oci/tools/submit_oci_backtest.py --start 2026-01-01 --end 2026-07-24 \
       --nodes 26 --max-parallel 160 --no-wait
python oci/tools/monitor_and_cleanup_backtest.py <run_id>
```

Flow: code tarball → `backtest-code` bucket → k8s Job → each pod pulls data from `backtest-cache`
→ results to `backtest-results` → downloaded as `backtest_<run_id>.zip` and extracted to
`oci/cloud_results/<run_id>/<session-date>/`.

Per session you get `trade_report.csv`, `analytics.jsonl`, `events.jsonl`, `agent.log.gz`.

**Notes**
- `oci/docker/entrypoint.py` is **baked into the Docker image**. Changing it requires a rebuild;
  the code tarball alone will not pick it up.
- **Always pass `--nodes N`.** The pool auto-scales to 0 after a run; submitting without it
  leaves every pod `Pending` forever (375 pods sat unschedulable for 47 minutes this way).
- Cost is trivial (~$0.22 for a 375-session run); wall clock ~35–40 min.
- Sessions with no trades write **no** `trade_report.csv`. Count session *directories*, not
  trade reports, when checking coverage.

**Slippage caveat:** runs execute at `fees_slippage_bps` (currently **5.0**), which is optimistic.
Measured slippage for illiquid shorts is 18.7 bp/side central, ~27.5 conservative, ~30 on recent
pools. Re-price results post-hoc; do not quote the raw run number.

---

## 6. Deployment — what actually runs

**Host:** `ubuntu@161.118.169.84`, key `ssh-key-2025-10-28.key`. Timezone **IST**.
**Two checkouts:** `/home/ubuntu/intraday_fixed/…` (intraday + overnight) and
`/home/ubuntu/multiday_cnc/…` (multi-day CNC), sharing the first one's venv.

**`python` does not exist on the VM** — only `python3` and `.venv/bin/python`. An interactive
shell activates the venv; a non-interactive `ssh host "…"` does not. Always use `.venv/bin/python`
in scripted commands or the process dies instantly.

### 6.1 tmux sessions (manual)

| session | what |
|---|---|
| `trading_fixed` | the intraday paper daemon |
| `api`, `frontend` | dashboard |

```bash
tmux new-session -d -s trading_fixed 'cd /home/ubuntu/intraday_fixed/intraday-trade-assistant && \
  .venv/bin/python main.py --paper-trading --health-port 8081 --ws-port 8091 \
  --admin-token <token> --risk-mode fixed --risk-value 1000 --data-source upstox'
```
Health check: `curl -s http://127.0.0.1:8081/health` → `{"status":"ok","state":"trading"}`.

### 6.2 cron (the live book)

```
45 8  * * 1-5  cron-refresh-mtf.sh                    # MTF universe refresh
26 15 * * 1-5  LIVE=1 cron-entry.sh                   # overnight ENTRY  (REAL MONEY)
28 15 * * 1-5  cron-multiday-exit.sh    (multiday_cnc)
35 15 * * 1-5  cron-upload-trading-session.sh
5  16 * * 1-5  LIVE=1 cron-place-exit.sh              # overnight exit placement (REAL MONEY)
5  16 * * 1-5  cron-multiday-entry.sh   (multiday_cnc)
10 16 * * 1-5  cron-archive-overnight.sh
30 9  * * 1-5  LIVE=1 cron-verify-exit.sh             # (REAL MONEY)
33 9  * * 1-5  cron-multiday-verify-entry.sh (multiday_cnc)
45 9  * * 1-5  cron-overnight-reconstruct.sh
30 10 * * 1-5  LIVE=1 cron-verify-exit.sh
*/15 9-15 * * 1-5  tools/funds_snapshot.py
0  9  * * 0    refresh-cap-segments.sh
```

`scripts/cron-entry.sh` defaults to `--paper-trading`; `LIVE=1` switches it to real Kite orders.
There is a scar comment in that file: `MODE_FLAGS=""` once silently fell back to paper and
"cost a live session on 2026-06-23". Do not reintroduce empty-string defaults.

### 6.3 What is real money right now

**Only the overnight/multi-day wing** (`close_dn_overnight_long`, MTF at ~₹10k/slot), via the
`LIVE=1` crons. The intraday book is paper. The intraday daemon is started **manually** — see §9.2.

---

## 7. Research methodology (`docs/setup_lifecycle.md`)

Stages 0–14, from idea to live, with five amendments added after specific failures:

| amendment | rule | why it exists |
|---|---|---|
| **A1** | forward-only fresh-pool one-shot decides | historical windows get re-used and burned |
| **A2** | every evaluation logged to `docs/experiment_ledger.jsonl`; M feeds a Harvey-Liu haircut | multiple-testing inflation |
| **A3** | illiquid data must be CA-adjusted + bad-print cleaned | a "reversal edge" was 100% bad prints |
| **A4** | factor budget — new setups must be orthogonal to the book | A2/C1/C4/C6 + panic_crash are ONE capitulation factor |
| **A5** | era-split results (pre/post 2025Q4); a pooled number that flips by era is not an edge | the illiquidity premium flipped sign in 2025Q4 |

**Standing hard rules**
1. Never tune a parameter on the data that revealed the problem. Widening a stop because losses
   appeared in `hard_sl` is both fitting *and* tautological (a stop-out is a loss by definition).
2. A pattern that flips sign across eras is noise, however good the pooled PF.
3. Prefer paper/production logs over research replications when they disagree — the replication
   usually has a population or object mismatch.
4. Mined filters must be pre-registered and tested on data that did not produce them.
5. `n` from a few weeks of paper cannot support ship/kill. Paper surfaces **anomalies**
   (parity breaks, universe mismatch, exit-mix divergence), not expectancy.

`docs/retired_setups.md` records ~20 retired setups with the reason each died — read it before
proposing anything, since many "new" ideas are already there.

---

## 8. The book today

`config/configuration.json → setups` (16 defined).

**Intraday, enabled:**

| setup | state |
|---|---|
| `up_spike_fade_short` | live-enabled, paper |
| `long_panic_gap_down` | live-enabled, paper |
| `or_window_failure_fade_short` | live-enabled, paper |
| `panic_crash_revert_long` | live-enabled, paper |
| `earnings_downshock_continuation_short` | **enabled 2026-07-31 for paper forward-run — see §8.1** |

**Overnight / multi-day:** `close_dn_overnight_long` (enabled + paper; the only intraday-adjacent
setup on **real money**). `crash2d_revert_long`, `low52_capitulation_revert_long`,
`mtf_capitulation_revert_long`, `zscore_oversold_revert_long` are `enabled:false, paper_enabled:true`
(multi-day cron path, which *does* honour `paper_enabled`).

**Disabled / parked:** `below_vwap_volume_revert_long` (paper validation failed at n=126, PF 0.84
vs 1.10 gate), `gap_fade_short` (retired after paper), `circuit_t1_fade_short`,
`delivery_pct_anomaly_short`, `pead_reaction_drift`, `xsec_momentum_demeaned`.

### 8.1 `earnings_downshock_continuation_short` — current focus

**Mechanism:** an analyst-uncovered, retail-held small-cap reports earnings and is punished with
a ≥8% down day. Selling is not complete at that close; the holder base digests overnight and the
stock keeps bleeding on T+1. Short at **09:20** (the close of the 09:15–09:20 bar, *not* the open
print), cover at **15:15** (bar labelled 15:10). No stop, no targets, catastrophe stop only.

**Stage-8 result** (production pipeline, official-close trigger, conservative 27.5 bp/side):

| window | n | mean/trade | PF | t |
|---|---|---|---|---|
| Discovery Apr-23→Dec-24 | 150 | +0.373% | 1.350 | +1.34 |
| OOS 2025 | 129 | +0.305% | 1.306 | +1.04 |
| Holdout 2026 Jan→Jul-24 | 106 | +0.404% | 1.397 | +1.23 |
| **true OOS (OOS+HO)** | **235** | **+0.349%** | **1.348** | **+1.60** |

Production reproduced research to three decimals on Discovery and OOS — **Stage-8 parity is
closed**. At measured 18.7 bp the true-OOS figure is +0.525% / PF 1.564 / t +2.41; at conservative
27.5 bp the CI crosses zero. Break-even is 46.2 bp/side.

**V1 was unholdable as a book:** 16/30 months positive, rolling-12m PF < 1.0 in 6 of 28 windows,
max DD −₹28,567, and a **15-month underwater stretch** (Oct-2023 → Feb-2025) sitting mid-sample.

**V2 (frozen + pre-registered 2026-07-31)** adds two era-stable, mechanism-backed filters:
- reaction **band [−12%, −8%]** (`shock_floor_pct`) — deeper moves are capitulation that reverts,
  and are the same cohort `panic_crash_revert_long` goes **long** on
- **`small_cap` only** — mid_caps carry coverage and price the news same-day (+0.30/−0.91/−0.20)

In-sample V2: n=210, +0.629%, PF 1.77, t +3.10, 21/27 months positive, DD −₹14,705.
Concurrency cap **derived**: V2 never exceeds 5 concurrent in 3.3 years → `max_concurrent_positions: 5`,
`capital_budget_pct: 20` (₹1L notional at ₹500k paper capital), truncating nothing.

**These filters were mined from the data they improve.** ~17 bucket evaluations; Bonferroni at
M=17 puts the bar near t≈3.0 and V2 sits at 3.10 — on the line, not clear of it — while discarding
45% of trades. Season and reporting-wave conditioning were also tested: both are real in V1 and
**absorbed** by V2 (ramp months go −0.013% → +0.817%), so they are recorded as monitored
dimensions in spec §10f, **not** filters. Regime conditioning is noise.

**Every archive window is burned.** Forward data from **2026-08-01** is the only admissible test:
one shot at n ≥ 40, conservative 27.5 bp, PASS floor +0.15%/trade, due ~Dec-2026. If V2 fails,
V1 is not resurrected — the candidate retires.

Spec: `specs/2026-07-29-brief-earnings_downshock_continuation_short.md` (§10 = V2).

---

## 9. Known gaps, traps and unfinished work

### 9.1 ⚠ Intraday has no paper-only gate
`plan_orchestrator.py:199` gates solely on `enabled`. `paper_enabled` is honoured **only** by
`overnight_handlers.py:317` and `mtf_capitulation_handlers.py:88`. So an intraday setup cannot be
paper-only by construction — `enabled: true` is the same flag live reads. `earnings_downshock` is
paper-only **by deployment** (the daemon runs `--paper-trading`), not by design. Anyone running
`python main.py` without the flag trades it with real money on zero forward evidence.

**Fix:** wire the intraday gate to `paper_enabled if paper_mode else enabled`, mirroring the
overnight pattern. `paper_enabled: true` is already set so no config change is needed after.

### 9.1b ⚠⚠ Event data is NOT deployed by git, and NOT refreshed by anything
`data/` is in `.gitignore`. `data/earnings_calendar/` therefore **never reached the VM at all** —
found on 2026-08-03, the setup's first live paper session, which logged
`DISPATCH_BUILD_UNIVERSE | earnings_downshock_continuation_short | 0 symbols`. The setup was
registered, dispatching, and structurally incapable of firing: no calendar → no reaction day is
ever flagged → empty universe, forever, with no error. 150 qualifying announcements existed on
2026-07-30/31; the day was simply lost.

Two separate problems, both still open:
1. **Deployment.** Event parquets must be copied to the VM by hand (`scp`). `asm_gsm_events.parquet`
   happened to be there; `earnings_events.parquet` was not. Nothing checks.
2. **Freshness.** The calendar is a static file. `earnings_downshock` needs the *previous trading
   day's* announcements, so a stale calendar silently produces zero fires. It was frozen at
   2026-07-28. **A daily refresh cron is required** (`tools/earnings_calendar/fetch_earnings.py
   --start <T-7> --end <T>`, which merges), plus a push to the VM.

Until that cron exists, this setup produces no forward data — which is the entire point of the
2026-08-01 pre-registration. Treat as blocking.

**Add a startup assertion**: any setup whose universe depends on an event feed should fail loudly
at load if the feed is missing or older than N days, rather than dispatching an empty universe
every session.

### 9.2 ⚠ Nothing restarts the intraday paper daemon
`main.py` runs one session and exits at EOD. No cron entry starts `trading_fixed`. It must be
started manually each trading day, or the forward data the pre-registration depends on will not
accrue. Needs a guarded wrapper script + cron entry (~08:30 weekdays).

### 9.3 Enrichment coverage unverified for 4 setups
The May–Jul 2026 monthly 5m files lack `vwap/adx/rsi`. Verified harmless for `earnings_downshock`;
**not** checked for `up_spike_fade_short`, `or_window_failure_fade_short`, `long_panic_gap_down`,
`panic_crash_revert_long`. Any backtest over that window for those setups may be silently degraded.

### 9.4 March/April 2026 replay universe is narrow
Their per-symbol 1m archives hold ~1,580 symbols vs ~2,196 elsewhere (built against the stale
instrument map). The bucket copies are better than a local rebuild, so they were left alone —
but those two months are not universe-comparable to their neighbours.

### 9.5 Order layer sends MARKET only
`EXECUTION_WORKING_UNSUPPORTED` fires on every `earnings_downshock` trade; working/limit entry
orders are configured but unimplemented on this path.

### 9.6 Edge-integrity monitors not fully wired
`jobs/check_edge_integrity.py` exists and `dry_run` is on. Cron wiring and a 1-week calibration
are pending. `services/cb_state.py` is now an **allowlist** (`enabled`, `forward_validation`) —
unknown values block. Note `cb_state: null` **blocks**, while a *missing* key defaults to active.

### 9.7 Scheduled reviews
- `close_dn_overnight_long` — August performance review ~Sep-1 (July: backtest 0.91 / mirror 1.04 /
  live 0.47). Tripwire may auto-pause ~Aug-25.
- C-09 fresh-pool re-shoot at n ≥ 85, ~Nov-2026.
- Multi-day composite selector is merged locally (`bf30013`) but **not pushed**; needs a paper run.

### 9.8 Repo hygiene
Dozens of `_tmp_*.py` research scripts and `2026*_full/` backtest dirs sit untracked at repo root.
`tests/` collection has ~11 pre-existing errors from retired setups whose modules were deleted, and
56 pre-existing test failures — establish that baseline before blaming a change.

---

## 10. Where to look

| topic | file |
|---|---|
| every parameter | `config/configuration.json` |
| project rules | `CLAUDE.md` |
| lifecycle + amendments | `docs/setup_lifecycle.md` |
| why setups died | `docs/retired_setups.md` |
| multiple-testing ledger | `docs/experiment_ledger.{md,jsonl}` |
| engine internals | `docs/ENGINE_ARCHITECTURE.md` |
| backtest internals | `docs/BACKTEST_ARCHITECTURE.md` |
| lessons from past mistakes | `tasks/lessons.md` |
| backtest issues / disabled gates | `analysis/backtest_findings.md` |
| per-setup briefs | `specs/` |

**Verification commands**
```bash
.venv/Scripts/python -m pytest tests/ -q --continue-on-collection-errors   # expect 56 pre-existing failures
python main.py --dry-run --session-date 2026-05-05                          # single-session smoke test
```

---

## 11. Book economics + parity findings (2026-08-11)

### 11a. Overnight (`close_dn_overnight_long`) — the live-vs-paper gap, decomposed

Live ran ~1% per trade below the paper mirror. It decomposes into three distinct
causes, only one of which is a bug:

| cause | size | status |
|---|---|---|
| **selection** — the `conviction` ranker | −0.474%/trade | **FIXED 2026-08-04** |
| **execution** — fills vs idealized | −0.101%/trade | measurable, partly structural |
| **validation optimism** — paper books an unreachable price | PF 3.69 → ~1.28 | structural |

**Selection.** `conviction` ordering (deepest |svr| first) was *anti-predictive*:
taken −0.306% vs skipped +0.624%, at the **0.0th percentile of 20,000 permutations
(p=0.0001)**. Replaced by `unbiased_hash` on 2026-08-04. Post-fix sits at the
**25.9th percentile (p=0.26)** — statistically random, exactly as designed. Only 9
post-fix trades exist, so this is "no longer broken", not "proven good".

**Execution.** Measured self-contained from the live ledger, which carries its own
`idealized_entry` / `idealized_exit` (no cross-ledger matching needed): entry
slippage median +3.8bp / mean +7.5bp, exit median 0.0bp / mean −2.8bp, **net
−0.101%/trade**. 39 of 79 trades land within ±0.25%; the damage is a tail of six
entries at **92–147bp**.

**The entry basis does not match between books.** In `overnight_handlers.py`
around line 668, the marketable-limit is built as `ref_px * (1 + buffer/100)`
where `ref_px` is the **live LTP at 15:26** (falling back to `plan.entry_price`
only when no quote is available). The paper mirror books `plan.entry_price` = the
**15:25 close**. So worst-case entry vs the mirror is `(15:25 to 15:26 drift) +
1%`, not 1% — which is how PIONRINV (2026-07-20) filled at +146.6bp against a
nominal 1% cap. The config comment claiming the buffer "covers the 15:25 to 15:26
drift" is **wrong**: it stacks on top of it.

`ENTRY_BASIS` logging was added 2026-08-11 (plan px, ref px, drift, buffer, limit,
and limit-vs-plan in bp) so the drift-vs-buffer split is attributable. **No pricing
change was made** — repricing off `plan.entry_price` would cap the gap but
introduce adverse selection, only filling when the name has *not* run. Decide that
once the logged drift is visible.

### 11b. Overnight economics — MTF is correct, do NOT switch to CNC

Per-trade, paper recon-era:

| product | n | lev | gross | fees | net | **return on capital** |
|---|---|---|---|---|---|---|
| CNC | 90 | 1.00 | 0.403% | 0.223% | 0.180% | **+0.180%** |
| MTF | 105 | 2.92 | 0.589% | 0.288% | 0.301% | **+0.882%** |

**MTF interest is only 0.065% of notional.** On the same capital MTF returns ~4.9x
CNC; on the counterfactual (same MTF trades booked as CNC) it is still +0.882% vs
+0.366%, because CNC needs 2.9x the capital for identical notional.

The irreducible cost is **delivery STT, 0.1% each side = 0.2% round-trip**, charged
identically on both products. No product choice escapes it on an overnight hold.

**Do not read "costs consume 71% of gross edge" as marginal economics** — that
measures return on *notional*, and this book deploys *margin*. On capital employed,
+0.882%/trade at ~1-day holds is a good return. Caveat: the two cohorts are not
randomly assigned (MTF gross 0.589% vs CNC 0.403%), so product correlates with
something; a clean test needs the same names run both ways.

### 11c. Tripwire is NOT affected by the ledger's mixed regimes

The paper ledger holds two eras — 93 `t1_settle` rows (2026-06-10 to 06-24,
pre-live, PF 3.625) and 195 `reconstructed_paper` rows (2026-06-29 onward, PF
1.728). Pooled they read PF 1.925, which is misleading for hand analysis — **but
`DecayTripwire._rolling_pf` slices `self._trades[-window_trades:]`, a trailing
30**, so the June rows fall far outside the window and have zero effect. Checked
before changing anything; **no fix needed**. Just segment when analysing by hand.

### 11d. Cross-setup correlation — capital allocation implications

Measured on daily P&L (intraday: 45 sessions; multiday: 22 entry days).

**Intraday setups do NOT share a factor.** Mean pairwise correlation −0.087,
diversification ratio 3.31, effective independent bets ~11 vs 6 nominal.
*Caveat:* with 45 observations the SE on a correlation is ~0.15, so the negative
mean is **indistinguishable from zero** — the honest claim is "uncorrelated", not
"actively hedging". Per-setup capital silos therefore under-use real
diversification: `long_panic_gap_down` is capped at 3 while firing p90=7, whilst
`panic_crash` holds an unused cap of 5.

**Multiday setups largely ARE one factor.** crash2d x zscore **+0.68**, crash2d x
mtf +0.50, mtf x zscore +0.38; only `low52` diversifies (−0.19 to +0.08).
Per-setup slots would let the book hold the same capitulation bet four times.
Book-level unique positions/day: median 4, **p90 8**.

**Multiday has no capital management at all** — `max_new_per_day: 100`,
`max_concurrent: 200` ("effectively take-all"), margin-pool arbitration explicitly
out of scope, `max_slots: None`. Its paper P&L is therefore an **unconstrained
upper bound and not achievable**: adding real slots forces a selection policy, and
the overnight book has already measured what a bad one costs (+0.182% to −0.465%).

**Any cross-setup multiday analysis must dedupe on (symbol, entry_date) first.**
The composite selector holds one book position but credits every contributing
setup's tripwire — 28% of pooled rows are duplicate attributions, weighted toward
high-consensus names.

### 11e. Day-of-week: investigated and REJECTED

"Multiday's worst day is Friday" is an **exit-day artifact** — by entry day Friday
is the *best* (+2.456%) and Wednesday the worst (−1.324%); with 2-day holds,
Wednesday entries exit Friday. Weekend-spanning holds are *better* (+0.367%,
t=+0.59), so weekend risk is not the mechanism.

Wednesday survives dedup at t=−2.07 / p=0.017 **but is not actionable**: only 4
distinct Wednesdays exist, **one day (07-29, three trades) is 76% of the loss**,
and the most recent and largest Wednesday (n=13) was positive. Excluding 07-29 the
gap collapses from −1.88% to −1.06%. The expiry hypothesis is untestable here —
NIFTY weekly expiry falls every Thursday, so every Wednesday entry faces it
equally. **Revisit at ~20 Wednesdays, not 4.**

### 11f. Capital management — what is derived and what is not

Standard and verifiable: diversification ratio `DR = sum(w_i * sigma_i) / sigma_p`
and `N_eff = DR^2` (Choueifaty & Coignard 2008); equal-weight constant-correlation
variance `sigma_p = sigma * sqrt((1 + (n-1)*rho) / n)`. At rho=0.5, n=8 the risk
ratio is **2.121x** versus independent — so a correlated sleeve must size down ~2x
for equal risk.

**Not derived — placeholders only:** every specific concurrency cap (p90 was
chosen by convenience), the per-setup concentration limit, and the target daily
vol. Sizing has **not** been derived from measured edge and variance; doing that
properly needs fractional Kelly with an uncertainty haircut plus a capacity model,
on mu estimates whose confidence intervals currently cross zero. **Do not treat
the earlier cap numbers as research.**
