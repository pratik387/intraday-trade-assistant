# Live Trading Incidents — Found and Fixed

Every defect below reached **live or paper execution**, not backtest. Each is
reconstructed from the inline code comment, the fix commit, or the session that
recorded it — sources are cited so each entry can be re-verified.

The organising fact: **almost none of these were strategy errors.** They are
execution, state, timing and accounting failures — and the majority were
*silent*, surfacing hours or days later as a different symptom in a different
log file.

Compiled 2026-08-19.

---

## 1. Order placement and broker mechanics (real money)

### 1.1 Per-instrument NSE tick size — 2026-07-02
**Symptom:** Kite rejected orders outright — *"Tick size for this scrip is 0.10"*.
**Cause:** prices were rounded to a blanket 0.05. NSE ticks are **per-scrip**
(0.01 / 0.05 / 0.10 bands).
**Cost:** the day's **top-2 ranked MTF picks** (NPST, MIDWESTLTD).
**Fix:** parse the tick out of Kite's own rejection text, re-round
**directionally** (BUY floors / SELL ceils, so it stays marketable and inside any
circuit clamp), retry once; plus `KiteBroker.get_tick_size()` reading the
instruments dump, wired into every order-pricing site.
`broker/kite/kite_broker.py:304,621` · commit `0747470`

### 1.2 Asynchronous REJECT burned the poll window — 2026-07-06
**Symptom:** a BUY LIMIT was accepted, then rejected asynchronously —
*"Insufficient funds. Margin required: 152010.12, available: 149081.20"*.
**Cause:** `_live_poll_fill` only checked for `COMPLETE`, so a rejected order
consumed the whole poll timeout, was misread as "did not fill", got no retry, and
left a **ghost `t0_open` slot** needing manual cleanup.
**Fix:** `_live_poll_fill_ex` stops immediately on REJECTED/CANCELLED; parse
required/available margin and place one reduced-qty retry;
`_rollback_slot_to_free` clears the ghost slot.
`overnight_handlers.py:2053` · commit `8424ff4`

### 1.3 Unfilled BUY pinned account-wide margin — 2026-07-09 (AMIRCHAND)
**Symptom:** **fired 1 of 3.** Every lower-ranked pick rejected for insufficient
funds.
**Cause:** an unfilled BUY left **OPEN** kept ~₹50k of margin blocked
*account-wide*. Compounding it, Kite's *"Margin required"* is **cumulative** (the
whole account's requirement), so the ratio-scaled retry from 1.2 barely shrank
the order and bounced too.
**Fix:** cancel a BUY still OPEN after its poll window, freeing margin for the
next candidate; and size the retry by **shortfall** —
`S = required − available`, `M = qty × price / leverage`,
`retry_qty = floor(qty × (M − S)/M × haircut)`.
`overnight_handlers.py:1942,2005` · commit `d0a5c7f`

### 1.4 Partial fill orphaned → unhedged overnight — 2026-07-15 (AJOONI)
**Symptom:** **1,751 real shares held overnight with no exit order.**
**Cause:** the entry BUY partially filled; the poll-timeout cancel killed the
remainder; the `CANCELLED` status was treated as "no position" and the slot
rolled back. Every dead-order path checked *status* and never `filled_quantity`.
**Recovery:** manual, next morning inside the AMO window.
**Fix:** `_partial_fill_from_status()` gates **all three** rollback paths — a
partial attaches as a real position, so place-exit hedges it at 16:05.
`overnight_handlers.py:1983` · commit `a1fe8a1`

### 1.5 Exit AMO rejected on the next-day circuit band — 2026-07-20 (SABEVENTS)
**Symptom:** exit AMO limit 7.37 sat inside Friday's band but **below Monday's
re-centred floor (7.50)** → rejected at the 08:58 exchange send → position left
unhedged. The 09:33 failsafe then rejected on zero holdings, because the user had
already flattened manually at 7.51.
**Fix:** clamp the AMO SELL limit to the **estimated next-day** lower circuit
(band fraction inferred from today's quote, re-centred on close); and scan the
orderbook for an out-of-band COMPLETE SELL matching symbol+qty and **adopt** it
instead of double-selling.
`overnight_handlers.py:200` · commit `1c9819f`

### 1.6 Stale Kite token failed everything silently — 2026-07-24
**Symptom:** verify-exit ran with a dead token; every status fetch, failsafe and
GTT cancel failed one by one.
**Cause:** `get_orders()` / `get_order_status()` swallow auth errors into
`[]` / `None`, so "no orders" and "dead token" were indistinguishable.
**Fix:** `KiteBroker.check_auth()` explicit probe (cheap margins call); verify
aborts cleanly so the 10:30 retry cron gets one consistent pass.
`broker/kite/kite_broker.py:576` · commit `96b4951`

### 1.7 Partially-filled exit AMO blocked its own unwind — 2026-07-24 (REGENCERAM)
**Symptom:** 286 of 1,388 filled at the floor price; the remainder sat **OPEN
above the market all morning**, locking the holdings so a second SELL would be
rejected.
**Fix:** `modify_order()` re-prices the existing order to marketable — the only
safe unwind; the failsafe sells **only the unsold remainder**.
`broker/kite/kite_broker.py:594` · commit `96b4951`

### 1.8 Cancel/fill race — the exchange filled after the cancel — 2026-07-24 (CREATIVEYE)
**Symptom:** **7,520 real shares unhedged across a weekend** (manually exited
Monday, net −₹1,646).
**Cause:** the entry poll timed out, the engine cancelled, and Kite's
cancel-confirm reported `CANCELLED / filled=0` — but the exchange had already
filled **7,520 / 7,587**. The order report lagged the snapshot. The phantom
margin also downsized the next pick (TFCILTD retry 571→495).
**Fix:** `_reconcile_unattached_buys` re-scans the day's orderbook for
engine-tagged BUYs with `filled_quantity > 0` that no slot knows about and
attaches them — at the end of `run_entry` and the start of `run_place_exit`, so a
caught fill still gets its exit AMO in the same 16:05 pass.
commit `2b3c392`

---

## 2. Cron timing and the placement budget

### 2.1 Entry loop blew the 15:26–15:30 window — 2026-07-30
**Symptom:** 7 fires became **2 dust partials** (BANG 461/1655, SAYAJIHOTL
5/172), 1 order placed into the close, and **4 "Markets are closed" rejections**.
**Cause:** a 101.5s batch fetch (hardcoded 20 rps over 2,047 symbols) pushed
first placement to ~15:28, and each candidate then held a **serial 60s** fill poll.
**Fix:** `entry_fetch_rps` 20→30, `fill_poll_timeout_sec` 60→20, and a hard
`entry_placement_cutoff_hhmmss = 15:29:20` — **LIVE only**, since it is an
exchange-acceptance deadline, not a signal decision. Paper/backtest skip it
(idealised instant fills).
commit `6fb9155`

### 2.2 The rps fix caused a 429 storm — 2026-07-31
**Symptom:** **5,148** 429-retries; 328 of 2,044 symbols in **409.9s**; fetch
finished 15:33; the cutoff correctly zeroed the day.
**Cause:** 30 rps collided with the co-located paper process on a shared IP.
**Fix:** reverted to 20 rps — the proven budget (101.5s, zero 429s).
commit `5afceea`

> A fix for one budget problem directly caused the next. The two are coupled
> through a shared external rate limit that neither change accounted for.

### 2.3 NameError staled the candidates snapshot — 2026-08-14 → 08-18
**Symptom:** the live overnight book **placed nothing for three sessions** —
6, 1 and 3 ranked signals forfeited.
**Cause:** an incomplete rename (`paper_enabled_setups` → `managed_setups`) done
with a **bounded line-range replace** left one reference past the range, inside
the best-effort baseline-build block. Its `except Exception` swallowed the
`NameError`, so exits kept settling and the run reported success — while
`candidates_latest.json` silently stopped updating. `run_entry` then found it
stale every day and fell back to a full universe rebuild (~2,344 symbols).
**Measured:** startup 0.8s; `run_entry` entered 15:26:03; its internal timer
started **15:29:55** — 232s in the fallback, past the 15:29:20 guard.
**Fix:** the reference; the failure now logs **CRITICAL** and lands in the run
summary; 35 static AST guards check every top-level function for undefined names
(no linter is installed, and no unit test reaches that branch).
commit `05a7a5f`

> Diagnosis took five wrong hypotheses. The decisive clue — *"it was working fine
> till Thursday"* — matched the last candidates file exactly.

---

## 3. Detectors that passed backtest and could not fire live

### 3.1 `close_dn` prior-day return returned None for every live candidate — 2026-06-09
Two bugs in sequence. The 2026-05-23 fix (`ce5f770`) corrected a literal
"prior day" reading — but computed **both** closes from `df_5m`. In production
`df_5m` is **today-only** (the Upstox intraday endpoint returns just the current
session), so the prior-session lookup was always empty and the method returned
`None` for **every production candidate**. Cell #5 had been unable to fire in
production at all since the previous "fix".
**Fix:** read yesterday's close from `ctx.df_daily` first, falling back to the
`df_5m` derivation for backtest.
`close_dn_overnight_long_structure.py:333-350`

### 3.2 Bar-label mismatch blocked every candidate — 2026-08-11 (earnings_downshock)
**Symptom:** **9 candidate-days, 0 fires.**
**Cause:** live evaluates the 09:15 bar at ~09:21–09:23 wall clock, by which time
Upstox has surfaced the **partial 09:20-labelled** bar — so a strict
`last bar == 09:15` window rejected everything. Backtest replay feeds bars only
to 09:15, so Stage-8 parity never exercised the path.
**Fix:** accept 09:20 as a **sentinel proving the 09:15 bar is complete**; its
data is never read. Same pattern as the overnight detector's `_ACTIVE_HHMMS`
(2026-06-09).
`earnings_downshock_continuation_short_structure.py:120` · commit `7db7a70`

### 3.3 Opening-bell universe collapse — 2026-06-04
At the 09:15 scan only 1–2 bars exist, and `min_bars=3` marked the whole universe
as failed: the SDK returned 1,525 dataframes (1,522 with `len=2`) and all were
discarded. `gap_fade_short` and `long_panic_gap_down` — both 09:15 setups — ran
on a tag-map subset of ~9–50 symbols instead of the full universe.
`services/screener_live.py:1400`

### 3.4 Event feed never deployed → empty universe all session — 2026-08-03
`earnings_downshock`'s first live paper session fired nothing: it logged
`DISPATCH_BUILD_UNIVERSE | 0 symbols` every bar because `data/earnings_calendar/`
had never been copied to the VM (`data/` is gitignored). **150 qualifying
announcements existed that day**, and nothing was raised.
**Fix:** `main.py::_refresh_event_feeds()` tops feeds up at daemon start
(paper/live only — never backtest, so the archive stays frozen) and logs
**CRITICAL** if a feed is missing or older than `max_staleness_days`.

---

## 4. Selection and sizing economics

### 4.1 Conviction slot ranking was anti-predictive — 2026-08-04
When more signals fire than slots, the choice of which to take was
deepest-`|svr|`-first — validated on Disc/OOS/HO, and it **reversed forward**:

| measure | value |
|---|---|
| July live picks | **−43.6 bps** mean |
| Fires the ranker skipped | **+72.0 bps** |
| Permutation test | **p < 0.001** |
| Actual picks vs 3,000 random draws | **0th percentile** |
| Attributable | ~**−₹11.5k** of the −₹17.9k July live net |

**Fix:** `slot_ranking_mode: unbiased_hash` — deterministic `sha1(date|symbol)`
ordering, expressing no view on fire quality, so live becomes a random sample of
the day's fire-set.
commit `4adcd63`

> Same forward-reversal signature as the parked monster-conditioning rule.

### 4.2 Sizing fell through to an accidental path — 2026-08-13
Setups that declared no `sizing_mode` inherited `qty = risk / stop_distance`,
where a **tight stop mints a huge position**: `or_window_failure_fade_short`
carried a **₹112,822 median notional** — 3.8× the other setups — and loses money
(−0.432%/trade). Nobody chose that size. Sizing also ran on **two code paths**
(orchestrator, then a silent executor override).

Separately, `max_allocation_per_trade: 0.2` applies 20% to **margin**, so at 5×
MIS it permits **₹500k of notional** per trade and never bound — the largest
observed position was **₹182,970, 37% of capital**.

**Fix:** `sizing_mode` required and validated per setup; one sizing path; a single
`[min, max]` **notional** clamp on every mode.
`services/risk/intraday_sizing.py` · commit `c1d4504`

### 4.3 Per-setup capital budget tracked but never enforced — 2026-08-17
`capital_budget_pct` is read from config, wired into `CapitalManager`, and its
usage incremented on open and decremented on release — but `setup_budgets_pct`
appears **three times** in the file: once to initialise, twice as a membership
test deciding whether to *track*. **Nothing ever compares usage to the budget.**
Four concurrent `earnings_downshock` positions therefore held ₹100k margin each —
**80% of ₹5L against a 20% budget**, 5× the footprint its brief validated.

**Status: found, implemented, PARKED** (`99ec1e1`, reverted in `404030c`) —
enforcing it at the current 10× multiplier converts a 5-position spread into one
concentrated position, which is a live decision, not a cleanup.

---

## 5. Position lifecycle and state

### 5.1 Disabling a setup stranded the positions it held — 2026-08-13
Turning `crash2d` off while it held **10 filled positions and 3 pending AMO
entries** removed it from `_eligible_multiday_setups` — the list that drove
**three** legs: new entries, entry-fill verification, **and exits**. The 10
positions lost their exit path entirely (NSE:CLSEL was due to square off that
day), and the 3 pending entries could neither fill nor be cleaned up.

The same defect was **latent in the overnight book**, where it is worse: the exit
legs read the slot-pool path from `setups[0]`, so disabling the only overnight
setup returns `[]` and an **open real-money position is never sold**.

**Fix:** `_managed_*` (everything that could be holding something, flags ignored)
drives wind-down; `_eligible_*` still gates new entries. Pending entries of a
disabled setup are **dropped, not filled**.
commits `019f876`, `a41d36b`, `bbfd3eb`

### 5.2 MTF approved-list delisting mid-hold — 2026-07-14
Refreshing an 8-week-stale snapshot revealed **3 of 11 held paper positions** had
been dropped from Zerodha's MTF approved list mid-hold. Live, Zerodha can
force-convert or square off such positions.
**Fix:** a daily check surfaces held names missing from the (daily-refreshed)
list on the day it happens, rather than at a broker rejection.

---

## 6. Accounting and observability

These changed no orders — they changed what we *believed*, which drove decisions.

### 6.1 Partial exits silently deleted realised profit — 2026-08-18
P&L was summed over `is_final_exit` rows only, counting just the **last leg** of a
multi-exit trade. 15 of 120 trades take a partial T1 exit, and T1 is the **profit
target**, so the winning half was discarded on every one.

| | reported | actual |
|---|---|---|
| Net P&L | −₹1,643 | **+₹10,182** |
| PF | 0.97 | **1.18** |

`is_final_exit` is the right filter for counting **trades** and the wrong one for
summing **money**. The consequent "% positive but rupees negative" anomaly — which
had been written into a module docstring as design justification — was purely this
artifact.

### 6.2 Retired setups pooled into "current book" figures — 2026-08-13/17
Per-session aggregates (`capital_report.json`) carry no setup dimension, so they
answer "what did the book do then", never "what does it do now". Twice: intraday
P&L quoted as −₹48,612 when the live book was −₹1,643 (retired `gap_fade` +
`below_vwap` were −₹47k of it), and capital utilisation quoted at 9.2% of ₹5L when
the active book was **4.39%**.

### 6.3 Capital-derived risk never reached sizing
`main.py` published it via `set_base_config_override` into
`config/pipelines/base_config.json` — which the orchestrator **never reads** (it
reads `configuration.json`). Both files held `1000.0`, so the disconnect was
invisible. Fixed with an explicit `set_runtime_capital()` seam.

### 6.4 Rupee P&L pooled across a book-size change — 2026-08-14
One session at an effective 5–10× moved the cumulative by **₹8,535 more** than the
same trades at 1×, making a **+₹12,515** record read as **+₹1,436**. The book had
not deteriorated; the yardstick changed mid-series. Fixed by splitting History
(current size era) from an Archive tab, and recording per-trade sizing provenance
so any trade can be restated at any size exactly.

### 6.5 Silent-logger and stale-value bugs (May 2026 sweep)
Five production bugs found in one investigation, including:

- **R-mismatch — HIGH:** stale `sizing.rps` after fill, corrupting fast-scalp
  R-multiples (`services/target_recalc.py`, `exit_executor.py:2086`)
- **`bar_scheduler.py` used a dead logger** — every admission rejection went to a
  handler-less logger, so drops were invisible
- **MIS list time-travel** — the live Zerodha sheet applied to historical dates,
  costing 31% of `circuit_release` signals

`analysis/backtest_findings.md` §8c

---

## 7. Live/backtest parity asymmetries (known, accepted)

### 7.1 `close_dn` corporate-action asymmetry — 2026-07-28
Live computes `prior_day_return_pct` from **unadjusted traded prices**; backtests
run on **back-adjusted** feathers. On split/bonus ex-dates live sees a spurious
−50% and can never fire cell #5, while the adjusted backtest sees continuity.
Dividend ex-dates suppress borderline fires across ~1,600 ex-date events/year.
**Direction is one-way — backtest may contain fires live would BLOCK, never false
fires** — so severity is low and it is documented rather than fixed.

### 7.2 WebSocket bar loss for illiquid names
The live WebSocket path missed **10–25% of bars** for illiquid stocks — the root
cause of a 20–30% paper/backtest divergence. Fixed by an API-first Stage-0 at
40 RPS.

---

## Patterns worth carrying forward

1. **Silent failure is the dominant mode.** 1.4, 1.6, 1.8, 2.3, 3.4, 4.3 and 6.5
   all reported success while failing. The recurring shape is a broad `except`
   around best-effort work whose *output* something else depends on. Isolation was
   applied in the wrong direction: it protected the caller from the sub-task, never
   the downstream consumer from the missing artifact.
2. **The cost lands somewhere else, later.** 2.3 failed at 09:30 and cost signals
   at 15:26 the next day, in a different log. Nothing connected them.
3. **Live-only surfaces are where detectors die.** 3.1, 3.2 and 3.3 all passed full
   backtest validation and could not fire in production — bar labelling, bar
   availability and today-only intraday data have no backtest analogue.
4. **Broker semantics are adversarial.** Cumulative margin (1.3), per-scrip ticks
   (1.1), async rejects (1.2), report lag behind the exchange (1.8) and errors
   swallowed into empty results (1.6) each broke a reasonable assumption.
5. **Order-management and data-prep jobs must not share an exit code** (2.3).
6. **A validated ranker can reverse forward** (4.1). Sizing and selection edges
   deserve the same forward-validation discipline as entry signals.
7. **Accounting bugs are decision bugs** (§6). Every one produced a
   confidently-stated wrong conclusion that drove a recommendation.
