# `close_dn_overnight_long` — Live Activation Design

**Date:** 2026-06-22
**Setup:** `close_dn_overnight_long` (system's first overnight / MTF setup)
**Status:** Design approved; spec → implementation plan next.
**Predecessor spec:** `specs/2026-05-21-close_dn_overnight_long-paper-trade-implementation-spec.md`
**Cell-lock (authoritative config):** `tools/sub9_research/close_dn_overnight_long_cell_lock.json`

## Goal

Take the paper-validated `close_dn_overnight_long` setup to **real-money live** trading on
Zerodha, at small pilot size, with broker-side catastrophe protection. The signal
pipeline, detector, slot pool, fee model, and decay tripwire are already built and
paper-validated; this spec covers **only the live-activation gap**: real order routing,
the AMO-window-correct cron architecture, GTT stop-loss, position sizing, and rollout.

## What already exists (do NOT rebuild)

- Detector `structures/close_dn_overnight_long_structure.py` (+ tests)
- `OvernightSlotPool` with JSON-persisted T0→T+2 lifecycle (`services/capital_manager.py`)
- Cron handlers `services/execution/overnight_handlers.py` (`run_entry`, `run_verify_exit`) — with
  paper *and* draft live branches
- MTF/CNC fee model `tools/sub7_validation/build_per_setup_pnl.py`
- Decay tripwire `services/risk/decay_tripwire.py`
- `KiteBroker.place_order` already supports `product=MTF` + `variety=amo`
  (`broker/kite/kite_broker.py`, test `tests/broker/test_kite_broker_mtf_amo.py`)
- Candidate pre-filter / baseline build (`services/execution/close_dn_baseline_build.py`)
- Cron wrappers `scripts/cron-entry.sh`, `scripts/cron-verify-exit.sh`

## Zerodha facts (researched 2026-06-22, with sources)

These drove the design; verify again if Zerodha policy changes.

1. **AMO window — NSE equity: 4:00 PM → 8:58 AM.** Standard market/limit AMOs sent to the
   exchange at 9:00 AM; SL/IOC types at 9:15. Regular orders placed after hours auto-convert
   to AMO. → The exit AMO must be placed in a **post-16:00 cron**, never inside the 15:26 entry run.
   Source: https://support.zerodha.com/category/trading-and-markets/charts-and-orders/order/articles/auto-amo
2. **Same-day-purchase sell restriction.** Without DDPI/POA, *CNC/MTF sell orders for same-day
   purchases can only be placed after 6:30 AM the next day.* **This account HAS DDPI enabled**
   (confirmed by owner 2026-06-22), so the exit AMO **can** be queued on T0 evening (≥16:00).
   Source: (same auto-amo article)
3. **GTT supports MTF — confirmed.** *"GTT is available only for MTF, CNC, and NRML."* GTT-OCO
   sets stop-loss + target together. App support lagged historically; the **API path works**.
   Sources: https://zerodha.com/z-connect/featured/mtf-updates ,
   https://support.zerodha.com/category/trading-and-markets/charts-and-orders/gtt/articles/what-is-the-good-till-triggered-gtt-feature
4. **MTF settlement is T+1.** *"Sale proceeds … only available on the next day."* → confirms the
   pool's T+2-from-entry release (= T+1-from-sale) is correct, not over-conservative.
   Source: https://support.zerodha.com/category/trading-and-markets/margins/margin-trading-facility/articles/margin-trading-facility-mtf-faqs
5. **MTF interest: 0.04%/day from T+1 until sold** (matches config `interest_pct_per_day: 0.0004`);
   intraday square-off = no interest; T0-buy/T+1-sell = 1 day interest (matches `hold_days = max(1, …)`).
   Auto-square-off RMS (~20% of funded amount → proportionate) is a backstop *behind* our 5% GTT.
   Source: (MTF FAQs + RMS policy)

## Architecture

### A. Hybrid broker (data vs orders)

Live reuses the **exact paper-validated data path** (Upstox) and only flips the order sink to Kite.

New `LiveOvernightBroker` composing:
- `UpstoxDataClient` exposed as `_data_sdk` → provides `async_fetch_intraday_5m_batch`,
  `get_intraday_5m`, `get_daily`, `list_symbols`, symbol/token maps, and feeds
  `close_dn_baseline_build` (which keys off `_data_sdk`, so the baseline build runs in live too —
  it is currently gated `if paper_mode:` and must be un-gated to also run when `_data_sdk` exists).
- `KiteBroker` → `place_order`, `get_order_status`, `get_ltp`, GTT methods.

Rationale: the signal computation, universe build, and candidate pre-filter are byte-identical to
what paper validated. Nothing in the signal path changes paper→live.

### B. Cron schedule (live; `MODE_FLAGS=""` → live + Kite)

| Time (IST) | Action | Work |
|---|---|---|
| **15:26 (T0)** | `entry` | compute signal → reserve slot → **MTF MARKET BUY** (regular) → poll fill via `get_order_status` → attach buy fill → persist (`t0_open`). **No AMO here.** |
| **16:05 (T0)** | `place-exit` *(new)* | for each `t0_open` slot with a buy fill and no `amo_sell_order_id`: place **AMO SELL** (product=MTF, variety=amo) for next-day open **+** place a **single-leg GTT stop-loss** (trigger = entry × `(1 - catastrophe_stop_pct/100)`, limit = trigger × `(1 - gtt_limit_buffer_pct/100)`); persist `amo_sell_order_id` + `gtt_id`. |
| **09:30 (T+1)** | `verify-exit` | verify AMO filled at open → compute fees/interest → `settle()` → **cancel the GTT** (`broker.cancel_gtt(slot.gtt_id)`) → record to decay tripwire → release slots whose T+2 cash-settle day has arrived. |

Morning fallback (no DDPI, or evening cron missed): `place-exit` may also run ~07:30 on T+1 (past the
6:30 AM same-day-sell gate, before the 8:58 cutoff). DDPI is enabled here so evening is the primary path;
the handler must be idempotent so a 07:30 re-run is safe.

### C. Position sizing

Unchanged math from paper (`plan_long_strategy`): `notional = margin_per_slot_inr × leverage`,
`qty = int(notional / entry_price)`. Only the rupee knobs change:

| Config key | Paper | Live |
|---|---|---|
| `margin_per_slot_inr` | 100000 | **10000** (₹10k base/trade → ~₹26k notional @ 2.64×) |
| `active_margin_inr` | 10000000 | **200000** (₹2L total) |
| `max_concurrent_slots` | 100 | **20** (`active_margin / margin_per_slot`) |
| `max_new_positions_per_day` | 100 | **20** (kept high — never drop a signal) |

The 2-day settlement lock (T0→T+2) means effective concurrent demand ≈ fires/day × ~2; ₹2L / 20 slots
gives headroom for ~10 fires/day. (₹1L/10 slots is the smaller alternative.)

### D. Risk — GTT catastrophe stop

- Single-leg GTT stop-loss placed at 16:05 alongside the AMO SELL; trigger = entry × 0.95
  (`catastrophe_stop_pct: 5.0`), limit a small buffer below trigger (`gtt_limit_buffer_pct`) to
  ensure fill. (Not OCO — the profit/exit leg is already the AMO at the open.)
- **Genuine role:** the AMO already exits at the next open, so the GTT is a *failsafe* for the case
  where the AMO does **not** fill (auction non-clear / rejection) and the position lingers into the day.
- **Mandatory cancel-on-settle:** `verify-exit` must cancel the GTT once the AMO fill is confirmed.
  A dangling GTT that later triggers on a flat position would open a **naked short**. Cancellation is
  part of the settle path, not best-effort.
- Implementation risk: if a specific symbol/leverage combination is rejected for GTT-MTF by the API at
  runtime, fall back to leaving only the AMO SELL (logged `GTT_PLACE_FAILED`), and rely on the morning
  failsafe market-SELL path already in `run_verify_exit`.

### E. Code deltas

1. `broker/live_overnight_broker.py` — NEW: composite (Upstox data + Kite orders).
2. `broker/kite/kite_broker.py` — add `get_order_status(order_id) -> {"status","average_price"}`
   (wrap existing `get_order_fill_price` / `kc.orders()`), and `place_gtt_stop(...)` (single-leg) +
   `cancel_gtt(gtt_id)` (wrap `kc.place_gtt` / `kc.delete_gtt` / `kc.get_gtts`).
3. `services/execution/overnight_handlers.py`:
   - `run_entry`: **remove** AMO-SELL placement (BUY + persist only).
   - `run_place_exit` (NEW): place AMO SELL + GTT for `t0_open` slots; idempotent.
   - `run_verify_exit`: cancel GTT after settle.
   - un-gate the candidate baseline build to run whenever `broker._data_sdk` exists (not `paper_mode` only).
4. `services/capital_manager.py` — `OvernightSlot.gtt_id: Optional[str]` field (+ persistence round-trip).
5. `main.py` — add `--action place-exit` (overnight mode), wire to `run_place_exit`.
6. `config/configuration.json` — live caps (table C), `catastrophe_stop_pct`, GTT limit-offset config;
   flip `enabled: true` **only after preconditions G pass**.
7. `scripts/cron-place-exit.sh` — NEW (16:05). `scripts/cron-entry.sh` — drop AMO note; live crontab
   variant documented (`MODE_FLAGS=""`).

### F. Known live-vs-paper delta (watch in pilot, not a blocker)

Paper fills entry at the 15:25 bar **close**; live is a MARKET BUY at ~15:26 → small entry-price gap.
Track median deviation in the pilot; add a slippage model only if it exceeds ~10 bps.

### G. Preconditions before flipping `enabled: true`

1. Confirm the **VM forward-paper ledger** exists and results are acceptable (local `state/` is
   backtest-seeded and stale — not evidence).
2. Confirm a **Kite daily access-token refresh** mechanism (reuse the live intraday daemon's); the
   15:26 / 16:05 crons need a valid `KITE_ACCESS_TOKEN` in env.
3. Dry-run E2E on a known Cell-#5 fire date (entry → place-exit → verify-exit) produces a clean trade.
4. **1-slot live pilot** (`max_concurrent_slots: 1`, `max_new_positions_per_day: 1`) for several
   sessions; watch real fills + GTT place/cancel; then scale to full caps (table C).

### H. Testing

- `LiveOvernightBroker` routes data→Upstox, orders→Kite (no Kite data calls, no Upstox order calls).
- `KiteBroker.get_order_status` shape; `place_gtt_stop` / `cancel_gtt` params.
- `run_place_exit`: places AMO + GTT for `t0_open` slots; idempotent on re-run; skips slots already
  carrying `amo_sell_order_id`.
- `run_verify_exit` cancels the GTT after settle; never leaves a dangling GTT on a flat slot.
- Sizing at `margin_per_slot_inr=10000` → correct qty/notional/margin.
- AMO-window timing guard: `place-exit` refuses to place AMO before 16:00 (and entry never places AMO).
- Regression: paper mode unchanged; intraday MIS path unaffected.

## Project rules honored (CLAUDE.md)

1. **No hardcoded defaults** — `catastrophe_stop_pct`, GTT offsets, live caps all in
   `config/configuration.json`.
2. **IST-naive timestamps** — all order/slot timestamps via `utils/time_util.py`.
3. **Live/paper/backtest compatible** — hybrid broker keeps the signal path identical; the simulator
   still drives time-stepping in backtest.
4. **Indian-market focus** — Zerodha AMO window, DDPI same-day-sell rule, MTF T+1 settlement, MTF
   interest, and GTT-MTF support are all baked to verified docs.

## Deferred past pilot

- Slippage model for the entry timing gap (F) — add only if pilot shows > ~10 bps median.
- Multi-overnight-setup slot-pool sharing (only this setup today).
- Weekly automated MTF approved-list refresh (`tools/scrape_zerodha_mtf.py` is manual).
