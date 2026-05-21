# `close_dn_overnight_long` — Paper-trade Implementation Spec (Cron Model)

**Date:** 2026-05-21 (revised same day)
**Branch:** `research/europe-open-13ist`
**SHIPPABLE record:** `specs/2026-05-21-close_dn_overnight_long-SHIPPABLE-cell-5.md`
**Cell-lock JSON (authoritative config):** `tools/sub9_research/close_dn_overnight_long_cell_lock.json`

## Architecture decision (2026-05-21 revision)

**Cron-triggered scripts, no persistent overnight daemon.** Reasoning:

Once the BUY fills at 15:30 IST and the AMO SELL is queued at the broker for next-day open, the broker holds all the state. Our system has zero work to do during the ~17 hours of overnight hold — no ticks to consume, no signal to recompute, no margin to monitor (broker handles that). Running a daemon during that window would burn RAM/CPU for nothing.

Two cron-triggered Python invocations per trading day:

```
15:25 IST  cron (Mon-Fri):  python main.py --mode overnight --action entry
09:30 IST  cron (Mon-Fri):  python main.py --mode overnight --action verify-exit
```

Each invocation:
1. Loads persisted state from JSON
2. Does its short-lived work (2-5 minutes per run)
3. Persists updated state
4. Exits

The intraday system continues to run as a long-lived daemon (`python main.py` without `--mode overnight`). Both processes share the same code repo; the `--mode` flag drives setup-registry loading.

## Goal

Wire `close_dn_overnight_long` Cell #5 into a **cron-triggered execution model** alongside the existing intraday daemon. This is the system's first MTF setup and first overnight setup. Paper-trade validation is the immediate next step; live activation gated on the cell-lock JSON's `decay_warning.tripwire`.

## Scope

Implementation breaks into **8 ordered tasks** (Task 0 + the original 7). Each is sized at ~1-3 commits. Dependencies form a tree:

```
Task 0 (cron entrypoint + --mode flag)
    │
    ├── Task 1 (fee helpers) ─────────┐
    ├── Task 2 (universe + MTF loader)┤
    └── Task 3 (capital manager state) ┤
                                       │
        Task 4 (detector) ──────────── ┤
                                       │
        Task 5 (order routing) ─────── ┤
                                       │
        Task 6 (entry + verify-exit handlers)
                                       │
        Task 7 (decay tripwire)
```

Tasks 1, 2, 3 are mostly independent of each other; 4 needs 1+2; 5 needs 3+4; 6 needs all of 0-5; 7 bolts on after 6.

## Project rules to honor (from CLAUDE.md)

1. **NO hardcoded defaults** — all thresholds from `config/configuration.json`. Cell-lock values become config entries.
2. **IST-naive timestamps** — use `utils/time_util.py`. Signal-bar timestamp = `15:25:00` naive IST.
3. **Live/backtest compatibility** — cron scripts must work in both paper and live modes. For backtest mode, the simulator drives time-stepping (cron isn't real).
4. **Indian-market focus** — MTF specifics (Zerodha 14.6% APR, NSE pre-open auction, T+1 settlement) hard-baked.

---

## Task 0: Cron entrypoint + `--mode` flag

**Files:**
- Modify: `main.py` — add `--mode {intraday,overnight}` and `--action {entry,verify-exit}` CLI flags
- Modify: `services/dispatch/setup_registry.py` — accept a mode filter when registering setups
- Create: `scripts/cron-entry.sh` — wrapper invoked by cron at 15:25 IST
- Create: `scripts/cron-verify-exit.sh` — wrapper invoked by cron at 09:30 IST
- Test: `tests/test_main_modes.py`

**Scope:**

`main.py` CLI extension:
```python
parser.add_argument("--mode", choices=["intraday", "overnight"], default="intraday")
parser.add_argument("--action", choices=["entry", "verify-exit", "run"], default="run")
```

Routing:
- `--mode intraday --action run` (default): existing long-lived daemon, registers only intraday setups (`gap_fade_short`, `long_panic_gap_down`, `circuit_t1_fade_short`, `or_window_failure_fade_short`, `delivery_pct_anomaly_short`, `below_vwap_volume_revert_long`).
- `--mode overnight --action entry`: short-lived run for 15:25 IST. Registers only overnight setups (`close_dn_overnight_long`). Compute signal, place BUY, place AMO SELL, persist state, exit.
- `--mode overnight --action verify-exit`: short-lived run for 09:30 IST. Load state, verify yesterday's AMOs filled, release slots, persist state, exit.
- `--mode overnight --action run`: deprecated/error — overnight is never a daemon.

`setup_registry.py`:
```python
def get_active_setups(mode: str) -> dict[str, SetupSpec]:
    """Return only setups whose `mode` field matches the requested mode."""
    return {name: spec for name, spec in SETUP_REGISTRY.items() if spec.mode == mode}
```

`SetupSpec` already has a `mode` field (added in Task 4 of original spec — keep here as a Task 0 prerequisite since it's load-bearing for filtering).

Crontab (deployment artifact, lives in `scripts/`):
```cron
# IST timezone assumed. Use `CRON_TZ=Asia/Kolkata` if needed.
25 15 * * 1-5  cd /opt/intraday-trade-assistant && .venv/Scripts/python main.py --mode overnight --action entry      >> logs/overnight_entry.log 2>&1
30 09 * * 1-5  cd /opt/intraday-trade-assistant && .venv/Scripts/python main.py --mode overnight --action verify-exit >> logs/overnight_verify.log 2>&1
```

**Test plan:**

1. `test_main_intraday_mode_default` — running without `--mode` defaults to intraday daemon path
2. `test_main_overnight_entry_mode_loads_only_overnight_setups` — registry exposes only setups with `mode == "overnight"`
3. `test_main_overnight_run_action_errors` — `--mode overnight --action run` raises with clear error
4. `test_cron_wrapper_scripts_exist_and_executable` — both `.sh` files present, +x bit set, correct shebang

**Acceptance gate:** all 4 tests pass; manual smoke: `python main.py --mode overnight --action entry` in a backtest harness exits cleanly within 30s on a no-signal day.

**Estimated effort:** 1 commit, ~150 lines.

---

## Task 1: Fee helpers (CNC + MTF) promoted to `build_per_setup_pnl.py`

**Files:**
- Modify: `tools/sub7_validation/build_per_setup_pnl.py`
- Reference: `tools/sub9_research/sanity_close_dn_overnight_long.py:calc_fee_cnc`
- Test: `tests/sub7_validation/test_fee_modes.py`

**Scope:**

Existing `calc_fee()` is intraday-MIS-only. Add:

```python
def calc_fee_cnc(buy_value_inr: float, sell_value_inr: float) -> float:
    """Round-trip CNC delivery fees. See cell_lock.json fee_model.cnc."""

def calc_fee_mtf(
    buy_value_inr: float,
    sell_value_inr: float,
    margin_inr: float,
    hold_days: int,
) -> float:
    """Round-trip MTF fees + overnight interest.
    interest = (buy_value - margin) * 0.0004 * hold_days
    """

# Existing calc_fee accepts a mode parameter:
def calc_fee(
    buy_value_inr: float,
    sell_value_inr: float,
    *,
    mode: str = "intraday_mis",
    margin_inr: float | None = None,
    hold_days: int = 0,
) -> float:
    """mode in {'intraday_mis', 'delivery_cnc', 'mtf'}."""
```

**Test plan:**

1. `test_calc_fee_cnc_baseline` — Rs 1L buy / Rs 1.005L sell → ~Rs 171 (matches sanity)
2. `test_calc_fee_mtf_one_night` — Rs 2.79L buy / Rs 2.80L sell / Rs 1L margin / 1 day → fee includes ~Rs 72 interest
3. `test_calc_fee_dispatches_by_mode` — `mode="intraday_mis"` matches old, `mode="delivery_cnc"` matches `calc_fee_cnc`, `mode="mtf"` matches `calc_fee_mtf`
4. `test_calc_fee_mtf_friday_to_monday_hold_days_3` — Friday BUY, Monday SELL → hold_days=3 → interest scales 3x

**Acceptance gate:** all 4 tests pass; existing intraday MIS unit tests still pass.

**Estimated effort:** 1 commit, ~150 lines.

---

## Task 2: Universe builder + MTF-list loader

**Files:**
- Create: `services/setup_universe.py::close_dn_overnight_long_universe`
- Create: `services/mtf_universe.py`
- Test: `tests/services/test_close_dn_overnight_universe.py`

**Scope:**

`services/mtf_universe.py` (new module):

```python
"""Loads Zerodha MTF approved-securities snapshot and provides lookup."""
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class MtfInfo:
    tradingsymbol: str
    isin: str
    category: str          # 'fo' | 'non_fo' | 'non_categorized' | 'etf'
    margin_pct: float      # 26.0 means 26% margin
    leverage: float        # 100 / margin_pct

class MtfUniverse:
    def __init__(self, snapshot_path: Path):
        # Loads JSON, builds dict keyed by tradingsymbol
        ...

    def lookup(self, bare_symbol: str) -> MtfInfo | None: ...

    def is_eligible(self, bare_symbol: str, *, exclude_etf: bool = True) -> bool:
        """In MTF list AND (if exclude_etf) not category=etf."""

    def snapshot_age_days(self) -> int: ...
```

`services/setup_universe.py::close_dn_overnight_long_universe(daily_dict, session_date, config) -> Set[str]`:

Filters:
1. `cap_segment in {large_cap, mid_cap, small_cap, unknown}` (NOT micro)
2. `mis_enabled` (proxy for "tradable")
3. Daily avg volume >= `config[min_daily_avg_volume]`
4. Trading-days coverage >= `config[min_trading_days_coverage_pct]`
5. **Symbol is either (a) MTF-eligible AND not ETF, OR (b) cap_segment in {large, mid, small} for CNC fallback** (skip unknown-cap that isn't MTF — avoids illiquid no-leverage names)
6. If MtfUniverse snapshot age > 7 days, log a `MTF_SNAPSHOT_STALE` warning (don't fail — paper-trade mode tolerates stale list)

**Test plan:**

1. `test_mtf_universe_loads_snapshot` — load JSON, verify 1,489 entries, `MtfInfo` populated
2. `test_mtf_universe_excludes_etf` — `is_eligible("BANKBEES", exclude_etf=True)` returns False; without flag returns True
3. `test_universe_builder_returns_eligible_symbols` — mock `daily_dict` + `nse_all.json` → all 5 conditions applied
4. `test_universe_builder_includes_non_mtf_fallback` — mock symbol absent from MTF list but `cap_segment in {large,mid,small}` → included for CNC fallback
5. `test_universe_builder_excludes_micro_cap` — micro_cap symbol excluded even if MTF-eligible
6. `test_universe_builder_excludes_unknown_cap_not_mtf` — unknown_cap + not in MTF list → excluded (no leverage + illiquid)
7. `test_mtf_snapshot_stale_warning` — set snapshot mtime to 10 days ago → builder logs warning

**Acceptance gate:** all 7 tests pass.

**Estimated effort:** 1-2 commits, ~280 lines.

---

## Task 3: Capital manager with JSON-persisted overnight state

**Files:**
- Modify: `services/capital_manager.py`
- Create: `state/overnight_slots.json` (gitignored runtime artifact; example committed to `state/overnight_slots.example.json`)
- Test: `tests/services/test_capital_manager_overnight.py`

**Scope:**

Existing `CapitalManager` handles MIS intraday budget. Add **persisted overnight slot state** (NOT an in-memory state machine — this is a cron model, state must survive process exits).

```python
@dataclass
class OvernightSlot:
    slot_id: int                       # 1..4 (matches max_concurrent_slots)
    status: str                        # 'free' | 't0_open' | 't1_settling'
    symbol: str | None
    product: str | None                # 'MTF' | 'CNC'
    leverage: float                    # 1.0 for CNC, 2.0-5.0 for MTF
    margin_inr: float
    notional_inr: float
    buy_fill_price: float | None
    buy_fill_ts: str | None            # ISO IST
    buy_order_id: str | None
    amo_sell_order_id: str | None
    expected_exit_date: str | None     # next trading day, ISO date
    sell_fill_price: float | None
    sell_fill_ts: str | None
    realized_pnl_inr: float | None
    fees_inr: float | None
    interest_inr: float | None

class OvernightSlotPool:
    def __init__(self, state_path: Path, max_slots: int, margin_per_slot: float, max_new_per_day: int):
        self._state_path = state_path
        self._max_slots = max_slots
        self._margin_per_slot = margin_per_slot
        self._max_new_per_day = max_new_per_day
        self._slots: list[OvernightSlot] = self._load()

    def reserve(self, symbol: str, product: str, leverage: float, today: date) -> OvernightSlot | None:
        """Reserve a free slot if (a) free slot exists, (b) new-per-day cap not hit."""

    def attach_buy_fill(self, slot_id: int, fill_price: float, fill_ts: datetime, order_id: str) -> None: ...

    def attach_amo_sell(self, slot_id: int, amo_order_id: str, expected_exit_date: date) -> None: ...

    def settle(self, slot_id: int, sell_fill_price: float, sell_fill_ts: datetime, fees: float, interest: float) -> None:
        """T+1 morning: AMO filled. Computes realized_pnl, transitions to t1_settling."""

    def release(self, slot_id: int, cash_back_date: date) -> None:
        """T+2 morning: cash settled, slot transitions to free."""

    def active(self) -> list[OvernightSlot]: ...
    def free_count(self) -> int: ...
    def new_today_count(self, today: date) -> int: ...

    def _load(self) -> list[OvernightSlot]: ...
    def _persist(self) -> None: ...
```

Config additions to `config/configuration.json`:

```json
{
  "setups": {
    "close_dn_overnight_long": {
      "enabled": false,
      "paper_enabled": true,
      "capital_allocation": {
        "active_margin_inr": 400000,
        "cushion_inr": 100000,
        "max_concurrent_slots": 4,
        "margin_per_slot_inr": 100000,
        "max_new_positions_per_day": 2,
        "state_file": "state/overnight_slots.json"
      },
      "mtf": {
        "approved_list_snapshot_path": "data/mtf_universe/approved_mtf_securities_2026-05-21.json",
        "interest_pct_per_day": 0.0004,
        "exclude_etf": true,
        "fallback_to_cnc_if_not_mtf": true,
        "stale_snapshot_warn_days": 7
      }
    }
  }
}
```

`.gitignore` additions:
```
state/overnight_slots.json
logs/overnight_entry.log
logs/overnight_verify.log
```

**Test plan:**

1. `test_pool_loads_empty_state_on_first_run` — no state file → all 4 slots free
2. `test_pool_persists_after_reserve` — reserve slot, instantiate new pool → loaded state matches
3. `test_reserve_returns_none_when_all_slots_occupied` — 4 slots reserved, 5th attempt → None
4. `test_max_new_per_day_cap` — reserve 2 on Mon; 3rd same-day → None even if a slot is free
5. `test_full_lifecycle_t0_t1_t2` — reserve → attach_buy_fill → attach_amo_sell → settle → release; assert state transitions
6. `test_settle_computes_realized_pnl_correctly` — given known buy/sell/fees/interest, verify realized_pnl_inr
7. `test_friday_to_monday_settle_uses_3_day_interest` — hold_days computed correctly across weekend
8. `test_corrupt_state_file_raises_explicit_error` — partial JSON → raise with clear message (don't silently start fresh)

**Acceptance gate:** all 8 tests pass; intraday MIS capital tracking unaffected.

**Estimated effort:** 2 commits, ~450 lines.

---

## Task 4: Detector implementation

**Files:**
- Create: `structures/close_dn_overnight_long_structure.py`
- Modify: `services/dispatch/setup_registry.py` — register with `mode="overnight"`
- Modify: `structures/__init__.py`
- Test: `tests/structures/test_close_dn_overnight_long_structure.py`

**Scope:**

Detector class follows the pattern of `below_vwap_volume_revert_long_structure.py`. Key specifics:

1. **Active window**: ONLY fires at `15:25:00` bar
2. **5-bar signal**: bars `15:00, 15:05, 15:10, 15:15, 15:20` (NOT 15:25 — see SHIPPABLE record look-ahead correction)
3. **Cell filter**: `closing_30m_volume_z >= 2.0` (extreme) AND `prior_day_return_pct >= 3.0` (up_gt_3pct)
4. **TradePlan output**: emits a plan with `metadata.product` set to "MTF" (if eligible) or "CNC" (fallback). Exit is `scheduled_amo` (broker-side). No SL/T1/T2.

The detector emits a TradePlan; the cron entry handler (Task 6) is responsible for actually placing orders.

```python
class CloseDnOvernightLongStructure(BaseStructure):
    def detect(self, df_5m_today, levels, regime, bar_ts, symbol) -> StructureEvent | None:
        if bar_ts.strftime("%H:%M") != "15:25":
            return None
        signal_bars = df_5m_today.between_time("15:00", "15:20")
        if len(signal_bars) < self.min_bar_count:
            return None

        svr = _signed_vol_ratio(signal_bars)
        if svr > self.signed_vol_ratio_max:
            return None
        zvol = _closing_25m_volume_z(signal_bars, prior_20d_baseline)
        if zvol < self.closing_volume_z_min:
            return None
        if zvol < 2.0:  # extreme bucket
            return None

        prior_ret = _prior_day_return_pct(df_5m_today, prior_day_close)
        if prior_ret < 3.0:
            return None

        mtf_info = self.mtf_universe.lookup(_bare(symbol))
        if mtf_info and mtf_info.category == "etf":
            return None  # ETFs explicitly excluded
        product = "MTF" if mtf_info else "CNC"
        leverage = mtf_info.leverage if mtf_info else 1.0

        next_day = next_trading_day(bar_ts.date())
        entry_price = signal_bars["close"].iloc[-1]  # informational only

        return StructureEvent(
            setup_type="close_dn_overnight_long",
            side="LONG",
            entry_price=entry_price,
            exit_levels=ExitLevels(
                exit_mode="scheduled_amo",
                scheduled_exit_at=naive_ist(next_day, "09:15"),
                hard_sl=None, targets=None,
            ),
            metadata={
                "product": product,
                "leverage": leverage,
                "signed_vol_ratio": float(svr),
                "closing_volume_z": float(zvol),
                "prior_day_return_pct": float(prior_ret),
            },
        )
```

Cross-day baseline (`prior_20d_baseline`): reuse `services/cross_day_rvol_enrichment.py` (the parquet built for `below_vwap_volume_revert_long`).

Setup registry entry:
```python
SETUP_REGISTRY["close_dn_overnight_long"] = SetupSpec(
    detector_class=CloseDnOvernightLongStructure,
    universe_builder=close_dn_overnight_long_universe,
    universe_trigger="session_start",
    active_window=("15:25", "15:25"),
    mode="overnight",                   # NEW field — filtered by Task 0
    side="LONG",
)
```

**Test plan:**

1. `test_only_fires_at_1525` — 14:00 → None; 15:25 valid → event
2. `test_rejects_below_signed_vol_threshold` — svr = -0.3 → None
3. `test_rejects_below_volume_z` — zvol = 0.5 → None
4. `test_rejects_non_extreme_volume_bucket` — zvol = 1.5 → None
5. `test_rejects_non_up_rally` — prior_ret = 1.0 → None
6. `test_emits_mtf_product_for_eligible_symbol` — RELIANCE mock in MTF list → product=MTF
7. `test_emits_cnc_product_for_non_mtf_symbol` — not-in-list → product=CNC
8. `test_rejects_etf_even_if_mtf_eligible` — BANKBEES (ETF) → None
9. `test_emits_scheduled_amo_exit` — verify exit_mode and scheduled_exit_at correctly set
10. `test_with_5_bars_matches_sanity` — real CSV row → detector emits matching signal stats

**Acceptance gate:** all 10 tests pass; backtest `python main.py --dry-run --session-date 2024-03-15 --mode overnight --action entry` (a known Cell #5 fire date) produces a clean detector event.

**Estimated effort:** 3 commits, ~500 lines.

---

## Task 5: Order placement — MTF + CNC routing

**Files:**
- Modify: `services/orders/order_placer.py` (or equivalent)
- Modify: `brokers/mock_broker.py` — simulate MTF margin + product=MTF
- Test: `tests/services/orders/test_order_routing.py`

**Scope:**

1. Add `product` and `variety` params:
```python
def place_order(
    symbol: str, qty: int, side: str,
    *, product: str, variety: str = "regular",
    order_type: str = "MARKET",
) -> str:
    """Returns broker_order_id."""
```

2. MTF order placement uses `product="MTF"`, variety="regular" for the BUY (during regular session)
3. AMO SELL uses `product="MTF"` (matching BUY), variety="amo"
4. CNC fallback uses `product="CNC"` for both

5. **Mock broker** simulates:
   - `product=MTF`: requires `notional / leverage` as margin, NOT full notional
   - `product=CNC`: requires full notional
   - `variety=amo`: order is queued, "executes" at the next-session's specified bar in backtest sim

**Test plan:**

1. `test_route_mtf_buy_calls_broker_with_product_mtf` — order placed with product="MTF"
2. `test_route_amo_sell_uses_variety_amo` — exit order has variety="amo"
3. `test_route_cnc_fallback` — TradePlan with product=CNC → placed with product="CNC"
4. `test_existing_mis_orders_unaffected` — regression: intraday MIS still routes product="MIS"
5. `test_mock_broker_mtf_margin_simulation` — margin reserved = notional / leverage
6. `test_mock_broker_amo_fills_at_next_session_open_in_backtest` — sim advances to next-day 09:15 → AMO fills at that bar's open

**Acceptance gate:** all 6 tests pass; mock broker correctly settles a Mon-BUY/Tue-SELL MTF round-trip with realistic margin + fee accounting.

**Estimated effort:** 2 commits, ~350 lines.

---

## Task 6: Cron handlers — entry + verify-exit

**Files:**
- Create: `services/execution/overnight_handlers.py` (replaces the daemon-based `overnight_exit_handler.py` from the previous spec version)
- Modify: `main.py` — wire `--action entry` and `--action verify-exit` to the new functions
- Test: `tests/services/execution/test_overnight_handlers.py`

**Scope:**

Two short-lived functions, called by `main.py` based on `--action`:

```python
def run_entry(now_ist: datetime, config: dict, broker, slot_pool, registry, mtf_universe) -> None:
    """Called at 15:25 IST every weekday.

    1. Compute signal via detector on today's df_5m up to 15:20 bar
       (cron runs at 15:25 — at that instant, the 15:25 bar is just starting,
       so we have access to bars 15:00-15:20 only, which is exactly what the
       5-bar signal needs — no look-ahead).
    2. For each symbol that fires:
       a. Reserve slot via slot_pool. If no slot or per-day cap hit, log skip.
       b. Place MOC BUY (product=MTF or CNC per detector metadata).
       c. Poll order status until filled or timeout (60s default).
       d. Attach buy fill to slot.
       e. Place AMO SELL for next-day pre-open. Attach amo_order_id to slot.
    3. Persist slot_pool state.
    4. Log summary; exit.
    """

def run_verify_exit(now_ist: datetime, config: dict, broker, slot_pool, fee_calc) -> None:
    """Called at 09:30 IST every weekday.

    1. For each slot with status 't0_open' (yesterday's BUY, AMO queued):
       a. Query broker for AMO order status.
       b. If filled: compute pnl, fees, interest. Call slot_pool.settle().
       c. If NOT filled (rare — AMO rejection or auction non-clear):
          - Place regular market SELL as fallback.
          - Wait for fill, then settle.
       d. If broker auto-squared overnight (margin call detection):
          - Use the auto-square fill price.
          - Log adversely (SETUP_AUTO_SQUARED event).
    2. For each slot with status 't1_settling' (yesterday's SELL waiting for cash):
       a. Verify cash is back (T+2 settlement).
       b. Call slot_pool.release().
    3. Persist slot_pool state.
    4. Log summary; exit.
    """
```

Backtest mode: the simulator drives time-stepping. `run_entry` and `run_verify_exit` are called at the simulated 15:25 and 09:30 timestamps respectively. The mock broker handles the AMO "fill" at the next-day 09:15 simulated bar's open price.

**Test plan:**

1. `test_run_entry_fires_detector_at_1525` — at 15:25, detector consulted; valid signals → slots reserved + orders placed
2. `test_run_entry_respects_per_day_cap` — 3 signals, max_new=2 → 2 placed, 1 logged-skipped
3. `test_run_entry_handles_no_signal_day` — no fires → no orders placed, state file unchanged
4. `test_run_entry_amo_submission_failure` — AMO rejected by broker → log alert, leave slot in t0_open (will be flagged at next verify-exit)
5. `test_run_verify_exit_settles_filled_amo` — AMO filled overnight → slot transitions to t1_settling, pnl/fees computed
6. `test_run_verify_exit_fallback_for_unfilled_amo` — AMO not filled → regular market SELL placed → settle on fill
7. `test_run_verify_exit_releases_t2_slots` — yesterday's t1_settling → released to free
8. `test_run_verify_exit_idempotent_on_repeat_call` — calling twice in a row produces same end state (recovers from missed runs)
9. `test_friday_entry_monday_verify` — Fri entry → Mon verify_exit settles (3 calendar days, but 1 trading day)
10. `test_backtest_simulator_end_to_end` — `python main.py --dry-run --session-date 2025-09-11 --mode overnight` runs entry on 09-11, verify-exit on 09-12, produces canonical trade record

**Acceptance gate:** all 10 tests pass; backtest E2E produces trades matching 5-bar sanity output for known fire dates.

**Estimated effort:** 3 commits, ~350 LOC (significantly less than the daemon model's ~400+ LOC).

---

## Task 7: Decay tripwire monitor

**Files:**
- Create: `services/risk/decay_tripwire.py`
- Modify: `services/dispatch/setup_registry.py` — consult tripwire at fire-time
- Test: `tests/services/risk/test_decay_tripwire.py`

**Scope:**

Per SHIPPABLE record decay_warning:
> "Pause if rolling 30-trade PF drops below 1.20 sustained for 6 weeks"

```python
class DecayTripwire:
    def __init__(
        self,
        setup_name: str,
        state_path: Path,
        window_trades: int = 30,
        pf_floor: float = 1.20,
        sustained_weeks: int = 6,
    ): ...

    def record_trade(self, net_pnl_inr: float, trade_ts: datetime) -> None:
        """Called by Task 6's run_verify_exit after settle()."""

    def is_paused(self) -> bool: ...
    def state_summary(self) -> dict: ...
    def reset(self) -> None:  # manual unpause
```

State persisted at `state/decay_tripwire_close_dn_overnight_long.json`.

Detector (or registry) checks `is_paused()` before emitting events. If paused, log `SETUP_PAUSED_DECAY` and skip.

**Test plan:**

1. `test_records_trades` — record 30 trades, query state
2. `test_pauses_when_pf_below_floor_for_6_weeks` — simulate 6 weeks of PF=1.0 → is_paused True
3. `test_unpauses_after_recovery` — record trades with PF > 1.20 for 2+ weeks → unpauses
4. `test_run_entry_skips_paused_setup` — paused → no events emitted
5. `test_state_persists_across_runs` — write state, instantiate new object → state loaded

**Acceptance gate:** all 5 tests pass.

**Estimated effort:** 2 commits, ~250 lines.

---

## Cross-cutting acceptance for paper-trade activation

After all 8 tasks merge:

1. **Pytest**: all suites pass; no regression on intraday MIS or below_vwap.
2. **Backtest 2024-03-15 (known Cell #5 fire day)**: `python main.py --dry-run --session-date 2024-03-15 --mode overnight --action entry` produces trade events. Next-day equivalent runs verify-exit and settles.
3. **Backtest 2025-09-11 (multi-fire HO day)**: capital_manager + per-day cap correctly clip to 2 new positions; remaining signal events logged as skipped.
4. **Paper-trade smoke (5 sessions)**:
   - Cron entry runs at simulated 15:25, places orders, persists state
   - Cron verify-exit runs at simulated 09:30, settles + releases
   - State file (`state/overnight_slots.json`) inspectable between runs
   - Mock broker simulates MTF margin + interest accurately
5. **MTF snapshot freshness**: stale (>7 day old) snapshot logs warning, doesn't block.
6. **Decay tripwire**: simulate 50 trades with PF=0.5 → tripwire pauses entry; reset → resume.
7. **Crash recovery**: kill the entry cron mid-run (after BUY filled, before AMO placed). Re-run entry — should detect orphaned BUY and either place the missing AMO or alert. Document the recovery path.

## Estimated total effort

- 8 tasks × ~2 commits avg = ~16 commits
- ~2,300 LOC including tests (down from 2,400 in the daemon model)
- 1-2 weeks of focused engineering work
- Significantly simpler operationally — 2 crontab lines vs systemd service + monitor

## Decisions deferred past paper-trade

1. **GTT-SL gap protection** — Zerodha supports GTT on MTF. Add post-paper if mechanism validates. SL at entry × 0.95.
2. **MTF list cron refresh** — `tools/scrape_zerodha_mtf.py` is manual today. Add weekly cron.
3. **Multi-setup overnight coexistence** — if a 2nd overnight setup is added, slot pool needs sharing logic. Today: only this setup.
4. **Pre-open slippage modeling** — sanity uses 09:15 official open. Real fill may differ for thin small-caps. Add slippage model if 4-week paper tracking shows >10 bps median deviation.
5. **Cron coordination with intraday daemon** — entry cron fires at 15:25 IST while intraday daemon is still running (15:30 EOD square-off). They share `state/overnight_slots.json` but write distinct keys. Verify lock-file or similar coordination if cross-mode resource contention surfaces in paper.

## File-of-record after merge

```
main.py                                                     ← +mode/action CLI
scripts/cron-entry.sh                                       ← NEW (deployment wrapper)
scripts/cron-verify-exit.sh                                 ← NEW (deployment wrapper)
config/configuration.json                                   ← setup config block
services/setup_universe.py                                  ← close_dn_overnight_long_universe
services/mtf_universe.py                                    ← NEW: MTF list loader
services/dispatch/setup_registry.py                         ← register + mode filter
services/capital_manager.py                                 ← OvernightSlotPool persistence
services/orders/order_placer.py                             ← product=MTF + variety=amo routing
services/execution/overnight_handlers.py                    ← NEW: run_entry + run_verify_exit
services/risk/decay_tripwire.py                             ← NEW: rolling-PF guard
structures/close_dn_overnight_long_structure.py             ← NEW: detector
brokers/mock_broker.py                                      ← +MTF + AMO simulation
tools/sub7_validation/build_per_setup_pnl.py                ← +calc_fee_cnc + calc_fee_mtf
state/overnight_slots.example.json                          ← NEW: state file template
state/overnight_slots.json                                  ← gitignored runtime artifact
data/mtf_universe/approved_mtf_securities_YYYY-MM-DD.json   ← already exists
tools/scrape_zerodha_mtf.py                                 ← already exists
```
