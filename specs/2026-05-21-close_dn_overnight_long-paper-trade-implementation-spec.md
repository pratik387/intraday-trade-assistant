# `close_dn_overnight_long` — Paper-trade Implementation Spec

**Date:** 2026-05-21
**Branch:** `research/europe-open-13ist`
**SHIPPABLE record:** `specs/2026-05-21-close_dn_overnight_long-SHIPPABLE-cell-5.md`
**Cell-lock JSON (authoritative config):** `tools/sub9_research/close_dn_overnight_long_cell_lock.json`

## Goal

Wire `close_dn_overnight_long` Cell #5 into the existing dispatch/execution pipeline as the system's **first MTF-overnight setup**, suitable for paper-trading validation.

## Scope

Implementation breaks into **7 ordered tasks** below. Each is sized at ~1-3 commits. Dependencies form a linear chain except where noted.

```
Task 1 (fee helpers) ── Task 2 (universe) ── Task 4 (detector) ── Task 5 (orders) ── Task 6 (exit handler) ── Task 7 (tripwire)
                                  │                                       │
                                  └────── Task 3 (capital manager) ───────┘
```

Tasks 1, 2, 3 are mostly independent; 4 needs 1+2; 5 needs 3+4; 6 needs 5; 7 is bolt-on after 6.

## Project rules to honor (from CLAUDE.md)

1. **NO hardcoded defaults** — all thresholds/parameters from `config/configuration.json`. Cell-lock values become config entries.
2. **IST-naive timestamps** — use `utils/time_util.py` helpers. Signal-bar timestamp = `15:25:00` naive IST.
3. **Live/backtest compatibility** — every code path must work in `--dry-run` (backtest) and `python main.py` (live) modes. Use tick timestamps, never `datetime.now()` for trading decisions.
4. **Indian-market focus** — MTF specifics (Zerodha rate card, NSE pre-open auction, T+1 settlement) baked in.

---

## Task 1: Fee helpers (CNC + MTF) promoted to `build_per_setup_pnl.py`

**Files:**
- Modify: `tools/sub7_validation/build_per_setup_pnl.py`
- Reference impl: `tools/sub9_research/sanity_close_dn_overnight_long.py:calc_fee_cnc`
- Test: `tests/sub7_validation/test_fee_modes.py`

**Scope:**

Existing `calc_fee()` in `build_per_setup_pnl.py` is intraday-MIS-only. Add:

```python
def calc_fee_cnc(buy_value_inr: float, sell_value_inr: float) -> float:
    """Round-trip CNC delivery fees in INR. See cell_lock.json fee_model.cnc."""

def calc_fee_mtf(
    buy_value_inr: float,
    sell_value_inr: float,
    margin_inr: float,
    hold_days: int,
) -> float:
    """Round-trip MTF fees + overnight interest. See cell_lock.json fee_model.mtf.

    interest = (buy_value - margin) * 0.0004 * hold_days
    Pledge fee Rs 15 + GST per ISIN per pledge; unpledge Rs 15 + GST per request.
    For close_dn_overnight setup with 1-night hold, hold_days = 1.
    """

# Existing calc_fee gets a `mode` parameter:
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

1. `test_calc_fee_cnc_baseline` — for Rs 1L buy, Rs 1L*1.005 sell, fee = ~Rs 171 (matches sanity calc)
2. `test_calc_fee_mtf_one_night` — Rs 2.79L buy, Rs 2.80L sell, Rs 1L margin, 1 hold day → fee includes ~Rs 72 interest
3. `test_calc_fee_dispatches_by_mode` — `mode="intraday_mis"` matches old behavior; `mode="delivery_cnc"` matches `calc_fee_cnc`; `mode="mtf"` matches `calc_fee_mtf`

**Acceptance gate:** all 3 tests pass + existing MIS unit tests still pass (no regression on intraday_mis mode).

**Estimated effort:** 1 commit, ~120 lines including tests.

---

## Task 2: Universe builder + MTF-list loader

**Files:**
- Create: `services/setup_universe.py::close_dn_overnight_long_universe`
- Create: `services/mtf_universe.py` (small new module — loads MTF JSON, provides lookup interface)
- Test: `tests/services/test_close_dn_overnight_universe.py`

**Scope:**

`services/mtf_universe.py`:

```python
"""Loads Zerodha MTF approved-securities snapshot and provides lookup."""
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class MtfInfo:
    tradingsymbol: str
    isin: str
    category: str          # 'fo' | 'non_fo' | 'non_categorized' | 'etf'
    margin_pct: float      # broker's haircut as % of notional (26.0 = 26%)
    leverage: float        # 100 / margin_pct (1.92x to 5.0x)

class MtfUniverse:
    def __init__(self, snapshot_path: Path):
        # Loads JSON, builds dict keyed by tradingsymbol (NSE bare name, no prefix)
        ...

    def lookup(self, bare_symbol: str) -> MtfInfo | None:
        """Return MtfInfo if symbol is MTF-eligible, None otherwise."""

    def is_eligible(self, bare_symbol: str, *, exclude_etf: bool = True) -> bool:
        """Eligible for our setup: in MTF list AND (if exclude_etf) not category=etf."""
```

`services/setup_universe.py::close_dn_overnight_long_universe(daily_dict, session_date, config) -> Set[str]`:

Filters:
1. `cap_segment in {large_cap, mid_cap, small_cap, unknown}` (NOT micro per Cell #5 brief)
2. `mis_enabled` flag (proxy for "tradable on NSE")
3. Daily avg volume >= 50,000
4. Trading-days coverage >= 80% (last 30 sessions)
5. **`mtf_universe.is_eligible(bare_symbol, exclude_etf=True) OR cap_segment != "unknown"`** — allows non-MTF for fallback CNC, but blocks unknown-cap that isn't MTF (avoids tiny illiquid names with no leverage)
6. Optional: skip if MtfUniverse snapshot is stale (mtime > 7 days) → log warning

Symbol metadata cached per-session. Reuses `services/symbol_metadata.get_cap_segment` and `get_mis_info`.

**Test plan:**

1. `test_mtf_universe_loads_snapshot` — load JSON, verify 1,489 entries, `MtfInfo` dataclass populated
2. `test_mtf_universe_excludes_etf` — `is_eligible("BANKBEES", exclude_etf=True)` returns False; without flag returns True
3. `test_universe_builder_returns_eligible_symbols` — feed mock `daily_dict` + `nse_all.json` → filter applies all 5 conditions
4. `test_universe_builder_includes_non_mtf_fallback` — mock symbol absent from MTF list but with `cap_segment != "unknown"` → still included (CNC fallback)
5. `test_universe_builder_excludes_micro_cap` — micro_cap symbol excluded even if MTF-eligible

**Acceptance gate:** all 5 tests pass.

**Estimated effort:** 1-2 commits, ~250 lines.

---

## Task 3: Capital manager extension for MTF + 2-day cycle

**Files:**
- Modify: `services/capital_manager.py`
- Test: `tests/services/test_capital_manager_mtf.py`

**Scope:**

Existing capital manager tracks MIS margin per-trade with same-day release (auto-square at 15:10). Add:

```python
class CapitalManager:
    # Existing MIS budget tracking unchanged.

    # New: per-trade overnight slot state machine
    # Each slot has: status in {free, t0_open, t1_settling, t2_released}
    # T+0 (buy day): slot enters t0_open
    # T+1 (sell day next session morning): slot enters t1_settling
    # T+2 (cash back morning): slot enters t2_released → free for new trade

    def reserve_overnight_slot(
        self,
        symbol: str,
        margin_inr: float,
        product: str,                    # 'MTF' | 'CNC'
        leverage: float,                 # 1.0 for CNC, 2.0-5.0 for MTF (per-stock)
        signal_date: date,
    ) -> int | None:
        """Reserve a slot if free; return slot_id or None if no slot available."""

    def release_overnight_slot(self, slot_id: int, cash_back_date: date) -> None:
        """Called on T+2 morning when cash settles."""

    def active_overnight_slots(self) -> list[OvernightSlot]:
        """For dashboard + cushion validation."""

    def overnight_capital_committed_inr(self) -> float:
        """Sum of margins of all not-yet-released slots."""
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
        "max_new_positions_per_day": 2
      },
      "mtf": {
        "approved_list_snapshot_path": "data/mtf_universe/approved_mtf_securities_2026-05-21.json",
        "interest_pct_per_day": 0.0004,
        "exclude_etf": true,
        "fallback_to_cnc_if_not_mtf": true
      }
    }
  }
}
```

**Test plan:**

1. `test_reserve_slot_with_free_capacity` — reserve 1 slot, returns slot_id; `active_overnight_slots()` has 1 entry
2. `test_reserve_slot_exhausted` — reserve 4 slots, 5th returns None
3. `test_release_slot_frees_capacity` — release slot 1, reserve again succeeds
4. `test_max_new_per_day_cap` — reserve 2 on Mon, 3rd same-Mon returns None (per-day cap)
5. `test_t2_cycle_simulation` — simulate Mon-Tue-Wed-Thu cycle; verify slot reuse on T+2

**Acceptance gate:** all 5 tests pass; intraday-MIS capital tracking unaffected (regression check).

**Estimated effort:** 2 commits, ~400 lines.

---

## Task 4: Detector implementation (`close_dn_overnight_long_structure.py`)

**Files:**
- Create: `structures/close_dn_overnight_long_structure.py`
- Modify: `services/dispatch/setup_registry.py` — add `mode` field; register `close_dn_overnight_long`
- Modify: `structures/__init__.py` — export new structure
- Test: `tests/structures/test_close_dn_overnight_long_structure.py`

**Scope:**

Detector class follows existing pattern (e.g., `below_vwap_volume_revert_long_structure.py`). Differences:

1. **Active window**: ONLY fires at `15:25:00` bar (single-bar window, not a range). Reject all other bars.
2. **5-bar signal**: uses bars `15:00, 15:05, 15:10, 15:15, 15:20` (NOT the 15:25 bar — see SHIPPABLE record look-ahead correction).
3. **Cell filter**: `closing_30m_volume_z_bin = "extreme"` AND `prior_day_return_bin = "up_gt_3pct"` checked at signal time.
4. **TradePlan output**: emits a new `mode="cnc_overnight"` or `mode="mtf_overnight"` plan (per-symbol via MTF lookup). No SL/T1/T2; `scheduled_exit_at = next_trading_day_09:15`.

Pseudo:

```python
class CloseDnOvernightLongStructure(BaseStructure):
    def detect(self, df_5m_today, levels, regime, bar_ts, symbol) -> StructureEvent | None:
        # Reject all bars except 15:25
        if bar_ts.strftime("%H:%M") != "15:25":
            return None

        # Build 5-bar signal window (15:00 to 15:20)
        signal_bars = df_5m_today.between_time("15:00", "15:20")
        if len(signal_bars) < self.min_bar_count:
            return None

        # Primary filter: signed_vol_ratio + closing_25m_volume_z + bar_count
        svr = _signed_vol_ratio(signal_bars)
        if svr > self.signed_vol_ratio_max:
            return None
        zvol = _closing_25m_volume_z(signal_bars, prior_20d_baseline)
        if zvol < self.closing_volume_z_min:
            return None

        # Cell filter: extreme volume_z + up_gt_3pct prior_ret
        if not (zvol >= 2.0):  # extreme bucket
            return None
        prior_ret = _prior_day_return_pct(df_5m_today, prior_day_close)
        if prior_ret < 3.0:
            return None

        # Build TradePlan
        mtf_info = self.mtf_universe.lookup(_bare(symbol))
        product = "MTF" if mtf_info and mtf_info.category != "etf" else "CNC"
        leverage = mtf_info.leverage if product == "MTF" else 1.0
        next_day = next_trading_day(bar_ts.date())

        return StructureEvent(
            setup_type="close_dn_overnight_long",
            side="LONG",
            entry_price=signal_bars["close"].iloc[-1],  # informational; actual fill at MOC
            exit_levels=ExitLevels(
                exit_mode="scheduled_amo",
                scheduled_exit_at=naive_ist(next_day, "09:15"),
                hard_sl=None, targets=None,
            ),
            metadata={
                "product": product,
                "leverage": leverage,
                "signed_vol_ratio": svr,
                "closing_volume_z": zvol,
                "prior_day_return_pct": prior_ret,
            },
        )
```

Cross-day baseline (`prior_20d_baseline`): read from the same `cross_day_rvol` parquet used by `below_vwap_volume_revert_long`. Either reuse `services/cross_day_rvol_enrichment.py` or compute on-the-fly from `df_5m_today`'s prior-20-session totals.

Setup registry entry (`services/dispatch/setup_registry.py`):

```python
SETUP_REGISTRY["close_dn_overnight_long"] = SetupSpec(
    detector_class=CloseDnOvernightLongStructure,
    universe_builder=close_dn_overnight_long_universe,
    universe_trigger="session_start",
    active_window=("15:25", "15:25"),  # single-bar window
    mode="cnc_overnight",               # NEW field — drives executor branching
    side="LONG",
)
```

**Test plan:**

1. `test_detector_only_fires_at_1525` — 14:00 bar → None; 15:25 bar with valid signal → event
2. `test_detector_rejects_below_signed_vol_threshold` — svr = -0.3 → None
3. `test_detector_rejects_below_volume_z` — zvol = 0.5 → None
4. `test_detector_rejects_non_extreme_volume_bucket` — zvol = 1.5 (high but not extreme) → None
5. `test_detector_rejects_non_up_rally` — prior_day_return_pct = 1.0 → None
6. `test_detector_emits_mtf_product_for_eligible_symbol` — mock RELIANCE in MTF list → product=MTF
7. `test_detector_emits_cnc_product_for_non_mtf_symbol` — mock not-in-list symbol → product=CNC
8. `test_detector_rejects_etf_even_if_mtf_eligible` — mock BANKBEES (ETF in MTF list) → None
9. `test_detector_emits_scheduled_amo_exit` — verify exit_mode and scheduled_exit_at correctly set
10. `test_detector_with_5_bars_matches_sanity` — use real CSV row's signal values, verify detector outputs identical signal stats

**Acceptance gate:** all 10 tests pass; live/backtest both run cleanly via `python main.py --dry-run --session-date 2024-03-15` (a day with known Cell #5 fires).

**Estimated effort:** 3 commits, ~500 lines.

---

## Task 5: Order placement — MTF + CNC routing

**Files:**
- Modify: `services/orders/order_placer.py` (or equivalent module)
- Modify: `services/dispatch/planner.py` — read `mode` from TradePlan and route
- Test: `tests/services/orders/test_order_routing.py`

**Scope:**

1. Add `product` param routing to broker order placement:
   ```python
   def place_buy_order(symbol, qty, *, product: str, variety: str = "regular"):
       # product in {"MIS", "CNC", "MTF", "NRML"}
       # variety in {"regular", "amo", "co", "iceberg"}
       broker_client.place(symbol=symbol, quantity=qty, transaction_type="BUY",
                            product=product, variety=variety, order_type="MARKET")
   ```

2. When detector emits TradePlan with `metadata.product = "MTF"`, planner routes to MTF order; else CNC.

3. **Mock broker for paper-trade**: existing MockBroker needs to handle `product=MTF`. Simulate MTF margin = notional / leverage. Track per-position to apply interest in PnL computation (Task 1 fee helper).

4. **Live broker**: Kite API call with `product="MTF"`. Already supported per Kite Connect docs.

**Test plan:**

1. `test_route_mtf_order_calls_broker_with_product_mtf` — TradePlan has metadata.product=MTF → broker called with product="MTF"
2. `test_route_cnc_order_calls_broker_with_product_cnc` — TradePlan has metadata.product=CNC → broker called with product="CNC"
3. `test_existing_mis_orders_unaffected` — regression: existing intraday MIS path still routes product=MIS
4. `test_mock_broker_simulates_mtf_margin` — for MTF order, margin reserved = notional / leverage; cash reserved = margin only
5. `test_paper_order_amo_variety` — exit order placed with variety="amo"

**Acceptance gate:** all 5 tests pass; paper-broker can correctly settle a Mon-BUY/Tue-SELL MTF round-trip.

**Estimated effort:** 2 commits, ~300 lines.

---

## Task 6: Exit handler — pre-open AMO + morning verification

**Files:**
- Create: `services/execution/overnight_exit_handler.py` (new module)
- Modify: `services/execution/exit_executor.py` — branch by exit_mode
- Test: `tests/services/execution/test_overnight_exit_handler.py`

**Scope:**

Existing `exit_executor.py` is tick-driven (SL/T1/T2 on intraday ticks). For overnight setups, we need a different flow:

1. **AMO submission**: When BUY fills (~15:30 IST), submit AMO SELL for the same quantity. Variety = "amo", product = same as BUY (MTF or CNC). The broker queues the AMO until next session.

2. **Morning verification**: At 09:30 IST next trading day (15 min after pre-open clears), check:
   - Was the AMO SELL executed? If yes, log fill, release slot via capital_manager.
   - If not, place a regular market SELL at current LTP. This is the failsafe for AMO non-execution (rare but happens with thin pre-open auction or broker-side rejection).
   - Track the actual fill price for PnL accuracy.

3. **Cash settlement event**: At 09:30 IST on T+2, the broker has settled the T+1 SELL. Notify capital_manager to mark slot as released.

Backtest mode: simulator advances bar-by-bar; the AMO "fills" at next-day 09:15 open price (matches sanity assumption). T+2 cash settlement modeled by advancing capital manager state on the simulated date.

**Test plan:**

1. `test_amo_placed_after_buy_fill` — BUY fills → AMO order created with correct symbol/qty/product
2. `test_morning_verification_logs_fill_if_amo_executed` — AMO already executed → no new order, slot released
3. `test_morning_verification_places_fallback_if_amo_not_executed` — AMO unexecuted → regular market SELL placed
4. `test_simulator_executes_amo_at_next_day_open_price` — backtest sim: AMO fills at exact 09:15 open price
5. `test_t2_settlement_releases_slot` — capital_manager.release_overnight_slot called on T+2 morning
6. `test_friday_amo_holds_to_monday` — Fri BUY → AMO valid through weekend, executes Mon 09:15 (Mon is T+2 for Fri fill)

**Acceptance gate:** all 6 tests pass; end-to-end paper-trade backtest of 2026-03-27 (a Cell #5 fire day per HO data) produces a clean trade record.

**Estimated effort:** 3 commits, ~400 lines.

---

## Task 7: Decay tripwire monitor

**Files:**
- Create: `services/risk/decay_tripwire.py`
- Modify: `services/dispatch/setup_registry.py` — check tripwire status at each fire
- Test: `tests/services/risk/test_decay_tripwire.py`

**Scope:**

Per the SHIPPABLE record:
> "Pause if rolling 30-trade PF (live or paper) drops below 1.20 sustained for 6 weeks"

```python
class DecayTripwire:
    def __init__(
        self,
        setup_name: str,
        window_trades: int = 30,
        pf_floor: float = 1.20,
        sustained_weeks: int = 6,
    ): ...

    def record_trade(self, net_pnl_inr: float, trade_ts: datetime) -> None: ...

    def is_paused(self) -> bool:
        """True if rolling 30-trade PF has been < pf_floor for >= sustained_weeks."""

    def state_summary(self) -> dict:
        """For dashboard: current rolling PF, weeks-below-floor counter, pause status."""
```

Persisted state (so it survives restarts): JSON file at `state/decay_tripwire_{setup_name}.json`.

Setup registry checks `is_paused()` before emitting events. If paused, log `SETUP_PAUSED_DECAY` and skip.

Manual unpause: delete the state file or call `decay_tripwire reset` CLI.

**Test plan:**

1. `test_tripwire_records_trades` — record 30 trades, query state
2. `test_tripwire_pauses_when_pf_below_floor_for_6_weeks` — simulate 6 weeks of PF=1.0 trades → is_paused returns True
3. `test_tripwire_unpauses_after_recovery` — after pause, record trades with PF > 1.20 for 2+ weeks → unpauses
4. `test_setup_registry_skips_paused_setup` — paused setup emits 0 events on signal day
5. `test_tripwire_state_persists_across_restart` — write state, instantiate new object, verify state loaded

**Acceptance gate:** all 5 tests pass; tripwire integrates cleanly with existing dispatch flow.

**Estimated effort:** 2 commits, ~250 lines.

---

## Cross-cutting acceptance for paper-trade activation

After all 7 tasks merge:

1. **Full pytest run**: `tests/services/`, `tests/structures/`, `tests/dispatch/`, `tests/sub7_validation/` — all pass.
2. **Backtest 2024-03-15** (Cell #5 fire day per HO data): `python main.py --dry-run --session-date 2024-03-15` produces trades for `close_dn_overnight_long` with MTF product on eligible symbols.
3. **Backtest 2025-09-11** (multi-fire HO day): verify multiple symbols dispatched, capital manager handles slot saturation, AMO exits placed.
4. **Paper-trade mode validation**:
   - Run `python main.py --paper-trading` for 5 consecutive sessions
   - Verify: detector fires only at 15:25; orders placed with product=MTF/CNC per routing; AMO SELL submitted; morning verification passes; capital cycle (T+0 → T+1 → T+2) tracked
5. **Decay tripwire bench**: simulate 50 trades with PF=0.5 → tripwire pauses; reset → resume.
6. **MTF list freshness check**: tools/scrape_zerodha_mtf.py runs cleanly, produces dated snapshot.

## Estimated total effort

- 7 tasks × avg 2 commits each = ~14 commits
- ~2,000 lines of new code + ~400 lines of tests
- 1-2 weeks of focused engineering work

## Decisions deferred to paper-trade phase

These don't block paper-trade kickoff but must be resolved before live capital:

1. **GTT-SL gap protection** — Zerodha supports GTT on MTF. Add as Task 8 if paper-trade validates the mechanism. Threshold: SL at entry * 0.95 (5% stop).
2. **Max-concurrent overnight cap** — currently 4 slots from Rs 5L. For larger capital deployment, capital_manager already supports configurable max via `max_concurrent_slots`.
3. **Multi-setup overnight coexistence** — if a second overnight setup is added later, capital_manager needs shared slot pool. Today: only `close_dn_overnight_long` uses overnight slots.
4. **Pre-open slippage modeling for paper-trade** — current model uses 09:15 official open. Real fill may differ if pre-open auction depth is thin (especially small_cap). Add a slippage model in paper-trade if 4-week tracking shows >10 bps median deviation.
5. **MTF approved-list cadence**: scraper is manual today. Move to a cron-based refresh once paper-trade graduates.

## File-of-record after merge

```
config/configuration.json                                  ← setup config block
services/setup_universe.py                                  ← close_dn_overnight_long_universe
services/mtf_universe.py                                    ← NEW: MTF list loader
services/dispatch/setup_registry.py                         ← register new setup + mode field
services/capital_manager.py                                 ← overnight slot state machine
services/orders/order_placer.py                             ← product=MTF routing
services/execution/exit_executor.py                         ← branch by exit_mode
services/execution/overnight_exit_handler.py                ← NEW: AMO + morning verification
services/risk/decay_tripwire.py                             ← NEW: rolling-PF guard
structures/close_dn_overnight_long_structure.py             ← NEW: detector
tools/sub7_validation/build_per_setup_pnl.py                ← calc_fee_cnc + calc_fee_mtf
data/mtf_universe/approved_mtf_securities_YYYY-MM-DD.json   ← already exists
tools/scrape_zerodha_mtf.py                                 ← already exists
```
