# Multi-Day Composite Selection & Ranking — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a cross-setup integrated-composite selection/ranking layer for the four `horizon=="multi_day"` CNC/MTF capitulation setups, so overlapping picks are deduped to one book position, ranked by a consensus-aware composite score, and each setup's standalone edge stays measurable.

**Architecture:** The per-setup `CrossSectionalRanker` emits one new normalized field (`cap_score`). A new pure module `MultiDayCompositeSelector` pools all setups' baskets, blends `cap_score` across setups (summing → consensus boost), dedupes to one row/symbol, drops already-held names, and returns a capped, composite-ranked basket. `mtf_capitulation_handlers._run_entries` is restructured from place-in-loop into collect→compose→place-once; the position lives in the owner setup's store but its realized PnL feeds every contributing setup's decay tripwire.

**Tech Stack:** Python 3, pandas, numpy, pytest. IST-naive timestamps. Config-driven (no hardcoded defaults).

**Spec:** `specs/2026-06-29-multiday-composite-selection-design.md`

---

## File Structure

- **Modify** `services/cross_sectional_ranker.py` — add `cap_score` to `rank()` output (Task 1).
- **Create** `services/multiday_composite_selector.py` — pure selection/dedupe/rank module (Task 2).
- **Modify** `config/configuration.json` — per-setup `composite_weight` + `cap_score_clip`; new `multi_day_portfolio` family block (Task 3).
- **Modify** `services/execution/mtf_capitulation_handlers.py` — collective composite entries + cross-day held-union filter (Task 4); multi-contributor exit attribution (Task 5); per-day selection diagnostics (Task 6).
- **Create** `tests/services/test_multiday_composite_selector.py` (Task 2).
- **Modify** `tests/services/test_cross_sectional_ranker.py` (Task 1).
- **Modify** `tests/services/execution/test_mtf_capitulation_handlers.py` (Tasks 4–6).
- **Create** `tests/config/test_multiday_config_keys.py` (Task 3).

---

## Task 1: Ranker emits `cap_score` (normalized capitulation magnitude)

**Files:**
- Modify: `services/cross_sectional_ranker.py` (inside `rank()`, after `rank_pct` is computed ~line 149; and the output dict ~line 170-180)
- Test: `tests/services/test_cross_sectional_ranker.py`

`cap_score` = cross-sectional z-score of the oriented signal (`magnitude = -signal`, so "more capitulated = larger" for all three modes), computed over the **full qualifying cross-section** (`today` after the universe/price/adv filters), **lower-clipped at 0** (tail only). The upper clip is applied later in the selector (family-level `cap_score_clip`), so this stays a pure per-setup normalization.

- [ ] **Step 1: Write the failing test**

Add to `tests/services/test_cross_sectional_ranker.py`:

```python
def test_rank_emits_cap_score_oriented_and_nonnegative():
    sd = date(2026, 3, 16)
    ranker = CrossSectionalRanker(_cfg(loser_pct=0.25))
    out = ranker.rank(_panel(sd), sd, mtf_eligible={f"LOSER{i}" for i in range(6)} | {f"FILL{i:02d}" for i in range(25)})
    assert out, "expected a non-empty basket"
    # every selected name carries a finite, non-negative cap_score
    for r in out:
        assert "cap_score" in r
        assert r["cap_score"] >= 0.0
        assert r["cap_score"] == r["cap_score"]  # not NaN
    # the deepest loser (most negative trailing return) has the largest cap_score
    deepest = min(out, key=lambda r: r["trail_ret"])
    assert deepest["cap_score"] == max(r["cap_score"] for r in out)
```

(The `_panel` helper already seeds a "deepest" loser; see the file's existing `loser(...)` builder.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/services/test_cross_sectional_ranker.py::test_rank_emits_cap_score_oriented_and_nonnegative -v`
Expected: FAIL with `KeyError: 'cap_score'`.

- [ ] **Step 3: Implement `cap_score` in `rank()`**

In `services/cross_sectional_ranker.py`, immediately AFTER the line:

```python
        today["rank_pct"] = today["signal"].rank(pct=True)
```

insert:

```python
        # Capitulation magnitude, oriented so MORE capitulated = larger, then
        # standardized cross-sectionally over the qualifying universe and
        # lower-clipped at 0 (tail only). The composite selector blends this
        # across setups; the UPPER clip (family-level cap_score_clip) is applied
        # there, so this stays a pure per-setup normalization (CLAUDE.md rule 1:
        # the clip value is not read here).
        mag = -today["signal"]
        mu = float(mag.mean())
        sd_mag = float(mag.std())
        if np.isfinite(sd_mag) and sd_mag > 0.0:
            today["cap_score"] = ((mag - mu) / sd_mag).clip(lower=0.0)
        else:
            today["cap_score"] = 0.0
```

Then in the output dict comprehension (the `out = [ {...} for _, r in sel.iterrows() ]` block), add the field:

```python
                "cap_score": float(r["cap_score"]),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/services/test_cross_sectional_ranker.py -v`
Expected: PASS (all existing ranker tests + the new one).

- [ ] **Step 5: Commit**

```bash
git add services/cross_sectional_ranker.py tests/services/test_cross_sectional_ranker.py
git commit -m "feat(ranker): emit cap_score (normalized capitulation magnitude) for composite blending"
```

---

## Task 2: `MultiDayCompositeSelector` (pure selection / dedupe / rank)

**Files:**
- Create: `services/multiday_composite_selector.py`
- Test: `tests/services/test_multiday_composite_selector.py`

Pure function of (baskets, held, weights, config) — no IO, no broker, no clock. Blends `cap_score` across setups by a weighted **sum** (consensus boost), dedupes to one row/symbol, drops held names, ranks by `(composite desc, tshock desc, bare-symbol asc)`, caps to `limit`.

- [ ] **Step 1: Write the failing test**

Create `tests/services/test_multiday_composite_selector.py`:

```python
from services.multiday_composite_selector import MultiDayCompositeSelector


def _cand(sym, cap_score, tshock=2.0, close=100.0, trail_ret=-0.1):
    return {"symbol": sym, "cap_score": cap_score, "tshock": tshock,
            "close": close, "trail_ret": trail_ret, "adv_tier": 1, "rank_pct": 0.02}


def _sel(clip=3.0):
    return MultiDayCompositeSelector({"max_new_per_day": 10, "max_concurrent": 50,
                                      "cap_score_clip": clip, "tiebreaker": "tshock"})


def test_consensus_sum_outranks_single_setup():
    # ABC flagged by two setups (1.0 + 1.0 = 2.0) beats XYZ flagged once at 1.5.
    baskets = {
        "A2": [_cand("ABC", 1.0), _cand("XYZ", 1.5)],
        "C1": [_cand("ABC", 1.0)],
    }
    chosen = _sel().select(baskets, held_symbols=set(),
                           weights={"A2": 1.0, "C1": 1.0}, limit=10)
    assert [c["bare"] for c in chosen] == ["ABC", "XYZ"]
    abc = chosen[0]
    assert abc["composite"] == 2.0
    assert sorted(abc["contributors"]) == ["A2", "C1"]
    assert abc["per_setup_cap_score"] == {"A2": 1.0, "C1": 1.0}


def test_deep_single_setup_can_outrank_mild_consensus():
    baskets = {"A2": [_cand("DEEP", 3.0)], "C1": [_cand("MILD", 0.4)],
               "C4": [_cand("MILD", 0.4)]}
    chosen = _sel().select(baskets, set(), {"A2": 1.0, "C1": 1.0, "C4": 1.0}, limit=10)
    assert chosen[0]["bare"] == "DEEP"  # 3.0 > 0.8


def test_dedup_owner_is_max_weighted_contributor():
    baskets = {"A2": [_cand("ABC", 0.5)], "C1": [_cand("ABC", 2.0)]}
    chosen = _sel().select(baskets, set(), {"A2": 1.0, "C1": 1.0}, limit=10)
    assert len(chosen) == 1
    assert chosen[0]["owner"] == "C1"  # higher weighted cap_score


def test_held_symbols_excluded():
    baskets = {"A2": [_cand("ABC", 2.0), _cand("XYZ", 1.0)]}
    chosen = _sel().select(baskets, held_symbols={"ABC"}, weights={"A2": 1.0}, limit=10)
    assert [c["bare"] for c in chosen] == ["XYZ"]


def test_nse_prefix_normalized_for_held_and_output():
    baskets = {"A2": [_cand("NSE:ABC", 2.0)]}
    chosen = _sel().select(baskets, held_symbols={"ABC"}, weights={"A2": 1.0}, limit=10)
    assert chosen == []  # NSE:ABC dedupes against bare-held ABC


def test_limit_caps_after_ranking():
    baskets = {"A2": [_cand("A", 3.0), _cand("B", 2.0), _cand("C", 1.0)]}
    chosen = _sel().select(baskets, set(), {"A2": 1.0}, limit=2)
    assert [c["bare"] for c in chosen] == ["A", "B"]


def test_cap_score_upper_clip_applied():
    baskets = {"A2": [_cand("ABC", 9.0)], "C1": [_cand("ABC", 9.0)]}
    chosen = _sel(clip=3.0).select(baskets, set(), {"A2": 1.0, "C1": 1.0}, limit=10)
    assert chosen[0]["composite"] == 6.0  # min(9,3)*1 + min(9,3)*1


def test_weights_scale_contribution():
    baskets = {"A2": [_cand("ABC", 1.0)], "C1": [_cand("XYZ", 1.0)]}
    chosen = _sel().select(baskets, set(), {"A2": 2.0, "C1": 1.0}, limit=10)
    assert chosen[0]["bare"] == "ABC" and chosen[0]["composite"] == 2.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/services/test_multiday_composite_selector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'services.multiday_composite_selector'`.

- [ ] **Step 3: Implement the module**

Create `services/multiday_composite_selector.py`:

```python
"""Cross-setup integrated-composite selector for the multi-day CNC/MTF family.

Pure function of (baskets, held, weights, config): blends each name's per-setup
`cap_score` into a single composite score (weighted SUM across the setups that
selected it — so multi-setup agreement raises rank), dedupes to one row per
symbol, drops names already held by any setup, and returns the top-`limit`
composite-ranked candidates. No IO, no broker, no clock (live/backtest-identical,
IST-naive by construction). NO hardcoded defaults (CLAUDE.md rule 1).

Spec: specs/2026-06-29-multiday-composite-selection-design.md
"""
from __future__ import annotations

from typing import Any, Dict, List, Set

from config.logging_config import get_agent_logger

logger = get_agent_logger()


def _bare(symbol: str) -> str:
    """Canonical bare ticker for cross-setup dedupe (strip NSE:, upper)."""
    return str(symbol).replace("NSE:", "").upper()


class MultiDayCompositeSelector:
    """Blend per-setup baskets into one deduped, consensus-ranked basket."""

    def __init__(self, config: Dict[str, Any]):
        # Fail-fast on every key (no silent defaults).
        self.max_new_per_day = int(config["max_new_per_day"])
        self.max_concurrent = int(config["max_concurrent"])
        self.cap_score_clip = float(config["cap_score_clip"])
        self.tiebreaker = str(config["tiebreaker"])
        if self.tiebreaker != "tshock":
            raise ValueError(f"unsupported tiebreaker {self.tiebreaker!r} (v1: 'tshock')")

    def select(
        self,
        baskets: Dict[str, List[Dict[str, Any]]],
        held_symbols: Set[str],
        weights: Dict[str, float],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Return the deduped, composite-ranked basket (≤ `limit` rows).

        Args:
            baskets: {setup_name: [ranker cand dict, ...]} — each cand carries
                `symbol`, `cap_score`, `tshock`, `close`, `trail_ret`.
            held_symbols: bare or NSE: symbols already held by ANY setup.
            weights: {setup_name: composite_weight}.
            limit: max names to return (caller computes from combined caps).

        Returns: [{symbol(NSE:), bare, composite, tshock, owner, contributors,
            per_setup_cap_score, close, trail_ret}], composite-desc.
        """
        held = {_bare(s) for s in held_symbols}
        agg: Dict[str, Dict[str, Any]] = {}
        for setup_name, cands in baskets.items():
            w = float(weights[setup_name])
            for cand in cands:
                bare = _bare(cand["symbol"])
                if bare in held:
                    continue
                contrib = w * min(float(cand["cap_score"]), self.cap_score_clip)
                a = agg.get(bare)
                if a is None:
                    a = {
                        "bare": bare, "composite": 0.0, "tshock": 0.0,
                        "contributors": [], "per_setup_cap_score": {},
                        "_owner_weighted": -1.0, "owner": None,
                        "close": float(cand["close"]),
                        "trail_ret": float(cand["trail_ret"]),
                    }
                    agg[bare] = a
                a["composite"] += contrib
                a["tshock"] = max(a["tshock"], float(cand["tshock"]))
                a["contributors"].append(setup_name)
                a["per_setup_cap_score"][setup_name] = float(cand["cap_score"])
                if contrib > a["_owner_weighted"]:
                    a["_owner_weighted"] = contrib
                    a["owner"] = setup_name
                    a["close"] = float(cand["close"])
                    a["trail_ret"] = float(cand["trail_ret"])

        rows = sorted(
            agg.values(),
            key=lambda a: (-a["composite"], -a["tshock"], a["bare"]),
        )
        capped = rows[: max(0, int(limit))]
        out: List[Dict[str, Any]] = []
        for a in capped:
            a.pop("_owner_weighted", None)
            a["symbol"] = f"NSE:{a['bare']}"
            a["contributors"] = sorted(set(a["contributors"]))
            out.append(a)
        logger.info(
            "composite_selector: %d unique candidates -> %d chosen (limit=%d, %d held excluded)",
            len(agg), len(out), limit, len(held),
        )
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/services/test_multiday_composite_selector.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add services/multiday_composite_selector.py tests/services/test_multiday_composite_selector.py
git commit -m "feat(multiday): MultiDayCompositeSelector — cross-setup blend/dedupe/rank"
```

---

## Task 3: Config keys (per-setup `composite_weight` + `cap_score_clip`; family block)

**Files:**
- Modify: `config/configuration.json` — each of the four `multi_day` setups + a new top-level-of-`setups`? NO — family block lives under a new top-level key `multi_day_portfolio` (sibling of `setups`).
- Test: `tests/config/test_multiday_config_keys.py`

- [ ] **Step 1: Write the failing test**

Create `tests/config/test_multiday_config_keys.py`:

```python
import json
from pathlib import Path

_CFG = Path(__file__).resolve().parents[2] / "config" / "configuration.json"
_MULTIDAY = ("mtf_capitulation_revert_long", "low52_capitulation_revert_long",
             "zscore_oversold_revert_long", "crash2d_revert_long")


def _load():
    return json.loads(_CFG.read_text(encoding="utf-8"))


def test_each_multiday_setup_has_composite_weight_and_clip():
    cfg = _load()
    for name in _MULTIDAY:
        block = cfg["setups"][name]
        assert isinstance(block["composite_weight"], (int, float))
        assert float(block["composite_weight"]) == 1.0  # equal-weight v1
        assert float(block["cap_score_clip"]) > 0.0


def test_multi_day_portfolio_family_block_present():
    cfg = _load()
    fam = cfg["multi_day_portfolio"]
    assert int(fam["max_new_per_day"]) > 0
    assert int(fam["max_concurrent"]) > 0
    assert float(fam["cap_score_clip"]) > 0.0
    assert fam["tiebreaker"] == "tshock"
    assert isinstance(fam["selection_log_path"], str) and fam["selection_log_path"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/config/test_multiday_config_keys.py -v`
Expected: FAIL with `KeyError: 'composite_weight'`.

- [ ] **Step 3: Add the keys**

In `config/configuration.json`, add to EACH of the four setup blocks (`mtf_capitulation_revert_long`, `low52_capitulation_revert_long`, `zscore_oversold_revert_long`, `crash2d_revert_long`) — place next to their existing `hold_days` key:

```json
      "composite_weight": 1.0,
      "_comment_composite_weight": "Equal-weight v1 (1/N). Promotion to IC-weighted gated on specs/2026-06-29-multiday-composite-selection-design.md section 6.1.",
      "cap_score_clip": 3.0,
      "_comment_cap_score_clip": "Upper clip on the per-setup cap_score before composite blending (caps a single extreme z so one setup can't dominate the sum).",
```

Then add a NEW top-level key (sibling of `"setups"`), `multi_day_portfolio`:

```json
  "multi_day_portfolio": {
    "_comment": "Cross-setup composite selection for horizon=multi_day setups. Technical selection only — margin-pool arbitration out of scope (paper testing). See specs/2026-06-29-multiday-composite-selection-design.md.",
    "max_new_per_day": 100,
    "max_concurrent": 200,
    "_comment_caps": "Generous selection caps in paper (effectively 'take all'); ordering is still real. Tighten when a setup goes live.",
    "cap_score_clip": 3.0,
    "tiebreaker": "tshock",
    "selection_log_path": "logs/multiday_selection.jsonl"
  },
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/config/test_multiday_config_keys.py -v`
Expected: PASS (2 tests).

Also confirm the JSON still parses end-to-end:
Run: `.venv/Scripts/python -c "import json; json.load(open('config/configuration.json', encoding='utf-8')); print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add config/configuration.json tests/config/test_multiday_config_keys.py
git commit -m "feat(config): composite_weight + cap_score_clip per multi-day setup; multi_day_portfolio family block"
```

---

## Task 4: Collective composite entries + cross-day held-union filter

**Files:**
- Modify: `services/execution/mtf_capitulation_handlers.py` — `run_eod` entries dispatch; replace the per-setup `_run_entries(name, raw, ...)` call with a single collective `_run_entries_composite(setups, ...)`. Keep `_run_exits` per-setup and unchanged (Task 5 modifies its tripwire feed only).
- Test: `tests/services/execution/test_mtf_capitulation_handlers.py`

The collective entry pass: rank each non-paused setup → basket (with `cap_score`); build `held_union` across ALL setups' stores; call the selector; place AMO BUYs for the chosen names via the **owner** setup's persistence, tagging `contributors` + `per_setup_cap_score` on the saved position.

- [ ] **Step 1: Write the failing test**

Add to `tests/services/execution/test_mtf_capitulation_handlers.py` (reuse that file's existing fixtures for `config`, `broker`, `monkeypatch` of `make_provider`/`MtfUniverse`; mirror the existing `test_run_eod_*` entry tests). If the file lacks a shared multi-setup fixture, add this self-contained test:

```python
def test_composite_entries_dedup_two_setups_one_position(monkeypatch, tmp_path):
    import services.execution.mtf_capitulation_handlers as mh

    # Two multi-day setups both flag SHARED; A2 also flags AONLY.
    cfg = _two_setup_config(tmp_path)  # helper below
    monkeypatch.setattr(mh, "_eligible_multiday_setups",
                        lambda config, *, paper_mode: [("A2", cfg["setups"]["A2"]),
                                                       ("C1", cfg["setups"]["C1"])])
    monkeypatch.setattr(mh, "_decay_paused", lambda name, raw: False)
    monkeypatch.setattr(mh, "_prewarm_daily_universe", lambda setups, broker: None)

    # Stub each setup's ranker output (cap_score-bearing baskets).
    def fake_rank_for(name):
        if name == "A2":
            return [{"symbol": "SHARED", "cap_score": 1.0, "tshock": 3.0, "close": 100.0,
                     "trail_ret": -0.12, "adv_tier": 1, "rank_pct": 0.01},
                    {"symbol": "AONLY", "cap_score": 0.5, "tshock": 2.5, "close": 50.0,
                     "trail_ret": -0.10, "adv_tier": 1, "rank_pct": 0.04}]
        return [{"symbol": "SHARED", "cap_score": 1.0, "tshock": 2.0, "close": 100.0,
                 "trail_ret": -0.09, "adv_tier": 1, "rank_pct": 0.02}]
    monkeypatch.setattr(mh, "_rank_basket_for_setup",
                        lambda name, raw, broker, today, ca_ex_dates, repo_root: fake_rank_for(name))

    broker = _stub_broker_amo()  # place_order returns a fake order id
    summary = mh.run_eod(cfg, broker, now_ist=pd.Timestamp("2026-06-22 15:35:00"),
                         paper_mode=True, phase="entries")

    # SHARED entered ONCE (deduped), owner = the higher weighted cap_score (tie -> deterministic).
    placed = [e for e in summary["events"]]
    symbols = sorted(e["symbol"] for e in placed)
    assert symbols == ["NSE:AONLY", "NSE:SHARED"]
    shared = next(e for e in placed if e["symbol"] == "NSE:SHARED")
    assert sorted(shared["contributors"]) == ["A2", "C1"]

    # SHARED persisted in exactly one store, with contributors tagged.
    from services.state.position_persistence import PositionPersistence
    a2_store = PositionPersistence(mh._position_state_dir(cfg["setups"]["A2"]))
    c1_store = PositionPersistence(mh._position_state_dir(cfg["setups"]["C1"]))
    in_a2 = a2_store.get_position("NSE:SHARED") is not None
    in_c1 = c1_store.get_position("NSE:SHARED") is not None
    assert in_a2 ^ in_c1  # exactly one store holds it
    owner_store = a2_store if in_a2 else c1_store
    pos = owner_store.get_position("NSE:SHARED")
    assert sorted(pos.state["contributors"]) == ["A2", "C1"]


def test_composite_entries_skip_held_union(monkeypatch, tmp_path):
    import services.execution.mtf_capitulation_handlers as mh
    cfg = _two_setup_config(tmp_path)
    # Pre-seed SHARED as held in C1's store (still inside its hold window).
    from services.state.position_persistence import PositionPersistence
    c1_store = PositionPersistence(mh._position_state_dir(cfg["setups"]["C1"]))
    c1_store.save_position(symbol="NSE:SHARED", side="BUY", qty=10, avg_price=100.0,
                           trade_id="t", entry_date="2026-06-20", exit_on_date="2026-06-24",
                           product="MTF", state={"qty": 10})
    monkeypatch.setattr(mh, "_eligible_multiday_setups",
                        lambda config, *, paper_mode: [("A2", cfg["setups"]["A2"]),
                                                       ("C1", cfg["setups"]["C1"])])
    monkeypatch.setattr(mh, "_decay_paused", lambda name, raw: False)
    monkeypatch.setattr(mh, "_prewarm_daily_universe", lambda setups, broker: None)
    monkeypatch.setattr(mh, "_rank_basket_for_setup",
                        lambda name, raw, broker, today, ca_ex_dates, repo_root:
                        [{"symbol": "SHARED", "cap_score": 5.0, "tshock": 3.0, "close": 100.0,
                          "trail_ret": -0.2, "adv_tier": 1, "rank_pct": 0.01}])
    broker = _stub_broker_amo()
    summary = mh.run_eod(cfg, broker, now_ist=pd.Timestamp("2026-06-22 15:35:00"),
                         paper_mode=True, phase="entries")
    assert summary["entered_count"] == 0  # SHARED already held -> excluded
```

Add these helpers to the test file (near the top, after imports):

```python
def _two_setup_config(tmp_path):
    def _block(state_name, weight=1.0):
        return {
            "horizon": "multi_day", "enabled": False, "paper_enabled": True,
            "selection_mode": "trailing_loser_decile", "lookback_days": 5, "loser_pct": 0.1,
            "adv_tier": 1, "adv_tier_count": 5, "turnover_shock_min": 2.0,
            "shock_lookback_days": 20, "adv_floor_inr": 2_000_000, "min_price": 5.0,
            "min_universe_symbols_per_day": 20, "hold_days": 2,
            "exclude_ca_in_hold_window": False, "ca_events_path": "",
            "composite_weight": weight, "cap_score_clip": 3.0,
            "mtf": {"approved_list_snapshot_path": "data/mtf_universe/approved_mtf_securities_2026-05-21.json",
                    "interest_pct_per_day": 0.0004, "exclude_etf": True,
                    "fallback_to_cnc_if_not_mtf": True, "stale_snapshot_warn_days": 7},
            "capital_allocation": {"state_file": str(tmp_path / f"{state_name}.json"),
                                   "max_concurrent_slots": 100, "margin_per_slot_inr": 100000,
                                   "max_new_positions_per_day": 100},
        }
    return {
        "setups": {"A2": _block("a2_slots"), "C1": _block("c1_slots")},
        "multi_day_portfolio": {"max_new_per_day": 100, "max_concurrent": 200,
                                "cap_score_clip": 3.0, "tiebreaker": "tshock",
                                "selection_log_path": str(tmp_path / "sel.jsonl")},
    }


def _stub_broker_amo():
    from unittest.mock import MagicMock
    b = MagicMock()
    b.place_order.return_value = "AMO1"
    b._dry_session_date = None
    return b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/services/execution/test_mtf_capitulation_handlers.py::test_composite_entries_dedup_two_setups_one_position -v`
Expected: FAIL — `AttributeError: module ... has no attribute '_rank_basket_for_setup'` (and the collective path not wired).

- [ ] **Step 3: Extract `_rank_basket_for_setup` and add `_run_entries_composite`**

In `services/execution/mtf_capitulation_handlers.py`:

(a) Refactor the ranking half of the existing `_run_entries` into a reusable helper that returns a basket (the AMO/universe/panel setup stays, the placement loop moves out). Add:

```python
def _rank_basket_for_setup(name, raw, broker, today, ca_ex_dates, repo_root):
    """Build one setup's ranked basket (cap_score-bearing) for `today`.

    Returns [] when the setup has no MTF universe, no panel, or an empty basket.
    The MTF prefetch + panel build are the same as the legacy per-setup path.
    """
    mtf_cfg = raw["mtf"]
    mtf = MtfUniverse(Path(str(mtf_cfg["approved_list_snapshot_path"])))
    exclude_etf = bool(mtf_cfg["exclude_etf"])
    eligible = {s for s in mtf.all_symbols() if mtf.is_eligible(s, exclude_etf=exclude_etf)}
    if not eligible:
        logger.warning("mtf_capitulation[%s]: empty MTF eligible set; no basket", name)
        return []
    if ca_ex_dates is None and bool(raw.get("exclude_ca_in_hold_window")):
        ca_ex_dates = _load_ca_ex_dates(raw, repo_root)

    if not _is_dry_run(broker) and hasattr(broker, "set_intraday_5m_prefetch"):
        sdk = getattr(broker, "_data_sdk", None)
        if sdk is not None and hasattr(sdk, "async_fetch_intraday_5m_batch"):
            existing = getattr(broker, "_intraday_5m_prefetch", {}) or {}
            need = [f"NSE:{s}" for s in eligible if f"NSE:{s}" not in existing and s not in existing]
            if need:
                import asyncio
                try:
                    fetched = asyncio.run(sdk.async_fetch_intraday_5m_batch(need, concurrency=30, rps=20.0))
                    merged = dict(existing); merged.update(fetched or {})
                    broker.set_intraday_5m_prefetch(merged)
                except Exception as e:
                    logger.exception("mtf_capitulation[%s]: 5m batch prewarm failed: %s", name, e)

    provider = make_provider(raw, dry_run=_is_dry_run(broker),
                             fetch_fn=getattr(broker, "fetch_daily_window", None),
                             mtf_symbols=eligible, repo_root=repo_root)
    panel = provider.get_panel(today)
    if panel is None or panel.empty:
        logger.warning("mtf_capitulation[%s]: empty daily panel for %s; no basket", name, today)
        return []
    basket = CrossSectionalRanker(raw).rank(panel, today, eligible, ca_ex_dates=ca_ex_dates)
    return basket or []
```

(b) Add the collective entry pass:

```python
def _held_union(setups, persistences):
    """Bare symbols currently held across ALL setups' stores (cross-day dedupe)."""
    held = set()
    for name, _raw in setups:
        for sym in persistences[name].load_snapshot().keys():
            held.add(str(sym).replace("NSE:", "").upper())
    return held


def _run_entries_composite(setups, broker, persistences, today, now, paper_mode,
                           summary, *, ca_ex_dates, repo_root, config):
    from services.multiday_composite_selector import MultiDayCompositeSelector

    active = [(name, raw) for name, raw in setups if not _decay_paused(name, raw)]
    for name, _raw in setups:
        if (name, _raw) not in active:
            summary.setdefault("by_setup", {}).setdefault(name, {})["decay_paused"] = True
    if not active:
        logger.info("mtf_capitulation: all multi-day setups decay-paused; no entries")
        return

    baskets = {}
    for name, raw in active:
        baskets[name] = _rank_basket_for_setup(name, raw, broker, today, ca_ex_dates, repo_root)
    if not any(baskets.values()):
        logger.info("mtf_capitulation: all baskets empty for %s; no entries", today)
        return

    fam = config["multi_day_portfolio"]
    selector = MultiDayCompositeSelector(fam)
    weights = {name: float(raw["composite_weight"]) for name, raw in active}
    held = _held_union(setups, persistences)
    total_held = sum(len(persistences[n].load_snapshot()) for n, _ in setups)
    limit = min(int(fam["max_new_per_day"]),
                max(0, int(fam["max_concurrent"]) - total_held))
    chosen = selector.select(baskets, held_symbols=held, weights=weights, limit=limit)
    if not chosen:
        return

    entry_date = _next_trading_day(today)
    for c in chosen:
        owner = c["owner"]
        raw = dict(active)[owner]
        exit_on_date = _add_trading_days(entry_date, int(raw["hold_days"]))
        bare = c["bare"]; symbol = c["symbol"]
        mtf = MtfUniverse(Path(str(raw["mtf"]["approved_list_snapshot_path"])))
        info = mtf.lookup(bare)
        if info is not None:
            product, leverage = "MTF", float(info.leverage)
        elif bool(raw["mtf"]["fallback_to_cnc_if_not_mtf"]):
            product, leverage = "CNC", 1.0
        else:
            summary["rejected_count"] += 1
            continue
        margin_per_slot = float(raw["capital_allocation"]["margin_per_slot_inr"])
        qty = int((margin_per_slot * leverage) // float(c["close"]))
        if qty <= 0:
            summary["rejected_count"] += 1
            continue
        trade_id = f"{owner}_{today.isoformat()}_{bare}"
        try:
            order_id = _place_amo_buy(broker, symbol, qty, product, trade_id)
        except Exception as e:
            logger.error("mtf_capitulation[%s]: AMO BUY failed for %s: %s", owner, symbol, e)
            summary["skipped_count"] += 1
            continue
        persistences[owner].save_position(
            symbol=symbol, side="BUY", qty=qty, avg_price=0.0, trade_id=trade_id,
            order_id=str(order_id), order_tag=trade_id,
            plan={"setup": owner, "trail_ret": c["trail_ret"], "tshock": c["tshock"],
                  "composite": c["composite"]},
            state={"pending_entry_fill": True, "qty": qty, "leverage": leverage,
                   "signal_close": float(c["close"]), "signal_date": today.isoformat(),
                   "contributors": c["contributors"],
                   "per_setup_cap_score": c["per_setup_cap_score"]},
            entry_date=entry_date.isoformat(), exit_on_date=exit_on_date.isoformat(),
            product=product)
        summary["entered_count"] += 1
        summary.setdefault("by_setup", {}).setdefault(owner, {"entered": 0})
        summary["by_setup"][owner]["entered"] = summary["by_setup"][owner].get("entered", 0) + 1
        summary["events"].append({
            "setup": owner, "symbol": symbol, "qty": qty, "product": product,
            "leverage": leverage, "entry_date": entry_date.isoformat(),
            "exit_on_date": exit_on_date.isoformat(), "amo_buy_order_id": str(order_id),
            "composite": c["composite"], "contributors": c["contributors"],
        })
```

(c) Rewire `run_eod`. Replace the per-setup entries dispatch. The current loop builds `persistence = PositionPersistence(_position_state_dir(raw))` per setup and calls `_run_exits` then `_run_entries`. Change to: build a `persistences` dict once, run exits per-setup, run entries collectively:

```python
    summary["by_setup"] = {}
    persistences = {name: PositionPersistence(_position_state_dir(raw)) for name, raw in setups}
    # ---- Phase A: exits due today (per-setup, pre-close) ----
    if phase in ("both", "exits"):
        for name, raw in setups:
            _run_exits(name, raw, broker, persistences[name], today, now, paper_mode, summary,
                       setups_by_name={n: r for n, r in setups})
    # ---- Phase B: rank + AMO BUY across the whole family (post-close) ----
    if phase in ("both", "entries"):
        _run_entries_composite(setups, broker, persistences, today, now, paper_mode, summary,
                               ca_ex_dates=ca_ex_dates, repo_root=repo_root, config=config)
```

(Delete the old `_run_entries` once `_rank_basket_for_setup` + `_run_entries_composite` cover it. The `_run_exits` signature gains `setups_by_name` in Task 5; add the parameter now with a default of `None` so this task's tests pass, then Task 5 uses it.)

For this task, update `_run_exits`' signature only:
```python
def _run_exits(name, raw, broker, persistence, today, now, paper_mode, summary,
               setups_by_name=None) -> None:
```
(body unchanged in Task 4.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/services/execution/test_mtf_capitulation_handlers.py -v`
Expected: PASS — new composite tests + existing handler tests (existing `run_eod` entry tests still pass because the single-setup path now routes through the collective pass with one basket).

- [ ] **Step 5: Commit**

```bash
git add services/execution/mtf_capitulation_handlers.py tests/services/execution/test_mtf_capitulation_handlers.py
git commit -m "feat(multiday): collective composite entries + cross-day held-union dedupe"
```

---

## Task 5: Multi-contributor decay-tripwire attribution on exit

**Files:**
- Modify: `services/execution/mtf_capitulation_handlers.py` — `_run_exits` feeds EVERY contributing setup's tripwire (not just the owner's), using the `contributors` tag on the position state and `setups_by_name` for each contributor's tripwire config.
- Test: `tests/services/execution/test_mtf_capitulation_handlers.py`

- [ ] **Step 1: Write the failing test**

```python
def test_exit_feeds_all_contributors_tripwires(monkeypatch, tmp_path):
    import services.execution.mtf_capitulation_handlers as mh
    from services.state.position_persistence import PositionPersistence
    from services.risk.decay_tripwire import DecayTripwire

    cfg = _two_setup_config(tmp_path)
    # give both setups a decay tripwire
    for n in ("A2", "C1"):
        cfg["setups"][n]["decay_tripwire"] = {
            "window_trades": 30, "pf_floor": 1.2, "sustained_weeks": 6,
            "state_file": str(tmp_path / f"tw_{n}.json")}
    # Seed an owned-by-A2 position that exits today, tagged contributors=[A2,C1].
    a2_store = PositionPersistence(mh._position_state_dir(cfg["setups"]["A2"]))
    a2_store.save_position(symbol="NSE:SHARED", side="BUY", qty=10, avg_price=100.0,
                           trade_id="A2_2026-06-20_SHARED", entry_date="2026-06-20",
                           exit_on_date="2026-06-22", product="MTF",
                           state={"qty": 10, "leverage": 2.5, "entry_fill_price": 100.0,
                                  "contributors": ["A2", "C1"]})
    monkeypatch.setattr(mh, "_eligible_multiday_setups",
                        lambda config, *, paper_mode: [("A2", cfg["setups"]["A2"]),
                                                       ("C1", cfg["setups"]["C1"])])
    monkeypatch.setattr(mh, "_paper_close_price", lambda b, s, d: 110.0)  # +10% exit
    broker = _stub_broker_amo()
    mh.run_eod(cfg, broker, now_ist=pd.Timestamp("2026-06-22 15:28:00"),
               paper_mode=True, phase="exits")
    # both A2 and C1 tripwires recorded one trade
    for n in ("A2", "C1"):
        tw = DecayTripwire(setup_name=n, state_path=tmp_path / f"tw_{n}.json",
                           window_trades=30, pf_floor=1.2, sustained_weeks=6)
        assert len(tw._trades) == 1  # noqa: SLF001
        assert tw._trades[0].net_pnl_inr > 0  # +10% gross, profitable
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/services/execution/test_mtf_capitulation_handlers.py::test_exit_feeds_all_contributors_tripwires -v`
Expected: FAIL — only A2's tripwire is fed (C1's `tw._trades` is empty).

- [ ] **Step 3: Feed all contributors in `_run_exits`**

In `services/execution/mtf_capitulation_handlers.py`, REPLACE the existing single-setup tripwire block at the end of `_run_exits` (the `tw_cfg = raw.get("decay_tripwire")` block) with a loop over contributors:

```python
        # Feed the realized trade to EVERY contributing setup's decay tripwire,
        # not just the owner's — the book holds the name once (owner store) but
        # each setup that flagged it must see the outcome for standalone-edge
        # measurement (spec section 5). Falls back to the owner when the position
        # predates contributor tagging.
        contributors = pos.state.get("contributors") or [name]
        lookup = setups_by_name or {name: raw}
        for cname in contributors:
            craw = lookup.get(cname)
            if craw is None:
                continue
            tw_cfg = craw.get("decay_tripwire")
            if tw_cfg is None:
                continue
            from services.risk.decay_tripwire import DecayTripwire
            DecayTripwire(
                setup_name=cname,
                state_path=Path(tw_cfg["state_file"]),
                window_trades=int(tw_cfg["window_trades"]),
                pf_floor=float(tw_cfg["pf_floor"]),
                sustained_weeks=int(tw_cfg["sustained_weeks"]),
            ).record_trade(
                net_pnl_inr=float(net), ts_iso=now.isoformat(),
                fees_inr=float(fees_only) + float(interest), gross_pnl_inr=float(gross),
                symbol=symbol, entry_price=float(entry_price),
                exit_price=float(sell_price), exit_reason="kday_close_moc", qty=int(qty),
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/services/execution/test_mtf_capitulation_handlers.py -v`
Expected: PASS (new attribution test + all prior).

- [ ] **Step 5: Commit**

```bash
git add services/execution/mtf_capitulation_handlers.py tests/services/execution/test_mtf_capitulation_handlers.py
git commit -m "feat(multiday): exit feeds every contributing setup's decay tripwire (standalone-edge attribution)"
```

---

## Task 6: Per-day selection diagnostics (jsonl, per setup/symbol/day `cap_score`)

**Files:**
- Modify: `services/execution/mtf_capitulation_handlers.py` — write one jsonl record per (setup, symbol, day) from the baskets, plus composite/owner/contributors/consensus, after selection in `_run_entries_composite`. Path from `multi_day_portfolio.selection_log_path`.
- Test: `tests/services/execution/test_mtf_capitulation_handlers.py`

Spec §7 requires per-(setup, symbol, day) `cap_score` persisted (the IC join in §6.1 needs it), logging **every flagged name**, not only chosen ones.

- [ ] **Step 1: Write the failing test**

```python
def test_selection_diagnostics_logged_per_setup_symbol(monkeypatch, tmp_path):
    import json as _json
    import services.execution.mtf_capitulation_handlers as mh
    cfg = _two_setup_config(tmp_path)
    log_path = tmp_path / "sel.jsonl"
    cfg["multi_day_portfolio"]["selection_log_path"] = str(log_path)
    monkeypatch.setattr(mh, "_eligible_multiday_setups",
                        lambda config, *, paper_mode: [("A2", cfg["setups"]["A2"]),
                                                       ("C1", cfg["setups"]["C1"])])
    monkeypatch.setattr(mh, "_decay_paused", lambda name, raw: False)
    monkeypatch.setattr(mh, "_prewarm_daily_universe", lambda setups, broker: None)
    monkeypatch.setattr(mh, "_rank_basket_for_setup",
                        lambda name, raw, broker, today, ca_ex_dates, repo_root:
                        ([{"symbol": "SHARED", "cap_score": 1.0, "tshock": 3.0, "close": 100.0,
                           "trail_ret": -0.12, "adv_tier": 1, "rank_pct": 0.01}]
                         if name == "C1" else
                         [{"symbol": "SHARED", "cap_score": 1.0, "tshock": 3.0, "close": 100.0,
                           "trail_ret": -0.12, "adv_tier": 1, "rank_pct": 0.01},
                          {"symbol": "AONLY", "cap_score": 0.5, "tshock": 2.5, "close": 50.0,
                           "trail_ret": -0.10, "adv_tier": 1, "rank_pct": 0.04}]))
    broker = _stub_broker_amo()
    mh.run_eod(cfg, broker, now_ist=pd.Timestamp("2026-06-22 15:35:00"),
               paper_mode=True, phase="entries")
    rows = [_json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    # one row per (setup, symbol): A2/SHARED, A2/AONLY, C1/SHARED
    keyed = {(r["setup"], r["symbol"]): r for r in rows}
    assert set(keyed) == {("A2", "SHARED"), ("A2", "AONLY"), ("C1", "SHARED")}
    assert keyed[("A2", "SHARED")]["cap_score"] == 1.0
    assert keyed[("A2", "SHARED")]["session_date"] == "2026-06-22"
    assert keyed[("A2", "SHARED")]["consensus_count"] == 2  # flagged by A2 + C1
    assert keyed[("A2", "AONLY")]["consensus_count"] == 1
    assert keyed[("A2", "SHARED")]["chosen"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/services/execution/test_mtf_capitulation_handlers.py::test_selection_diagnostics_logged_per_setup_symbol -v`
Expected: FAIL — `FileNotFoundError` (log not written).

- [ ] **Step 3: Write the diagnostics record**

Add a helper to `services/execution/mtf_capitulation_handlers.py`:

```python
def _log_selection_diagnostics(baskets, chosen, today, log_path_str):
    """One jsonl row per (setup, symbol, day): cap_score + composite/owner/
    contributors/consensus + chosen flag. Feeds the section 6.1 IC analysis.
    Logs EVERY flagged name (not only chosen), so forward-return IC can be
    computed for picks that were capped out. Never breaks the cron.
    """
    import json as _json
    try:
        # consensus + composite views keyed by bare symbol
        consensus, by_sym = {}, {}
        for _setup, cands in baskets.items():
            for c in cands:
                b = str(c["symbol"]).replace("NSE:", "").upper()
                consensus[b] = consensus.get(b, 0) + 1
        for c in chosen:
            by_sym[c["bare"]] = c
        path = Path(log_path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for setup_name, cands in baskets.items():
                for c in cands:
                    b = str(c["symbol"]).replace("NSE:", "").upper()
                    ch = by_sym.get(b)
                    f.write(_json.dumps({
                        "session_date": today.isoformat(),
                        "setup": setup_name, "symbol": b,
                        "cap_score": float(c["cap_score"]), "tshock": float(c["tshock"]),
                        "trail_ret": float(c["trail_ret"]), "rank_pct": float(c["rank_pct"]),
                        "consensus_count": int(consensus[b]),
                        "chosen": ch is not None,
                        "composite": (float(ch["composite"]) if ch else None),
                        "owner": (ch["owner"] if ch else None),
                        "contributors": (ch["contributors"] if ch else None),
                    }) + "\n")
    except Exception as e:  # pragma: no cover - diagnostics must not break the cron
        logger.warning("mtf_capitulation: selection diagnostics log failed: %s", e)
```

Then call it inside `_run_entries_composite`, right AFTER `chosen = selector.select(...)`:

```python
    _log_selection_diagnostics(baskets, chosen, today, fam["selection_log_path"])
    if not chosen:
        return
```

(Move the existing `if not chosen: return` to AFTER the log call so an all-capped day is still recorded.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/services/execution/test_mtf_capitulation_handlers.py -v`
Expected: PASS (diagnostics test + all prior).

- [ ] **Step 5: Commit**

```bash
git add services/execution/mtf_capitulation_handlers.py tests/services/execution/test_mtf_capitulation_handlers.py
git commit -m "feat(multiday): per-(setup,symbol,day) selection diagnostics jsonl for IC analysis"
```

---

## Task 7: Full-suite regression + dry-run replay parity

**Files:**
- No new code. Verification only.

- [ ] **Step 1: Run the multi-day + ranker + config suites**

Run:
```
.venv/Scripts/python -m pytest tests/services/test_cross_sectional_ranker.py tests/services/test_multiday_composite_selector.py tests/services/execution/test_mtf_capitulation_handlers.py tests/config/test_multiday_config_keys.py -v
```
Expected: all PASS.

- [ ] **Step 2: Confirm no regression in the broader execution suite**

Run: `.venv/Scripts/python -m pytest tests/services/execution/ -v`
Expected: all PASS (overnight handler tests unaffected — separate family).

- [ ] **Step 3: Dry-run replay (enable the family in paper, one historical date)**

Temporarily confirm two multi-day setups are `paper_enabled: true` (they already are), then:
Run: `.venv/Scripts/python main.py --dry-run --session-date 2025-01-20`
Expected: process exits 0; if the multi-day cron path runs, `logs/multiday_selection.jsonl` gains rows and the run log shows `composite_selector: N unique candidates -> M chosen`. Inspect overlap: rows with `consensus_count >= 2` quantify real cross-setup duplication on that date.

- [ ] **Step 4: Commit any doc note (optional)**

If the dry-run surfaces a real overlap rate worth recording, append it to `analysis/backtest_findings.md`. Otherwise no commit.

---

## Self-Review

**Spec coverage:**
- §3 architecture (collect→compose→place) → Tasks 4.
- §4 Step A normalize (`cap_score`) → Task 1; Step B blend (sum, equal-weight, tshock tiebreak) → Task 2; Step C select (held-filter, top-N) → Tasks 2 + 4.
- §5 dedupe/owner/attribution + held-union → Tasks 4 (dedupe/owner/held) + 5 (multi-contributor tripwire).
- §6 config (`composite_weight`, family block) → Task 3; §6.1 promotion criterion is analysis-time (no code now) — its data dependency (`cap_score` per setup/symbol/day) → Task 6.
- §7 diagnostics → Task 6.
- §8 testing → Tasks 1–7 (TDD throughout) + Task 7 (parity).
- §9 live/backtest symmetry → selector is pure (Task 2); no clock use.
- §10 out-of-scope respected (no margin-pool work; equal weights; selection-based blend; close_dn untouched).

**Type consistency:** `cap_score` (float, ≥0) emitted Task 1, consumed Tasks 2/6. Selector output keys (`symbol`, `bare`, `composite`, `tshock`, `owner`, `contributors`, `per_setup_cap_score`, `close`, `trail_ret`) defined Task 2, consumed Tasks 4/6. `contributors` persisted to `state` Task 4, read Task 5. `setups_by_name` added to `_run_exits` signature Task 4, used Task 5. `multi_day_portfolio` keys defined Task 3, read Tasks 4/6.

**Placeholder scan:** none — every step has concrete code/commands.
