# Brief: STT-Aware Gap-Fade Re-Optimization (Candidate 5)

**Branch:** `research/post-sebi-edge-setups` (do not switch)
**Drafted:** 2026-05-14
**Status:** AUDIT-ONLY. No sweep run. No code changed. No commit.
**Verdict:** **SKIP** — premise does not hold for equity intraday MIS.

---

## Premise verification

### Does the Apr 1, 2026 STT hike affect equity intraday MIS trades?

**No.** The Apr 1, 2026 STT hike applies **only to F&O segment**:
- Futures: 0.02% → 0.05% (sell side, notional)
- Options on premium: 0.10% → 0.15% (sell side)

The equity cash-segment STT for **delivery** (0.1% both sides) and **intraday**
(0.025% sell side) is **untouched** in Budget 2026 — and has been stable since
the previous structural shift in 2008 (when "true-to-label" was introduced and
intraday equity STT became sell-only at 0.025%).

Authoritative confirmation:
1. `data/sebi_calendar/rule_changes.csv` row `2026-04-01` lists `affects` as
   `"STT_drag;F&O_speculation;option_premium"` — F&O only; no equity tag.
2. `tools/report_utils.py` line 24: `STT_RATE = 0.00025` for equity intraday.
   No "post-Apr-2026" variant exists because the rate did not change.
3. Source linked in the rule_changes row (1Finance summary of Budget 2026)
   describes futures + options hike. Budget memorandum makes no mention of
   equity cash-segment STT.
4. gap_fade_short is equity-cash SHORT (per the sweep tool comments and the
   roadmap own note: "Note: gap_fade_short is equity-cash SHORT, not F&O").

### Quantitative impact on a representative trade

Trade: entry ₹500, exit ₹490, base qty 1000, MIS leverage 5x.
Leveraged turnover (sell): ₹490 × 5000 = ₹2,450,000.

| STT regime | Rate | STT on trade | Delta |
|---|---|---|---|
| Pre-Apr-2026 (equity intraday) | 0.025% | ₹612.50 | — |
| Post-Apr-2026 (equity intraday) | **0.025%** (unchanged) | **₹612.50** | **₹0.00** |
| Hypothetical: F&O futures rate applied (0.05%) | 0.05% | ₹1,225.00 | +₹612.50 |
| Hypothetical: futures pre-Apr rate (0.02%) | 0.02% | ₹490.00 | -₹122.50 |

**The actual fee delta on gap_fade_short trades from the Apr 1, 2026 STT
change is zero rupees.** Re-running the sweep with "post-Apr-2026 fees" would
produce identical results to the current sweep because no fee constant has
changed for the equity cash segment.

### Audit of other 2024-2026 fee/regulation changes affecting equity intraday MIS

Walked the full `data/sebi_calendar/rule_changes.csv` for anything tagged
`MIS_leverage`, `STT_drag`, or equity-segment:
- `2025-02-01` — full option premium upfront. Affects MIS only on F&O long
  options. **Does not touch cash-segment MIS.**
- `2025-04-28` — T+1 margin collection. Tightens margin collection timing for
  brokers; does not change Zerodha's published MIS leverage on small-cap cash
  shorts. **No P&L impact.**
- `2025-10-01` — MWPL formula change. F&O position-limit driven. Affects
  *which symbols* are F&O-eligible (knock-on to single-stock-FO universe),
  not the fees on equity-cash gap fades.
- `2024-10-01` STT hike — futures + options only. Same conclusion.

Conclusion: no equity-cash MIS fee parameter (brokerage, STT, exchange, SEBI,
IPFT, stamp duty, GST) has shifted in the Oct-2024 → May-2026 window.

---

## Existing tool audit

### `tools/report_utils.py::calculate_single_trade_charges`

- Fee constants defined as module-level globals at lines 22-29.
- `STT_RATE = 0.00025` is the **current and correct** equity-intraday rate.
- No `is_post_apr_2026` parameter exists — and none is needed for equity.
- `calculate_single_trade_charges` correctly applies MIS leverage to notional
  before computing fees (lines 474-491). Verified fee model is leverage-aware.
- No code branch or feature flag would need to be added for Apr-1-2026
  equity intraday. The fee model already reflects the correct post-Apr-2026
  rates (because they are identical to pre-Apr-2026 rates).

### `tools/sub9_research/_gap_fade_short_sl_target_sweep.py`

- Imports `calc_fee` from `tools/sub7_validation/build_per_setup_pnl.py`
  (line 56) — **not** from `report_utils`. The sweep uses an independent
  fee implementation that mirrors `report_utils` constants.
- `calc_fee` signature is `(entry_price, exit_price, qty, side,
  mis_leverage=1.0)` — leverage-aware.
- **Bug**: the sweep calls `calc_fee(entry_price, exit_price, qty, "SELL")`
  at lines 304-305 and 308 **without passing `mis_leverage`**. This means
  the sweep is using `mis_leverage=1.0` (default), undercounting brokerage by
  ~2-3x on typical small-cap MIS trades. This is the same defect the
  2026-05-13 fix to `calculate_per_trade_final_pnl` was created to prevent.
  **This is a separate, real bug that affects the sweep's PF estimates today
  — independent of the STT question.** It biases the sweep toward
  PF-overstatement on the locked combo because the locked combo's PF was
  computed under under-counted fees.
- Fee parameterization: there is no `is_post_apr_2026` flag in the sweep
  tool or in `calc_fee`. Adding one would be a one-line change to the
  constants module — but adding it would be a no-op for equity because the
  equity rates did not change. The roadmap line 192-194 saying "re-run with
  `is_post_apr_2026=True` in the fee model" is based on a wrong premise.

---

## Pre-registered falsifiers

Standard registration is not applicable because the premise fails at step 1.
If we *were* to proceed anyway (treating this as a generic re-sweep), the
falsifier from the roadmap stands:

- If no SL/target/time-stop combo achieves **PF ≥ 1.30** on post-SEBI
  (Oct 2025 → Apr 2026) gap_fade_short trades using the **leverage-aware**
  fee model (i.e. after fixing the `mis_leverage` bug above), gap_fade
  should be sized down or retired.
- WR floor: ≥ 40% on n ≥ 50 holdout trades.
- Daily-PF distribution: bottom-decile day must net > -3R.

But these would be tested under the **same fee structure** for both pre- and
post-Apr-2026 equity intraday — so this exercise reduces to "do we still
trust the locked combo's PF when fees are correctly counted?", which is a
fee-bug-fix audit, not a regulatory-regime re-optimization.

---

## Effort estimate

| Path | Effort | Outcome |
|---|---|---|
| Re-sweep with `is_post_apr_2026=True` for equity | 0.5 day | **Identical** to current sweep results (no rate change) |
| Fix the `mis_leverage` bug in the sweep then re-run all 3 phases | 0.5 day implementation + ~3-6 hrs compute (3 phases × ~1-2 hrs each based on prior sweep wall-clock) | Different result — locked combo may shift. **This is the real test.** |
| Pure audit + brief (this deliverable) | 2-3 hrs | Premise refuted, ready to redirect attention |

The 1-day estimate in the roadmap is correct for the *audit-then-fix* path
but for the wrong reason — the work would be a fee-correctness fix, not an
STT-regime adaptation.

---

## Recommendation: SKIP, with one carve-out

**Skip Candidate 5 as scoped.** The premise that Apr 1, 2026 STT hike shifts
gap_fade's break-even is empirically false: the hike is F&O-only, gap_fade
is equity-cash. Re-running the sweep with a "post-Apr-2026 fee flag" would
return identical numbers because no equity-cash fee constant changed.

**Carve-out — modify scope to a fee-correctness audit instead:**
While doing this audit I found that `_gap_fade_short_sl_target_sweep.py`
calls `calc_fee` without passing `mis_leverage`, leaving it at the default
`1.0`. The same bug was found and fixed in `report_utils` /
`calculate_per_trade_final_pnl` on 2026-05-13 (see commit message reference
in `report_utils.py` line 561-565). That fix undercounted fees by 2-3x on
typical small-cap MIS gap_fade trades and overstated NET by hundreds of
thousands of rupees on a 2-year OCI Discovery.

**Recommended replacement task:** "Patch the `mis_leverage` bug in
`_gap_fade_short_sl_target_sweep.py` (3 call sites), then re-run Discovery /
OOS / Holdout sweeps and confirm whether the locked combo (`stop_buffer=0.10,
atr_mult=2.0, time_stop=13:00, partial_mode=...`) still dominates under
leverage-aware fees." This is a 1-day total and produces decision-grade
information. Same falsifiers (PF ≥ 1.30 on holdout) but tested under the
**actually-correct** fee model.

If we want a regulatory-regime gap_fade test, the right one is **MWPL/intraday
ban regime effects** on small-cap gap-up exhaustion — not STT.

---

## References (NSE/SEBI circulars on equity STT)

- Budget 2026 STT hike (F&O only): https://1finance.co.in/blog/stt-futures-options-increased-budget-2026-for-fno-investors/
  (referenced in `rule_changes.csv` row 16)
- Budget 2024 STT hike (F&O only, Oct 1 2024): https://www.icicidirect.com/research/equity/finace/new-stt-rules-in-futures-and-options-trading
- Zerodha statutory charges (equity intraday STT 0.025% sell side, unchanged
  since the "true-to-label" reform): https://zerodha.com/charges
- Last structural change to equity-intraday STT: 2008 reform when intraday
  equity STT moved to sell-side only at 0.025%. No subsequent reform has
  touched this rate. SEBI / Income Tax Act references in Finance Act 2008
  (Section 98 of Finance Act 2004 as amended).

---

## Files referenced

- `E:\Codebase\intraday-trade-assistant\specs\2026-05-14-research-post-sebi-edges.md`
- `E:\Codebase\intraday-trade-assistant\tools\report_utils.py` (lines 22-29, 448-502)
- `E:\Codebase\intraday-trade-assistant\tools\sub9_research\_gap_fade_short_sl_target_sweep.py` (lines 56, 297-309)
- `E:\Codebase\intraday-trade-assistant\tools\sub7_validation\build_per_setup_pnl.py` (lines 22-68)
- `E:\Codebase\intraday-trade-assistant\data\sebi_calendar\rule_changes.csv` (rows for 2024-10-01, 2025-02-01, 2025-04-28, 2025-10-01, 2026-04-01)
