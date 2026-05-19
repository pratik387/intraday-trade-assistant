# Brief: `mwpl_recalc_forced_rebalance_fade`

**Research track:** post-SEBI edge setups (`research/post-sebi-edge-setups`)
**Status:** DRAFT — pre-registration phase. Falsifiers locked before any data look.
**Date:** 2026-05-14
**Predecessor:** `specs/2026-05-14-research-post-sebi-edges.md` (Candidate 1 of roadmap)

---

## Mechanism (one sentence)

After the new SEBI MWPL formula (effective 2025-10-01) re-anchors single-stock MWPL to the *lower of* 15% free-float OR 65× rolling-3-month ADDV (floor 10% free-float) and is recalculated **quarterly**, names whose pre-recalc FutEq open interest sits close to (or above) the NEW MWPL must shed positions in the days following the cutover — this forced single-stock-FO speculator-deleveraging produces a directional downside drift over T+0 to T+10 trading days that can be faded SHORT.

## Direction

**SHORT-only** on names where `current_FutEq_OI / new_MWPL >= 0.85` at the moment the new MWPL becomes effective. Long side is rejected:
- Existing speculator-unwind literature (Bessembinder & Seguin; Indian event-study tradition) and the empirical signature that broke `delivery_pct_anomaly_short` post-Oct-2025 both point to *downward* drift when the F&O speculator inventory is unwound, not upward.
- Sub-9 priors: long-bias setups have systematically lost on Indian intraday flow (SEBI FY23 study: 70%/93% loss rates concentrated on long side).
- Excess OI relative to a *binding* limit creates a one-way exit door (no fresh longs allowed past the threshold), structurally consistent with short-side edge.

## Regulatory dependencies

- **regulatory_sensitivity:** `rule_creating`
  - Setup exists ONLY because of the Oct 1 2025 MWPL re-formula. Pre-Oct-2025 MWPL used a different denominator (lower of 30× avg cash volume or 20% non-promoter shares) — different threshold, different binding stocks, no comparable "recalc cliff" because the old formula didn't refresh quarterly on rolling ADDV.
- **depends_on:** `["MWPL", "single_stock_FO", "F&O_speculation", "single_stock_FO_speculator_unwind"]`
- **reasoning:** The mechanism is the *intersection* of (a) new MWPL formula publishing a quarterly-refreshing scalar, (b) single-stock position caps (10/20/30%) that bite when MWPL drops, and (c) speculator-unwind behavioural pattern. Removing any of the three breaks the setup. Strongly recommend ALSO declaring `STT_drag` as a secondary dep because post-2026-04-01 STT hike materially affects break-even for any low-volume, short-hold flavor of this setup.

## Cells (pre-registered, must NOT change after seeing data)

These cell definitions are LOCKED. Discovery cell-mining is restricted to choosing between cells listed here; new cells discovered ex-post are out-of-scope.

| Param | Value | Why |
|---|---|---|
| `cap_segment` (universe) | `mid_cap` and `small_cap` only | Large-caps rarely OI-binding (high free-float scales MWPL); micro-caps not F&O-eligible. |
| `universe_base` | NSE single-stock F&O-eligible list as of recalc date | New MWPL only applies to F&O segment names. |
| `OI_threshold_for_qualification` | `current_FutEq_OI / new_MWPL >= 0.85` at recalc effective date | Pre-registered cliff. Stocks below 0.85 are unconstrained → no forced unwind expected. 0.85 vs 0.95 (ban threshold) gives 10 ppt buffer for events that *should* unwind without entering ban. |
| `entry_window` | T+0 to T+5 trading days after MWPL recalc effective date | Most forced deleveraging completes within first 5 sessions (consistent with general MIS/F&O monthly rollover patterns). |
| `entry_time_intraday` | 10:30 IST single-bar (same anchor as `circuit_t1_fade_short`) | Avoids open-auction noise; aligns with daily pattern where speculators reposition mid-morning. |
| `entry_confirmation` | Latest 5m bar red AND stock down >= 0 from prior-day close (no fade against an up-day open) | Defends against entering on a sympathy bounce within the unwind window. |
| `direction` | SHORT only | Per mechanism. |
| `hard_stop` | Higher of: (a) prior session high + 0.5%, (b) entry × 1.015 (qty-inflation guard) | Single-day worst-case bound. |
| `target_T1` (50%) | 1.0% below entry | Locks in expected-value half. |
| `target_T2` (50%) | 2.5% below entry | Captures the larger-drift days. |
| `time_stop` | 15:15 IST same-session (MIS hold) | Day-trade only. No overnight. |
| `latch` | One fire per (symbol, T+0..T+5 window) | Avoid stacking on the same forced-unwind event. |
| `event_filter_exclusions` | Same-session earnings / results / corp-action / circuit-band-hit on prior day | Forced-unwind thesis must dominate the price action, not fundamentals. |

These cells are intentionally CONSERVATIVE (tight universe, single-bar entry, intraday-only). The objective at sanity is to test whether the *mechanism* prints — not to optimize cells.

## Falsification thresholds (pre-registered)

Per roadmap §"Validation Methodology (Post-SEBI Aware)":

| Phase | Window | Min sample (n) | Pass criterion | Auto-retire if |
|---|---|---|---|---|
| **Discovery** | 2025-10-01 .. 2025-12-31 (clean post-SEBI, pre-war) | n ≥ 30 trades | NET PF ≥ 1.30 AND WR ≥ 38% | n < 20 OR PF < 1.10 |
| **OOS (pre-war)** | 2026-01-01 .. 2026-02-28 | n ≥ 20 | NET PF ≥ 1.20 | PF < 1.00 |
| **OOS (war window)** | 2026-03-01 .. 2026-04-30 (war volatility tailwind) | n ≥ 20 | NET PF ≥ 1.00 (war is tailwind; do NOT double-count) | PF < 0.90 |
| **Combined OOS** | 2026-01-01 .. 2026-04-30 | n ≥ 40 | NET PF ≥ 1.15 | combined PF < 1.05 |
| **Holdout** | 2026-05-14 onwards via paper-trade | 60-day forward | NET PF ≥ 1.10 forward | drawdown > 8% AUM |

**Hard kill criteria (any one triggers retirement, no debate):**
1. Sample too thin: total trades across Discovery + OOS combined < 50 → can't statistically validate; retire.
2. Direction asymmetry collapses: if the same cells run LONG also produce PF > 1.10, the signal is volatility-feeding-both-sides, not speculator-unwind-driven; mechanism falsified, retire.
3. Win-skew failure: median trade > 0 but mean trade < 0 (or vice versa with the wrong sign) for the qualifying group; not a tradeable distribution.
4. Sample concentration: > 60% of events in any single MWPL recalc quarter (i.e., one-off event, not repeatable mechanism).
5. War-window PF carries the combined PF: if war-window PF > 1.40 AND pre-war OOS PF < 1.05, the edge is volatility-driven not mechanism-driven, retire.

## Data requirements

### Required new data (NOT in repo as of 2026-05-14)

| Data | Source | Format / Endpoint | Granularity | Backfill cost |
|---|---|---|---|---|
| **Per-stock MWPL value, current+history** | NSE archives `nseclearing.in/clearing-settlement/equity-derivatives` + NSE circulars listing quarterly recalc values | XLS / PDF circular per recalc + daily MWPL XLS via `https://nsearchives.nseindia.com/content/nsccl/fao_security_in_ban_for_the_day_<DDMMYYYY>.csv` (ban-list publishes MWPL inline) | Per-symbol, per-day | ~2 days engineering. Recalc-event values from quarterly circulars (~4 PDFs/yr); daily MWPL via ban-list CSV is already a known endpoint. |
| **Per-stock FutEq Open Interest, daily** | NSE F&O UDiFF bhavcopy `https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_<YYYYMMDD>_F_0000.csv.gz` (post 2024-07-08 format) | CSV.gz daily zip, contract-level OI rows; aggregate to stock-level FutEq via delta sum | Per-symbol, per-day (EOD) | ~2 days engineering. Same pattern as existing `tools/delivery_pct/fetch_delivery.py`; UDiFF parser is straightforward. |
| **MWPL recalc effective dates** | NSE circulars + SEBI circular ref `SEBI/HO/MRD/TPD-1/P/CIR/2025/79` | Scrape circular index for "MWPL revision" subjects | Per-event date | < 0.5 day. Few events per year, manual is acceptable. |
| **Optional: pre-Oct-2025 MWPL** (for "regime-break audit" comparison) | NSE archives | XLS | N/A | Skip — pre-rule data not directly comparable per roadmap §1. |

### Data already in repo and reusable

| Data | Path | Use |
|---|---|---|
| 5m equity feathers | `backtest-cache-download/.../*_5minutes.feather` | Intraday entry/exit pricing |
| Daily delivery_pct (NSE bhavcopy MTO + Pd) | `data/delivery_pct/delivery_history.parquet` (built by `tools/delivery_pct/fetch_delivery.py`) | Optional secondary filter — high-delivery names less likely to be retail-FOMO unwinds |
| `nse_all.json` cap-segment + MIS-leverage | repo root | Universe filter (cap_segment via `services.symbol_metadata`) |
| ASM/GSM history | `data/asm_gsm_history/asm_gsm_events.parquet` | Exclude ASM Stage-IV+ names (already restricted-trade) |
| Earnings calendar | `data/earnings_calendar/` | Exclude same-day earnings events |
| SEBI rule changes | `data/sebi_calendar/rule_changes.csv` | Mandatory pre-flight via `services.regime_break_detector` |

### Sample availability estimate (the load-bearing concern)

- **MWPL recalc events in our research window (Oct 2025 – Apr 2026):**
  - Q4-2025 recalc effective ~2025-10-01 (already counts as the rule-launch event)
  - Q1-2026 recalc effective ~2026-01-01
  - Q2-2026 recalc effective ~2026-04-01
  - **Total: 3 recalc cliffs across Discovery + OOS.**
- **Qualifying stocks per recalc:** Estimating from public commentary at the Oct 2025 launch (BusinessToday, 5paisa coverage), 15-30 single-stock F&O names saw materially binding MWPL cuts at the formula change (mostly mid-caps with thin ADDV). Subsequent quarterly refreshes are routine and likely produce 5-15 binding names per recalc.
- **Events × stocks × entry-window-days:** 3 events × ~15 binding stocks × 5-day window × 1 fire per (symbol, event) = **~225 raw events**. After cap-segment + ASM/earnings exclusion + 10:30-confirmation filter, expect **40-80 actual entries** across Oct 2025 – Apr 2026.
- **This is thin.** It puts the candidate at the borderline of the `n >= 50` Discovery+OOS hard-floor. Half the events (Oct-2025 cohort) are the *one-time launch shock* — they may behave differently from steady-state quarterly recalcs and bias the discovery distribution. The Q1-2026 and Q2-2026 cohorts are the "true" steady-state events.

## Implementation sketch

### Universe filter (pre-event)

```
For each MWPL recalc effective date D:
    candidate_universe = nse_fo_eligible(D)
                         & cap_segment in {mid_cap, small_cap}
                         & not ASM Stage-IV+ (from asm_gsm_events)
                         & not in F&O ban (D-1)
                         & no scheduled earnings in [D, D+5]
    For each symbol S in candidate_universe:
        oi_ratio = futeq_oi(S, D-1) / mwpl_new(S, D)
        if oi_ratio >= 0.85:
            qualified.add((S, D, D+5))   # window [D .. D+5]
```

### Detector signal (per session, per symbol within window)

```
At 10:30 IST on each session t in [D, D+5]:
    if (S, t) qualified by event window AND not already fired:
        last_bar = 5m bar ending 10:30
        if last_bar.close < last_bar.open  AND  last_bar.close < prior_day_close:
            FIRE short
            latch (S, event_window)  # one fire per event
```

### Plan (entry / stop / target)

- **Entry price**: 10:30 5m bar close
- **Direction**: SHORT
- **Hard SL**: max(prior-session high + 0.5%, entry × 1.015) — fixed at entry
- **T1**: entry × 0.99 (50% qty)
- **T2**: entry × 0.975 (50% qty)
- **Time stop**: 15:15 IST same session
- **Latch key**: `(symbol, recalc_event_id)` — prevents re-entry across the entire 5-day window after one fire

### Sanity script outline (NOT yet written; only after APPROVE)

`tools/research/post_sebi/sanity_mwpl_recalc_forced_rebalance_fade.py` — mirrors the pattern of `tools/sub9_research/sanity_circuit_t1_fade_short.py`. Inputs:

1. List of 3 recalc effective dates (Oct-2025, Jan-2026, Apr-2026) — supplied as a constant.
2. Per-event qualifying-stock list (`futeq_oi / mwpl_new >= 0.85`) — produced by a small standalone enrichment script that reads UDiFF bhavcopy + MWPL ban-list CSV.
3. 5m feathers for the qualified stocks across each 5-day event window.

Outputs:
- Per-trade CSV at `reports/post_sebi/mwpl_recalc_trades.csv`
- Aggregate PF / WR / NET PnL with Indian fee model (NOTE: must use post-Apr-1-2026 STT for Q2 event, pre-STT-hike for Oct-2025 and Jan-2026 events — the fee model needs an `is_post_apr_2026` switch keyed off trade date).

### Detector implementation (only post-sanity-pass)

- `tools/mwpl_data/fetch_mwpl_history.py` — quarterly recalc + daily MWPL parquet builder
- `tools/fno_oi_data/fetch_futeq_oi.py` — daily FutEq OI parquet builder from UDiFF bhavcopy
- `services/mwpl_event_enricher.py` — produces per-symbol qualifying-event list
- `structures/mwpl_recalc_forced_rebalance_fade_structure.py` — detector
- `plan_short_strategy` extension for plan args

### Pre-flight (mandatory)

```python
from services.regime_break_detector import check_window

check_window(
    strategy_name="mwpl_recalc_forced_rebalance_fade",
    depends_on=["MWPL", "single_stock_FO", "F&O_speculation"],
    window_label="Discovery",
    start=date(2025, 10, 1),
    end=date(2025, 12, 31),
)
# This will FLAG the 2025-10-01 critical rule change which is intentional —
# the setup REQUIRES being on the post-rule side. Acceptable per roadmap:
# "rule_creating sensitivity means the rule LAUNCH is the START of the
# Discovery window, not a break within it." Need a `start_on_rule_launch=True`
# bypass in the detector (one-time exception coded in the script).
```

## Risks / open questions

### Mechanism-level risks
1. **The mechanism may be priced in by the announcement date (2025-05-29), not the effective date (2025-10-01).** Pros knew 4 months in advance which stocks would be MWPL-binding. The unwind may have happened in Aug-Sep 2025 (pre-Discovery window). If so, by Oct 1 the OI ratios are already below 0.85 and there's nothing left to fade.
2. **Stocks at OI/MWPL ≥ 0.85 are exactly the names that enter the F&O ban (≥ 0.95)**, which has its OWN well-known price-impact literature (already covered by Candidate 2 in the roadmap). Risk: this setup is just a slow-motion version of Candidate 2's "F&O ban entry fade" and lacks independent edge.
3. **Forced unwind may produce upward price impact, not downward.** If speculator inventory is net SHORT (high SI as in stressed names), forced position cut means SHORT COVERING → price UP, not down. The brief assumes net-long speculator positioning. Need to verify per-symbol long-vs-short OI balance before fading short.
4. **War-window contamination.** Q1 2026 recalc effective ~2026-01-01 is INSIDE the war-volatility window. War-period elevated vol may overwhelm the structural MWPL-unwind signal. Roadmap §"Hard Learnings 3" warns that war is a tailwind to fade-shorts — could falsely flatter results. The pre-war-split falsifier (above) addresses this.

### Sample-size risks
5. **n = 40-80 across the full research window** is thin. Statistical power for PF ≥ 1.20 at α=0.05 with realistic variance requires n > 100 typically. The roadmap accepts thinner samples (n ≥ 50 explicit) but acknowledges in §"Open Questions" this is a known limitation.
6. **Only 2 truly "steady-state" recalc events** in the window (Jan-2026, Apr-2026) once we discount the one-time Oct-2025 launch shock. Two events is below any reasonable statistical threshold for the "quarterly recalc" hypothesis specifically.

### Data risks
7. **MWPL data quality.** NSE publishes MWPL via the daily ban-list CSV but the historical archive of *quarterly recalc events* needs reconstruction from circular PDFs. Risk: 1-2 of the 3 recalc effective dates may be ambiguous or differ by a few days from the round-quarter date.
8. **UDiFF FutEq OI schema instability.** The post-July-2024 UDiFF format is relatively new; column names and aggregation rules for delta-adjusted OI are not as battle-tested as legacy formats. Risk of silent misalignment between contract-level OI sum and the FutEq-OI value NSE publishes for MWPL utilization. Mitigation: cross-check our computed FutEq OI against the ban-list CSV's published utilization % at several spot-check dates before running sanity.

### Open questions for user before APPROVE
- **Q1:** Do we want to scope-include the Oct-2025 launch shock as an event, or treat it as out-of-scope and use only the steady-state quarterly recalcs (Jan-2026, Apr-2026)? Including it gives more sample; excluding it gives cleaner mechanism test. Recommend INCLUDING with a "first-event" flag in the trade log so we can split-test later.
- **Q2:** Is `current_OI / new_MWPL >= 0.85` the right pre-registered threshold? The roadmap proposed > 0.85. F&O-ban threshold is 0.95. A more aggressive 0.90 threshold would give fewer but cleaner events. Locking at 0.85 (roadmap default) for falsifiability — but worth user confirmation.
- **Q3:** Should the sanity also produce LONG-side numbers (control test) to verify direction asymmetry per hard-kill #2?

## References

### SEBI / NSE primary sources
- SEBI circular **SEBI/HO/MRD/TPD-1/P/CIR/2025/79** (announced 2025-05-29, effective 2025-10-01): new MWPL formula, single-stock position limits, FutEq-OI methodology. Source: <https://www.5paisa.com/news/sebi-implements-stricter-fo-rules-from-october-1-to-strengthen-market-stability>
- NSE Position Limits portal: <https://www.nseindia.com/static/position-limit>
- NSE MWPL / Sec-Ban portal: <https://www.nseindia.com/products-services/equity-derivatives-risk-management-sec-ban>
- NSE daily MWPL utilization (CSV per day): `https://nsearchives.nseindia.com/content/nsccl/fao_security_in_ban_for_the_day_<DDMMYYYY>.csv`
- NSE F&O UDiFF bhavcopy: `https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_<YYYYMMDD>_F_0000.csv.gz` (post 2024-07-08 format)
- NSE Clearing UDiFF format reference: <https://www.nseclearing.in/udiff>

### Industry / explainer sources (mechanism)
- BusinessToday — *Big F&O shake up: SEBI introduces FutEq OI, tweaks ban rules, position limits* (2025-05-29): <https://www.businesstoday.in/markets/story/sebi-equity-fo-rules-futeq-method-mwpl-risk-management-478321-2025-05-29>
- Business Standard — *F&O trading faces reset as Sebi revisits open interest formula*: <https://www.business-standard.com/markets/news/market-regulator-sebi-proposes-overhaul-of-open-interest-calculation-125022500885_1.html>
- Outlook Business — *SEBI Shakes Up Derivatives Game With New Rules*: <https://www.outlookbusiness.com/markets/sebi-shakes-up-derivatives-game-with-new-rules-for-measuring-open-interest-mwpl>
- Bajaj Broking — *SEBI Notifies New F&O Rules With Higher Index Limits and Risk Measures*: <https://www.bajajbroking.in/blog/sebi-notifies-new-f-and-o-rules-with-higher-index-limits-and-risk-measures>
- Kotak Neo — *SEBI's New F&O Rules Explained: Delta-Based OI, New MWPL*: <https://www.kotakneo.com/bulletins/sebi-new-f-o-rules-explained-delta-based-oi-new-mwpl-what-it-means-for-your-trading/>
- Zerodha Z-Connect — *A short brief on the new SEBI measures for the F&O space*: <https://zerodha.com/z-connect/kite/a-short-brief-on-the-new-sebi-measures-for-the-fo-space>

### Academic / empirical literature (price-limit & forced-unwind analogues)
NOTE: Direct peer-reviewed research on *MWPL-recalc-forced-unwind price impact* in Indian markets is **not available** as of 2026-05-14 (the rule itself is only 7 months old). The closest analogues:
- Chen, Petukhov, Wang — *Magnet Effect of Price Limits* (MIT working paper): <https://web.mit.edu/wangj/www/pap/ChenPetukhovWang18.pdf> — establishes that approaching a binding limit causes accelerated flow toward the limit; the obverse (forced exits from a binding cap) has the symmetric reverse implication that this brief leverages.
- Guo et al. — *Journal of International Financial Markets* (2023): <https://www.sciencedirect.com/science/article/abs/pii/S1386418123000381> — Indian-equity natural-experiment on price bands; documents that limits *delay* but don't eliminate price discovery; volatility migrates to subsequent sessions (supports the T+0 .. T+5 entry window).
- Sehgal et al. — *Pacific-Basin Finance Journal* (2024): <https://www.sciencedirect.com/science/article/abs/pii/S0927538X23002640> — Indian-momentum/reversal evidence; upper-circuit hits show next-day continuation in low-float names while lower-circuit hits show reclaim in liquid F&O names. The MWPL-forced-unwind setup is most consistent with the high-OI mid-cap segment, NOT the liquid-F&O reclaim group.
- Bessembinder & Seguin — *Journal of Financial and Quantitative Analysis* (1992) — classic study of speculator/hedger position changes and futures volatility; provides the macro mechanism for "forced position cuts produce next-period drift."

**No direct empirical Indian-market study of MWPL forced-unwind price impact was found in the web search** — the literature gap is itself a finding: this setup may be a genuine post-rule edge if it works, or a data-mined artifact if it doesn't. The pre-registered falsifiers above are the defense against the latter.

### Cross-references in this repo
- `data/sebi_calendar/rule_changes.csv` row `2025-10-01 critical MWPL` — source for `regime_break_detector` pre-flight
- `services/delivery_pct_enrichment.py` — pattern for daily-EOD-data parquet enrichment
- `tools/delivery_pct/fetch_delivery.py` — pattern for NSE-archive scraper
- `tools/sub9_research/sanity_circuit_t1_fade_short.py` — pattern for sanity-check script structure
- `services/regime_break_detector.py` — mandatory pre-flight gate
- `specs/2026-05-14-research-post-sebi-edges.md` — parent roadmap (Candidate 1)
- `specs/2026-05-08-sub-project-9-brief-nse_delivery_pct_anomaly.md` — adjacent NSE-disclosure mechanic brief (delivery %); useful template for SEBI-anchored briefs

---

## Decision required (per sub-9 §3.3 template)

- [ ] APPROVED — proceed to data-engineering for MWPL + FutEq-OI parquet, then sanity script
- [ ] REJECTED — reason (sample too thin, mechanism overlaps Candidate 2, etc.)
- [ ] REVISE — specify what's missing / wrong

**Verdict at draft time (author's recommendation): MARGINAL.** The mechanism is theoretically sound and uniquely post-SEBI-Oct-2025, satisfying the `rule_creating` test. However the achievable sample (n ≈ 40-80 across 7 months) is at or below statistical-power floor, and the mechanism may already be captured more cleanly by Candidate 2 (intraday-ban-entry fade) which has a much richer event stream (200+ events vs ~3 recalc cliffs). Recommend prioritizing Candidate 2 first; revisit this candidate only if Candidate 2 falsifies cleanly and we have residual research capacity. If proceeding, the strict pre-registered falsifiers above are the protection against the small-sample data-mining risk.
