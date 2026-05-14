# SEBI / NSE Regulatory Calendar

Manually maintained log of dated SEBI/NSE rule changes that may affect strategy
edge. The gauntlet methodology should NEVER span an `effective_date` row in
this file without explicit per-strategy attribution.

## Why this exists

Our Sep 2025 gauntlet failed to detect that SEBI's Oct 1, 2025 F&O rule changes
(MWPL formula + single-stock position limits) had been announced May 29, 2025
and would land inside our Holdout window. We discovered the regime break only
when `delivery_pct_anomaly_short` collapsed in production replay: OOS PF 1.245
→ Holdout PF 0.879, sharp Oct 2025 inflection. Root cause: the Holdout straddled
a regulatory cutover that broke the setup's mechanism (F&O speculator inventory
unwind).

This calendar exists so future gauntlet runs auto-flag any validation window
spanning a relevant rule change.

## Schema (rule_changes.csv)

| Column          | Description |
|-----------------|-------------|
| `effective_date`| ISO date when the rule becomes binding. Use this for split detection. |
| `announced_date`| ISO date of the announcing circular. Useful for "did pros know in advance?". |
| `category`      | One of: `margin`, `fno_structure`, `fno_position`, `fno_monitoring`, `settlement`, `surveillance`, `tax`, `eligibility`, `dpr_circuit` |
| `severity`      | `low` / `medium` / `high` / `critical`. Critical = multi-setup mechanism-breaking. |
| `affects`       | Semicolon-separated dependency tags (see below). Match against strategy `depends_on`. |
| `description`   | One-sentence summary of what changed. |
| `circular_ref`  | SEBI/NSE circular number when known. |
| `source_url`    | URL for verification. |

## Dependency tag vocabulary

Use these as `affects` tags AND as `setups.<name>.depends_on` values:

| Tag | Meaning | Setups that should declare it |
|-----|---------|------------------------------|
| `MWPL` | Market Wide Position Limit math | Any setup using F&O speculator inventory |
| `single_stock_FO` | Single-stock F&O availability | Setups trading or fading single-stock F&O flows |
| `single_stock_FO_speculator_unwind` | Pro speculator forced rebalance behavior | delivery_pct_anomaly_short (mechanism) |
| `F&O_speculation` | Generic F&O speculator behavior | Most fade-the-amateur setups |
| `option_premium` | Option pricing structure | Any premium-decay or vol-based setup |
| `intraday_position_limit` | Intraday position cap rules | Any setup running multiple concurrent positions |
| `MIS_leverage` | Broker intraday leverage | Almost everything (PnL scales with leverage) |
| `STT_drag` | STT-driven break-even cost | Fee-sensitive setups (low average win) |
| `cash_settlement` | T+0/T+1/T+2 settlement | Setups holding overnight or T+1 strategies |
| `weekly_expiry` | Weekly option expiry flow | Setups fading expiry-day pin or hedge unwind |
| `index_options` | NIFTY/BANKNIFTY option dynamics | Any index-option strategy |
| `index_derivatives` | Index futures + options | Any index derivative strategy |
| `non_benchmark_indices` | Non-NIFTY/BANKNIFTY indices | Strategies on sector or thematic indices |
| `BankNifty_flow` | BankNifty-specific liquidity | Anything that uses BankNifty as proxy |
| `surveillance_list` | ESM/T2T/ASM/GSM list mechanics | Any setup that trades surveillance-list stocks |
| `intraday_ban` | F&O intraday-ban mechanics | Setups fading F&O ban entries |
| `dpr_circuit` | Daily price range / circuit rules | circuit_t1_fade_short |

## Severity guide

- **critical** — mechanism break. Strategy must be re-validated before re-shipping.
- **high** — material edge shift. PF/WR may move 10-20% in either direction. Re-tune required.
- **medium** — parameter sensitivity. Stop loss, target, or time-stop may need adjustment.
- **low** — cosmetic. Maybe wider universe or marginal noise.

## How the gauntlet should use this

Pseudocode for the regime-break detector:

```python
def gauntlet_regime_check(strategy, discovery_window, oos_window, holdout_window):
    calendar = load_rule_changes()
    deps = set(strategy.depends_on)
    for window_label, window in [("Discovery", discovery_window),
                                  ("OOS", oos_window),
                                  ("Holdout", holdout_window)]:
        for row in calendar:
            if not (deps & set(row.affects.split(";"))):
                continue
            if window.start <= row.effective_date <= window.end:
                if row.severity in ("high", "critical"):
                    raise GauntletRegimeBreakError(
                        f"{window_label} window contains {row.severity} rule "
                        f"change on {row.effective_date} affecting "
                        f"{deps & set(row.affects.split(';'))}: {row.description}"
                    )
```

## Maintenance

This file is **manually maintained**. Update it when:

1. SEBI or NSE issues a circular affecting any tagged dependency.
2. The `effective_date` of any pending change is confirmed.
3. A retired rule is reinstated or amended.

Source of truth: read `sebi.gov.in` circulars and `nseindia.com/regulations/circulars`
monthly. Cross-reference with Zerodha Z-Connect, ICICI Direct, Bajaj Broking blog
for plain-English summaries.

## History (notable entries)

- **2024-10-01** STT hike on F&O (futures 0.0125%→0.02%, options 0.0625%→0.1% on
  premium). High severity — reset fee-sensitive backtests.
- **2024-11-20** Single weekly expiry per exchange. High severity — destroyed
  multi-index weekly-expiry flow patterns; lots of strategies died here.
- **2024-12-24/26** Lot size doubling/tripling (BANKNIFTY 15→30, NIFTY 25→75).
  High severity — material flow shift from F&O retail back to cash segment.
- **2025-02-01** Full option premium upfront. High severity — ended option-
  buying-on-leverage retail strategies.
- **2025-10-01** **Critical**: MWPL formula tightened, single-stock position
  limits cut. Broke `delivery_pct_anomaly_short` (mechanism: F&O speculator
  inventory unwind no longer exists at same magnitude post-rule).
- **2026-04-01** **Critical**: STT another 150% hike on futures, 50% on options.
  All F&O-sensitive setups need re-validation post this date.

## Open items

- Add 2022/2023 rule changes if you need Discovery period before 2023-Q4
- Track NSE quarterly F&O eligibility revisions (added/removed stocks list)
- Track ASM/GSM list changes (weekly cadence — may need separate file)
