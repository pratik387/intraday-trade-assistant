# Multiday capitulation longs: results-day skip + bulk-deal size tilt — design

**Status:** design, not built. User asked for research before any filter (2026-10-06) and then for a design of both findings. Nothing here is deployed.

## 1. What the evidence says (research baselines, 9,012 trades 2023-01..2026-04, net Rs at research size)

Source scripts: scratchpad `md_news_direction.py`, data `news/announcements.parquet` (NSE corporate announcements, 675 ledger symbols, 166,421 filings), `data/earnings_calendar/earnings_events.parquet`, `data/bulk_deals_cache/nse_bulk_deals_2023_2026.parquet`. Memory: `project_multiday_news_direction`.

| finding | trades | per year vs same-year quiet baseline | holds in |
|---|---|---|---|
| results-cluster drop (results filing / investor call / board meeting / calendar) | 28% | -1.6k / -6.0k / -5.8k / +2.6k | 3 of 4 years, all 4 setups |
| ...of which ON the reaction day or the day before | 8% | mean -1.0k..-3.3k, win 33-35% (2023-25) | |
| ...AND the drop is shallow (> -20% vs 50-day SMA) = **rule R3** | 7% | skip delta +169k / +648k / +723k / +3k | **every year, every setup** |
| bulk deal (either side) in the window | 1.5% | +12.1k / +2.3k / +4.7k / +18.4k | every year, every setup |
| pledge, resignation, fund raise, order win, rating, clarification, surveillance | each < 2% | sign flips | none |

Deep drops (< -20%) on results still revert and are kept. Depth without the news test fails (2 years worse). Results trades in 2023 and 2026H1 were positive only through 10 trades each.

## 2. The two rules

### R3 — results-day skip
Skip a multiday BUY signal for symbol S on signal day T when **all** of:
1. S has a results event with reaction day R such that T == R or T == R - 1 business day
   (R from realised filings) **or** S has a scheduled results board meeting on T or T+1 (from the event calendar; this is the "day before the pre-results fall" leg);
2. close(T) / SMA50(T) - 1 > -20% (the drop has not gone deep);
3. the symbol is NOT bulk-deal tagged in [T-1, T+1] (bulk overrides, see §1).

Config (per multiday setup, no defaults, missing key = startup error):
```
"news_gate": {
  "enabled": true,
  "skip_results_reaction_day": true,
  "results_days_before": 1,          # T == R-1 counts
  "results_days_after": 0,           # T == R+1 does NOT count (near-neutral in data)
  "shallow_drop_sma50_pct": -20.0,   # skip only if dist > this
  "bulk_overrides_skip": true
}
```

### Bulk tilt — size multiplier
When S is bulk-deal tagged (NSE bulk deals, buy or sell side, in [T-1, T+1]), multiply the per-position risk budget in PHASE 2 sizing by `bulk_risk_multiplier` (2.0), still capped by `risk_budget.max_notional_inr` and the cluster caps. Config under `multi_day_portfolio.risk_budget`:
```
"bulk_deal_risk_multiplier": 2.0,
"bulk_deal_window_bdays": 1
```
No change to selection order: evidence is on size, and the cluster caps (3+2 new/day) already bind on the composite score.

## 3. Data feeds (three; all reuse `services/event_feeds.py` contract: `refresh_module --start --end --sleep-secs`, parquet with a date column, staleness check)

| feed | source | refresh | rows/day | status |
|---|---|---|---|---|
| `data/news/nse_announcements.parquet` | `GET nseindia.com/api/corporate-announcements?index=equities&from_date&to_date` with NO symbol = whole market (tested: 2 days -> 1,070 rows / 804 symbols) | one call per day of lookback, `tools/news_feed/fetch_nse_announcements.py` (new; cookie bootstrap on nseindia.com, plain `requests` works from PC and VM) | ~500 | new |
| `data/news/nse_event_calendar.parquet` | `GET nseindia.com/api/event-calendar?index=equities&from_date&to_date` (tested: 154 scheduled meetings Oct 6-31, purpose "Financial Results") | same module, `--forward-days 10` | ~10 | new |
| `data/earnings_calendar/earnings_events.parquet` | existing `tools.earnings_calendar.fetch_earnings` | existing | | **gap: Jul-Aug 2026 has 512 events vs ~1,500 expected; fix scrape before relying on it forward.** The announcements feed with `desc` in {Financial Results, Outcome of Board Meeting} is the redundant realised leg. |
| `data/bulk_deals/nse_bulk_deals.parquet` | NSE bulk-deals API, same Akamai-cookie scraper as `tools/block_deal_calendar/fetch_block_deals.py` (`optionType=bulk_deals`) | extend that module, `--deal-type bulk` | ~30 | extend; current cache ends 2026-05-24 |

Each multiday setup declares `event_feed` blocks for the three feeds (same shape as earnings_downshock's). **The multiday cron (`main.py --mode multi_day --action entry`) does not call `_refresh_event_feeds()` today and the multiday checkout on the VM has no `data/earnings_calendar/` at all.** `run_eod(phase='entries')` must call `refresh_and_validate_all` for its setups first; a stale or missing feed FAILS OPEN for the skip (trade as today, log `NEWS_GATE | feed stale | gate off`) and fails CLOSED for the tilt (multiplier 1.0). Never block entries on a scrape.

## 4. Tagging module — `services/news_tags.py` (pure, testable)
```
tag_symbol_day(symbol, day, *, announcements, event_calendar, earnings, bulk, cfg) -> NewsTags
NewsTags = {results_reaction: bool, results_scheduled_tomorrow: bool, bulk_deal: bool, categories: set[str]}
```
- Category regexes = the ones in the research script (results | investor/analyst | board meeting ...). Keep them in config under `news_gate.results_regex` so research and production use one list.
- Reaction day R for a realised filing: filing timestamp before 15:30 -> R = that day; after 15:30 -> R = next trading day (same convention as `earnings_reaction_enrichment`).
- Business-day arithmetic via `utils/time_util` and the NSE holiday calendar; all timestamps IST-naive.

## 5. Where it hooks (`services/execution/mtf_capitulation_handlers.py`)
1. `_rank_basket_for_setup`: after `CrossSectionalRanker.rank`, add `dist_sma50_pct` to every candidate. The ranker's panel must be >= 60 trading days deep (zscore uses 20, low52 uses `low_lookback_days`; **verify the shared provider window at build time**; if shallower, compute SMA50 from `fetch_daily_window`).
2. New step between basket build and `selector.select`: `_apply_news_gate(baskets, today, feeds, cfg)` removes R3 names from every basket (so the cluster slot goes to the next name) and attaches `news.bulk_deal` to the rest. One log line per removal: `NEWS_GATE | SKIP | sym | reason=results_reaction_day dist_sma50=-7.2% | owner=zscore`.
3. PHASE 2 sizing: `risk_inr *= bulk_deal_risk_multiplier` when tagged; log `NEWS_TILT | sym | bulk_deal x2.0 | notional a -> b`.
4. Selection diagnostics jsonl (`logs/multiday_selection.jsonl`) gains `news_tags` and `gated` fields so the skip is auditable per day without a rerun.

## 6. Tests
- `tests/services/test_news_tags.py`: truth table for R3 (reaction day / day before / day after / deep drop / bulk override / scheduled tomorrow), after-15:30 filing rolls to next day, stale-feed fail-open.
- Handler tests: gated name is absent from baskets and the slot is taken by the next composite name; bulk name sized x2 but capped at max_notional; config keys read with `[]`, missing key raises.
- Replay gate (per `docs/setup_lifecycle.md`): run the dry-run replay harness over 2026-01..2026-09 with the gate on and off; the on-minus-off delta must match the research script's R3 delta for the overlapping window within tolerance (parity of the tagging, not of the P&L).

## 7. Rollout
1. Feeds + tagging module + tests, deployed to the multiday checkout; gate `enabled: false`, tilt multiplier 1.0 -> **observe only** for 10 sessions: every entry logs its tags and what the gate WOULD have done.
2. Compare the observed would-skip set with the research rate (7% of signals; 33-35% win on the skipped).
3. Enable the skip in paper. Tilt stays 1.0 until the skip has 20 sessions.
4. Enable the tilt in paper. Live only after both have a full results season (Jan-Feb 2027) behind them.
Kill switch: `news_gate.enabled=false` restores today's behaviour exactly.

## 8. Open items before build
- Earnings scraper Jul-Aug 2026 gap (root cause in `fetch_earnings.py`).
- Panel depth for SMA50 (see §5.1).
- Bulk-deal scraper needs `curl_cffi` on the VM (block-deal tool's dependency); confirm installed in the shared venv.
- The research tagging used the filing DATE, not the 15:30 cutoff; production must use the cutoff. Re-run the research script with the cutoff before the replay gate so the comparison is like-for-like.
