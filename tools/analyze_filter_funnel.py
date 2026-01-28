#!/usr/bin/env python
"""
Analyze the filtering funnel to understand why big movers don't reach events.jsonl.

Flow:
1. nse_all.json (1,992 stocks) - stock universe
2. scanning.jsonl - scanner stage (momentum/mean reversion scoring)
3. screening.jsonl - screener stage (ORB levels, filters)
4. events.jsonl - decision gate stage (structure + regime + news)

Find: Where do big movers get filtered out?
