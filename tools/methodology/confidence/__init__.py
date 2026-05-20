"""Confidence framework for per-trade strategy validation.

Three independent statistical pillars, each backed by primary sources:

1. Bootstrap BCa CI on PF / expectancy / win rate (Efron & Tibshirani 1993)
2. Per-regime decomposition (Lopez de Prado tactical paradigm 2019)
3. Selection-bias correction via Harvey-Liu Sharpe haircut + ONC clustering
   (Harvey & Liu 2015; Lopez de Prado & Lewis 2019)

The framework outputs INTERVALS, not binary ship/no-ship verdicts. Researcher
applies judgment to the intervals. Thresholds are NOT baked in (per
_per_trade_validation_research.md: literature is silent on ship thresholds
for per-trade discrete strategies).

See: docs/methodology_confidence_framework.md (to be written)
"""
