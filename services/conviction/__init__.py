"""XGBoost-based conviction scorer + top-N selection gate.

Per sub-project #2 design spec (2026-04-21). Ranks Stage-5c-filter-surviving
candidates by predicted R-multiple, selects top 50/day above calibration
threshold. Training data = 74 validation-gate-surviving rules (Discovery
trades matching those rules).
"""
