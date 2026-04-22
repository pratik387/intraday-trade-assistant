# Conviction model validation report (2025 OOS)

Model: `2026-04-22-universal-xgboost.json`
OOS period: 2025-01-01 to 2025-09-30
Trades evaluated: 73,245

## Results

| Test | Required | Actual | Passed |
|---|---|---|---|
| 1. OOS PF lift (top-50 vs random) | PF_ml > PF_random | PF_ml=8.331, PF_random=6.289 | PASS |
| 2. Calibration monotonicity | >=7/9 transitions non-decreasing | 5/9 | FAIL |
| 3. SHAP stability | >=7/10 features overlap | 10/10 | PASS |
| 4. Per-session Spearman | median rho > 0.05, p < 0.05 | rho=0.018, p=0.0509 | FAIL |

**All passed: NO**
