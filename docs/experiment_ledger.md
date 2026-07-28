# Experiment Ledger

**Purpose:** honest selection-bias accounting. Every evaluation of a candidate/variant against
OOS, Holdout, or post-freeze data burns statistical validity of that window. The Harvey-Liu
haircut M in confidence cards must count *all* such evaluations, not just the survivors. Before
this ledger existed, M was set to "ship-eligible count" (4-8) while the true count across a year
of research was in the hundreds — a ~100× understatement that let noise-floor candidates ship
(retrospective 2026-07-27; all three 2026-05-14 shipped intraday setups later died on forward data).

**The rule (lifecycle amendment A2):** any run that computes a performance statistic (PF, WR,
expectancy, Sharpe — anything used to judge the candidate) on OOS, Holdout, or any post-freeze
window appends one line to `docs/experiment_ledger.jsonl` **in the same session**, regardless of
outcome. Discovery-only exploration does not need logging (Discovery is development data by
definition). If in doubt, log it.

## Schema (`experiment_ledger.jsonl`, one JSON object per line)

| field | type | meaning |
|---|---|---|
| `date` | `YYYY-MM-DD` (IST) | when the evaluation was run |
| `setup` | string | setup/candidate name (use the brief's name; `<name>__vN` for variants) |
| `variant` | string | short human label of what distinguishes this variant (cell, geometry, filter) |
| `stage` | string | lifecycle stage at evaluation time (`phase5_oos`, `phase5_holdout`, `post_freeze`, `revival`, `adhoc`) |
| `windows` | list of strings | which evaluation windows were touched (`oos_2025`, `holdout_oct25_apr26`, `post_freeze:<YYYY-MM-DD>`) |
| `verdict` | string | `pass` / `kill` / `marginal` |
| `evidence` | string | path to the trades CSV / report / script that produced the number |
| `notes` | string | optional; anything a future M-auditor needs |

## How M is derived

M for a confidence card = count of ledger lines whose `windows` overlap the card's decisive window,
grouped per setup-family where variants are near-duplicates (ONC clustering may reduce raw count to
effective M — but the *raw* count is what goes in the ledger; reduction happens transparently in the
card with both numbers reported).

## Historical baseline (pre-ledger)

The first line of the JSONL is a `_baseline` marker, not an experiment. Everything evaluated before
2026-07-27 against `oos_2025` / `holdout_oct25_apr26` is undercounted here: ~20 retired setups
(`docs/retired_setups.md`), ~10 shipped/paper setups, plus per-setup cell/R-sweeps and the root-level
`_tmp_*` variant scripts — conservatively **M ≥ 200** for those windows. This is why lifecycle
amendment A1 demotes them to development data for new candidates.
