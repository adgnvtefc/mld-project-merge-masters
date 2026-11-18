# 20261118-2012_merging_ties_hz

Hypothesis: TIES (top-20% trim, sign election) over the five task vectors beats plain task arithmetic.

Status: done | dev mean 3.7196

## Outcome

Best lam 0.4 (pool-val 3.7472); dev 3.7196, worse than the seed (3.4684) and the generalist. Does not support the hypothesis.

## Observations

- Pool-val loss rose monotonically with lam (0.4: 3.75 ... 1.5: 9.15); the chosen value is the lowest in the grid, so smaller lam was not tried.
- Settings: top 20% kept per tensor, all five task vectors.

## Caveat (added 2026-11-18)

Settings and pool-val numbers in this run used the first n pool-val windows in a row. For math these were 79% one GMAT reading-comprehension guide, where the math expert is barely better than the generalist, so math pulled every choice toward the generalist. Dev scores are correct. Fixed in commit d725d3f; this run has not been re-run. See error_analysis.md.
