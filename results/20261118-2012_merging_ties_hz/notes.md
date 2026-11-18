# 20261118-2012_merging_ties_hz

Hypothesis: TIES (top-20% trim, sign election) over the five task vectors beats plain task arithmetic.

Status: done | dev mean 3.7196

## Outcome

Best lam 0.4 (pool-val 3.7472); dev 3.7196, worse than the seed (3.4684) and the generalist. Does not support the hypothesis.

## Observations

- Pool-val loss rose monotonically with lam (0.4: 3.75 ... 1.5: 9.15); the chosen value is the lowest in the grid, so smaller lam was not tried.
- Settings: top 20% kept per tensor, all five task vectors.
