# 20261118-2014_merging_regmean_hz

Hypothesis: RegMean is the best closed-form merge and beats the generalist (3.156).

Status: done | dev mean 3.4446

## Outcome

Best alpha 0.5 (pool-val 3.4805); dev 3.4446, worse than the generalist by 0.289. Does not support the hypothesis.

## Observations

- Pool-val loss rose with alpha: 0.5 -> 3.48, 0.7 -> 4.06, 0.9 -> 6.73, 1.0 -> 9.77. 0.5 is the lowest alpha in the grid, so smaller values were not tried.
- Grams from 1M pool-train tokens per source (each expert on its domain, the generalist on an equal mix); non-Linear tensors are the plain mean of the five models.

## Caveat (added 2026-11-18)

Settings and pool-val numbers in this run used the first n pool-val windows in a row. For math these were 79% one GMAT reading-comprehension guide, where the math expert is barely better than the generalist, so math pulled every choice toward the generalist. Dev scores are correct. Fixed in commit d725d3f; this run has not been re-run. See error_analysis.md.

Re-run as 20261119-0512_merging_regmean_hz after the fix (2026-11-19): same choice and same dev score, so the result above stands.
