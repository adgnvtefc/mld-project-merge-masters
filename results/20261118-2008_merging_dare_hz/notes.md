# 20261118-2008_merging_dare_hz

Hypothesis: DARE (drop 90%, rescale) before task arithmetic reduces interference and beats plain task arithmetic.

Status: done | dev mean 3.4684

## Outcome

The best pool-val point was lam_e = 0, lam_g = 0, which is the seed itself; dev 3.4684. Does not support the hypothesis.

## Observations

- DARE (drop 90%, rescale x10) made every non-zero point much worse: the DARE'd generalist alone at lam_g 1.0 scored 20.19 on pool-val, vs 3.19 without DARE. Larger coefficients were monotonically worse (up to 36.1).
