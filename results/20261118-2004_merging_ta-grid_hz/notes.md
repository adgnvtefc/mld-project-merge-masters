# 20261118-2004_merging_ta-grid_hz

Hypothesis: Seed-relative task arithmetic including the generalist's task vector, with coefficients picked on pool-val, beats the generalist (3.156).

Status: done | dev mean 3.1555

## Outcome

The best of 28 grid points on pool-val was lam_e = 0, lam_g = 1, which is exactly the generalist; dev 3.1555 (Δ 0.0000). Does not support the hypothesis.

## Observations

- Any expert weight on top of the full generalist vector hurt sharply on pool-val: lam_e 0.1 / lam_g 1.0 -> 5.36, 0.2 / 1.0 -> 11.66. With lam_g 0.75, lam_e 0.1 -> 3.29 (vs 3.27 for lam_e 0).
- The chosen point sits on the grid edge (lam_e at its minimum, lam_g at its maximum); points with 0 < lam_e < 0.1 or lam_g > 1 were not tried.
- Generalist on pool-val (64 windows/domain): 3.1887, vs 3.1555 on dev.
