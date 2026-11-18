# 20261118-2107_merging_low-rank_hz

Hypothesis: Adding each expert's top-r change from the generalist (no extra parameters) beats the generalist and the learned mix (3.141).

Status: done | dev mean 3.1700

## Outcome

Dev 3.1700, worse than the generalist by 0.0145 (95% CI [+0.0142, +0.0148]) and worse on every domain. Does not support the hypothesis.

## Observations

- Best pool-val point: r 16, lam 0.1, others moved by lam times the experts' mean change -> 3.2041, already above the generalist's 3.1888. Both r and lam are the smallest values in the grid, and lam = 0 (the generalist itself) was not in the grid.
- Larger lam was always worse (for example r 128: lam 0.1 -> 3.22, 0.25 -> 3.36-3.40, 0.5 -> 4.05-4.25).
- Per domain vs the generalist: math +0.0081, code +0.0237, medicine +0.0145, history +0.0117. No extra parameters (125.2M). 204 s, 0.057 A10G-hours.
