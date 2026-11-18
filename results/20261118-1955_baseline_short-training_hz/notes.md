# 20261118-1955_baseline_short-training_hz

Hypothesis: Reference: the starter's generalist + extra layer, 200 steps (about 3.16).

Status: done | dev mean 3.1629

## Outcome

Dev 3.1629, about the handout's 3.16, and 0.0074 worse than the generalist (95% CI [+0.0063, +0.0085]).

## Observations

- 200 steps x 32 x 2,048 tokens (13.1M tokens) in 365 s, 0.10 A10G-hours. Improves math and code slightly, worsens medicine (+0.048) and history (+0.024) vs the generalist.
