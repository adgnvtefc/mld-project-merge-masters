# 20261118-2130_analysis_scale_hz

Hypothesis: Whether failed merges are mostly miscalibrated (fixable by output temperature) or have shrunken hidden states.

Status: done

## Outcome

Merges are not merely miscalibrated: the best output temperature is 1.0 for every model except the overshooting mix, where it recovers only 0.22 of 2.18.

## Observations

- Pool-val: generalist 3.209, average of experts 3.591, learned-lambda 3.200, ta lam_e 0.1 lam_g 1.0 5.392 (5.173 at T 0.8).
- Hidden-state RMS per block matches the generalist for the average and the learned mix; the overshooting mix's RMS is about 40% lower in every block (0.04-0.12 vs 0.06-0.19).
- Run before commit d725d3f, so its pool-val windows were the first n in a row. For math those were 79% one 104K-token GMAT reading-comprehension guide, where the math expert is barely better than the generalist. Math columns here describe that book, not math; code, medicine and history are unaffected.
