# 20261121-0736_routed_heal-all-rows8k-ffnlr03-kd05-1600_hz

Hypothesis: Spending the rest of the 2-hour recipe budget on distillation (1,600 steps instead of 800, about 101 minutes for the whole script) improves on dev 2.8994, as twice the steps did without distillation (-0.0067).

Status: done | dev mean 2.8880

Parent: 20261121-0619_routed_heal-all-rows8k-ffnlr03-kd05_hz

## Outcome

Supported. Dev 2.8880, the team's best: 0.0115 better than the same recipe at 800 steps (2.8994), a bigger gain from doubling the steps than without distillation (0.0067). 71% of the gap from the generalist to the own-expert ceiling.

## Observations

- Same as 20261121-0619 (Stacks + own rows, all trained, lr 1e-4 with 200 warmup, FFNs at 0.3x, distillation 0.5 from each stack's own unpruned expert, same data stream), 1,600 steps instead of 800.
- Dev: math 2.782, code 2.804, medicine 2.686, history 3.279. vs 800 steps: -0.0115 (95% range -0.0122 to -0.0104), better in every domain (-0.010 / -0.010 / -0.011 / -0.015) and on 97-99% of windows. vs 1,600 steps without distillation (20261121-0506): -0.0295. vs generalist: -0.2675, better on 95/97/89/92% of math/code/medicine/history windows.
- Held-out with the router: 2.958, 2.897, 2.880, 2.869, 2.863 (steps 0, 400, 800, 1200, 1600). Still falling at the end, by 0.006 over the last 400 steps.
- Script time: 99.3 minutes (build 2.2 min, training 96.9 min at 18.1K tok/s, plus validation); 1.66 A10G-h with dev scoring. Inside the 2-hour recipe limit with about 20 minutes to spare; 1,800 steps would take about 112 minutes, too tight for a fresh machine.
- Valid: 299,994,524 parameters, 148.6M active, causal. One seed. Weights fingerprint of the untrained build c8125bbd2bbcc065.

