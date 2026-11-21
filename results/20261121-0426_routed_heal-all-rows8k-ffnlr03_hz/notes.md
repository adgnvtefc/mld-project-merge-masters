# 20261121-0426_routed_heal-all-rows8k-ffnlr03_hz

Hypothesis: With the experts' FFNs at 0.3x the learning rate, all trained no longer loses ground at the peak learning rate (+0.016 at step 200 before) and beats 2.9269 on dev.

Status: done

Parent: 20261120-1753_routed_heal-all-rows8k-lr1e-4_hz

## Outcome

Supported. Dev 2.9242, the team's best: -0.0027 vs FFNs at the full rate (95% range -0.0033 to -0.0020). 61% of the gap.

## Observations

- Same as 20261120-1753 (Stacks + own rows, all trained, lr 1e-4, 200 warmup, 800 steps) except the experts' FFNs train at 0.3x the learning rate.
- Dev: math 2.813, code 2.835, medicine 2.721, history 3.328.
- Valid: 299,994,524 parameters, 148.6M active, causal. One seed.
- Held-out with the router: 2.958 (step 0), 2.934, 2.933 (step 200), then down to 2.900 at 800. The rise at the peak learning rate seen with FFNs at the full rate (2.935 to 2.951 at step 200) is gone.
