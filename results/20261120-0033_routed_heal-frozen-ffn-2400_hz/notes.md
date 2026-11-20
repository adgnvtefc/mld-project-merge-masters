# 20261120-0033_routed_heal-frozen-ffn-2400_hz

Hypothesis: Stacks with FFNs frozen keeps improving with 3x the training (2,400 steps, about 90 minutes, same schedule shape) and beats its 800-step dev score of 2.973.

Status: done

Parent: 20261119-2206_routed_heal-frozen-ffn_hz

## Outcome

Supported, but the gain is small. Dev 2.9630, the team's best: 0.0099 better than 800 steps (95% range 0.0080 to 0.0118) for 3x the training. 0.026 better than the best Shared-Attention MoE (0.022 to 0.030); 0.193 better than the generalist, 51% of the gap.

## Observations

- Same as its 800-step parent except --steps 2400 --val-every 200: same start (fingerprint 46ae47f84a72da85), data stream, warmup (30 steps) and peak learning rate; the linear decay is stretched over 2,400 steps. 157.3M training tokens.
- Dev: math 2.849, code 2.878, medicine 2.761, history 3.364. Each domain gained 0.008-0.011 over 800 steps.
- Held-out loss with the router got worse first (3.011 to 3.022 at step 200, with the longer time at peak learning rate), passed its start at step 600, and fell to 2.934 at 2,400. The last 200 steps gained 0.004.
- 86.5 min of training; with the 1.5-min build it fits a 2-hour final recipe. 1.48 A10G-h with dev scoring.
- Valid: 299,995,804 parameters, 148.5M active, causal. One seed.
