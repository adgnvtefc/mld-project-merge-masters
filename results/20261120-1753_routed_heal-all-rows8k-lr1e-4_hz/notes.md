# 20261120-1753_routed_heal-all-rows8k-lr1e-4_hz

Hypothesis: At the lower learning rate, training every weight no longer loses ground early (+0.11 at 3e-4), so letting the pruned FFNs adapt makes all trained beat FFNs frozen.

Status: done

Parent: 20261120-0750_routed_heal-frozen-ffn-rows8k_hz

## Outcome

Supported, narrowly. Dev 2.9269, the team's best: 0.0018 better than FFNs frozen at the same learning rate (95% range 0.0012 to 0.0025). At 3e-4, all trained lost to FFNs frozen by 0.019; at 1e-4 it no longer does.

## Observations

- Same as 20261120-0750 (Stacks, rank-32 fix + own rows for the top 8,192 tokens, 800 steps, same data stream) except the learning rate: peak 1e-4 with a 200-step warmup, instead of 3e-4 with 30. Every weight trained.
- Held-out with the router: 2.958, 2.935 (step 100), then up to 2.951 at step 200 as the learning rate reached its 1e-4 peak, then down to 2.903 at 800. So training the FFNs still costs a little at the peak learning rate; still falling at the end (2.905 to 2.903 over the last 100 steps).
- Dev: math 2.816, code 2.839, medicine 2.724, history 3.329. vs best Shared-Attention MoE (2.989): -0.062. vs generalist: -0.229, 61% of the gap.
- Valid: 299,994,524 parameters, 148.6M active, causal. 0.61 A10G-h with dev scoring. One seed.
