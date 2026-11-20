# 20261120-1715_routed_heal-frozen-ffn-rows8k-lr1e-4_hz

Hypothesis: A lower peak learning rate (1e-4, 200-step warmup instead of 3e-4, 30) avoids the early damage (+0.038 in the first 100 steps at 3e-4), keeps more of the own rows' head start, and beats 2.957 on dev with the same data and steps.

Status: done

Parent: 20261120-0750_routed_heal-frozen-ffn-rows8k_hz

## Outcome

Supported. Dev 2.9286: 0.028 better than the same run at 3e-4 (95% range 0.027 to 0.030), by the same amount in every domain. 60% of the gap closed; 0.227 better than the generalist.

## Observations

- Same as 20261120-0750 (Stacks, rank-32 fix + own rows for the top 8,192 tokens, 800 steps, same data stream) except the learning rate: peak 1e-4 with a 200-step warmup, instead of 3e-4 with 30. FFNs frozen.
- Held-out with the router fell at every validation: 2.958 (step 0), 2.935, 2.931, 2.926, 2.917, 2.911, 2.910, 2.906, 2.904 (step 800). At 3e-4 it first rose to 2.996 and ended at 2.932.
- Dev: math 2.817, code 2.838, medicine 2.726, history 3.334. Medicine 0.061 better than the generalist.
- Valid: 299,994,524 parameters, 148.6M active, causal. 0.58 A10G-h with dev scoring. One seed.
