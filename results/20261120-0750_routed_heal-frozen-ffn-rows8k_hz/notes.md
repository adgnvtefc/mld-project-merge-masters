# 20261120-0750_routed_heal-frozen-ffn-rows8k_hz

Hypothesis: Stacks with a rank-32 fix plus own rows for each expert's top 8,192 tokens (0.062 better untrained) keeps most of that lead after the same 30-minute training, beating 2.973 on dev.

Status: done

Parent: 20261119-2206_routed_heal-frozen-ffn_hz

## Outcome

Supported. Dev 2.9569, the team's best: 0.016 better than the same training with the rank-128 fix, and 0.006 better than that model trained 3x longer.

## Observations

- Same training as 20261119-2206 (800 steps, same data stream and schedule, FFNs frozen); only the embeddings differ: rank-32 fix plus each expert's own rows for its 8,192 most frequent tokens, instead of a rank-128 fix. 50.4% of FFN neurons kept (was 52.4%).
- Dev: math 2.845, code 2.868, medicine 2.753, history 3.362. vs the rank-128 run: -0.016 (95% range -0.018 to -0.014), better in every domain. vs the 90-minute rank-128 run (2.9630): -0.006 (-0.008 to -0.004). vs best MoE: -0.032. vs generalist: -0.199, 53% of the gap.
- Held-out with the router: 2.958 at step 0 (rank-128 build: 3.011), worse first (2.996 at step 100), then falling to 2.932 at 800. Before training the lead was 0.053 on these 32 windows; after, 0.014.
- Valid: 299,994,524 parameters (299,961,756 trainable plus 32,768 token-id entries), 148.6M active (1% under the limit), causal. 32.0 min training; 0.57 A10G-h with dev scoring. One seed.
