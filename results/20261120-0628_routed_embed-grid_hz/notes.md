# 20261120-0628_routed_embed-grid_hz

Hypothesis: At a fixed 300M, moving each expert's embedding budget from the low-rank fix into its own rows for its most frequent tokens lowers the untrained Stacks' loss, despite the FFN neurons it costs.

Status: done

Parent: 20261120-0618_routed_embed-check_hz

## Outcome

Supported. Moving each expert's budget from the low-rank fix into its own rows for its most frequent tokens lowers the untrained loss by up to 0.062 at the same 300M, but only while some low-rank fix remains.

## Observations

- Untrained Stacks, 300M total for every split; FFN neurons take what the fix (rank r) and own rows (top K tokens) leave. 128 pool-val windows per domain. Router: best commit point within 150M active.
- Router mean, rank:rows: 128:0 3.0248 (the current split), 64:4096 2.9694, 32:6144 2.9650, 32:8192 2.9632, 64:8192 2.9712, 0:8192 3.0429, 0:16384 3.0415. Perfect routing: 3.0187, 2.9646, 2.9613, 2.9581, 2.9670, 3.0360, 3.0381.
- Best: 32:8192, 0.062 better than the current split before training, keeping 50.4% of FFN neurons (was 52.4%). 148.6M active at commit 224.
- Without any low-rank fix (rank 0), rare tokens break: history gets worse than the current split (3.676 vs 3.489 with 8,192 rows).
- Per domain at 32:8192 (perfect routing): math 2.767, code 2.837, medicine 2.760, history 3.469 (current split: 2.841, 2.908, 2.836, 3.489).
- 0.2285 A10G-h.
