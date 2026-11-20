# 20261120-0618_routed_embed-check_hz

Hypothesis: Most of what sharing the embeddings costs the untrained Stacks is in each expert's most frequent tokens, so its own rows for its top few thousand tokens recover most of it, and the input side (free in active parameters) recovers more than the output side.

Status: done

Parent: 20261119-1936_routed_design-a-det_hz

## Outcome

Partly supported. Own rows for frequent tokens recover most of the cost (top 16,384: 0.160 of 0.187), but the cost is on the output side, not the input side: the input-only versions recover almost nothing.

## Observations

- Untrained Stacks (deterministic build, rank-128 fix, 52% of FFN neurons), label routing, 128 pool-val windows per domain. Mean: shared + fix 3.0186; own tables (ceiling) 2.8320 (-0.187).
- Output side carries the cost: own word table on the output only 2.8626 (-0.156); own tables on the input only 3.0004 (-0.018). Own position table: no change (+0.0005).
- Own rows for each expert's most frequent tokens (ranked on 20M pool train tokens), input and output: top 1,024 2.9798 (-0.039, 0.79M params per expert), top 4,096 2.9283 (-0.090, 3.15M), top 16,384 2.8590 (-0.160, 12.6M). Input only: -0.007, -0.011, -0.017.
- The top 1,024 / 4,096 / 16,384 tokens cover 64-72% / 80-88% / 94-98% of pool-val tokens.
- Per million parameters per expert: the first 4,096 own rows buy about 0.03; the low-rank fix from rank 64 to 128 bought about 0.007 (pruning runs); FFN neurons near 52% kept buy about 0.015. So budget should move from the low-rank fix to own rows for frequent tokens.
- Output rows cost active parameters (K x 768 per token); input rows do not.
- 0.0669 A10G-h.
