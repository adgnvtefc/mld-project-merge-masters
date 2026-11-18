# 20261118-2105_merging_routed-oracle_hz

Hypothesis: Measures how close each domain gets to its own expert when given only that expert's rank-r change from the generalist, and which other tensors (LayerNorms, biases, embeddings) the change needs.

Status: done

## Outcome

No model; diagnostic. With perfect routing, no rank-r piece (r <= 160) beat the generalist on pool-val (3.1888), whatever the other tensors came from. Only the full expert (2.9149) did.

## Observations

- Pieces with every other tensor from the generalist got worse as r grew: rank 16 -> 3.4427, 64 -> 3.5303, 128 -> 3.6016, 160 -> 3.6230; the whole matrix change -> 3.5697.
- Taking the expert's LayerNorms and biases as well made every row worse (rank 16 -> 4.1233 ... full -> 4.7390).
- Taking the expert's LayerNorms, biases and embeddings: rank 16 -> 3.3041, 64 -> 3.2879, 128 -> 3.2589, 160 -> 3.2394, full -> 2.9149 (identical to the expert). This row improves with rank but is still 0.05 above the generalist at rank 160.
- Rank 160 is about the most the 150M active limit allows (123.5M + 0.147M per rank). Separate embeddings per expert would also cost about 40M each against the 300M total.
- Pool-val, 64 windows per domain; each domain scored only on its own text.
