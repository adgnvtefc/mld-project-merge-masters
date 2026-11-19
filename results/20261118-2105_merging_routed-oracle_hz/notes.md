# 20261118-2105_merging_routed-oracle_hz

Hypothesis: Measures how close each domain gets to its own expert when given only that expert's rank-r change from the generalist, and which other tensors (LayerNorms, biases, embeddings) the change needs.

Status: done

## Outcome

No model; diagnostic. Corrected 2026-11-18: the first version of this note said no rank-r piece (r <= 160) beat the generalist on pool-val. That came from the 4-domain mean, which the biased math windows (see Caveat) dragged up. Per domain, with the expert's LayerNorms, biases and embeddings, rank 160 does beat the generalist on code (3.1669 vs 3.2242) and history (3.5069 vs 3.5402), not on medicine (2.8570 vs 2.8105). It recovers only about 12% (code) and 8% (history) of the expert's lead; the full expert reaches 2.7558 and 3.1335. With the generalist's other tensors, every rank is worse than the generalist in every domain.

## Observations

- Pieces with every other tensor from the generalist got worse as r grew: rank 16 -> 3.4427, 64 -> 3.5303, 128 -> 3.6016, 160 -> 3.6230; the whole matrix change -> 3.5697.
- Taking the expert's LayerNorms and biases as well made every row worse (rank 16 -> 4.1233 ... full -> 4.7390).
- Taking the expert's LayerNorms, biases and embeddings: rank 16 -> 3.3041, 64 -> 3.2879, 128 -> 3.2589, 160 -> 3.2394, full -> 2.9149 (identical to the expert). This row improves with rank but is still 0.05 above the generalist at rank 160.
- Rank 160 is about the most the 150M active limit allows (123.5M + 0.147M per rank). Separate embeddings per expert would also cost about 40M each against the 300M total.
- Pool-val, 64 windows per domain; each domain scored only on its own text.

## Caveat (added 2026-11-18)

Settings and pool-val numbers in this run used the first n pool-val windows in a row. For math these were 79% one GMAT reading-comprehension guide, where the math expert is barely better than the generalist, so math pulled every choice toward the generalist. Dev scores are correct. Fixed in commit d725d3f; this run has not been re-run. See error_analysis.md.

Re-run as 20261119-0535_merging_routed-oracle_hz after the fix (2026-11-19): see that run for the corrected table; its conclusions are in error_analysis.md.
