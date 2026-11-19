# 20261119-0535_merging_routed-oracle_hz

Hypothesis: Re-run of 20261118-2105_merging_routed-oracle_hz after the pool-val fix (d725d3f: windows spread over the whole slice, not the first 64 in a row); same grid and settings. Original hypothesis: Measures how close each domain gets to its own expert when given only that expert's rank-r change from the generalist, and which other tensors (LayerNorms, biases, embeddings) the change needs.

Status: done

## Outcome

Diagnostic re-run with fixed pool-val windows; replaces the math-biased table of 20261118-2105_merging_routed-oracle_hz.

## Observations

- With the fixed windows, the math rows are now valid. Generalist (no piece) per domain: math 3.0769, code 3.1156, medicine 2.8073, history 3.5445, mean 3.1361.
- Pieces with the generalist's other tensors: worse than the generalist at every rank, in every domain (rank 16 mean 3.4096, rank 160 3.5566, full 3.4021). With the expert's LayerNorms and biases too: worse still (3.95-4.38).
- Pieces with the expert's LayerNorms, biases and embeddings: rank 160 beats the generalist on math (3.0244), code (3.0803) and history (3.5184), not medicine (2.8452); mean 3.1171, 0.019 better than the generalist. That recovers about 5% of the gap to the full expert (mean 2.7468): math 10%, code 8%, history 7%, medicine -18% (worse than the generalist).
- Full-rank matrices with all of the expert's other tensors reproduce the expert exactly (2.7468).
