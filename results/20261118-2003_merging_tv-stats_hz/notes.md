# 20261118-2003_merging_tv-stats_hz

Hypothesis: Records the norms and pairwise cosines of the five task vectors, to explain how the merges behave.

Status: done

## Outcome

No model; diagnostics only (see train.log for the full tables).

## Observations

- Relative norm ||tau||/||seed||: 0.41 (generalist) to 0.52 (code expert) overall; about 0.73-0.85 on embeddings, 0.61-0.68 on MLP weights, 0.45-0.51 on attention, 0.22-0.24 on LayerNorms.
- Pairwise cosines are high: 0.64-0.79 overall, 0.58-0.74 on embeddings, 0.77-0.86 on attention, 0.59-0.72 on MLP, 0.99+ on LayerNorms. Most similar pair: medicine ~ generalist (0.79); least: code ~ history (0.64).
