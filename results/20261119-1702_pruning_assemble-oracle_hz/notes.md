# 20261119-1702_pruning_assemble-oracle_hz

Hypothesis: With label routing and no training, design A (shared embeddings, each expert's own attention and 65% of its FFN) beats design B (shared attention, full FFNs), but shared embeddings cost most of the experts' lead; a small per-expert embedding correction recovers part of it.

Status: done

Parent: 20261119-1623_pruning_sweep-fine_hz

## Outcome

Supported, and more strongly than expected. Design A beats design B by a wide margin. Shared embeddings alone break the experts, but a rank-64 per-expert correction of the embeddings recovers most of it: with expert-average embeddings the mean is 3.022, against 3.136 for the generalist and 3.122 for the learned mix.

## Observations

- Budgets (all at most 300M total): rank 0 keeps 64.5% of FFN neurons, rank 16 63.0%, rank 64 58.4%; about 103M active. Design B: 295.2M total, 123.5M active.
- Pool-val mean, routed by label: generalist 3.136, learned mix 3.122, own experts 2.747, own experts pruned to 64.5% 2.748.
- Shared embeddings with no correction: 3.73 (expert average) to 4.88 (seed), far worse than the generalist. Unpruned gives the same numbers, so the pruning costs nothing here; the embeddings are the whole problem.
- Correction rank 16: 3.078-3.153 for the generalist, learned and expert-average sources (seed 3.548). Rank 64: generalist 3.033, learned 3.028, expert average 3.022, seed 3.357.
- Per domain at rank 64 with expert-average embeddings: math 2.799, code 2.914, medicine 2.888, history 3.485 (generalist 3.077, 3.116, 2.807, 3.545). Medicine is still worse than the generalist; with generalist embeddings medicine is 2.793, slightly better than it.
- Design B (shared attention and LayerNorms, full expert FFNs), no training: 3.443 (generalist source) and 3.456 (learned source), worse than the generalist.
- Routing is by the true domain label (an oracle router), on pool-val (64 windows per domain), with no training. This is not a submittable model: a real router misroutes some early tokens, and there is no dev score.
- Rules: design A keeps each expert nearly whole and shares only the embeddings, which is close to what the 300M limit is meant to stop ("a classifier which picks a provided expert model per document"). It needs the same Piazza question as Method 2, more urgently.
- 0.05 A10G-h.
