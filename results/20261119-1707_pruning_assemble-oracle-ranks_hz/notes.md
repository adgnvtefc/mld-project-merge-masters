# 20261119-1707_pruning_assemble-oracle-ranks_hz

Hypothesis: Design A's best embedding-correction rank is above 64: more rank keeps helping until the FFN neurons it costs (58% kept at rank 64, about 40% at rank 256) start to hurt.

Status: done

Parent: 20261119-1702_pruning_assemble-oracle_hz

## Outcome

Supported. The best rank is 96-128: more rank helps until the FFN neurons it costs start to hurt. Best mean 2.999 (expert-average embeddings, rank 128), 0.137 below the generalist: 35% of the gap from the generalist to the experts.

## Observations

- Budgets: rank 64 / 96 / 128 / 192 / 256 keep 58.4 / 55.4 / 52.4 / 46.3 / 40.2% of FFN neurons.
- Pool-val mean by rank, expert-average embeddings: 3.022 / 3.002 / 2.999 / 3.022 / 3.077. Learned embeddings: 3.028 / 3.022 / 3.025 / 3.053 / 3.109. Generalist embeddings: 3.033 / 3.028 / 3.031 / 3.058 / 3.114.
- Best per domain (expert average, rank 128): math 2.798, code 2.894, medicine 2.861, history 3.442 (generalist 3.077, 3.116, 2.807, 3.545). Medicine is still 0.05 worse than the generalist. With generalist embeddings, medicine is best (2.793 at rank 64) and the others are worse.
- Pruning alone, own embeddings, at each budget: 2.768 / 2.788 / 2.812 / 2.874 / 2.958, against 2.747 unpruned. At rank 128 the remaining gap to the experts (0.252) is about 0.065 from pruning and 0.187 from sharing the embeddings.
- Routing is by the true domain label (an oracle router), on pool-val (64 windows per domain), with no training. This is not a submittable model: a real router misroutes some early tokens, and there is no dev score.
- Rules: design A keeps each expert nearly whole and shares only the embeddings, which is close to what the 300M limit is meant to stop ("a classifier which picks a provided expert model per document"). It needs the same Piazza question as Method 2, more urgently.
- 0.05 A10G-h.
