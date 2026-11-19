# 20261119-1845_routed_design-a_hz

Hypothesis: Design A with a real causal router (mix all experts for the first k tokens, then commit to the expert with the lowest prefix loss) loses little against oracle routing, and beats the generalist on dev (3.156) by a wide margin with no training.

Status: done

Parent: 20261119-1707_pruning_assemble-oracle-ranks_hz

## Outcome

Supported. Dev 3.0466, valid (299,995,804 parameters, 129.1M active, causal), with no training: 0.109 better than the generalist (95% range 0.102 to 0.116) and 0.095 better than the learned mix. The router costs 0.014 against oracle routing on pool-val.

## Observations

- Built from the provided models in about 1.5 minutes: expert-average shared embeddings, rank-128 corrections, 52.4% of FFN neurons per expert (layers 1-3 and 11 keep 154-456 of 3,072; layers 5-10 keep 2,050-2,850).
- Pool-val, oracle routing: 2.9988, the same as the Phase 2 check (20261119-1707), so the model reproduces that construction.
- Router accuracy on pool-val (lowest prefix loss = true domain): about 50% after 1 token, 75-86% after 8, 86-92% after 32, 94-98% after 128. Math stays lowest (92-94% after 128 tokens); medicine and history reach 100% by 256-511.
- Pool-val mean by commit point: 32 3.037 (109.7M active), 64 3.025 (116.2M), 128 3.013 (129.1M); 128 chosen.
- Dev vs generalist: math -0.227, code -0.215, history -0.066, medicine +0.072 (worse; better on only 17% of medicine windows, against 90-92% for math and code).
- Causality passed the official test plus extra cuts inside the mixed prefix (64) and just after the commit point (129).
- Weights stored in fp32 (1.2 GB in runs/ on EC2).
- 0.03 A10G-h including dev scoring.
