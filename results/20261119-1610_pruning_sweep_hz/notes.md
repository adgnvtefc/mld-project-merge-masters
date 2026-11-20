# 20261119-1610_pruning_sweep_hz

Hypothesis: Each expert can lose a large share of its FFN neurons (35% or more) and keep most of its own-domain lead over the generalist, if neurons are chosen by own-domain importance and the layer output is repaired; this decides whether pruned experts with their own attention fit the 300M limit.

Status: done

## Outcome

Supported, with global allocation. With one ranking over all layers, every expert keeps its whole own-domain lead at 65% of FFN neurons (loss +0.0004 to +0.0008 with the least-squares refit, +0.001 to +0.003 with no fix). With the same share removed from every layer it keeps only 40-75% (medicine 40%). Below 65% the loss rises fast.

## Observations

- Pool-val own-domain loss, generalist / expert: math 3.077 / 2.572, code 3.116 / 2.676, medicine 2.807 / 2.597, history 3.545 / 3.142. Sanity: keep 1.0 with the refit reproduces each expert (within 0.0001).
- Best setting: taylor scores, global allocation, least-squares refit. Loss increase over the expert at 90 / 80 / 65 / 50 / 35 / 20% kept: about 0 / 0 / +0.0005 / +0.08 to +0.09 / +0.28 to +0.32 / +0.70 to +0.80, about the same in all four domains.
- Global allocation at 65% keeps almost no neurons in layers 1, 2, 3 and 11 (5%, 10%, 18%, 15%), about 74% of layers 0 and 4, and 93-100% of layers 5-10. Taylor (loss units) and activation (output size) rankings give the same per-layer split.
- Same share per layer, 65% kept: 40-75% of the lead with the refit (best taylor; medicine 40%), and the expert is no better than the generalist without it.
- Repairs: the least-squares refit always helps and matters more the more is pruned; folding the mean into the bias helps a little. Random choice with the refit keeps 42-53% at 65% per layer, except medicine (-8%).
- Criteria: taylor >= activation > magnitude > random throughout.
- Kept neurons overlap between experts more than chance (Jaccard 0.66-0.70 at 65% per layer, chance 0.48): the experts tend to keep the same neurons.
- Medicine's retained share falls fastest, but only because its lead is smallest (0.21): in absolute loss, pruning costs medicine about the same as the others.
- Pool-val only (64 windows per domain), each expert with its own embeddings and attention; no dev score. 0.19 A10G-h.
