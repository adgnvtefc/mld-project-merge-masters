# 20261119-1623_pruning_sweep-fine_hz

Hypothesis: The near-free 35% of global pruning comes from neurons that are silent on the expert's own text, mostly in layers 1-3 and 11, inherited from the seed; and the loss stays near the expert's until close to 55-60% kept.

Status: done

Parent: 20261119-1610_pruning_sweep_hz

## Outcome

Half supported. The loss stays near the expert's down to about 60% kept (+0.011 to +0.013 with the refit), then rises (+0.04 to +0.05 at 55%). But the free pruning is not mostly silent neurons, and the silent ones are not inherited from the seed.

## Observations

- Taylor, global, least-squares refit, loss increase over the expert at 65 / 60 / 55 / 50% kept: math +0.0004 / +0.013 / +0.047 / +0.092, code +0.0006 / +0.011 / +0.039 / +0.077, medicine +0.0004 / +0.013 / +0.044 / +0.086, history +0.0009 / +0.013 / +0.047 / +0.093.
- Silent neurons (contribution below 1% of their layer's median): about 768 per model, 25% of layer 4 and almost none anywhere else. The same neurons are silent in the generalist and in all four experts (766-772 of each expert's silent set), and none are silent in the seed. So a quarter of layer 4 went silent during the shared continued training, before or apart from domain specialization.
- Those 768 account for only about 2% of all FFN neurons. The rest of the near-free 35% are neurons that are active but matter little, concentrated in layers 1-3 and 11.
- Activation ranking without a refit falls apart below 65% (medicine 0.05 retained at 60%); taylor plus the refit is the most robust.
- 0.08 A10G-h.
