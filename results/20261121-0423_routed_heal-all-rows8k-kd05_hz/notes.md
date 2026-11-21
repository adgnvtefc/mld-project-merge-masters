# 20261121-0423_routed_heal-all-rows8k-kd05_hz

Hypothesis: Adding distillation from each stack's own unpruned expert (loss = 0.5 next-token + 0.5 KL) to the best recipe recovers more of the pruning and embedding damage than labels alone and beats 2.9269 on dev.

Status: failed

Parent: 20261120-1753_routed_heal-all-rows8k-lr1e-4_hz

## Outcome

Failed: CUDA out of memory in the distillation loss after the model build (the full-vocabulary KL held several 4 x 2048 x 50,257 float tensors at once). No result.

## Observations

- Fixed in 60579df: the gradient of the combined loss is formed one sequence at a time and pushed through the network in one backward call (checked against autograd, relative error 5.5e-7). Re-run on the best recipe as heal-all-rows8k-ffnlr03-kd05.
