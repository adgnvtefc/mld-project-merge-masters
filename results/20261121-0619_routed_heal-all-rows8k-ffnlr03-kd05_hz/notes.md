# 20261121-0619_routed_heal-all-rows8k-ffnlr03-kd05_hz

Hypothesis: Adding distillation from each stack's own unpruned expert (0.5 next-token + 0.5 KL) to the best recipe (lr 1e-4, FFNs 0.3x) recovers more of the pruning and embedding damage than labels alone and beats 2.9242 on dev.

Status: done

Parent: 20261121-0426_routed_heal-all-rows8k-ffnlr03_hz

## Outcome

Supported, strongly. Dev 2.8994, the team's best: 0.025 better than the same recipe without distillation, and 0.018 better than training without it twice as long.

## Observations

- Same as 20261121-0426 (Stacks + own rows, all trained, lr 1e-4 with 200 warmup, FFNs at 0.3x, 800 steps, same data stream) plus distillation: loss = 0.5 next-token CE + 0.5 KL(own unpruned expert || stack). Teachers: the four provided experts in bf16, each teaching its own domain's stack.
- Dev: math 2.791, code 2.814, medicine 2.698, history 3.295. vs the same recipe without distillation: -0.0248 (95% range -0.0253 to -0.0243), better in every domain. vs 1,600 steps without it: -0.018. vs best Shared-Attention MoE: -0.090. vs generalist: -0.256, 68% of the gap; better on 95/97/88/91% of math/code/medicine/history windows.
- Held-out with the router fell at every validation: 2.958, 2.923, 2.912, 2.902, 2.894, 2.887, 2.882, 2.878, 2.875 (steps 0 to 800); without distillation it ended at 2.900.
- Throughput 17.8K tok/s (26K without the teachers); 51 minutes of training. 0.86 A10G-h with dev scoring.
- The first attempt (20261121-0423) ran out of memory; this run uses the per-sequence gradient from 60579df.
- Valid: 299,994,524 parameters, 148.6M active, causal. One seed.
