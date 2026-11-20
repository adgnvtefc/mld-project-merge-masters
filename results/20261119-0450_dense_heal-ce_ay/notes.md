# 20261119-0450_dense_heal-ce_ay

Hypothesis: the generalist trained the same way improves less than the MoE, so the MoE's gain is not just extra training

Status: done | dev mean 3.1186

## Outcome

Supported. The generalist, trained with the same loop, data (52.4M pool train tokens, equal per domain), steps,
schedule and learning rate as the MoE healing runs, improves by 0.037. The MoE runs improve on the generalist
by 0.147 (heal-all) and 0.166 (heal-shared), and beat this control by 0.110 and 0.129.

Controls for `20261119-0352_moe_heal-all_ay` and `20261119-0423_moe_heal-shared_ay` (launched without `--parent`).

Dev, with the difference and its paired bootstrap 95% interval (negative = this run is better):

| | math | code | medicine | history | mean |
|---|---|---|---|---|---|
| dev | 3.0558 | 3.0865 | 2.8219 | 3.5103 | **3.1186** |
| vs generalist (3.1555) | -0.086 [-0.091, -0.081] | -0.082 [-0.087, -0.078] | +0.035 [+0.033, +0.037] | -0.014 [-0.019, -0.010] | -0.037 [-0.039, -0.035] |
| vs learned-lambda (3.1411) | -0.044 [-0.048, -0.040] | -0.058 [-0.062, -0.054] | +0.009 [+0.007, +0.011] | +0.004 [-0.000, +0.007] | -0.022 [-0.024, -0.021] |
| windows better than generalist | 95% | 95% | 3% | 64% | |

Valid: 125,226,240 parameters, 123,532,032 active, causal. 1,445 s training (0.401 A10G-h).

## Observations

- Pool-val got worse before it got better: 3.1305 (step 0), 3.2138 (step 100), 3.1984, 3.1778, 3.1567, 3.1381,
  3.1185, 3.1019, 3.0916 (step 800). It passed its starting score between steps 500 and 600. The MoE runs fell
  at every validation.
- Medicine got worse on dev (+0.035, worse on 97% of windows); math and code improved.
- Beats learned-lambda (3.1411) by 0.022, so further training of the generalist alone already beats every
  training-free merge so far.
- Dev is 0.027 above the final pool-val score.
- Throughput 36.5K tok/s. One seed; one learning rate (3e-4 peak), chosen for the MoE runs, not tuned for this.
