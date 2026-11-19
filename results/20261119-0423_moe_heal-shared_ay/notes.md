# 20261119-0423_moe_heal-shared_ay

Hypothesis: freezing the experts' MLPs and training only the shared parts keeps more of the experts' gains than training everything

Status: done | dev mean 2.9893

## Outcome

Supported on score: with the four experts' MLPs frozen, it beats training everything (`heal-all`) on every
domain, by 0.020 on the mean. Same start, data, steps and schedule as `20261119-0352_moe_heal-all_ay`; only
embeddings, attention, LayerNorms and embedding maps trained (70.9M of 297.6M parameters). Best dev score on the
team so far.

Dev, with the difference and its paired bootstrap 95% interval (negative = this run is better):

| | math | code | medicine | history | mean |
|---|---|---|---|---|---|
| dev | 2.8653 | 2.9156 | 2.7809 | 3.3953 | **2.9893** |
| vs heal-all (3.0089) | -0.022 [-0.023, -0.021] | -0.017 [-0.018, -0.016] | -0.020 [-0.021, -0.019] | -0.019 [-0.021, -0.018] | -0.020 [-0.020, -0.019] |
| vs generalist (3.1555) | -0.276 [-0.292, -0.261] | -0.253 [-0.264, -0.242] | -0.006 [-0.014, +0.003] | -0.129 [-0.142, -0.117] | -0.166 [-0.172, -0.160] |
| vs learned-lambda (3.1411) | -0.235 [-0.250, -0.220] | -0.229 [-0.239, -0.218] | -0.032 [-0.040, -0.023] | -0.111 [-0.123, -0.100] | -0.152 [-0.158, -0.146] |
| vs dense control (3.1186) | -0.190 [-0.203, -0.179] | -0.171 [-0.178, -0.163] | -0.041 [-0.048, -0.034] | -0.115 [-0.124, -0.106] | -0.129 [-0.134, -0.125] |
| windows better than heal-all | 95% | 93% | 96% | 96% | |
| windows better than generalist | 96% | 98% | 62% | 86% | |

Valid: 297,596,164 parameters, 124,714,752 active, causal. 1,510 s training (0.419 A10G-h).

## Observations

- Pool-val (router): 3.1828 (step 0), 3.0483, 3.0274, 3.0122, 2.9978, 2.9842, 2.9743, 2.9656, 2.9611 (step 800).
  It fell faster than heal-all early (3.048 vs 3.143 at step 100) and slower late (-0.0045 vs -0.015 over the
  last 100 steps). The gap to heal-all shrank from 0.094 at step 100 to 0.018 at step 800.
- Oracle routing at step 800: 2.9482; the prefix router costs 0.013. Router accuracy 0.955 to 0.959.
- Medicine is level with the generalist (-0.006, interval includes 0).
- Own-expert dev per domain: 2.675, 2.710, 2.587, 3.145. This run is 0.19-0.25 above each. The per-domain
  own-expert mean is 2.779, so this run covers 44% of the generalist's 0.377 gap to it.
- Dev is 0.028 above the final pool-val router score.
- Throughput 35.3K tok/s (heal-all 31.2K).
- One seed.
