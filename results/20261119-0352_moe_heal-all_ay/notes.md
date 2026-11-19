# 20261119-0352_moe_heal-all_ay

Hypothesis: training the whole MoE with the next-token loss brings it below the generalist and learned-lambda (3.141)

Status: done | dev mean 3.0089

## Outcome

Supported. Starting from the init-check model (mean5 shared parts + embedding maps, dev 3.2068), 800 steps of
32 x 2,048 tokens (52.4M tokens, equal per domain, routed by domain label) with every weight trained bring the
dev mean to 3.0089.

Dev, with the difference and its paired bootstrap 95% interval (negative = this run is better):

| | math | code | medicine | history | mean |
|---|---|---|---|---|---|
| dev | 2.8873 | 2.9324 | 2.8013 | 3.4148 | **3.0089** |
| vs generalist (3.1555) | -0.254 [-0.270, -0.239] | -0.236 [-0.248, -0.225] | +0.014 [+0.006, +0.023] | -0.110 [-0.122, -0.098] | -0.147 [-0.153, -0.141] |
| vs learned-lambda (3.1411) | -0.213 [-0.228, -0.198] | -0.212 [-0.223, -0.201] | -0.011 [-0.020, -0.002] | -0.092 [-0.104, -0.080] | -0.132 [-0.138, -0.126] |
| vs dense control (3.1186) | -0.169 [-0.181, -0.157] | -0.154 [-0.161, -0.147] | -0.021 [-0.028, -0.013] | -0.096 [-0.104, -0.087] | -0.110 [-0.114, -0.105] |
| vs init-check (3.2068) | -0.164 | -0.236 | -0.166 | -0.226 | -0.198 [-0.202, -0.194] |
| windows better than generalist | 95% | 96% | 52% | 83% | |

Dense control: `20261119-0450_dense_heal-ce_ay` (same loop, data, schedule and learning rate).

Valid: 297,596,164 parameters, 124,714,752 active, causal. 1,706 s training (0.474 A10G-h).

## Observations

- Pool-val (router) fell at every validation: 3.1828 (step 0), 3.1425, 3.1195, 3.0986, 3.0717, 3.0407, 3.0142,
  2.9934, 2.9786 (step 800). Still falling 0.015 per 100 steps at the end, with the learning rate decayed to zero.
- Oracle routing at step 800: 2.9670; the prefix router costs 0.012. Router accuracy (refitted each validation)
  0.955 at step 0, 0.961 at step 800.
- Medicine is the only domain not better than the generalist (+0.014). The generalist's medicine (2.787) was
  already closest to its expert's (2.587).
- Own-expert dev per domain: 2.675, 2.710, 2.587, 3.145. This run is 0.19-0.27 above each.
- Dev is 0.030 above the final pool-val router score.
- Throughput 31.2K tok/s, below the dense control's 36.5K and heal-shared's 35.3K.
- Every weight trained (297.6M), experts at the same learning rate as the shared parts (3e-4 peak, x0.25 for
  hidden matrices by muP). One seed.
