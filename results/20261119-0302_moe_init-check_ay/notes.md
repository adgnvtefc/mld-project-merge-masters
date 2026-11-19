# 20261119-0302_moe_init-check_ay

Hypothesis: an untrained MoE of the experts' MLPs is worse than the generalist, and frequency-weighted embedding maps recover part of the gap

Status: done | dev mean 3.2068

## Outcome

Both parts supported. No weights were trained; the router and the embedding maps were fitted in closed form
on pool train data. Six configurations were scored on pool-val (64 windows per domain); the best by prefix-
router mean was saved and scored on dev.

- **Untrained MoE is worse than the generalist.** Every configuration is worse than the generalist on pool-val
  (3.136). The saved best scores 3.2068 on dev, against 3.1555 for the generalist: +0.051, paired bootstrap
  95% interval [+0.047, +0.056]. Against learned-lambda (3.1411): +0.066 [+0.062, +0.070].
- **The maps recover part of the gap.** With averaged shared parts they recover most of the extra loss over the
  generalist; with the generalist's shared parts, about a third.

pool-val (references on the same windows: generalist 3.1361, own expert per domain 2.7468):

| Shared parts | Maps | Oracle routing | Prefix router | Map residual |
|---|---|---|---|---|
| generalist | off | 3.4453 | 3.4564 | |
| generalist | on | 3.3436 | 3.3569 | 0.35-0.40 |
| mean5 (4 experts + generalist) | off | 3.4415 | 3.4526 | |
| **mean5** | **on** | **3.1636** | **3.1793** | 0.30-0.33 |
| mean4 (4 experts) | off | 3.4897 | 3.5011 | |
| mean4 | on | 3.1920 | 3.2088 | 0.28-0.32 |

Saved model (mean5 + maps) on dev, with the difference from the generalist and its paired bootstrap 95%
interval:

| | math | code | medicine | history | mean |
|---|---|---|---|---|---|
| dev | 3.0509 | 3.1682 | 2.9677 | 3.6403 | 3.2068 |
| vs generalist | -0.091 [-0.102, -0.080] | -0.001 [-0.008, +0.006] | +0.181 [+0.174, +0.189] | +0.116 [+0.107, +0.124] | +0.051 [+0.047, +0.056] |
| windows where MoE is better | 79% | 53% | 0.2% | 8% | |

Valid: 297,596,164 parameters, ~124.7M active, causal. 114 s of GPU time including dev scoring (0.032 A10G-h).

## Observations

- The same configuration (mean5 + maps) was best by both oracle and router mean, so the choice does not depend
  on routing.
- The maps help much more with averaged shared parts (-0.28 for mean5, -0.30 for mean4) than with the
  generalist's (-0.10). The averaged embeddings also have lower map residuals (about 0.30 against 0.38).
- Without maps, averaging the shared parts trades domains: math and code improve over generalist-shared, medicine
  and history get worse. With maps, mean5 beats generalist-shared on math, code and history (oracle), and is about
  level on medicine (2.954 vs 2.940).
- Math is better than the generalist on dev with no training (-0.091, about 19% of the math expert's 0.467 lead
  on dev). Code is level. Medicine and history are clearly worse, medicine on 99.8% of its windows.
- Routing costs about 0.011-0.017 over oracle routing. Router accuracy over all positions is 0.945-0.956; by
  position (mean5 router): 0.40 at token 0, 0.59 at token 7, 0.84 at 63, 0.94 at 255, 0.97 at 2047.
- The dev score is 0.028 above the saved model's pool-val router score; the generalist's is 0.020 above its
  pool-val score.
- Not tested: the learned-lambda merge as the shared parts (needs that run's merge_settings.json on this host).
- One seed; 64 pool-val windows per domain.
