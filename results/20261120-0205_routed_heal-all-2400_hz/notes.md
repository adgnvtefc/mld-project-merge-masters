# 20261120-0205_routed_heal-all-2400_hz

Hypothesis: Stacks with all weights trained gains more from 3x the training than FFNs frozen does (it was still improving 5x faster at step 800), and may overtake it at 2,400 steps.

Status: done

Parent: 20261119-2241_routed_heal-all_hz

## Outcome

Not supported: training everything did not overtake. Dev 2.9819, 0.0101 better than its 800-step run (0.0074 to 0.0126) but still 0.019 behind FFNs frozen at 2,400 steps (0.017 to 0.021), the same gap as at 800 steps.

## Observations

- Same as its 800-step parent except --steps 2400 --val-every 200: same start (fingerprint 46ae47f84a72da85), data stream, warmup (30 steps) and peak learning rate; the linear decay is stretched over 2,400 steps. 157.3M training tokens.
- Dev: math 2.870, code 2.897, medicine 2.780, history 3.380.
- Held-out loss with the router jumped from 3.011 to 3.121 at step 200 and only passed its start between steps 1,600 and 1,800. It fell to 2.952 at 2,400, still falling 0.010 over the last 200 steps.
- 93.1 min of training. 1.59 A10G-h with dev scoring. Valid, 148.5M active, causal. One seed.
