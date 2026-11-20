# 20261119-2241_routed_heal-all_hz

Hypothesis: Design A with every weight trained (same data and schedule) does better or worse than with the FFNs frozen, as a check on which parts need training.

Status: done

Parent: 20261119-1936_routed_design-a-det_hz

## Outcome

Training everything is worse than freezing the FFNs: dev 2.9919, 0.019 behind 20261119-2206 (95% range 0.017 to 0.021), and level with Ayush's best MoE (+0.003, -0.001 to +0.007). Same pattern as his runs: it first gets worse, then improves faster, and is behind at 800 steps.

## Observations

- Matched to Ayush's healing runs: same seed and data stream (800 steps x 32 windows = 52.4M pool train tokens, one window per domain per micro-batch, routed by label), schedule (3e-4 peak, 30 warmup, linear decay), AdamW and clipping. His dense control 20261119-0450 (3.1186) applies. Start: the deterministic design-A build (fingerprint 46ae47f84a72da85, dev 3.0416), commit 224.
- Every weight trained (300.0M).
- Dev: math 2.879, code 2.906, medicine 2.791, history 3.392. vs generalist -0.164 (95% range -0.171 to -0.156); medicine +0.005 (level). vs Ayush's train-everything MoE (3.0089): -0.017.
- Pool-val with the router: 3.011 at step 0, 3.101 at step 100 (0.09 worse: the experts' FFNs start moving), then falling to 2.963 at 800. Still falling 0.014 over the last 100 steps, against 0.003 for the frozen run, so longer training could change the order.
- Valid: 299,995,804 parameters, 148.5M active, causal. 31.6 min training at 27.8K tok/s; 0.56 A10G-h with the build and dev scoring.
