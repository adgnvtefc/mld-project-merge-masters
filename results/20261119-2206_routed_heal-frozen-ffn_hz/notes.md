# 20261119-2206_routed_heal-frozen-ffn_hz

Hypothesis: Design A trained like Ayush's best MoE run (experts' FFNs frozen; embeddings, corrections, attention and LayerNorms trained; same data and schedule) keeps its 0.165 head start and beats his 2.9893 on dev.

Status: done

Parent: 20261119-1936_routed_design-a-det_hz

## Outcome

Supported. Dev 2.9729, the team's best: 0.016 better than Ayush's best MoE (20261119-0423, 2.9893; 95% range 0.013 to 0.020) and 0.183 better than the generalist (0.175 to 0.190), 48% of the gap to the per-domain experts. But most of design A's 0.165 head start over the untrained MoE did not survive training: training gained 0.069 here against 0.218 for his MoE.

## Observations

- Matched to Ayush's healing runs: same seed and data stream (800 steps x 32 windows = 52.4M pool train tokens, one window per domain per micro-batch, routed by label), schedule (3e-4 peak, 30 warmup, linear decay), AdamW and clipping. His dense control 20261119-0450 (3.1186) applies. Start: the deterministic design-A build (fingerprint 46ae47f84a72da85, dev 3.0416), commit 224.
- Trained 181.3M of 300.0M parameters (every expert's FFN frozen): hidden matrices 113.2M (lr x0.25), embeddings and corrections 67.7M, vectors 0.3M.
- Dev: math 2.858, code 2.887, medicine 2.772, history 3.375. vs generalist: -0.284, -0.281, -0.015, -0.150. Medicine beats the generalist (on 65% of windows), which no earlier model did. vs Ayush's best: better on 72 / 87 / 63 / 81% of math / code / medicine / history windows.
- Pool-val with the router (32 windows per domain): 3.011 at step 0, 3.014 at 100, then falling every validation to 2.946 at 800 (oracle 2.932; the router costs 0.014). The last 100 steps gained 0.003 (router) and 0.006 (oracle), with the learning rate decayed to zero.
- Dev is 0.027 above the final pool-val router score (Ayush's runs: 0.027-0.030).
- Valid: 299,995,804 parameters, 148.5M active, causal. 29.4 min training at 29.8K tok/s; 0.53 A10G-h with the build and dev scoring.
