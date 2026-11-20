# 20261119-1936_routed_design-a-det_hz

Hypothesis: Grouping the taylor statistic by 4 windows again (as in the original sweep) recovers the earlier build's dev score (about 3.041) while staying deterministic.

Status: done

Parent: 20261119-1921_routed_design-a-det_hz

## Outcome

Supported. Deterministic build with the taylor statistic grouped as in the original sweep: dev 3.0416, the same as the earlier nondeterministic build 20261119-1850 (3.0413; difference +0.0002, 95% range -0.0000 to +0.0007). This is the reference design-A build.

## Observations

- Weights fingerprint 46ae47f84a72da85. 299,995,804 parameters, 148.5M active (commit 224), causal.
- Dev: math 2.913, code 2.948, medicine 2.853, history 3.453. vs generalist -0.114 (95% range -0.121 to -0.107).
- vs Ayush's runs on the same dev windows: 0.165 better than his untrained MoE (20261119-0302, 3.2068); 0.052 worse than his MoE trained with experts frozen (20261119-0423, 2.9893) and 0.033 worse than trained in full (20261119-0352, 3.0089); 0.077 better than his dense control (20261119-0450, 3.1186).
- Pool-val now uses 256 windows per domain, so its numbers are not comparable with the 64-window runs (oracle 3.03-3.04 here vs 2.999); dev is unaffected.
- Commit point chosen on pool-val: 224 (pool-val falls steadily from 128 to 224 on 256 windows, which the 64-window runs could not show).
- 0.10 A10G-h including dev scoring.
