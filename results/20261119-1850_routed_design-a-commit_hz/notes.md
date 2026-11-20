# 20261119-1850_routed_design-a-commit_hz

Hypothesis: Committing later than 128 tokens helps a little more (fewer misroutes), up to the 150M active limit near 232 tokens.

Status: done

Parent: 20261119-1845_routed_design-a_hz

## Outcome

Weakly supported. Commit 224 has the lowest pool-val mean (3.0119) and dev 3.0413, 0.005 better than commit 128's run. But the pool-val differences between commit points (under 0.007) are about the size of the run-to-run difference seen here, so the choice is not well determined.

## Observations

- Pool-val mean by commit point: 128 3.0149, 160 3.0199, 192 3.0132, 224 3.0119; active 129.1 / 135.6 / 142.0 / 148.5M. 224 is 1% under the 150M limit (staff check it with the same counter).
- Dev: 3.0413 (math 2.913, code 2.947, medicine 2.853, history 3.452). vs generalist -0.114 (95% range -0.121 to -0.107); vs the learned mix -0.100. Medicine still +0.066 worse than the generalist.
- Run to run: the same build at commit 128 gave pool-val 3.0149 here against 3.0125 in 20261119-1845 (oracle routing identical to 1e-4). Likely cause: non-deterministic GPU arithmetic in the calibration pass changes a few near-tied routing decisions. Worth pinning down before a final recipe, which must reproduce within 0.005.
- 0.03 A10G-h including dev scoring.
