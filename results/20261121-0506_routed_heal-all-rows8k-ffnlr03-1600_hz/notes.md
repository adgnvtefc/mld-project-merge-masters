# 20261121-0506_routed_heal-all-rows8k-ffnlr03-1600_hz

Hypothesis: At the gentler learning rates (1e-4, FFNs 0.3x), twice the training (1,600 steps, about 60 minutes) keeps improving past 2.9242 on dev, more than the 0.010 that 3x training bought at 3e-4.

Status: done

Parent: 20261121-0426_routed_heal-all-rows8k-ffnlr03_hz

## Outcome

Supported. Dev 2.9175, the team's best, 63% of the gap. Twice the training gained 0.0067 at these learning rates.

## Observations

- Same as 20261121-0426 (Stacks + own rows, all trained, lr 1e-4 with 200 warmup, FFNs at 0.3x) except 1,600 steps (105M tokens) with validation every 200; the decay is stretched to match.
- Held-out with the router fell at every validation: 2.958, 2.933, 2.927, 2.916, 2.912, 2.906, 2.899, 2.895, 2.892 (steps 0 to 1,600). Still falling 0.003 over the last 200 steps.
- Dev: math 2.807, code 2.830, medicine 2.714, history 3.320. vs 800 steps: -0.0067 (95% range -0.0075 to -0.0056). vs generalist: -0.238.
- Training time about 70 minutes; with the 1.5-minute build it fits the 2-hour final recipe. 1.16 A10G-h with dev scoring.
- Valid: 299,994,524 parameters, 148.6M active, causal. One seed.
