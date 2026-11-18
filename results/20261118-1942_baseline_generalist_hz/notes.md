# 20261118-1942_baseline_generalist_hz

Hypothesis: Reference: the provided generalist, scored unchanged (handout: 3.156).

Status: done | dev mean 3.1555

## Outcome

Reproduces the handout's generalist score: dev mean 3.1555 vs 3.156 listed. Per domain: math 3.1415,
code 3.1689, medicine 2.7868, history 3.5248. This is the reference for "Δ vs generalist" on the dashboard.

## Observations

- First run through the full pipeline (launch, official eval, per-window dump, pull); the per-window mean
  matches the official score to 1.9e-7.
- 125.2M parameters, about 123.5M active, causal. No training, so 0 A10G-hours.
- Host: plain Ubuntu 26.04 g5.xlarge, NVIDIA driver 580.178.04 (installed by hand), torch 2.10.0+cu128.
