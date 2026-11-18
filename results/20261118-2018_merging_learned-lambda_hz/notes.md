# 20261118-2018_merging_learned-lambda_hz

Hypothesis: 375 per-layer coefficients learned on pool train beat the best single-coefficient merge.

Status: done | dev mean 3.1411

## Outcome

Dev 3.1411, the only merge below the generalist: Δ -0.0144, 95% CI [-0.0149, -0.0140] (paired bootstrap over dev windows). Supports beating the best single-coefficient merge (which was the generalist itself).

## Observations

- Per domain vs the generalist: math -0.0413, code -0.0243, history -0.0180, medicine +0.0258 (worse). CIs exclude 0 for every domain.
- Pool-val went 3.1887 -> 3.1800 over 300 steps (16 x 2,048 tokens each, 9.8M tokens, 398 s, 0.11 A10G-hours); most of the drop came in the first 100 steps.
- Coefficients stayed near the generalist start: mean 0.026 for expert vectors, 0.978 for the generalist vector. Full table in runs/<run_id>/merge_settings.json on the host.
- Init was the generalist (lam_e 0, lam_g 1), the script default; other inits were not tried.
