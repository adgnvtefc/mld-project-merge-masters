# 20261119-0527_merging_learned-lambda_hz

Hypothesis: Re-run of 20261118-2018_merging_learned-lambda_hz after the pool-val fix (d725d3f: windows spread over the whole slice, not the first 64 in a row); same grid and settings. Original hypothesis: 375 per-layer coefficients learned on pool train beat the best single-coefficient merge. Training does not read pool-val, so the model and dev score should reproduce exactly.

Status: done | dev mean 3.1411

## Outcome

Same choice and same dev score as 20261118-2018_merging_learned-lambda_hz: the pool-val fix changed nothing that was decided. Dev 3.1411 (original 3.1411).

## Observations

- Chosen on fixed pool-val: step 300: pool-val math 3.0351 | code 3.0931 | medicine 2.8349 | history 3.5259 | mean 3.1222
- Original, on the biased windows: step 300: pool-val math 3.1635 | code 3.1994 | medicine 2.8353 | history 3.5218 | mean 3.1800
- Pool-val is now lower in math and code because the windows no longer sit on one GMAT reading guide; the merges are still worse than the generalist in every domain, so the same edge point wins.
- Reproducibility: dev mean 3.141069 vs 3.141050 in the original (+1.9e-5; per domain at most 9.4e-5). Not bit-identical: per-window differences have median 5.8e-4 and max 5.1e-3, from non-deterministic GPU arithmetic during training. Well inside the course's 0.005 tolerance.
- Final coefficients identical to 3 decimals (mean 0.026 for experts, 0.978 for the generalist); final train loss 3.0194 vs 3.0192.
