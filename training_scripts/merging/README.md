# merging
Description: weight-space merges of the provided models (task arithmetic, TIES, DARE, RegMean, learned coefficients); no training of the weights
Owner: hz
Status: active

Task vectors are relative to the seed: tau = model - seed, for the four experts and the generalist.
Coefficients are always picked on pool-val (lib/data.py), never on dev.

| Script | Variant | What it does |
|---|---|---|
| `task_arithmetic.py --scheme ta` | `ta-grid` | seed + lam_e * sum(tau_experts) + lam_g * tau_generalist, grid over both |
| `task_arithmetic.py --scheme dare` | `dare` | same grid after DARE (drop p, rescale) |
| `task_arithmetic.py --scheme ties` | `ties` | TIES over all 5 task vectors, grid over lam |
| `regmean.py` | `regmean` | closed-form per-Linear merge from input Gram matrices, alpha picked on pool-val |
| `learned_lambda.py` | `learned-lambda` | 375 coefficients (source x tensor group) trained with Adam on pool train |
| `task_vector_stats.py` | `tv-stats` | norms and cosines of the task vectors (no model; launch with --no-eval) |
| `low_rank.py` | `low-rank` | generalist + lam * sum of each expert's top-r SVD change (no extra parameters) |
| `low_rank.py --orthogonalize` | `tsv` | the same, with the experts' kept directions orthogonalized (TSV-M) |
| `routed_oracle.py` | `routed-oracle` | diagnostic: each domain scored with its own expert's rank-r piece (no model; --no-eval) |

    ./lab run --method merging --variant ta-grid --hypothesis "..." \
        -- python training_scripts/merging/task_arithmetic.py --scheme ta --out '$LAB_OUT'

`merging.py` holds the shared loading, combining, scoring and saving code. Each run writes the chosen
settings to `runs/<run_id>/merge_settings.json` and prints them in train.log.
