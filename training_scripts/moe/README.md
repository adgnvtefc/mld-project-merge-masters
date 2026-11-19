# moe
Description: one model with shared embeddings and attention and the four experts' MLPs, each token routed to one expert (top-1) by a causal prefix router
Owner: ay
Status: active

`moe_model.py` holds the model, the builder from the provided checkpoints, the closed-form router and
embedding-map fits, and save/load. It is also every MoE solution's `model_def.py`. Budget: ~295M total and
~124M active (one expert MLP per token), or ~297M total with the embedding maps.

| Script | Variant | What it does |
|---|---|---|
| `init_check.py` | `init-check` | no training: for each source of the shared parts, with and without embedding maps, scores the MoE on pool-val with perfect routing and with the prefix router; saves the best as a solution (scored on dev) |
| `test_moe.py` | (not a run) | CPU checks on a tiny config: equivalence to gpt.GPT, dispatch, causality, active parameters, save/load, router fit |

    # on a laptop, before pushing (any Python with torch 2.10.0)
    python training_scripts/moe/test_moe.py

    ./lab run --method moe --variant init-check --hypothesis "..." \
        -- python training_scripts/moe/init_check.py --out '$LAB_OUT'
    # add --lambda-settings runs/<learned-lambda run>/merge_settings.json to include the learned merge

Every number used to choose is on pool-val (lib/data.py). dev is never read here.
