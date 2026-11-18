# analysis
Description: diagnostics that explain results (no models submitted); every run is launched with --no-eval
Owner: hz
Status: active

| Script | Variant | What it measures |
|---|---|---|
| `diagnostics.py --which paths` | `paths` | loss along straight lines between models (generalist to each expert, expert to expert) |
| `diagnostics.py --which scale` | `scale` | best output temperature and hidden-state size for merged models vs their sources |
| `diagnostics.py --which tokens` | `tokens` | which tokens, frequencies and positions carry each expert's gain over the generalist |
| `diagnostics.py --which swaps` | `swaps` | each expert's gain when one part (embeddings, LayerNorms, attention, MLP) is swapped in or out |
| `diagnostics.py --which slices` | `slices` | generalist and expert loss at 10 places in each pool file, to check the files are not ordered |

All of them use pool-val (lib/data.py), never dev. Tables go to train.log; the last line of each is the
same numbers as JSON.
