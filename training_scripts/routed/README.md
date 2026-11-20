# routed
Description: pruned experts with their own attention, shared embeddings plus per-expert low-rank corrections, and a causal prefix router
Owner: hz
Status: active

Pruned Expert Stacks (see CLAUDE.md, Names): the model from `training_scripts/pruning/`, made submittable. `build.py` makes it from the provided
models with no training, and writes a solution directory whose `model_def.py` is `routed_model_def.py`.

The model:

- **Shared:** word and position embeddings (the average of the four experts').
- **Per expert:** its own attention and LayerNorms; its FFN neurons pruned to fit the 300M total (global taylor
  ranking on its own pool train text, least-squares refit of each layer's output); a rank-r correction of the
  shared word and position embeddings toward its own (truncated SVD of the difference; also used by the tied
  output layer).
- **Router, no parameters:** for the first `commit` tokens of a sequence all four experts run, and the output mixes
  their predictions, weighted by each expert's likelihood of the text so far (a posterior with a uniform prior).
  At token `commit` the sequence commits to the expert with the lowest loss on the prefix, which then predicts
  every later token. Each expert needs its own attention state for earlier tokens, so the committed expert reruns
  the prefix. Active parameters per 2,048-token window: (4 x commit + 2048) / 2048 times one expert's.

The build is deterministic: calibration runs in PyTorch's deterministic mode with the plain-math attention kernel
(the fused kernels' backward accumulates in a varying order), and the log prints a fingerprint of the saved
weights, so two builds can be compared. `commit` is chosen on pool-val among `--commits`, on 256 windows per
domain (with 64, one misrouted window moves the mean by about 0.002); rank and embedding source come from the pruning runs
(`20261119-1707_pruning_assemble-oracle-ranks_hz`).

| Script | Variant | What it does |
|---|---|---|
| `build.py` | `design-a`, `design-a-commit` | builds the model, reports routing accuracy by position and pool-val loss for each commit point, saves the best |
| `build.py` | `design-a-det` | the same, deterministic (calibration in deterministic mode; prints a weights fingerprint) |
| `heal.py --train frozen-ffn` | `heal-frozen-ffn` | trains Stacks with the next-token loss, matched to `moe/heal.py` (same data stream, steps, schedule); each expert's FFN frozen |
| `heal.py --train all` | `heal-all` | the same with every weight trained |

Results: dev 3.0466 at commit 128 (`20261119-1845`), 3.0413 at commit 224 (`20261119-1850`, 148.5M active),
against 3.1555 for the generalist. See analysis.md section 8.
