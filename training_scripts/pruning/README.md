# pruning
Description: structured pruning of FFN neurons in the provided experts, to free parameters for routed designs
Owner: hz
Status: active

A neuron is one FFN hidden unit: its row of `mlp.c_fc` (and bias) and its column of `mlp.c_proj`. Pruned
neurons are zeroed in place, so the model keeps its shape; this scores the same as physically removing
them. Everything is scored on pool-val (lib/data.py), never on dev. Calibration uses pool train.

| Script | Variant | What it does |
|---|---|---|
| `prune_sweep.py` | `sweep` | own-domain pool-val loss of each expert with a share of its FFN neurons removed, over criteria, output fixes, allocations and keep fractions (no model; --no-eval) |

Criteria (score per neuron; low scores are pruned):

- `random`: seeded random order (control)
- `magnitude`: norm of the neuron's c_fc row times norm of its c_proj column
- `activation`: RMS of the neuron's activation on own-domain pool text times its c_proj column norm (the size of
  its contribution to the block output)
- `taylor`: |sum of activation x gradient of the loss| over each batch of 4 windows, summed over batches: a
  first-order estimate of the loss change from removing it

Fixes (after removing neurons):

- `none`: nothing
- `bias`: add the removed neurons' mean output to the c_proj bias
- `lstsq`: refit the kept c_proj columns and bias by least squares to reproduce the full layer's output on the
  calibration text (closed form, from the activation Gram matrix)

Allocations: `layer` keeps the same share in every layer; `global` ranks neurons across all layers (only for
`activation` and `taylor`, whose scores are comparable across layers), with at least 5% kept per layer.
