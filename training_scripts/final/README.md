# final
Description: the submitted recipe, one standalone script from the provided files to --out in under 2 h
Owner: both
Status: recipe chosen, script not written

The recipe must run without ./lab (staff run it directly), fix every seed, and carry over only settings
written into the script (at most 1 MB). Time it end to end on a fresh g5.xlarge with no network.

Chosen recipe (November 21, 2026): Pruned Expert Stacks + own rows + distillation, 1,600 steps, dev 2.8880
(`results/20261121-0736_routed_heal-all-rows8k-ffnlr03-kd05-1600_hz`). It is `training_scripts/routed/build.py`'s
model plus `training_scripts/routed/heal.py` with `--train all --rank 32 --top-rows 8192 --commit 224 --lr 1e-4
--warmup 200 --ffn-lr-mult 0.3 --kd 0.5 --steps 1600`. Measured 99 minutes on our A10G (build 2.2, training
96.9), which leaves about 20 minutes for setup on a fresh machine. See `results_summary.md`.
