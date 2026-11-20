# 10-605 mini-project: one model from four experts

Course handout: `README`. Team workflow: the `lab-workflow` skill. Load it before writing a
training script, launching or monitoring a run, touching EC2, syncing files, or reading or
recording results.

Hard rules, even if the skill isn't loaded:
1. Run `./lab whereami` before running anything. Edit code only on a laptop, never on EC2.
2. Reach EC2 only through `./lab` (never raw ssh/scp/rsync with an IP).
3. Launch every experiment with `./lab run`, so it gets a run ID, commit and record in `results/`.
4. Never overwrite or delete a `results/<run_id>/` directory, and never touch `dev/` in training.

## Names

Use exactly these names in docs, notes, logs, pages and replies. Never use "design A/B", "Method 1/2/3" or
"Phase N"; they are confusing. Older run notes still contain them; the last table translates.

| Name (short form) | What it is | Code | Run IDs |
|---|---|---|---|
| **Weight Mixing** (Mixing) | Blend the five provided models' weights into one 125M model. No routing. Variants: Learned Mix, Simple Mix, RegMean, TIES, DARE, low-rank, TSV | `training_scripts/merging/` | `*_merging_*` |
| **Shared-Attention MoE** (MoE) | Shared embeddings and attention; the four experts' full FFNs; a per-token prefix router; a 768×768 embedding map per expert | `training_scripts/moe/` | `*_moe_*` |
| **Pruned Expert Stacks** (Stacks) | Four experts, each keeping its own attention and a pruned FFN; shared embeddings plus a low-rank fix per expert; a prefix router that commits each text to one stack | `training_scripts/pruning/` (sweeps), `training_scripts/routed/` (build, training) | `*_pruning_*`, `*_routed_*` |
| **Distillation** | Planned: train one dense model on the experts' predictions | — | — |

Other terms:
- **Generalist**, **Seed**, **Math / Code / Medicine / History expert**: the provided models.
- **Own-expert ceiling** (2.779 on dev): each domain scored by its own expert. **Gap closed** = share of
  generalist (3.156) minus ceiling.
- **Trained Generalist (control)**: the generalist trained with the same loop and data (`dense/`).
- **Untrained / trained**: before or after next-token training (scripts call it `heal.py`, variants `heal-*`).
  **FFNs frozen** = the experts' FFN weights do not train (`heal-shared`, `heal-frozen-ffn`); **all trained** =
  every weight trains (`heal-all`).
- **Dev** = the course scoring set (compare only). **Held-out** = pool-val, the last 2M tokens of each pool file.

| Old name in notes | Means |
|---|---|
| Method 1 | Weight Mixing |
| Method 2, MoE Phase 0–3 | Shared-Attention MoE (Phase 3 = its training) |
| Method 3 | Distillation |
| design A, pruning Phase 0–2 | Pruned Expert Stacks (and the sweeps that led to it) |
| design B | Shared-Attention MoE built from the pruning scripts, untrained, without embedding maps |
