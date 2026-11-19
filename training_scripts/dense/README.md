# dense
Description: matched dense controls: the provided generalist trained with the same loop, data and steps as an MoE run, so a gain can be told apart from "more training on the pool"
Owner: ay
Status: active

Each control runs the MoE script with `--arch dense`, so the loop, data order, learning-rate schedule and
validation are the same as the MoE run it controls. Use the same `--steps`, `--batch-size`, `--lr` and
`--seed`, and name the MoE run with `--parent`.

| Script | Variant | Controls for |
|---|---|---|
| `../moe/heal.py --arch dense` | `heal-ce` | `moe` `heal-*` runs (Phase 3, next-token loss) |

    ./lab run --method dense --variant heal-ce --hypothesis "..." --parent <moe run id> \
        -- python training_scripts/moe/heal.py --arch dense --out '$LAB_OUT'
