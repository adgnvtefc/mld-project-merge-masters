"""
Where does sharing the embeddings cost Pruned Expert Stacks, and which fix is worth its parameters? No training.

Builds the untrained Stacks (build.py, deterministic) and scores each domain with its own stack (label routing) on
pool-val, with the stack's word and position embeddings replaced in different ways:

    shared + fix        what the model uses: shared table + the stack's rank-r fix (input and output, tied)
    own                 the expert's own full tables, input and output (over budget: the ceiling)
    own input only      own tables for the input lookup; output stays shared + fix
    own output only     own word table for the output layer; input stays shared + fix
    own positions       own position table; word table stays shared + fix
    top-K rows          the expert's own rows for its K most frequent tokens (in its pool train text), shared + fix
                        for the rest; input and output, or input only

A lookup costs no active parameters; output rows do (K x 768 per token). Reports the parameters each option adds
per expert and the share of pool-val tokens the top K cover. dev is never read. No model is saved; --no-eval.

    python3 training_scripts/routed/embed_check.py --out runs/x
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build as b  # noqa: E402  (sets CUBLAS_WORKSPACE_CONFIG before CUDA starts)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

m = b.m
DOMAINS = m.DOMAINS
VOCAB = 50257


@torch.no_grad()
def stack_loss(model, e, windows, wte_in, wpe_in, wte_out, batch=4):
    ex, device = model.experts[e], next(model.parameters()).device
    total = 0.0
    for i in range(0, len(windows), batch):
        xy = windows[i:i + batch].to(device)
        x, y = xy[:, :-1], xy[:, 1:]
        with torch.autocast(device.type, dtype=torch.bfloat16):
            h = F.embedding(x, wte_in) + wpe_in[:x.shape[1]]
            for block in ex.h:
                h = block(h)
            logits = F.linear(ex.ln_f(h) / model.config.width_mult, wte_out)
        total += F.cross_entropy(logits.float().reshape(-1, VOCAB), y.reshape(-1), reduction="sum").item()
    return total / (len(windows) * (windows.shape[1] - 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--tops", default="1024,4096,16384", help="K values for the top-K own rows")
    parser.add_argument("--val-windows", type=int, default=128)
    parser.add_argument("--count-tokens", type=int, default=20_000_000, help="pool train tokens used to rank tokens")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    log = m.Logger(args.out)
    log(f"settings: {json.dumps(vars(args))}")
    dev = args.device
    model, config, _, _, _ = b.build_design_a(args.rank, "expert-avg", 128, 1, args.seed, "fp32", dev, log)
    C = model.config.n_embd
    tops = [int(k) for k in args.tops.split(",") if k]
    rows, out = {}, {}
    for e, d in enumerate(DOMAINS):
        _, ep = m.load_checkpoint(f"expert_{d}", dev)
        own_wte, own_wpe = ep["transformer.wte.weight"][:VOCAB].float(), ep["transformer.wpe.weight"].float()
        del ep
        ex = model.experts[e]
        fix_wte = model.wte.weight + (ex.wte_u @ ex.wte_v if ex.rank else 0)
        fix_wpe = model.wpe.weight + (ex.wpe_u @ ex.wpe_v if ex.rank else 0)
        counts = np.bincount(np.asarray(m.data.pool(d, "train")[:args.count_tokens], dtype=np.int64), minlength=VOCAB)
        order = torch.from_numpy(np.argsort(-counts[:VOCAB]).copy()).to(dev)
        val = m.data.val_windows(d, args.val_windows)
        val_counts = np.bincount(val[:, 1:].reshape(-1).numpy(), minlength=VOCAB)
        configs = {"shared + fix": (fix_wte, fix_wpe, fix_wte), "own": (own_wte, own_wpe, own_wte),
                   "own input only": (own_wte, own_wpe, fix_wte), "own output only": (fix_wte, fix_wpe, own_wte),
                   "own positions": (fix_wte, own_wpe, fix_wte)}
        for k in tops:
            ids = order[:k]
            mixed = fix_wte.clone()
            mixed[ids] = own_wte[ids]
            configs[f"top-{k} own rows"] = (mixed, fix_wpe, mixed)
            configs[f"top-{k} own rows, input only"] = (mixed, fix_wpe, fix_wte)
            out.setdefault("coverage", {}).setdefault(d, {})[k] = float(val_counts[ids.cpu().numpy()].sum() / val_counts.sum())
        for i, (name, (wi, pi, wo)) in enumerate(configs.items()):
            rows.setdefault(name, {})[d] = stack_loss(model, e, val, wi, pi, wo)
            m.metrics.log(step=e * 100 + i, domain=d, config=name, loss=rows[name][d])
        log(f"{d}: " + " | ".join(f"{n} {rows[n][d]:.4f}" for n in configs))
        del own_wte, own_wpe, fix_wte, fix_wpe, configs
        torch.cuda.empty_cache()

    cost = {"shared + fix": 0, "own": None, "own input only": None, "own output only": None, "own positions": 2048 * C}
    for k in tops:
        cost[f"top-{k} own rows"] = cost[f"top-{k} own rows, input only"] = k * C
    base = {d: rows["shared + fix"][d] for d in DOMAINS}
    log(f"pool-val, each domain scored by its own untrained stack (label routing), {args.val_windows} windows:")
    log(f"  {'embeddings':<32}" + "".join(f"{d:>10}" for d in DOMAINS) + f"{'mean':>10}{'vs shared+fix':>15}{'params/expert':>15}")
    for name, r in rows.items():
        r["mean"] = float(np.mean([r[d] for d in DOMAINS]))
        c = cost[name]
        log(f"  {name:<32}" + "".join(f"{r[d]:>10.4f}" for d in DOMAINS) + f"{r['mean']:>10.4f}"
            + f"{r['mean'] - np.mean(list(base.values())):>+15.4f}" + (f"{c / 1e6:>14.2f}M" if c else f"{'(over budget)' if c is None else '-':>15}"))
    for k in tops:
        log(f"  top-{k} tokens cover " + ", ".join(f"{d} {out['coverage'][d][k]:.1%}" for d in DOMAINS) + " of pool-val tokens")
    log("json " + json.dumps({"rows": rows, "cost": cost, "coverage": out["coverage"]}))


if __name__ == "__main__":
    main()
