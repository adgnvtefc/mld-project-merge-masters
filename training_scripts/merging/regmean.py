"""
RegMean (Jin et al., 2023): merge each Linear layer in closed form so the merged layer best reproduces every
source model's outputs on that model's own inputs:

    W* = (sum_i G_i)^-1 sum_i G_i W_i,   G_i = X_i^T X_i   (inputs of the layer in model i)
    G_i -> alpha * G_i + (1 - alpha) * diag(G_i)          (off-diagonals shrunk, for stability)

X_i comes from --tokens of pool train data per source: each expert on its own domain, the generalist on an
equal mix. Everything that is not a block Linear weight (embeddings, LayerNorms, biases) is the plain mean
of the sources. alpha is picked on pool-val from --alpha.

    python3 training_scripts/merging/regmean.py --out runs/x
"""

import argparse
import json
import os

import numpy as np
import torch

import merging as m

LINEARS = ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sources", default=",".join(m.SOURCES), help="models to merge")
    parser.add_argument("--tokens", type=int, default=1_000_000, help="pool train tokens per source for the Grams")
    parser.add_argument("--alpha", default="0.5,0.7,0.9,1.0", help="grid for the off-diagonal scale")
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--val-windows", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def grams(model, domains, tokens, micro_batch, rng, device):
    """{'h.<i>.<linear>': X^T X} over `tokens` tokens of pool train, cycling through `domains`."""
    out, hooks = {}, []
    for i, block in enumerate(model.transformer.h):
        for name in LINEARS:
            module = block.get_submodule(name)
            key = f"{i}.{name}"

            def hook(_mod, inputs, key=key):
                x = inputs[0].reshape(-1, inputs[0].shape[-1]).float()
                g = x.T @ x
                out[key] = out[key] + g if key in out else g

            hooks.append(module.register_forward_pre_hook(hook))
    seen, step = 0, 0
    while seen < tokens:
        batch_domains = [domains[(step * micro_batch + j) % len(domains)] for j in range(micro_batch)]
        x, _ = m.data.batch(batch_domains, 2048, rng, "train", device)
        with torch.autocast(device, dtype=torch.bfloat16):
            model(x)
        seen += x.numel()
        step += 1
    for h in hooks:
        h.remove()
    return out


@torch.no_grad()
def regmean(params, gram, alpha, device):
    """Merged params: closed-form block Linear weights, mean of everything else."""
    names = list(params)
    merged = {k: sum(params[s][k] for s in names) / len(names) for k in params[names[0]]}
    for key in gram[names[0]]:
        layer, linear = key.split(".", 1)
        wname = f"transformer.h.{layer}.{linear}.weight"
        lhs, rhs = 0, 0
        for s in names:
            g = gram[s][key].double()
            g = alpha * g + (1 - alpha) * torch.diag(torch.diagonal(g))
            lhs = lhs + g
            rhs = rhs + g @ params[s][wname].double().T  # nn.Linear stores W as [out, in]
        lhs = lhs + 1e-6 * torch.diagonal(lhs).mean() * torch.eye(lhs.shape[0], device=device, dtype=lhs.dtype)
        merged[wname] = torch.linalg.solve(lhs, rhs).T.float()
    return merged


def main():
    args = parse_args()
    log = m.Logger(args.out)
    log(f"settings: {json.dumps(vars(args))}")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    sources = args.sources.split(",")

    header, params, gram = None, {}, {}
    for s in sources:
        header, params[s] = m.load_checkpoint(s, args.device)
        model = m.set_params(m.new_model(header, args.device), params[s])
        domains = [s.removeprefix("expert_")] if s.startswith("expert_") else m.DOMAINS
        gram[s] = grams(model, domains, args.tokens, args.micro_batch, rng, args.device)
        log(f"{s}: Grams of {len(gram[s])} Linear layers from {args.tokens:,} tokens of {', '.join(domains)}")
        del model

    model = m.new_model(header, args.device)
    best = None
    for i, alpha in enumerate(m.floats(args.alpha)):
        losses = m.score(model, regmean(params, gram, alpha, args.device), args.val_windows)
        m.metrics.log(step=i, alpha=alpha, **{f"val_{k}": v for k, v in losses.items()})
        log(f"alpha {alpha}: pool-val {m.fmt(losses)}")
        if best is None or losses["mean"] < best[1]["mean"]:
            best = (alpha, losses)

    alpha, losses = best
    log(f"best on pool-val: alpha {alpha} -> {m.fmt(losses)}")
    with open(os.path.join(args.out, "merge_settings.json"), "w") as f:
        json.dump({"scheme": "regmean", "best": {"alpha": alpha}, "pool_val": losses, "settings": vars(args)}, f,
                  indent=2)
    m.save_solution(args.out, header, regmean(params, gram, alpha, args.device), log)


if __name__ == "__main__":
    main()
