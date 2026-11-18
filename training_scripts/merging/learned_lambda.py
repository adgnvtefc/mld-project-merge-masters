"""
Learned merge coefficients ("supervised AdaMerging"): one coefficient per (source, tensor group), trained
with Adam on the real next-token loss of pool train data; the weights themselves stay fixed.

    merged[k] = seed[k] + sum_s lam[s, group(k)] * tau_s[k]

Groups: each block's ln_1, attn.c_attn, attn.c_proj, ln_2, mlp.c_fc, mlp.c_proj (weight and bias
together), plus wte, wpe and ln_f: 75 groups x 5 sources = 375 coefficients. The default start is the
generalist (lam = 1 for it, 0 for the experts); --init-experts / --init-generalist change it, for example to
the best task-arithmetic point. Each micro-batch holds one window per domain. pool-val is scored every
--val-every steps; the final coefficients are saved, with the table in merge_settings.json.

    python3 training_scripts/merging/learned_lambda.py --out runs/x
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call

import merging as m


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16, help="sequences per step (a multiple of 4)")
    parser.add_argument("--micro-batch", type=int, default=4, help="sequences per forward pass (one per domain)")
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--init-experts", type=float, default=0.0)
    parser.add_argument("--init-generalist", type=float, default=1.0)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--val-windows", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def group_of(name):
    """transformer.h.3.attn.c_attn.weight -> h.3.attn.c_attn; transformer.wte.weight -> wte."""
    parts = name.split(".")
    return ".".join(parts[1:-1]) if parts[1] == "h" else parts[1]


def main():
    args = parse_args()
    assert args.batch_size % args.micro_batch == 0 and args.micro_batch % len(m.DOMAINS) == 0
    log = m.Logger(args.out)
    log(f"settings: {json.dumps(vars(args))}")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    header, seed, taus = m.load_task_vectors(m.SOURCES, args.device)
    for p in list(seed.values()) + [t for tau in taus.values() for t in tau.values()]:
        p.requires_grad_(False)
    model = m.new_model(header, args.device)
    for p in model.parameters():
        p.requires_grad_(False)
    vocab = model.config.vocab_size

    groups = sorted({group_of(k) for k in seed})
    index = {g: i for i, g in enumerate(groups)}
    init = torch.tensor([args.init_generalist if s == "generalist" else args.init_experts for s in m.SOURCES])
    lam = init[:, None].repeat(1, len(groups)).to(args.device).requires_grad_(True)
    log(f"{lam.numel()} coefficients: {len(m.SOURCES)} sources x {len(groups)} groups")

    def merged():
        out = {}
        for k in seed:
            g = index[group_of(k)]
            w = seed[k] + sum(lam[i, g] * taus[s][k] for i, s in enumerate(m.SOURCES))
            out[k] = w[:vocab] if k == "transformer.wte.weight" else w
        return out

    def validate(step):
        with torch.no_grad():
            losses = m.score(model, merged(), args.val_windows)
        m.metrics.log(step=step, **{f"val_{k}": v for k, v in losses.items()})
        log(f"step {step}: pool-val {m.fmt(losses)}")
        return losses

    optimizer = torch.optim.Adam([lam], lr=args.lr)
    accum = args.batch_size // args.micro_batch
    validate(0)
    for step in range(1, args.steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = args.lr * min(1.0, step / args.warmup)
        step_loss = 0.0
        for _ in range(accum):
            domains = [m.DOMAINS[j % len(m.DOMAINS)] for j in range(args.micro_batch)]
            x, y = m.data.batch(domains, 2048, rng, "train", args.device)
            with torch.autocast(args.device, dtype=torch.bfloat16):
                logits = functional_call(model, merged(), (x,))
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
            (loss / accum).backward()
            step_loss += loss.item() / accum
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % args.log_every == 0 or step == args.steps:
            m.metrics.log(step=step, tokens=step * args.batch_size * 2048, train_loss=step_loss,
                          lam_experts_mean=lam[:4].mean().item(), lam_generalist_mean=lam[4].mean().item())
            log(f"step {step}/{args.steps} | loss {step_loss:.4f} | mean lam experts {lam[:4].mean().item():.3f} "
                f"generalist {lam[4].mean().item():.3f}")
        if step % args.val_every == 0 and step != args.steps:
            validate(step)

    losses = validate(args.steps)
    table = {s: {g: round(lam[i, j].item(), 5) for j, g in enumerate(groups)} for i, s in enumerate(m.SOURCES)}
    with open(os.path.join(args.out, "merge_settings.json"), "w") as f:
        json.dump({"scheme": "learned-lambda", "pool_val": losses, "lambda": table, "settings": vars(args)}, f,
                  indent=2)
    with torch.no_grad():
        final = {k: seed[k] + sum(lam[i, index[group_of(k)]] * taus[s][k] for i, s in enumerate(m.SOURCES))
                 for k in seed}
    m.save_solution(args.out, header, final, log)


if __name__ == "__main__":
    main()
