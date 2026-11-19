"""
Phase 3, healing: train the starting MoE (common.build_initial: shared parts + the experts' MLPs, embedding
maps and router fitted in closed form) on pool train with the ordinary next-token loss, so the shared parts
and the copied expert MLPs fit together again.

    --arch moe     the MoE (default)
    --arch dense   the matched control: the provided generalist, trained with the same loop, data, steps and
                   learning rate. Without it, a gain could just be "more training on the pool".

What is trained (MoE):
    --train all      every weight; the experts' MLPs at --expert-lr-mult times the learning rate
    --train shared   embeddings, attention, LayerNorms and embedding maps; the experts' MLPs stay frozen
The router is never trained by the LM loss (it reads detached embeddings). The embeddings move during
training, so it is refitted in closed form before every validation and before saving.

Routing during training: every sequence goes to its own domain's expert (the label). Each micro-batch holds one
window per domain, so every expert sees the same number of tokens. Validation scores both oracle routing and
the prefix router on pool-val; the saved model routes with the prefix router. dev is never read.

    python3 training_scripts/moe/heal.py --out runs/x [--train shared] [--steps 800]
    python3 training_scripts/moe/heal.py --arch dense --out runs/y
"""

import argparse
import json
import os
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

import common as c
from common import DOMAINS, data, gpt, mm
from lib import metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--arch", default="moe", choices=["moe", "dense"])
    parser.add_argument("--shared", default="mean5", help="moe: source of the shared parts (Phase 1's best)")
    parser.add_argument("--maps", type=int, default=1, help="moe: 1 = embedding maps (Phase 1's best)")
    parser.add_argument("--lambda-settings", default="", help="moe, --shared learned: merge_settings.json")
    parser.add_argument("--train", default="all", choices=["all", "shared"], help="moe: which weights train")
    parser.add_argument("--expert-lr-mult", type=float, default=1.0, help="moe, --train all: experts' lr factor")
    parser.add_argument("--steps", type=int, default=800, help="optimizer steps")
    parser.add_argument("--batch-size", type=int, default=32, help="2,048-token sequences per step")
    parser.add_argument("--micro-batch", type=int, default=4, help="sequences per forward pass, a multiple of 4")
    parser.add_argument("--lr", type=float, default=3e-4, help="peak learning rate (before the muP factors)")
    parser.add_argument("--warmup", type=int, default=30, help="linear warmup steps, then linear decay to zero")
    parser.add_argument("--weight-decay", type=float, default=0.3816, help="the provided models' own")
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--val-windows", type=int, default=32, help="pool-val windows per domain")
    parser.add_argument("--router-windows", type=int, default=32, help="pool train windows per domain")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    assert args.batch_size % args.micro_batch == 0 and args.micro_batch % len(DOMAINS) == 0
    return args


def param_groups(model, weight_decay, expert_lr_mult):
    """gpt.optimizer_groups' muP groups (hidden matrices lr / width_mult and decay x width_mult; embeddings
    plain; vectors undecayed), with the experts' tensors split off at expert_lr_mult. Frozen tensors are left
    out."""
    w = model.config.width_mult
    spec = {"hidden": (1 / w, weight_decay * w), "embedding": (1.0, weight_decay), "vector": (1.0, 0.0),
            "expert_hidden": (expert_lr_mult / w, weight_decay * w), "expert_vector": (expert_lr_mult, 0.0)}
    groups = {k: [] for k in spec}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        expert = ".experts." in name
        if p.dim() < 2:
            kind = "expert_vector" if expert else "vector"
        elif name.startswith(("transformer.wte", "transformer.wpe")):
            kind = "embedding"
        else:
            kind = "expert_hidden" if expert else "hidden"
        groups[kind].append(p)
    return [{"params": ps, "lr_mult": spec[k][0], "weight_decay": spec[k][1], "name": k}
            for k, ps in groups.items() if ps]


class Dense(torch.nn.Module):
    """gpt.GPT with the MoE call signature (route is ignored), so both archs share the loop and scoring."""

    def __init__(self, model):
        super().__init__()
        self.inner = model
        self.config = model.config

    def forward(self, x, route=None):
        return self.inner(x)


def main():
    args = parse_args()
    log = c.Logger(args.out)
    log(f"settings: {json.dumps(vars(args))}")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    device = args.device

    # ---- 1. Model ---------------------------------------------------------------------------------------
    windows = c.router_windows(args.router_windows, args.seed) if args.arch == "moe" else None
    if args.arch == "moe":
        model, residuals = c.build_initial(args.shared, bool(args.maps), device, lambda_settings=args.lambda_settings,
                                           windows=windows, seed=args.seed)
        if residuals:
            log("map relative residuals: " + ", ".join(f"{d} {r:.3f}" for d, r in zip(DOMAINS, residuals)))
        model.router.requires_grad_(False)
        if args.train == "shared":
            for block in model.transformer.h:
                block.experts.requires_grad_(False)
    else:
        model = Dense(gpt.load(c.model_path("generalist"), device))
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())
    log(f"{args.arch}: {total:,} parameters, {sum(p.numel() for p in trainable):,} trained")

    # ---- 2. Validation on pool-val: oracle routing and the (refitted) prefix router ---------------------
    val = {d: data.val_windows(d, args.val_windows) for d in DOMAINS}

    def validate(step):
        model.eval()
        out = {}
        if args.arch == "moe":
            c.fit_router(model, windows, args.seed)  # the embeddings moved: refit before scoring
            out["oracle"] = c.with_mean({d: c.domain_loss(model, val[d], route=e) for e, d in enumerate(DOMAINS)})
            accuracy = mm.router_accuracy(model, [val[d][:, :-1] for d in DOMAINS])
            out["router_accuracy"] = accuracy["all"]
        out["router"] = c.with_mean({d: c.domain_loss(model, val[d]) for d in DOMAINS})
        metrics.log(step=step, **{f"val_{d}": out["router"][d] for d in DOMAINS + ["mean"]},
                    **({"val_oracle_mean": out["oracle"]["mean"], "router_accuracy": out["router_accuracy"]}
                       if "oracle" in out else {}))
        if "oracle" in out:
            log(f"step {step}: pool-val oracle {c.fmt(out['oracle'])}")
        log(f"step {step}: pool-val {'router' if args.arch == 'moe' else 'dense '} {c.fmt(out['router'])}"
            + (f" | router accuracy {out['router_accuracy']:.3f}" if "router_accuracy" in out else ""))
        model.train()
        return out

    # ---- 3. Optimizer: AdamW with the muP groups, linear warmup then linear decay to zero ----------------
    groups = param_groups(model.inner if args.arch == "dense" else model, args.weight_decay, args.expert_lr_mult)
    log("optimizer groups: " + ", ".join(f"{g['name']} ({sum(p.numel() for p in g['params']):,}, lr x{g['lr_mult']:.3g})"
                                         for g in groups))
    optimizer = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
                                  fused=device == "cuda")

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        return args.lr * (1 - (step - args.warmup) / max(1, args.steps - args.warmup))

    # ---- 4. Train loop ----------------------------------------------------------------------------------
    history = [validate(0)]
    accum = args.batch_size // args.micro_batch
    domains = [DOMAINS[j % len(DOMAINS)] for j in range(args.micro_batch)]
    route = torch.tensor([j % len(DOMAINS) for j in range(args.micro_batch)], device=device)  # the labels
    start, tokens = time.time(), 0
    for step in range(args.steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step) * group["lr_mult"]
        step_loss, per_domain = 0.0, torch.zeros(len(DOMAINS))
        for _ in range(accum):
            x, y = data.batch(domains, data.T, rng, "train", device)
            with torch.autocast(device, dtype=torch.bfloat16):
                logits = model(x, route=route)
            ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
            loss = ce.mean()
            (loss / accum).backward()
            step_loss += loss.item() / accum
            rows = ce.detach().view(len(x), -1).mean(dim=1).cpu()
            for j in range(len(x)):
                per_domain[j % len(DOMAINS)] += rows[j] / (accum * args.micro_batch / len(DOMAINS))
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        tokens += args.batch_size * data.T
        if (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            elapsed = time.time() - start
            rate = tokens / elapsed
            eta = (args.steps - step - 1) * elapsed / (step + 1)
            metrics.log(step=step + 1, tokens=tokens, train_loss=step_loss, lr=lr_at(step), tokens_per_s=rate,
                        **{f"train_{d}": per_domain[i].item() for i, d in enumerate(DOMAINS)})
            log(f"step {step + 1}/{args.steps} | loss {step_loss:.4f} | "
                + " ".join(f"{d} {per_domain[i]:.3f}" for i, d in enumerate(DOMAINS))
                + f" | lr {lr_at(step):.2e} | {rate / 1e3:.1f}K tok/s | eta {eta / 60:.1f} min")
        if (step + 1) % args.val_every == 0 and step + 1 != args.steps:
            history.append(validate(step + 1))

    # ---- 5. Final validation and save -------------------------------------------------------------------
    final = validate(args.steps)  # for the MoE, this also refits the router on the trained embeddings
    history.append(final)
    model.eval()
    if args.arch == "moe":
        mm.save_solution(model, args.out)
    else:
        gpt.save(model.inner, os.path.join(args.out, "model.bin"), gpt.read_header(c.model_path("generalist")))
        shutil.copy(os.path.join(c.ROOT, "solutions", "gpt_model_def.py"), os.path.join(args.out, "model_def.py"))
    with open(os.path.join(args.out, "heal_settings.json"), "w") as f:
        json.dump({"settings": vars(args), "final": final, "history": history, "tokens": tokens,
                   "train_seconds": round(time.time() - start)}, f, indent=2)
    log(f"saved {args.out} ({tokens / 1e6:.1f}M training tokens, {(time.time() - start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
