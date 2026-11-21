"""
Train Pruned Expert Stacks with the next-token loss, matched to Ayush's MoE healing runs (training_scripts/moe/heal.py):
the same data stream (seeded pool train windows, one per domain per micro-batch, each routed to its own domain's
expert by label), steps, batch, learning-rate schedule, AdamW settings, gradient clipping and validation cadence.
So its dense control (20261119-0450_dense_heal-ce_ay) controls this run too.

    --train all         every weight
    --train frozen-ffn  each expert's pruned FFN frozen; embeddings, corrections, attention and LayerNorms train
                        (the analogue of heal.py --train shared, which froze the experts' MLPs)

The starting model is build.py's deterministic Pruned Expert Stacks. Validation on pool-val every --val-every steps: oracle
routing (label) and the prefix router at --commit. The saved solution routes with the prefix router; it has no
router parameters, so nothing needs refitting. dev is never read.

    python3 training_scripts/routed/heal.py --train frozen-ffn --out runs/x
"""

import argparse
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build as b  # noqa: E402  (sets CUBLAS_WORKSPACE_CONFIG before CUDA starts)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

m = b.m
DOMAINS = m.DOMAINS
VOCAB = 50257


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train", default="frozen-ffn", choices=["all", "frozen-ffn"])
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--top-rows", type=int, default=0, help="own word-embedding rows for each expert's top tokens")
    parser.add_argument("--source", default="expert-avg", choices=["expert-avg", "generalist"])
    parser.add_argument("--commit", type=int, default=224, help="prefix router commit point (build.py's choice)")
    parser.add_argument("--calib-windows", type=int, default=128)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=32, help="2,048-token sequences per step")
    parser.add_argument("--micro-batch", type=int, default=4, help="sequences per forward pass, a multiple of 4")
    parser.add_argument("--lr", type=float, default=3e-4, help="peak learning rate (before the muP factors)")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--kd", type=float, default=0.0,
                        help="distillation weight: loss = (1 - kd) * next-token CE + kd * KL(own unpruned expert || stack)")
    parser.add_argument("--ffn-lr-mult", type=float, default=1.0, help="learning-rate factor for the experts' FFNs")
    parser.add_argument("--weight-decay", type=float, default=0.3816, help="the provided models' own")
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--val-windows", type=int, default=32, help="pool-val windows per domain")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    assert args.batch_size % args.micro_batch == 0 and args.micro_batch % len(DOMAINS) == 0
    return args


def param_groups(model, weight_decay, ffn_lr_mult=1.0):
    """gpt.optimizer_groups' muP rules: hidden matrices lr / width_mult and decay x width_mult; embeddings plain;
    vectors undecayed. The embedding corrections (wte_u/v, wpe_u/v) are parts of embedding tables, so they go with
    the embeddings. Frozen tensors are left out."""
    w = model.config.width_mult
    spec = {"hidden": (1 / w, weight_decay * w), "embedding": (1.0, weight_decay), "vector": (1.0, 0.0),
            "ffn_hidden": (ffn_lr_mult / w, weight_decay * w), "ffn_vector": (ffn_lr_mult, 0.0)}
    groups = {k: [] for k in spec}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ".mlp." in name:
            kind = "ffn_vector" if p.dim() < 2 else "ffn_hidden"
        elif p.dim() < 2:
            kind = "vector"
        elif name in ("wte.weight", "wpe.weight") or name.endswith(("_u", "_v", "row_emb")):
            kind = "embedding"
        else:
            kind = "hidden"
        groups[kind].append(p)
    return [{"params": ps, "lr_mult": spec[k][0], "weight_decay": spec[k][1], "name": k}
            for k, ps in groups.items() if ps]


@torch.no_grad()
def domain_loss(model, windows, route=None, batch=4):
    device = next(model.parameters()).device
    total = 0.0
    for i in range(0, len(windows), batch):
        xy = windows[i:i + batch].to(device)
        x, y = xy[:, :-1], xy[:, 1:]
        kw = {} if route is None else {"route": torch.full((len(x),), route, device=device)}
        with torch.autocast(device.type, dtype=torch.bfloat16):
            logits = model(x, **kw)
        total += F.cross_entropy(logits[..., :VOCAB].float().reshape(-1, VOCAB), y.reshape(-1), reduction="sum").item()
    return total / (len(windows) * (windows.shape[1] - 1))


def with_mean(losses):
    return {**losses, "mean": float(np.mean([losses[d] for d in DOMAINS]))}


def main():
    args = parse_args()
    log = m.Logger(args.out)
    log(f"settings: {json.dumps(vars(args))}")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    device = args.device

    # ---- 1. Model: build.py's deterministic Stacks -------------------------------------------------------
    model, config, state, digest, _ = b.build_design_a(args.rank, args.source, args.calib_windows, 1, args.seed,
                                                       "fp32", device, log, rows=args.top_rows)
    model.commit = config["commit"] = args.commit
    del state
    if args.train == "frozen-ffn":
        for ex in model.experts:
            for block in ex.h:
                block.mlp.requires_grad_(False)
    teachers = []
    if args.kd > 0:  # each domain's own unpruned expert, frozen, in bf16
        for d in DOMAINS:
            header, params = m.load_checkpoint(f"expert_{d}", device)
            teacher = m.set_params(m.new_model(header, device), params).to(torch.bfloat16).eval()
            teacher.requires_grad_(False)
            teachers.append(teacher)
            del params
        log(f"distillation: weight {args.kd} on KL(own unpruned expert || stack)")
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    log(f"Stacks: {total:,} parameters, {sum(p.numel() for p in trainable):,} trained ({args.train})")

    # ---- 2. Validation on pool-val: oracle routing (label) and the prefix router --------------------------
    val = {d: m.data.val_windows(d, args.val_windows) for d in DOMAINS}

    def validate(step):
        model.eval()
        out = {"oracle": with_mean({d: domain_loss(model, val[d], route=e) for e, d in enumerate(DOMAINS)}),
               "router": with_mean({d: domain_loss(model, val[d]) for d in DOMAINS})}
        m.metrics.log(step=step, **{f"val_{d}": out["router"][d] for d in DOMAINS + ["mean"]},
                      val_oracle_mean=out["oracle"]["mean"])
        log(f"step {step}: pool-val oracle {m.fmt(out['oracle'])}")
        log(f"step {step}: pool-val router {m.fmt(out['router'])}")
        model.train()
        return out

    # ---- 3. Optimizer: AdamW with the muP groups, linear warmup then linear decay to zero ----------------
    groups = param_groups(model, args.weight_decay, args.ffn_lr_mult)
    log("optimizer groups: " + ", ".join(f"{g['name']} ({sum(p.numel() for p in g['params']):,}, lr x{g['lr_mult']:.3g})"
                                         for g in groups))
    optimizer = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=device == "cuda")

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        return args.lr * (1 - (step - args.warmup) / max(1, args.steps - args.warmup))

    # ---- 4. Train loop (the same data stream as moe/heal.py: one window per domain per micro-batch) -------
    history = [validate(0)]
    accum = args.batch_size // args.micro_batch
    domains = [DOMAINS[j % len(DOMAINS)] for j in range(args.micro_batch)]
    route = torch.tensor([j % len(DOMAINS) for j in range(args.micro_batch)], device=device)
    start, tokens = time.time(), 0
    for step in range(args.steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step) * group["lr_mult"]
        step_loss, per_domain = 0.0, torch.zeros(len(DOMAINS))
        for _ in range(accum):
            x, y = m.data.batch(domains, m.data.T, rng, "train", device)
            with torch.autocast(device, dtype=torch.bfloat16):
                logits = model(x, route=route)
            ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
            loss = ce.mean()
            if teachers:
                with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
                    t_logits = torch.cat([teachers[j % len(DOMAINS)](x[j:j + 1]) for j in range(len(x))])
                t_logp = F.log_softmax(t_logits[..., :VOCAB].float(), -1)
                s_logp = F.log_softmax(logits[..., :VOCAB].float(), -1)
                kl = (t_logp.exp() * (t_logp - s_logp)).sum(-1).mean()
                loss = (1 - args.kd) * loss + args.kd * kl
                del t_logits, t_logp
            (loss / accum).backward()
            step_loss += loss.item() / accum
            rows = ce.detach().view(len(x), -1).mean(dim=1).cpu()
            for j in range(len(x)):
                per_domain[j % len(DOMAINS)] += rows[j] / (accum * args.micro_batch / len(DOMAINS))
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        tokens += args.batch_size * m.data.T
        if (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            elapsed = time.time() - start
            rate = tokens / elapsed
            eta = (args.steps - step - 1) * elapsed / (step + 1)
            m.metrics.log(step=step + 1, tokens=tokens, train_loss=step_loss, lr=lr_at(step), tokens_per_s=rate,
                          **{f"train_{d}": per_domain[i].item() for i, d in enumerate(DOMAINS)})
            log(f"step {step + 1}/{args.steps} | loss {step_loss:.4f} | "
                + " ".join(f"{d} {per_domain[i]:.3f}" for i, d in enumerate(DOMAINS))
                + f" | lr {lr_at(step):.2e} | {rate / 1e3:.1f}K tok/s | eta {eta / 60:.1f} min")
        if (step + 1) % args.val_every == 0 and step + 1 != args.steps:
            history.append(validate(step + 1))

    # ---- 5. Final validation and save -------------------------------------------------------------------
    final = validate(args.steps)
    history.append(final)
    model.eval()
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, os.path.join(args.out, "model.pt"))
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=1)
    shutil.copy(os.path.join(HERE, "routed_model_def.py"), os.path.join(args.out, "model_def.py"))
    with open(os.path.join(args.out, "heal_settings.json"), "w") as f:
        json.dump({"settings": vars(args), "start_fingerprint": digest.hexdigest(), "final": final, "history": history,
                   "tokens": tokens, "train_seconds": round(time.time() - start)}, f, indent=2)
    log(f"saved {args.out} ({tokens / 1e6:.1f}M training tokens, {(time.time() - start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
