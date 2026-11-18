"""
Training-free merges from task vectors (tau = model - seed), with coefficients picked on pool-val.

    --scheme ta    seed + lam_e * sum(tau_experts) + lam_g * tau_generalist, grid over (lam_e, lam_g)
    --scheme dare  the same grid, after DARE: drop each task-vector entry with prob p, rescale by 1/(1-p)
    --scheme ties  TIES on all sources: keep each tau's top-k fraction by magnitude (per tensor), elect a
                   sign per entry, average the entries that agree; then seed + lam * tau_ties, grid over lam

    python3 training_scripts/merging/task_arithmetic.py --scheme ta --out runs/x

Every grid point's pool-val loss goes to metrics.jsonl (step = grid index); the best point is saved.
dev is never read here.
"""

import argparse
import itertools
import json
import os

import torch

import merging as m


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scheme", required=True, choices=["ta", "dare", "ties"])
    parser.add_argument("--lam-experts", default="0,0.1,0.2,0.25,0.3,0.4,0.5", help="ta/dare: grid for lam_e")
    parser.add_argument("--lam-generalist", default="0,0.5,0.75,1.0", help="ta/dare: grid for lam_g")
    parser.add_argument("--lam", default="0.4,0.6,0.8,1.0,1.2,1.5", help="ties: grid for lam")
    parser.add_argument("--drop", type=float, default=0.9, help="dare: drop probability p")
    parser.add_argument("--keep", type=float, default=0.2, help="ties: fraction of each tau kept (top-k)")
    parser.add_argument("--val-windows", type=int, default=64, help="pool-val windows per domain per grid point")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def dare(tau, drop, generator):
    out = {}
    for k, t in tau.items():
        keep = torch.rand(t.shape, generator=generator, device=t.device) >= drop
        out[k] = t * keep / (1 - drop)
    return out


@torch.no_grad()
def ties(taus, keep):
    """One merged task vector from several (Yadav et al., 2023): trim, elect sign, disjoint mean."""
    merged = {}
    for k in taus[0]:
        stack = torch.stack([tau[k] for tau in taus])  # [n, ...]
        flat = stack.reshape(len(taus), -1).abs()
        drop_n = int(flat.shape[1] * (1 - keep))
        if drop_n > 0:
            threshold = flat.kthvalue(drop_n, dim=1).values.view(-1, *[1] * (stack.dim() - 1))
            stack = torch.where(stack.abs() > threshold, stack, torch.zeros_like(stack))
        sign = torch.sign(stack.sum(dim=0))
        agree = (torch.sign(stack) == sign) & (stack != 0)
        merged[k] = (stack * agree).sum(dim=0) / agree.sum(dim=0).clamp(min=1)
    return merged


def main():
    args = parse_args()
    log = m.Logger(args.out)
    log(f"settings: {json.dumps(vars(args))}")
    torch.manual_seed(args.seed)
    header, seed, taus = m.load_task_vectors(m.SOURCES, args.device)
    model = m.new_model(header, args.device)
    log(f"loaded seed and {len(taus)} task vectors on {args.device}")

    if args.scheme == "dare":
        g = torch.Generator(device=args.device).manual_seed(args.seed)
        taus = {name: dare(tau, args.drop, g) for name, tau in taus.items()}
        log(f"DARE: dropped {args.drop:.0%} of each task vector, rescaled the rest")
    if args.scheme == "ties":
        tau_ties = ties([taus[s] for s in m.SOURCES], args.keep)
        del taus
        log(f"TIES: kept top {args.keep:.0%} of each task vector per tensor, elected signs, disjoint mean")
        grid = [{"lam": lam} for lam in m.floats(args.lam)]
        build = lambda p: m.combine(seed, [(p["lam"], tau_ties)])  # noqa: E731
    else:
        experts_sum = {k: sum(taus[e][k] for e in m.EXPERTS) for k in seed}
        generalist = taus["generalist"]
        del taus
        grid = [{"lam_e": e, "lam_g": g} for e, g in itertools.product(m.floats(args.lam_experts),
                                                                        m.floats(args.lam_generalist))]
        build = lambda p: m.combine(seed, [(p["lam_e"], experts_sum), (p["lam_g"], generalist)])  # noqa: E731

    best = None
    for i, point in enumerate(grid):
        losses = m.score(model, build(point), args.val_windows)
        m.metrics.log(step=i, **point, **{f"val_{k}": v for k, v in losses.items()})
        log(f"grid {i + 1}/{len(grid)} {point}: pool-val {m.fmt(losses)}")
        if best is None or losses["mean"] < best[1]["mean"]:
            best = (point, losses)

    point, losses = best
    log(f"best on pool-val: {point} -> {m.fmt(losses)}")
    with open(os.path.join(args.out, "merge_settings.json"), "w") as f:
        json.dump({"scheme": args.scheme, "best": point, "pool_val": losses, "settings": vars(args)}, f, indent=2)
    m.save_solution(args.out, header, build(point), log)


if __name__ == "__main__":
    main()
