"""
Low-rank merge on the generalist (training-free, no extra parameters). For each expert e and each of the 48
block weight matrices, delta_e = W_e - W_generalist is cut to its top-r singular directions, and the four
cut deltas are added back at strength lam:

    W = W_generalist + lam * sum_e U_e,r S_e,r V_e,r^T

--orthogonalize applies Task Singular Vectors merging (TSV-M, Gargiulo et al., 2025): the four experts'
kept directions are concatenated and replaced by the nearest orthonormal sets (U <- P Q^T from the SVD of
[U_1..U_4], same for V), so the experts stop pushing along shared directions. This needs 4r <= 768.

Tensors that are not block matrices (embeddings, biases, LayerNorms) either stay at the generalist
(others 0) or move by lam times the experts' mean change (others 1). r, lam and others are all picked on
pool-val; dev is never read.

    python3 training_scripts/merging/low_rank.py --out runs/x [--orthogonalize]
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
    parser.add_argument("--orthogonalize", action="store_true", help="TSV-M: orthogonalize the kept directions")
    parser.add_argument("--ranks", default="16,64,128,192", help="grid for r (at most 192 with --orthogonalize)")
    parser.add_argument("--lam", default="0.1,0.25,0.5,0.75,1.0", help="grid for lam")
    parser.add_argument("--others", default="0,1", help="grid: 0 keeps non-matrix tensors at the generalist")
    parser.add_argument("--val-windows", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def nearest_orthonormal(a):
    """The matrix with orthonormal columns closest to a (polar factor: P Q^T from a = P S Q^T)."""
    p, _, qh = torch.linalg.svd(a, full_matrices=False)
    return p @ qh


@torch.no_grad()
def low_rank_sum(svds, r, orthogonalize):
    """sum_e U_e,r S_e,r V_e,r^T for one matrix, from each expert's (U, S, Vh) cut to r."""
    u = torch.cat([s[0][:, :r] for s in svds], dim=1)
    sv = torch.cat([s[1][:r] for s in svds])
    vh = torch.cat([s[2][:r] for s in svds], dim=0)
    if orthogonalize:
        u, vh = nearest_orthonormal(u), nearest_orthonormal(vh.T).T
    return (u * sv) @ vh


def main():
    args = parse_args()
    log = m.Logger(args.out)
    log(f"settings: {json.dumps(vars(args))}")
    torch.manual_seed(args.seed)
    ranks = [int(r) for r in m.floats(args.ranks)]
    if args.orthogonalize and max(ranks) * len(m.EXPERTS) > 768:
        raise SystemExit("--orthogonalize needs 4r <= 768, so r <= 192")

    header, base = m.load_checkpoint("generalist", args.device)
    svds, mean_delta = {k: [] for k in base if m.is_matrix(k)}, {k: 0 for k in base if not m.is_matrix(k)}
    for e in m.EXPERTS:
        _, params = m.load_checkpoint(e, args.device)
        for k in base:
            delta = params[k] - base[k]
            if m.is_matrix(k):
                u, s, vh = torch.linalg.svd(delta, full_matrices=False)
                svds[k].append((u[:, :max(ranks)], s[:max(ranks)], vh[:max(ranks)]))
            else:
                mean_delta[k] = mean_delta[k] + delta / len(m.EXPERTS)
        del params
        log(f"{e}: top-{max(ranks)} SVD of its change from the generalist in {len(svds)} matrices")

    model = m.new_model(header, args.device)
    best, step = None, 0
    for r in ranks:
        merged = {k: low_rank_sum(svds[k], r, args.orthogonalize) for k in svds}
        for lam, others in itertools.product(m.floats(args.lam), m.floats(args.others)):
            params = {k: base[k] + lam * merged[k] if k in merged else base[k] + others * lam * mean_delta[k]
                      for k in base}
            losses = m.score(model, params, args.val_windows)
            point = {"r": r, "lam": lam, "others": others}
            m.metrics.log(step=step, **point, **{f"val_{k}": v for k, v in losses.items()})
            log(f"r {r} lam {lam} others {others:g}: pool-val {m.fmt(losses)}")
            if best is None or losses["mean"] < best[1]["mean"]:
                best = (point, losses)
            step += 1
        del merged

    point, losses = best
    log(f"best on pool-val: {point} -> {m.fmt(losses)}")
    with open(os.path.join(args.out, "merge_settings.json"), "w") as f:
        json.dump({"scheme": "tsv" if args.orthogonalize else "low-rank", "best": point, "pool_val": losses,
                   "settings": vars(args)}, f, indent=2)
    merged = {k: low_rank_sum(svds[k], point["r"], args.orthogonalize) for k in svds}
    params = {k: base[k] + point["lam"] * merged[k] if k in merged
              else base[k] + point["others"] * point["lam"] * mean_delta[k] for k in base}
    m.save_solution(args.out, header, params, log)


if __name__ == "__main__":
    main()
