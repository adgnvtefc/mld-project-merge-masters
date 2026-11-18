"""
Diagnostics, no model: the size of each task vector (tau = model - seed) and the cosine similarity between
every pair, overall and per tensor kind. Tables go to train.log; launch with --no-eval.

    python3 training_scripts/merging/task_vector_stats.py --out runs/x
"""

import argparse
import itertools
import json

import torch

import merging as m

KINDS = {"embedding": ("wte", "wpe"), "attention": ("attn.",), "mlp": ("mlp.",), "layernorm": ("ln_",)}


def kind_of(name):
    return next(k for k, keys in KINDS.items() if any(key in name for key in keys))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    log = m.Logger(args.out)
    _, seed, taus = m.load_task_vectors(m.SOURCES, args.device)

    def flat(params, kind=None):
        return torch.cat([v.flatten() for k, v in params.items() if kind is None or kind_of(k) == kind]).double()

    stats = {"norm": {}, "relative_norm": {}, "cosine": {}}
    for kind in [None, *KINDS]:
        label = kind or "all"
        base = flat(seed, kind).norm().item()
        for s in m.SOURCES:
            norm = flat(taus[s], kind).norm().item()
            stats["norm"][f"{s}/{label}"] = norm
            stats["relative_norm"][f"{s}/{label}"] = norm / base
        for a, b in itertools.combinations(m.SOURCES, 2):
            va, vb = flat(taus[a], kind), flat(taus[b], kind)
            stats["cosine"][f"{a}~{b}/{label}"] = (va @ vb / (va.norm() * vb.norm())).item()

    log("relative norm ||tau|| / ||seed||, by tensor kind:")
    log(f"{'source':<18}" + "".join(f"{k:>11}" for k in ["all", *KINDS]))
    for s in m.SOURCES:
        log(f"{s:<18}" + "".join(f"{stats['relative_norm'][f'{s}/{k}']:>11.4f}" for k in ["all", *KINDS]))
    log("cosine similarity of task vectors, by tensor kind:")
    log(f"{'pair':<34}" + "".join(f"{k:>11}" for k in ["all", *KINDS]))
    for a, b in itertools.combinations(m.SOURCES, 2):
        log(f"{a + ' ~ ' + b:<34}" + "".join(f"{stats['cosine'][f'{a}~{b}/{k}']:>11.4f}" for k in ["all", *KINDS]))
    log("json " + json.dumps(stats))


if __name__ == "__main__":
    main()
