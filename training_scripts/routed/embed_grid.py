"""
How to spend each expert's embedding budget in Pruned Expert Stacks, with the 300M total fixed. No training.

Each configuration "r:K" gives every expert a rank-r fix of the shared embeddings plus its own word-embedding rows
for its K most frequent tokens; the FFN neurons take what is left of the 300M (fewer neurons when r or K grows).
For each: label-routed (oracle) pool-val loss, and the prefix router's pool-val loss at each commit point that stays
within 150M active parameters. dev is never read. No model is saved; launch with --no-eval.

    python3 training_scripts/routed/embed_grid.py --configs 128:0,64:4096,0:8192 --out runs/x
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

m, official = b.m, b.official
DOMAINS = m.DOMAINS


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--configs", default="128:0,64:4096,32:6144,32:8192,0:8192,64:8192,0:16384",
                        help="comma-separated rank:rows pairs")
    parser.add_argument("--commits", default="64,128,160,192,224")
    parser.add_argument("--val-windows", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    log = m.Logger(args.out)
    log(f"settings: {json.dumps(vars(args))}")
    dev = args.device
    val = {d: m.data.val_windows(d, args.val_windows) for d in DOMAINS}
    window = val["math"][0, :official.T].to(dev)
    commits = [int(c) for c in args.commits.split(",") if c]
    results = {}
    for step, spec in enumerate(c for c in args.configs.split(",") if c):
        rank, rows = (int(x) for x in spec.split(":"))
        model, config, _, digest, total = b.build_design_a(rank, "expert-avg", 128, 1, args.seed, "fp32", dev, log,
                                                           rows=rows)
        n_params = official.count_parameters(model)
        assert n_params <= official.MAX_PARAMETERS, n_params
        kept = sum(sum(w) for w in config["widths"]) / (4 * 12 * 3072)
        oracle = {}
        for e, d in enumerate(DOMAINS):
            model.force = e
            oracle[d] = m.data.val_loss(model, windows=args.val_windows, batch_size=4, device=dev, domains=[d])[d]
        model.force = None
        oracle["mean"] = float(np.mean([oracle[d] for d in DOMAINS]))
        router = {}
        for k in commits:
            model.commit = k
            active = official.active_parameters(model, window, dev)
            if active > 150e6:
                continue
            router[k] = {"losses": m.data.val_loss(model, windows=args.val_windows, batch_size=4, device=dev),
                         "active": active}
        best = min(router, key=lambda k: router[k]["losses"]["mean"])
        results[spec] = {"rank": rank, "rows": rows, "ffn_kept": kept, "parameters": n_params, "oracle": oracle,
                         "router": {str(k): v for k, v in router.items()}, "best_commit": best,
                         "fingerprint": digest.hexdigest()}
        m.metrics.log(step=step, config=spec, ffn_kept=kept, oracle_mean=oracle["mean"],
                      router_mean=router[best]["losses"]["mean"], commit=best, active=router[best]["active"])
        log(f"{spec}: FFN kept {kept:.1%}, {n_params:,} params | oracle {m.fmt(oracle)} | router at {best}: "
            f"{router[best]['losses']['mean']:.4f} ({router[best]['active'] / 1e6:.1f}M active)")
        del model
        torch.cuda.empty_cache()
    log(f"summary (pool-val, {args.val_windows} windows per domain, untrained):")
    log(f"  {'rank:rows':<12}{'FFN kept':>10}{'oracle':>10}{'router':>10}{'commit':>8}{'active':>10}")
    for spec, r in results.items():
        bc = str(r["best_commit"])
        log(f"  {spec:<12}{r['ffn_kept']:>10.1%}{r['oracle']['mean']:>10.4f}{r['router'][bc]['losses']['mean']:>10.4f}"
            f"{bc:>8}{r['router'][bc]['active'] / 1e6:>9.1f}M")
    log("json " + json.dumps(results))


if __name__ == "__main__":
    main()
