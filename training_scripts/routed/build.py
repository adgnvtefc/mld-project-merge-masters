"""
Build a routed design-A model from the provided checkpoints, with no training (see README.md), and save it as a
solution directory: model.pt, config.json, model_def.py.

Steps: prune each expert's FFN to the budget left by the embedding corrections (global taylor ranking on its own
pool train text, least-squares refit), compute each expert's rank-r embedding corrections, check the parameter
count, then on pool-val: the oracle-routed loss (each domain forced to its own expert), the router's accuracy by
position, and the loss and active parameters for each commit point. The commit point with the lowest pool-val loss
within 150M active parameters is saved. Dev is never read here.

    python3 training_scripts/routed/build.py --out runs/x
"""

import argparse
import hashlib
import json
import os
import shutil
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS for the calibration pass

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(SCRIPTS, "pruning"), os.path.join(SCRIPTS, "merging")]
import assemble_oracle as ao  # noqa: E402
import merging as m  # noqa: E402
import prune_sweep as ps  # noqa: E402
import routed_model_def as rmd  # noqa: E402

sys.path.insert(0, os.path.join(m.ROOT, "code"))
import create_submission as official  # noqa: E402

POSITIONS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 511]


def factors(shared, own, rank):
    """(U * S, Vh) of the rank-r truncated SVD of own - shared."""
    U, S, Vh = torch.linalg.svd((own - shared).float(), full_matrices=False)
    return (U[:, :rank] * S[:rank]).contiguous(), Vh[:rank].contiguous()


def physical_expert(params, masks, L):
    """{per-layer key: tensor} with pruned neurons removed: c_fc rows and bias, c_proj columns."""
    out = {}
    for i, keep in enumerate(masks):
        p = f"transformer.h.{i}."
        for name in ("ln_1.weight", "ln_1.bias", "ln_2.weight", "ln_2.bias", "attn.c_attn.weight", "attn.c_attn.bias",
                     "attn.c_proj.weight", "attn.c_proj.bias", "mlp.c_proj.bias"):
            out[f"h.{i}.{name}"] = params[p + name]
        out[f"h.{i}.mlp.c_fc.weight"] = params[p + "mlp.c_fc.weight"][keep]
        out[f"h.{i}.mlp.c_fc.bias"] = params[p + "mlp.c_fc.bias"][keep]
        out[f"h.{i}.mlp.c_proj.weight"] = params[p + "mlp.c_proj.weight"][:, keep]
    out["ln_f.weight"], out["ln_f.bias"] = params["transformer.ln_f.weight"], params["transformer.ln_f.bias"]
    return out


@torch.no_grad()
def routing_accuracy(model, windows, device):
    """Share of pool-val windows whose lowest-loss expert at each position is the true domain."""
    old = model.commit
    model.commit = POSITIONS[-1] + 1
    acc = {}
    for e, d in enumerate(m.DOMAINS):
        xy = m.data.val_windows(d, windows).to(device)
        hits = []
        for i in range(0, len(xy), 4):
            with torch.autocast(device, dtype=torch.bfloat16):
                model(xy[i:i + 4, :model.commit])
            hits.append((model.last_nll[:, :, POSITIONS].argmin(0) == e).float().cpu())
        acc[d] = torch.cat(hits).mean(0).tolist()
    model.commit = old
    return acc


def build_design_a(rank, source, calib_windows, calib_batch_size, seed, store_dtype, dev, log):
    """Design A from the provided checkpoints, deterministically. Returns (model on dev in eval mode, config dict,
    saved-state dict on CPU, weights fingerprint, total parameter budget). config["commit"] is 0; set it."""
    header, gen = m.load_checkpoint("generalist", dev)
    s = ao.sizes(header)
    L, H, V = s["L"], s["H"], s["V"]
    kept, total, active = ao.budget_a(s, rank)
    log(f"budget: {kept:,} FFN neurons per expert ({kept / (L * H):.1%}), total {total:,}, one expert's active "
        f"{active / 1e6:.2f}M")
    experts = {d: m.load_checkpoint(f"expert_{d}", dev)[1] for d in m.DOMAINS}
    if source == "generalist":
        shared = {k: gen[k] for k in ao.EMBEDDINGS}
    else:
        shared = {k: sum(experts[d][k] for d in m.DOMAINS) / len(m.DOMAINS) for k in ao.EMBEDDINGS}
    del gen

    work = m.new_model(header, dev)
    state = {"wte.weight": shared["transformer.wte.weight"][:V], "wpe.weight": shared["transformer.wpe.weight"]}
    widths = []
    for e, d in enumerate(m.DOMAINS):
        stats = ps.calibration_stats(work, experts[d], d, calib_windows, calib_batch_size, seed, dev,
                                     deterministic=True)
        masks = ps.keep_masks(ps.scores("taylor", experts[d], stats, L, seed), kept / (L * H), "global")
        params = ps.pruned_params(experts[d], masks, "lstsq", stats)
        widths.append([int(k.sum()) for k in masks])
        log(f"{d}: FFN widths {widths[-1]} ({sum(widths[-1]):,} neurons)")
        for k, v in physical_expert(params, masks, L).items():
            state[f"experts.{e}.{k}"] = v
        if rank:
            for name, key in (("wte", "transformer.wte.weight"), ("wpe", "transformer.wpe.weight")):
                u, v = factors(shared[key], experts[d][key], rank)
                state[f"experts.{e}.{name}_u"] = u[:V] if name == "wte" else u
                state[f"experts.{e}.{name}_v"] = v
        del stats, params
        torch.cuda.empty_cache()
    del work, experts

    cfg = m.gpt.config_from_header(header)
    config = {"gpt_config": {"block_size": cfg.block_size, "vocab_size": cfg.vocab_size, "n_layer": cfg.n_layer,
                             "n_head": cfg.n_head, "n_embd": cfg.n_embd, "width_mult": cfg.width_mult,
                             "attn_mult": cfg.attn_mult},
              "widths": widths, "rank": rank, "commit": 0, "domains": m.DOMAINS,
              "source": source}
    model = rmd.build(config)
    # evaluate exactly what is saved
    store = torch.bfloat16 if store_dtype == "bf16" else torch.float32
    state = {k: (v.to(store) if v.dim() >= 2 else v).cpu() for k, v in state.items()}
    model.load_state_dict({k: v.float() for k, v in state.items()})
    model = model.to(dev).eval()
    digest = hashlib.sha256()
    for k in sorted(state):
        digest.update(k.encode())
        digest.update(state[k].contiguous().view(torch.uint8).numpy().tobytes())
    log(f"weights fingerprint (sha256 of every saved tensor): {digest.hexdigest()[:16]}")
    return model, config, state, digest, total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rank", type=int, default=128, help="embedding correction rank per expert")
    parser.add_argument("--source", default="expert-avg", choices=["expert-avg", "generalist"],
                        help="shared embeddings")
    parser.add_argument("--commits", default="32,64,128", help="commit points to compare on pool-val")
    parser.add_argument("--calib-windows", type=int, default=128)
    parser.add_argument("--calib-batch-size", type=int, default=1,
                        help="calibration runs in deterministic mode, whose attention needs batch 1 to fit")
    parser.add_argument("--val-windows", type=int, default=256,
                        help="pool-val windows per domain (one misrouted window moves a 64-window mean by ~0.002)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--store-dtype", default="fp32", choices=["bf16", "fp32"],
                        help="dtype of saved matrices (autocast feeds bf16 to the matmuls anyway)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    log = m.Logger(args.out)
    log(f"settings: {json.dumps(vars(args))}")
    dev = args.device
    torch.manual_seed(args.seed)

    model, config, state, digest, total = build_design_a(args.rank, args.source, args.calib_windows,
                                                         args.calib_batch_size, args.seed, args.store_dtype, dev, log)
    n_params = official.count_parameters(model)
    log(f"parameters: {n_params:,} (budget {total:,}; limit {official.MAX_PARAMETERS:,})")
    assert n_params <= official.MAX_PARAMETERS

    # oracle routing: each domain forced to its own expert (should match the Phase 2 number)
    oracle = {}
    for e, d in enumerate(m.DOMAINS):
        model.force = e
        oracle[d] = m.data.val_loss(model, windows=args.val_windows, batch_size=args.batch_size, device=dev,
                                    domains=[d])[d]
    model.force = None
    oracle["mean"] = float(np.mean([oracle[d] for d in m.DOMAINS]))
    log("pool-val, oracle routing: " + m.fmt(oracle))
    m.metrics.log(step=0, config="oracle", **{f"val_{k}": v for k, v in oracle.items()})

    acc = routing_accuracy(model, args.val_windows, dev)
    log("router accuracy on pool-val by position (share of windows whose lowest-loss expert is the true domain):")
    log(f"  {'position':<10}" + "".join(f"{p:>7}" for p in POSITIONS))
    for d in m.DOMAINS:
        log(f"  {d:<10}" + "".join(f"{a:>7.2f}" for a in acc[d]))

    window = m.data.val_windows("math", 1)[0, :official.T].to(dev)
    results = {}
    for step, k in enumerate(int(c) for c in args.commits.split(",") if c):
        model.commit = k
        losses = m.data.val_loss(model, windows=args.val_windows, batch_size=args.batch_size, device=dev)
        active_k = official.active_parameters(model, window, dev)
        results[k] = {"losses": losses, "active": active_k}
        log(f"commit {k:>4}: pool-val " + m.fmt(losses) + f" | active {active_k / 1e6:.1f}M")
        m.metrics.log(step=step + 1, commit=k, active=active_k, **{f"val_{x}": v for x, v in losses.items()})
    ok = {k: r for k, r in results.items() if r["active"] <= 150e6}
    best = min(ok, key=lambda k: ok[k]["losses"]["mean"])
    model.commit = config["commit"] = best
    log(f"chosen commit point: {best} (pool-val mean {ok[best]['losses']['mean']:.4f})")
    cuts = tuple(sorted(c for c in {max(1, best // 2), best + 1, 512, 1024, 1536} if c < window.numel()))
    is_causal = official.causal(model, window, cuts=cuts)
    log(f"causal check (official test, cuts {cuts}, including inside the mixed prefix): {'yes' if is_causal else 'NO'}")
    assert is_causal

    torch.save(state, os.path.join(args.out, "model.pt"))
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=1)
    shutil.copy(os.path.join(HERE, "routed_model_def.py"), os.path.join(args.out, "model_def.py"))
    log(f"saved {args.out}/model.pt, config.json and model_def.py")
    log("json " + json.dumps({"oracle": oracle, "accuracy": acc, "positions": POSITIONS,
                              "commits": {str(k): r for k, r in results.items()}, "chosen": best,
                              "parameters": n_params, "fingerprint": digest.hexdigest()}))


if __name__ == "__main__":
    main()
