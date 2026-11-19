"""
How much of each expert's own-domain lead survives when a share of its FFN neurons is removed.

For each expert: collect per-neuron statistics on its own pool train text (activation Gram matrix, mean, and
activation x gradient), rank neurons by each criterion, zero the lowest-ranked ones, optionally repair the layer
output, and score the expert's own domain on pool-val. The reported "retained" share is
(generalist - pruned) / (generalist - expert): 1 keeps the whole lead, 0 is no better than the generalist.
See README.md for the criteria, fixes and allocations. No model is saved; launch with --no-eval.

    python3 training_scripts/pruning/prune_sweep.py --out runs/x
"""

import argparse
import json
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # lets deterministic mode use cuBLAS (set before CUDA starts)

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "merging"))
import merging as m  # noqa: E402

VOCAB = 50257
GLOBAL_OK = ("activation", "taylor")  # scores comparable across layers
MIN_LAYER_KEEP = 0.05


def fc(i):
    return f"transformer.h.{i}.mlp.c_fc"


def proj(i):
    return f"transformer.h.{i}.mlp.c_proj"


def calibration_stats(model, params, domain, windows, batch_size, seed, device, deterministic=False):
    """See _calibration_stats. deterministic=True gives bit-identical statistics from run to run: PyTorch's
    deterministic mode (an error on any nondeterministic op) and the plain-math attention kernel, whose backward,
    unlike the fused kernels, does not accumulate in a varying order. The math kernel keeps every layer's attention
    matrix for the backward pass, so use batch_size 1 at 2,048 tokens."""
    if not deterministic:
        return _calibration_stats(model, params, domain, windows, batch_size, seed, device)
    from torch.nn.attention import SDPBackend, sdpa_kernel
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        with sdpa_kernel(SDPBackend.MATH):
            return _calibration_stats(model, params, domain, windows, batch_size, seed, device)
    finally:
        torch.use_deterministic_algorithms(previous)


TAYLOR_GROUP = 4  # windows per |sum h*g| term, fixed so the statistic does not depend on the batch size


def _calibration_stats(model, params, domain, windows, batch_size, seed, device):
    """Per layer: Gram E[h h^T] (fp64), mean E[h], and taylor: the sum over consecutive groups of TAYLOR_GROUP
    own-domain pool train windows of |sum of h*g over the group|."""
    m.set_params(model, params)
    for p in model.parameters():
        p.requires_grad_(False)
    model.transformer.wte.weight.requires_grad_(True)  # so the graph reaches every activation
    tokens = m.data.pool(domain, "train")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(tokens) - m.data.T - 1, size=windows)
    L, H = len(model.transformer.h), model.transformer.h[0].mlp.c_fc.out_features
    gram = [torch.zeros(H, H, dtype=torch.float64, device=device) for _ in range(L)]
    total = [torch.zeros(H, dtype=torch.float64, device=device) for _ in range(L)]
    taylor = [torch.zeros(H, dtype=torch.float64, device=device) for _ in range(L)]
    group = [torch.zeros(H, dtype=torch.float64, device=device) for _ in range(L)]
    captured = {}
    hooks = [block.mlp.c_proj.register_forward_hook(lambda _m, inp, _o, i=i: captured.__setitem__(i, inp[0]))
             for i, block in enumerate(model.transformer.h)]
    n = 0
    for b in range(0, windows, batch_size):
        xy = torch.from_numpy(np.stack([np.asarray(tokens[s:s + m.data.T + 1], dtype=np.int64)
                                        for s in starts[b:b + batch_size]])).to(device)
        with torch.autocast(device, dtype=torch.bfloat16):
            logits = model(xy[:, :-1])
        loss = F.cross_entropy(logits[..., :VOCAB].float().reshape(-1, VOCAB), xy[:, 1:].reshape(-1),
                               reduction="sum")
        hs = [captured[i] for i in range(L)]
        grads = torch.autograd.grad(loss, hs)
        for i, (h, g) in enumerate(zip(hs, grads)):
            h2 = h.detach().reshape(-1, H).float()
            gram[i] += (h2.T @ h2).double()
            total[i] += h2.sum(0).double()
            per_window = (h.detach().float() * g.float()).sum(1).to(torch.float64)  # [windows in batch, H]
            for w, row in enumerate(per_window):
                group[i] += row
                if (b + w + 1) % TAYLOR_GROUP == 0 or b + w + 1 == windows:
                    taylor[i] += group[i].abs()
                    group[i].zero_()
        n += h2.shape[0]
        captured.clear()
    for hook in hooks:
        hook.remove()
    model.transformer.wte.weight.requires_grad_(False)
    return {"gram": [g / n for g in gram], "mean": [t / n for t in total], "taylor": taylor, "tokens": n}


def scores(criterion, params, stats, L, seed):
    out = []
    for i in range(L):
        w_in = params[f"{fc(i)}.weight"].double()   # [H, C]
        w_out = params[f"{proj(i)}.weight"].double()  # [C, H]
        if criterion == "random":
            g = torch.Generator().manual_seed(seed * 100 + i)
            s = torch.rand(w_in.shape[0], generator=g, dtype=torch.float64).to(w_in.device)
        elif criterion == "magnitude":
            s = w_in.norm(dim=1) * w_out.norm(dim=0)
        elif criterion == "activation":
            s = stats["gram"][i].diagonal().clamp_min(0).sqrt() * w_out.norm(dim=0)
        elif criterion == "taylor":
            s = stats["taylor"][i]
        else:
            raise ValueError(criterion)
        out.append(s)
    return out


def silent(stats, params, L, threshold):
    """Per layer: boolean mask of neurons whose output contribution (activation RMS x c_proj column norm) is below
    `threshold` times the layer's median contribution."""
    out = []
    for i, s in enumerate(scores("activation", params, stats, L, 0)):
        out.append(s < threshold * s.median())
    return out


def keep_masks(layer_scores, keep, allocation):
    H = layer_scores[0].numel()
    if allocation == "layer":
        k = max(1, round(keep * H))
        return [torch.zeros(H, dtype=torch.bool, device=s.device).scatter_(0, s.topk(k).indices, True)
                for s in layer_scores]
    # global: one ranking over all layers, with at least MIN_LAYER_KEEP of every layer kept
    floor = max(1, round(MIN_LAYER_KEEP * H))
    masks = [torch.zeros(H, dtype=torch.bool, device=s.device).scatter_(0, s.topk(floor).indices, True)
             for s in layer_scores]
    budget = round(keep * H * len(layer_scores)) - floor * len(layer_scores)
    if budget > 0:
        flat = torch.cat([s.masked_fill(mk, -float("inf")) for s, mk in zip(layer_scores, masks)])
        chosen = torch.zeros_like(flat, dtype=torch.bool).scatter_(0, flat.topk(budget).indices, True)
        masks = [mk | c for mk, c in zip(masks, chosen.split(H))]
    return masks


def pruned_params(params, masks, fix, stats, ridge=1e-6):
    out = dict(params)
    for i, keep in enumerate(masks):
        w = params[f"{proj(i)}.weight"].double()  # [C, H]
        b = params[f"{proj(i)}.bias"].double()
        drop = ~keep
        if fix == "lstsq":  # solved even when nothing is dropped, so the keep-1.0 sanity check tests it
            # augment h with a constant 1; refit [W_K, b] to reproduce [W, b] h~ on the calibration text
            H = w.shape[1]
            G = torch.zeros(H + 1, H + 1, dtype=torch.float64, device=w.device)
            G[:H, :H], G[:H, H], G[H, :H], G[H, H] = stats["gram"][i], stats["mean"][i], stats["mean"][i], 1.0
            kk = torch.cat([keep, torch.ones(1, dtype=torch.bool, device=w.device)])
            A = G[kk][:, kk]
            A = A + ridge * A.diagonal().mean() * torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
            target = torch.cat([w, b[:, None]], 1) @ G[:, kk]  # [C, K+1]
            sol = torch.linalg.solve(A, target.T).T
            w_new = torch.zeros_like(w)
            w_new[:, keep] = sol[:, :-1]
            b_new = sol[:, -1]
        else:
            w_new = w * keep
            b_new = b + (w[:, drop] @ stats["mean"][i][drop] if fix == "bias" else 0.0)
        out[f"{proj(i)}.weight"] = w_new.to(params[f"{proj(i)}.weight"].dtype)
        out[f"{proj(i)}.bias"] = b_new.to(params[f"{proj(i)}.bias"].dtype)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--experts", default=",".join(m.DOMAINS))
    parser.add_argument("--keeps", default="0.9,0.8,0.65,0.5,0.35,0.2")
    parser.add_argument("--criteria", default="random,magnitude,activation,taylor")
    parser.add_argument("--fixes", default="none,bias,lstsq")
    parser.add_argument("--allocations", default="layer,global")
    parser.add_argument("--calib-windows", type=int, default=128, help="own-domain pool train windows per expert")
    parser.add_argument("--val-windows", type=int, default=64, help="pool-val windows per domain")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--reference-models", default="",
                        help="also measure silent neurons of these models (e.g. seed,generalist) on each domain")
    parser.add_argument("--silent-threshold", type=float, default=0.01,
                        help="a neuron is silent below this share of its layer's median contribution")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    log = m.Logger(args.out)
    log(f"settings: {json.dumps(vars(args))}")
    torch.manual_seed(args.seed)
    keeps = m.floats(args.keeps)
    header, gen = m.load_checkpoint("generalist", args.device)
    model = m.new_model(header, args.device)
    L = len(model.transformer.h)
    results, kept_sets, step = {}, {}, 0
    for d in [x for x in args.experts.split(",") if x]:
        _, expert = m.load_checkpoint(f"expert_{d}", args.device)
        base = m.score(model, gen, args.val_windows, domains=[d])[d]
        full = m.score(model, expert, args.val_windows, domains=[d])[d]
        log(f"{d}: pool-val generalist {base:.4f}, expert {full:.4f}, lead {base - full:.4f}")
        stats = calibration_stats(model, expert, d, args.calib_windows, args.batch_size, args.seed, args.device)
        log(f"{d}: calibration on {stats['tokens']:,} own-domain pool train tokens")
        check = m.score(model, pruned_params(expert, keep_masks(scores("activation", expert, stats, L, args.seed),
                                                                   1.0, "layer"), "lstsq", stats),
                        args.val_windows, domains=[d])[d]
        log(f"{d}: sanity, keep 1.0 + lstsq refit {check:.4f} (expert {full:.4f})")
        quiet = {f"expert_{d}": silent(stats, expert, L, args.silent_threshold)}
        for ref in [x for x in args.reference_models.split(",") if x]:
            _, ref_params = m.load_checkpoint(ref, args.device)
            ref_stats = calibration_stats(model, ref_params, d, args.calib_windows, args.batch_size, args.seed,
                                          args.device)
            quiet[ref] = silent(ref_stats, ref_params, L, args.silent_threshold)
            del ref_params, ref_stats
        log(f"{d}: share of neurons silent on {d} text (contribution < {args.silent_threshold} x layer median), "
            "by layer:")
        for name, masks in quiet.items():
            log(f"  {name:<18}" + "".join(f"{float(q.float().mean()):>6.2f}" for q in masks)
                + f"  | all {float(torch.cat(masks).float().mean()):.3f}")
        for name, masks in quiet.items():
            if name != f"expert_{d}":
                mine = quiet[f"expert_{d}"]
                both = sum(int((a & b).sum()) for a, b in zip(mine, masks))
                log(f"  silent in expert_{d} and in {name}: {both:,} of {sum(int(a.sum()) for a in mine):,}")
        res_silent = {name: [float(q.float().mean()) for q in masks] for name, masks in quiet.items()}
        res = {"generalist": base, "expert": full, "curves": {}, "silent": res_silent}
        for criterion in [c for c in args.criteria.split(",") if c]:
            s = scores(criterion, expert, stats, L, args.seed)
            for allocation in [a for a in args.allocations.split(",") if a]:
                if allocation == "global" and criterion not in GLOBAL_OK:
                    continue
                for keep in keeps:
                    masks = keep_masks(s, keep, allocation)
                    if criterion == "activation" and allocation == "layer" and keep == 0.65:
                        kept_sets[d] = [mk.cpu().numpy() for mk in masks]
                    for fix in [f for f in args.fixes.split(",") if f]:
                        loss = m.score(model, pruned_params(expert, masks, fix, stats), args.val_windows,
                                       domains=[d])[d]
                        retained = (base - loss) / (base - full)
                        name = f"{criterion}/{allocation}/{fix}"
                        res["curves"].setdefault(name, {})[str(keep)] = {"loss": loss, "retained": retained}
                        if allocation == "global":
                            res["curves"][name][str(keep)]["layer_keep"] = [round(float(mk.float().mean()), 3)
                                                                            for mk in masks]
                        m.metrics.log(step=step, expert=d, criterion=criterion, allocation=allocation, fix=fix,
                                      keep=keep, loss=loss, retained=retained)
                        step += 1
        results[d] = res
        log(f"{d}: share of the lead retained (rows: criterion/allocation/fix; columns: share of neurons kept)")
        log(f"  {'':<28}" + "".join(f"{k:>8}" for k in keeps))
        for name, curve in res["curves"].items():
            log(f"  {name:<28}" + "".join(f"{curve[str(k)]['retained']:>8.2f}" for k in keeps))
        for name, curve in res["curves"].items():
            if "global" in name and name.endswith("/lstsq"):
                log(f"  {name} per-layer keep at 0.65: {curve['0.65']['layer_keep']}")
        del expert, stats
        torch.cuda.empty_cache()
    if len(kept_sets) > 1:
        log("overlap of kept neurons between experts (activation/layer, keep 0.65): mean Jaccard over layers; "
            f"random sets would give {0.65 / (2 - 0.65):.2f}")
        names = list(kept_sets)
        overlap = {}
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                j = [float((x & y).sum() / (x | y).sum()) for x, y in zip(kept_sets[names[a]], kept_sets[names[b]])]
                overlap[f"{names[a]}~{names[b]}"] = j
                log(f"  {names[a]:>8} ~ {names[b]:<8} {np.mean(j):.2f} (layers {min(j):.2f}-{max(j):.2f})")
        results["overlap"] = overlap
    log("json " + json.dumps(results))


if __name__ == "__main__":
    main()
