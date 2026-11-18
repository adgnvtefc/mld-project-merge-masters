"""
Diagnostic, no submission: how well routed low-rank expert deltas could do with perfect routing. For each
domain d and rank r, score on d's pool-val only the model

    W = W_generalist + U_d,r S_d,r V_d,r^T   (the 48 block weight matrices)

where U S V^T is the top-r SVD of expert d's change from the generalist ("full" uses the whole change).
--others says where every other tensor comes from, to show where the expert's knowledge lives:
generalist (all from the generalist), small (the expert's LayerNorms and biases, about 0.1M per expert)
or all (also the expert's token and position embeddings, about 40M per expert).

The mean over domains is the score of a model that sends every text to its own expert's piece; a real
routed model must also guess the domain from the text, so it can only do worse. Rank r costs about 0.147M
active parameters per token (r <= ~170 under the 150M limit). Launch with --no-eval.

    python3 training_scripts/merging/routed_oracle.py --out runs/x
"""

import argparse
import json

import torch

import merging as m


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ranks", default="16,64,128,160,full")
    parser.add_argument("--others", default="generalist,small,all")
    parser.add_argument("--val-windows", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    log = m.Logger(args.out)
    log(f"settings: {json.dumps(vars(args))}")
    ranks = args.ranks.split(",")
    rows = [(r, o) for o in args.others.split(",") for r in ranks]
    embeddings = ("transformer.wte.weight", "transformer.wpe.weight")

    header, base = m.load_checkpoint("generalist", args.device)
    model = m.new_model(header, args.device)
    table = {"generalist": m.score(model, base, args.val_windows)}
    table.update({f"{r}/{o}": {} for r, o in rows})
    table["expert"] = {}
    for step, d in enumerate(m.DOMAINS):
        _, expert = m.load_checkpoint(f"expert_{d}", args.device)
        table["expert"][d] = m.score(model, expert, args.val_windows, domains=[d])[d]
        svd = {k: torch.linalg.svd(expert[k] - base[k], full_matrices=False) for k in base if m.is_matrix(k)}
        for r, o in rows:
            params = {k: expert[k] if (o == "all" or (o == "small" and k not in embeddings)) else base[k]
                      for k in base}
            for k, (u, s, vh) in svd.items():
                n = s.numel() if r == "full" else int(r)
                params[k] = base[k] + (u[:, :n] * s[:n]) @ vh[:n]
            row = f"{r}/{o}"
            table[row][d] = m.score(model, params, args.val_windows, domains=[d])[d]
            m.metrics.log(step=step, domain=d, rank=-1 if r == "full" else int(r), others=o, val_loss=table[row][d])
            log(f"{d}: rank {r}, others from {o} -> {table[row][d]:.4f}")
        log(f"{d}: generalist {table['generalist'][d]:.4f}, expert {table['expert'][d]:.4f}")
        del expert, svd

    names = [f"{r}/{o}" for r, o in rows]
    for row in [*names, "expert"]:
        table[row]["mean"] = sum(table[row][d] for d in m.DOMAINS) / len(m.DOMAINS)
    log("pool-val loss with perfect routing (each domain scored with its own expert's piece):")
    log(f"{'model':<34}" + "".join(f"{d:>10}" for d in [*m.DOMAINS, "mean"]))
    for row in ["generalist", *names, "expert"]:
        label = {"generalist": "generalist (no piece)", "expert": "the expert itself"}.get(row) or \
            f"rank {row.split('/')[0]}, others: {row.split('/')[1]}"
        log(f"{label:<34}" + "".join(f"{table[row][d]:>10.4f}" for d in [*m.DOMAINS, "mean"]))
    log("json " + json.dumps(table))


if __name__ == "__main__":
    main()
