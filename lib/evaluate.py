"""
Score a finished run on dev, in two passes:
1. the official code/create_submission.py on runs/<run_id>/ (it writes submission.log for Gradescope);
2. the same windows again, keeping the loss of every 2,048-token window (results/<run_id>/window_losses.npz),
   which the dashboard's paired bootstrap needs. Its mean must match pass 1.

    python3 -m lib.evaluate <run_id> [--batch 4]

The official numbers, plus the check, go into run.json under "eval". Needs a GPU.
"""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

from lib.common import DOMAINS, RESULTS, ROOT, RUNS, parse_run_id, update_run, utc_now

sys.path.insert(0, os.path.join(ROOT, "code"))
import create_submission as official  # noqa: E402
import data  # noqa: E402


@torch.no_grad()
def window_losses(model, tokens, batch, device):
    """Mean next-token cross-entropy of each non-overlapping window, exactly as official.file_loss slices them."""
    T, V = official.T, official.VOCAB
    n = (len(tokens) - 1) // T
    tokens = torch.from_numpy(tokens[:n * T + 1].astype("int64")).to(device)
    out = []
    for start in range(0, n, batch):
        b = min(batch, n - start)
        chunk = tokens[start * T:(start + b) * T + 1]
        x, y = chunk[:-1].view(b, T), chunk[1:].view(b, T)
        with torch.autocast(device, dtype=torch.bfloat16):
            logits = official.logits_of(model, x)
        ce = F.cross_entropy(logits[..., :V].float().reshape(-1, V), y.reshape(-1), reduction="none")
        out.append(ce.view(b, T).mean(dim=1).cpu())
    return torch.cat(out).numpy()


def load_solution(directory, device):
    """The run's model, loaded the way create_submission.py does (model_def.py may import code/ and its own dir)."""
    sys.path.insert(0, directory)
    spec = importlib.util.spec_from_file_location("model_def", os.path.join(directory, "model_def.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_model(directory, device).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_id")
    parser.add_argument("--batch", type=int, default=4, help="windows per forward pass")
    args = parser.parse_args()
    parse_run_id(args.run_id)
    out_dir = os.path.join(RUNS, args.run_id)
    results_dir = os.path.join(RESULTS, args.run_id)
    if not os.path.exists(os.path.join(out_dir, "model_def.py")):
        raise SystemExit(f"runs/{args.run_id}/model_def.py not found: is this the host the run ran on?")

    # ---- 1. Official score -------------------------------------------------------------------------------
    subprocess.run([sys.executable, os.path.join(ROOT, "code", "create_submission.py"), out_dir,
                    "--batch", str(args.batch)], check=True, cwd=ROOT)
    shutil.copy(os.path.join(out_dir, "submission.log"), results_dir)
    with open(os.path.join(out_dir, "submission.log")) as f:
        result = json.loads([line for line in f if line.startswith("RESULT ")][-1][len("RESULT "):])

    # ---- 2. Per-window losses ----------------------------------------------------------------------------
    device = "cuda"
    model = load_solution(out_dir, device)
    windows = {d: window_losses(model, data.read_tokens(os.path.join(ROOT, "dev", f"{d}.bin")), args.batch, device)
               for d in DOMAINS}
    np.savez_compressed(os.path.join(results_dir, "window_losses.npz"), **windows)
    gap = max(abs(float(windows[d].mean()) - result["losses"][d]) for d in DOMAINS)
    print(f"per-window means match the official score within {gap:.2e}", flush=True)

    update_run(results_dir, eval={**result, "window_mean_gap": gap, "evaluated_at": utc_now()})


if __name__ == "__main__":
    main()
