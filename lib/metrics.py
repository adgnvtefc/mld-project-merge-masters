"""
Training curves for the dashboard. Call from a training script; outside `./lab run` it does nothing, so
scripts (including the final recipe) still run standalone.

    from lib import metrics
    metrics.log(step=step, tokens=tokens_seen, train_loss=loss, lr=lr)   # any numeric fields

Each call appends one JSON line to results/<run_id>/metrics.jsonl with the time since launch (elapsed_s).
"""

import json
import os
import time


def _plain(value):
    if hasattr(value, "item"):  # torch / numpy scalars
        value = value.item()
    return value


def log(**values):
    results = os.environ.get("LAB_RESULTS")
    if not results:
        return
    line = {"elapsed_s": round(time.time() - float(os.environ.get("LAB_START", time.time())), 1)}
    line.update({k: _plain(v) for k, v in values.items()})
    with open(os.path.join(results, "metrics.jsonl"), "a") as f:
        f.write(json.dumps(line) + "\n")
