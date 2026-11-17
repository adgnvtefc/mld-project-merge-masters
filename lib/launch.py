"""
Run one experiment under the recorder: it gets a results/<run_id>/ record, its output goes to runs/<run_id>/,
and on success it is scored on dev. `./lab run` calls this on the host (inside tmux); you rarely call it
yourself.

    python3 -m lib.launch --run-id 20261118-1930_moe_ridge-router_hz --hypothesis "..." \
        -- python3 training_scripts/moe/train.py --out '$LAB_OUT'

The command runs from the repository root with these environment variables:
    LAB_RUN_ID   the run ID
    LAB_OUT      runs/<run_id>/, the solution output directory ('$LAB_OUT' in the command is replaced by it)
    LAB_RESULTS  results/<run_id>/, where lib.metrics.log appends metrics.jsonl
"""

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

from lib.common import (METHODS, RESULTS, ROOT, RUNS, git, git_commit, git_dirty_files, host_identity,
                        parse_run_id, read_json, update_run, utc_now)

HEARTBEAT_EVERY_S = 60


def last_metrics(results_dir):
    """The step/tokens of the newest metrics line, for the progress column."""
    try:
        with open(os.path.join(results_dir, "metrics.jsonl"), "rb") as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        return None
    for line in reversed(lines):
        try:
            last = json.loads(line)
        except ValueError:
            continue
        return {k: last[k] for k in ("step", "tokens", "elapsed_s") if k in last}
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--parent", action="append", default=[], help="run ID this run builds on (repeatable)")
    parser.add_argument("--allow-dirty", action="store_true", help="run uncommitted code; the diff is saved")
    parser.add_argument("--no-eval", action="store_true", help="skip the dev evaluation after training")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- then the command to run")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("no command: put it after --")

    # ---- 1. Checks: a known method, a fresh run ID, committed code ---------------------------------------
    _, method, variant, owner = parse_run_id(args.run_id)
    if not os.path.isfile(os.path.join(METHODS, method, "README.md")):
        raise SystemExit(f"no training_scripts/{method}/README.md: create the method folder first")
    dirty = git_dirty_files()
    if dirty and not args.allow_dirty:
        raise SystemExit(f"uncommitted changes ({', '.join(dirty[:5])}{', ...' if len(dirty) > 5 else ''}): "
                         "commit them, or pass --allow-dirty to save the diff with the run")
    results_dir = os.path.join(RESULTS, args.run_id)
    out_dir = os.path.join(RUNS, args.run_id)
    os.makedirs(RESULTS, exist_ok=True)
    try:
        os.makedirs(results_dir)  # never reuse a run ID
    except FileExistsError:
        raise SystemExit(f"results/{args.run_id} already exists; run IDs are never reused")
    os.makedirs(out_dir, exist_ok=True)
    if dirty:
        with open(os.path.join(results_dir, "code.diff"), "w") as f:
            f.write(git("diff", "HEAD") + "\n")
            untracked = git("ls-files", "--others", "--exclude-standard", "--", ".", ":(exclude)results").splitlines()
            for path in untracked:  # new files, as diffs against nothing (git diff HEAD leaves them out)
                if os.path.getsize(os.path.join(ROOT, path)) < 1_000_000:
                    f.write(subprocess.run(["git", "diff", "--no-index", "/dev/null", path], cwd=ROOT,
                                           capture_output=True, text=True).stdout)
                else:
                    f.write(f"# untracked file over 1 MB, not included: {path}\n")

    command = [a.replace("${LAB_OUT}", out_dir).replace("$LAB_OUT", out_dir) for a in command]
    if os.path.basename(command[0]) in ("python", "python3"):
        command[0] = sys.executable  # the host's configured Python, not whatever `python` is on PATH
    ident = host_identity() or {}
    started = time.time()
    lock = threading.Lock()

    def record(**fields):
        with lock:
            return update_run(results_dir, **fields)

    record(run_id=args.run_id, method=method, variant=variant, owner=owner, hypothesis=args.hypothesis,
           parents=args.parent, command=command, host=ident.get("host", socket.gethostname()),
           git_commit=git_commit(), git_dirty=bool(dirty), status="running", pid=os.getpid(),
           started=utc_now(), heartbeat=utc_now(), ended=None, exit_code=None,
           train_seconds=None, a10g_hours=None, eval=None, progress=None)

    # ---- 2. Run the command, copying its output to stdout.log -------------------------------------------
    env = dict(os.environ, LAB_RUN_ID=args.run_id, LAB_OUT=out_dir, LAB_RESULTS=results_dir,
               LAB_START=str(started), PYTHONUNBUFFERED="1",
               PYTHONPATH=os.pathsep.join(p for p in (ROOT, os.environ.get("PYTHONPATH")) if p))
    log = open(os.path.join(results_dir, "stdout.log"), "ab")

    def tee(proc):
        for line in iter(proc.stdout.readline, b""):
            log.write(line)
            log.flush()
            sys.stdout.buffer.write(line)
            sys.stdout.flush()
        return proc.wait()

    killed = threading.Event()
    child = {}

    def on_signal(signum, _frame):
        killed.set()
        if child.get("proc") and child["proc"].poll() is None:
            os.killpg(child["proc"].pid, signal.SIGTERM)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, on_signal)

    stop_beat = threading.Event()

    def heartbeat():
        while not stop_beat.wait(HEARTBEAT_EVERY_S):
            record(heartbeat=utc_now(), progress=last_metrics(results_dir))

    threading.Thread(target=heartbeat, daemon=True).start()
    child["proc"] = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     start_new_session=True)
    code = tee(child["proc"])
    train_seconds = time.time() - started
    status = "killed" if killed.is_set() else ("done" if code == 0 else "failed")
    record(exit_code=code, train_seconds=round(train_seconds), a10g_hours=round(train_seconds / 3600, 4),
           progress=last_metrics(results_dir), heartbeat=utc_now(), status=status)
    if os.path.exists(os.path.join(out_dir, "train.log")):
        shutil.copy(os.path.join(out_dir, "train.log"), results_dir)

    # ---- 3. Score on dev ---------------------------------------------------------------------------------
    if status == "done" and not args.no_eval:
        if os.path.exists(os.path.join(out_dir, "model_def.py")):
            record(status="evaluating")
            child["proc"] = subprocess.Popen([sys.executable, "-m", "lib.evaluate", args.run_id], cwd=ROOT, env=env,
                                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
            eval_code = tee(child["proc"])
            if killed.is_set():
                status = "killed"
            elif eval_code != 0 and not (read_json(os.path.join(results_dir, "run.json"), {}).get("eval")):
                record(eval={"error": f"lib.evaluate exited with {eval_code}; see stdout.log"})
        else:
            record(eval={"error": "no model_def.py in the output directory"})
    stop_beat.set()
    record(status=status, ended=utc_now(), heartbeat=utc_now())
    log.close()
    sys.exit(code)


if __name__ == "__main__":
    main()
