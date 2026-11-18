"""
./lab: one command line for the lab workflow (see .claude/skills/lab-workflow/SKILL.md).

Laptop commands reach EC2 through tools/ec2.py; on an EC2 host (one with .lab_host), run/runs/logs/kill/eval
act on the local checkout and the laptop-only commands refuse.
"""

import argparse
import glob
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

from lib.common import (HOSTS_CONFIG, METHODS, RESULTS, ROOT, effective_status, git, git_dirty_files,
                        host_identity, load_toml, local_config, new_run_id, owner, parse_run_id, read_json, write_json)

LIST_RUNS = r"""
import glob, json, os
runs = []
for path in glob.glob(os.path.expanduser("~/mld/*/results/*/run.json")):
    try:
        runs.append(json.load(open(path)))
    except Exception:
        pass
print(json.dumps(runs))
"""
SYNC_DIR = os.path.join(RESULTS, ".sync")  # when each host was last pulled (gitignored)


# ---- helpers ------------------------------------------------------------------------------------------

def on_ec2():
    return host_identity() is not None


def laptop_only(command):
    if on_ec2():
        raise SystemExit(f"`./lab {command}` runs on a laptop; this is EC2 host {host_identity()['host']}")


def host_for(name):
    from tools.ec2 import Host
    return Host(name or local_config()["default_host"])


def host_of_run(run_id, name):
    """The host a run lives on: --host, else what a pulled run.json says, else the default host."""
    pulled = read_json(os.path.join(RESULTS, run_id, "run.json"), {}).get("host")
    if pulled not in load_toml(HOSTS_CONFIG).get("hosts", {}):
        pulled = None
    return host_for(name or pulled)


def age(stamp):
    if not stamp:
        return "-"
    s = time.time() - datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    return f"{s:.0f}s" if s < 120 else f"{s / 60:.0f}m" if s < 7200 else f"{s / 3600:.1f}h"


def print_runs(runs, limit):
    runs = sorted(runs, key=lambda r: r.get("started") or "")[-limit:]
    if not runs:
        print("no runs")
        return
    print(f"{'run_id':<52} {'status':<10} {'step':>7} {'beat':>6} {'A10G-h':>7} {'dev':>7}")
    for r in runs:
        progress = r.get("progress") or {}
        ev = r.get("eval") or {}
        dev = f"{ev['mean']:.4f}" if "mean" in ev else ("err" if "error" in ev else "-")
        hours = r.get("a10g_hours")
        print(f"{r.get('run_id', '?'):<52} {effective_status(r):<10} {progress.get('step', '-'):>7} "
              f"{age(r.get('heartbeat')):>6} {hours if hours is not None else '-':>7} {dev:>7}")


def launch_command(python, run_id, args):
    cmd = [python, "-m", "lib.launch", "--run-id", run_id, "--hypothesis", args.hypothesis]
    for p in args.parent:
        cmd += ["--parent", p]
    cmd += ["--allow-dirty"] * args.allow_dirty + ["--no-eval"] * args.no_eval + ["--", *args.command]
    return shlex.join(cmd)


def tmux_launch(run_id, launch):
    """Start the launcher in a detached tmux session named after the run; its own errors go to .lab_launch/."""
    return (f"mkdir -p .lab_launch && tmux new-session -d -s {run_id} "
            + shlex.quote(f"{launch} 2>> .lab_launch/{run_id}.err"))


# ---- commands -----------------------------------------------------------------------------------------

def cmd_whereami(args):
    ident = host_identity()
    if ident:
        print(f"ec2:{ident['host']}  owner {ident['owner']}  checkout {ROOT}  commit {git('rev-parse', '--short', 'HEAD')}")
        print("This is an EC2 host: run things here, but edit code on a laptop and ./lab push.")
        return
    cfg = local_config()
    dirty = git_dirty_files()
    print(f"laptop  owner {cfg['owner']}  default host {cfg['default_host']}  "
          f"commit {git('rev-parse', '--short', 'HEAD')}{f'  ({len(dirty)} uncommitted files)' if dirty else ''}")


def cmd_hosts(args):
    laptop_only("hosts")
    me = local_config()["owner"]
    local_head = git("rev-parse", "--short", "HEAD")
    for name in load_toml(HOSTS_CONFIG).get("hosts", {}):
        try:
            host = host_for(name)
            state, ip = host.describe()
        except SystemExit as e:
            print(f"{name:<10} {e}")
            continue
        line = f"{name:<10} {state:<9} {ip or '-':<16}"
        if state == "running":
            out = host.ssh("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits "
                           "2>/dev/null | head -1; echo ---; for d in mld/*/; do [ -d $d.git ] && echo "
                           "$(basename $d) $(git -C $d rev-parse --short HEAD); done; echo ---; "
                           "tmux ls -F '#S' 2>/dev/null", check=False) or "---\n---\n"
            gpu, checkouts, sessions = (out.split("---") + ["", ""])[:3]
            line += f" GPU {gpu.strip() or '?'} (util %, MiB)"
            for entry in checkouts.split("\n"):
                if entry.strip():
                    who, head = entry.split()
                    flag = "" if who != me else (" = local" if head == local_head else f" != local {local_head}")
                    line += f" | {who}@{head}{flag}"
            running = [s for s in sessions.split() if s]
            line += f" | tmux: {', '.join(running) or 'none'}"
        print(line)


def cmd_start(args):
    laptop_only("start")
    host_for(args.host).start()


def cmd_stop(args):
    laptop_only("stop")
    host = host_for(args.host)
    runs = [r for r in remote_runs(host) if effective_status(r) in ("running", "evaluating")]
    if runs:
        raise SystemExit(f"{host.name} has active runs: {', '.join(r['run_id'] for r in runs)}. Kill them first.")
    if not args.yes:
        raise SystemExit(f"stopping {host.name} ends every session on it. Re-run with --yes to confirm.")
    host.stop()


def cmd_setup(args):
    laptop_only("setup")
    from tools.ec2 import DATA_DIR, remote_dir
    host, me = host_for(args.host), local_config()["owner"]
    print(f"pushing code to {host.name}:~/{remote_dir(me)}", flush=True)
    host.push(me)
    marker = json.dumps({"host": host.name, "owner": me})
    py = host.python
    script = f"""set -e
cd ~/{remote_dir(me)}
echo {shlex.quote(marker)} > .lab_host
mkdir -p ~/{DATA_DIR}/models ~/{DATA_DIR}/pool
for d in models pool; do [ -L $d ] || {{ [ -e $d ] && echo "$d exists and is not a link; leaving it" || ln -s ~/{DATA_DIR}/$d $d; }}; done
command -v tmux >/dev/null || sudo apt-get install -y tmux
nvidia-smi -L
case "{py}" in */bin/python) [ -x "{py}" ] || python3 -m venv "$(dirname "$(dirname "{py}")")";; esac
{py} -c "import torch, sys; sys.exit(0 if torch.__version__.startswith('2.10.0') and torch.cuda.is_available() else 1)" \\
  || {py} -m pip install --no-cache-dir torch==2.10.0 numpy huggingface_hub
[ -f ~/{DATA_DIR}/.verified ] || {{ {py} code/download_data.py && touch ~/{DATA_DIR}/.verified; }}
{py} -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "setup done: ~/{remote_dir(me)} on {host.name}"
"""
    host.ssh(f"bash -c {shlex.quote(script)}", capture=False)


def cmd_push(args):
    laptop_only("push")
    host, me = host_for(args.host), local_config()["owner"]
    dirty = git_dirty_files()
    if dirty:
        print(f"note: pushing {len(dirty)} uncommitted files; ./lab run will still refuse them without --allow-dirty")
    host.push(me)
    print(f"pushed {git('rev-parse', '--short', 'HEAD')} to {host.name}:~/mld/{me}")


def cmd_run(args):
    if not args.command:
        raise SystemExit("no command: ./lab run [host] --method M --variant V --hypothesis '...' -- <command>")
    if not os.path.isfile(os.path.join(METHODS, args.method, "README.md")):
        raise SystemExit(f"no training_scripts/{args.method}/README.md: create the method folder first "
                         f"(existing: {', '.join(sorted(os.listdir(METHODS)))})")
    if not any("LAB_OUT" in a for a in args.command):
        print("warning: the command never mentions $LAB_OUT, so its outputs won't land in runs/<run_id>/", file=sys.stderr)
    dirty = git_dirty_files()
    if dirty and not args.allow_dirty:
        raise SystemExit(f"uncommitted changes ({', '.join(dirty[:5])}): commit them, or pass --allow-dirty")
    run_id = new_run_id(args.method, args.variant, owner())

    if on_ec2():
        if not args.force and subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                             capture_output=True, text=True).stdout.strip():
            raise SystemExit("the GPU is in use (nvidia-smi); pass --force to launch anyway")
        subprocess.run(tmux_launch(run_id, launch_command(sys.executable, run_id, args)), shell=True, cwd=ROOT,
                       check=True)
        check = lambda: os.path.exists(os.path.join(RESULTS, run_id, "run.json"))  # noqa: E731
        err = lambda: open(os.path.join(ROOT, ".lab_launch", f"{run_id}.err")).read()  # noqa: E731
    else:
        host, me = host_for(args.host), local_config()["owner"]
        host.push(me)
        busy = host.ssh("nvidia-smi --query-compute-apps=pid --format=csv,noheader", check=False)
        if busy and busy.strip() and not args.force:
            raise SystemExit(f"the GPU on {host.name} is in use; check ./lab hosts, or pass --force")
        host.ssh(f"cd mld/{me} && " + tmux_launch(run_id, launch_command(host.python, run_id, args)))
        check = lambda: os.environ.get("LAB_DRY_RUN") or host.ssh(  # noqa: E731
            f"test -f mld/{me}/results/{run_id}/run.json && echo yes", check=False).strip() == "yes"
        err = lambda: host.ssh(f"cat mld/{me}/.lab_launch/{run_id}.err", check=False)  # noqa: E731
    for _ in range(10):
        if check():
            print(f"launched {run_id}")
            print(f"  ./lab logs {run_id} -f    ./lab runs    ./lab pull")
            return
        time.sleep(1)
    raise SystemExit(f"{run_id} did not start:\n{err()}")


def remote_runs(host):
    from tools.ec2 import run
    out = run(["ssh", *host.ssh_options(), host.target(), "python3", "-"], stdin=LIST_RUNS)
    return json.loads(out or "[]")


def cmd_runs(args):
    if on_ec2():
        runs = [read_json(p, {}) for p in glob.glob(os.path.join(ROOT, "..", "*", "results", "*", "run.json"))]
    else:
        runs = remote_runs(host_for(args.host))
    print_runs([r for r in runs if args.owner in (None, r.get("owner"))], args.limit)


def cmd_logs(args):
    _, _, _, who = parse_run_id(args.run_id)
    rel = f"results/{args.run_id}/stdout.log"
    tail = ["tail", "-n", str(args.lines), *(["-f"] if args.follow else [])]
    if on_ec2():
        os.execvp("tail", tail + [os.path.join(ROOT, "..", who, rel)])
    host = host_of_run(args.run_id, args.host)
    host.ssh(f"{shlex.join(tail)} mld/{who}/{rel}", capture=False, check=False, tty=args.follow)


def cmd_kill(args):
    _, _, _, who = parse_run_id(args.run_id)
    if who != owner() and not args.force:
        raise SystemExit(f"{args.run_id} belongs to {who}; pass --force only if the user asked to kill it")
    if on_ec2():
        rec = read_json(os.path.join(ROOT, "..", who, "results", args.run_id, "run.json"), {})
        os.kill(int(rec["pid"]), 15)
    else:
        host = host_of_run(args.run_id, args.host)
        host.ssh(f"kill -TERM $(python3 -c \"import json; print(json.load(open('mld/{who}/results/{args.run_id}/"
                 f"run.json'))['pid'])\")")
    print(f"sent SIGTERM to {args.run_id}; its status becomes 'killed' once it exits")


def cmd_eval(args):
    _, _, _, who = parse_run_id(args.run_id)
    if on_ec2():
        subprocess.run([sys.executable, "-m", "lib.evaluate", args.run_id], cwd=os.path.join(ROOT, "..", who),
                       check=True)
    else:
        host = host_of_run(args.run_id, args.host)
        host.ssh(f"cd mld/{who} && {host.python} -m lib.evaluate {args.run_id}", capture=False)


def cmd_pull(args):
    laptop_only("pull")
    host = host_for(args.host)
    os.makedirs(RESULTS, exist_ok=True)
    for who in host.owners():
        if host.ssh(f"test -d mld/{who}/results && echo yes", check=False).strip() == "yes":
            host.pull_results(who, RESULTS)
    os.makedirs(SYNC_DIR, exist_ok=True)
    write_json(os.path.join(SYNC_DIR, f"{host.name}.json"), {"host": host.name, "pulled": time.time()})
    stubs = []
    for path in glob.glob(os.path.join(RESULTS, "*", "run.json")):
        rec = read_json(path, {})
        notes = os.path.join(os.path.dirname(path), "notes.md")
        if rec.get("status") in ("done", "failed", "killed") and not os.path.exists(notes):
            ev = rec.get("eval") or {}
            score = f" | dev mean {ev['mean']:.4f}" if "mean" in ev else ""
            with open(notes, "w") as f:
                f.write(f"# {rec['run_id']}\n\nHypothesis: {rec.get('hypothesis', '')}\n\n"
                        f"Status: {rec['status']}{score}\n\n"
                        "## Outcome\n\nTODO: what happened, and does it support the hypothesis?\n\n"
                        "## Observations\n\n")
            stubs.append(rec["run_id"])
    print(f"pulled results from {host.name}" + (f"; new notes.md stubs: {', '.join(stubs)}" if stubs else ""))
    git_status = git("status", "--porcelain", "--", "results")
    if git_status:
        print("uncommitted results:\n" + git_status)


def cmd_ssh(args):
    laptop_only("ssh")
    host, me = host_for(args.host), local_config()["owner"]
    os.execvp("ssh", ["ssh", *host.ssh_options(), "-t", host.target(), f"cd mld/{me} 2>/dev/null; exec $SHELL -l"])


def cmd_dashboard(args):
    laptop_only("dashboard")
    if shutil.which("streamlit") is None:
        raise SystemExit("streamlit is not installed: pip install streamlit plotly pandas numpy")
    os.execvp("streamlit", ["streamlit", "run", os.path.join(ROOT, "tools", "dashboard", "app.py"),
                            "--server.port", str(args.port), "--", "--results", args.results])


# ---- argument parsing ---------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(prog="./lab", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_, host=False):
        p = sub.add_parser(name, help=help_)
        p.set_defaults(fn=fn)
        if host:
            p.add_argument("host", nargs="?", help="host name from lab_hosts.toml (default: default_host)")
        return p

    add("whereami", cmd_whereami, "laptop or ec2:<host>, owner, default host")
    add("hosts", cmd_hosts, "each host: state, IP, GPU, checkouts and their commits, tmux sessions")
    add("start", cmd_start, "start an instance", host=True)
    add("stop", cmd_stop, "stop an instance (needs --yes)", host=True).add_argument("--yes", action="store_true")
    add("setup", cmd_setup, "first-time setup of your checkout on a host (idempotent)", host=True)
    add("push", cmd_push, "sync code and .git to your checkout on a host", host=True)

    p = add("run", cmd_run, "push, then launch a command under the recorder in tmux", host=True)
    p.add_argument("--method", required=True, help="a folder in training_scripts/")
    p.add_argument("--variant", required=True, help="what makes this run different, e.g. ridge-router")
    p.add_argument("--hypothesis", required=True, help="one sentence: what this run is meant to show")
    p.add_argument("--parent", action="append", default=[], help="run ID this run builds on (repeatable)")
    p.add_argument("--allow-dirty", action="store_true", help="run uncommitted code; the diff is saved")
    p.add_argument("--no-eval", action="store_true", help="skip the dev evaluation")
    p.add_argument("--force", action="store_true", help="launch even if the GPU is in use")
    p.epilog = "After --, the command to run; pass '$LAB_OUT' (quoted) as its --out."

    p = add("runs", cmd_runs, "runs on a host with status, step, heartbeat age, dev score", host=True)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--owner", help="only this owner's runs")

    p = add("logs", cmd_logs, "tail a run's stdout.log")
    p.add_argument("run_id")
    p.add_argument("-f", "--follow", action="store_true")
    p.add_argument("-n", "--lines", type=int, default=40)
    p.add_argument("--host")

    p = add("kill", cmd_kill, "stop a run (SIGTERM); its status becomes killed")
    p.add_argument("run_id")
    p.add_argument("--host")
    p.add_argument("--force", action="store_true", help="kill another owner's run")

    p = add("eval", cmd_eval, "(re)score a finished run on dev")
    p.add_argument("run_id")
    p.add_argument("--host")

    add("pull", cmd_pull, "copy results/ from a host (no weights); create notes.md stubs", host=True)
    add("ssh", cmd_ssh, "interactive shell in your checkout on a host", host=True)
    p = add("dashboard", cmd_dashboard, "open the dashboard on localhost")
    p.add_argument("--port", type=int, default=8605)
    p.add_argument("--results", default=RESULTS)

    # Split at the first --, so the command's own words never fill [host] or other options.
    argv = sys.argv[1:]
    cut = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:cut])
    args.command = argv[cut + 1:]
    args.fn(args)
