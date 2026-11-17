"""
Shared paths, names and file formats for the lab workflow (see .claude/skills/lab-workflow/SKILL.md).
Standard library only: this runs on laptops (no torch) and on EC2.
"""

import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
RUNS = os.path.join(ROOT, "runs")
METHODS = os.path.join(ROOT, "training_scripts")
HOST_MARKER = os.path.join(ROOT, ".lab_host")  # written by `./lab setup` on EC2 only
LOCAL_CONFIG = os.path.join(ROOT, "lab.local.toml")
HOSTS_CONFIG = os.path.join(ROOT, "lab_hosts.toml")

DOMAINS = ["math", "code", "medicine", "history"]
SLUG = re.compile(r"^[a-z0-9][a-z0-9-]*$")
RUN_ID = re.compile(r"^(\d{8}-\d{4})_([a-z0-9-]+)_([a-z0-9-]+)_([a-z0-9-]+)$")
HEARTBEAT_STALE_S = 600

# Dev scores from the handout, drawn as reference lines; not runs of ours.
REFERENCE = {
    "average of experts": 3.516,
    "best single expert (math)": 3.331,
    "generalist": 3.156,
    "per-domain experts (oracle)": 2.78,
}


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id(method, variant, owner):
    for name, value in (("method", method), ("variant", variant), ("owner", owner)):
        if not SLUG.match(value):
            raise SystemExit(f"{name} {value!r} must be lowercase letters, digits and dashes")
    return f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}_{method}_{variant}_{owner}"


def parse_run_id(run_id):
    """(timestamp, method, variant, owner), or SystemExit if run_id is not one of ours."""
    m = RUN_ID.match(run_id)
    if not m:
        raise SystemExit(f"{run_id!r} is not a run ID (<YYYYMMDD-HHMM>_<method>_<variant>_<owner>)")
    return m.groups()


def method_names():
    return sorted(d for d in os.listdir(METHODS)
                  if os.path.isfile(os.path.join(METHODS, d, "README.md"))) if os.path.isdir(METHODS) else []


# ---- run.json ----------------------------------------------------------------------------------------

def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path, obj):
    """Atomic write, so the dashboard or a pull never sees half a file."""
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def update_run(results_dir, **fields):
    path = os.path.join(results_dir, "run.json")
    record = read_json(path, {})
    record.update(fields)
    write_json(path, record)
    return record


def effective_status(record, now=None):
    """The status to show: a 'running' run whose heartbeat stopped is 'lost'."""
    status = record.get("status", "unknown")
    if status in ("running", "evaluating") and record.get("heartbeat"):
        beat = datetime.strptime(record["heartbeat"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        if (now or time.time()) - beat.timestamp() > HEARTBEAT_STALE_S:
            return "lost"
    return status


# ---- where am I ---------------------------------------------------------------------------------------

def host_identity():
    """{'host': name, 'owner': tag} on EC2 (from the marker `./lab setup` wrote), else None (a laptop)."""
    return read_json(HOST_MARKER)


def load_toml(path):
    import tomllib  # Python 3.11+
    with open(path, "rb") as f:
        return tomllib.load(f)


def local_config():
    if not os.path.exists(LOCAL_CONFIG):
        raise SystemExit(f"no {os.path.basename(LOCAL_CONFIG)}: copy lab.local.toml.example to lab.local.toml "
                         "and fill in your owner tag, ssh key and default host")
    return load_toml(LOCAL_CONFIG)


def owner():
    ident = host_identity()
    return ident["owner"] if ident else local_config()["owner"]


# ---- git ----------------------------------------------------------------------------------------------

def git(*args, cwd=ROOT):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True).stdout.strip()


def git_commit():
    return git("rev-parse", "HEAD") or None


def git_dirty_files():
    """Uncommitted changes that would make a run's commit lie: anything but results/ (results aren't code)."""
    out = git("status", "--porcelain", "--", ".", ":(exclude)results")
    return [line[3:] for line in out.splitlines() if line.strip()]
