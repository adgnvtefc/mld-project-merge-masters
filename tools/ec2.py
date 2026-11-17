"""
The EC2 connector: finds an instance by its ID (public IPs change on every restart) and runs ssh/rsync
against it. Standard library only; needs the AWS CLI (`brew install awscli`, then `aws configure`).

Hosts are listed in lab_hosts.toml (committed); keys and the AWS profile in lab.local.toml (per laptop).
On a host, each owner has a checkout at ~/mld/<owner>; models/ and pool/ are links to ~/mld/_data.
Set LAB_DRY_RUN=1 to print the aws/ssh/rsync commands instead of running them.
"""

import json
import os
import shlex
import subprocess
import sys

from lib.common import HOSTS_CONFIG, ROOT, load_toml, local_config

DRY_RUN = bool(os.environ.get("LAB_DRY_RUN"))
DATA_DIR = "mld/_data"

# Never pushed, and never deleted on the host by a push: data, outputs, host-only state.
PUSH_EXCLUDES = ["/models", "/pool", "/runs", "/results", "/.lab_host", "/.lab_launch", "/lab.local.toml",
                 "/.cache", "__pycache__", ".DS_Store", "*.tmp*"]


def run(cmd, capture=True, check=True, stdin=None):
    if DRY_RUN:
        print("[dry-run]", shlex.join(cmd), file=sys.stderr)
        return ""
    proc = subprocess.run(cmd, capture_output=capture, text=True, input=stdin)
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip() if capture else ""
        raise SystemExit(f"{cmd[0]} failed (exit {proc.returncode}){': ' + detail if detail else ''}")
    return proc.stdout if capture else proc.returncode


def remote_dir(owner):
    return f"mld/{owner}"  # relative to the remote home directory


class Host:
    def __init__(self, name):
        hosts = load_toml(HOSTS_CONFIG).get("hosts", {})
        if name not in hosts:
            raise SystemExit(f"unknown host {name!r}; lab_hosts.toml lists: {', '.join(hosts) or 'none'}")
        cfg, local = hosts[name], local_config()
        self.name = name
        self.instance_id = cfg["instance_id"]
        self.region = cfg.get("region", "us-east-1")
        self.user = cfg.get("user", "ubuntu")
        self.python = cfg.get("python", "python3")
        self.key = os.path.expanduser(local.get("ssh_keys", {}).get(name) or local["ssh_key"])
        self.profile = local.get("aws_profile")
        self._ip = None
        if self.instance_id.startswith("i-REPLACE"):
            raise SystemExit(f"host {name!r} has no instance ID yet: fill it in lab_hosts.toml")

    # ---- AWS ----------------------------------------------------------------------------------------------
    def aws(self, *args):
        cmd = ["aws", "--region", self.region, "--output", "json", *(["--profile", self.profile] if self.profile else []),
               "ec2", *args]
        try:
            out = run(cmd)
        except FileNotFoundError:
            raise SystemExit("the AWS CLI is not installed: brew install awscli, then aws configure")
        return json.loads(out) if out else {}

    def describe(self):
        """(state, public IP or None)."""
        if DRY_RUN:
            return "running", "203.0.113.10"
        out = self.aws("describe-instances", "--instance-ids", self.instance_id)
        inst = out["Reservations"][0]["Instances"][0]
        return inst["State"]["Name"], inst.get("PublicIpAddress")

    def ip(self):
        if self._ip is None:
            state, ip = self.describe()
            if state != "running" or not ip:
                raise SystemExit(f"host {self.name} is {state}; start it with ./lab start {self.name}")
            self._ip = ip
        return self._ip

    def start(self):
        self.aws("start-instances", "--instance-ids", self.instance_id)
        print(f"starting {self.name}; waiting until it is running...", flush=True)
        self.aws("wait", "instance-running", "--instance-ids", self.instance_id)
        self._ip = None
        print(f"{self.name} is running at {self.ip()}")

    def stop(self):
        self.aws("stop-instances", "--instance-ids", self.instance_id)
        print(f"{self.name} is stopping")

    # ---- ssh / rsync --------------------------------------------------------------------------------------
    def ssh_options(self):
        # Known-hosts entries keyed by instance ID, so a new IP after a restart is not a "changed host key".
        return ["-i", self.key, "-o", f"HostKeyAlias={self.instance_id}", "-o", "StrictHostKeyChecking=accept-new",
                "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=30"]

    def target(self):
        return f"{self.user}@{self.ip()}"

    def ssh(self, command, capture=True, check=True, tty=False):
        """Run a shell command on the host (from the home directory)."""
        return run(["ssh", *self.ssh_options(), *(["-t"] if tty else []), self.target(), command],
                   capture=capture, check=check)

    def rsync(self, src, dst, *flags):
        ssh = shlex.join(["ssh", *self.ssh_options()])
        return run(["rsync", "-az", "-e", ssh, *flags, src, dst], capture=False)

    def push(self, owner):
        """Mirror this checkout (code and .git, not data or outputs) to ~/mld/<owner> on the host."""
        self.ssh(f"mkdir -p {remote_dir(owner)}")
        excludes = [f"--exclude={p}" for p in PUSH_EXCLUDES]
        self.rsync(ROOT + "/", f"{self.target()}:{remote_dir(owner)}/", "--delete", *excludes)

    def owners(self):
        """Owners with a checkout on this host."""
        out = self.ssh("ls -1 mld 2>/dev/null || true")
        return [o for o in out.split() if o != "_data"]

    def pull_results(self, owner, dest):
        """Copy ~/mld/<owner>/results/ into dest; never notes.md (written on laptops) or weights."""
        self.rsync(f"{self.target()}:{remote_dir(owner)}/results/", dest + "/", "--exclude=notes.md",
                   "--exclude=*.tmp*")
