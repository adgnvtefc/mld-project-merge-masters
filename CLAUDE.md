# 10-605 mini-project: one model from four experts

Course handout: `README`. Team workflow: the `lab-workflow` skill. Load it before writing a
training script, launching or monitoring a run, touching EC2, syncing files, or reading or
recording results.

Hard rules, even if the skill isn't loaded:
1. Run `./lab whereami` before running anything. Edit code only on a laptop, never on EC2.
2. Reach EC2 only through `./lab` (never raw ssh/scp/rsync with an IP).
3. Launch every experiment with `./lab run`, so it gets a run ID, commit and record in `results/`.
4. Never overwrite or delete a `results/<run_id>/` directory, and never touch `dev/` in training.
