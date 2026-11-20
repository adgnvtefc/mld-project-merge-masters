# 20261119-0259_baseline_generalist_ay

Hypothesis: my host reproduces the generalist's 3.1555

Status: done | dev mean 3.1555

## Outcome

Supported. The provided generalist, unchanged, scored dev mean 3.155494363978505 on host `ayush` (commit
eef4534). That is identical, to every digit, to Hanmo's run on `hanmo` (20261118-1942_baseline_generalist_hz).
Per domain: math 3.1415, code 3.1689, medicine 2.7868, history 3.5248.

## Observations

- First run on host `ayush` (g5.xlarge, Deep Learning Base OSS Nvidia Driver GPU AMI, Ubuntu 24.04, venv
  created by `./lab setup`). The setup and scoring path give the same numbers as `hanmo`, so runs from the
  two hosts can be compared directly.
- Smoke test only; not a method result.
