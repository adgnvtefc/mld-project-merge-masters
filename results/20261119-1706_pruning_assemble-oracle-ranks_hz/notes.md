# 20261119-1706_pruning_assemble-oracle-ranks_hz

Hypothesis: Design A's best embedding-correction rank is above 64: more rank keeps helping until the FFN neurons it costs (58% kept at rank 64, about 40% at rank 256) start to hurt.

Status: failed

Parent: 20261119-1702_pruning_assemble-oracle_hz

## Outcome

Crashed after scoring the references: the pruned-expert ceiling row assumed rank 0 was in --ranks. Fixed in the next commit and re-run as 20261119-1707_pruning_assemble-oracle-ranks_hz. No result.

## Observations

- KeyError ('math', 0) in assemble_oracle.py; the ceiling is now reported at every listed rank's budget.
