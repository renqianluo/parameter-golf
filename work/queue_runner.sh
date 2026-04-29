#!/bin/bash
# Run a list of experiments sequentially.
# Usage: bash work/queue_runner.sh exp1@seed1@env1 exp2@seed2@env2 ...
# Example: bash work/queue_runner.sh p1500_42@42@PHASED_TTT_PREFIX_DOCS=1500 p1500_314@314@PHASED_TTT_PREFIX_DOCS=1500
set -eo pipefail
WORKDIR=/workspace/pgolf
cd "$WORKDIR"

for spec in "$@"; do
    IFS='@' read -r name seed env <<< "$spec"
    [[ -z "$seed" ]] && seed=1337
    echo ""
    echo "############################################"
    echo "# [$(date)] EXPERIMENT: $name seed=$seed env: $env"
    echo "############################################"
    bash work/run_seeds.sh "$name" "$seed" $env || echo "WARNING: $name exit code $?"
done
echo "[$(date)] QUEUE DONE"
