#!/bin/bash
# Run a list of experiments sequentially.
# Usage: bash work/queue_runner.sh exp1:env1 exp2:env2 ...
# Example: bash work/queue_runner.sh prefix1500:PHASED_TTT_PREFIX_DOCS=1500 minlr05:MIN_LR=0.05
set -eo pipefail
WORKDIR=/workspace/pgolf
cd "$WORKDIR"

for spec in "$@"; do
    name="${spec%%:*}"
    env="${spec#*:}"
    [[ "$env" == "$name" ]] && env=""
    echo ""
    echo "############################################"
    echo "# [$(date)] EXPERIMENT: $name (env: $env)"
    echo "############################################"
    bash work/run_seeds.sh "$name" 1337 $env || echo "WARNING: $name exit code $?"
done
echo "[$(date)] QUEUE DONE"
