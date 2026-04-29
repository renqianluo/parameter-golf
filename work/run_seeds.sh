#!/bin/bash
# Run an experiment across multiple seeds sequentially.
# Usage: bash work/run_seeds.sh <exp_name> [SEED1,SEED2,...] [VAR=val ...]
# Example: bash work/run_seeds.sh minlr05 1337,42,314 MIN_LR=0.05
set -eo pipefail
EXP_NAME=${1:-"runpod_exp"}
SEEDS=${2:-"1337"}
shift 2 || true

WORKDIR=/workspace/pgolf
cd "$WORKDIR"
LOGDIR="$WORKDIR/work/logs"
mkdir -p "$LOGDIR"

IFS=',' read -r -a SEED_LIST <<< "$SEEDS"
echo "Running $EXP_NAME with seeds: ${SEED_LIST[@]}"
echo "Extra env: $@"
for SEED in "${SEED_LIST[@]}"; do
    LOGFILE="$LOGDIR/${EXP_NAME}_seed${SEED}.log"
    echo "=== [$(date)] Starting $EXP_NAME seed=$SEED ===" | tee "$LOGFILE"
    env DATA_DIR="$WORKDIR/data" \
        MAX_WALLCLOCK_SECONDS=600 \
        SEED=$SEED \
        PYTHONUNBUFFERED=1 \
        "$@" \
        torchrun --standalone --nproc_per_node=8 "$WORKDIR/work/train_gpt.py" 2>&1 | tee -a "$LOGFILE"
    echo "=== [$(date)] Finished $EXP_NAME seed=$SEED ===" | tee -a "$LOGFILE"
done
echo "=== ALL SEEDS DONE for $EXP_NAME ==="
