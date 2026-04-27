#!/bin/bash
# RunPod launch script for parameter-golf experiments
set -eo pipefail
EXP_NAME=${1:-"runpod_exp"}
shift

WORKDIR=/workspace/pgolf
cd "$WORKDIR"

LOGDIR="$WORKDIR/work/logs"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/${EXP_NAME}.log"

SNAPSHOT="$LOGDIR/${EXP_NAME}_train_gpt.py"
if [ -f "$SNAPSHOT" ]; then
    TRAIN_SCRIPT="$SNAPSHOT"
else
    TRAIN_SCRIPT="$WORKDIR/work/train_gpt.py"
fi

echo "[$(date)] Starting experiment: $EXP_NAME" | tee "$LOGFILE"
echo "Extra args: $@" | tee -a "$LOGFILE"
echo "Train script: $TRAIN_SCRIPT" | tee -a "$LOGFILE"

DATA_DIR="$WORKDIR/data"
N=8
W=600

env DATA_DIR="$DATA_DIR" \
    MAX_WALLCLOCK_SECONDS=$W \
    SEED=1337 \
    PYTHONUNBUFFERED=1 \
    "$@" \
    torchrun --standalone --nproc_per_node=$N "$TRAIN_SCRIPT" 2>&1 | tee -a "$LOGFILE"
EXIT_CODE=${PIPESTATUS[0]}
echo "[$(date)] Experiment $EXP_NAME complete (exit=$EXIT_CODE)" | tee -a "$LOGFILE"
