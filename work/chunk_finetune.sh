#!/bin/bash
# Fine-tune around chunk36 (current best at 1.06900):
# chunk34 (between 32 over-budget and 36 sweet-spot — might be even better)
# chunk38 (refine between 36 and 40 outlier)
# chunk36+phases4 (more phases with smaller chunks)
# chunk36+loralr0.00012 (slight LR bump now that grad steps land more often)
set -eo pipefail
cd /workspace/pgolf

EXPS=(
  "chunk34@1337@TTT_CHUNK_SIZE=34"
  "chunk38@1337@TTT_CHUNK_SIZE=38"
  "chunk36_phases4@1337@TTT_CHUNK_SIZE=36 PHASED_TTT_NUM_PHASES=4"
  "chunk36_loralr12@1337@TTT_CHUNK_SIZE=36 TTT_LORA_LR=0.00012"
)

echo "=== CHUNK FINETUNE BATCH STARTED [$(date)] ==="
for spec in "${EXPS[@]}"; do
  IFS='@' read -r name seed env <<< "$spec"
  echo ""
  echo "############################################"
  echo "# [$(date)] EXPERIMENT: $name seed=$seed env=$env"
  echo "############################################"
  bash work/run_seeds.sh "$name" "$seed" $env 2>&1 || echo "WARNING: $name failed"
done
echo "=== CHUNK FINETUNE DONE [$(date)] ==="
for spec in "${EXPS[@]}"; do
  IFS='@' read -r name seed env <<< "$spec"
  log="work/logs/${name}_seed${seed}.log"
  if [ -f "$log" ]; then
    bpb=$(grep -aE 'quantized_ttt_phased.*val_bpb' "$log" 2>/dev/null | tail -1 | grep -oE 'val_bpb:[0-9.]+' | head -1 || true)
    eval_t=$(grep -aE 'total_eval_time' "$log" 2>/dev/null | tail -1 || true)
    echo "$name seed=$seed: $bpb | $eval_t"
  else
    echo "$name seed=$seed: NO LOG"
  fi
done
