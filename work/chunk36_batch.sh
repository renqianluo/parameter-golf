#!/bin/bash
# Verify chunk36 multi-seed + test rank96+chunk36 combo
set -eo pipefail
cd /workspace/pgolf

EXPS=(
  # Multi-seed chunk36 (seed 1337 was 1.06938, best single-seed result)
  "chunk36_s42@42@TTT_CHUNK_SIZE=36"
  "chunk36_s314@314@TTT_CHUNK_SIZE=36"

  # Combo: rank96 + chunk36 — both helped alone, see if they stack
  "rank96_chunk36@1337@TTT_LORA_RANK=96 TTT_CHUNK_SIZE=36"
)

echo "=== CHUNK36 BATCH STARTED [$(date)] ==="
for spec in "${EXPS[@]}"; do
  IFS='@' read -r name seed env <<< "$spec"
  echo ""
  echo "############################################"
  echo "# [$(date)] EXPERIMENT: $name seed=$seed env=$env"
  echo "############################################"
  bash work/run_seeds.sh "$name" "$seed" $env 2>&1 || echo "WARNING: $name failed"
done
echo "=== CHUNK36 BATCH DONE [$(date)] ==="
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
