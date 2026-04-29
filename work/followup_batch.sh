#!/bin/bash
# Follow-up batch: multi-seed rank96 (best from night batch) + extend rank/chunk sweeps
set -eo pipefail
cd /workspace/pgolf

EXPS=(
  # Multi-seed verification of rank96 (best single-seed at 1.06964)
  "rank96_s42@42@TTT_LORA_RANK=96"
  "rank96_s314@314@TTT_LORA_RANK=96"

  # Extend rank sweep further (rank96 was better, try lower)
  "rank64@1337@TTT_LORA_RANK=64"
  "rank80@1337@TTT_LORA_RANK=80"
  "rank112@1337@TTT_LORA_RANK=112"

  # Chunk sweep — chunk32 was best but timed out (600.2s); chunk40 should be safe and good
  "chunk40@1337@TTT_CHUNK_SIZE=40"
  "chunk36@1337@TTT_CHUNK_SIZE=36"

  # Combo: rank96 + chunk40 (potentially stacked improvement)
  "rank96_chunk40@1337@TTT_LORA_RANK=96 TTT_CHUNK_SIZE=40"
)

echo "=== FOLLOWUP BATCH STARTED [$(date)] ==="
for spec in "${EXPS[@]}"; do
  IFS='@' read -r name seed env <<< "$spec"
  echo ""
  echo "############################################"
  echo "# [$(date)] EXPERIMENT: $name seed=$seed env=$env"
  echo "############################################"
  bash work/run_seeds.sh "$name" "$seed" $env 2>&1 || echo "WARNING: $name failed"
done
echo "=== FOLLOWUP BATCH DONE [$(date)] ==="
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
