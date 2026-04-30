#!/bin/bash
# More chunk36 explorations:
# - Multi-seed rank96+chunk36 combo (single-seed at 1337 was 1.06954, worse than chunk36 alone but might be better at other seeds)
# - chunk36 with different prefix (less prefix → more suffix for TTT)
# - chunk36 with phases=2 (fewer phase boundaries, simpler schedule)
# - chunk36 with alpha=160 (higher alpha might help at smaller chunks)
set -eo pipefail
cd /workspace/pgolf

EXPS=(
  "rank96_chunk36_s42@42@TTT_LORA_RANK=96 TTT_CHUNK_SIZE=36"
  "rank96_chunk36_s314@314@TTT_LORA_RANK=96 TTT_CHUNK_SIZE=36"
  "chunk36_prefix1200@1337@TTT_CHUNK_SIZE=36 PHASED_TTT_PREFIX_DOCS=1200"
  "chunk36_phases2@1337@TTT_CHUNK_SIZE=36 PHASED_TTT_NUM_PHASES=2"
  "chunk36_alpha160@1337@TTT_CHUNK_SIZE=36 TTT_LORA_ALPHA=160"
  "chunk36_wd15@1337@TTT_CHUNK_SIZE=36 TTT_WEIGHT_DECAY=1.5"
)

echo "=== CHUNK36 COMBO BATCH STARTED [$(date)] ==="
for spec in "${EXPS[@]}"; do
  IFS='@' read -r name seed env <<< "$spec"
  echo ""
  echo "############################################"
  echo "# [$(date)] EXPERIMENT: $name seed=$seed env=$env"
  echo "############################################"
  bash work/run_seeds.sh "$name" "$seed" $env 2>&1 || echo "WARNING: $name failed"
done
echo "=== CHUNK36 COMBO DONE [$(date)] ==="
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
