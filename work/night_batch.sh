#!/bin/bash
# Autonomous overnight batch (24 experiments × ~25 min ≈ 10 hours).
# All experiments are seed=1337 single-seed screening on top of prefix=1500 baseline.
# Runs sequentially. Each experiment varies ONE knob from the 1.06934 record defaults.
set -eo pipefail
cd /workspace/pgolf

# Format: name@seed@env_overrides
EXPS=(
  # === LoRA rank sweep ===
  "rank96@1337@TTT_LORA_RANK=96"
  "rank160@1337@TTT_LORA_RANK=160"
  "rank192@1337@TTT_LORA_RANK=192"
  "rank256@1337@TTT_LORA_RANK=256"

  # === LoRA alpha sweep (lower than 144) ===
  "alpha120@1337@TTT_LORA_ALPHA=120"
  "alpha100@1337@TTT_LORA_ALPHA=100"

  # === LoRA LR sweep (around 0.0001) ===
  "loralr05@1337@TTT_LORA_LR=0.00005"
  "loralr07@1337@TTT_LORA_LR=0.00007"
  "loralr12@1337@TTT_LORA_LR=0.00012"

  # === Chunk size sweep (default 48) ===
  "chunk32@1337@TTT_CHUNK_SIZE=32"
  "chunk64@1337@TTT_CHUNK_SIZE=64"

  # === Phase count sweep (default 3) ===
  "phases2@1337@PHASED_TTT_NUM_PHASES=2"
  "phases4@1337@PHASED_TTT_NUM_PHASES=4"
  "phases5@1337@PHASED_TTT_NUM_PHASES=5"

  # === Optimizer + momentum ===
  "ttt_adamw@1337@TTT_OPTIMIZER=adamw"
  "beta1_05@1337@TTT_BETA1=0.5"
  "beta1_09@1337@TTT_BETA1=0.9"
  "beta2_995@1337@TTT_BETA2=0.995"

  # === WD sweep (default 2.0) ===
  "wd175@1337@TTT_WEIGHT_DECAY=1.75"
  "wd25@1337@TTT_WEIGHT_DECAY=2.5"
  "wd30@1337@TTT_WEIGHT_DECAY=3.0"

  # === LoRA placement (turn off one) ===
  "no_mlp_lora@1337@TTT_MLP_LORA=0"
  "no_o_lora@1337@TTT_O_LORA=0"

  # === Prefix docs further-tuning around 1500 ===
  "prefix1750@1337@PHASED_TTT_PREFIX_DOCS=1750"
)

echo "=== NIGHT BATCH STARTED [$(date)] ==="
echo "Total experiments: ${#EXPS[@]}"
echo "Estimated runtime: $((${#EXPS[@]} * 25)) min = $((${#EXPS[@]} * 25 / 60)) hours"
echo ""

for spec in "${EXPS[@]}"; do
  IFS='@' read -r name seed env <<< "$spec"
  echo ""
  echo "############################################"
  echo "# [$(date)] EXPERIMENT: $name seed=$seed env=$env"
  echo "############################################"
  bash work/run_seeds.sh "$name" "$seed" $env 2>&1 || echo "WARNING: $name failed (exit=$?)"
done

echo ""
echo "=== NIGHT BATCH DONE [$(date)] ==="
echo "Results summary:"
for spec in "${EXPS[@]}"; do
  IFS='@' read -r name seed env <<< "$spec"
  log="work/logs/${name}_seed${seed}.log"
  if [ -f "$log" ]; then
    bpb=$(grep -aE 'quantized_ttt_phased.*val_bpb' "$log" | tail -1 | grep -oE 'val_bpb:[0-9.]+' | head -1)
    echo "$name: $bpb"
  else
    echo "$name: NO LOG"
  fi
done
