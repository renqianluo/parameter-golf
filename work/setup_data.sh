#!/bin/bash
# Setup data on RunPod: download HF docs + tokenize SP8192
set -e
cd /workspace/pgolf
echo "=== Step 1: download HF docs ==="
python3 work/download_docs.py
echo "=== Step 2: tokenize SP8192 ==="
# Ensure tokenizer is reachable at $DATA_DIR/tokenizers/ where train_gpt.py expects it
mkdir -p /workspace/pgolf/data/tokenizers
ln -sf /workspace/pgolf/work/data/tokenizers/fineweb_8192_bpe.model /workspace/pgolf/data/tokenizers/fineweb_8192_bpe.model
ln -sf /workspace/pgolf/work/data/tokenizers/fineweb_8192_bpe.vocab /workspace/pgolf/data/tokenizers/fineweb_8192_bpe.vocab
python3 work/standalone_tokenize.py
echo "=== Step 3: verify shards ==="
ls -la /workspace/pgolf/data/datasets/fineweb10B_sp8192/ | head -10
du -sh /workspace/pgolf/data/datasets/fineweb10B_sp8192/
echo "=== DONE ==="
