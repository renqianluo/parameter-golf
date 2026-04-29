#!/bin/bash
# Setup data on RunPod: download HF docs + tokenize SP8192
set -e
cd /workspace/pgolf
echo "=== Step 1: download HF docs ==="
python3 work/download_docs.py
echo "=== Step 2: tokenize SP8192 ==="
python3 work/standalone_tokenize.py
echo "=== Step 3: verify shards ==="
ls -la /workspace/pgolf/data/datasets/fineweb10B_sp8192/ | head -10
du -sh /workspace/pgolf/data/datasets/fineweb10B_sp8192/
echo "=== DONE ==="
