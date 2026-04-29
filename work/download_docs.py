"""Download docs_selected.jsonl from HF willdepueoai/parameter-golf and place at expected path."""
import os, sys, time
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "willdepueoai/parameter-golf"
FILENAME = "docs_selected.jsonl"
SUBFOLDER = "datasets"
TARGET = Path("/workspace/pgolf/data/docs_selected.jsonl")
HF_CACHE = "/workspace/.cache/huggingface"

os.makedirs(HF_CACHE, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_HUB_CACHE"] = HF_CACHE + "/hub"

t0 = time.time()
print(f"Downloading {FILENAME} from {REPO_ID}...", flush=True)
try:
    local = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        subfolder=SUBFOLDER,
        repo_type="dataset",
        cache_dir=HF_CACHE + "/hub",
    )
    print(f"[{time.time()-t0:.1f}s] Downloaded to: {local}", flush=True)
except Exception as e:
    print(f"Subfolder lookup failed: {e}", flush=True)
    print("Trying without subfolder...", flush=True)
    local = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        repo_type="dataset",
        cache_dir=HF_CACHE + "/hub",
    )
    print(f"[{time.time()-t0:.1f}s] Downloaded to: {local}", flush=True)

TARGET.parent.mkdir(parents=True, exist_ok=True)
if TARGET.exists() or TARGET.is_symlink():
    TARGET.unlink()

# Try hard link first (local disk supports it), fall back to symlink
try:
    os.link(local, TARGET)
    print(f"[{time.time()-t0:.1f}s] Hard-linked to {TARGET}", flush=True)
except OSError as e:
    print(f"Hard link failed ({e}), creating symlink...", flush=True)
    os.symlink(local, TARGET)
    print(f"[{time.time()-t0:.1f}s] Symlinked to {TARGET}", flush=True)

import subprocess
size = subprocess.run(["du", "-h", str(TARGET.resolve())], capture_output=True, text=True).stdout.strip()
print(f"File size: {size}", flush=True)
print(f"DONE in {time.time()-t0:.1f}s", flush=True)
