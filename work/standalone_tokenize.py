"""Standalone SP8192 tokenizer for FineWeb. Reads docs_selected.jsonl directly."""
import json, sys, time
from pathlib import Path
import numpy as np
import sentencepiece as spm

JSONL = Path("/workspace/pgolf/data/docs_selected.jsonl")
TOKENIZER = "/workspace/pgolf/work/data/tokenizers/fineweb_8192_bpe.model"
OUT_DIR = Path("/workspace/pgolf/data/datasets/fineweb10B_sp8192")

DATAFILE_MAGIC = 20240520
DATAFILE_VERSION = 1
SHARD_SIZE = 10**8
NUM_VAL_DOCS = 50_000
APPEND_EOS = False

def write_shard(path, toks_list):
    toks = np.concatenate([np.asarray(t, dtype=np.uint16) for t in toks_list])
    if len(toks) >= 2**31:
        raise ValueError("token count too large")
    header = np.zeros(256, dtype="<i4")
    header[0] = DATAFILE_MAGIC
    header[1] = DATAFILE_VERSION
    header[2] = len(toks)
    with path.open("wb") as f:
        f.write(header.tobytes())
        f.write(toks.astype("<u2", copy=False).tobytes())
    return len(toks)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER)
    bos_id = sp.bos_id() if sp.bos_id() >= 0 else 1
    print(f"vocab={sp.vocab_size()}, bos_id={bos_id}", flush=True)

    val_buf = []
    val_count = 0
    train_buf = []
    train_shard_idx = 0
    train_buf_tokens = 0
    val_done = False

    t0 = time.time()
    docs_processed = 0
    with JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                text = obj.get("text") or obj.get("content") or ""
            except Exception:
                continue
            if not text:
                continue
            tokens = sp.encode(text, out_type=int)
            tokens.insert(0, bos_id)  # prepend BOS
            arr = np.array(tokens, dtype=np.uint16)
            docs_processed += 1
            if not val_done and val_count < NUM_VAL_DOCS:
                val_buf.append(arr)
                val_count += 1
                if val_count >= NUM_VAL_DOCS:
                    val_done = True
                    val_path = OUT_DIR / "fineweb_val_000000.bin"
                    n = write_shard(val_path, val_buf)
                    print(f"[{time.time()-t0:.1f}s] wrote val shard: {n} tokens, {val_count} docs", flush=True)
                    val_buf = None
            else:
                train_buf.append(arr)
                train_buf_tokens += len(arr)
                if train_buf_tokens >= SHARD_SIZE:
                    path = OUT_DIR / f"fineweb_train_{train_shard_idx:06d}.bin"
                    n = write_shard(path, train_buf)
                    print(f"[{time.time()-t0:.1f}s] wrote train shard {train_shard_idx}: {n} tokens, {docs_processed} docs total", flush=True)
                    train_buf = []
                    train_buf_tokens = 0
                    train_shard_idx += 1
    if train_buf:
        path = OUT_DIR / f"fineweb_train_{train_shard_idx:06d}.bin"
        n = write_shard(path, train_buf)
        print(f"[{time.time()-t0:.1f}s] wrote final train shard {train_shard_idx}: {n} tokens", flush=True)
    print(f"[{time.time()-t0:.1f}s] DONE: {docs_processed} docs, val: {val_count}, train shards: {train_shard_idx + 1}", flush=True)

if __name__ == "__main__":
    main()
