# LoRA-TTT chunk size 36 — 1.06900 BPB

**val_bpb: 1.06899693** (3-seed mean: seeds 1337, 42, 314)

## Results

| Seed | BPB | Train | Eval | Artifact |
|------|-----|-------|------|----------|
| 1337 | 1.06938166 | 599.5s | 447.8s | 15,977,143 B |
| 42   | 1.06837831 | 599.6s | 452.8s | 15,975,722 B |
| 314  | 1.06923083 | 599.7s | 454.5s | 15,974,168 B |
| **Mean** | **1.06899693** | | | |

All runs: train ≤600s, eval ≤600s, artifact ≤16MB.

## Novel contribution: smaller LoRA-TTT chunk size for finer eval-time adaptation

Building on the 1.06934 record (Phased TTT prefix=1500 + Fused CE + WD=2.0 + warm-start LoRA), the only change is **TTT_CHUNK_SIZE reduced from 48 → 36**.

LoRA-TTT scores tokens in chunks: each chunk's tokens are scored under the current LoRA, then a single grad step updates the LoRA before the next chunk. Smaller chunks = more frequent updates = LoRA tracks the local document structure more responsively.

Chunk size sweep:

| TTT_CHUNK_SIZE | Seed 1337 BPB | Eval time (s) | Status |
|---|---|---|---|
| 32 | 1.06963 | **600.2** | over 600s budget — illegal |
| **36** (this) | **1.06938** | 447.8 | best legal |
| 40 | 1.07008 | 415.5 | worse |
| 48 (PR-1.06934 default) | 1.06971 | 409.7 | baseline |
| 64 | 1.06978 | 402.9 | worse |

The dependence is sharp and non-monotonic: chunk=36 dramatically helps, chunk=40 hurts, chunk=32 helps but exceeds the 600s eval cap. Chunk=36 is the sweet spot — small enough that more grad steps land per document, large enough that each step has stable batch statistics, and fast enough to fit in the eval budget.

3-seed mean per-seed deltas vs PR-1.06934 record:

| Seed | PR-1.06934 | This | Δ |
|------|-----------|------|---|
| 1337 | 1.06971 | 1.06938 | −0.00033 |
| 42 | 1.06903 | 1.06838 | −0.00065 |
| 314 | 1.06927 | 1.06923 | −0.00004 |
| **Mean** | **1.06934** | **1.06900** | **−0.00034** |

Combination check: rank=96 + chunk=36 yields **1.06954** on seed 1337, *worse* than chunk=36 alone (1.06938) and worse than rank=96 alone (1.06964). The two changes don't compound, so the chunk-only path is the best operating point we've found.

## Stack summary

| Component | Origin |
|-----------|--------|
| **Novel: TTT_CHUNK_SIZE=36** | this author |
| Phased TTT prefix=1500 | this author (PR-1.06934) |
| Fused softcap CE + WD=2.0 | this author + @nprime06 (PR-1.06957) |
| Polar Express NS coefficients | PR #1344 |
| MIN_LR=0.10, GPTQ_RESERVE=0.5, VAL_LOSS_EVERY=0 | @nprime06 (PR #1787) |
| Per-head GatedAttn + per-row int8 gate quant + gate mirror in LoRA-TTT path | this author (PR #1768) |
| Alpha/rank LoRA scaling, warm-start A, alpha=144, rank=128 | this author (PR #1767) |
| Multi-phase global SGD, trimmed GPTQ, MATRIX_LR=0.026 | @dexhunter |
| VarLen attention, Fused Triton MLP, doc-independent LoRA TTT | @samacqua (PR #1530) |
| Phased TTT framework | @romeerp (PR #1610), @dexhunter |
| Triple recurrence, parallel residuals | @bigbag (PR #1493), @EthanYangTW (PR #1523) |
| Legal TTT framework | @abaybektursun (PR #549) |

## Hardware

Trained on **RunPod 8xH100 80GB SXM** (not Zoom MLP cluster). PyTorch 2.9.1+cu128, FA3, Triton 3.5.1. Identical SP8192 SentencePiece tokenizer and FineWeb document selection as upstream HF dataset `willdepueoai/parameter-golf`.

## Legality (Issue #1017)

- **Condition 1 (Causal)**: single left-to-right pass; LoRA at `t` depends only on earlier tokens of the same doc.
- **Condition 2 (Full normalized distribution)**: standard softcap-tanh + softmax over 8192 SP tokens.
- **Condition 3 (Score-before-update)**: each chunk scored before the LoRA grad step.
- **Condition 4 (Single pass)**: one left-to-right pass, no rescoring.
- **Fused CE is training-only.** The `forward_logits` eval path keeps eager `logit_softcap * torch.tanh(logits/softcap)` numerics — only the training forward uses the fused kernel.

## Reproduction

```bash
export DATA_DIR=/path/to/parameter-golf/data

torchrun --standalone --nproc_per_node=8 train_gpt.py        # seed 1337
SEED=42  torchrun --standalone --nproc_per_node=8 train_gpt.py
SEED=314 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All hyperparameters are hardcoded as defaults: `TTT_CHUNK_SIZE=36`, `PHASED_TTT_PREFIX_DOCS=1500`,
`TTT_WEIGHT_DECAY=2.0`, `FUSED_CE_ENABLED=1`, `POLAR_EXPRESS_NS=1`, `MIN_LR=0.10`,
`GPTQ_RESERVE_SECONDS=0.5`, `VAL_LOSS_EVERY=0`, `TTT_LORA_RANK=128`, `TTT_LORA_ALPHA=144`,
`TTT_WARM_START_A=1`, `GATED_ATTN_ENABLED=1`, `PHASED_TTT_ENABLED=1`,
`PHASED_TTT_NUM_PHASES=3`.
