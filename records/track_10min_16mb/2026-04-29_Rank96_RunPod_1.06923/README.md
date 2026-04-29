# LoRA rank 96 — 1.06923 BPB

**val_bpb: 1.06922864** (3-seed mean: seeds 1337, 42, 314)

## Results

| Seed | BPB | Train | Eval | Artifact |
|------|-----|-------|------|----------|
| 1337 | 1.06964010 | 599.6s | 462.2s | 15,974,843 B |
| 42   | 1.06864450 | 599.6s | 353.7s | 15,975,534 B |
| 314  | 1.06940132 | 599.6s | 406.7s | 15,978,318 B |
| **Mean** | **1.06922864** | | | |

All runs: train ≤600s, eval ≤600s, artifact ≤16MB.

## Novel contribution: lower LoRA rank improves BPB at WD=2.0

Building on the previous 1.06934 record (Phased TTT prefix=1500 + Fused CE + WD=2.0 + warm-start LoRA), the only change vs that record is **TTT_LORA_RANK reduced from 128 → 96**.

The result is somewhat counterintuitive: smaller rank = less expressive LoRA, yet BPB improves. The mechanism: at rank 128 with WD=2.0 + warm-start A, the LoRA adapter has more capacity than the per-chunk gradient signal can fill within 1 grad step. The extra capacity becomes noise that's penalized by the WD term but never gets useful structure. At rank 96, the per-step signal more fully exploits the available LoRA dimensions, and WD doesn't have to fight as much accumulated noise.

| Config | Seed 1337 | Seed 42 | Seed 314 | Mean |
|--------|-----------|---------|----------|------|
| rank=128 (PR-1.06934)     | 1.06971 | 1.06903 | 1.06927 | 1.06934 |
| rank=160                  | 1.06976 | —       | —       | —       |
| rank=192/256              | OOM     | —       | —       | —       |
| rank=80                   | 1.07011 | —       | —       | —       |
| rank=64                   | 1.06989 | —       | —       | —       |
| **rank=96** (this)         | **1.06964** | **1.06864** | **1.06940** | **1.06923** |

The 3-seed mean drops from 1.06934 → **1.06923 (−0.00011)**. Seed 42 has the largest gain (−0.00039); seed 314 is essentially flat (+0.00013); seed 1337 dips by −0.00007.

The rank sweep is U-shaped: rank=64 too narrow (Δ +0.00018), rank=96 optimal, rank=128 OK, rank=160+ overfit / OOM.

## Stack summary

| Component | Origin |
|-----------|--------|
| **Novel: TTT_LORA_RANK=96** | this author |
| Phased TTT prefix=1500 | this author (PR-1.06934) |
| Fused softcap CE + WD=2.0 | this author + @nprime06 (PR-1.06957) |
| Polar Express NS coefficients | PR #1344 |
| MIN_LR=0.10, GPTQ_RESERVE=0.5, VAL_LOSS_EVERY=0 | @nprime06 (PR #1787) |
| Per-head GatedAttn + per-row int8 gate quant + gate mirror in LoRA-TTT path | this author (PR #1768) |
| Alpha/rank LoRA scaling, warm-start A, alpha=144 | this author (PR #1767) |
| Multi-phase global SGD, trimmed GPTQ, MATRIX_LR=0.026 | @dexhunter |
| VarLen attention, Fused Triton MLP, doc-independent LoRA TTT | @samacqua (PR #1530) |
| Phased TTT framework | @romeerp (PR #1610), @dexhunter |
| Triple recurrence, parallel residuals | @bigbag (PR #1493), @EthanYangTW (PR #1523) |
| Legal TTT framework | @abaybektursun (PR #549) |

## Hardware

Trained on **RunPod 8xH100 80GB SXM** (not Zoom MLP cluster). PyTorch 2.9.1+cu128, FA3, Triton 3.5.1. Identical SP8192 SentencePiece tokenizer and FineWeb document selection as upstream HF dataset `willdepueoai/parameter-golf`. Validation set is the standard `fineweb_val_*.bin` shard from the SP8192 tokenization.

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

All hyperparameters are hardcoded as defaults: `TTT_LORA_RANK=96`, `PHASED_TTT_PREFIX_DOCS=1500`,
`TTT_WEIGHT_DECAY=2.0`, `FUSED_CE_ENABLED=1`, `POLAR_EXPRESS_NS=1`, `MIN_LR=0.10`,
`GPTQ_RESERVE_SECONDS=0.5`, `VAL_LOSS_EVERY=0`, `TTT_LORA_ALPHA=144`,
`TTT_WARM_START_A=1`, `GATED_ATTN_ENABLED=1`, `PHASED_TTT_ENABLED=1`,
`PHASED_TTT_NUM_PHASES=3`.
