# Phased TTT prefix=1500 — 1.06934 BPB

**val_bpb: 1.06933770** (3-seed mean: seeds 1337, 42, 314)

## Results

| Seed | BPB | Train | Eval | Artifact |
|------|-----|-------|------|----------|
| 1337 | 1.06971119 | 599.6s | 409.7s | 15,975,596 B |
| 42   | 1.06902789 | 599.5s | 409.3s | 15,978,587 B |
| 314  | 1.06927402 | 599.6s | 367.7s | 15,971,896 B |
| **Mean** | **1.06933770** | | | |

All runs: train ≤600s, eval ≤600s, artifact ≤16MB.

## Novel contribution: shorter Phased-TTT prefix unlocks more eval-time learning

Building on PR #1768/PR-1.06957 stack (Fused CE + WD=2.0 + warm-start LoRA + GatedAttn),
the only change vs that record is **PHASED_TTT_PREFIX_DOCS reduced from 2000 → 1500**.

Phased TTT splits the 50K validation documents into a *prefix* (base-model warmup, no
TTT updates) and a *suffix* (split into K=3 phases, with full LoRA-TTT updates +
between-phase global SGD on the base model). The prefix exists to let the LoRA
warm-start pool stabilize before the actual scored phases begin.

PHASED_TTT_PREFIX_DOCS=2000 (the previous default) wastes ~10% of the eval budget
on docs that are already easier-to-predict (the base model is well-trained on them via
warm-start A's accumulated features). Cutting prefix to 1500:

| Config | Seed 42 | Seed 314 | Seed 1337 | Mean |
|--------|---------|----------|-----------|------|
| prefix=2000 (PR #1.06957)   | 1.06920 | 1.06942 | 1.07010 | 1.06957 |
| prefix=2500                 | —       | —       | 1.06988 | —       |
| **prefix=1500** (this)      | **1.06903** | **1.06927** | **1.06971** | **1.06934** |

All 3 seeds improve (Δ −0.00015 to −0.00039 each). Mean delta: −0.00023.

The mechanism: with 1500 prefix docs, the LoRA-TTT loop sees more docs in the suffix
(48000 → 48500), and the per-phase boundaries shift correspondingly:

- prefix=2000: boundaries [666, 1333, 2000] → 16000 docs/phase in suffix
- prefix=1500: boundaries [499, 999, 1500] → 16166 docs/phase in suffix

The extra 500 docs in the suffix get full TTT updates instead of just being scored
against a frozen base model.

## Stack summary

| Component | Origin |
|-----------|--------|
| **Novel: PHASED_TTT_PREFIX_DOCS=1500** | this author |
| Fused softcap CE + WD=2.0 | this author + @nprime06 (PR #1.06957) |
| Polar Express NS coefficients | PR #1344 |
| MIN_LR=0.10, GPTQ_RESERVE=0.5, VAL_LOSS_EVERY=0 | @nprime06 (PR #1787) |
| Per-head GatedAttn + per-row int8 gate quant + gate mirror in LoRA-TTT path | this author (PR #1768) |
| Alpha/rank LoRA scaling, warm-start A, alpha=144 | this author (PR #1767) |
| Multi-phase global SGD, trimmed GPTQ, MATRIX_LR=0.026 | @dexhunter |
| VarLen attention, Fused Triton MLP, doc-independent LoRA TTT | @samacqua (PR #1530) |
| Phased TTT | @romeerp (PR #1610), @dexhunter |
| Triple recurrence, parallel residuals | @bigbag (PR #1493), @EthanYangTW (PR #1523) |
| Legal TTT framework | @abaybektursun (PR #549) |

## Hardware

Trained on **RunPod 8xH100 80GB SXM** (not Zoom MLP cluster). PyTorch 2.9.1+cu128, FA3, Triton 3.5.1. Identical SP8192 SentencePiece tokenizer and FineWeb document selection as upstream HF dataset `willdepueoai/parameter-golf`. Validation set is the standard `fineweb_val_*.bin` shard from the SP8192 tokenization.

## Legality (Issue #1017)

- **Condition 1 (Causal)**: single left-to-right pass; LoRA at `t` depends only on earlier tokens of the same doc.
- **Condition 2 (Full normalized distribution)**: standard softcap-tanh + softmax over 8192 SP tokens.
- **Condition 3 (Score-before-update)**: each chunk scored before the LoRA grad step.
- **Condition 4 (Single pass)**: one left-to-right pass, no rescoring.
- **Phased TTT prefix is unscored warmup**: the first 1500 documents are processed by the base model only (no TTT updates) and ARE scored as part of the validation BPB. The "prefix" terminology refers to phase scheduling for the LoRA optimizer, not to dropping any tokens from the BPB calculation. All 50K validation docs are scored.
- **Fused CE is training-only.** The `forward_logits` eval path keeps eager `logit_softcap * torch.tanh(logits/softcap)` numerics — only the training forward uses the fused kernel.

## Reproduction

```bash
export DATA_DIR=/path/to/parameter-golf/data

torchrun --standalone --nproc_per_node=8 train_gpt.py        # seed 1337
SEED=42  torchrun --standalone --nproc_per_node=8 train_gpt.py
SEED=314 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All hyperparameters are hardcoded as defaults: `PHASED_TTT_PREFIX_DOCS=1500`,
`TTT_WEIGHT_DECAY=2.0`, `FUSED_CE_ENABLED=1`, `POLAR_EXPRESS_NS=1`, `MIN_LR=0.10`,
`GPTQ_RESERVE_SECONDS=0.5`, `VAL_LOSS_EVERY=0`, `TTT_LORA_RANK=128`,
`TTT_LORA_ALPHA=144`, `TTT_WARM_START_A=1`, `GATED_ATTN_ENABLED=1`,
`PHASED_TTT_ENABLED=1`, `PHASED_TTT_NUM_PHASES=3`.
