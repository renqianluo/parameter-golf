# RunPod Experiment Results

Baseline: **1.06957** (3-seed mean)
- seed 1337: 1.07010
- seed 42:   1.06920
- seed 314:  1.06942

| Exp | Seed | BPB | Δ vs base | Train(s) | Eval(s) | Status | Notes |
|-----|------|-----|-----------|----------|---------|--------|-------|
| baseline | 1337 | 1.07024 | +0.00014 | 599.6 | 481.1 | ok | reproduces record (1.07010) within noise |
| loralr15e5 | 1337 | 1.07023 | +0.00013 | 599.6 | 410.4 | discard | TTT_LORA_LR=0.00015 — no improvement (noise vs baseline) |
| **prefix1500** | 1337 | **1.06971** | **-0.00053** | 599.6 | 409.7 | **NEW RECORD** | PHASED_TTT_PREFIX_DOCS=1500 |
| **prefix1500** | 42 | **1.06903** | -0.00121 (vs RP base est) | 599.6 | 409.3 | **NEW RECORD** | seed 42 of new record |
| **prefix1500** | 314 | **1.06927** | -0.00097 (vs RP base est) | 599.6 | 367.7 | **NEW RECORD** | seed 314 of new record |
| **prefix1500 mean** | — | **1.06934** | **−0.00023** vs old record 1.06957 | — | — | **RECORD** | 3-seed mean — submit to PR |
| prefix2500 | 1337 | 1.06988 | -0.00036 | 599.6 | 409.2 | maybe | PHASED_TTT_PREFIX_DOCS=2500 — also helps but less than 1500 |
| minlr05 | 1337 | 1.07059 | +0.00035 | 599.6 | 369.2 | discard | MIN_LR=0.05 — hurts |
| p1000s1337 | 1337 | 1.06985 | -0.00039 (worse than p1500) | 599.6 | 396.9 | discard | prefix=1000 too short; U-shape with 1500 optimal |

## Direction: prefix=1500 is now baked into work/train_gpt.py default. Next sweep new knobs on top of 1.06934.

## Night batch (24 single-seed experiments, all on top of prefix=1500)
Baseline seed 1337 = 1.06971

| Exp | seed1337 BPB | Δ vs 1.06971 | eval(s) | Verdict |
|-----|--------------|--------------|---------|---------|
| **rank96** | **1.06964** | **-0.00007** | 462.2 | BEST single-seed; multi-seed |
| chunk32 | 1.06963 | -0.00008 | **600.2** | ILLEGAL (over 600s eval cap) |
| wd30 | 1.06971 | 0 | 372.9 | tied; possibly noise |
| rank160 | 1.06976 | +0.00005 | 498.7 | slightly worse |
| alpha100 | 1.06976 | +0.00005 | 371.7 | slightly worse |
| chunk64 | 1.06978 | +0.00007 | 402.9 | slightly worse |
| wd175 | 1.06984 | +0.00013 | 376.3 | worse |
| no_mlp_lora | 1.06986 | +0.00015 | 449.8 | worse |
| phases4 | 1.06989 | +0.00018 | 385.3 | worse |
| wd25 | 1.06990 | +0.00019 | 372.9 | worse |
| loralr12 | 1.06990 | +0.00019 | 369.1 | worse |
| prefix1750 | 1.06993 | +0.00022 | 376.8 | worse |
| phases5 | 1.06996 | +0.00025 | 395.1 | worse |
| beta2_995 | 1.06997 | +0.00026 | 368.4 | worse |
| phases2 | 1.06997 | +0.00026 | 351.8 | worse |
| beta1_05 | 1.07001 | +0.00030 | 369.1 | worse |
| alpha120 | 1.07004 | +0.00033 | 406.6 | worse |
| no_o_lora | 1.07005 | +0.00034 | 451.1 | worse |
| ttt_adamw | 1.07009 | +0.00037 | 375.7 | worse |
| loralr07 | 1.07013 | +0.00042 | 375.8 | worse |
| loralr05 | 1.07022 | +0.00051 | 377.3 | worse |
| beta1_09 | 1.07092 | +0.00121 | 371.7 | much worse |
| rank192 | CRASH | — | — | OOM |
| rank256 | CRASH | — | — | OOM |

## Followup batch (in progress, started 18:12 UTC, ~3.75h)
- rank96 seeds 42, 314 (multi-seed verify)
- rank64, rank80, rank112 (extend rank trend down)
- chunk36, chunk40 (between 32 and 48, safer eval time)
- rank96+chunk40 combo
