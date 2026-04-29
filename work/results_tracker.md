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
