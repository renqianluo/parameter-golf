# RunPod Experiment Results

Baseline: **1.06957** (3-seed mean)
- seed 1337: 1.07010
- seed 42:   1.06920
- seed 314:  1.06942

| Exp | Seed | BPB | Δ vs base | Train(s) | Eval(s) | Status | Notes |
|-----|------|-----|-----------|----------|---------|--------|-------|
| baseline | 1337 | 1.07024 | +0.00014 | 599.6 | 481.1 | ok | reproduces record (1.07010) within noise |
| loralr15e5 | 1337 | 1.07023 | +0.00013 | 599.6 | 410.4 | discard | TTT_LORA_LR=0.00015 — no improvement (noise vs baseline) |
| **prefix1500** | 1337 | **1.06971** | **-0.00053** | 599.6 | 409.7 | **PROMISING** | PHASED_TTT_PREFIX_DOCS=1500 — best knob found, run multi-seed |
| prefix2500 | 1337 | 1.06988 | -0.00036 | 599.6 | 409.2 | maybe | PHASED_TTT_PREFIX_DOCS=2500 — also helps but less than 1500 |
| minlr05 | 1337 | 1.07059 | +0.00035 | 599.6 | 369.2 | discard | MIN_LR=0.05 — hurts |
