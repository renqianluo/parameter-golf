# RunPod Experiment Results

Baseline: **1.06957** (3-seed mean)
- seed 1337: 1.07010
- seed 42:   1.06920
- seed 314:  1.06942

| Exp | Seed | BPB | Δ vs base | Train(s) | Eval(s) | Status | Notes |
|-----|------|-----|-----------|----------|---------|--------|-------|
| baseline | 1337 | 1.07024 | +0.00014 | 599.6 | 481.1 | ok | reproduces record (1.07010) within noise |
| loralr15e5 | 1337 | 1.07023 | +0.00013 | 599.6 | 410.4 | discard | TTT_LORA_LR=0.00015 — no improvement (noise vs baseline) |
