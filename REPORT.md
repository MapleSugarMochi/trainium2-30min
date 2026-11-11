# Trainium R1 Report

Nov 11, 2026 · Xuzhou Dong

## Result

The final config reaches **val_bpb 1.0605**, down **0.199** from the baseline's 1.2591. Two changes did most of it. First, the training loop was made fast: device batch 8, `torch.compile`, compiled optimizers, an NKI kernel and a time-based schedule took it to 1.1102. Second, a precision bug was found: an fp32 parameter promoted the whole bf16 residual stream to fp32, so every matmul ran in fp32. Fixing it doubled throughput, and the extra speed went into a bigger model (depth 14) and a 6x bigger batch (384K tokens per step). *Single seed; seed-to-seed std is ±0.004, and four one-knob variants of the final config scored 1.0602–1.0614, so it sits on a flat region.*

|  | Baseline | Phase 1 (fp32 matmuls) | Final |
| --- | --- | --- | --- |
| val_bpb (public eval) | 1.2591 | 1.1102 | **1.0605** |
| Model | depth 10, 640 wide, 59.6M params | depth 12, 768 wide | depth 14, 896 wide, 149.6M params |
| Tokens per step | 65,536 | 131,072 | 393,216 |
| Optimizer steps in 30 min | 1000 (step cap hit at 1590 s) | 1183 | 679 |
| s/step | 1.60 | 1.52 | 2.66 |
| Tokens per second | 41K | 86K | 148K |
| Model FLOP/s (share of ~667 TFLOP/s BF16 peak) | — | ~52 TFLOP/s (8%) | ~149 TFLOP/s (22%) |

The launch command is the plain one (no environment variables or flags); all settings are defaults in `train.py`, and `prepare.py check` passes:

```
NEURON_LOGICAL_NC_CONFIG=1 torchrun --standalone --nproc_per_node=8 train.py
```

## What worked

Every row is a full 30-minute run on `lnc=1 × 8`, scored with the public eval.

| Step | val_bpb | Change |
| --- | --- | --- |
| Baseline | 1.2591 | — |
| Phase 1 final | 1.1102 | −0.149: fast loop + schedule + depth 12 + 128K (details below) |
| bf16 matmul inputs | 1.0872 | −0.023: same config, 2x faster steps (0.78 vs 1.53 s) |
| Depth 14 (896 wide) | 1.0805 | −0.007 |
| Total batch 192K | 1.0757 | −0.005 |
| Total batch 256K | 1.0667 | −0.009 |
| **Total batch 384K (final)** | **1.0605** | −0.006. 512K was worse (1.0687), 320K tied (1.0625) |

| Change | Why it helps |
| --- | --- |
| Time-based linear warmdown over the last 60%; LR and Muon momentum as 0-d device tensors; no 1000-step cap | Decaying LR to zero settles the model at a lower loss. A changing Python-float LR recompiled every step on Neuron (1.5 to 8 s/step). The baseline stopped 210 s early. |
| Device batch 8, `torch.compile` of the model and both optimizers | Per-op launch cost dominated, so fewer, larger ops win: backward 0.74 to 0.30 s, optimizer 0.20 to 0.05 s per step. |
| NKI fused RoPE + QK-RMSNorm | One HBM pass per 128-row tile instead of several: forward 37% faster, 7% more steps. |
| **bf16 matmul inputs** | A `[1]`-shaped fp32 scalar (`resid_lambda`) multiplied the bf16 residual stream, and type promotion kept it fp32 from there on, so every Linear ran in fp32. Found with forward hooks printing each Linear's input dtype. Casting Linear inputs to bf16 halved the step time (1.53 to 0.78 s). |
| Depth 14, total batch 384K, Muon LR 0.035 | Faster steps pay for a bigger model and batch. Depth 16 does not fit at device batch 8; 512K leaves too few steps; Muon LR is a sharp optimum (0.028: 1.0710, 0.045: 1.0649). |

## What did not work

| Change | Result | Why |
| --- | --- | --- |
| Launch shape lnc=2 × 4 procs | 1.68 vs 1.42 s/step | Slower in both phases, also with device batch 16 per logical core. |
| Other kernels and fusions: example relu², q/k written as `[B,H,T,D]`, NKI RMSNorm, fused QKV, bucketed all-reduce | 0 to +27% step time | The compiler already fuses these, or the extra copies cost more than they save. |
| Compiler flags `-O3`, `--model-type=transformer` | within 2% | `-O3` doubled compile time. |
| Device batch 12 or 16; depth 16 or U-net at device batch 8 | crash | Out of device memory (12 GiB per NeuronCore). |
| Depth 16 or U-net at device batch 4; depth 15 or 13 at width 896 | 1.0784, 1.0847, 1.0737, 1.0708 | Depth 14 is the 30-minute optimum. |

Recipe changes on the final config (1.0605), one knob each: weight decay 0.05 (1.0602) or 0 (1.0606), warmdown 70% (1.0608) or 50% (1.0615), embedding LR 0.2 (1.0609), logit softcap 12 (1.0614) all tied; total batch 320K (1.0625), lm\_head LR ×0.67 (1.0635), 1-sqrt warmdown (1.0638), Muon LR 0.04 (1.0629), 0.045 (1.0649) or 0.028 (1.0710), total batch 512K (1.0687) and embedding/lm\_head/scalar LRs ×1.5 (1.0748) were worse; gradient clipping does not compile on Neuron (`aten::_foreach_norm`).

## Where step time goes, and the remaining bottleneck

In the final config the chip runs at about **22% of its BF16 peak**: 1.0 GFLOP per token (from the training log) × 148K tokens/s ≈ 149 TFLOP/s, against about 667 TFLOP/s. The remaining bottleneck is **forward and backward compute made of many mid-sized ops, with device memory capping the per-core batch**. Evidence:

- Memory is the cap: device batch 16, depth 16 and U-net skips fail to allocate at 12 GiB per NeuronCore, and the device-batch-4 workarounds are slower per token.
- Not communication or host: all-reduce was 7% of the step, and lnc=2 (half the host processes) was slower.

Next levers: a flash-attention NKI kernel and activation checkpointing to free memory for a bigger device batch or depth; a fused MLP kernel to cut HBM traffic.

## Method notes

- **Noise.** Same config and seed reproduce val\_bpb exactly (±0.001 from time-based stopping); seed std is 0.004. Phase 1 confirmed small changes on two seeds; phase 2 kept a change only if it beat the best by 0.008, or by 0.004–0.008 in line with a trend. Every kept result is a full 30-minute run (10-minute screens favour small models). All 130 runs are in `results.tsv`; logs and the runner are in [`experiments/`](experiments/README.md).
- **Anything that changes the forward pass must be in the checkpoint.** A softcap set only by environment variable was evaluated with the default (invalid 1.1123); it is now in the config.
