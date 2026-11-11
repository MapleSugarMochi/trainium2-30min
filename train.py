"""
Neuron Competition R1 baseline training script.

Modified by Xuzhou Dong (2026) from the AWS Trainium Frontier R1 starter kit
(Apache-2.0). The original is in baseline/train.py; REPORT.md lists the changes.

Participants edit this file. It is intentionally self-contained and readable.
The fixed data/tokenizer/eval utilities live in prepare.py.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Proxy for process launch, captured before the heavy imports below. Used to
# align the self-managed budget clock with the runner's startup allowance.
_PROCESS_STARTED = time.monotonic()

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    EVAL_TIMEOUT_SECONDS,
    TOKENIZER_VOCAB_SIZE,
    TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS,
    TRAIN_STARTUP_STEPS_EXCLUDED,
    TRAIN_TIME_BUDGET_SECONDS,
    Tokenizer,
    causality_check,
    ensure_tokenizer,
    evaluate_bpb,
    get_dist_info,
    make_dataloader,
    print0,
    record_step_boundary,
)


# -----------------------------------------------------------------------------
# Baseline hyperparameters. Model-related config lives here, not in prepare.py.

def _env(name: str, default):
    """Hyperparameter override from the environment (NEURON_HP_<NAME>), for sweeps."""
    raw = os.environ.get(f"NEURON_HP_{name}")
    if raw is None:
        return default
    if isinstance(default, bool):
        # bool("0") is True, so parse flags explicitly.
        return raw.strip().lower() not in ("0", "false", "no", "off", "")
    return type(default)(raw)


SEQ_LEN = _env("SEQ_LEN", 1024)
DEPTH = _env("DEPTH", 14)
ASPECT_RATIO = _env("ASPECT_RATIO", 64)
HEAD_DIM = 128

DEVICE_BATCH_SIZE = _env("DEVICE_BATCH_SIZE", 8)
TOTAL_BATCH_SIZE = _env("TOTAL_BATCH_SIZE", 393216)
# Time-based training: the run ends when the 30-minute budget is used up, so the
# step cap is only a safety net (the baseline's 1000 could end a fast run early).
NUM_STEPS = _env("NUM_STEPS", 1_000_000)
# The official runner charges wall time only after the first five completed
# optimizer steps (the startup/compilation allowance, capped). The baseline
# restarts its budget clock at that boundary so it can use the full charged
# budget even when compilation makes the first steps slow.
MAX_TRAIN_SECONDS = TRAIN_TIME_BUDGET_SECONDS
CHECKPOINT_RESERVE_SECONDS = 5.0
# Covers the launch -> module-import gap this process cannot observe when
# checking its startup time against the allowance cap.
STARTUP_LAUNCH_MARGIN_SECONDS = 30.0

LOGIT_SOFTCAP = _env("LOGIT_SOFTCAP", 15.0)

EMBEDDING_LR = _env("EMBEDDING_LR", 0.30)
UNEMBEDDING_LR = _env("UNEMBEDDING_LR", 0.006)
MATRIX_LR = _env("MATRIX_LR", 0.035)
SCALAR_LR = _env("SCALAR_LR", 0.20)
WEIGHT_DECAY = _env("WEIGHT_DECAY", 0.10)
WARMUP_STEPS = _env("WARMUP_STEPS", 20)
# Fraction of the (time) budget spent linearly decaying LR to FINAL_LR_FRAC.
WARMDOWN_FRAC = _env("WARMDOWN_FRAC", 0.6)
FINAL_LR_FRAC = _env("FINAL_LR_FRAC", 0.0)
WARMDOWN_SHAPE = _env("WARMDOWN_SHAPE", "linear")  # "linear" or "sqrt" (1-sqrt cooldown)
GRAD_CLIP = _env("GRAD_CLIP", 0.0)


# -----------------------------------------------------------------------------
# Device / distributed setup

def neuron_available() -> bool:
    try:
        torch.device("neuron")
        return True
    except Exception:
        return False


def detect_device_type(requested: str = "") -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if neuron_available():
        return "neuron"
    return "cpu"


def init_runtime(device_type: str):
    assert device_type in {"cpu", "cuda", "neuron"}
    torch.manual_seed(_env("SEED", 42))
    ddp, rank, local_rank, world_size = get_dist_info()
    if ddp and device_type == "cuda":
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", device_id=device)
    elif ddp and device_type == "neuron":
        dist.init_process_group("neuron")
        device = torch.device("neuron")
    elif ddp:
        dist.init_process_group("gloo")
        device = torch.device("cpu")
    else:
        device = torch.device(device_type)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return ddp, rank, local_rank, world_size, device


def cleanup_runtime() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "neuron" and hasattr(torch, "neuron"):
        torch.neuron.synchronize()


def compute_dtype_for(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "neuron":
        return torch.bfloat16
    return torch.float32


# -----------------------------------------------------------------------------
# Model

@dataclass
class GPTConfig:
    sequence_len: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    # Architecture options (stored in the checkpoint so load_for_eval rebuilds the same model).
    value_resid: bool = False  # every layer's v mixes in layer 0's v (learned 2 scalars per layer)
    unet: bool = False         # U-net skips: layer i's output feeds layer n-1-i (learned weight each)
    logit_softcap: float = 15.0  # tanh softcap on the logits (stored so eval uses the trained value)
    # Forward-path switches that were env-only (NEURON_HP_*). Stored so load_for_eval
    # runs the forward the model was trained with, not whatever the eval env says.
    # Defaults equal the env defaults, so older checkpoints load unchanged.
    bf16_matmul: bool = True
    qkv_cat: bool = False
    nki_qknorm: bool = True
    nki_rmsnorm: bool = False


COMPUTE_DTYPE: torch.dtype = torch.bfloat16


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


# -----------------------------------------------------------------------------
# Example: a custom NKI kernel (optional, default OFF)
#
# This is an ILLUSTRATION to get you started writing your own Trainium kernels.
# It implements the MLP activation `relu(x)**2` as a fused NKI kernel. The eager
# baseline does not use it; enable it with `--nki-relu2` (Neuron only).
#
# Three things worth copying from this example:
#   1. The kernel body: load a tile from HBM into SBUF, compute, store back.
#   2. Tiling over the 128-partition limit: a NeuronCore processes at most 128
#      rows (the "partition" dim) at once, so we loop over chunks of 128 rows.
#   3. Autograd: a raw NKI kernel is NOT differentiable. To use one inside the
#      training forward pass you must wrap it in a torch.autograd.Function and
#      provide the backward yourself, or gradients silently stop flowing.
#
# To write your own: change the per-tile math in `_relu_sq_fwd` (and the matching
# derivative in `_relu_sq_bwd`), or fuse a different elementwise op.

_NKI_PARTITION = 128  # NeuronCore SBUF partition limit (== nki.language.tile_size.pmax)

try:
    import nki
    import nki.language as nl
    import nki.isa as nisa

    @nki.jit
    def _relu_sq_fwd(x):
        """Fused y = relu(x)**2 for a 2-D [M, N] tensor, tiled over rows."""
        M, N = x.shape
        out = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)
        for t in nl.affine_range((M + _NKI_PARTITION - 1) // _NKI_PARTITION):
            rows = nl.ds(t * _NKI_PARTITION, _NKI_PARTITION)
            xb = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=xb, src=x[rows])
            rb = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=rb, data=xb, op0=nl.maximum, operand0=0.0)  # relu
            yb = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=yb, data1=rb, data2=rb, op=nl.multiply)     # square
            nisa.dma_copy(dst=out[rows], src=yb)
        return out

    @nki.jit
    def _relu_sq_bwd(x, grad_out):
        """Backward of relu(x)**2: grad_in = grad_out * 2 * relu(x)."""
        M, N = x.shape
        out = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)
        for t in nl.affine_range((M + _NKI_PARTITION - 1) // _NKI_PARTITION):
            rows = nl.ds(t * _NKI_PARTITION, _NKI_PARTITION)
            xb = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=xb, src=x[rows])
            gb = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=gb, src=grad_out[rows])
            rb = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=rb, data=xb, op0=nl.maximum, operand0=0.0)   # relu(x)
            two_r = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=two_r, data=rb, op0=nl.multiply, operand0=2.0)
            yb = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=yb, data1=two_r, data2=gb, op=nl.multiply)
            nisa.dma_copy(dst=out[rows], src=yb)
        return out

    class _ReluSquaredNKI(torch.autograd.Function):
        """Makes the NKI relu**2 kernel differentiable so it can train."""

        @staticmethod
        def forward(ctx, x):
            y = _relu_sq_fwd(x.reshape(-1, x.shape[-1])).reshape(x.shape)
            ctx.save_for_backward(x)
            return y

        @staticmethod
        def backward(ctx, grad_out):
            (x,) = ctx.saved_tensors
            flat_x = x.reshape(-1, x.shape[-1])
            flat_g = grad_out.reshape(-1, x.shape[-1]).contiguous()
            return _relu_sq_bwd(flat_x, flat_g).reshape(x.shape)

    _HAS_NKI = True
except Exception:  # nki unavailable (e.g. CPU/GPU dev box) -- eager path still works
    _HAS_NKI = False

# Toggled by the --nki-relu2 flag in main(). Default OFF: the shipped baseline is
# numerically identical to the plain eager `relu(x)**2`.
USE_NKI_RELU2 = False


def relu_squared(x: torch.Tensor) -> torch.Tensor:
    """relu(x)**2, via the example NKI kernel when enabled, else eager."""
    # The example kernel tiles rows in blocks of 128, so it needs a row count that
    # is a multiple of 128 (true for the baseline: B*T = 1*1024). Otherwise fall
    # back to eager so the knob is always safe to flip.
    if USE_NKI_RELU2 and _HAS_NKI and x.device.type == "neuron" and (x.numel() // x.shape[-1]) % _NKI_PARTITION == 0:
        return _ReluSquaredNKI.apply(x)
    return F.relu(x).square()


BF16_MATMUL = os.environ.get("NEURON_HP_BF16_MATMUL", "1") != "0"


class Linear(nn.Linear):
    """Linear without bias that casts weight to input dtype (allows bf16 input with f32 weight).

    The baseline's residual stream is fp32 in practice: resid_lambda/x0_lambda are
    fp32 [1]-shaped parameters, and `resid_lambda * x` promotes the bf16 activations
    to fp32, so every matmul ran in fp32. With BF16_MATMUL the matmul input (and so
    the weight) is cast to COMPUTE_DTYPE; the residual stream itself stays fp32."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if BF16_MATMUL:
            x = x.to(COMPUTE_DTYPE)
        return F.linear(x, self.weight.to(dtype=x.dtype), None)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), dim=-1)


# -----------------------------------------------------------------------------
# Custom NKI kernel: fused RoPE + QK-RMSNorm (NEURON_HP_NKI_QKNORM=1)
#
# Why: per-component timing (exp/opbench, depth 12, batch 8, compiled) showed the
# attention block at ~240 ms/microstep vs ~65 ms for the MLP block, although the
# MLP does 2x the matmul FLOPs. Only ~75 ms is the attention core (sdpa); the
# rest is RoPE (slice + cat), the q/k RMSNorm and layout copies - elementwise
# work that round-trips HBM several times. This kernel does
#     y = rms_norm(rope(x))   for x = q or k, viewed as rows [B*T*H, head_dim]
# in one pass per 128-row tile (one HBM read of x, one write of y).
#
# Trick: RoPE rotates pairs (x1_i, x2_i), which preserves each pair's norm, so
# rms(rope(x)) == rms(x): the sum of squares is taken on the raw input.
#
# Backward (r = rope(x), y = r * rstd, rstd = 1/sqrt(mean(x^2) + eps)):
#     dr = (dy - y * mean(dy * y)) * rstd
#     dx1 = dr1 * cos - dr2 * sin ;  dx2 = dr1 * sin + dr2 * cos   (R^T dr)
# Rows of one sequence position are consecutive (H heads), so the cos/sin row
# tables are [T*H, head_dim/2]; T*H is a multiple of 128 for T = 1024.

_QK_EPS = float(torch.finfo(COMPUTE_DTYPE).eps)  # F.rms_norm(eps=None) on COMPUTE_DTYPE inputs
QKV_CAT = os.environ.get("NEURON_HP_QKV_CAT", "0") != "0"
USE_NKI_QKNORM = os.environ.get("NEURON_HP_NKI_QKNORM", "1") != "0"
KERNEL_SELFTEST = os.environ.get("NEURON_HP_KERNEL_SELFTEST", "0") != "0"

if _HAS_NKI:
    @nki.jit
    def _qk_rope_norm_fwd(x, cos_rows, sin_rows):
        N, D = x.shape
        TH, Dh = cos_rows.shape
        P = _NKI_PARTITION
        out = nl.ndarray((N, D), dtype=x.dtype, buffer=nl.shared_hbm)
        for i in nl.affine_range(N // TH):
            for j in nl.affine_range(TH // P):
                rows = nl.ds(i * TH + j * P, P)
                crow = nl.ds(j * P, P)
                x1 = nl.ndarray((P, Dh), dtype=x.dtype, buffer=nl.sbuf)
                x2 = nl.ndarray((P, Dh), dtype=x.dtype, buffer=nl.sbuf)
                c = nl.ndarray((P, Dh), dtype=cos_rows.dtype, buffer=nl.sbuf)
                sn = nl.ndarray((P, Dh), dtype=sin_rows.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=x1, src=x[rows, nl.ds(0, Dh)])
                nisa.dma_copy(dst=x2, src=x[rows, nl.ds(Dh, Dh)])
                nisa.dma_copy(dst=c, src=cos_rows[crow, :])
                nisa.dma_copy(dst=sn, src=sin_rows[crow, :])
                # sum of squares of the raw input (rotation-invariant)
                sq = nl.ndarray((P, Dh), dtype=nl.float32, buffer=nl.sbuf)
                ss1 = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                ss2 = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=sq, data1=x1, data2=x1, op=nl.multiply)
                nisa.tensor_reduce(dst=ss1, op=nl.add, data=sq, axis=1, keepdims=True)
                nisa.tensor_tensor(dst=sq, data1=x2, data2=x2, op=nl.multiply)
                nisa.tensor_reduce(dst=ss2, op=nl.add, data=sq, axis=1, keepdims=True)
                ss = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=ss, data1=ss1, data2=ss2, op=nl.add)
                rstd = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=rstd, op=nl.rsqrt, data=ss, scale=1.0 / D, bias=_QK_EPS)
                # rotation
                a = nl.ndarray((P, Dh), dtype=nl.float32, buffer=nl.sbuf)
                b = nl.ndarray((P, Dh), dtype=nl.float32, buffer=nl.sbuf)
                r1 = nl.ndarray((P, Dh), dtype=nl.float32, buffer=nl.sbuf)
                r2 = nl.ndarray((P, Dh), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=a, data1=x1, data2=c, op=nl.multiply)
                nisa.tensor_tensor(dst=b, data1=x2, data2=sn, op=nl.multiply)
                nisa.tensor_tensor(dst=r1, data1=a, data2=b, op=nl.add)          # x1 c + x2 s
                nisa.tensor_tensor(dst=a, data1=x2, data2=c, op=nl.multiply)
                nisa.tensor_tensor(dst=b, data1=x1, data2=sn, op=nl.multiply)
                nisa.tensor_tensor(dst=r2, data1=a, data2=b, op=nl.subtract)     # x2 c - x1 s
                y1 = nl.ndarray((P, Dh), dtype=x.dtype, buffer=nl.sbuf)
                y2 = nl.ndarray((P, Dh), dtype=x.dtype, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=y1, data=r1, op0=nl.multiply, operand0=rstd)
                nisa.tensor_scalar(dst=y2, data=r2, op0=nl.multiply, operand0=rstd)
                nisa.dma_copy(dst=out[rows, nl.ds(0, Dh)], src=y1)
                nisa.dma_copy(dst=out[rows, nl.ds(Dh, Dh)], src=y2)
        return out

    @nki.jit
    def _qk_rope_norm_bwd(x, dy, cos_rows, sin_rows):
        N, D = x.shape
        TH, Dh = cos_rows.shape
        P = _NKI_PARTITION
        out = nl.ndarray((N, D), dtype=x.dtype, buffer=nl.shared_hbm)
        for i in nl.affine_range(N // TH):
            for j in nl.affine_range(TH // P):
                rows = nl.ds(i * TH + j * P, P)
                crow = nl.ds(j * P, P)
                x1 = nl.ndarray((P, Dh), dtype=x.dtype, buffer=nl.sbuf)
                x2 = nl.ndarray((P, Dh), dtype=x.dtype, buffer=nl.sbuf)
                g1 = nl.ndarray((P, Dh), dtype=x.dtype, buffer=nl.sbuf)
                g2 = nl.ndarray((P, Dh), dtype=x.dtype, buffer=nl.sbuf)
                c = nl.ndarray((P, Dh), dtype=cos_rows.dtype, buffer=nl.sbuf)
                sn = nl.ndarray((P, Dh), dtype=sin_rows.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=x1, src=x[rows, nl.ds(0, Dh)])
                nisa.dma_copy(dst=x2, src=x[rows, nl.ds(Dh, Dh)])
                nisa.dma_copy(dst=g1, src=dy[rows, nl.ds(0, Dh)])
                nisa.dma_copy(dst=g2, src=dy[rows, nl.ds(Dh, Dh)])
                nisa.dma_copy(dst=c, src=cos_rows[crow, :])
                nisa.dma_copy(dst=sn, src=sin_rows[crow, :])
                sq = nl.ndarray((P, Dh), dtype=nl.float32, buffer=nl.sbuf)
                ss1 = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                ss2 = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=sq, data1=x1, data2=x1, op=nl.multiply)
                nisa.tensor_reduce(dst=ss1, op=nl.add, data=sq, axis=1, keepdims=True)
                nisa.tensor_tensor(dst=sq, data1=x2, data2=x2, op=nl.multiply)
                nisa.tensor_reduce(dst=ss2, op=nl.add, data=sq, axis=1, keepdims=True)
                ss = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=ss, data1=ss1, data2=ss2, op=nl.add)
                rstd = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=rstd, op=nl.rsqrt, data=ss, scale=1.0 / D, bias=_QK_EPS)
                # recompute y = rope(x) * rstd
                a = nl.ndarray((P, Dh), dtype=nl.float32, buffer=nl.sbuf)
                b = nl.ndarray((P, Dh), dtype=nl.float32, buffer=nl.sbuf)
                y1 = nl.ndarray((P, Dh), dtype=nl.float32, buffer=nl.sbuf)
                y2 = nl.ndarray((P, Dh), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=a, data1=x1, data2=c, op=nl.multiply)
                nisa.tensor_tensor(dst=b, data1=x2, data2=sn, op=nl.multiply)
                nisa.tensor_tensor(dst=y1, data1=a, data2=b, op=nl.add)
                nisa.tensor_tensor(dst=a, data1=x2, data2=c, op=nl.multiply)
                nisa.tensor_tensor(dst=b, data1=x1, data2=sn, op=nl.multiply)
                nisa.tensor_tensor(dst=y2, data1=a, data2=b, op=nl.subtract)
                nisa.tensor_scalar(dst=y1, data=y1, op0=nl.multiply, operand0=rstd)
                nisa.tensor_scalar(dst=y2, data=y2, op0=nl.multiply, operand0=rstd)
                # dsum = sum(dy * y) ; k = dsum / D
                d1 = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                d2 = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=a, data1=g1, data2=y1, op=nl.multiply)
                nisa.tensor_reduce(dst=d1, op=nl.add, data=a, axis=1, keepdims=True)
                nisa.tensor_tensor(dst=b, data1=g2, data2=y2, op=nl.multiply)
                nisa.tensor_reduce(dst=d2, op=nl.add, data=b, axis=1, keepdims=True)
                k = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=k, data1=d1, data2=d2, op=nl.add)
                nisa.tensor_scalar(dst=k, data=k, op0=nl.multiply, operand0=1.0 / D)
                # dr = (dy - y * k) * rstd
                nisa.tensor_scalar(dst=y1, data=y1, op0=nl.multiply, operand0=k)
                nisa.tensor_scalar(dst=y2, data=y2, op0=nl.multiply, operand0=k)
                h1 = nl.ndarray((P, Dh), dtype=nl.float32, buffer=nl.sbuf)
                h2 = nl.ndarray((P, Dh), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=h1, data1=g1, data2=y1, op=nl.subtract)
                nisa.tensor_tensor(dst=h2, data1=g2, data2=y2, op=nl.subtract)
                nisa.tensor_scalar(dst=h1, data=h1, op0=nl.multiply, operand0=rstd)
                nisa.tensor_scalar(dst=h2, data=h2, op0=nl.multiply, operand0=rstd)
                # dx = R^T dr
                o1 = nl.ndarray((P, Dh), dtype=x.dtype, buffer=nl.sbuf)
                o2 = nl.ndarray((P, Dh), dtype=x.dtype, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=a, data1=h1, data2=c, op=nl.multiply)
                nisa.tensor_tensor(dst=b, data1=h2, data2=sn, op=nl.multiply)
                nisa.tensor_tensor(dst=o1, data1=a, data2=b, op=nl.subtract)     # dr1 c - dr2 s
                nisa.tensor_tensor(dst=a, data1=h1, data2=sn, op=nl.multiply)
                nisa.tensor_tensor(dst=b, data1=h2, data2=c, op=nl.multiply)
                nisa.tensor_tensor(dst=o2, data1=a, data2=b, op=nl.add)          # dr1 s + dr2 c
                nisa.dma_copy(dst=out[rows, nl.ds(0, Dh)], src=o1)
                nisa.dma_copy(dst=out[rows, nl.ds(Dh, Dh)], src=o2)
        return out

    class _QKRopeNormNKI(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, cos_rows, sin_rows):
            D = x.shape[-1]
            flat = x.reshape(-1, D).contiguous()
            y = _qk_rope_norm_fwd(flat, cos_rows, sin_rows)
            ctx.save_for_backward(flat, cos_rows, sin_rows)
            ctx.shape = x.shape
            return y.reshape(x.shape)

        @staticmethod
        def backward(ctx, dy):
            flat, cos_rows, sin_rows = ctx.saved_tensors
            g = _qk_rope_norm_bwd(flat, dy.reshape(flat.shape).contiguous(), cos_rows, sin_rows)
            return g.reshape(ctx.shape), None, None



# -----------------------------------------------------------------------------
# Custom NKI kernel 2: fused RMSNorm forward + backward (NEURON_HP_NKI_RMSNORM=1)
#
# The two pre-norms per block (24 calls at depth 12) work on [B*T, n_embd] rows.
# One 128-row tile is a [128, 768] SBUF tile; the kernel reads x once and writes
# y once (backward: reads x and dy once, writes dx once), recomputing rstd.
#     y  = x * rstd,  rstd = 1/sqrt(mean(x^2) + eps)
#     dx = (dy - y * mean(dy * y)) * rstd

USE_NKI_RMSNORM = os.environ.get("NEURON_HP_NKI_RMSNORM", "0") != "0"

# NKI traces module-level constants but not closure variables, so each input
# dtype gets its own kernel pair with eps = finfo(dtype).eps (like F.rms_norm(eps=None)).
_EPS_F32 = float(torch.finfo(torch.float32).eps)
_EPS_BF16 = float(torch.finfo(torch.bfloat16).eps)

if _HAS_NKI:
    @nki.jit
    def _rmsnorm_fwd_f32(x):
        N, D = x.shape
        P = _NKI_PARTITION
        out = nl.ndarray((N, D), dtype=x.dtype, buffer=nl.shared_hbm)
        for i in nl.affine_range(N // P):
            rows = nl.ds(i * P, P)
            xt = nl.ndarray((P, D), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=xt, src=x[rows, :])
            sq = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
            ss = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=sq, data1=xt, data2=xt, op=nl.multiply)
            nisa.tensor_reduce(dst=ss, op=nl.add, data=sq, axis=1, keepdims=True)
            rstd = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=rstd, op=nl.rsqrt, data=ss, scale=1.0 / D, bias=_EPS_F32)
            yt = nl.ndarray((P, D), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=yt, data=xt, op0=nl.multiply, operand0=rstd)
            nisa.dma_copy(dst=out[rows, :], src=yt)
        return out

    @nki.jit
    def _rmsnorm_bwd_f32(x, dy):
        N, D = x.shape
        P = _NKI_PARTITION
        out = nl.ndarray((N, D), dtype=x.dtype, buffer=nl.shared_hbm)
        for i in nl.affine_range(N // P):
            rows = nl.ds(i * P, P)
            xt = nl.ndarray((P, D), dtype=x.dtype, buffer=nl.sbuf)
            gt = nl.ndarray((P, D), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=xt, src=x[rows, :])
            nisa.dma_copy(dst=gt, src=dy[rows, :])
            sq = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
            ss = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=sq, data1=xt, data2=xt, op=nl.multiply)
            nisa.tensor_reduce(dst=ss, op=nl.add, data=sq, axis=1, keepdims=True)
            rstd = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=rstd, op=nl.rsqrt, data=ss, scale=1.0 / D, bias=_EPS_F32)
            y = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=y, data=xt, op0=nl.multiply, operand0=rstd)
            # k = mean(dy * y)
            k = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=sq, data1=gt, data2=y, op=nl.multiply)
            nisa.tensor_reduce(dst=k, op=nl.add, data=sq, axis=1, keepdims=True)
            nisa.tensor_scalar(dst=k, data=k, op0=nl.multiply, operand0=1.0 / D)
            # dx = (dy - y * k) * rstd
            nisa.tensor_scalar(dst=y, data=y, op0=nl.multiply, operand0=k)
            h = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=h, data1=gt, data2=y, op=nl.subtract)
            ot = nl.ndarray((P, D), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=ot, data=h, op0=nl.multiply, operand0=rstd)
            nisa.dma_copy(dst=out[rows, :], src=ot)
        return out

    @nki.jit
    def _rmsnorm_fwd_bf16(x):
        N, D = x.shape
        P = _NKI_PARTITION
        out = nl.ndarray((N, D), dtype=x.dtype, buffer=nl.shared_hbm)
        for i in nl.affine_range(N // P):
            rows = nl.ds(i * P, P)
            xt = nl.ndarray((P, D), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=xt, src=x[rows, :])
            sq = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
            ss = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=sq, data1=xt, data2=xt, op=nl.multiply)
            nisa.tensor_reduce(dst=ss, op=nl.add, data=sq, axis=1, keepdims=True)
            rstd = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=rstd, op=nl.rsqrt, data=ss, scale=1.0 / D, bias=_EPS_BF16)
            yt = nl.ndarray((P, D), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=yt, data=xt, op0=nl.multiply, operand0=rstd)
            nisa.dma_copy(dst=out[rows, :], src=yt)
        return out

    @nki.jit
    def _rmsnorm_bwd_bf16(x, dy):
        N, D = x.shape
        P = _NKI_PARTITION
        out = nl.ndarray((N, D), dtype=x.dtype, buffer=nl.shared_hbm)
        for i in nl.affine_range(N // P):
            rows = nl.ds(i * P, P)
            xt = nl.ndarray((P, D), dtype=x.dtype, buffer=nl.sbuf)
            gt = nl.ndarray((P, D), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=xt, src=x[rows, :])
            nisa.dma_copy(dst=gt, src=dy[rows, :])
            sq = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
            ss = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=sq, data1=xt, data2=xt, op=nl.multiply)
            nisa.tensor_reduce(dst=ss, op=nl.add, data=sq, axis=1, keepdims=True)
            rstd = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=rstd, op=nl.rsqrt, data=ss, scale=1.0 / D, bias=_EPS_BF16)
            y = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=y, data=xt, op0=nl.multiply, operand0=rstd)
            # k = mean(dy * y)
            k = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=sq, data1=gt, data2=y, op=nl.multiply)
            nisa.tensor_reduce(dst=k, op=nl.add, data=sq, axis=1, keepdims=True)
            nisa.tensor_scalar(dst=k, data=k, op0=nl.multiply, operand0=1.0 / D)
            # dx = (dy - y * k) * rstd
            nisa.tensor_scalar(dst=y, data=y, op0=nl.multiply, operand0=k)
            h = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=h, data1=gt, data2=y, op=nl.subtract)
            ot = nl.ndarray((P, D), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=ot, data=h, op0=nl.multiply, operand0=rstd)
            nisa.dma_copy(dst=out[rows, :], src=ot)
        return out



def _rmsnorm_kernels(dtype: torch.dtype):
    """The kernel pair for this input dtype."""
    if dtype == torch.float32:
        return _rmsnorm_fwd_f32, _rmsnorm_bwd_f32
    return _rmsnorm_fwd_bf16, _rmsnorm_bwd_bf16


class _RMSNormNKI(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        D = x.shape[-1]
        flat = x.reshape(-1, D).contiguous()
        ctx.save_for_backward(flat)
        ctx.shape = x.shape
        fwd, _ = _rmsnorm_kernels(flat.dtype)
        return fwd(flat).reshape(x.shape)

    @staticmethod
    def backward(ctx, dy):
        (flat,) = ctx.saved_tensors
        _, bwd = _rmsnorm_kernels(flat.dtype)
        return bwd(flat, dy.reshape(flat.shape).to(flat.dtype).contiguous()).reshape(ctx.shape)


def block_norm(x: torch.Tensor) -> torch.Tensor:
    """Pre-norm inside a block; NKI-fused when enabled and the shape allows."""
    if (USE_NKI_RMSNORM and _HAS_NKI and x.device.type == "neuron"
            and x.dtype in (torch.float32, torch.bfloat16) and (x.numel() // x.shape[-1]) % _NKI_PARTITION == 0):
        return _RMSNormNKI.apply(x)
    return norm(x)

def rope_rows_tables(cos: torch.Tensor, sin: torch.Tensor, n_head: int):
    """[1,T,1,Dh] tables -> [T*H, Dh] row tables matching x.view(-1, D) row order."""
    return (cos[0, :, 0, :].repeat_interleave(n_head, dim=0).contiguous(),
            sin[0, :, 0, :].repeat_interleave(n_head, dim=0).contiguous())


def rope_norm_reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return norm(apply_rotary_emb(x, cos, sin))


def kernel_selftest(model: "GPT", device: torch.device) -> None:
    """Compare the NKI kernel with the PyTorch reference (forward and input grad)."""
    if not (_HAS_NKI and device.type == "neuron"):
        print0("kernel_selftest: skipped (no NKI / not on Neuron)")
        return
    cfg = model.config
    H, D, T = cfg.n_head, cfg.n_embd // cfg.n_head, cfg.sequence_len
    cos, sin = model.cos[:, :T], model.sin[:, :T]
    g = torch.Generator(device="cpu").manual_seed(1234)
    x = torch.randn(2, T, H, D, generator=g).to(device=device, dtype=COMPUTE_DTYPE)
    dy = torch.randn(2, T, H, D, generator=g).to(device=device, dtype=COMPUTE_DTYPE)
    xa = x.clone().requires_grad_(True)
    # fp32 reference with the same eps the bf16 training path uses (finfo(bf16).eps)
    ya = F.rms_norm(apply_rotary_emb(xa.float(), cos.float(), sin.float()), (D,), eps=_QK_EPS)
    (ga,) = torch.autograd.grad(ya, xa, dy.float())
    xb = x.clone().requires_grad_(True)
    yb = _QKRopeNormNKI.apply(xb, model.cos_rows, model.sin_rows)
    (gb,) = torch.autograd.grad(yb, xb, dy)
    fe = (ya - yb.float()).abs().max().item()
    be = (ga.float() - gb.float()).abs().max().item()
    print0(f"kernel_selftest: fwd max|err| {fe:.4g} (|y| max {ya.abs().max().item():.3g}), "
           f"bwd max|err| {be:.4g} (|g| max {ga.abs().max().item():.3g})")
    if not (fe < 0.05 and be < 0.1 * max(1.0, ga.abs().max().item())):
        raise RuntimeError("kernel_selftest FAILED")
    for rdt in ((torch.float32, COMPUTE_DTYPE) if USE_NKI_RMSNORM else ()):
        C = cfg.n_embd
        xr = torch.randn(2, T, C, generator=g).to(device=device, dtype=rdt)
        dyr = torch.randn(2, T, C, generator=g).to(device=device, dtype=rdt)
        x1 = xr.clone().float().requires_grad_(True)
        y1 = F.rms_norm(x1, (C,), eps=float(torch.finfo(rdt).eps))
        (g1,) = torch.autograd.grad(y1, x1, dyr.float())
        x2 = xr.clone().requires_grad_(True)
        y2 = _RMSNormNKI.apply(x2)
        (g2,) = torch.autograd.grad(y2, x2, dyr)
        fe3 = (y1 - y2.float()).abs().max().item()
        be3 = (g1 - g2.float()).abs().max().item()
        print0(f"kernel_selftest[rmsnorm {rdt}]: fwd max|err| {fe3:.4g} (|y| max {y1.abs().max().item():.3g}), "
               f"bwd max|err| {be3:.4g} (|g| max {g1.abs().max().item():.3g})")
        if not (fe3 < 0.05 and be3 < 0.1 * max(1.0, g1.abs().max().item())):
            raise RuntimeError("kernel_selftest[rmsnorm] FAILED")


def rope_norm(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rows) -> torch.Tensor:
    """norm(rope(x)) for x [B, T, H, D]; NKI-fused when enabled and shapes allow."""
    if (USE_NKI_QKNORM and _HAS_NKI and x.device.type == "neuron" and rows is not None
            and (x.shape[1] * x.shape[2]) % _NKI_PARTITION == 0):
        return _QKRopeNormNKI.apply(x, rows[0], rows[1])
    return rope_norm_reference(x, cos, sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        assert self.head_dim * self.n_head == self.n_embd
        self.c_q = Linear(config.n_embd, config.n_embd, bias=False)
        self.c_k = Linear(config.n_embd, config.n_embd, bias=False)
        self.c_v = Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rows=None, v0=None, v_mix=None):
        b, t, c = x.shape
        if QKV_CAT:
            # One matmul for q, k, v (same parameters, same math; fewer launches).
            # This path bypasses Linear.forward, so apply its bf16 input cast here too
            # (without it the fused matmul silently ran in fp32 on the fp32 residual).
            xin = x.to(COMPUTE_DTYPE) if BF16_MATMUL else x
            w = torch.cat((self.c_q.weight, self.c_k.weight, self.c_v.weight), dim=0).to(dtype=xin.dtype)
            q, k, v = F.linear(xin, w, None).split(c, dim=-1)
            q = q.reshape(b, t, self.n_head, self.head_dim)
            k = k.reshape(b, t, self.n_head, self.head_dim)
            v = v.reshape(b, t, self.n_head, self.head_dim)
        else:
            q = self.c_q(x).view(b, t, self.n_head, self.head_dim)
            k = self.c_k(x).view(b, t, self.n_head, self.head_dim)
            v = self.c_v(x).view(b, t, self.n_head, self.head_dim)
        v_raw = v
        if v0 is not None:
            v = v_mix[0] * v + v_mix[1] * v0
        q = rope_norm(q, cos, sin, rows)
        k = rope_norm(k, cos, sin, rows)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.c_proj(y), v_raw


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(relu_squared(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.resid_lambda = nn.Parameter(torch.ones(1))
        # x0 skip: mixes initial embedding into each layer's input (stabilizes deep nets)
        self.x0_lambda = nn.Parameter(torch.zeros(1))
        if config.value_resid:
            self.v_lambda = nn.Parameter(torch.tensor([0.5, 0.5]))

    def forward(self, x, x0, cos, sin, rows=None, v0=None):
        x = self.resid_lambda * x + self.x0_lambda * x0
        mix = self.v_lambda if v0 is not None else None
        a, v = self.attn(block_norm(x), cos, sin, rows, v0, mix)
        x = x + a
        x = x + self.mlp(block_norm(x))
        return x, v


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        padded_vocab_size = ((config.vocab_size + 63) // 64) * 64
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        if config.unet:
            self.skip_weights = nn.Parameter(torch.ones(config.n_layer // 2))
        cos, sin = self._precompute_rotary_embeddings(config.sequence_len, config.n_embd // config.n_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        cr, sr = rope_rows_tables(cos, sin, config.n_head)
        self.register_buffer("cos_rows", cr, persistent=False)
        self.register_buffer("sin_rows", sr, persistent=False)

    @torch.no_grad()
    def init_weights(self) -> None:
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = math.sqrt(3.0) * self.config.n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -0.4 * s, 0.4 * s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            block.resid_lambda.fill_(1.0)
            block.x0_lambda.fill_(0.05)
            if self.config.value_resid:
                block.v_lambda.copy_(torch.tensor([0.5, 0.5]))
        if self.config.unet:
            self.skip_weights.fill_(1.0)
        cos, sin = self._precompute_rotary_embeddings(self.config.sequence_len, self.config.n_embd // self.config.n_head)
        self.cos.copy_(cos)
        self.sin.copy_(sin)
        cr, sr = rope_rows_tables(cos, sin, self.config.n_head)
        self.cos_rows.copy_(cr)
        self.sin_rows.copy_(sr)

    def _precompute_rotary_embeddings(self, seq_len: int, head_dim: int, base: int = 100000):
        device = self.transformer.wte.weight.device
        freqs = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (freqs / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        table = torch.outer(t, inv_freq)
        cos = table.cos().to(COMPUTE_DTYPE)[None, :, None, :]
        sin = table.sin().to(COMPUTE_DTYPE)[None, :, None, :]
        return cos, sin

    def get_device(self) -> torch.device:
        return self.transformer.wte.weight.device

    def estimate_flops(self) -> int:
        """Estimate FLOPs per token (forward + backward). Attention QK^T and AV are O(T) per token."""
        nparams = sum(p.numel() for p in self.parameters())
        # Parameters that are not matmul weights: the embedding table and the learned
        # scalars (resid/x0 lambdas, and v_lambda / skip_weights when enabled).
        non_matmul = self.transformer.wte.weight.numel() + sum(
            p.numel() for p in self.parameters() if p.dim() < 2)
        h = self.config.n_head
        d = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        # QK^T: 2*h*d*T per token, AV: 2*h*d*T per token, x3 for fwd+bwd.
        # PaLM convention: the causal mask is not discounted, so this counts the
        # full T x T score matrix (about 2x the attention work actually needed).
        attn_flops = self.config.n_layer * 12 * h * d * t
        return 6 * (nparams - non_matmul) + attn_flops

    def setup_optimizer(self):
        scalar_params = []
        matrix_params = []
        for block in self.transformer.h:
            scalar_params.extend([block.resid_lambda, block.x0_lambda])
            if self.config.value_resid:
                scalar_params.append(block.v_lambda)
            matrix_params.extend(block.attn.parameters())
            matrix_params.extend(block.mlp.parameters())
        if self.config.unet:
            scalar_params.append(self.skip_weights)
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        dmodel_lr_scale = (self.config.n_embd / 768) ** -0.5
        param_groups = [
            dict(kind="adamw", params=embedding_params, lr=EMBEDDING_LR * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind="adamw", params=lm_head_params, lr=UNEMBEDDING_LR * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind="adamw", params=scalar_params, lr=SCALAR_LR, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(kind="muon", params=params, lr=MATRIX_LR, momentum=0.95, beta2=BETA2_MUON, ns_steps=5, weight_decay=WEIGHT_DECAY))
        opt = MuonAdamW(param_groups)
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
            opt.set_lr(group, group["lr"])
            if group["kind"] == "muon":
                opt.set_momentum(group, group["momentum"])
        return opt

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, loss_reduction: str = "mean"):
        _b, t = idx.shape
        assert t <= self.config.sequence_len
        cos, sin = self.cos[:, :t], self.sin[:, :t]
        x = self.transformer.wte(idx).to(COMPUTE_DTYPE)
        x = norm(x)
        x0 = x
        rows = None
        if USE_NKI_QKNORM and t == self.config.sequence_len:
            rows = (self.cos_rows, self.sin_rows)
        n = self.config.n_layer
        skips = []
        v0 = None
        for i, block in enumerate(self.transformer.h):
            if self.config.unet and i >= n - n // 2:
                x = x + self.skip_weights[n - 1 - i] * skips.pop()
            x, v = block(x, x0, cos, sin, rows, v0 if self.config.value_resid else None)
            if i == 0:
                v0 = v
            if self.config.unet and i < n // 2:
                skips.append(x)
        x = norm(x)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        cap = self.config.logit_softcap
        logits = cap * torch.tanh(logits / cap)
        if targets is None:
            return logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-1,
            reduction=loss_reduction,
        )
        if loss_reduction == "none":
            return loss.view_as(targets)
        return loss


def build_config(depth: int = DEPTH) -> GPTConfig:
    base_dim = depth * ASPECT_RATIO
    n_embd = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    n_head = n_embd // HEAD_DIM
    return GPTConfig(
        sequence_len=SEQ_LEN,
        vocab_size=TOKENIZER_VOCAB_SIZE,
        n_layer=depth,
        n_head=n_head,
        n_embd=n_embd,
        value_resid=_env("VALUE_RESID", 0) != 0,
        unet=_env("UNET", 0) != 0,
        logit_softcap=float(LOGIT_SOFTCAP),
        bf16_matmul=BF16_MATMUL,
        qkv_cat=QKV_CAT,
        nki_qknorm=USE_NKI_QKNORM,
        nki_rmsnorm=USE_NKI_RMSNORM,
    )


def apply_forward_flags(config: GPTConfig) -> None:
    """Set the module-level forward switches from a (checkpoint) config."""
    global BF16_MATMUL, QKV_CAT, USE_NKI_QKNORM, USE_NKI_RMSNORM
    BF16_MATMUL = bool(config.bf16_matmul)
    QKV_CAT = bool(config.qkv_cat)
    USE_NKI_QKNORM = bool(config.nki_qknorm)
    USE_NKI_RMSNORM = bool(config.nki_rmsnorm)


# -----------------------------------------------------------------------------
# Optimizer

POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


BETA2_MUON = 0.90  # NorMuon second-moment EMA; the default for each Muon group's beta2


def muon_core(g, w, momentum_buffer, second, lr, momentum, wd: float, ns_steps: int, beta2: float = BETA2_MUON):
    """One Muon update on a stack of same-shape matrices, in place on w and the
    optimizer buffers. lr and momentum are 0-d device tensors so the compiled
    graph never specializes on their values. beta2 is a per-run constant (the
    group's beta2), so specializing on it does not cause recompiles."""
    shape = g.shape
    momentum_buffer.lerp_(g, 1 - momentum)
    g = g.lerp_(momentum_buffer, momentum)
    x = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g
    x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if x.shape[-2] > x.shape[-1]:
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            xtx = x.mT @ x
            x = a * x + x @ (b * xtx + c * (xtx @ xtx))
    else:
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            xxt = x @ x.mT
            x = a * x + (b * xxt + c * (xxt @ xxt)) @ x
    g = x
    red_dim = -1 if shape[-2] >= shape[-1] else -2
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size).sqrt()
    second.lerp_(v_mean, 1 - beta2)
    step_size = second.clamp_min(1e-10).rsqrt()
    scaled = (v_mean * red_dim_size) * step_size.square()
    v_norm_new = scaled.sum(dim=(-2, -1), keepdim=True).sqrt()
    g = g * (step_size * (v_norm / v_norm_new.clamp_min(1e-10))).to(g.dtype)
    mask = (g * w) >= 0
    w.sub_(lr * g + lr * wd * w * mask)


def adamw_core(params, grads, exp_avgs, exp_avg_sqs, lr, bias1, bias2, wd: float, beta1: float, beta2: float, eps: float):
    """AdamW on a list of tensors, in place. lr / bias1 / bias2 are 0-d device
    tensors so a compiled graph does not specialize on their per-step values."""
    for p, g, m, v in zip(params, grads, exp_avgs, exp_avg_sqs):
        p.mul_(1 - lr * wd)
        m.lerp_(g, 1 - beta1)
        v.lerp_(g.square(), 1 - beta2)
        update = m / bias1 / ((v / bias2).sqrt() + eps)
        p.sub_(update * lr)


# Set in main(): torch.compile(muon_core) when NEURON_HP_COMPILE_OPT=1, else eager.
_MUON_CORE = muon_core
_ADAMW_CORE = adamw_core
COMPILE_ADAMW = os.environ.get("NEURON_HP_COMPILE_ADAMW", "1") != "0"
COMPILE_OPT = os.environ.get("NEURON_HP_COMPILE_OPT", "1") != "0"
BUCKET_ALLREDUCE = os.environ.get("NEURON_HP_BUCKET_AR", "0") != "0"


class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})

    @torch.no_grad()
    def _adamw_step_fused(self, group):
        """Group-level AdamW through adamw_core (compiled when COMPILE_ADAMW)."""
        beta1, beta2 = group["betas"]
        params = [p for p in group["params"] if p.grad is not None]
        if not params:
            return
        for p in params:
            st = self.state[p]
            if not st:
                st["exp_avg"] = torch.zeros_like(p)
                st["exp_avg_sq"] = torch.zeros_like(p)
        group["step_count"] = group.get("step_count", 0) + 1
        t = group["step_count"]
        host = torch.tensor([1 - beta1 ** t, 1 - beta2 ** t], dtype=torch.float32)
        if "bias_t" not in group:
            group["bias_t"] = host.to(params[0].device)
        else:
            group["bias_t"].copy_(host)
        _ADAMW_CORE(params, [p.grad for p in params],
                    [self.state[p]["exp_avg"] for p in params],
                    [self.state[p]["exp_avg_sq"] for p in params],
                    group["lr_t"], group["bias_t"][0], group["bias_t"][1],
                    group["weight_decay"], beta1, beta2, group["eps"])

    @torch.no_grad()
    def _adamw_step(self, group):
        if COMPILE_ADAMW:
            return self._adamw_step_fused(group)
        beta1, beta2 = group["betas"]
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            lr = group["lr_t"]  # 0-d device tensor, see set_lr()
            p.mul_(1 - lr * group["weight_decay"])
            exp_avg.lerp_(grad, 1 - beta1)
            exp_avg_sq.lerp_(grad.square(), 1 - beta2)
            bias1 = 1 - beta1 ** state["step"]
            bias2 = 1 - beta2 ** state["step"]
            update = exp_avg / bias1 / ((exp_avg_sq / bias2).sqrt() + group["eps"])
            p.sub_(update * lr)

    @torch.no_grad()
    def _muon_step(self, group):
        params = group["params"]
        if not params:
            return
        p0 = params[0]
        state = self.state[p0]
        shape = p0.shape
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(len(params), *shape, dtype=p0.dtype, device=p0.device)
        if "second_momentum_buffer" not in state:
            state_shape = (len(params), shape[-2], 1) if shape[-2] >= shape[-1] else (len(params), 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=torch.float32, device=p0.device)
        if "stacked_grads" not in state:
            state["stacked_grads"] = torch.empty(len(params), *shape, dtype=p0.dtype, device=p0.device)
        if "stacked_params" not in state:
            state["stacked_params"] = torch.empty(len(params), *shape, dtype=p0.dtype, device=p0.device)

        g = state["stacked_grads"]
        w = state["stacked_params"]
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
        torch._foreach_copy_(list(g.unbind(0)), grads)
        torch._foreach_copy_(list(w.unbind(0)), list(params))
        lr = group["lr_t"] * (max(1.0, shape[-2] / shape[-1]) ** 0.5)
        _MUON_CORE(g, w, state["momentum_buffer"], state["second_momentum_buffer"], lr,
                   group["mom_t"], group["weight_decay"], group["ns_steps"], group["beta2"])
        torch._foreach_copy_(params, list(w.unbind(0)))

    @torch.no_grad()
    def set_lr(self, group, lr: float) -> None:
        """Store the LR as a 0-d device tensor filled by a host->device copy.

        Neuron eager mode specializes compiled kernels on Python scalar operands,
        so a Python-float LR that changes every step (warmup / warmdown) forces a
        recompile each step (~1.5 s -> ~8 s per step). Passing the LR as tensor
        data keeps the kernels fixed; only the value moves."""
        group["lr"] = lr
        p0 = group["params"][0]
        host = torch.tensor(lr, dtype=torch.float32)
        if "lr_t" not in group:
            group["lr_t"] = host.to(p0.device)
        else:
            group["lr_t"].copy_(host)

    @torch.no_grad()
    def set_momentum(self, group, momentum: float) -> None:
        """Muon momentum as a 0-d device tensor (same reason as set_lr)."""
        group["momentum"] = momentum
        p0 = group["params"][0]
        host = torch.tensor(momentum, dtype=torch.float32)
        if "mom_t" not in group:
            group["mom_t"] = host.to(p0.device)
        else:
            group["mom_t"].copy_(host)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group["kind"] == "adamw":
                self._adamw_step(group)
            elif group["kind"] == "muon":
                self._muon_step(group)
            else:
                raise ValueError(f"unknown optimizer kind: {group['kind']}")


def sync_gradients(model: nn.Module) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if BUCKET_ALLREDUCE:
        # One collective over a flat buffer instead of one per parameter.
        flat = torch.cat([g.reshape(-1) for g in grads])
        dist.all_reduce(flat, op=dist.ReduceOp.AVG)
        torch._foreach_copy_(grads, [t.view_as(g) for t, g in zip(flat.split([g.numel() for g in grads]), grads)])
        return
    for g in grads:
        dist.all_reduce(g, op=dist.ReduceOp.AVG)


# -----------------------------------------------------------------------------
# Checkpoint / scoring hook

def save_checkpoint(path: Path, model: GPT, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": asdict(model.config),
        "meta": meta,
    }
    torch.save(payload, path)


def torch_load_weights(path: str | Path, map_location):
    return torch.load(path, map_location=map_location, weights_only=True)


def load_for_eval(checkpoint_path: str, device: torch.device) -> nn.Module:
    global COMPUTE_DTYPE
    COMPUTE_DTYPE = compute_dtype_for(device)
    payload = torch_load_weights(checkpoint_path, map_location="cpu")
    config = GPTConfig(**payload["model_config"])
    apply_forward_flags(config)
    model = GPT(config)
    model.init_weights()
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Training

PROFILE = os.environ.get("NEURON_HP_PROFILE", "0") != "0"


class PhaseTimer:
    """Per-phase wall-clock timing with device syncs (profiling only: the syncs
    themselves slow training, so never enable this for a scored run)."""

    def __init__(self, device: torch.device, enabled: bool):
        self.device, self.enabled = device, enabled
        self.totals: dict[str, float] = {}
        self.t = 0.0

    def start(self) -> None:
        if self.enabled:
            synchronize(self.device)
            self.t = time.monotonic()

    def lap(self, name: str) -> None:
        if self.enabled:
            synchronize(self.device)
            now = time.monotonic()
            self.totals[name] = self.totals.get(name, 0.0) + now - self.t
            self.t = now

    def report(self, steps: int) -> str:
        total = sum(self.totals.values()) or 1.0
        return " | ".join(f"{k} {v / max(steps, 1):.3f}s ({100 * v / total:.0f}%)" for k, v in self.totals.items())


def lr_multiplier(step: int, progress: float) -> float:
    """Warmup by steps, then constant, then linear warmdown over the last
    WARMDOWN_FRAC of the time budget. `progress` is the fraction of the
    charged training time already used (0 before the budget clock starts)."""
    if step < WARMUP_STEPS:
        return (step + 1) / max(1, WARMUP_STEPS)
    if WARMDOWN_FRAC <= 0 or progress < 1.0 - WARMDOWN_FRAC:
        return 1.0
    frac = min(1.0, max(0.0, (1.0 - progress) / WARMDOWN_FRAC))
    if WARMDOWN_SHAPE == "sqrt":
        # "1-sqrt" cooldown: 1 - sqrt(elapsed fraction of the warmdown)
        frac = 1.0 - math.sqrt(1.0 - frac)
    return FINAL_LR_FRAC + (1.0 - FINAL_LR_FRAC) * frac


def muon_momentum(step: int) -> float:
    frac = min(step / 300, 1.0)
    return (1 - frac) * 0.85 + frac * 0.95


def main() -> None:
    parser = argparse.ArgumentParser(description="Neuron Competition R1 baseline")
    parser.add_argument("--device-type", type=str, default="")
    parser.add_argument("--num-steps", type=int, default=int(os.environ.get("NEURON_COMPETITION_R1_NUM_STEPS", NUM_STEPS)))
    parser.add_argument("--max-train-seconds", type=int, default=int(os.environ.get("NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS", MAX_TRAIN_SECONDS)))
    parser.add_argument("--out-dir", type=str, default=os.environ.get("NEURON_COMPETITION_R1_OUT_DIR", "out"))
    parser.add_argument("--eval-public", action=argparse.BooleanOptionalAction, default=os.environ.get("NEURON_COMPETITION_R1_EVAL_PUBLIC", "1") != "0")
    parser.add_argument("--eval-tokens", type=int, default=int(os.environ.get("NEURON_COMPETITION_R1_EVAL_TOKENS", str(4 * 524288))))
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=os.environ.get("NEURON_COMPETITION_R1_COMPILE", "1") != "0",
                        help="enable torch.compile on the full model")
    parser.add_argument("--compile-only", action="store_true",
                        help="run one synthetic optimizer step to warm the compile cache, then exit")
    parser.add_argument("--nki-relu2", action="store_true", default=os.environ.get("NEURON_COMPETITION_R1_NKI_RELU2", "0") != "0",
                        help="use the example fused NKI relu(x)**2 kernel in the MLP (Neuron only; illustration)")
    args = parser.parse_args()

    global USE_NKI_RELU2
    USE_NKI_RELU2 = args.nki_relu2

    device_type = detect_device_type(args.device_type)
    ddp, rank, _local_rank, world_size, device = init_runtime(device_type)
    global COMPUTE_DTYPE
    COMPUTE_DTYPE = compute_dtype_for(device)

    print0(f"device={device} dtype={COMPUTE_DTYPE} world_size={world_size}")
    print0(f"training target={args.num_steps} steps max_seconds={args.max_train_seconds} official_budget={TRAIN_TIME_BUDGET_SECONDS}s")

    tokenizer = ensure_tokenizer(build_if_missing=False)
    config = build_config()
    print0(f"model_config={json.dumps(asdict(config), indent=2)}")
    model = GPT(config)
    model.init_weights()
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    flops_per_token = model.estimate_flops()
    print0(f"num_params={num_params:,} flops_per_token={flops_per_token:e}")

    global _MUON_CORE, _ADAMW_CORE
    if COMPILE_ADAMW:
        _ADAMW_CORE = torch.compile(adamw_core, backend="neuron" if device_type == "neuron" else "inductor", dynamic=False)
        print0("adamw_core compiled")
    if COMPILE_OPT:
        _MUON_CORE = torch.compile(muon_core, backend="neuron" if device_type == "neuron" else "inductor", dynamic=False)
        print0("muon_core compiled")
    print0(f"bucket_allreduce={BUCKET_ALLREDUCE} compile_opt={COMPILE_OPT}")

    if KERNEL_SELFTEST:
        kernel_selftest(model, device)

    orig_model = model
    if args.compile:
        compile_backend = "neuron" if device_type == "neuron" else "inductor"
        model = torch.compile(model, backend=compile_backend, dynamic=False)
        print0(f"torch.compile enabled (backend={compile_backend})")

    world_tokens_per_micro = DEVICE_BATCH_SIZE * SEQ_LEN * world_size
    if TOTAL_BATCH_SIZE % world_tokens_per_micro != 0:
        raise ValueError(f"TOTAL_BATCH_SIZE={TOTAL_BATCH_SIZE} must divide micro tokens={world_tokens_per_micro}")
    grad_accum_steps = TOTAL_BATCH_SIZE // world_tokens_per_micro
    print0(f"micro_tokens={world_tokens_per_micro:,} grad_accum_steps={grad_accum_steps}")

    if args.compile_only:
        synthetic_x = torch.zeros((DEVICE_BATCH_SIZE, SEQ_LEN), dtype=torch.int32).to(device)
        synthetic_y = torch.zeros((DEVICE_BATCH_SIZE, SEQ_LEN), dtype=torch.long).to(device)
        optimizer = model.setup_optimizer()
        model.zero_grad(set_to_none=True)
        for _ in range(grad_accum_steps):
            loss = model(synthetic_x, synthetic_y)
            (loss / grad_accum_steps).backward()
        # Average across ranks first, then clip on the global gradient.
        sync_gradients(model)
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        synchronize(device)
        print0("compile-only synthetic training step completed")
        cleanup_runtime()
        return

    train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, SEQ_LEN, "train", device)
    x, y, loader_state = next(train_loader)
    optimizer = model.setup_optimizer()
    loop_started = time.monotonic()
    budget_started = loop_started

    smooth_loss = 0.0
    total_train_tokens = 0
    step = 0
    recent_step_times: list[float] = []
    startup_allowance_est = 0.0

    timer = PhaseTimer(device, PROFILE)
    profiled_steps = 0
    while step < args.num_steps:
        synchronize(device)
        t0 = time.monotonic()
        # Only time steady-state steps (after compilation).
        timer.enabled = PROFILE and step >= 10
        timer.start()
        optimizer.zero_grad(set_to_none=True)
        last_loss = None
        for _ in range(grad_accum_steps):
            loss = model(x, y)
            last_loss = loss.detach()
            timer.lap("forward")
            (loss / grad_accum_steps).backward()
            timer.lap("backward")
            x, y, loader_state = next(train_loader)
            timer.lap("data")

        # Average across ranks first, then clip on the global gradient (clipping
        # each rank's local gradient by its own norm is not the intended update).
        sync_gradients(model)
        timer.lap("grad_allreduce")
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        # Progress through the charged budget. Before the clock restarts at the
        # 5th step boundary this is ~0, so warmdown cannot trigger during compile.
        if step >= TRAIN_STARTUP_STEPS_EXCLUDED:
            progress = (time.monotonic() - budget_started) / args.max_train_seconds
        else:
            progress = 0.0
        lrm = lr_multiplier(step, progress)
        for group in optimizer.param_groups:
            optimizer.set_lr(group, group["initial_lr"] * lrm)
            if group["kind"] == "muon":
                optimizer.set_momentum(group, muon_momentum(step))
                group["weight_decay"] = WEIGHT_DECAY
        optimizer.step()
        timer.lap("optimizer")

        synchronize(device)
        record_step_boundary(step)
        dt = time.monotonic() - t0
        # Only steady-state steps feed the step-time estimate used by the predictive
        # stop: the excluded startup steps include compilation, and a long first-step
        # compile could otherwise end a short run right after the clock restarts.
        if step >= TRAIN_STARTUP_STEPS_EXCLUDED:
            recent_step_times.append(dt)
        train_loss = float(last_loss.item())
        if step + 1 == TRAIN_STARTUP_STEPS_EXCLUDED:
            # The runner excludes launch through this boundary (capped), so
            # restart the budget clock here to use the full charged budget.
            # If startup exceeded the cap, the excess is already charged:
            # back-date the clock by that excess, with a margin for the
            # launch -> import gap this process cannot observe.
            now = time.monotonic()
            observed_startup = now - _PROCESS_STARTED
            startup_allowance_est = min(observed_startup, TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS)
            excess = max(
                0.0,
                observed_startup + STARTUP_LAUNCH_MARGIN_SECONDS - TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS,
            )
            budget_started = now - excess
        elapsed_after_step = time.monotonic() - budget_started
        # Fold the deadline decision into the existing health collective. A
        # local wall-clock check can otherwise make one rank begin another
        # optimizer step while its peers head for the checkpoint barrier.
        # The budget clock is only meaningful once it has been reset at the
        # excluded-startup boundary; before that, elapsed_after_step is measured
        # from loop start and a slow first-step compile could trip the predictive
        # stop and break before five steps complete -- leaving the run unscorable.
        # Gate the time-based stop behind reaching the excluded-step count so the
        # run is always scorable; an explicit num_steps budget still stops early.
        reached_min_steps = step + 1 >= TRAIN_STARTUP_STEPS_EXCLUDED
        stop_after_step = (
            step + 1 >= args.num_steps
            or (
                reached_min_steps
                and (
                    elapsed_after_step >= args.max_train_seconds
                    or elapsed_after_step
                    + max(recent_step_times[-20:], default=dt)
                    + CHECKPOINT_RESERVE_SECONDS
                    >= args.max_train_seconds
                )
            )
        )
        control = torch.tensor(
            [
                1 if (math.isnan(train_loss) or train_loss > 100) else 0,
                1 if stop_after_step else 0,
            ],
            device=device,
            dtype=torch.int32,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(control, op=dist.ReduceOp.MAX)
        if int(control[0].item()) != 0:
            raise RuntimeError(f"bad loss detected: {train_loss}")
        timer.lap("control")
        if timer.enabled:
            profiled_steps += 1
            if rank == 0 and profiled_steps % 20 == 0:
                print(f"profile[{profiled_steps} steps] {timer.report(profiled_steps)}", flush=True)

        total_train_tokens += TOTAL_BATCH_SIZE
        smooth_loss = 0.9 * smooth_loss + 0.1 * train_loss
        debiased = smooth_loss / (1 - 0.9 ** (step + 1))
        tok_per_sec = TOTAL_BATCH_SIZE / max(dt, 1e-6)
        if rank == 0 and (step < 5 or step % 10 == 0):
            remaining_steps = args.num_steps - step - 1
            remaining_time = max(0.0, args.max_train_seconds - (time.monotonic() - budget_started))
            print(
                f"step {step:05d} | loss {debiased:.4f} | lrm {lrm:.3f} | "
                f"dt {dt:.2f}s | tok/s {tok_per_sec:,.0f} | remaining_steps {remaining_steps} | "
                f"remaining_time {remaining_time:.0f}s | "
                f"epoch {loader_state['epoch']}",
                flush=True,
            )

        if step == 0:
            # After first step all persistent objects are allocated; freeze to avoid GC pauses
            gc.collect()
            gc.freeze()
            gc.disable()
        step += 1
        if int(control[1].item()) != 0:
            break

    checkpoint_path = Path(args.out_dir) / "final.pt"
    budget_elapsed_before_save = time.monotonic() - budget_started
    if rank == 0:
        save_checkpoint(checkpoint_path, orig_model, {
            "step": step,
            "total_train_tokens": total_train_tokens,
            "budget_elapsed_before_save": budget_elapsed_before_save,
            "world_size": world_size,
            "compiled": args.compile,
        })
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    budget_elapsed_after_save = time.monotonic() - budget_started

    public_eval = None
    causal = None
    if args.eval_public:
        causal = causality_check(orig_model, tokenizer, DEVICE_BATCH_SIZE, SEQ_LEN, device)
        public_eval = evaluate_bpb(
            orig_model,
            tokenizer,
            DEVICE_BATCH_SIZE,
            SEQ_LEN,
            device,
            split="public_val",
            max_tokens=args.eval_tokens,
            timeout_seconds=EVAL_TIMEOUT_SECONDS,
        )

    if rank == 0:
        print("---")
        print(f"checkpoint_path:    {checkpoint_path}")
        # The budget clock restarts at the fifth completed step to mirror the
        # runner's charged window, so post-restart elapsed time IS the charged
        # estimate (conservative: the runner also excludes pre-loop startup).
        print(f"training_seconds:  {budget_elapsed_before_save:.1f}")
        print(f"budget_seconds:    {TRAIN_TIME_BUDGET_SECONDS}")
        print(f"target_steps:      {args.num_steps}")
        print(f"max_train_seconds: {args.max_train_seconds}")
        print(f"startup_allowance_est: {startup_allowance_est:.1f}")
        print(f"charged_seconds_est:   {budget_elapsed_after_save:.1f}")
        print(f"over_budget:       {budget_elapsed_after_save > TRAIN_TIME_BUDGET_SECONDS}")
        print(f"total_seconds:     {time.monotonic() - loop_started:.1f}")
        print(f"compiled:          {args.compile}")
        print(f"nki_relu2:         {args.nki_relu2}")
        print(f"num_steps:         {step}")
        print(f"total_tokens_M:    {total_train_tokens / 1e6:.1f}")
        print(f"world_size:        {world_size}")
        print(f"lnc:               {os.environ.get('NEURON_LOGICAL_NC_CONFIG', '')}")
        print(f"seq_len:           {SEQ_LEN}")
        print(f"num_params_M:      {num_params / 1e6:.1f}")
        if causal is not None:
            print(f"causality_passed:  {causal['passed']}")
            print(f"causality_max_abs: {causal['max_abs_diff']:.6f}")
        if public_eval is not None:
            print(f"public_val_bpb:    {public_eval['val_bpb']:.6f}")
            print(f"eval_seconds:      {public_eval['eval_seconds']:.1f}")
            print(f"eval_timeout:      {public_eval['eval_timeout']}")
            print(f"eval_over_timeout: {public_eval['eval_over_timeout']}")

    cleanup_runtime()


if __name__ == "__main__":
    main()
