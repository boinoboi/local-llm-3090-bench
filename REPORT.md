# Local LLM benchmark — RTX 3090 + RTX 5060 Ti

Measured, not estimated. Single stream, one request at a time, one card.

## Hardware

| | |
|---|---|
| GPU0 | RTX 3090 24 GB, sm_86, 936 GB/s, 82 SMs |
| GPU1 | RTX 5060 Ti 16 GB, sm_120 (unused for inference) |
| CPU | i9-14900K, 24c/32t, PL1 95 W / PL2 125 W |
| RAM | 125 GiB |
| OS | CachyOS, kernel 7.2.9, CUDA 13.4.92 |
| Display | 2× 1920×1080 (Hyprland/Wayland) |

## Method

- Cold prefill uses a unique random prefix so the prefix cache cannot contribute.
- Warm prefill repeats the identical prompt.
- Prompt length is calibrated against the server's own tokenizer, not estimated.
- llama.cpp numbers come from a server build with `-fa on -ctk q4_0 -ctv q4_0`.
- megakernel numbers come from the engine's own reported decode rate, which
  excludes load and prefill.
- Power sweeps use one fresh engine process per measurement (10 reps per level).

## Headline: engine comparison, Qwen3.8-27B

| | decode 8k | decode 32k | decode 98k |
|---|---|---|---|
| llama.cpp + MTP, IQ2_XS | 62 | 62 | 36 (at 131k) |
| **megakernel + MTP 7, UD-Q4_K_M** | **152** | **134** | **96** |

2.4x faster *and* 4x more bits per weight.

The megakernel is a single CUDA kernel per speculative cycle: the model's own MTP
head drafts 7 tokens, and one int8 tensor-core pass verifies all 8 rows.

## Power sweep — 8k context, 10 runs per level

| limit | decode t/s | peak W | tok/s per kW | energy/token | gain |
|---|---|---|---|---|---|
| 230 | 109.39 | 232 | 470.9 | 2.12 J | — |
| 250 | 126.77 | 251 | 505.6 | 1.98 J | +15.9% |
| **270** | 141.16 | 271 | **521.8** | **1.92 J** | +11.4% |
| 290 | *invalid* | 230 | — | — | limit failed to apply |
| 310 | 148.70 | 310 | 480.1 | 2.08 J | +37.5% |
| **330** | 150.31 | 330 | 455.6 | 2.20 J | +1.1% |
| 350 | 151.36 | 348 | 435.2 | 2.30 J | +0.7% |
| 370 | 152.31 | 370 | 411.9 | 2.43 J | +0.6% |
| 385 | 152.31 | 385 | 396.1 | 2.53 J | +0.0% |

- **Efficiency optimum: 270 W** (best tok/s per watt, lowest energy per token).
- **Speed knee: 330 W** — 98.7% of maximum speed at 86% of maximum power.
- **385 W buys 1.3% more speed for 17% more power.** Not worth it.

Draft acceptance stays flat at 49.9% across every level, so the power curve is
pure clock scaling, not a speculation effect.

## Context scaling — megakernel

| context | decode t/s | KV |
|---|---|---|
| 1k | 133.6 | q8_0 |
| 8k | 152.3 | q8_0 |
| 32k | 134.4 | q8_0 |
| 64k | 89.1 | q8_0 |
| 131k | 57.6 | q8_0 |
| 131k | 87.5 | preset F |

## Full 262k on 24 GB

KV presets (`MK_KV` + `MK_KV_WIN=2048`) store old positions at 5-6 bits and keep
the newest 2048 positions at q8:

| preset | K/V bits | max context | peak VRAM |
|---|---|---|---|
| q8_0 | 8 / 8 | ~120k | 22,383 MiB |
| **F** | 5r / 5 | **262,144** | **23,432 MiB measured** |

Measured cold read of a 229,978-token prompt: **355 s = 647 tok/s** (at 230 W).

## Design facts found by reading configs and tensors

- Qwen3.6-35B-A3B: 40 layers = **30 linear_attention + 10 full_attention**, 2 KV heads,
  head_dim 256, 256 experts / 8 active, ctx 262144, MTP 1 (`nextn` tensor present).
- Qwen3.8-27B: 64 layers = 48 linear + 16 full, 4 KV heads, head_dim 256, MTP 1.
- MiMo-V2.6-9B: 32 layers, hybrid (`ssm_*` tensors), 4 KV heads, vocab 248320, ctx 262144, MTP 1.
- KV at 262k with hybrid attention is far smaller than uniform-attention maths predicts:
  35B-A3B needs 20 KB/token, not 256 KB.

## Cost (MSEDCL LT-I(B), Pune, FY 2026-27)

Marginal unit ₹20.04 (~$0.215) at 301-500 kWh/month, USD/INR 93.44.

| | per 1M output tokens |
|---|---|
| 270 W | ₹11.6 (**$0.124**) |
| 385 W | ₹13.2 (**$0.141**) |

Idle draw dominates a 24/7 machine: 3090 27 W + 5060 Ti 13 W + CPU 8 W, plus two
1080p panels at 20-30 W each.

## Reproduce

```bash
# llama.cpp
./run_llama.sh <label> <model.gguf> "draft-mtp" "" Q2_K 262144

# megakernel
make LLAMA=~/src/llama.cpp qwen38_mk libqwen38mk.so
MK_KV='K*=5r,V*=5' MK_KV_WIN=2048 MK_PF_CAP=1024 ./qwen38_mk -m model.gguf -c 262144 -d 7 --kv q8

# sweeps
python3 power_cli_sweep.py
python3 merge_csv.py
```

## Files

| file | contents |
|---|---|
| `results_all.csv` | every measurement, normalised |
| `results_llama.csv` | llama.cpp context sweeps |
| `results_mk.csv` | megakernel context sweeps |
| `results_power.csv` | power sweep, 10 reps per level |
| `power_cli_sweep.py` | power sweep driver |
| `measure.py` | engine-agnostic context sweep client |
| `merge_csv.py` | normaliser |
