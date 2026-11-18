# GPU inference benchmark — 3090 + 5060 Ti

## Goal
Measure real decode tok/s and prefill tok/s vs context fill (1k → 262k) for the three
local model families, pick the best engine per model, emit `results.csv`.

## Hardware (fixed)
- GPU0 RTX 3090 24GB, sm_86, 936 GB/s — power limit capped at **230 W** by user (default 350 W)
- GPU1 RTX 5060 Ti 16GB, sm_120 — limit 150 W (min), max 200 W
- i9-14900K, 125 GiB RAM, NVMe with ~670 GiB free

## Baseline (restore after tests)
- 3090 power.limit = 230 W (default 350, min 100, max 385)
- 5060 Ti power.limit = 150 W (default 180, min 150, max 200)
- stored: ~/bench/baseline_power.csv

## Artifacts
| path | what |
|---|---|
| ~/models/qwen3.6/ | huihui 35B-A3B Q2_K (MTP verified), llmfan46 Q3_K_M heretic (KL 0.0015, MTP), mmproj x2 |
| ~/models/qwen3.8/ | HauhauCS 27B Q2_K_P (MTP), NInfer 27B DFlash2 (14 GiB container) |
| ~/models/mimo2.6/ | heretic IQ2_M (MTP LOST — needs reconvert), mmproj |
| ~/src/ninfer | spk-c fork, sm_86, CUDA 13.4 |
| ~/src/llama.cpp | upstream, to build with GGML_CUDA |

## Key design facts (verified, not assumed)
- Qwen3.6-35B-A3B: 40 layers = 30 linear_attention + 10 full_attention; 2 KV heads, hd 256; 256 experts, 8 active; ctx 262144; MTP 1 (`nextn` tensor present in Q2_K).
- Qwen3.8-27B: 64 layers = 48 linear + 16 full; 4 KV heads, hd 256; ctx 262144; MTP 1.
- MiMo-V2.6-9B: 32 layers, hybrid (ssm_* tensors), 4 KV heads, hd 256, vocab 248320, ctx 262144, MTP 1.
- KV @262k: 35B-A3B 20 KB/tok → 1.35 GB @q4; 27B 64 KB/tok → 4.3 GB @q4; MiMo ~32 KB/tok → 2.2 GB @q4.

## Plan
1. [x] free VRAM (ollama stopped+disabled)
2. [ ] build ninfer sm_86
3. [ ] build llama.cpp CUDA (sm_86 + sm_120)
4. [ ] harness: context sweep 1k..262k, cold prefill / warm prefill / decode / TTFT / VRAM
5. [ ] runs: 35B Q2_K, 35B Q3_K_M, 27B Q2_K_P, 27B NInfer, MiMo IQ2_M
6. [ ] spec on/off (MTP, DFlash2, ngram, cache-reuse) deltas
7. [ ] power A/B at 230 W vs 350 W
8. [ ] results.csv + README
9. [ ] restore power limits

## Decisions
- Keep ComfyUI alive (368 MiB, interactive app, not a service).
- Cold prefill = unique random filler prefix (defeats prefix cache). Warm = identical repeat.
- Use llama-server `timings` fields for prompt/predicted rates.
- 200 GiB disk / 1 TB network budget.
