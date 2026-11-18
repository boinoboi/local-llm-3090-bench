# Local LLM inference on one RTX 3090 — measured

**→ Read the study: [STUDY.md](STUDY.md)**

A measurement study of single-stream LLM inference on a consumer workstation: which engine is
fastest, how speed decays with context length up to 262,144 tokens, what it costs in watts and
dollars, and which of our assumptions turned out to be wrong.

## Headline numbers

| | |
|---|---|
| Fastest configuration measured | **250.6 tok/s** — Qwen3.6-35B-A3B (Q3_K_M) + MTP on `llama.cpp`, 8 K context, 230 W |
| Fastest at full native context | **94.0 tok/s** at 262,144 tokens, 21.2 GB VRAM |
| Fastest single-stream 27B | **152.8 tok/s** with a single-kernel CUDA engine at 385 W; 141.2 at 270 W |
| Most efficient operating point | **270 W** — 521.8 tok/s per kW, **$0.114 per million output tokens** |
| Engine advantage, same model | **1.54×** at equal power, at ~2× the weight precision |
| Prefix caching value | a 256 K follow-up costs **0.22 s** instead of 198 s (**904×**) |
| 2-bit vs 4-bit weights | 2-bit is **slower at every depth** — 58 vs 94 tok/s at 262 K |
| Accuracy probe | 27B **60/60**, 35B-A3B **53/60** (z = 2.73, p = 0.006) |

## What is in here

```
STUDY.md                 the full write-up (~1,700 lines, 40 tables, 6 figures, 17 refs)
REPORT.md                short summary
PROCESS.md               running notes / decisions / open items

results_power.csv        115 power-sweep rows across 2 workloads, with validity flags
results_accuracy.csv     180 graded answers, 6 model/seed configurations
results_longctx.csv      does 64 K of padding hurt accuracy?
results_llama.csv        llama.cpp context sweeps
results_llama_rep3.csv   the replicated 35B series (4 runs per depth)
results_mk*.csv          megakernel context data
results_27b_spec.csv     27B speculation ladder, 1 K → 262 K
results_all.csv          normalised performance rows

measure.py               engine-agnostic context sweep with tokenizer calibration
power_cli_sweep.py       power sweep with draw validation (rejects runs where -pl didn't apply)
accuracy_eval.py         auto-graded probe, ground truth computed by program execution
longctx_eval.py          accuracy under context padding
assert_paper.py          21 assertions: every headline number checked against the raw data
verify_paper.py          recomputes every table from CSVs
plot_*.py                figure generation
figures/                 the six published figures
```

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install matplotlib
.venv/bin/python assert_paper.py     # 21 checks against the committed data
.venv/bin/python verify_paper.py     # regenerate every table
```

## Method notes worth reading even if you skip the paper

- **`nvidia-smi -pl` fails silently.** In 17 of 108 attempts the card kept behaving as if capped
  lower, with exit code 0 and the driver reporting the requested value. Every power measurement here
  is validated against actual draw.
- **Speculative throughput is content-dependent.** Draft acceptance ranged 54–73 % on identical
  context, giving up to 13 % spread. Single-run benchmarks measure luck.
- **Quantisation is not free.** Fewer bits per token produced fewer tokens per second, by 54 % at
  128 K.
- **Check the tensor table, not the filename.** Two popular GGUF distributions silently dropped the
  MTP block that makes them fast.

## Environment

RTX 3090 24 GB (SM 8.6) · i9-14900K · 128 GB DDR5 · CachyOS / Linux 7.2.9 · CUDA 13.4 ·
`llama.cpp` b43bd7a · `qwen38-megakernel` ed0d237 · MSEDCL tariff FY 2026-27 · USD/INR 93.44

All numbers are single-stream, batch size 1, measured 18 November 2026.
