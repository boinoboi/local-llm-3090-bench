# GPU inference benchmark — 3090 + 5060 Ti

## Goal
Measure real decode tok/s and prefill tok/s vs context fill (1 K → 262 K) for the local model
families, pick the best engine per model, emit CSVs and a publishable research write-up.

**Status: study written and published.** Document: `~/Videos/Learn/RTX-3090-Inference-Study.md`
(and `STUDY.md` in the repo).

## Hardware
- GPU0 RTX 3090 24 GB, sm_86, 936 GB/s — power limit set to **270 W** (user's choice; default 350)
- GPU1 RTX 5060 Ti 16 GB — unused for inference
- i9-14900K (PL1 95 W / PL2 125 W), 125 GiB RAM, ~670 GiB free

## Deliverables
| path | what |
|---|---|
| `~/Videos/Learn/RTX-3090-Inference-Study.md` | the paper, ~1400 lines, 12k words |
| https://github.com/boinoboi/local-llm-3090-bench | data + code + figures |
| `results_all.csv` | everything, normalised |
| `results_power.csv` | power sweep, 2 workloads, validity flags |
| `results_accuracy.csv` | auto-graded accuracy, 6 model/seed runs |
| `results_llama.csv` | llama.cpp context sweeps |
| `results_mk.csv`, `results_mk_context.csv` | megakernel context data |
| `results_27b_spec.csv` | 27B speculation ladder |
| `plots/`, `figures/` | 5 generated figures |

## Key findings
1. Megakernel 152.8 tok/s vs llama.cpp 71.1 at equal power — 1.54× at ~2× the precision.
2. Power/performance frontier peaks in efficiency at **270 W** on two independent workloads;
   speed keeps rising to 385 W. Last 115 W = +8 % speed for +42 % power.
3. Full 262,144-token context fits 24 GB via 5-bit KV + 8-bit recency window (23,432 MiB).
4. 27B scored 60/60 on auto-graded tasks vs 35B-A3B 53/60 (z = 2.73, p = 0.0064).
5. Speculative throughput is content-dependent: acceptance 54–73 % → up to 13 % spread.
6. `nvidia-smi -pl` fails silently ~16 % of the time; always validate draw.

## Open items
- [ ] 131 K megakernel replication — blocked by server instability past ~64 K
- [ ] `draft-dflash2,draft-mtp` stacking never completed
- [ ] 5060 Ti never evaluated (deliberate: no NVLink)

## Decisions
- Keep ComfyUI alive (368 MiB, interactive).
- Stop ollama permanently during measurements.
- Report single measurements as indicative only; 3–10 reps where claims depend on them.
