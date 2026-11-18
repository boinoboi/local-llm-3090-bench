# AGENTS.md

Guidance for AI agents working in this repository.

## What this is

A measurement study, not an application. The deliverable is a set of numbers that must stay
true to the raw data in this directory. Code here exists to produce and verify measurements.

- **`STUDY.md`** — the paper (~1,630 lines). Mirror of `~/Videos/Learn/RTX-3090-Inference-Study.md`.
- **`README.md`** — landing page and headline numbers.
- **`results_*.csv`** — the data. Everything else is derived.

If you change a number in `STUDY.md` that also appears in a CSV, you have created a bug.

## The gate

```bash
.venv/bin/python assert_paper.py     # must print "30 passed, 0 failed"
```

Every headline figure is asserted against the CSVs. Run it before and after any edit that touches
data, prose containing numbers, or a sweep script. Do not report a change as done until it exits 0.

Other checks:

```bash
.venv/bin/python verify_paper.py              # regenerate tables from CSVs
python3 check_tables.py STUDY.md # table column consistency
```

`.venv` already exists with matplotlib. If missing: `python3 -m venv .venv && .venv/bin/pip install matplotlib`.

## Hard-won invariants

These were all discovered by getting them wrong. Breaking one produces plausible-looking garbage.

1. **`nvidia-smi -pl` fails silently.** Roughly 16 % of the time the card keeps behaving as if
   capped lower, with exit code 0 and the driver reporting the requested value. **Always validate
   `draw_W_peak ≥ 0.93 × limit`** and drop the run otherwise. `results_power.csv` carries a `valid`
   column for exactly this reason.
2. **Never average across `workload`.** `results_power.csv` contains two prompt families (`code`,
   `prose`). They have different draft acceptance and therefore different absolute throughput.
   Filter on `workload` before computing any statistic.
3. **Calibrate prompts against the tokenizer, never a word count.** Random-word filler tokenises at
   ~1.8 tokens/word, which inflated an entire early sweep by 20 % and overflowed the context.
   `measure.py::build_prompt` does this iteratively; keep it.
4. **Single runs are indicative only.** Draft acceptance ranged 54–73 % on identical context,
   giving up to 13 % spread. Claims that matter use 3–10 repetitions.
5. **Greedy decoding is clock-invariant.** In the energy sweep, acceptance was byte-identical at
   34.4 % across all 21 runs — that is what makes the J/token comparison valid.

## Environment assumptions

| | |
|---|---|
| GPU 0 | RTX 3090 24 GB, SM 8.6 — the only device used for inference |
| GPU 1 | RTX 5060 Ti — **never use it**; 36 SMs, and the megakernel needs ≥ 80 |
| Pin the device | `CUDA_VISIBLE_DEVICES=0`, or `-dev CUDA0` for llama.cpp |
| Power limit | 270 W is the user's choice. It **resets to 230 W** on driver reload, so set it explicitly before measuring: `sudo nvidia-smi -i 0 -pl 270`. Keep `power-profiles-daemon` stopped or it also reverts |
| Stop first | `ollama` is disabled; a stray server holding VRAM shows up as "failed to load model" |
| ComfyUI | left running (368 MiB) by the user's choice |

## Engines

**llama.cpp** — `~/src/llama.cpp/build/bin/llama-server`. Reference configuration:

```bash
llama-server -m <model>.gguf -ngl 99 -dev CUDA0 -c 262144 -np 1 \
  -ctk q4_0 -ctv q4_0 -fa on --spec-type draft-mtp --spec-draft-n-max 3
```

`-ctk/-ctv q4_0` is what makes 262 K fit. `--spec-type` accepts `draft-mtp`, `draft-dflash2`,
`ngram-simple`, and comma-separated combinations.

**megakernel** — `~/src/megakernel/`. Needs preset env vars for full context:

```bash
export MK_KV='K*=5r,V*=5' MK_KV_WIN=2048 MK_PF_CAP=1024 MK_KEEP_WARM_S=0
./qwen38_mk -m model.gguf -c 262144 -d 7 --kv q8
```

- Only accepts `Q4_K_M` / `UD-Q4_K_XL`. Other quants fall back to llama.cpp.
- The **server** dies past ~64 K with repeated prompts; the **CLI** is stable. Do not treat a server
  failure as a model limitation.
- `MK_KEEP_WARM_S=0` is mandatory on a 24/7 box.

## Data schemas

| File | Rows | Notes |
|---|---|---|
| `results_llama.csv` | 87 | single-run context sweeps |
| `results_llama_rep3.csv` | 64 | the replicated 35B series behind §8.1 — 4 runs/depth |
| `results_power.csv` | 115 | `valid` + `workload` columns; filter both |
| `results_energy.csv` | 21 | integrated decode-window energy, 3 runs/level |
| `results_accuracy.csv` | 180 | graded answers; `model` may carry an `@s<seed>` suffix |
| `results_longctx.csv` | 60 | padded-context probe at `max_tokens = 1200` |
| `results_27b_spec.csv` | 46 | speculation ladder |
| `results_all.csv` | 193 | older normalised aggregate — **stale**, do not cite |

## Editing the study

- Keep `STUDY.md` and `~/Videos/Learn/RTX-3090-Inference-Study.md` identical (`cp` one to the other).
- Use `[M]` measured / `[R]` reported / `[D]` derived tags; do not present a vendor figure as measured.
- Figures regenerate from `plot_*.py` into `plots/`; the published copies live in `figures/`.
  Copy after regenerating.
- When a claim is withdrawn or corrected, **say so in place**. The document keeps its mistakes
  visible (`§12`, `§13.2`, §8.4d) because that is the point of it.

## Do not

- Do not push to the remote without being asked. The user uploads manually.
- Do not change the GPU power limit, stop services, or reboot without explicit instruction.
- Do not run a benchmark while another server holds VRAM. Check `nvidia-smi` first.
- Do not "improve" a number to make a claim cleaner. If a measurement contradicts a section,
  the section is wrong.

## Style

No emoji in the study. Plain British spelling. Numbers to a consistent precision per table.
Prose claims state their measurement basis ("integrated", "peak-draw", "single run").
