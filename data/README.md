# data/

Every measurement in the study, as CSV. Nothing here is derived from a model or simulated — these are
rows written by the harness in `../scripts/` as it ran.

**If a number appears in `STUDY.md` and also here, they must agree.** `../analysis/assert_paper.py`
enforces that for 30 headline figures and exits non-zero on any mismatch.

## Files

| File | Rows | What it is |
|---|---|---|
| `results_llama_rep3.csv` | 64 | **The replicated 35B series** behind §8.1 — 4 runs per depth, Q2_K vs Q3_K_M, 1 K → 262 K |
| `results_power.csv` | 115 | Power sweep, 230–385 W, **two prompt workloads**, with validity flags |
| `results_energy.csv` | 21 | Integrated decode-window energy — 7 limits × 3 runs, 10 Hz sampling |
| `results_accuracy.csv` | 180 | Auto-graded answers, 6 model/seed configurations × 30 tasks |
| `results_longctx.csv` | 60 | Padded-context accuracy probe at `max_tokens = 1200` |
| `results_longctx_900cap.csv` | 60 | The earlier 900-token pilot, superseded; kept as evidence for the §9.6 footnote |
| `results_llama.csv` | 87 | Single-run context sweeps across 6 configurations |
| `results_mk.csv` | 16 | megakernel context points parsed from engine logs |
| `results_mk_context.csv` | 10 | megakernel replicated context points |
| `results_27b_spec.csv` | 46 | 27B speculation ladder — none / MTP / DFlash2, 1 K → 262 K |
| `results_27b_270w.csv` | 32 | 27B at the recommended 270 W operating point |
| `results_all.csv` | 193 | Older normalised aggregate. **Stale — do not cite.** |

## Traps

1. **`results_power.csv` needs two filters.** `valid == "yes"` drops runs where `nvidia-smi -pl`
   silently failed to apply; `workload` separates the code prompt from the prose prompt. Averaging
   across either produces a number that is wrong but plausible.
2. **`results_accuracy.csv` model names can carry an `@s<seed>` suffix.** Sum by prefix, not by exact
   string, or you will count one seed instead of two.
3. **Single-run cells are indicative only.** Draft acceptance varies with what the model writes; where
   it matters the study uses 3–10 repetitions, and the repeated files are named `_rep3`.
4. **Absolute throughput is not comparable across files** unless the prompt family, power limit and KV
   setting match. Each row records what it can; the study states the basis at every table.

## Generating

```bash
cd ..                                   # scripts resolve paths from the repo root
python3 scripts/power_cli_sweep.py      # -> data/results_power.csv   (validates draw)
python3 scripts/energy_sweep.py         # -> data/results_energy.csv  (10 Hz, decode window only)
```

See `../AGENTS.md` for the invariants these scripts depend on.
