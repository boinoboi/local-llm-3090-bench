# Status and decisions

Living document. Update at the end of each working session rather than re-deriving state.

## Current status

**The study is complete and published.** `STUDY.md` (1,631 lines, 43 tables, 6 figures, 17
references) is the deliverable; `README.md` is the showcase; `data/` holds every measurement.

Last verified: `analysis/assert_paper.py` → **30 passed, 0 failed**.

## Machine state

| | |
|---|---|
| GPU 0 power limit | **270 W** — the measured efficiency optimum. Resets to 230 W on driver reload |
| `power-profiles-daemon` | **stopped** — it silently reverts the power limit |
| `ollama` | stopped and disabled |
| Display blanking | 5 min, display only — no lock, no suspend |
| ComfyUI | left running (368 MiB) at the user's request |

## Decisions

1. **270 W, not 385 W.** Integrated energy, not peak draw, is the denominator. 270 W is the minimum
   joules-per-token point on two independent workloads; 385 W costs ~29 % more per token for ~8 %
   more speed.
2. **Report a cost range, not a point.** $0.11–0.17 per million output tokens, because acceptance
   (hence joules/token) depends on prompt predictability. A single figure was overstated.
3. **4-bit weights, 5-bit KV.** 2-bit was slower at every depth *and* lower quality; bits belong in
   the cache, which is what makes 262 K fit.
4. **Discard invalid power runs rather than average them.** `nvidia-smi -pl` fails silently ~16 % of
   the time; the CSV carries a `valid` column and the study reports the drop rate.
5. **Keep corrections visible.** Withdrawn claims stay in the text with their replacement, rather than
   being quietly deleted. That is why §8.4d, §12 and §13.2 read the way they do.
6. **The engine comparison is indicative.** Prompt family, draft count and quant differ between the
   two stacks, so the ratio is stated as ~1.5× and never as a controlled result.

## Open questions

Listed with the experiment that would settle each, in §13.5 of the study. The two most promising:

- **Server vs CLI gap.** The megakernel's server reads slower than its CLI at depth, but the prompts
  differed, so the gap is unproven. Needs identical prompts through both paths.
- **Where the 2-bit penalty comes from.** Acceptance, dequantisation cost or kernel maturity — we
  observed the effect but could not apportion it.

## Not done

- No multi-user / continuous-batching measurements: everything is batch size 1.
- The RTX 5060 Ti was never evaluated (no NVLink, no benefit for fitting models).
- `draft-dflash2,draft-mtp` stacking never completed.
- 131 K megakernel replication exists only via the CLI path, not the server.
