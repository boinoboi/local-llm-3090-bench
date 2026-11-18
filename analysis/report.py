#!/usr/bin/env python3
"""Turn ~/bench/results.csv into a readable markdown report."""
import csv, os, sys
from collections import defaultdict

CSV = os.path.expanduser("~/bench/data/results.csv")
OUT = os.path.expanduser("~/bench/REPORT.md")


def load():
    if not os.path.exists(CSV):
        sys.exit(f"no {CSV}")
    with open(CSV) as f:
        return list(csv.DictReader(f))


def f(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def pivot(rows):
    # (model, ctx, phase) -> row
    d = {}
    for r in rows:
        d[(r["model"], int(r["ctx_target"]), r["phase"])] = r
    return d


def main():
    rows = load()
    if not rows:
        sys.exit("empty csv")
    d = pivot(rows)
    models = sorted({r["model"] for r in rows})
    ctxs = sorted({int(r["ctx_target"]) for r in rows})

    lines = ["# Local inference benchmark — RTX 3090 + RTX 5060 Ti", "",
             "Measured, not estimated. Cold prefill uses a unique random prefix so the",
             "prefix cache cannot contribute; warm prefill repeats the identical prompt.",
             "", f"Rows: {len(rows)}", ""]

    lines += ["## Decode speed (tok/s) — the number that decides everything", "",
              "| model | " + " | ".join(f"{c//1024}k" for c in ctxs) + " |",
              "|---|" + "---|" * len(ctxs)]
    for m in models:
        cells = []
        for c in ctxs:
            r = d.get((m, c, "warm")) or d.get((m, c, "cold"))
            cells.append(f"{f(r['decode_tok_s']):.0f}" if r else "–")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")

    lines += ["", "## Cold prefill (tok/s) — prompt processing from zero cache", "",
              "| model | " + " | ".join(f"{c//1024}k" for c in ctxs) + " |",
              "|---|" + "---|" * len(ctxs)]
    for m in models:
        cells = []
        for c in ctxs:
            r = d.get((m, c, "cold"))
            cells.append(f"{f(r['prefill_tok_s']):.0f}" if r else "–")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")

    lines += ["", "## Time to first token, cold (s)", "",
              "| model | " + " | ".join(f"{c//1024}k" for c in ctxs) + " |",
              "|---|" + "---|" * len(ctxs)]
    for m in models:
        cells = []
        for c in ctxs:
            r = d.get((m, c, "cold"))
            cells.append(f"{f(r['ttft_s']):.1f}" if r else "–")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")

    lines += ["", "## Peak VRAM at 262k (MiB)", "", "| model | VRAM | power W | limit W |", "|---|---|---|"]
    for m in models:
        r = d.get((m, 262144, "cold")) or d.get((m, 262144, "warm"))
        if r:
            lines.append(f"| {m} | {f(r['vram_mib']):.0f} | {f(r['power_w']):.0f} | {f(r['power_limit_w']):.0f} |")

    lines += ["", "## Prefix-cache effect (warm vs cold prefill)", "",
              "| model | ctx | cold t/s | warm t/s | cached tokens | speedup |",
              "|---|---|---|---|---|---|"]
    for m in models:
        for c in ctxs:
            rc, rw = d.get((m, c, "cold")), d.get((m, c, "warm"))
            if rc and rw and f(rc["prefill_tok_s"]) > 0:
                sp = f(rw["prefill_tok_s"]) / f(rc["prefill_tok_s"])
                if sp > 1.05:
                    lines.append(f"| {m} | {c//1024}k | {f(rc['prefill_tok_s']):.0f} | "
                                 f"{f(rw['prefill_tok_s']):.0f} | {f(rw['cached_tokens']):.0f} | {sp:.1f}x |")

    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(lines)} lines) from {len(rows)} rows")


if __name__ == "__main__":
    main()
