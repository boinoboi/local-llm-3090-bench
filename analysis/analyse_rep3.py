#!/usr/bin/env python3
"""Aggregate the 3-repetition llama.cpp replications for the 35B-A3B."""
import csv, os, statistics, collections

B = os.path.expanduser("~/bench")
rows = list(csv.DictReader(open(f"{B}/data/results_llama.csv")))

want = [m for m in {r["model"] for r in rows} if m.endswith("_rep3")]
print("replicated models:", want)

for model in sorted(want):
    sub = [r for r in rows if r["model"] == model]
    by = collections.defaultdict(list)
    pre = {}
    for r in sub:
        ctx = int(r["ctx_target"])
        by[ctx].append(float(r["decode_tok_s"]))
        if r["phase"] == "cold":
            pre[ctx] = float(r["prefill_tok_s"])
    print(f"\n### {model}")
    print(f"{'ctx':>8} {'n':>3} {'decode median':>14} {'range':>16} {'spread':>7} {'prefill':>9}")
    for ctx in sorted(by):
        v = by[ctx]
        print(f"{ctx//1024:>7}k {len(v):>3} {statistics.median(v):>14.1f} "
              f"{min(v):>7.1f}-{max(v):<7.1f} {max(v)-min(v):>7.1f} {pre.get(ctx, 0):>9.0f}")
