#!/usr/bin/env python3
"""Rebuild results_energy.csv with full provenance parsed from the engine logs.

Records, per run: model, context, prompt id, tokens generated, decode seconds, decode tok/s,
draft acceptance, tokens per cycle, mean board power inside the decode window, and J/token.
"""
import csv, glob, os, re, statistics, collections

OUT = os.path.expanduser("~/bench/data/data/results_energy.csv")
RE = re.compile(
    r"decode (\d+) tok in ([0-9.]+) s = ([0-9.]+) tok/s \| cycles (\d+), "
    r"accepted (\d+)/(\d+) drafts \(([0-9.]+)%\), ([0-9.]+) tok/cycle")

MODEL = "Qwen3.8-27B-UD-Q4_K_M.gguf"
CTX = 8192
PROMPT_ID = "os-scheduler-explainer"
NPRED = 512

FIELDS = ["PL_W", "rep", "model", "ctx", "prompt_id", "n_predict", "tokens", "decode_s",
          "decode_tok_s", "accept_pct", "tok_per_cycle", "mean_decode_W", "J_per_token"]

prev = {}
if os.path.exists(OUT):
    for r in csv.DictReader(open(OUT)):
        prev[(int(float(r["PL_W"])), int(float(r["rep"])))] = r

rows = []
for log in sorted(glob.glob("/tmp/opencode/en_*_*.log")):
    base = os.path.basename(log)[3:-4]
    try:
        pl, rep = base.split("_")
        pl, rep = int(pl), int(rep)
    except ValueError:
        continue
    txt = open(log, errors="replace").read()
    m = RE.findall(txt)
    if not m:
        continue
    ntok, dsec, tps, cyc, acc, tot, pct, tpc = m[-1]
    key = (pl, rep)
    if key not in prev:
        continue
    rows.append(dict(PL_W=pl, rep=rep, model=MODEL, ctx=CTX, prompt_id=PROMPT_ID,
                     n_predict=NPRED, tokens=int(ntok), decode_s=float(dsec),
                     decode_tok_s=float(tps), accept_pct=float(pct),
                     tok_per_cycle=float(tpc),
                     mean_decode_W=float(prev[key]["mean_decode_W"]),
                     J_per_token=float(prev[key]["J_per_token"])))

rows.sort(key=lambda r: (r["PL_W"], r["rep"]))
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows)

by = collections.defaultdict(list)
for r in rows:
    by[r["PL_W"]].append(r)

print(f"{'W':>4} {'n':>2} {'tok/s':>7} {'accept':>7} {'tok/cyc':>8} {'W':>6} "
      f"{'J/tok':>7} {'J spread':>10} {'accept spread':>14}")
for pl in sorted(by):
    rs = by[pl]
    j = [r["J_per_token"] for r in rs]
    a = [r["accept_pct"] for r in rs]
    print(f"{pl:>4} {len(rs):>2} {statistics.median(r['decode_tok_s'] for r in rs):>7.1f} "
          f"{statistics.median(a):>6.1f}% {statistics.median(r['tok_per_cycle'] for r in rs):>8.2f} "
          f"{statistics.median(r['mean_decode_W'] for r in rs):>6.1f} "
          f"{statistics.median(j):>7.3f} {min(j):.3f}-{max(j):<6.3f} {min(a):.1f}-{max(a):<6.1f}")
print(f"\n{len(rows)} rows -> {OUT}")
