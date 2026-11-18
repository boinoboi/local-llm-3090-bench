#!/usr/bin/env python3
"""Parse the recovered 27B speculation ladder (driver_27b.out) into CSV + a figure."""
import csv, os, re, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HOME = os.path.expanduser("~")
SRC = f"{HOME}/bench/driver_27b.out"
OUT = f"{HOME}/bench/data/results_27b_spec.csv"
FIELDS = ["model", "spec", "ctx", "phase", "prompt_tokens", "cached", "prefill_tok_s",
          "decode_tok_s", "ttft_s", "vram_mib"]

cur = None
rows = []
for line in open(SRC, errors="replace"):
    m = re.match(r"=+ (\S+) =+", line.strip())
    if m:
        cur = m.group(1)
        continue
    m = re.match(r"\s*ctx=(\d+)\s+(cold|warm)\s+prompt=(\d+)\s+cached=(\d+)\s+prefill=([\d.]+)\s+"
                 r"decode=([\d.]+)\s+ttft=([\d.]+)\s*vram=([\d.]+)", line)
    if m and cur:
        spec = "none"
        if "mtp" in cur and "dflash2" not in cur:
            spec = "mtp"
        elif "dflash2_mtp" in cur:
            spec = "dflash2+mtp"
        elif "dflash2" in cur:
            spec = "dflash2"
        rows.append(dict(model="27B_Turbo_IQ2_XS", spec=spec, ctx=int(m.group(1)),
                         phase=m.group(2), prompt_tokens=int(m.group(3)),
                         cached=int(m.group(4)), prefill_tok_s=float(m.group(5)),
                         decode_tok_s=float(m.group(6)), ttft_s=float(m.group(7)),
                         vram_mib=float(m.group(8))))

with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"{len(rows)} rows -> {OUT}")

# aggregate warm decode per spec/ctx
agg = {}
for r in rows:
    if r["phase"] != "warm":
        continue
    agg.setdefault(r["spec"], {})[r["ctx"]] = r["decode_tok_s"]

order = ["none", "mtp", "dflash2"]
colors = {"none": "#64748b", "mtp": "#5eead4", "dflash2": "#f0abfc", "dflash2+mtp": "#fde047"}
labels = {"none": "no speculation", "mtp": "native MTP", "dflash2": "DFlash2 block-diffusion"}

plt.rcParams.update({
    "figure.facecolor": "#0f1117", "axes.facecolor": "#161923",
    "axes.edgecolor": "#2b3040", "axes.labelcolor": "#c9d1e0",
    "text.color": "#c9d1e0", "xtick.color": "#8b93a7", "ytick.color": "#8b93a7",
    "grid.color": "#242938", "font.size": 11, "axes.titlesize": 13,
    "axes.titleweight": "bold", "figure.dpi": 150,
})
fig, ax = plt.subplots(figsize=(9.5, 5.6))
for spec in order:
    if spec not in agg:
        continue
    xs = sorted(agg[spec])
    ys = [agg[spec][x] for x in xs]
    ax.plot(xs, ys, "-o", color=colors[spec], lw=2.4, ms=6, label=labels[spec])
    ax.annotate(f"{ys[-1]:.0f}", xy=(xs[-1], ys[-1]), xytext=(6, -3),
                textcoords="offset points", color=colors[spec], fontsize=9, fontweight="bold")

ax.set_xscale("log", base=2)
ax.set_xticks([1024, 4096, 8192, 16384, 32768, 65536, 131072, 262144])
ax.set_xticklabels(["1k", "4k", "8k", "16k", "32k", "64k", "131k", "262k"])
ax.set_xlabel("context length (tokens)")
ax.set_ylabel("decode throughput (tokens/s)")
ax.set_title("Qwen3.8-27B IQ2_XS: speculation backends across context")
ax.grid(True, alpha=0.4, lw=0.7)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=10, loc="lower left")
fig.text(0.5, 0.015, "Single measurements per point, 230 W, greedy, 96 generated tokens",
         ha="center", fontsize=8.5, color="#6b7488")
fig.tight_layout(rect=[0, 0.035, 1, 1])
p = f"{HOME}/bench/plots/context_spec.png"
fig.savefig(p, facecolor=fig.get_facecolor())
print("wrote", p)
