#!/usr/bin/env python3
"""Figure 6: the 2-bit model is slower than the 4-bit model at every depth."""
import csv, os, statistics, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HOME = os.path.expanduser("~")
rows = list(csv.DictReader(open(f"{HOME}/bench/results.csv")))
rows = [r for r in rows if r["model"].endswith("_rep3")]

by = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    by[r["model"]][int(r["ctx_target"])].append(float(r["decode_tok_s"]))

labels = {
    "35B-A3B_llmfan_Q3KM_mtp_rep3": ("Q3_K_M  (~3.9 bpw)", "#5eead4"),
    "35B-A3B_huihui_Q2K_mtp_rep3": ("Q2_K  (~2.3 bpw)", "#f0abfc"),
}

plt.rcParams.update({
    "figure.facecolor": "#0f1117", "axes.facecolor": "#161923",
    "axes.edgecolor": "#2b3040", "axes.labelcolor": "#c9d1e0",
    "text.color": "#c9d1e0", "xtick.color": "#8b93a7", "ytick.color": "#8b93a7",
    "grid.color": "#242938", "font.size": 11, "axes.titlesize": 13,
    "axes.titleweight": "bold", "figure.dpi": 150,
})

fig, ax = plt.subplots(figsize=(10, 5.6))
for model, (name, col) in labels.items():
    if model not in by:
        continue
    xs = sorted(by[model])
    med = [statistics.median(by[model][x]) for x in xs]
    lo = [min(by[model][x]) for x in xs]
    hi = [max(by[model][x]) for x in xs]
    ax.fill_between(xs, lo, hi, color=col, alpha=0.16)
    ax.plot(xs, med, "-o", color=col, lw=2.6, ms=7, label=name)
    for x, m in zip(xs, med):
        if x in (1024, 131072, 262144):
            ax.annotate(f"{m:.0f}", xy=(x, m), xytext=(0, 9), textcoords="offset points",
                        ha="center", color=col, fontsize=9.5, fontweight="bold")

ax.set_xscale("log", base=2)
ax.set_xticks([1024, 4096, 8192, 16384, 32768, 65536, 131072, 262144])
ax.set_xticklabels(["1k", "4k", "8k", "16k", "32k", "64k", "131k", "262k"])
ax.set_xlabel("context length (tokens)")
ax.set_ylabel("decode throughput (tokens/s)")
ax.set_title("Less bytes per token, fewer tokens per second")
ax.grid(True, alpha=0.4, lw=0.7)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=10.5, loc="lower left")

ax.annotate("2.4× gap at 128 K", xy=(131072, 104), xytext=(5000, 62),
            fontsize=10.5, color="#e08fa0", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#e08fa0", lw=1.5,
                            connectionstyle="arc3,rad=-0.15"))

fig.text(0.5, 0.015,
         "Qwen3.6-35B-A3B, llama.cpp, q4_0 KV, 230 W. Four runs per point; band is the observed min–max.",
         ha="center", fontsize=8.5, color="#6b7488")
fig.tight_layout(rect=[0, 0.035, 1, 1])
p = f"{HOME}/bench/plots/quant_speed.png"
fig.savefig(p, facecolor=fig.get_facecolor())
print("wrote", p)
