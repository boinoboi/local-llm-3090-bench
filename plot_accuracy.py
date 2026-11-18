#!/usr/bin/env python3
"""Figure 4: auto-graded accuracy by task family."""
import csv, collections, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HOME = os.path.expanduser("~")
rows = list(csv.DictReader(open(f"{HOME}/bench/results_accuracy.csv")))

pick = {
    "27B_Turbo_IQ2XS": "27B  IQ2_XS",
    "35B-A3B_Q3KM_heretic_tok1200": "35B-A3B  Q3_K_M",
}
cats = ["arith", "code", "count"]
data = {}
for r in rows:
    if r["model"] in pick:
        data.setdefault(pick[r["model"]], collections.defaultdict(list))
        data[pick[r["model"]]][r["category"]].append(int(r["correct"]))

plt.rcParams.update({
    "figure.facecolor": "#0f1117", "axes.facecolor": "#161923",
    "axes.edgecolor": "#2b3040", "axes.labelcolor": "#c9d1e0",
    "text.color": "#c9d1e0", "xtick.color": "#8b93a7", "ytick.color": "#8b93a7",
    "grid.color": "#242938", "font.size": 11, "axes.titlesize": 13,
    "axes.titleweight": "bold", "figure.dpi": 150,
})

models = list(pick.values())
x = np.arange(len(cats))
w = 0.36
colors = ["#5eead4", "#f0abfc"]

fig, ax = plt.subplots(figsize=(9, 5.2))
bars_by_model = []
for i, m in enumerate(models):
    vals = [100 * sum(data[m][c]) / len(data[m][c]) for c in cats]
    bars = ax.bar(x + (i - 0.5) * w, vals, w, label=m, color=colors[i], edgecolor="none")
    bars_by_model.append(bars[0])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}", ha="center",
                fontsize=10, fontweight="bold", color=colors[i])

ax.set_xticks(x)
ax.set_xticklabels(["arithmetic\n(10 tasks)", "code output\n(10 tasks)", "character count\n(10 tasks)"])
ax.set_ylabel("accuracy (%)")
ax.set_ylim(0, 118)
ax.set_title("Auto-graded accuracy: the whole gap is one task family")
ax.grid(True, axis="y", alpha=0.4, lw=0.7)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

ax.text(0.01, 0.97, "overall  30/30  vs  26/30    z = 2.07, p = 0.038",
        transform=ax.transAxes, va="top", fontsize=9.5, color="#8b93a7")
fig.legend(bars_by_model, models, loc="lower center", ncol=2, frameon=False,
           fontsize=10.5, bbox_to_anchor=(0.5, 0.055))
fig.text(0.5, 0.008,
         "Same engine, same settings, greedy, thinking disabled. Ground truth computed by program execution.",
         ha="center", fontsize=8.5, color="#6b7488")
fig.tight_layout(rect=[0, 0.10, 1, 1])
p = f"{HOME}/bench/plots/accuracy.png"
fig.savefig(p, facecolor=fig.get_facecolor())
print("wrote", p)
