#!/usr/bin/env python3
"""Figure 5: the efficiency optimum holds across two independent workloads."""
import csv, os, statistics, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HOME = os.path.expanduser("~")
rows = [r for r in csv.DictReader(open(f"{HOME}/bench/results_power.csv"))
        if r.get("valid", "yes") == "yes"]

data = {}
for r in rows:
    wl = r.get("workload", "code")
    data.setdefault(wl, collections.defaultdict(list))
    data[wl][int(float(r["PL_W"]))].append(r)

series = {}
for wl, by in data.items():
    xs, eff, dec = [], [], []
    for pl in sorted(by):
        v = [float(x["decode_tok_s"]) for x in by[pl]]
        pk = statistics.median(float(x["draw_W_peak"]) for x in by[pl])
        d = statistics.median(v)
        xs.append(pl); eff.append(d / pk * 1000); dec.append(d)
    series[wl] = (xs, eff, dec)

plt.rcParams.update({
    "figure.facecolor": "#0f1117", "axes.facecolor": "#161923",
    "axes.edgecolor": "#2b3040", "axes.labelcolor": "#c9d1e0",
    "text.color": "#c9d1e0", "xtick.color": "#8b93a7", "ytick.color": "#8b93a7",
    "grid.color": "#242938", "font.size": 11, "axes.titlesize": 13,
    "axes.titleweight": "bold", "figure.dpi": 150,
})

fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.5, 5.4))
fig.suptitle("The efficiency optimum is a property of the card, not of the prompt",
             fontsize=14, fontweight="bold", color="#e8edf7", y=0.98)

colors = {"code": "#5eead4", "prose": "#f0abfc"}
names = {"code": "code-generation prompt", "prose": "technical prose prompt"}

for wl, (xs, eff, dec) in series.items():
    a1.plot(xs, eff, "-o", color=colors[wl], lw=2.4, ms=6, label=names[wl])
    best = eff.index(max(eff))
    a1.scatter([xs[best]], [eff[best]], s=180, facecolors="none",
               edgecolors="#fde047", lw=2.4, zorder=5)
    a1.annotate(f"{xs[best]} W", xy=(xs[best], eff[best]),
                xytext=(-6, -26 if wl == "code" else 16), textcoords="offset points",
                color="#fde047", fontsize=10, fontweight="bold", ha="center")
    a2.plot(xs, dec, "-o", color=colors[wl], lw=2.4, ms=6, label=names[wl])

a1.set_title("Efficiency: both workloads peak at 270 W")
a1.set_xlabel("power limit (W)")
a1.set_ylabel("tokens / second / kW")
a1.legend(frameon=False, fontsize=9.5, loc="lower left")

a2.set_title("Raw speed: same shape, different height")
a2.set_xlabel("power limit (W)")
a2.set_ylabel("decode throughput (tokens/s)")
a2.legend(frameon=False, fontsize=9.5, loc="lower right")

for a in (a1, a2):
    a.grid(True, alpha=0.4, lw=0.7)
    a.set_axisbelow(True)
    a.axvline(270, color="#fde047", ls="--", lw=1.2, alpha=0.6)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)

fig.text(0.5, 0.015,
         "Qwen3.8-27B UD-Q4_K_M, megakernel, 6–10 runs per level, greedy, 8 K context. "
         "The prose sweep covers four levels only.",
         ha="center", fontsize=8.5, color="#6b7488")
fig.tight_layout(rect=[0, 0.035, 1, 0.95])
p = f"{HOME}/bench/plots/workload_replication.png"
fig.savefig(p, facecolor=fig.get_facecolor())
print("wrote", p)
