#!/usr/bin/env python3
"""Why the efficiency peak and the speed peak are at different watts."""
import csv, os, statistics, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HOME = os.path.expanduser("~")
rows = [r for r in csv.DictReader(open(f"{HOME}/bench/results_power.csv"))
        if r.get("valid", "yes") == "yes"]
by = collections.defaultdict(list)
for r in rows:
    by[int(float(r["PL_W"]))].append(r)

xs, d, p = [], [], []
for pl in sorted(by):
    v = [float(x["decode_tok_s"]) for x in by[pl]]
    xs.append(pl); d.append(statistics.median(v))
    p.append(statistics.median(float(x["draw_W_peak"]) for x in by[pl]))

eff = [dd / pp * 1000 for dd, pp in zip(d, p)]
marg_x, marg = [], []
for i in range(1, len(xs)):
    marg_x.append((xs[i - 1] + xs[i]) / 2)
    marg.append((d[i] - d[i - 1]) / (p[i] - p[i - 1]))

plt.rcParams.update({
    "figure.facecolor": "#0f1117", "axes.facecolor": "#161923",
    "axes.edgecolor": "#2b3040", "axes.labelcolor": "#c9d1e0",
    "text.color": "#c9d1e0", "xtick.color": "#8b93a7", "ytick.color": "#8b93a7",
    "grid.color": "#242938", "font.size": 11, "axes.titlesize": 13,
    "axes.titleweight": "bold", "figure.dpi": 150,
})

fig, ax = plt.subplots(1, 2, figsize=(15, 5.4))
fig.suptitle("Speed always rises. Efficiency does not. They peak at different watts.",
             fontsize=14, fontweight="bold", color="#e8edf7", y=0.98)

# panel 1: speed and efficiency, both normalised to 100 at their own peak
a = ax[0]
ds = [v / max(d) * 100 for v in d]
es = [v / max(eff) * 100 for v in eff]
a.plot(xs, ds, "-o", color="#5eead4", lw=2.6, ms=6, label="decode speed (% of its own max)")
a.plot(xs, es, "-o", color="#f0abfc", lw=2.6, ms=6, label="efficiency (% of its own max)")
a.axvline(270, color="#fde047", ls="--", lw=1.3, alpha=0.85)
a.axvline(385, color="#5eead4", ls="--", lw=1.3, alpha=0.85)
a.annotate("efficiency peaks\n270 W", xy=(270, 100), xytext=(292, 84),
           color="#fde047", fontsize=10, fontweight="bold",
           arrowprops=dict(arrowstyle="->", color="#fde047", lw=1.4))
a.annotate("speed peaks 385 W", xy=(385, 100), xytext=(322, 62),
           color="#5eead4", fontsize=10, fontweight="bold",
           arrowprops=dict(arrowstyle="->", color="#5eead4", lw=1.4))
a.set_title("Normalised: raw speed vs speed-per-watt")
a.set_xlabel("power limit (W)")
a.set_ylabel("% of that metric's own maximum")
a.set_ylim(55, 108)
a.legend(frameon=False, fontsize=9.5, loc="lower left")

# panel 2: marginal return
a = ax[1]
cols = ["#5eead4" if m > 0.3 else ("#fde047" if m > 0.1 else "#7f2d3f") for m in marg]
a.bar(marg_x, marg, width=8.5, color=cols, edgecolor="none")
a.axhline(0, color="#3b4256", lw=1)
for x, m in zip(marg_x, marg):
    if m > 0.5:
        a.text(x, m + 0.02, f"{m:.2f}", ha="center", fontsize=8.5, color="#0f1117", fontweight="bold")
a.set_title("What each extra watt actually buys")
a.set_xlabel("power limit (W)")
a.set_ylabel("extra tokens/s per extra watt")
a.axvspan(270, 385, color="#7f2d3f", alpha=0.12)
a.text(327, 0.86, "spending watts here\nis mostly waste", ha="center",
       color="#e08fa0", fontsize=9.5, fontweight="bold")

for a in ax:
    a.grid(True, alpha=0.4, lw=0.7)
    a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)

fig.tight_layout(rect=[0, 0.02, 1, 0.95])
pth = f"{HOME}/bench/plots/efficiency_explained.png"
fig.savefig(pth, facecolor=fig.get_facecolor())
print("wrote", pth)
