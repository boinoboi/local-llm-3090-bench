#!/usr/bin/env python3
"""Plot the power-limit sweep: speed, efficiency, and cost."""
import csv, os, statistics, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HOME = os.path.expanduser("~")
SRC = f"{HOME}/bench/data/results_power.csv"
OUTDIR = f"{HOME}/bench/plots"
os.makedirs(OUTDIR, exist_ok=True)

INR_PER_KWH = 20.04          # MSEDCL LT-I(B) 301-500 slab, incl wheeling + 16% duty
USD_PER_INR = 1 / 93.44

rows = list(csv.DictReader(open(SRC)))
by = collections.defaultdict(list)
for r in rows:
    if r.get("valid", "yes") != "yes":
        continue
    by[int(float(r["PL_W"]))].append(r)

xs, med, lo, hi, eff, cost = [], [], [], [], [], []
for pl in sorted(by):
    rs = by[pl]
    v = [float(x["decode_tok_s"]) for x in rs]
    pk = statistics.median(float(x["draw_W_peak"]) for x in rs)
    d = statistics.median(v)
    xs.append(pl); med.append(d); lo.append(min(v)); hi.append(max(v))
    eff.append(d / pk * 1000)                     # tok/s per kW
    jtok = pk / d                                 # joules per token
    kwh_per_m = jtok * 1e6 / 3.6e6
    cost.append(kwh_per_m * INR_PER_KWH * USD_PER_INR)

plt.rcParams.update({
    "figure.facecolor": "#0f1117", "axes.facecolor": "#161923",
    "axes.edgecolor": "#2b3040", "axes.labelcolor": "#c9d1e0",
    "text.color": "#c9d1e0", "xtick.color": "#8b93a7", "ytick.color": "#8b93a7",
    "grid.color": "#242938", "font.size": 11, "axes.titlesize": 13,
    "axes.titleweight": "bold", "figure.dpi": 150,
})

fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
fig.suptitle("RTX 3090 · Qwen3.8-27B Q4_K_M · single stream · 8k context",
             fontsize=14, fontweight="bold", color="#e8edf7", y=0.99)

ACCENT, ACCENT2, DIM = "#5eead4", "#f0abfc", "#3b4256"

# --- 1. speed -------------------------------------------------------------
a = ax[0]
a.fill_between(xs, lo, hi, color=ACCENT, alpha=0.15, label="run min–max")
a.plot(xs, med, "-o", color=ACCENT, lw=2.4, ms=6, label="median decode")
knee = 330
a.axvline(knee, color=ACCENT2, ls="--", lw=1.4, alpha=0.8)
a.annotate(f"knee {knee} W\n{med[xs.index(knee)]:.1f} t/s",
           xy=(knee, med[xs.index(knee)]), xytext=(knee - 78, 128),
           color=ACCENT2, fontsize=10, fontweight="bold",
           arrowprops=dict(arrowstyle="->", color=ACCENT2, lw=1.4))
a.set_title("Decode speed")
a.set_xlabel("power limit (W)")
a.set_ylabel("tokens / second")
a.legend(frameon=False, fontsize=9, loc="lower right")
a.axhline(med[-1], color=DIM, ls=":", lw=1)
a.text(xs[0] + 2, med[-1] + 2.2, f"max {med[-1]:.1f} t/s", color="#8b93a7", fontsize=8.5)

# --- 2. efficiency --------------------------------------------------------
a = ax[1]
best = eff.index(max(eff))
a.plot(xs, eff, "-o", color=ACCENT2, lw=2.4, ms=6)
a.scatter([xs[best]], [eff[best]], s=170, facecolors="none",
          edgecolors="#fde047", lw=2.4, zorder=5)
a.annotate(f"most efficient\n{xs[best]} W · {eff[best]:.0f} tok/s/kW",
           xy=(xs[best], eff[best]), xytext=(xs[best] + 12, eff[best] + 22),
           color="#fde047", fontsize=10, fontweight="bold",
           arrowprops=dict(arrowstyle="->", color="#fde047", lw=1.4))
a.set_title("Efficiency (higher = better)")
a.set_xlabel("power limit (W)")
a.set_ylabel("tokens / second / kW")

# --- 3. cost --------------------------------------------------------------
a = ax[2]
a.plot(xs, cost, "-o", color="#7dd3fc", lw=2.4, ms=6)
a.fill_between(xs, cost, min(cost), color="#7dd3fc", alpha=0.12)
a.set_title("Cost per 1M output tokens")
a.set_xlabel("power limit (W)")
a.set_ylabel("USD per million tokens")
a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:.3f}"))
a.annotate(f"${cost[-1]:.3f}", xy=(xs[-1], cost[-1]), xytext=(xs[-1] - 46, cost[-1] - 0.006),
           color="#7dd3fc", fontsize=10, fontweight="bold")
a.annotate(f"${cost[best]:.3f}", xy=(xs[best], cost[best]), xytext=(xs[best] + 8, cost[best] + 0.004),
           color="#fde047", fontsize=10, fontweight="bold")

for a in ax:
    a.grid(True, alpha=0.45, lw=0.7)
    a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)

fig.text(0.5, 0.015,
         f"Cost = {INR_PER_KWH} INR/kWh (MSEDCL 301-500 slab incl. wheeling + 16% duty)  ·  "
         f"USD/INR 93.44  ·  17 runs dropped where the limit silently failed to apply",
         ha="center", fontsize=8.5, color="#6b7488")

fig.tight_layout(rect=[0, 0.035, 1, 0.96])
p = f"{OUTDIR}/power_curve.png"
fig.savefig(p, facecolor=fig.get_facecolor())
print("wrote", p)
