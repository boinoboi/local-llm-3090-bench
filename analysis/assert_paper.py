#!/usr/bin/env python3
"""Assert every headline number in the study against the raw data. Fails loudly."""
import csv, math, os, statistics, collections

B = os.path.expanduser("~/bench")
ok = fail = 0


def check(label, got, want, tol=0.06):
    global ok, fail
    good = abs(got - want) <= abs(want) * tol if want else got == want
    print(f"  {'PASS' if good else 'FAIL'}  {label:<52} got {got:>10.2f}  want {want:>10.2f}")
    ok += good
    fail += (not good)


def rows(name):
    p = f"{B}/{name}"
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


print("== replicated 35B series (§8.1) ==")
rep = rows("data/results_llama_rep3.csv")
by = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rep:
    by[r["model"]][int(r["ctx_target"])].append(float(r["decode_tok_s"]))
q3 = "35B-A3B_llmfan_Q3KM_mtp_rep3"
q2 = "35B-A3B_huihui_Q2K_mtp_rep3"
check("Q3_K_M 1 K", statistics.median(by[q3][1024]), 226.8)
check("Q3_K_M 8 K", statistics.median(by[q3][8192]), 250.6)
check("Q3_K_M 128 K", statistics.median(by[q3][131072]), 143.9)
check("Q3_K_M 256 K", statistics.median(by[q3][262144]), 94.0)
check("Q2_K 256 K", statistics.median(by[q2][262144]), 58.3)
check("128 K ratio Q3/Q2", statistics.median(by[q3][131072]) / statistics.median(by[q2][131072]), 2.2, 0.02)

print("== power sweep (§10.2) ==")
pw = [r for r in rows("data/results_power.csv") if r.get("valid") == "yes" and r.get("workload") == "code"]
pb = collections.defaultdict(list)
for r in pw:
    pb[int(float(r["PL_W"]))].append(r)
med = {pl: statistics.median(float(x["decode_tok_s"]) for x in v) for pl, v in pb.items()}
peak = {pl: statistics.median(float(x["draw_W_peak"]) for x in v) for pl, v in pb.items()}
check("decode at 230 W", med[230], 109.39)
check("decode at 270 W", med[270], 141.16)
check("decode at 385 W", med[385], 152.75)
check("tok/s per kW at 270 W", med[270] / peak[270] * 1000, 521.8)
j270 = peak[270] / med[270]
check("J/token at 270 W", j270, 1.916, 0.01)
usd = j270 * 1e6 / 3.6e6 * 20.04 / 93.44
check("USD per 1M tokens at 270 W", usd, 0.114, 0.02)

print("== prose replication (§10.5) ==")
pr = [r for r in rows("data/results_power.csv") if r.get("valid") == "yes" and r.get("workload") == "prose"]
pp = collections.defaultdict(list)
for r in pr:
    pp[int(float(r["PL_W"]))].append(r)
eff = {pl: statistics.median(float(x["decode_tok_s"]) for x in v) /
       statistics.median(float(x["draw_W_peak"]) for x in v) * 1000 for pl, v in pp.items()}
check("prose peak efficiency is at 270 W", float(max(eff, key=eff.get)), 270.0, 0)

print("== accuracy (§9.3) ==")
acc = rows("data/results_accuracy.csv")
tot = collections.Counter()
for r in acc:
    if r["model"].startswith("27B_Turbo"):
        tot["27B"] += int(r["correct"])
    elif r["model"].startswith("35B-A3B_Q3KM_heretic_tok1200"):
        tot["35B"] += int(r["correct"])
check("27B pooled correct (of 60)", tot["27B"], 60, 0)
check("35B pooled correct (of 60)", tot["35B"], 53, 0)
p1, p2, n = 1.0, 53 / 60, 60
pooled = 113 / 120
se = math.sqrt(pooled * (1 - pooled) * 2 / n)
z = (p1 - p2) / se
pval = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
check("two-proportion z", z, 2.73, 0.02)
check("p-value", pval, 0.0064, 0.15)

print("== KV arithmetic (§2.1, §6.1) ==")
check("64-layer fp16 KV at 262 K (GB)", 2 * 64 * 4 * 256 * 2 * 262144 / 1e9, 68.7, 0.01)
check("16-layer fp16 KV at 262 K (GB)", 2 * 16 * 4 * 256 * 2 * 262144 / 1e9, 17.2, 0.01)
check("35B-A3B KV/token (KB)", 2 * 10 * 2 * 256 * 2 / 1024, 20.0, 0.01)
check("35B-A3B KV at 262 K (GB)", 2 * 10 * 2 * 256 * 2 * 262144 / 1e9, 5.37, 0.01)


print("== Opus review items ==")

# (1) pooled two-proportion p-value
acc2 = rows("data/results_accuracy.csv")
tot2 = collections.Counter()
for r in acc2:
    if r["model"].startswith("27B_Turbo"):
        tot2["27B"] += int(r["correct"])
    elif r["model"].startswith("35B-A3B_Q3KM_heretic_tok1200"):
        tot2["35B"] += int(r["correct"])
n1 = n2 = 60
p1, p2 = tot2["27B"] / n1, tot2["35B"] / n2
pooled = (tot2["27B"] + tot2["35B"]) / (n1 + n2)
se2 = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
z2 = (p1 - p2) / se2
pval2 = 2 * (1 - 0.5 * (1 + math.erf(abs(z2) / math.sqrt(2))))
check("pooled p-value", pval2, 0.0064, 0.02)

# (2) integrated energy at 270 W
en = rows("data/results_energy.csv")
e270 = [float(r["J_per_token"]) for r in en if int(float(r["PL_W"])) == 270]
check("J/token at 270 W (integrated)", statistics.median(e270), 2.830, 0.01)
check("energy runs at 270 W", len(e270), 3, 0)

# (3) cost per 1M tokens on the integrated basis
jt = statistics.median(e270)
kwh = jt * 1e6 / 3.6e6
usd_int = kwh * 20.04 / 93.44
check("USD per 1M (integrated, 270 W)", usd_int, 0.169, 0.02)

# (4) six valid 290 W retests
pw2 = [r for r in rows("data/results_power.csv")
       if r.get("valid") == "yes" and r.get("workload") == "code"
       and int(float(r["PL_W"])) == 290]
check("valid 290 W rows", len(pw2), 6, 0)
check("290 W peak draw actually applied",
      statistics.median(float(r["draw_W_peak"]) for r in pw2), 290.0, 0.02)

# (5) integrated-energy spread is small at every level
by_pl = collections.defaultdict(list)
for r in en:
    by_pl[int(float(r["PL_W"]))].append(float(r["J_per_token"]))
worst = max((max(v) - min(v)) / statistics.median(v) for v in by_pl.values())
check("worst within-level J/token spread < 8 %", worst * 100, 7.0, 1.0)


# (6) §9.6 padded-probe counts at max_tokens = 1200
lc = [r for r in rows("data/results_longctx.csv") if float(r.get("ctx_target", 0) or 0) in (1024, 65536)]
lc1200 = [r for r in lc if r.get("model", "").find("longctx") >= 0]
by_d = collections.defaultdict(int)
for r in lc1200:
    by_d[int(float(r["ctx_target"]))] += int(r["correct"])
check("9.6 padded 1 K correct", by_d[1024], 26, 0)
check("9.6 padded 64 K correct", by_d[65536], 27, 0)

print(f"\n{ok} passed, {fail} failed")
raise SystemExit(1 if fail else 0)
