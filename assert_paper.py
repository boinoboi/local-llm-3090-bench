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
rep = rows("results_llama_rep3.csv")
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
pw = [r for r in rows("results_power.csv") if r.get("valid") == "yes" and r.get("workload") == "code"]
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
pr = [r for r in rows("results_power.csv") if r.get("valid") == "yes" and r.get("workload") == "prose"]
pp = collections.defaultdict(list)
for r in pr:
    pp[int(float(r["PL_W"]))].append(r)
eff = {pl: statistics.median(float(x["decode_tok_s"]) for x in v) /
       statistics.median(float(x["draw_W_peak"]) for x in v) * 1000 for pl, v in pp.items()}
check("prose peak efficiency is at 270 W", float(max(eff, key=eff.get)), 270.0, 0)

print("== accuracy (§9.3) ==")
acc = rows("results_accuracy.csv")
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

print(f"\n{ok} passed, {fail} failed")
raise SystemExit(1 if fail else 0)
