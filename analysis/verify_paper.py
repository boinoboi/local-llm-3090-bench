#!/usr/bin/env python3
"""Recompute every table in the study from raw CSVs, so the prose can be checked against data."""
import csv, os, statistics, collections

B = os.path.expanduser("~/bench")


def rd(name):
    p = f"{B}/{name}"
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


print("=" * 78)
print("POWER TABLE  (data/results_power.csv, workload=code, valid rows only)")
print("=" * 78)
rows = [r for r in rd("data/results_power.csv")
        if r.get("valid", "yes") == "yes" and r.get("workload") == "code"]
by = collections.defaultdict(list)
for r in rows:
    by[int(float(r["PL_W"]))].append(r)
print(f"{'W':>4} {'n':>3} {'median':>8} {'min':>7} {'max':>7} {'peakW':>6} {'tok/s/kW':>9} {'J/tok':>7}")
for pl in sorted(by):
    v = [float(x["decode_tok_s"]) for x in by[pl]]
    pk = statistics.median(float(x["draw_W_peak"]) for x in by[pl])
    d = statistics.median(v)
    print(f"{pl:>4} {len(v):>3} {d:8.2f} {min(v):7.2f} {max(v):7.2f} {pk:5.0f}W {d/pk*1000:9.1f} {pk/d:7.3f}")

print()
print("=" * 78)
print("COST PER 1M OUTPUT TOKENS  (INR 20.04/kWh, USD/INR 93.44)")
print("=" * 78)
KWH_INR, INR_USD = 20.04, 1 / 93.44
print(f"{'W':>4} {'J/token':>8} {'kWh/1M':>8} {'INR/1M':>9} {'USD/1M':>8}")
for pl in sorted(by):
    v = [float(x["decode_tok_s"]) for x in by[pl]]
    pk = statistics.median(float(x["draw_W_peak"]) for x in by[pl])
    jt = pk / statistics.median(v)
    kwh = jt * 1e6 / 3.6e6
    print(f"{pl:>4} {jt:8.3f} {kwh:8.3f} {kwh*KWH_INR:9.2f} {kwh*KWH_INR*INR_USD:8.3f}")

print()
print("=" * 78)
print("LLAMA.CPP DECODE (warm) and COLD PREFILL")
print("=" * 78)
ll = rd("data/results_llama.csv")
d = {}
for r in ll:
    d[(r["model"], int(r["ctx_target"]), r["phase"])] = r
models = sorted({r["model"] for r in ll})
ctxs = sorted({int(r["ctx_target"]) for r in ll})
print("DECODE  " + f"{'model':<34}" + "".join(f"{c//1024:>7}k" for c in ctxs))
for m in models:
    print(f"        {m:<34}" + "".join(
        f"{float(d[(m,c,'warm')]['decode_tok_s']):>8.0f}" if (m, c, "warm") in d else f"{'-':>8}"
        for c in ctxs))
print("PREFILL " + f"{'model':<34}" + "".join(f"{c//1024:>7}k" for c in ctxs))
for m in models:
    print(f"        {m:<34}" + "".join(
        f"{float(d[(m,c,'cold')]['prefill_tok_s']):>8.0f}" if (m, c, "cold") in d else f"{'-':>8}"
        for c in ctxs))

print()
print("=" * 78)
print("MEGAKERNEL (data/results_mk.csv)")
print("=" * 78)
for r in rd("data/results_mk.csv"):
    print(f"  {r['model']:<28} ctx={r['ctx_target']:<7} prompt={r['prompt_tokens']:<7} "
          f"prefill={r['prefill_tok_s']:<7} decode={r['decode_tok_s']:<7} "
          f"acc={r['draft_accept_pct']:<5}")

print()
print("=" * 78)
print("DERIVED KV BUDGET")
print("=" * 78)
for name, layers, kvh, hd in [("35B-A3B", 10, 2, 256), ("27B", 16, 4, 256), ("MiMo-9B", 8, 4, 256)]:
    b = 2 * layers * kvh * hd * 2
    print(f"{name:<10} full-attn {layers:>2}  KV/token {b/1024:6.1f} KB   "
          f"32K {b*32768/2**30:5.2f} GB  131K {b*131072/2**30:5.2f} GB  262K {b*262144/2**30:6.2f} GB")
