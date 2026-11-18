#!/usr/bin/env python3
"""Does padding the context degrade accuracy?

Same questions, same engine, same model — only the amount of irrelevant text in front of the
question changes. A shared padding prefix is used throughout so prefix caching keeps the cost
manageable after the first request at each depth.
"""
import csv, json, os, random, re, statistics, sys, time
import urllib.request

sys.path.insert(0, os.path.expanduser("~/bench"))
from accuracy_eval import build_tasks, grade, PROMPT_TMPL  # noqa: E402

PORT = int(sys.argv[1])
LABEL = sys.argv[2]
OUT = os.path.expanduser("~/bench/results_longctx.csv")
FIELDS = ["model", "ctx_target", "task", "expected", "correct", "got", "latency_s"]
DEPTHS = [1024, 65536]
N_TASKS = 30

PARA = ("The depot ledger records every crate by weight, by the initials of the clerk who counted "
        "it, and by the day it left the yard. Entries are checked twice and signed once. Rain in "
        "the fourth week turned the road to clay and the carts waited. ")


def pad_for(n_tokens):
    return PARA * max(1, int(n_tokens / 38))


def chat(prompt, maxtok=900, timeout=3600):
    body = {"messages": [{"role": "user", "content": prompt}],
            "max_tokens": maxtok, "temperature": 0,
            "chat_template_kwargs": {"thinking": False, "enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        o = json.loads(r.read())
    return (o["choices"][0].get("message") or {}).get("content") or "", time.time() - t0


def main():
    tasks = build_tasks(per_cat=10)[:N_TASKS]
    new = not os.path.exists(OUT)
    fh = open(OUT, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        w.writeheader()
    try:
        for depth in DEPTHS:
            pad = pad_for(depth)
            ok_n = 0
            for cat, idx, q, a in tasks:
                prompt = (f"Below is a long excerpt from a warehouse ledger. It is background only; "
                          f"the question that follows it is what matters.\n\n{pad}\n\n"
                          + PROMPT_TMPL.format(question=q))
                try:
                    txt, dt = chat(prompt)
                except Exception as e:
                    print(f"  {depth} {cat}{idx} ERR {e}", flush=True)
                    continue
                ok, got = grade(txt, a)
                ok_n += int(ok)
                w.writerow(dict(model=LABEL, ctx_target=depth, task=f"{cat}{idx}",
                                expected=a, correct=int(ok), got=got, latency_s=round(dt, 2)))
                fh.flush()
                print(f"  ctx={depth:<6} {cat}{idx:<3} {'OK ' if ok else 'MISS'} exp={a:<5} got={got[:30]}",
                      flush=True)
            print(f"  -> depth {depth}: {ok_n}/{len(tasks)}", flush=True)
    finally:
        fh.close()


if __name__ == "__main__":
    main()
