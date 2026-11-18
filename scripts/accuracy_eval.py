#!/usr/bin/env python3
"""Auto-graded accuracy benchmark with programmatically computed ground truth.

Three categories, all with exact answers, so grading is objective and reproducible:

  arith  multi-step word problem   -> integer, computed
  code   predict a snippet's output -> computed by executing the snippet
  count  count a character in text  -> computed by str.count

Every prompt demands the final value on a line of the form  ANSWER: <value>
so grading is a regex, not an LLM judge.
"""
import json, os, random, re, subprocess, sys, time
import urllib.request

FIELDS = ["model", "engine", "task", "category", "correct", "got", "expected", "latency_s"]

# ---------------------------------------------------------------- generators


def gen_arith(rng):
    a = rng.randint(120, 900)
    b = rng.randint(30, 400)
    c = rng.randint(10, 150)
    ans = a + b - c + 3 * c
    q = (f"A warehouse starts with {a} crates. It receives {b} more, then ships out {c}. "
         f"Finally it receives three times as many crates as it shipped out. "
         f"How many crates does the warehouse hold now?")
    return q, str(ans)


def gen_code(rng):
    n = rng.randint(6, 14)
    k = rng.randint(2, 5)
    kind = rng.choice(["sum_sq", "count_mod", "max_pair"])
    if kind == "sum_sq":
        src = f"xs=[i*i for i in range({n}) if i % {k} != 0]\nprint(sum(xs))"
    elif kind == "count_mod":
        src = f"xs=[i for i in range({n}) if i % {k} == 0]\nprint(len(xs))"
    else:
        src = (f"xs=[i*{k} for i in range({n})]\n"
               f"print(max(xs[i]+xs[i+1] for i in range(len(xs)-1)))")
    out = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True).stdout.strip()
    q = f"What is printed by this Python program?\n\n```python\n{src}\n```"
    return q, out


WORDS = ("lantern meadow copper orchard velvet granite whisper hollow juniper "
         "cinder marble ribbon tunnel pumpkin saffron drizzle compass thicket").split()


def gen_count(rng):
    words = [rng.choice(WORDS) for _ in range(rng.randint(12, 20))]
    target = rng.choice("aeiou")
    text = " ".join(words)
    ans = str(text.count(target))
    q = (f'How many times does the letter "{target}" appear in the following text? '
         f"Count every occurrence, including inside words and any repeats.\n\n{text}")
    return q, ans


GENS = [("arith", gen_arith), ("code", gen_code), ("count", gen_count)]

PROMPT_TMPL = """{question}

Think it through, then give the final answer on its own last line in exactly this format:
ANSWER: <value>
"""

# ---------------------------------------------------------------- plumbing


def post(port, payload, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ask(port, prompt, maxtok, thinking_off=True):
    body = {"messages": [{"role": "user", "content": prompt}],
            "max_tokens": maxtok, "temperature": 0, "top_p": 1}
    if thinking_off:
        body["chat_template_kwargs"] = {"thinking": False, "enable_thinking": False}
    t0 = time.time()
    o = post(port, body)
    dt = time.time() - t0
    txt = (o["choices"][0].get("message") or {}).get("content") or ""
    return txt, dt


def grade(text, expected):
    m = re.findall(r"ANSWER:\s*([^\n]+)", text, flags=re.I)
    if not m:
        return False, "<no ANSWER line>"
    raw = m[-1].strip().strip("`*_ ").rstrip(".")
    norm = re.sub(r"[,\s]", "", raw)
    if re.fullmatch(r"-?\d+(\.0+)?", norm):
        norm = str(int(float(norm)))
    ok = norm.lower() == str(expected).strip().lower()
    return ok, raw[:60]


def build_tasks(seed=int(os.environ.get("ACC_SEED", "20261118")), per_cat=10):
    rng = random.Random(seed)
    tasks = []
    for cat, g in GENS:
        for i in range(per_cat):
            q, a = g(rng)
            tasks.append((cat, i, q, a))
    return tasks


def main():
    port = int(sys.argv[1])
    label = sys.argv[2]
    engine = sys.argv[3] if len(sys.argv) > 3 else "llama.cpp"
    out = os.path.expanduser("~/bench/data/results_accuracy.csv")
    label = label + ("" if os.environ.get("ACC_SEED") in (None, "20261118") else f"@s{os.environ.get('ACC_SEED')}")
    maxtok = int(os.environ.get("ACC_MAXTOK", "700"))
    thinking_off = os.environ.get("ACC_THINKING_OFF", "1") == "1"

    tasks = build_tasks()
    new = not os.path.exists(out)
    import csv
    fh = open(out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        w.writeheader()

    bycat = {}
    for cat, idx, q, a in tasks:
        prompt = PROMPT_TMPL.format(question=q)
        try:
            txt, dt = ask(port, prompt, maxtok, thinking_off)
        except Exception as e:
            print(f"  {cat}#{idx} ERROR {e}", flush=True)
            continue
        ok, got = grade(txt, a)
        w.writerow(dict(model=label, engine=engine, task=f"{cat}{idx}", category=cat,
                        correct=int(ok), got=got, expected=a, latency_s=round(dt, 2)))
        fh.flush()
        bycat.setdefault(cat, []).append(int(ok))
        print(f"  {cat}{idx:<3} {'OK ' if ok else 'MISS'} expected={a:<6} got={got[:40]}", flush=True)
    fh.close()

    print()
    total_c = sum(sum(v) for v in bycat.values())
    total_n = sum(len(v) for v in bycat.values())
    for cat in sorted(bycat):
        print(f"  {cat:<8} {sum(bycat[cat])}/{len(bycat[cat])}")
    print(f"  {'TOTAL':<8} {total_c}/{total_n} = {100*total_c/max(1,total_n):.1f}%  [{label}]")


if __name__ == "__main__":
    main()
