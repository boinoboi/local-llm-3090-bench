#!/usr/bin/env python3
"""Engine-agnostic benchmark client for OpenAI-compatible servers.

Speaks both /v1/completions and /v1/chat/completions, so llama.cpp and NInfer
are measured by identical code. Streams everything so TTFT is exact, derives
rates from wall time + usage (works whether or not the engine reports
llama.cpp-style `timings`), and calibrates prompt length against the server's
real tokenizer instead of guessing.

Appends one CSV row per measurement; a crash never loses collected data.
"""
import argparse, csv, json, os, random, subprocess, time
import urllib.request

FIELDS = ["ts","engine","model","quant","ctx_target","phase","spec","prompt_tokens",
          "cached_tokens","prefill_tok_s","decode_tok_s","ttft_s","wall_s",
          "vram_mib","power_w","power_limit_w","notes"]

WORDS = ("amber basalt cinder dapple ember fathom gantry hollow ingot juniper "
         "kestrel lantern marrow nectar obelisk plinth quiver riven sable "
         "tallow umbra vellum warden xenon yarrow zephyr anvil brook cipher "
         "dune fennel garnet haft iris jetty kelp loom mica nickel onyx").split()

OCR = ("The ledger for the northern depot was signed on a grey morning in the "
       "fourth week of autumn. Each entry listed a crate, a weight, and a name. "
       "The clerk counted twice, then once more, and wrote the total at the foot "
       "of the page in a hand that leaned slightly to the right. Rain began at "
       "dusk and did not stop until the following Thursday, by which time the "
       "road had turned to a river of clay and the carts had to wait.")


def gpu(gpu_id):
    try:
        o = subprocess.run(["nvidia-smi", f"--id={gpu_id}",
                            "--query-gpu=memory.used,power.draw,power.limit",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10).stdout.strip()
        m, d, l = [x.strip() for x in o.split(",")]
        return float(m), float(d), float(l)
    except Exception:
        return None, None, None


def _post(port, path, payload, timeout):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    return urllib.request.urlopen(req, timeout=timeout)


def count_tokens(port, text):
    """Exact token count from the server's tokenizer, if it exposes one."""
    try:
        with _post(port, "/tokenize", {"content": text}, 120) as r:
            o = json.loads(r.read())
        t = o.get("tokens")
        if isinstance(t, list):
            return len(t)
        if isinstance(o.get("length"), int):
            return o["length"]
    except Exception:
        pass
    return None


def probe_ratio(port, api):
    """Fallback: one tiny request tells us tokens-per-word for this tokenizer."""
    sample = " ".join(WORDS * 8)
    try:
        if api == "chat":
            body = {"messages": [{"role": "user", "content": sample}],
                    "max_tokens": 1, "temperature": 0, "stream": False}
            path = "/v1/chat/completions"
        else:
            body = {"prompt": sample, "n_predict": 1, "temperature": 0, "stream": False}
            path = "/v1/completions"
        with _post(port, path, body, 600) as r:
            o = json.loads(r.read())
        n = (o.get("usage") or {}).get("prompt_tokens")
        if n:
            return n / len(WORDS), n
    except Exception as e:
        print(f"  ratio probe failed: {e}", flush=True)
    return 1.6, None


def build_prompt(port, target, seed, api):
    """Grow word count until the real tokenizer agrees we are at `target` tokens."""
    rng = random.Random(seed)
    per_word = None

    def gen(words):
        return " ".join(rng.choice(WORDS) for _ in range(int(words) + 8))

    if count_tokens(port, "amber basalt cinder") is not None:
        words = max(8, int(target / 1.5))
        text = gen(words)
        for _ in range(4):
            got = count_tokens(port, text)
            if got and abs(got - target) <= max(64, target * 0.02):
                return text, got
            if not got:
                break
            words = max(8, int(words * target / got))
            text = gen(words)
        return text, count_tokens(port, text)

    if per_word is None:
        per_word, _ = probe_ratio(port, api)
    text = gen(max(8, int(target / per_word)))
    got = None
    try:
        if api == "chat":
            body = {"messages": [{"role": "user", "content": text}],
                    "max_tokens": 1, "temperature": 0, "stream": False}
            path = "/v1/chat/completions"
        else:
            body = {"prompt": text, "n_predict": 1, "temperature": 0, "stream": False}
            path = "/v1/completions"
        with _post(port, path, body, 1800) as r:
            got = (json.loads(r.read()).get("usage") or {}).get("prompt_tokens")
    except Exception:
        pass
    return text, got


def stream_once(port, text, n_predict, api, timeout=7200):
    if api == "chat":
        body = {"messages": [{"role": "user", "content": text}], "max_tokens": n_predict,
                "temperature": 0, "stream": True,
                "stream_options": {"include_usage": True}}
        path = "/v1/chat/completions"
    else:
        body = {"prompt": text, "n_predict": n_predict, "temperature": 0,
                "cache_prompt": True, "stream": True,
                "stream_options": {"include_usage": True}}
        path = "/v1/completions"
    t0 = time.time()
    ttft = None
    ntok = 0
    timings, usage = {}, {}
    with _post(port, path, body, timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            timings = obj.get("timings") or timings
            usage = obj.get("usage") or usage
            for ch in obj.get("choices") or []:
                piece = ch.get("text")
                if piece is None:
                    d = ch.get("delta") or {}
                    piece = d.get("content")
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    ntok += 1
    return dict(ttft=ttft, wall=time.time() - t0, ntok=ntok,
                timings=timings, usage=usage)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--engine", default="llama.cpp")
    ap.add_argument("--api", default="completions", choices=["completions", "chat"])
    ap.add_argument("--quant", default="")
    ap.add_argument("--spec", default="off")
    ap.add_argument("--ctx", default="1024,4096,8192,16384,32768,65536,131072,262144")
    ap.add_argument("--n-predict", type=int, default=96)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--csv", default=os.path.expanduser("~/bench/data/results.csv"))
    ap.add_argument("--notes", default="")
    ap.add_argument("--max-prompt", type=int, default=0)
    ap.add_argument("--reps", type=int, default=1, help="cold repeats for decode variance")
    a = ap.parse_args()

    new = not os.path.exists(a.csv)
    fh = open(a.csv, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        w.writeheader()
    try:
        for ctx in [int(x) for x in a.ctx.split(",")]:
            tgt = min(ctx, a.max_prompt) if a.max_prompt else ctx
            seed = random.randrange(1 << 30)
            p, exact = build_prompt(a.port, tgt, seed, a.api)
            print(f"  target={ctx} tokenized={exact}", flush=True)
            phases = [("cold", p), ("warm", p)]
            for i in range(max(0, a.reps - 1)):
                phases.append((f"cold{i+2}", p))
            for phase, text in phases:
                try:
                    r = stream_once(a.port, text, a.n_predict, a.api)
                except Exception as e:
                    print(f"  {ctx} {phase} ERROR {e}", flush=True)
                    continue
                t = r["timings"]
                pn = t.get("prompt_n") or r["usage"].get("prompt_tokens") or 0
                cn = t.get("cache_n") or 0
                decode = r["ntok"] / max(1e-6, r["wall"] - (r["ttft"] or 0))
                if t.get("predicted_per_second"):
                    decode = t["predicted_per_second"]
                if t.get("prompt_per_second"):
                    prefill = t["prompt_per_second"]
                elif r["ttft"]:
                    prefill = (pn or tgt) / r["ttft"]
                else:
                    prefill = 0
                mem, draw, lim = gpu(a.gpu)
                row = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), engine=a.engine,
                           model=a.label, quant=a.quant, ctx_target=ctx, phase=phase,
                           spec=a.spec, prompt_tokens=pn, cached_tokens=cn,
                           prefill_tok_s=round(prefill, 1), decode_tok_s=round(decode, 1),
                           ttft_s=round(r["ttft"] or 0, 3), wall_s=round(r["wall"], 3),
                           vram_mib=mem, power_w=draw, power_limit_w=lim, notes=a.notes)
                w.writerow(row); fh.flush()
                print(f"  ctx={ctx:<7}{phase:<8} prompt={pn:<7}cached={cn:<7}"
                      f"prefill={row['prefill_tok_s']:<8}decode={row['decode_tok_s']:<7}"
                      f"ttft={row['ttft_s']:<7}vram={mem}", flush=True)
    finally:
        fh.close()


if __name__ == "__main__":
    main()
