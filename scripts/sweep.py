#!/usr/bin/env python3
"""Context-fill sweep against an OpenAI-compatible llama.cpp server.

For each target context size it measures, on this exact machine:
  * cold prefill   -> unique filler prefix, defeats the prefix cache
  * warm prefill   -> identical repeat, measures prefix-cache hit
  * decode         -> predicted tokens/sec after the prompt
  * TTFT, wall, VRAM peak, power draw

Appends one row per measurement to results.csv so a crash never loses data.
"""
import argparse, csv, json, os, random, signal, socket, subprocess, sys, time
import urllib.request, urllib.error

CSV_FIELDS = [
    "ts", "engine", "model", "quant", "ctx_target", "phase", "spec",
    "prompt_tokens", "cached_tokens", "prefill_tok_s", "decode_tok_s",
    "ttft_s", "wall_s", "vram_mib", "power_w", "power_limit_w", "notes",
]

WORDS = ("amber basalt cinder dapple ember fathom gantry hollow ingot juniper "
         "kestrel lantern marrow nectar obelisk plinth quiver riven sable "
         "tallow umbra vellum warden xenon yarrow zephyr anvil brook cipher "
         "dune fennel garnet haft iris jetty kelp loom mica nickel onyx").split()


def make_prompt(n_tokens: int, seed: int) -> str:
    """Unique filler text. ~1.35 tokens/word for this vocab; overshoot then trim by words."""
    rng = random.Random(seed)
    n_words = int(n_tokens * 1.0) + 8
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def wait_port(port, timeout=1800, proc=None):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server died early rc={proc.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), 1):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"port {port} never opened")


def post(port, path, payload, timeout=7200):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def gpu_stats(gpu):
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={gpu}",
             "--query-gpu=memory.used,power.draw,power.limit",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        mem, draw, lim = [x.strip() for x in out.split(",")]
        return float(mem), float(draw), float(lim)
    except Exception:
        return None, None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--model-label", required=True)
    p.add_argument("--quant", default="")
    p.add_argument("--engine", default="llama.cpp")
    p.add_argument("--extra", default="", help="extra server flags, space separated")
    p.add_argument("--spec", default="off")
    p.add_argument("--port", type=int, default=8199)
    p.add_argument("--gpu", default="0")
    p.add_argument("--ctx", default="1024,4096,8192,16384,32768,65536,131072,262144")
    p.add_argument("--n-predict", type=int, default=128)
    p.add_argument("--max-ctx", type=int, default=262144)
    p.add_argument("--csv", default=os.path.expanduser("~/bench/data/results.csv"))
    p.add_argument("--kv-type", default="q4_0")
    p.add_argument("--notes", default="")
    args = p.parse_args()

    flags = [args.server, "-m", args.model, "-ngl", "99", "-c", str(args.max_ctx),
             "--host", "127.0.0.1", "--port", str(args.port), "-np", "1",
             "--cache-type-k", args.kv_type, "--cache-type-v", args.kv_type]
    if "mmproj" not in args.extra and args.extra:
        pass
    flags += [f for f in args.extra.split() if f]

    log = open(os.path.expanduser(f"~/bench/server_{args.model_label}.log"), "w")
    print("launch:", " ".join(flags), flush=True)
    srv = subprocess.Popen(flags, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_port(args.port, proc=srv)
        time.sleep(2)
        new = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            if new:
                w.writeheader()
            for ctx in [int(x) for x in args.ctx.split(",")]:
                for phase, seed in (("cold", random.randrange(1 << 30)), ("warm", None)):
                    if phase == "warm":
                        seed = last_seed
                    else:
                        last_seed = seed
                    prompt = make_prompt(ctx, seed)
                    t0 = time.time()
                    try:
                        r = post(args.port, "/v1/completions", {
                            "prompt": prompt,
                            "n_predict": args.n_predict,
                            "temperature": 0,
                            "cache_prompt": True,
                            "stream": False,
                        })
                    except Exception as e:
                        print(f"  {ctx} {phase} ERROR {e}", flush=True)
                        break
                    wall = time.time() - t0
                    t = r.get("timings", {}) or {}
                    pn = t.get("prompt_n", 0) or 0
                    cn = t.get("cache_n", 0) or 0
                    pred_ms = t.get("predicted_ms", 0) or 0
                    ttft = max(0.0, wall - pred_ms / 1000.0)
                    mem, draw, lim = gpu_stats(args.gpu)
                    row = {
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "engine": args.engine, "model": args.model_label,
                        "quant": args.quant, "ctx_target": ctx, "phase": phase,
                        "spec": args.spec, "prompt_tokens": pn, "cached_tokens": cn,
                        "prefill_tok_s": round(t.get("prompt_per_second", 0) or 0, 1),
                        "decode_tok_s": round(t.get("predicted_per_second", 0) or 0, 1),
                        "ttft_s": round(ttft, 3), "wall_s": round(wall, 3),
                        "vram_mib": mem, "power_w": draw, "power_limit_w": lim,
                        "notes": args.notes,
                    }
                    w.writerow(row); fh.flush()
                    print(f"  ctx={ctx:<7} {phase:<5} prompt={pn:<7} cached={cn:<7} "
                          f"prefill={row['prefill_tok_s']:<8} decode={row['decode_tok_s']:<7} "
                          f"ttft={row['ttft_s']:<7} vram={mem}", flush=True)
    finally:
        srv.send_signal(signal.SIGINT)
        try:
            srv.wait(timeout=60)
        except subprocess.TimeoutExpired:
            srv.kill()
        log.close()


if __name__ == "__main__":
    main()
