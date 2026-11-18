#!/usr/bin/env python3
"""megakernel context sweep with repetitions, reading the engine's own decode report.

Fills the weakest table in the study: §8.3 previously had single measurements and two
disagreeing 131 K points. Uses preset F so the sweep runs to the full native window.
"""
import csv, json, os, re, statistics, subprocess, sys, threading, time
import urllib.request

MK = os.path.expanduser("~/src/megakernel")
MODEL = os.path.expanduser("~/models/qwen3.8/megakernel/Qwen3.8-27B-UD-Q4_K_M.gguf")
OUT = os.path.expanduser("~/bench/data/results_mk_context.csv")
PORT = 8977
CTXS = [1024, 8192, 32768, 65536, 131072]
REPS = 3
MAXCTX = 262144
NPRED = 96
LOG = "/tmp/opencode/mkctx_server.log"

RE_REPLY = re.compile(r"#(\d+) reply: (\d+) tok in ([0-9.]+) s = ([0-9.]+) tok/s.*?([0-9.]+) tok/cycle, (\d+)% of (\d+) drafts accepted")
RE_PROMPT = re.compile(r"#(\d+) prompt: ([\d,]+) cached \+ ([\d,]+) prefilled in (.*?) \(([\d,]+) tok/s\)")

PARA = ("A warehouse ledger from the northern depot lists crates by weight and by the initials "
        "of the clerk who counted them. Each line ends with a running total, checked twice "
        "before the page is signed. ")


def start():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", MK_EXACT="0", MK_KEEP_WARM_S="0",
               MK_KV="K*=5r,V*=5", MK_KV_WIN="2048", MK_PF_CAP="1024")
    f = open(LOG, "w")
    p = subprocess.Popen(
        ["python3", f"{MK}/server/mk_server.py", "-m", MODEL, "--lib", f"{MK}/libqwen38mk.so",
         "-c", str(MAXCTX), "--kv", "q8", "-d", "7", "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=MK, env=env, stdout=f, stderr=subprocess.STDOUT)
    for _ in range(900):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2).read()
            return p
        except Exception:
            if p.poll() is not None:
                sys.exit("server died:\n" + open(LOG).read()[-1500:])
            time.sleep(1)
    sys.exit("no server")


def chat(prompt, maxtok=NPRED):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                       "max_tokens": maxtok, "temperature": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=3600) as r:
        r.read()


def make(n, nonce):
    body = PARA * (n // 40 + 8)
    return f"Read the following and answer with one word.\n\n{body}\n\n[{nonce}] Ready?"


def main():
    srv = start()
    fields = ["ctx", "rep", "prompt_tokens", "prefill_tok_s", "decode_tok_s",
              "accept_pct", "tok_per_cycle"]
    new = not os.path.exists(OUT)
    fh = open(OUT, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=fields)
    if new:
        w.writeheader()
    try:
        for ctx in CTXS:
            got = []
            for rep in range(1, REPS + 1):
                off = os.path.getsize(LOG)
                prompt = make(ctx, f"{ctx}-{rep}-{time.time()}")
                try:
                    chat(prompt)
                except Exception as e:
                    print(f"  {ctx} rep{rep} ERR {e}", flush=True)
                    continue
                with open(LOG, errors="replace") as f:
                    f.seek(off)
                    txt = f.read()
                m = RE_REPLY.search(txt)
                if not m:
                    print(f"  {ctx} rep{rep} no reply line", flush=True)
                    continue
                pm = RE_PROMPT.search(txt)
                ptok = int(pm.group(2).replace(",", "")) + int(pm.group(3).replace(",", "")) if pm else 0
                pps = float(pm.group(5).replace(",", "")) if pm else 0.0
                row = dict(ctx=ctx, rep=rep, prompt_tokens=ptok, prefill_tok_s=pps,
                           decode_tok_s=float(m.group(4)), accept_pct=float(m.group(6)),
                           tok_per_cycle=float(m.group(5)))
                w.writerow(row); fh.flush()
                got.append(row["decode_tok_s"])
                print(f"  ctx={ctx:<7} rep{rep}  {row['decode_tok_s']:7.2f} tok/s  "
                      f"accept {row['accept_pct']:.0f}%  prefill {pps:.0f}", flush=True)
            if got:
                print(f"  -> {ctx}: median {statistics.median(got):.2f} "
                      f"(n={len(got)}, spread {max(got)-min(got):.2f})", flush=True)
    finally:
        fh.close()
        srv.terminate()
        try:
            srv.wait(timeout=30)
        except Exception:
            srv.kill()


if __name__ == "__main__":
    main()
