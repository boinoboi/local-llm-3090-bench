"""SUPERSEDED by ../power_cli_sweep.py. This is the original sweep that produced the first
91 rows of data/results_power.csv. It divides by peak draw and does not validate that the
power limit actually applied. Kept only so the provenance of those rows is recoverable.
"""

#!/usr/bin/env python3
"""Power-limit sweep: decode tok/s at 8k context, N runs per limit.

Efficiency-optimal point = best tok/s per watt, not simply the fastest.
The engine's own log line is the source of truth for decode rate.
"""
import json, os, re, statistics, subprocess, sys, threading, time
import urllib.request

MK = os.path.expanduser("~/src/megakernel")
MODEL = os.path.expanduser("~/models/qwen3.8/megakernel/Qwen3.8-27B-UD-Q4_K_M.gguf")
PORT = 8811
LIMITS = [int(x) for x in os.environ.get("PL_LIMITS","230,250,270,290,310,330,350,370,385").split(",")]
N_RUNS = 10
CTX = 16384
MAXTOK = 256
LOG = "/tmp/opencode/pwsweep_server.log"


def set_pl(w):
    r = subprocess.run(["sudo", "-n", "nvidia-smi", "-i", "0", "-pl", str(w)],
                       capture_output=True, text=True)
    return r.returncode == 0


def power_now():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits", "-i", "0"],
                           capture_output=True, text=True, timeout=5).stdout.strip()
        return float(o)
    except Exception:
        return 0.0


def start_server():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", MK_EXACT="0", MK_KEEP_WARM_S="0")
    logf = open(LOG, "w")
    p = subprocess.Popen(
        ["python3", f"{MK}/server/mk_server.py", "-m", MODEL, "--lib", f"{MK}/libqwen38mk.so",
         "-c", str(CTX), "--kv", "q8", "-d", "7", "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=MK, env=env, stdout=logf, stderr=subprocess.STDOUT)
    for _ in range(600):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2).read()
            return p
        except Exception:
            if p.poll() is not None:
                sys.exit("server died:\n" + open(LOG).read()[-2000:])
            time.sleep(1)
    sys.exit("server never came up")


def make_prompt(tokens_hint=8000):
    para = ("A warehouse ledger from the northern depot lists crates by weight and by "
            "the initials of the clerk who counted them. Each line ends with a running "
            "total, checked twice before the page is signed. ")
    return para * (tokens_hint // 40 + 10)


def chat(prompt, maxtok=MAXTOK):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                       "max_tokens": maxtok, "temperature": 0,
                       "chat_template_kwargs": {"thinking": False}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        r.read()
    return time.time() - t0


RE_REPLY = re.compile(r"reply: (\d+) tok in ([0-9.]+) s = ([0-9.]+) tok/s.*?([0-9.]+) tok/cycle, (\d+)%")


def parse_since(offset):
    with open(LOG, errors="replace") as f:
        f.seek(offset)
        txt = f.read()
    return txt, txt


def main():
    srv = start_server()
    prompt = make_prompt()
    out = []
    try:
        # warm the cache once so every level measures decode only
        chat(prompt, 32)
        time.sleep(1)
        for pl in LIMITS:
            if not set_pl(pl):
                print(f"!! cannot set {pl} W, skipping")
                continue
            time.sleep(2)
            actual = subprocess.run(["nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader,nounits", "-i", "0"],
                                    capture_output=True, text=True).stdout.strip()
            rates, accs, pows = [], [], []
            stop = threading.Event()

            def sampl():
                while not stop.is_set():
                    pows.append(power_now())
                    time.sleep(0.2)

            th = threading.Thread(target=sampl); th.start()
            for i in range(N_RUNS):
                off = os.path.getsize(LOG)
                try:
                    chat(prompt + f" Point {pl} sampling {i} here.")
                except Exception as e:
                    print("  err", e); continue
                txt, _ = parse_since(off)
                m = RE_REPLY.search(txt)
                if m:
                    rates.append(float(m.group(3)))
                    accs.append(float(m.group(5)))
            stop.set(); th.join()
            if not rates:
                print(f"{pl:>4} W (actual {actual}) : no data")
                continue
            med = statistics.median(rates)
            avg_p = statistics.median(pows) if pows else 0
            out.append((pl, actual, med, avg_p, statistics.mean(accs)))
            print(f"{pl:>4} W (actual {actual}) : median {med:6.1f} tok/s | "
                  f"draw {avg_p:5.0f} W | {med/avg_p*1000:6.1f} tok/s/kW | acc {statistics.mean(accs):4.1f}% | "
                  f"min {min(rates):.1f} max {max(rates):.1f}", flush=True)
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=30)
        except Exception:
            srv.kill()

    print("\nPL_W,actual_W,median_tok_s,median_draw_W,tok_s_per_kW,accept_pct")
    for r in out:
        print(f"{r[0]},{r[1]},{r[2]:.2f},{r[3]:.0f},{r[2]/r[3]*1000:.1f},{r[4]:.1f}")


if __name__ == "__main__":
    main()
