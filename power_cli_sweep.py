#!/usr/bin/env python3
"""Power-limit sweep using one fresh engine process per measurement.

A fresh process per run is slower than a server but immune to the engine's
checkpoint-resume crashes, and the reported decode figure excludes load and
prefill, so it is the cleanest comparison available.

Writes ~/bench/results_power.csv
  PL_W          power limit requested
  actual_W      limit the driver reports afterwards
  run           repetition index
  decode_tok_s  engine-reported decode rate
  accept_pct    share of MTP drafts accepted
  tok_per_cycle tokens produced per speculative cycle
  draw_W_peak   peak GPU draw sampled during the run
  draw_W_med    median GPU draw during the run
"""
import csv, os, re, statistics, subprocess, threading, time

MK = os.path.expanduser("~/src/megakernel")
MODEL = os.path.expanduser("~/models/qwen3.8/megakernel/Qwen3.8-27B-UD-Q4_K_M.gguf")
OUT = os.path.expanduser("~/bench/results_power.csv")
LIMITS = [int(x) for x in os.environ.get("PL_LIMITS", "230,250,270,290,310,330,350,370,385").split(",")]
REPS = int(os.environ.get("PL_REPS", "10"))
CTX = int(os.environ.get("PL_CTX", "8192"))
NPRED = int(os.environ.get("PL_NPRED", "256"))
PROMPT = ("Write a C function that merges two sorted linked lists and explain each "
          "line of it in detail, then give the complexity of every step.")

RE_DEC = re.compile(r"decode (\d+) tok in ([0-9.]+) s = ([0-9.]+) tok/s \| cycles (\d+), accepted (\d+)/(\d+) drafts \(([0-9.]+)%\), ([0-9.]+) tok/cycle")
FIELDS = ["PL_W", "actual_W", "run", "decode_tok_s", "accept_pct", "tok_per_cycle",
          "draw_W_peak", "draw_W_med"]


def set_pl(w):
    subprocess.run(["sudo", "-n", "nvidia-smi", "-i", "0", "-pl", str(w)],
                   capture_output=True, text=True)


def actual_pl():
    o = subprocess.run(["nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader,nounits", "-i", "0"],
                       capture_output=True, text=True).stdout.strip()
    return float(o) if o else 0.0


def power_now():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits", "-i", "0"],
                           capture_output=True, text=True, timeout=5).stdout.strip()
        return float(o)
    except Exception:
        return 0.0


def run_once(pl, i):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", MK_KEEP_WARM_S="0")
    log = f"/tmp/opencode/pw_{pl}_{i}.log"
    with open(log, "w") as fh:
        p = subprocess.Popen(
            [f"{MK}/qwen38_mk", "-m", MODEL, "-p", PROMPT, "-n", str(NPRED), "-d", "7",
             "--kv", "q8", "-c", str(CTX), "--draft-vocab", f"{MK}/draft_vocab_32k.bin", "--raw"],
            cwd=MK, env=env, stdout=fh, stderr=subprocess.STDOUT)
        draws, stop = [], threading.Event()

        def samp():
            while not stop.is_set():
                draws.append(power_now())
                time.sleep(0.15)

        th = threading.Thread(target=samp); th.start()
        rc = p.wait(timeout=1800)
        stop.set(); th.join()
    txt = open(log, errors="replace").read()
    m = RE_DEC.search(txt)
    if not m:
        return None
    peak = max(draws) if draws else 0
    med = statistics.median(draws) if draws else 0
    return dict(decode_tok_s=float(m.group(3)), accept_pct=float(m.group(7)),
                tok_per_cycle=float(m.group(8)), draw_W_peak=peak, draw_W_med=med, rc=rc)


def main():
    new = not os.path.exists(OUT)
    fh = open(OUT, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        w.writeheader()
    try:
        for pl in LIMITS:
            set_pl(pl); time.sleep(2)
            act = actual_pl()
            got = []
            for i in range(1, REPS + 1):
                r = run_once(pl, i)
                if not r:
                    print(f"{pl}W run{i}: no decode line", flush=True)
                    continue
                row = dict(PL_W=pl, actual_W=act, run=i, **{k: r[k] for k in
                           ("decode_tok_s", "accept_pct", "tok_per_cycle", "draw_W_peak", "draw_W_med")})
                w.writerow(row); fh.flush()
                got.append(r["decode_tok_s"])
            if got:
                print(f"{pl:>4} W | median {statistics.median(got):7.2f} tok/s | "
                      f"min {min(got):.1f} max {max(got):.1f} | n={len(got)}", flush=True)
    finally:
        fh.close()
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
