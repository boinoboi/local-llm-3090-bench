#!/usr/bin/env python3
"""Measure energy per output token properly: integrate power over the decode window only.

The earlier sweep divided by *peak* draw, which tracks the power limit by construction and
therefore makes efficiency look like speed/limit. Here we sample board power at 10 Hz and keep
only the samples inside the engine's own reported decode interval, so J/token is real energy.

Writes results_energy.csv
"""
import csv, os, re, statistics, subprocess, threading, time

MK = os.path.expanduser("~/src/megakernel")
MODEL = os.path.expanduser("~/models/qwen3.8/megakernel/Qwen3.8-27B-UD-Q4_K_M.gguf")
OUT = os.path.expanduser("~/bench/results_energy.csv")
LIMITS = [int(x) for x in os.environ.get("EN_LIMITS", "230,250,270,300,330,350,385").split(",")]
REPS = int(os.environ.get("EN_REPS", "3"))
CTX = 8192
NPRED = 512
PROMPT = ("Write a detailed technical explanation of how a modern operating system scheduler "
          "decides which process to run next, covering scheduling classes, priority inheritance, "
          "load balancing across cores, and the trade-off between latency and throughput.")

RE_DEC = re.compile(r"decode (\d+) tok in ([0-9.]+) s = ([0-9.]+) tok/s")
FIELDS = ["PL_W", "rep", "tokens", "decode_s", "decode_tok_s", "mean_decode_W", "J_per_token", "n_samples"]


def set_pl(w):
    subprocess.run(["sudo", "-n", "nvidia-smi", "-i", "0", "-pl", str(w)],
                   capture_output=True, text=True)


def pwr():
    try:
        return float(subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=4).stdout.strip())
    except Exception:
        return None


def main():
    new = not os.path.exists(OUT)
    fh = open(OUT, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        w.writeheader()
    try:
        for pl in LIMITS:
            set_pl(pl)
            time.sleep(6)          # let the ceiling settle; -pl often lags
            vals = []
            for rep in range(1, REPS + 1):
                log = f"/tmp/opencode/en_{pl}_{rep}.log"
                samples = []
                stop = threading.Event()

                def samp():
                    while not stop.is_set():
                        v = pwr()
                        if v:
                            samples.append((time.time(), v))
                        time.sleep(0.1)

                th = threading.Thread(target=samp)
                th.start()
                t_end = None
                with open(log, "w") as f:
                    p = subprocess.Popen(
                        [f"{MK}/qwen38_mk", "-m", MODEL, "-p", PROMPT, "-n", str(NPRED), "-d", "7",
                         "--kv", "q8", "-c", str(CTX), "--draft-vocab", f"{MK}/draft_vocab_32k.bin",
                         "--raw"], cwd=MK,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES="0", MK_KEEP_WARM_S="0"),
                        stdout=f, stderr=subprocess.STDOUT)
                    p.wait()
                    t_end = time.time()
                stop.set(); th.join()

                txt = open(log, errors="replace").read()
                m = RE_DEC.search(txt)
                if not m:
                    print(f"  {pl} rep{rep}: no decode line", flush=True)
                    continue
                ntok, dsec, tps = int(m.group(1)), float(m.group(2)), float(m.group(3))
                # keep samples inside the decode interval
                window = [v for t, v in samples if t >= t_end - dsec - 0.15]
                if len(window) < 5:
                    print(f"  {pl} rep{rep}: too few samples ({len(window)})", flush=True)
                    continue
                meanw = statistics.mean(window)
                jtok = meanw * dsec / ntok
                row = dict(PL_W=pl, rep=rep, tokens=ntok, decode_s=round(dsec, 3),
                           decode_tok_s=round(tps, 2), mean_decode_W=round(meanw, 1),
                           J_per_token=round(jtok, 4), n_samples=len(window))
                w.writerow(row); fh.flush()
                vals.append(jtok)
                print(f"  PL={pl:>3}W rep{rep}  {tps:7.2f} tok/s  {meanw:5.1f} W  "
                      f"{jtok:6.4f} J/tok", flush=True)
            if vals:
                print(f"  -> {pl} W median {statistics.median(vals):.4f} J/token (n={len(vals)})", flush=True)
    finally:
        fh.close()
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
