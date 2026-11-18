#!/usr/bin/env python3
"""Parse megakernel server/CLI logs into CSV rows.

The engine reports its own timings, which are more trustworthy than wall-clock
arithmetic from the client: prefill and decode are measured inside the engine,
and draft acceptance is reported directly.
"""
import csv, glob, os, re, sys

FIELDS = ["ts","engine","model","quant","ctx_target","phase","spec","prompt_tokens",
          "cached_tokens","prefill_tok_s","decode_tok_s","ttft_s","wall_s",
          "vram_mib","power_w","power_limit_w","notes","draft_accept_pct","tok_per_cycle"]

RE_PROMPT = re.compile(r"#(\d+) prompt: ([\d,]+) cached \+ ([\d,]+) prefilled in ([0-9.]+) ?(?:s|min) .*?\(([\d,]+) tok/s\)")
RE_PROMPT_MIN = re.compile(r"#(\d+) prompt: ([\d,]+) cached \+ ([\d,]+) prefilled in (.*?) \(([\d,]+) tok/s\)")
RE_REPLY = re.compile(r"#(\d+) reply: (\d+) tok in ([0-9.]+) s = ([0-9.]+) tok/s.*?([0-9.]+) tok/cycle, (\d+)% of (\d+) drafts accepted")
RE_CTX = re.compile(r"ctx (\d+), (\d+) drafts")
RE_KV = re.compile(r"KV cache ([^,]+), ([0-9.]+) GB for (\d+) positions")
RE_CLI_PROMPT = re.compile(r"\[mk\] prompt (\d+) tok in ([0-9.]+) s \(([0-9.]+) tok/s\)")
RE_CLI_DECODE = re.compile(r"\[mk\] decode (\d+) tok in ([0-9.]+) s = ([0-9.]+) tok/s \| cycles (\d+), accepted (\d+)/(\d+) drafts \(([0-9.]+)%\), ([0-9.]+) tok/cycle")
RE_LOAD = re.compile(r"\[mk\] load ([0-9.]+) s")


def secs(txt):
    txt = txt.strip()
    m = re.match(r"(?:(\d+) min )?([0-9.]+) s", txt)
    if m:
        return (int(m.group(1) or 0) * 60) + float(m.group(2))
    return 0.0


def parse(path):
    rows = []
    txt = open(path, errors="replace").read()
    name = os.path.basename(path)
    kv = ""
    m = RE_KV.search(txt)
    if m:
        kv = m.group(1).strip()
    ctx = ""
    m = RE_CTX.search(txt)
    if m:
        ctx = m.group(1)

    # server-style
    prompts = {}
    for line in txt.splitlines():
        m = RE_PROMPT_MIN.search(line)
        if m:
            prompts[int(m.group(1))] = (int(m.group(2).replace(",", "")),
                                        int(m.group(3).replace(",", "")),
                                        secs(m.group(4)),
                                        float(m.group(5).replace(",", "")))
        m = RE_REPLY.search(line)
        if m:
            rid = int(m.group(1))
            pc, pp, psec, pps = prompts.get(rid, (0, 0, 0, 0))
            rows.append(dict(engine="megakernel", model=name.split("_", 2)[-1].rsplit("_", 1)[0],
                             quant="UD-Q4_K_M", ctx_target=ctx or "", phase=f"req{rid}",
                             spec=f"mtp{kv}", prompt_tokens=pc + pp, cached_tokens=pc,
                             prefill_tok_s=round(pps, 1), decode_tok_s=float(m.group(4)),
                             ttft_s=round(psec, 2), wall_s=round(float(m.group(3)), 3),
                             vram_mib="", power_w="", power_limit_w="", notes=f"kv={kv}",
                             draft_accept_pct=m.group(6), tok_per_cycle=m.group(5)))

    # CLI-style
    m = RE_CLI_PROMPT.search(txt)
    d = RE_CLI_DECODE.search(txt)
    if m and d:
        rows.append(dict(engine="megakernel-cli", model=name, quant="UD-Q4_K_M",
                         ctx_target=ctx or "", phase="cli", spec=f"mtp{kv}",
                         prompt_tokens=m.group(1), cached_tokens=0,
                         prefill_tok_s=float(m.group(3)), decode_tok_s=float(d.group(3)),
                         ttft_s=float(m.group(2)), wall_s="", vram_mib="",
                         power_w="", power_limit_w="", notes=f"kv={kv}",
                         draft_accept_pct=d.group(7), tok_per_cycle=d.group(8)))
    return rows


def main():
    out = os.path.expanduser("~/bench/results_mk.csv")
    rows = []
    for p in sorted(glob.glob(os.path.expanduser("~/bench/mk_*.log"))):
        try:
            rows.extend(parse(p))
        except Exception as e:
            print(f"skip {p}: {e}", file=sys.stderr)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
