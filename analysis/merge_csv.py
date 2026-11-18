#!/usr/bin/env python3
"""Merge every measurement taken in this session into one normalised CSV."""
import csv, os, glob

HOME = os.path.expanduser("~")
OUT = f"{HOME}/bench/data/data/results_all.csv"

FIELDS = ["source", "engine", "model", "quant", "kv", "spec", "ctx_target", "phase",
          "prompt_tokens", "cached_tokens", "prefill_tok_s", "decode_tok_s",
          "ttft_s", "wall_s", "vram_mib", "power_w", "power_limit_w",
          "draft_accept_pct", "tok_per_cycle", "notes"]

SOURCES = [
    (f"{HOME}/bench/data/results_llama.csv", "llama.cpp"),
    (f"{HOME}/bench/data/results_mk.csv", "megakernel"),
]


def norm_llama(r):
    return dict(source="llama.cpp", engine=r.get("engine", "llama.cpp"),
                model=r.get("model", ""), quant=r.get("quant", ""),
                kv="q4_0", spec=r.get("spec", ""),
                ctx_target=r.get("ctx_target", ""), phase=r.get("phase", ""),
                prompt_tokens=r.get("prompt_tokens", ""),
                cached_tokens=r.get("cached_tokens", ""),
                prefill_tok_s=r.get("prefill_tok_s", ""),
                decode_tok_s=r.get("decode_tok_s", ""),
                ttft_s=r.get("ttft_s", ""), wall_s=r.get("wall_s", ""),
                vram_mib=r.get("vram_mib", ""), power_w=r.get("power_w", ""),
                power_limit_w=r.get("power_limit_w", ""),
                draft_accept_pct=r.get("draft_accept_pct", ""),
                tok_per_cycle=r.get("tok_per_cycle", ""),
                notes=r.get("notes", ""))


def norm_mk(r):
    return dict(source="megakernel", engine=r.get("engine", ""),
                model=r.get("model", ""), quant=r.get("quant", ""),
                kv="presetF" if "q4" in (r.get("notes") or "") else "q8",
                spec=r.get("spec", ""),
                ctx_target=r.get("ctx_target", ""), phase=r.get("phase", ""),
                prompt_tokens=r.get("prompt_tokens", ""),
                cached_tokens=r.get("cached_tokens", ""),
                prefill_tok_s=r.get("prefill_tok_s", ""),
                decode_tok_s=r.get("decode_tok_s", ""),
                ttft_s=r.get("ttft_s", ""), wall_s=r.get("wall_s", ""),
                vram_mib=r.get("vram_mib", ""), power_w=r.get("power_w", ""),
                power_limit_w=r.get("power_limit_w", ""),
                draft_accept_pct=r.get("draft_accept_pct", ""),
                tok_per_cycle=r.get("tok_per_cycle", ""),
                notes=r.get("notes", ""))


def main():
    rows = []
    for path, kind in SOURCES:
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            for r in csv.DictReader(fh):
                rows.append(norm_llama(r) if kind == "llama.cpp" else norm_mk(r))

    pw = f"{HOME}/bench/data/results_power.csv"
    if os.path.exists(pw):
        with open(pw) as fh:
            for r in csv.DictReader(fh):
                rows.append(dict(source="power-sweep", engine="megakernel",
                                 model="27B_UD-Q4_K_M", quant="UD-Q4_K_M", kv="q8",
                                 spec="mtp7", ctx_target="8192", phase=f"run{r.get('run','')}",
                                 prompt_tokens="", cached_tokens="",
                                 prefill_tok_s="", decode_tok_s=r.get("decode_tok_s", ""),
                                 ttft_s="", wall_s="", vram_mib="",
                                 power_w=r.get("draw_W_med", ""),
                                 power_limit_w=r.get("actual_W", ""),
                                 draft_accept_pct=r.get("accept_pct", ""),
                                 tok_per_cycle=r.get("tok_per_cycle", ""),
                                 notes=f"PL={r.get('PL_W','')} peak={r.get('draw_W_peak','')}"))

    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"{len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
