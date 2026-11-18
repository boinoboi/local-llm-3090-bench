#!/usr/bin/env bash
# 27B Turbo IQ2_XS: spec-off vs MTP vs DFlash2 vs both, across context fill.
set -uo pipefail
cd ~/bench
M=~/models/qwen3.8/27B_Turbo_50X_IQ2XS_mtp.gguf
D=~/models/qwen3.8/dflash2/Qwen3.8-27B-DFlash2-Q8_0.gguf
CTX=1024,8192,32768,65536,131072,262144
export CTX

for p in $(seq 8300 8500); do fuser -k "$p/tcp" 2>/dev/null || true; done
sleep 2

echo "================ 27B_Turbo_IQ2XS_nospec ================"
./run_llama.sh "27B_Turbo_IQ2XS_nospec" "$M" "none" "" "IQ2_XS" 262144

echo "================ 27B_Turbo_IQ2XS_mtp ================"
./run_llama.sh "27B_Turbo_IQ2XS_mtp" "$M" "draft-mtp" "" "IQ2_XS" 262144

echo "================ 27B_Turbo_IQ2XS_dflash2 ================"
./run_llama.sh "27B_Turbo_IQ2XS_dflash2" "$M" "draft-dflash2" "-md $D" "IQ2_XS" 262144

echo "================ 27B_Turbo_IQ2XS_dflash2_mtp ================"
./run_llama.sh "27B_Turbo_IQ2XS_dflash2_mtp" "$M" "draft-dflash2,draft-mtp" "-md $D" "IQ2_XS" 262144

echo "ALL DONE 27B"
