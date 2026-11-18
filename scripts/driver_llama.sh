#!/usr/bin/env bash
# Full llama.cpp matrix. Sequential on purpose: one GPU, one measurement at a time.
set -uo pipefail
cd ~/bench
rm -f data/results_llama.csv
R=./run_llama.sh
M36=~/models/qwen3.6
M38=~/models/qwen3.8
MM=~/models/mimo2.6

run() { # label model spec extra quant
  echo "================ $1 ================"
  CTX=1024,4096,8192,16384,32768,65536,131072,262144 \
    $R "$1" "$2" "$3" "$4" "$5" 262144 2>&1 | tee -a driver.log
}

run "35B-A3B_huihui_Q2K_nospec" "$M36/Huihui-Qwen3.6-35B-A3B-abliterated-ggml-model-Q2_K.gguf" "none" "" "Q2_K"
run "35B-A3B_huihui_Q2K_mtp"    "$M36/Huihui-Qwen3.6-35B-A3B-abliterated-ggml-model-Q2_K.gguf" "draft-mtp" "" "Q2_K"
run "35B-A3B_huihui_Q2K_ngram"  "$M36/Huihui-Qwen3.6-35B-A3B-abliterated-ggml-model-Q2_K.gguf" "ngram-simple" "" "Q2_K"
run "35B-A3B_huihui_Q2K_mtp_ngram" "$M36/Huihui-Qwen3.6-35B-A3B-abliterated-ggml-model-Q2_K.gguf" "draft-mtp,ngram-simple" "" "Q2_K"
run "35B-A3B_llmfan_heretic_Q3KM_mtp" "$M36/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-Q3_K_M.gguf" "draft-mtp" "" "Q3_K_M"
run "27B_HauhauCS_Q2KP_nospec" "$M38/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q2_K_P.gguf" "none" "" "Q2_K_P"
run "27B_HauhauCS_Q2KP_mtp"    "$M38/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q2_K_P.gguf" "draft-mtp" "" "Q2_K_P"
run "MiMo9B_heretic_IQ2M_nospec" "$MM/MiMo-V2.6-Distill-Qwen-9B-heretic.i1-IQ2_M.gguf" "none" "" "IQ2_M"
run "MiMo9B_heretic_IQ2M_ngram"  "$MM/MiMo-V2.6-Distill-Qwen-9B-heretic.i1-IQ2_M.gguf" "ngram-simple" "" "IQ2_M"

cp data/results.csv data/results_llama.csv
echo "ALL DONE"
