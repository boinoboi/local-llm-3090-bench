#!/usr/bin/env bash
# run_llama.sh <label> <model.gguf> <spec> <extra-server-flags> [quant] [max-ctx] [ctx-list]
set -euo pipefail
BIN=~/src/llama.cpp/build/bin
LABEL=$1; MODEL=$2; SPEC=${3:-none}; EXTRA=${4:-}; QUANT=${5:-}; MAXCTX=${6:-262144}
CTX=${7:-1024,4096,8192,16384,32768,65536,131072,262144}
PORT=$((8300 + RANDOM % 200))
LOG=~/bench/server_${LABEL}_$(date +%H%M%S).log

{ pgrep -f 'llama[-]server' | xargs -r kill 2>/dev/null || true; }; sleep 2

echo ">>> $LABEL  spec=$SPEC  ctxmax=$MAXCTX"
# pin to GPU0 only so the 5060 Ti never enters the measurement
$BIN/llama-server -m "$MODEL" -ngl 99 -dev CUDA0 -c "$MAXCTX" -np 1 \
  --host 127.0.0.1 --port "$PORT" -ctk q4_0 -ctv q4_0 -fa on \
  --spec-type "$SPEC" $EXTRA > "$LOG" 2>&1 &
SRV=$!

for i in $(seq 1 1800); do
  kill -0 $SRV 2>/dev/null || { echo "SERVER DIED"; tail -30 "$LOG"; exit 1; }
  curl -sf "127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  sleep 1
done
sleep 3

python3 ~/bench/measure.py --port "$PORT" --label "$LABEL" --engine llama.cpp \
  --quant "$QUANT" --spec "$SPEC" --ctx "$CTX" --gpu 0 --notes "$EXTRA" \
  --max-prompt $((MAXCTX - 256)) --reps ${REPS:-1}

kill -INT $SRV 2>/dev/null || true
wait $SRV 2>/dev/null || true
echo ">>> done $LABEL"
