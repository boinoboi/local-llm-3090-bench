#!/usr/bin/env bash
# launch a llama.cpp server, run the accuracy eval against it, tear down
set -uo pipefail
LABEL=$1
MODEL=$2
SPEC=${3:-draft-mtp}
EXTRA=${4:-}
BIN=~/src/llama.cpp/build/bin
PORT=$((8900 + RANDOM % 90))
LOG=/tmp/opencode/acc_${LABEL}.log

for p in $(seq 8900 8990); do fuser -k "$p/tcp" 2>/dev/null || true; done
sleep 2

"$BIN/llama-server" -m "$MODEL" -ngl 99 -dev CUDA0 -c 16384 -np 1 \
  --host 127.0.0.1 --port "$PORT" -ctk q8_0 -ctv q8_0 -fa on \
  --spec-type "$SPEC" $EXTRA > "$LOG" 2>&1 &
SRV=$!

for i in $(seq 1 600); do
  if ! kill -0 "$SRV" 2>/dev/null; then echo "SERVER DIED"; tail -20 "$LOG"; exit 1; fi
  curl -sf "127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  sleep 1
done
sleep 2

python3 ~/bench/accuracy_eval.py "$PORT" "$LABEL" llama.cpp

kill -INT "$SRV" 2>/dev/null || true
wait "$SRV" 2>/dev/null || true
