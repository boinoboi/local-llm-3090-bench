#!/usr/bin/env bash
# accuracy eval against the megakernel server (chat API)
set -uo pipefail
LABEL=$1
MK=~/src/megakernel
PORT=$((8800 + RANDOM % 90))
LOG=/tmp/opencode/accmk_${LABEL}.log

for p in $(seq 8800 8890); do fuser -k "$p/tcp" 2>/dev/null || true; done
sleep 2

export CUDA_VISIBLE_DEVICES=0 MK_KEEP_WARM_S=0 MK_EXACT=0
cd "$MK"
python3 server/mk_server.py -m ~/models/qwen3.8/megakernel/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --lib "$MK/libqwen38mk.so" -c 16384 --kv q8 -d 7 \
  --host 127.0.0.1 --port "$PORT" > "$LOG" 2>&1 &
SRV=$!

for i in $(seq 1 600); do
  if ! kill -0 "$SRV" 2>/dev/null; then echo "SERVER DIED"; tail -20 "$LOG"; exit 1; fi
  curl -sf "127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  curl -sf "127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  sleep 1
done
sleep 3

ACC_THINKING_OFF=0 python3 ~/bench/accuracy_eval.py "$PORT" "$LABEL" megakernel

kill -INT "$SRV" 2>/dev/null || true
wait "$SRV" 2>/dev/null || true
