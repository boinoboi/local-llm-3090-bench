#!/usr/bin/env bash
# run_ninfer.sh <label> <artifact.ninfer> <spec> <extra-flags> [max-ctx] [ctx-list]
set -uo pipefail
BIN=~/src/ninfer/build/apps
LABEL=$1; MODEL=$2; SPEC=${3:-none}; EXTRA=${4:-}; MAXCTX=${5:-262144}
CTX=${6:-1024,4096,8192,16384,32768,65536,131072,262144}
PORT=$((8500 + RANDOM % 200))
LOG=~/bench/ninfer_${LABEL}_$(date +%H%M%S).log

{ pgrep -f 'ninfer[-]serve' | xargs -r kill 2>/dev/null || true; }; sleep 2

echo ">>> $LABEL  spec=$SPEC  ctxmax=$MAXCTX  port=$PORT"
$BIN/ninfer-serve "$MODEL" \
  --host 127.0.0.1 --port "$PORT" --device 0 \
  --max-context "$MAXCTX" --kv-capacity auto --kv-headroom-mib 1024 \
  --kv-dtype rk4v4 --prefill-chunk 1024 \
  --model-id "$LABEL" --spec "$SPEC" $EXTRA > "$LOG" 2>&1 &
SRV=$!

for i in $(seq 1 1800); do
  kill -0 $SRV 2>/dev/null || { echo "SERVER DIED"; tail -40 "$LOG"; exit 1; }
  curl -sf "127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 1
done
sleep 3

python3 ~/bench/scripts/measure.py --port "$PORT" --label "$LABEL" --engine ninfer \
  --api chat --quant "${QUANT:-IQ3_S}" --spec "$SPEC" --ctx "$CTX" --gpu 0 \
  --notes "$EXTRA" --max-prompt $((MAXCTX - 256))

kill -INT $SRV 2>/dev/null || true
wait $SRV 2>/dev/null || true
echo ">>> done $LABEL"
