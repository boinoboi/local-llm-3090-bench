#!/usr/bin/env bash
# run_mk.sh <label> <ctx-tokens> <drafts> <kv-mode: q8|presetD|presetF> <ctx-list>
set -uo pipefail
MK=~/src/megakernel
LABEL=$1; CTXN=${2:-262144}; DRAFT=${3:-7}; KVMODE=${4:-q8}
CTX=${5:-1024,8192,32768,65536,131072,262144}
PORT=$((8600 + RANDOM % 200))
LOG=~/bench/mk_${LABEL}_$(date +%H%M%S).log

for p in $(seq 8600 8800); do fuser -k "$p/tcp" 2>/dev/null || true; done
sleep 2

export MK_EXACT=1
export CUDA_VISIBLE_DEVICES=0
export MK_PF_CAP=${MK_PF_CAP:-1024}
case "$KVMODE" in
  presetD) export MK_KV='K*=6,V*=5' MK_KV_WIN=2048 ;;
  presetF) export MK_KV='K*=5r,V*=5' MK_KV_WIN=2048 ;;
  *) unset MK_KV MK_KV_WIN || true ;;
esac

echo ">>> $LABEL ctx=$CTXN drafts=$DRAFT kv=$KVMODE port=$PORT"
cd "$MK"
python3 server/mk_server.py -m ~/models/qwen3.8/megakernel/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --lib "$MK/libqwen38mk.so" -c "$CTXN" --kv q8 -d "$DRAFT" \
  --host 127.0.0.1 --port "$PORT" > "$LOG" 2>&1 &
SRV=$!

for i in $(seq 1 1800); do
  kill -0 $SRV 2>/dev/null || { echo "SERVER DIED"; tail -30 "$LOG"; exit 1; }
  curl -sf "127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  curl -sf "127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  sleep 1
done
sleep 3

python3 ~/bench/scripts/measure.py --port "$PORT" --label "$LABEL" --engine megakernel \
  --api chat --quant "UD-Q4_K_M" --spec "mtp${DRAFT}$([ "$KVMODE" != q8 ] && echo "-$KVMODE")" \
  --ctx "$CTX" --gpu 0 --notes "kv=$KVMODE" --max-prompt $((CTXN - 512))

kill -INT $SRV 2>/dev/null || true
wait $SRV 2>/dev/null || true
echo ">>> done $LABEL"
