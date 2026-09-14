#!/bin/bash
# Qwen3.8-27B-VL (UD-IQ2_XXS + mmproj-Q8_0) on CMP 90HX 10GB
# 用法: bash /data3/gguf/run_llama_server.sh [额外参数]
#   SPEC=off      关闭 ngram-mod 投机解码（默认 on）
#   PORT=8081     换端口（默认 8080）
set -euo pipefail
BIN=/home/a3165/llama.cpp/build/bin
M=/data3/gguf/Qwen3.8-27B-UD-IQ2_XXS.gguf
MM=/data3/gguf/mmproj-Q8_0.gguf
SPEC=${SPEC:-on}
PORT=${PORT:-8080}

SPEC_ARGS=()
[ "$SPEC" = "on" ] && SPEC_ARGS=(--spec-type ngram-mod)

exec $BIN/llama-server \
  -m "$M" --mmproj "$MM" \
  --alias qwen3.8-27b \
  -ngl 99 \
  -c 32768 -ctk q8_0 -ctv q8_0 \
  -b 256 -ub 256 \
  --host 0.0.0.0 --port "$PORT" \
  --parallel 1 \
  --jinja \
  "${SPEC_ARGS[@]}" \
  "$@"
