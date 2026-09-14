#!/bin/bash
# Qwen3.8-27B-VL (UD-IQ2_XXS + mmproj-Q8_0) on CMP 90HX 10GB
# 用法: bash /data3/gguf/run_llama_server.sh [额外参数]
#   SPEC=off 可关闭 ngram-mod 投机解码
set -euo pipefail
BIN=/home/a3165/llama.cpp/build/bin
M=/data3/gguf/Qwen3.8-27B-UD-IQ2_XXS.gguf
MM=/data3/gguf/mmproj-Q8_0.gguf
SPEC=${SPEC:-on}

SPEC_ARGS=()
[ "$SPEC" = "on" ] && SPEC_ARGS=(--spec-type ngram-mod)

exec $BIN/llama-server \
  -m "$M" --mmproj "$MM" \
  --alias qwen3.8-27b \
  -ngl 99 \
  -c 32768 -ctk q8_0 -ctv q8_0 \
  -b 256 -ub 256 \
  --host 0.0.0.0 --port 8080 \
  --parallel 1 \
  --jinja \
  "${SPEC_ARGS[@]}" \
  "$@"
