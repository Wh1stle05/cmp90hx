#!/bin/bash
# bench_ab.sh —— 同一张卡上做「baseline vs ngram-mod 投机解码」A/B，跑完自动恢复 systemd 服务
#
# 用法（在 ubuntu1 上）：
#   bash /data3/gguf/bench_ab.sh                # 默认 reps=3
#   REPS=1 bash /data3/gguf/bench_ab.sh         # 快速验证
#   MODES="off on off2" ...                     # 一般用不到
#
# 它会：停掉 llama-qwen38 → 用 SPEC=off 起临时服务跑一遍 → 杀掉 → 用 SPEC=on 起 → 跑 → 杀掉
#       → 最后把 llama-qwen38 重新拉起来（保持你原来的状态）
set -u
BIN=/home/a3165/llama.cpp/build/bin
RUN=/data3/gguf/run_llama_server.sh
PY=/data3/ComfyUI/venv/bin/python
PY=${PY:-python3}
OUT=/data3/gguf/bench_ab_results
REPS=${REPS:-3}
PORT=${PORT:-8080}
mkdir -p "$OUT"

wait_up() {  # 等 /health
  for i in $(seq 1 120); do
    curl -s -m 2 "http://127.0.0.1:$PORT/health" | grep -q ok && return 0
    sleep 1
  done
  echo "!! 服务没起来，看 $1"; return 1
}

run_one() {
  local mode=$1 label=$2
  echo "=============================================================="
  echo ">>> 配置: SPEC=$mode   label=$label"
  echo "=============================================================="
  SPEC=$mode nohup bash "$RUN" > "/data3/gguf/bench_server_$label.log" 2>&1 &
  local pid=$!
  if ! wait_up "/data3/gguf/bench_server_$label.log"; then kill $pid 2>/dev/null; return 1; fi
  sleep 2
  python3 /data3/gguf/bench_llama.py all \
      --base-url "http://127.0.0.1:$PORT/v1" \
      --label "$label" --reps "$REPS" \
      --json "$OUT/$label.json" | tee "$OUT/$label.txt"
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  sleep 6
}

sudo systemctl stop llama-qwen38 2>/dev/null
sleep 4

run_one off baseline
run_one on  ngram_mod

echo
echo "=== 恢复 systemd 服务 llama-qwen38 ==="
sudo systemctl start llama-qwen38
sleep 3
echo "llama=$(systemctl is-active llama-qwen38)"
echo
echo "=== 对比汇总（decode + 投机任务）==="
python3 - "$OUT/baseline.json" "$OUT/ngram_mod.json" <<'PYEOF'
import json, sys
b = json.load(open(sys.argv[1])); s = json.load(open(sys.argv[2]))
print(f"{'测试':<24}{'baseline':>12}{'ngram-mod':>12}{'倍率':>8}")
def row(name, bv, sv):
    r = f"{sv/bv:.2f}x" if bv and sv else "-"
    print(f"{name:<24}{bv:>12.1f}{sv:>12.1f}{r:>8}")
row("prefill 8000 tok (tok/s)", b["prefill"][-1]["prefill_tps"], s["prefill"][-1]["prefill_tps"])
row("decode 512 tok (tok/s)", b["decode"][-1]["gen_tps_tpot"], s["decode"][-1]["gen_tps_tpot"])
for i, t in enumerate(b["spec"]):
    st = s["spec"][i]
    row(f"spec: {t['task']} (tok/s)", t["gen_tps"], st["gen_tps"])
    if st.get("accept_rate"):
        print(f"{'    ↳ 草稿接受率':<24}{'':>24}{st['accept_rate']*100:>11.1f}%")
print()
for k in ("baseline",):
    pass
print("峰值显存/功耗:", b.get("gpu"), "|", s.get("gpu"))
PYEOF
