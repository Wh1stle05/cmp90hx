#!/usr/bin/env bash
# CMP 90HX GPU1 压力测试
# 用法: sudo ./pstress.sh [时长秒] [轮询间隔秒]
# 默认 600 秒 / 5 秒
set -uo pipefail

GPU_ID="${GPU_ID:-1}"
BENCH="${BENCH:-$HOME/dual-90hx-research/bench}"
DURATION="${1:-600}"
POLL="${2:-5}"
OUT="$HOME/dual-90hx-research/pstress"
mkdir -p "$OUT"
TS="$(date +%Y%m%d_%H%M%S)"
DMON="$OUT/dmon_gpu${GPU_ID}_${TS}.log"
PTS="$OUT/points_gpu${GPU_ID}_${TS}.csv"

[[ -x "$BENCH" ]] || { echo "no bench at $BENCH"; exit 1; }

echo "=== 压测 GPU${GPU_ID} ${DURATION}s (poll ${POLL}s) ==="
echo "start: $(date '+%H:%M:%S')"

# 1) dmon 全程低频采样(1s) 背景
nvidia-smi dmon -i "$GPU_ID" -s pumt -d 1 -o DT > "$DMON" &
DMON_PID=$!
sleep 2

# 2) 跑满: bench 各模式循环, 后台
(
  i=0
  while true; do
    i=$((i+1))
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$BENCH" fp32 >/dev/null 2>&1
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$BENCH" fp16 >/dev/null 2>&1
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$BENCH" int8 >/dev/null 2>&1
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$BENCH" copydev >/dev/null 2>&1
    echo "round $i done $(date '+%H:%M:%S')" >> "$OUT/rounds_${TS}.log"
  done
) &
STRESS_PID=$!

# 3) 打点记录
echo "elapsed_s,temp_c,power_w,sm_pct,mem_pct,clk_sm_mhz,clk_mem_mhz,mem_gb,pcirx_gbs" > "$PTS"
t0=$(date +%s)
fail=0
while true; do
  now=$(date +%s); el=$((now-t0))
  [[ $el -ge $DURATION ]] && break
  line=$(nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu,utilization.memory,clocks.sm,clocks.mem,memory.used --format=csv,noheader,nounits -i "$GPU_ID" 2>/dev/null | tr -d ' ')
  # 顺带打 pcie 状态
  pgen=$(nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv,noheader,nounits -i "$GPU_ID" 2>/dev/null | tr -d ' ')
  echo "$el,$line,$pgen" >> "$PTS"
  # 检测掉速(gen降到1)或温度异常
  g=$(echo "$pgen" | cut -d, -f1)
  if [ "$g" = "1" ]; then
    echo "!! gen 掉到 1 at ${el}s" | tee -a "$OUT/pstress_${TS}.log"
    fail=1
  fi
  sleep "$POLL"
done

# 4) 收尾
kill $STRESS_PID 2>/dev/null; wait $STRESS_PID 2>/dev/null
sleep 1
kill $DMON_PID 2>/dev/null; wait $DMON_PID 2>/dev/null

echo "=== 汇总 (GPU${GPU_ID}, ${DURATION}s) ==="
python3 - "$PTS" <<'PY'
import sys, csv
rows=list(csv.DictReader(open(sys.argv[1])))
if not rows: print("no data"); sys.exit(0)
def g(k,i): 
    try: return float(rows[i][k])
    except: return 0.0
n=len(rows)
temps=[g('temp_c',i) for i in range(n)]
pw=[g('power_w',i) for i in range(n)]
sm=[g('sm_pct',i) for i in range(n)]
clk=[g('clk_sm_mhz',i) for i in range(n)]
print(f"samples={n} ({rows[-1]['elapsed_s']}s)")
print(f"temp  max={max(temps):.0f}C avg={sum(temps)/n:.1f}C")
print(f"power max={max(pw):.0f}W avg={sum(pw)/n:.1f}W  (>245W samples: {sum(1 for p in pw if p>245)})")
print(f"SM    avg={sum(sm)/n:.1f}%  (>90% samples: {sum(1 for s in sm if s>90)}/{n})")
print(f"clk   min={min(clk):.0f}Hz max={max(clk):.0f}MHz")
print("temp 分布:", " ".join(f"{c}C:{int(sum(1 for t in temps if t==c)*100/n)}%" for c in sorted(set(int(t) for t in temps))))
PY
echo "done: $(date '+%H:%M:%S')  logs: $OUT/dmon_gpu${GPU_ID}_${TS}.log $PTS"
[[ $fail -eq 1 ]] && echo "!! 检测到 PCIe 掉速, 查看 $PTS"