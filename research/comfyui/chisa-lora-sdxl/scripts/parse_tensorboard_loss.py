#!/usr/bin/env python3
"""解析 kohya 的 tensorboard 日志，还原两次 LoRA 训练的步数/时长/loss 曲线"""
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob, os, datetime

for d in sorted(glob.glob("/data3/kohya/lora/logs/*/network_train")):
    ev = glob.glob(os.path.join(d, "events.out.tfevents.*"))
    if not ev:
        continue
    ea = EventAccumulator(d, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    print(f"\n=== {d}")
    print("  run 目录名(起始):", os.path.basename(os.path.dirname(d)))
    print("  scalar tags:", tags)
    for tag in tags:
        s = ea.Scalars(tag)
        if not s:
            continue
        first, last = s[0], s[-1]
        t0 = datetime.datetime.fromtimestamp(first.wall_time)
        t1 = datetime.datetime.fromtimestamp(last.wall_time)
        dur = last.wall_time - first.wall_time
        print(f"  [{tag}] 点数={len(s)}  步 {first.step} → {last.step}")
        print(f"      值 {first.value:.5f} → {last.value:.5f} (min {min(x.value for x in s):.5f} / max {max(x.value for x in s):.5f})")
        print(f"      时间 {t0:%H:%M:%S} → {t1:%H:%M:%S}  跨度 {dur/60:.1f} 分钟")
        # 每 10% 抽一个点
        n = len(s)
        picks = [s[int(n * p / 10)] if int(n * p / 10) < n else s[-1] for p in range(11)]
        print("      曲线: " + "  ".join(f"step{x.step}:{x.value:.3f}" for x in picks))
