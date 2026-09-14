#!/usr/bin/env python3
"""导出 kohya 两次训练的 loss 曲线（每 10 步采样）为 CSV"""
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import csv, os

RUNS = [("20260911055337", "run1_failed"), ("20260911071327", "run2_final")]
os.makedirs("/data3/gguf/lora_logs", exist_ok=True)

for tag_dir, name in RUNS:
    d = f"/data3/kohya/lora/logs/{tag_dir}/network_train"
    ea = EventAccumulator(d, size_guidance={"scalars": 0})
    ea.Reload()
    cur = {s.step: s.value for s in ea.Scalars("loss/current")}
    avg = {s.step: s.value for s in ea.Scalars("loss/average")}
    ep = {s.step: s.value for s in ea.Scalars("loss/epoch_average")}
    steps = sorted(cur)
    out = f"/data3/gguf/lora_logs/lora_{name}_curve.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "loss_current", "loss_average"])
        for s in steps:
            if s % 10 == 0 or s == steps[-1]:
                w.writerow([s, round(cur[s], 5), round(avg.get(s, float("nan")), 5)])
    ep_out = f"/data3/gguf/lora_logs/lora_{name}_epoch_avg.csv"
    with open(ep_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "epoch_average_loss"])
        for e in sorted(ep):
            w.writerow([e, round(ep[e], 5)])
    print(f"{name}: {len(steps)} 点, loss {steps[0]}→{steps[-1]} 步 "
          f"avg {avg.get(steps[0]):.4f}→{avg.get(steps[-1]):.4f}, epoch平均 "
          + " ".join(f"{e}:{ep[e]:.3f}" for e in sorted(ep)))
    print("   ->", out, "+", ep_out)
