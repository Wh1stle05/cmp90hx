#!/usr/bin/env python3
import json, os, sys, time
sys.path.insert(0, "/tmp")
from comfy_bench import graph_t2i, run_one, wait_health, RES

SCEN = [
    ("lora_epoch2", "1024² LoRA 训练第2轮权重 (epoch 2/12) · 24步",
     graph_t2i(lora="chisa_e02.safetensors", w=1024, h=1024, steps=24, seed=1234567890,
               lora_w=1.3, prefix="bench/lora_e2")),
    ("lora_epoch6", "1024² LoRA 训练第6轮权重 (epoch 6/12) · 24步",
     graph_t2i(lora="chisa_e06.safetensors", w=1024, h=1024, steps=24, seed=1234567890,
               lora_w=1.3, prefix="bench/lora_e6")),
]
if __name__ == "__main__":
    wait_health()
    results = []
    for name, desc, g in SCEN:
        print(f"\n>>> {name}", flush=True)
        r = run_one(name, desc, g)
        print(json.dumps(r, ensure_ascii=False), flush=True)
        results.append(r)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RES, f"gpu_comfyui_bench3_{stamp}.json")
    json.dump({"date": stamp, "results": results}, open(path, "w"), ensure_ascii=False, indent=2)
    print("SAVED", path, flush=True)
