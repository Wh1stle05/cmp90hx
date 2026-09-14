#!/usr/bin/env python3
"""补充测试：img2img 修复版 + LoRA 强度扫描 + LoRA 训练 epoch 对比"""
import json, os, sys, time
sys.path.insert(0, "/tmp")
from comfy_bench import (graph_t2i, graph_img2img, run_one, wait_health, RES, POS, NEG)

SCEN = [
    ("img2img_1mp", "img2img (16.jpg→1MP) · denoise 0.81 · 28步", graph_img2img()),
    ("lora_w0.6", "1024² LoRA 强度 0.6 · 24步", graph_t2i(w=1024, h=1024, steps=24, seed=1234567890,
                                                        lora_w=0.6, prefix="bench/lora_w06")),
    ("lora_w1.0", "1024² LoRA 强度 1.0 · 24步", graph_t2i(w=1024, h=1024, steps=24, seed=1234567890,
                                                        lora_w=1.0, prefix="bench/lora_w10")),
    ("lora_epoch2", "1024² LoRA 第2轮权重 (000002) · 24步",
     graph_t2i(lora="chisa_illustrious_v1-000002.safetensors", w=1024, h=1024, steps=24,
               seed=1234567890, lora_w=1.3, prefix="bench/lora_e2")),
    ("lora_epoch6", "1024² LoRA 第6轮权重 (000006) · 24步",
     graph_t2i(lora="chisa_illustrious_v1-000006.safetensors", w=1024, h=1024, steps=24,
               seed=1234567890, lora_w=1.3, prefix="bench/lora_e6")),
]

if __name__ == "__main__":
    wait_health()
    results = []
    for name, desc, g in SCEN:
        print(f"\n>>> {name}: {desc}", flush=True)
        r = run_one(name, desc, g)
        print(json.dumps(r, ensure_ascii=False), flush=True)
        results.append(r)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RES, f"gpu_comfyui_bench2_{stamp}.json")
    json.dump({"date": stamp, "results": results}, open(path, "w"), ensure_ascii=False, indent=2)
    print("SAVED", path, flush=True)
