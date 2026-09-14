#!/usr/bin/env python3
"""视频用：基模 vs +LoRA，1024x1024 / 24 步，各 3 次取平均"""
import json, subprocess, sys, threading, time, urllib.request
sys.path.insert(0, "/tmp")
from comfy_bench import graph_t2i, wait_health, log_tail
import re

API = "http://127.0.0.1:8188"
SEED = 1234567890


def run(steps, lora, tag):
    g = graph_t2i(lora=lora, w=1024, h=1024, steps=steps, seed=SEED, lora_w=1.3,
                  prefix=f"bench/video_{tag}")
    mon = {"v": 0, "pw": 0.0}
    stop = False

    def m():
        while not stop:
            try:
                o = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,power.draw",
                                             "--format=csv,noheader"], text=True).strip()
                p = [x.strip() for x in o.split(",")]
                mon["v"] = max(mon["v"], int(p[0].split()[0])); mon["pw"] = max(mon["pw"], float(p[1].split()[0]))
            except Exception:
                pass
            time.sleep(0.2)
    threading.Thread(target=m, daemon=True).start()
    t0 = time.time()
    body = json.dumps({"prompt": g}).encode()
    pid = json.loads(urllib.request.urlopen(urllib.request.Request(
        API + "/prompt", data=body, headers={"Content-Type": "application/json"}), timeout=30).read())["prompt_id"]
    while True:
        time.sleep(0.3)
        if pid in json.loads(urllib.request.urlopen(f"{API}/history/{pid}", timeout=30).read()):
            break
    wall = time.time() - t0
    stop = True
    log = log_tail(t0)
    rep = re.findall(r"Prompt executed in ([0-9.]+) seconds", log)
    comfy = float(rep[-1]) if rep else None
    print(f"{tag:<22} wall={wall:5.2f}s  comfy={comfy:5.2f}s  vram={mon['v']}MiB  pw={mon['pw']:.0f}W", flush=True)
    return wall, comfy


if __name__ == "__main__":
    wait_health()
    run(8, None, "warmup")          # 预热（加载底模）
    res = {"base": [], "lora": []}
    for i in range(3):
        res["base"].append(run(24, None, f"base_{i+1}"))
        res["lora"].append(run(24, "chisa_illustrious_v1.safetensors", f"lora_{i+1}"))
    b = [x[0] for x in res["base"]]; l = [x[0] for x in res["lora"]]
    print(f"\n基模   1024²/24步: {b} → 平均 {sum(b)/len(b):.2f}s ({min(b):.2f}~{max(b):.2f})")
    print(f"基模+LoRA 1024²/24步: {l} → 平均 {sum(l)/len(l):.2f}s ({min(l):.2f}~{max(l):.2f})")
    print(f"LoRA 增量: {sum(l)/len(l)-sum(b)/len(b):+.2f}s")
