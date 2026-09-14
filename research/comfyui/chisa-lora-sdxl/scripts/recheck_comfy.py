#!/usr/bin/env python3
"""复核 1024² 出图耗时：与 research/comfyui/README.md 的 6.0s 首测对比（含功耗/频率监控）"""
import csv, io, json, os, re, subprocess, sys, threading, time, urllib.request
sys.path.insert(0, "/tmp")
from comfy_bench import graph_t2i, wait_health, log_tail

API = "http://127.0.0.1:8188"


def gpu():
    o = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,clocks.sm,power.draw,temperature.gpu",
                                 "--format=csv,noheader"], text=True).strip()
    p = [x.strip() for x in o.split(",")]
    return int(p[0].split()[0]), int(p[1].split()[0]), float(p[2].split()[0]), int(p[3].split()[0])


def submit(graph):
    body = json.dumps({"prompt": graph}).encode()
    req = urllib.request.Request(API + "/prompt", data=body, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())["prompt_id"]


def run(steps, lora, tag):
    g = graph_t2i(lora=lora, w=1024, h=1024, steps=steps, seed=1234567890, lora_w=1.3,
                  prefix=f"bench/recheck_{tag}")
    mon = {"vram": 0, "clk": 0, "pw": 0.0, "temp": 0}
    stop = False

    def m():
        while not stop:
            try:
                v, c, p, t = gpu()
                mon["vram"] = max(mon["vram"], v); mon["clk"] = max(mon["clk"], c)
                mon["pw"] = max(mon["pw"], p); mon["temp"] = max(mon["temp"], t)
            except Exception:
                pass
            time.sleep(0.2)
    th = threading.Thread(target=m, daemon=True); th.start()
    t0 = time.time()
    pid = submit(g)
    while True:
        time.sleep(0.4)
        d = json.loads(urllib.request.urlopen(f"{API}/history/{pid}", timeout=30).read())
        if pid in d:
            break
    wall = time.time() - t0
    stop = True; time.sleep(0.3)
    log = log_tail(t0)
    rep = re.findall(r"Prompt executed in ([0-9.]+) seconds", log)
    print(f"{tag:<28} wall={wall:6.2f}s  comfy={rep[-1] if rep else '?':>7}s  "
          f"vram={mon['vram']}MiB  sm={mon['clk']}MHz  pw={mon['pw']:.0f}W  temp={mon['temp']}C", flush=True)
    return wall


if __name__ == "__main__":
    wait_health()
    print("warmup ...", flush=True)
    run(8, None, "warmup")                      # 冷启动
    run(28, None, "28step_nolora")
    run(28, "chisa_illustrious_v1.safetensors", "28step_lora")
    run(24, None, "24step_nolora")
    run(24, "chisa_illustrious_v1.safetensors", "24step_lora")
    run(28, None, "28step_nolora_again")        # 再测一次确认稳定
