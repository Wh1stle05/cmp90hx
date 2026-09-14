#!/usr/bin/env python3
"""llama.cpp 侧数据：冷启动加载、TTFT/预填充、显存、OpenAI 接口时延"""
import json, subprocess, time, urllib.request, urllib.error, os, uuid

BIN = "/home/a3165/llama.cpp/build/bin"
M = "/data3/gguf/Qwen3.8-27B-UD-IQ2_XXS.gguf"
RES = "/data3/dual-90hx-research/results"
API = "http://127.0.0.1:8080"
SENT = "The on-disk row cache keeps embedding rows on disk and fetches only the rows that a token needs. "


def sh(c, check=True):
    return subprocess.run(c, shell=True, capture_output=True, text=True, check=check).stdout.strip()


def vram():
    return int(sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits").split()[0])


def post(path, body=None, timeout=2400):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(API + path, data=data, headers={"Content-Type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    except urllib.error.HTTPError as e:
        print("   HTTP", e.code, e.read().decode()[:200], flush=True)
        raise


def tok(t):
    return len(post("/tokenize", {"content": t})["tokens"])


out = {"date": time.strftime("%Y%m%d_%H%M%S"), "model": M, "binary": BIN}

print("=== 1) 停 comfyui / 确认 GPU 空闲", flush=True)
sh("sudo systemctl stop comfyui || true", check=False)
sh("sudo systemctl stop llama-qwen38 || true", check=False)
for _ in range(60):
    if vram() < 500:
        break
    time.sleep(2)
out["vram_idle_mib"] = vram()
print("   idle vram:", out["vram_idle_mib"], flush=True)

print("=== 2) llama-bench（f16 KV / q8_0 KV）", flush=True)
benches = {}
for kvt in ("f16", "q8_0"):
    args = f'{BIN}/llama-bench -m {M} -ngl 99 -p 512,4096 -n 64 -r 1'
    if kvt != "f16":
        args += f" -ctk {kvt} -ctv {kvt}"
    log = sh(args)
    benches[kvt] = log
    print(log, flush=True)
out["llama_bench"] = benches

print("=== 3) 冷启动（service start → /health ok）", flush=True)
t0 = time.time()
sh("sudo systemctl start llama-qwen38")
ok_at = None
while time.time() - t0 < 300:
    try:
        if b"ok" in urllib.request.urlopen(API + "/health", timeout=2).read():
            ok_at = time.time() - t0
            break
    except Exception:
        pass
    time.sleep(0.5)
out["cold_start_s"] = round(ok_at, 2) if ok_at else None
out["vram_loaded_mib"] = vram()
print(f"   cold start = {out['cold_start_s']}s, vram = {out['vram_loaded_mib']} MiB", flush=True)

print("=== 4) TTFT / 预填充（不同输入长度）", flush=True)
ttfts = []
for target in (100, 2500, 30000):
    lo, hi = 1, 1
    while tok(SENT * hi) < target:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if tok(SENT * mid) < target:
            lo = mid + 1
        else:
            hi = mid
    text = SENT * max(1, lo - 1)
    n_in = tok(text)
    body = {"model": "qwen3.8-27b", "temperature": 0, "max_tokens": 16,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": f"[{uuid.uuid4().hex[:6]}] 一句话总结：\n" + text}]}
    t0 = time.time()
    d = post("/v1/chat/completions", body)
    wall = time.time() - t0
    t = d.get("timings", {})
    row = {"input_tokens": d["usage"]["prompt_tokens"], "ttft_s": round(t.get("prompt_ms", 0) / 1000, 2),
           "total_s": round(wall, 2), "prefill_tps": round(t.get("prompt_per_second", 0), 1),
           "decode_tps": round(t.get("predicted_per_second", 0), 1)}
    ttfts.append(row)
    print("  ", json.dumps(row), flush=True)
out["ttft"] = ttfts

print("=== 5) 短问答时延（含思考关闭）", flush=True)
t0 = time.time()
d = post("/v1/chat/completions", {"model": "qwen3.8-27b", "temperature": 0, "max_tokens": 128,
                                  "chat_template_kwargs": {"enable_thinking": False},
                                  "messages": [{"role": "user", "content": "17*23 等于多少？只回答数字。"}]})
out["short_qa"] = {"wall_s": round(time.time() - t0, 2), "answer": d["choices"][0]["message"]["content"][:40],
                   "prompt_tokens": d["usage"]["prompt_tokens"],
                   "completion_tokens": d["usage"]["completion_tokens"],
                   "decode_tps": round(d["timings"]["predicted_per_second"], 1)}
print("  ", json.dumps(out["short_qa"], ensure_ascii=False), flush=True)

json.dump(out, open(os.path.join(RES, f"gpu_llama_bench_{out['date']}.json"), "w"),
          ensure_ascii=False, indent=2)
print("SAVED", os.path.join(RES, f"gpu_llama_bench_{out['date']}.json"), flush=True)
