#!/usr/bin/env python3
"""ComfyUI benchmark：chisa LoRA + 之前配置的 4 套 workflow，产出可写入研究报告的数据。"""
import json, os, re, shutil, subprocess, threading, time, uuid, urllib.request

API = "http://127.0.0.1:8188"
RES = "/data3/dual-90hx-research/results"
IMGDIR = os.path.join(RES, "images")
os.makedirs(IMGDIR, exist_ok=True)

POS = ("chisa, 1girl, solo, long black hair, red eyes, red hair ribbon, black choker, black serafuku, "
       "black shirt, black pleated skirt, sailor collar, red neckerchief, looking at viewer, cityscape, "
       "sunset, masterpiece, best quality")
NEG = ("lowres, worst quality, low quality, bad anatomy, bad hands, extra fingers, jpeg artifacts, "
       "watermark, signature")
POS_I2I = ("chisa, 1girl, solo, long black hair, red eyes, red hair ribbon, black choker, white sundress, "
           "standing, cityscape, sunset, masterpiece, best quality")
POS_INP = ("chisa, 1girl, solo, long black hair, red eyes, red hair ribbon, black choker, white pleated skirt, "
           "kneehigs, masterpiece, best quality")


def ckpt(name):
    return {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": name}}


def clip(text, src):
    return {"class_type": "CLIPTextEncode", "inputs": {"text": text, "clip": src}}


def ksampler(model, pos, neg, latent, seed, steps, cfg=7.0, sampler="euler_ancestral",
             sched="normal", denoise=1.0):
    return {"class_type": "KSampler", "inputs": {
        "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler, "scheduler": sched,
        "denoise": denoise, "model": model, "positive": pos, "negative": neg, "latent_image": latent}}


def save(images, prefix):
    return {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": images}}


def graph_t2i(ckpt_name="Illustrious-XL-v2.0.safetensors", lora="chisa_illustrious_v1.safetensors",
              lora_w=1.3, w=1024, h=1024, steps=24, seed=1234567890, batch=1,
              pos=POS, neg=NEG, prefix="bench/t2i"):
    g = {"1": ckpt(ckpt_name)}
    model, clip_src = ["1", 0], ["1", 1]
    if lora:
        g["2"] = {"class_type": "LoraLoader", "inputs": {
            "lora_name": lora, "strength_model": lora_w, "strength_clip": lora_w,
            "model": ["1", 0], "clip": ["1", 1]}}
        model, clip_src = ["2", 0], ["2", 1]
    g["3"] = clip(pos, clip_src)
    g["4"] = clip(neg, clip_src)
    g["5"] = {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": batch}}
    g["6"] = ksampler(model, ["3", 0], ["4", 0], ["5", 0], seed, steps)
    g["7"] = {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}}
    g["8"] = save(["7", 0], prefix)
    return g


def graph_hires4k(w=1344, h=768, steps=30, seed=16781277634538, upscale="4x-UltraSharp.pth",
                  out_w=3840, out_h=2160, lora_w=1.0):
    g = graph_t2i(w=w, h=h, steps=steps, seed=seed, lora_w=lora_w, prefix="bench/hires_src")
    g["9"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": upscale}}
    g["10"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["9", 0], "image": ["7", 0]}}
    g["11"] = {"class_type": "ImageScale", "inputs": {"image": ["10", 0], "upscale_method": "lanczos",
                                                      "width": out_w, "height": out_h, "crop": "disabled"}}
    g["12"] = save(["11", 0], "bench/hires_4k")
    return g


def graph_img2img(image="16.jpg", megapixels=1.0, steps=28, seed=982656716057339, denoise=0.81, lora_w=1.3):
    g = {"1": ckpt("Illustrious-XL-v2.0.safetensors"),
         "2": {"class_type": "LoraLoader", "inputs": {"lora_name": "chisa_illustrious_v1.safetensors",
                                                      "strength_model": lora_w, "strength_clip": lora_w,
                                                      "model": ["1", 0], "clip": ["1", 1]}},
         "3": clip(POS_I2I, ["2", 1]),
         "4": clip(NEG, ["2", 1]),
         "5": {"class_type": "LoadImage", "inputs": {"image": image}},
         "6": {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": ["5", 0],
                                                                   "upscale_method": "lanczos",
                                                                   "megapixels": megapixels,
                                                                   "resolution_steps": 1}},
         "7": {"class_type": "VAEEncode", "inputs": {"pixels": ["6", 0], "vae": ["1", 2]}},
         "8": ksampler(["2", 0], ["3", 0], ["4", 0], ["7", 0], seed, steps, denoise=denoise),
         "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["1", 2]}},
         "10": save(["9", 0], "bench/img2img")}
    return g


def graph_inpaint(image="test_src.jpg", steps=28, seed=42, grow=6, lora_w=1.3):
    g = {"1": ckpt("Illustrious-XL-v2.0.safetensors"),
         "2": {"class_type": "LoraLoader", "inputs": {"lora_name": "chisa_illustrious_v1.safetensors",
                                                      "strength_model": lora_w, "strength_clip": lora_w,
                                                      "model": ["1", 0], "clip": ["1", 1]}},
         "3": clip(POS_INP, ["2", 1]),
         "4": clip(NEG, ["2", 1]),
         "5": {"class_type": "LoadImage", "inputs": {"image": image}},
         "6": {"class_type": "VAEEncodeForInpaint", "inputs": {"pixels": ["5", 0], "vae": ["1", 2],
                                                               "mask": ["5", 1], "grow_mask_by": grow}},
         "7": ksampler(["2", 0], ["3", 0], ["4", 0], ["6", 0], seed, steps),
         "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}},
         "9": save(["8", 0], "bench/inpaint")}
    return g


SCENARIOS = [
    ("warmup_512", "预热(加载底模)", graph_t2i(lora=None, w=512, h=512, steps=8, seed=1, prefix="bench/warmup")),
    ("t2i_1024_chisa_lora", "1024² txt2img · chisa LoRA 1.3 · 24步",
     graph_t2i(w=1024, h=1024, steps=24, seed=1234567890, lora_w=1.3, prefix="bench/t2i_lora")),
    ("t2i_1024_no_lora", "1024² txt2img · 无 LoRA（同种子）· 24步",
     graph_t2i(lora=None, w=1024, h=1024, steps=24, seed=1234567890, prefix="bench/t2i_nolora")),
    ("t2i_1024_animagine", "1024² txt2img · animagine-xl-4.0 · 24步",
     graph_t2i(ckpt_name="animagine-xl-4.0.safetensors", lora=None, w=1024, h=1024, steps=24,
               seed=1234567890, prefix="bench/t2i_anima")),
    ("batch4_1024_chisa", "1024² ×4 批量 · chisa LoRA · 24步",
     graph_t2i(w=1024, h=1024, steps=24, seed=777, batch=4, lora_w=1.3, prefix="bench/batch4")),
    ("img2img_1mp", "img2img (16.jpg→1MP) · denoise 0.81 · 28步", graph_img2img()),
    ("inpaint_28", "inpaint (test_src.jpg) · 28步", graph_inpaint()),
    ("hires_4k", "1344×768 → 4x-UltraSharp → 3840×2160 · 30步", graph_hires4k()),
    ("direct_4k", "3840×2160 直接生成 · 24步",
     graph_t2i(w=3840, h=2160, steps=24, seed=1234567890, prefix="bench/direct4k")),
]


def wait_health(timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(API + "/system_stats", timeout=5):
                return True
        except Exception:
            time.sleep(2)
    return False


class VramMon(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak = 0
        self.stop = False

    def run(self):
        while not self.stop:
            try:
                out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                               "--format=csv,noheader,nounits"], text=True)
                self.peak = max(self.peak, int(out.split()[0]))
            except Exception:
                pass
            time.sleep(0.3)


def log_tail(since_epoch):
    try:
        out = subprocess.check_output(["journalctl", "-u", "comfyui", "--since", f"@{int(since_epoch)}",
                                       "--no-pager"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return ""
    return out


def run_one(name, desc, graph):
    mon = VramMon(); mon.start()
    mon.peak = 0
    t0 = time.time()
    body = json.dumps({"prompt": graph, "client_id": f"bench-{uuid.uuid4().hex[:8]}"}).encode()
    req = urllib.request.Request(API + "/prompt", data=body, headers={"Content-Type": "application/json"})
    try:
        pid = json.loads(urllib.request.urlopen(req, timeout=60).read())["prompt_id"]
    except urllib.error.HTTPError as e:
        mon.stop = True
        return {"scenario": name, "desc": desc, "error": e.read().decode()[:500]}
    hist = None
    while True:
        time.sleep(1)
        try:
            d = json.loads(urllib.request.urlopen(f"{API}/history/{pid}", timeout=30).read())
            if pid in d:
                hist = d[pid]; break
        except Exception:
            pass
        if time.time() - t0 > 2400:
            mon.stop = True
            return {"scenario": name, "desc": desc, "error": "timeout 2400s"}
    wall = time.time() - t0
    mon.stop = True; time.sleep(0.4)
    log = log_tail(t0)
    rep = re.findall(r"Prompt executed in ([0-9.]+) seconds", log)
    its = [round(float(x), 2) for x in re.findall(r"([0-9.]+)it/s", log)]
    st = hist.get("status", {})
    imgs = []
    for _nid, out in (hist.get("outputs") or {}).items():
        for im in out.get("images", []):
            src = os.path.join("/data3/ComfyUI/output", im.get("subfolder", ""), im["filename"])
            dst = os.path.join(IMGDIR, f"{name}_{im['filename']}")
            try:
                shutil.copy(src, dst); imgs.append(dst)
            except Exception as e:
                imgs.append(f"COPY_FAIL {src} {e}")
    return {"scenario": name, "desc": desc, "status": st.get("status_str"),
            "wall_s": round(wall, 2), "comfy_reported_s": float(rep[-1]) if rep else None,
            "peak_vram_mib": mon.peak, "its_samples": its[-3:], "images": imgs,
            "error": None if st.get("status_str") != "error" else json.dumps(st.get("messages"))[:400]}


if __name__ == "__main__":
    wait_health()
    results = []
    for name, desc, g in SCENARIOS:
        print(f"\n>>> {name}: {desc}", flush=True)
        r = run_one(name, desc, g)
        print(json.dumps(r, ensure_ascii=False), flush=True)
        results.append(r)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RES, f"gpu_comfyui_bench_{stamp}.json")
    json.dump({"date": stamp, "gpu": "NVIDIA CMP 90HX 10GB (single card)",
               "comfyui_version": json.loads(urllib.request.urlopen(API + "/system_stats",
                                                                    timeout=10).read())["system"]["comfyui_version"],
               "results": results}, open(path, "w"), ensure_ascii=False, indent=2)
    print("\nSAVED", path, flush=True)
