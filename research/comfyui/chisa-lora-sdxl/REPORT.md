# chisa LoRA + SDXL 出图实测（CMP 90HX 单卡）

日期：2026-09-14　机器：ubuntu1　GPU：CMP 90HX 10GB（BDF `03:00.0`，Gen2 x16）
脚本：`scripts/comfy_bench.py`（主场景）、`comfy_bench2.py`（补充）、`comfy_bench3.py`（训练轮次）
原始数据：`data/gpu_comfyui_bench_20260914_083047.json`、`..._bench2_...json`、`..._bench3_...json`
出图：`images/`（1024² 全部 + 4K）　对比图：`previews/`

## 1. 环境与 workflow

| 项目 | 值 |
|---|---|
| ComfyUI | 0.35.0（systemd `comfyui.service`，`0.0.0.0:8188`），torch 2.14.0+cu130，`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| 底模 | `Illustrious-XL-v2.0.safetensors`、`animagine-xl-4.0.safetensors` |
| 自训 LoRA | `chisa_illustrious_v1.safetensors`（dim16/alpha8，见 §5） |
| 放大模型 | `4x-UltraSharp.pth`（另有 RealESRGAN_x4plus_anime_6B.pth） |
| workflow | `角色_chisa_SDXL`（直接 4K）、`角色_chisa_高清4K`（1344×768 + 放大）、`图生图_img2img`、`局部重绘_inpaint`（均为之前配置好的原版，脚本按其参数以 API 格式复现） |

提示词（SFW，取自原 workflow）：
`chisa, 1girl, solo, long black hair, red eyes, red hair ribbon, black choker, black serafuku, black shirt, black pleated skirt, sailor collar, red neckerchief, looking at viewer, cityscape, sunset, masterpiece, best quality`

## 2. 出图性能总表（Illustrious + chisa LoRA 1.3，同种子）

| 场景 | 分辨率 / 步数 | 耗时 | 显存峰值 | 功耗 |
|---|---|---|---|---|
| 预热（加载底模，512²/8） | 512² / 8 | 16.2 s | 7012 MiB | — |
| **标准出图** | **1024² / 24** | **10.9–11.1 s** | 9668 MiB | 250 W |
| 无 LoRA 对照（同种子） | 1024² / 24 | 9.1–9.3 s | 9668 MiB | 250 W |
| 换底模 animagine-xl-4.0 | 1024² / 24 | 21.0 s | 8132 MiB | 250 W |
| **批量 4 张** | 1024² ×4 / 24 | 44.7 s → **11.2 s/张** | 8646 MiB | 250 W |
| img2img | 1MP / 28 · denoise 0.81 | 12.2 s | 9670 MiB | 250 W |
| 局部重绘 inpaint | 800×1200 / 28 · grow 6 | 11.3 s | 9478 MiB | 250 W |
| **高清 4K（推荐）** | 1344×768 /30 → 4x-UltraSharp → 3840×2160 | **19.4 s** | 9670 MiB | 250 W |
| **直接 4K** | 3840×2160 / 24 | **107.1 s** | **9862 MiB** | 250 W |

> 全场景 GPU 稳定在 **SM 1830 MHz / 250 W 顶墙 / 54–59 °C**，无降频。

### 4K 两条路线

| 路线 | 耗时 | 峰值显存 | 评价 |
|---|---|---|---|
| A. 直接 4K 采样（3840×2160 一次出） | 107.1 s | 9862 MiB（几乎顶满 10240） | 细节自己生成，构图/手部风险高，接近显存上限 |
| **B. 两步法**（1344×768 采样 → 4x-UltraSharp → lanczos 到 4K） | **19.4 s（快 5.5×）** | 9670 MiB | 结构稳定、锐化放大，**日常推荐** |

对比图：`previews/compare_hires_4k.jpg`、`previews/preview_direct_4k.jpg`、`previews/preview_hires_4k.jpg`

## 3. LoRA 效果验证

| 对比 | 图 | 结论 |
|---|---|---|
| 强度扫描：no LoRA → 0.6 → 1.0 → 1.3 | `previews/compare_lora_strength.jpg` | 无 LoRA 时发色/服设漂移；**w ≥ 1.0 后黑长直 + 红丝带 + 红瞳 + 水手服稳定复现**（1.3 是原 workflow 默认） |
| 训练轮次：epoch 2 → 6 → 12 | `previews/compare_lora_epoch.jpg` | epoch 2 领口/领结不稳，epoch 6 收敛，epoch 12 服设最干净 → 12 轮有必要 |
| 底模对比：Illustrious+LoRA vs animagine-4.0 | `previews/compare_checkpoint.jpg` | 同提示词/种子下画风差异明显 |

LoRA 带来的额外耗时：1024²/24 步 **9.25 → 11.11 s（+1.9 s，+20%）**。

## 4. 与 2026-09-09 首测数据的差异说明（重要）

`research/comfyui/README.md` 记录 2026-09-09 首测「Illustrious 1024² **28 步 = 6.0 s**」。
本次在同一台机器上复核（`scripts/recheck_comfy.py`，含功耗/频率监控）**无法复现**：

| 配置 | 本次复核（wall / ComfyUI 自报） |
|---|---|
| 1024² / 28 步 / 无 LoRA | 9.25 s / 9.13 s，10.61 s / 10.31 s（两次） |
| 1024² / 24 步 / 无 LoRA | 9.25 s / 9.11 s |
| 1024² / 28 步 / LoRA 1.3 | 12.27 s / 11.84 s |
| 1024² / 24 步 / LoRA 1.3 | 11.11 s / 10.67 s |

复核期间 GPU 为 **SM 1830 MHz、250 W 顶墙（power limit 250 W）**，即满速运行、无降频，
显存 9668 MiB。差异来源应为环境/口径变化（09-09 首测用的是当时的 `~/comfyui-env`，
之后按要求重建过 venv；ComfyUI/attention 后端实现也可能不同）。

**本目录数据（2026-09-14）为当前环境下的可复现结果，建议以本表为准。**

## 5. 自训 LoRA（chisa）训练数据

| 项目 | 值 |
|---|---|
| 底模 | Illustrious-XL-v2.0（SDXL） |
| 训练集 | **16 张图 + 16 caption**（29 MB），1024²，开 bucket（768–1024 / step 64） |
| 网络 | LoRA `network_dim=16`、`network_alpha=8` |
| 轮次 | **12 epoch = 960 step** |
| 速度 / 时长 | **3.50 s/step**，总 **56 分 01 秒**（单卡 90HX） |
| batch / 精度 | batch 1 × grad-accum 2；fp16 + gradient checkpointing + SDPA + `--cache_latents` |
| 学习率 | UNet 1e-4 / TE 1e-5，cosine_with_restarts(3) + warmup 100 |
| 优化器/正则 | AdamW8bit、`min_snr_gamma=5`、`max_grad_norm=1.0`、`shuffle_caption` + `keep_tokens=1`、seed 42 |
| 收敛 | avr_loss ≈ **0.16 → 0.10**（最低单步 0.008） |
| 产出 | 每 2 轮存一次共 6 个 checkpoint（114 MB/个），最终 `chisa_illustrious_v1.safetensors` |

训练脚本见 `~/kohya/train.sh`（kohya-ss/sd-scripts）。

## 6. 复现

```bash
sudo systemctl start comfyui                 # http://192.168.124.10:8188
python3 scripts/comfy_bench.py               # §2 主场景
python3 scripts/comfy_bench2.py              # LoRA 强度 / img2img
python3 scripts/comfy_bench3.py              # 训练轮次（需先把 epoch 权重拷进 models/loras）
/data3/ComfyUI/venv/bin/python scripts/mk_previews.py   # 生成 previews/
```

> 注：LLM 服务（27B，常驻 9.0 GB）与 SDXL（峰值 9.7 GB）**不能在同一张 10GB 卡上同时常驻**，
> 跑出图前需 `sudo systemctl stop llama-qwen38`。
