# ComfyUI 0.35 部署 + SDXL 文生图首测（CMP 90HX 单卡）

日期：2026-09-09　机器：ubuntu1　GPU：CMP 90HX 10GB（GPU1，Gen2 x8）

## 环境

| 项目 | 值 |
|---|---|
| ComfyUI | 0.35.0（2026-09-08 master），Web UI + API 双可用 |
| venv | `~/comfyui-env`，Python 3.12，torch 2.14.0+cu130（sm_86 正常） |
| 启动 | `CUDA_VISIBLE_DEVICES=1 python main.py --listen 0.0.0.0 --port 8188` |
| 显存模式 | NORMAL_VRAM（10G 全量加载，不走 lowvram） |
| 模型 | `Illustrious-XL-v2.0.safetensors`、`animagine-xl-4.0.safetensors`（各 6.5G） |

依赖要点：torch 走 pytorch cu130 wheel；`requirements.txt` 里 torchsde 等
源码包用 `--only-binary` 跳过可省大量编译时间；ComfyUI 动态 VRAM +
comfy-aimdo 已自动启用（日志可见 `aimdo inited`）。

## 首图实测（Illustrious-XL v2.0）

workflow：1024×1024，28 步，euler / normal，cfg 7.0，正负提示词各一。

```
生成耗时: 6.0 秒（API 提交 → 出图）
GPU:      SM 100% / 249W（顶 250W 墙）/ 显存 7.2G / 51°C
回落后:   87W / 42°C（模型驻留显存）
```

- SDXL 全家（UNet + 双 CLIP + VAE）**7.2G 全量进显存**，无需 lowvram，
  Gen2 PCIe 不参与计算，速度无衰减。
- 显存预算验证：10G 下 SDXL 全量运行绰绰有余（剩 ~2.8G）。

## API 用法（自动化/压测）

POST `http://127.0.0.1:8188/prompt` JSON workflow → 轮询
`/history/{prompt_id}` 取图。首图脚本 `tx_test.py` 即该流程（含耗时统计）。

## 对比参考

| 模型 | 分辨率 | 步数 | 耗时 |
|---|---|---|---|
| Illustrious-XL v2.0 | 1024² | 28 | **6.0 s** |
| animagine-xl-4.0 | 待测 | — | — |
| （vLLM 侧参考：Qwen3.5-4B-AWQ decode 159 tok/s） | | | |

两者都属 SDXL 生态，动漫/NSFW 向；加载 6.5G 权重一次约数秒。

## 待办

- [ ] animagine-xl-4.0 同条件对比
- [ ] 512² / 多步数 / 不同调度器基准
- [ ] 压测脚本接入（pstress.sh 已验证 GPU1 30min 249W 稳定）