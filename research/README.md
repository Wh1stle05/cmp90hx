# Research — CMP 90HX 单卡实测（ubuntu1）

解锁之后这台机器能做什么、各环节实测多少，全部归档在这里。

> **2026-09-09 硬件变更**：双卡改为**单卡（GPU1 保留）**。PCIe 是双卡
> 时 x16/x8 的实测；单卡场景下 PCIe 只影响模型加载（几秒），推理/绘图
> 时不参与。P2P/NCCL 部分保留为双卡历史数据。

## 目录

| 子目录 | 内容 | 关键结论 |
|---|---|---|
| `AI-capability-2026-09-14.md` | **单卡 AI 能力总汇**（LLM + 出图） | 27B-2bit 35 tok/s、SDXL 1024² 11s、4K 19s |
| `llamacpp/` | llama.cpp 跑 Qwen3.8-27B IQ2_XXS（+mmproj），上下文/投机/多模态 | 冷启动 3.7s；pp 840 tok/s、tg 35.5 tok/s；32k 上限 |
| `comfyui/chisa-lora-sdxl/` | chisa LoRA + SDXL 出图、4K 两条路线、LoRA 训练与强度/轮次对比 | 1024² 11s、批量 11.2s/张、4K 19.4s（两步法） |
| `notes/` | 上下文上限、ngram 投机解码、分层带宽（RAM/SSD/HDD）、Flash-Next 量化与 MTP | 复制类任务投机 2.45–2.71× |
| `p2p/` | 双卡 P2P / NCCL / 拷贝带宽（torch 手写 + nccl-tests） | TP 不可行（all_reduce ~2GB/s），PP 更合适 |
| `vllm/` | vLLM 0.28 跑 Qwen3.5-4B-AWQ prefill/decode/显存 | decode 159 tok/s（CUDA Graph），prefill 7.3k tok/s |
| `comfyui/` | ComfyUI 0.35 部署 + SDXL 文生图 | Illustrious 1024² 28步 = 6.0s，SM 100% |
| `pstress.sh` | 压力测试脚本（dmon + 打点 + 掉速检测） | 30 分钟 249W 顶墙无掉速 |

## 机器状态（2026-09）

- ubuntu1 192.168.124.10，Ubuntu 24.04，kernel 6.8，驱动 610.43.03 open
- **1× CMP 90HX 10GB**（GA102 / 50SM / cc8.6，BDF `03:00.0`），Gen2 x16 全速解锁
- FP32 ~18 TFLOPS、TF32 ~41、FP16 ~78、BF16 ~61、INT8 ~45 TOPS
- 双卡历史：GPU0 03:00.0 Gen2 x16，GPU1 05:00.0 Gen2 x8

## 对 AI 应用的结论

1. **单卡推理/绘图：PCIe 零影响**，只有加载时差几秒。
2. **TP 张量并行不可行**：双卡连线 ~2GB/s，每层 2 次 all_reduce，
   TP=2 比单卡还慢；**PP 流水线并行**更合适（搬运量小）。
3. **PD 分离**可行：KV 搬运量小（Mamba state 定长），但单卡 10G
   本身就把 prefill/decode 都塞得下，实际意义有限。
4. **10GB 显存预算**：SDXL 全家 7.2G 全量进显存；4B-AWQ 模型 3G +
   长上下文 KV（32KiB/token）也够用。

## 复现

每个子目录内有自己的 README / 脚本，环境要求见各目录。