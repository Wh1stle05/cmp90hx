# Qwen3.5-4B-AWQ 单卡 prefill / decode 报告（GPU1）

日期：2026-09-09　目录：`~/dual-90hx-research/vllm_test/qwen35_4b_bench/`

## 1. 环境

| 项目 | 值 |
|---|---|
| Host | ubuntu1（192.168.124.10），Ubuntu 24.04，kernel 6.8 |
| GPU | 2× NVIDIA CMP 90HX 10GB（GA102，50SM，cc8.6），驱动 610.43.03 open |
| 测试卡 | **GPU1 = 0000:05:00.0，Gen2 x8**（idle 37°C/75W；对照 GPU0 03:00.0 Gen2 x16 idle 60°C/83W） |
| 模型 | kumar2235/Qwen3.5-4B-AWQ，3.13GB，compressed-tensors W4A16，Marlin 内核 |
| 架构 | Qwen3_5ForCausalLM，32 层（24× linear/GDN + 8× full attention），纯文本无 Vision |
| 软件 | vLLM 0.28.0 + transformers 5.16.1 + torch 2.13 cu130 |
| vLLM 配置 | TP=1，max_model_len=8192，**CUDA Graph 开（enforce_eager=False）**，gpu util 0.9，FlashAttention-2，FlashInfer sampler 禁用（flashinfer JIT 与系统 nvcc12 不兼容，PATH 已切 cu13 nvcc；eager 模式仅用于对照） |
| 显存占用 | 权重 3.04 GiB，KV 4.43 GiB（102,400 tokens），空闲 9.48/9.65 GiB |

## 2. 方法

* `bench_prefill_decode.py`：随机 token 构造精确长度 prompt（`[500,200000)` 避开特殊 token），每组 2–3 次取 **min**（去 JIT/调度抖动）
* **关键修正**：vLLM 默认开 prefix caching + 第一版复用同一 seed，导致短 prompt 全命中缓存（128 和 4096 都是 0.09s 的假数据）；已设 `enable_prefix_caching=False` 且每 rep 换 seed
* prefill 组：prompt 128/512/2048/4096/8000，`max_tokens=1`；decode 组：prompt 128，输出 128/512（TPOT 已扣掉 prefill 时间）
* **关键修正 #2（性能根因）**：初版用 `enforce_eager=True`（因 flashinfer JIT 崩溃而加），decode 仅 **17.7 tok/s**，GPU 只吃 115W / SM 17%。实际 flashinfer 崩溃只在 **sampler**，用 `VLLM_USE_FLASHINFER_SAMPLER=0` + cu13 nvcc 即解；CUDA Graph 本身在 90HX 可用。关掉 eager 后 decode **156 tok/s（6.4ms）**，9 倍提升，GPU 顶满 249W / SM 100% / 1785MHz。eager 模式每 token 32 层×多次 kernel launch 的发射开销全部串行化，是唯一瓶颈。
* `nvidia-smi dmon -i 1` 全程记录功耗/温度（`dmon_gpu1.log`）

## 3. 结果

### prefill（TTFT≈输出1 token 总耗时）

| prompt_len | wall（min/3次） | 吞吐 |
|---|---|---|
| 128 | 0.088 s | ~90 ms 引擎开销主导 |
| 512 | 0.091 s | ~90 ms 引擎开销主导 |
| 2048 | 0.299 s | 6857 tok/s |
| 4096 | 0.603 s | 6798 tok/s |
| 8000 | 1.212 s | 6598 tok/s |

### decode

| 配置 | 输出长度 | wall | TPOT | 速度 |
|---|---|---|---|---|
| **enforce_eager（旧）** | 128 | 7.28 s | 56.6 ms | 17.7 tok/s |
| **enforce_eager（旧）** | 512 | 28.96 s | 56.5 ms | 17.7 tok/s |
| **CUDA Graph ✅** | 512 | 3.26 s | **6.4 ms** | **156 tok/s** |
| **CUDA Graph ✅ batch1** | 256 | 1.67 s | ~6.5 ms | 153 tok/s |
| **CUDA Graph ✅ batch2** | 512 | 1.82 s | — | 281 agg / 141 per |
| **CUDA Graph ✅ batch4** | 1024 | 1.91 s | — | 537 agg / 134 per |
| **CUDA Graph ✅ batch8** | 2048 | 2.17 s | — | **943 agg** / 118 per |
| 256（自然文本，旧eager） | 256 | 14.57 s | — | 17.6 tok/s |

### 功耗/温度（GPU1，dmon 2s 采样）

eager 段：平均 115W / SM 17%（半睡）。**CUDA Graph 段：249W 顶着 250W 墙、SM 100%、显存 87%、SM 时钟 1785MHz**。温度最高 50°C，跑完回落 39°C/75W。

## 4. 分析

1. **prefill 线性，无二次爆炸**：2k 以上严格线性 ~6.7k tok/s，8k 上下文 1.2 秒吃完。24/32 层是 linear/GDN attention 是主因——只有 8 层 full attention 带 O(N²)，Mamba 类 prefill 基本是线性账。做 RAG/长文，这是 90HX 上最舒服的区间。
2. **prefill 数字是 eager 模式测的**（首版脚本），2048 以上线性区 ~6.7k tok/s；短 prompt ~90ms 引擎 floor 也是 eager 代价，CUDA Graph 下会明显降低——小请求看 TTFT 请以 Graph 模式复测为准。
3. **decode：必须开 CUDA Graph（默认），勿用 `enforce_eager`**。eager 17.7 tok/s vs Graph **156 tok/s**，9 倍差。Graph 下 GPU 吃满 249W/SM100%，权重流 ~283GB/s，接近带宽上限；batch8 聚合 **943 tok/s**。此卡 decode 上限受 `GPU0 x16` 全速 ~1.4GB INT4 权重 + 显存带宽主导，单流 156 tok/s 属正常水平（llama.cpp 同级 ~100-150 tok/s）。
4. **卡很凉**：即使 Graph 全速 249W，温度仍 50°C 封顶。4B-AWQ 单卡 10G 里还剩 ~1G，`max_model_len` 拉到 16k 也还有余量。
5. **对 TP/PD 的意义**：
   * PD：Mamba 层的 recurrent state 是定长的，随上下文增长的只有 8 层 full-attention 的 KV——P→D 搬运量比同规模 Dense 小一个量级，`P2P 2.1GB/s` 搬得动；8k prompt 的 prefill 也只要 1.2s，P 卡压力小。
   * TP：decode 本来就 17 tok/s，`all_reduce 2GB/s` 再切一刀更慢；4B 这种小模型单卡跑就是最优解，别拆。

## 5. 文件与复现

```
qwen35_4b_bench/
  bench_prefill_decode.py  # 测试脚本
  results.json             # 原始数据
  bench_run.log            # 运行日志（含 sample 全文）
  dmon_gpu1.log            # 功耗/温度 trace
  REPORT.md                # 本报告
```

```bash
cd ~/dual-90hx-research/vllm_test/qwen35_4b_bench
source ~/vllm-env/bin/activate
export PATH=~/vllm-env/lib/python3.12/site-packages/nvidia/cu13/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_VISIBLE_DEVICES=1
# 注意: 不要加 enforce_eager! CUDA Graph 必须开, 否则 decode 掉 9 倍
python bench_prefill_decode.py
# 对照: diag_decode.py (eager vs graph + batch 缩放 + clock 采样)
```
