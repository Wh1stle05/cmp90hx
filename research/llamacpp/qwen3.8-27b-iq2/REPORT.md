# Qwen3.8-27B（2bit GGUF）单卡 llama.cpp 实测报告

日期：2026-09-14　机器：ubuntu1（192.168.124.10）　GPU：CMP 90HX 10GB（BDF `03:00.0`，Gen2 x16）
脚本：`scripts/llama_data.py`　原始数据：`data/gpu_llama_bench_20260914_084045.json`

## 1. 环境

| 项目 | 值 |
|---|---|
| 运行时 | **llama.cpp b10920**，源码自编译（`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86`，CUDA 13.0，nvcc 在 `/usr/local/cuda-13.0`） |
| 模型 | `unsloth/Qwen3.8-27B-GGUF` → **UD-IQ2_XXS 6.76 GiB**（2.0625 bpw / 26.90B 参数） |
| 视觉塔 | `ggml-org/Qwen3.8-27B-GGUF` → `mmproj-Qwen3.8-27B-Q8_0.gguf` 0.63 GB |
| 服务参数 | `-ngl 99 -c 32768 -ctk q8_0 -ctv q8_0 -b 256 -ub 256 --parallel 1 --jinja --spec-type ngram-mod` |
| 驱动 | 610.43.03（已解锁算力 + PCIe Gen2） |

### 为什么必须 2bit

| 方案 | 权重体积 | 10GB 卡 |
|---|---|---|
| 官方 fp16/bf16 | **55.6 GB** | ❌ 需 ≥64GB 显存 |
| AWQ/INT4、EXL3-3.5bpw | 16–19 / 11.5 GB | ❌ |
| GGUF Q4_K_M | 19 GB | ❌ |
| **IQ2_XXS（本方案）** | **6.76 GiB** | ✅ 峰值 9.0–9.9 GB |

> Qwen3.8-27B 是 dense 模型（64 层 = 48×Gated DeltaNet 线性注意力 + 16×Gated Attention），
> 另带视觉塔 0.46B 与 MTP 头 0.43B。**dense 每 token 都要读全部权重，没有"冷权重"可下放**，
> 所以 10GB 下量化到 2bit 是唯一出路（详见 `../notes/moe_offload_bandwidth.md`）。

## 2. 加载与显存

| 指标 | 实测 |
|---|---|
| 冷启动（systemd start → `/health` ok） | **3.69 s**（含从 NVMe 读 6.76 GiB + CUDA 初始化） |
| 常驻显存 | **9012 MiB / 10240** |
| 显存峰值（图像输入 + 32k 上下文） | 9258 MiB |
| 主机内存 | ≈3 GB（含 ngram 投机池） |

## 3. 速度

### llama-bench（`-ngl 99` 全量上卡）

| KV 精度 | pp512 | pp4096 | tg（生成） |
|---|---|---|---|
| f16 | 797.3 tok/s | 841.8 tok/s | **35.80 tok/s** |
| **q8_0（生产配置）** | 793.4 tok/s | 838.8 tok/s | **35.43 tok/s** |

有效显存带宽：6.76 GiB × 35.5/s ≈ **258 GB/s**（标称 760 GB/s，2bit 反量化 kernel 只用出约 1/3）。

### 首 token 延迟（TTFT）

| 输入长度 | TTFT | 预填充 | 生成 |
|---|---|---|---|
| 112 token | **0.40 s** | 280 tok/s | 33.3 tok/s |
| 2,511 token | **3.29 s** | 764 tok/s | 33.0 tok/s |
| 30,011 token | **43.3 s** | 694 tok/s | 26.7 tok/s |

短问答：`17*23=?` → 4 token，**1.16 s** 返回「391」。

## 4. 上下文与单次请求上限（实测边界）

| 项目 | 数值 |
|---|---|
| 上下文窗口 | **32,768 token**（模型原生 262,144，受 10GB 显存限制） |
| 最大输入 | **≈32,745**，超出直接 HTTP 400 `exceed_context_size_error`（实测 32831 被拒） |
| 最大输出 | 32,768 − 输入 − 4；实测 input 31027 + output 1737 = **32764 顶格**（finish_reason=length） |
| f16 KV 可跑到 | 8k / 16k ✅；32k ❌ OOM（**必须 q8_0 KV**）；64k 即使 q8_0 也 OOM |
| pi（编程 Agent）固定开销 | ≈2,500 token（系统提示 + 工具定义 + AGENTS.md） |
| 图像 token 开销 | **每 32×32 像素 = 1 token**；768×1024 → 787；1280×1800 → 2259；大图自动缩放到 ≈4,000 上限 |

## 5. ngram-mod 投机解码（零显存、零下载）

同模型同配置，仅加 `--spec-type ngram-mod`：

| 任务类型 | baseline | ngram-mod | 加速比 |
|---|---|---|---|
| **逐字复述**（原样重复输入） | 35.31 | **86.57** | **2.45×** |
| 按提示补全（续写上下文原句） | 35.46 | **49.99** | 1.41× |
| 代码改写（变量改名，代码照抄） | 35.23 | **43.12** | 1.22× |
| 逐句翻译（英→中） | 35.37 | 35.43 | 1.00× |
| 开放推理生成 | 35.57 | 34.62 | 0.97× |
| 调参 `--spec-ngram-mod-n-match 8` 后逐字复述 | 35.31 | **95.83** | **2.71×** |
| 极限案例（`ignore_eos` 长输出 + 重复内容） | ~35 | **231.4** | **6.6×** |

结论：**复制/改写/抽取类工作流收益巨大，自由生成损失 ≤3%，建议常开**。完整基准见 `../notes/ngram_mod_bench.md`。

## 6. 多模态

- 输入 `/data3/原画/8.jpg`（768×1024 动漫图），问「头发上系着什么颜色的丝带？」→ 答 **「红色。」**（正确）
- 耗时 3.3 s（836 prompt token，prefill 451 tok/s，decode 34.6 tok/s），显存峰值 9258 MiB
- 完整描述测试：黑长直+红丝带、红瞳、黑 choker、水手服红领结、深色夹克带绑带、百褶裙、
  背景虚化绿叶粉花、逆光 —— **与真实图片逐项核对全部正确，无幻觉**

## 7. 落地形态

- `llama-server` 提供 OpenAI 兼容 `/v1`：流式、usage、`reasoning_content`（思考链）、`tools`（函数调用）全可用
- 已接入 **pi** 编程 Agent：模型自己调用 bash 工具执行 `uname -r` 并正确回答 → 工具调用链路验证通过
- 思考开关：`chat_template_kwargs.enable_thinking=false` 或 `reasoning_effort=low|medium|xhigh`（模板支持值）
- 服务别名 `qwen3.8-27b`，监听 `0.0.0.0:8080`

## 8. 复现

```bash
# 编译（CUDA 13）
cd ~/llama.cpp && CUDACXX=/usr/local/cuda/bin/nvcc cmake -B build -G Ninja \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build -j 20 --target llama-cli llama-server llama-mtmd-cli llama-bench

# 启动服务
bash scripts/run_llama_server.sh      # 等价于 scripts/ 内的参数说明

# 采集本报告数据
python3 scripts/llama_data.py
```
