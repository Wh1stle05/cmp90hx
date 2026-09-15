中文 | [English](README.en.md)

# CMP 90HX 全解锁总仓库 —— 算力 + PCIe Gen2 + 单卡实测

一个仓库搞定 NVIDIA **CMP 90HX** 矿卡（`10de:220d` / 子系统 `10de:1555`，
VBIOS `94.02.74.00.01`/`.05`）在 Linux + NVIDIA **open** 内核模块
`610.43.03` 下的全部解锁，以及解锁后的实测数据与可用工作负载
（AI 推理 / 文生图 / 压力测试）。

| 项目 | 锁定状态 | 解锁后 |
| --- | --- | --- |
| **SM 算力**（FP32） | 0.72 TFLOPS | **18.24 TFLOPS**（满发射率） |
| 混合精度 | — | TF32 40.6 / FP16 77.7 / BF16 60.5 TFLOPS，INT8 45.2 TOPS |
| **PCIe** | Gen1 x16 / x8，H2D 3.08 / 1.59 GB/s | **Gen2 x16 / x8，H2D 6.29 / 3.18 GB/s** |
| 显存 | 10GB GDDR6X，9501MHz | 不变 |
| **L2 缓存** | 2.5 MiB（320-bit GA102 标称 5 MiB，实为一半） | ❌ 不可解锁（GSP/VBIOS 决定，见 [docs/L2-FINDINGS.md](docs/L2-FINDINGS.md)） |

不刷 VBIOS、不写 OTP/熔丝——只有运行时寄存器写入 + 打补丁的内核模块。

## 解锁后的 AI 实测（单卡 10GB，2026-09-14）

| 能力 | 实测结果 | 详见 |
|---|---|---|
| **Qwen3.8-27B（2bit GGUF）** | 冷启动 **3.7 s**、生成 **35.5 tok/s**、预填充 **840 tok/s**、上下文 **32k**、显存 9.0 GB | [research/llamacpp](research/llamacpp/qwen3.8-27b-iq2/REPORT.md) |
| 多模态（看图） | 图片描述准确，768×1024 图 ≈ 787 token | 同上 |
| ngram-mod 投机解码 | 复制/改写类任务 **2.45–2.71×**（零显存） | [research/notes](research/notes/ngram_mod_bench.md) |
| **SDXL 出图** | 1024² / 24 步 **10.9 s**，批量 4 张 **11.2 s/张** | [research/comfyui](research/comfyui/chisa-lora-sdxl/REPORT.md) |
| **4K 出图** | 直接 4K **107 s**；两步法（1344×768 + 4x-UltraSharp）**19.4 s** | 同上 |
| 自训角色 LoRA | 16 张图 / 960 步 / **56 分钟**，角色一致性明显 | 同上 |

> 出图全程 GPU 稳定 **SM 1830 MHz / 250 W 顶墙 / <60 °C**，无降频。
> 限制：10GB 显存下 27B-LLM（9.0 GB）与 SDXL（9.7 GB）**不能同时常驻**；上下文 32k 是显存边界。

## 仓库结构

```
├── scripts/ systemd/ tools/ patches/ masks/   解锁工具链（算力 + PCIe Gen2）
├── docs/                                      解锁原理、基准、Gen2 找坑记录、L2 排查结论
└── research/                                  单卡实测：AI 推理 / 文生图 / 压测
    ├── AI-capability-2026-09-14.md  单卡 AI 能力总汇（LLM + 出图，含视频用结论）
    ├── llamacpp/   llama.cpp 跑 Qwen3.8-27B(2bit)：35 tok/s 生成 / 32k 上下文 / 多模态
    ├── comfyui/    ComfyUI 0.35 + SDXL：部署首测 + chisa LoRA 出图与 4K 实测
    ├── vllm/       vLLM 0.28 + Qwen3.5-4B-AWQ prefill/decode 报告
    ├── p2p/        双卡 P2P/NCCL 历史数据（TP 不可行，见 README）
    ├── notes/      专题：上下文上限、ngram 投机解码、分层带宽、Flash-Next 量化/MTP
    └── pstress.sh  压力测试脚本
```

## 原理

```
NVIDIA open 内核模块源码 610.43.03
  + pearlfortune 0014/0015   -> 算力解锁（V67 链每次开机打开 FEAT/SS0/SS1）
  + jdowning100 0016         -> 内核态写 PCIe Gen2 速率路径
  + 本仓库 0017              -> 多轮链路重训（修复回落 Gen1 的竞争问题）
        |
        v
补丁 nvidia*.ko 装入 /usr/lib/modules/$(uname -r)/updates/...
        |
        +-- 算力：每次模块加载自动生效（单/多卡均适用）
        +-- PCIe Gen2：开机服务每卡写 2 个权限掩码，再由模块完成重训
```

* **算力**无需每次开机额外操作：V67 链在每次开机的首次驱动加载时自动重开
  发射率选择器（`SS0=0x88888888`、`SS1=0x8`）。
* **PCIe Gen2** 每卡需要 2 次掩码写入（最多 2 次驱动重载/卡）。掩码有时能跨
  warm reboot 保留、断电必然重锁，所以开机服务每次检查、只补缺的：
  热重启几秒，冷启动约 1–3 分钟。

## 安装

```sh
sudo ./scripts/install.sh      # 下载上游、编译、安装、启用服务
sudo systemctl start cmp90hx-gen2.service   # 或直接重启
./scripts/verify.sh            # 验证算力 + PCIe（需要 root）
```

`install.sh` 在安装时拉取上游（不 vendor）：

* pearlfortune/cmpunlocker v0.1.28 90HX stockflow（MIT）——补丁 `0014`/`0015`
* jdowning100/cmpunlocker rejoin16 补丁 `0016`（GPL-2.0，固定 commit）
* NVIDIA open 内核模块源码 `610.43.03`

并应用本仓库的 `patches/0017-cmp90hx-gen2-retrain-retry.patch`。

## 验证 / 基准

```sh
sudo ./scripts/verify.sh                    # 寄存器 + 链路状态
sudo ./scripts/verify.sh --bench            # 额外跑 cuBLAS + 拷贝带宽
```

预期输出（当前为单卡，示例为双卡历史）：

```
compute  GPU0 SS0=0x88888888 SS1=0x00000008  OK (full)
compute  GPU1 SS0=0x88888888 SS1=0x00000008  OK (full)
pcie     GPU0 5.0 GT/s x16  nvidia-smi gen=2 OK
pcie     GPU1 5.0 GT/s x8   nvidia-smi gen=2 OK
```

`tools/bench.cu`（verify.sh 会自动编译）测 FP32/TF32/FP16/BF16/INT8 GEMM 与
H2D/D2H/板载带宽，参考数据见 `docs/RESULTS.md`；解锁后的实际工作负载
（vLLM 推理 / ComfyUI 文生图 / 压力测试）见 `research/`。

## 环境要求

* CMP 90HX `10de:220d` / `1555`，VBIOS `94.02.74.00.01` 或 `.05`
* NVIDIA **open** 内核模块 `610.43.03`（runfile 安装；脚本会检查
  `modinfo -F version nvidia`）
* 内核头文件、`make`、`gcc`、`patch`、`setpci`、`python3`
* **关闭 Secure Boot**（补丁模块未签名）

## 卸载

```sh
sudo ./scripts/uninstall.sh   # 然后重启
```

## 注意

* 绑定驱动 **610.43.03 open**；换驱动版本需 rebase 补丁并重新验证 2 掩码。
* **冷启动顺序很重要**：补丁模块**不能**是一个电源周期里第一个加载的 nvidia
  驱动——它的 V67 链要替换 stock GSP boot 留下的 signature memdesc，抢先加载会
  `RmInitAdapter failed (0x62:0x40:2119)` 并把 GPU 卡在 WPR2（只能重启恢复）。
  因此 install.sh 会装 `/etc/modprobe.d/cmp90hx-gen2-noauto.conf` 禁止开机
  自动加载，改由服务 `cmp90hx-gen2-handoff.sh` **先用 stock 模块把卡点起来，
  再交接给补丁模块**，最后才写掩码。
* **GPU 负载必须排在解锁服务【之后】启动**：解锁服务要卸载/重载 nvidia 模块，
  如果有进程占着 `/dev/nvidia*`（ComfyUI 一启动初始化 CUDA 就会），
  `modprobe -r` 会失败 —— 而 `rejoin16-cycle.sh` 已经先把算力选择器重置为 0，
  结果是**算力解锁丢失 + 掩码写不进 + PCIe 停在 Gen1**（实测症状：
  服务日志 `open 0x00823800 FAIL (readback=0xffffff8f)`，dmesg 里只有 2 次
  `NVRM: loading`，没有第 3 次重载）。
  正确做法是给 GPU 服务加 drop-in：
  ```ini
  [Unit]
  After=cmp90hx-gen2.service
  Wants=cmp90hx-gen2.service
  ```
  ⚠️ 但本服务的单元里**不能**写 `After=multi-user.target`（与 `WantedBy=multi-user.target`
  构成顺序环，systemd 会直接删除 GPU 服务的启动任务）。本仓库的单元已移除该行。
* **stock 驱动用 `--dkms` 安装时**，其模块位于 `updates/dkms/`，depmod 会按模块名
  去重并优先选它，导致补丁模块被静默覆盖（脚本报成功、实际没解锁）。
  `install.sh` 会写入 `/etc/depmod.d/cmp90hx-gen2.conf` 强制补丁版优先，
  `uninstall.sh` 会一并删除；可用 `modprobe --show-depends nvidia` 自检。
* **内核升级后**：DKMS 会自动重编 stock 模块，但补丁模块不会——必须重跑
  `install.sh`，否则重启回落到 stock 模块（0.72 TFLOPS + Gen1）。
* apply 期间会多次重载驱动，别在 GPU 有任务时跑。
* **L2 缓存已排查，不可解锁**：实测 **2.5 MiB**（320-bit GA102 标称 5 MiB，正好一半），
  由 GSP-RM 固件从 VSI（VBIOS/熔丝）决定。host/驱动写被 PLM 拒、特权 V67 写会被 GSP
  重写、FBPA 组硬锁——不是掩码型软锁。详见 [docs/L2-FINDINGS.md](docs/L2-FINDINGS.md)。
* 不在范围内：CMP 170HX（GA100）、50HX、30HX/40HX/70HX、VBIOS `.07`
  （另一套解锁路径）、专有模块 flavor。

## 致谢与许可

见 `CREDITS.md`。GPL-2.0（`LICENSE`）——本项目修改了 GPL-2.0 代码
（jdowning100/cmpunlocker）。
