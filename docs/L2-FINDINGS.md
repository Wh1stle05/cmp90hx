# CMP 90HX L2 缓存排查：为什么打不开、以及结论

> 环境：CMP 90HX `10de:220d` / `1555`，VBIOS `94.02.74.00.01`，NVIDIA open
> `610.43.03`，kernel `6.8.0-139-generic`；算力 + PCIe Gen2 均已解锁。
> 排查日期：2026-09-15。

## TL;DR

* **实测 L2 = 2.5 MiB**（2,621,440 B）；pointer-chase 微基准的延迟拐点也在 2.5 MiB 左右，
  说明这是**真实可用容量**，不是 CUDA 误报。
* 同样是 320-bit 的 GA102（RTX 3080）标称 **5 MiB** → 90HX 只有**一半**。
* 这个值由 **GSP-RM 固件**从 **VSI（Video System Information，来源 VBIOS/熔丝）** 决定，
  存在 `pVSI->sizeL2Cache / ltcMask / ltsCount` 里。
* **驱动层 / 掩码 / 特权 V67 写都无法改变它**：
  - host（CPU/驱动）写 LTC 寄存器被 **PLM 拒绝**；
  - 特权 V67 写能落进去，但 **GSP 初始化时会重写回去**；
  - FBPA 相关寄存器（PLM 与 CFG1）对特权写也是**硬锁**（NO-EFFECT）。
* 结论：**L2 不是像 SM 算力（SS0/SS1）或 PCIe 那样的 PLM 门控软锁，驱动层解不开**。
  要动它只能改 GSP 固件或熔丝，都超出驱动层范围。

---

## 1. 实测

### 1.1 CUDA / torch API

```
torch.cuda.get_device_properties(0).L2_cache_size == 2621440   # 2.50 MiB
cudaDeviceProp.memoryBusWidth                  == 320 bits
cudaDeviceProp.multiProcessorCount             == 50
```

API 可信度校验：本机用同一 API 在 RTX 2080（TU104）上读到 `4194304`（4 MiB），
正好是 TU104 的标称 L2 → 该 API 报的是完整 L2，不存在"系统性减半"。

### 1.2 pointer-chase 微基准（`tools/l2bench.cu`）

单线程 dependent load，工作集 = `n × 4 B`，4M 次 chase，CUDA event 计时：

| 工作集 | cycles/acc | ns/acc |
|---:|---:|---:|
| 64 K | 42.6 | 24.9 |
| 128 K | 132.0 | 73.1 |
| 256 K | 198.0 | 107.3 |
| 512 K | 216.0 | 117.1 |
| 1024 K | 223.5 | 121.2 |
| 1536 K | 226.0 | 122.5 |
| 2048 K | 227.5 | 123.3 |
| 2304 K | 229.9 | 124.6 |
| **2560 K** | **258.0** | **139.8** |
| 2816 K | 319.1 | 173.0 |
| 3072 K | 357.4 | 193.7 |
| 3584 K | 398.7 | 216.1 |
| 4096 K | 419.6 | 227.4 |
| 8192 K | 463.7 | 251.4 |
| 16384 K | 477.2 | 258.6 |
| 32768 K | 482.6 | 261.6 |

拐点：≤ 2.25 MiB 稳定在 ~230 cyc（L2 命中），**2.56 MiB 起明显爬升到 ~480 cyc（DRAM）**
→ 有效 L2 ≈ **2.5 MiB**。

### 1.3 标称对比

| | L2 |
|---|---|
| CMP 90HX（实测） | **2.5 MiB** |
| CMP 90HX（TechPowerUp / technical.city 标称） | 5 MB |
| RTX 3080（GA102-200，同样 320-bit） | 5 MiB |
| RTX 3090（GA102-300，384-bit） | 6 MiB |

→ 实测正好是标称的一半，与"GA102-100 是筛过的矿卡 SKU"一致。

---

## 2. 来源：为什么不是掩码能开的软锁

在完整 RM 源码（`NVIDIA-kernel-module-source-610.43.03`）里：

```
src/nvidia/src/kernel/gpu/mem_sys/kern_mem_sys_ctrl.c
  93:  NV2080_CTRL_FB_INFO_INDEX_LTC_COUNT   -> nvPopCount64(pVSI->ltcMask)
  98:  NV2080_CTRL_FB_INFO_INDEX_LTS_COUNT   -> pVSI->ltsCount
 113:  NV2080_CTRL_FB_INFO_INDEX_L2CACHE_SIZE-> pVSI->sizeL2Cache
1719:  pParams->ltcCount = pVSI->fbLtcInfoForFbp[fbpIndex].ltcCount
```

* L2 分片信息全部来自 **VSI**；GSP 模式下 VSI 由 **GSP-RM 固件**填好后交给主机
  （host 侧只有 vgpu/ctrl 在读它）。
* GA102 的分片配置见
  `src/nvidia/src/kernel/gpu/mem_sys/arch/ampere/kern_mem_sys_ga102.c`
  （按 `ltsPerLtcCount × ltcCount` ∈ {48, 40, 32, 24} 区分）。
* **90HX 的驱动源码里没有任何写 `0x00100ce0` / `0x009a0204` 的代码**（已 grep 全树）。
  → 这两个值是 **GSP/SEC2** 写的，所以"特权写先进去、GSP 再盖回来"完全说得通。
* 对比 170HX（GA100）：它的 `sec2-postbl-plm-ss-cfg.patch` 在
  `kgspSec2PostblTiming` 里写 `FBPA_CFG1` + `LTC_LMR`，**前提是先打开 FBPA PLM
  (`0x009a0148`)**。90HX 的 FBPA PLM 是硬锁，照搬不了。

---

## 3. 试写实验（驱动/特权路径）

方法：用现成的 V67 特权 BAR0 写通道
（`scripts/rejoin16-cycle.sh <addr> <value>`，每次驱动重载让 Falcon 载荷做一次特权写，
可绕过 PLM 保护），每次重载后读回 + 量 L2 + 查显存。

| 目标寄存器 | 写入值 | V67 特权写 | 重载后读回 | L2 |
|---|---|---|---|---|
| `FEAT2 0x00823b00` | `0xffffffff` | ✅ OK（readback=0xffffffff） | **0x00004bff**（被 GSP 重写） | 2.5 MiB |
| `FBPA PLM 0x009a0148` | `0xffffffff` | ❌ NO-EFFECT | **0xffffff0f**（硬锁） | 2.5 MiB |
| `LTC_LMR 0x00100ce0` | `0x0000028a` | ✅ OK（readback=0x0000028a） | **0x00000288**（被 GSP 重写） | 2.5 MiB |
| `FBPA_CFG1 0x009a0204` | `0x4266a001`（探测） | ❌ NO-EFFECT | **0x4266a000**（硬锁） | 2.5 MiB |
| `0x00100ce0`（CPU 直写） | 任意 | ❌ REJECTED（PLM 保护） | 不变 | 2.5 MiB |

基线寄存器（解锁后、未做任何试写）：

```
0x00823800 FEAT_OVR_ECC_PLM = 0xffffffff   (open)
0x00823804 FEAT             = 0xffffffff   (open)
0x0082381c SS0              = 0x88888888   (full)
0x00823820 SS1              = 0x00000008   (full)
0x00823b00 FEAT2            = 0x00004b8f
0x00823810 PCIE_FUSE        = 0x002aaaaa
0x009a0148 FBPA PLM         = 0xffffff0f
0x009a0204 FBPA_CFG1        = 0x4266a000
0x00100ce0 LTC_LMR          = 0x00000288
```

**关键现象**：特权写确实落进了寄存器（readback 立刻变成新值），但 GSP 初始化时把它重写回
原值 → 说明这个寄存器由 GSP 在初始化阶段按 VSI 重新编程，早期写没有意义；
而 FBPA 那一组连特权写都硬锁。

---

## 4. 结论

1. CMP 90HX 的 **L2 = 2.5 MiB**，是 320-bit GA102 标称值（5 MiB）的一半。
2. 该配置由 **GSP-RM 固件 + VBIOS/熔丝**决定，经 `pVSI` 上报主机。
3. 驱动层（host 写 / V67 特权写 / PLM 掩码）**都改不了**：
   * host 写被 PLM 拒；
   * 特权写会被 GSP 重写；
   * FBPA PLM/CFG1 硬锁。
4. 与 SM 算力（`SS0/SS1`，软锁）和 PCIe 速率（PLM 掩码，软锁）不同，**L2 不是掩码型软锁**。
   上排 170HX 那套 `FBPA_CFG1/LTC_LMR` 内存几何解锁在 90HX 上不可复制。
5. 理论上的剩余路径只有 **改 GSP 固件**（签名）或 **改熔丝**（不可写）——都超出驱动层，
   实际不可行。**不建议继续投入**（刷 VBIOS 也解决不了：L2 配置在 GSP/熔丝里，且 Ampere
   VBIOS 有签名校验、变砖风险高）。

---

## 5. 复现

```sh
# L2 数值
python3 -c "import torch;print(torch.cuda.get_device_properties(0).L2_cache_size)"

# 容量拐点
nvcc -O3 -arch=sm_86 tools/l2bench.cu -o /tmp/l2bench && /tmp/l2bench

# 试写（一次一个驱动重载周期；先停掉占用 /dev/nvidia* 的进程，如 nvtop）
sudo CMP90_BDF=0000:03:00.0 bash scripts/rejoin16-cycle.sh 0x00100ce0 0x0000028a
sudo python3 scripts/maskread.py 0000:03:00.0 0x00100ce0     # 重载后读回

# 清理（避免下次开机残留写入）
sudo rm -f /var/lib/cmpunlocker-rs/rejoin16-next-write.bin
```

**注意事项**

* 全部试写都是**易失**的：重载/冷启动即恢复，不碰 OTP/熔丝，**不会造成永久损伤**。
* 试写期间出现过一次 `RmInitAdapter failed! (0x62:0x62:2119)`（FBPA_CFG1 探测那次），
  经 rejoin15 的串行重试后自动恢复；事后已用寄存器读回 + CUDA 显存自检确认正常。
* GPIO/CPU 直写这些寄存器会被 PLM 拒绝；只有 V67 特权路径（Falcon）能绕过 PLM。
