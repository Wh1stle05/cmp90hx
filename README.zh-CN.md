# CMP 90HX PCIe Gen2 解锁 —— 最小 2 掩码版

给 NVIDIA CMP 90HX 矿卡（`10de:220d` / 子系统 `10de:1555`，VBIOS
`94.02.74.00.01`/`.05`）解锁 **PCIe Gen2（5.0 GT/s）**，基于 NVIDIA **Open**
内核模块 `610.43.03`，叠加在 pearlfortune 算力解锁之上。

本仓库把公开的 **34 掩码** 方案压缩为**每卡 2 个掩码**，并修复了部分主板上
"链路训不到 Gen2"的重训竞争问题。

* 不刷 VBIOS、不写 OTP/熔丝，只做运行时寄存器写入。
* 算力解锁全程保持（每一步都验证 `PASS_CMP90HX_ALL_TARGETS_FULL_SPEED`）。

## 实测结果（Ubuntu 24.04，内核 6.8.0-139-generic，250W 上限）

| | 解锁前 | 解锁后 |
| --- | --- | --- |
| GPU0 `03:00.0` | Gen1 x16，H2D 3.08 GB/s | **Gen2 x16，H2D 6.29 / D2H 6.70 GB/s** |
| GPU1 `05:00.0` | Gen1 x8，H2D 1.59 GB/s | **Gen2 x8，H2D 3.18 / D2H 3.35 GB/s** |
| 算力（两张） | full | full |

`nvidia-smi` 的 `pcie.link.gen.current` 变为 `2`。

## 两个掩码

| 地址 | 名称 | 作用 |
| --- | --- | --- |
| `0x00823800` | `FEAT_OVR_ECC_PLM` | 补丁模块写 Gen2 速率路径前的总闸门 |
| `0x00088fe8` | XVE 权限掩码 | 解锁 `PRIV_MISC_1` / `LTSSM` 速率路径寄存器 |

原 rejoin16 表里其余 32 个掩码（XP3G×17、OPTB×10、XVE×5）**对 Gen2 非必需**，
二分过程见 `docs/FINDINGS.md`。

## 安装

```sh
sudo ./scripts/install.sh
sudo systemctl start cmp90hx-gen2.service   # 或直接重启
```

`install.sh` 会为当前内核现场编译补丁模块并装入
`/usr/lib/modules/$(uname -r)/updates/cmpunlocker-90hx-stockflow`（depmod 覆盖
stock 模块），然后安装开机服务。

## 运行机制

* **每次开机**服务都会检查两张卡：若 2 个掩码已开（warm reboot 有时会保留），
  只做重训，几秒完成。
* 若掩码被重新锁上（断电冷启动必然如此），对每个关闭的掩码跑一次驱动重载
  （每卡最多 2 次，每次约 20 秒），随后补丁模块在内核态完成重训。
  **两张卡冷启动恢复通常 1–3 分钟。**
* BDF 每次自动探测——本机上第二张卡的 BDF 会随重启变化。

## 验证

```sh
cat /sys/bus/pci/devices/$(lspci -Dnn | awk '/10de:220d/{print $1; exit}')/current_link_speed   # 5.0 GT/s
nvidia-smi --query-gpu=index,pcie.link.gen.current --format=csv                                   # 2
sudo ./cmpunlocker-rs compute90hx-v67 verify --all-cmp90hx --expect full                          # PASS
```

## 卸载

```sh
sudo ./scripts/uninstall.sh   # 然后重启
```

## 注意

* 换内核或换驱动后必须重跑 `install.sh` 重新编译。
* apply 期间会多次重载驱动，别在 GPU 有任务时跑。
* 解锁是运行时状态、不写固件；靠开机服务跨重启存活。

## 致谢

* [pearlfortune/cmpunlocker](https://github.com/pearlfortune/cmpunlocker) ——
  90HX 算力解锁（rejoin14/15）。
* [jdowning100/cmpunlocker](https://github.com/jdowning100/cmpunlocker) ——
  rejoin16 PCIe Gen2 补丁、`bar0poke.c` 与 34 掩码表。
* [studebaker8/cmp170hx-gen2](https://github.com/studebaker8/cmp170hx-gen2) ——
  Retrain-Link 思路。
* NVIDIA —— Open 内核模块 610.43.03。

## 许可

GPL-2.0（见 `LICENSE`）——因为本项目基于并修改 GPL-2.0 代码
（jdowning100/cmpunlocker）。
