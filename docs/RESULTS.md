# Measurements

Host: Jingyue X99 Titanium D3, Ubuntu 24.04, kernel `6.8.0-139-generic`,
NVIDIA open `610.43.03`, 2x CMP 90HX (`10de:220d` / `1555`, VBIOS
`94.02.74.00.01`), 250 W power cap.

## PCIe link and copy bandwidth

`bench copyh2d` / `copyd2h` = 1 GiB pinned host buffer, repeated for 10 s.

| card | state | link | H2D | D2H |
| --- | --- | --- | --- | --- |
| GPU0 `03:00.0` | locked | Gen1 x16 (2.5 GT/s) | 3.08 GB/s | 3.35 GB/s |
| GPU0 | unlocked | **Gen2 x16 (5.0 GT/s)** | **6.29 GB/s** | **6.70 GB/s** |
| GPU1 `05:00.0` | locked | Gen1 x8 | 1.59 GB/s | 1.68 GB/s |
| GPU1 | unlocked | **Gen2 x8** | **3.18 GB/s** | **3.35 GB/s** |

Theoretical 8b/10b ceilings: Gen1 x16 3.2, Gen2 x16 8.0, Gen2 x8 4.0 GB/s.
Achieved ≈ 79–84 % of the Gen2 ceiling, same efficiency class as the Gen1
baseline (96 % of 3.2).

`nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.width.current`:

```
index, pcie.link.gen.current, pcie.link.width.current
0, 2, 16
1, 2, 8
```

## Compute (unchanged by the PCIe unlock)

`cmpunlocker-rs compute90hx-v67 verify --all-cmp90hx --expect full`:

```
TARGET_BDF=0000:03:00.0  result=PASS  PASS_CMP90HX_FULL_SPEED
TARGET_BDF=0000:05:00.0  result=PASS  PASS_CMP90HX_FULL_SPEED
PASS_CMP90HX_ALL_TARGETS_FULL_SPEED
```

cuBLAS 8192^3 GEMM, 250 W cap, memory 9501 MHz (GPU0):

| mode | rate | notes |
| --- | --- | --- |
| FP32 | 18.24 TFLOPS | @ ~1700 MHz, 84 % of 50-SM peak |
| TF32 | 40.55 TFLOPS | tensor cores |
| FP16 | 77.73 TFLOPS | tensor cores |
| BF16 | 60.50 TFLOPS | tensor cores |
| INT8 | 45.16 TOPS | tensor cores |
| device copy | 390.9 GB/s | on-GPU |

For reference, the locked state is ~0.72 TFLOPS FP32.

## Cold-boot recovery timing

Boot service on a boot where the masks were locked (worst case observed):

```
10:07:33  GPU0 masks feat_ecc=0xffffff8f xve=0xffffffcf
10:08:23  GPU0 open 0x00823800 OK (try 6)
10:08:39  GPU0 open 0x00088fe8 OK (try 1)
10:08:47  GPU0 link=5.0 GT/s gen=2
10:09:22  GPU1 open 0x00823800 OK (try 1)
10:10:17  GPU1 open 0x00088fe8 OK (try 2)
10:10:27  GPU1 link=5.0 GT/s gen=2
```

~3 minutes including flake retries; ~40 s when every cycle lands first try.
When the masks happen to survive a reboot, the service finishes in ~20 s.
