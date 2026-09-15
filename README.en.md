[中文](README.md) | English

# CMP 90HX full unlock — compute + PCIe Gen2

One repo for unlocking NVIDIA **CMP 90HX** mining cards
(`10de:220d` / subsystem `10de:1555`, VBIOS `94.02.74.00.01`/`.05`) on Linux
with the NVIDIA **open** kernel modules `610.43.03`:

| Feature | Locked | Unlocked |
| --- | --- | --- |
| **SM compute** (FP32) | 0.72 TFLOPS | **18.24 TFLOPS** (full issue rate) |
| Mixed precision | — | TF32 40.6 / FP16 77.7 / BF16 60.5 TFLOPS, INT8 45.2 TOPS |
| **PCIe** | Gen1 x16 / x8, 3.08 / 1.59 GB/s H2D | **Gen2 x16 / x8, 6.29 / 3.18 GB/s H2D** |
| Memory | 10 GB GDDR6X, 9501 MHz | unchanged |
| **L2 cache** | 2.5 MiB (half of a 320-bit GA102's nominal 5 MiB) | ❌ not unlockable (GSP/VBIOS-determined, see [docs/L2-FINDINGS.md](docs/L2-FINDINGS.md)) |

No VBIOS flashing, no OTP/fuse writes — runtime register writes and a patched
kernel module only.

## Repo layout

```
scripts/ systemd/ tools/ patches/ masks/   unlock toolchain (compute + PCIe Gen2)
docs/                                      unlock internals, benchmarks, Gen2 notes, L2 findings
research/                                  single-GPU measurements: AI inference,
                                           text-to-image, stress test
    AI-capability-2026-09-14.md  full AI capability summary (LLM + image gen)
    llamacpp/   Qwen3.8-27B (2-bit GGUF): 35 tok/s decode, 32k ctx, multimodal
    comfyui/    SDXL text-to-image + self-trained chisa LoRA, 4K pipelines
    vllm/       vLLM 0.28 + Qwen3.5-4B-AWQ prefill/decode report
    p2p/        dual-GPU P2P/NCCL history (TP impractical, see README)
    notes/      context limits, ngram spec decoding, tiered bandwidth, Flash-Next quants
    pstress.sh  stress-test script
```

## AI workloads after unlock (single 10 GB card, 2026-09-14)

| Workload | Measured | Details |
|---|---|---|
| **Qwen3.8-27B (2-bit GGUF, llama.cpp)** | cold start **3.7 s**, decode **35.5 tok/s**, prefill **840 tok/s**, 32k ctx, 9.0 GB VRAM | [research/llamacpp](research/llamacpp/qwen3.8-27b-iq2/REPORT.md) |
| Vision (image Q&A) | accurate captions; 768x1024 image ~= 787 tokens | same |
| ngram-mod speculative decoding | **2.45-2.71x** on copy/rewrite tasks (zero VRAM cost) | [research/notes](research/notes/ngram_mod_bench.md) |
| **SDXL text-to-image** | 1024^2 / 24 steps **10.9 s**; batch of 4 **11.2 s/img** | [research/comfyui](research/comfyui/chisa-lora-sdxl/REPORT.md) |
| **4K output** | direct 4K **107 s**; two-step (1344x768 + 4x-UltraSharp) **19.4 s** | same |
| Self-trained character LoRA | 16 images / 960 steps / **56 min** | same |

> GPU held **SM 1830 MHz / 250 W (power cap) / <60 C** throughout, no throttling.
> Limit: on 10 GB, the 27B LLM (9.0 GB) and SDXL (9.7 GB) cannot be resident at the same time.

## How it works

```
NVIDIA open kernel module source 610.43.03
  + pearlfortune 0014/0015   -> compute unlock (V67 chain opens FEAT/SS0/SS1 each boot)
  + jdowning100 0016         -> PCIe Gen2 speed-path writes in kernel
  + this repo 0017           -> multi-round link retrain (fixes Gen1 fallback)
        |
        v
patched nvidia*.ko installed to /usr/lib/modules/$(uname -r)/updates/...
        |
        +-- compute: applied automatically on every module load (both cards)
        +-- PCIe Gen2: boot service opens 2 privilege masks per card, then
            the module retrains the link (see docs/PCIE-GEN2-FINDINGS.md)
```

* **Compute** needs no per-boot action: the V67 chain fires on the first driver
  load of every boot and reopens the issue-rate selectors (`SS0=0x88888888`,
  `SS1=0x8`).
* **PCIe Gen2** needs 2 mask writes per card (max 2 driver reloads/card).
  Masks survive warm reboots sometimes, never a real power cycle, so a boot
  service re-checks and applies only what is missing: seconds on warm reboot,
  ~1–3 minutes after a cold boot.

## Install

```sh
sudo ./scripts/install.sh      # downloads upstream, builds, installs, enables service
sudo systemctl start cmp90hx-gen2.service   # or reboot
./scripts/verify.sh            # compute + PCIe checks (run as root)
```

`install.sh` fetches the upstream pieces at install time (not vendored):

* pearlfortune/cmpunlocker v0.1.28 90HX stockflow (MIT) — patches `0014`/`0015`
* jdowning100/cmpunlocker rejoin16 patch `0016` (GPL-2.0, pinned commit)
* NVIDIA open kernel module source `610.43.03`

and applies this repo's `patches/0017-cmp90hx-gen2-retrain-retry.patch`.

## Verify / benchmark

```sh
sudo ./scripts/verify.sh                    # registers + link state
sudo ./scripts/verify.sh --bench            # also run cuBLAS + copy benchmark
```

Expected output:

```
compute  GPU0 SS0=0x88888888 SS1=0x00000008  OK (full)
compute  GPU1 SS0=0x88888888 SS1=0x00000008  OK (full)
pcie     GPU0 5.0 GT/s x16  nvidia-smi gen=2 OK
pcie     GPU1 5.0 GT/s x8   nvidia-smi gen=2 OK
```

`tools/bench.cu` (compiled by verify.sh) measures FP32/TF32/FP16/BF16/INT8 GEMM
throughput and H2D/D2H/on-device bandwidth. Reference numbers in
`docs/RESULTS.md`.

## Requirements

* CMP 90HX `10de:220d` / `1555`, VBIOS `94.02.74.00.01` or `.05`
* NVIDIA **open** kernel modules `610.43.03` installed (runfile; the installer
  checks `modinfo -F version nvidia`)
* kernel headers, `make`, `gcc`, `patch`, `setpci`, `python3`
* Secure Boot **off** (patched modules are unsigned)

## Uninstall

```sh
sudo ./scripts/uninstall.sh   # then reboot
```

## Caveats

* Bound to driver **610.43.03 open**. A different driver version needs the
  patches rebased and the 2-mask set re-verified.
* **Cold-boot order matters**: the patched module must NOT be the first nvidia
  driver load of a power cycle. Its V67 chain replaces the signature memdesc a
  plain stock GSP boot leaves behind; loading it first fails
  `RmInitAdapter (0x62:0x40:2119)` and wedges the GPU ("unexpected WPR2 already
  up") until reboot. `install.sh` therefore installs
  `/etc/modprobe.d/cmp90hx-gen2-noauto.conf` to suppress boot-time auto-load and
  lets `cmp90hx-gen2-handoff.sh` bring the card up on the stock module, hand it
  over to the patched one, and only then run the mask apply.
* **If the stock driver was installed with `--dkms`**, its modules live in
  `updates/dkms/` and depmod dedups by module name, preferring that path over
  the patched copies — the unlock is then silently lost (install.sh still
  passes). `install.sh` writes `/etc/depmod.d/cmp90hx-gen2.conf` to force the
  patched path to win, and `uninstall.sh` removes it again. Self-check with
  `modprobe --show-depends nvidia`.
* After a **kernel upgrade**, DKMS rebuilds the stock module but not the
  patched one — re-run `install.sh`, otherwise the card falls back to the
  locked stock module (0.72 TFLOPS + Gen1).
* Do not run the apply while the GPUs are busy (it reloads the driver).
* **L2 cache is a dead end**: measured **2.5 MiB** (half of a 320-bit GA102's nominal
  5 MiB) and determined by the GSP-RM firmware from the VSI (VBIOS/fuses). Host/driver
  writes are rejected by the PLM, privileged V67 writes get rewritten by the GSP, and the
  FBPA registers are hard-locked. Not a mask-gated soft lock. See
  [docs/L2-FINDINGS.md](docs/L2-FINDINGS.md).
* Out of scope: CMP 170HX (GA100), 50HX, 30HX/40HX/70HX, VBIOS `.07`
  (different unlock path), proprietary module flavour.

## Credits & license

See `CREDITS.md`. GPL-2.0 (`LICENSE`) — this work patches GPL-2.0 code
(jdowning100/cmpunlocker).
