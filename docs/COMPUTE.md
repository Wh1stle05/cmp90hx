# Compute unlock (CMP 90HX)

The CMP 90HX ships with its SM issue rate clamped: FP32 measures ~**0.72
TFLOPS**, i.e. a few percent of the 50-SM GA102's capability. The clamp is
enforced by the same PLM-protected selector block that also holds the PCIe
speed path:

| register | name | locked | unlocked |
| --- | --- | --- | --- |
| `0x0082381c` | `SS0` (issue-rate selector 0) | `0x0` | `0x88888888` |
| `0x00823820` | `SS1` (issue-rate selector 1) | `0x0` | `0x00000008` |
| `0x00823804` | `FEAT_OVR_PLM` gate | `0x0`/locked | `0xffffffff` |
| `0x00823830` | `GFX_SPEED_SELECT` (gfx clock bins) | `0x3` | `0x4` (optional, rendering) |

## How this repo unlocks it

The patched module carries pearlfortune's **V67 chain** (`0014`/`0015`,
rejoin14 multigpu-state + rejoin15 serialized-start): during GSP boot the
driver injects a crafted payload into the SEC2 Booter signature buffer, which
performs one privileged write per boot that opens `FEAT_OVR_PLM`. With the gate
open, the driver writes the full issue-rate selectors.

* This happens **automatically on the first module load of every boot** — no
  service, no user interaction, both cards.
* `cmpunlocker-rs compute90hx-v67 verify --all-cmp90hx --expect full` reports
  `PASS_CMP90HX_ALL_TARGETS_FULL_SPEED` (9/9 issue rates `full`).
* The same mechanism is what makes the PCIe mask writes possible: the module's
  Gen2 config only runs when `0x823800` (FEAT_OVR_ECC_PLM) is open, and the
  per-load "extra Booter write" slot is what the PCIe apply uses.

## Verify without external tools

`scripts/verify.sh` reads the selectors directly:

```
SS0 == 0x88888888 and SS1 == 0x00000008  -> full compute
SS0 == 0x00000000                        -> locked
```

## Measured performance (250 W cap, memory 9501 MHz)

cuBLAS 8192^3 GEMM, `tools/bench.cu`, GPU0 (50 SM, GA102):

| mode | locked | unlocked |
| --- | --- | --- |
| FP32 | ~0.72 TFLOPS | **18.24 TFLOPS** @ ~1700 MHz (84 % of 50-SM peak) |
| TF32 | — | 40.55 TFLOPS |
| FP16 | — | 77.73 TFLOPS |
| BF16 | — | 60.50 TFLOPS |
| INT8 | — | 45.16 TOPS |
| on-device copy | — | 390.9 GB/s |

For context: 50 SM / 6400 CUDA cores ≈ 74 % of an RTX 3080's core count, and
the unlocked card lands at roughly RTX 3070 Ti class FP32 throughput while
keeping the 3080's 10 GB GDDR6X / 760 GB/s memory configuration.

## Optional: GFX clock bins

`0x00823830` (`GFX_SPEED_SELECT`, stock `0x3`) clamps the graphics clock domain
to its two lowest bins. Writing `0x4` (after opening its PLM at `0x823b04`)
restores the full gfx issue rate — only relevant for desktop/rendering use,
irrelevant headless. This repo does not touch it by default.

## Scope

* Verified on `10de:220d` / `1555`, VBIOS `94.02.74.00.01`, open `610.43.03`.
* The compute unlock itself comes from
  [pearlfortune/cmpunlocker](https://github.com/pearlfortune/cmpunlocker);
  this repo only packages it together with the PCIe Gen2 work.
