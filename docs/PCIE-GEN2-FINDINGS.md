# Findings

Environment: CMP 90HX `10de:220d` / `10de:1555`, VBIOS `94.02.74.00.01`,
NVIDIA open `610.43.03`, kernel `6.8.0-139-generic`, board: Jingyue X99
Titanium D3 (slots Gen3 x16/x8/x8).

## 1. Minimal mask set: 2 of 34

Method: masks, once opened, cannot be closed without a reboot, so instead of
independent subsets we used an **ordered list and checked the link state after
every reload cycle**. The patched module applies the Gen2 speed path as soon as
the required masks are open, so the first cycle after which the link reports
`5.0 GT/s` + `pcie.link.gen.current = 2` reveals a sufficient prefix.

Observed (clean card, masks all locked):

| set opened (cumulative, in order) | cycles | link |
| --- | --- | --- |
| `{0x823800}` | 1 | 2.5 GT/s |
| `{0x823800, 0x88fe8}` | 2 | **5.0 GT/s, gen=2** |
| `{0x823800, 0x88fe8, 0x88fec}` | 3 | 5.0 GT/s |
| `{0x823800} + XVE x6` | 7 | 5.0 GT/s |
| `{0x823800} + XVE x6 + XP3G x17` (no OPTB) | 24 | 5.0 GT/s |

Decisive single-mask test on a second clean card: `{0x823800}` alone never
reaches Gen2 (final retrain leaves 2.5 GT/s) → the XVE mask `0x88fe8` is
required. Therefore:

* **required:** `0x823800` (`FEAT_OVR_ECC_PLM`, module gate) and `0x88fe8`
  (XVE privilege mask covering `PRIV_MISC_1` @ `0x8841c` and the LTSSM override
  @ `0x8872c`).
* **not required for Gen2:** all XP3G masks (`0x8e1b0`–`0x8e1f0`), all OPTB
  masks (`0x8200d0`–`0x8200f4`), the other XVE masks (`0x88fec`, `0x88ff0`,
  `0x88ff4`, `0x88ff8`, `0x88ab4`), and `0x823b04` (that one only matters for
  the GFX clock-bin select, not PCIe).

Result: 34 → 2 reload cycles per card. Cold-boot apply drops from ~10 min/card
to ~20–40 s/card.

## 2. In-kernel retrain race (patch 0017)

The original rejoin16 module retrains the link once, polling LNKSTA for 2 s.
On this board that always lost the race against GSP-RM link management:

```
NVRM: GPU 0000:03:00.0: CMP90 Gen2: retrain completed without Gen2 link (status=0x1101)
```

Userspace `setpci` retrains (target speed + bridge RL bit) then succeeded, but
the link only reached Gen2 **after** RM had already initialised at Gen1, so
`nvidia-smi` still reported gen 1 and DMA stayed at Gen1 throughput
(3.08 GB/s = Gen1 x16 ceiling) even though config space showed 5.0 GT/s.

Fix (`patches/0017-cmp90hx-gen2-retrain-retry.patch`): retrain in up to 5
rounds, each round re-asserting the upstream bridge Retrain-Link bit and
polling for 4 s. With this, the module's own retrain succeeds during the
reload window, RM initialises at Gen2, and DMA doubles:

```
copyh2d 3.08 -> 6.29 GB/s, copyd2h 3.35 -> 6.70 GB/s   (GPU0, x16)
copyh2d 1.59 -> 3.18 GB/s, copyd2h 1.68 -> 3.35 GB/s   (GPU1, x8)
```

## 3. Other environment fixes

* `rejoin16-apply-all.sh` preflight read BAR0 through a plain `read()` of
  `resource0`; on Ubuntu 6.8 that returns `EIO`. The apply scripts here use
  `mmap()` (`maskread.py`).
* `rejoin16-cycle.sh` used `insmod` on the artifact, which does not resolve
  module dependencies — `nvidia` failed with `Unknown symbol
  crypto_ecdh_shared_secret`/`ecc_*` when the crypto modules were not already
  loaded. The version here uses `modprobe nvidia`, which pulls `ecdh`/`ecc`
  (and DRM) automatically.
* The second card's PCIe BDF changes across reboots on this board
  (`04:00.0` ↔ `05:00.0`), so every script auto-detects BDFs via
  `lspci -Dnn | awk '/10de:220d/'`.
* Warm reboots may or may not keep the masks latched (observed both), so the
  boot service always re-checks and opens only what is missing.
* 2026-09-09: `rejoin16-cycle.sh` re-lock moved **before** `modprobe -r`
  (with `maskread.py` readback check + `restore_full` safety net). After an
  `apt --fix-broken` pulled in `libnvidia-compute-580/535` and regenerated
  initramfs, the device drops into a low-power state once the driver is
  unloaded (BAR0 reads `0xffffffff`, every write `REJECTED`), so the old
  order (unload → re-lock) silently left SS0/SS1 full, V67 logged
  "already present" and skipped the Booter chain (`(no REJOIN16 lines!)`
  + PCIe FAIL on every cycle). Pre-unload re-lock makes REJOIN16 fire on
  every cycle. Side observation: the first Booter write to a fresh mask
  often reports `NO-EFFECT (polls=1000)` while an identical second write
  reports `OK (polls=40-54)` — the 6-tries-per-mask loop in
  `cmp90hx-gen2-minimal.sh` covers this.

## 4. Board topology

The two cards sit in the x16 and x8 slots; both train Gen2 after unlock:

```
03:00.0  LnkCap: Speed 5GT/s, Width x16   LnkSta: Speed 5GT/s, Width x16
05:00.0  LnkCap: Speed 5GT/s, Width x16   LnkSta: Speed 5GT/s, Width x8 (downgraded)
```

The card caps at Gen2; the Gen3-capable slots are not a bottleneck, the x8
slot simply cannot give x16 lanes.
