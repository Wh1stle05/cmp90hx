# Credits

| Who | What this project uses |
| --- | --- |
| [pearlfortune/cmpunlocker](https://github.com/pearlfortune/cmpunlocker) (MIT) | 90HX stockflow compute unlock, patches `0014`/`0015`, build harness |
| [jdowning100/cmpunlocker](https://github.com/jdowning100/cmpunlocker) (GPL-2.0) | rejoin16 PCIe Gen2 patch `0016`, `bar0poke.c`, 34-mask table, research notes |
| [studebaker8/cmp170hx-gen2](https://github.com/studebaker8/cmp170hx-gen2) | Retrain-Link hammer approach |
| [NatsumeAi/CMP-90HX-Compute-PCIE2.0-unlock](https://github.com/NatsumeAi/CMP-90HX-Compute-PCIE2.0-unlock) | `.07` VBIOS measurements and pointers |
| NVIDIA | Open GPU kernel modules 610.43.03 |

## What this repo adds

* **Packaging**: one installer that builds compute unlock + PCIe Gen2 into the
  same patched module and verifies both (`scripts/verify.sh`).
* **Minimal mask set**: 2 of 34 masks are sufficient for PCIe Gen2
  (`docs/FINDINGS.md`).
* **`patches/0017-cmp90hx-gen2-retrain-retry.patch`**: multi-round in-kernel
  link retrain, fixing boards where the single 2 s retrain always lost the race
  against GSP-RM and the card stayed at Gen1 throughput.
* **Robust apply tooling**: `mmap` BAR0 reads (Ubuntu 6.8 returns `EIO` on
  `read()`), `modprobe` instead of `insmod` (crypto `ecdh`/`ecc` deps),
  per-write retries, BDF auto-detection (BDFs change across reboots),
  warm-boot fast path.
