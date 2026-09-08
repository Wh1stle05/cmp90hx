# CMP 90HX PCIe Gen2 unlock — minimal 2-mask apply

Unlocks **PCIe Gen 2 (5.0 GT/s)** on NVIDIA CMP 90HX mining cards
(`10de:220d` / subsystem `10de:1555`, VBIOS `94.02.74.00.01`/`.05`) running the
NVIDIA **Open** kernel modules `610.43.03`, on top of the pearlfortune compute
unlock.

This repo reduces the PCIe part from the published **34-mask** apply to
**2 masks per card**, and fixes a retrain race that prevented the link from
coming up at Gen2 on some boards.

* No VBIOS flashing, no OTP/fuse writes. Runtime register writes only.
* Compute unlock (full SM issue rate) is preserved — verified after every step.

## Results (measured, Ubuntu 24.04, kernel 6.8.0-139-generic, 250 W cap)

| | before (locked) | after |
| --- | --- | --- |
| GPU0 `03:00.0` | Gen1 x16, 3.08 GB/s H2D | **Gen2 x16, 6.29 GB/s H2D / 6.70 GB/s D2H** |
| GPU1 `05:00.0` | Gen1 x8, 1.59 GB/s H2D | **Gen2 x8, 3.18 GB/s H2D / 3.35 GB/s D2H** |
| compute (both) | full | full (`PASS_CMP90HX_ALL_TARGETS_FULL_SPEED`) |

`nvidia-smi --query-gpu=pcie.link.gen.current` reports `2` after the apply.

## The 2 masks

| address | name | why |
| --- | --- | --- |
| `0x00823800` | `FEAT_OVR_ECC_PLM` | gate checked by the patched module before it writes the Gen2 speed path |
| `0x00088fe8` | XVE privilege mask | unprotects `PRIV_MISC_1` / `LTSSM` speed-path registers |

The remaining 32 masks from the original rejoin16 table (XP3G x17, OPTB x10,
XVE x5) are **not** needed for Gen2. See `docs/FINDINGS.md` for the bisection.

## Requirements

* CMP 90HX, `10de:220d`, subsystem `10de:1555`
* NVIDIA **open** kernel modules `610.43.03` installed and the pearlfortune
  compute unlock working (`cmpunlocker-rs compute90hx-v67 verify --expect full`)
* matching kernel headers, `make`, `gcc`, `patch`, `setpci`, `python3`
* Secure Boot **off** (patched modules are unsigned)

## Install

```sh
sudo ./scripts/install.sh
sudo systemctl start cmp90hx-gen2.service   # or reboot
```

`install.sh` builds the patched `nvidia*.ko` for the running kernel and
installs it to `/usr/lib/modules/$(uname -r)/updates/cmpunlocker-90hx-stockflow`
(depmod override), then installs the boot service.

## How it runs

* **Every boot** the service checks both cards. If the 2 masks are already
  open (masks sometimes survive a warm reboot) it only retrains the link and
  finishes in seconds.
* If the masks are locked (always after a real power cycle), it runs one
  driver-reload cycle per closed mask (max 2 per card, ~20 s each), then the
  patched module retrains the link in kernel context. **Typical cold-boot
  recovery: ~1–3 minutes total for two cards.**
* BDFs are auto-detected every run — on this board the second card's BDF
  changes across reboots.

## Verify

```sh
cat /sys/bus/pci/devices/$(lspci -Dnn | awk '/10de:220d/{print $1; exit}')/current_link_speed   # 5.0 GT/s
nvidia-smi --query-gpu=index,pcie.link.gen.current --format=csv                                   # 2
sudo ./cmpunlocker-rs compute90hx-v67 verify --all-cmp90hx --expect full                          # PASS
```

## Uninstall

```sh
sudo ./scripts/uninstall.sh   # then reboot
```

## Caveats

* After a kernel or driver change you must re-run `install.sh` (module rebuilt
  for the new kernel).
* Do not run the apply while workloads are using the GPUs — it reloads the
  driver several times.
* The unlock is runtime state, not persistent firmware; the boot service is
  what makes it survive reboots.

## Credits

* [pearlfortune/cmpunlocker](https://github.com/pearlfortune/cmpunlocker) —
  90HX stockflow compute unlock (rejoin14/rejoin15) that this builds on.
* [jdowning100/cmpunlocker](https://github.com/jdowning100/cmpunlocker) —
  rejoin16 PCIe Gen2 patch (`90hx/0016-...pcie-jtag-plm.patch`), `bar0poke.c`,
  and the 34-mask table used as the starting point.
* [studebaker8/cmp170hx-gen2](https://github.com/studebaker8/cmp170hx-gen2) —
  Retrain-Link approach.
* NVIDIA — open GPU kernel modules 610.43.03.

## License

GPL-2.0 (see `LICENSE`), because this work derives from and patches
GPL-2.0 code (jdowning100/cmpunlocker).
