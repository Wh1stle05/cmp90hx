#!/usr/bin/env bash
# Drive one rejoin16 crafted-Booter write per module reload.
#
# The GA102 V67 chain fires only once per FLR-separated module load, so each
# mask register needs its own unload/reload cycle. The patched module reads
# /var/lib/cmpunlocker-rs/rejoin16-next-write.bin (8 bytes LE: addr, value) in
# the canary-success branch and fires exactly one write.
#
# Usage: ./rejoin16-cycle.sh <addr> <value>
#   env CMP90_BDF  target card (default: first CMP 90HX)
set -uo pipefail

ADDR="$1"
VALUE="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BDF="${CMP90_BDF:-$(lspci -Dnn | awk '/10de:220d/ {print $1; exit}')}"
SPEC=/var/lib/cmpunlocker-rs/rejoin16-next-write.bin
POKE="${CMP90_POKE:-${SCRIPT_DIR}/bar0poke}"

[[ -n "$BDF" ]] || { echo "FATAL: no CMP 90HX (10de:220d) found; set CMP90_BDF"; exit 2; }
[[ -n "$ADDR" && -n "$VALUE" ]] || { echo "usage: $0 <addr> <value>"; exit 2; }

mkdir -p /var/lib/cmpunlocker-rs
python3 - "$ADDR" "$VALUE" "$SPEC" <<'PY'
import struct, sys
addr, val, path = int(sys.argv[1], 0), int(sys.argv[2], 0), sys.argv[3]
with open(path, "wb") as f:
    f.write(struct.pack("<II", addr, val))
PY

# Unload the whole stack (nvidia_drm/nvidia_modeset may be held by udev).
modprobe -r nvidia_drm nvidia_modeset nvidia_uvm nvidia_peermem nvidia 2>/dev/null || {
    sleep 2
    modprobe -r nvidia_drm nvidia_modeset nvidia_uvm nvidia_peermem nvidia 2>/dev/null
} || { echo "FATAL: cannot unload nvidia stack"; exit 1; }

# Re-lock the compute selectors so the canary path runs on the next load.
"$POKE" "$BDF" wr 0x0082381c 0x0 >/dev/null
"$POKE" "$BDF" wr 0x00823820 0x0 >/dev/null

dmesg -C 2>/dev/null || true

# modprobe (not insmod) so kernel crypto deps (ecdh/ecc) and DRM resolve
# automatically. The patched build is installed in
# /usr/lib/modules/$(uname -r)/updates/cmpunlocker-90hx-stockflow with a depmod
# override, so this loads the patched nvidia.ko.
modprobe nvidia || { echo "FATAL: modprobe nvidia failed"; exit 1; }
sleep 1
nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1   # trigger RM init
sleep 1
modprobe nvidia_uvm 2>/dev/null || true

echo "--- REJOIN16 result ---"
dmesg | grep -E "REJOIN16: (spec|fwsec|write|refill)" || echo "(no REJOIN16 lines!)"
echo "--- readback ---"
"$POKE" "$BDF" wr "$ADDR" "$VALUE" 2>/dev/null | sed 's/^/  (cpu-write probe) /' || true
