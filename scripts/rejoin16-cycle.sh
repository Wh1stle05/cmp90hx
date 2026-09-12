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

# Re-lock the compute selectors BEFORE unloading, while BAR0 is still
# accessible. After `modprobe -r` the device drops into a low-power state
# (BAR0 reads back 0xffffffff, every write REJECTED), so the old order
# (unload first, re-lock after) silently left SS0/SS1 full, the V67 canary
# saw "already present" and skipped the Booter chain entirely
# ("(no REJOIN16 lines!)" + PCIe FAIL every cycle). Root trigger was a
# power-management behavior change after an `apt --fix-broken` pulled in
# libnvidia-compute-580/535 + regenerated initramfs; the kernel module
# itself is still 610.43.03. Verified 2026-09-09 on ubuntu1: pre-unload
# re-lock makes REJOIN16 fire on every cycle.
relock() {  # re-lock SS0/SS1 to 0 with readback check; 0 on success
    local try cur0 cur1 reader="${SCRIPT_DIR}/maskread.py"
    for try in 1 2 3; do
        "$POKE" "$BDF" wr 0x0082381c 0x0 >/dev/null 2>&1
        "$POKE" "$BDF" wr 0x00823820 0x0 >/dev/null 2>&1
        if [[ -f "$reader" ]]; then
            read -r cur0 cur1 <<<"$(python3 "$reader" "$BDF" 0x0082381c 0x00823820 2>/dev/null)"
            [[ "$cur0" == "0x00000000" && "$cur1" == "0x00000000" ]] && return 0
        else
            return 0  # no reader available; assume writes landed (legacy path)
        fi
        sleep 1
    done
    return 1
}

restore_full() {  # best-effort restore of full selectors (compute safety net)
    "$POKE" "$BDF" wr 0x0082381c 0x88888888 >/dev/null 2>&1 || true
    "$POKE" "$BDF" wr 0x00823820 0x00000008 >/dev/null 2>&1 || true
}

relock || { echo "FATAL: cannot re-lock selectors while driver loaded; aborting before unload"; restore_full; exit 1; }

# Unload the whole stack (nvidia_drm/nvidia_modeset may be held by udev).
modprobe -r nvidia_drm nvidia_modeset nvidia_uvm nvidia_peermem nvidia 2>/dev/null || {
    sleep 2
    modprobe -r nvidia_drm nvidia_modeset nvidia_uvm nvidia_peermem nvidia 2>/dev/null
} || {
    # 卸载失败(通常是有进程占着 /dev/nvidia*, 例如 ComfyUI 已初始化 CUDA)。
    # 此时上面的 relock 已经把 SS0/SS1 重置为 0, 若不恢复就会【静默丢失算力解锁】
    # (实测: 服务读到 masks=0xffffff8f, 掩码写不进, PCIe 停在 Gen1)。
    echo "FATAL: cannot unload nvidia stack; restoring full selectors"
    restore_full
    exit 1
}

# Re-lock the compute selectors so the canary path runs on the next load.
# (Done above, before unload — see relock(). Writes after unload are
# REJECTED because BAR0 is inaccessible once the driver is gone.)

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
