#!/usr/bin/env bash
# Make the PATCHED nvidia module the active one, safely.
#
# The patched module must not be the *first* nvidia driver load of a power
# cycle. Its V67 chain replaces the signature memdesc that a plain *stock* GSP
# boot leaves behind, so loading it first fails:
#
#   NVRM: GPU0 nvCheckFailedNoLog: Check failed:
#         s_cmp90PcStockSignatureMemdescByGpu[cmp90GpuSlot] == NULL @ kernel_gsp.c:5986
#   NVRM: GPU 0000:03:00.0: RmInitAdapter failed! (0x62:0x40:2119)
#
# and the half-booted GSP leaves WPR2 up, after which every retry in that same
# boot also fails ("_kgspBootGspRm: unexpected WPR2 already up ... the GPU is
# likely in a bad state and may need to be reset"). Only a reboot clears it,
# so a bad first load costs a reboot and the unlock never happens.
#
# Fix: bring the card up with the stock module first, then hand it over to the
# patched module. Pair with /etc/modprobe.d/cmp90hx-gen2-noauto.conf so udev
# does not auto-load the patched module before this runs.
set -uo pipefail

KREL="$(uname -r)"
PATCHED_DIR="/usr/lib/modules/${KREL}/updates/cmpunlocker-90hx-stockflow"
PATCHED_SRCV="$(modinfo -F srcversion "${PATCHED_DIR}/nvidia.ko" 2>/dev/null || true)"
UNLOAD=(nvidia_drm nvidia_modeset nvidia_uvm nvidia_peermem nvidia)

log() { echo "cmp90hx-handoff: $*"; }
loaded_srcv() { cat /sys/module/nvidia/srcversion 2>/dev/null || true; }
gpu_ok() { nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; }

wait_gpu() {  # <tries> (2 s each)
    local i
    for i in $(seq 1 "${1:-30}"); do gpu_ok && return 0; sleep 2; done
    return 1
}

unload_all() {
    modprobe -r "${UNLOAD[@]}" 2>/dev/null || { sleep 2; modprobe -r "${UNLOAD[@]}" 2>/dev/null; }
}

find_stock() {  # echo path of an unpatched nvidia.ko, if any
    local p
    for p in "/usr/lib/modules/${KREL}/updates/dkms/nvidia.ko" \
             "/lib/modules/${KREL}/updates/dkms/nvidia.ko"; do
        [[ -f "$p" ]] && { echo "$p"; return 0; }
    done
    p="$(find "/lib/modules/${KREL}" -name nvidia.ko 2>/dev/null | grep -v -- "$PATCHED_DIR" | head -1)"
    [[ -n "$p" ]] && echo "$p"
}

[[ -n "$PATCHED_SRCV" ]] || { log "FATAL: patched nvidia.ko missing at $PATCHED_DIR"; exit 1; }
[[ -c /dev/nvidiactl || -d /sys/module/nvidia ]] || true

# Fast path: patched module already active with a live GPU.
if [[ "$(loaded_srcv)" == "$PATCHED_SRCV" ]] && gpu_ok; then
    log "patched module already active (srcversion $PATCHED_SRCV)"
    exit 0
fi

STOCK="$(find_stock)"
if [[ -n "$STOCK" ]]; then
    log "priming GPU with stock module: $STOCK"
    unload_all
    modprobe ecc 2>/dev/null || true
    insmod "$STOCK" 2>/dev/null || log "WARNING: insmod stock module failed"
    if wait_gpu 20; then
        log "stock module brought the GPU up"
    else
        log "WARNING: stock module did not bring the GPU up"
    fi
else
    log "WARNING: no stock nvidia.ko found; patched module may fail as first load"
fi

log "handing over to the patched module"
unload_all
sleep 2
modprobe nvidia || { log "FATAL: modprobe nvidia failed"; exit 1; }
loaded="$(loaded_srcv)"
if [[ "$loaded" != "$PATCHED_SRCV" ]]; then
    log "WARNING: loaded srcversion '${loaded:-none}' != patched '$PATCHED_SRCV'"
fi
if wait_gpu 30; then
    modprobe nvidia_uvm 2>/dev/null || true
    log "patched module active (srcversion ${loaded:-?})"
    exit 0
fi
log "FATAL: GPU did not come up on the patched module"
exit 1
