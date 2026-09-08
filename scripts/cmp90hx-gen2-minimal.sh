#!/usr/bin/env bash
# CMP 90HX PCIe Gen2 unlock - minimal 2-mask apply (per card, auto BDF detect).
#
# Required masks (see docs/FINDINGS.md):
#   0x00823800  FEAT_OVR_ECC_PLM   (module Gen2-config gate)
#   0x00088fe8  XVE privilege mask (PRIV_MISC_1 / LTSSM speed-path gate)
#
# warm boot with masks still open  -> seconds (retrain only)
# cold boot with masks locked      -> max 2 reload cycles per card, ~40 s/card
set -uo pipefail

PREFIX="${CMP90_PREFIX:-/opt/cmp90hx-gen2}"
CYCLE="${CMP90_CYCLE:-$PREFIX/rejoin16-cycle.sh}"
READER="${CMP90_READER:-$PREFIX/maskread.py}"
MASK_FEAT_ECC=0x00823800
MASK_XVE=0x00088fe8

log() { echo "cmp90hx-gen2: $*"; }

mask_val() {  # <bdf> <addr> -> value
    python3 "$READER" "$1" "$2" 2>/dev/null | awk '{print $1}'
}

retrain() {   # <bdf>
    local bdf="$1" up lc nc
    up="$(basename "$(readlink -f "/sys/bus/pci/devices/${bdf}/..")")"
    setpci -s "$bdf" CAP_EXP+2c.w=0x0002
    setpci -s "$up"  CAP_EXP+2c.w=0x0002
    lc="$(setpci -s "$up" CAP_EXP+10.w)"
    printf -v nc '0x%x' $(( (16#$lc) | 0x20 ))
    setpci -s "$up" CAP_EXP+10.w="$nc"
    sleep 2
}

open_mask() { # <bdf> <addr>
    local bdf="$1" addr="$2" try cur
    for try in 1 2 3 4 5 6; do
        CMP90_BDF="$bdf" bash "$CYCLE" "$addr" 0xffffffff >/dev/null 2>&1
        cur="$(mask_val "$bdf" "$addr")"
        [[ "$cur" == "0xffffffff" ]] && { log "  $bdf open $addr OK (try $try)"; return 0; }
        sleep 1
    done
    log "  $bdf open $addr FAIL (readback=${cur:-none})"
    return 1
}

mapfile -t BDFS < <(lspci -Dnn | awk '/10de:220d/ {print $1}')
if [[ "${#BDFS[@]}" -eq 0 ]]; then
    log "no CMP 90HX found; nothing to do"
    exit 0
fi

for bdf in "${BDFS[@]}"; do
    m_feat="$(mask_val "$bdf" "$MASK_FEAT_ECC")"
    m_xve="$(mask_val "$bdf" "$MASK_XVE")"
    log "$bdf masks feat_ecc=$m_feat xve=$m_xve"

    if [[ "$m_feat" == "0xffffffff" && "$m_xve" == "0xffffffff" ]]; then
        log "  masks already open - no reload cycles needed"
    else
        [[ "$m_feat" == "0xffffffff" ]] || open_mask "$bdf" "$MASK_FEAT_ECC" || true
        [[ "$m_xve"  == "0xffffffff" ]] || open_mask "$bdf" "$MASK_XVE"      || true
    fi

    retrain "$bdf"
    spd="$(cat "/sys/bus/pci/devices/${bdf}/current_link_speed" 2>/dev/null)"
    gen="$(nvidia-smi --query-gpu=pcie.link.gen.current,pci.bus_id --format=csv,noheader,nounits 2>/dev/null \
           | awk -v b="${bdf#0000:}" '$2 ~ b {print $1; exit}')"
    log "  $bdf link=${spd:-?} gen=${gen:-?}"
done
log "done"
exit 0
