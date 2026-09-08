#!/usr/bin/env bash
# Verify CMP 90HX compute + PCIe Gen2 unlock state.
# usage: sudo ./scripts/verify.sh [--bench] [--strict]
#   --bench   also run the cuBLAS/copy benchmark (needs nvcc)
#   --strict  exit non-zero if anything is not unlocked
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${CMP90_PREFIX:-/opt/cmp90hx-gen2}"
READER="$PREFIX/maskread.py"
[[ -f "$READER" ]] || READER="$SCRIPT_DIR/maskread.py"

BENCH=0; STRICT=0
for a in "$@"; do
    case "$a" in
        --bench) BENCH=1 ;;
        --strict) STRICT=1 ;;
        *) echo "unknown arg: $a" >&2; exit 2 ;;
    esac
done

[[ "$(id -u)" -eq 0 ]] || { echo "run as root (BAR0 reads)" >&2; exit 1; }

fail=0
mapfile -t BDFS < <(lspci -Dnn | awk '/10de:220d/ {print $1}')
[[ "${#BDFS[@]}" -gt 0 ]] || { echo "no CMP 90HX found"; exit 1; }

echo "== compute =="
for bdf in "${BDFS[@]}"; do
    # SS0=0x82381c, SS1=0x823820, FEAT_OVR_PLM=0x823804
    read -r ss0 ss1 feat <<<"$(python3 "$READER" "$bdf" 0x0082381c 0x00823820 0x00823804 2>/dev/null)"
    if [[ "$ss0" == "0x88888888" && "$ss1" == "0x00000008" ]]; then
        printf 'compute  %-14s SS0=%s SS1=%s  OK (full)\n' "$bdf" "$ss0" "$ss1"
    else
        printf 'compute  %-14s SS0=%s SS1=%s  FAIL (locked)\n' "$bdf" "${ss0:-?}" "${ss1:-?}"
        fail=1
    fi
done

echo "== pcie =="
for bdf in "${BDFS[@]}"; do
    spd="$(cat "/sys/bus/pci/devices/${bdf}/current_link_speed" 2>/dev/null)"
    wid="$(cat "/sys/bus/pci/devices/${bdf}/current_link_width" 2>/dev/null)"
    gen="$(nvidia-smi --query-gpu=pcie.link.gen.current,pci.bus_id --format=csv,noheader,nounits 2>/dev/null \
           | awk -v b="${bdf#0000:}" '$2 ~ b {print $1; exit}')"
    if [[ "$spd" == 5.0* && "$gen" == "2" ]]; then
        printf 'pcie     %-14s %s x%s  nvidia-smi gen=%s OK\n' "$bdf" "$spd" "$wid" "$gen"
    else
        printf 'pcie     %-14s %s x%s  nvidia-smi gen=%s FAIL (run: systemctl start cmp90hx-gen2)\n' \
               "$bdf" "${spd:-?}" "${wid:-?}" "${gen:-?}"
        fail=1
    fi
done

if [[ "$BENCH" -eq 1 ]]; then
    echo "== benchmark =="
    BENCH_BIN="$PREFIX/bench"
    if [[ ! -x "$BENCH_BIN" ]]; then
        if command -v nvcc >/dev/null && [[ -f "$SCRIPT_DIR/../tools/bench.cu" ]]; then
            nvcc -O2 -arch=sm_86 "$SCRIPT_DIR/../tools/bench.cu" -lcublas -o /tmp/cmp90-bench 2>/dev/null \
                && BENCH_BIN=/tmp/cmp90-bench
        fi
    fi
    if [[ -x "$BENCH_BIN" ]]; then
        for i in "${!BDFS[@]}"; do
            echo "-- GPU$i (${BDFS[$i]})"
            CUDA_VISIBLE_DEVICES="$i" "$BENCH_BIN" fp32 | sed 's/^/   /'
            CUDA_VISIBLE_DEVICES="$i" "$BENCH_BIN" copyh2d | sed 's/^/   /'
            CUDA_VISIBLE_DEVICES="$i" "$BENCH_BIN" copyd2h | sed 's/^/   /'
        done
    else
        echo "   (bench binary not available; install nvcc or run from the repo root)"
    fi
fi

echo
if [[ "$fail" -eq 0 ]]; then
    echo "ALL OK: compute full + PCIe Gen2"
else
    echo "NOT fully unlocked (see FAIL lines above)"
    [[ "$STRICT" -eq 1 ]] && exit 1
fi
exit 0
