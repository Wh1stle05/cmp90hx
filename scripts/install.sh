#!/usr/bin/env bash
# Build + install the patched NVIDIA open 610.43.03 module (compute + PCIe
# Gen2 unlock) and the boot service.
#
# Upstream pieces are fetched at install time (not vendored):
#   - pearlfortune/cmpunlocker v0.1.28 90HX stockflow bundle (MIT): patches 0014/0015
#   - jdowning100/cmpunlocker rejoin16 patch 0016 (GPL-2.0, pinned commit)
#   - NVIDIA open kernel module source 610.43.03
# This repo adds patches/0017 (retrain retry) and the minimal-2-mask apply.
set -Eeuo pipefail

DRIVER_EXPECT=610.43.03
NV_SRC_SHA256=7e118923c7a23edc36114d63273a46e3e04e9af98695a42203e7ac2dfe9fc1dc
PEARL_URL=https://github.com/pearlfortune/cmpunlocker/releases/download/v0.1.28
PEARL_ASSET=cmpunlocker-v0.1.28-linux-x64-90hx-stockflow
JD_COMMIT=2cdc63cf1ab80d8128cf9e77216e3f3c132b212f
JD_0016_URL=https://raw.githubusercontent.com/jdowning100/cmpunlocker/${JD_COMMIT}/90hx/0016-6104303-cmp90hx-stockflow-rejoin16-pcie-jtag-plm.patch
NV_SRC_URL=https://download.nvidia.com/XFree86/NVIDIA-kernel-module-source/NVIDIA-kernel-module-source-${DRIVER_EXPECT}.tar.xz

PREFIX=/opt/cmp90hx-gen2
WORK="${WORK:-/var/tmp/cmp90hx-gen2-build}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KREL="$(uname -r)"
UPDATES="/usr/lib/modules/${KREL}/updates/cmpunlocker-90hx-stockflow"

die() { echo "[FAIL] $*" >&2; exit 1; }
info() { echo "[INFO] $*"; }

[[ "$(id -u)" -eq 0 ]] || die "run as root"
[[ -d "/lib/modules/${KREL}/build" ]] || die "kernel headers missing: /lib/modules/${KREL}/build"
for c in make gcc patch depmod modinfo lspci setpci python3; do
    command -v "$c" >/dev/null || die "required command missing: $c"
done
lspci -Dnn > /tmp/.cmp90-lspci.$$ 2>/dev/null || true
grep -q '10de:220d' /tmp/.cmp90-lspci.$$ || die "no CMP 90HX (10de:220d) found"
rm -f /tmp/.cmp90-lspci.$$

cur="$(modinfo -F version nvidia 2>/dev/null || true)"
[[ "$cur" == "$DRIVER_EXPECT" ]] || die "nvidia driver is '$cur', expected $DRIVER_EXPECT (open modules)"
lic="$(modinfo -F license nvidia 2>/dev/null || true)"
case "$lic" in *MIT*|*GPL*) ;; *) info "WARNING: module license '$lic' does not look open" ;; esac

mkdir -p "$WORK"
cd "$WORK"

# 1. pearlfortune bundle (patches 0014/0015)
if [[ ! -f "${PEARL_ASSET}.tar.gz" ]]; then
    info "downloading pearlfortune ${PEARL_ASSET}"
    curl -fL --retry 3 -o "${PEARL_ASSET}.tar.gz" "${PEARL_URL}/${PEARL_ASSET}.tar.gz"
    curl -fL --retry 3 -o SHA256SUMS "${PEARL_URL}/SHA256SUMS"
    sha256sum -c SHA256SUMS --ignore-missing
fi
[[ -d "${PEARL_ASSET}" ]] || tar xzf "${PEARL_ASSET}.tar.gz"
BUNDLE="${WORK}/${PEARL_ASSET}/stockflow/${DRIVER_EXPECT}"
[[ -d "$BUNDLE/patches" ]] || die "unexpected bundle layout: $BUNDLE"

# 2. NVIDIA open kernel module source
if [[ ! -f "NVIDIA-kernel-module-source-${DRIVER_EXPECT}.tar.xz" ]]; then
    info "downloading NVIDIA kernel module source ${DRIVER_EXPECT}"
    curl -fL --retry 3 -o "NVIDIA-kernel-module-source-${DRIVER_EXPECT}.tar.xz" "$NV_SRC_URL"
fi
echo "${NV_SRC_SHA256}  NVIDIA-kernel-module-source-${DRIVER_EXPECT}.tar.xz" | sha256sum -c -

# 3. jdowning100 0016 patch
if [[ ! -f 0016-jd.patch ]]; then
    info "downloading jdowning100 rejoin16 patch (${JD_COMMIT:0:12})"
    curl -fL --retry 3 -o 0016-jd.patch "$JD_0016_URL"
fi

# 4. build tree
info "extracting + patching source"
rm -rf "src-${KREL}" && mkdir -p "src-${KREL}"
tar xf "NVIDIA-kernel-module-source-${DRIVER_EXPECT}.tar.xz" -C "src-${KREL}" --strip-components=1
(
    cd "src-${KREL}"
    patch -p1 -s < "$BUNDLE/patches/0014-"*.patch
    patch -p1 -s < "$BUNDLE/patches/0015-"*.patch
    patch -p1 -s < "$WORK/0016-jd.patch"
    patch -p1 -s < "$REPO/patches/0017-cmp90hx-gen2-retrain-retry.patch"
    # low-memory friendly build of the huge generated firmware blob
    if ! grep -qF CMP90_LOW_MEM_G_BINDATA src/nvidia/Makefile; then
        printf '\n# low-mem: compile generated/g_bindata.c at O0\n' >> src/nvidia/Makefile
        printf '$(call BUILD_OBJECT_LIST,generated/g_bindata.c): CFLAGS := $(filter-out -O2,$(CFLAGS)) -O0\n' >> src/nvidia/Makefile
    fi
    make -s modules -j"${JOBS:-$(nproc)}" KERNEL_UNAME="$KREL"
)

ART="src-${KREL}/kernel-open"
[[ -f "$ART/nvidia.ko" ]] || die "build did not produce nvidia.ko"
[[ "$(modinfo -F version "$ART/nvidia.ko")" == "$DRIVER_EXPECT" ]] || die "built module version mismatch"
[[ "$(modinfo -F vermagic "$ART/nvidia.ko" | awk '{print $1}')" == "$KREL" ]] || die "vermagic mismatch"
strings "$ART/nvidia.ko" > /tmp/.cmp90-strings.$$ 2>/dev/null || true
grep -q CMP90_STOCKFLOW_REJOIN16 /tmp/.cmp90-strings.$$ || die "rejoin16 marker missing"
rm -f /tmp/.cmp90-strings.$$

# 5. install modules (backup any previous content)
info "installing modules to $UPDATES"
if [[ -d "$UPDATES" ]]; then
    mv "$UPDATES" "${UPDATES}.bak.$(date +%Y%m%d%H%M%S)"
fi
mkdir -p "$UPDATES"
for m in nvidia nvidia-uvm nvidia-modeset nvidia-drm nvidia-peermem; do
    [[ -f "$ART/$m.ko" ]] && install -m0644 "$ART/$m.ko" "$UPDATES/"
done
depmod -a "$KREL"
command -v update-initramfs >/dev/null && update-initramfs -u -k "$KREL" || true

# 6. install helper scripts + bar0poke
info "installing helper scripts to $PREFIX"
mkdir -p "$PREFIX"
install -m0755 "$REPO/scripts/cmp90hx-gen2-minimal.sh" "$PREFIX/"
install -m0755 "$REPO/scripts/rejoin16-cycle.sh" "$PREFIX/"
install -m0644 "$REPO/scripts/maskread.py" "$PREFIX/"
install -m0644 "$REPO/tools/bench.cu" "$PREFIX/"
gcc -O2 -o "$PREFIX/bar0poke" "$REPO/tools/bar0poke.c"
if command -v nvcc >/dev/null 2>&1; then
    nvcc -O2 -arch=sm_86 "$REPO/tools/bench.cu" -lcublas -o "$PREFIX/bench" 2>/dev/null \
        && info "bench binary installed" || info "WARNING: bench build failed (nvcc/cuBLAS missing?)"
else
    info "WARNING: nvcc not found; bench binary not built (verify.sh --bench will try to build it)"
fi

# 7. systemd unit
info "installing systemd unit"
install -m0644 "$REPO/systemd/cmp90hx-gen2.service" /etc/systemd/system/cmp90hx-gen2.service
systemctl daemon-reload
systemctl enable cmp90hx-gen2.service >/dev/null 2>&1 || true

cat <<EOF

[PASS] installed.
  modules : $UPDATES (backup: ${UPDATES}.bak.* if any)
  scripts : $PREFIX
  service : cmp90hx-gen2.service (enabled)

Next:
  sudo systemctl start cmp90hx-gen2.service   # apply now (or reboot)
  cat /sys/bus/pci/devices/\$(lspci -Dnn | awk '/10de:220d/{print \$1; exit}')/current_link_speed
  nvidia-smi --query-gpu=index,pcie.link.gen.current --format=csv
EOF
