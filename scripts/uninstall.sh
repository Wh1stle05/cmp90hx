#!/usr/bin/env bash
# Remove the patched module + boot service (compute unlock modules are also
# removed; reinstall the pearlfortune stockflow if you want to keep compute
# unlocked without PCIe Gen2).
set -Eeuo pipefail

PREFIX=/opt/cmp90hx-gen2
KREL="$(uname -r)"
UPDATES="/usr/lib/modules/${KREL}/updates/cmpunlocker-90hx-stockflow"

[[ "$(id -u)" -eq 0 ]] || { echo "run as root" >&2; exit 1; }

systemctl disable --now cmp90hx-gen2.service 2>/dev/null || true
rm -f /etc/systemd/system/cmp90hx-gen2.service
systemctl daemon-reload

if [[ -d "$UPDATES" ]]; then
    rm -rf "$UPDATES"
fi
depmod -a "$KREL"
command -v update-initramfs >/dev/null && update-initramfs -u -k "$KREL" || true
rm -rf "$PREFIX"

echo "[PASS] removed. Reboot to fall back to the stock nvidia module."
