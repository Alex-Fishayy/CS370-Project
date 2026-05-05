#!/bin/bash
# =============================================================================
# firstrun.sh  —  Runs ONCE on the Pi's very first boot from this USB
# Placed in /boot/firstrun.sh  (the FAT32 boot partition, writable from Windows)
# Referenced by /boot/cmdline.txt (added automatically by FLASH_INSTRUCTIONS)
# =============================================================================
set -e
exec > >(tee -a /var/log/firstrun.log) 2>&1

echo "============================================================"
echo "[firstrun] Starting first-boot setup at $(date)"
echo "============================================================"

INSTALL_DIR="/home/pi/attention"
BOOT_DIR="/boot/firmware"   # Pi OS Bookworm mounts boot here
# Fallback for older Pi OS where boot is at /boot
[ -d "$BOOT_DIR" ] || BOOT_DIR="/boot"

# ── 1. Install software ──────────────────────────────────────────────────────
echo "[firstrun] Copying software to $INSTALL_DIR ..."
mkdir -p "$INSTALL_DIR"
cp -r "$BOOT_DIR/attention/." "$INSTALL_DIR/"
chown -R pi:pi "$INSTALL_DIR"

echo "[firstrun] Running setup.sh (installs Miniforge + packages) ..."
cd "$INSTALL_DIR"
sudo -u pi bash setup.sh

# ── 2. Copy helper scripts ────────────────────────────────────────────────────
echo "[firstrun] Installing helper scripts ..."
cp "$BOOT_DIR/usb_setup/wifi-scan.sh"             /usr/local/bin/wifi-scan.sh
chmod +x /usr/local/bin/wifi-scan.sh

# ── 3. Configure known WiFi networks via NetworkManager ──────────────────────
echo "[firstrun] Configuring WiFi networks ..."

NM_DIR="/etc/NetworkManager/system-connections"
mkdir -p "$NM_DIR"

# --- Network 1: 5G District at Campus West ---
UUID1=$(cat /proc/sys/kernel/random/uuid)
cat > "$NM_DIR/campus-west.nmconnection" << EOF
[connection]
id=5G District at Campus West
uuid=$UUID1
type=wifi
autoconnect=true
autoconnect-priority=20

[wifi]
ssid=5G District at Campus West
mode=infrastructure

[wifi-security]
key-mgmt=wpa-psk
psk=It~Brown-Octopus

[ipv4]
method=auto

[ipv6]
addr-gen-mode=default
method=auto
EOF
chmod 600 "$NM_DIR/campus-west.nmconnection"

# --- Network 2: Jackson’s iPhone ---
UUID2=$(cat /proc/sys/kernel/random/uuid)
cat > "$NM_DIR/jacksons-iphone.nmconnection" << EOF
[connection]
id=Jackson’s iPhone
uuid=$UUID2
type=wifi
autoconnect=true
autoconnect-priority=10

[wifi]
ssid=Jackson’s iPhone
mode=infrastructure

[wifi-security]
key-mgmt=wpa-psk
psk=jackson12

[ipv4]
method=auto

[ipv6]
addr-gen-mode=default
method=auto
EOF
chmod 600 "$NM_DIR/jacksons-iphone.nmconnection"

# Reload NetworkManager so it picks up the new profiles
systemctl reload NetworkManager 2>/dev/null || systemctl restart NetworkManager 2>/dev/null || true

# ── 4. Install systemd services ───────────────────────────────────────────────
echo "[firstrun] Installing systemd services ..."

cp "$BOOT_DIR/usb_setup/attention-tracker.service" /etc/systemd/system/
cp "$BOOT_DIR/usb_setup/wifi-autoconnect.service"  /etc/systemd/system/

systemctl daemon-reload
systemctl enable attention-tracker.service
systemctl enable wifi-autoconnect.service

# ── 5. Remove firstrun from cmdline.txt so it doesn't re-run ─────────────────
echo "[firstrun] Removing firstrun hook from cmdline.txt ..."
CMDLINE="$BOOT_DIR/cmdline.txt"
sed -i 's| systemd\.run=/boot/firmware/firstrun\.sh||g' "$CMDLINE" 2>/dev/null || true
sed -i 's| systemd\.run=/boot/firstrun\.sh||g'          "$CMDLINE" 2>/dev/null || true
sed -i 's| systemd\.run_success_action=reboot||g'       "$CMDLINE" 2>/dev/null || true
sed -i 's| systemd\.unit=kernel-command-line\.target||g' "$CMDLINE" 2>/dev/null || true

echo "[firstrun] ✓ Setup complete — rebooting in 5 seconds ..."
sleep 5
reboot
