#!/usr/bin/env bash
# =============================================================================
# wifi-scan.sh  —  Runs as a systemd service (wifi-autoconnect.service)
# Every 30 seconds:
#   1. Checks if the Pi has internet/local connectivity
#   2. If not connected, tries known saved networks first
#   3. Then scans for any open (passwordless) WiFi networks and connects
# =============================================================================

log() { echo "[wifi-scan $(date '+%H:%M:%S')] $*"; }

log "WiFi auto-connect daemon started"

while true; do
    # ── Check current connectivity ──────────────────────────────────────────
    STATE=$(nmcli -t -f STATE general 2>/dev/null || echo "unknown")

    if [[ "$STATE" == "connected" ]]; then
        # Already connected — log which network and sleep
        ACTIVE=$(nmcli -t -f NAME connection show --active 2>/dev/null | head -1)
        log "Connected to: ${ACTIVE:-unknown}"
        sleep 30
        continue
    fi

    log "Not connected (state=$STATE) — scanning for networks ..."

    # ── Trigger a fresh scan ────────────────────────────────────────────────
    nmcli device wifi rescan 2>/dev/null || true
    sleep 4   # give scan time to populate

    # ── Try saved (known) networks first ────────────────────────────────────
    log "Attempting saved networks ..."
    nmcli connection up "5G District at Campus West" 2>/dev/null && {
        log "Connected to: 5G District at Campus West"; sleep 30; continue
    }
    nmcli connection up "Jackson’s iPhone" 2>/dev/null && {
        log "Connected to: Jackson’s iPhone"; sleep 30; continue
    }

    # ── Scan for open (no-password) networks ────────────────────────────────
    log "Scanning for open networks ..."
    # nmcli returns SECURITY as "--" for open networks
    mapfile -t OPEN_SSIDS < <(
        nmcli -t -f SSID,SECURITY device wifi list 2>/dev/null \
            | awk -F':' '{ security=$NF; ssid=$0; sub(/:[^:]*$/, "", ssid); if (security == "--" && ssid != "") print ssid }' \
            | sort -u
    )

    if [[ ${#OPEN_SSIDS[@]} -eq 0 ]]; then
        log "No open networks found — will retry in 30s"
    else
        for SSID in "${OPEN_SSIDS[@]}"; do
            log "Trying open network: \"$SSID\""
            if nmcli device wifi connect "$SSID" 2>/dev/null; then
                log "Connected to open network: \"$SSID\""
                break
            fi
        done
    fi

    sleep 30
done
