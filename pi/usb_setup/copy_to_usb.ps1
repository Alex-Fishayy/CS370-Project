# copy_to_usb.ps1
# Run this from the project root AFTER flashing Pi OS to the USB.
# Replace X with your actual USB drive letter.
param(
    [Parameter(Mandatory=$true)]
    [string]$DriveLetter   # e.g.  .\copy_to_usb.ps1 -DriveLetter E
)

$Root  = Split-Path -Parent $MyInvocation.MyCommand.Path
$USB   = "${DriveLetter}:"

Write-Host "[1/4] Copying firstrun.sh ..."
Copy-Item "$Root\pi\usb_setup\firstrun.sh" "$USB\firstrun.sh" -Force

Write-Host "[2/4] Copying usb_setup scripts ..."
New-Item -ItemType Directory -Force "$USB\usb_setup" | Out-Null
Copy-Item "$Root\pi\usb_setup\wifi-scan.sh"              "$USB\usb_setup\" -Force
Copy-Item "$Root\pi\usb_setup\attention-tracker.service" "$USB\usb_setup\" -Force
Copy-Item "$Root\pi\usb_setup\wifi-autoconnect.service"  "$USB\usb_setup\" -Force

Write-Host "[3/4] Copying pi\ software to attention\ ..."
if (Test-Path "$USB\attention") { Remove-Item "$USB\attention" -Recurse -Force }
Copy-Item "$Root\pi" "$USB\attention" -Recurse -Force

Write-Host "[4/4] Copying YOLOv8 model ..."
if (Test-Path "$Root\yolov8n.pt") {
    Copy-Item "$Root\yolov8n.pt" "$USB\attention\yolov8n.pt" -Force
} else {
    Write-Host "  (yolov8n.pt not found — skipping, Pi version uses TFLite anyway)"
}

Write-Host ""
Write-Host "Done!  Now open ${USB}\cmdline.txt in Notepad and append to the end of the single line:"
Write-Host ""
Write-Host "  systemd.run=/boot/firmware/firstrun.sh systemd.run_success_action=reboot systemd.unit=kernel-command-line.target"
Write-Host ""
Write-Host "Save, eject, and plug into the Pi."
