# 1-Click Launcher for OmniFed Interactive Diagnostic Studio
# Run from c:\Users\quenc\Desktop\NetProject

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "   LAUNCHING OMNIFED INTERACTIVE DIAGNOSTIC STUDIO                " -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan

# Check if port 5000 is open/in use
$portInUse = Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue
if ($portInUse) {
    Write-Host "Port 5000 is currently occupied. Freeing port 5000..." -ForegroundColor Yellow
    Get-Process -Id (Get-NetTCPConnection -LocalPort 5000).OwningProcess -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 1
}

Write-Host "`nStarting FastAPI Diagnostic Server on RTX 5070 Ti (CUDA)..." -ForegroundColor Green
Start-Process "http://localhost:5000"

& ".\.venv\Scripts\python.exe" src/app.py
