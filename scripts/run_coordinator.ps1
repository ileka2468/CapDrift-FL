param(
    [int]$MinClients = 4,
    [int]$Rounds = 6,
    [int]$Port = 8080
)

$ErrorActionPreference = "Stop"

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "   Starting Central Federated Learning Coordinator (5070 Ti)     " -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

# Find current LAN IP address to display to user
$lanIp = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -like '*Ethernet*' -or $_.InterfaceAlias -like '*Wi-Fi*' } | Select-Object -First 1).IPAddress

Write-Host "`n>>> Coordinator IP Address on LAN: $lanIp" -ForegroundColor Green
Write-Host ">>> Edge nodes should connect to: ${lanIp}:${Port}" -ForegroundColor Yellow
Write-Host ">>> Waiting for $MinClients clients before starting rounds...`n" -ForegroundColor Yellow

$venvPython = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Virtual environment not found. Running python from PATH..." -ForegroundColor Gray
    $venvPython = "python"
}

& $venvPython src/server.py --rounds $Rounds --min_clients $MinClients --port $Port
