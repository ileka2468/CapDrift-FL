# Launch 5070 Ti as Client (Hospital C or Local Testing)
# Run from c:\Users\quenc\Desktop\NetProject

param(
    [string]$Server = "127.0.0.1:8080",
    [string]$NodeId = "node_3",
    [int]$Epochs = 1
)

$venvPython = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    $venvPython = "python"
}

Write-Host "Starting RTX 5070 Ti Client [$NodeId] connecting to $Server..." -ForegroundColor Cyan
& $venvPython src/client.py --server $Server --node_id $NodeId --epochs $Epochs
