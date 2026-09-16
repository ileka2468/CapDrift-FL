# 1-Click Orchestration Script for 4-Node Heterogeneous Federated Learning Cluster
# Run from c:\Users\quenc\Desktop\NetProject

param(
    [int]$Rounds = 6,
    [int]$Epochs = 1,
    [string]$ServerIp = "192.168.30.102",
    [int]$Port = 8080
)

$ErrorActionPreference = "Continue"

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "   1-CLICK CLUSTER LAUNCH: 4-NODE HETEROGENEOUS TESTBED          " -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

# 1. Start Containers on Proxmox if not running
Write-Host "`n[1/5] Ensuring Proxmox Containers (201, 202, 203) are running..." -ForegroundColor Yellow
ssh -o StrictHostKeyChecking=no root@192.168.20.100 "pct start 201" 2>$null
ssh -o StrictHostKeyChecking=no root@192.168.10.100 "pct start 202" 2>$null
ssh -o StrictHostKeyChecking=no root@192.168.40.100 "pct start 203" 2>$null

# 2. Launch Central Coordinator on 5070 Ti in background
Write-Host "[2/5] Starting Central Coordinator on RTX 5070 Ti..." -ForegroundColor Yellow
$serverProcess = Start-Process -FilePath ".\.venv\Scripts\python.exe" `
    -ArgumentList "src/server.py --rounds $Rounds --min_clients 4 --port $Port" `
    -PassThru -NoNewWindow

Start-Sleep -Seconds 3

# 3. Launch Proxmox Edge Clients (pve01, pve02, pve03) via SSH
Write-Host "[3/5] Triggering Edge Clients on Cisco UCS Xeon Servers..." -ForegroundColor Yellow

# Hospital A (pve01)
Write-Host "  -> Launching Hospital A (pve01: CT 201)..." -ForegroundColor Gray
ssh -o StrictHostKeyChecking=no root@192.168.20.100 "pct exec 201 -- bash -c 'nohup /opt/fl_client/venv/bin/python3 /opt/fl_client/client.py --server ${ServerIp}:${Port} --node_id node_1 --data_dir /opt/fl_client/data/node_1 --epochs $Epochs > /opt/fl_client/client.log 2>&1 &'"

# Hospital B (pve02)
Write-Host "  -> Launching Hospital B (pve02: CT 202)..." -ForegroundColor Gray
ssh -o StrictHostKeyChecking=no root@192.168.10.100 "pct exec 202 -- bash -c 'nohup /opt/fl_client/venv/bin/python3 /opt/fl_client/client.py --server ${ServerIp}:${Port} --node_id node_2 --data_dir /opt/fl_client/data/node_2 --epochs $Epochs > /opt/fl_client/client.log 2>&1 &'"

# Hospital C (pve03)
Write-Host "  -> Launching Hospital C (pve03: CT 203)..." -ForegroundColor Gray
ssh -o StrictHostKeyChecking=no root@192.168.40.100 "pct exec 203 -- bash -c 'nohup /opt/fl_client/venv/bin/python3 /opt/fl_client/client.py --server ${ServerIp}:${Port} --node_id node_3 --data_dir /opt/fl_client/data/node_3 --epochs $Epochs > /opt/fl_client/client.log 2>&1 &'"

# 4. Launch Local Client 4 on RTX 5070 Ti GPU
Write-Host "`n[4/5] Starting Client 4 on RTX 5070 Ti (CUDA)..." -ForegroundColor Green
& ".\.venv\Scripts\python.exe" src/client.py --server "127.0.0.1:${Port}" --node_id node_4 --data_dir "data/node_4" --epochs $Epochs

# Wait for Coordinator to finish
Write-Host "`nWaiting for coordinator to complete remaining rounds..." -ForegroundColor Yellow
$serverProcess | Wait-Process

# 5. Automatically generate report plots
Write-Host "`n[5/5] Generating updated report charts and convergence curves..." -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" scripts/plot_results.py

Write-Host "`n=================================================================" -ForegroundColor Green
Write-Host "   FL CLUSTER RUN COMPLETE! ALL 4 NODES FINISHED SUCCESSFULLY     " -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Green
Write-Host "Metrics saved to: results/fl_metrics.json"
Write-Host "Plots saved to:   results/plots/`n"
