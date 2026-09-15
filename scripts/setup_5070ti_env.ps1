# Setup script for RTX 5070 Ti Workstation (Windows 11 / PowerShell)
# Run from c:\Users\quenc\Desktop\NetProject

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "   Setting up Environment for NVIDIA GeForce RTX 5070 Ti Build   " -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

# 1. Verify Python
Write-Host "[1/4] Verifying Python installation..." -ForegroundColor Yellow
$pyVersion = python --version 2>&1
Write-Host "Found: $pyVersion" -ForegroundColor Green

# 2. Setup Virtual Environment
if (-not (Test-Path ".venv")) {
    Write-Host "[2/4] Creating virtual environment (.venv)..." -ForegroundColor Yellow
    python -m venv .venv
} else {
    Write-Host "[2/4] Virtual environment (.venv) already exists." -ForegroundColor Green
}

$venvPython = ".\.venv\Scripts\python.exe"
$venvPip = ".\.venv\Scripts\pip.exe"

# 3. Install PyTorch with CUDA 12.6/13 Support
Write-Host "[3/4] Installing PyTorch with CUDA support for RTX 5070 Ti..." -ForegroundColor Yellow
& $venvPip install --upgrade pip
& $venvPip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130 --force-reinstall

# 4. Install Flower & Project Dependencies
Write-Host "[4/4] Installing Flower and project dependencies..." -ForegroundColor Yellow
& $venvPip install -r requirements.txt

# Smoke Test
Write-Host "`nRunning GPU Smoke Test on RTX 5070 Ti..." -ForegroundColor Cyan
& $venvPython -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); x = torch.randn(1000, 1000, device='cuda'); print('Tensor multiplication test on 5070 Ti: OK! Shape:', (x @ x).shape)"

Write-Host "`nEnvironment setup complete! You are ready to run the FL Coordinator." -ForegroundColor Green
