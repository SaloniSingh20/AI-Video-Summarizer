$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Test-Path .venv)) {
    py -3.11 -m venv .venv
}

./.venv/Scripts/python.exe -m pip install --upgrade pip setuptools wheel

try {
    ./.venv/Scripts/python.exe -m pip install -r requirements.txt
}
catch {
    Write-Host "Standard install failed. Retrying with CPU-only torch index..." -ForegroundColor Yellow
    ./.venv/Scripts/python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
    ./.venv/Scripts/python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
}

Write-Host "Backend environment ready." -ForegroundColor Green
