Get-Process -Name python,node -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Write-Host "Stopped local app processes." -ForegroundColor Yellow
