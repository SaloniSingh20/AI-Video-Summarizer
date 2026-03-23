Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object {
    $_.Path -like "*AI Video Summarizer*"
} | Stop-Process -Force -ErrorAction SilentlyContinue

Get-Process -Name node -ErrorAction SilentlyContinue | Where-Object {
    $_.Path -like "*AI Video Summarizer*"
} | Stop-Process -Force -ErrorAction SilentlyContinue

Write-Host "Stopped local backend/frontend related processes (if running)." -ForegroundColor Yellow
