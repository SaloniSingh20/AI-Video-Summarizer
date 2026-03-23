$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$backendPath = Join-Path $projectRoot "backend"
$frontendPath = Join-Path $projectRoot "frontend"
if (-not (Test-Path $frontendPath)) {
    $frontendPath = Join-Path (Split-Path -Parent $projectRoot) "frontend"
}

if (-not (Test-Path $frontendPath)) {
    throw "Frontend folder not found. Expected either '$projectRoot\\frontend' or sibling '../frontend'."
}

$envPath = Join-Path $projectRoot ".env"
if (-not (Test-Path $envPath)) {
    throw "Missing .env file at project root."
}

function Read-DotEnvValue([string]$Key) {
    $line = Get-Content $envPath | Where-Object { $_ -match "^\s*$Key\s*=" } | Select-Object -First 1
    if (-not $line) { return "" }
    $value = ($line -split "=", 2)[1].Trim()
    return $value.Trim('"').Trim("'")
}

$openaiKey = Read-DotEnvValue "OPENAI_API_KEY"
$whisperModel = Read-DotEnvValue "WHISPER_MODEL"
if (-not $whisperModel) { $whisperModel = "base" }
$ollamaModel = Read-DotEnvValue "OLLAMA_MODEL"
if (-not $ollamaModel) { $ollamaModel = "llama3" }
$ollamaUrl = Read-DotEnvValue "OLLAMA_API_URL"
if (-not $ollamaUrl) { $ollamaUrl = "http://127.0.0.1:11434" }

$ffmpegPath = (Get-Command ffmpeg -ErrorAction SilentlyContinue).Source
if (-not $ffmpegPath) {
    $wingetFfmpeg = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe"
    if (Test-Path $wingetFfmpeg) {
        $ffmpegPath = $wingetFfmpeg
    }
}

if (-not $ffmpegPath) {
    throw "ffmpeg is not available. Install ffmpeg and retry."
}

Write-Host "Starting backend and frontend in local Whisper + Ollama mode..." -ForegroundColor Cyan

if (-not (Test-Path (Join-Path $backendPath ".venv\Scripts\python.exe"))) {
    throw "Backend virtualenv not found. Run backend setup first."
}

$backendCmd = "Set-Location '$backendPath'; `$env:OPENAI_API_KEY='$openaiKey'; `$env:WHISPER_MODEL='$whisperModel'; `$env:OLLAMA_MODEL='$ollamaModel'; `$env:OLLAMA_API_URL='$ollamaUrl'; `$env:FFMPEG_PATH='$ffmpegPath'; ./.venv/Scripts/python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
$frontendCmd = "Set-Location '$frontendPath'; `$env:API_PROXY_TARGET='http://127.0.0.1:8000'; npm run dev -- --port 5173"

Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd | Out-Null
Start-Sleep -Seconds 2
Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCmd | Out-Null

Write-Host "Backend:  http://localhost:8000" -ForegroundColor Green
Write-Host "Docs:     http://localhost:8000/docs" -ForegroundColor Green
Write-Host "Frontend: http://localhost:5173" -ForegroundColor Green
